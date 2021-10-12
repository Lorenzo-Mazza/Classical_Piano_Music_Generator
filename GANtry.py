
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.initializers.initializers_v2 import RandomNormal
from keras.optimizer_v2.adam import Adam
from keras.optimizer_v2.rmsprop import RMSprop
from music21 import *
import time
import os
import pandas as pd
import numpy as np
import tensorflow as tf
import keras
from keras.utils import np_utils

import LoadPianoroll
from data import *
import tensorflow as tf
from keras.layers import LSTM, Input, Dropout, Dense, Activation, Embedding, Concatenate, Reshape
from keras.layers import Flatten, RepeatVector, Permute, TimeDistributed
from keras.layers import Multiply, Lambda, Softmax
import keras.backend as K
from keras.models import Model
import tensorflow.keras.optimizers as opt
from keras.layers import Input, Dense, Reshape, Flatten, Dropout
from keras.layers import BatchNormalization, Activation, ZeroPadding2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D, Conv1D, Conv2DTranspose, Conv3D
from keras.models import Sequential, Model


FIXED_NUMBER_OF_BARS= 64
QUANTIZATION = 32
BATCH_SIZE = 50
latent_dimension = 256


class RNNGAN:
    def __init__(self, input_shape, discriminator_lr, generator_lr,
                 optimiser, z_dim, batch_size, quantization):

        self.discriminator_lr = discriminator_lr
        self.generator_lr = generator_lr
        self.optimiser = optimiser

        self.input_shape = input_shape  # 128
        self.z_dim = z_dim  # size of encoding

        self.n_bars= FIXED_NUMBER_OF_BARS  # Generates only fixed length music
        self.n_steps_per_bar= quantization
        self.weight_init = RandomNormal(mean=0., stddev=0.02)  # 'he_normal' #RandomNormal(mean=0., stddev=0.02)
        self.batch_size = batch_size

        self.d_losses = []
        self.g_losses = []
        self.epoch = 0


        """ Input Shape: N_SEQUENCES X SEQ_LENGTH for each feature (pitch,duration,velocity)"""
        """ Output Shape: N_SEQUENCES X N_DIFFERENT CLASSES for each feature (pitch,duration,velocity)"""

        # Build and compile the discriminator
        self.discriminator = self.build_discriminator()
        # Build the generator
        self.generator = self.build_generator()
        self.build_adversarial()


    @staticmethod
    def wasserstein(y_true, y_pred):
        return K.mean(y_true * y_pred)

    def conv_t(self, x, f, k, s, a, p, bn):
        x = Conv2DTranspose(
            filters=f
            , kernel_size=k
            , padding=p
            , strides=s
            , kernel_initializer=self.weight_init)(x)
        if bn:
            x = BatchNormalization(momentum=0.9)(x)
        if a == 'relu':
            x = Activation(a)(x)
        elif a == 'lrelu':
            x = LeakyReLU()(x)
        return x

    def conv(self, x, f, k, s, a, p):
        x = Conv3D(
            filters=f
            , kernel_size=k
            , padding=p
            , strides=s
            , kernel_initializer=self.weight_init)(x)
        if a == 'relu':
            x = Activation(a)(x)
        elif a == 'lrelu':
            x = LeakyReLU()(x)
        return x


    def build_generator(self):
        """ create the structure of the neural network """
        input_layer = Input(shape=(self.z_dim,))
        #x = Embedding(self.input_shape, self.z_dim)(input_layer)
        x = Dense(1024)(input_layer)
        x = BatchNormalization(momentum=0.9)(x)
        x = Activation('relu')(x)
        x = Reshape([2, 1, 512])(x)
        x = self.conv_t(x, f=512, k=(4, 1), s=(4, 1), a='relu', p='same', bn=True)
        x = self.conv_t(x, f=512, k=(2, 1), s=(2, 1), a='relu', p='same', bn=True)
        x = self.conv_t(x, f=256, k=(2, 1), s=(2, 1), a='relu', p='same', bn=True)
        x = self.conv_t(x, f=256, k=(1, 8), s=(1, 8), a='relu', p='same', bn=True)
        x = self.conv_t(x, f=self.n_bars, k=(1, 16), s=(1, 16), a='tanh', p='same', bn=False)
        output_layer = Reshape([self.n_bars, self.n_steps_per_bar, self.input_shape,1])(x)
        return Model(input_layer, output_layer)

    def build_discriminator(self):

        critic_input = Input(shape=(self.n_bars,self.n_steps_per_bar,self.input_shape,1), name='critic_input')
        x = critic_input
        x = self.conv(x, f=128, k=(2, 1, 1), s=(1, 1, 1), a='lrelu', p='valid')
        x = self.conv(x, f=128, k=(int(self.n_bars/4), 1, 1), s=(1, 1, 1), a='lrelu',p='valid')
        x = self.conv(x, f=128, k=(8, 1, 12), s=(8, 1, 12), a='lrelu', p='same')
        x = self.conv(x, f=128, k=(1, 1, 7), s=(1, 1, 7), a='lrelu', p='same')
        x = self.conv(x, f=128, k=(1, 2, 1), s=(1, 2, 1), a='lrelu', p='same')
        x = self.conv(x, f=128, k=(1, 2, 1), s=(1, 2, 1), a='lrelu', p='same')
        x = self.conv(x, f=256, k=(1, 4, 1), s=(1, 2, 1), a='lrelu', p='same')
        x = self.conv(x, f=512, k=(1, 3, 1), s=(1, 2, 1), a='lrelu', p='same')
        x = Flatten()(x)
        x = Dense(1024, kernel_initializer=self.weight_init)(x)
        x = LeakyReLU()(x)
        critic_output = Dense(1, activation=None, kernel_initializer=self.weight_init)(x)

        return Model(critic_input, critic_output)


    def get_opti(self, lr):
        if self.optimiser == 'adam':
            opti = Adam(learning_rate=lr, beta_1=0.5, beta_2=0.9)
        elif self.optimiser == 'rmsprop':
            opti = RMSprop(learning_rate=lr)
        else:
            opti = Adam(learning_rate=lr)
        return opti


    def set_trainable(self, m, val):
        m.trainable = val
        for l in m.layers:
            l.trainable = val



    def build_adversarial(self):

        # Freeze generator's layers while training critic
        self.set_trainable(self.generator, False)
        # Image input (real sample)
        real_img = Input(shape=(self.n_bars, self.n_steps_per_bar, self.input_shape,1))

        # Fake image
        noise_input = Input(shape=(self.z_dim,), name='noise_input')
        fake_img = self.generator(noise_input)

        # discriminator determines validity of the real and fake images
        fake = self.discriminator(fake_img)
        valid = self.discriminator(real_img)

        self.critic_model = Model(inputs=[real_img, noise_input],outputs=[valid, fake])

        self.critic_model.compile(
            loss=[self.wasserstein, self.wasserstein]
            , optimizer=self.get_opti(self.discriminator_lr)
            , loss_weights=[1, 1])

        # For the generator we freeze the critic's layers
        self.set_trainable(self.discriminator, False)
        self.set_trainable(self.generator, True)

        # Sampled noise for input to generator
        noise_input = Input(shape=(self.z_dim,), name='noise_input')

        # Generate images based of noise
        img = self.generator(noise_input)
        # Discriminator determines validity
        model_output = self.discriminator(img)
        # Defines generator model
        self.model = Model(noise_input, model_output)

        self.model.compile(optimizer=self.get_opti(self.generator_lr)
                           , loss=self.wasserstein
                           )
        self.set_trainable(self.discriminator, True)


    def train_discriminator(self, x_train, batch_size, using_generator, x_train_iter):

        valid = np.ones((batch_size, 1), dtype=np.float32)
        fake = -np.ones((batch_size, 1), dtype=np.float32)
        #dummy = np.zeros((batch_size, 1), dtype=np.float32)  # Dummy gt for gradient penalty
        if using_generator:
            true_imgs = next(x_train_iter)[0]
            if true_imgs.shape[0] != batch_size:
                true_imgs = next(x_train_iter)[0]
        else:
            idx = np.random.randint(0, len(x_train))
            true_imgs = x_train[idx]

        noise = np.random.normal(0, 1, (batch_size, self.z_dim))
        #noise = np.random.uniform(0, 128, (batch_size, self.z_dim))
        d_loss = self.critic_model.train_on_batch([true_imgs, noise],[valid, fake])
        return d_loss


    def train_generator(self, batch_size):
        valid = np.ones((batch_size, 1), dtype=np.float32)

        noise = np.random.normal(0, 1, (batch_size, self.z_dim))
        return self.model.train_on_batch(noise, valid)



    def train(self, x_train, batch_size, epochs, n_critic=5, using_generator=False):
        x_train_iter= x_train.as_numpy_iterator()
        x_train= list(x_train_iter)
        for epoch in range(self.epoch, self.epoch + epochs):
            if epoch % 100 == 0:
                critic_loops = 5
            else:
                critic_loops = n_critic
            for _ in range(critic_loops):
                d_loss = self.train_discriminator(x_train, batch_size, using_generator, x_train_iter)
            g_loss = self.train_generator(batch_size)
            print("%d (%d, %d) [D loss: (%.1f)(R %.1f, F %.1f, G %.1f)] [G loss: %.1f]" % (
             epoch, critic_loops, 1, d_loss[0], d_loss[1], d_loss[2], d_loss[3], g_loss))
            self.d_losses.append(d_loss)
            self.g_losses.append(g_loss)
            self.epoch += 1



"""    def train2(self, Y_train, epochs, batch_size, save_interval=50):

        # Rescale -1 to 1
        # np.interp(X_train, (X_train.min(), X_train.max()), (-1, +1))
        # X_train = X_train / 127.5 - 1.
        # Y_train = np.expand_dims(Y_train, axis=3)

        # Adversarial ground truths
        valid = -np.ones((batch_size, 1))
        fake = np.ones((batch_size, 1))

        for epoch in range(epochs):

            # ---------------------
            #  Train Discriminator
            # ---------------------

            # Select a random half of images
            idx = np.random.randint(0, Y_train[0].shape[0], batch_size)
            real_out = [Y_train[0][idx], Y_train[1][idx], Y_train[2][idx]]

            # Sample noise and generate a batch of new music sequences
            noises = [np.random.randint(0, 2, (batch_size, self.n_notes)),
                      np.random.randint(0, 2, (batch_size, self.n_durations)),
                      np.random.randint(0, 2, (batch_size, self.n_velocities))]
            # Get the music sequences predicted by the generator
            gen_seq = self.generator.predict(noises)

            # Train the discriminator (real classified as ones and generated as zeros)
            d_loss_real = self.discriminator.train_on_batch(real_out, valid)
            d_loss_fake = self.discriminator.train_on_batch(gen_seq, fake)
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

            # ---------------------
            #  Train Generator
            # ---------------------

            # Train the generator (wants discriminator to mistake his sequences as real)
            g_loss = self.generator.train_on_batch(noises, real_out)

            # g_loss = self.combined.train_on_batch(noises, valid)

            # Plot the progress
            print("%d [D loss: %f] [G loss: %f]" % (epoch, 1 - d_loss[0], 1 - g_loss[0]))

            # If at save interval => save generated image samples
            if epoch % save_interval == 0:
                a = 0"""




fixed_timesteps= FIXED_NUMBER_OF_BARS*QUANTIZATION

training_data = LoadPianoroll.load_data(fixed_timesteps)
input_shape= training_data[0].shape[2]  # notes= 128
training_data=LoadPianoroll.create_batches(training_data,BATCH_SIZE)
#training_data = load_data()

gan = RNNGAN(input_shape=training_data.element_spec.shape[3],discriminator_lr=0.001
             ,generator_lr=0.001, optimiser='adam', z_dim=latent_dimension
             ,batch_size=BATCH_SIZE, quantization=QUANTIZATION)
gan.generator.summary()
gan.discriminator.summary()
gan.critic_model.summary()
gan.model.summary()

EPOCHS = 6000
PRINT_EVERY_N_BATCHES = 10
gan.epoch = 0


gan.train(
    training_data
    , batch_size = BATCH_SIZE
    , epochs = EPOCHS)
