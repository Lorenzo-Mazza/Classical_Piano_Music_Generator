from functools import partial

import pypianoroll
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.initializers.initializers_v2 import RandomNormal
from keras.optimizer_v2.adam import Adam
from keras.optimizer_v2.rmsprop import RMSprop

import LoadPianoroll
import tensorflow as tf
import numpy as np
from keras.layers import LSTM, Input, Dropout, Dense, Activation, Embedding, Concatenate, Reshape
from keras.layers import Flatten, RepeatVector, Permute, TimeDistributed
from keras.layers import Multiply, Lambda, Softmax
import keras.backend as K
from keras.layers.merge import _Merge
from keras.models import Model
import tensorflow.keras.optimizers as opt
from keras.layers import Input, Dense, Reshape, Flatten, Dropout
from keras.layers import BatchNormalization, Activation, ZeroPadding2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D, Conv1D, Conv2DTranspose, Conv3D
from keras.models import Sequential, Model


# W-GAN that generates fixed length, 4/4 music.
FIXED_NUMBER_OF_BARS= 32
FIXED_NUMBER_OF_QUARTERS= 4*FIXED_NUMBER_OF_BARS
QUANTIZATION = 8
BATCH_SIZE = 32


latent_dimension = 128
physical_devices = tf.config.list_physical_devices('GPU')
for device in physical_devices:
    tf.config.experimental.set_memory_growth(device, True)



class MuseGAN:
    def __init__(self, input_shape, discriminator_lr, generator_lr,
                 optimiser, z_dim, batch_size, quantization):

        self.discriminator_lr = discriminator_lr
        self.generator_lr = generator_lr
        self.optimiser = optimiser
        self.clip_value = 0.01

        self.input_shape = input_shape  # 128
        self.z_dim = z_dim  # size of encoding

        self.n_bars= int(FIXED_NUMBER_OF_QUARTERS/4 ) # Generates only fixed length music
        self.n_steps_per_bar= 4*quantization
        self.weight_init ='he_normal' #RandomNormal(mean=0., stddev=0.02)
        self.batch_size = batch_size

        self.d_losses = []
        self.g_losses = []
        self.epoch = 0


        # Build and compile the critic
        self.critic = self.build_discriminator()
        # Build the generator
        self.generator = self.build_generator()

        self.critic.compile(loss=self.wasserstein,
                            optimizer=optimizer,
                            metrics=['accuracy'])
        # The generator takes noise as input and generated imgs
        z = Input(shape=(self.z_dim,))
        img = self.generator(z)

        # For the combined model we will only train the generator
        self.critic.trainable = False

        # The critic takes generated images as input and determines validity
        valid = self.critic(img)

        # The combined model  (stacked generator and critic)
        self.combined = Model(z, valid)
        self.combined.compile(loss=self.wasserstein,
                              optimizer=optimizer,
                              metrics=['accuracy'])

    @staticmethod
    def wasserstein(y_true, y_pred):
        return K.mean(y_true * y_pred)


    def get_activation(self, activation):
        if activation == 'leaky_relu':
            layer = LeakyReLU(alpha = 0.2)
        else:
            layer = Activation(activation)
        return layer


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
        #x = Embedding(self.input_shape, self.latent_dim)(input_layer)
        x = Dense(1024)(input_layer)
        x = BatchNormalization(momentum=0.8)(x)
        x = Activation('relu')(x)
        x = Reshape([2, 1, 512])(x)
        x = self.conv_t(x, f=128, k=(16, 1), s=(16, 1), a='relu', p='same', bn=True)
        x = self.conv_t(x, f=64, k=(4, 1), s=(4, 1), a='relu', p='same', bn=True)
        x = self.conv_t(x, f=32, k=(1, 2), s=(1, 2), a='relu', p='same', bn=True)
        x = self.conv_t(x, f=self.n_bars, k=(1, 16), s=(1, 16), a='softmax', p='same', bn=False)
        output_layer = Reshape([self.n_bars, self.n_steps_per_bar, self.input_shape,1])(x)
        return Model(input_layer, output_layer)

    def build_discriminator(self):

        critic_input = Input(shape=(self.n_bars,self.n_steps_per_bar,self.input_shape,1), name='critic_input')
        x = critic_input
        x = self.conv(x, f=32, k=(2, 1, 1), s=(1, 1, 1), a='lrelu', p='valid')
        x= (Dropout(0.25))(x)
        x = self.conv(x, f=64, k=(int(self.n_bars/2), 1, 1), s=(1, 1, 1), a='lrelu',p='valid')
        x= (Dropout(0.25))(x)
        x = self.conv(x, f=128, k=(8, 1, 12*7), s=(8, 1, 12*7), a='lrelu', p='same')
        x= (Dropout(0.25))(x)
        x = self.conv(x, f=256, k=(1, 4, 1), s=(1, 2, 1), a='lrelu', p='same')
        x= (Dropout(0.25))(x)
        x = self.conv(x, f=512, k=(1, 3, 1), s=(1, 2, 1), a='lrelu', p='same')
        x= (Dropout(0.25))(x)
        x = Flatten()(x)
        #x = Dense(1024, kernel_initializer=self.weight_init)(x)
        #x = LeakyReLU()(x)
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

    def train(self, X_train, batch_size, epochs, n_critic=5):
        # Load the dataset
        X_train= list(X_train.as_numpy_iterator())
        # Rescale -1 to 1 and fill the incomplete batches
        for i in range(len(X_train)):
            X_train[i]=2 * X_train[i] - 1
            if X_train[i].shape[0]%batch_size!=0:
                fill=-np.ones((batch_size-X_train[i].shape[0]%batch_size,*X_train[i].shape[1:]))
                X_train[i]=np.concatenate((X_train[i],fill),axis=0)
       # X_train=[2 * batch - 1 for batch in X_train]

        # Adversarial ground truths
        valid = -np.ones((batch_size, 1))
        fake = np.ones((batch_size, 1))

        early_stopping=0
        d_loss_best=np.inf
        d_loss_prev= np.inf
        for epoch in range(epochs):

            for _ in range(n_critic):

                # ---------------------
                #  Train Discriminator
                # ---------------------

                # Select a random batch of images
                n_batches= len(X_train)
                idx = np.random.randint(0, n_batches)
                imgs = X_train[idx]
                #imgs= np.squeeze(imgs,axis=0)

                # Sample noise as generator input
                noise = np.random.normal(0, 1, (batch_size, self.z_dim))

                # Generate a batch of new images
                gen_imgs = self.generator.predict(noise)

                # Save previous loss
                if epoch!=0:
                    d_loss_prev= d_loss[0]
                # Train the critic
                d_loss_real = self.critic.train_on_batch(imgs, valid)
                d_loss_fake = self.critic.train_on_batch(gen_imgs, fake)
                d_loss = 0.5 * np.add(d_loss_fake, d_loss_real)
                if np.abs(d_loss_prev)<np.abs(d_loss[0]):
                    early_stopping+=1
                else:
                    early_stopping=0

                # Clip critic weights
                for l in self.critic.layers:
                    weights = l.get_weights()
                    weights = [np.clip(w, -self.clip_value, self.clip_value) for w in weights]
                    l.set_weights(weights)

            # ---------------------
            #  Train Generator
            # ---------------------

            g_loss = self.combined.train_on_batch(noise, valid)

            # Plot the progress
            print("%d [D loss: %f] [G loss: %f]" % (epoch, 1 - d_loss[0], 1 - g_loss[0]))
            if np.abs(d_loss[0]) < np.abs(d_loss_best):
                d_loss_best= d_loss[0]
                self.generator.save_weights('best model')
            #if early_stopping==10:
            #    break
        return d_loss_best


fixed_timesteps= FIXED_NUMBER_OF_QUARTERS * QUANTIZATION

print ("quantization is %d"%QUANTIZATION)
training_data = LoadPianoroll.load_data(fixed_timesteps)
input_shape= training_data[0].shape[2]  # notes= 128
training_data=LoadPianoroll.create_batches(training_data,BATCH_SIZE)
optimizer= RMSprop(learning_rate=0.0005)  # base=0.0005
gan = MuseGAN(input_shape=training_data.element_spec.shape[3], discriminator_lr=0.00005
              , generator_lr=0.00005, optimiser=optimizer, z_dim=latent_dimension
              , batch_size=BATCH_SIZE, quantization=QUANTIZATION)
gan.generator.summary()
gan.critic.summary()

EPOCHS = 6000
PRINT_EVERY_N_BATCHES = 10
gan.epoch = 0

d_loss_best= gan.train(
    training_data
    , batch_size = BATCH_SIZE
    , epochs = EPOCHS)

print("best loss is %f"% d_loss_best)
for counter in range(10):
    pred_noise = np.random.normal(0, 1, (1, gan.z_dim))
    gan.generator.load_weights('best model')
    gen_scores = gan.generator.predict(pred_noise)
    gen_scores = np.squeeze(gen_scores)
    gen_scores= np.reshape(gen_scores,(fixed_timesteps,-1))
    gen_scores=np.where(gen_scores>0,67,0)
    track= pypianoroll.StandardTrack(pianoroll=gen_scores)
    multi= pypianoroll.Multitrack(tracks=[track],resolution=QUANTIZATION)
    pypianoroll.write(path='try%d.mid'% counter,multitrack=multi)

