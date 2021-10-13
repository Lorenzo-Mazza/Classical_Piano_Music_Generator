from keras.callbacks import EarlyStopping
import os
import pandas as pd
import numpy as np

import tensorflow as tf
from keras.layers import LSTM, Input, Dropout, Dense, Activation, Embedding, Concatenate, Reshape
from keras.layers import Flatten, RepeatVector, Permute
from keras.layers import Multiply, Lambda, Softmax
import keras.backend as K
import tensorflow.keras.optimizers as opt
from keras.layers import Input, Dense, Reshape, Flatten, Dropout
from keras.layers import BatchNormalization, Activation
from keras.models import Sequential, Model
import pypianoroll



physical_devices = tf.config.list_physical_devices('GPU')
for device in physical_devices:
    tf.config.experimental.set_memory_growth(device, True)



def create_rnn(n_notes,seq_length, embed_size, rnn_units, use_attention=False):
    """ create the structure of the neural network """

    notes_in = Input(shape=(None,n_notes))

    #x = Embedding(input_dim=1, output_dim=embed_size)(notes_in)

    x = LSTM(rnn_units, input_shape=(None, n_notes), return_sequences=True)(notes_in)
    # x = Dropout(0.2)(x)

    if use_attention:

        x = LSTM(rnn_units, return_sequences=True)(x)
        # x = Dropout(0.2)(x)

        e = Dense(1, activation='tanh')(x)
        e = Reshape([-1])(e)
        alpha = Activation('softmax')(e)

        alpha_repeated = Permute([2, 1])(RepeatVector(rnn_units)(alpha))

        c = Multiply()([x, alpha_repeated])
        c = Lambda(lambda xin: K.sum(xin, axis=1), output_shape=(rnn_units,))(c)

    else:
        c = LSTM(rnn_units)(x)
        # c = Dropout(0.2)(c)

    notes_out = Dense(n_notes, activation='softmax', name='pitch')(c)
    model = Model([notes_in], [notes_out])
    if use_attention:
        att_model = Model([notes_in], alpha)
    else:
        att_model = None

    opti = opt.RMSprop(learning_rate=0.001)
    model.compile(loss=['categorical_crossentropy'], optimizer=opti)
    return model, att_model


def load_data( path='maestro-v2.0.0'):
    os.chdir(path)
    infos = pd.read_csv("maestro-v2.0.0.csv")
    songs=[]
    counter=0
    for name, author in zip(infos['midi_filename'], infos['canonical_composer']):
        pr= pypianoroll.read(name)
        pr.set_resolution(QUANTIZATION)
        piano_roll = pr.tracks[0].pianoroll[1000:3000, :]
        piano_roll= np.where(piano_roll>0,1,0)
        songs.append(piano_roll)
        counter+=1

        if counter==50:
             break
    return songs

def create_sequences(songs,seq_length):
    input_sequence=[]
    output_sequence=[]
    x_size= songs[0].shape[1]
    for s in songs:
        s = np.insert(s,0,np.zeros((seq_length,x_size)),axis=0)
        for idx in range(len(s) - seq_length):
            sequence_in = s[idx:idx+seq_length,:]
            sequence_out= s[idx+seq_length,:]
            #input_sequence.append([ts for ts in sequence_in])
            input_sequence.append(sequence_in)
            output_sequence.append(sequence_out)

    n_batches = len(input_sequence)
    input_sequence = np.reshape(input_sequence, (n_batches, seq_length,x_size))
    output_sequence = np.reshape(output_sequence, (n_batches,x_size))

    return input_sequence,output_sequence

QUANTIZATION = 4
# generator_model params
embedding_size = 100
rnn_units = 256
sequence_length=64
use_attention = True

training_data = load_data()
n_notes= training_data[0].shape[1]  # notes= 128
x_tr,y_tr=create_sequences(training_data,sequence_length)
model, att_model = create_rnn(n_notes,sequence_length, embedding_size, rnn_units, use_attention)
model.summary()
early_stopping = EarlyStopping(
    monitor='loss',
    restore_best_weights=True,
    patience=10
)
model.fit(x_tr, y_tr
          , epochs=1, batch_size=32
          , validation_split=0.2
          , callbacks=early_stopping
          )

max_extra_notes = 512
max_seq_len = 64

prediction_output = []
prediction_input = []
prediction_input.extend(np.zeros((sequence_length,n_notes)))
overall_preds = []
att_matrix = np.zeros(shape=(max_extra_notes + sequence_length, max_extra_notes))

for note_index in range(max_extra_notes):

    prediction_input = np.array(prediction_input)
    prediction_input = np.expand_dims(prediction_input,axis=0)
    pred = model.predict(prediction_input, verbose=0)
    if use_attention:
        att_prediction = att_model.predict(prediction_input, verbose=0)[0]
        att_matrix[(note_index - len(att_prediction) + sequence_length):
                   (note_index + sequence_length),note_index] = att_prediction

    pred=np.where(pred>=np.max(pred),1,0)
    prediction_output.append(pred)
    prediction_input= np.squeeze(prediction_input, axis=0)
    prediction_input= np.append(prediction_input,pred,axis=0)
    if len(prediction_input) > max_seq_len:
        prediction_input = prediction_input[1:,:]


print('Generated sequence of {} notes'.format(len(prediction_output)))
gen_scores = np.squeeze(np.array(prediction_output))
#gen_scores= np.reshape(gen_scores,(gen_scores.shape[0]*gen_scores.shape[1],-1))
#THRESHOLD*=np.max(gen_scores)
#gen_scores=np.where(gen_scores>THRESHOLD,70,0)
track= pypianoroll.StandardTrack(pianoroll=gen_scores)
multi= pypianoroll.Multitrack(tracks=[track])
#multi.set_resolution(QUANTIZATION)
pypianoroll.write(path='rnn_try.mid',multitrack=multi)

