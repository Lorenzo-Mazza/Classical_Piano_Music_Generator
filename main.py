from keras.callbacks import ModelCheckpoint, EarlyStopping
from music21 import *
import time
import os
import pandas as pd
import numpy as np
import tensorflow as tf
import keras
from keras.utils import np_utils

import Network
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
from keras.layers.convolutional import UpSampling2D, Conv2D, Conv1D
from keras.models import Sequential, Model


def sample_with_temp(preds, temperature):

    if temperature == 0:
        return np.argmax(preds)
    else:
        preds = np.log(preds) / temperature
        exp_preds = np.exp(preds)
        preds = exp_preds / np.sum(exp_preds)
        return np.random.choice(len(preds), p=preds)


def create_network(n_notes, n_durations,n_velocities, embed_size=100, rnn_units=256, use_attention=False):
    """ create the structure of the neural network """

    notes_in = Input(shape=(None,))
    durations_in = Input(shape=(None,))
    velocities_in = Input(shape=(None,))

    x1 = Embedding(n_notes, embed_size)(notes_in)
    x2 = Embedding(n_durations, embed_size)(durations_in)
    x3 = Embedding(n_velocities, embed_size)(velocities_in)

    x = Concatenate()([x1, x2, x3])

    x = LSTM(rnn_units, return_sequences=True)(x)
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
    durations_out = Dense(n_durations, activation='softmax', name='duration')(c)
    vel_out = Dense(n_velocities, activation='softmax', name='velocity')(c)
    model = Model([notes_in, durations_in,velocities_in], [notes_out, durations_out,vel_out])
    if use_attention:
        att_model = Model([notes_in, durations_in,velocities_in], alpha)
    else:
        att_model = None

    opti = opt.RMSprop(learning_rate=0.001)
    model.compile(loss=['categorical_crossentropy', 'categorical_crossentropy','categorical_crossentropy'], optimizer=opti)

    return model, att_model


BATCH_SIZE = 32
# model params
embedding_size = 100
rnn_units = 256
use_attention = True
training_data = load_data()
train_inputs, train_outputs = process_data(training_data)
output_shape= (np.shape(train_outputs[0])[1], np.shape(train_outputs[1])[1],np.shape(train_outputs[2])[1])
n_notes=train_outputs[0].shape[1]
n_durations= train_outputs[1].shape[1]
n_velocities= train_outputs[2].shape[1]
model, att_model = create_network(n_notes, n_durations,n_velocities, embedding_size, rnn_units, use_attention)
model.summary()
early_stopping = EarlyStopping(
    monitor='loss',
    restore_best_weights=True,
    patience=10
)
model.fit(train_inputs, train_outputs
          , epochs=1, batch_size=32
          , validation_split = 0.2
          , callbacks=early_stopping
          , shuffle=True
          )


# prediction params
notes_temp=0.5
duration_temp = 0.5
velocity_temp= 0.5
vel_temp=0.5
max_extra_notes = 500
max_seq_len = 32
seq_len = 32
notes = ['START']
durations = [0]
velocities=[0]
if seq_len is not None:
    notes = ['START'] * (seq_len - len(notes)) + notes
    durations = [0] * (seq_len - len(durations)) + durations
    velocities = [0]* (seq_len - len(velocities)) + velocities

sequence_length = len(notes)
prediction_output = []
notes_input_sequence = []
durations_input_sequence = []
velocities_input_sequence=[]

overall_preds = []
note_to_int, int_to_note= preprocessing.create_dict(np.unique(training_data[0]))
duration_to_int, int_to_duration= preprocessing.create_dict(np.unique(training_data[1]))
vel_to_int, int_to_vel= preprocessing.create_dict(np.unique(training_data[2]))

for n, d, v in zip(notes, durations,velocities):
    note_int = note_to_int[n]
    duration_int = duration_to_int[d]
    vel_int= vel_to_int[v]
    notes_input_sequence.append(note_int)
    durations_input_sequence.append(duration_int)
    velocities_input_sequence.append(vel_int)
    prediction_output.append([n, d, v])

    if n != 'START':
        midi_note = note.Note(n)

        new_note = np.zeros(128)
        new_note[midi_note.pitch.midi] = 1
        overall_preds.append(new_note)

att_matrix = np.zeros(shape=(max_extra_notes + sequence_length, max_extra_notes))

for note_index in range(max_extra_notes):

    prediction_input = [
        np.array([notes_input_sequence])
        , np.array([durations_input_sequence])
        , np.array([velocities_input_sequence])
    ]

    notes_prediction, durations_prediction, velocities_prediction = model.predict(prediction_input, verbose=0)
    if use_attention:
        att_prediction = att_model.predict(prediction_input, verbose=0)[0]
        att_matrix[(note_index - len(att_prediction) + sequence_length):(note_index + sequence_length),
        note_index] = att_prediction

    new_note = np.zeros(128)

    for idx, n_i in enumerate(notes_prediction[0]):
        try:
            note_name = int_to_note[idx]
            midi_note = note.Note(note_name)
            new_note[midi_note.pitch.midi] = n_i
        except:
            pass

    overall_preds.append(new_note)

    i1 = sample_with_temp(notes_prediction[0], notes_temp)
    i2 = sample_with_temp(durations_prediction[0], duration_temp)
    i3 = sample_with_temp(velocities_prediction[0], velocity_temp)
    note_result = int_to_note[i1]
    duration_result = int_to_duration[i2]
    velocity_result= int_to_vel[i3]
    prediction_output.append([note_result, duration_result, velocity_result])

    notes_input_sequence.append(i1)
    durations_input_sequence.append(i2)
    velocities_input_sequence.append(i3)
    if len(notes_input_sequence) > max_seq_len:
        notes_input_sequence = notes_input_sequence[1:]
        durations_input_sequence = durations_input_sequence[1:]
        velocities_input_sequence = velocities_input_sequence[1:]
    #     print(note_result)
    #     print(duration_result)

    if note_result == 'START':
        break

overall_preds = np.transpose(np.array(overall_preds))
print('Generated sequence of {} notes'.format(len(prediction_output)))


output_folder = os.getcwd()

midi_stream = stream.Stream()

# create note and chord objects based on the values generated by the model
for pattern in prediction_output:
    note_pattern, duration_pattern, velocity_pattern = pattern
    # pattern is a chord
    if ('.' in note_pattern):
        notes_in_chord = note_pattern.split('.')
        chord_notes = []
        for current_note in notes_in_chord:
            new_note = note.Note(current_note)
            new_note.duration = duration.Duration(duration_pattern)
            new_note.volume.velocity = velocity_pattern
            new_note.storedInstrument = instrument.Piano()
            chord_notes.append(new_note)
        new_chord = chord.Chord(chord_notes)
        midi_stream.append(new_chord)
    elif note_pattern == 'rest':
    # pattern is a rest
        new_note = note.Rest()
        new_note.duration = duration.Duration(duration_pattern)
        new_note.storedInstrument = instrument.Piano()
        midi_stream.append(new_note)
    elif note_pattern != 'START':
    # pattern is a note
        new_note = note.Note(note_pattern)
        new_note.volume.velocity = velocity_pattern
        new_note.duration = duration.Duration(duration_pattern)
        new_note.storedInstrument = instrument.Piano()
        midi_stream.append(new_note)

#midi_stream = midi_stream.chordify()
timestr = time.strftime("%Y%m%d-%H%M%S")
midi_stream.write('midi', fp=os.path.join(output_folder, 'output-' + timestr + '.mid'))




#gan = Network.DCGAN(train_outputs)


#gan.train(train_outputs,epochs=200,batch_size=32)

#print("end")


"""gan.fit(train_inputs, train_outputs,
          epochs=2000000, batch_size=32,
          validation_split=0.2,
          callbacks=early_stopping,
          shuffle=True
          )
"""
