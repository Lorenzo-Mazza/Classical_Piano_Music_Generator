import numpy as np
import tensorflow as tf
import keras
from keras.utils import np_utils


def create_dict(sequence):
    """Create Element to Int and Int to Element dictionaries from a list of elements"""
    return (dict((element, number) for number, element in enumerate(sequence)),
            dict((number, element)for number, element in enumerate(sequence)))


def prepare_batches(notes, durations, velocities, lookups, classes, batch_size=32):
    """ Prepare sequences for the embedding """

    note_to_int = lookups[0][0]
    duration_to_int = lookups[1][0]
    velocities_to_int = lookups[2][0]
    note_class, duration_class, velocity_class = classes

    notes_inp = []
    notes_out = []
    durations_inp = []
    durations_out = []
    vel_inp = []
    vel_out = []

    # create input sequences and the corresponding real_rnn_sequences
    for i in range(len(notes) - batch_size):
        notes_sequence_in = notes[i:i + batch_size]
        notes_sequence_out = notes[i + batch_size]
        notes_inp.append([note_to_int[char] for char in notes_sequence_in])
        notes_out.append(note_to_int[notes_sequence_out])

        durations_sequence_in = durations[i:i + batch_size]
        durations_sequence_out = durations[i + batch_size]
        durations_inp.append([duration_to_int[char] for char in durations_sequence_in])
        durations_out.append(duration_to_int[durations_sequence_out])

        vel_sequence_in = velocities[i:i + batch_size]
        vel_sequence_out = velocities[i + batch_size]
        vel_inp.append([velocities_to_int[char] for char in vel_sequence_in])
        vel_out.append(velocities_to_int[vel_sequence_out])


    n_batches = len(notes_inp)

    # reshape the inputs into a format compatible with LSTM layers
    # convert the real_rnn_sequences into a proper format
    inputs = [np.reshape(notes_inp, (n_batches, batch_size)),
              np.reshape(durations_inp, (n_batches, batch_size)),
              np.reshape(vel_inp, (n_batches, batch_size))]

    outputs = [np_utils.to_categorical(notes_out, num_classes=len(note_class)),
               np_utils.to_categorical(durations_out, num_classes=len(duration_class)),
               np_utils.to_categorical(vel_out, num_classes=len(velocity_class))]

    return inputs, outputs
