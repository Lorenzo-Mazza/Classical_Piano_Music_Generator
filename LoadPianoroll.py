import os
import string

import pandas as pd
import tensorflow as tf

import preprocessing
import numpy as np
import pypianoroll

QUANTIZATION = 8

def load_data(max_timesteps, path='maestro-v2.0.0'):
    os.chdir(path)
    infos = pd.read_csv("maestro-v2.0.0.csv")
    songs=[]
    counter=0
    for name, author in zip(infos['midi_filename'], infos['canonical_composer']):
        pr= pypianoroll.read(name)
        pr.set_resolution(QUANTIZATION)
        if pr.tracks[0].pianoroll.shape[0]>max_timesteps:
            piano_roll= pr.tracks[0].pianoroll[0:max_timesteps, :]
            piano_roll= np.where(piano_roll>0,1,0)
            for semitone in range(-6,6):
                transposed= piano_roll.transpose(semitone)
                transposed= np.reshape(transposed,(-1,4*QUANTIZATION,128))
                transposed= np.expand_dims(transposed,axis=3)
                songs.append(transposed)
            counter+=1

        #if counter==64:
        #     break
    return songs

def create_batches(data,batch_size):
    dataset= (tf.data.Dataset.from_tensor_slices(data)
              .shuffle(batch_size * 100000)
              .batch(batch_size))
    return dataset
