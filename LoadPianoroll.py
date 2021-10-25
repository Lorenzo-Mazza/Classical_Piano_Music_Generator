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

            #augmenting the piece, taking 6 different transpositions, baseline= no transposition
            for semitone in range(0,6):
                if semitone==0:
                    piano_roll = pr.tracks[0].transpose(-2).pianoroll[0:max_timesteps, :]
                else:
                    piano_roll= pr.tracks[0].transpose(1).pianoroll[0:max_timesteps, :]
                piano_roll = np.where(piano_roll > 0, 1, 0)
                piano_roll= np.reshape(piano_roll,(-1,4*QUANTIZATION,128))
                piano_roll= np.expand_dims(piano_roll,axis=3)
                songs.append(piano_roll)
                counter+=1
                print("song %d added"% counter)


        #if counter==64:
        #     break
    return songs

def create_batches(data,batch_size):
    dataset= (tf.data.Dataset.from_tensor_slices(data)
              .shuffle(batch_size * 100000)
              .batch(batch_size))
    return dataset
