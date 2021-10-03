from music21 import *
import os
import pandas as pd
import numpy as np
import tensorflow as tf
import keras
from keras.utils import np_utils
from data import *


BATCH_SIZE = 32
# model params
embedding_size = 100
rnn_units = 256
use_attention = True
training_data= load_data()
train_inputs,train_outputs= process_data(training_data)



print("end")
