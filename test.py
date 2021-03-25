import pickle
import numpy as np
import sklearn as skl
import tensorflow as tf
import matplotlib.pyplot as plt


# Set up GPU
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)


# Testing hyperparameters
DATA_DIR = 'data'           # pre-processed data path


# Load the pre-processed data to pickle files
x_train = pickle.load(open(DATA_DIR + '/x_train.p', 'rb'))
y_train = pickle.load(open(DATA_DIR + '/y_train.p', 'rb'))
x_test = pickle.load(open(DATA_DIR + '/x_test.p', 'rb'))
y_test = pickle.load(open(DATA_DIR + '/y_test.p', 'rb'))
print('x_train.shape = {0}, x_test.shape = {1}'.format(x_train.shape, x_test.shape))
print('y_train.shape = {0}, y_test.shape = {1}'.format(y_train.shape, y_test.shape))


# load the model
model = tf.keras.models.load_model('models/model.h5')


# print the model
print(model.summary())


# Evaluate the model
model.evaluate(x_test, y_test, verbose=2)
