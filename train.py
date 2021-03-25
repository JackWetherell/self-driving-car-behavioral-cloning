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


# Training hyperparameters
DATA_DIR = 'data'              # pre-processed data path
BATCH_SIZE = 128               # Batch size
EPOCHS = 5                     # Number of epochs
LEARNING_RATE = 0.0005         # Learning rate for adam optimizer


# Load the pre-processed data to pickle files
x_train = pickle.load(open(DATA_DIR + '/x_train.p', 'rb'))
y_train = pickle.load(open(DATA_DIR + '/y_train.p', 'rb'))
x_test = pickle.load(open(DATA_DIR + '/x_test.p', 'rb'))
y_test = pickle.load(open(DATA_DIR + '/y_test.p', 'rb'))
print('x_train.shape = {0}, x_test.shape = {1}'.format(x_train.shape, x_test.shape))
print('y_train.shape = {0}, y_test.shape = {1}'.format(y_train.shape, y_test.shape))


# Build the model
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Lambda(lambda x: x / 255.0 - 0.5, input_shape=(x_train.shape[1], x_train.shape[2], x_train.shape[3])))
model.add(tf.keras.layers.Cropping2D(cropping=((50,20), (0,0))))
model.add(tf.keras.layers.Conv2D(filters=24, kernel_size=(5, 5), strides=(2,2), activation="relu"))
model.add(tf.keras.layers.Conv2D(filters=36, kernel_size=(5, 5), strides=(2,2), activation="relu"))
model.add(tf.keras.layers.Conv2D(filters=48, kernel_size=(5, 5), strides=(2,2), activation="relu"))
model.add(tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), activation="relu"))
model.add(tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), activation="relu"))
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(units=1164, activation="relu"))
model.add(tf.keras.layers.Dense(units=100, activation="relu"))
model.add(tf.keras.layers.Dense(units=50, activation="relu"))
model.add(tf.keras.layers.Dense(units=y_train.shape[1], activation="linear"))


# Compile the model
model.compile(loss='mse', optimizer=tf.keras.optimizers.Adam(lr=LEARNING_RATE))


# Train the model
history = model.fit(x_train, y_train, validation_split=0.2, batch_size=BATCH_SIZE, epochs=EPOCHS, shuffle=True)


# Save the model
model.save('models/model.h5')


# Plot the training
plt.plot(history.history['loss'], label='training')
plt.plot(history.history['val_loss'], label='validation')
plt.ylabel('MSE loss')
plt.xlabel('epoch')
plt.legend()
plt.savefig('results/train.pdf')
