import pickle
import numpy as np
import pandas as pd
import cv2 as cv
import matplotlib.pyplot as plt


# Pre-processing hyperparameters
DATA_DIR = 'data'             # Raw simulator data path
IMAGE_DIM = (160,320)         # Dimentions of images
N_TRAIN = 3000                # Number of training samples (this is multipied by 3 as there are three cameras)
N_TOTAL = 3500                # Total size of dataset to use (this is multipied by 3 as there are three cameras)
CORRECTION = 0.25             # Steering correction for left and right images


# Load in the x data (center images)
df = pd.read_csv(DATA_DIR + '/driving_log.csv')
print('number of raw frames = {}'.format(df.shape[0]))
df = df.loc[:N_TOTAL-1]
x = np.zeros(shape=(df.shape[0], IMAGE_DIM[0], IMAGE_DIM[1], 3), dtype=float)
for i, file_name in enumerate(df.astype('str').iloc[:,0]):
    x[i,:,:,:] =  cv.cvtColor(cv.imread(file_name.replace(' ', '')), cv.COLOR_BGR2RGB)
x = x[:N_TOTAL,:,:,:]
x_train, x_test = x[:N_TRAIN,:,:,:], x[N_TRAIN:,:,:,:]
del x


# Load in the x data (left images)
x = np.zeros(shape=(df.shape[0], IMAGE_DIM[0], IMAGE_DIM[1], 3), dtype=float)
for i, file_name in enumerate(df.astype('str').iloc[:,1]):
    x[i,:,:,:] =  cv.cvtColor(cv.imread(file_name.replace(' ', '')), cv.COLOR_BGR2RGB)
x = x[:N_TOTAL,:,:,:]
x_train_left, x_test_left = x[:N_TRAIN,:,:,:], x[N_TRAIN:,:,:,:]
x_train = np.concatenate((x_train, x_train_left), axis=0)
x_test = np.concatenate((x_test, x_test_left), axis=0)
del x_train_left, x_test_left, x


# Load in the x data (right images)
x = np.zeros(shape=(df.shape[0], IMAGE_DIM[0], IMAGE_DIM[1], 3), dtype=float)
for i, file_name in enumerate(df.astype('str').iloc[:,2]):
    x[i,:,:,:] =  cv.cvtColor(cv.imread(file_name.replace(' ', '')), cv.COLOR_BGR2RGB)
x = x[:N_TOTAL,:,:,:]
x_train_right, x_test_right = x[:N_TRAIN,:,:,:], x[N_TRAIN:,:,:,:]
x_train = np.concatenate((x_train, x_train_right), axis=0)
x_test = np.concatenate((x_test, x_test_right), axis=0)
del x_train_right, x_test_right, x


# Load in the y data (car controls) (for centered camera)
y = df.iloc[:, [3,4,5]].to_numpy()
y = y[:N_TOTAL,:]
y_train, y_test = y[:N_TRAIN,:], y[N_TRAIN:,:]


# Load in the y data (car controls) (for left camera)
y = df.iloc[:, [3,4,5]].to_numpy() + CORRECTION
y = y[:N_TOTAL,:]
y_train_left, y_test_left = y[:N_TRAIN,:], y[N_TRAIN:,:]
y_train = np.concatenate((y_train, y_train_left), axis=0)
y_test = np.concatenate((y_test, y_test_left), axis=0)
del y_train_left, y_test_left, y


# Load in the y data (car controls) (for right camera)
y = df.iloc[:, [3,4,5]].to_numpy() - CORRECTION
y = y[:N_TOTAL,:]
y_train_right, y_test_right = y[:N_TRAIN,:], y[N_TRAIN:,:]
y_train = np.concatenate((y_train, y_train_right), axis=0)
y_test = np.concatenate((y_test, y_test_right), axis=0)
del y_train_right, y_test_right, y


# Print final shapes of dataset
print('x_train.shape = {0}, x_test.shape = {1}'.format(x_train.shape, x_test.shape))
print('y_train.shape = {0}, y_test.shape = {1}'.format(y_train.shape, y_test.shape))


# Save the pre-processed data to pickle files
pickle.dump(x_train, open(DATA_DIR + '/x_train.p', 'wb'))
pickle.dump(y_train, open(DATA_DIR + '/y_train.p', 'wb'))
pickle.dump(x_test, open(DATA_DIR + '/x_test.p', 'wb'))
pickle.dump(y_test, open(DATA_DIR + '/y_test.p', 'wb'))


# Plot some training data
fig, axs = plt.subplots(5, 5)
np.random.seed(0)
indices = np.random.randint(0, 3*N_TRAIN, size=25)
k = 0
for i in range(axs.shape[0]):
    for j in range(axs.shape[1]):
        axs[i,j].imshow(x_train[indices[k], :, :, :].astype(int), interpolation='hamming')
        axs[i,j].set_title('s = {0:.2f}, t = {1:.2f}, b = {2:.2f}'.format(y_train[indices[k],0], y_train[indices[k],1], y_train[indices[k],2]))
        axs[i,j].set_yticklabels([])
        axs[i,j].set_xticklabels([])
        axs[i,j].set_yticks([])
        axs[i,j].set_xticks([])
        k = k + 1
plt.savefig('results/preproc.pdf')
