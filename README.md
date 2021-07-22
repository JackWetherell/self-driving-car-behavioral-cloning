# Behavioural Cloning

[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)

The goals of this project were the following:

* Generate and augment a behavioural cloning dataset by driving in a simulator.
* Build a deep-learning end-to-end driving model that predicts driving actions from camera data.
* Test, train and validate the model using the simulator driving data.
* Apply the model in the simulator, recording a video of the completion of one lap of the track.

![Demo](./demo.gif)

## Requirements

- `numpy`
- `scipy`
- `pandas`
- `sklearn`
- `matplotlib`
- `seaborn`
- `opencv`
- `jupyterlab`
- `tensorflow`

In addition, this project requires the Term 1 Udacity simulator: https://github.com/udacity/self-driving-car-sim

## Usage
To train the model:
1. Use the simulator to drive vehicle in manual mode and record data.
2. `python preproc.py`
3. `python model.py`

To run the trained model:
1. Download the relevant release of the Udacity simulator for your platform.
2. Run the driving model: `python drive.py models/model.h5 run1`.
3. Run the simulator.
