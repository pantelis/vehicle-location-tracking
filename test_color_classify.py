import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import cv2
import glob
import time
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import helper


import os


car_classify_data_dir = '../datasets/udacity-vehicle-tracking'


# Read in car and non-car images
cars = []
notcars = []
cars = helper.list_files(os.path.join(car_classify_data_dir, 'vehicles'))
notcars = helper.list_files(os.path.join(car_classify_data_dir, 'non-vehicles'))

# performs under different binning scenarios
spatial = 16
histbin = 32

car_features = helper.extract_features(cars, convert='RGB2YCrCb', spatial_size=(spatial, spatial),
                        hist_bins=histbin, orient=9,
                        pix_per_cell=8, cell_per_block=2, hog_channels=[0],
                        spatial_feat=True, hist_feat=True, hog_feat=False)

notcar_features = helper.extract_features(notcars, convert='RGB2YCrCb', spatial_size=(spatial, spatial),
                        hist_bins=histbin, orient=9,
                        pix_per_cell=8, cell_per_block=2, hog_channels=[0],
                        spatial_feat=True, hist_feat=True, hog_feat=False)

# Create an array stack of feature vectors
X = np.vstack((car_features, notcar_features)).astype(np.float64)

# Fit a per-column scaler
X_scaler = StandardScaler().fit(X)

# Apply the scaler to X
scaled_X = X_scaler.transform(X)

# Define the labels vector
y = np.hstack((np.ones(len(car_features)), np.zeros(len(notcar_features))))

# Split up data into randomized training and test sets
rand_state = np.random.randint(0, 100)
X_train, X_test, y_train, y_test = train_test_split(
    scaled_X, y, test_size=0.2, random_state=rand_state)

print('Using spatial binning of:',spatial,
    'and', histbin,'histogram bins')

print('Feature vector length:', len(X_train[0]))

# Use a linear SVC
svc = LinearSVC()

# Check the training time for the SVC
t = time.time()
svc.fit(X_train, y_train)
t2 = time.time()
print(round(t2-t, 2), 'Seconds to train SVC...')

# Check the score of the SVC
print('Test Accuracy of SVC = ', round(svc.score(X_test, y_test), 4))