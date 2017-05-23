import configparser
from moviepy.editor import VideoFileClip
import glob
import os
import helper
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pickle
import numpy as np
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.externals import joblib
import time
from ast import literal_eval

# Parameters
config = configparser.ConfigParser()
config.read('config.ini')
car_classify_data_dir = config['Classifier']['car_classify_data_dir']
features_filename = config['FeaturesGenerator']['features_filename']
model_filename = config['Classifier']['model_filename']


if __name__ == "__main__":

    cars = []
    notcars = []

    # Feature Generation
    t10 = time.time()
    if config['Project'].getboolean('regenerate_features'):
        car_images = helper.list_files(os.path.join(car_classify_data_dir, 'vehicles'))
        notcar_images = helper.list_files(os.path.join(car_classify_data_dir, 'non-vehicles'))

        car_features = helper.extract_features(car_images,
                                               colorspace_conv=config['FeaturesGenerator']['colorspace_conv'],
                                               spatial_size=literal_eval(config['FeaturesGenerator']['spatial_size']),
                                               hist_bins=config['FeaturesGenerator'].getint('hist_bins'),
                                               orient=config['FeaturesGenerator'].getint('orient'),
                                               pix_per_cell=config['FeaturesGenerator'].getint('pix_per_cell'),
                                               cell_per_block=config['FeaturesGenerator'].getint('cell_per_block'),
                                               hog_channels=config['FeaturesGenerator']['hog_channels'],
                                               spatial_feat=config['FeaturesGenerator'].getboolean('spatial_feat'),
                                               hist_feat=config['FeaturesGenerator'].getboolean('hist_feat'),
                                               hog_feat=config['FeaturesGenerator'].getboolean('hog_feat'))

        notcar_features = helper.extract_features(notcar_images,
                                                  colorspace_conv=config['FeaturesGenerator']['colorspace_conv'],
                                                  spatial_size=literal_eval(config['FeaturesGenerator']['spatial_size']),
                                                  hist_bins=config['FeaturesGenerator'].getint('hist_bins'),
                                                  orient=config['FeaturesGenerator'].getint('orient'),
                                                  pix_per_cell=config['FeaturesGenerator'].getint('pix_per_cell'),
                                                  cell_per_block=config['FeaturesGenerator'].getint('cell_per_block'),
                                                  hog_channels=config['FeaturesGenerator']['hog_channels'],
                                                  spatial_feat=config['FeaturesGenerator'].getboolean('spatial_feat'),
                                                  hist_feat=config['FeaturesGenerator'].getboolean('hist_feat'),
                                                  hog_feat=config['FeaturesGenerator'].getboolean('hog_feat'))

        features = {}
        features["car"] = car_features
        features["notcar"] = notcar_features
        features["feature_generator_parameters"] = {
            'spatial_feat': config['FeaturesGenerator']['spatial_feat'],
            'hist_feat': config['FeaturesGenerator']['hist_feat'],
            'hog_feat': config['FeaturesGenerator']['hog_feat'],
            'colorspace_conv': config['FeaturesGenerator']['colorspace_conv'],
            'spatial_size': config['FeaturesGenerator']['spatial_size'],
            'hist_bins': config['FeaturesGenerator']['hist_bins'],
            'orient': config['FeaturesGenerator']['orient'],
            'pix_per_cell': config['FeaturesGenerator']['pix_per_cell'],
            'cell_per_block': config['FeaturesGenerator']['cell_per_block'],
            'hog_channels': config['FeaturesGenerator']['hog_channels']}

        joblib.dump(features, open(os.path.join(car_classify_data_dir, features_filename), 'wb'), protocol=pickle.HIGHEST_PROTOCOL)

    else:
        # read the car features from the pickled dict
        features = joblib.load(open(os.path.join(car_classify_data_dir, features_filename), 'rb'))

        car_features = features['car']
        notcar_features = features['notcar']

        feature_generator_parameters = features['feature_generator_parameters']
        feature_generator_parameters = features['feature_generator_parameters']
        orient = int(feature_generator_parameters["orient"])
        pix_per_cell = int(feature_generator_parameters["pix_per_cell"])
        cell_per_block = int(feature_generator_parameters["cell_per_block"])
        spatial_size = literal_eval(feature_generator_parameters["spatial_size"])
        hist_bins = int(feature_generator_parameters["hist_bins"])
        spatial_feat = bool(feature_generator_parameters['spatial_feat'])
        hist_feat = bool(feature_generator_parameters['hist_feat'])
        hog_feat = bool(feature_generator_parameters['hog_feat'])

    t20 = time.time()
    print('Time to extract features = ', round(t20-t10, 2))

    # Training
    if config['Project'].getboolean('retrain'):

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

        # print('Feature vector length:', len(X_train[0]))

        # Use a linear SVC
        svc = LinearSVC()

        # Check the training time for the SVC
        t30 = time.time()

        svc.fit(X_train, y_train)
        model = {'svc': svc, 'X_scaler': X_scaler}

        t40 = time.time()
        print(round(t40 - t30, 2), 'Seconds to train SVC...')

        # save the model to disk
        joblib.dump(model, open(os.path.join(car_classify_data_dir, model_filename), 'wb'), protocol=pickle.HIGHEST_PROTOCOL)

        # Check the score of the SVC
        print('Test Accuracy of SVC = ', round(svc.score(X_test, y_test), 4))

    else:
        # load the model from disk
        model = joblib.load(open(os.path.join(car_classify_data_dir, model_filename), 'rb'))

        svc = model["svc"]
        X_scaler = model["X_scaler"]

    # Main processing functions - for images and video
    if config.get('Project', 'media_type') == 'images':

        test_directory = './test_images'
        images = glob.glob(os.path.join(test_directory, 'test?.jpg'))

        h, axs = plt.subplots(3, 2, figsize=(24, 9))
        h.tight_layout()
        axs = axs.ravel()

        for fname, i in zip(images, range(len(images))):
            image = mpimg.imread(fname)
            detected_vehicles_img = helper.process_image(image, config['FeaturesGenerator']['colorspace_conv'],
                                            literal_eval(config['Test']['y_start_stop']),
                                            float(config['Test']['scale']), svc, X_scaler, orient, pix_per_cell,
                                            cell_per_block, spatial_size, hist_bins,
                                            config['FeaturesGenerator']['hog_channels'], spatial_feat,
                                            hist_feat, hog_feat)
            axs[i].imshow(detected_vehicles_img)
            axs[i].set_title(fname)
            plt.savefig(os.path.join(test_directory, 'detection_output.jpg'), bbox_inches='tight')
        plt.show()

    elif config['Project']['media_type'] == 'video':

        videos_filenames = ['test_video.mp4', './project_video.mp4']
        t100 = time.time()

        for fname in videos_filenames:
            # read the project video
            project_video_clip = VideoFileClip(fname)
            project_video_output_fname = 'output_' + os.path.basename(fname)

            output_clip = project_video_clip.fx(helper.process_video, config['FeaturesGenerator']['colorspace_conv'],
                                            literal_eval(config['Test']['y_start_stop']),
                                            float(config['Test']['scale']), svc, X_scaler, orient, pix_per_cell,
                                            cell_per_block, spatial_size, hist_bins,
                                            config['FeaturesGenerator']['hog_channels'], spatial_feat,
                                            hist_feat, hog_feat)

            output_clip.write_videofile(project_video_output_fname, audio=False)

        t110 = time.time()
        print(round(t110 - t100, 2), 'Seconds to process the videos ...')


