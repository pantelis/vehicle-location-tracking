import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
from ast import literal_eval
import cv2
import helper
import configparser
import os
from sklearn.externals import joblib


# Parameters
config = configparser.ConfigParser()
config.read('config.ini')
car_classify_data_dir = config['Classifier']['car_classify_data_dir']
model_filename = config['Classifier']['model_filename']
features_filename = config['FeaturesGenerator']['features_filename']

# read the car features from the pickled dict
features = joblib.load(open(os.path.join(car_classify_data_dir, features_filename), 'rb'))
feature_generator_parameters = features['feature_generator_parameters']


orient = int(feature_generator_parameters["orient"])
pix_per_cell = int(feature_generator_parameters["pix_per_cell"])
cell_per_block = int(feature_generator_parameters["cell_per_block"])
spatial_size = literal_eval(feature_generator_parameters["spatial_size"])
hist_bins = int(feature_generator_parameters["hist_bins"])
spatial_feat = bool(feature_generator_parameters['spatial_feat'])
hist_feat = bool(feature_generator_parameters['hist_feat'])
hog_feat = bool(feature_generator_parameters['hog_feat'])


# read the SVM Classifier from  the model and the
model = joblib.load(open(os.path.join(car_classify_data_dir, model_filename), 'rb'))
svc = model["svc"]
X_scaler = model["X_scaler"]
img = mpimg.imread('./test_images/test1.jpg')


# Define a single function that can extract features using hog sub-sampling and make predictions
def find_cars(img, ystart, ystop, scale, svc, X_scaler, orient, pix_per_cell, cell_per_block, spatial_size, hist_bins, spatial_feat, hist_feat, hog_feat):
    '''
    The find_cars only has to extract hog features once and then can be sub-sampled to get all of its overlaying windows. Each window is defined by a scaling factor where a scale of 1 would result in a window that's 8 x 8 cells then the overlap of each window is in terms of the cell distance. This means that a cells_per_step = 2 would result in a search window overlap of 75%. Its possible to run this same function multiple times for different scale values to generate multiple-scaled search windows.
    
    :param img: 
    :param ystart: 
    :param ystop: 
    :param scale: 
    :param svc: 
    :param X_scaler: 
    :param orient: 
    :param pix_per_cell: 
    :param cell_per_block: 
    :param spatial_size: 
    :param hist_bins: 
    :return: 
    
    '''
    draw_img = np.copy(img)
    img = img.astype(np.float32) / 255

    img_tosearch = img[ystart:ystop, :, :]
    ctrans_tosearch = helper.convert_color(img_tosearch, conv='RGB2YCrCb')

    if scale != 1:
        imshape = ctrans_tosearch.shape
        ctrans_tosearch = cv2.resize(ctrans_tosearch, (np.int(imshape[1] / scale), np.int(imshape[0] / scale)))

    ch1 = ctrans_tosearch[:, :, 0]
    ch2 = ctrans_tosearch[:, :, 1]
    ch3 = ctrans_tosearch[:, :, 2]

    # Define blocks and steps as above
    nxblocks = (ch1.shape[1] // pix_per_cell) - cell_per_block + 1
    nyblocks = (ch1.shape[0] // pix_per_cell) - cell_per_block + 1
    nfeat_per_block = orient * cell_per_block * cell_per_block

    # 64 was the orginal sampling rate, with 8 cells and 8 pix per cell
    window = 64
    nblocks_per_window = (window // pix_per_cell) - cell_per_block + 1
    cells_per_step = 2  # Instead of overlap, define how many cells to step
    nxsteps = (nxblocks - nblocks_per_window) // cells_per_step
    nysteps = (nyblocks - nblocks_per_window) // cells_per_step

    # Compute individual channel HOG features for the entire image
    hog1 = helper.get_hog_features(ch1, orient, pix_per_cell, cell_per_block, feature_vec=False)
    hog2 = helper.get_hog_features(ch2, orient, pix_per_cell, cell_per_block, feature_vec=False)
    hog3 = helper.get_hog_features(ch3, orient, pix_per_cell, cell_per_block, feature_vec=False)

    for xb in range(nxsteps):
        for yb in range(nysteps):
            ypos = yb * cells_per_step
            xpos = xb * cells_per_step

            if hog_feat:
                # Extract HOG for this patch
                hog_feat1 = hog1[ypos:ypos + nblocks_per_window, xpos:xpos + nblocks_per_window].ravel()
                hog_feat2 = hog2[ypos:ypos + nblocks_per_window, xpos:xpos + nblocks_per_window].ravel()
                hog_feat3 = hog3[ypos:ypos + nblocks_per_window, xpos:xpos + nblocks_per_window].ravel()
                hog_features = np.hstack((hog_feat1, hog_feat2, hog_feat3))

            xleft = xpos * pix_per_cell
            ytop = ypos * pix_per_cell

            # Extract the image patch
            subimg = cv2.resize(ctrans_tosearch[ytop:ytop + window, xleft:xleft + window], (64, 64))

            # Get color features
            if spatial_feat:
                spatial_features = helper.bin_spatial(subimg, size=spatial_size)
            if hist_feat:
                hist_features = helper.color_hist(subimg, nbins=hist_bins)

            # Scale features and make a prediction
            test_features = X_scaler.transform(
                np.hstack((spatial_features, hist_features, hog_features)).reshape(1, -1))
            test_prediction = svc.predict(test_features)

            if test_prediction == 1:
                xbox_left = np.int(xleft * scale)
                ytop_draw = np.int(ytop * scale)
                win_draw = np.int(window * scale)
                cv2.rectangle(draw_img, (xbox_left, ytop_draw + ystart),
                              (xbox_left + win_draw, ytop_draw + win_draw + ystart), (0, 0, 255), 6)

    return draw_img


ystart = 400
ystop = 656
scale = 1.5

out_img = find_cars(img, ystart, ystop, scale, svc, X_scaler, orient, pix_per_cell, cell_per_block, spatial_size,
                    hist_bins, spatial_feat, hist_feat, hog_feat)

plt.imshow(out_img)
plt.show()