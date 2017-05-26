import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import helper
import os
import configparser

car_classify_data_dir = '../datasets/udacity-vehicle-tracking'

config = configparser.ConfigParser()
config.read('config.ini')

# Read in car and non-car images
cars = helper.list_files(os.path.join(car_classify_data_dir, 'vehicles'))
notcars = helper.list_files(os.path.join(car_classify_data_dir, 'non-vehicles'))

# Play with these values to see how your classifier
colorspace_conv = 'RGB2YCrCb' # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
orient = 9
pix_per_cell = 8
cell_per_block = 2
hog_channel = [0, 1, 2]
spatial_feat = False
hist_feat = False
hog_feat = True

# pick up an image
image = mpimg.imread(cars[100])

single_car_hog, hog_image = helper.get_hog_features(image[:, :, 0], orient, pix_per_cell, cell_per_block, vis=True)

h, (h1, h2) = plt.subplots(1, 2, figsize=(15, 9))
h.tight_layout()
h1.imshow(image)
h2.imshow(hog_image)
plt.savefig(os.path.join('./test_images', 'hog_output.jpg'), bbox_inches='tight')