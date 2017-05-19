import cv2
import numpy as np
import helper

# Read a color image
img = cv2.imread("./test_images/000275.png")

features = helper.bin_spatial(img, 'HLS')
