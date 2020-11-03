#!/usr/bin/env python

'''
'''

# Python 2/3 compatibility
from __future__ import print_function

from matplotlib.colors import hsv_to_rgb
from skimage.transform import resize
from skimage.feature import hog
from skimage import exposure
import numpy as np
import cv2 as cv
import math
import matplotlib.pyplot as plt
import imutils
import time
import re
from numpy import savetxt

def scale(img, scale_percent = 50):# percent of original size
	width = int(img.shape[1] * scale_percent / 100)
	height = int(img.shape[0] * scale_percent / 100)
	dim = (width, height)
	return cv.resize(img, dim, interpolation= cv.INTER_AREA)

def pyramid(image, scale=1.5, minSize=(30,30)):
	yield image

	while True:
		w = int(image.shape[1]/scale)
		image = imutils.resize(image, width=w)
		if image.shape[0] < minSize[1] or image.shape[1] < minSize[0]:
			break
		yield image

def sliding_window(image, stepSize, windowSize):
	# slide a window across the image
	for y in range(0, image.shape[0], stepSize):
		for x in range(0, image.shape[1], stepSize):
			# yield the current window
			yield (x, y, image[y:y + windowSize[1], x:x + windowSize[0]])
def HOG(crop_img, orient=9, pixels_per_cell=(8,8), cells_per_block=(2,2)):
	return hog(crop_img, orientations=orient, pixels_per_cell=pixels_per_cell, cells_per_block=cells_per_block, visualize=True, multichannel=True)
	
def main():

	image = cv.imread("../data/puzzle_part.jpg")

	image = scale(image, 20)
	print(image.shape)

	'''
	for (i, resized) in enumerate(pyramid(image)):
		# show the resized image
		cv.imshow("Layer {}".format(i + 1), resized)
	        cv.waitKey(0)
	'''
	f = open("../data/features_hog.csv", "w")
	(winW, winH) = (64,128)
	# loop over the image pyramid
	for resized in pyramid(image, scale=2, minSize=(64,128)):
		# loop over the sliding window for each layer of the pyramid
		for (x, y, window) in sliding_window(resized, stepSize=32, windowSize=(winW, winH)):
			# if the window does not meet our desired window size, ignore it
			if window.shape[0] != winH or window.shape[1] != winW:
				continue
			#print("resized size {}".format(resized.shape))
			#print("window size {}".format(window.shape))
			fd, hog_image = HOG(window)
			#print(window)
			f.write(','.join(map(str,fd)))
			f.write('\n')
			#cv.imshow('window', window)
			#cv.imshow('hog', hog_image)
			#cv.waitKey(0)

			# we'll just draw the window
			#clone = resized.copy()
			#cv.rectangle(clone, (x, y), (x + winW, y + winH), (0, 255, 0), 2)
			#cv.imshow("Window", clone)
			#cv.waitKey(1)
			#time.sleep(0.025)

	f.close()		
        print('Done')


if __name__ == '__main__':
    print(__doc__)
    main()
    cv.destroyAllWindows()


