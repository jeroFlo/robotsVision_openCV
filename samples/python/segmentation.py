#!/usr/bin/env python
# Python 2/3 compatibility
'''
	Speeded-Up Robust Features
'''
from __future__ import print_function

import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
import random as rng

rng.seed(12345)

def scale(img, scale_percent = 50):# percent of original size
	width = int(img.shape[1] * scale_percent / 100)
	height = int(img.shape[0] * scale_percent / 100)
	dim = (width, height)
	return cv.resize(img, dim, interpolation= cv.INTER_AREA)

def binarization(img, thres = 127, otsu=False, inv=True):
	gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
	#cv.imshow('gray', gray)
	if not otsu:
		if inv:
			ret1,thresh = cv.threshold(gray,thres,255,cv.THRESH_BINARY_INV)
		else:
			ret1,thresh = cv.threshold(gray,thres,255,cv.THRESH_BINARY)
		return thresh

	if inv:
		ret, thresh = cv.threshold(gray,0,255,cv.THRESH_BINARY_INV+cv.THRESH_OTSU)
	else:
		ret, thresh = cv.threshold(gray,0,255,cv.THRESH_BINARY+cv.THRESH_OTSU)
	return thresh

def erosion(img, size = 3, iter=1):
    kernel = np.ones((size,size), np.uint8) 
    # The first parameter is the original image, 
    # kernel is the matrix with which image is  
    # convolved and third parameter is the number  
    # of iterations, which will determine how much  
    # you want to erode/dilate a given image.  
    img_erosion = cv.erode(img, kernel, iterations=iter) 
    return img_erosion

def dilation(img, size = 3, iter=1):
    kernel = np.ones((size,size), np.uint8) 
    img_dilation = cv.dilate(img, kernel, iterations=iter) 
    return img_dilation

def closing(img, size =3 ,iter = 1):
	for i_iter in range(iter):
		img = erosion(dilation(img, size), size)
	return img

def opening(img, size =3 ,iter = 1):
	for i_iter in range(iter):
		img = dilation(erosion(img, size), size)
	return img

def main():
	#image_path = '../data/puzzle_part_edit.jpg'
	image_path = '../data/one_piece_puzzle.jpeg'

	img = cv.imread(image_path)
	print(img.shape)
	img = scale(img, 20)
	print(img.shape)
	#cv.imshow('Scaled', img)

	binary = binarization(img, 110)
	#binary_n = binarization(img, 105, inv=False)
	#cv.imshow('binary not inverted', binary_n)
	
	binary = dilation(binary, size=6, iter=2)
	#cv.imshow('binary', binary)
	sure_bg = closing(binary, 4, iter=2)
	#binary = dilation(binary, size=4)
	cv.imshow('closing', binary)
	
	
	dist = cv.distanceTransform(sure_bg,distanceType=cv.DIST_L2,maskSize=3)
	cv.normalize(dist, dist, 0, 1.0, cv.NORM_MINMAX)
	cv.imshow('dist', dist)
	_, binary = cv.threshold(dist, 0.29, 1.0, cv.THRESH_BINARY)

	cv.imshow('binary normalize', binary)
	#binary = opening(binary)
	sure_fg = erosion(binary, 4)
	cv.imshow('opening', sure_fg)


	# Finding unknown region
	sure_fg = np.uint8(sure_fg)
	unknown = cv.subtract(sure_bg,sure_fg)
	# Marker labelling
	ret, markers = cv.connectedComponents(sure_fg)
	# Add one to all labels so that sure background is not 0, but 1
	markers = markers+1
	# Now, mark the region of unknown with zero
	markers[unknown==255] = 0
	#markers = np.uint8(markers)
	#cv.imshow('markers', markers)
	markers = cv.watershed(img,markers)
	img[markers == -1] = [0,255,0]
	cv.imshow('result',img)
	'''
	# Marker labelling
	binary = np.uint8(binary)
	contours, _= cv.findContours(binary, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

	#print(binary)
	#ret, markers = cv.connectedComponents(binary)
	# Add one to all labels so that sure background is not 0, but 1
	#print(markers[0][20])
	#markers = markers+1
	# Now, mark the region of unknown with zero
	#markerrs[unknown==255] = 0
	markers = cv.watershed(img,markers)
	print(markers)
	
	img[markers == -1] = [255,0,0]
	cv.imshow('x', img)
	'''
	# Create the CV_8U version of the distance image
	# It is needed for findContours()
	#dist_8u = binary.astype('uint8')
	# Find total markers
	#contours, _= cv.findContours(dist_8u, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
	#print(contours)
	# Create the marker image for the watershed algorithm
	#markers = np.zeros(binary.shape, dtype=np.int32)
	# Draw the foreground markers
	#for i in range(len(contours)):
	#	print(i)
	#	cv.drawContours(markers, contours, i, (i+1), -1)
	# Draw the background marker
	#cv.circle(markers, (5,5), 3, (255,0,0), -1)

	#markers = cv.watershed(img, markers)
	#img[markers == -1] = [0,255,0]

	#print(markers)
	#cv.imshow('x', img)
	'''
	#mark = np.zeros(markers.shape, dtype=np.uint8)
	mark = markers.astype('uint8')
	mark = cv.bitwise_not(mark)
	# uncomment this if you want to see how the mark
	# image looks like at that point
	cv.imshow('Markers_v2', mark)
	# Generate random colors
	colors = []
	for contour in contours:
	    colors.append((rng.randint(0,256), rng.randint(0,256), rng.randint(0,256)))
	# Create the result image
	dst = np.zeros((markers.shape[0], markers.shape[1], 3), dtype=np.uint8)
	# Fill labeled objects with random colors
	for i in range(markers.shape[0]):
	    for j in range(markers.shape[1]):
	        index = markers[i,j]
	        if index > 0 and index <= len(contours):
	            dst[i,j,:] = colors[index-1]
	# Visualize the final image
	cv.imshow('Final Result', dst)
	#img[markers==-1] = [255,0,0]
	#cv.imshow('Markers', markers)
	'''
	cv.waitKey()
if __name__ == '__main__':
    print(__doc__)
    main()
    cv.destroyAllWindows()