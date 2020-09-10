#!/usr/bin/env python

'''
sample for disctrete fourier transform (dft)

USAGE:
    dft.py <image_file>
'''


# Python 2/3 compatibility
from __future__ import print_function

import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
import math
import sys


def shift_dft(src, dst=None):
    '''
        Rearrange the quadrants of Fourier image so that the origin is at
        the image center. Swaps quadrant 1 with 3, and 2 with 4.

        src and dst arrays must be equal size & type
    '''

    if dst is None:
        dst = np.empty(src.shape, src.dtype)
    elif src.shape != dst.shape:
        raise ValueError("src and dst must have equal sizes")
    elif src.dtype != dst.dtype:
        raise TypeError("src and dst must have equal types")

    if src is dst:
        ret = np.empty(src.shape, src.dtype)
    else:
        ret = dst

    h, w = src.shape[:2]

    cx1 = cx2 = w // 2
    cy1 = cy2 = h // 2

    # if the size is odd, then adjust the bottom/right quadrants
    if w % 2 != 0:
        cx2 += 1
    if h % 2 != 0:
        cy2 += 1

    # swap quadrants

    # swap q1 and q3
    ret[h-cy1:, w-cx1:] = src[0:cy1 , 0:cx1 ]   # q1 -> q3
    ret[0:cy2 , 0:cx2 ] = src[h-cy2:, w-cx2:]   # q3 -> q1

    # swap q2 and q4
    ret[0:cy2 , w-cx2:] = src[h-cy2:, 0:cx2 ]   # q2 -> q4
    ret[h-cy1:, 0:cx1 ] = src[0:cy1 , w-cx1:]   # q4 -> q2

    if src is dst:
        dst[:,:] = ret

    return dst

def highPassFilter(shape, filter_dims = 60):
	rows, cols = shape[0], shape[1]
	crow,ccol = rows//2 , cols//2
	# create a mask first, center square is 0, remaining all ones
	mask = np.ones((rows,cols,2),np.uint8)
	mask[crow-filter_dims:crow+filter_dims, ccol-filter_dims:ccol+filter_dims] = 0
	return mask

def lowPassFilter(shape, filter_dims = 30):
    rows, cols = shape[0], shape[1]
    crow = rows//2
    ccol = cols//2
    # create a mask first, center square is 1, remaining all zeros
    mask = np.zeros((rows,cols,2),np.uint8)
    mask[crow-filter_dims:crow+filter_dims, ccol-filter_dims:ccol+filter_dims] = 1
    return mask		

def DFT(img):
    
    f = np.fft.fft2(img)
    fshift = np.fft.fftshift(f)
    magnitude_spectrum = 20*np.log(np.abs(fshift))
    #magnitude_spectrum = (np.abs(fshift))
    plt.subplot(121),plt.imshow(img, cmap = 'gray')
    plt.title('Input filter'), plt.xticks([]), plt.yticks([])
    plt.subplot(122),plt.imshow(magnitude_spectrum, cmap = 'gray')
    plt.title('Magnitude Spectrum'), plt.xticks([]), plt.yticks([])
    plt.show()

def iDFTwithFilter(dft_A):
	if sys.argv[2] == '-h':
	    # Apply mask and inverse DFT
	    sidft = np.fft.fftshift(dft_A) * highPassFilter(dft_A.shape, int(sys.argv[3]))
	    type_f = "High"
	elif sys.argv[2] == '-l':
	    # Apply mask and inverse DFT
	    sidft = np.fft.fftshift(dft_A) * lowPassFilter(dft_A.shape, int(sys.argv[3]))
	    type_f = "Low"

	idft = np.fft.ifftshift(sidft)
	img_back = cv.idft(idft)
	img_back = cv.magnitude(img_back[:,:,0],img_back[:,:,1])
	#cv.imshow(type_f +' pass Filter applied', img_back)
	return img_back, type_f

def gaussianMatrix(size = 7, variance = 1):
    sum = 0.0
    limit = (size-1)/2
    gaus_kernel = [[0] * size] * size
    # generating sizexsize kernel 
    for x_var in range(-limit,limit+1): 
       for y_var in range(-limit,limit+1):
           exp_number = (x_var**2 + y_var**2) / (2.0 * variance)
           gaus_kernel[x_var + limit][y_var + limit] = math.exp(-exp_number) / (math.pi * 2.0 * variance) 
           sum += gaus_kernel[x_var + limit][y_var + limit]
       
    # normalising the Kernel 
    for i_var in range(size):
       for j_var in range(size):
           gaus_kernel[i_var][j_var] /= sum

    return np.array(gaus_kernel)

def main():
    
    if sys.argv[1] == '--image':
	fname = '../../../practice_1_2_fft/resources/gato_2.jpg'
	#im = cv.imread(fname)
	im = gaussianMatrix()
	print(im)
	DFT(im)
	#cv.imshow('Gray scaled original image', img_gray)
	#cv.imshow('Magnitude', log_spectrum)
	#img_back, type_f = iDFTwithFilter(dft)
	#cv.imshow(type_f +' pass Filter applied', img_back)
	cv.waitKey(0)
    elif sys.argv[1] == '--video':
	cap = cv.VideoCapture('../../../practice_1_2_fft/resources/yo_1.avi')
	width = int(cap.get(3))
	height = int(cap.get(4))
	frames = cap.get(cv.CAP_PROP_FRAME_COUNT)
	print("{}, {}, {}".format(width,height,frames))
	fourcc = cv.VideoWriter_fourcc(*'XVID')
	out = cv.VideoWriter('../../../practice_1_2_fft/resources/output.avi',fourcc, frames, (width,height))
	
	while(cap.isOpened()):
		ret, frame = cap.read()
		if ret == True:
			mg_gray, log_spectrum, dft = DFT(frame)
			img_back, type_f = iDFTwithFilter(dft)
			cv.imshow('frame', img_back)
			out.write(np.uint8(img_back))
			if cv.waitKey(1) % 0xFF == ord('q'):
				break
		else:
			break
		
	
    
    print('Done')


if __name__ == '__main__':
    print(__doc__)
    main()
    cv.destroyAllWindows()
