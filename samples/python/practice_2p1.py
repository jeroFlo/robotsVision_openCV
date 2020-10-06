#!/usr/bin/env python

'''
'''

# Python 2/3 compatibility
from __future__ import print_function

import numpy as np
import cv2 as cv
import math

bins = np.arange(256).reshape(256,1)

def hist_curve(im):
    h = np.zeros((300,256,3))
    if len(im.shape) == 2:
        color = [(255,255,255)]
    elif im.shape[2] == 3:
        color = [ (255,0,0),(0,255,0),(0,0,255) ]
    for ch, col in enumerate(color):
        hist_item = cv.calcHist([np.uint8(im)],[ch],None,[256],[0,256])
        cv.normalize(hist_item,hist_item,0,255,cv.NORM_MINMAX)
        hist=np.int32(np.around(hist_item))
        pts = np.int32(np.column_stack((bins,hist)))
        cv.polylines(h,[pts],False,col)
    y=np.flipud(h)
    return y

def binary(img, threshold = 127):
    im = img
    im = cv.cvtColor(im,cv.COLOR_BGR2GRAY)
    #blur = cv.GaussianBlur(im,(3,3),0)
    #ret3,th3 = cv.threshold(blur,0,255,cv.THRESH_BINARY+cv.THRESH_OTSU)
    ret1,th3 = cv.threshold(im,threshold,255,cv.THRESH_BINARY)
    return th3

def contrast(im, alpha):
    img = np.where(im*alpha < im, 255, alpha*im)
    return img

def brightness(im, beta):
    if beta > 0:
        img = np.where(im+beta < im, 255, im+beta)
    elif beta < 0:
        img = np.where(im+beta < 0, 0, im+beta)
        return np.uint8(img)
    else:
        return im 

    return img

def main():
    import sys

    if len(sys.argv)>1:
        fname = sys.argv[1]
        #../../../practice_1_2_fft/resources/gato_2.jpg
    else :
        fname = 'lena.jpg'
        print("usage : python hist.py <image_file>")

    im = cv.imread(cv.samples.findFile(fname))

    if im is None:
        print('Failed to load image file:', fname)
        sys.exit(1)

    gray = cv.cvtColor(im,cv.COLOR_BGR2GRAY)


    print(''' Histogram plotting \n
    show histogram for color image in curve mode \n
    show binarization from the input image \n
    Esc - exit \n
    ''')

    cv.imshow('image',im)
   
    curve = hist_curve(im)      #obtener el histograma
    cv.imshow('histogram original image',curve)#mostrar el histograma
            
        
    img = contrast(im, 2)
    cv.imshow('contrast', img)
    curve = hist_curve(img)      #obtener el histograma
    cv.imshow('histogram contrast',curve)
    
    img = brightness(im, -100)
    cv.imshow('brightness mine', img)
    curve = hist_curve(img)      #obtener el histograma
    cv.imshow('histogram brightness mine',curve)
   
    
    #for contrast and brightness given function
    img = cv.convertScaleAbs(im, alpha=1, beta=100)
    cv.imshow('brightness', img)
    curve = hist_curve(img)      #obtener el histograma
    cv.imshow('histogram brightness',curve)


    #Edge detection
    #gaussiana = cv2.GaussianBlur(gris, (n,n), 0)
    img_gauss = cv.GaussianBlur(gray, (3,3), 0) # 3x3 kernel 

    img = binary(im, 150)
    cv.imshow('image',img)
    
    # Canny
    #canny = cv2.Canny(imagen, umbral_minimo histeresis, umbral_maximo)
    img_canny = cv.Canny(img, 100, 200)
    cv.imshow("Canny", img_canny)

    # Sobel 
    img_sobelx = cv.Sobel(img_gauss, cv.CV_8U, 1, 0, ksize=3)
    img_sobely = cv.Sobel(img_gauss, cv.CV_8U, 0, 1, ksize=3)
    img_sobel = img_sobelx + img_sobely
    cv.imshow("Sobel X", img_sobelx)
    cv.imshow("Sobel Y", img_sobely)
    cv.imshow("Sobel", img_sobel)

    # Prewitt
    kernelx = np.array([[1,1,1],[0,0,0],[-1,-1,-1]])
    kernely = np.array([[-1,0,1],[-1,0,1],[-1,0,1]])
    img_prewittx = cv.filter2D(img_gauss, -1, kernelx)
    img_prewitty = cv.filter2D(img_gauss, -1, kernely)
    cv.imshow("Prewitt X", img_prewittx)
    cv.imshow("Prewitt Y", img_prewitty)
    cv.imshow("Prewitt", img_prewittx + img_prewitty)

    cv.waitKey(0)
    print('Done')


if __name__ == '__main__':
    print(__doc__)
    main()
    cv.destroyAllWindows()
