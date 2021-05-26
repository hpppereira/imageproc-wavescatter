
import cv2
import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from skimage import measure
import imageio
from scipy import ndimage, misc

plt.close('all')

pathname = os.environ['HOME'] + \
           '/GoogleDrive/Wave_Scatter/data/DERIVA_RANDOMICA/VIDEO/CAM1/T100/'

# filename = 'T100_010100_CAM1.avi'
# filename = 'T100_010100.CAM1_ISOPOR.avi'
filename = 'T100_020100_CAM1.avi'


# ---------------------------------------------------------------------- #

# le o video
cap = cv2.VideoCapture(pathname+filename)

# ---------------------------------------------------------------------- #

# escolhe o frame inicial
contf = 0
for ff in np.arange(1000,1500,1):

    contf += 1

    cap.set(cv2.CAP_PROP_POS_FRAMES, ff)

    # take first frame of the video
    ret, frame = cap.read()  
    ret, frame1 = cap.read()

    img = frame[250:-130,200:]
    img1 = frame1[250:-130,200:]

    # convert to gray scale
    gray = cv2.cvtColor(img ,cv2.COLOR_BGR2GRAY)
    gray1 = cv2.cvtColor(img1 ,cv2.COLOR_BGR2GRAY)

    # threshold com otsu
    ret, thresh = cv2.threshold(gray,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
    ret, thresh1 = cv2.threshold(gray1,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
    # ret, thresh = cv.threshold(img,127,255,0)


    # so acha o contorno e da as cores no primeiro frame, depois soh faz o loop
    # de uma imagem menor a outra
    if contf == 1:

        # Find contours at a constant value of 0.8
        # contours = measure.find_contours(thresh, 100)
        im2, contours, hierarchy = cv2.findContours(thresh, 1, 2)

        # stop

        # plt.figure(figsize=(12,8))
        # plt.imshow(thresh)

        # matriz que sera a imagem com os contornos de cores 
        # especificas para rastrear
        cimg = np.zeros_like(thresh) + 255

        c = -1
        cor = 0
        for cnt in contours:

            area = cv2.contourArea(cnt)

            if area > 1000:

                print 'Area maior que 1000'

                c +=1
                cor += 10

                print 'Contour n: %i' %c

                print 'Area: %.1f' %area

                # cnt = contours[0]

                # Moments
                # Image moments help you to calculate some features like center 
                # of mass of the object, area of the object etc. Check out the
                # wikipedia page on Image Moments
                # The function cv.moments() gives a dictionary of all moment
                # values calculated. See below:

                M = cv2.moments(cnt)
                # print( M )

                cx = int(M['m10']/M['m00'])
                cy = int(M['m01']/M['m00'])


                # Contour Area
                # Contour area is given by the function cv.contourArea() or from moments, M['m00'].

                area = cv2.contourArea(cnt)

                # print 'Area: %.1f' %area


                # Contour Perimeter
                # It is also called arc length. It can be found out using cv.arcLength()
                # function. Second argument specify whether shape is a closed contour 
                # (if passed True), or just a curve.

                perimeter = cv2.arcLength(cnt,True)

                print 'Perimeter: %.1f' %perimeter

                print '# -------------------------- #'

                # Contour Approximation
                # It approximates a contour shape to another shape with less number of
                # vertices depending upon the precision we specify. It is an implementation
                # of Douglas-Peucker algorithm. Check the wikipedia page for algorithm and demonstration.
                # To understand this, suppose you are trying to find a square in an image, 
                # but due to some problems in the image, you didn't get a perfect square, but
                # a "bad shape" (As shown in first image below). Now you can use this function
                # to approximate the shape. In this, second argument is called epsilon, which
                # is maximum distance from contour to approximated contour. It is an accuracy
                # parameter. A wise selection of epsilon is needed to get the correct output.

                # epsilon = 0.1*cv2.arcLength(cnt,True)
                # approx = cv2.approxPolyDP(cnt,epsilon,True)
                # cnt1 = approx
                # if area > 1000:
                # stop

                # a variavel cimg vai eh a variavel com as bolinhas coloridas
                # as bolinhas estao com pixel de 10 a x a cada 10
                # Create a mask image that contains the contour filled in
                # cimg = np.zeros_like(thresh)
                cv2.drawContours(cimg, contours, c, color=cor, thickness=-1)

        imres = thresh1 - cimg + 255


                # plt.plot(cnt[:,0,:][:,0],cnt[:,0,:][:,1])
                # plt.plot(cnt1[:,0,:][:,0],cnt1[:,0,:][:,1])
                # plt.close('all')

    # else:


    # imagem colorida da subtracao do frame colorido menor o preto/branco
    imres = thresh1 - imres + 255

    plt.figure()
    plt.imshow(imres)
    plt.savefig('/home/hp/Documents/teste/teste_%s' %ff)
    plt.close('all')



        # plt.show()
        # stop


