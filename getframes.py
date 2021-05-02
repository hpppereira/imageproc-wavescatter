# =============================================================================================== #

"""
ONR Project - WaveDrift

Image Processing:
- Save frames from an video
# - Track object path

Modifications:
2017/10/09 - Henrique, Uggo e Nelson
"""

# =============================================================================================== #
# Import Libraries

import os
import pandas as pd
# import cv2
import numpy as np
from matplotlib import pyplot as plt

plt.close('all')

# =============================================================================================== #
# Enable/Disable functions (True or False)

gf = True  # get frames
mf = False  # make filter (laplacian)
imd = False # image difference

# =============================================================================================== #
# Input Files

pathname = os.environ['HOME'] + '/Dropbox/Random_Drift/data/DERIVA_RANDOMICA/VIDEO/CAM1/T100/'
# framedir = 'frames'
framedirlap = 'frames_laplacian'
filename = 'T100_010100_CAM1.avi'  # list of files
framei = '00:00:0.000'  # first frame
# framei = '00:00:04.000'  # first frame
framef = '00:02:0.000'  # last frame
freq = '300ms'  # fps
fileout = 'img'  # filename

# =============================================================================================== #
# Functions

# ----------------------------------------------------------------------------------------------- #

def getframes(pathname, moviefile, framei, framef, freq, fileout):

    """
    Get frames from a video
    :param pathname:
    :param moviefile:
    :param framei:
    :param framef:
    :param freq:
    :param fileout:
    :return: saved frames
    """

    frames1 = pd.date_range(framei, framef, freq=freq)

    def function1(x):
        return x.strftime('%H:%M:%S.%f')

    frames = frames1.format(formatter=function1)

    cont = 0
    for frame in frames:
        cont += 1
        os.system('cd ' + pathname + '\n' +
                  'ffmpeg -i ' + moviefile + ' -ss ' + frame +
                  ' -f image2 -vframes 1 ' + pathname + 'frames/'
                  + fileout + '_' + frame + '_' + str(cont).zfill(3) + '.png')

# ----------------------------------------------------------------------------------------------- #

def imagefilter(pathname, filename, method='Laplacian'):

    """
    Apply image filter
    :param pathname:
    :param filename:
    :param filter: 'Normal', 'Laplacian','Slobel_X', Slobel_Y'
    :return: laplacian
    """

    global image

    # read the data
    img0 = cv2.imread(pathname+filename)

    # converting to gray scale
    gray = cv2.cvtColor(img0, cv2.COLOR_BGR2GRAY)

    # remove noise
    img = cv2.GaussianBlur(gray, (3, 3), 0)

    if method == 'Laplacian':

        print ('Making Laplacian filter')

        # convolute with proper kernels
        image = cv2.Laplacian(img, cv2.CV_64F)

    elif method=='Slobel_X':

        print ('Making Slobel_X filter')

        image = cv2.Sobel(img,cv2.CV_64F,1,0,ksize=5)  # x

    elif method == 'Slobel_Y':

        print ('Making Slobel_Y filter')

        image = cv2.Sobel(img,cv2.CV_64F,0,1,ksize=5)  # y

    else:

        image = img0

    return image

# ----------------------------------------------------------------------------------------------- #

def imagediff(image1, image2):

    """
    Difference between posterior and anterior image
    :param image1: first image
    :param image2: second image
    :return: subtracted image
    """
    imdif = image2 - image1

    return imdif

# =============================================================================================== #
# =============================================================================================== #
# =============================================================================================== #
# Start Execution
# =============================================================================================== #
# =============================================================================================== #
# =============================================================================================== #

# ----------------------------------------------------------------------------------------------- #
# save original frames

if gf:

    getframes(pathname=pathname,
              moviefile=filename,
              framei=framei,
              framef=framef,
              freq=freq,
              fileout=fileout + '_' + filename)

# ----------------------------------------------------------------------------------------------- #
# save processed frames with laplace filter


filelist = np.sort(os.listdir(pathname+'frames/'))

for f in range(len(filelist)-1):

    print f

    # ------------------ #
    # make image filter
    if mf:

        df1 = imagefilter(pathname+'frames/', filelist[f], method='Laplacian')
        df2 = imagefilter(pathname+'frames/', filelist[f+1], method='Laplacian')

    # adjust values
    # df[df>10] = df[df>10]+1000
    # df[df<10] = df[df<10]-1000

    # plot figure
    # plt.figure()
    # plt.imshow(df)
    # plt.colorbar()

    # ------------------ #
    # image difference
    if imd == True:

        imdif = imagediff(df1, df2)


plt.show()


# =============================================================================================== #























