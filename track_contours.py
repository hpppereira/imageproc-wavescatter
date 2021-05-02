"""
===============
Contour finding
===============

We use a marching squares method to find constant valued contours in an image.
In ``skimage.measure.find_contours``, array values are linearly interpolated
to provide better precision of the output contours. Contours which intersect
the image edge are open; all others are closed.

The `marching squares algorithm
<http://www.essi.fr/~lingrand/MarchingCubes/algo.html>`__ is a special case of
the marching cubes algorithm (Lorensen, William and Harvey E. Cline. Marching
Cubes: A High Resolution 3D Surface Construction Algorithm. Computer Graphics
(SIGGRAPH 87 Proceedings) 21(4) July 1987, p. 163-170).
"""

import cv2
import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from skimage import measure
import imageio
from scipy import ndimage, misc
from PIL import Image


plt.close('all')
cv2.destroyAllWindows()

pathname = os.environ['HOME'] + \
           '/Documents/wavescatter_videos/DERIVA_RANDOMICA/VIDEO/CAM1/T100/'

filename = 'T100_010300_CAM1.avi'

def fig2data ( fig ):
    """
    @brief Convert a Matplotlib figure to a 4D numpy array with RGBA channels and return it
    @param fig a matplotlib figure
    @return a numpy 3D array of RGBA values
    """
    # draw the renderer
    fig.canvas.draw ( )
 
    # Get the RGBA buffer from the figure
    w,h = fig.canvas.get_width_height()
    buf = np.fromstring ( fig.canvas.tostring_argb(), dtype=np.uint8 )
    buf.shape = ( w, h,4 )
 
    # canvas.tostring_argb give pixmap in ARGB mode. Roll the ALPHA channel to have it in RGBA mode
    buf = np.roll ( buf, 3, axis = 2 )

    w, h, d = buf.shape

    mat = Image.frombytes( "RGBA", ( w ,h ), buf.tostring( ) )


    return mat


# ---------------------------------------------------------------------- #

# le o video
cap = cv2.VideoCapture(pathname+filename)

# ---------------------------------------------------------------------- #

for ff in np.arange(1000,2000,10):

    cap.set(cv2.CAP_PROP_POS_FRAMES, ff)

    # take first frame of the video
    ret, frame1 = cap.read()  
    ret, frame2 = cap.read()

    img1 = frame1[250:-130,200:]
    img2 = frame2[250:-130,200:]

    # convert to gray scale
    gray1 = cv2.cvtColor(img1 ,cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2 ,cv2.COLOR_BGR2GRAY)

    # threshold com otsu
    ret, thresh1 = cv2.threshold(gray1,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
    ret, thresh2 = cv2.threshold(gray2,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)

    # Find contours at a constant value of
    contours1 = measure.find_contours(thresh1, 100)
    contours2 = measure.find_contours(thresh2, 100)

    # im1, contours1, hierarchy1 = cv2.findContours(thresh1,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    # im2, contours2, hierarchy2 = cv2.findContours(thresh1,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

    # contours = measure.find_contours(thresh1-thresh, 100)
    # contours1 = measure.find_contours(thresh, 100)


    # lista com comprimento dos contornos, para deixar eles em ordem crescente
    # length_contours = []
    # big_contours = [] # contornos com mais que x pontos
    # left_contours = []
    # cor = 0
    # for n, contour in enumerate(contours):

    #     # condicao para pegar apenas contornos maiores que 90 pontos
    #     # if len(contour) > 5 and len(contour) < 335:
    #     if len(contour) > 100 and len(contour) < 1000:

    #         print 'Contorno numero: %s' %n
    #         print 'Numero de pontos: %s' %len(contour)

    #         big_contours.append(contour)
    #         length_contours.append(len(contour))
    #         left_contours.append(contour[:,0][0])

    #         # coloca cor especifica em um contorno
    #         thresh[contour[:,0].mean().astype(int), contour[:,1].mean().astype(int)] = cor
    #         cor += 10

    # stop


    # acha contornos apenas maiores que 90 pontos e coloca eles em orgem
    # crescente para identificar cada contorno
    # contours_sort = np.array(big_contours)[list(np.argsort(length_contours))]
    # contours_sort = np.array(big_contours)[list(np.argsort(left_contours))]

    # cria lista com valores do centro do cluster (media nas posicoes x e y)
    # e plota os valores



    # cv2.drawContours(frame1,contours1,-1,(0,255,0),3)
    # cv2.imshow("Contour",frame1)

    # stop

    # Display the image and plot all contours found
    # fig1 = plt.figure(figsize=(12,8))
    # ax1 = fig1.add_subplot(111)
    # ax1.imshow(thresh1, interpolation='nearest', cmap=plt.cm.gray)

    fig2 = plt.figure(figsize=(7,17.2))
    ax2 = fig2.add_subplot(111)
    ax2.imshow(thresh2, interpolation='nearest', cmap=plt.cm.gray)

    # plota contornos na figura 1
    for cs in contours2:
        ax2.fill(cs[:, 1], cs[:, 0], linewidth=2)

    # # plota contornos na figura 2
    # for cs in contours2:
    #     cont += 1
    #     ax.fill(cs[:, 1], cs[:, 0], linewidth=2)
    #     ax.plot(cs[:, 1].mean(), cs[:, 0].mean(), 'k.')

    # converte figuras para array (imagem)
    # mat1 = fig2data(fig1)
    mat2 = fig2data(fig2)

    plt.close('all')

    # mat1 = np.array(mat1[:,:,2])
    mat2 = np.array(mat2)[:,:,2]


    # ax.axis('image')
    # ax.set_xticks([])
    # ax.set_yticks([])

    # fig.savefig('/home/hp/Documents/teste3/teste_%s' %ff)


    stop

# When everything done, release the capture
cap.release()

    # plt.show()

    # ff += 1


# plt.show()
