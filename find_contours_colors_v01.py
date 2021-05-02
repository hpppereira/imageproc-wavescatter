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

pathname = os.environ['HOME'] + \
           '/Documents/wavescatter_videos/DERIVA_RANDOMICA/VIDEO/CAM1/T100/'

filename = 'T100_010300_CAM1.avi'

def fig2data ( fig ):
    """
    @brief Convert a Matplotlib figure to a 4D numpy array with RGBA channels and return it
    @param fig a matplotlib figure
    @return a numpy 3D array of RGBA values
    """/
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



for ff in np.arange(1000,2000,10):

    cap.set(cv2.CAP_PROP_POS_FRAMES, ff)

    # take first f rame of the video
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

    # Find contours at a constant value of 0.8
    contours = measure.find_contours(thresh, 100)
    # contours = measure.find_contours(thresh1-thresh, 100)
    # contours1 = measure.find_contours(thresh, 100)

    # lista com comprimento dos contornos, para deixar eles em ordem crescente
    length_contours = []
    big_contours = [] # contornos com mais que x pontos
    left_contours = []
    cor = 0
    for n, contour in enumerate(contours):

        # condicao para pegar apenas contornos maiores que 90 pontos
        # if len(contour) > 5 and len(contour) < 335:
        if len(contour) > 100 and len(contour) < 1000:

            print 'Contorno numero: %s' %n
            print 'Numero de pontos: %s' %len(contour)

            big_contours.append(contour)
            length_contours.append(len(contour))
            left_contours.append(contour[:,0][0])

            # coloca cor especifica em um contorno
            thresh[contour[:,0].mean().astype(int), contour[:,1].mean().astype(int)] = cor
            cor += 10

    # stop


    # acha contornos apenas maiores que 90 pontos e coloca eles em orgem
    # crescente para identificar cada contorno
    # contours_sort = np.array(big_contours)[list(np.argsort(length_contours))]
    contours_sort = np.array(big_contours)[list(np.argsort(left_contours))]



    # cria lista com valores do centro do cluster (media nas posicoes x e y)
    # e plota os valores

    # Display the image and plot all contours found
    fig = plt.figure(figsize=(12,8))
    ax = fig.add_subplot(111)
    ax.imshow(thresh , interpolation='nearest', cmap=plt.cm.gray)

    cont = 0 #identificador
    for cs in contours_sort:

        cont += 1

        ax.fill(cs[:, 1], cs[:, 0], linewidth=2)
        # ax.plot(cs[:, 1], cs[:, 0], linewidth=2)
        ax.text(cs[:, 1].mean(), cs[:, 0].mean(), str(cont), fontsize=16)
        ax.plot(cs[:, 1].mean(), cs[:, 0].mean(), 'k.')


    ax.axis('image')
    ax.set_xticks([])
    ax.set_yticks([])

    fig.savefig('/home/hp/Documents/teste2/teste_%s' %ff)

    mat = fig2data ( fig )

    # stop



    # plt.show()
    plt.close('all')

    ff += 1


# plt.show()

def gerar_imagens_a_partir_do_video(arquivo_avi):
    cap = cv2.VideoCapture(arquivo_avi)

    # ---------------------------------------------------------------------- #

    for ff in np.arange(1000,2000,10):

        cap.set(cv2.CAP_PROP_POS_FRAMES, ff)

        # take first f rame of the video
        ret = True
        while ret:
            ret, frame = cap.read()
            yield frame

def processar_contornos_da_imagem(img):
    pass

def identificar_ponto_central(contorno):
    pass
    return []


if __name__ == '__main__':
    arquivo_avi = os.path.join(os.environ['HOME'], '/Documents/wavescatter_videos/DERIVA_RANDOMICA/VIDEO/CAM1/T100/', 'T100_010300_CAM1.avi')
    for frame in gerar_imagens_a_partir_do_video(arquivo_avi)[:5]:
        contornos = processar_contornos_da_imagem(frame)
        pontos_centrais = [ identificar_ponto_central(c) for c in processar_contornos_da_imagem(img) ]
