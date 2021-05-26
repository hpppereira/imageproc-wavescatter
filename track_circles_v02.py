"""
Acha circulos e cria caminho dos circulos encontrados
- o numero de circulos a serem rastrados sera a quantidade
de circulos encontrados no primeiro frame

Henrique Pereira
ONR
03/05/2018

with the arguments:

src_gray: Input image (grayscale)
circles: A vector that stores sets of 3 values: x_{c}, y_{c}, r for each detected circle.
CV_HOUGH_GRADIENT: Define the detection method. Currently this is the only one available in OpenCV
dp = 1: The inverse ratio of resolution
min_dist = src_gray.rows/8: Minimum distance between detected centers
param_1 = 200: Upper threshold for the internal Canny edge detector
param_2 = 100*: Threshold for center detection.
min_radius = 0: Minimum radio to be detected. If unknown, put zero as default.
max_radius = 0: Maximum radius to be detected. If unknown, put zero as default
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import cv2
import pandas as pd
from scipy import spatial

cv2.destroyAllWindows()

def carrega_video(pathname, nome_video):
    cap = cv2.VideoCapture(pathname+nome_video)
    fps = cap.get(cv2.CAP_PROP_FPS)
    return cap, fps

def acha_frame_inicial_e_final(dict_videos, nome_video, fps):
    # tempo inical e final do filme
    dtime = pd.to_datetime(dict_videos[nome_video], format='%M:%S')
    # tempo em timedelta (para converter para total_seconds)
    timei = dtime[0] - pd.Timestamp('1900')
    timef = dtime[1] - pd.Timestamp('1900')
    # duracao do video em time delta
    dur = dtime[1] - dtime[0]
    # duracao do video em segundos
    durs = dur.total_seconds()
    # numero do frame inicial e final (maquina de 30 fps)
    nframei = int(timei.total_seconds() * fps)
    nframef = int(timef.total_seconds() * fps)
    return nframei, nframef

def carrega_frame(cap, ff):
    cap.set(cv2.CAP_PROP_POS_FRAMES, ff)
    ret, frame = cap.read()
    frame = frame[250:-130,200:]
    return frame

def acha_circulos(frame):
    # circles = cv2.HoughCircles(image=frame, method=cv2.HOUGH_GRADIENT, dp=1, minDist=50, param1=10, param2=5, minRadius=8, maxRadius=10)
    circles = cv2.HoughCircles(image=frame,
                               method=cv2.HOUGH_GRADIENT,
                               dp=1,
                               minDist=30,
                               param1=10,
                               param2=5,
                               minRadius=10,
                               maxRadius=20)
    return circles


if __name__ == '__main__':

    # Dados de entrada
    pathname_video = os.environ['HOME'] + '/Documents/wavescatter_videos/' \
                   'DERIVA_RANDOMICA/VIDEO/CAM1/T100/'

    nome_video = 'T100_010300_CAM1.avi'

    # dicionario com nome do video e tempo inicial e final em que as
    # bolinhas estao dentro da tela e ja se separaram (quando em cluster)
    dict_videos = {'T100_010100.CAM1.avi': ['00:38','01:54'],
                   'T100_010100.CAM1_ISOPOR.avi': ['00:13','01:38'],
                   'T100_010100_CAM1.avi': ['00:12','01:58'],
                   'T100_010200_CAM1.avi': ['00:11','01:56'],
                   'T100_010300_CAM1.avi': ['00:08','01:37'],
                   'T100_020100_CAM1.avi': ['00:08','01:30']}

    cap, fps = carrega_video(pathname_video, nome_video)
    nframei, nframef = acha_frame_inicial_e_final(dict_videos, nome_video, fps)

    # cria variavel lista de paths com id das bolas
    paths = {}
    numero_bolas = 30
    for b in range(1,numero_bolas):
        paths['ball_%s' %str(b).zfill(2)] = []

    for ff in range(nframei, nframei+20):

        # print (ff)
        # stop

        frame1 = carrega_frame(cap, ff)
        frame2 = carrega_frame(cap, ff+1)

        # frame original para plotagem do resultado
        output1 = frame1.copy()
        output2 = frame2.copy()
        
        # frame2 = carrega_frame(cap, nframei+1)
        frame1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
        frame2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

        frame1 = cv2.GaussianBlur(frame1,(5,5),0)
        frame2 = cv2.GaussianBlur(frame2,(5,5),0)

        ret, frame1 = cv2.threshold(frame1,70,255,cv2.THRESH_BINARY)
        ret, frame2 = cv2.threshold(frame2,70,255,cv2.THRESH_BINARY)

        # lista de circulos
        circles1 = acha_circulos(frame1)
        circles2 = acha_circulos(frame2)

        # xy_circles_frame1 = circles1[0,:,:2]
        # xy_circles_frame2 = circles2[0,:,:2]


        cont_ball = 0
        # varia o numero de bolas encontradas
        for pt in circles1[0,:,:2]:# in range(len(xy1)):

            # contador de bolas
            cont_ball += 1

            # nome da variavel no dicionario de paths
            str_ball = 'ball_%s' %str(cont_ball).zfill(2)

            # acha valor mais proximo
            A = circles2[0,:,:2]
            distance, index = spatial.KDTree(A).query(pt)

            # print (index)

            pos = list(A[index])
            paths[str_ball].append(pos)

            # print (paths['ball_01'])

            # stop
            # stop


            # # se for o primeiro frame, coloca os valores do xy em cada bola
            # if i == 0:
            #     paths[str_ball] = 

            # # coloca no dicionario os 
            # pt_xy = xy1[i,:]
            # distance, index = spatial.KDTree(xy2).query(pt_xy)


            # balls_xy[str_ball] = [list(xy1[i])]

            # acha valor mais proximo
            # pt = circles1[0,0,:2]
            # pt = np.array(balls_xy[str_ball])
            # A = circles2[0,:,:][:,:2]
            # A = xy2
            # distance,index = spatial.KDTree(A).query(pt)
            # pos = list(A[index])
            # stop

            # balls_xy[str_ball].append(pos)

            # stop

######################################################################################
    # # plota_circulos(frame1, circles1)

    # # ensure at least some circles were found
    # if circles is not None:
    #     # convert the (x, y) coordinates and radius of the circles to integers
    #     circles = np.round(circles[0, :]).astype("int")
     
    #     # loop over the (x, y) coordinates and radius of the circles
    #     idb = 0
    #     for (x, y, r) in circles[np.argsort(circles[:,-1]),:]: #np.sort(circles):
    #         idb += 1
    #         # draw the circle in the output image, then draw a rectangle
    #         # corresponding to the center of the circle
    #         cv2.circle(output, (x, y), r, (0, 255, 0), 4)
    #         font = cv2.FONT_HERSHEY_SIMPLEX
    #         cv2.putText(output,str(idb),(x-10, y+10), font, 0.6, (255,255,255), 2, cv2.LINE_AA)

    #         # cv2.rectangle(output, (x - 5, y - 5), (x + 5, y + 5), (0, 128, 255), -1)
    #         # cv2.plot(output, (x, y), (0, 255, 0), 4)
     
    #     # show the output image
    #     # cv2.imshow("output", np.hstack([frame, output]))
######################################################################################


# cv2.imshow('detected circles',output)