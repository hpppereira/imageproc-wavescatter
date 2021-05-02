# coding: utf-8

import cv2
import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from skimage import measure
import imageio
from scipy import ndimage, misc
from PIL import Image


def gerar_imagens_a_partir_do_video(arquivo_avi):
    cap = cv2.VideoCapture(arquivo_avi)

    for ff in np.arange(1000,2000,10):

        cap.set(cv2.CAP_PROP_POS_FRAMES, ff)

        # take first f rame of the video
        ret = True
        while ret:
            ret, frame = cap.read()    
            yield frame

def processar_contornos_da_imagem(frame):
    img = frame[250:-130,200:]
    gray = cv2.cvtColor(img ,cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(gray,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)

    # Find contours at a constant value ofzio 0.8
    contours = measure.find_contours(thresh, 100)
    return contours

def identificar_ponto_central(contorno):
    x = contorno[:, 1].mean()
    y = contorno[:, 0].mean()
    return x, y

def calcular_o_ponto_mais_proximo_de(a, lista_de_pontos):
    distancia_pra_a = []
    for p in lista_de_pontos:
        distancia_pra_a.append( ((p[0]-a[0])**2 + (p[1]-a[1])**2, p) )
    return sorted(distancia_pra_a)[0]
    
def cria_arquivo_de_saida(id_bola, n_frame, x, y):

    return saida

if __name__ == '__main__':
    n = 20 # numero de frames
    lista_de_pontos_centrais = []
    arquivo_avi = os.path.join('/home/hp/Documents/wavescatter_videos/DERIVA_RANDOMICA/VIDEO/CAM1/T100/', 'T100_010300_CAM1.avi') 
    for index, frame in enumerate(gerar_imagens_a_partir_do_video(arquivo_avi), 0):
        contornos = processar_contornos_da_imagem(frame)
        pontos_centrais = [ identificar_ponto_central(c) for c in processar_contornos_da_imagem(frame) ]
        lista_de_pontos_centrais.append(pontos_centrais)
        if index == n: break

    id_bola = []
    n_frame = []
    pos_x = []
    pos_y = []
    l = range(len(lista_de_pontos_centrais))
    for frame1, frame2 in zip(l, l[1:]):
        cont_bola = 0
        for pa in lista_de_pontos_centrais[frame1]:
            cont_bola += 1
            pb = calcular_o_ponto_mais_proximo_de(pa, lista_de_pontos_centrais[frame2])

            s1 =  u'Bola ID: {cont_bola}'.format(cont_bola=cont_bola)
            print (s1)

            s = u'O ponto equivalente do ponto ({xa}, {ya}) no frame {f1}, é o ponto ({xb}, {yb}) no frame {f2}'.format(xa=pa[0], ya=pa[1], f1=frame1, f2=frame2, xb=pb[0], yb=pb[1] )
            print (s)

            id_bola.append(cont_bola)
            n_frame.append(frame1)
            pos_x.append(pb[0])
            pos_y.append(pb[1])



