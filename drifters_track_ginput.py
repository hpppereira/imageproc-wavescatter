"""
Fazer o track de cada bola (ou cluster) manualmente, utilizando
a ferramenta ginput

Henrique Pereira
"""

import os
import numpy as np
from glob import glob
import matplotlib.pyplot as plt
import pandas as pd
import cv2

plt.close('all')

if __name__ == '__main__':

    pathname = os.environ['HOME'] + '/Documents/wavescatter_data/T100_040300_CAM1.avi'

    cap = cv2.VideoCapture(pathname)

    fps = cap.get(cv2.CAP_PROP_FPS)

    # intervalo entre frames (camera 30 fps)
    intf = 1 * 30.0

    # intervalo de amostragem
    timei = 3 * 30 # 3 segundos
    timef = 2 * 60 * 38 * 30 # 2:38

    # loop para calcular a posicao xy em cada frame
    xy = [] # lista com posicoes xy da bola

    for ff in np.arange(timei ,timei + 1):#timef, intf):

        # frame = carrega_e_salva_frame(cap, nome_video, ff, pathname_out)
        cap.set(cv2.CAP_PROP_POS_FRAMES, ff)
        ret, frame = cap.read()
        frame = frame[250:-130,:]

        # salva frame
        # cv2.imwrite(pathname_out + 'frame_%s.png' %str(int(ff)).zfill(6), frame)

        # p = acha_xy_ginput(frame)
        plt.figure(figsize=(8,6))
        plt.imshow(frame)

    xy = pd.DataFrame(xy[:-1], columns=['nframe','x','y'], index=None)
    xy = xy.astype(np.int)
    xy = xy.set_index('nframe')

    # salva paths
    # xy.to_csv('data/paths_T100_040300_CAM1_ball_31_manual.csv')
    # xy.to_pickle('data/paths_T100_040300_CAM1_ball_31_manual.pkl')
