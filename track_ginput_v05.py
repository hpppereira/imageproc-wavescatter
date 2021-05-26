"""
Fazer o track de cada bola (ou cluster) manualmente, utilizando
a ferramenta ginput

maquina de 30 fps
mudar o id da bola para cada video

henrique pereira
15/05/2018
"""

import os
import numpy as np
from glob import glob
import matplotlib.pyplot as plt
import pandas as pd
import cv2

plt.close('all')

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

def carrega_e_salva_frame(cap, nome_video, ff, pathname_out):
    """
    Entrada:
        framei - frame inicial (int)
    Saida:
        frame - matriz 3d do frame
    """
    cap.set(cv2.CAP_PROP_POS_FRAMES, ff)
    ret, frame = cap.read()

    # salva frame
    cv2.imwrite(pathname_out + 'frame_%s.png' %str(int(ff)).zfill(6), frame)
    return frame

def acha_xy_ginput(frame):

    # plotagem de cada frame
    plt.figure(figsize=(8,6))
    plt.imshow(frame)

    # acha a posicao xy
    p = plt.ginput(1)
    plt.close('all')
    return p

def salva_xy_bola(xy, nome_video, id_ball,pathname_out):
    xy = pd.DataFrame(xy[:-1], columns=['nframe','x','y'], index=None)
    xy = xy.astype(np.int)
    xy = xy.set_index('nframe')
    xy.to_csv(pathname_out + 'path_ball_%s.csv' %str(id_ball).zfill(2))
    return

def junta_paths_xy(pathname):
    balls_path = {}
    listarqs = np.sort(os.listdir(pathname))
    listout = []
    for fileout in listarqs:
        if fileout.endswith('.csv'):
            df = pd.read_csv(pathname + fileout, index_col='nframe')
            balls_path[fileout] = df
    return balls_path

def plota_xy_no_video(balls_path, frame):
    pass

if __name__ == '__main__':

    #id da bola a ser seguida (mudar para cada bola. o arquivo e salvo com esse id)
    id_ball = input('Entre com o ID da bola: ')

    # Dados de entrada
    nome_video = 'T100_010300_CAM1.avi'
    pathname_video = os.environ['HOME'] + '/Documents/wavescatter_videos/DERIVA_RANDOMICA/VIDEO/CAM1/T100/'
    pathname_out = os.environ['HOME'] + '/Documents/wavescatter/out/' + nome_video[:-4] + '/'

    get_xy = True
    plot_xy = True

    # dicionario com nome do video e tempo inicial e final em que as
    # bolinhas estao dentro da tela e ja se separaram (quando em cluster)
    dict_videos = {'T100_010100.CAM1.avi': ['00:38','01:54'],
                   'T100_010100.CAM1_ISOPOR.avi': ['00:13','01:38'],
                   'T100_010100_CAM1.avi': ['00:12','01:58'],
                   'T100_010200_CAM1.avi': ['00:11','01:56'],
                   'T100_010300_CAM1.avi': ['00:08','01:37'],
                   'T100_020100_CAM1.avi': ['00:08','01:30']}

    # ---------------------------------------------------------------------- #
    if get_xy == True:


        # intervalo entre frames em segundos
        ints = 5

        # execucao das funcoes
        cap, fps = carrega_video(pathname_video, nome_video)

        # intervalo entre frames (camera 30 fps)
        intf = ints * fps

        nframei, nframef = acha_frame_inicial_e_final(dict_videos, nome_video, fps)

        # loop para calcular a posicao xy em cada frame
        xy = [] # lista com posicoes xy da bola
        for ff in np.arange(nframei,nframef,intf):
            frame = carrega_e_salva_frame(cap, nome_video, ff, pathname_out)
            p = acha_xy_ginput(frame)

            # cria lista com posicoes
            xy.append([ff,p[0][0],p[0][1]])

            # finaliza o loop se clicar em x<20
            if p[0][0] < 20:
                break

        salva_xy_bola(xy, nome_video[:-4], id_ball, pathname_out)

    # ---------------------------------------------------------------------- #
    if plot_xy == True:

        #lista de frames
        lista_frames = np.sort(glob(pathname_out + '*.png'))

        # junta todos os paths num arquivo de dicionario
        balls_path = junta_paths_xy(pathname_out)

        # plotagem do path de todas as bolas em cada frame
        # em um frame,  plotar todos os paths
        f = 0
        for frame in lista_frames:
            f += 1
            plt.figure(figsize=(12,10))
            img = plt.imread(frame)
            plt.imshow(img)
            for nb in balls_path.keys():
                plt.plot(balls_path[nb].x[:f], balls_path[nb].y[:f],'-*')
                plt.savefig('%strack_%s.png' %(pathname_out, str(f).zfill(3)))
                # plt.show()
        plt.close('all')


