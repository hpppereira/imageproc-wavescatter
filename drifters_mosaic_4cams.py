"""
track_drifter.py - 9.7 kB

Description: Track drifters released in a wave tank.
Input: video file with initial and final time
Output: paths with x and y position

Functions:
    - read_video
    - read_frame
    - find_first_and_last_frames
    - frames_preproc
    - find_circles
    - find_initial_balls
    - track_min_dist
    - make_gif
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import cv2
import pandas as pd
from scipy import spatial
#import imutils

plt.close('all')
cv2.destroyAllWindows()

def read_video(pathname, filename):
    cap = cv2.VideoCapture(pathname + filename)
    fps = cap.get(cv2.CAP_PROP_FPS)
    return cap, fps

def read_frame(cap, ff, ncam):
    cap.set(cv2.CAP_PROP_POS_FRAMES, ff)
    ret, frame = cap.read()
    if ncam == 1:
        frame1 = frame
        # frame = frame[250:-130,:]
    elif ncam == 2:
        # frame1 = frame
        frame1 = cv2.rotate(frame, rotateCode = cv2.ROTATE_90_COUNTERCLOCKWISE)
        # frame = frame[:950,:]
        # frame1 = imutils.rotate(frame, 90)
        # frame = frame_rot[:,430:1420]
    elif ncam == 3:
        # frame1 = frame
        frame1 = cv2.rotate(frame, rotateCode = cv2.ROTATE_180)
    elif ncam == 4:
        # frame1 = frame
        frame1 = frame

    return frame1

def find_first_and_last_frames(dict_videos, filename, fps):

    # first and final time in datetime format
    dtime = pd.to_datetime(dict_videos[filename], format='%M:%S')

    dtimevec = pd.date_range(dtime[0], dtime[1], freq='.1S')

    # time in timedelta (to convert to total_seconds)
    timei = dtime[0] - pd.Timestamp('1900')
    timef = dtime[1] - pd.Timestamp('1900')
    dtimevec = dtimevec - pd.Timestamp('1900')

    # video duration in time_delta format
    dur = dtime[1] - dtime[0]

    # video duration in seconds
    durs = dur.total_seconds()

    # number of first and last frames to be reaed (based of fps)
    nframei = int(timei.total_seconds() * fps)
    nframef = int(timef.total_seconds() * fps)

    return nframei, nframef, dtimevec

def create_mosaic(g, gg, ggg, gggg, ):

    """
    g1, g2 = frames em tons de cinza
    """

    m0 = np.zeros((2160, 4920)) * np.nan

    # mosaico sem sobreposicao
    # m0[540:1620, 0:1920] = g
    # m0[120:2040, 1920:3000] = gg
    # m0[0:1080, 3000:4920] = ggg
    # m0[1080:2160, 3000:4920] = gggg

    m0[540:1620, 0:1920] = g
    m0[120-101:2040-101, 1920-401:3000-401] = gg
    m0[0+106:1080+106, 3000-505:4920-505] = ggg
    m0[1080-120:2160-120, 3000-376:4920-376] = gggg

    # m0 = np.zeros((g2.shape[0], g1.shape[0] + g2.shape[0]))# * np.nan
    # m0[420+112-20+11+2-10:-420+112-20+11+2-10,:1920] = g1
    # m0[:,1920-357-5+4:-357-5+4] = g2
    return m0

def frames_preproc(gray1, gray2):

    # convert frames to gray scale
    # gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    # gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

    # smooth the frames
    frame1 = cv2.GaussianBlur(gray1,(5,5),0)
    frame2 = cv2.GaussianBlur(gray2,(5,5),0)

    # remove the background using threshold
    ret, frame1 = cv2.threshold(frame1,70,255,cv2.THRESH_BINARY)
    ret, frame2 = cv2.threshold(frame2,70,255,cv2.THRESH_BINARY)

    return gray1, gray2, frame1, frame2

def find_circles(frame):
    """
    Find circles with HoughCircles
    """

    circles = cv2.HoughCircles(image=frame,
                               method=cv2.HOUGH_GRADIENT,
                               dp=1,
                               minDist=40,
                               param1=10,
                               param2=5,
                               minRadius=10,
                               maxRadius=20)

    return circles

def find_initial_balls(circles1):
    """
    Find initial balls
    """

    # list of circles to be tracked
    balls_xy = circles1[0,:,:2]
    for ball_id in range(len(balls_xy)):
        paths['ball_{ball_id}'.format(ball_id=str(ball_id).zfill(2))]  = [list(balls_xy[ball_id].astype(int))]

    return paths

def track_min_dist(paths, ball, circles2):
    """
    Track balls with minimum distance
    """

    # point x and y for one  ball for the frame_i
    pt = paths[ball][-1]

    # all point x and y for one ball for the frame_i+1
    A = circles2[0,:,:2]

    # calculates the distance and index
    distance, index = spatial.KDTree(A).query(pt)

    # print (distance, index)

    if distance > 30:
        # print (distance)
        # pass
        xy_ball = pt # pega o ponto anterior
    else:
        xy_ball = list(A[index].astype(int))

    return xy_ball

if __name__ == '__main__':

    track_balls = True
    plot_balls_paths = True
    save_balls_paths = True # save dict in csv and pck file

    if track_balls:
      print ('Fazer rastremento: OK')
    if plot_balls_paths:
      print ('Plotagem do rastreio: OK')
    if save_balls_paths:
      print ('Salva rastreio: OK')

    # dicionario com nome do video e tempo inicial e final em que as
    # bolinhas estao dentro da tela e ja se separaram (quando em cluster)
    dict_videos = {
                   'T100_040300': ['00:09','05:00'],
                   }

    for experiment in dict_videos.keys():

        print ('Iniciando processamento no experimento: {}'.format(experiment))

        # numero da camera
#        ncam = int(filename_video[-5])

#        pathname_video = os.environ['HOME'] + '/gdrive/coppe/lioc/wavescatter_data/DERIVA_RANDOMICA/VIDEO/CAM{}/T100/'.format(ncam)
        pathname_frames = os.environ['HOME'] + '/Documents/wavescatter/frames/{}/'.format(experiment)
        pathname_output = os.environ['HOME'] + '/Documents/wavescatter/output/{}/'.format(experiment)

        # create directory for fig outuput
        os.system('mkdir {}'.format(pathname_frames))
        os.system('mkdir {}'.format(pathname_output))

        # ------------------------------------------------------------------------ #
        # Start program

        if track_balls == True:

            print ('Carregando parametros das cameras...')

            filename_v1 = experiment + '_CAM1.avi'
            filename_v2 = experiment + '_CAM2.avi'
            filename_v3 = experiment + '_CAM3.avi'
            filename_v4 = experiment + '_CAM4.avi'

            pathname_v1 = os.environ['HOME'] + '/Documents/wavescatter_data/DERIVA_RANDOMICA/VIDEO/CAM1/T100/'
            pathname_v2 = os.environ['HOME'] + '/Documents/wavescatter_data/DERIVA_RANDOMICA/VIDEO/CAM2/T100/'
            pathname_v3 = os.environ['HOME'] + '/Documents/wavescatter_data/DERIVA_RANDOMICA/VIDEO/CAM3/T100/'
            pathname_v4 = os.environ['HOME'] + '/Documents/wavescatter_data/DERIVA_RANDOMICA/VIDEO/CAM4/T100/'

            cap1, fps1 = read_video(pathname=pathname_v1, filename=filename_v1)
            cap2, fps2 = read_video(pathname=pathname_v2, filename=filename_v2)
            cap3, fps3 = read_video(pathname=pathname_v3, filename=filename_v3)
            cap4, fps4 = read_video(pathname=pathname_v4, filename=filename_v4)

            if cap1.isOpened():
              print ('Leitura da camera 1 OK')
            if cap2.isOpened():
              print ('Leitura da camera 2 OK')
            if cap3.isOpened():
              print ('Leitura da camera 3 OK')
            if cap4.isOpened():
              print ('Leitura da camera 4 OK')

            # achar os frames inicial e final para um dos videos
            nframei, nframef, dtimevec = find_first_and_last_frames(dict_videos, experiment, fps2)

            cont_frame = 0
            # cont_frame1 = 0
            # cont_frame2 = 0
            paths = {}

            # for ff in range(nframei, nframef, 1):
            # for i in np.arange(len(dtimevec))[:-30]:
            for i in np.arange(0,len(dtimevec),1):

                # acha o frame para cada um dos videos
                # nframei1, nframef1 = find_first_and_last_frames(dict_videos, experiment, fps1)
                # nframei2, nframef2 = find_first_and_last_frames(dict_videos, experiment, fps2)
                nframe1 = int(dtimevec[i].total_seconds() * fps1)
                nframe11 = int(dtimevec[i].total_seconds() * fps2)
                nframe111 = int(dtimevec[i].total_seconds() * fps3)
                nframe1111 = int(dtimevec[i].total_seconds() * fps4)

                nframe2 = int(dtimevec[i+1].total_seconds() * fps1)
                nframe22 = int(dtimevec[i+1].total_seconds() * fps2)
                nframe222 = int(dtimevec[i+1].total_seconds() * fps3)
                nframe2222 = int(dtimevec[i+1].total_seconds() * fps4)

                # print (nframei1, nframei2)

                cont_frame += 1
               # cont_frame1 += 1
#                cont_frame2 += 1 
#                cont_frame2 = cont_frame2 - 6

                # stop

                print ('Iniciando o tracking frame a frame...')

                # cria mosaico frame 'i'
                f1 = read_frame(cap1, nframe1, ncam=1)
                print ('Leitura do video 1 - OK')
                f11 = read_frame(cap2, nframe11, ncam=2)
                print ('Leitura do video 2 - OK')
                f111 = read_frame(cap3, nframe111, ncam=3)
                print ('Leitura do video 3 - OK')
                f1111 = read_frame(cap4, nframe1111, ncam=4)
                print ('Leitura do video 4 - OK')

                g1 = cv2.cvtColor(f1, cv2.COLOR_BGR2GRAY)
                g11 = cv2.cvtColor(f11, cv2.COLOR_BGR2GRAY)
                g111 = cv2.cvtColor(f111, cv2.COLOR_BGR2GRAY)
                g1111 = cv2.cvtColor(f1111, cv2.COLOR_BGR2GRAY)

                frame1 = create_mosaic(g1, g11, g111, g1111)

                # cria mosaico frame 'i+1'
                f2 = read_frame(cap1, nframe2, ncam=1)
                f22 = read_frame(cap2, nframe22, ncam=2)
                f222 = read_frame(cap3, nframe222, ncam=3)
                f2222 = read_frame(cap4, nframe2222, ncam=4)

                g2 = cv2.cvtColor(f2, cv2.COLOR_BGR2GRAY)
                g22 = cv2.cvtColor(f22, cv2.COLOR_BGR2GRAY)
                g222 = cv2.cvtColor(f222, cv2.COLOR_BGR2GRAY)
                g2222 = cv2.cvtColor(f2222, cv2.COLOR_BGR2GRAY)

                frame2 = create_mosaic(g2, g22, g222, g2222)

                # coloca zeros nos escritos do frame
                # frame1[1450:, :1560] = 0
                # frame1[:, 2500:] = 0
                # frame2[1450:, :1560] = 0
                # frame2[:, 2500:] = 0

                frame1 = frame1.astype('uint8')
                frame2 = frame2.astype('uint8')

                print ('Mosaico criado!')

                #print ('frame {ff} de {nframef}'.format(ff=ff, nframef=nframef))

                # contador de bolas

                # read two consecutives frames
                #frame1 = read_frame(cap, ff, ncam)
                #frame2 = read_frame(cap, ff+1, ncam)

                # save a copy of original frames
                output1 = frame1.copy()
                output2 = frame2.copy()

                # preprocessing the frames to detect circles
                gray1, gray2, frame1, frame2 = frames_preproc(frame1, frame2)

                print ('Pre-processamento dos frames realizado!')

                # list of circles (balls)
                circles1 = find_circles(frame1)
                circles2 = find_circles(frame2)

                print ('Circulos detectados!')

                # only the circles identified in the first frame will be tracked along the video
                if cont_frame == 1:
                    paths = find_initial_balls(circles1)

                # condicao para parar quando nao tiver mais bola na imagem
                if circles2 is not None:

                    # plot figure to be saved
                    if plot_balls_paths == True:
                        plt.figure(figsize=(10,10))
                        plt.imshow(output1)
                        plt.tight_layout()

                    # loop to track each ball
                    for ball in paths.keys():
                        # xy position of each ball
                        xy_ball = track_min_dist(paths, ball, circles2)
                        # create a list with paths for each ball
                        paths[ball].append(xy_ball)

                        # plot figure with paths (plot for each frame all balls)
                        if plot_balls_paths == True:
                            a = np.array(paths[ball])
                            plt.plot(a[:,0], a[:,1],'-', linewidth=2.5)
                            plt.text(a[:,0][-1], a[:,1][-1], ball[-2:], color='w')

                    # loop to convert list of xy to array (to save csv with dataframe)
                    paths_xy = {}
                    for path_key in paths.keys():
                        paths_xy[path_key] = np.vstack(paths[path_key])

                    # save the figure in png
                    if plot_balls_paths == True:                        
                        print ('Salvando Figura')
                        plt.savefig(pathname_frames + 'frame_{cont_frame}'.format(cont_frame=str(cont_frame).zfill(4)), bbox_inches='tight')
                        # plt.show()
                        plt.close('all')

            # realease the video obj
            cap1.release()
            cap2.release()

        # save path with xy for each ball
        if save_balls_paths:
            df = pd.DataFrame(paths)
            # df.index = np.arange(nframei, nframei+30+1)/30
            df.index.name = 'nframe'
            df.to_csv(pathname_output + 'paths_%s.csv' %experiment[:-4])
            df.to_pickle(pathname_output + 'paths_%s.pkl' %experiment[:-4])
