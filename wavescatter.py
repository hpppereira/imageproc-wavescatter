# funcoes para processamento dos dados do wavescatter

import numpy as np
import pandas as pd
import cv2
from scipy import spatial

def qualified_balls():
    """
    Linha dos experimentos e bolas qualificadas
    balls without errors
    """
    balls_tracked = {
    'T100_010300_CAM1': ['ball_04', 'ball_22', 'ball_00', 'ball_16', 'ball_06',
                               'ball_02', 'ball_01', 'ball_15',
                               'ball_11', 'ball_05', 'ball_10', 'ball_13', 'ball_12', 
                                'ball_03', 'ball_08'],

    'T100_020100_CAM1': ['ball_19', 'ball_21', 'ball_24', 'ball_12', 'ball_26',
                               'ball_31', 'ball_01', 'ball_00', 'ball_02', 'ball_06',
                               'ball_05','ball_03','ball_11'],

    'T100_020201_CAM1': ['ball_00', 'ball_13', 'ball_01', 'ball_04', 'ball_15',
                               'ball_25', 'ball_14', 'ball_05', 'ball_03',
                               'ball_11', 'ball_23', 'ball_06', 'ball_29', 'ball_28',
                               'ball_09'],

    'T100_020300_CAM1': ['ball_07', 'ball_15', 'ball_03', 'ball_16', 'ball_09',
                               'ball_13', 'ball_14', 'ball_11', 'ball_06', 'ball_00',],

    'T100_030100_CAM1': ['ball_27', 'ball_04', 'ball_00', 'ball_24', 'ball_31',
                               'ball_20', 'ball_05', 'ball_10', 'ball_32', 'ball_25',
                               'ball_30', 'ball_06', 'ball_22', 'ball_12', 'ball_17',
                               'ball_16', 'ball_33', 'ball_13', 'ball_26', 'ball_14',
                               'ball_03', 'ball_28', 'ball_07', 'ball_08', 'ball_02',
                               'ball_01', 'ball_19',],

    'T100_030200_CAM1': ['ball_10', 'ball_06', 'ball_22', 'ball_24', 'ball_13',
                               'ball_03', 'ball_08', 'ball_09', 'ball_20', 'ball_25',
                               'ball_07', 'ball_15', 'ball_23', 'ball_11', 'ball_17'],

    'T100_030300_CAM1': ['ball_14', 'ball_17', 'ball_19', 'ball_24', 'ball_23',
                               'ball_04', 'ball_27', 'ball_30', 'ball_00',
                                'ball_32', 'ball_26', 'ball_01',
                               'ball_29', 'ball_13', 'ball_28', 'ball_12', 'ball_22',],
                               
    'T100_040100_CAM1': ['ball_10', 'ball_02', 'ball_26', 'ball_03',
                               'ball_34', 'ball_00', 'ball_08', 'ball_11',
                               'ball_28', 'ball_35', 'ball_32', 'ball_21', 'ball_27',
                               'ball_06'],

    'T100_040300_CAM1': ['ball_19', 'ball_01', 'ball_04', 'ball_08',
                               'ball_12', 'ball_17', 'ball_27', 'ball_14',
                               'ball_09', 'ball_26', 'ball_18', 'ball_21', 'ball_24',
                               'ball_06', 'ball_31', 'ball_00', 'ball_22'],

    'T100_050100_CAM1': ['ball_25', 'ball_04', 'ball_06', 'ball_02', 'ball_01',
                               'ball_15', 'ball_20', 'ball_21', 'ball_27', 'ball_30',
                               'ball_16', 'ball_11', 'ball_00', 'ball_29',
                               'ball_13', 'ball_08', 'ball_24'],

    'T100_050200_CAM1': ['ball_26', 'ball_05', 'ball_16',
                               'ball_01', 'ball_00', 'ball_08', 'ball_11',
                               'ball_04', 'ball_02', 'ball_07', 'ball_19', 'ball_27',
                               'ball_22', 'ball_28', 'ball_09', 'ball_18', 'ball_32',
                               'ball_03'],

    'T100_050300_CAM1': ['ball_02', 'ball_01', 'ball_24', 'ball_13', 'ball_17',
                               'ball_18', 'ball_12', 'ball_27', 'ball_16', 'ball_08']}

    time_videos = {
                   'T100_010300_CAM1': ['00:08','00:53'],
                   'T100_020100_CAM1': ['00:08','00:55'], 
                   'T100_020201_CAM1': ['00:09','00:57'],
                   'T100_020300_CAM1': ['00:08','00:51'],
                   'T100_030100_CAM1': ['00:12','00:54'],
                   'T100_030200_CAM1': ['00:09','01:00'],
                   'T100_030300_CAM1': ['00:08','00:53'],
                   'T100_040100_CAM1': ['00:08','01:37'],
                   'T100_040300_CAM1': ['00:09','01:14'],
                   'T100_050100_CAM1': ['00:07','01:04'],
                   'T100_050200_CAM1': ['00:09','00:50'],
                   'T100_050300_CAM1': ['00:08','00:58'],
                   }
    return balls_tracked, time_videos

def read_video(pathname, filename):
    cap = cv2.VideoCapture(pathname + filename)
    fps = cap.get(cv2.CAP_PROP_FPS)
    return cap, fps

def read_frame(cap, ff):
    cap.set(cv2.CAP_PROP_POS_FRAMES, ff)
    ret, frame = cap.read()
    frame1 = frame[250:-130,:]
    # if ncam == 1:
        # frame1 = frame
        # frame1 = frame[250:-130,:]
    # elif ncam == 2:
        # frame1 = frame
        # frame1 = cv2.rotate(frame, rotateCode = cv2.ROTATE_90_COUNTERCLOCKWISE)
        # frame = frame[:950,:]
        # frame1 = imutils.rotate(frame, 90)
        # frame = frame_rot[:,430:1420]
    return frame1

def read_paths_pickle(pathname):
    """
    Leitura dos dados das trajetorias xy em
    pickle
    """
    # informacoes de escala e frequencia de amostragem dos videos
    # escala com base no tamanho da bola de plastico 1 px = 1.8 ou vice versa (verificar)
    pxmm = 1.8

    # read pickle with xy position
    xy = pd.read_pickle(pathname)

    # create paths xy for each ball
    path_pkl = {}
    for ball in xy.keys():
        path_pkl[ball] = np.array(xy[ball].tolist())# * pxmm
    return path_pkl

def find_first_and_last_frames(time_videos, filename, fps):
    """
    Acha o numero do frame inicial e final 
    de cada video com base nos minituos e segundos
    escolhidos pelo dicionario 'time_videos'
    """
    # first and final time in datetime format
    dtime = pd.to_datetime(time_videos[filename], format='%M:%S')
    # time in timedelta (to convert to total_seconds)
    timei = dtime[0] - pd.Timestamp('1900')
    timef = dtime[1] - pd.Timestamp('1900')
    # video duration in time_delta format
    dur = dtime[1] - dtime[0]
    # video duration in seconds
    durs = dur.total_seconds()
    # number of first and last frames to be reaed (based of fps)
    nframei = int(timei.total_seconds() * fps)
    nframef = int(timef.total_seconds() * fps)
    return nframei, nframef, durs

def frames_preproc(frame1, frame2):

    # convert frames to gray scale
    gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

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
    paths = {}
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
    if distance > 30:
        xy_ball = pt # pega o ponto anterior
    else:
        xy_ball = list(A[index].astype(int))
    return xy_ball