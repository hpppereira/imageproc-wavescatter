# Description: Track drifters released in a wave tank.
# Input: video file with initial and final time
# Output: paths with x and y position

import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import cv2
from importlib import reload  
import wavescatter
reload(wavescatter)

if __name__ == '__main__':

    # pathname
    pathname = '/media/hp/HIGGINS/wavescatter/'

    # cria dicionario com as bolas qualificadas
    balls_tracked, time_videos = wavescatter.qualified_balls()

    # loop dos experimentos
    for filename in list(time_videos.keys()):

        # create directory path and fig outuput
        os.system('mkdir {}paths/'.format(pathname))
        os.system('mkdir {}figs/frames/{}'.format(pathname, filename))

        # leitura do video
        cap, fps = wavescatter.read_video(pathname + '/VIDEOS/', filename + '.avi')

        # numeros dos frames iniciais e finais do video e a duracao total em segundos
        nframei, nframef, durs = wavescatter.find_first_and_last_frames(time_videos, filename, fps)

        # imprime tempo do video em segundos
        print ('{} -- Total seconds: {}'.format(filename, durs))

        # loop para cada frame
        cont_frame = 0
        for ff in range(nframei, nframef, 1):

            # imprime frame a ser processado
            print ('frame {ff} de {nframef}'.format(ff=ff, nframef=nframef))

            # contador de frames
            cont_frame += 1

            # read two consecutives frames
            frame1 = wavescatter.read_frame(cap, ff)
            frame2 = wavescatter.read_frame(cap, ff+1)

            # copy of original frames
            output1 = frame1.copy()
            output2 = frame2.copy()

            # preprocessing the frames to detect circles
            gray1, gray2, frame1, frame2 = wavescatter.frames_preproc(frame1, frame2)

            # list of circles (balls)
            circles1 = wavescatter.find_circles(frame1)
            circles2 = wavescatter.find_circles(frame2)

            # only the circles identified in the first frame will
            # be tracked along the video
            if cont_frame == 1:
                paths = wavescatter.find_initial_balls(circles1)

            # condicao para processar enquanto tiver bola na imagem
            if circles2 is not None:

                # plot figure to be saved
                fig = plt.figure(figsize=(10,10))
                ax = fig.add_subplot(111)
                ax.imshow(output1)
                fig.tight_layout()

                # loop to track each ball
                for ball in paths.keys():

                    # xy position of each ball
                    xy_ball = wavescatter.track_min_dist(paths, ball, circles2)

                    # create a list with paths for each ball
                    paths[ball].append(xy_ball)

                    # cria array com os trajetos de cada bola
                    a = np.array(paths[ball])

                    plt.plot(a[:,0], a[:,1],'-', linewidth=2.5)
                    plt.text(a[:,0][-1], a[:,1][-1], ball[-2:], color='w')

                # loop to convert list of xy to array (to save csv with dataframe)
                paths_xy = {}
                for path_key in paths.keys():
                    paths_xy[path_key] = np.vstack(paths[path_key])

                # save the figure in png
                fig.savefig('{}figs/frames/{}/{}_{}'.format(pathname, filename, filename,
                            str(cont_frame).zfill(4)), bbox_inches='tight')
                plt.close('all')

        # realease the video obj
        cap.release()

        # save path with xy for each ball
        df = pd.DataFrame(paths)
        df.index.name = 'nframe'
        df.to_csv('{}paths/paths_raw_{}.csv'.format(pathname, filename))
        df.to_pickle('{}paths/paths_raw_{}.pkl'.format(pathname, filename))