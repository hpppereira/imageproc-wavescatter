# Calcula distancia entre bolas
#
# Henrique Pereira (pereira.henriquep@gmail.com) 
# data de criação: 15/07/2019

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from proc_drifters import qualified_balls, calc_path
plt.close('all')

if __name__ == "__main__":

    # image scale (1px = 1.8 mm)
    pxmm = 1.8

    balls_tracked = qualified_balls()

    for experiment in list(balls_tracked.keys())[:1]:
        print (experiment)

        # path and filename
        pathname = os.environ['HOME'] + '/Documents/coppe/lioc/wavescatter/' \
                                        'output/paper_onr/paths/{}/'.format(experiment[6:])
        filename = '{}.pkl'.format(experiment)

        # read pickle with xy position
        xy = pd.read_pickle(pathname + filename)

    # # run all balls
    # if balls_tracked[experiment] == []:
    #     balls_qc = list(xy.keys())
    # # run specific balls
    # else:
    #     balls_qc = balls_tracked[experiment]

    # stop

    # create paths xy for each ball in mm
    path = {}
    for ball in balls_tracked[experiment]:
        path[ball] = calc_path(xy, ball, pxmm)
        # stop

    # dist = np.array([np.sqrt((path[i+1,0] - path[i,0])**2 + (path[i+1,1] - path[i,1])**2) for i in range(len(path)-1)])

    x1 = path['ball_04'][:,0]
    y1 = path['ball_04'][:,1]

    x2 = path['ball_22'][:,0]
    y2 = path['ball_22'][:,1]

    dist = np.sqrt( (x2 - x1)**2 + (y2 - y1)**2 )

    fig = plt.figure(figsize=(8,8))
    ax1 = fig.add_subplot(211)
    ax1.plot(x1, y1, label='ball_04')
    ax1.plot(x2, y2, label='ball_22')
    ax1.set_xlabel('tank dimension (millimeter)')
    ax1.set_ylabel('tank dimension (millimeter)')
    ax1.grid()
    ax1.legend()

    ax2 = fig.add_subplot(212)
    ax2.plot(dist, label='ball_04')
    ax2.set_xlabel('number of frames')
    ax2.set_ylabel('distance between balls (millimeter)')
    ax2.grid()
    # ax2.legend()

    plt.show()
