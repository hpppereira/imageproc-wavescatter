"""
Main functions for drifters processing
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from proc_paths_xy import qualified_balls

plt.close('all')

def calc_path(xy, ball, pxmm):
    """
    Calcula as posicoes x e y de cada derivador
    e coloca escala em milimetros
    """
    path = np.array(xy[ball].tolist()) * pxmm
    return path

def calc_dist_veloc(path, fps=30):
    """
    Calcula distancias e velocidades entre 
    2 frames consecutivos em milimetros    
    """
    # time berween 2 frames
    t = 1. / fps

    # calculate distance between two consecutive frames
    dist = np.array([np.sqrt((path[i+1,0] - path[i,0])**2 + (path[i+1,1] - path[i,1])**2) for i in range(len(path)-1)])
    dist0 = np.array([np.sqrt((path[i+1,0] - path[0,0])**2 + (path[i+1,1] - path[0,1])**2) for i in range(len(path)-1)])

    # calculate velocity
    velo = np.array(dist) / t
    velo0 = np.array(dist0) / t

    return dist, dist0, velo, velo0

def autocorr(x):
    result = np.correlate(x, x, mode='full')
    return result[int(result.size/2):]

def velo_autocorr(x):
    """
    Calcula a autocorrelacao com base no
    artigo que o nelson mandou
    """
    tals = np.arange(x.shape[0]-1)

    R = []
    for tal in tals:
        R.append(x)
    return ac

# def calculate_paths_dists_vels(xy, ball, pxmm, fps):
#     """
#     Calculate paths, distances (mm) and velocities (mm/s) in a 2D dimension
#     xy - position xy in time
#     ball - string of ball name
#     numframes - total number of frames
#     distances in milimeters
#     position in milimeters
#     vels in milimters/second
#     """

#     # take a sample with initial and final frame number
#     # xy = xy[numframes[0]:numframes[1]]

#     # position x and y for each ball
#     x, y = np.array(xy[ball].tolist()).T

#     # calculate distance between two consecutive frames
#     dists_xy = [np.sqrt((x[i+1] - x[i])**2 + (y[i+1] - y[i])**2) for i in range(len(x)-1)]
#     dists_x = [x[i+1] - x[i] for i in range(len(x)-1)]
#     dists_y = [y[i+1] - y[i] for i in range(len(y)-1)]

#     # distance relative to x=x0
#     dists_xy_t0 = [np.sqrt((x[i] - x[0])**2 + (y[i] - y[0])**2) for i in range(len(x)-1)]
#     dists_x_t0 = [x[i] - x[0] for i in range(len(x)-1)]
#     dists_y_t0 = [y[i] - y[0] for i in range(len(y)-1)]

#     # derivada da distancia (velocity) for quality control of paths_xy
#     dists_dif = pd.Series(np.diff(np.array([dists_xy[0] + dists_xy]))[0])

#     # quality control

#     # indicies dos arquivos qualificados
#     index_qc = np.where((dists_dif<105) & (dists_dif>-105) & (pd.Series(dists_xy)>=0.0) & (pd.Series(dists_xy)<1000))[0]

#     # distancias qualificadas
#     dists_qc = dists_dif.loc[index_qc]

#     # dictionary with distances and paths xy for qualified time series of paths and convert to mm
#     paths_xy = np.array(xy[ball].tolist())[dists_qc.index] * pxmm

#     dists_xy = pd.Series(dists_xy)[dists_qc.index] * pxmm
#     dists_xy_t0 = pd.Series(dists_xy_t0)[dists_qc.index] * pxmm

#     dists_x = pd.Series(dists_x)[dists_qc.index] * pxmm
#     dists_x_t0 = pd.Series(dists_x_t0)[dists_qc.index] * pxmm

#     dists_y = pd.Series(dists_y)[dists_qc.index] * pxmm
#     dists_y_t0 = pd.Series(dists_y_t0)[dists_qc.index] * pxmm

#     vels_xy = dists_xy / (1./fps)
#     vels_x = dists_x / (1./fps)
#     vels_y = dists_y / (1./fps)

#     return paths_xy, dists_xy, dists_xy_t0, vels_xy, dists_x, dists_x_t0, vels_x, dists_y, dists_y_t0, vels_y

def calculate_mean_path_xy(paths_xy, fps):
    """
    Calculate the mean position x y - position x and y
    """
    x = []
    y = []
    for ball in paths_xy.keys():
        x.append(paths_xy[ball][:,0])
        y.append(paths_xy[ball][:,1])
    x = np.vstack(x).mean(axis=0)
    y = np.vstack(y).mean(axis=0)
    mean_path_xy = pd.DataFrame(np.array([x, y]).T)
    mean_path_xy.index = mean_path_xy.index / fps
    return mean_path_xy

def calculate_relative_dispersion(paths_xy, mean_path_xy):
    """
    Calculate the relative dispersion
    - For each ball
    - Mean of all balls
    """
    # mean square distance for each ball
    rel_disp = {}
    rel_disp_t0 = {} # D(t) com a referencia do M(t=0)
    dist = {}
    dist_t0 = {}
    for ball in paths_xy.keys():
        # print (ball)
        x, y = paths_xy[ball][:,[0,1]].T

        # relative dispersion
        rel_disp[ball] = (np.sqrt((x - mean_path_xy.iloc[:,0])**2 + (y - mean_path_xy.iloc[:,1])**2))**2
        rel_disp_t0[ball] = (np.sqrt((x - mean_path_xy.iloc[0,0])**2 + (y - mean_path_xy.iloc[0,1])**2))**2

        # distance 
        dist[ball] = np.sqrt((x - mean_path_xy.iloc[:,0])**2 + (y - mean_path_xy.iloc[:,1])**2)
        dist_t0[ball] = np.sqrt((x - mean_path_xy.iloc[0,0])**2 + (y - mean_path_xy.iloc[0,1])**2)

    # mean distance for all balls
    df_rel_disp = pd.DataFrame(rel_disp)
    df_mean_rel_disp = df_rel_disp.mean(axis=1)

    df_rel_disp_t0 = pd.DataFrame(rel_disp_t0, index=df_rel_disp.index)
    df_mean_rel_disp_t0 = df_rel_disp_t0.mean(axis=1)

    df_dist = pd.DataFrame(dist, index=df_rel_disp.index)
    df_mean_dist = df_dist.mean(axis=1)

    df_dist_t0 = pd.DataFrame(dist_t0, index=df_rel_disp.index)
    df_mean_dist_t0 = df_dist_t0.mean(axis=1)
    # return , mean_rel_disp, rel_disp_t0, mean_rel_disp_t0
    return df_rel_disp, df_mean_rel_disp, df_rel_disp_t0, df_mean_rel_disp_t0, df_dist, df_mean_dist, df_dist_t0, df_mean_dist_t0


# def calculate_velocity_statistics(vels_xy):
#     """
#     Calculate mean velocity, max and standard deviation for all frames for all buoys
#     input: velocities for all buoys during the video
#     """

#     # media da serie temporal de todas as bolas
#     mean_vel_balls_xy = {}
#     for ball in vels_xy.keys():
#         mean_vel_balls_xy[ball] = vels_xy[ball].mean()
#         print ('Ball {} - Mean: {}'.format(ball, mean_vel_balls_xy[ball]))

#     # media das bolas
#     mean_values = np.array([mean_vel_balls_xy[key] for key in mean_vel_balls_xy.keys()])

#     # statistical values for balls velocities
#     mean_vel_xy = np.mean(mean_values)
#     std_vel_xy = np.std(mean_values)
#     min_vel_xy = np.min(mean_values)
#     max_vel_xy = np.max(mean_values)

#     return mean_vel_xy, std_vel_xy, min_vel_xy, max_vel_xy

# def exponential_func(t, x):

#     return t ** x

# def adjust_fit_rel_disp(xdata, ydata):
#     """
#     Adjust an exponential fit
#     """
#     popt, pcov = curve_fit(exponential_func, xdata, ydata)
#     yy = exponential_func(xdata, *popt)

#     return popt[0], pcov, yy


# def plot_distances(dist0):

#     fig1 = plt.figure(figsize=(12,6))
#     ax1 = fig1.add_subplot(131)
#     ax1.grid()
#     ax1.set_xlabel('Time (seconds)')
#     ax1.set_ylabel('Distance (mm)')
#     ax1.set_title('Distance')
#     ax1.plot(dist0)
    # ax1.set_ylim(0,2500)

    # ax2 = fig1.add_subplot(132)
    # ax2.grid()
    # ax2.set_xlabel('Time (seconds)')
    # ax2.set_title('X')
    # ax2.plot(dists_x_t0)
    # ax2.set_ylim(0,2500)

    # ax3 = fig1.add_subplot(133)
    # ax3.grid()
    # ax3.set_xlabel('Time (seconds)')
    # ax3.set_title('Y')
    # ax3.plot(dists_y_t0)
    # ax3.set_ylim(0,2500)
    # fig1.savefig('img/dists{}.png'.format(filename[-17:]))
    # return fig

# def plot_distances_log(filename, dists_xy_t0, dists_x_t0, dists_y_t0):

#     fig1 = plt.figure(figsize=(12,6))
#     fig1.suptitle('Distance from t=t0 - {}'.format(filename))

#     ax1 = fig1.add_subplot(131)
#     ax1.grid()
#     ax1.set_xlabel('Time (seconds)')
#     ax1.set_ylabel('Distance (mm)')
#     ax1.set_title('XY')
#     ax1.loglog(dists_xy_t0)
#     ax1.set_ylim(0,2500)

#     ax2 = fig1.add_subplot(132)
#     ax2.grid()
#     ax2.set_xlabel('Time (seconds)')
#     ax2.set_title('X')
#     ax2.loglog(dists_x_t0)
#     ax2.set_ylim(0,2500)

#     ax3 = fig1.add_subplot(133)
#     ax3.grid()
#     ax3.set_xlabel('Time (seconds)')
#     ax3.set_title('Y')
#     ax3.loglog(dists_y_t0)
#     ax3.set_ylim(0,2500)
#     # fig1.savefig('img/dists_loglog{}.png'.format(filename[-17:]))

#    return

# def plot_mean_distances(filename, dists_xy_t0):

#     fig1 = plt.figure(figsize=(10,6))
#     fig1.suptitle('Distance from t=t0 - {}'.format(filename))

#     ax1 = fig1.add_subplot(121)
#     # ax1.grid()
#     ax1.set_xlabel('Time (seconds)')
#     ax1.set_ylabel('Distance (mm)')
#     # ax1.set_title('XY')
#     ax1.loglog(dists_xy_t0)
#     ax1.set_ylim(0,2500)

#     ax2 = fig1.add_subplot(122)
#     # ax2.grid()
#     ax2.set_xlabel('Time (seconds)')
#     # ax2.set_title('X')
#     ax2.loglog(dists_x_t0.mean(axis=1))
#     ax2.set_ylim(0,2500)
#     # fig1.savefig('img/mean_dists{}.png'.format(filename[-17:]))

    # return

def plot_paths(experiment, paths_xy, mean_path_xy):
    """
    Plot paths for each ball
    """

    fig = plt.figure(figsize=(10,5))
    ax1 = fig.add_subplot(111)
    ax1.grid()
    # ax1.set_xlabel('Position X')
    # ax1.set_ylabel('Position Y \n Wavemaker')
    ax1.set_title(experiment)
    for ball in paths_xy.keys():
        ax1.plot(paths_xy[ball][:,0], paths_xy[ball][:,1]*-1,'b-', linewidth=0.4)
        # ax1.text(paths_xy[ball][-1,0], paths_xy[ball][-1,1]*-1, ball[-2:])
    ax1.plot(mean_path_xy.iloc[:,0], mean_path_xy.iloc[:,1]*-1, 'r')
    ax1.plot(mean_path_xy.iloc[0,0], mean_path_xy.iloc[0,1]*-1, 'ko', markersize=6)
    # fig1.savefig('img/paths{}.png'.format(filename[-17:]), bbox_inches='tight')

    return fig

# def plot_dist(t, dist):
#     """
#     """
#     fig = plt.figure(figsize=(10,5))
#     ax1 = fig.add_subplot(111)
#     ax1.grid()
#     ax1.set_xlabel('Time (seconds)')
#     ax1.set_ylabel('Distance (mm)')
#     ax1.set_title(experiment)
#     for ball in dist.keys():
#         ax1.plot(t, dist[ball],'b-', linewidth=0.4)
#         # ax1.text(paths_xy[ball][-1,0], paths_xy[ball][-1,1]*-1, ball[-2:])
#     ax1.plot(t, pd.DataFrame(dist).mean(axis=1), 'r')
#     return fig

def plot_dist(dist):
    """
    """
    fig = plt.figure(figsize=(10,5))
    ax1 = fig.add_subplot(111)
    ax1.grid()
    ax1.set_xlabel('Time (seconds)')
    ax1.set_ylabel('Distance (mm)')
    ax1.set_title(experiment)
    for ball in dist.keys():
        ax1.plot(dist[ball].index, dist[ball],'b-', linewidth=0.4)
        # ax1.text(dist.index[-1], dist[ball].iloc[-1], ball[-2:])
    ax1.plot(dist[ball].index, dist.mean(axis=1), 'r')
    return fig

def plot_velo(velo):
    """
    """
    fig = plt.figure(figsize=(10,5))
    ax1 = fig.add_subplot(111)
    ax1.grid()
    ax1.set_xlabel('Time (seconds)')
    ax1.set_ylabel('Velocity (mm/s)')
    ax1.set_title(experiment)
    for ball in velo.keys():
        ax1.plot(velo[ball].index, velo[ball],'b-', linewidth=0.4)
        # ax1.text(paths_xy[ball][-1,0], paths_xy[ball][-1,1]*-1, ball[-2:])
    ax1.plot(velo[ball].index, velo.mean(axis=1), 'r')
    return fig

def plot_autocorr(x):
    """
    """
    # plot_acf(x, ax=None, lags=None, alpha=0.05, use_vlines=True, unbiased=False, fft=False, title='Autocorrelation', zero=True, **kwargs)
    fig = plt.figure(figsize=(10,5))
    ax1 = fig.add_subplot(111)
    ax1.grid()
    ax1.set_xlabel('Lag')
    ax1.set_ylabel('Autocorrelation')
    ax1.set_title(experiment)
    ax1.plot(x/x.max())
    return fig


def plot_rel_disp(experiment, rel_disp, mean_rel_disp, rel_disp_t0, mean_rel_disp_t0):
    """
    """
    fig1 = plt.figure(figsize=None)
    fig1.suptitle('Relative Dispersion \n {}'.format(experiment))
    ax1 = fig1.add_subplot(111)
    ax1.set_xlabel('Time (seconds)')
    ax1.set_ylabel('D(t)')
    # ax1.set_title(filename)
    rel_disp.plot(loglog=False, ax=ax1, legend=None, color='b', alpha=0.5)
    mean_rel_disp.plot(loglog=False, ax=ax1, legend=None, color='r', alpha=1)
    ax1.grid()

    fig2 = plt.figure(figsize=None)
    fig2.suptitle('Relative Dispersion (D(t=0)) \n {}'.format(experiment))
    ax1 = fig2.add_subplot(111)
    ax1.set_xlabel('Time (seconds)')
    ax1.set_ylabel('D(t)')
    # ax1.set_title(filename)
    rel_disp_t0.plot(loglog=False, ax=ax1, legend=None, color='b', alpha=0.5)
    mean_rel_disp_t0.plot(loglog=False, ax=ax1, legend=None, color='r', alpha=1)
    ax1.grid()

    # fig3 = plt.figure(figsize=None)
    # fig3.suptitle('Relative Dispersion \n {}'.format(experiment))
    # ax1 = fig3.add_subplot(111)
    # ax1.set_xlabel('Time (seconds)')
    # ax1.set_ylabel('D(t)')
    # # ax1.set_title(filename)
    # rel_disp.plot(loglog=True, ax=ax1, legend=None, color='b', alpha=0.5)
    # mean_rel_disp.plot(loglog=True, ax=ax1, legend=None, color='r', alpha=1)
    # ax1.grid()

    # fig4 = plt.figure(figsize=None)
    # fig4.suptitle('Relative Dispersion (D(t=0)) \n {}'.format(experiment))
    # ax1 = fig4.add_subplot(111)
    # ax1.set_xlabel('Time (seconds)')
    # ax1.set_ylabel('D(t)')
    # # ax1.set_title(filename)
    # rel_disp_t0.plot(loglog=True, ax=ax1, legend=None, color='b', alpha=0.5)
    # mean_rel_disp_t0.plot(loglog=True, ax=ax1, legend=None, color='r', alpha=1)
    # ax1.grid()

    return fig1, fig2#, fig3, fig4

def plot_r(experiment, dist, mean_dist, dist_t0, mean_dist_t0):
    """
    """
    fig1 = plt.figure(figsize=None)
    fig1.suptitle('Distance from mean position \n {}'.format(experiment))
    ax1 = fig1.add_subplot(111)
    ax1.set_xlabel('Time (seconds)')
    ax1.set_ylabel('mm')
    # ax1.set_title(filename)
    dist.plot(loglog=False, ax=ax1, legend=None, color='b', alpha=0.5)
    mean_dist.plot(loglog=False, ax=ax1, legend=None, color='r', alpha=1)
    ax1.grid()

    fig2 = plt.figure(figsize=None)
    fig2.suptitle('Distance from mean position at t=t0 \n {}'.format(experiment))
    ax1 = fig2.add_subplot(111)
    ax1.set_xlabel('Time (seconds)')
    ax1.set_ylabel('mm')
    # ax1.set_title(filename)
    dist_t0.plot(loglog=False, ax=ax1, legend=None, color='b', alpha=0.5)
    mean_dist_t0.plot(loglog=False, ax=ax1, legend=None, color='r', alpha=1)
    ax1.grid()
    return fig1, fig2

# def plot_rel_disp_loglog(experiment, rel_disp, mean_rel_disp):

#     fig = plt.figure(figsize=None)
#     fig.suptitle('Relative Dispersion \n {}'.format(experiment))
#     ax1 = fig.add_subplot(111)
#     ax1.set_xlabel('Time (seconds)')
#     ax1.set_ylabel('D(t)')
#     # ax1.set_title(filename)
#     rel_disp.plot(loglog=True, ax=ax1, legend=None, color='b', alpha=0.5)
#     mean_rel_disp.plot(loglog=True, ax=ax1, legend=None, color='r', alpha=1)
#     ax1.grid()

    # return fig

# def plot_rel_disp1(experiment, rel_disp, mean_rel_disp):

#     fig = plt.figure(figsize=(10,6))
#     fig.suptitle('Relative Dispersion \n {}'.format(experiment))
#     ax1 = fig.add_subplot(121)
#     ax1.set_xlabel('Time (seconds)')
#     ax1.set_ylabel('MSD')
#     # ax1.set_title(filename)
#     rel_disp.plot(loglog=True, ax=ax1, legend=None)
#     ax1.grid()

#     ax2 = fig.add_subplot(122)
#     # ax2.set_ylabel('Mean Relative Dispersion')
#     # ax2.set_title(filename)
#     mean_rel_disp.plot(loglog=True, ax=ax2, legend=None)
#     ax2.set_xlabel('Time (seconds)')
#     ax2.grid()
#     # fig1.savefig('img/rel_disp{}.png'.format(filename[-17:]))

#     return fig

# def plot_adjust_rel_disp(filename, mean_rel_disp, xdata, popt):

#     fig1 = plt.figure(figsize=(10,6))
#     ax1 = fig1.add_subplot(111)
#     ax1.set_title(filename)
#     ax1.loglog(mean_rel_disp)
#     ax2 = ax1.twinx()
#     ax2.loglog(xdata, xdata ** popt, 'r')
#     # fig1.savefig('img/fit_rel_disp{}.png'.format(filename[-17:]))
#     return

# def plot_adjust_dist_t0(filename, mean_dists_xy, xdata, popt):

#     fig1 = plt.figure(figsize=(10,6))
#     ax1 = fig1.add_subplot(111)
#     ax1.set_title(filename)
#     ax1.loglog(mean_dists_xy)
#     ax2 = ax1.twinx()
#     ax2.loglog(xdata, xdata ** popt, 'r')
#     # fig1.savefig('img/fit_dist_t0{}.png'.format(filename[-17:]))

#     return


# if __name__ == '__main__':

#     balls_tracked = qualified_balls()

#     # image scale (1px = 1.8 mm)
#     # pxmm = 1.54
#     pxmm = 1.8

#     # frames per second
#     fps = 30.0

#     for experiment in list(balls_tracked.keys()):

#         print (experiment)

#         # path and filename
#         pathname = '/home/hp/gdrive/coppe/lioc/wavescatter/output/CAM1/{}/'.format(experiment[6:])
#         # pathname_fig = os.environ['HOME'] + '/gdrive/coppe/lioc/wavescatter/output/paper_onr/20191216_v07/fig/'
#         # pathname_out = os.environ['HOME'] + '/gdrive/coppe/lioc/wavescatter/output/paper_onr/20191216_v07/out/'
#         filename = '{}.pkl'.format(experiment)

#         # get values from a dict keys
#         # balls_tracked = balls_tracked_keys[experiment]
#         # print (len(balls_tracked))
#         # numframes = numframes_keys[filename]

#         # read pickle with xy position
#         xy = pd.read_pickle(pathname + filename)


        # teste para avaliar os tracks
        balls_tracked[experiment] = []


        # xy = xy[numframes[experiment][0]:numframes[experiment][1]]

        # list with balls
        # balls = list(xy.keys())

        # run all balls
        if balls_tracked[experiment] == []:
            balls_qc = list(xy.keys())
        # run specific balls
        else:
            balls_qc = balls_tracked[experiment]

        print (len(balls_qc))

        # variables create inside a function
        path = {}
        dist = {}
        dist0 = {}
        # dists_xy_t0 = {}
        velo = {}
        velo0 = {}
        # dists_x = {}
        # dists_x_t0 = {}
        # vels_x = {}
        # dists_y = {}
        # dists_y_t0 = {}
        # vels_y = {}

        fig = plt.figure(figsize=(10,4))
        ax1 = fig.add_subplot(121)
        ax2 = fig.add_subplot(122)

        # create paths xy for each ball
        for ball in balls_qc:
            print (ball)
            path[ball] = calc_path(xy, ball, pxmm)
            # dist[ball], dist0[ball], velo[ball], velo0[ball] = calc_dist_veloc(path[ball], fps=30.0)
            # print ('Bola: ' + ball)
            # print ('Numero de amostragens em cada bola: {}'.format(paths_xy[ball].shape))
            # ax.plot(path[ball][:,0], path[ball][:,1])
            # ax.plot(velo[ball])

        #     if np.max(np.abs(np.diff(dist[ball]))) < 50: # or (np.diff(dist[ball]).all() == 0) ):

        #         # if 0 not in np.sum(np.diff(dist0[ball][-100:]):

        #         fig.suptitle(experiment)
        #         ax1.plot(dist[ball], label=ball)
        #         ax1.set_title('Distance per frame')
        #         ax1.set_xlabel('Number of frames')
        #         ax1.set_ylabel('Distance (mm)')

        #         ax2.plot(dist0[ball], label=ball)
        #         ax2.set_title('Distance from initial position')
        #         ax2.set_xlabel('Number of frames')
        #         ax2.set_ylabel('Distance (mm)')

        # fig.savefig('distances.pdf')

        # plt.show()



        # calculate path for each ball (loop for each ball)
        # for ball in balls:
        #     if ball in balls_tracked:
        #         # print ('{}..Processed'.format(ball))
        #         paths_xy[ball], dists_xy[ball], dists_xy_t0[ball], vels_xy[ball], \
        #         dists_x[ball], dists_x_t0[ball], vels_x[ball], \
        #         dists_y[ball], dists_y_t0[ball], vels_y[ball] = calculate_paths_dists_vels(xy, ball, pxmm, fps)
        #     else:
        #         # print ('{}..Error'.format(ball))
        #         pass

        # # create dataframes with times as index
        # dists_xy = pd.DataFrame(dists_xy)
        # dists_x = pd.DataFrame(dists_x)
        # dists_y = pd.DataFrame(dists_y)

        # dists_xy_t0 = pd.DataFrame(dists_xy_t0)
        # dists_x_t0 = pd.DataFrame(dists_x_t0)
        # dists_y_t0 = pd.DataFrame(dists_y_t0)

        # vels_xy = pd.DataFrame(vels_xy)
        # vels_x = pd.DataFrame(vels_x)
        # vels_y = pd.DataFrame(vels_y)

        # dists_xy.index = dists_xy.index / fps
        # dists_xy_t0.index = dists_xy_t0.index / fps
        # vels_xy.index = vels_xy.index / fps
        # dists_x.index = dists_x.index / fps
        # dists_x_t0.index = dists_x_t0.index / fps
        # vels_x.index = vels_x.index / fps
        # dists_y.index = dists_y.index / fps
        # dists_y_t0.index = dists_y_t0.index / fps
        # vels_y.index = vels_y.index / fps

        # time vector
        # times = np.array(dists_x.index)

        # calculate mean path
        # mean_path = calculate_mean_path_xy(path, fps)

        # calculate relative dispersion for each ball and mean
        # rel_disp, mean_rel_disp, rel_disp_t0, mean_rel_disp_t0 = calculate_relative_dispersion(paths_xy, mean_path_xy)
        # df_rel_disp, df_mean_rel_disp, df_rel_disp_t0, df_mean_rel_disp_t0, df_dist, df_mean_dist, df_dist_t0, df_mean_dist_t0 = calculate_relative_dispersion(path, mean_path)

        # converte velocidade para dataframe com index igual ao tempo em segundos
        # df_velo = pd.DataFrame(velo, index=df_rel_disp.index[:-1])
        # df_velo0 = pd.DataFrame(velo0, index=df_rel_disp.index[:-1])
        # df_dist = pd.DataFrame(dist, index=df_rel_disp.index[:-1])
        # df_dist0 = pd.DataFrame(dist0, index=df_rel_disp.index[:-1])

        # fig = plot_velo(df_velo)
        # fig.savefig(pathname_fig + 'velo_{}.png'.format(experiment[6:]))

        # a = autocorr(df_velo.mean(axis=1))

        # fig = plot_autocorr(a)
        # fig.savefig(pathname_fig + 'autocorr_{}.png'.format(experiment[6:]))


        # D(t)
        # adjust fit for mean relative dispersion
        # a, b = adjust_keys[filename]
        # xdata = mean_rel_disp[a:b].index.values
        # ydata = mean_rel_disp[a:b].values[:,0]
        # popt_dt, pcov, yy = adjust_fit_rel_disp(xdata, ydata)
        # print ('D(t): {:.1f}'.format(popt_dt))

        # M(t)
        # a = 0.2
        # xdata = dists_xy_t0[a:].index.values
        # ydata = dists_xy_t0[a:].mean(axis=1)
        # popt_mt, pcov, yy = adjust_fit_rel_disp(xdata, ydata)
        # print ('M(t): {:.1f}'.format(popt_mt))

        # calculate statistics from velocity time series of each ball
        # mean_vel_xy, std_vel_xy, min_vel_xy, max_vel_xy = calculate_velocity_statistics(vels_xy)
        # print ('Mean: {:.2f}, STD: {:.2f}, Min: {:.2f}, Max: {:.2f}'.format(mean_vel_xy, std_vel_xy, min_vel_xy, max_vel_xy))

        # plot_adjust_rel_disp(filename, mean_rel_disp, xdata, popt_dt)
        # plot_adjust_dist_t0(filename, dists_xy_t0.mean(axis=1), xdata, popt_mt)

        # plot_distances(filename, dists_xy_t0, dists_x_t0, dists_y_t0)

        # plot_mean_distances(filename, dists_xy_t0)

        # plota os paths
        # fig1 = plot_paths(experiment, path, mean_path)

        # plota Dt
        # fig2, fig3 = plot_rel_disp(experiment, df_rel_disp, df_mean_rel_disp, df_rel_disp_t0, df_mean_rel_disp_t0)

        # fig4, fig5 = plot_r(experiment, df_dist, df_mean_dist, df_dist_t0, df_mean_dist_t0)

        # fig6 = plot_dist(df_dist0)

        # plota sem retirar a media
        # fig4 = plot_rel_disp_loglog(experiment, df_rel_disp, df_mean_rel_disp)

        # plota retirando a media
        # fig4 = plot_rel_disp(experiment, rel_disp1, mean_rel_disp1)

        # plota retirando a media com loglog
        # fig5 = plot_rel_disp_loglog(experiment, rel_disp1, mean_rel_disp1)

        # calcula velocidade media
        # dist_total = dists_xy_t0.mean(axis=1).iloc[-1]
        # time_total = dists_xy_t0.index[-1]
        # mean_vel_total = (dist_total / time_total) / 1000.0 # em metros
        # print ('Velocidade media total (m/s): {:.3f}'.format(mean_vel_total))

        # create paths_xy qualified
        # paths_xy_qc = {}
        # for ball in paths_xy.keys():
        #     paths_xy_qc[ball] = paths_xy[ball].tolist()
        # paths_xy_qc = pd.DataFrame(paths_xy_qc)

        # # save paths qualified
        # paths_xy_qc.to_csv('output/paths_qc_{}.csv'.format(experiment[6:]))
        # paths_xy_qc.to_pickle('output/paths_qc_{}.pkl'.format(experiment[6:]))
        # dists_xy.to_csv('data/qc/dists{}.csv'.format(filename[-17:]))
        # dists_xy_t0.to_csv('data/dists_xy_t0{}.csv'.format(filename[-17:]))
        # vels_xy.to_csv('data/qc/vels{}.csv'.format(filename[-17:]))

        # save figures
        # pathname_fig = 'fig/'
        # fig1.savefig(pathname_fig + 'paths_{}.png'.format(experiment[6:]))
        # fig2.savefig(pathname_fig + 'rel_disp_{}.png'.format(experiment[6:]))
        # fig3.savefig(pathname_fig + 'rel_disp_t0_{}.png'.format(experiment[6:]))
        # fig4.savefig(pathname_fig + 'dist_{}.png'.format(experiment[6:]))
        # fig5.savefig(pathname_fig + 'dist_t0_{}.png'.format(experiment[6:]))
        # fig6.savefig(pathname_fig + 'dist0_{}.png'.format(experiment[6:]))

        # save mean relative dispersion
        # pathname_out = 'out/'

        # df_rel_disp.to_csv('{}rel_disp_{}.csv'.format(pathname_out, experiment), index_label='%time')
        # df_mean_rel_disp.to_csv('{}mean_rel_disp_{}.csv'.format(pathname_out, experiment), index_label='%time', header=['mean_rel_disp'])

        # df_rel_disp_t0.to_csv('{}rel_disp_t0_{}.csv'.format(pathname_out, experiment), index_label='%time')
        # df_mean_rel_disp_t0.to_csv('{}mean_rel_disp_t0_{}.csv'.format(pathname_out, experiment), index_label='%time', header=['mean_rel_disp'])

        # df_dist.to_csv('{}dist_{}.csv'.format(pathname_out, experiment), index_label='%time')
        # df_mean_dist.to_csv('{}mean_dist_{}.csv'.format(pathname_out, experiment), index_label='%time', header=['mean_rel_disp'])

        # df_dist_t0.to_csv('{}dist_t0_{}.csv'.format(pathname_out, experiment), index_label='%time')
        # df_mean_dist_t0.to_csv('{}mean_dist_t0_{}.csv'.format(pathname_out, experiment), index_label='%time', header=['mean_rel_disp'])

        # df_dist0.to_csv('{}dist0_{}.csv'.format(pathname_out, experiment), index_label='%time')
        # df_velo.to_csv('{}velo_{}.csv'.format(pathname_out, experiment), index_label='%time')
        # mean_path.to_csv('{}mean_path_{}.csv'.format(pathname_out, experiment), index_label='%time', header=['x', 'y'])

        # plt.close('all')
        # plt.show()