"""
proc_drifter.py - 19.3 kB

Description: Qualify the paths and calculate the mean
             and relative dispersion of the drifters.
Input: dictionary with path file and valid balls
Output: qualified paths

Functions:
    - calculate_paths_dists_vels
    - calculate_mean_path_xy
    - calculate_relative_dispersion
    - calculate_velocity_statistics
    - exponential_func
    - adjust_fit_rel_disp
    - plot_paths_vels
    - plot_distances
    - plot_distances_log
    - plot_mean_distances
    - plot_rel_disp
    - plot_adjust_rel_disp
    - plot_adjust_dist_t0
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import warnings
warnings.filterwarnings('ignore')
plt.close('all')

def calculate_paths_xy(xy, ball):
    paths_xy = np.array(xy[ball].tolist())
    return paths_xy

def calculate_paths_dists_vels(xy, ball, pxmm, fps):
    """
    Calculate paths, distances (mm) and velocities (mm/s) in a 2D dimension
    xy - position xy in time
    ball - string of ball name
    numframes - total number of frames
    distances in milimeters
    position in milimeters
    vels in milimters/second
    """

    # take a sample with initial and final frame number
    # xy = xy[numframes[0]:numframes[1]]

    # position x and y for each ball
    # x, y = np.array(xy[ball].tolist()).T

    # calculate distance between two consecutive frames
    # dists_xy = [np.sqrt((x[i+1] - x[i])**2 + (y[i+1] - y[i])**2) for i in range(len(x)-1)]
    # dists_x = [x[i+1] - x[i] for i in range(len(x)-1)]
    # dists_y = [y[i+1] - y[i] for i in range(len(y)-1)]

    # distance relative to x=x0
    # dists_xy_t0 = [np.sqrt((x[i] - x[0])**2 + (y[i] - y[0])**2) for i in range(len(x)-1)]
    # dists_x_t0 = [x[i] - x[0] for i in range(len(x)-1)]
    # dists_y_t0 = [y[i] - y[0] for i in range(len(y)-1)]

    # derivada da distancia (velocity) for quality control of paths_xy
    # dists_dif = pd.Series(np.diff(np.array([dists_xy[0] + dists_xy]))[0])

    # quality control

    # indicies dos arquivos qualificados
    # index_qc = np.where((dists_dif<105) & (dists_dif>-105) & (pd.Series(dists_xy)>=0.0) & 
    #                     (pd.Series(dists_xy)<1000))[0]

    # sem qualificacao
    # index_qc = np.where(dists_dif)[0]

    # distancias qualificadas
    # dists_qc = dists_dif.loc[index_qc]

    # dictionary with distances and paths xy for qualified time series of paths and convert to mm
    # paths_xy = np.array(xy[ball].tolist())[dists_qc.index] * pxmm

    # dists_xy = pd.Series(dists_xy)[dists_qc.index] * pxmm
    # dists_xy_t0 = pd.Series(dists_xy_t0)[dists_qc.index] * pxmm

    # dists_x = pd.Series(dists_x)[dists_qc.index] * pxmm
    # dists_x_t0 = pd.Series(dists_x_t0)[dists_qc.index] * pxmm

    # dists_y = pd.Series(dists_y)[dists_qc.index] * pxmm
    # dists_y_t0 = pd.Series(dists_y_t0)[dists_qc.index] * pxmm

    # vels_xy = dists_xy / (1./fps)
    # vels_x = dists_x / (1./fps)
    # vels_y = dists_y / (1./fps)

    # return paths_xy, dists_xy, dists_xy_t0, vels_xy, dists_x, dists_x_t0, vels_x, dists_y, dists_y_t0, vels_y, x, y, index_qc
    pass

def calculate_mean_path_xy(paths_xy):
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

    mean_path_xy = pd.DataFrame(np.array([x, y]).T, columns=['x','y'])
    mean_path_xy.index = mean_path_xy.index

    return mean_path_xy

def calculate_relative_dispersion(paths_xy, mean_path_xy):
    """
    Calculate the relative dispersion
    - For each ball
    - Mean of all balls
    """

    # mean square distance for each ball
    rel_disp = {}
    for ball in paths_xy.keys():
        # print (ball)
        x, y = paths_xy[ball][:,[0,1]].T
        rel_disp[ball] = (np.sqrt((x - mean_path_xy.iloc[:,0])**2 + (y - mean_path_xy.iloc[:,1])**2))**2

    # mean distance for all balls
    mean_rel_disp = pd.DataFrame(np.vstack(rel_disp.values()).mean(axis=0), index=mean_path_xy.index, columns=['Dt'])
    rel_disp = pd.DataFrame(rel_disp)

    return rel_disp, mean_rel_disp

def calculate_velocity_statistics(vels_xy):
    """
    Calculate mean velocity, max and standard deviation for all frames for all buoys
    input: velocities for all buoys during the video
    """

    # media da serie temporal de todas as bolas
    mean_vel_balls_xy = {}
    for ball in vels_xy.keys():
        mean_vel_balls_xy[ball] = vels_xy[ball].mean()
        print ('Ball {} - Mean: {}'.format(ball, mean_vel_balls_xy[ball]))

    # media das bolas
    mean_values = np.array([mean_vel_balls_xy[key] for key in mean_vel_balls_xy.keys()])

    # statistical values for balls velocities
    mean_vel_xy = np.mean(mean_values)
    std_vel_xy = np.std(mean_values)
    min_vel_xy = np.min(mean_values)
    max_vel_xy = np.max(mean_values)

    return mean_vel_xy, std_vel_xy, min_vel_xy, max_vel_xy

def exponential_func(t, x):

    return t ** x

def adjust_fit_rel_disp(xdata, ydata):
    """
    Adjust an exponential fit
    """
    popt, pcov = curve_fit(exponential_func, xdata, ydata)
    yy = exponential_func(xdata, *popt)

    return popt[0], pcov, yy

def plot_paths(experiment, paths_xy, mean_path_xy):
    """
    Plot paths for each ball
    """

    fig = plt.figure(figsize=(10,5))
    ax1 = fig.add_subplot(111)
    ax1.grid()
    ax1.set_xlabel('Position X (px)')
    ax1.set_ylabel('Position Y (px)')
    ax1.set_title(experiment)
    for ball in paths_xy.keys():
        ax1.plot(paths_xy[ball][:,0], paths_xy[ball][:,1]*-1,'b-', linewidth=0.4)
        ax1.text(paths_xy[ball][-1,0], paths_xy[ball][-1,1]*-1,ball[-2:])
    ax1.plot(mean_path_xy.iloc[:,0], mean_path_xy.iloc[:,1]*-1, 'r')
    # fig1.savefig('img/paths{}.png'.format(filename[-17:]), bbox_inches='tight')

    return fig

def plot_distances(filename, dists_xy_t0, dists_x_t0, dists_y_t0):

    fig1 = plt.figure(figsize=(12,6))
    fig1.suptitle('Distance from t=t0 - {}'.format(filename))

    ax1 = fig1.add_subplot(131)
    ax1.grid()
    ax1.set_xlabel('Time (frames)')
    ax1.set_ylabel('Distance (mm)')
    ax1.set_title('XY')
    ax1.plot(dists_xy_t0)
    ax1.set_ylim(0,2500)

    ax2 = fig1.add_subplot(132)
    ax2.grid()
    ax2.set_xlabel('Time (frames)')
    ax2.set_title('X')
    ax2.plot(dists_x_t0)
    ax2.set_ylim(0,2500)

    ax3 = fig1.add_subplot(133)
    ax3.grid()
    ax3.set_xlabel('Time (frames)')
    ax3.set_title('Y')
    ax3.plot(dists_y_t0)
    ax3.set_ylim(0,2500)
    # fig1.savefig('img/dists{}.png'.format(filename[-17:]))

    return

def plot_distances_log(filename, dists_xy_t0, dists_x_t0, dists_y_t0):

    fig1 = plt.figure(figsize=(12,6))
    fig1.suptitle('Distance from t=t0 - {}'.format(filename))

    ax1 = fig1.add_subplot(131)
    ax1.grid()
    ax1.set_xlabel('Time (frames)')
    ax1.set_ylabel('Distance (mm)')
    ax1.set_title('XY')
    ax1.loglog(dists_xy_t0)
    ax1.set_ylim(0,2500)

    ax2 = fig1.add_subplot(132)
    ax2.grid()
    ax2.set_xlabel('Time (frames)')
    ax2.set_title('X')
    ax2.loglog(dists_x_t0)
    ax2.set_ylim(0,2500)

    ax3 = fig1.add_subplot(133)
    ax3.grid()
    ax3.set_xlabel('Time (frames)')
    ax3.set_title('Y')
    ax3.loglog(dists_y_t0)
    ax3.set_ylim(0,2500)
    # fig1.savefig('img/dists_loglog{}.png'.format(filename[-17:]))

    return

def plot_mean_distances(filename, dists_xy_t0):

    fig1 = plt.figure(figsize=(10,6))
    fig1.suptitle('Distance from t=t0 - {}'.format(filename))

    ax1 = fig1.add_subplot(121)
    # ax1.grid()
    ax1.set_xlabel('Time (frames)')
    ax1.set_ylabel('Distance (mm)')
    # ax1.set_title('XY')
    ax1.loglog(dists_xy_t0)
    ax1.set_ylim(0,2500)

    ax2 = fig1.add_subplot(122)
    # ax2.grid()
    ax2.set_xlabel('Time (frames)')
    # ax2.set_title('X')
    ax2.loglog(dists_x_t0.mean(axis=1))
    ax2.set_ylim(0,2500)
    # fig1.savefig('img/mean_dists{}.png'.format(filename[-17:]))

    return

def plot_rel_disp(filename, rel_disp, mean_rel_disp):

    fig = plt.figure(figsize=(10,6))
    fig.suptitle('Relative Dispersion \n {}'.format(filename))
    ax1 = fig.add_subplot(121)
    ax1.set_xlabel('Time (frames)')
    ax1.set_ylabel('MSD (px)')
    # ax1.set_title(filename)
    rel_disp.plot(loglog=False, ax=ax1, legend=None)
    ax1.grid()

    ax2 = fig.add_subplot(122)
    # ax2.set_ylabel('Mean Relative Dispersion')
    # ax2.set_title(filename)
    mean_rel_disp.plot(loglog=False, ax=ax2, legend=None)
    ax2.set_xlabel('Time (frames)')
    ax2.grid()
    # fig1.savefig('img/rel_disp{}.png'.format(filename[-17:]))

    return fig

def plot_adjust_rel_disp(filename, mean_rel_disp, xdata, popt):

    fig1 = plt.figure(figsize=(10,6))
    ax1 = fig1.add_subplot(111)
    ax1.set_title(filename)
    ax1.loglog(mean_rel_disp)
    ax2 = ax1.twinx()
    ax2.loglog(xdata, xdata ** popt, 'r')
    # fig1.savefig('img/fit_rel_disp{}.png'.format(filename[-17:]))
    return

def plot_adjust_dist_t0(filename, mean_dists_xy, xdata, popt):

    fig1 = plt.figure(figsize=(10,6))
    ax1 = fig1.add_subplot(111)
    ax1.set_title(filename)
    ax1.loglog(mean_dists_xy)
    ax2 = ax1.twinx()
    ax2.loglog(xdata, xdata ** popt, 'r')
    # fig1.savefig('img/fit_dist_t0{}.png'.format(filename[-17:]))

    return

def qualified_balls():

    # list of balls without errors
    balls_tracked = {
                        'paths_T100_010300': ['ball_03', 'ball_37'],
                        'paths_T100_020100': ['ball_34', 'ball_13', 'ball_31'],
                        'paths_T100_020201': ['ball_45', 'ball_18', 'ball_05', 'ball_20'],
                        'paths_T100_020300': ['ball_38', 'ball_26'],
                        'paths_T100_030100': ['ball_32', 'ball_17', 'ball_10', 'ball_26', 'ball_22', 'ball_14', 'ball_03'],
                        'paths_T100_030200': ['ball_33', 'ball_15', 'ball_26', 'ball_04', 'ball_08'],
                        'paths_T100_030300': ['ball_35', 'ball_25'],
                        'paths_T100_040100': ['ball_45', 'ball_44', 'ball_13', 'ball_03', 'ball_12'],
                        'paths_T100_040300': ['ball_33', 'ball_11', 'ball_32', 'ball_43', 'ball_26', 'ball_18', 'ball_44'],
                        'paths_T100_050100': ['ball_04', 'ball_29', 'ball_28', 'ball_03', 'ball_22', 'ball_20', 'ball_33'],
                        'paths_T100_050300': ['ball_08', 'ball_33', 'ball_21', 'ball_15'],
                        }

    # # initial and final frame (number of frames)
    numframes = {
                'paths_T100_010300': [0, 500],
                'paths_T100_020100': [0, 621],
                'paths_T100_020201': [0, 500],
                'paths_T100_020300': [0, 611],
                'paths_T100_030100': [0, 611],
                'paths_T100_030200': [0, 771],
                'paths_T100_030300': [0, 631],
                'paths_T100_040100': [0, 1251],
                'paths_T100_040300': [0, 1081],
                'paths_T100_050100': [0, 721],
                'paths_T100_050300': [0, 721],
                }

    # # initial and final time for adjustment (frames)
    # adjust_keys = {
    #                'paths_T100_010300_CAM1': [7, 16],
    #                'paths_T100_020100_CAM1': [10, 35],
    #                'paths_T100_020201_CAM1': [3, 31],
    #                'paths_T100_020300_CAM1': [10, 22],
    #                'paths_T100_030100_CAM1': [11, 24],
    #                'paths_T100_030200_CAM1': [10, 40],
    #                'paths_T100_030300_CAM1': [10, 33],
    #                'paths_T100_040100_CAM1': [11, 32],
    #                'paths_T100_040300_CAM1': [10, 31],
    #                'paths_T100_050100_CAM1': [17, 21],
    #                'paths_T100_050200_CAM1': [2, 10], #X
    #                'paths_T100_050300_CAM1': [14, 25]
    #                }

    return balls_tracked, numframes


if __name__ == '__main__':

    balls_tracked, numframes = qualified_balls()

    # image scale (1px = 1.8 mm)
#    pxmm = 1.54

    # frames per second
#    fps = 30.0

    for experiment in list(balls_tracked.keys()):

        # path and filename
        pathname = os.environ['HOME'] + '/gdrive/coppe/lioc/wavescatter/mosaico/output/{}/'.format(experiment[6:])
        pathname_fig = os.environ['HOME'] + '/gdrive/coppe/lioc/wavescatter/mosaico/figs/'
        filename = '{}.pkl'.format(experiment[:-4])

        # read pickle with xy position
        xy = pd.read_pickle(pathname + filename)

        # escolhe frames iniciais e finais
        xy = xy[numframes[experiment][0]:numframes[experiment][1]]

        # list with balls
        # balls = list(xy.keys())

        # run all balls
        if balls_tracked[experiment] == []:
            balls_qc = list(xy.keys())
        else:
            balls_qc = balls_tracked[experiment]

        # numframes = numframes_keys[filename]

        # variables create inside a function
        paths_xy = {}

        # create paths xy for each ball
        for ball in balls_qc:
            paths_xy[ball] = calculate_paths_xy(xy, ball)
            # print ('Bola: ' + ball)
            # print ('Numero de amostragens em cada bola: {}'.format(paths_xy[ball].shape))

        # dists_xy = {}
        # dists_xy_t0 = {}
        # vels_xy = {}
        # dists_x = {}
        # dists_x_t0 = {}
        # vels_x = {}
        # dists_y = {}
        # dists_y_t0 = {}
        # vels_y = {}

        # # calculate path for each ball (loop for each ball)
        # for ball in balls:
        #     if ball in balls_tracked:
        #         # print ('{}..Processed'.format(ball))
        #         paths_xy[ball], dists_xy[ball], dists_xy_t0[ball], vels_xy[ball], \
        #         dists_x[ball], dists_x_t0[ball], vels_x[ball], \
        #         dists_y[ball], dists_y_t0[ball], vels_y[ball], x, y, index_qc = calculate_paths_dists_vels(xy, ball, pxmm, fps)
        #     else:
        #         # print ('{}..Error'.format(ball))
        #         pass

        # create dataframes with times as index
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
        mean_path_xy = calculate_mean_path_xy(paths_xy)
        # stop

        # calculate relative dispersion for each ball and mean
        rel_disp, mean_rel_disp = calculate_relative_dispersion(paths_xy, mean_path_xy)

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
        fig1 = plot_paths(experiment, paths_xy, mean_path_xy)
        # plot_distances(filename, dists_xy_t0, dists_x_t0, dists_y_t0)
        # plot_mean_distances(filename, dists_xy_t0)
        fig2 = plot_rel_disp(experiment, rel_disp, mean_rel_disp)

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

        # save paths qualified
        # paths_xy_qc.to_csv(pathname + 'paths_qc_{}.csv'.format(experiment[6:]))
        # paths_xy_qc.to_csv(pathname + 'paths_qc_{}.csv'.format(experiment[6:]))
        # paths_xy_qc.to_pickle(pathname + 'paths_qc_{}.pkl'.format(experiment[6:]))
        # dists_xy.to_csv('data/qc/dists{}.csv'.format(filename[-17:]))
        # dists_xy_t0.to_csv('data/dists_xy_t0{}.csv'.format(filename[-17:]))
        # vels_xy.to_csv('data/qc/vels{}.csv'.format(filename[-17:]))

        # save figures
        # fig1.savefig(pathname_fig + 'paths_{}.png'.format(experiment[6:]))
        # fig2.savefig(pathname_fig + 'reldisp_{}.png'.format(experiment[6:]))

        # save mean relative dispersion
        mean_rel_disp.to_csv('mean_rel_disp_{}.csv'.format(experiment), index_label='nframe')
        # mean_rel_disp.to_csv(pathname + 'mean_rel_disp_{}.csv'.format(experiment))


    plt.show()
    # plt.close('all')
