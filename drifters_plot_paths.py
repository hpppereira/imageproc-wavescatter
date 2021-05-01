# Plota a trajetoria qualificada junto com os frames

import pandas as pd
from importlib import reload  
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import wavescatter
reload(wavescatter)
plt.close('all')

if __name__ == "__main__":

    pathname1 = '/media/hp/HIGGINS/ensino/coppe/lioc/wavescatter/VIDEOS/'
    pathname2 = '/media/hp/HIGGINS/ensino/coppe/lioc/wavescatter/output/paths/'
    pathname3 = '/media/hp/HIGGINS/ensino/coppe/lioc/wavescatter/output/figs/painel_posicoes/'

    # cria dicionario com as bolas qualificadas
    balls_tracked, time_videos = wavescatter.qualified_balls()

    # loop dos experimentos
    for experiment in list(balls_tracked.keys()):
        print (experiment)

        # leitura do viedeo
        cap, fps = wavescatter.read_video(pathname1, experiment + '.avi')
        nframei, nframef, durs = wavescatter.find_first_and_last_frames(time_videos, experiment, fps)
        frame = wavescatter.read_frame(cap, nframef)

        # leitura das trajetorias
        paths_qc = wavescatter.read_paths_pickle(pathname2 + 'paths_qc_{}.pkl'.format(experiment))

        fig = plt.figure(figsize=(8,6))
        spec2 = gridspec.GridSpec(ncols=2, nrows=2, figure=fig)
        ax1 = fig.add_subplot(spec2[0, 0:2])
        ax2 = fig.add_subplot(spec2[1, 0])
        ax3 = fig.add_subplot(spec2[1, 1])
        ax1.set_title(experiment)
        # ax1.imshow(frame)
        for ball in paths_qc.keys():
            ax1.plot(paths_qc[ball][:,0], paths_qc[ball][:,1], linewidth=.7)
            ax1.text(paths_qc[ball][:,0][-1], paths_qc[ball][:,1][-1], ball[-2:], color='w')
            ax2.plot(paths_qc[ball][:,0])
            ax3.plot(paths_qc[ball][:,1])
        ax1.invert_yaxis()
        ax1.grid()        
        ax1.set_ylabel('Frames em X')
        ax1.set_ylabel('Frames em Y')
        ax2.set_ylim(0, 2000)
        ax2.set_ylabel('Frames em X')
        ax2.set_xlabel('Num. Frames')
        ax2.grid()
        ax3.set_ylim(0, 2000)
        ax3.set_ylabel('Frames em Y')
        ax3.set_xlabel('Num. Frames')
        ax3.grid()
        fig.tight_layout()
        fig.savefig(pathname3 + 'painel_posicoes_xy_{}'.format(experiment))
        # plt.close('all')
        plt.show()
        stop