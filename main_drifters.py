"""
Main program to processing difters data
Henrique Pereira
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from glob import glob
plt.close('all')

if __name__ == "__main__":

    # pathname do arquivo qualificado com os paths
    pathname1 = '/media/hp/HIGGINS/ensino/coppe/lioc/wavescatter/output/paths/'
    pathname2 = '/media/hp/HIGGINS/ensino/coppe/lioc/wavescatter/output/figs/paths_x0/'

    # lista dos paths dos arquivos
    experimentos = glob(pathname1 + 'paths_qc*.pkl')

    for experimento in experimentos:

        # nome do experimento
        nome = experimento.split('/')[-1][9:-4]

        # carrega dados com os paths de cada bola
        paths = pd.read_pickle(experimento)

        # cria vetor de tempo de acordo com o experimento para 30 fps
        N = paths[list(paths.keys())[0]].shape[0]
        fps = 30.0
        t=np.arange(0, N/fps, 1/fps)

        # figura
        fig, (ax1) = plt.subplots(1, 1, figsize=(8,5))

        for ball in paths.keys():

            # cria dataframe com o path x e y
            df = pd.DataFrame(paths[ball], index=t, columns=['x', 'y'])

            # inicia todos no posicao xy=0
            df['x'] = df.x - df.x[0]
            df['y'] = df.y - df.y[0]

            # para para escala de metros
            df = df * 1.8 / 1000.0

            # remove os primeiros segundos
            # df = df.loc[8:,:]

            # calculo media movel
            df = df.rolling(window=6, center=True, axis=0).mean()

            # calculo da derivada
            ddf = pd.DataFrame(np.diff(df, axis=0), index=df.index[1:], columns=['x','y'])

            ax1.plot(df.x, df.y)
            ax1.set_title(nome)
            ax1.set_xlabel('Posição X [metros]')
            ax1.set_ylabel('Posição Y [metros]')
            ax1.grid()
            # ax1.plot(df.x, df.y, '.', color='r')
            ax1.invert_yaxis()

            fig.savefig(pathname2 + 'paths_xy0_' + nome + '.png')
            plt.close('all')

            # ax2.plot(ddf.x, ddf.y)


        # quiveropts = dict(color='black', headlength=1, pivot='middle', scale=0.1, 
        #                   linewidth=.5, units='xy', width=.5, headwidth=1)
        # fig, ax = plt.subplots()
        # ax.quiver(ddf.x, ddf.y, headaxislength=4.5, **quiveropts)

        plt.show()