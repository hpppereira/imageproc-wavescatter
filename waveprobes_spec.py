# Calculo do espectro dos waveprobes

import os
from matplotlib import mlab
import matplotlib.pyplot as plt
import pandas as pd
import scipy.io as sio
from glob import glob
import numpy as np
import matplotlib.gridspec as gridspec
plt.close('all')

def read_waveprobe(d):
    """
    Read matlab file (.mat) waveprobe from LabOceano
    """
    # convert to dataframe
    df = {}
    for w in ['TIME','WAVE1_C','WAVE2_C','WAVE3_C']:
        df[w] = d[w][2000:-2000:,0]
    df = pd.DataFrame(df, index=df['TIME'])
    df = df.set_index('TIME')
    df.index = df.index - df.index[0]
    return df

def calcula_espectro(df):
    """
    calculo do espectro para os 3 waveprobes
    """
    df_spec = pd.DataFrame()
    for wp in df.columns:
        s, f = mlab.psd(df[wp].values, NFFT=int(nfft), Fs=fs, detrend=mlab.detrend_mean,
                        window=mlab.window_hanning, noverlap=nfft/2)
        df_spec[wp] = s

    # monta frequencia como indice
    df_spec.index = f
    df_spec.index.name = 'freq'

    return df_spec

def plot_serie_espectro(df, df_spec):
    """
    plotagem
    """
    fig = plt.figure(figsize=(10,3))
    spec2 = gridspec.GridSpec(ncols=3, nrows=1, figure=fig)
    ax1 = fig.add_subplot(spec2[0, :2])
    ax2 = fig.add_subplot(spec2[0, 2])

    ax1.plot(df)
    # ax1.set_title(experimento)
    ax1.set_xlim(0, 30)
    ax1.set_ylim(-.15, .15)
    ax1.grid()
    ax1.set_xlabel('Time (sec)')
    ax1.set_ylabel('Elevation (meters)')
    ax1.legend(df.columns, ncol=3, fontsize=6, loc=1)

    ax2.plot(df_spec)
    # ax2.set_title(experimento)
    ax2.set_xlim(0.2, 2.5)
    ax2.set_ylim(0, 0.007)
    ax2.grid()
    ax2.set_xlabel('Frequency (Hz)')
    ax2.set_ylabel('Energy (m²/Hz)')

    fig.suptitle(experimento)
    fig.tight_layout()
    return fig

if __name__ == "__main__":

    lista_arquivos = np.sort(glob('/media/hp/HIGGINS/wavescatter/dados/DADOS_DO_ENSAIO/*.mat'))

    fs = 60.0 #frequencia de amostragem
    N = 24800.0 # numero de registros do df
    nfft = N / 8.0 # 16 graus de liberdade

    for arquivo in lista_arquivos:

        # nome do experimento
        experimento = arquivo.split('/')[-1].split('.')[0]

        # load data in dictionary
        d = sio.loadmat(arquivo)

        # leitura do arquivo
        df = read_waveprobe(d)

        # calculo do espectro para os 3 waveprobes
        df_spec = calcula_espectro(df)

        fig = plot_serie_espectro(df, df_spec)

        fig.savefig('serie_espectro_' + experimento + '.png')
        plt.close('all')

        # stop

        # plot do espectro
        # fig = plot_spectra(f, s, tt=experimento)
        
        # sp = pd.DataFrame([f,s]).T

        # fig.savefig(pathout + 'spec_' + filename[:11] + '.png')
        # sp.to_csv(pathout + 'spec_' + filename[:11] + '.csv', float_format='%.7f', header=['%freq', 'spec'], index=False)

    plt.show()