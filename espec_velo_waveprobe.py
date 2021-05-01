"""
Comparacao do espectro da serie de velocidade com
a seria do waveprobe
"""

import numpy as np
import scipy.io as sio
import pandas as pd
from matplotlib import mlab
import matplotlib.pyplot as plt
plt.close('all')

def read_waveprobe(pathname):
    """
    """
    # leitura dos dados do waveprobe    
    d = sio.loadmat(pathname)

    # convert to dataframe
    df = {}
    for w in ['TIME','WAVE1_C','WAVE2_C','WAVE3_C']:
        df[w] = d[w][2000:-2000,0]
    df = pd.DataFrame(df, index=df['TIME'])
    df = df.set_index('TIME')

    return df

def read_drifter(pathname):
    """
    """
    df = pd.read_csv(pathname, index_col='Unnamed: 0')

    return df

def calculate_spec(x, fs, nfft):
    """
    """
    # dt = df.index[3] - df.index[2]

    # # specra parameters
    # fs = 1. / dt
    # nfft = int(df.shape[0]/11)

    # calculo do espectro do waveprobe
    s, f = mlab.psd(x, NFFT=int(nfft), Fs=fs, detrend=mlab.detrend_mean,
                    window=mlab.window_hanning, noverlap=nfft/2)

    return s, f

def plot_spectra(fp, fwp, swp, fdr, sdr1, sdr2, sdr3, sdr4,
                 f03, sd03, f04, sd04, f05, sd05):

    fig = plt.figure(figsize=(15.3,5))
    ax1 = fig.add_subplot(121)
    ax1.plot([fp, fp], [0,1000], 'k:')
    # plt.plot(fwp, swp, 'k', label='water elevation')
    ax1.plot(fdr, sdr1, 'k-', label='drifter displacement', alpha=.4)
    ax1.plot(fdr, sdr2, 'k-', alpha=.4)
    ax1.plot(fdr, sdr3, 'k-', alpha=.4)
    ax1.plot(fdr, sdr4, 'k-', alpha=.4)
    ax1.set_xlim(0.3, 2)
    ax1.set_ylim(0,1000)
    ax1.grid()
    # plt.ylabel('Normalized Energy \n m²/Hz -- mm²/Hz')
    ax1.set_ylabel('Spectral density (mm²/Hz)', fontsize=18)
    ax1.set_xlabel('Frequency (Hz)', fontsize=18)
    ax1.tick_params(axis='both', which='major', labelsize=18)
    ax2 = fig.add_subplot(122)
    ax2.plot(f03, sd03, 'k-', label='Exp.1')    
    ax2.plot([0.62, 0.62], [0,sd03.max()], 'k:')
    ax2.plot(f04, sd04, 'k--', label='Exp.2')    
    ax2.plot([0.75, 0.75], [0,sd04.max()], 'k:')
    ax2.plot(f05, sd05, 'k-.', label='Exp.3')    
    ax2.plot([1.02, 1.02], [0,sd05.max()], 'k:')
    ax2.set_ylabel('Spectral density (mm²/Hz)', fontsize=18)
    ax2.set_xlabel('Frequency (Hz)', fontsize=18)
    ax2.tick_params(axis='both', which='major', labelsize=18)
    ax2.set_xlim(0.3, 2)
    ax2.set_ylim(0, sd03.max())
    ax2.legend(fontsize=18)
    ax2.grid()
    fig.savefig('img/compara_espec.pdf', bbox_inches='tight')
    plt.show()

    return

if __name__ == '__main__':

    # pathnames
    pathname_waveprobe = 'data/T100_040300.pro.mat'
    pathname_d03 = 'data/dists_xy_t0_T100_030100_CAM1.csv'
    pathname_d04 = 'data/dists_xy_t0_T100_040300_CAM1.csv'
    pathname_d05 = 'data/dists_xy_t0_T100_050100_CAM1.csv'

    # read data
    wp = read_waveprobe(pathname_waveprobe)
    d03 = read_drifter(pathname_d03)
    d04 = read_drifter(pathname_d04)
    d05 = read_drifter(pathname_d05)

    # calculate spectra
    # waveprobe
    swp, fwp = calculate_spec(wp.WAVE1_C, 60, len(wp)/13)

    # normaliza o espec
    # swp = swp/swp.max()

    # calculo do espectro dos drifters do mesmo experimento
    sdr1, fdr = calculate_spec(d04.ball_00, 30, len(d04)/4)
    sdr2, fdr = calculate_spec(d04.ball_01, 30, len(d04)/4)
    sdr3, fdr = calculate_spec(d04.ball_05, 30, len(d04)/4)
    sdr4, fdr = calculate_spec(d04.ball_07, 30, len(d04)/4)

    # calculo do espectro de uma bola para 3 diferentes experimentos
    sd03, f03 = calculate_spec(d03.ball_00, 30, len(d03)/4)
    sd04, f04 = calculate_spec(d04.ball_00, 30, len(d04)/4)
    sd05, f05 = calculate_spec(d05.ball_00, 30, len(d05)/4)

    # zera os valores de baixa freq
    sdr1[:5] = 0
    sdr2[:5] = 0
    sdr3[:5] = 0
    sdr4[:5] = 0
    sd03[:5] = 0
    sd04[:5] = 0
    sd05[:5] = 0

    # normaliza a energia dos espectros
    # sdr1 = sdr1/sdr1.max()
    # sdr2 = sdr2/sdr2.max()
    # sdr3 = sdr3/sdr3.max()
    # sdr4 = sdr4/sdr4.max()

    # calculo da frequencia de pico dos derivadores
    fp = fdr[np.where(sdr1 == sdr1.max())[0][0]]

    # plot spectra
    plot_spectra(fp, fwp, swp, fdr, sdr1, sdr2, sdr3, sdr4,
                 f03, sd03, f04, sd04, f05, sd05)

