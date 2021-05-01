"""
Program to procss wave probes data from LabOceano
henrique pereira
uggo pinho
nelson violante
coppe/ufrj
2018-11-02
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import mlab
import scipy.io as sio
from glob import glob
from scipy.optimize import curve_fit
from numpy import (atleast_1d, sqrt, ones_like, zeros_like, arctan2, where,
                   tanh, sin, cos, sign, inf,
                   flatnonzero, cosh)
plt.close('all')

def w2k(w, theta=0.0, h=inf, g=9.81, count_limit=100, rtol=1e-7, atol=1e-14):
    """
    Translates from frequency to wave number using the dispersion relation

    Parameters
    ----------
    w : array-like
        angular frequency [rad/s].
    theta : array-like, optional
        direction [rad].
    h : real scalar, optional
        water depth [m].
    g : real scalar or array-like of size 2.
        constant of gravity [m/s**2] or 3D normalizing constant

    Returns
    -------
    k1, k2 : ndarray
        wave numbers [rad/m] along dimension 1 and 2.

    Description
    -----------
    Uses Newton Raphson method to find the wave number k in the dispersion
    relation
        w**2= g*k*tanh(k*h).
    The solution k(w) => k1 = k(w)*cos(theta)
                         k2 = k(w)*sin(theta)
    The size of k1,k2 is the common shape of w and theta according to numpy
    broadcasting rules. If w or theta is scalar it functions as a constant
    matrix of the same shape as the other.

    Example
    -------
    >>> import pylab as plb
    >>> import wafo.wave_theory.dispersion_relation as wsd
    >>> w = plb.linspace(0,3);
    >>> wsd.w2k(range(4))[0]
    array([ 0.        ,  0.1019368 ,  0.4077472 ,  0.91743119])
    >>> wsd.w2k(range(4),h=20)[0]
    array([ 0.        ,  0.10503601,  0.40774726,  0.91743119])

    h = plb.plot(w,w2k(w)[0])
    plb.close('all')

    See also
    --------
    k2w
    """
    gi = atleast_1d(g)
    wi, th, hi = np.broadcast_arrays(w, theta, h)
    if wi.size == 0:
        return zeros_like(wi)

    k = 1.0 * sign(wi) * wi ** 2.0 / gi[0]  # deep water
    if (hi > 1e25).all():
        k2 = k * sin(th) * gi[0] / gi[-1]  # size np x nf
        k1 = k * cos(th)
        return k1, k2
    _assert(gi.size == 1, 'Finite depth in combination with 3D normalization'
            ' (len(g)=2) is not implemented yet.')

    oshape = k.shape
    wi, k, hi = wi.ravel(), k.ravel(), hi.ravel()

    # Newton's Method
    # Permit no more than count_limit iterations.
    hi = hi * ones_like(k)
    hn = zeros_like(k)
    ix = flatnonzero(((wi < 0) | (0 < wi)) & (hi < 1e25))

    # Break out of the iteration loop for three reasons:
    #  1) the last update is very small (compared to k*rtol)
    #  2) the last update is very small (compared to atol)
    #  3) There are more than 100 iterations. This should NEVER happen.
    count = 0
    while (ix.size > 0 and count < count_limit):
        ki = k[ix]
        kh = ki * hi[ix]
        coshkh2 = lazywhere(np.abs(kh) < 350, (kh, ),
                            lambda kh: cosh(kh) ** 2.0, fillvalue=np.inf)
        hn[ix] = (ki * tanh(kh) - wi[ix] ** 2.0 / gi) / (tanh(kh) + kh / coshkh2)
        knew = ki - hn[ix]
        # Make sure that the current guess is not zero.
        # When Newton's Method suggests steps that lead to zero guesses
        # take a step 9/10ths of the way to zero:
        ksmall = flatnonzero((np.abs(knew) == 0) | (np.isnan(knew)) )
        if ksmall.size > 0:
            knew[ksmall] = ki[ksmall] / 10.0
            hn[ix[ksmall]] = ki[ksmall] - knew[ksmall]

        k[ix] = knew
        # disp(['Iteration ',num2str(count),'  Number of points left:  '
        # num2str(length(ix)) ]),

        ix = flatnonzero((np.abs(hn) > rtol * np.abs(k)) *
                         (np.abs(hn) > atol))
        count += 1
    max_err = np.max(hn[ix]) if np.any(ix) else 0
    _assert_warn(count < count_limit, 'W2K did not converge. '
                 'Maximum error in the last step was: %13.8f' % max_err)

    k.shape = oshape

    k2 = k * sin(th)
    k1 = k * cos(th)
    return k1, k2

def read_waveprobe(pathname):
    """
    Read matlab file (.mat) waveprobe from LabOceano
    """
    # load data in dictionary
    d = sio.loadmat(pathname)

    # convert to dataframe
    df = {}
    for w in ['TIME','WAVE1_C','WAVE2_C','WAVE3_C']:
        df[w] = d[w][2000:-2000,0]
    df = pd.DataFrame(df, index=df['TIME'])
    df = df.set_index('TIME')
    return df

def calculate_spectra(x, fs, nfft):

    spec = mlab.psd(df.WAVE1_C,NFFT=int(nfft),Fs=fs,detrend=mlab.detrend_mean,
                  window=mlab.window_hanning,noverlap=nfft/2)

    # frequency and spectrum vector
    f, s = spec[1], spec[0]

    # indice para frequencia menor que 2 hz
    # idx = np.where(f<10)[0][-1]
    # f, s = f[20:idx], s[20:idx]

    # calculate angular frequency
    w = 2 * np.pi * f
    return f, w, s

def calculate_wave_parameters(f, s):
    """
    Calculate wave height and peak period in
    the frequency domain
    """
    # significant wave height
    m0 = np.sum(s) * (f[3]-f[2])
    hm0 = 4.01 * np.sqrt(m0)

    # peak frequency index
    tp = 1/f[np.where(s==s.max())[0][0]]
    return hm0, tp

def exponential_func(t, x):
    return t ** x

def adjust_fit_rel_disp(xdata, ydata):
    """
    Adjust an exponential fit
    """
    popt, pcov = curve_fit(exponential_func, xdata, ydata)
    yy = exponential_func(xdata, *popt)
    return popt[0], pcov, yy

def plot_spectra(k, specs):

    fig = plt.figure(figsize=(7,5))
    ax1 = fig.add_subplot(111)
    # for filename in paths:
    for n in specs.keys():
        plt.loglog(k, specs[n], label=n)
    plt.xlim(10**-1, 10**2)
    plt.grid()
    plt.xlabel('Wavenumber (rad/m)')
    plt.ylabel('Energy (m²/(rad/m))')
    plt.legend()
    return

def plot_freq_decay():
    """
    Plot spectra decaiment
    """
    # frequencia angular
    # w = 2 * np.pi * f

    # calcula do numero de onda (com theta igual a 0, apenas o k1 tem valores positivos)
    # k, k2 = w2k(w, theta=0.0, h=inf, g=9.81, count_limit=100, rtol=1e-7, atol=1e-14)

    # peak frequency index
    pki = np.where(sp==sp.max())[0][0]

    # high part of the spectrum
    w1 = w[pki:]
    k1 = k[pki:]
    sp1 = sp[pki:]

    # angular frequency
#     wd4 = w1 ** -4
#     wd5 = w1 ** -5

    # wave number
    kd2 = k1 ** -2.5
    kd3 = k1 ** -3.0

    # normalization
#     wd5 = wd4 / wd4.max()
#     wd5 = wd5 / wd5.max()
    kd2 = kd2 / kd2.max()
    kd3 = kd3 / kd3.max()

    # ----------------------------------------------- #

    # espectro de frequencia
#     plt.figure(figsize=(8,5))
#     plt.loglog(w, sp)
#     plt.grid()
#     plt.xlabel('Ang. Frequency (rad/s)')
#     plt.ylabel('Energy (m²/(rad/s))')
#     plt.twinx()
#     plt.loglog(w1, wd4, 'r')
#     plt.loglog(w1, wd5, 'g')
#     plt.legend(['w-4','w-5'])
#     plt.yticks(visible=False)

    # espectro de numero de onda
    plt.figure(figsize=(6,4))
    plt.title(i)
    plt.loglog(k1, sp1)
    plt.grid()
    plt.xlabel('Wave Number (rad/m)')
    plt.ylabel('Energy (m²/(rad/m))')
    plt.twinx()
    plt.loglog(k1, kd2, 'r')
    plt.loglog(k1, kd3, 'g')
    plt.legend(['k-2.5','k-3.0'])
    plt.yticks(visible=False)
    return


if __name__ == '__main__':

    # pathname dos dados
    pathnames = os.environ['HOME'] + '/Documents/wavescatter_data/DERIVA_RANDOMICA/DADOS_DO_ENSAIO/'
    pathnames = np.sort(glob(pathnames + 'T100_??0100*'))

    # dict with spectra for all experiments
    specs = {}

    fig = plt.figure(figsize=(12,6))

    for pathname in pathnames:

        # create name with experiment
        name = pathname.split('/')[-1].split('.')[0]

        print (pathname)

        # read wave probes (wave1_c, wave2_c and wave3_c)
        df = read_waveprobe(pathname)

        # sample time
        dt = df.index[3] - df.index[2]

        # specra parameters
        fs = 1. / dt
        nfft = int(df.shape[0]/6)

        # calculate frequency spectra
        f, w, s = calculate_spectra(x=df.WAVE1_C, fs=fs, nfft=nfft)

        # calcula do numero de onda (com theta igual a 0, apenas o k1 tem valores positivos)
        k, k2 = w2k(w, theta=0.0, h=inf, g=9.81, count_limit=100, rtol=1e-7, atol=1e-14)

        # pega trecho do espectro e vetores apenas ate ondas de 5 Hz (0.2 s)
        ind_max = np.where(f<=5)[0][-1] # ind freq pico

        f, w, k, s = f[:ind_max], w[:ind_max], k[:ind_max], s[:ind_max]
        # stop

        # calculate wave parameters
        hm0, tp = calculate_wave_parameters(f, s)
        print ('Hm0: {:.3f}, Tp: {:.3f}'.format(hm0, tp))

        # salva valores de espectro no dict
        specs[name] = s

        # indice da frequencia de pico (para iniciar o ajuste)
        ind_wp = np.where(s==s.max())[0][0] # ind freq pico

        # omega de pico
        fp = f[ind_wp]
        wp = w[ind_wp]
        kp = k[ind_wp]

        # normaliza frequencia pela frequencia de pico
        wn = w / wp
        kn = k / kp

        # w de transicao
        # omega de transicao eh a frequencia em que ocorre a transicao
        # do decaimento de alta frequencia de f-4 a f-5 (paper alex).
        # wp ate wt --> f-4  ; wt ate w_final --> f-5
        wt1 = 1.5 * wp
        wt2 = 2.0 * wp

        # indice do w de transicao
        ind_wt1 = np.where(w < wt1)[0][-1]
        ind_wt2 = np.where(w < wt2)[0][-1]

        # idx1 = np.where(f>7)[0][0] # freq pico
        # idx1 = 35
        # idx1 = 7
        # idx2 = np.where(f<2)[0][-1] # freq 1
        # stop

        # xdata_w = w[ind_wp:ind_wt]
        # ydata_w = s[ind_wp:ind_wt]

        # indice para ajuste e plotagem
        ind_final = 1000

        # xdata_w = wn[ind_wp:ind_final]
        # ydata_w = s[ind_wp:ind_final]

        # xdata_k = kn[ind_wp:ind_final]
        # ydata_k = np.copy(ydata_w)

        xdata_w1 = wn[ind_wp:ind_wt2]
        ydata_w1 = s[ind_wp:ind_wt2]

        xdata_w2 = wn[ind_wt1:ind_final]
        ydata_w2 = s[ind_wt1:ind_final]

        xdata_k1 = kn[ind_wp:ind_wt2]
        ydata_k1 = np.copy(ydata_w1)

        xdata_k2 = kn[ind_wt1:ind_final]
        ydata_k2 = np.copy(ydata_w2)

        # xdata_k = kn[ind_wp:ind_final]
        # ydata_k = np.copy(ydata_w)

        # xdata_k = k[ind_wt:]#ind_final]
        # ydata_k = s[ind_wt:]#ind_final]

        # wd_f4 = w[ind_wp:ind_wt] ** -4
        # wd_f5 = w[ind_wt:] ** -5
        # wd_f4 = wd_f4 / wd_f4.max()
        # wd_f5 = wd_f5 / wd_f5.max()

        # xdata_w = w[ind_wp:ind_wt]
        # ydata_w = s[ind_wp:ind_wt]

        #### teste funcao exp
        # xdata = np.arange(1,11).astype(float)
        # ydata = xdata ** -5
        # popt, pcov, yy = adjust_fit_rel_disp(xdata, ydata)

        # plt.figure()
        # # plt.title('Angular frequency spectra')
        # plt.loglog(xdata, ydata, label='{}'.format(name))
        # # plt.loglog(xdata_w, xdata_w**-4, 'r', label='w^-4')
        # # plt.loglog(xdata_w, ydata_w, 'y', label='')
        # plt.loglog(xdata, xdata**popt, 'r-.', label='w^{:.1f}'.format(popt))
        # plt.xlabel('Angular Frequency (rad/s)')
        # plt.ylabel('Energy (m^2/(rad/s))')
        # plt.legend(fontsize=11)
        # plt.show()


        # stop


        # a_w, b_w = 150, 600
        # a_k, b_k = 200, 400
        # xdata_w = w[a_w:b_w]
        # xdata_k = k[a_k:b_k]
        # ydata_w = s[a_w:b_w]
        # ydata_k = s[a_k:b_k]

        # adjust fit for freq and wavenumber spectra
        # popt_w, pcov, yy = adjust_fit_rel_disp(xdata_w, ydata_w)
        # popt_k, pcov, yy = adjust_fit_rel_disp(xdata_k, ydata_k)
        # print (popt_w)
        # print (popt_k)

    # fig1 = plt.figure(figsize=(10,6))
    # fig1.suptitle('Relative Dispersion \n {}'.format(filename))
    # ax1 = fig1.add_subplot(121)
    # ax1.grid()
    # ax1.set_xlabel('Time (seconds)')
    # ax1.set_ylabel('MSD')
    # # ax1.set_title(filename)
    # rel_disp.plot(loglog=True, ax=ax1, legend=None)

    # ax2 = fig1.add_subplot(122)
    # ax2.grid()
    # ax2.set_xlabel('Time (seconds)')
    # # ax2.set_ylabel('Mean Relative Dispersion')
    # # ax2.set_title(filename)
    # mean_rel_disp.plot(loglog=True, ax=ax2, legend=None)

    # fig1.savefig('../img/rel_disp{}.png'.format(filename[-17:]))
    # plt.show()

        # normaliza decaimento
        mm5 = xdata_w2 ** -5
        mm5 = mm5 / mm5.max()
        mm5 = mm5 * s[ind_wp]

        mm4 = xdata_w1 ** -4
        mm4 = mm4 / mm4.max()
        mm4 = mm4 * s[ind_wp]

        mm3 = xdata_k2 ** -3.5
        mm3 = mm3 / mm3.max()
        mm3 = mm3 * s[ind_wp]

        mm2 = xdata_k1 ** -3.
        mm2 = mm2 / mm2.max()
        mm2 = mm2 * s[ind_wp]

        # plot ang freq spectra

        # fig = plt.figure(121)
        ax1 = fig.add_subplot(121)
        ax1.set_title('Angular frequency spectra')
        ax1.loglog(wn, s, label='{}'.format(name))
        # plt.loglog(xdata_w, ydata_w, 'r', label='')
        # plt.loglog(xdata_w, ydata_w, 'y', label='')
        # plt.loglog(xdata_w, xdata_w**popt_w, 'k-.', label='w^{:.3f}'.format(popt_w))
        # plt.loglog(xdata_w, xdata_w**popt_w, 'k-.', label='w^{:.3f}'.format(popt_w))
        # plt.twinx()
        # plt.loglog(xdata_w, xdata_w**-5.0+10**-6, 'r-.', label='')
        # plt.twinx()
        # plt.loglog(w[ind_wp:ind_wt], wd_f4, 'r-.', label='w^-4')
        # plt.loglog(w[ind_wt:], wd_f5, 'r-.', label='w^-5')
        # plt.xlim(1,xdata_w[-1])

        # plot ang wavenumber spectra
        ax2 = fig.add_subplot(122)
        ax2.set_title('Wavenumber spectra')
        ax2.loglog(kn, s, label='{}'.format(name))
        # plt.loglog(xdata_k, xdata_k**-3, 'r', label='k^-5')
        # plt.loglog(xdata_k, ydata_k, 'y', label='')
        # plt.loglog(xdata_k, xdata_k**popt_k, 'k--', label='k^{:.3f}'.format(popt_k))
        # plt.loglog(xdata_k, xdata_k**-5, 'r-', label='k^-5')
        # plt.loglog(k[ind_wp:ind_wt], k[ind_wp:ind_wt]**-2.0, 'r-.', label='k^-4')
        # plt.loglog(k[ind_wt:], k[ind_wt:]**-2.5, 'r-.', label='k^-5')
        # plt.savefig('../img/espec_decay_w_k.png', bbox_inches='tight')

    ax1.grid()
    ax2.grid()

    # ax1.loglog(xdata_w1, mm4, 'k-', linewidth=2)#, label='w^-4')
    # ax1.loglog(xdata_w2, mm5, 'k-', linewidth=2)#, label='w^-5')
    # ax1.text(xdata_w1[-1], mm4[-1], ' w^-4')
    # ax1.text(xdata_w2[-1], mm5[-1], ' w^-5')
    ax1.set_xlabel('w/wp (rad/s)')
    ax1.set_ylabel('Energy (m^2/(rad/s))')
    # ax1.legend(fontsize=11)
    # ax1.set_xlim(10**-4, 10**2)

    # ax2.loglog(xdata_k1, mm2, 'k-', linewidth=2)#, label='w^-4')
    # ax2.loglog(xdata_k2, mm3, 'k-', linewidth=2)#, label='w^-5')
    # ax2.text(xdata_k1[-1], mm2[-1], ' k^-2.5')
    # ax2.text(xdata_k2[-1], mm3[-1], ' k^-3')
    ax2.set_xlabel('k/kp (rad/m^2)')
    ax2.set_ylabel('Energy (m^2/(rad/m^2))')
    # ax2.legend()
    # ax2.legend(fontsize=11)
    # ax2.set_xlim(10**-4, 10**2)
    fig.savefig('../img/espec_decay_w_k.png', bbox_inches='tight')

    # plot_spectra(k, specs)
    plt.show()
