# Concatena valores dos trajetos das 2 cameras

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import signal
from scipy.signal import butter, lfilter, freqz
plt.close('all')


def butter_lowpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a

def butter_lowpass_filter(data, cutoff, fs, order=5):
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = lfilter(b, a, data)
    return y

if __name__ == '__main__':

  pathname = os.environ['HOME'] + '/gdrive/coppe/lioc/wavescatter/output/'
  pathname_fig = os.environ['HOME'] + '/gdrive/coppe/lioc/wavescatter/figs/'

  experiments = ['T100_010300',
                 # 'T100_020100',
                 # 'T100_020200',
                 # 'T100_020201',
                 # 'T100_020300',
                 # 'T100_030100',
                 # 'T100_030200',
                 # 'T100_030300',
                 # 'T100_040100',
                 # 'T100_040300',
                 # 'T100_050100',
                 # 'T100_050300'
                 ]

  teste = 'all_balls'

  for experiment in experiments:

      d1 = pd.read_csv(pathname + '{}_CAM1/'.format(experiment) + 'mean_rel_disp_{}_{}_CAM1.csv'.format(teste, experiment)
        , header=0, usecols=[1], names=['Dt'])
      d2 = pd.read_csv(pathname + '{}_CAM2/'.format(experiment) + 'mean_rel_disp_{}_{}_CAM2.csv'.format(teste, experiment),
        header=0, usecols=[1], names=['Dt'])

      # retira do valor inicial
      d22 = d2 - d2.iloc[0]

      d = pd.concat((d1, d2), axis=0, ignore_index=True)
      d_offset = pd.concat((d1, d22 + d1.iloc[-30:].mean()), axis=0, ignore_index=True)
      d['time'] = np.linspace(0,len(d)/30.0,len(d))
      d_offset['time'] = np.linspace(0,len(d_offset)/30.0,len(d))
      d.set_index('time', inplace=True)
      d_offset.set_index('time', inplace=True)

      order = 5
      fs = 30.0       # sample rate, Hz
      cutoff = 0.4  # desired cutoff frequency of the filter, Hz

      b, a = butter_lowpass(cutoff, fs, order)
      d['y'] = butter_lowpass_filter(d.values[:,0], cutoff, fs, order)

      plt.figure()
      plt.plot(d.index[:len(d1)], d.Dt.iloc[:len(d1)])
      plt.plot(d.index[-len(d2):], d.Dt.iloc[-len(d2):])
      plt.title('mean_rel_disp_concat_{}'.format(experiment))
      plt.grid()
      plt.show()
      plt.savefig(pathname_fig + 'mean_rel_disp_concat_{}_{}.png'.format(teste, experiment))

      plt.figure()
      plt.plot(d.index[:len(d1)], d_offset.Dt.iloc[:len(d1)])
      plt.plot(d.index[-len(d2):], d_offset.Dt.iloc[-len(d2):])
      plt.title('mean_rel_disp_concat_offset_{}'.format(experiment))
      plt.grid()
      plt.show()
      plt.savefig(pathname_fig + 'mean_rel_disp_concat_offset_{}_{}.png'.format(teste, experiment))

      d.to_csv(pathname + 'Dt/' + 'mean_rel_disp_concat_{}_{}.csv'.format(teste, experiment))
      d_offset.to_csv(pathname + 'Dt/' + 'mean_rel_disp_concat_offset_{}_{}.csv'.format(teste, experiment))