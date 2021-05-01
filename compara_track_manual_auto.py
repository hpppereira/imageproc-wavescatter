"""
Comparacao track manual e automatico

# 1 px --> 1.81 mm <br>
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
plt.close('all')

ma = pd.read_pickle('data/paths_T100_040300_CAM1_ball_31_manual.pkl')
au = pd.read_pickle('data/paths_T100_040300_CAM1.pkl')
au['x'] = np.array(au['ball_31'].tolist()).T[0,:]
au['y'] = np.array(au['ball_31'].tolist()).T[1,:] + 250 # +250 para corrigir a janela do frame
au = au[['x','y']]

# reamostra em 1 fps
au = au.iloc[ma.index]

# calcula as distancias
dists_xy_au_t0 = [np.sqrt((au.x.iloc[i] - au.x.iloc[0])**2 + (au.y.iloc[i] - au.y.iloc[0])**2) for i in range(len(au)-1)]
dists_xy_ma_t0 = [np.sqrt((ma.x.iloc[i] - ma.x.iloc[0])**2 + (ma.y.iloc[i] - ma.y.iloc[0])**2) for i in range(len(ma)-1)]

# converte pixel para m
dists_xy_au_t0 = np.array(dists_xy_au_t0) * 1.81 / 1000
dists_xy_ma_t0 = np.array(dists_xy_ma_t0) * 1.81 / 1000

# calcula coeficiente de correlação
r = pearsonr(dists_xy_au_t0, dists_xy_ma_t0)[0]
print ('correlação: %r' %r)

# calcula bias
b = (dists_xy_ma_t0 - dists_xy_au_t0).mean()
print ('bias: %r' %b)

# calcula rmse
rmse = np.sqrt( np.sum( (dists_xy_ma_t0 - dists_xy_au_t0) ** 2 ) / len(dists_xy_ma_t0) )
print ('rmse: %r' %rmse)

# calcula scatter index
si = rmse / dists_xy_au_t0.mean()
print ('scatter index: %r' %si)

# calcula as velocidades
va = np.abs(np.diff(dists_xy_au_t0))
vm = np.abs(np.diff(dists_xy_ma_t0))

# plt.figure()
# plt.plot(au.x, au.y, label='auto')
# plt.plot(ma.x, ma.y, label='manu')
# plt.legend()

# plt.figure()
# plt.plot(dists_xy_au_t0, label='auto')
# plt.plot(dists_xy_ma_t0, label='manu')
# plt.legend()

# plt.figure()
fig = plt.figure(figsize=(4,4))
ax1 = fig.add_subplot(111)
ax1.plot(dists_xy_au_t0, dists_xy_ma_t0, 'ko', alpha=0.5)
ax1.plot([0, 2.5], [0, 2.5], 'k--')
# plt.title('Distance from initial position')
ax1.set_xlabel('Automatic detection (m)')
ax1.set_ylabel('Manual detection (m)')
ax1.grid()
ax1.set_xlim(0,2.5)
ax1.set_ylim(0,2.5)
# plt.legend(['r = %.3f' %r])
fig.savefig('img/compara_track.pdf', bbox_inches='tight')

# plt.legend()

plt.show()