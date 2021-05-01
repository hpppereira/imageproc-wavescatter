# Analise da deriva de stokes
# Henrique Pereira
# 11/10/2020

import numpy as np
import matplotlib.pyplot as plt
plt.close('all')

def stokes_drift_aguas_profundas(a, T):
    """
    Equacao de stokes para aguas profundas
    alfa - amplitude de onda
    T - período de onda
    """
    z = 0.0
    L = 1.56 * T ** 2
    k = 2 * np.pi / L
    sigma = 2 * np.pi / T
    epslon = a * k
    Us = epslon ** 2 * (sigma/k) * np.exp(2 * k * z)
    return Us

if __name__ == "__main__":

    TT = np.arange(4,20,0.1)
    aa = np.arange(0.1,10,0.1)

    Us_a = [stokes_drift_aguas_profundas(a=x, T=10.0) for x in aa]
    Us_T = [stokes_drift_aguas_profundas(a=2.0, T=x) for x in TT]
   
    fig = plt.figure(figsize=(6,12))
    ax1 = fig.add_subplot(211)
    ax1.plot(aa, Us_a)
    ax1.set_title('Stokes Drift para T=10 s')
    ax1.set_xlabel('Alura de onda (m)')
    ax1.set_ylabel('Stokes drift (m/s)')

    ax2 = fig.add_subplot(212)
    ax2.plot(TT, Us_T)
    ax2.set_title('Stokes Drift para a=1 m')
    ax2.set_xlabel('Período de onda (m)')
    ax2.set_ylabel('Stokes drift (m/s)')

    fig.tight_layout()




    plt.show()

