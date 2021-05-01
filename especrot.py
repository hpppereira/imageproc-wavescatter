# Processamento dos dados dos derivadores utilizando
# o espectro rotatorio

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from glob import glob

if __name__ == "__main__":

    lista_experimentos = np.sort(glob('pickle/*.pkl'))

    for experimento in lista_experimentos:

        paths = pd.read_pickle(experimento)

        plt.figure()

        for ball in list(paths.keys()):

            x, y = paths[ball].T

            plt.plot(x,y)