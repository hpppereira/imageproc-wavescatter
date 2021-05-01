# Controle de qualidade dos dados dos derivadores
# converte para escala em centimetros?

import numpy as np
import pandas as pd
from datetime import datetime
from importlib import reload  
import wavescatter
reload(wavescatter)

if __name__ == "__main__":

    pathname = '/media/hp/HIGGINS/ensino/coppe/lioc/wavescatter/output/paths/'

    # cria dicionario com as bolas qualificadas
    balls_tracked, time_videos = wavescatter.qualified_balls()

    # loop dos experimentos
    for experiment in list(balls_tracked.keys()):
        print (experiment)

        # leitura dos dados em pickle transformado em arrays (saida da rotina drifters_track.py)
        paths_raw = wavescatter.read_paths_pickle(pathname + 'paths_raw_{}.pkl'.format(experiment))

        # bolas qualificadas de cada experimento
        balls_qc = balls_tracked[experiment]

        # loop para criar arquivo com paths apenas das bolas qualificadas
        # que foram colocadas na funcao balls_tracked
        paths_qc = {}
        for ball in balls_qc:
            paths_qc[ball] = paths_raw[ball]

        # salva arquivos com os paths qualificados
        pd.to_pickle(paths_qc, pathname + 'paths_qc_{}.pkl'.format(experiment))