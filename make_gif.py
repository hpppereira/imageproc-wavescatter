# Programa para criar gif 

import os
import numpy as np

pathname = os.environ['HOME'] + '/Documents/wavescatter_results/'

list_paths = np.sort(os.listdir(pathname))

def make_gif():
    string = "ffmpeg -framerate 10 -i %*.png output.mp4"
    os.system(string)
    return


for p in list_paths:
    pathname_arq = pathname + p
    string = "cd {}; ffmpeg -framerate 10 -i %*.png {}.mp4".format(pathname_arq, p)
    os.system(string)



