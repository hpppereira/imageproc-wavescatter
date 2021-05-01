"""
Cria mosaico com as cameras 1 e 2

x = 2026 - 1669 = 357
y = 1114 - 1002 = 112
"""

import os
import matplotlib.pyplot as plt
from importlib import reload
import track_drifters_mosaic
reload(track_drifters_mosaic)
from track_drifters_mosaic import *
#from scipy import ndimage
plt.close('all')

def get_frame_gray(pathname_video, filename, dict_videos, ncam):
    """
    """
    cap, fps = read_video(pathname_video, filename)
    nframei, nframef, dtimevec = find_first_and_last_frames(dict_videos, filename, fps)
    f = read_frame(cap, nframei, ncam)
    g = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)

    return g

if __name__ == "__main__":

    dict_videos = {'T100_040300_CAM1.avi': ['00:00', '00:00'],
                   'T100_040300_CAM2.avi': ['00:00', '00:00'],
                   'T100_040300_CAM3.avi': ['00:00', '00:00'],
                   'T100_040300_CAM4.avi': ['00:00', '00:00']}

    pathname_video = os.environ['HOME'] + '/Documents/wavescatter_data/DERIVA_RANDOMICA/VIDEO/'

    pathname_v1 = pathname_video + 'CAM1/T100/'
    pathname_v2 = pathname_video + 'CAM2/T100/'
    pathname_v3 = pathname_video + 'CAM3/T100/'
    pathname_v4 = pathname_video + 'CAM4/T100/'

    filename_v1 = list(dict_videos.keys())[0]
    filename_v2 = list(dict_videos.keys())[1]
    filename_v3 = list(dict_videos.keys())[2]
    filename_v4 = list(dict_videos.keys())[3]

    g1 = get_frame_gray(pathname_v1, filename_v1, dict_videos, ncam=1)
    g2 = get_frame_gray(pathname_v2, filename_v2, dict_videos, ncam=2)
    g3 = get_frame_gray(pathname_v3, filename_v3, dict_videos, ncam=3)
    g4 = get_frame_gray(pathname_v4, filename_v4, dict_videos, ncam=4)

    # cria matriz de zeros com o tamanho do mosaico
    # m0 = np.zeros((g2.shape[0], g1.shape[0] + g2.shape[0])) * np.nan
    m0 = np.zeros((2160, 4920)) * np.nan

    # mosaico sem sobreposicao
    m0[540:1620, 0:1920] = g1
    m0[120-101:2040-101, 1920-401:3000-401] = g2
    m0[0+106:1080+106, 3000-505:4920-505] = g3
    m0[1080-120:2160-120, 3000-376:4920-376] = g4

    # m0[540:1620, 0:1920] = g1
    # m0[120-109:2040-109, 1920-357:3000-357] = g2
    # m0[0+101:1080+101, 3000-460:4920-460] = g3
    # m0[1080:2160, 3000:4920] = g4



    # m0[540:1620,0:1920]
    # m0[420+112-20:-420+112-20,:1920] = g1
    # m0[:,1920-357-5:-357-5] = g2
    # m0[420+112-20+11+2-10:-420+112-20+11+2-10,:1920] = g1
    # m0[:,1920-357-5+4:-357-5+4] = g2
#    m00[420:-420,:1920] = g1

    # m0[int(g1.shape[0]/2):-int(g1.shape[0]/2), :g1.shape[1]] = g1
    # m0[int(g2.shape[0]/2):g2.shape[0] + int(g2.shape[0]/2), g2.shape[1]:] = g2

    plt.figure(figsize=(15,12))
    plt.imshow(m0)

#    plt.figure()
#    plt.imshow(m00)

    plt.show()