"""
===============
Contour finding
===============

We use a marching squares method to find constant valued contours in an image.
In ``skimage.measure.find_contours``, array values are linearly interpolated
to provide better precision of the output contours. Contours which intersect
the image edge are open; all others are closed.

The `marching squares algorithm
<http://www.essi.fr/~lingrand/MarchingCubes/algo.html>`__ is a special case of
the marching cubes algorithm (Lorensen, William and Harvey E. Cline. Marching
Cubes: A High Resolution 3D Surface Construction Algorithm. Computer Graphics
(SIGGRAPH 87 Proceedings) 21(4) July 1987, p. 163-170).
"""

import cv2
import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from skimage import measure
import imageio
from scipy import ndimage, misc

pathname = os.environ['HOME'] + \
           '/Documents/wavescatter_videos/DERIVA_RANDOMICA/VIDEO/CAM1/T100/'

filename = 'T100_010100_CAM1.avi'

# ---------------------------------------------------------------------- #

# le o video
cap = cv2.VideoCapture(pathname+filename)

# ---------------------------------------------------------------------- #

# escolhe o frame inicial
cap.set(cv2.CAP_PROP_POS_FRAMES, 1000)


# take first frame of the video
ret, frame = cap.read()

gray = cv2.cvtColor(frame ,cv2.COLOR_BGR2GRAY)

# r = misc.imread(image_path)

# Construct some test data
# x, y = np.ogrid[-np.pi:np.pi:100j, -np.pi:np.pi:100j]
# r = np.sin(np.exp((np.sin(x)**3 + np.cos(y)**2)))

# Find contours at a constant value of 0.8
contours = measure.find_contours(gray, 100)

# stop
# Display the image and plot all contours found
fig, ax = plt.subplots()
ax.imshow(frame , interpolation='nearest', cmap=plt.cm.gray)

for n, contour in enumerate(contours):
    ax.plot(contour[:, 1], contour[:, 0], linewidth=2)

ax.axis('image')
ax.set_xticks([])
ax.set_yticks([])
plt.show()
