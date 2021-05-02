"""
Detecta os conjuntos de bolas e faz um contorno neles

Henrique Pereira
02/11/2017
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import glob
import imageio
from scipy import ndimage, misc
from PIL import Image

###############################################################################

plt.close('all')

# pathname = os.environ['HOME'] + \
# 		   '/Dropbox/WaveDrift/data/DERIVA_RANDOMICA/VIDEO/CAM1/T100/frames/'

pathname = 'frames/'

for image_path in glob.glob(pathname + '*.png'):
    image = imageio.imread(image_path)
    print (image.shape)
    print (image.dtype)

###############################################################################

# im = Image.open(image_path) #Can be many different formats.
# pix = im.load()
# print im.size #Get the width and hight of the image for iterating over
# print pix[0,0] #Get the RGBA Value of the a pixel of an image
# pix[0,0] = (255, 155, 40) # Set the RGBA Value of the image (tuple)
# im.save("alive_parrot.png") # Save the modified pixels as png

# width, height = im.size

# pixel_values = list(im.getdata())


# def get_image(image_path):
#     """Get a numpy array of an image so that one can access values[x][y]."""
#     image = Image.open(image_path, 'r')
#     width, height = image.size
#     pixel_values = list(image.getdata())
#     if image.mode == 'RGB':
#         channels = 3
#     elif image.mode == 'L':
#         channels = 1
#     else:
#         print("Unknown mode: %s" % image.mode)
#         return None
#     pixel_values = np.array(pixel_values).reshape((width, height, channels))
#     return pixel_values


###############################################################################


list_frames = np.sort(os.listdir(pathname))

for f in range(len(list_frames)-1):

	im1 = misc.imread(pathname + list_frames[f])
	im2 = misc.imread(pathname + list_frames[f+1])

	im = im2[:,:,0] - im1[:,:,0]

	plt.figure()
	plt.imshow(im, cmap=plt.cm.gray,  vmin=30, vmax=200)

	plt.show()




# im = misc.imread(image_path)
# image_path = pathname + 'ame_01.png.png'

# im = misc.imread(image_path)

# im = im[:,:,0]

# im = ndimage.gaussian_filter(im, 8)

# f = misc.face(gray=True)  # retrieve a grayscale image

# plt.imshow(im, cmap=plt.cm.gray,  vmin=30, vmax=200)

# plt.contour(im, [200], color='yellow')

# Use a gradient operator (Sobel) to find high intensity variations:

# sx = ndimage.sobel(im, axis=0, mode='constant')

# sy = ndimage.sobel(im, axis=1, mode='constant')

# sob = np.hypot(sx, sy)

# plt.figure()
# plt.imshow(sob, cmap=plt.cm.gray)

# plt.show()