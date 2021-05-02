
import os
import numpy as np
import matplotlib.pyplot as plt
import cv2

plt.close('all')

pathname = 'fig/frames/'

listfiles = np.sort(os.listdir(pathname))

# lista com posicoes
pos = []
for filename in listfiles[50::10]:

	print (filename)
	
	img = cv2.imread(pathname + filename)

	plt.figure(figsize=(12,8))
	plt.imshow(img)

	x = plt.ginput(1)

	pos.append(x[0])

	plt.close('all')

	if pos[-1][0] < 200:
		break


pos1 = np.array(pos)

np.savetxt('ball_01_pos.txt', pos1, fmt='%i')