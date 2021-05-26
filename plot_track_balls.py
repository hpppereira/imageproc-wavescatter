"""
Plotagem do caminho de cada bola calculada
pela rotina 'get_speed_ginput.py'
"""



import os
import numpy as np
import matplotlib.pyplot as plt
import cv2

plt.close('all')

pathname = 'fig/frames/'

pos = np.loadtxt('ball_01_pos.txt')

listfiles = np.sort(os.listdir(pathname))

# lista com posicoes
# pos = []

cont = 0
for filename in listfiles[50::10]:

	print (filename)

	img = cv2.imread(pathname + filename)

	cont += 1

	# l, c = pos[:cont]
	
	plt.figure(figsize=(12,8))
	plt.imshow(img)
	plt.plot(pos[:cont,0], pos[:cont,1], '-*')

	plt.savefig('fig/teste/%s' %filename)

# pos1 = np.array(pos)

plt.close('all')
plt.show()
# np.savetxt('ball_01_pos.txt', pos1, fmt='%i')