"""
plotagem dos paths calculados na
rotina 'get_circles_paths.py'
"""


import pandas as pd
import matplotlib.pyplot as plt
import cv2

pathname = pathname = 'fig/frames/'

left = pd.read_csv('left.csv')
right = pd.read_csv('right.csv')
down = pd.read_csv('down.csv')
up = pd.read_csv('up.csv')

cont = 0
for filename in left.frame:

	cont += 1

	img = cv2.imread(pathname + filename)

	plt.figure(figsize=(12,8))
	plt.imshow(img)
	plt.plot(left.x[:cont], left.y[:cont], 'w-*')
	plt.plot(right.x[:cont], right.y[:cont], 'g-*')
	plt.plot(down.x[:cont], down.y[:cont], 'y-*')
	plt.plot(up.x[:cont], up.y[:cont], 'b-*')
	plt.axis('tight')

	plt.savefig('fig/teste2/%s' %filename, bbox_inches='tight')


plt.close('all')
