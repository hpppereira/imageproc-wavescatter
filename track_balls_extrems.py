import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
import pandas as pd

plt.close('all')
cv2.destroyAllWindows()

# # ---------------------------------------------- #

# #BGR cor da bolinha
# b, g, r = 255, 189, 64

# # nframei = 100
# # nframef = 1000

pathname = 'fig/frames/'

listfiles = np.sort(os.listdir(pathname))

# ---------------------------------------------- #
# limite para achar a bola
lim = 40

# RGB das bolas
r, g, b = 255, 189, 64 # laranja

# RGB para  pintar as bolas
r1, g1,  b1 = 255,255,255

# ---------------------------------------------- #
#remove background
def remove_background(img):

	l, c, p = img.shape

	for cc in range(c):
		for ll in range(l):
			if (img[ll, cc] > (b-lim, g-lim, r-lim)).all() and (img[ll, cc] < (b+lim, g+lim, r+lim)).all():
				img[ll, cc] = np.array([b1, g1, r1])
			else:
				img[ll, cc] = np.array([0, 0, 0]) #preto
	return img
# ---------------------------------------------- #

left = []
right = []
down = []
up = []

for filename in listfiles[50::]:

	# 	print (filename)
		
	# 	# imagem original
	imgo = cv2.imread(pathname + filename)

	# imgo = imgo[::2,::2]

	# imagem sem o fundo
	img = np.copy(imgo)

	# remove backgroud
	img = remove_background(img)

	# convert to gray scale
	img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

	# find circle
	circles = cv2.HoughCircles(img,cv2.HOUGH_GRADIENT,1,5,param1=10,
		param2=5,minRadius=8,maxRadius=10)
	# 	# circles = cv2.HoughCircles(img, cv2.HOUGH_GRADIENT, 2, 5)

	# circles = np.uint16(np.around(circles))

	# circulos em nos extremos x e y
	circlesx = np.array([circles[0,np.argsort(circles[0,:,0]),:]])
	circlesy = np.array([circles[0,np.argsort(circles[0,:,1]),:]])

	#posicoes da bola maior e menor distancia em x
	for i in circlesx[0,:][:1]:

		ii = circlesx[0,:][-1]

		print str(i) + '--' + str(ii)

	# 	# draw the outer circle
	# 	cv2.circle(img,(i[0],i[1]),i[2],(0,255,0),2)

		# draw the center of the circle (left side)
		cv2.circle(imgo,(i[0],i[1]),2,(0,0,255),3)

		# draw the center of the circle (right side)
		cv2.circle(imgo,(ii[0],ii[1]),2,(0,0,255),3)

		# create object with x and y
		left.append([filename,i[0],i[1]])
		right.append([filename,ii[0],ii[1]])

	#posicoes da bola maior e menor distancia em y
	for i in circlesy[0,:][:1]:

		ii = circlesy[0,:][-1]

		print str(i) + '--' + str(ii)

	# 	# draw the outer circle
	# 	cv2.circle(img,(i[0],i[1]),i[2],(0,255,0),2)

		# draw the center of the circle (left side)
		cv2.circle(imgo,(i[0],i[1]),2,(0,0,255),3)

		# draw the center of the circle (right side)
		cv2.circle(imgo,(ii[0],ii[1]),2,(0,0,255),3)

		# create object with x and y
		down.append([filename,i[0],i[1]])
		up.append([filename,ii[0],ii[1]])


	# cv2.imshow(filename,img)
	cv2.imwrite('fig/teste1/' + filename, imgo)


# save path for each of four balls
left = pd.DataFrame(np.array(left))
right = pd.DataFrame(np.array(right))
down = pd.DataFrame(np.array(down))
up = pd.DataFrame(np.array(up))

left.iloc[:,1:] = left.iloc[:,1:].astype(float)
right.iloc[:,1:] = right.iloc[:,1:].astype(float)
down.iloc[:,1:] = down.iloc[:,1:].astype(float)
up.iloc[:,1:] = up.iloc[:,1:].astype(float)

left.to_csv('left.csv', header=['frame','x','y'], index=False)
right.to_csv('right.csv', header=['frame','x','y'], index=False)
down.to_csv('down.csv', header=['frame','x','y'], index=False)
up.to_csv('up.csv', header=['frame','x','y'], index=False)

# 	# ---------------------------------------------- #

# 	# cv2.imshow('detected circles',img1)
	# cv2.imwrite('fig/' + filename + '_detected circles',imgo)

# 	# img = img.