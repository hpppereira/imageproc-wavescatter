'''
Read LabOceano .mat files

Henrique P. P. Pereira
2017/06/08
'''

#load libraries
import os
import numpy as np
import scipy.io as sio

#pathname of data
pathname = os.environ['HOME'] + '/Dropbox/phd/data/DADOS_DO_ENSAIO/'

#list .mat files
list_mat = []
for f in os.listdir(pathname):
	if f.endswith('.mat'):
		list_mat.append(f)

#dd = sio.loadmat