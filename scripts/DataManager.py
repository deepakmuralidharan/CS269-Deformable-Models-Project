''' Data Manager script '''

__author__ = "deepak_muralidharan"
__email__ = "deepakmuralidharan2308@gmail.com"

import numpy as np
import scipy.io
import os,sys

path = "/Users/deepakmuralidharan/Documents/data/"

def mat2numpy(file_name):

	full_file_name = path + file_name
	val = scipy.io.loadmat(full_file_name)
	return (val['subImage'],val['v'])


def mat2numpy_test(path, file_name):

	full_file_name = path + file_name
	val = scipy.io.loadmat(full_file_name)
	return (val['subImage'])


def getBatch(batch_size, step, dirs):

	#dirs = os.listdir(path)
	dir_batch_list = dirs[step*batch_size:(step+1)*batch_size]
	#print(dir_batch_list)
	x_list = []
	y_list = []
	for i in dir_batch_list:
		(x, y) = mat2numpy(i)
		x_list.append(x)
		y_list.append(y)

	x_array = np.asarray(x_list)
	y_array = np.asarray(y_list)

	return (x_array, y_array)

def getValidationSet(dirs):

	#dirs = os.listdir(path)
	dir_batch_list = dirs[-10000:]
	#print(dir_batch_list)
	x_list = []
	y_list = []
	for i in dir_batch_list:
		(x, y) = mat2numpy(i)
		x_list.append(x)
		y_list.append(y)

	x_array = np.asarray(x_list)
	y_array = np.asarray(y_list)

	return (x_array, y_array)

def getTestSet():

	test_path = "../tmp/patches/"
	dirs = os.listdir(test_path)
	dirs = sorted(dirs, key=lambda x: int(x.split('.')[0]))
	x_list = []
	for i in dirs:

		x = mat2numpy_test(test_path,i)
		x_list.append(x)

	x_array = np.asarray(x_list)

	return x_array
