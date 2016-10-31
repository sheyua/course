import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data

def plotImage(data, idx, pred=-1):
	arr1 = data.images[idx,:]
	in_dim = np.sqrt(arr1.shape[0]).astype(int)
	arr2 = arr1.reshape((in_dim,in_dim))
	plt.ion()
	plt.figure()
	plt.pcolor(arr2)
	plt.xlim((0,in_dim))
	plt.xticks([])
	plt.xlabel('true: '+str(data.labels[idx].argmax()), fontsize=25)
	plt.ylim((0,in_dim))
	plt.gca().invert_yaxis()
	plt.yticks([])
	if pred>=0 and pred<10:
		plt.title('pred: '+str(pred), fontsize=25)
	plt.tight_layout()

# load data and define parameters
mnist = input_data.read_data_sets('MNIST_data/', one_hot=True)
in_dim = np.sqrt(mnist.train.images.shape[1]).astype(int)
num_of_train, out_dim = mnist.train.labels.shape
