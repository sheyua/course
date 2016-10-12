#!/usr/bin/python3 -i
'''
A very simple MNIST classifier.
See extensive documentation at
http://tensorflow.org/tutorials/mnist/beginners/index.md
'''
import argparse
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
	plt.gca().invert_yaxis()
	plt.xlim((0,in_dim))
	plt.xticks([])
	plt.xlabel('true: '+str(data.labels[idx]), fontsize=25)
	plt.ylim((0,in_dim))
	plt.yticks([])
	if pred>=0 and pred<10:
		plt.title('pred: '+str(pred), fontsize=25)
	plt.tight_layout()

def load_and_train(options):
	# load data and define parameters
	mnist = input_data.read_data_sets('MNIST_data/')
	in_dim = np.sqrt(mnist.train.images.shape[1]).astype(int)
	num_of_train = mnist.train.labels.shape[0]
	
	# construst the model
	x = tf.placeholder(tf.float32, [None, in_dim*in_dim])
	W = tf.Variable(tf.zeros([in_dim*in_dim, 1]))
	b = tf.Variable(tf.zeros([1, 1]))
	h = 1/(1+tf.exp(-tf.matmul(x,W)-b))
	
	# supervise with known labels
	y = tf.placeholder(tf.float32, [None, 1])
	
	# define loss function
	log_likelihood = tf.reduce_mean( y*tf.log(h) + (1.-y)*tf.log(1.0-h) )
	train_step = tf.train.GradientDescentOptimizer(0.5).minimize(-log_likelihood)
	
	# init and train
	with tf.Session() as sess:
		tf.initialize_all_variables().run()
		for idx in range(options.num_epoch):
			batch_pos = options.batch_size
			err = 0.
			num_batch = 0
			while(batch_pos <= num_of_train):
				x_data = mnist.train.images[(batch_pos-options.batch_size):batch_pos,:]
				y_data = mnist.train.labels[(batch_pos-options.batch_size):batch_pos] == options.digit
				y_data = y_data.reshape((options.batch_size,1)).astype(np.float32)
				sess.run(train_step, feed_dict={x:x_data, y:y_data})
				batch_pos += options.batch_size
				# test the current model
				num_batch += 1
				this_err = tf.reduce_mean(
						tf.cast(
							tf.not_equal( tf.greater(h,options.thres), tf.greater(y,options.thres) )
							, tf.float32)
						)
				err += this_err.eval({x:x_data, y:y_data})
			print('epoch', idx, 'err: ', err/num_batch)
		# in the test set
		pred = tf.greater(h, options.thres).eval({x:mnist.test.images})
		obs = mnist.test.labels == options.digit
		err = np.mean(pred != obs)
		print('err in test: ', err)
		return (W.eval(),b.eval())


def build_options():
	parser = argparse.ArgumentParser()
	parser.add_argument('--thres', type=float,
		dest='thres', help='threshold in logistic regression for being true',
		metavar='THRES', default=0.5)
	parser.add_argument('--batch-size', type=int,
		dest='batch_size', help='number of digit images to be feeded at one time',
		metavar='BATCH_SIZE', default=5000)
	parser.add_argument('--num-epoch', type=int,
		dest='num_epoch', help='number of epochs in traning',
		metavar='NUM_EPOCH', default=50)
	parser.add_argument('--digit', type=int,
		dest='digit', help='which digit to do the regression',
		metavar='DIGIT', default=5)
	return parser.parse_args()

# main function
if __name__ == '__main__':
	options = build_options()
	W, b = load_and_train(options)
	class Wrapper:
		images = None
		labels = None
	coeff = Wrapper()
	coeff.images = W.reshape((1,W.shape[0]))
	coeff.labels = b.reshape((1,))
	plotImage(coeff,0)
	plt.title('W matrix', fontsize=25)
	plt.xlabel("b: %.2f" % coeff.labels, fontsize=20)
	plt.colorbar()
	plt.tight_layout()
