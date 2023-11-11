#!/usr/bin/python3 -i
'''
A very simple linear regression.
Generate a random linear model and fit.
'''
import argparse
import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt

def gen_coeff(ndim):
	W = np.random.rand(1,ndim)
	b = np.random.rand(1,1)
	return (W.astype(np.float32),b.astype(np.float32))

def gen_sample(num_data, coeff, epsilon = 0.1):
	W = coeff[0]
	b = coeff[1]
	ndim = W.shape[1]
	x_data = np.random.rand(ndim, num_data)
	y_data = np.dot(W, x_data) + b + epsilon*np.random.rand(1, num_data)
	return (x_data.astype(np.float32), y_data.astype(np.float32))

def load_and_train(options):
	# generate data
	coeff_true = gen_coeff(options.ndim)
	x_data, y_data = gen_sample(options.num_data, coeff_true)
	
	# construst the model
	x = tf.placeholder(tf.float32, [options.ndim, None] )
	W = tf.Variable(tf.zeros([1, options.ndim]))
	b = tf.Variable(tf.zeros([1,1]))
	h = tf.matmul(W, x) + b
	
	# supervise with known data
	y = tf.placeholder(tf.float32, [1, None])
	
	# define loss function
	log_likelihood = -tf.reduce_sum( tf.square(h - y) ) # should be equivalent as (h-y)**2
	err = tf.sqrt( tf.reduce_mean(tf.square(h-y)) )
	train_step = tf.train.RMSPropOptimizer(0.01).minimize(-log_likelihood)
	
	# init and train
	with tf.Session() as sess:
		tf.initialize_all_variables().run()
		for idx in range(options.num_epoch):
			sess.run(train_step, feed_dict={x:x_data, y:y_data})
			print('epoch', idx, 'err in training:', err.eval({x:x_data, y:y_data}))
		return ((W.eval(), b.eval()), coeff_true)

def build_options():
	parser = argparse.ArgumentParser()
	parser.add_argument('--num-dim', type=int,
		dest='ndim', help='number of dimension of the input vector',
		metavar='NDIM', default=1)
	parser.add_argument('--num-data', type=int,
		dest='num_data', help='number of data points in the sample set',
		metavar='NUM_DATA', default=50)
	parser.add_argument('--num-epoch', type=int,
		dest='num_epoch', help='number of epochs in traning',
		metavar='NUM_EPOCH', default=300)
	return parser.parse_args()

# main function
if __name__ == '__main__':
	options = build_options()
	coeff_fit, coeff_true = load_and_train(options)
	print('True Coefficients:')
	print('W:',coeff_true[0], 'b:', coeff_true[1])
	print('Fitted Coefficients:')
	print('W:',coeff_fit[0], 'b:', coeff_fit[1])
