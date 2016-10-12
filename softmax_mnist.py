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

def plotImage(data, idx):
	arr1 = data.images[idx,:]
	in_dim = np.sqrt(arr1.shape[0]).astype(int)
	out_dim = data.labels.shape[1]
	print( np.sum( np.arange(0,out_dim,1)*data.labels[idx] ).astype(int) )
	arr2 = arr1.reshape((in_dim,in_dim))
	plt.ion()
	plt.figure()
	plt.pcolor(arr2)
	plt.gca().invert_yaxis()

def load_and_train(options):
	# load data and define parameters
	mnist = input_data.read_data_sets('MNIST_data/', one_hot=True)
	in_dim = np.sqrt(mnist.train.images.shape[1]).astype(int)
	num_of_train, out_dim = mnist.train.labels.shape
	
	# construst the model
	x = tf.placeholder(tf.float32, [None, in_dim*in_dim])
	W = tf.Variable(tf.zeros([in_dim*in_dim, out_dim]))
	b = tf.Variable(tf.zeros([1,out_dim]))
	h = tf.nn.softmax( tf.matmul(x, W) + b )
	
	# supervise with known labels
	y = tf.placeholder(tf.float32, [None, out_dim])
	
	# define loss function
	cross_entropy = tf.reduce_mean( -tf.reduce_sum( tf.log(h)*y, reduction_indices=[1] ) )
	train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)
	
	# init and train
	if num_of_train < options.train_size:
		options.train_size = num_of_train
	num_of_loops = np.floor( options.train_size / options.batch_size ).astype(int)
	with tf.Session() as sess:
		tf.initialize_all_variables().run()
		for idx in range(num_of_loops):
			batch_xs, batch_ys = mnist.train.next_batch(options.batch_size)
			sess.run(train_step, feed_dict={x:batch_xs, y:batch_ys})
			# test the current model
			correct_prediction = tf.equal( tf.argmax(h,1), tf.argmax(y,1) )
			accuracy = tf.reduce_mean( tf.cast(correct_prediction, tf.float32) )
			print('step', idx, 'acc:', accuracy.eval({x: batch_xs, y: batch_ys}))
	
		# evaluate the final results
		results = tf.argmax(h,1).eval({x:mnist.test.images})
	# return test set for evaluation
	return (mnist.test, results)

def build_options():
	parser = argparse.ArgumentParser()
	parser.add_argument('--batch-size', type=int,
		dest='batch_size', help='number of digit images to be feeded at one time',
		metavar='BATCH_SIZE', default=50)
	parser.add_argument('--train-size', type=int,
		dest='train_size', help='total number of digit images to be trained',
		metavar='TRAIN_SIZE', default=5000)
	return parser.parse_args()

# main function
if __name__ == '__main__':
	options = build_options()
	xys, hs = load_and_train(options)
