#!/usr/bin/python3 -i
'''
A very linear regression with dropout layer.
Fit stock weekly return.
'''
import sys; sys.path.append('./utils')
import sym
import argparse
import pickle
import random
import numpy as np
import pandas as pd
import tensorflow as tf
from matplotlib import pyplot as plt

def download_trends(options):
	if options.down_tre:
		tre_lst = sym.trend_lst()
		sym.restart_trends_day(syms=tre_lst, loc=options.data_loc)
		sym.resume_trends_day(syms=tre_lst, loc=options.data_loc)

def download_syms(sym_lst, options):
	if options.down_sym:
		sym.restart_syms_day(syms=sym_lst, loc=options.data_loc)

def load_data(options):
	# symbol and data loc
	symbol = options.symbol
	loc = options.data_loc
	# download data
	download_trends(options)
	download_syms([symbol], options)
	# build input
	tre_lst = sym.trend_lst()
	df_sym = sym.day2week(pickle.load(open(loc+'/'+symbol+'.dat','rb')))
	df_tre_lst = {}
	interset = df_sym['Week']
	for trend in tre_lst:
		df_tre_lst[trend] = sym.day2week(pickle.load(open(loc+'/'+trend+'.dat','rb')))
		interset = np.intersect1d(interset, df_tre_lst[trend]['Week'])
	df = df_sym[df_sym['Week'].isin(interset)]
	for trend in tre_lst:
		df[trend] = df_tre_lst[trend]['Ret'][df_tre_lst[trend]['Week'].isin(interset)].values
	x_df = df[[col for col in df.columns if col != 'Week' ]]
	y_df = df['Ret'][1:]
	return x_df.values.astype(np.float32), np.append(y_df.values,np.nan).astype(np.float32), df['Week'].values, [col for col in x_df.columns]

#def load_and_train(options):
#	# generate data and get dimensions
#	x_data, y_data, weeks, _ = load_data(options)
#	num_data, ndim = x_data.shape
#	num_data -= 1
#	num_cross = np.floor(num_data * options.cross).astype(int)
#	if num_cross == 0 or num_cross == num_data:
#		print('Not enough data for training and cross-validation!')
#		return
#	else:
#		y_data = y_data.reshape((num_data+1, 1))
#	
#	# construst the model
#	x = tf.placeholder(tf.float32, [None, ndim])
#	x_drop = tf.nn.dropout(x, 1.-options.dropout) # keep probablity is the arg
#	W = tf.Variable(tf.zeros([ndim, 1]))
#	b = tf.Variable(tf.zeros([1,1]))
#	h = tf.matmul(x, W) + b
#	
#	# supervise with known data
#	y = tf.placeholder(tf.float32, [None, 1])
#	
#	# define loss function
#	log_likelihood = -tf.reduce_sum( tf.square(h - y) )
#	err = tf.sqrt( tf.reduce_mean(tf.square(h-y)) )
#	train_step = tf.train.RMSPropOptimizer(1e-5).minimize(-log_likelihood)
#	
#	# init and train
#	with tf.Session() as sess:
#		tf.initialize_all_variables().run()
#		for idx in range(options.num_epoch):
#			cross = random.sample(range(num_data), num_cross)
#			not_cross = [i for i in range(num_data) if i not in cross]
#			sess.run(train_step, feed_dict={
#			  x:x_data[not_cross,:], y:y_data[not_cross,:]
#			})
#			print('epoch', idx, 'err in cross:', err.eval({
#			  x:x_data[cross,:], y:y_data[cross,:]
#			}))
#		pred = h.eval({x:x_data})
#		W_data = W.eval()
#		b_data = b.eval()
#	
#	# report err metrics
#	print('RMS of weekly return:', np.sqrt(np.mean(y_data[:-1,0]**2)))
#	print('RMSE of prediction:', np.sqrt(np.mean((pred-y_data)[:-1,0]**2)))
#	print(weeks[-1], ':', pred[-1])
#	return (W_data, b_data)

def build_options():
	parser = argparse.ArgumentParser()
	parser.add_argument('--symbol', type=str,
		dest='symbol', help='stock symbol',
		metavar='SYM', default='NVDA')
	parser.add_argument('--down-sym', type=bool,
		dest='down_sym', help='whether or not to download daily stock data again',
		metavar='DOWN_SYM', default=False)
	parser.add_argument('--down-tre', type=bool,
		dest='down_tre', help='whether or not to download daily trend data again',
		metavar='DOWN_TRE', default=False)
	parser.add_argument('--data-loc', type=str,
		dest='data_loc', help='where stock data and goodle trends are stored',
		metavar='DATA_LOC', default='./data/day')
	parser.add_argument('--dropout', type=float,
		dest='dropout', help='probability of disabling an input channel',
		metavar='DROPOUT', default=0.5)
	parser.add_argument('--num-epoch', type=int,
		dest='num_epoch', help='number of epochs in traning',
		metavar='NUM_EPOCH', default=5000)
	return parser.parse_args()

# main function
if __name__ == '__main__':
	options = build_options()
#	df = load_and_train(options)

# generate data and get dimensions
x_data, y_data, weeks, _ = load_data(options)
num_data, ndim = x_data.shape
num_data -= 1
y_data = y_data.reshape((num_data+1, 1))

# construst the model
x = tf.placeholder(tf.float32, [None, ndim])

# supervise with known data
y = tf.placeholder(tf.float32, [None, 1])
