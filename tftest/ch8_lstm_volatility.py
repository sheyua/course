#!/usr/bin/python3 -i
'''
An implementation of arxiv paper 1512.04916
Deep Learning Stock Volatility with Google Domestic Trends
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

def pack_data(xdata, ydata, max_lag = 20):
	num_data, in_dim = xdata.shape
	_, out_dim = ydata.shape
	num_data -= max_lag
	xdata_lstm = np.empty([num_data, max_lag, in_dim], dtype = np.float32)
	ydata_lstm = ydata[max_lag:, :].astype(np.float32)
	for idx in range(num_data):
		xdata_lstm[idx, :, :] = xdata[idx:(idx+max_lag), :]
	return xdata_lstm, ydata_lstm

def load_data(options):
	# download data
	download_trends(options)
	download_syms([options.symbol], options)
	# build input
	tre_lst = sym.trend_lst()
	df_sym = sym.day2week(pickle.load(open(options.data_loc+'/'+options.symbol+'.dat','rb')))
	df_tre_lst = {}
	interset = df_sym['Week']
	for trend in tre_lst:
		df_tre_lst[trend] = sym.day2week(pickle.load(open(options.data_loc+'/'+trend+'.dat','rb')))
		interset = np.intersect1d(interset, df_tre_lst[trend]['Week'])
	df = df_sym[df_sym['Week'].isin(interset)]
	for trend in tre_lst:
		df[trend] = df_tre_lst[trend]['Ret'][df_tre_lst[trend]['Week'].isin(interset)].values
	# pack into ordered time series
	in_col = [col for col in df.columns if col != 'Week' ]
	out_col = ['Sigma']
	xdata_all, ydata_all = pack_data( df[in_col][:-1].values, df[out_col][1:].values, options.max_lag )
	# shuffle
	num_data, _ = ydata_all.shape
	shuf_order = np.arange(num_data)
	np.random.shuffle(shuf_order)
	cut_loc = int((1.0-options.cross_frac)*num_data)
	assert( cut_loc>0 and cut_loc<num_data )
	# split into train cross
	xdata = {}
	ydata = {}
	xdata['train'] = xdata_all[shuf_order[:cut_loc], :]
	ydata['train'] = ydata_all[shuf_order[:cut_loc], :]
	xdata['cross']  = xdata_all[shuf_order[cut_loc:], :]
	ydata['cross']  = ydata_all[shuf_order[cut_loc:], :]
	xdata['last']  = df[in_col][-options.max_lag:].values.astype(np.float32).reshape([1, options.max_lag, -1])
	return xdata, ydata, in_col, out_col


def build_options():
	parser = argparse.ArgumentParser()
	parser.add_argument('--symbol', type=str,
		dest='symbol', help='stock symbol',
		metavar='SYM', default='NVDA')
	parser.add_argument('--num-units', type=int,
		dest='num_units', help='dimension of LSTM hidden/memory state',
		metavar='NUM_UNITS', default=1)
	parser.add_argument('--max-lag', type=int,
		dest='max_lag', help='maximum lag of weeks to keep lstm memory',
		metavar='MAX_LAG', default=20)
	parser.add_argument('--cross-frac', type=float,
		dest='cross_frac', help='fraction of data used as cross validation',
		metavar='CROSS_FRAC', default=0.2)
	parser.add_argument('--num-epoch', type=int,
		dest='num_epoch', help='number of epochs in traning',
		metavar='NUM_EPOCH', default=2000)
	parser.add_argument('--data-loc', type=str,
		dest='data_loc', help='where stock data and goodle trends are stored',
		metavar='DATA_LOC', default='./data/day')
	parser.add_argument('--down-sym', type=bool,
		dest='down_sym', help='whether or not to download daily stock data again',
		metavar='DOWN_SYM', default=True)
	parser.add_argument('--down-tre', type=bool,
		dest='down_tre', help='whether or not to download daily trend data again',
		metavar='DOWN_TRE', default=True)
	return parser.parse_args()

# main function
if __name__ == '__main__':
	options = build_options()
#	df = load_and_train(options)
options.down_sym = False
options.down_tre = False


# generate data and get dimensions
xdata, ydata, in_col, out_col = load_data(options)
in_dim = len(in_col)
out_dim = len(out_col)
print('Weekly Volatility mean in cross validation data:', np.mean(ydata['cross']) )

# construst the model
x = tf.placeholder(tf.float32, [None, options.max_lag, in_dim])
cell = tf.nn.rnn_cell.BasicLSTMCell(options.num_units, state_is_tuple = True, forget_bias = 0.0)
all_hiddens, last_states = tf.nn.dynamic_rnn(
	cell = cell,
	dtype = tf.float32,
	inputs = x
)
W = tf.Variable(tf.zeros([options.num_units, out_dim]))
b = tf.Variable(tf.zeros(out_dim))
h = tf.matmul(last_states[1], W) + b

# supervise with known labels
y = tf.placeholder(tf.float32, [None, out_dim])

# define loss function
cost_function = tf.reduce_mean( tf.abs( (h-y) / y ) )
train_step = tf.train.AdamOptimizer(0.005).minimize(cost_function)

with tf.Session() as sess:
	tf.initialize_all_variables().run()
	for idx in range(options.num_epoch):
		sess.run(train_step, feed_dict={x:xdata['train'], y:ydata['train']})
		print('epoch', idx, 
			': cross valid MAPE', cost_function.eval(feed_dict={x:xdata['cross'], y:ydata['cross']})
		)
	print('Next week volatility prediction:', h.eval(feed_dict={x:xdata['last']})[0,0])
