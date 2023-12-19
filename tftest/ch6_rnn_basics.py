#!/usr/bin/python3 -i
'''
A simple example showing how to construct simple BasicRNNCell and BasicLSTMCell.
Cell variables are initialized as random numbers and can be accessed with the correct variable name.
This example shows how to evaluate these rnn cells both manually and by tf graph computation.
'''
import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt; plt.ion()

# $x_t \in R^{input\_size}$
batch_size = 2;
time_size = 10;
input_size = 8;
xdata = np.random.randn(batch_size, time_size, input_size)
# each $x_t$ has different length and is padded with zeros
xdata_length = [np.random.randint(1, input_size) for idx in range(batch_size)]
for idx in range(batch_size):
	xdata[idx, xdata_length[idx]:] = 0.0
xdata = xdata.astype(np.float32)
print('x_t:', batch_size, 'x', time_size, 'x', input_size)

# $rnn_h_t \in R^{num\_units}$
num_units = 4
cell = tf.nn.rnn_cell.BasicRNNCell(num_units)
print('h_t:', '?', 'x', cell.state_size)
# construct vanilla simple rnn
all_hiddens, last_states = tf.nn.dynamic_rnn(
	cell = cell,
	dtype = tf.float32,
	sequence_length = xdata_length,
	inputs = xdata
)
with tf.variable_scope('RNN') as vs:
	rnn_vars = [v for v in tf.all_variables() if v.name.startswith(vs.name)]
for v in rnn_vars:
	print(v.name, 'dimension:', v.get_shape().as_list())

# evaluate by tf ops
with tf.Session() as sess:
	# run the rnn by evaluating outputs
	tf.initialize_all_variables().run()
	W = rnn_vars[0].eval()
	b = rnn_vars[1].eval()
	out = all_hiddens.eval()
# evaluate manually
myout = np.zeros([batch_size, time_size, num_units], dtype=np.float32)
for batch in range(batch_size):
	h = np.zeros([1, num_units], dtype=np.float32)
	for t in range(xdata_length[batch]):
		in_and_hidden = np.zeros([1, input_size+num_units], dtype=np.float32)
		# input
		in_and_hidden[0, :input_size] = xdata[batch, t, :]
		# hidden state
		in_and_hidden[0, input_size:] = h
		# manual output
		myout[batch, t, :] = np.tanh( np.matmul(in_and_hidden, W)+b )[0,:]
		h = myout[batch, t, :]
# compare these two methods
fig, axarr = plt.subplots(batch_size, 2)
plt.suptitle('BasicRNNCell')
for batch in range(batch_size):
	axarr[batch, 0].pcolormesh(out[batch,:], vmin=-1, vmax=1)
	axarr[batch, 0].set_xlabel('num_units')
	axarr[batch, 0].set_ylabel('timesteps')
	axarr[batch, 0].set_title('tf ops, batch: '+str(batch))
	axarr[batch, 1].pcolormesh(myout[batch,:], vmin=-1, vmax=1)
	axarr[batch, 1].set_xlabel('num_units')
	axarr[batch, 1].set_ylabel('timesteps')
	axarr[batch, 1].set_title('manual, batch: '+str(batch))
plt.tight_layout()

# $lstm_h_t \in R^{2\times{}num\_units}$
num_units = 4
cell = tf.nn.rnn_cell.BasicLSTMCell(num_units, state_is_tuple=True, forget_bias=0.0)
print('c_t, h_t:', '?', 'x', cell.state_size)
# construct vanilla simple rnn
all_hiddens, last_states = tf.nn.dynamic_rnn(
	cell = cell,
	dtype = tf.float32,
	sequence_length = xdata_length,
	inputs = xdata
)
with tf.variable_scope('RNN') as vs:
	lstm_vars = [v for v in tf.all_variables() if v.name.startswith(vs.name) and v.name.find('LSTM') != -1]
for v in rnn_vars:
	print(v.name, 'dimension:', v.get_shape().as_list())

# evaluate by tf ops
with tf.Session() as sess:
	# run the rnn by evaluating outputs
	tf.initialize_all_variables().run()
	W = lstm_vars[0].eval()
	b = lstm_vars[1].eval()
	out = all_hiddens.eval()
	last_c = last_states[0].eval()
# evaluate manually
myout = np.zeros([batch_size, time_size, num_units], dtype=np.float32)
for batch in range(batch_size):
	c_and_h = np.zeros([1, 2*num_units], dtype=np.float32)
	for t in range(xdata_length[batch]):
		in_and_h = np.zeros([1, input_size+num_units], dtype=np.float32)
		# input
		in_and_h[0, :input_size] = xdata[batch, t, :]
		# c, hidden statetuple
		in_and_h[0, input_size:] = c_and_h[0,num_units:]
		# manual output
		linear = np.matmul(in_and_h, W) + b
		i = 1/(1+np.exp(-linear[0,:num_units])) # input_gate
		g = np.tanh(linear[0,num_units:(2*num_units)]) # new_input
		f = 1/(1+np.exp(-linear[0,(2*num_units):(3*num_units)])) # forget_gate
		o = 1/(1+np.exp(-linear[0,(3*num_units):])) # output_gate
		c_and_h[0,:num_units] = c_and_h[0,:num_units]*f + i*g
		c_and_h[0,num_units:] = o*np.tanh(c_and_h[0,:num_units])
		myout[batch, t, :] = c_and_h[0, num_units:]
# compare these two methods
fig, axarr = plt.subplots(batch_size, 2)
plt.suptitle('BasicLSTMCell')
for batch in range(batch_size):
	axarr[batch, 0].pcolormesh(out[batch,:], vmin=-1, vmax=1)
	axarr[batch, 0].set_xlabel('num_units')
	axarr[batch, 0].set_ylabel('timesteps')
	axarr[batch, 0].set_title('tf ops, batch: '+str(batch))
	axarr[batch, 1].pcolormesh(myout[batch,:], vmin=-1, vmax=1)
	axarr[batch, 1].set_xlabel('num_units')
	axarr[batch, 1].set_ylabel('timesteps')
	axarr[batch, 1].set_title('manual, batch: '+str(batch))
plt.tight_layout()
