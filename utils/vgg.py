import numpy as np
import tensorflow as tf

class VGG19:
	def __init__(self, path):
		"""
		The original caffemodel:
			www.robots.ox.ac.uk/~vgg/research/very_deep/
		has been translated in to numpy's ndarray
			https://mega.nz/#!xZ8glS6J!MAnE91ND_WyfZ_8mvkuSa2YcA7q-1ehfSm-Q1fxOvvs
		and can be loaded here
		"""
		print('loading pre-trained VGG coefficients...')
		self.data_dict = np.load(path, encoding='latin1').item()
		print('initialized!')
	
	# 'D' and 'E' weights layers - 19 in total
	# either conv or fc
	def load_weights(self, name):
		return tf.constant(self.data_dict[name][0], name=name+'_weights')
	
	def load_bias(self, name):
		return tf.constant(self.data_dict[name][1], name=name+'_bias')
	
	# layer constructors
	def input_bgr(self, height, width):
		# batch size undetermined, 3 channel must be BGR
		return tf.placeholder(tf.float32, [None, height, width, 3])
	
	def conv_layer(self, bottom, name, if_relu=True):
		# strides=[1, ?, ?, 1] for 2d images
		# all vgg19 conv layers have ?, ? = 1, 1
		conv = tf.nn.conv2d( bottom, self.load_weights(name), strides=[1,1,1,1], padding='SAME', name=name)
		conv = tf.nn.bias_add(conv, self.load_bias(name), name=name+'_biased')
		if if_relu:
			return tf.nn.relu(conv, name=name+'_biased'+'_relu')
		else:
			return conv
	
	def avg_pool(self, bottom, name):
		# kernel_size=[1, ?, ?, 1] for 2d images
		# strides=[1, ?, ?, 1] for 2d images
		# all vgg19 pool layers have ?, ? = 2, 2
		return tf.nn.avg_pool(bottom, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name=name)
	
	def max_pool(self, bottom, name):
		# kernel_size=[1, ?, ?, 1] for 2d images
		# strides=[1, ?, ?, 1] for 2d images
		# all vgg19 pool layers have ?, ? = 2, 2
		return tf.nn.max_pool(bottom, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name=name)
	
	def fc_layer(self, bottom, name):
		with tf.variable_scope(name):
			shape = bottom.get_shape().as_list()
			dim = 1
			for d in shape[1:]:
				dim *= d
			fc_weights = self.load_fc_weights(name)
			fc_bias = self.load_fc_bias(name)
			fc = tf.matmul( tf.reshape(bottom, [-1, dim]), fc_weights)
			fc = tf.nn.bias_add(fc, fc_bias)
			return fc
