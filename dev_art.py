import sys; sys.path.append('./utils')
import numpy as np
import tensorflow as tf
import img
import vgg
from matplotlib import pyplot as plt

bgr = np.array( [img.convert_img(img.load_img('./data/img/art.jpg'))] )
_, height, width, _ = bgr.shape
vgg19 = vgg.VGG19('./data/vgg19.npy')

x_bgr = vgg19.input_bgr(height, width)
'''
layers {
	bottom: "data"
	top: "conv1_1"
	name: "conv1_1"
	type: CONVOLUTION
	convolution_param {
		num_output: 64
		pad: 1
		kernel_size: 3
	}
}
layers {
	bottom: "conv1_1"
	top: "conv1_1"
	name: "relu1_1"
	type: RELU
}
'''
conv1_1 = vgg19.conv_layer(x_bgr, 'conv1_1')
'''
layers {
	bottom: "conv1_1"
	top: "conv1_2"
	name: "conv1_2"
	type: CONVOLUTION
	convolution_param {
		num_output: 64
		pad: 1
		kernel_size: 3
	}
}
layers {
	bottom: "conv1_2"
	top: "conv1_2"
	name: "relu1_2"
	type: RELU
}
'''
conv1_2 = vgg19.conv_layer(conv1_1, 'conv1_2')
'''
layers {
	bottom: "conv1_2"
	top: "pool1"
	name: "pool1"
	type: POOLING
	pooling_param {
		pool: MAX # Changed to AVG by 1508.06576
		kernel_size: 2
		stride: 2
	}
}
'''
pool1 = vgg19.avg_pool(conv1_2, 'pool1')
'''
layers {
	bottom: "pool1"
	top: "conv2_1"
	name: "conv2_1"
	type: CONVOLUTION
	convolution_param {
		num_output: 128
		pad: 1
		kernel_size: 3
	}
}
layers {
	bottom: "conv2_1"
	top: "conv2_1"
	name: "relu2_1"
	type: RELU
}
'''
conv2_1 = vgg19.conv_layer(pool1, 'conv2_1')

tf.InteractiveSession()
