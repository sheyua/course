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
conv1_1 = vgg19.conv_layer(x_bgr, 'conv1_1')
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
conv1_2 = vgg19.conv_layer(conv1_1, 'conv1_2')
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
pool1 = vgg19.avg_pool(conv1_2, 'pool1')
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
conv2_1 = vgg19.conv_layer(pool1, 'conv2_1')
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
conv2_2 = vgg19.conv_layer(conv2_1, 'conv2_2')
'''
layers {
	bottom: "conv2_1"
	top: "conv2_2"
	name: "conv2_2"
	type: CONVOLUTION
	convolution_param {
		num_output: 128
		pad: 1
		kernel_size: 3
	}
}
layers {
	bottom: "conv2_2"
	top: "conv2_2"
	name: "relu2_2"
	type: RELU
}
'''
pool2 = vgg19.avg_pool(conv2_2, 'pool2')
'''
layers {
	bottom: "conv2_2"
	top: "pool2"
	name: "pool2"
	type: POOLING
	pooling_param {
		pool: MAX # Changed to AVG by 1508.06576
		kernel_size: 2
		stride: 2
	}
}
'''
conv3_1 = vgg19.conv_layer(pool2, 'conv3_1')
'''
layers {
	bottom: "pool2"
	top: "conv3_1"
	name: "conv3_1"
	type: CONVOLUTION
	convolution_param {
		num_output: 256
		pad: 1
		kernel_size: 3
	}
}
layers {
	bottom: "conv3_1"
	top: "conv3_1"
	name: "relu3_1"
	type: RELU
}
'''
conv3_2 = vgg19.conv_layer(conv3_1, 'conv3_2')
'''
layers {
	bottom: "conv3_1"
	top: "conv3_2"
	name: "conv3_2"
	type: CONVOLUTION
	convolution_param {
		num_output: 256
		pad: 1
		kernel_size: 3
	}
}
layers {
	bottom: "conv3_2"
	top: "conv3_2"
	name: "relu3_2"
	type: RELU
}
'''
conv3_3 = vgg19.conv_layer(conv3_2, 'conv3_3')
'''
layers {
	bottom: "conv3_2"
	top: "conv3_3"
	name: "conv3_3"
	type: CONVOLUTION
	convolution_param {
		num_output: 256
		pad: 1
		kernel_size: 3
	}
}
layers {
	bottom: "conv3_3"
	top: "conv3_3"
	name: "relu3_3"
	type: RELU
}
'''
conv3_4 = vgg19.conv_layer(conv3_3, 'conv3_4')
'''
layers {
	bottom: "conv3_3"
	top: "conv3_4"
	name: "conv3_4"
	type: CONVOLUTION
	convolution_param {
		num_output: 256
		pad: 1
		kernel_size: 3
	}
}
layers {
	bottom: "conv3_4"
	top: "conv3_4"
	name: "relu3_4"
	type: RELU
}
'''
pool3 = vgg19.avg_pool(conv3_4, 'pool3')
'''
layers {
	bottom: "conv3_4"
	top: "pool3"
	name: "pool3"
	type: POOLING
	pooling_param {
		pool: MAX # Changed to AVG by 1508.06576
		kernel_size: 2
		stride: 2
	}
}
'''
conv4_1 = vgg19.conv_layer(pool3, 'conv4_1')
'''
layers {
	bottom: "pool3"
	top: "conv4_1"
	name: "conv4_1"
	type: CONVOLUTION
	convolution_param {
		num_output: 512
		pad: 1
		kernel_size: 3
	}
}
layers {
	bottom: "conv4_1"
	top: "conv4_1"
	name: "relu4_1"
	type: RELU
}
'''
conv4_2 = vgg19.conv_layer(conv4_1, 'conv4_2')
'''
layers {
	bottom: "conv4_1"
	top: "conv4_2"
	name: "conv4_2"
	type: CONVOLUTION
	convolution_param {
		num_output: 512
		pad: 1
		kernel_size: 3
	}
}
layers {
	bottom: "conv4_2"
	top: "conv4_2"
	name: "relu4_2"
	type: RELU
}
'''
conv4_3 = vgg19.conv_layer(conv4_2, 'conv4_3')
'''
layers {
	bottom: "conv4_2"
	top: "conv4_3"
	name: "conv4_3"
	type: CONVOLUTION
	convolution_param {
		num_output: 512
		pad: 1
		kernel_size: 3
	}
}
layers {
	bottom: "conv4_3"
	top: "conv4_3"
	name: "relu4_3"
	type: RELU
}
'''
conv4_4 = vgg19.conv_layer(conv4_3, 'conv4_4')
'''
layers {
	bottom: "conv4_3"
	top: "conv4_4"
	name: "conv4_4"
	type: CONVOLUTION
	convolution_param {
		num_output: 512
		pad: 1
		kernel_size: 3
	}
}
layers {
	bottom: "conv4_4"
	top: "conv4_4"
	name: "relu4_4"
	type: RELU
}
'''
pool4 = vgg19.avg_pool(conv4_4, 'pool4')
'''
layers {
	bottom: "conv4_4"
	top: "pool4"
	name: "pool4"
	type: POOLING
	pooling_param {
		pool: MAX # Changed to AVG by 1508.06576
		kernel_size: 2
		stride: 2
	}
}
'''
conv5_1 = vgg19.conv_layer(pool4, 'conv5_1')
'''
layers {
	bottom: "pool4"
	top: "conv5_1"
	name: "conv5_1"
	type: CONVOLUTION
	convolution_param {
		num_output: 512
		pad: 1
		kernel_size: 3
	}
}
layers {
	bottom: "conv5_1"
	top: "conv5_1"
	name: "relu5_1"
	type: RELU
}
'''
conv5_2 = vgg19.conv_layer(conv5_1, 'conv5_2')
'''
layers {
	bottom: "conv5_1"
	top: "conv5_2"
	name: "conv5_2"
	type: CONVOLUTION
	convolution_param {
		num_output: 512
		pad: 1
		kernel_size: 3
	}
}
layers {
	bottom: "conv5_2"
	top: "conv5_2"
	name: "relu5_2"
	type: RELU
}
'''
conv5_3 = vgg19.conv_layer(conv5_2, 'conv5_3')
'''
layers {
	bottom: "conv5_2"
	top: "conv5_3"
	name: "conv5_3"
	type: CONVOLUTION
	convolution_param {
		num_output: 512
		pad: 1
		kernel_size: 3
	}
}
layers {
	bottom: "conv5_3"
	top: "conv5_3"
	name: "relu5_3"
	type: RELU
}
'''
conv5_4 = vgg19.conv_layer(conv5_3, 'conv5_4')
'''
layers {
	bottom: "conv5_3"
	top: "conv5_4"
	name: "conv5_4"
	type: CONVOLUTION
	convolution_param {
		num_output: 512
		pad: 1
		kernel_size: 3
	}
}
layers {
	bottom: "conv5_4"
	top: "conv5_4"
	name: "relu5_4"
	type: RELU
}
'''
pool5 = vgg19.avg_pool(conv5_4, 'pool5')
'''
layers {
	bottom: "conv5_4"
	top: "pool5"
	name: "pool5"
	type: POOLING
	pooling_param {
		pool: MAX # Changed to AVG by 1508.06576
		kernel_size: 2
		stride: 2
	}
}
'''


tf.InteractiveSession()
