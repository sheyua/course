#!/usr/bin/python3 -i
'''
A demo of how to use VGG19 for object classification.
See utils.vgg for more information.
The original caffemodel:
	www.robots.ox.ac.uk/~vgg/research/very_deep/
has been translated in to numpy's ndarray:
	https://mega.nz/#!xZ8glS6J!MAnE91ND_WyfZ_8mvkuSa2YcA7q-1ehfSm-Q1fxOvvs
This implementation is adapted from :
	https://github.com/machrisaa/tensorflow-vgg.git
'''
import sys; sys.path.append('./utils')
import numpy as np
import tensorflow as tf
import img
import vgg

# load images and vgg19 coefficients
bgr = np.array( [
	img.convert_img(
		img.resize_img(
			img.load_img('./data/img/tiger.jpg')
		)
	),
	img.convert_img(
		img.resize_img(
			img.load_img('./data/img/file.jpg')
		)
	),
] )
vgg19 = vgg.VGG19('./data/vgg19.npy')

# build vgg19
_, height, width, _ = bgr.shape
x_bgr = vgg19.input_bgr(height, width)
vgg19.build_upto(x_bgr, 'prob')

# object classification
with tf.Session() as sess:
	prob = vgg19.layers['prob'].eval(feed_dict={x_bgr:bgr})
vgg19.predict(prob)
