#!/usr/bin/python3 -i
'''
An implementation of arxiv paper 1508.06576
A Neural Algorithm of Artistic Style
'''
import sys; sys.path.append('./utils')
import argparse
import scipy.misc
import numpy as np
import tensorflow as tf
import img
import vgg
from matplotlib import pyplot as plt

# load images and convert to vgg19 style
def load_sty_imgs(imgs=['./data/img/art1.jpg']):
	sty_imgs = []
	for image in imgs:
		sty_imgs.append(np.array([
			img.convert_img(
				img.load_img(image)
			)
		]))
	return sty_imgs

# load content image, resize and convert to vgg19 style
def load_cont_img(image='./data/img/file.jpg', scale=0.1):
	content = img.load_img(image)
	height, width, _ = content.shape
	height = int(height*scale)
	width = int(width*scale)
	content = img.resize_img(content, height, width)
	return np.array([img.convert_img(content)])

# default style features given by 1508.06576
def load_sty_features():
	return ['relu1_1', 'relu2_1', 'relu3_1', 'relu4_1', 'relu5_1']

# default content feature given by 1508.06576
def load_cont_feature():
	return 'relu4_2'

# mixing style features into gram matrix
def comp_sty_gram( sty_imgs, sty_weights, sty_features, vgg_obj ):
	# make sure all weights sums up to 1
	sty_weights = np.abs(sty_weights)
	sty_weights = sty_weights/np.sum(sty_weights)
	# build gram matrix
	sty_gram = []
	with tf.Session() as sess:
		print('')
		for idx in range(len(sty_imgs)):
			this_sty_gram = {}
			vgg_obj.build_upto( tf.constant(sty_imgs[idx]),'pool5',False )
			for style in sty_features:
				this_layer = vgg_obj.layers[style].eval()
				this_shape = this_layer.shape
				this_Ml = this_shape[1]*this_shape[2]
				reshaped = np.reshape(this_layer, (-1, this_shape[3]))
				this_sty_gram[style] = np.matmul( np.transpose(reshaped), reshaped ) / (this_Ml**2)
			sty_gram.append(this_sty_gram)
			print('Style features computed for image', idx,'!')
		print('')
	
	gram = {}
	gram_val = {}
	gram_coef = {}
	# weight over different styles
	for style in sty_features:
		gram_val[style] = sty_gram[0][style] * sty_weights[0]
		for idx in range(1, len(sty_imgs)):
			gram_val[style] += sty_gram[idx][style] * sty_weights[idx]
		gram_coef[style] = np.float32( 0.5/gram_val[style].shape[0]**2 )
		gram[style] = tf.constant(gram_val[style].astype(np.float32))
	
	return gram, gram_coef

# compute the tensor at the content feature layer
def comp_cont_ten( cont_img, cont_feature, vgg_obj ):
	vgg_obj.build_upto(tf.constant(cont_img), 'pool5', False)
	with tf.Session() as sess:
		print('')
		print('Content features computed!')
		print('')
		return tf.constant(vgg_obj.layers[cont_feature].eval())


def load_and_train(options):
	# unpack parameters
	sty_imgs = options.sty_imgs
	sty_weights = np.array(options.sty_weights)
	cont_img = options.cont_img
	output_file = options.output_file
	output_scale = options.output_scale
	learn_rate = options.learn_rate
	alpha = np.float32(options.alpha)
	beta = np.float32(options.beta)
	num_epoch = options.num_epoch
	vgg19_loc = options.vgg19_loc
	# load images and vgg19 coefficients
	sty_features = load_sty_features()
	cont_feature = load_cont_feature()
	cont = load_cont_img(cont_img, output_scale)
	vgg_obj = vgg.VGG19(vgg19_loc)
	cont_ten = comp_cont_ten( cont, cont_feature, vgg_obj )
	gram, gram_coef = comp_sty_gram( load_sty_imgs(sty_imgs), sty_weights, sty_features, vgg_obj )
	# model
	cont_remix = tf.Variable(cont)
	vgg_obj.build_upto(cont_remix, 'pool5', False)
	# style loss function
	gamma = np.float32(1.0/len(sty_features))
	gram_style = {}
	for style in sty_features:
		this_shape = vgg_obj.layers[style].get_shape().as_list()
		this_Ml = this_shape[1]*this_shape[2]
		reshaped = tf.reshape(vgg_obj.layers[style], (-1, this_shape[3]))
		gram_style[style] = tf.matmul(tf.transpose(reshaped), reshaped) / (this_Ml**2)
	loss_style = tf.constant(np.float32(0.0))
	for style in sty_features:
		loss_style += tf.reduce_sum( tf.square(gram_style[style] - gram[style]) )*gram_coef[style]
	# content loss function
	loss_content = tf.reduce_mean( tf.square(vgg_obj.layers[cont_feature] - cont_ten) )
	# punish local pixel noise
	loss_noise = tf.reduce_mean( tf.abs( 
		tf.nn.max_pool(cont_remix, ksize=[1,3,3,1], strides=[1,1,1,1], padding='VALID') - 
		tf.nn.max_pool(-cont_remix, ksize=[1,3,3,1], strides=[1,1,1,1], padding='VALID')
	))
	# train step
	loss = gamma*loss_style + alpha*loss_content + beta*loss_noise
	err = float('inf')
	train_step = tf.train.AdamOptimizer(learn_rate).minimize(loss)
	with tf.Session() as sess:
		tf.initialize_all_variables().run()
		for idx in range(num_epoch):
			sess.run(train_step)
			# list all errors
			this_loss_content = alpha*loss_content.eval()
			this_loss_style = gamma*loss_style.eval()
			this_loss_noise = beta*loss_noise.eval()
			this_err = this_loss_content + this_loss_style + this_loss_noise
			print('epoch', idx, 
				': content loss', this_loss_content, 
				'style loss', this_loss_style,
				'noise loss', this_loss_noise
			)
			if this_err < err:
				err = this_err
				output = cont_remix.eval()[0,:,:,:]
	# save image
	img.save_img(output_file, img.revert_img(output))

def build_options():
	parser = argparse.ArgumentParser()
	parser.add_argument('--sty-imgs',
		dest='sty_imgs', nargs='+', help='one or more style images',
		metavar='STY_IMGS', default=['./data/img/art1.jpg', './data/img/art2.jpg'])
	parser.add_argument('--sty-weights',
		dest='sty_weights', nargs='+', help='mixing weights of style images',
		metavar='STY_WEIGHTS', default=[0.5, 0.5])
	parser.add_argument('--cont-img', type=str,
		dest='cont_img', help='content picture which you wish to change its style',
		metavar='CONT_IMG', default='./data/img/file.jpg')
	parser.add_argument('--output-file', type=str,
		dest='output_file', help='location to save the output file',
		metavar='VGG19_LOC', default='./data/img/output.jpg')
	parser.add_argument('--output-scale', type=float,
		dest='output_scale', help='a factor to rescale the output picture',
		metavar='OUTPUT_SCALE', default=0.1)
	parser.add_argument('--learn-rate', type=float,
		dest='learn_rate', help='learning rate for mixing content and styles',
		metavar='LEARN_RATE', default=10.0)
	parser.add_argument('--alpha', type=float,
		dest='alpha', help='mixing weight of the content picture',
		metavar='ALPHA', default=1e-5)
	parser.add_argument('--beta', type=float,
		dest='beta', help='mixing weight of picture noise',
		metavar='BETA', default=1e-2)
	parser.add_argument('--num-epoch', type=int,
		dest='num_epoch', help='number of epochs in traning',
		metavar='NUM_EPOCH', default=1000)
	parser.add_argument('--vgg19-loc', type=str,
		dest='vgg19_loc', help='location of the vgg19 *.npy file',
		metavar='VGG19_LOC', default='./data/vgg19.npy')
	return parser.parse_args()

# main function
if __name__ == '__main__':
	options = build_options()
	load_and_train(options)
