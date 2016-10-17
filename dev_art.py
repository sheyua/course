import sys; sys.path.append('./utils')
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

# load images and vgg19 coefficients
sty_features = load_sty_features()
cont_feature = load_cont_feature()
cont = load_cont_img()
_, height, width, _ = cont.shape

vgg_obj = vgg.VGG19('./data/vgg19.npy')
gram, gram_coef = comp_sty_gram( load_sty_imgs(), np.array([1.0]), sty_features, vgg_obj )
cont_ten = comp_cont_ten( cont, cont_feature, vgg_obj )

cont_remix = tf.Variable(cont)
vgg_obj.build_upto(cont_remix, 'pool5', False)
loss_1 = 1e-5 * tf.reduce_mean( tf.square(vgg_obj.layers[cont_feature] - cont_ten) )
loss_2 = tf.constant(np.float32(0.0))
gram_style = {}
for style in sty_features:
	this_shape = vgg_obj.layers[style].get_shape().as_list()
	this_Ml = this_shape[1]*this_shape[2]
	reshaped = tf.reshape(vgg_obj.layers[style], (-1, this_shape[3]))
	gram_style[style] = tf.matmul(tf.transpose(reshaped), reshaped) / (this_Ml**2)

for style in sty_features:
	loss_2 += tf.reduce_sum( tf.square(gram_style[style] - gram[style]) )*gram_coef[style]

loss = loss_1 + 0.2*loss_2
err = float('inf')
train_step = tf.train.AdamOptimizer(10.0).minimize(loss)
with tf.Session() as sess:
	tf.initialize_all_variables().run()
	for idx in range(1000):
		sess.run(train_step)
		print(idx, loss_1.eval(), loss_2.eval())
		this_err = loss.eval()
		if this_err < err:
			err = this_err
			output = cont_remix.eval()[0,:,:,:]

import scipy.misc
scipy.misc.imsave('./data/img/output.jpg', img.revert_img(output).astype(np.uint8))
