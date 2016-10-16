import numpy as np
import scipy.misc
import argparse
from matplotlib import pyplot as plt

#def build_options():
#	parser = argparse.ArgumentParser()
#	'''
#	style images options
#	'''
#	parser.add_argument('--style_imgs',
#		dest='style_imgs', help='one or more style images',
#		metavar='STYLES_IMGS', nargs='+', required=True)
#	parser.add_argument('--style_scales', type=float,
#		dest='style_scales', help='one or more scales that downsize the style images',
#		metavar='STYLE_SCALES', nargs='+')
#	parser.add_argument('--style_scale', type=float,
#		dest='style_scale', help='one scale that downsizes all rest of style images',
#		metavar='STYLE_SCALE', default=1.0)
#	'''
#	output image options
#	'''
#	parser.add_argument('--width', type=int,
#		dest='width', help='width of the output image',
#		metavar='WIDTH')
#	'''
#	training parameters
#	'''
#	parser.add_argument('--vgg19_npy_path',
#		dest='vgg19_npy_path', help='path to the vgg 19 numpy ndarray file',
#		metavar='VGG19_NPY_PATH', default='vgg_19_npy/vgg19.npy')
#	# clean up to get options
#	options = parser.parse_args()
#	length = len(options.style_imgs)
#	if options.style_scales is not None:
#		while(len(options.style_scales)<=length):
#			options.style_scales.append(options.style_scale)
#		option.style_scales = option.style_scales[:length]
#	else:
#		options.style_scales = [options.style_scale for idx in range(length)]
#	return options

def load_img(path):
	return scipy.misc.imread(path).astype(np.float32)

# convert image to VGG style, BGR - mean
def convert_img(img):
	# R G B
	vgg_mean = np.array([123.68, 116.779, 103.939]).astype(np.float32)
	red, green, blue = [ img[:,:,idx] - vgg_mean[idx] for idx in range(3) ]
	res_img = np.empty(img.shape, np.float32)
	# B G R
	res_img[:,:,0] = blue
	res_img[:,:,1] = green
	res_img[:,:,2] = red
	return res_img

def resize_img(img, height=224, width=224):
	return scipy.misc.imresize(img, (height, width, 3)).astype(np.float32)

def view_layer(layer_output, label_chan=False):
	batch_size, height, width, nchan = layer_output.shape
	nx = np.ceil(np.sqrt(nchan)).astype(int)
	ny = np.ceil(1.0*nchan / nx).astype(int)
	for fdx in range(batch_size):
		plt.ion()
		fig, axarr = plt.subplots(ny, nx)
		for ydx in range(ny):
			for xdx in range(nx):
				chdx = ydx*nx + xdx
				if chdx >= nchan:
					axarr[ydx, xdx].axis('off')
				else:
					axarr[ydx, xdx].imshow(layer_output[fdx, :,:, chdx])
					axarr[ydx, xdx].set_xticks([])
					axarr[ydx, xdx].set_yticks([])
					if label_chan:
						axarr[ydx, xdx].text(0.5, 0.9, str(chdx+1),
						  horizontalalignment='center', transform=axarr[ydx, xdx].transAxes)
		plt.tight_layout()
		plt.subplots_adjust(hspace=0, wspace=0)
