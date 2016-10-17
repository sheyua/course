# Tensorflow Tutorial

This repository contains a few examples of basic tensorflow usage and

- **Object Classification**

	utils/vgg.py implements caffe VGG19 for tensorflow
		www.robots.ox.ac.uk/~vgg/research/very_deep/
	execution of *basic_vgg19.py* should give

		tiger, Panthera tigris prob: 0.569161
		tiger cat prob: 0.421618
		zebra prob: 0.00868218

	for image
	![image](data/img/tiger.jpg?raw=true)
	and

		Samoyed, Samoyede prob: 0.833102
		Siberian husky prob: 0.0217978
		malamute, malemute, Alaskan malamute prob: 0.0196398
	
	for image
	![image](data/img/file.jpg?raw=true)

- **A Neural Algorithm of Artistic Style**

	*dev_art.py* implements a neural algorithm of artistic style transformation
		arxiv.org/abs/1508.06576

	![image](data/img/art1.jpg?raw=true)
