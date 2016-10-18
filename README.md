# Tensorflow Tutorial

This repository contains a few examples of basic tensorflow usage and

- **Object Classification**

	*utils/vgg.py* implements caffe model VGG19 for tensorflow
	www.robots.ox.ac.uk/~vgg/research/very_deep/.
	execution of *basic_vgg19.py* should give:

		tiger, Panthera tigris prob: 0.569161
		tiger cat prob: 0.421618
		zebra prob: 0.00868218
		for image
<p align="center">
  <img src="https://github.com/sy0302/TensorflowTutorial/blob/master/data/img/tiger.jpg" width="250">
</p>

	and

		Samoyed, Samoyede prob: 0.833102
		Siberian husky prob: 0.0217978
		malamute, malemute, Alaskan malamute prob: 0.0196398
		for image
<p align="center">
  <img src="https://github.com/sy0302/TensorflowTutorial/blob/master/data/img/file.jpg" height="250"/>.
</p>

- **A Neural Algorithm of Artistic Style**

	*neural_styler.py* implements a neural algorithm of artistic style transformation
	arxiv.org/abs/1508.06576
	mixing the artistic style of Georges Seurat's A Sunday Afternoon on the Island of La Grande Jatte
<p align="center">
  <img src="https://github.com/sy0302/TensorflowTutorial/blob/master/data/img/art1.jpg" height="200"/>
</p>
	and Vincent van Gogh's The Starry Night
<p align="center">
  <img src="https://github.com/sy0302/TensorflowTutorial/blob/master/data/img/art2.jpg" height="200"/>
</p>
	into the above image and obtain
<p align="center">
  <img src="https://github.com/sy0302/TensorflowTutorial/blob/master/data/img/output.jpg" height="250"/>.
</p>
