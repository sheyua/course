import numpy as np
import tensorflow as tf

class VGG19:
	def __init__(self, path):
		"""
		The original caffemodel:
			www.robots.ox.ac.uk/~vgg/research/very_deep/
		has been translated in to numpy's ndarray:
			https://mega.nz/#!xZ8glS6J!MAnE91ND_WyfZ_8mvkuSa2YcA7q-1ehfSm-Q1fxOvvs
		This implementation is adapted from :
			https://github.com/machrisaa/tensorflow-vgg.git
		"""
		print('loading pre-trained VGG coefficients...')
		self.data_dict = np.load(path, encoding='latin1').item()
		print('initialized!')
		self.layer_lst = [
			'conv1_1', 'relu1_1', 'conv1_2', 'relu1_2', 'pool1',
			'conv2_1', 'relu2_1', 'conv2_2', 'relu2_1', 'pool2',
			'conv3_1', 'relu3_1', 'conv3_2', 'relu3_2', 'conv3_3', 'relu3_3', 'conv3_4', 'relu3_4', 'pool3',
			'conv4_1', 'relu4_1', 'conv4_2', 'relu4_2', 'conv4_3', 'relu4_3', 'conv4_4', 'relu4_4', 'pool4',
			'conv5_1', 'relu5_1', 'conv5_2', 'relu5_2', 'conv5_3', 'relu5_3', 'conv5_4', 'relu5_4', 'pool5',
			'fc6', 'relu6', 'drop6',
			'fc7', 'relu7', 'drop6',
			'fc8',
			'prob'
		]
	
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
	
	def conv_layer(self, bottom, name):
		# strides=[1, ?, ?, 1] for 2d images
		# all vgg19 conv layers have ?, ? = 1, 1
		conv = tf.nn.conv2d( bottom, self.load_weights(name), strides=[1,1,1,1], padding='SAME', name=name)
		return tf.nn.bias_add(conv, self.load_bias(name), name=name+'_biased')
	
	def relu_layer(self, bottom, name):
		return tf.nn.relu(bottom, name=name)
	
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
		# make sure that dimension matches for using fc layers
		dim = 1
		for d in bottom.get_shape().as_list()[1:]:
			dim *= d
		fc_weights = self.load_weights(name)
		assert dim == fc_weights.get_shape().as_list()[0]
		# construct
		fc = tf.matmul( tf.reshape(bottom, [-1, dim]), fc_weights, name=name )
		return tf.nn.bias_add( fc, self.load_bias(name), name=name+'_biased' )
	
	def drop_layer(self, bottom, name):
		# dropout probablity always 0.5
		return tf.nn.dropout(bottom, keep_prob=0.5, name=name)
	
	def build_upto(self, bottom, top_layer, use_max_pool=True):
		self.layers = {}
		prev_layer = ''
		for layer in self.layer_lst:
			# first conv
			if layer == 'conv1_1':
				self.layers[layer] = self.conv_layer(bottom, layer)
			elif layer[:2] == 'co':
				self.layers[layer] = self.conv_layer(self.layers[prev_layer], layer)
			elif layer[:2] == 're':
				self.layers[layer] = self.relu_layer(self.layers[prev_layer], layer)
			elif layer[:2] == 'po':
				if use_max_pool:
					self.layers[layer] = self.max_pool(self.layers[prev_layer], layer)
				else:
					self.layers[layer] = self.avg_pool(self.layers[prev_layer], layer)
			elif layer[:2] == 'fc':
				self.layers[layer] = self.fc_layer(self.layers[prev_layer], layer)
			elif layer[:2] == 'dr':
				self.layers[layer] = self.drop_layer(self.layers[prev_layer], layer)
			elif layer[:2] == 'pr':
				self.layers[layer] = tf.nn.softmax(self.layers[prev_layer], name=layer)
			# update to the next layer
			prev_layer = layer
			if layer == top_layer:
				break
	
	def predict(self, prob, num=5):
		ncase, nobj = prob.shape
		assert nobj == 1000
		if not hasattr(self, 'obj_lst'):
			self.creat_obj_lst()
		for idx in range(ncase):
			print('Image',idx,':')
			top_num = prob[idx,:].argsort()[::-1][:num]
			for jdx in top_num:
				print(self.obj_lst[jdx],'prob:',prob[idx,jdx])
			print('')

	# [None, 224, 224, 3]
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
	# [None, 112, 112, 64]
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
	# [None, 112, 112, 64]
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
	# [None, 112, 112, 64]
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
	# [None, 56, 56, 128]
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
	# [None, 56, 56, 128]
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
	# [None, 56, 56, 128]
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
	# [None, 28, 28, 256]
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
	# [None, 28, 28, 256]
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
	# [None, 28, 28, 256]
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
	# [None, 28, 28, 256]
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
	# [None, 28, 28, 256]
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
	# [None, 14, 14, 512]
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
	# [None, 14, 14, 512]
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
	# [None, 14, 14, 512]
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
	# [None, 14, 14, 512]
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
	# [None, 14, 14, 512]
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
	# [None, 7, 7, 512]
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
	# [None, 7, 7, 512]
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
	# [None, 7, 7, 512]
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
	# [None, 7, 7, 512]
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
	# [None, 7, 7, 512]
	'''
	layers {
		bottom: "pool5"
		top: "fc6"
		name: "fc6"
		type: INNER_PRODUCT
		inner_product_param {
			num_output: 4096
		}
	}
	layers {
		bottom: "fc6"
		top: "fc6"
		name: "relu6"
		type: RELU
	}
	'''
	# [None, 4096]
	'''
	layers {
		bottom: "fc6"
		top: "fc6"
		name: "drop6"
		type: DROPOUT
		dropout_param {
			dropout_ratio: 0.5
		}
	}
	'''
	# [None, 4096]
	'''
	layers {
		bottom: "fc6"
		top: "fc7"
		name: "fc7"
		type: INNER_PRODUCT
		inner_product_param {
			num_output: 4096
		}
	}
	layers {
		bottom: "fc7"
		top: "fc7"
		name: "relu7"
		type: RELU
	}
	'''
	# [None, 4096]
	'''
	layers {
		bottom: "fc7"
		top: "fc7"
		name: "drop7"
		type: DROPOUT
		dropout_param {
			dropout_ratio: 0.5
		}
	}
	'''
	# [None, 4096]
	'''
	layers {
		bottom: "fc7"
		top: "fc8"
		name: "fc8"
		type: INNER_PRODUCT
		inner_product_param {
			num_output: 1000
		}
	}
	'''
	# [None, 1000]
	'''
	layers {
		bottom: "fc8"
		top: "prob"
		name: "prob"
		type: SOFTMAX
	}
	'''
	def creat_obj_lst(self):
		self.obj_lst = [
			"tench, Tinca tinca",
			"goldfish, Carassius auratus",
			"great white shark, white shark, man-eater, man-eating shark, Carcharodon carcharias",
			"tiger shark, Galeocerdo cuvieri",
			"hammerhead, hammerhead shark",
			"electric ray, crampfish, numbfish, torpedo",
			"stingray",
			"cock",
			"hen",
			"ostrich, Struthio camelus",
			"brambling, Fringilla montifringilla",
			"goldfinch, Carduelis carduelis",
			"house finch, linnet, Carpodacus mexicanus",
			"junco, snowbird",
			"indigo bunting, indigo finch, indigo bird, Passerina cyanea",
			"robin, American robin, Turdus migratorius",
			"bulbul",
			"jay",
			"magpie",
			"chickadee",
			"water ouzel, dipper",
			"kite",
			"bald eagle, American eagle, Haliaeetus leucocephalus",
			"vulture",
			"great grey owl, great gray owl, Strix nebulosa",
			"European fire salamander, Salamandra salamandra",
			"common newt, Triturus vulgaris",
			"eft",
			"spotted salamander, Ambystoma maculatum",
			"axolotl, mud puppy, Ambystoma mexicanum",
			"bullfrog, Rana catesbeiana",
			"tree frog, tree-frog",
			"tailed frog, bell toad, ribbed toad, tailed toad, Ascaphus trui",
			"loggerhead, loggerhead turtle, Caretta caretta",
			"leatherback turtle, leatherback, leathery turtle, Dermochelys coriacea",
			"mud turtle",
			"terrapin",
			"box turtle, box tortoise",
			"banded gecko",
			"common iguana, iguana, Iguana iguana",
			"American chameleon, anole, Anolis carolinensis",
			"whiptail, whiptail lizard",
			"agama",
			"frilled lizard, Chlamydosaurus kingi",
			"alligator lizard",
			"Gila monster, Heloderma suspectum",
			"green lizard, Lacerta viridis",
			"African chameleon, Chamaeleo chamaeleon",
			"Komodo dragon, Komodo lizard, dragon lizard, giant lizard, Varanus komodoensis",
			"African crocodile, Nile crocodile, Crocodylus niloticus",
			"American alligator, Alligator mississipiensis",
			"triceratops",
			"thunder snake, worm snake, Carphophis amoenus",
			"ringneck snake, ring-necked snake, ring snake",
			"hognose snake, puff adder, sand viper",
			"green snake, grass snake",
			"king snake, kingsnake",
			"garter snake, grass snake",
			"water snake",
			"vine snake",
			"night snake, Hypsiglena torquata",
			"boa constrictor, Constrictor constrictor",
			"rock python, rock snake, Python sebae",
			"Indian cobra, Naja naja",
			"green mamba",
			"sea snake",
			"horned viper, cerastes, sand viper, horned asp, Cerastes cornutus",
			"diamondback, diamondback rattlesnake, Crotalus adamanteus",
			"sidewinder, horned rattlesnake, Crotalus cerastes",
			"trilobite",
			"harvestman, daddy longlegs, Phalangium opilio",
			"scorpion",
			"black and gold garden spider, Argiope aurantia",
			"barn spider, Araneus cavaticus",
			"garden spider, Aranea diademata",
			"black widow, Latrodectus mactans",
			"tarantula",
			"wolf spider, hunting spider",
			"tick",
			"centipede",
			"black grouse",
			"ptarmigan",
			"ruffed grouse, partridge, Bonasa umbellus",
			"prairie chicken, prairie grouse, prairie fowl",
			"peacock",
			"quail",
			"partridge",
			"African grey, African gray, Psittacus erithacus",
			"macaw",
			"sulphur-crested cockatoo, Kakatoe galerita, Cacatua galerita",
			"lorikeet",
			"coucal",
			"bee eater",
			"hornbill",
			"hummingbird",
			"jacamar",
			"toucan",
			"drake",
			"red-breasted merganser, Mergus serrator",
			"goose",
			"black swan, Cygnus atratus",
			"tusker",
			"echidna, spiny anteater, anteater",
			"platypus, duckbill, duckbilled platypus, duck-billed platypus, Ornithorhynchus anatinus",
			"wallaby, brush kangaroo",
			"koala, koala bear, kangaroo bear, native bear, Phascolarctos cinereus",
			"wombat",
			"jellyfish",
			"sea anemone, anemone",
			"brain coral",
			"flatworm, platyhelminth",
			"nematode, nematode worm, roundworm",
			"conch",
			"snail",
			"slug",
			"sea slug, nudibranch",
			"chiton, coat-of-mail shell, sea cradle, polyplacophore",
			"chambered nautilus, pearly nautilus, nautilus",
			"Dungeness crab, Cancer magister",
			"rock crab, Cancer irroratus",
			"fiddler crab",
			"king crab, Alaska crab, Alaskan king crab, Alaska king crab, Paralithodes camtschatica",
			"American lobster, Northern lobster, Maine lobster, Homarus americanus",
			"spiny lobster, langouste, rock lobster, crawfish, crayfish, sea crawfish",
			"crayfish, crawfish, crawdad, crawdaddy",
			"hermit crab",
			"isopod",
			"white stork, Ciconia ciconia",
			"black stork, Ciconia nigra",
			"spoonbill",
			"flamingo",
			"little blue heron, Egretta caerulea",
			"American egret, great white heron, Egretta albus",
			"bittern",
			"crane",
			"limpkin, Aramus pictus",
			"European gallinule, Porphyrio porphyrio",
			"American coot, marsh hen, mud hen, water hen, Fulica americana",
			"bustard",
			"ruddy turnstone, Arenaria interpres",
			"red-backed sandpiper, dunlin, Erolia alpina",
			"redshank, Tringa totanus",
			"dowitcher",
			"oystercatcher, oyster catcher",
			"pelican",
			"king penguin, Aptenodytes patagonica",
			"albatross, mollymawk",
			"grey whale, gray whale, devilfish, Eschrichtius gibbosus, Eschrichtius robustus",
			"killer whale, killer, orca, grampus, sea wolf, Orcinus orca",
			"dugong, Dugong dugon",
			"sea lion",
			"Chihuahua",
			"Japanese spaniel",
			"Maltese dog, Maltese terrier, Maltese",
			"Pekinese, Pekingese, Peke",
			"Shih-Tzu",
			"Blenheim spaniel",
			"papillon",
			"toy terrier",
			"Rhodesian ridgeback",
			"Afghan hound, Afghan",
			"basset, basset hound",
			"beagle",
			"bloodhound, sleuthhound",
			"bluetick",
			"black-and-tan coonhound",
			"Walker hound, Walker foxhound",
			"English foxhound",
			"redbone",
			"borzoi, Russian wolfhound",
			"Irish wolfhound",
			"Italian greyhound",
			"whippet",
			"Ibizan hound, Ibizan Podenco",
			"Norwegian elkhound, elkhound",
			"otterhound, otter hound",
			"Saluki, gazelle hound",
			"Scottish deerhound, deerhound",
			"Weimaraner",
			"Staffordshire bullterrier, Staffordshire bull terrier",
			"American Staffordshire terrier, Staffordshire terrier, American pit bull terrier, pit bull terrier",
			"Bedlington terrier",
			"Border terrier",
			"Kerry blue terrier",
			"Irish terrier",
			"Norfolk terrier",
			"Norwich terrier",
			"Yorkshire terrier",
			"wire-haired fox terrier",
			"Lakeland terrier",
			"Sealyham terrier, Sealyham",
			"Airedale, Airedale terrier",
			"cairn, cairn terrier",
			"Australian terrier",
			"Dandie Dinmont, Dandie Dinmont terrier",
			"Boston bull, Boston terrier",
			"miniature schnauzer",
			"giant schnauzer",
			"standard schnauzer",
			"Scotch terrier, Scottish terrier, Scottie",
			"Tibetan terrier, chrysanthemum dog",
			"silky terrier, Sydney silky",
			"soft-coated wheaten terrier",
			"West Highland white terrier",
			"Lhasa, Lhasa apso",
			"flat-coated retriever",
			"curly-coated retriever",
			"golden retriever",
			"Labrador retriever",
			"Chesapeake Bay retriever",
			"German short-haired pointer",
			"vizsla, Hungarian pointer",
			"English setter",
			"Irish setter, red setter",
			"Gordon setter",
			"Brittany spaniel",
			"clumber, clumber spaniel",
			"English springer, English springer spaniel",
			"Welsh springer spaniel",
			"cocker spaniel, English cocker spaniel, cocker",
			"Sussex spaniel",
			"Irish water spaniel",
			"kuvasz",
			"schipperke",
			"groenendael",
			"malinois",
			"briard",
			"kelpie",
			"komondor",
			"Old English sheepdog, bobtail",
			"Shetland sheepdog, Shetland sheep dog, Shetland",
			"collie",
			"Border collie",
			"Bouvier des Flandres, Bouviers des Flandres",
			"Rottweiler",
			"German shepherd, German shepherd dog, German police dog, alsatian",
			"Doberman, Doberman pinscher",
			"miniature pinscher",
			"Greater Swiss Mountain dog",
			"Bernese mountain dog",
			"Appenzeller",
			"EntleBucher",
			"boxer",
			"bull mastiff",
			"Tibetan mastiff",
			"French bulldog",
			"Great Dane",
			"Saint Bernard, St Bernard",
			"Eskimo dog, husky",
			"malamute, malemute, Alaskan malamute",
			"Siberian husky",
			"dalmatian, coach dog, carriage dog",
			"affenpinscher, monkey pinscher, monkey dog",
			"basenji",
			"pug, pug-dog",
			"Leonberg",
			"Newfoundland, Newfoundland dog",
			"Great Pyrenees",
			"Samoyed, Samoyede",
			"Pomeranian",
			"chow, chow chow",
			"keeshond",
			"Brabancon griffon",
			"Pembroke, Pembroke Welsh corgi",
			"Cardigan, Cardigan Welsh corgi",
			"toy poodle",
			"miniature poodle",
			"standard poodle",
			"Mexican hairless",
			"timber wolf, grey wolf, gray wolf, Canis lupus",
			"white wolf, Arctic wolf, Canis lupus tundrarum",
			"red wolf, maned wolf, Canis rufus, Canis niger",
			"coyote, prairie wolf, brush wolf, Canis latrans",
			"dingo, warrigal, warragal, Canis dingo",
			"dhole, Cuon alpinus",
			"African hunting dog, hyena dog, Cape hunting dog, Lycaon pictus",
			"hyena, hyaena",
			"red fox, Vulpes vulpes",
			"kit fox, Vulpes macrotis",
			"Arctic fox, white fox, Alopex lagopus",
			"grey fox, gray fox, Urocyon cinereoargenteus",
			"tabby, tabby cat",
			"tiger cat",
			"Persian cat",
			"Siamese cat, Siamese",
			"Egyptian cat",
			"cougar, puma, catamount, mountain lion, painter, panther, Felis concolor",
			"lynx, catamount",
			"leopard, Panthera pardus",
			"snow leopard, ounce, Panthera uncia",
			"jaguar, panther, Panthera onca, Felis onca",
			"lion, king of beasts, Panthera leo",
			"tiger, Panthera tigris",
			"cheetah, chetah, Acinonyx jubatus",
			"brown bear, bruin, Ursus arctos",
			"American black bear, black bear, Ursus americanus, Euarctos americanus",
			"ice bear, polar bear, Ursus Maritimus, Thalarctos maritimus",
			"sloth bear, Melursus ursinus, Ursus ursinus",
			"mongoose",
			"meerkat, mierkat",
			"tiger beetle",
			"ladybug, ladybeetle, lady beetle, ladybird, ladybird beetle",
			"ground beetle, carabid beetle",
			"long-horned beetle, longicorn, longicorn beetle",
			"leaf beetle, chrysomelid",
			"dung beetle",
			"rhinoceros beetle",
			"weevil",
			"fly",
			"bee",
			"ant, emmet, pismire",
			"grasshopper, hopper",
			"cricket",
			"walking stick, walkingstick, stick insect",
			"cockroach, roach",
			"mantis, mantid",
			"cicada, cicala",
			"leafhopper",
			"lacewing, lacewing fly",
			"dragonfly, darning needle, devil's darning needle, sewing needle, snake feeder, snake doctor, mosquito hawk, skeeter hawk",
			"damselfly",
			"admiral",
			"ringlet, ringlet butterfly",
			"monarch, monarch butterfly, milkweed butterfly, Danaus plexippus",
			"cabbage butterfly",
			"sulphur butterfly, sulfur butterfly",
			"lycaenid, lycaenid butterfly",
			"starfish, sea star",
			"sea urchin",
			"sea cucumber, holothurian",
			"wood rabbit, cottontail, cottontail rabbit",
			"hare",
			"Angora, Angora rabbit",
			"hamster",
			"porcupine, hedgehog",
			"fox squirrel, eastern fox squirrel, Sciurus niger",
			"marmot",
			"beaver",
			"guinea pig, Cavia cobaya",
			"sorrel",
			"zebra",
			"hog, pig, grunter, squealer, Sus scrofa",
			"wild boar, boar, Sus scrofa",
			"warthog",
			"hippopotamus, hippo, river horse, Hippopotamus amphibius",
			"ox",
			"water buffalo, water ox, Asiatic buffalo, Bubalus bubalis",
			"bison",
			"ram, tup",
			"bighorn, bighorn sheep, cimarron, Rocky Mountain bighorn, Rocky Mountain sheep, Ovis canadensis",
			"ibex, Capra ibex",
			"hartebeest",
			"impala, Aepyceros melampus",
			"gazelle",
			"Arabian camel, dromedary, Camelus dromedarius",
			"llama",
			"weasel",
			"mink",
			"polecat, fitch, foulmart, foumart, Mustela putorius",
			"black-footed ferret, ferret, Mustela nigripes",
			"otter",
			"skunk, polecat, wood pussy",
			"badger",
			"armadillo",
			"three-toed sloth, ai, Bradypus tridactylus",
			"orangutan, orang, orangutang, Pongo pygmaeus",
			"gorilla, Gorilla gorilla",
			"chimpanzee, chimp, Pan troglodytes",
			"gibbon, Hylobates lar",
			"siamang, Hylobates syndactylus, Symphalangus syndactylus",
			"guenon, guenon monkey",
			"patas, hussar monkey, Erythrocebus patas",
			"baboon",
			"macaque",
			"langur",
			"colobus, colobus monkey",
			"proboscis monkey, Nasalis larvatus",
			"marmoset",
			"capuchin, ringtail, Cebus capucinus",
			"howler monkey, howler",
			"titi, titi monkey",
			"spider monkey, Ateles geoffroyi",
			"squirrel monkey, Saimiri sciureus",
			"Madagascar cat, ring-tailed lemur, Lemur catta",
			"indri, indris, Indri indri, Indri brevicaudatus",
			"Indian elephant, Elephas maximus",
			"African elephant, Loxodonta africana",
			"lesser panda, red panda, panda, bear cat, cat bear, Ailurus fulgens",
			"giant panda, panda, panda bear, coon bear, Ailuropoda melanoleuca",
			"barracouta, snoek",
			"eel",
			"coho, cohoe, coho salmon, blue jack, silver salmon, Oncorhynchus kisutch",
			"rock beauty, Holocanthus tricolor",
			"anemone fish",
			"sturgeon",
			"gar, garfish, garpike, billfish, Lepisosteus osseus",
			"lionfish",
			"puffer, pufferfish, blowfish, globefish",
			"abacus",
			"abaya",
			"academic gown, academic robe, judge's robe",
			"accordion, piano accordion, squeeze box",
			"acoustic guitar",
			"aircraft carrier, carrier, flattop, attack aircraft carrier",
			"airliner",
			"airship, dirigible",
			"altar",
			"ambulance",
			"amphibian, amphibious vehicle",
			"analog clock",
			"apiary, bee house",
			"apron",
			"ashcan, trash can, garbage can, wastebin, ash bin, ash-bin, ashbin, dustbin, trash barrel, trash bin",
			"assault rifle, assault gun",
			"backpack, back pack, knapsack, packsack, rucksack, haversack",
			"bakery, bakeshop, bakehouse",
			"balance beam, beam",
			"balloon",
			"ballpoint, ballpoint pen, ballpen, Biro",
			"Band Aid",
			"banjo",
			"bannister, banister, balustrade, balusters, handrail",
			"barbell",
			"barber chair",
			"barbershop",
			"barn",
			"barometer",
			"barrel, cask",
			"barrow, garden cart, lawn cart, wheelbarrow",
			"baseball",
			"basketball",
			"bassinet",
			"bassoon",
			"bathing cap, swimming cap",
			"bath towel",
			"bathtub, bathing tub, bath, tub",
			"beach wagon, station wagon, wagon, estate car, beach waggon, station waggon, waggon",
			"beacon, lighthouse, beacon light, pharos",
			"beaker",
			"bearskin, busby, shako",
			"beer bottle",
			"beer glass",
			"bell cote, bell cot",
			"bib",
			"bicycle-built-for-two, tandem bicycle, tandem",
			"bikini, two-piece",
			"binder, ring-binder",
			"binoculars, field glasses, opera glasses",
			"birdhouse",
			"boathouse",
			"bobsled, bobsleigh, bob",
			"bolo tie, bolo, bola tie, bola",
			"bonnet, poke bonnet",
			"bookcase",
			"bookshop, bookstore, bookstall",
			"bottlecap",
			"bow",
			"bow tie, bow-tie, bowtie",
			"brass, memorial tablet, plaque",
			"brassiere, bra, bandeau",
			"breakwater, groin, groyne, mole, bulwark, seawall, jetty",
			"breastplate, aegis, egis",
			"broom",
			"bucket, pail",
			"buckle",
			"bulletproof vest",
			"bullet train, bullet",
			"butcher shop, meat market",
			"cab, hack, taxi, taxicab",
			"caldron, cauldron",
			"candle, taper, wax light",
			"cannon",
			"canoe",
			"can opener, tin opener",
			"cardigan",
			"car mirror",
			"carousel, carrousel, merry-go-round, roundabout, whirligig",
			"carpenter's kit, tool kit",
			"carton",
			"car wheel",
			"cash machine, cash dispenser, automated teller machine, automatic teller machine, automated teller, automatic teller, ATM",
			"cassette",
			"cassette player",
			"castle",
			"catamaran",
			"CD player",
			"cello, violoncello",
			"cellular telephone, cellular phone, cellphone, cell, mobile phone",
			"chain",
			"chainlink fence",
			"chain mail, ring mail, mail, chain armor, chain armour, ring armor, ring armour",
			"chain saw, chainsaw",
			"chest",
			"chiffonier, commode",
			"chime, bell, gong",
			"china cabinet, china closet",
			"Christmas stocking",
			"church, church building",
			"cinema, movie theater, movie theatre, movie house, picture palace",
			"cleaver, meat cleaver, chopper",
			"cliff dwelling",
			"cloak",
			"clog, geta, patten, sabot",
			"cocktail shaker",
			"coffee mug",
			"coffeepot",
			"coil, spiral, volute, whorl, helix",
			"combination lock",
			"computer keyboard, keypad",
			"confectionery, confectionary, candy store",
			"container ship, containership, container vessel",
			"convertible",
			"corkscrew, bottle screw",
			"cornet, horn, trumpet, trump",
			"cowboy boot",
			"cowboy hat, ten-gallon hat",
			"cradle",
			"crane",
			"crash helmet",
			"crate",
			"crib, cot",
			"Crock Pot",
			"croquet ball",
			"crutch",
			"cuirass",
			"dam, dike, dyke",
			"desk",
			"desktop computer",
			"dial telephone, dial phone",
			"diaper, nappy, napkin",
			"digital clock",
			"digital watch",
			"dining table, board",
			"dishrag, dishcloth",
			"dishwasher, dish washer, dishwashing machine",
			"disk brake, disc brake",
			"dock, dockage, docking facility",
			"dogsled, dog sled, dog sleigh",
			"dome",
			"doormat, welcome mat",
			"drilling platform, offshore rig",
			"drum, membranophone, tympan",
			"drumstick",
			"dumbbell",
			"Dutch oven",
			"electric fan, blower",
			"electric guitar",
			"electric locomotive",
			"entertainment center",
			"envelope",
			"espresso maker",
			"face powder",
			"feather boa, boa",
			"file, file cabinet, filing cabinet",
			"fireboat",
			"fire engine, fire truck",
			"fire screen, fireguard",
			"flagpole, flagstaff",
			"flute, transverse flute",
			"folding chair",
			"football helmet",
			"forklift",
			"fountain",
			"fountain pen",
			"four-poster",
			"freight car",
			"French horn, horn",
			"frying pan, frypan, skillet",
			"fur coat",
			"garbage truck, dustcart",
			"gasmask, respirator, gas helmet",
			"gas pump, gasoline pump, petrol pump, island dispenser",
			"goblet",
			"go-kart",
			"golf ball",
			"golfcart, golf cart",
			"gondola",
			"gong, tam-tam",
			"gown",
			"grand piano, grand",
			"greenhouse, nursery, glasshouse",
			"grille, radiator grille",
			"grocery store, grocery, food market, market",
			"guillotine",
			"hair slide",
			"hair spray",
			"half track",
			"hammer",
			"hamper",
			"hand blower, blow dryer, blow drier, hair dryer, hair drier",
			"hand-held computer, hand-held microcomputer",
			"handkerchief, hankie, hanky, hankey",
			"hard disc, hard disk, fixed disk",
			"harmonica, mouth organ, harp, mouth harp",
			"harp",
			"harvester, reaper",
			"hatchet",
			"holster",
			"home theater, home theatre",
			"honeycomb",
			"hook, claw",
			"hoopskirt, crinoline",
			"horizontal bar, high bar",
			"horse cart, horse-cart",
			"hourglass",
			"iPod",
			"iron, smoothing iron",
			"jack-o'-lantern",
			"jean, blue jean, denim",
			"jeep, landrover",
			"jersey, T-shirt, tee shirt",
			"jigsaw puzzle",
			"jinrikisha, ricksha, rickshaw",
			"joystick",
			"kimono",
			"knee pad",
			"knot",
			"lab coat, laboratory coat",
			"ladle",
			"lampshade, lamp shade",
			"laptop, laptop computer",
			"lawn mower, mower",
			"lens cap, lens cover",
			"letter opener, paper knife, paperknife",
			"library",
			"lifeboat",
			"lighter, light, igniter, ignitor",
			"limousine, limo",
			"liner, ocean liner",
			"lipstick, lip rouge",
			"Loafer",
			"lotion",
			"loudspeaker, speaker, speaker unit, loudspeaker system, speaker system",
			"loupe, jeweler's loupe",
			"lumbermill, sawmill",
			"magnetic compass",
			"mailbag, postbag",
			"mailbox, letter box",
			"maillot",
			"maillot, tank suit",
			"manhole cover",
			"maraca",
			"marimba, xylophone",
			"mask",
			"matchstick",
			"maypole",
			"maze, labyrinth",
			"measuring cup",
			"medicine chest, medicine cabinet",
			"megalith, megalithic structure",
			"microphone, mike",
			"microwave, microwave oven",
			"military uniform",
			"milk can",
			"minibus",
			"miniskirt, mini",
			"minivan",
			"missile",
			"mitten",
			"mixing bowl",
			"mobile home, manufactured home",
			"Model T",
			"modem",
			"monastery",
			"monitor",
			"moped",
			"mortar",
			"mortarboard",
			"mosque",
			"mosquito net",
			"motor scooter, scooter",
			"mountain bike, all-terrain bike, off-roader",
			"mountain tent",
			"mouse, computer mouse",
			"mousetrap",
			"moving van",
			"muzzle",
			"nail",
			"neck brace",
			"necklace",
			"nipple",
			"notebook, notebook computer",
			"obelisk",
			"oboe, hautboy, hautbois",
			"ocarina, sweet potato",
			"odometer, hodometer, mileometer, milometer",
			"oil filter",
			"organ, pipe organ",
			"oscilloscope, scope, cathode-ray oscilloscope, CRO",
			"overskirt",
			"oxcart",
			"oxygen mask",
			"packet",
			"paddle, boat paddle",
			"paddlewheel, paddle wheel",
			"padlock",
			"paintbrush",
			"pajama, pyjama, pj's, jammies",
			"palace",
			"panpipe, pandean pipe, syrinx",
			"paper towel",
			"parachute, chute",
			"parallel bars, bars",
			"park bench",
			"parking meter",
			"passenger car, coach, carriage",
			"patio, terrace",
			"pay-phone, pay-station",
			"pedestal, plinth, footstall",
			"pencil box, pencil case",
			"pencil sharpener",
			"perfume, essence",
			"Petri dish",
			"photocopier",
			"pick, plectrum, plectron",
			"pickelhaube",
			"picket fence, paling",
			"pickup, pickup truck",
			"pier",
			"piggy bank, penny bank",
			"pill bottle",
			"pillow",
			"ping-pong ball",
			"pinwheel",
			"pirate, pirate ship",
			"pitcher, ewer",
			"plane, carpenter's plane, woodworking plane",
			"planetarium",
			"plastic bag",
			"plate rack",
			"plow, plough",
			"plunger, plumber's helper",
			"Polaroid camera, Polaroid Land camera",
			"pole",
			"police van, police wagon, paddy wagon, patrol wagon, wagon, black Maria",
			"poncho",
			"pool table, billiard table, snooker table",
			"pop bottle, soda bottle",
			"pot, flowerpot",
			"potter's wheel",
			"power drill",
			"prayer rug, prayer mat",
			"printer",
			"prison, prison house",
			"projectile, missile",
			"projector",
			"puck, hockey puck",
			"punching bag, punch bag, punching ball, punchball",
			"purse",
			"quill, quill pen",
			"quilt, comforter, comfort, puff",
			"racer, race car, racing car",
			"racket, racquet",
			"radiator",
			"radio, wireless",
			"radio telescope, radio reflector",
			"rain barrel",
			"recreational vehicle, RV, R.V.",
			"reel",
			"reflex camera",
			"refrigerator, icebox",
			"remote control, remote",
			"restaurant, eating house, eating place, eatery",
			"revolver, six-gun, six-shooter",
			"rifle",
			"rocking chair, rocker",
			"rotisserie",
			"rubber eraser, rubber, pencil eraser",
			"rugby ball",
			"rule, ruler",
			"running shoe",
			"safe",
			"safety pin",
			"saltshaker, salt shaker",
			"sandal",
			"sarong",
			"sax, saxophone",
			"scabbard",
			"scale, weighing machine",
			"school bus",
			"schooner",
			"scoreboard",
			"screen, CRT screen",
			"screw",
			"screwdriver",
			"seat belt, seatbelt",
			"sewing machine",
			"shield, buckler",
			"shoe shop, shoe-shop, shoe store",
			"shoji",
			"shopping basket",
			"shopping cart",
			"shovel",
			"shower cap",
			"shower curtain",
			"ski",
			"ski mask",
			"sleeping bag",
			"slide rule, slipstick",
			"sliding door",
			"slot, one-armed bandit",
			"snorkel",
			"snowmobile",
			"snowplow, snowplough",
			"soap dispenser",
			"soccer ball",
			"sock",
			"solar dish, solar collector, solar furnace",
			"sombrero",
			"soup bowl",
			"space bar",
			"space heater",
			"space shuttle",
			"spatula",
			"speedboat",
			"spider web, spider's web",
			"spindle",
			"sports car, sport car",
			"spotlight, spot",
			"stage",
			"steam locomotive",
			"steel arch bridge",
			"steel drum",
			"stethoscope",
			"stole",
			"stone wall",
			"stopwatch, stop watch",
			"stove",
			"strainer",
			"streetcar, tram, tramcar, trolley, trolley car",
			"stretcher",
			"studio couch, day bed",
			"stupa, tope",
			"submarine, pigboat, sub, U-boat",
			"suit, suit of clothes",
			"sundial",
			"sunglass",
			"sunglasses, dark glasses, shades",
			"sunscreen, sunblock, sun blocker",
			"suspension bridge",
			"swab, swob, mop",
			"sweatshirt",
			"swimming trunks, bathing trunks",
			"swing",
			"switch, electric switch, electrical switch",
			"syringe",
			"table lamp",
			"tank, army tank, armored combat vehicle, armoured combat vehicle",
			"tape player",
			"teapot",
			"teddy, teddy bear",
			"television, television system",
			"tennis ball",
			"thatch, thatched roof",
			"theater curtain, theatre curtain",
			"thimble",
			"thresher, thrasher, threshing machine",
			"throne",
			"tile roof",
			"toaster",
			"tobacco shop, tobacconist shop, tobacconist",
			"toilet seat",
			"torch",
			"totem pole",
			"tow truck, tow car, wrecker",
			"toyshop",
			"tractor",
			"trailer truck, tractor trailer, trucking rig, rig, articulated lorry, semi",
			"tray",
			"trench coat",
			"tricycle, trike, velocipede",
			"trimaran",
			"tripod",
			"triumphal arch",
			"trolleybus, trolley coach, trackless trolley",
			"trombone",
			"tub, vat",
			"turnstile",
			"typewriter keyboard",
			"umbrella",
			"unicycle, monocycle",
			"upright, upright piano",
			"vacuum, vacuum cleaner",
			"vase",
			"vault",
			"velvet",
			"vending machine",
			"vestment",
			"viaduct",
			"violin, fiddle",
			"volleyball",
			"waffle iron",
			"wall clock",
			"wallet, billfold, notecase, pocketbook",
			"wardrobe, closet, press",
			"warplane, military plane",
			"washbasin, handbasin, washbowl, lavabo, wash-hand basin",
			"washer, automatic washer, washing machine",
			"water bottle",
			"water jug",
			"water tower",
			"whiskey jug",
			"whistle",
			"wig",
			"window screen",
			"window shade",
			"Windsor tie",
			"wine bottle",
			"wing",
			"wok",
			"wooden spoon",
			"wool, woolen, woollen",
			"worm fence, snake fence, snake-rail fence, Virginia fence",
			"wreck",
			"yawl",
			"yurt",
			"web site, website, internet site, site",
			"comic book",
			"crossword puzzle, crossword",
			"street sign",
			"traffic light, traffic signal, stoplight",
			"book jacket, dust cover, dust jacket, dust wrapper",
			"menu",
			"plate",
			"guacamole",
			"consomme",
			"hot pot, hotpot",
			"trifle",
			"ice cream, icecream",
			"ice lolly, lolly, lollipop, popsicle",
			"French loaf",
			"bagel, beigel",
			"pretzel",
			"cheeseburger",
			"hotdog, hot dog, red hot",
			"mashed potato",
			"head cabbage",
			"broccoli",
			"cauliflower",
			"zucchini, courgette",
			"spaghetti squash",
			"acorn squash",
			"butternut squash",
			"cucumber, cuke",
			"artichoke, globe artichoke",
			"bell pepper",
			"cardoon",
			"mushroom",
			"Granny Smith",
			"strawberry",
			"orange",
			"lemon",
			"fig",
			"pineapple, ananas",
			"banana",
			"jackfruit, jak, jack",
			"custard apple",
			"pomegranate",
			"hay",
			"carbonara",
			"chocolate sauce, chocolate syrup",
			"dough",
			"meat loaf, meatloaf",
			"pizza, pizza pie",
			"potpie",
			"burrito",
			"red wine",
			"espresso",
			"cup",
			"eggnog",
			"alp",
			"bubble",
			"cliff, drop, drop-off",
			"coral reef",
			"geyser",
			"lakeside, lakeshore",
			"promontory, headland, head, foreland",
			"sandbar, sand bar",
			"seashore, coast, seacoast, sea-coast",
			"valley, vale",
			"volcano",
			"ballplayer, baseball player",
			"groom, bridegroom",
			"scuba diver",
			"rapeseed",
			"daisy",
			"yellow lady's slipper, yellow lady-slipper, Cypripedium calceolus, Cypripedium parviflorum",
			"corn",
			"acorn",
			"hip, rose hip, rosehip",
			"buckeye, horse chestnut, conker",
			"coral fungus",
			"agaric",
			"gyromitra",
			"stinkhorn, carrion fungus",
			"earthstar",
			"hen-of-the-woods, hen of the woods, Polyporus frondosus, Grifola frondosa",
			"bolete",
			"ear, spike, capitulum",
			"toilet tissue, toilet paper, bathroom tissue"
		]
