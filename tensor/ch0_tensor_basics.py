#!/usr/bin/python3 -i
'''
A demo of very simple tensorflow usage.
See extensive documentation at
https://www.tensorflow.org/versions/master/get_started/basic_usage.html
'''
import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt

matrix1 = tf.constant([[3.,3.], [4., 4.], [5., 5.]])
matrix2 = tf.constant([[2., 2.]])
matrix3 = matrix1 * matrix2 # not equivalent as tf.matmul(matrix1, matrix2)

# open and close session in the with block
with tf.Session() as sess:
	result1 = matrix1.eval()
	result2 = matrix2.eval()
	result3 = matrix3.eval()
	result4 = sess.run(matrix3)

# lauch interactive session
sess = tf.InteractiveSession()
vec1 = tf.Variable([1.,2.])
vec2 = tf.constant([3.,3.])
vec3 = vec1 - vec2 # equivalent as tf.sub(vec1, vec2)
# init and eval
vec1.initializer.run() # or tf.initialize_all_variables().run()
# after initialization, vec1.eval() can be called
result5 = vec3.eval()
result6 = vec3.eval({vec1:np.array([5.,5.])})
# close the interactive session
sess.close()

# use a session
num = 5
sess = tf.Session(config = tf.ConfigProto(log_device_placement=False))
state = tf.Variable(1)
tf.initialize_all_variables().run(session=sess)
for idx in range(1,num+1):
	mulval = tf.constant(idx)
	update = tf.assign(state, state*mulval) # equivalent as tf.mul(state, mulval)
	sess.run(update) # equivalent as update.eval(session=sess)
result7 = state.eval(session=sess)
sess.close()

# placeholder and feed_dict
x = tf.placeholder(tf.float32, [2,2])
y = tf.placeholder(tf.float32, [2,1])
x_mul_y = tf.matmul(x, y)
with tf.Session() as sess:
	x_case = np.array([[2.,2.],[2.,2.]])
	y_case = np.array([[3.],[3.]])
	my_feed = {x: x_case, y: y_case}
	result8 = x_mul_y.eval(feed_dict=my_feed)
	result9 = sess.run(x_mul_y, feed_dict=my_feed)
