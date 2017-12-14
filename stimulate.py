# -*- coding: utf-8 -*-
"""
Created on Mon Nov 20 10:12:51 2017

@author: milittle
"""

import tensorflow as tf
import numpy as np
import readImg

def weights_variable(shape, name):
    initial = tf.truncated_normal(shape, stddev = 0.01)
    return tf.Variable(initial, name = name)

def biases_variable(shape, name):
    initial = tf.constant(0., shape = shape)
    return tf.Variable(initial, name = name)

def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides = [1, 1, 1, 1], padding = 'VALID')

def max_pool_2X2(x):
    return tf.nn.max_pool(x, ksize = [1, 2, 2, 1], strides = [1, 2, 2, 1], padding = 'SAME')

def compute_accuracy(v_xs, v1_xs, v_ys):
    global prediction
    y_pre = sess.run(prediction, feed_dict={xs: v_xs, xs_1: v1_xs})
    y_pre_r = tf.round(y_pre)
    correct_prediction = tf.equal(y_pre_r, v_ys)
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    result = sess.run(accuracy, feed_dict={xs: v_xs, xs_1: v1_xs, ys: v_ys})
    return result
    


xs = tf.placeholder(tf.float32, [None, 14700])
xs_1 = tf.placeholder(tf.float32, [None, 14700])
ys = tf.placeholder(tf.float32, [None, 2])
keep_prob = tf.placeholder(tf.float32)

x_image = tf.reshape(xs, shape = [-1, 210, 70, 1])
x_1_image = tf.reshape(xs_1, shape = [-1, 210, 70, 1])

w_1_p = weights_variable([7, 7, 1, 16], name = 'w_1_p')
b_1_p = biases_variable([16], name = 'b_1_p')
h_conv1 = tf.nn.relu(conv2d(x_image, w_1_p) + b_1_p)

h_conv1_nor = tf.nn.local_response_normalization(h_conv1, depth_radius = 5, bias = 2, alpha = 1e-4, beta = 0.75)

h_pool1 = max_pool_2X2(h_conv1_nor)

w_1_g = weights_variable([7, 7, 1, 16], name = 'w_1_g')
b_1_g = biases_variable([16], name = 'b_1_g')
h_conv_1_1 = tf.nn.relu(conv2d(x_1_image, w_1_g) + b_1_g)

h_conv_1_1_nor = tf.nn.local_response_normalization(h_conv_1_1, depth_radius = 5, bias = 2, alpha = 1e-4, beta = 0.75)

h_pool_1_1 = max_pool_2X2(h_conv_1_1_nor)

h_conbination = np.add(h_pool1, h_pool_1_1) #融合两个feature map

w_2 = weights_variable([7, 7, 16, 64], name = 'w_2');
b_2 = biases_variable([64], name = 'b_2')

h_conv2 = tf.nn.relu(conv2d(h_conbination, w_2) + b_2)

h_conv2_nor = tf.nn.local_response_normalization(h_conv2, depth_radius = 5, bias = 2, alpha = 1e-4, beta = 0.75)

h_pool2 = max_pool_2X2(h_conv2_nor)

w_3 = weights_variable([7, 7, 64, 256], 'w_3');
b_3 = biases_variable([256], name = 'b_3')

h_conv3 = tf.nn.relu(conv2d(h_pool2, w_3) + b_3)

h_pool2_flat = tf.reshape(h_conv3, [-1, 7 * 42 * 256])

# w_fc1 = weights_variable([7 * 17 * 256, 1024], 'w_fc')
# b_fc1 = biases_variable([1024])

# h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, w_fc1) + b_fc1)
# h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

w_softmax = weights_variable([7 * 42 * 256, 2], 'w_softmax')
b_softmax = biases_variable([2], name = 'b_softmax')

prediction = tf.nn.softmax(tf.matmul(h_pool2_flat, w_softmax) + b_softmax, name = 'predic')
cross_entropy = tf.reduce_mean(-tf.reduce_sum(ys * tf.log(prediction), reduction_indices = [1]))
train_step = tf.train.AdamOptimizer(1e-5).minimize(cross_entropy)


init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

#读出在NM-01情况下的所有角度的GEI


data_xs = readImg.readNM_()[1:506:11, :, :].reshape(46, 14700)
data_xs_1 = readImg.readNM_()[2:253:11, :, :].reshape(23, 14700)
data_xs_2 = readImg.readNM_()[264:506:11, :, :].reshape(22, 14700)
data_xs_3 = readImg.readNM_()[0, :, :].reshape(1, 14700)
data_xs_1_list = list(data_xs_1)
data_xs_2_list = list(data_xs_2)
data_xs_3_list = list(data_xs_3)
data_xs_1_list.extend(data_xs_2_list)
data_xs_1_list.extend(data_xs_3_list)



for i in range(10000):
    data_list_1, data_list_2, data_ys = readImg.nextBatch()
    sess.run(train_step, feed_dict = {xs: data_list_1, xs_1: data_list_2, ys: data_ys})
    print(sess.run(prediction, feed_dict = {xs: data_list_1, xs_1: data_list_2, ys: data_ys}))
    print(sess.run(cross_entropy, feed_dict = {xs: data_list_1, xs_1: data_list_2, ys: data_ys}))
    if i % 100 == 0:
        test_xs, test_xs1, test_ys = readImg.nextTestBatch()
        print(compute_accuracy(test_xs, test_xs1, test_ys))



