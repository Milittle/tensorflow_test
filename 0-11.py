# -*- coding: utf-8 -*-
"""
Created on Thu Nov 16 08:57:11 2017

@author: milittle
"""

import tensorflow as tf

from sklearn.datasets import load_digits
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import LabelBinarizer

keep_prob = tf.placeholder(tf.float32)



def add_layer(inputs, in_size, out_size, n_layer, activation_function = None):
    layer_name = 'layer_name %s' % n_layer
    with tf.name_scope('layer'):
        with tf.name_scope('Weights'):
            Weights = tf.Variable(tf.random_normal([in_size, out_size]), name = 'w')
            tf.summary.histogram(layer_name + '/Weights', Weights)
        with tf.name_scope('biases'):
            biases = tf.Variable(tf.zeros([1, out_size]) + 0.1, name = 'b')
            tf.summary.histogram(layer_name + '/biases', biases)
        with tf.name_scope('Wx_plus_b'):
            Wx_plus_b = tf.add(tf.matmul(inputs, Weights), biases)
            Wx_plus_b = tf.nn.dropout(Wx_plus_b, keep_prob)
        if activation_function is None:
            outputs = Wx_plus_b
        else:
            outputs = activation_function(Wx_plus_b, )
        tf.summary.histogram(layer_name + '/outputs', outputs)
        return outputs

digits = load_digits()
X = digits.data
y = digits.target
y = LabelBinarizer().fit_transform(y)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = .3)
xs = tf.placeholder(tf.float32, [None, 64])
ys = tf.placeholder(tf.float32, [None, 10])
l1 = add_layer(xs, 64, 50, n_layer = 'L1', activation_function = tf.nn.tanh)
prediction = add_layer(l1, 50, 10, n_layer = 'L2', activation_function = tf.nn.softmax)
with tf.name_scope('loss'):
    cross_entropy = tf.reduce_mean(-tf.reduce_sum(ys * tf.log(prediction), reduction_indices = [1]))
    tf.summary.scalar('loss', cross_entropy)

with tf.name_scope('train_step'):
    train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

init = tf.global_variables_initializer()
sess = tf.Session()
merged = tf.summary.merge_all()
writer = tf.summary.FileWriter("logs/", sess.graph)
sess.run(init)

for i in range (501):
    sess.run(train_step, feed_dict = {xs: X_train, ys: y_train, keep_prob: 0.5})
    if i % 50 == 0:
        print(sess.run(cross_entropy, feed_dict = {xs: X_test, ys: y_test, keep_prob: 0.5}))
        rs = sess.run(merged, feed_dict = {xs: X_train, ys: y_train, keep_prob: 0.5})
        writer.add_summary(rs, i)

