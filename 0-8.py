# -*- coding: utf-8 -*-
"""
Created on Wed Nov 15 17:26:27 2017

@author: milittle
"""

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plot

xs = tf.placeholder(tf.float32, [None, 1], name = 'x_in')
ys = tf.placeholder(tf.float32, [None, 1], name = 'y_in')

with tf.name_scope('inputs'):
    xs = tf.placeholder(tf.float32, [None, 1])
    ys = tf.placeholder(tf.float32, [None, 1])

def add_layer(inputs, in_size, out_size, activation_function=None):
    '''
    inputs data
    in_size： input size
    out_size output size
    activation_function: 激活函数
    '''
    with tf.name_scope('layer'):
        with tf.name_scope('Weights'):
            Weights = tf.Variable(tf.random_normal([in_size, out_size]))
    with tf.name_scope('biases'):
        biases = tf.Variable(tf.zeros([1, out_size]) + 0.1)
    with tf.name_scope('Wx_plus_b'):
        Wx_plus_b = tf.add(tf.matmul(inputs, Weights), biases)
    if activation_function is None:
        outputs = Wx_plus_b
    else:
        outputs = activation_function(Wx_plus_b, )
    return outputs

x_data = np.linspace(-1, 1, 300, dtype = np.float32)[:, np.newaxis]
noise = np.random.normal(0, 0.05, x_data.shape).astype(np.float32)
y_data = np.square(x_data) - 0.5 + noise

#全局显示
fig = plot.figure()
ax = fig.add_subplot(1, 1, 1)
ax.scatter(x_data, y_data)
plot.ion()
plot.show()

xs = tf.placeholder(tf.float32, [None, 1])
ys = tf.placeholder(tf.float32, [None, 1])

l1 = add_layer(xs, 1, 10, activation_function = tf.nn.relu)

prediction = add_layer(l1, 10, 1, activation_function = None)

with tf.name_scope('loss'):
    loss = tf.reduce_mean(tf.reduce_sum(tf.square(ys - prediction),
                     reduction_indices = [1]))

with tf.name_scope('train_stpp'):
    train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)

init = tf.global_variables_initializer()
lines = []
sess = tf.Session()
writer = tf.summary.FileWriter("logs/", sess.graph)
sess.run(init)
for i in range(1000):
    sess.run(train_step, feed_dict = {xs: x_data, ys: y_data})
    if i % 50 ==0:
        print(i)
        try:
            ax.lines.remove(lines[0])
        except Exception:
            pass
        prediction_values = sess.run(prediction, feed_dict = {xs: x_data})
        lines = ax.plot(x_data, prediction_values, 'r-', lw = 5)
        plot.pause(0.1)
