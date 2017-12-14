# -*- coding: utf-8 -*-
"""
Created on Wed Nov 22 16:07:06 2017

@author: milittle
"""

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

tf.set_random_seed(1)

mnist = input_data.read_data_sets('MNIST_data', one_hot = True)

#hyper parameter

lr = 0.001
traing_iters = 100000
batch_size = 128
n_inputs = 28
n_steps = 28
n_hidden_units = 128
n_classes = 10

xs = tf.placeholder(tf.float32, [None, n_steps, n_inputs])
ys = tf.placeholder(tf.float32, [None, n_classes])

weights = {
        'in':tf.Variable(tf.random_normal([n_inputs, n_hidden_units])), 
        'out':tf.Variable(tf.random_normal([n_hidden_units, n_classes]))
        }

biases = {
        'in':tf.Variable([n_hidden_units, ]), 
        'out':tf.Variable([n_classes, ])
        }

