# -*- coding: utf-8 -*-
"""
Created on Thu Nov 16 18:59:37 2017

@author: milittle
"""
import tensorflow as tf
import numpy as np

w = tf.Variable(np.arange(800).reshape([5, 5, 1, 32]), dtype = tf.float32, name = 'w_conv1')

saver = tf.train.Saver()

sess = tf.Session()



saver.restore(sess, 'my_net/save_net.ckpt')

b = sess.graph.get_tensor_by_name('w_fc1')

print('weight:', sess.run('w_conv1:0'))