# -*- coding: utf-8 -*-
"""
Created on Fri Mar  9 16:39:25 2018

@author: milittle
"""

import tensorflow as tf

x1 = tf.constant(1.0, shape=[1, 3, 3, 1])

x2 = tf.constant(1.0, shape=[1, 6, 6, 3])

x3 = tf.constant(1.0, shape=[1, 5, 5, 3])

kernel = tf.constant(1.0, shape=[3, 3, 3, 1])

y1 = tf.nn.conv2d_transpose(x1, kernel, output_shape=[1, 6, 6, 3],
                            strides=[1, 2, 2, 1], padding="SAME")

y2 = tf.nn.conv2d(x3, kernel, strides=[1, 2, 2, 1], padding="SAME")

y3 = tf.nn.conv2d_transpose(y2, kernel, output_shape=[1, 5, 5, 3],
                            strides=[1, 2, 2, 1], padding="SAME")

y4 = tf.nn.conv2d(x2, kernel, strides=[1, 2, 2, 1], padding="SAME")

''''' 
Wrong!!This is impossible
y5 = tf.nn.conv2d_transpose(x1,kernel,output_shape=[1,10,10,3],strides=[1,2,2,1],padding="SAME") 
'''
sess = tf.Session()
tf.global_variables_initializer().run(session=sess)
x1_p, x2_p, x3_p, kernel_p = sess.run([x1, x2, x3, kernel])
x1_decov, x3_cov, y2_decov, x2_cov = sess.run([y1, y2, y3, y4])
print(x3_cov.shape)
print(x3_cov)
print(y2_decov)
print(y2_decov.shape)