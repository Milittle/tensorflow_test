# -*- coding: utf-8 -*-
"""
Created on Wed Nov 15 13:16:10 2017

@author: milittle
"""

import tensorflow as tf

test1 = tf.placeholder(tf.float32)
test2 = tf.placeholder(tf.float32)

output = tf.multiply(test1, test2)

with tf.Session() as sess:
    print(sess.run(output, feed_dict = {test1:[2], test2:[9]}))