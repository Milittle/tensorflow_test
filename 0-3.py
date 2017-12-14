# -*- coding: utf-8 -*-
"""
Created on Wed Nov 15 11:18:24 2017

@author: milittle
"""

import tensorflow as tf

state = tf.Variable(0, name = 'counter')
con = tf.constant(1)

new_val = tf.add(state, con)

update = tf.assign(state, new_val)

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    for _ in range(3):
        print(sess.run(update))
        print(sess.run(state))