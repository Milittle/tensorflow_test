# -*- coding: utf-8 -*-
"""
Created on Wed Nov 15 10:56:52 2017

@author: milittle
"""

import tensorflow as tf

mat1 = tf.constant([[2, 2]]) #行向量
mat2 = tf.constant([[3], [3]]) #列向量
product = tf.matmul(mat1, mat2) #行列向量相乘

with tf.Session() as sess:
    result = sess.run(product)
    print(result)