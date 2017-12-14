# -*- coding: utf-8 -*-
"""
Created on Wed Nov 15 10:56:52 2017

@author: milittle
"""

import tensorflow as tf
import numpy as np

#构造标准函数映射
x_data = np.random.rand(100).astype(np.float32)
y_data = x_data * 0.1 + 0.3

#建立预测回归函数
Weights = tf.Variable(tf.random_uniform([1], -1.0, 1.0))
biases = tf.Variable(tf.zeros([1]))
y = x_data * Weights + biases

#计算loss值
loss = tf.reduce_mean(tf.square(y - y_data))
optimizer = tf.train.GradientDescentOptimizer(0.5)
train = optimizer.minimize(loss)
init = tf.global_variables_initializer()

#构造session对象
sess = tf.Session()
sess.run(init)

#开始迭代类型
for i in range(201):
    sess.run(train)
    if i % 20 == 0:
        print(i, sess.run(Weights), sess.run(biases))
print('hello')
