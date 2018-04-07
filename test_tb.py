import tensorflow as tf
a = tf.constant(2)
b = tf.constant(3)
c = tf.Variable(1)

with tf.name_scope('C'):
    c = tf.add(a,b)
    tf.summary.scalar('C:/c', c)

sess = tf.Session()


init = tf.global_variables_initializer()

rs = tf.summary.merge_all()
FWriter = tf.summary.FileWriter('logs/', sess.graph)

sess.run(init)

for i in range(1000):
    print(sess.run(c))
    res = sess.run(rs)
    FWriter.add_summary(res, i)