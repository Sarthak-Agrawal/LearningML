import tensorflow as tf

a = tf.constant(2, name='a')
b = tf.constant(3, name='b')
x = tf.add(a, b)
c = tf.constant([2, 3], name='c')
d = tf.constant([5, 6], name='d')
y = tf.add_n([c, d, d])
p = tf.placeholder(tf.float32, shape=2)
q = tf.constant([1, 1], tf.float32, name='q')
z = p + q
v = tf.Variable([2.0, 3.0], name='v')
# p+q will give an error as p is a placeholder and has not been initialised

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    print(sess.run(v+q))
    writer = tf.summary.FileWriter('./graphs', sess.graph)
    print(sess.run(x))
    print(sess.run(y))
    print(sess.run(z, {p: [1, 2]}))
# tensorboard --logdirs='./graphs' used to start server in terminal
# close the writer when you’re done using it
writer.close()
