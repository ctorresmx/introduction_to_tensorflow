import tensorflow as tf

a = tf.constant(5)
b = tf.constant(3)

c = a + b

with tf.Session() as sess:
    print(sess.run(c))
