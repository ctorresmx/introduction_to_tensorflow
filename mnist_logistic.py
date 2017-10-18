#!/usr/bin/env python

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

logs_path = '/tmp/tensorboard/'
model_directory = 'saved_model/'
epochs = 100
batch_size = 100

# Load the MNIST data set
mnist_data = input_data.read_data_sets("MNIST_data/", one_hot=True)

# The basic neural network graph
x = tf.placeholder(tf.float32, shape=[None, 784], name='input')
W = tf.Variable(tf.random_normal([784, 10]), name='weights')
b = tf.Variable(tf.zeros([10]), name='biases')
y = tf.nn.softmax(tf.matmul(x, W) + b, name='prediction')

# The placeholder for the correct result
real_y = tf.placeholder(tf.float32, [None, 10], name='expected_prediction')

# Cost function
cost = tf.reduce_mean(-tf.reduce_sum(
    real_y * tf.log(y), axis=[1]), name='cost'
)

# Optimization
optimization = tf.train.GradientDescentOptimizer(0.5).minimize(cost)

# Initialization
init = tf.global_variables_initializer()


tf.summary.scalar('cost', cost)
summarize = tf.summary.merge_all()

iteration = 0

with tf.Session() as session:
    session.run(init)

    writer = tf.summary.FileWriter(logs_path, graph=tf.get_default_graph())

    for epoch in range(epochs):
        total_batch = int(mnist_data.train.num_examples / batch_size)
        for i in range(total_batch):
            batch_x, batch_y = mnist_data.train.next_batch(100)

            _, cost_mini_batch, summary = session.run(
                [optimization, cost, summarize], feed_dict={
                    x: batch_x, real_y: batch_y
                }
            )

            writer.add_summary(summary, epoch * total_batch + i)

            if iteration % 1000 == 0:
                print('Cost at mini-batch {}: {:.2f}'.format(
                    iteration, cost_mini_batch)
                )

            iteration += 1

    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(real_y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    network_accuracy = session.run(
        accuracy,
        feed_dict={x: mnist_data.test.images, real_y: mnist_data.test.labels}
    )

    print('Accuracy is {:.2f}%'.format(network_accuracy * 100))

    print('Saving model...')
    saver = tf.train.Saver()
    saver.save(session, '{}mnist_trained'.format(model_directory))
