import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf

from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("data/", one_hot=True)

nNodesHl11 = 500
nNodesHl12 = 500
nNodesHl13 = 500

nClasses = 10
batchSize = 100

x = tf.placeholder('float', [None, 784])
y = tf.placeholder('float')

def neuralNetworkModel(data):
    hiddenLayer1 = {'weights':tf.Variable(tf.random_normal([784, nNodesHl11])),
                    'biases':tf.Variable(tf.random_normal([nNodesHl11]))}
    hiddenLayer2 = {'weights': tf.Variable(tf.random_normal([nNodesHl11, nNodesHl12])),
                    'biases': tf.Variable(tf.random_normal([nNodesHl12]))}
    hiddenLayer3 = {'weights': tf.Variable(tf.random_normal([nNodesHl12, nNodesHl13])),
                    'biases': tf.Variable(tf.random_normal([nNodesHl13]))}
    outputLayer = {'weights': tf.Variable(tf.random_normal([nNodesHl13, nClasses])),
                    'biases': tf.Variable(tf.random_normal([nClasses]))}
    l1 = tf.add(tf.matmul(data, hiddenLayer1['weights']), hiddenLayer1['biases'])
    l1 = tf.nn.relu(l1)

    l2 = tf.add(tf.matmul(l1, hiddenLayer2['weights']), hiddenLayer2['biases'])
    l2 = tf.nn.relu(l2)

    l3 = tf.add(tf.matmul(l2, hiddenLayer3['weights']), hiddenLayer3['biases'])
    l3 = tf.nn.relu(l3)

    output = tf.add(tf.matmul(l3, outputLayer['weights']), outputLayer['biases'])
    return output


def trainNeuralNetwork(x):
    prediction = neuralNetworkModel(x)
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=y))
    optimizer = tf.train.AdamOptimizer().minimize(cost)
    hmEpochs = 10
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for epoch in range(hmEpochs):
            epochLoss = 0
            for _ in range(int(mnist.train.num_examples/batchSize)):
                XBatch, YBatch = mnist.train.next_batch(batchSize)
                _, l = sess.run([optimizer, cost], feed_dict={x: XBatch, y:YBatch})
                epochLoss += l

            print('Epoch', epoch, 'completed out of', hmEpochs, 'loss:', epochLoss/int(mnist.train.num_examples/batchSize))

        correctPreds = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1), name="CorrectPreds")
        accuracy = tf.reduce_mean(tf.cast(correctPreds, 'float'))
        print("Accuracy : ", accuracy.eval({x:mnist.test.images, y:mnist.test.labels}))


trainNeuralNetwork(x)
