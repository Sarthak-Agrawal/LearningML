import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
import numpy as np

from NeuralNetworks.createSentimentFeaturesets import createFeaturesetsAndLabels

trainX, trainY,testX, testY = createFeaturesetsAndLabels(
        'data/pos.txt', 'data/neg.txt'
    )

nNodesHl11 = 500
nNodesHl12 = 700
nNodesHl13 = 800

nClasses = 2
batchSize = 100

x = tf.placeholder('float')
y = tf.placeholder('float')

def neuralNetworkModel(data):
    hiddenLayer1 = {'weights':tf.Variable(tf.random_normal([len(trainX[0]), nNodesHl11])),
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
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=y, logits=prediction))
    optimizer = tf.train.AdamOptimizer(learning_rate=0.001).minimize(cost)
    hmEpochs = 10
    print(len(trainX))
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        for epoch in range(hmEpochs):
            epochLoss = 0
            i = 0
            while i < len(trainX):
                start = i
                end = i+ batchSize

                batchX = np.array(trainX[start:end])
                batchY = np.array(trainY[start:end])
                _, c = sess.run([optimizer, cost], feed_dict={x: batchX, y: batchY})
                epochLoss += c

                i += batchSize

            print('Epoch', epoch+1, 'completed out of', hmEpochs,
                  'loss:', epochLoss/int(len(trainX)/batchSize))

        correctPreds = tf.equal(tf.argmax(prediction, axis=1), tf.argmax(y, axis=1), name="CorrectPreds")
        accuracy = tf.reduce_mean(tf.cast(correctPreds, 'float'))
        print("Accuracy : ", accuracy.eval({x:testX, y:testY}))


trainNeuralNetwork(x)
