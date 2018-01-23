import time
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

MNIST = input_data.read_data_sets('data/mnist', one_hot=True)

learningRate = 0.01
batchSize = 128
numberOfEpochs = 10

X = tf.placeholder(tf.float32, [batchSize, 784], name="image")
Y = tf.placeholder(tf.float32, [batchSize, 10], name="label")

w = tf.Variable(tf.random_normal(shape=[784, 10], mean=0.0, stddev=0.01), name="weights")
b = tf.Variable(tf.zeros([1, 10]), name="bias")

''' 
logits = X * w + b
Y_predicted = softmax(logits)
loss = cross_entropy(Y, Y_predicted)
'''
logits = tf.matmul(X, w) + b
entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=Y)
loss = tf.reduce_mean(entropy)

optimizer = tf.train.GradientDescentOptimizer(learning_rate=learningRate).minimize(loss)

with tf.Session() as sess:
    startTime = time.time()

    sess.run(tf.global_variables_initializer())
    numberOfBatches = int(MNIST.train.num_examples / batchSize)

    writer = tf.summary.FileWriter('./graphs/LogisticRegression', sess.graph)

    for i in range(numberOfEpochs):
        totalLoss = 0

        for _ in range(numberOfBatches):
            XBatch, YBatch = MNIST.train.next_batch(batchSize)
            _,l = sess.run([optimizer, loss], feed_dict={X:XBatch, Y:YBatch})
            totalLoss += l
        print('Average loss epoch {0}: {1}'.format(i, totalLoss / numberOfBatches))

    print("Total time : {0}".format(time.time() - startTime))

    # test
    numberOfBatches = int(MNIST.test.num_examples/batchSize)
    preds = tf.nn.softmax(logits, name='testPreds')
    correctPreds = tf.equal(tf.argmax(preds, 1), tf.argmax(Y, 1), name="CorrectPreds")
    accuracy = tf.reduce_sum(tf.cast(correctPreds, tf.float32), name='accuracy')

    totalCorrectPreds=0
    for i in range(numberOfBatches):
        XBatch, YBatch = MNIST.test.next_batch(batchSize)
        accuracyPerBatch = sess.run([accuracy], feed_dict={X:XBatch, Y:YBatch})
        totalCorrectPreds += accuracyPerBatch[0]

    print("Accuracy : {0}".format(totalCorrectPreds/MNIST.test.num_examples))
    writer.close()
