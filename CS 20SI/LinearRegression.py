import matplotlib.pyplot as plt
import tensorflow as tf
import pandas as pd
import numpy as np

dataFile = "data/fireAndTheft.xls"

df = pd.read_excel(dataFile)
data = df.as_matrix()
numberOfSamples = len(data)

X = tf.placeholder(tf.float32, name='X')
Y = tf.placeholder(tf.float32, name='Y')

XInput = np.linspace(-1, 1, 100)
YInput = XInput*3 + np.random.randn(XInput.shape[0]) * 0.5

w = tf.Variable(0.0, name='weights')
# w2 = tf.Variable(0.0, name='weight2')
b = tf.Variable(0.0, name='bias')

yPredicted = X * w + b

loss = tf.square(Y - yPredicted, name='loss')

def huber_loss(labels, predictions, delta=1.0):
    residual = tf.abs(predictions - labels)
    condition = tf.less(residual, delta)
    small_res = 0.5 * tf.square(residual)
    large_res = delta * residual - 0.5 * tf.square(delta)
    return tf.where(condition, small_res, large_res)

# loss1 = tf.losses.huber_loss(Y, yPredicted)
loss1 = huber_loss(Y, yPredicted)

optimizer = tf.train.AdamOptimizer(learning_rate=0.001).minimize(loss)
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    writer = tf.summary.FileWriter('./graphs/LinearRegression', sess.graph)

    for epoch in range(100):
        totalLoss = 0
        for x, y in zip(XInput, YInput):
            _, l = sess.run([optimizer, loss1], feed_dict={X: x, Y: y})
            totalLoss += l
        print('Epoch {0}:loss: {1}'.format(epoch, totalLoss/numberOfSamples))

    w, b = sess.run([w, b])
    print('w1: {0}, b: {1}'.format(w, b))
    # print(loss.eval(session=sess))
    writer.close()

X, Y = XInput, YInput
plt.plot(X, Y, 'b.', label='Real Data')
plt.plot(X, X * w + b, 'r', label='Predicted data')
plt.title("AdamOptimizer")
plt.legend()
plt.show()
