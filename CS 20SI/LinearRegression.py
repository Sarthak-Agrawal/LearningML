import matplotlib.pyplot as plt
import tensorflow as tf
import pandas as pd

dataFile = "fireAndTheft.xls"

df = pd.read_excel(dataFile)
data = df.as_matrix()
numberOfSamples = len(data)

X = tf.placeholder(tf.float32, name='X')
Y = tf.placeholder(tf.float32, name='Y')

w = tf.Variable(0.0, name='weights')
b = tf.Variable(0.0, name='bias')

yPredicted = X * w + b

loss = tf.square(Y - yPredicted, name='loss')

optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001).minimize(loss)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    writer = tf.summary.FileWriter('./graphs/LinearRegression', sess.graph)

    for epoch in range(100):
        totalLoss = 0
        for x, y in data:
            _, l = sess.run([optimizer, loss], feed_dict={X: x, Y: y})
            totalLoss += l
        print('Epoch {0}:loss: {1}'.format(epoch, totalLoss/numberOfSamples))

    w, b = sess.run([w, b])
    print('w: {0}, b: {1}'.format(w, b))
    # print(loss.eval(session=sess))
    writer.close()

X, Y = data.T[0], data.T[1]
plt.plot(X, Y, 'b.', label='Real Data')
plt.plot(X, X * w + b, 'r', label='Predicted data')
plt.legend()
plt.show()
