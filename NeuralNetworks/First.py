import tensorflow as tf

# creates nodes in a graph
# "construction phase"
x1 = tf.constant(5)
x2 = tf.constant(6)

result = tf.multiply(x1, x2)
# This result is still an abstract tensor
print(result)
# Each operation done is just a node created in our computation graph
# To actually see the result, we need to run the session.

# defines our session and launches graph
session = tf.Session()
# runs result
output = session.run(result)
print(output)
session.close()
