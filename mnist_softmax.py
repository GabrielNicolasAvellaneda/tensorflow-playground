import mnist_data
import tensorflow as tf


mnist = mnist_data.read_data_sets('MNIST_data', one_hot=True)

sess = tf.InteractiveSession()




x = tf.placeholder("float", shape=[None, 784]) # input images, 784 == 28**2 flattened
y_ = tf.placeholder("float", shape=[None, 10]) # target output class


W = tf.Variable(tf.zeros([784,10]))
b = tf.Variable(tf.zeros([10]))


sess.run(tf.initialize_all_variables())


y = tf.nn.softmax(tf.matmul(x,W) + b)

cross_entropy = -tf.reduce_sum(y_*tf.log(y))

train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)

for i in range(1000):
  batch = mnist.train.next_batch(50)
  train_step.run(feed_dict={x: batch[0], y_: batch[1]})

  # Since evaluating the accuracy is expensive, only run it every 50 steps
  if i % 50 == 0:
    correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    print accuracy.eval(feed_dict={x: mnist.test.images, y_: mnist.test.labels})

'''
Sample output

0.2847
0.8575
0.8778
0.8739
0.8945
0.8994
0.8972
0.8783
0.9031
0.9017
0.9015
0.9026
0.9109
0.906
0.9007
0.9038
0.8901
0.9053
0.9061
0.9148
'''
