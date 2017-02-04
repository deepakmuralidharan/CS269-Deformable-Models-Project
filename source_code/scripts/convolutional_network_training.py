'''
Training a Convolutional Network for Deep Active Contour
'''
from __future__ import print_function

__author__ = "deepak_muralidharan"
__email__ = "deepakmuralidharan2308@gmail.com"


import tensorflow as tf
import DataManager
import numpy as np
import sys, os
import time
from random import shuffle
# Parameters
learning_rate = 0.001
training_iters = 200000
batch_size = 16
display_step = 1
max_epochs = 100
path = "/Users/deepakmuralidharan/Documents/data/"

# Network Parameters
n_input = 4096 # image patch input (64x64)
n_classes = 2 # size of vector (2,1)
dropout = 0.9 # Dropout, probability to keep units

# tf Graph input
x = tf.placeholder(tf.float32, [None, 64, 64, 3])
y = tf.placeholder(tf.float32, [None, 2])
keep_prob = tf.placeholder(tf.float32) #dropout (keep probability)


# Create some wrappers for simplicity

def conv2d(x, W, b, strides=1):
    # Conv2D wrapper, with bias and relu activation
    x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding='SAME')
    x = tf.nn.bias_add(x, b)
    return tf.nn.relu(x)


def maxpool2d(x, k=2):
    # MaxPool2D wrapper
    return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1],
                          padding='SAME')


# Create model
def conv_net(x, weights, biases, dropout):
    # Reshape input picture
    x = tf.reshape(x, shape=[-1, 64, 64, 3])

    # Convolution Layer
    conv1 = conv2d(x, weights['wc1'], biases['bc1'])
    # Max Pooling (down-sampling)
    conv1 = maxpool2d(conv1, k=2)

    # Convolution Layer
    conv2 = conv2d(conv1, weights['wc2'], biases['bc2'])
    # Max Pooling (down-sampling)
    conv2 = maxpool2d(conv2, k=2)

    # Convolution Layer
    conv3 = conv2d(conv2, weights['wc3'], biases['bc3'])
    # Max Pooling (down-sampling)
    conv3 = maxpool2d(conv3, k=2)

    # Convolution Layer
    conv4 = conv2d(conv3, weights['wc4'], biases['bc4'])
    # Max Pooling (down-sampling)
    conv4 = maxpool2d(conv4, k=2)


    # Fully connected layer
    # Reshape conv2 output to fit fully connected layer input
    fc1 = tf.reshape(conv4, [-1, weights['wd1'].get_shape().as_list()[0]])
    fc1 = tf.add(tf.matmul(fc1, weights['wd1']), biases['bd1'])
    fc1 = tf.nn.relu(fc1)
    # Apply Dropout
    fc1 = tf.nn.dropout(fc1, dropout)

    # Output, class prediction
    out = tf.add(tf.matmul(fc1, weights['out']), biases['out'])
    return out

# Store layers weight & bias
weights = {
    # 5x5 conv, 1 input, 32 outputs
    'wc1': tf.Variable(tf.random_normal([3, 3, 3, 32],stddev = 0.1)),
    # 5x5 conv, 32 inputs, 64 outputs
    'wc2': tf.Variable(tf.random_normal([3, 3, 32, 64],stddev = 0.1)),
    # fully connected, 7*7*64 inputs, 1024 outputs
    'wc3': tf.Variable(tf.random_normal([3, 3, 64, 128],stddev = 0.1)),
    'wc4': tf.Variable(tf.random_normal([3, 3, 128, 256],stddev = 0.1)),
    'wd1': tf.Variable(tf.random_normal([4*4*256, 2048],stddev = 0.1)),
    # 1024 inputs, 10 outputs (class prediction)
    'out': tf.Variable(tf.random_normal([2048, n_classes],stddev = 0.1))
}

biases = {
    'bc1': tf.Variable(tf.random_normal([32],stddev = 0.1)),
    'bc2': tf.Variable(tf.random_normal([64],stddev = 0.1)),
    'bc3': tf.Variable(tf.random_normal([128],stddev = 0.1)),
    'bc4': tf.Variable(tf.random_normal([256],stddev = 0.1)),
    'bd1': tf.Variable(tf.random_normal([2048],stddev = 0.1)),
    'out': tf.Variable(tf.random_normal([n_classes],stddev = 0.1))
}

# Construct model
pred = conv_net(x, weights, biases, keep_prob)

# Define loss and optimizer
#cost = tf.reduce_mean(tf.nn.l2_loss(y - pred))
cost = tf.reduce_mean(tf.square(pred - y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

# Initializing the variables
init = tf.initialize_all_variables()
saver = tf.train.Saver()

# Launch the graph
with tf.Session() as sess:
    sess.run(init)

    best_validation_epoch = float("inf")
    training_loss_arr = []
    # Keep training until reach max iterations
    for epoch in xrange(max_epochs):

        dirs = os.listdir(path)
        shuffle(dirs)
    	total_steps = int(round(len(dirs)/(batch_size*1.0)));
        #total_steps = 3000;
        print('Epoch: {}'.format(epoch))

        for step in range(0,total_steps - 1):

            total_loss = []
            verbose = 1

            (batch_xs, batch_ys) = DataManager.getBatch(batch_size, step, dirs)

            batch_x = np.copy(batch_xs).astype('float32')
            batch_x /= 255
            #print(batch_x)
            batch_y = np.squeeze(np.copy(batch_ys),axis=(2,)).astype('float32')

            nan_rows = []
            for it in range(0,batch_y.shape[0]):
                if np.isnan(np.sum(batch_y[it,])) == True:
                    nan_rows.append(it)

            nan_rows = np.asarray(nan_rows)
            batch_x = np.delete(batch_x, nan_rows, 0)
            batch_y = np.delete(batch_y, nan_rows, 0)
            #print(batch_x.shape)
            #print(batch_y.shape)

            #if(np.isnan(np.sum(batch_y)) == False):

            # Run optimization op (backprop)
            sess.run(optimizer, feed_dict={x: batch_x, y: batch_y,
                                       keep_prob: dropout})
            # Calculate batch loss and accuracy
            [loss,y_true,y_pred] = sess.run([cost,y,pred], feed_dict={x: batch_x,
                                             y: batch_y,
                                             keep_prob: 1.})


            total_loss.append(loss)

            if verbose and step % verbose == 0:
                sys.stdout.write('\r{} / {} : loss = {}'.format(
                step, total_steps, np.mean(total_loss)))
                sys.stdout.flush()

        if verbose:
            sys.stdout.write('\r')

        training_loss = np.mean(total_loss)
        print('Training loss: {}'.format(training_loss))
        print(y_true)
        print(y_pred)


        """Testing on Validation dataset"""

        (valid_xs, valid_ys) = DataManager.getValidationSet(dirs)
        valid_x = np.copy(valid_xs).astype('float32')
        valid_x /= 255
        valid_y = np.squeeze(np.copy(valid_ys),axis=(2,)).astype('float32')

        nan_rows = []
        for it in range(0,valid_y.shape[0]):
            if np.isnan(np.sum(valid_y[it,])) == True:
                nan_rows.append(it)

        nan_rows = np.asarray(nan_rows)
        valid_x = np.delete(valid_x, nan_rows, 0)
        valid_y = np.delete(valid_y, nan_rows, 0)

        #print(valid_x)
        #print(valid_y)

        [validation_loss,valid_true,valid_predicted] = sess.run([cost,y,pred], feed_dict={x: valid_x,
                                                    y: valid_y,
                                                    keep_prob: 1.})

        print('Validation loss: {}'.format(validation_loss))
        #print(valid_true)
        #print(valid_predicted)

        if validation_loss < best_validation_epoch:
            saver.save(sess, 'bear/cnn.weights')
            best_validation_epoch = validation_loss

        training_loss_arr.append(training_loss)


    print("Optimization Finished!")
