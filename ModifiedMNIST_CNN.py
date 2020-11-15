import tensorflow as tf
import numpy as np
import LoadTFRecord as load
import sys
import matplotlib.pyplot as plt

# Model helper
x = tf.placeholder(tf.float32, shape=[None,64,64,1])
y_true = tf.placeholder(tf.float32, shape=[None,40])
hold_prob = tf.placeholder(tf.float32)

batch_size = 200
buffer_size = 400
test_batch_size = 100
train_x, train_y_read = load.input_fn('my_train2.tfrecord', False, batch_size)

train_x = tf.transpose(train_x, [0,2,1])
train_x = tf.reshape(train_x, [batch_size,64,64,1])
train_x /= 255

train_y = tf.cast(train_y_read, tf.int32)
train_y = load.one_hot_labels(train_y)

test_x, test_y = load.input_fn('my_test2.tfrecord', False, test_batch_size)
test_x = tf.transpose(test_x, [0,2,1])
test_x = tf.reshape(test_x, [-1,64,64,1])
test_x /= 255

test_y = tf.cast(test_y, tf.int32)
test_y = load.one_hot_labels(test_y)

# use training data to test accurcy of model
train_as_test_x, train_as_test_y = load.input_fn('my_train2.tfrecord', False, test_batch_size)
train_as_test_x = tf.transpose(train_as_test_x, [0,2,1])
train_as_test_x = tf.reshape(train_as_test_x, [-1,64,64,1])
train_as_test_x /= 255

train_as_test_y = tf.cast(train_as_test_y, tf.int32)
train_as_test_y = load.one_hot_labels(train_as_test_y)

def init_weights(shape):
    #stddev = np.sqrt(2 / (shape[0] + shape[1]))
    #init_random_dist = tf.truncated_normal(shape, stddev=stddev)
    init_random_dist = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(init_random_dist)

def init_bias(shape):
    init_bias_vals = tf.constant(0.05, shape=shape)
    return tf.Variable(init_bias_vals)

def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides = [1,1,1,1], padding='SAME')

def max_pool_2by2(x):
    return tf.layers.max_pooling2d(inputs=x, pool_size=[3, 3], strides=2)

def convolutional_layer(input_x, shape):
    W = init_weights(shape)
    b = init_bias([shape[3]])
    return tf.nn.relu(conv2d(input_x, W) + b)

def convolutional_layer2(input_x, shape):
    conv = tf.layers.conv2d(
      inputs=input_x,
      filters=shape[3],
      kernel_size=[shape[0], shape[1]],
      padding="same",
      kernel_initializer=tf.truncated_normal_initializer(stddev=0.05),
      bias_initializer=tf.random_uniform_initializer(minval=0.0, maxval=0.05),
      activation=tf.nn.relu)
    return conv

def normal_full_layer(input_layer, size):
    input_size = int(input_layer.get_shape()[1])
    W = init_weights([input_size, size])
    b = init_bias([size])
    return tf.matmul(input_layer, W) + b

# Create the Layers
convo_1 = convolutional_layer2(x,shape=[9,9,1,40])
convo_1_pooling = max_pool_2by2(convo_1)
convo_1_normalized = tf.layers.batch_normalization(convo_1_pooling)

convo_2 = convolutional_layer2(convo_1_normalized,shape=[5,5,40,64])
convo_2_pooling = max_pool_2by2(convo_2)
convo_2_normalized = tf.layers.batch_normalization(convo_2_pooling)

convo_3 = convolutional_layer2(convo_2_normalized,shape=[3,3,64,64])
convo_3_pooling = max_pool_2by2(convo_3)
convo_3_normalized = tf.layers.batch_normalization(convo_3_pooling)

convo_4 = convolutional_layer2(convo_3_normalized,shape=[3,3,64,96])
convo_4_pooling = max_pool_2by2(convo_4)
convo_4_normalized = tf.layers.batch_normalization(convo_4_pooling)

convo_1_flat = tf.reshape(convo_4_normalized, shape=[-1,3*3*96])
full_layer_1 = tf.nn.relu(normal_full_layer(convo_1_flat,1024))
dropout_1 = tf.nn.dropout(full_layer_1, keep_prob=hold_prob)

full_layer_2 = tf.nn.relu(normal_full_layer(dropout_1,512))
dropout_2 = tf.nn.dropout(full_layer_2, keep_prob=hold_prob)

y_pred = normal_full_layer(dropout_2,40)

# # # simple model
# # # Layers
# # convo_1 = convolutional_layer(x, shape=[5,5,1,32])
# # convo_1_pooling = max_pool_2by2(convo_1)

# # convo_2 = convolutional_layer(convo_1_pooling, shape = [5,5,32,64])
# # convo_2_pooling = max_pool_2by2(convo_2)

# # convo_1_flat = tf.reshape(convo_2_pooling, shape=[-1,16384])
# # fully_layer_one = tf.nn.relu(normal_full_layer(convo_1_flat,1024))

# # # Dropout
# # full_one_dropout = tf.nn.dropout(fully_layer_one,keep_prob = hold_prob)

# # y_pred = normal_full_layer(full_one_dropout, 40)

cross_entropy_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits (labels=y_true, logits=y_pred))

optimizer = tf.train.AdamOptimizer(learning_rate=0.0005)
train = optimizer.minimize(cross_entropy_loss)

init = tf.global_variables_initializer()

steps = 20000

loss_list = []
acc_list =[]
test_acc_index = []
train_as_test_acc_list = []
train_acc_index = []
with tf.Session() as sess:
    
    sess.run(init)

    for i in range(steps):
        
        batch_x, batch_y = sess.run([train_x, train_y])
        _, loss = sess.run((train, cross_entropy_loss),feed_dict={x:batch_x,y_true:batch_y,hold_prob:0.35})
        
        print('Performing step {}. Loss is {}'.format(i, loss), end='\r')

        loss_list.append(loss)
        
        if i > 10000:
	        plt.figure(1)
	        plt.clf()
	        plt.title('Cross entropy loss',)
	        plt.plot(loss_list)
	        plt.draw()
	        plt.pause(0.01)

        # PRINT OUT A MESSAGE EVERY 50 STEPS
        if i%100 == 0:
            print()
            print('Currently on step {}'.format(i))
            print('Accuracy is:')

            matches = tf.equal(tf.argmax(y_pred,1),tf.argmax(y_true,1))
            all_matches = []

            for j in range(100):
                # Test the Train Model
                test_batch_x, test_batch_y = sess.run([test_x, test_y])
                all_matches.append(sess.run(matches, feed_dict={x:test_batch_x,y_true:test_batch_y,hold_prob:1.0}))
                #all_matches = tf.shape(tf.concat(all_matches, sess.run(matches, feed_dict={x:test_batch_x,y_true:test_batch_y,hold_prob:1.0})))

            all_matches = tf.stack(all_matches)
            all_matches = tf.reshape(all_matches, [1,10000])
            acc = tf.reduce_mean(tf.cast(all_matches,tf.float32))
            acc_eval = sess.run(acc)
            acc_list.append(acc_eval)
            test_acc_index.append(i)
            
            if i > 10000:
	            plt.figure(2)
	            plt.clf()
	            plt.title('Accuracy of model on the test data')
	            plt.plot(test_acc_index, acc_list)
	            plt.draw()
	            plt.pause(0.01)
            
            print(acc_eval)
            print('\n')

        if i%1000 == 0:

            print('Currently on step {}'.format(i))
            print('Accuracy on train data is:')

            # check the accuracy of model against all training data
            matches = tf.equal(tf.argmax(y_pred,1),tf.argmax(y_true,1))
            all_matches = []
            
            for k in range(500):
                # Test the Train Model
                train_as_test_batch_x, train_as_test_batch_y = sess.run([train_as_test_x, train_as_test_y])
                all_matches.append(sess.run(matches, feed_dict={x:train_as_test_batch_x,y_true:train_as_test_batch_y,hold_prob:1.0}))

            all_matches = tf.stack(all_matches)
            all_matches = tf.reshape(all_matches, [1,50000])
            acc = tf.reduce_mean(tf.cast(all_matches,tf.float32))
            acc_eval = sess.run(acc)
            train_as_test_acc_list.append(acc_eval)
            train_acc_index.append(i)
            
            print(acc_eval)
            print('\n')


            if i>10000:
	            plt.figure(3)
	            plt.clf()
	            plt.title('Accuracy of model on the training data')
	            plt.plot(train_acc_index, train_as_test_acc_list)
	            plt.draw()
	            plt.pause(0.01)
    
    inputs = {
            "features_placeholder": x,
            "labels_placeholder": y_true,
            }
    outputs = {"prediction": y_pred}
    tf.saved_model.simple_save(sess, 'model1', inputs, outputs)

plt.show()