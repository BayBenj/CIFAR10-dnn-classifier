
import pandas as pd
import numpy as np
from pandas import DataFrame
import tensorflow as tf
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

def nbimage( data ):
    from IPython.display import display, Image
    from PIL.Image import fromarray
    from StringIO import StringIO

    s = StringIO()
    fromarray( data ).save( s, 'png' )
    display( Image( s.getvalue() ) )

def conv( x, filter_size=3, stride=2, num_filters=64, is_output=False, name="conv" ):
  print("\nCONV LAYER: {}".format(name))
  with tf.name_scope( name ) as scope:
      with tf.variable_scope( name ):
        x_dims = x.get_shape().as_list() #[batch, in_height, in_width, in_channels]
        print("x shape: {}".format(x_dims))

        batch = 1
        in_height = x_dims[1]
        in_width = x_dims[2]
        in_channels = x_dims[3]

        W = tf.get_variable(name = "W", shape=[filter_size, filter_size, in_channels, num_filters], initializer = tf.contrib.layers.variance_scaling_initializer())
        print("W shape: {}".format(W.get_shape().as_list()))

        B = tf.get_variable(name = "B", shape=[1,1,1,num_filters], initializer = tf.contrib.layers.variance_scaling_initializer())
        print("B shape: {}".format(B.get_shape().as_list()))

        strides = [1,stride,stride,1]

        activation_map = tf.nn.conv2d(x, W, strides, padding = "SAME")
        activation_map += B

        if (not is_output):
            activation_map = tf.nn.relu(activation_map)

        print("activation_map shape: {}".format(activation_map.get_shape().as_list()))
        return activation_map

def fc( x, out_size=50, is_output=False, name="fc" ):#perceptron/gradient descent?
  print("\nFULLY-CONNECTED LAYER: {}".format(name))
  with tf.name_scope( name ) as scope:
      with tf.variable_scope( name ):
        x_flat = tf.reshape(x,[-1, 1])
        print("x_flat shape: {}".format(x_flat.get_shape().as_list()))
        x_dims = x_flat.get_shape().as_list()
        W = tf.get_variable(name = "W", shape=[out_size, x_dims[0]], initializer = tf.contrib.layers.variance_scaling_initializer())#len(x), out_size
        print("W shape: {}".format(W.get_shape().as_list()))
        B = tf.get_variable(name = "B", shape=[out_size, 1], initializer = tf.contrib.layers.variance_scaling_initializer())#len(x), out_size
        print("B shape: {}".format(B.get_shape().as_list()))
        result = tf.matmul(W, x_flat) + B
        if (not is_output):
          result = tf.nn.relu(result)
        print("result shape:{}".format(result.get_shape().as_list()))
        return result

def unpickle( file ):
    import cPickle
    fo = open(file, 'rb')
    dict = cPickle.load(fo)
    fo.close()
    return dict

def loadCIFAR_10():
    data = unpickle( 'cifar-10-batches-py/data_batch_1' )
    features = data['data']
    labels = data['labels']
    labels = np.atleast_2d( labels ).T
    # squash classes 0-4 into class 0, and squash classes 5-9 into class 1
    # labels[ labels < 5 ] = 0
    # labels[ labels >= 5 ] = 1
    return data, features, labels

def split_train_test(train, features, labels):
    train_n = int(len(features) * train)

    state = np.random.get_state()
    np.random.shuffle(features)
    np.random.set_state(state)
    np.random.shuffle(labels)

    train_f = features[:train_n]
    test_f = features[train_n:]
    train_l = labels[:train_n]
    test_l = labels[train_n:]

    return train_f, test_f, train_l, test_l

def loss(logits, label):
  logits = tf.transpose(logits)
  label = tf.cast(label, tf.int64)
  print("label: {}".format(label))
  cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=label, logits=logits, name='cross_entropy')
  cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy_mean')
  return cross_entropy_mean

input_data = tf.placeholder(tf.float32, [1,32,32,3])
tf_label = tf.placeholder(tf.int64)
learning_rate = tf.placeholder(tf.float32)

#DNN topology
  #convolution layers
h0 = conv( input_data, name="h0" )
h1 = conv( h0, name="h1" )
h2 = conv( h1, name="h2" )

  #fully-connected layers
fc0 = fc( h2, out_size=256, name="fc0" )
fc1 = fc( fc0, out_size=64, name="fc1" )
logits = fc( fc1, out_size=10, is_output=True, name="fc2" )

loss = loss(logits, tf_label)
train = tf.train.AdamOptimizer(learning_rate=learning_rate, name = "train").minimize(loss)
predict = tf_label - tf.argmax(logits)

init = tf.global_variables_initializer()

with tf.Session() as sess:
  writer = tf.summary.FileWriter("/Users/Benjamin/Desktop/byu/Semester 8/CS501R/lab5", sess.graph)
  sess.run(init)

  c = 0.00001
  CIFAR_10_IMAGE_DIMS = [1,32,32,3]
  data, features, labels = loadCIFAR_10()

  # training_features = training_set["features"]
  # training_labels = training_set["labels"]
  # testing_features = testing_set["features"]
  # testing_labels = testing_set["labels"]

  #train
  # print("TRAINING FEATURE ROWS: {}".format(len(training_features)))
  # print("TRAINING FEATURE COLS: {}".format(len(training_features[0])))
  # print("TRAINING LABEL LEN: {}".format(len(training_labels)))
  # print("TRAINING LABELS: {}".format(training_labels))
  # print("training_labels[0]: {}".format(training_labels[0]))
  EPOCHS = 100
  train_accuracies = []
  test_accuracies = []
  train_losses = []
  test_losses = []
  for n in range(EPOCHS):
    training_features, testing_features, training_labels, testing_labels = split_train_test(0.8, features, labels)
    i = 0
    all_guesses = []
    train_loss_sum = 0
    for row in training_features:
        image = np.reshape(row, CIFAR_10_IMAGE_DIMS)
        _, train_loss, train_prediction = sess.run([train, loss, predict], feed_dict = {input_data: image, tf_label: training_labels[i], learning_rate: c})
        if train_prediction == 0:
            image_guess = 1.0
        else:
            image_guess = 0.0
        all_guesses.append(image_guess)
        train_loss_sum += train_loss
        i += 1
    mean_epoch_train_loss = float(train_loss_sum) / float(EPOCHS)
    train_losses.append(mean_epoch_train_loss)
    train_accuracy = float(sum(all_guesses)) / float(len(all_guesses))
    train_accuracies.append(train_accuracy)
    print("EPOCH {} TRAINING ACCURACY: {}".format(n, train_accuracy))

  # print("TESTING ROWS: {}".format(len(testing_features)))
  # print("TESTING COLS: {}".format(len(testing_features[0])))
  # print("TESTING LABEL LEN: {}".format(len(testing_labels)))
    all_guesses = []
    i = 0
    test_loss_sum = 0
    for row in testing_features:
      image = np.reshape(row, CIFAR_10_IMAGE_DIMS)
      test_loss, test_prediction = sess.run([loss, predict], feed_dict = {input_data: image, tf_label: testing_labels[i], learning_rate: c})
      if test_prediction == 0:
          image_guess = 1.0
      else:
          image_guess = 0.0
      all_guesses.append(image_guess)
      test_loss_sum += test_loss
      i += 1
    mean_epoch_test_loss = float(test_loss_sum) / float(EPOCHS)
    test_losses.append(mean_epoch_test_loss)
    test_accuracy = float(sum(all_guesses)) / float(len(all_guesses))
    test_accuracies.append(test_accuracy)
    print("EPOCH {} TESTING ACCURACY: {}".format(n, test_accuracy))

  writer.close

plt.figure(1)
plt.plot(range(1, EPOCHS + 1), train_accuracies)
plt.plot(range(1, EPOCHS + 1), test_accuracies)
plt.legend(['train','test'], loc='center left', bbox_to_anchor=(1, 0.5))
plt.ylabel('Classification Accuracy (%)')
plt.xlabel('Epochs')
plt.show()

plt.figure(2)
plt.plot(range(1, EPOCHS + 1), train_losses)
plt.plot(range(1, EPOCHS + 1), test_losses)
plt.legend(['train','test'], loc='center left', bbox_to_anchor=(1, 0.5))
plt.ylabel('Loss')
plt.xlabel('Epochs')
plt.show()

from IPython import display
from PIL import Image
from matplotlib.pyplot import imshow
pil_im = Image.open('comp-graph.png', 'r')

# display the image
#nbimage( data )
nbimage(np.asarray(pil_im))



#When predicting, use arg_max to find index of highest score
# cross_entropy loss function -> tf.nn.sparse_softmax_cross_entropy_with_logits()
# Train the network using tf.train.AdamOptimizer <- give it the number you want to minimize
#normalize RGB values from 0 to 1
#1hot encoding = 1 only at the index that's correct
