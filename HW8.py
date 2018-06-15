import os
import struct
import numpy as np
import tensorflow as tf
import sys
import gzip
import matplotlib.pyplot as plt


def load_mnist(path, kind='train'):
    """Load MNIST data from `path`"""
    labels_path = os.path.join(path,
                               '%s-labels-idx1-ubyte' % kind)
    images_path = os.path.join(path,
                               '%s-images-idx3-ubyte' % kind)

    with open(labels_path, 'rb') as lbpath:
        magic, n = struct.unpack('>II',
                                 lbpath.read(8))
        labels = np.fromfile(lbpath,
                             dtype=np.uint8)

    with open(images_path, 'rb') as imgpath:
        magic, num, rows, cols = struct.unpack(">IIII",
                                               imgpath.read(16))
        images = np.fromfile(imgpath,
                             dtype=np.uint8).reshape(len(labels), 784)
        images = ((images / 255.) - .5) * 2

    return images, labels


# unzips mnist


if (sys.version_info > (3, 0)):
    writemode = 'wb'
else:
    writemode = 'w'

zipped_mnist = [f for f in os.listdir('./') if f.endswith('ubyte.gz')]
for z in zipped_mnist:
    with gzip.GzipFile(z, mode='rb') as decompressed, open(z[:-3], writemode) as outfile:
        outfile.write(decompressed.read())

X, y = load_mnist('', kind='train')

X_train = X[:55000,:]
y_train = y[:55000]
X_valid = X[55000:60000,:]
y_valid = y[55000:60000]

X_test, y_test = load_mnist('', kind='t10k')
X_test = X_test[:10000,:]
y_test = y_test[:10000]

print('Training set : \nRows: %d, columns: %d' % (X_train.shape[0], X_train.shape[1]))
print('Validation set : \nRows: %d, columns: %d' % (X_valid.shape[0], X_valid.shape[1]))
print('Testing set : \nRows: %d, columns: %d' % (X_test.shape[0], X_test.shape[1]))


def batch_generator(X, y, batch_size=64,
                    shuffle=False, random_seed=None):
    idx = np.arange(y.shape[0])

    if shuffle:
        rng = np.random.RandomState(random_seed)
        rng.shuffle(idx)
        X = X[idx]
        y = y[idx]

    for i in range(0, X.shape[0], batch_size):
        yield (X[i:i + batch_size, :], y[i:i + batch_size])


mean_vals = np.mean(X_train, axis=0)
std_val = np.std(X_train)

X_train_centered = (X_train - mean_vals) / std_val
X_valid_centered = X_valid - mean_vals
X_test_centered = (X_test - mean_vals) / std_val

del X_train, X_valid, X_test




## Implementing a CNN in TensorFlow low-level API

## wrapper functions
def conv_layer(input_tensor, name,
               kernel_size, n_output_channels,
               padding_mode='SAME', strides=(1, 1, 1, 1)):
    with tf.variable_scope(name):
        ## get n_input_channels:
        ##   input tensor shape:
        ##   [batch x width x height x channels_in]
        input_shape = input_tensor.get_shape().as_list()
        n_input_channels = input_shape[-1]

        weights_shape = (list(kernel_size) +
                         [n_input_channels, n_output_channels])

        weights = tf.get_variable(name='_weights',
                                  shape=weights_shape)
        print(weights)
        biases = tf.get_variable(name='_biases',
                                 initializer=tf.zeros(
                                     shape=[n_output_channels]))
        print(biases)
        conv = tf.nn.conv2d(input=input_tensor,
                            filter=weights,
                            strides=strides,
                            padding=padding_mode)
        print(conv)
        conv = tf.nn.bias_add(conv, biases,
                              name='net_pre-activation')
        print(conv)
        conv = tf.nn.relu(conv, name='activation')
        print(conv)

        return conv


# Fully Connected Layer
def fc_layer(input_tensor, name,
             n_output_units, activation_fn=None):
    with tf.variable_scope(name):
        input_shape = input_tensor.get_shape().as_list()[1:]
        n_input_units = np.prod(input_shape)
        if len(input_shape) > 1:
            input_tensor = tf.reshape(input_tensor,
                                      shape=(-1, n_input_units))

        weights_shape = [n_input_units, n_output_units]

        weights = tf.get_variable(name='_weights',
                                  shape=weights_shape)
        print(weights)
        biases = tf.get_variable(name='_biases',
                                 initializer=tf.zeros(
                                     shape=[n_output_units]))
        print(biases)
        layer = tf.matmul(input_tensor, weights)
        print(layer)
        layer = tf.nn.bias_add(layer, biases,
                               name='net_pre-activation')
        print(layer)
        if activation_fn is None:
            return layer

        layer = activation_fn(layer, name='activation')
        print(layer)
        return layer



def build_cnn(LnReg = 0, LnBeta = 0):
    # L1 혹은 L2 regularization을 위해서 만든 두 parameter.
    # Param
    #   LnReg = 0, 1, 2
    #       0인 경우 Regularization을 하지 않고
    #       1인 경우 L1,
    #       2인 경우 L2 Regularization term을 loss function에 추가.
    #   LnBeta = float
    #       LnReg 가 1 혹은 2인 경우 beta 값으로 사용.
    ## Placeholders for X and y:
    tf_x = tf.placeholder(tf.float32, shape=[None, 784],
                          name='tf_x')
    tf_y = tf.placeholder(tf.int32, shape=[None],
                          name='tf_y')

    # reshape x to a 4D tensor:
    # [batchsize, width, height, 1]
    tf_x_image = tf.reshape(tf_x, shape=[-1, 28, 28, 1],
                            name='tf_x_reshaped')
    ## One-hot encoding:
    tf_y_onehot = tf.one_hot(indices=tf_y, depth=10,
                             dtype=tf.float32,
                             name='tf_y_onehot')

    ## 1st layer: Conv_1
    print('\nBuilding 1st layer: ')
    h1 = conv_layer(tf_x_image, name='conv_1',
                    kernel_size=(5, 5),
                    padding_mode='VALID',
                    n_output_channels=32)
    ## MaxPooling
    h1_pool = tf.nn.max_pool(h1,
                             ksize=[1, 2, 2, 1],
                             strides=[1, 2, 2, 1],
                             padding='SAME')
    ## 2n layer: Conv_2
    print('\nBuilding 2nd layer: ')
    h2 = conv_layer(h1_pool, name='conv_2',
                    kernel_size=(5, 5),
                    padding_mode='VALID',
                    n_output_channels=64)
    ## MaxPooling
    h2_pool = tf.nn.max_pool(h2,
                             ksize=[1, 2, 2, 1],
                             strides=[1, 2, 2, 1],
                             padding='SAME')

    ## 3rd layer: Fully Connected
    print('\nBuilding 3rd layer:')
    h3 = fc_layer(h2_pool, name='fc_3',
                  n_output_units=1024,
                  activation_fn=tf.nn.relu)

    ## Dropout
    keep_prob = tf.placeholder(tf.float32, name='fc_keep_prob')
    h3_drop = tf.nn.dropout(h3, keep_prob=keep_prob,
                            name='dropout_layer')

    ## 4th layer: Fully Connected (linear activation)
    print('\nBuilding 4th layer:')
    h4 = fc_layer(h3_drop, name='fc_4',
                  n_output_units=10,
                  activation_fn=None)

    ## Prediction
    predictions = {
        'probabilities': tf.nn.softmax(h4, name='probabilities'),
        'labels': tf.cast(tf.argmax(h4, axis=1), tf.int32,
                          name='labels')
    }

    ## Loss Function and Optimization
    if LnReg == 0: # Ln Regularization 없음.
        cross_entropy_loss = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits_v2(
                logits=h4, labels=tf_y_onehot),
            name='cross_entropy_loss')
    elif LnReg == 1: # L1 Regularization
        with tf.variable_scope('fc_3',reuse=True):
            ln_values = tf.get_variable('_weights')
        cross_entropy_loss = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits_v2(
                logits=h4, labels=tf_y_onehot)
            + LnBeta*tf.reduce_sum(tf.abs(ln_values)),
            name='cross_entropy_loss')
    elif LnReg == 2: # L2 Regularization
        with tf.variable_scope('fc_3',reuse=True):
            ln_values = tf.get_variable('_weights')
        cross_entropy_loss = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits_v2(
                logits=h4, labels=tf_y_onehot)
            + LnBeta*tf.nn.l2_loss(ln_values),
            name='cross_entropy_loss')
    else:
        raise Exception('LnReg value must be either 0, 1, 2')

    ## Optimizer:
    optimizer = tf.train.AdamOptimizer(learning_rate)
    optimizer = optimizer.minimize(cross_entropy_loss, name='train_op')

    ## Computing the prediction accuracy
    correct_predictions = tf.equal(
        predictions['labels'],
        tf_y, name='correct_preds')

    accuracy = tf.reduce_mean(
        tf.cast(correct_predictions, tf.float32),
        name='accuracy')


def save(saver, sess, epoch, path='./model/'):
    if not os.path.isdir(path):
        os.makedirs(path)
    print('Saving model in %s' % path)
    saver.save(sess, os.path.join(path, 'cnn-model.ckpt'),
               global_step=epoch)


def load(saver, sess, path, epoch):
    print('Loading model from %s' % path)
    saver.restore(sess, os.path.join(
        path, 'cnn-model.ckpt-%d' % epoch))


def train(sess, training_set, validation_set=None,
          initialize=True, epochs=20, shuffle=True,
          dropout=0.5, batch_size=64, random_seed=None):
    X_data = np.array(training_set[0])
    y_data = np.array(training_set[1])
    training_loss = []

    ## initialize variables
    if initialize:
        sess.run(tf.global_variables_initializer())

    np.random.seed(random_seed)  # for shuflling in batch_generator
    for epoch in range(1, epochs + 1):
        batch_gen = batch_generator(
            X_data, y_data, batch_size=batch_size,
            shuffle=shuffle)
        avg_loss = 0.0
        for i, (batch_x, batch_y) in enumerate(batch_gen):
            feed = {'tf_x:0': batch_x,
                    'tf_y:0': batch_y,
                    'fc_keep_prob:0': dropout}
            loss, _ = sess.run(
                ['cross_entropy_loss:0', 'train_op'],
                feed_dict=feed)
            avg_loss += loss

        training_loss.append(avg_loss / (i + 1))
        print('Epoch %02d Training Avg. Loss: %7.3f' % (
            epoch, avg_loss), end=' ')
        if validation_set is not None:
            feed = {'tf_x:0': validation_set[0],
                    'tf_y:0': validation_set[1],
                    'fc_keep_prob:0': 1.0}
            valid_acc = sess.run('accuracy:0', feed_dict=feed)
            print(' Validation Acc: %7.3f' % valid_acc)
        else:
            print()


def predict(sess, X_test, return_proba=False):
    feed = {'tf_x:0': X_test,
            'fc_keep_prob:0': 1.0}
    if return_proba:
        return sess.run('probabilities:0', feed_dict=feed)
    else:
        return sess.run('labels:0', feed_dict=feed)




#############################################################
#############   Initialize & Train the CNN   ################
#############################################################

## Define hyperparameter sets
param_dropout = [0.1, 0.5, 1.0]
param_L2 = [0, 0.1, 0.001]
param_L1 = [0, 0.1, 0.001]
param_batch = [1,4,16,64,128,1024,55000]


## Define fixed hyperparameters
learning_rate = 1e-4
random_seed = 123
epoch = 6

np.random.seed(random_seed)



######################################################################################


## create a graph
g = tf.Graph()
with g.as_default():
    tf.set_random_seed(random_seed)
    ## build the graph
    build_cnn(LnReg = 0, LnBeta = 0)
    ## saver:
    saver = tf.train.Saver()

## crearte a TF session and train the CNN model

with tf.Session(graph=g) as sess:
    train(sess,
          training_set=(X_train_centered, y_train),
          validation_set=(X_valid_centered, y_valid),
          initialize=True,
          epochs=epoch,
          shuffle=True,
          dropout=0.5,
          batch_size=64,
          random_seed=123)
    save(saver, sess, epoch=epoch)



### Calculate prediction accuracy
### on test set
### restoring the saved model

for i in range(5):
    del g

    ## create a new graph
    ## and build the model
    g = tf.Graph()
    with g.as_default():
        tf.set_random_seed(random_seed)
        ## build the graph
        build_cnn()

        ## saver:
        saver = tf.train.Saver()

    ## create a new session
    ## and restore the model
    with tf.Session(graph=g) as sess:
        load(saver, sess,
             epoch=i+1, path='./model/')

        preds = predict(sess, X_test_centered,
                        return_proba=False)

        print('Test Accuracy: %.3f%%' % (100 * np.sum(preds == y_test) / len(y_test)))
#
#
# ## run the prediction on
# ##  some test samples
#
# np.set_printoptions(precision=2, suppress=True)
#
# with tf.Session(graph=g2) as sess:
#     load(saver, sess,
#          epoch=20, path='./model/')
#
#     print(predict(sess, X_test_centered[:10],
#                   return_proba=False))
#
#     print(predict(sess, X_test_centered[:10],
#                   return_proba=True))

#
#
#
# ## continue training for 20 more epochs
# ## without re-initializing :: initialize=False
# ## create a new session
# ## and restore the model
# with tf.Session(graph=g2) as sess:
#     load(saver, sess,
#          epoch=20, path='./model/')
#
#     train(sess,
#           training_set=(X_train_centered, y_train),
#           validation_set=(X_valid_centered, y_valid),
#           initialize=False,
#           epochs=20,
#           random_seed=123)
#
#     save(saver, sess, epoch=40, path='./model/')
#
#     preds = predict(sess, X_test_centered,
#                     return_proba=False)
#
#     print('Test Accuracy: %.3f%%' % (100 * np.sum(preds == y_test) / len(y_test)))
#
