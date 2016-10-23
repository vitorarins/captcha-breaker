import tensorflow as tf
import glob, os
import numpy as np
import cv2
import random

class OCR_recognizer(object):
    def __init__(self, model_filename, num_classes=36, num_channels=1, num_chars=5, resize_height=24, resize_width=88):
        self.model_filename = model_filename
        self.num_classes = num_classes
        self.num_channels = num_channels
        self.num_chars = num_chars
        self.resize_height = resize_height
        self.resize_width = resize_width
        self._imgs = []
        self._labels = []
        # tf Graph input
        fc_num_outputs = 4096
        self.l2_beta_param = 3e-4
        self.initial_learning_rate = 0.01
        self.decay_steps = 1000
        self.decay_rate = 0.9
        self.momentum = 0.9
        self.dropout_keep_prob = 0.5
        self.x = tf.placeholder(tf.float32, [None, resize_height, resize_width, num_channels])
        self.y = tf.placeholder(tf.float32, [None, num_chars*num_classes])
        self.weights = {
	    'wc1': self.weight_variable('wc1',[5, 5, num_channels, 64]),
	    'wc2': self.weight_variable('wc2',[5, 5, 64, 128]),
	    'wc3': self.weight_variable('wc3',[5, 5, 128, 256]),
	    'wc4': self.weight_variable('wc4',[3, 3, 256, 512]),
	    'wd1': self.weight_variable('wd1',[(resize_height/8)*(resize_width/8)*512, fc_num_outputs]),
	    'out1': self.weight_variable('out1',[fc_num_outputs, num_classes]),
	    'out2': self.weight_variable('out2',[fc_num_outputs, num_classes]),
	    'out3': self.weight_variable('out3',[fc_num_outputs, num_classes]),
	    'out4': self.weight_variable('out4',[fc_num_outputs, num_classes]),
	    'out5': self.weight_variable('out5',[fc_num_outputs, num_classes])
        }
        self.biases = {
	    'bc1': self.bias_variable([64]),
	    'bc2': self.bias_variable([128]),
	    'bc3': self.bias_variable([256]),
            'bc4': self.bias_variable([512]),
	    'bd1': self.bias_variable([fc_num_outputs]),
	    'out1': self.bias_variable([num_classes]),
	    'out2': self.bias_variable([num_classes]),
	    'out3': self.bias_variable([num_classes]),
	    'out4': self.bias_variable([num_classes]),
	    'out5': self.bias_variable([num_classes])
        }
        self.logits = self.ocr_net(self.x, self.weights, self.biases, self.dropout_keep_prob)
        self.define_graph()

    def __enter__(self):
        return self

    def __exit__(self, *err):
        tf.reset_default_graph()

    def create_captcha(self, pathAndFilename):
        img = cv2.imread(pathAndFilename, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img,(self.resize_width, self.resize_height), interpolation=cv2.INTER_AREA)
        filename, ext = os.path.splitext(os.path.basename(pathAndFilename))
        label = self.create_label(filename)
        return (img, label)

    def create_label(self, filename):
        label = []
        for c in filename:
            ascii_code = ord(c)
            if ascii_code < 58:
                char_value = ascii_code - 48
            else:
                char_value = ascii_code - 87
            label.append(char_value)
        return self.dense_to_one_hot(label)

    def dense_to_one_hot(self, labels_dense):
	num_labels = len(labels_dense)
	index_offest = np.arange(num_labels) * self.num_classes
	labels_one_hot = np.zeros((num_labels, self.num_classes))
	labels_one_hot.flat[index_offest + labels_dense] = 1
	labels_one_hot = labels_one_hot.reshape(num_labels*self.num_classes)
	return labels_one_hot
        
    def get_prediction_string(self, one_hot_predictions):
        reshaped = one_hot_predictions.reshape(self.num_chars, self.num_classes)
        correct_pred = np.argmax(reshaped, 1)
        final_string = ""
        for char_value in correct_pred:
            if char_value > 9:
                final_string += chr(char_value + 87)
            else:
                final_string += chr(char_value + 48)
        return final_string

    def image_to_batch(self, image_path):
	img, label = self.create_captcha(image_path)
        self._imgs.append(img)
        self._labels.append(label)
        self._imgs = np.array(self._imgs).reshape((-1, self.resize_height, self.resize_width, self.num_channels)).astype(np.float32)
        self._labels = np.array(self._labels)
        return self._imgs[0:1], self._labels[0:1]

    def weight_variable(self, name, shape):
        return tf.get_variable(name, shape, initializer=tf.contrib.layers.xavier_initializer())

    def bias_variable(self, shape):
	initial = tf.constant(0.0, shape=shape)
	return tf.Variable(initial, trainable=True)

    def max_pool(self, x, k, name):
	return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1], padding='SAME', name=name)

    def conv2d(self, x, W, B, name):
    	conv = tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')
	bias = tf.nn.bias_add(conv, B)
	conv = tf.nn.relu(bias, name=name)
	return conv

    def ocr_net(self, _x, _weights, _biases, keep_prob):
	_x = tf.reshape(_x, shape=[-1, self.resize_height, self.resize_width, self.num_channels])

	conv1 = self.conv2d(_x, _weights['wc1'], _biases['bc1'], 'conv1')
	lrn1 = tf.nn.local_response_normalization(conv1)
	pool1 = self.max_pool(lrn1, k=2, name='pool1')

	conv2 = self.conv2d(pool1, _weights['wc2'], _biases['bc2'], 'conv2')
        lrn2 = tf.nn.local_response_normalization(conv2)
        pool2 = self.max_pool(lrn2, k=2, name='pool2')

	conv3 = self.conv2d(pool2, _weights['wc3'], _biases['bc3'], 'conv3')
        lrn3 = tf.nn.local_response_normalization(conv3)
        pool3 = self.max_pool(lrn3, k=2, name='pool3')

	conv4 = self.conv2d(pool3, _weights['wc4'], _biases['bc4'], 'conv4')

        dropout = tf.nn.dropout(conv4, keep_prob)

        shape = dropout.get_shape().as_list()
	reshaped = tf.reshape(dropout, [-1, _weights['wd1'].get_shape().as_list()[0]])

	fc1 = tf.nn.relu(tf.matmul(reshaped, _weights['wd1']) + _biases['bd1'], name='fc1')

	fc21 = tf.nn.relu(tf.matmul(fc1, _weights['out1']) + _biases['out1'], name='fc21')

	fc22 = tf.nn.relu(tf.matmul(fc1, _weights['out2']) + _biases['out2'], name='fc22')

	fc23 = tf.nn.relu(tf.matmul(fc1, _weights['out3']) + _biases['out3'], name='fc23')

	fc24 = tf.nn.relu(tf.matmul(fc1, _weights['out4']) + _biases['out4'], name='fc24')

	fc25 = tf.nn.relu(tf.matmul(fc1, _weights['out5']) + _biases['out5'], name='fc25')

	return [fc21, fc22, fc23, fc24, fc25]

    def softmax_joiner(self, logits):
        return tf.transpose(tf.pack([tf.nn.softmax(logits[0]), tf.nn.softmax(logits[1]), \
                                     tf.nn.softmax(logits[2]), tf.nn.softmax(logits[3]), \
                                     tf.nn.softmax(logits[4])]), perm = [1,0,2])

    def define_graph(self):
        self.saver = tf.train.Saver()
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(self.logits[0],self.y[:,0:36])) +\
               tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(self.logits[1],self.y[:,36:72])) +\
               tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(self.logits[2],self.y[:,72:108])) +\
               tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(self.logits[3],self.y[:,108:144])) +\
               tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(self.logits[4],self.y[:,144:180]))

        # adding regularizers
        regularizers = (tf.nn.l2_loss(self.weights['wc1']) + tf.nn.l2_loss(self.biases['bc1']) +
                        tf.nn.l2_loss(self.weights['wc2']) + tf.nn.l2_loss(self.biases['bc2']) +
                        tf.nn.l2_loss(self.weights['wc3']) + tf.nn.l2_loss(self.biases['bc3']) +
                        tf.nn.l2_loss(self.weights['wc4']) + tf.nn.l2_loss(self.biases['bc4']))
        # Add the regularization term to the loss.
        self.loss = loss + self.l2_beta_param * regularizers
            
        global_step = tf.Variable(0)
        learning_rate = tf.train.exponential_decay(self.initial_learning_rate, global_step, self.decay_steps, self.decay_rate)

        self.optimizer = tf.train.MomentumOptimizer(learning_rate, self.momentum).minimize(loss, global_step=global_step)
        self.pred = self.softmax_joiner(self.logits)

    def recognize(self, image_path):        
        with tf.Session() as sess:
            init = tf.initialize_all_variables()
	    sess.run(init)
            self.saver.restore(sess, self.model_filename)

	    batch = self.image_to_batch(image_path)
	    # Fit training using batch data
	    _, predictions, l = sess.run([self.optimizer, self.pred, self.loss], feed_dict={self.x: batch[0], self.y: batch[1]})
        return self.get_prediction_string(predictions)

