import tensorflow as tf
import time
import numpy as np
from captcha_data import OCR_data
# Parameters
initial_learning_rate = 0.01
decay_steps = 1000
decay_rate = 0.9
momentum = 0.9
l2_beta_param = 3e-4
dropout_keep_prob = 0.75
training_iters = 200000
batch_size = 64
display_step = 20
summaries_dir = 'logs'

# Network Parameters
width = 180
resize_width = 88
height = 60
resize_height = 24
color_channels = 1
n_chars = 5
n_classes = 36  # 10+26
n_training_samples = 180000
n_test_samples = 100
fc_num_outputs = 4096

# Calculate Elapsed time
start_time = time.time()

data_train = OCR_data(n_training_samples, './images/train', n_classes)
data_test = OCR_data(n_test_samples, './images/test', n_classes)

# tf Graph input
x = tf.placeholder(tf.float32, [None, resize_height, resize_width, color_channels])
y = tf.placeholder(tf.float32, [None, n_chars*n_classes])
# x_test = tf.placeholder(tf.float32, [n_test_samples, resize_height, resize_width, color_channels])
# y_test = tf.placeholder(tf.float32, [n_test_samples, n_chars*n_classes])

def print_activations(t):
	print(t.op.name, t.get_shape().as_list())

def weight_variable(name,shape):
        return tf.get_variable(name, shape, initializer=tf.contrib.layers.xavier_initializer())

def bias_variable(shape):
	initial = tf.constant(0.0, shape=shape)
	return tf.Variable(initial, trainable=True)

def max_pool(x, k, name):
	return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1], padding='SAME', name=name)

def variable_summaries(var, name):
        """Attach a lot of summaries to a Tensor."""
        with tf.name_scope('summaries'):
                mean = tf.reduce_mean(var)
                tf.scalar_summary('mean/' + name, mean)
                with tf.name_scope('stddev'):
                        stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
                tf.scalar_summary('stddev/' + name, stddev)
                tf.scalar_summary('max/' + name, tf.reduce_max(var))
                tf.scalar_summary('min/' + name, tf.reduce_min(var))
                tf.histogram_summary(name, var)

def conv2d(x, W, B, name):
	with tf.name_scope(name) as scope:
                with tf.name_scope('weights'):
                        variable_summaries(W, name + '/weights')
                with tf.name_scope('biases'):
                        variable_summaries(B, name + '/biases')
		conv = tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')
		bias = tf.nn.bias_add(conv, B)
                with tf.name_scope('Wx_plus_b'):
                        tf.histogram_summary(name + '/pre_activatioins', bias)
		conv = tf.nn.relu(bias, name=scope)
                tf.histogram_summary(name + '/activatioins', conv)
		return conv


def ocr_net(_x, _weights, _biases, keep_prob):
	_x = tf.reshape(_x, shape=[-1, resize_height, resize_width, color_channels])
        tf.image_summary('input', _x, 10)

	conv1 = conv2d(_x, _weights['wc1'], _biases['bc1'], 'conv1')
	print_activations(conv1)
        lrn1 = tf.nn.local_response_normalization(conv1)
	pool1 = max_pool(lrn1, k=2, name='pool1')
	print_activations(pool1)

	conv2 = conv2d(pool1, _weights['wc2'], _biases['bc2'], 'conv2')
	print_activations(conv2)
        lrn2 = tf.nn.local_response_normalization(conv2)
	pool2 = max_pool(lrn2, k=2, name='pool2')
	print_activations(pool2)

	conv3 = conv2d(pool2, _weights['wc3'], _biases['bc3'], 'conv3')
	print_activations(conv3)
        lrn3 = tf.nn.local_response_normalization(conv3)
	pool3 = max_pool(lrn3, k=2, name='pool3')
	print_activations(pool3)

	conv4 = conv2d(pool3, _weights['wc4'], _biases['bc4'], 'conv4')
	print_activations(conv4)

        dropout = tf.nn.dropout(conv4, keep_prob)

        shape = dropout.get_shape().as_list()
	reshaped = tf.reshape(dropout, [-1, _weights['wd1'].get_shape().as_list()[0]])

	fc1 = tf.nn.relu(tf.matmul(reshaped, _weights['wd1']) + _biases['bd1'], name='fc1')
	print_activations(fc1)

	fc21 = tf.nn.relu(tf.matmul(fc1, _weights['out1']) + _biases['out1'], name='fc21')
	print_activations(fc21)

	fc22 = tf.nn.relu(tf.matmul(fc1, _weights['out2']) + _biases['out2'], name='fc22')
	print_activations(fc22)

	fc23 = tf.nn.relu(tf.matmul(fc1, _weights['out3']) + _biases['out3'], name='fc23')
	print_activations(fc23)

	fc24 = tf.nn.relu(tf.matmul(fc1, _weights['out4']) + _biases['out4'], name='fc24')
	print_activations(fc24)

	fc25 = tf.nn.relu(tf.matmul(fc1, _weights['out5']) + _biases['out5'], name='fc25')
	print_activations(fc25)

	# out = tf.concat(1, [fc21, fc22, fc23, fc24, fc25], name='out')
	# print_activations(out)
	return [fc21, fc22, fc23, fc24, fc25]

def accuracy_func(predictions, labels):
        # predictions = tf.reshape(predictions, shape=[-1, n_chars, n_classes])
        # labels = tf.reshape(labels, shape=[-1, n_chars, n_classes])
        # truth_table = np.zeros((predictions.get_shape().as_list()[0],5))
        # for i in range(5):
        #         truth_table[:,i] = tf.argmax(predictions[:,i,:], 1) == tf.argmax(labels[:,i,:], 1)
        # return (100 * np.sum(np.amin(truth_table, axis = 1))/predictions.get_shape().as_list()[0])
        with tf.name_scope('accuracy'):
	        y = tf.reshape(labels, shape=[-1, n_chars, n_classes])
                with tf.name_scope('prediction'):
	                pred = tf.reshape(predictions, shape=[-1, n_chars, n_classes])
                with tf.name_scope('correct_prediction'):
	                correct_pred = tf.equal(tf.argmax(pred,2), tf.argmax(y,2))
                with tf.name_scope('accuracy'):
                        accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
                tf.scalar_summary('accuracy',accuracy)
	        return accuracy * 100.0

def softmax_joiner(logits):      
        return tf.transpose(tf.pack([tf.nn.softmax(logits[0]), tf.nn.softmax(logits[1]), \
                                     tf.nn.softmax(logits[2]), tf.nn.softmax(logits[3]), \
                                     tf.nn.softmax(logits[4])]), perm = [1,0,2])

weights = {
	'wc1': weight_variable('wc1',[5, 5, color_channels, 64]),
	'wc2': weight_variable('wc2',[5, 5, 64, 128]),
	'wc3': weight_variable('wc3',[5, 5, 128, 256]),
	'wc4': weight_variable('wc4',[3, 3, 256, 512]),
	'wd1': weight_variable('wd1',[(resize_height/8)*(resize_width/8)*512, fc_num_outputs]),
	'out1': weight_variable('out1',[fc_num_outputs, n_classes]),
	'out2': weight_variable('out2',[fc_num_outputs, n_classes]),
	'out3': weight_variable('out3',[fc_num_outputs, n_classes]),
	'out4': weight_variable('out4',[fc_num_outputs, n_classes]),
	'out5': weight_variable('out5',[fc_num_outputs, n_classes])    
}

biases = {
	'bc1': bias_variable([64]),
	'bc2': bias_variable([128]),
	'bc3': bias_variable([256]),
        'bc4': bias_variable([512]),
	'bd1': bias_variable([fc_num_outputs]),
	'out1': bias_variable([n_classes]),
	'out2': bias_variable([n_classes]),
	'out3': bias_variable([n_classes]),
	'out4': bias_variable([n_classes]),
	'out5': bias_variable([n_classes])
}

def train():
        with tf.Session() as sess:
                logits = ocr_net(x, weights, biases, dropout_keep_prob)

                with tf.name_scope('loss'):
                        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits[0],y[:,0:36])) +\
                               tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits[1],y[:,36:72])) +\
                               tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits[2],y[:,72:108])) +\
                               tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits[3],y[:,108:144])) +\
                               tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits[4],y[:,144:180]))

                        # adding regularizers
                        regularizers = (tf.nn.l2_loss(weights['wc1']) + tf.nn.l2_loss(biases['bc1']) +
                                        tf.nn.l2_loss(weights['wc2']) + tf.nn.l2_loss(biases['bc2']) +
                                        tf.nn.l2_loss(weights['wc3']) + tf.nn.l2_loss(biases['bc3']) +
                                        tf.nn.l2_loss(weights['wc4']) + tf.nn.l2_loss(biases['bc4'])
                        )
                        # Add the regularization term to the loss.
                        loss += l2_beta_param * regularizers
                        tf.scalar_summary('loss', loss)

                global_step = tf.Variable(0)
                learning_rate = tf.train.exponential_decay(initial_learning_rate, global_step, decay_steps, decay_rate)
                with tf.name_scope('train'):
                        # optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)
                        optimizer = tf.train.MomentumOptimizer(learning_rate, momentum).minimize(loss, global_step=global_step)
                        # optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)
                pred = softmax_joiner(logits)
                accuracy = accuracy_func(pred, y)

                init = tf.initialize_all_variables()
                merged = tf.merge_all_summaries()
                train_writer = tf.train.SummaryWriter(summaries_dir + '/train', sess.graph)

	        sess.run(init)
	        step = 1# Keep training until reach max iterations
	        while step * batch_size < training_iters:
		        batch = data_train.next_batch(batch_size)
                        batch_labels = batch[1]
		        # Fit training using batch data
		        smry, _, l = sess.run([merged, optimizer, loss], feed_dict={x: batch[0], y: batch_labels})
		        if step % display_step == 0:
			        # Calculate batch accuracy
                                run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
                                run_metadata = tf.RunMetadata()
			        summary, acc = sess.run([merged,accuracy],
                                                        feed_dict={x: batch[0], y: batch[1]},
                                                        options=run_options,
                                                        run_metadata=run_metadata)
                                train_writer.add_run_metadata(run_metadata, 'step%03d' % step)
                                train_writer.add_summary(summary, step)
			        print "Iter " + str(step*batch_size) + ", Minibatch Loss= " + "{:.6f}".format(l) + ", Training Accuracy= " + "{:.2f}%".format(acc)
                        else:
                                train_writer.add_summary(smry, step)
		        step += 1
	        print "Optimization Finished!"
                train_writer.close()
                elapsed_time = time.time() - start_time
                hours = elapsed_time / 3600
                minutes = (elapsed_time % 3600) / 60
                seconds = (elapsed_time % 3600) % 60
                print "Total time was: " + "{:.0f}h".format(hours) + ", {:.0f}m".format(minutes) + ", {:.0f}s".format(seconds)
                
                test_writer = tf.train.SummaryWriter(summaries_dir + '/test')
	        test_batch = data_test.next_batch(n_test_samples)
                summ, acc = sess.run([merged,accuracy], feed_dict={x: test_batch[0], y: test_batch[1]})
	        print "Testing Accuracy: " + "{:.2f}%".format(acc)
                test_writer.add_summary(summ,step)

if tf.gfile.Exists(summaries_dir):
        tf.gfile.DeleteRecursively(summaries_dir)
tf.gfile.MakeDirs(summaries_dir)
train()
