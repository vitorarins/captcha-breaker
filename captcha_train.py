import tensorflow as tf
from captcha_data import OCR_data
# Parameters
learning_rate = 0.001
training_iters = 200000
batch_size = 100
display_step = 20

# Network Parameters
width = 180
resize_width = 88
height = 60
resize_height = 24
color_channels = 1
n_chars = 5
# n_input = 7200  # 60*180*3
n_classes = 36  # 10+26
n_training_samples = 180000
n_test_samples = 100
fc_num_outputs = 4096

data_train = OCR_data(n_training_samples, './images/train', n_classes)
data_test = OCR_data(n_test_samples, './images/test', n_classes)

# tf Graph input
x = tf.placeholder(tf.float32, [None, resize_height, resize_width])
y = tf.placeholder(tf.float32, [None, n_chars*n_classes])

def print_activations(t):
	print(t.op.name, t.get_shape().as_list())

def weight_variable(name,shape):
        return tf.get_variable(name, shape, initializer=tf.contrib.layers.xavier_initializer())

def bias_variable(shape):
	initial = tf.constant(0.0, shape=shape)
	return tf.Variable(initial, trainable=True)

def conv2d(x, W, B, name):
	with tf.name_scope(name) as scope:
		conv = tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')
		bias = tf.nn.bias_add(conv, B)
		conv = tf.nn.relu(bias, name=scope)
		return conv

def max_pool(x, k, name):
	return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1], padding='SAME', name=name)

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

def ocr_net(_x, _weights, _biases):
	_x = tf.reshape(_x, shape=[-1, resize_height, resize_width, color_channels])

	conv1 = conv2d(_x, _weights['wc1'], _biases['bc1'], 'conv1')
	print_activations(conv1)
	pool1 = max_pool(conv1, k=2, name='pool1')
	print_activations(pool1)

	conv2 = conv2d(pool1, _weights['wc2'], _biases['bc2'], 'conv2')
	print_activations(conv2)
	pool2 = max_pool(conv2, k=2, name='pool2')
	print_activations(pool2)

	conv3 = conv2d(pool2, _weights['wc3'], _biases['bc3'], 'conv3')
	print_activations(conv3)
	pool3 = max_pool(conv3, k=2, name='pool3')
	print_activations(pool3)

	conv4 = conv2d(pool3, _weights['wc4'], _biases['bc4'], 'conv4')
	print_activations(conv4)

	conv4_flat = tf.reshape(conv4, [-1, _weights['wd1'].get_shape().as_list()[0]])
	fc1 = tf.nn.relu(tf.matmul(conv4_flat, _weights['wd1']) + _biases['bd1'], name='fc1')
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

	out = tf.concat(1, [fc21, fc22, fc23, fc24, fc25], name='out')
	print_activations(out)
	return out

def accuracy_func(_pred, _y):
	y = tf.reshape(_y, shape=[-1, n_chars, n_classes])
	pred = tf.reshape(_pred, shape=[-1, n_chars, n_classes])
	correct_pred = tf.equal(tf.argmax(pred,2), tf.argmax(y,2))
	accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
	return accuracy

pred = ocr_net(x, weights, biases)

cross_entropy = tf.nn.softmax_cross_entropy_with_logits(pred,y)
cost = tf.reduce_mean(cross_entropy)

optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

accuracy = accuracy_func(pred, y)

init = tf.initialize_all_variables()

with tf.Session() as sess:
	sess.run(init)
	step = 1# Keep training until reach max iterations
	while step * batch_size < training_iters:
		batch = data_train.next_batch(batch_size)
		# Fit training using batch data
		sess.run(optimizer, feed_dict={x: batch[0], y: batch[1]})
		if step % display_step == 0:
			# Calculate batch accuracy
			acc = sess.run(accuracy, feed_dict={x: batch[0], y: batch[1]})
			# Calculate batch loss
			loss = sess.run(cost, feed_dict={x: batch[0], y: batch[1]})
			print "Iter " + str(step*batch_size) + ", Minibatch Loss= " + "{:.6f}".format(loss) + ", Training Accuracy= " + "{:.5f}".format(acc)
		step += 1
	print "Optimization Finished!"

	test_batch = data_test.next_batch(n_test_samples)
	print "Testing Accuracy:", sess.run(accuracy, feed_dict={x: test_batch[0], y: test_batch[1]})
