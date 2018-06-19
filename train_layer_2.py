
#!/usr/local/bin/python

#########################################################################
#   							#
# Author: Deep Chakraborty												#
#
# 		    		   													#
#########################################################################
import numpy as np
import tensorflow as tf

import pickle


"""
Everything is same as train_layer_1.py
This time we are taking the weights from parameters_mfcc_1.pkl 
which were created by layer 1.
Using them to train our next layer.
"""

path = "./data/"
classes = np.loadtxt("classes.txt", dtype='str')
n_classes = classes.size


train = np.loadtxt(path + 'multi_train_processed.dat')
train_X = np.copy(train[:,:(-1*n_classes)]) 
train_labels_dense = np.copy(train[:,(-1*n_classes):]) 
train_labels_dense = train_labels_dense.astype(int)
train_y = train_labels_dense


print("Data Loaded and processed ...")
################## Neural Networks Training #################################

print("Selecting Neural Network Parameters ...")
# Parameters
training_epochs = 100
batch_size = 64
total_batch = int(train_X.shape[0]/batch_size)
display_step = 5

starter_learning_rate = 0.003
global_step = tf.Variable(0, trainable = False)
learning_rate = tf.train.exponential_decay(starter_learning_rate, global_step, 100 * total_batch, 0.9, staircase = True)

# Network Parameters
n_hidden_1 = 256 # 1st layer num features
n_hidden_2 = 256 # 2nd layer num features
# n_hidden_3 = 256
n_input = 585 # input dimensionality

x = tf.placeholder("float", [None, n_input])
y = tf.placeholder("float", [None, n_classes])

# Create model
def multilayer_perceptron(_X, _weights, _biases):
	layer_1 = tf.nn.relu(tf.add(tf.matmul(_X, _weights['h1']), _biases['b1']))
	layer_2 = tf.nn.relu(tf.add(tf.matmul(layer_1, _weights['h2']), _biases['b2']))
	return tf.matmul(layer_2, _weights['out']) + _biases['out']

print("Loading saved Weights for layer 1 ...") #loadin weights of layer 1
file_ID = "./data/parameters_mfcc_1.pkl"
f = open(file_ID, "rb")
W = pickle.load(f)
b = pickle.load(f)
f.close()

weights = {
	'h1': tf.Variable(W['h1']), #insted of intializing them randomly we are using the existing weights.
	'h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
	'out': tf.Variable(tf.random_normal([n_hidden_2, n_classes]))
}
biases = {
	'b1': tf.Variable(b['b1']),
	'b2': tf.Variable(tf.random_normal([n_hidden_2])),
	'out': tf.Variable(tf.random_normal([n_classes]))
}


pred = multilayer_perceptron(x, weights, biases)


#cost function, sigmoid_cross_entroy_with_logits
cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits = pred, labels=y))
# Adam Optimizer
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost, global_step = global_step)

# Create a summary to monitor cost function
tf.contrib.deprecated.scalar_summary("loss", cost)

# Merge all summaries to a single operator
merged_summary_op = tf.summary.merge_all()

print("Training the Neural Network")
init = tf.initialize_all_variables()

with tf.Session() as sess:
	sess.run(init)

	# Set logs writer into folder /tmp/tensorflow_logs
	summary_writer = tf.summary.FileWriter('home/v_vivek/tmp/tensorflow_logs', graph_def=sess.graph_def)
	# Training cycle
	avg_cost = 100.
	epoch = 0
	while avg_cost > 0.00001 and epoch < training_epochs:
	# for epoch in range(training_epochs):
		avg_cost = 0.
		# Loop over all batches
		for i in range(total_batch):
			batch_xs, batch_ys = train_X[i*batch_size:(i+1)*batch_size,:], train_y[i*batch_size:(i+1)*batch_size,:]
			# Fit training using batch data
			sess.run(optimizer, feed_dict={x: batch_xs, y: batch_ys})
			# Compute average loss
			avg_cost += sess.run(cost, feed_dict={x: batch_xs, y: batch_ys})/total_batch
			# Write logs at every iteration
			summary_str = sess.run(merged_summary_op, feed_dict={x: batch_xs, y: batch_ys})
			summary_writer.add_summary(summary_str, epoch*total_batch + i)


		# Display logs per epoch step
		if epoch % display_step == 0:
			print("Epoch:", '%04d' % (epoch+1), "cost=", "{:.9f}".format(avg_cost), "learning rate= %.4f" % sess.run(learning_rate), "step= %d" % sess.run(global_step))

		epoch += 1

	print("Optimization Finished!")
	W = {
	'h1': sess.run(weights['h1']),
	'h2': sess.run(weights['h2']),
	'out': sess.run(weights['out'])
	}

	b = {
	'b1': sess.run(biases['b1']),
	'b2': sess.run(biases['b2']),
	'out': sess.run(biases['out'])
	}


	file_ID = path + "parameters_mfcc_2.pkl"
	f = open(file_ID, "wb")
	pickle.dump(W, f, protocol=pickle.HIGHEST_PROTOCOL)
	pickle.dump(b, f, protocol=pickle.HIGHEST_PROTOCOL)
	f.close()

