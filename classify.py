"@author:V_Vivek"
import sys
import numpy as np
import tensorflow as tf
import os
import pickle
from scipy.stats import mode

################    All Constants and paths used    #####################
path = "./data/"
classes = np.loadtxt("classes.txt", dtype='str')
n_classes = classes.size
parametersFileDir = "./data/parameters_mfcc_2.pkl"
relativePathForTest = "./data/melfilter48/multi_test/"
testFilesExtension = '.mfcc'
confMatFileDirectory = './data/confusion_multi.txt'
classifiedTestDirectory = './testOutput/'


################## Neural Networks Training #################################

print("Verifying Neural Network Parameters ...")

# Network Parameters
n_hidden_1 = 256 # 1st layer num features
n_hidden_2 = 256 # 2nd layer num features
n_input = 585 # input dimensionality

x = tf.placeholder("float32", [None, n_input])
y = tf.placeholder("float32", [None, n_classes])

# Create model
def multilayer_perceptron(_X, _weights, _biases):
    #Hidden layer with RELU activation
	layer_1 = tf.nn.relu(tf.add(tf.matmul(_X, _weights['h1']), _biases['b1']))
    #Hidden layer with sigmoid activation
	layer_2 = tf.nn.relu(tf.add(tf.matmul(layer_1, _weights['h2']), _biases['b2']))
	#layer_3 = tf.nn.relu(tf.add(tf.matmul(layer_2, _weights['h3']), _biases['b3']))
	#layer_4 = tf.nn.relu(tf.add(tf.matmul(layer_3, _weights['h4']), _biases['b4']))
	#layer_5 = tf.nn.sigmoid(tf.add(tf.matmul(layer_4, _weights['h5']), _biases['b5']))
	#return tf.nn.softmax(tf.matmul(layer_2, _weights['out']) + _biases['out'])
	return tf.nn.sigmoid(tf.matmul(layer_2, _weights['out']) + _biases['out'])

print("Loading saved Weights ...")
file_ID = parametersFileDir
f = open(file_ID, "rb")
W = pickle.load(f)
b = pickle.load(f)


weights = {
	'h1': tf.Variable(W['h1']),
	'h2': tf.Variable(W['h2']),
	#'h3': tf.Variable(W['h3']),
	#'h4': tf.Variable(W['h4']),
	#'h5': tf.Variable(W['h5']),
	'out': tf.Variable(W['out'])
	}

biases = {
	'b1': tf.Variable(b['b1']),
	'b2': tf.Variable(b['b2']),
	#'b3': tf.Variable(b['b3']),
	#'b4': tf.Variable(b['b4']),
	#'b5': tf.Variable(b['b5']),
	'out': tf.Variable(b['out'])
}

f.close()


pred = multilayer_perceptron(x, weights, biases)

print("Testing the Neural Network")
init = tf.initialize_all_variables()

with tf.Session() as sess:
	sess.run(init)
	# Test model
	print("Computing...")
	if len(sys.argv) == 1:
		for root, dirs, files in os.walk(relativePathForTest, topdown=False):
			list_actual_class = [] #intializing list of actual classes
			list_predicted_class = [] #intializing list of predicted classes
			for name in dirs:
				parts = []
				parts += [each for each in os.listdir(os.path.join(root,name)) if each.endswith(testFilesExtension)]
				directory = os.path.dirname(classifiedTestDirectory)
				if not os.path.exists(directory):
					os.makedirs(directory)
				for part in parts:
					#print("Part : ",part)
					example = np.loadtxt(os.path.join(root,name,part)) #loading the mfcc test file
					i = 0
					rows, cols = example.shape
					context = np.zeros((rows-14,15*cols)) # 15 contextual frames
					part_mat = np.zeros((rows-14,10))
					see = tf.reduce_sum(pred,0)
					while i <= (rows - 15):
						ex = example[i:i+15,:].ravel() # flattening the context vector
						ex = np.reshape(ex,(1,ex.shape[0]))
						part_mat[i:i+1,:] = sess.run(pred, feed_dict={x:ex})
						context[i:i+1,:] = ex # appending the context vector to an array
						i += 1
					product = np.argmax(np.asarray(see.eval({x: context}))) # feeding the context to neural network
																			# getting the argmax of the output array
					product_array = (np.asarray(see.eval({x: context}))) # getting the output array
					max_product_array = np.amax(np.asarray(see.eval({x: context}))) # getting the max value of output array
					threshold_factor = 0.5 # setting a threshold factor
					threshold = max_product_array*threshold_factor # multipling the max value of an array with threshold factor
					"""
					All the classes that have greater value that the threshold are predicted
					"""
					predicted_class = []
					for j in range(n_classes):
						if(product_array[j] >= threshold):
							predicted_class.append(j+1)
					print("predicted class", predicted_class)
					
					actual_class_temp = name.split('_')
					actual_class = []
					for j in actual_class_temp:
						actual_class.append(int(j[1:]))
					print("actual class", actual_class)
					list_actual_class.append(actual_class)
					list_predicted_class.append(predicted_class)
		accuracy_multi = 0
		recall = 0
		precision = 0
		hamming_loss = 0
		"""
		Metrics are calculated using IoU method
		if the predicted class = [1, 2, 3]
		and actual class is = [2, 3, 5]
		then intersection = [2, 3]
		Union = [1, 2, 3, 5]
		IoU = len(intersection)/len(Union)
			= 2/4 = 1/2
		Accuracy is calculated by adding IoUs of all the examples.

		"""
		for j in range(len(list_actual_class)):
			intersection = len(set(list_actual_class[j]) & set(list_predicted_class[j]))
			union = len(set(list_actual_class[j]) | set(list_predicted_class[j]))
			accuracy_multi += (intersection/union)
			if len(list_actual_class) > 0:
				recall += intersection/len(list_actual_class[j])
			if len(list_predicted_class) > 0:
				precision += intersection/len(list_predicted_class[j])
			hamming_loss += (union - intersection)/n_classes
		accuracy_multi = (accuracy_multi/len(list_actual_class))*100
		recall = (recall/len(list_actual_class))
		precision = (precision/len(list_actual_class))
		hamming_loss = (hamming_loss/len(list_actual_class))
		print("~ Results ~")
		print("Accuracy is %.4f " % accuracy_multi)
		print("Recall is %.4f " % recall)
		print("Precision is %.4f " % precision)
		print("Hamming Loss is %.4f " % hamming_loss)
	else:
		path = sys.argv[1] # for a single file if the path is given as system argument
		example = np.loadtxt(path)
		i = 0
		rows, cols = example.shape
		context = np.zeros((rows-14,15*cols)) 
		part_mat = np.zeros((rows-14,10))
		see = tf.reduce_sum(pred,0)
		while i <= (rows - 15):
			ex = example[i:i+15,:].ravel()
			ex = np.reshape(ex,(1,ex.shape[0]))
			part_mat[i:i+1,:] = sess.run(pred, feed_dict={x:ex})
			context[i:i+1,:] = ex
			i += 1
		product = np.argmax(np.asarray(see.eval({x: context})))
		product_count = (np.asarray(see.eval({x: context})))
		max_product_count = np.amax(np.asarray(see.eval({x: context})))
		threshold_factor = 0.5
		avg_classes = 2
		threshold = max_product_count*threshold_factor;
		predicted_class = []
		for j in range(n_classes):
			if(product_count[j] >= threshold):
				predicted_class.append(j+1)
		print("predicted classes", predicted_class)
		print("Names:")
		for k in predicted_class:
			print(classes[k-1])
		

		

	
