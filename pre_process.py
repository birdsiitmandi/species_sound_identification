import numpy as np
import pickle
import os

def get_one_hot_encoding(name, n_classes):
	"""
	Get array from c01_c07_c03 etc.
	[0,6,2]
	use this to make one hot array.
	if there are 10 classes
	one hot array would be [1,0,1,0,0,0,1,0,0,0]
	"""
	actual_class_temp = name.split('_')
	classes_present = []
	for j in actual_class_temp:
		classes_present.append(int(j[1:])-1)
	one_hot_array = np.zeros(n_classes)
	for i in classes_present:
		one_hot_array[i] = 1
	return one_hot_array
	
cl = 0.
sum = 0
classes = np.loadtxt("classes.txt", dtype='str')
n_classes = classes.size
"""
classes.txt has all the class names.
As you may have noticed the folder names of data are named as c01_c02 etc
This is done because it's harder to deal with full bird names.

Make sure there is correspondence between the new names i.e c01, c02 and the full bird names.

blackandyellow_groasbeak c01
blackcrested_tit c02
black_throated_tit c03
etc

And the order in classes.txt should be same as c01, c02 and so on.

"""
f_handle = open('./data/multi_train_processed_temp.dat', 'ab') #opening a temp dat file

#walking through folders in data
for root, dirs, files in os.walk("./data/melfilter48/multi_train", topdown=False):
		for name in dirs:
			parts = []
			parts += [each for each in os.listdir(os.path.join(root,name)) if each.endswith('.mfcc')]
			print(name, "...")
			one_hot_encoding = get_one_hot_encoding(name, n_classes) #getting all the one hot encoding
			print(one_hot_encoding)
			for part in parts:
				example = np.loadtxt(os.path.join(root,name,part))
				i = 0
				rows = example.shape[0]
				while i <= (rows - 15):
					"""
					Taking 15 rows from the mfcc file and flattening it to form
					a context vector. Doing this for every row.
					For n*39 vector mfcc, you will get 585 (39X15) size context vector.
					For n*40 mel vector, you will get 600 (40X15) size context vector.
					Append all these context vectors, column wise to one array.
					"""
					context = example[i:i+15,:].ravel() 
					ex = np.append(context,one_hot_encoding) #appending the one hot encoding
					ex = np.reshape(ex,(1,ex.shape[0]))
					np.savetxt(f_handle, ex)
					sum += 1
					i += 1
				print("No. of context windows: %d" % i)
			cl += 1
print("No. of Training examples: %d" % sum)
f_handle.close()

A = np.loadtxt('./data/multi_train_processed_temp.dat') # saving the data.
np.random.shuffle(A) #shuffle the data, shuffling is done to elminate the any bias towards last classes.
np.savetxt('./data/multi_train_processed.dat',A,delimiter = ' ') #save the shuffled data




