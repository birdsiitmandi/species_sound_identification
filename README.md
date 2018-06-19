# DNNbasedMultiLabelspeciessoundidentificationusingSignalProcessing
DNN based MultiLabel species sound identification using Signal Processing
# Multi Label classification of bird songs


##Setting up

### Getting started with deep learning

Stanford Lectures on Deep learning (CS231n): [link](https://www.youtube.com/watch?v=vT1JzLTH4G4&list=PLC1qU-LWwrF64f4QKQT-Vg5Wr4qEE1Zxk)
Andrew Ng lectures: [link](https://www.youtube.com/playlist?list=PLkDaE6sCZn6Gl29AoE31iwdVwSG-KnDzF) 
Rabiner Speech recognition: [link](https://www.ece.ucsb.edu/Faculty/Rabiner/ece259/)

### Installing Anaconda

download Anaconda-latest-Linux-x86_64.sh from 
[Here](https://www.google.com "Anaconda install")

Install this file on server using 

```
bash Anaconda-latest-Linux-x86_64.sh
```

Make sure that "export PATH="/home/b14138/anaconda3/bin:$PATH"
is added in bashrc file. Use "vim ~/.bashrc" to edit the file. 
Anaconda usually does it for you.

Check installation by

```
conda --version
```

### Creating virtual env

To create:
```
conda create --name yourenvname
```
To activate:
```
source activate yourenvname
```
To deactivate:
```
source deactivate
```
To remove:
```
conda remove -n yourenvname -all
```
## Code Explanation.

There are three parts to the code
1. Pre-processing (pre_process.py)
2. Training our network (train_layer_1.py & train_layer_2.py)
3. Testing our network (classify.py)

Make sure you have "classes.txt" which contains the class names. The order is important as the order of output probability array you get in the end corresponds to the order in which the class names were saved in the "classes.txt" file. 

Make sure that your train data is kept in "./data/melfilter48/multi_train" folder. And your test data in "./data/melfilter48/multi_test"

The classes are named as c01, c02, c03 for convenience. The order is the same as it is in "classes.txt" file.

MFCC features are extracted using feature_extract.m


### Pre-processing the data.

Input: All MFCC files in the train data
Ouput: multi_train_processed.dat

Let's say a MFCC file is of N*39 dimension, pre_process.py takes this file and applies a 15 frame context window on this array with stride=1. That means it takes every 15 frames and flattens it. As a frame is of length 39, you get a 585 (15*39) length context window. Every context window is concatenated with it's multi hot notation. All the context windows from every training MFCC file are attached into an array and saved into "multi_train_processed_temp.dat" file. This is again loaded, and a final shuffled array is produced with the name "multi_train_processed.dat". Shuffling is done to elminate any biases towards last classes.


### Training the network.

Input: multi_train_processed.dat
Ouput: parameters_mfcc_1.pkl, parameters_mfcc_2.pkl	

There two files: "train_layer_1.py" and "train_layer_2.py". "train_layer_1.py" trains network of one hidden layer and outputs its weights into "parameters_mfcc_1.pkl" file. "train_layer_2.py" uses the weights trained by "train_layer_1.py" and trains network of two hidden layers. Weights of the newly trained two layers are saved into "parameters_mfcc_2.pkl" file.

### Testing the network

Input: Test data, parameters_mfcc_2.pkl
Ouput: Metrics

Testing is done by classify.py file. There are two parts in this file, either you can test a single file by providing the path as a sys arg or you can get metrics for your test data. Just like training a context matrix is formed by attaching every 15 frames from an MFCC file. This is then fed into the neural network and you get probability array of length n_classes. Multiplying a threshold factor (=0.5) to the max in the array would give you a threshold value. Using this threshold you can predict the multiple classes present the given test file. For metrics refer to our Final report. 


## How to run the code
After making sure the train and test data are in right folders. 
To pre process the data:
```
python3 pre_process.py
```
To train the first layer:
```
python3 train_layer_1.py
```
To train the second layer:
```
python3 train_layer_2.py
```
To test:
```
python3 train_layer_1.py
```






