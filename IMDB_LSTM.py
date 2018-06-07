# Copyright (c) Hu Zhiming JimmyHu@pku.edu.cn 2018/6/7 All Rights Reserved.

# the LSTM model for IMDB dataset.



# import future libs.
from __future__ import division, print_function, absolute_import


# import libs.
import tflearn
from tflearn.data_utils import to_categorical, pad_sequences
import string
import codecs
import collections    
import random
import glob
import numpy as np

#######################################
# Usage: 
# Accuracy of the model on test set is 82.20%.
# Run this code to test the accuracy of our model.
# If you want to retrain the model, set trainingMode to True to train your own model based on the pre-trained model.
#######################################


# If use training mode
trainingMode = False

# the size of our vocabulary.
VocabularySize = 89528
# the max length of an example, i.e., the max number of integers/words of an example.
MaxLen = 200
# the number of classes to classify.
NumClasses = 2
# the length of a word embedding.
EmbeddingLen = 128
# the number of Lstm units.
LstmUnits = 128
# Lstm layer's dropout rate.
LstmDropoutRate = 0.6
# Learning Rate
LearningRate = 0.001
# training epochs.
Epochs = 40
# the Batch Size.
BatchSize = 100
# the number of steps that we snapshot the models.
snapshot_step = 0
# Show Metric or Not
ShowMetric = True
#  The minimum validation accuracy that needs to be achieved before a model weight's are saved to the best_checkpoint_path.
BestValAccuracy = 0.82



# load the training, validation and test datasets.
trainX = np.load('Data/trainX.npy')
trainY = np.load('Data/trainY.npy')
if len(trainX)!=len(trainY):
	print('\n\nTraining Dataset\'s X and Y are not matched!\n\n')
print('\n\nTraining Data Size:')
print(len(trainX))

validX = np.load('Data/validX.npy')
validY = np.load('Data/validY.npy')
if len(validX)!=len(validY):
	print('\n\nValidation Dataset\'s X and Y are not matched!\n\n')
print('\n\nValidation Data Size:')
print(len(validX))

testX = np.load('Data/testX.npy')
testY = np.load('Data/testY.npy')
if len(testX)!=len(testY):
	print('\n\nTest Dataset\'s X and Y are not matched!\n\n')
print('\n\nTest Data Size:')
print(len(testX))


# Prepare the X and Y data for LSTM.
trainX = pad_sequences(trainX, maxlen = MaxLen, value=0.)
validX = pad_sequences(validX, maxlen = MaxLen, value=0.)
testX = pad_sequences(testX, maxlen = MaxLen, value=0.)
# Converting labels to binary vectors
trainY = to_categorical(trainY, nb_classes = NumClasses)
validY = to_categorical(validY, nb_classes = NumClasses)
testY = to_categorical(testY, nb_classes = NumClasses)


# Network building
net = tflearn.input_data([None, MaxLen])
net = tflearn.embedding(net, input_dim = VocabularySize, output_dim = EmbeddingLen)
net = tflearn.lstm(net, LstmUnits, dropout = LstmDropoutRate)
net = tflearn.fully_connected(net, NumClasses, activation='softmax')
net = tflearn.regression(net, optimizer='adam', learning_rate = LearningRate,
                         loss='categorical_crossentropy')

# Build the model.
model = tflearn.DNN(net, best_checkpoint_path = 'Models/', best_val_accuracy = BestValAccuracy,max_checkpoints = 2, tensorboard_verbose=0)

# Load pre-trained model.
model.load('Models/8628')

# train the model.

if trainingMode:
	model.fit(trainX, trainY, n_epoch= Epochs, validation_set=(validX, validY), show_metric = ShowMetric,
			batch_size = BatchSize, snapshot_step = snapshot_step)

# save the model.		
# model.save('IMDBLstm.tfl')		  

# evaluate the model.
evaluation = model.evaluate(testX, testY)
# Calculate the accuracy on Test Dataset.
accuracy = evaluation[0]*100
print('\n\nTest Set accuracy: {:0.2f}%\n\n'.format(accuracy))	
