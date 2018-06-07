# Copyright (c) Hu Zhiming JimmyHu@pku.edu.cn 2018/6/6 All Rights Reserved.

# Convert the original words to integers and save them. 

# import future libs.
from __future__ import division, print_function, absolute_import

# import libs.
import numpy as np
import string
import codecs
import collections    
import random
import glob


vocabulary_size = 89528

# valid_rate = valid_dataset/(valid_dataset+training_dataset).
valid_rate = 0.1


# Read File and Convert Words To Integers.
def readFileToConvertWordsToIntegers(dictionary, fileName, allDocuments, allLabels, label):
	# read the file.
	file = codecs.open(fileName, encoding='utf-8')
	document = []
	for line in file:
		line = line.lower()
		words = line.split()
		# delete the punctuations.
		map = str.maketrans('', '', string.punctuation)
		# transform words to integers.
		for word in words:
			word = word.translate(map)
			if word in dictionary:
				index = dictionary[word]
			else:
				index = 0  # dictionary['UNK'] 
			document.append(index)
			
	allDocuments.append(document)
	allLabels.append(label)
	file.close()



# load the vocabulary.
def LoadVocab(fileName):
	
	vocabulary = dict()
	# init the vocabulary.
	vocabulary['UNK'] = 0
	
	# open a file with utf-8 code.
	file = codecs.open(fileName, encoding='utf-8')
	
	# delete punctuations in the lines.
	# map = str.maketrans('', '', string.punctuation)
	for line in file:
		words = line.split()
		word = words[0]
		vocabulary[word] = len(vocabulary)
	
	# close the file.
	file.close()
	# return the vocabulary.
	return vocabulary


# read the vocabulary.
vocabulary = LoadVocab('imdb.vocab')
print('Vocabulary Size: ')
print(len(vocabulary))


# save the training and validation dataset.
TrainDocuments = []
TrainLabels = []
# read the training data and convert words to integers. 
fileList = glob.glob("train/neg/*.txt")
for file in fileList:
    readFileToConvertWordsToIntegers(vocabulary, file, TrainDocuments, TrainLabels, 0)

fileList = glob.glob("train/pos/*.txt")
for file in fileList:
    readFileToConvertWordsToIntegers(vocabulary, file, TrainDocuments, TrainLabels, 1)

trainData = list(zip(TrainDocuments, TrainLabels)) 
# shuffle the data.
random.shuffle(trainData)
# unzip.
TrainDocuments, TrainLabels = zip(*trainData)

# print('Total Training Data Size:')
# print(len(TrainDocuments))
# print(len(TrainLabels))

trainingSize = len(TrainDocuments)
validSize = int(valid_rate * trainingSize)

# Split train data into train data and valid data.
validX = TrainDocuments[:validSize]
trainX = TrainDocuments[validSize:]

validY = TrainLabels[:validSize]
trainY = TrainLabels[validSize:]

if len(validX)!=len(validY):
	print('Validation Dataset\'s X and Y are not matched!')
if len(trainX)!=len(trainY):
	print('Training Dataset\'s X and Y are not matched!')

print('Training Data Size:')
print(len(trainX))
print('Validation Data Size:')
print(len(validX))

# save the training and validation dataset.
np.save('trainX.npy', trainX)
np.save('trainY.npy', trainY)
np.save('validX.npy', validX)
np.save('validY.npy', validY)


# save the test dataset.
TestDocuments = []
TestLabels = []
# read the test data and convert words to integers. 
fileList = glob.glob("test/neg/*.txt")
for file in fileList:
    readFileToConvertWordsToIntegers(vocabulary, file, TestDocuments, TestLabels, 0)

fileList = glob.glob("test/pos/*.txt")
for file in fileList:
    readFileToConvertWordsToIntegers(vocabulary, file, TestDocuments, TestLabels, 1)

testX = TestDocuments
testY = TestLabels


if len(testX)!=len(testY):
	print('Test Dataset\'s X and Y are not matched!')


print('Test Data Size:')
print(len(testX))


# save the test dataset.
np.save('testX.npy', testX)
np.save('testY.npy', testY)


