# Copyright (c) Hu Zhiming JimmyHu@pku.edu.cn 2018/6/7 All Rights Reserved.
# Preprocess the IMDB dataset.


# directories & files:
test & train dir: stores the original test and train data from IMDB dataset.
imdb.vocab: the vocabulary that is provided by IMDB dataset.
Words2Integers.py: transform the words into integers and save the results.

Environment:
Python 3.6+

# Usage:

Step1:
Download the full dataset from http://ai.stanford.edu/~amaas/data/sentiment/?spm=a2c4e.11153940.blogcont221671.14.4de510813WjGn3.

Step2: 
Run Words2Integers.py and you will get trainX.npy, trainY.npy, validX.npy, validY.npy, testX.npy and testY.npy files.
Copy the .npy files into '../Data/' dir for later use.