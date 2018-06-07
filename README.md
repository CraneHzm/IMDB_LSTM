# Copyright (c) Hu Zhiming JimmyHu@pku.edu.cn 2018/6/7 All Rights Reserved.

# Sentiment Analysis: Tensorflow/TFLearn LSTM network for IMDB dataset. 


# Directories & Files:
'Data/' dir: the dir to store our datasets. 
'DataPreprocessing/' dir: the dir to save the preprocess code.
'Models/' dir: the dir to save our trained model.
'IMDB_LSTM.py': the main code.


# Environments:
Python 3.6+
tensorflow 1.7+
TFLearn


# Usage:

Step 1: Check the 'Data/' directory to confirm whether the dataset files(.npy files) exist. 
If not, run the codes in 'DataPreprocessing/' to create the dataset files.
UnZip the Models.7z file to use our model.

Step 2: Run 'IMDB_LSTM.py' to test the model.
The Accuracy of our model on the Test Dataset is 82.02%.
