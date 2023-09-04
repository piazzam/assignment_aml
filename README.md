# assignment_aml

Collection of some assignment developed for my master's degree course: Advanced Machine Learning. The course deals with deep learning both in theory and practice terms.

All the code is developed in Python with Keras library.

## First assignment
The assignment consists of the "default payments" prediction using a neural network. This assignment is developed in the form of Jupyter notebook, where there is the source code and some comments on the choices.

The provided dataset contains information on default payments (y_train), demographic factors, credit data, history of payment, and bill statements of credit card clients (X_train, X_test) in Taiwan from April 2005 to September 2005.

The dataset is contained ih the "dataset" folder. You will find a description of the featured dataset in the "Data Dictionary" attached pdf.
The provided data comprises the training set that can be used for the training (and for the validation) and the unlabelled test set.

## Second assignment
The assignment consists on the prediction of grayscale images of letters P - Z.
This assignment is divided in three part: 
1. The implementation of a traditional Neural Network to predict the letters.
2. The implementation of an auto-encoder to create the images.
3. Using the encoded part predict letters with a traditional classifier. 
The provided data comprises the training set that can be used for the training (and for the validation) and the unlabelled balanced test set.
This assignment is implemented in the form of Python script and there is a relation that explain the work.

## Third assignment
The task of the assignment 3 is the design of a CNN architecture and its training.

Input dataset: MNIST digits (input size 28x28x1, number of classes: 10).
The dataset is not distributed since can be easily downloaded directly from Keras.

The CNN has to be designed with the aim of reaching the maximum possible accuracy on the test set, with the hard constraint of a maximum of 6K learnable parameters. Refer to the code developed in today's class as a skeleton on which to build your solution.

## Fourth assignment
The task of the assignment #4 is Transfer Learning using a CNN pretrained on IMAGENET.

The suggested architecture is the VGG16.

The CNN should be used as fixed feature extractor on a new task of your choice containing a number of classes in the range from 2 to 10. 
