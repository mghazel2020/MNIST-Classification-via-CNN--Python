# MNIST Hand-written Digits Classification using Convolutional Neural Networks (CNN)

<img src="images/banner.png" width = 500/>

## 1. Objective

The objective of this section is to develop a Convolutional Neural Network (CNN) to classify hand-written digits using the widely used MNIST data set.

## 2. Motivation
The MNIST handwritten digit classification problem is a standard dataset used in computer vision and deep learning.

Although the dataset is effectively solved, it can be used as the basis for learning and practicing how to develop, evaluate, and use convolutional neural networks for image classification from scratch. 

In this section, we shall demonstrate how to develop convolutional neural network for handwritten digit classification from scratch, including:

* How to prepare the input training and test data 
* How to deploy the model
* How to use the trained model to make predictions
* How to evaluate its performance

## 3. Data

The MNIST database of handwritten digits, is widely used for training and evaluating various supervised machine and deep learning models [1]:

* It has a training set of 60,000 examples
* It has test set of 10,000 examples
* It is a subset of a larger set available from NIST. 
* The digits have been size-normalized and centered in a fixed-size image.
* It is a good database for people who want to try learning techniques and pattern recognition methods on real-world data while spending minimal efforts on preprocessing and formatting.
* The original black and white images from NIST were size normalized and resized to 28x28 binary images.

Sample images from the MNIST data set are illustrated next:
* There are significant variations how digits are handwritten by different people
* The same digit may be written quite differently by different people
* More significantly, different handwritten digits may appear similar, such as the 0, 5 and 6 or the 7 and 9.


