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

<img src="images/MNIST-sample-images.png" width = 1000/>

## 4. Development

In this section, we shall demonstrate how to develop a Convolutional Neural Network (CNN) for handwritten digit classification from scratch, including:

* How to prepare the input training and test data 
* How to deploy the model
* How to use the trained model to make predictions
* How to evaluate its performance

* Author: Mohsen Ghazel (mghazel)
* Date: April 5th, 2021
* Project: MNIST Handwritten Digits Classification using Convolutional Neural Networks (CNN):


The objective of this project is to demonstrate how to develop a convolutional neural network to classify images of hand-written digits, from 0-9:

We shall apply the standard Machine and Deep Learning model development and evaluation process, with the following steps:

* Load the MNIST dataset of handwritten digits:
  * 60,000 labelled training examples
  * 10,000 labelled test examples
  * Each handwritten example is 28x28 pixels binary image.
* Build a simple CNN model
* Train the selected ML model
* Deploy the trained on the test data
* Evaluate the performance of the trained model using evaluation metrics:
  * Accuracy
  * Confusion Matrix
  * Other metrics derived form the confusion matrix.

### 4.1. Part 1: Imports and global variables
#### 4.1.1. Standard scientific Python imports:


<pre style="color:#000000;background:#e6ffff;font-size:10px;line-height:1.5;"><span style="color:#696969; "># numpy</span>
<span style="color:#800000; font-weight:bold; ">import</span> numpy <span style="color:#800000; font-weight:bold; ">as</span> np
<span style="color:#696969; "># matplotlib</span>
<span style="color:#800000; font-weight:bold; ">import</span> matplotlib<span style="color:#808030; ">.</span>pyplot <span style="color:#800000; font-weight:bold; ">as</span> plt
<span style="color:#696969; "># - import sklearn to use the confusion matrix function</span>
<span style="color:#800000; font-weight:bold; ">from</span> sklearn<span style="color:#808030; ">.</span>metrics <span style="color:#800000; font-weight:bold; ">import</span> confusion_matrix
<span style="color:#696969; "># import itertools</span>
<span style="color:#800000; font-weight:bold; ">import</span> itertools
<span style="color:#696969; "># opencv</span>
<span style="color:#800000; font-weight:bold; ">import</span> cv2
<span style="color:#696969; "># tensorflow</span>
<span style="color:#800000; font-weight:bold; ">import</span> tensorflow <span style="color:#800000; font-weight:bold; ">as</span> tf

<span style="color:#696969; "># keras input layer</span>
<span style="color:#800000; font-weight:bold; ">from</span> tensorflow<span style="color:#808030; ">.</span>keras<span style="color:#808030; ">.</span>layers <span style="color:#800000; font-weight:bold; ">import</span> <span style="color:#400000; ">Input</span>
<span style="color:#696969; "># keras conv2D layer</span>
<span style="color:#800000; font-weight:bold; ">from</span> tensorflow<span style="color:#808030; ">.</span>keras<span style="color:#808030; ">.</span>layers <span style="color:#800000; font-weight:bold; ">import</span> Conv2D
<span style="color:#696969; "># keras MaxPooling2D layer</span>
<span style="color:#800000; font-weight:bold; ">from</span> tensorflow<span style="color:#808030; ">.</span>keras<span style="color:#808030; ">.</span>layers <span style="color:#800000; font-weight:bold; ">import</span> MaxPooling2D
<span style="color:#696969; "># keras Dense layer</span>
<span style="color:#800000; font-weight:bold; ">from</span> tensorflow<span style="color:#808030; ">.</span>keras<span style="color:#808030; ">.</span>layers <span style="color:#800000; font-weight:bold; ">import</span> Dense
<span style="color:#696969; "># keras Flatten layer</span>
<span style="color:#800000; font-weight:bold; ">from</span> tensorflow<span style="color:#808030; ">.</span>keras<span style="color:#808030; ">.</span>layers <span style="color:#800000; font-weight:bold; ">import</span> Flatten
<span style="color:#696969; "># keras Dropout layer</span>
<span style="color:#800000; font-weight:bold; ">from</span> tensorflow<span style="color:#808030; ">.</span>keras<span style="color:#808030; ">.</span>layers <span style="color:#800000; font-weight:bold; ">import</span> Dropout
<span style="color:#696969; "># keras model</span>
<span style="color:#800000; font-weight:bold; ">from</span> tensorflow<span style="color:#808030; ">.</span>keras<span style="color:#808030; ">.</span>models <span style="color:#800000; font-weight:bold; ">import</span> Model
<span style="color:#696969; "># keras sequential model</span>
<span style="color:#800000; font-weight:bold; ">from</span> tensorflow<span style="color:#808030; ">.</span>keras<span style="color:#808030; ">.</span>models <span style="color:#800000; font-weight:bold; ">import</span> Sequential
<span style="color:#696969; "># optimizers</span>
<span style="color:#800000; font-weight:bold; ">from</span> tensorflow<span style="color:#808030; ">.</span>keras<span style="color:#808030; ">.</span>optimizers <span style="color:#800000; font-weight:bold; ">import</span> SGD

<span style="color:#696969; "># random number generators values</span>
<span style="color:#696969; "># seed for reproducing the random number generation</span>
<span style="color:#800000; font-weight:bold; ">from</span> random <span style="color:#800000; font-weight:bold; ">import</span> seed
<span style="color:#696969; "># random integers: I(0,M)</span>
<span style="color:#800000; font-weight:bold; ">from</span> random <span style="color:#800000; font-weight:bold; ">import</span> randint
<span style="color:#696969; "># random standard unform: U(0,1)</span>
<span style="color:#800000; font-weight:bold; ">from</span> random <span style="color:#800000; font-weight:bold; ">import</span> random
<span style="color:#696969; "># time</span>
<span style="color:#800000; font-weight:bold; ">import</span> datetime
<span style="color:#696969; "># I/O</span>
<span style="color:#800000; font-weight:bold; ">import</span> os
<span style="color:#696969; "># sys</span>
<span style="color:#800000; font-weight:bold; ">import</span> sys

<span style="color:#696969; "># check for successful package imports and versions</span>
<span style="color:#696969; "># python</span>
<span style="color:#800000; font-weight:bold; ">print</span><span style="color:#808030; ">(</span><span style="color:#0000e6; ">"Python version : {0} "</span><span style="color:#808030; ">.</span>format<span style="color:#808030; ">(</span>sys<span style="color:#808030; ">.</span>version<span style="color:#808030; ">)</span><span style="color:#808030; ">)</span>
<span style="color:#696969; "># OpenCV</span>
<span style="color:#800000; font-weight:bold; ">print</span><span style="color:#808030; ">(</span><span style="color:#0000e6; ">"OpenCV version : {0} "</span><span style="color:#808030; ">.</span>format<span style="color:#808030; ">(</span>cv2<span style="color:#808030; ">.</span>__version__<span style="color:#808030; ">)</span><span style="color:#808030; ">)</span>
<span style="color:#696969; "># numpy</span>
<span style="color:#800000; font-weight:bold; ">print</span><span style="color:#808030; ">(</span><span style="color:#0000e6; ">"Numpy version  : {0}"</span><span style="color:#808030; ">.</span>format<span style="color:#808030; ">(</span>np<span style="color:#808030; ">.</span>__version__<span style="color:#808030; ">)</span><span style="color:#808030; ">)</span>
<span style="color:#696969; "># tensorflow</span>
<span style="color:#800000; font-weight:bold; ">print</span><span style="color:#808030; ">(</span><span style="color:#0000e6; ">"Tensorflow version  : {0}"</span><span style="color:#808030; ">.</span>format<span style="color:#808030; ">(</span>tf<span style="color:#808030; ">.</span>__version__<span style="color:#808030; ">)</span><span style="color:#808030; ">)</span>

Python version <span style="color:#808030; ">:</span> <span style="color:#008000; ">3.7</span><span style="color:#808030; ">.</span><span style="color:#008c00; ">10</span> <span style="color:#808030; ">(</span>default<span style="color:#808030; ">,</span> Feb <span style="color:#008c00; ">20</span> <span style="color:#008c00; ">2021</span><span style="color:#808030; ">,</span> <span style="color:#008c00; ">21</span><span style="color:#808030; ">:</span><span style="color:#008c00; ">17</span><span style="color:#808030; ">:</span><span style="color:#008c00; ">23</span><span style="color:#808030; ">)</span> 
<span style="color:#808030; ">[</span>GCC <span style="color:#008000; ">7.5</span><span style="color:#808030; ">.</span><span style="color:#008c00; ">0</span><span style="color:#808030; ">]</span> 
OpenCV version <span style="color:#808030; ">:</span> <span style="color:#008000; ">4.1</span><span style="color:#808030; ">.</span><span style="color:#008c00; ">2</span> 
Numpy version  <span style="color:#808030; ">:</span> <span style="color:#008000; ">1.19</span><span style="color:#808030; ">.</span><span style="color:#008c00; ">5</span>
Tensorflow version  <span style="color:#808030; ">:</span> <span style="color:#008000; ">2.4</span><span style="color:#808030; ">.</span><span style="color:#008c00; ">1</span>
</pre>

#### 4.1.2. Global variables:


<pre style="color:#000000;background:#e6ffff;font-size:10px;line-height:1.5;"><span style="color:#696969; "># -set the random_state seed = 100 for reproducibilty</span>
random_state_seed <span style="color:#808030; ">=</span> <span style="color:#008c00; ">100</span>

<span style="color:#696969; "># the number of visualized images</span>
num_visualized_images <span style="color:#808030; ">=</span> <span style="color:#008c00; ">25</span>
</pre>


### 4.2. Part 2: Load MNIST Dataset

#### 4.2.1. Load the MNIST dataset :
* Load the MNIST dataset of handwritten digits:
  * 60,000 labelled training examples
  * 10,000 labelled test examples
  * Each handwritten example is 28x28 pixels binary image.

<pre style="color:#000000;background:#e6ffff;font-size:10px;line-height:1.5;"><span style="color:#696969; "># Load in the data: MNIST</span>
mnist <span style="color:#808030; ">=</span> tf<span style="color:#808030; ">.</span>keras<span style="color:#808030; ">.</span>datasets<span style="color:#808030; ">.</span>mnist
<span style="color:#696969; "># mnist.load_data() automatically splits traing and test data sets</span>
<span style="color:#808030; ">(</span>x_train<span style="color:#808030; ">,</span> y_train<span style="color:#808030; ">)</span><span style="color:#808030; ">,</span> <span style="color:#808030; ">(</span>x_test<span style="color:#808030; ">,</span> y_test<span style="color:#808030; ">)</span> <span style="color:#808030; ">=</span> mnist<span style="color:#808030; ">.</span>load_data<span style="color:#808030; ">(</span><span style="color:#808030; ">)</span>
</pre>


#### 4.2.2. Explore training and test images:

##### 4.2.2.1. Display the number and shape of the training and test subsets:

<pre style="color:#000000;background:#e6ffff;font-size:10px;line-height:1.5;"><span style="color:#696969; ">#------------------------------------------------------</span>
<span style="color:#696969; "># Training data:</span>
<span style="color:#696969; ">#------------------------------------------------------</span>
<span style="color:#696969; "># the number of training images</span>
num_train_images <span style="color:#808030; ">=</span> x_train<span style="color:#808030; ">.</span>shape<span style="color:#808030; ">[</span><span style="color:#008c00; ">0</span><span style="color:#808030; ">]</span>
<span style="color:#800000; font-weight:bold; ">print</span><span style="color:#808030; ">(</span><span style="color:#0000e6; ">"----------------------------------------------"</span><span style="color:#808030; ">)</span>
<span style="color:#800000; font-weight:bold; ">print</span><span style="color:#808030; ">(</span><span style="color:#0000e6; ">"Training data:"</span><span style="color:#808030; ">)</span>
<span style="color:#800000; font-weight:bold; ">print</span><span style="color:#808030; ">(</span><span style="color:#0000e6; ">"----------------------------------------------"</span><span style="color:#808030; ">)</span>
<span style="color:#800000; font-weight:bold; ">print</span><span style="color:#808030; ">(</span><span style="color:#0000e6; ">"x_train.shape: "</span><span style="color:#808030; ">,</span> x_train<span style="color:#808030; ">.</span>shape<span style="color:#808030; ">)</span>
<span style="color:#800000; font-weight:bold; ">print</span><span style="color:#808030; ">(</span><span style="color:#0000e6; ">"Number of training images: "</span><span style="color:#808030; ">,</span> num_train_images<span style="color:#808030; ">)</span>
<span style="color:#800000; font-weight:bold; ">print</span><span style="color:#808030; ">(</span><span style="color:#0000e6; ">"Image size: "</span><span style="color:#808030; ">,</span> x_train<span style="color:#808030; ">.</span>shape<span style="color:#808030; ">[</span><span style="color:#008c00; ">1</span><span style="color:#808030; ">:</span><span style="color:#808030; ">]</span><span style="color:#808030; ">)</span>

<span style="color:#696969; ">#------------------------------------------------------</span>
<span style="color:#696969; "># Test data:</span>
<span style="color:#696969; ">#------------------------------------------------------</span>
<span style="color:#696969; "># the number of test images</span>
num_test_images <span style="color:#808030; ">=</span> x_test<span style="color:#808030; ">.</span>shape<span style="color:#808030; ">[</span><span style="color:#008c00; ">0</span><span style="color:#808030; ">]</span>
<span style="color:#800000; font-weight:bold; ">print</span><span style="color:#808030; ">(</span><span style="color:#0000e6; ">"----------------------------------------------"</span><span style="color:#808030; ">)</span>
<span style="color:#800000; font-weight:bold; ">print</span><span style="color:#808030; ">(</span><span style="color:#0000e6; ">"Test data:"</span><span style="color:#808030; ">)</span>
<span style="color:#800000; font-weight:bold; ">print</span><span style="color:#808030; ">(</span><span style="color:#0000e6; ">"----------------------------------------------"</span><span style="color:#808030; ">)</span>
<span style="color:#800000; font-weight:bold; ">print</span><span style="color:#808030; ">(</span><span style="color:#0000e6; ">"x_test.shape: "</span><span style="color:#808030; ">,</span> x_test<span style="color:#808030; ">.</span>shape<span style="color:#808030; ">)</span>
<span style="color:#800000; font-weight:bold; ">print</span><span style="color:#808030; ">(</span><span style="color:#0000e6; ">"Number of test images: "</span><span style="color:#808030; ">,</span> num_test_images<span style="color:#808030; ">)</span>
<span style="color:#800000; font-weight:bold; ">print</span><span style="color:#808030; ">(</span><span style="color:#0000e6; ">"Image size: "</span><span style="color:#808030; ">,</span> x_test<span style="color:#808030; ">.</span>shape<span style="color:#808030; ">[</span><span style="color:#008c00; ">1</span><span style="color:#808030; ">:</span><span style="color:#808030; ">]</span><span style="color:#808030; ">)</span>
<span style="color:#800000; font-weight:bold; ">print</span><span style="color:#808030; ">(</span><span style="color:#0000e6; ">"----------------------------------------------"</span><span style="color:#808030; ">)</span>

<span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span>
Training data<span style="color:#808030; ">:</span>
<span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span>
x_train<span style="color:#808030; ">.</span>shape<span style="color:#808030; ">:</span>  <span style="color:#808030; ">(</span><span style="color:#008c00; ">60000</span><span style="color:#808030; ">,</span> <span style="color:#008c00; ">28</span><span style="color:#808030; ">,</span> <span style="color:#008c00; ">28</span><span style="color:#808030; ">)</span>
Number of training images<span style="color:#808030; ">:</span>  <span style="color:#008c00; ">60000</span>
Image size<span style="color:#808030; ">:</span>  <span style="color:#808030; ">(</span><span style="color:#008c00; ">28</span><span style="color:#808030; ">,</span> <span style="color:#008c00; ">28</span><span style="color:#808030; ">)</span>
<span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span>
Test data<span style="color:#808030; ">:</span>
<span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span>
x_test<span style="color:#808030; ">.</span>shape<span style="color:#808030; ">:</span>  <span style="color:#808030; ">(</span><span style="color:#008c00; ">10000</span><span style="color:#808030; ">,</span> <span style="color:#008c00; ">28</span><span style="color:#808030; ">,</span> <span style="color:#008c00; ">28</span><span style="color:#808030; ">)</span>
Number of test images<span style="color:#808030; ">:</span>  <span style="color:#008c00; ">10000</span>
Image size<span style="color:#808030; ">:</span>  <span style="color:#808030; ">(</span><span style="color:#008c00; ">28</span><span style="color:#808030; ">,</span> <span style="color:#008c00; ">28</span><span style="color:#808030; ">)</span>
<span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span>
</pre>

##### 4.2.2.2. Reshape the training and test images to 3D:

* The training and test images are 2D grayscale/binary:
  * CNN expect the images to be of shape:
    * height x width x color
  * We need to add a fourth color dimension to:
    * the training images: x_train
    * the test images: x_test
    

<pre style="color:#000000;background:#e6ffff;font-size:10px;line-height:1.5;"><span style="color:#696969; ">#------------------------------------------------------</span>
<span style="color:#696969; "># reshape the x_train and x_test images 4D:</span>
<span style="color:#696969; ">#------------------------------------------------------</span>
<span style="color:#696969; "># Add a fourth color dimension to x_train</span>
x_train <span style="color:#808030; ">=</span> np<span style="color:#808030; ">.</span>expand_dims<span style="color:#808030; ">(</span>x_train<span style="color:#808030; ">,</span> <span style="color:#44aadd; ">-</span><span style="color:#008c00; ">1</span><span style="color:#808030; ">)</span> 
<span style="color:#696969; "># add a fourth color dimension to x_test</span>
x_test <span style="color:#808030; ">=</span> np<span style="color:#808030; ">.</span>expand_dims<span style="color:#808030; ">(</span>x_test<span style="color:#808030; ">,</span> <span style="color:#44aadd; ">-</span><span style="color:#008c00; ">1</span><span style="color:#808030; ">)</span>
<span style="color:#696969; ">#------------------------------------------------------</span>
<span style="color:#696969; "># display the new shapes of x_train and x_test</span>
<span style="color:#696969; ">#------------------------------------------------------</span>
<span style="color:#800000; font-weight:bold; ">print</span><span style="color:#808030; ">(</span><span style="color:#0000e6; ">"----------------------------------------------"</span><span style="color:#808030; ">)</span>
<span style="color:#800000; font-weight:bold; ">print</span><span style="color:#808030; ">(</span><span style="color:#0000e6; ">"Re-shaped x_train:"</span><span style="color:#808030; ">)</span>
<span style="color:#800000; font-weight:bold; ">print</span><span style="color:#808030; ">(</span><span style="color:#0000e6; ">"----------------------------------------------"</span><span style="color:#808030; ">)</span>
<span style="color:#800000; font-weight:bold; ">print</span><span style="color:#808030; ">(</span><span style="color:#0000e6; ">"x_train.shape: "</span><span style="color:#808030; ">,</span> x_train<span style="color:#808030; ">.</span>shape<span style="color:#808030; ">)</span>
<span style="color:#800000; font-weight:bold; ">print</span><span style="color:#808030; ">(</span><span style="color:#0000e6; ">"----------------------------------------------"</span><span style="color:#808030; ">)</span>
<span style="color:#800000; font-weight:bold; ">print</span><span style="color:#808030; ">(</span><span style="color:#0000e6; ">"Re-shaped x_test:"</span><span style="color:#808030; ">)</span>
<span style="color:#800000; font-weight:bold; ">print</span><span style="color:#808030; ">(</span><span style="color:#0000e6; ">"----------------------------------------------"</span><span style="color:#808030; ">)</span>
<span style="color:#800000; font-weight:bold; ">print</span><span style="color:#808030; ">(</span><span style="color:#0000e6; ">"x_test.shape: "</span><span style="color:#808030; ">,</span> x_test<span style="color:#808030; ">.</span>shape<span style="color:#808030; ">)</span>
<span style="color:#800000; font-weight:bold; ">print</span><span style="color:#808030; ">(</span><span style="color:#0000e6; ">"----------------------------------------------"</span><span style="color:#808030; ">)</span>

<span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span>
Re<span style="color:#44aadd; ">-</span>shaped x_train<span style="color:#808030; ">:</span>
<span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span>
x_train<span style="color:#808030; ">.</span>shape<span style="color:#808030; ">:</span>  <span style="color:#808030; ">(</span><span style="color:#008c00; ">60000</span><span style="color:#808030; ">,</span> <span style="color:#008c00; ">28</span><span style="color:#808030; ">,</span> <span style="color:#008c00; ">28</span><span style="color:#808030; ">,</span> <span style="color:#008c00; ">1</span><span style="color:#808030; ">,</span> <span style="color:#008c00; ">1</span><span style="color:#808030; ">)</span>
<span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span>
Re<span style="color:#44aadd; ">-</span>shaped x_test<span style="color:#808030; ">:</span>
<span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span>
x_test<span style="color:#808030; ">.</span>shape<span style="color:#808030; ">:</span>  <span style="color:#808030; ">(</span><span style="color:#008c00; ">10000</span><span style="color:#808030; ">,</span> <span style="color:#008c00; ">28</span><span style="color:#808030; ">,</span> <span style="color:#008c00; ">28</span><span style="color:#808030; ">,</span> <span style="color:#008c00; ">1</span><span style="color:#808030; ">,</span> <span style="color:#008c00; ">1</span><span style="color:#808030; ">)</span>
<span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span>
</pre>

##### 4.2.2.3. Display the targets/classes:

* The classification of the digits should be: 0 to 9



<pre style="color:#000000;background:#e6ffff;font-size:10px;line-height:1.5;"><span style="color:#800000; font-weight:bold; ">print</span><span style="color:#808030; ">(</span><span style="color:#0000e6; ">"----------------------------------------------"</span><span style="color:#808030; ">)</span>
<span style="color:#800000; font-weight:bold; ">print</span><span style="color:#808030; ">(</span><span style="color:#0000e6; ">"Classes/labels:"</span><span style="color:#808030; ">)</span>
<span style="color:#800000; font-weight:bold; ">print</span><span style="color:#808030; ">(</span><span style="color:#0000e6; ">"----------------------------------------------"</span><span style="color:#808030; ">)</span>
<span style="color:#800000; font-weight:bold; ">print</span><span style="color:#808030; ">(</span><span style="color:#0000e6; ">'The target labels: '</span> <span style="color:#44aadd; ">+</span> <span style="color:#400000; ">str</span><span style="color:#808030; ">(</span>np<span style="color:#808030; ">.</span>unique<span style="color:#808030; ">(</span>y_train<span style="color:#808030; ">)</span><span style="color:#808030; ">)</span><span style="color:#808030; ">)</span>
<span style="color:#800000; font-weight:bold; ">print</span><span style="color:#808030; ">(</span><span style="color:#0000e6; ">"----------------------------------------------"</span><span style="color:#808030; ">)</span>

<span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span>
Classes<span style="color:#44aadd; ">/</span>labels<span style="color:#808030; ">:</span>
<span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span>
The target labels<span style="color:#808030; ">:</span> <span style="color:#808030; ">[</span><span style="color:#008c00; ">0</span> <span style="color:#008c00; ">1</span> <span style="color:#008c00; ">2</span> <span style="color:#008c00; ">3</span> <span style="color:#008c00; ">4</span> <span style="color:#008c00; ">5</span> <span style="color:#008c00; ">6</span> <span style="color:#008c00; ">7</span> <span style="color:#008c00; ">8</span> <span style="color:#008c00; ">9</span><span style="color:#808030; ">]</span>
<span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span>
</pre>


##### 4.2.2.4. Examine the number of images for each class of the training and testing subsets:


<pre style="color:#000000;background:#e6ffff;font-size:10px;line-height:1.5;"><span style="color:#696969; "># Create a histogram of the number of images in each class/digit:</span>
<span style="color:#800000; font-weight:bold; ">def</span> plot_bar<span style="color:#808030; ">(</span>y<span style="color:#808030; ">,</span> loc<span style="color:#808030; ">=</span><span style="color:#0000e6; ">'left'</span><span style="color:#808030; ">,</span> relative<span style="color:#808030; ">=</span><span style="color:#074726; ">True</span><span style="color:#808030; ">)</span><span style="color:#808030; ">:</span>
    width <span style="color:#808030; ">=</span> <span style="color:#008000; ">0.35</span>
    <span style="color:#800000; font-weight:bold; ">if</span> loc <span style="color:#44aadd; ">==</span> <span style="color:#0000e6; ">'left'</span><span style="color:#808030; ">:</span>
        n <span style="color:#808030; ">=</span> <span style="color:#44aadd; ">-</span><span style="color:#008000; ">0.5</span>
    <span style="color:#800000; font-weight:bold; ">elif</span> loc <span style="color:#44aadd; ">==</span> <span style="color:#0000e6; ">'right'</span><span style="color:#808030; ">:</span>
        n <span style="color:#808030; ">=</span> <span style="color:#008000; ">0.5</span>
     
    <span style="color:#696969; "># calculate counts per type and sort, to ensure their order</span>
    unique<span style="color:#808030; ">,</span> counts <span style="color:#808030; ">=</span> np<span style="color:#808030; ">.</span>unique<span style="color:#808030; ">(</span>y<span style="color:#808030; ">,</span> return_counts<span style="color:#808030; ">=</span><span style="color:#074726; ">True</span><span style="color:#808030; ">)</span>
    sorted_index <span style="color:#808030; ">=</span> np<span style="color:#808030; ">.</span>argsort<span style="color:#808030; ">(</span>unique<span style="color:#808030; ">)</span>
    unique <span style="color:#808030; ">=</span> unique<span style="color:#808030; ">[</span>sorted_index<span style="color:#808030; ">]</span>
     
    <span style="color:#800000; font-weight:bold; ">if</span> relative<span style="color:#808030; ">:</span>
        <span style="color:#696969; "># plot as a percentage</span>
        counts <span style="color:#808030; ">=</span> <span style="color:#008c00; ">100</span><span style="color:#44aadd; ">*</span>counts<span style="color:#808030; ">[</span>sorted_index<span style="color:#808030; ">]</span><span style="color:#44aadd; ">/</span><span style="color:#400000; ">len</span><span style="color:#808030; ">(</span>y<span style="color:#808030; ">)</span>
        ylabel_text <span style="color:#808030; ">=</span> <span style="color:#0000e6; ">'% count'</span>
    <span style="color:#800000; font-weight:bold; ">else</span><span style="color:#808030; ">:</span>
        <span style="color:#696969; "># plot counts</span>
        counts <span style="color:#808030; ">=</span> counts<span style="color:#808030; ">[</span>sorted_index<span style="color:#808030; ">]</span>
        ylabel_text <span style="color:#808030; ">=</span> <span style="color:#0000e6; ">'count'</span>
         
    xtemp <span style="color:#808030; ">=</span> np<span style="color:#808030; ">.</span>arange<span style="color:#808030; ">(</span><span style="color:#400000; ">len</span><span style="color:#808030; ">(</span>unique<span style="color:#808030; ">)</span><span style="color:#808030; ">)</span>
    plt<span style="color:#808030; ">.</span>bar<span style="color:#808030; ">(</span>xtemp <span style="color:#44aadd; ">+</span> n<span style="color:#44aadd; ">*</span>width<span style="color:#808030; ">,</span> counts<span style="color:#808030; ">,</span> align<span style="color:#808030; ">=</span><span style="color:#0000e6; ">'center'</span><span style="color:#808030; ">,</span> alpha<span style="color:#808030; ">=</span><span style="color:#008000; ">.7</span><span style="color:#808030; ">,</span> width<span style="color:#808030; ">=</span>width<span style="color:#808030; ">)</span>
    plt<span style="color:#808030; ">.</span>xticks<span style="color:#808030; ">(</span>xtemp<span style="color:#808030; ">,</span> unique<span style="color:#808030; ">,</span> rotation<span style="color:#808030; ">=</span><span style="color:#008c00; ">45</span><span style="color:#808030; ">)</span>
    plt<span style="color:#808030; ">.</span>xlabel<span style="color:#808030; ">(</span><span style="color:#0000e6; ">'digit'</span><span style="color:#808030; ">)</span>
    plt<span style="color:#808030; ">.</span>ylabel<span style="color:#808030; ">(</span>ylabel_text<span style="color:#808030; ">)</span>
 
plt<span style="color:#808030; ">.</span>suptitle<span style="color:#808030; ">(</span><span style="color:#0000e6; ">'Frequency of images per digit'</span><span style="color:#808030; ">)</span>
plot_bar<span style="color:#808030; ">(</span>y_train<span style="color:#808030; ">,</span> loc<span style="color:#808030; ">=</span><span style="color:#0000e6; ">'left'</span><span style="color:#808030; ">)</span>
plot_bar<span style="color:#808030; ">(</span>y_test<span style="color:#808030; ">,</span> loc<span style="color:#808030; ">=</span><span style="color:#0000e6; ">'right'</span><span style="color:#808030; ">)</span>
plt<span style="color:#808030; ">.</span>legend<span style="color:#808030; ">(</span><span style="color:#808030; ">[</span>
    <span style="color:#0000e6; ">'train ({0} images)'</span><span style="color:#808030; ">.</span>format<span style="color:#808030; ">(</span><span style="color:#400000; ">len</span><span style="color:#808030; ">(</span>y_train<span style="color:#808030; ">)</span><span style="color:#808030; ">)</span><span style="color:#808030; ">,</span> 
    <span style="color:#0000e6; ">'test ({0} images)'</span><span style="color:#808030; ">.</span>format<span style="color:#808030; ">(</span><span style="color:#400000; ">len</span><span style="color:#808030; ">(</span>y_test<span style="color:#808030; ">)</span><span style="color:#808030; ">)</span><span style="color:#808030; ">,</span> 
<span style="color:#808030; ">]</span><span style="color:#808030; ">)</span><span style="color:#808030; ">;</span>
</pre>

<img src="images/train-test-images-distributions.png" width = 1000/>

##### 4.2.2.5. Visualize some of the training and test images and their associated targets:

* First implement a visualization functionality to visualize the number of randomly selected images:

<pre style="color:#000000;background:#e6ffff;font-size:10px;line-height:1.5;"><span style="color:#696969; ">"""</span>
<span style="color:#696969; "># A utility function to visualize multiple images:</span>
<span style="color:#696969; ">"""</span>
<span style="color:#800000; font-weight:bold; ">def</span> visualize_images_and_labels<span style="color:#808030; ">(</span>num_visualized_images <span style="color:#808030; ">=</span> <span style="color:#008c00; ">25</span><span style="color:#808030; ">,</span> dataset_flag <span style="color:#808030; ">=</span> <span style="color:#008c00; ">1</span><span style="color:#808030; ">)</span><span style="color:#808030; ">:</span>
  <span style="color:#696969; ">"""To visualize images.</span>
<span style="color:#696969; "></span>
<span style="color:#696969; ">&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Keyword arguments:</span>
<span style="color:#696969; ">&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;- num_visualized_images -- the number of visualized images (deafult 25)</span>
<span style="color:#696969; ">&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;- dataset_flag -- 1: training dataset, 2: test dataset</span>
<span style="color:#696969; ">&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Return:</span>
<span style="color:#696969; ">&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;- None</span>
<span style="color:#696969; ">&nbsp;&nbsp;"""</span>
  <span style="color:#696969; ">#--------------------------------------------</span>
  <span style="color:#696969; "># the suplot grid shape:</span>
  <span style="color:#696969; ">#--------------------------------------------</span>
  num_rows <span style="color:#808030; ">=</span> <span style="color:#008c00; ">5</span>
  <span style="color:#696969; "># the number of columns</span>
  num_cols <span style="color:#808030; ">=</span> num_visualized_images <span style="color:#44aadd; ">//</span> num_rows
  <span style="color:#696969; "># setup the subplots axes</span>
  fig<span style="color:#808030; ">,</span> axes <span style="color:#808030; ">=</span> plt<span style="color:#808030; ">.</span>subplots<span style="color:#808030; ">(</span>nrows<span style="color:#808030; ">=</span>num_rows<span style="color:#808030; ">,</span> ncols<span style="color:#808030; ">=</span>num_cols<span style="color:#808030; ">,</span> figsize<span style="color:#808030; ">=</span><span style="color:#808030; ">(</span><span style="color:#008c00; ">8</span><span style="color:#808030; ">,</span> <span style="color:#008c00; ">10</span><span style="color:#808030; ">)</span><span style="color:#808030; ">)</span>
  <span style="color:#696969; "># set a seed random number generator for reproducible results</span>
  seed<span style="color:#808030; ">(</span>random_state_seed<span style="color:#808030; ">)</span>
  <span style="color:#696969; "># iterate over the sub-plots</span>
  <span style="color:#800000; font-weight:bold; ">for</span> row <span style="color:#800000; font-weight:bold; ">in</span> <span style="color:#400000; ">range</span><span style="color:#808030; ">(</span>num_rows<span style="color:#808030; ">)</span><span style="color:#808030; ">:</span>
      <span style="color:#800000; font-weight:bold; ">for</span> col <span style="color:#800000; font-weight:bold; ">in</span> <span style="color:#400000; ">range</span><span style="color:#808030; ">(</span>num_cols<span style="color:#808030; ">)</span><span style="color:#808030; ">:</span>
        <span style="color:#696969; "># get the next figure axis</span>
        ax <span style="color:#808030; ">=</span> axes<span style="color:#808030; ">[</span>row<span style="color:#808030; ">,</span> col<span style="color:#808030; ">]</span><span style="color:#808030; ">;</span>
        <span style="color:#696969; "># turn-off subplot axis</span>
        ax<span style="color:#808030; ">.</span>set_axis_off<span style="color:#808030; ">(</span><span style="color:#808030; ">)</span>
        <span style="color:#696969; ">#--------------------------------------------</span>
        <span style="color:#696969; "># if the dataset_flag = 1: Training data set</span>
        <span style="color:#696969; ">#--------------------------------------------</span>
        <span style="color:#800000; font-weight:bold; ">if</span> <span style="color:#808030; ">(</span> dataset_flag <span style="color:#44aadd; ">==</span> <span style="color:#008c00; ">1</span> <span style="color:#808030; ">)</span><span style="color:#808030; ">:</span> 
          <span style="color:#696969; "># generate a random image counter</span>
          counter <span style="color:#808030; ">=</span> randint<span style="color:#808030; ">(</span><span style="color:#008c00; ">0</span><span style="color:#808030; ">,</span>num_train_images<span style="color:#808030; ">)</span>
          <span style="color:#696969; "># get the training image</span>
          image <span style="color:#808030; ">=</span> np<span style="color:#808030; ">.</span>squeeze<span style="color:#808030; ">(</span>x_train<span style="color:#808030; ">[</span>counter<span style="color:#808030; ">,</span><span style="color:#808030; ">:</span><span style="color:#808030; ">]</span><span style="color:#808030; ">)</span>
          <span style="color:#696969; "># get the target associated with the image</span>
          label <span style="color:#808030; ">=</span> y_train<span style="color:#808030; ">[</span>counter<span style="color:#808030; ">]</span>
        <span style="color:#696969; ">#--------------------------------------------</span>
        <span style="color:#696969; "># dataset_flag = 2: Test data set</span>
        <span style="color:#696969; ">#--------------------------------------------</span>
        <span style="color:#800000; font-weight:bold; ">else</span><span style="color:#808030; ">:</span> 
          <span style="color:#696969; "># generate a random image counter</span>
          counter <span style="color:#808030; ">=</span> randint<span style="color:#808030; ">(</span><span style="color:#008c00; ">0</span><span style="color:#808030; ">,</span>num_test_images<span style="color:#808030; ">)</span>
          <span style="color:#696969; "># get the test image</span>
          image <span style="color:#808030; ">=</span> np<span style="color:#808030; ">.</span>squeeze<span style="color:#808030; ">(</span>x_test<span style="color:#808030; ">[</span>counter<span style="color:#808030; ">,</span><span style="color:#808030; ">:</span><span style="color:#808030; ">]</span><span style="color:#808030; ">)</span>
          <span style="color:#696969; "># get the target associated with the image</span>
          label <span style="color:#808030; ">=</span> y_test<span style="color:#808030; ">[</span>counter<span style="color:#808030; ">]</span>
        <span style="color:#696969; ">#--------------------------------------------</span>
        <span style="color:#696969; "># display the image</span>
        <span style="color:#696969; ">#--------------------------------------------</span>
        ax<span style="color:#808030; ">.</span>imshow<span style="color:#808030; ">(</span>image<span style="color:#808030; ">,</span> cmap<span style="color:#808030; ">=</span>plt<span style="color:#808030; ">.</span>cm<span style="color:#808030; ">.</span>gray_r<span style="color:#808030; ">,</span> interpolation<span style="color:#808030; ">=</span><span style="color:#0000e6; ">'nearest'</span><span style="color:#808030; ">)</span>
        <span style="color:#696969; "># set the title showing the image label</span>
        ax<span style="color:#808030; ">.</span>set_title<span style="color:#808030; ">(</span><span style="color:#0000e6; ">'y ='</span> <span style="color:#44aadd; ">+</span> <span style="color:#400000; ">str</span><span style="color:#808030; ">(</span>label<span style="color:#808030; ">)</span><span style="color:#808030; ">,</span> size <span style="color:#808030; ">=</span> <span style="color:#008c00; ">8</span><span style="color:#808030; ">)</span>
</pre>

###### 4.2.2.5.1. Visualize some of the training images and their associated targets:


<img src="images/25-random-train-images.png" width = 1000/>

###### 4.2.2.5.2. Visualize some of the test images and their associated targets:


<img src="images/25-random-test-images.png" width = 1000/>

#### 4.2.3. Normalize the training and test images to the interval: [0, 1]:


<pre style="color:#000000;background:#e6ffff;font-size:10px;line-height:1.5;"><span style="color:#696969; "># Normalize the training images</span>
x_train <span style="color:#808030; ">=</span> x_train <span style="color:#44aadd; ">/</span> <span style="color:#008000; ">255.0</span>
<span style="color:#696969; "># Normalize the test images</span>
x_test <span style="color:#808030; ">=</span> x_test <span style="color:#44aadd; ">/</span> <span style="color:#008000; ">255.0</span>
</pre>


### 4.3. Part 3: Develop the CNN model architecture

#### 4.3.1. Design the structure of the CNN model to classify the MINIST images:

<pre style="color:#000000;background:#e6ffff;font-size:10px;line-height:1.5;"><span style="color:#696969; ">#-------------------------------------------------------------------------------</span>
<span style="color:#696969; "># Build the sequential CNN model</span>
<span style="color:#696969; ">#-------------------------------------------------------------------------------</span>
model <span style="color:#808030; ">=</span> Sequential<span style="color:#808030; ">(</span><span style="color:#808030; ">)</span>
<span style="color:#696969; ">#-------------------------------------------------------------------------------</span>
<span style="color:#696969; "># Layer # 1: Convolutional layer</span>
<span style="color:#696969; ">#-------------------------------------------------------------------------------</span>
<span style="color:#696969; "># - 32 filters of size: 3x3</span>
<span style="color:#696969; "># - relu activation function </span>
<span style="color:#696969; ">#-------------------------------------------------------------------------------</span>
model<span style="color:#808030; ">.</span>add<span style="color:#808030; ">(</span>Conv2D<span style="color:#808030; ">(</span><span style="color:#008c00; ">32</span><span style="color:#808030; ">,</span> <span style="color:#808030; ">(</span><span style="color:#008c00; ">3</span><span style="color:#808030; ">,</span> <span style="color:#008c00; ">3</span><span style="color:#808030; ">)</span><span style="color:#808030; ">,</span> activation<span style="color:#808030; ">=</span><span style="color:#0000e6; ">'relu'</span><span style="color:#808030; ">,</span> 
                 kernel_initializer<span style="color:#808030; ">=</span><span style="color:#0000e6; ">'he_uniform'</span><span style="color:#808030; ">,</span> 
                 input_shape<span style="color:#808030; ">=</span><span style="color:#808030; ">(</span><span style="color:#008c00; ">28</span><span style="color:#808030; ">,</span> <span style="color:#008c00; ">28</span><span style="color:#808030; ">,</span> <span style="color:#008c00; ">1</span><span style="color:#808030; ">)</span><span style="color:#808030; ">)</span><span style="color:#808030; ">)</span>
<span style="color:#696969; ">#-------------------------------------------------------------------------------</span>
<span style="color:#696969; "># Layer # 2: Max-pooling layer</span>
<span style="color:#696969; ">#-------------------------------------------------------------------------------</span>
<span style="color:#696969; "># - filter size: 2x2</span>
<span style="color:#696969; ">#-------------------------------------------------------------------------------</span>
model<span style="color:#808030; ">.</span>add<span style="color:#808030; ">(</span>MaxPooling2D<span style="color:#808030; ">(</span><span style="color:#808030; ">(</span><span style="color:#008c00; ">2</span><span style="color:#808030; ">,</span> <span style="color:#008c00; ">2</span><span style="color:#808030; ">)</span><span style="color:#808030; ">)</span><span style="color:#808030; ">)</span>
<span style="color:#696969; ">#-------------------------------------------------------------------------------</span>
<span style="color:#696969; "># Layer # 3: Convolutional layer</span>
<span style="color:#696969; ">#-------------------------------------------------------------------------------</span>
<span style="color:#696969; "># - 64 filters of size: 3x3</span>
<span style="color:#696969; "># - relu activation function </span>
<span style="color:#696969; ">#-------------------------------------------------------------------------------</span>
model<span style="color:#808030; ">.</span>add<span style="color:#808030; ">(</span>Conv2D<span style="color:#808030; ">(</span><span style="color:#008c00; ">64</span><span style="color:#808030; ">,</span> <span style="color:#808030; ">(</span><span style="color:#008c00; ">3</span><span style="color:#808030; ">,</span> <span style="color:#008c00; ">3</span><span style="color:#808030; ">)</span><span style="color:#808030; ">,</span> activation<span style="color:#808030; ">=</span><span style="color:#0000e6; ">'relu'</span><span style="color:#808030; ">,</span> kernel_initializer<span style="color:#808030; ">=</span><span style="color:#0000e6; ">'he_uniform'</span><span style="color:#808030; ">)</span><span style="color:#808030; ">)</span>
<span style="color:#696969; ">#-------------------------------------------------------------------------------</span>
<span style="color:#696969; "># Layer # 4: Convolutional layer</span>
<span style="color:#696969; ">#-------------------------------------------------------------------------------</span>
<span style="color:#696969; "># - 64 filters of size: 3x3</span>
<span style="color:#696969; "># - relu activation function </span>
<span style="color:#696969; ">#-------------------------------------------------------------------------------</span>
model<span style="color:#808030; ">.</span>add<span style="color:#808030; ">(</span>Conv2D<span style="color:#808030; ">(</span><span style="color:#008c00; ">64</span><span style="color:#808030; ">,</span> <span style="color:#808030; ">(</span><span style="color:#008c00; ">3</span><span style="color:#808030; ">,</span> <span style="color:#008c00; ">3</span><span style="color:#808030; ">)</span><span style="color:#808030; ">,</span> activation<span style="color:#808030; ">=</span><span style="color:#0000e6; ">'relu'</span><span style="color:#808030; ">,</span> kernel_initializer<span style="color:#808030; ">=</span><span style="color:#0000e6; ">'he_uniform'</span><span style="color:#808030; ">)</span><span style="color:#808030; ">)</span>
<span style="color:#696969; ">#-------------------------------------------------------------------------------</span>
<span style="color:#696969; "># Layer # 5: Max-pooling layer</span>
<span style="color:#696969; ">#-------------------------------------------------------------------------------</span>
<span style="color:#696969; "># - filter size: 2x2</span>
<span style="color:#696969; ">#-------------------------------------------------------------------------------</span>
model<span style="color:#808030; ">.</span>add<span style="color:#808030; ">(</span>MaxPooling2D<span style="color:#808030; ">(</span><span style="color:#808030; ">(</span><span style="color:#008c00; ">2</span><span style="color:#808030; ">,</span> <span style="color:#008c00; ">2</span><span style="color:#808030; ">)</span><span style="color:#808030; ">)</span><span style="color:#808030; ">)</span>
<span style="color:#696969; ">#-------------------------------------------------------------------------------</span>
<span style="color:#696969; "># Layer # 6: Flatten to connect to Fully-Connected layer</span>
<span style="color:#696969; ">#-------------------------------------------------------------------------------</span>
model<span style="color:#808030; ">.</span>add<span style="color:#808030; ">(</span>Flatten<span style="color:#808030; ">(</span><span style="color:#808030; ">)</span><span style="color:#808030; ">)</span>
<span style="color:#696969; ">#-------------------------------------------------------------------------------</span>
<span style="color:#696969; "># Layer # 7: Dense layer</span>
<span style="color:#696969; ">#-------------------------------------------------------------------------------</span>
<span style="color:#696969; "># - 100 neurons</span>
<span style="color:#696969; "># - relu activation function </span>
<span style="color:#696969; ">#-------------------------------------------------------------------------------</span>
model<span style="color:#808030; ">.</span>add<span style="color:#808030; ">(</span>Dense<span style="color:#808030; ">(</span><span style="color:#008c00; ">100</span><span style="color:#808030; ">,</span> activation<span style="color:#808030; ">=</span><span style="color:#0000e6; ">'relu'</span><span style="color:#808030; ">,</span> kernel_initializer<span style="color:#808030; ">=</span><span style="color:#0000e6; ">'he_uniform'</span><span style="color:#808030; ">)</span><span style="color:#808030; ">)</span>
<span style="color:#696969; ">#-------------------------------------------------------------------------------</span>
<span style="color:#696969; "># Layer # 8: Output layer</span>
<span style="color:#696969; ">#-------------------------------------------------------------------------------</span>
<span style="color:#696969; "># - 10 neurons: output: 0-9</span>
<span style="color:#696969; "># - Activation function: Softmax for multi-class classification</span>
<span style="color:#696969; ">#-------------------------------------------------------------------------------</span>
model<span style="color:#808030; ">.</span>add<span style="color:#808030; ">(</span>Dense<span style="color:#808030; ">(</span><span style="color:#008c00; ">10</span><span style="color:#808030; ">,</span> activation<span style="color:#808030; ">=</span><span style="color:#0000e6; ">'softmax'</span><span style="color:#808030; ">)</span><span style="color:#808030; ">)</span>
</pre>


#### 4.3.2. Print the designed model summary


<pre style="color:#000000;background:#e6ffff;font-size:10px;line-height:1.5;"><span style="color:#696969; "># print the model summary</span>
model<span style="color:#808030; ">.</span>summary<span style="color:#808030; ">(</span><span style="color:#808030; ">)</span>
</pre>



<pre style="color:#000000;background:#e6ffff;font-size:10px;line-height:1.5;">Model<span style="color:#808030; ">:</span> <span style="color:#0000e6; ">"sequential_4"</span>
_________________________________________________________________
Layer <span style="color:#808030; ">(</span><span style="color:#400000; ">type</span><span style="color:#808030; ">)</span>                 Output Shape              Param <span style="color:#696969; ">#   </span>
<span style="color:#44aadd; ">==</span><span style="color:#44aadd; ">==</span><span style="color:#44aadd; ">==</span><span style="color:#44aadd; ">==</span><span style="color:#44aadd; ">==</span><span style="color:#44aadd; ">==</span><span style="color:#44aadd; ">==</span><span style="color:#44aadd; ">==</span><span style="color:#44aadd; ">==</span><span style="color:#44aadd; ">==</span><span style="color:#44aadd; ">==</span><span style="color:#44aadd; ">==</span><span style="color:#44aadd; ">==</span><span style="color:#44aadd; ">==</span><span style="color:#44aadd; ">==</span><span style="color:#44aadd; ">==</span><span style="color:#44aadd; ">==</span><span style="color:#44aadd; ">==</span><span style="color:#44aadd; ">==</span><span style="color:#44aadd; ">==</span><span style="color:#44aadd; ">==</span><span style="color:#44aadd; ">==</span><span style="color:#44aadd; ">==</span><span style="color:#44aadd; ">==</span><span style="color:#44aadd; ">==</span><span style="color:#44aadd; ">==</span><span style="color:#44aadd; ">==</span><span style="color:#44aadd; ">==</span><span style="color:#44aadd; ">==</span><span style="color:#44aadd; ">==</span><span style="color:#44aadd; ">==</span><span style="color:#44aadd; ">==</span><span style="color:#808030; ">=</span>
conv2d_7 <span style="color:#808030; ">(</span>Conv2D<span style="color:#808030; ">)</span>            <span style="color:#808030; ">(</span><span style="color:#074726; ">None</span><span style="color:#808030; ">,</span> <span style="color:#008c00; ">26</span><span style="color:#808030; ">,</span> <span style="color:#008c00; ">26</span><span style="color:#808030; ">,</span> <span style="color:#008c00; ">32</span><span style="color:#808030; ">)</span>        <span style="color:#008c00; ">320</span>       
_________________________________________________________________
max_pooling2d_4 <span style="color:#808030; ">(</span>MaxPooling2 <span style="color:#808030; ">(</span><span style="color:#074726; ">None</span><span style="color:#808030; ">,</span> <span style="color:#008c00; ">13</span><span style="color:#808030; ">,</span> <span style="color:#008c00; ">13</span><span style="color:#808030; ">,</span> <span style="color:#008c00; ">32</span><span style="color:#808030; ">)</span>        <span style="color:#008c00; ">0</span>         
_________________________________________________________________
conv2d_8 <span style="color:#808030; ">(</span>Conv2D<span style="color:#808030; ">)</span>            <span style="color:#808030; ">(</span><span style="color:#074726; ">None</span><span style="color:#808030; ">,</span> <span style="color:#008c00; ">11</span><span style="color:#808030; ">,</span> <span style="color:#008c00; ">11</span><span style="color:#808030; ">,</span> <span style="color:#008c00; ">64</span><span style="color:#808030; ">)</span>        <span style="color:#008c00; ">18496</span>     
_________________________________________________________________
conv2d_9 <span style="color:#808030; ">(</span>Conv2D<span style="color:#808030; ">)</span>            <span style="color:#808030; ">(</span><span style="color:#074726; ">None</span><span style="color:#808030; ">,</span> <span style="color:#008c00; ">9</span><span style="color:#808030; ">,</span> <span style="color:#008c00; ">9</span><span style="color:#808030; ">,</span> <span style="color:#008c00; ">64</span><span style="color:#808030; ">)</span>          <span style="color:#008c00; ">36928</span>     
_________________________________________________________________
max_pooling2d_5 <span style="color:#808030; ">(</span>MaxPooling2 <span style="color:#808030; ">(</span><span style="color:#074726; ">None</span><span style="color:#808030; ">,</span> <span style="color:#008c00; ">4</span><span style="color:#808030; ">,</span> <span style="color:#008c00; ">4</span><span style="color:#808030; ">,</span> <span style="color:#008c00; ">64</span><span style="color:#808030; ">)</span>          <span style="color:#008c00; ">0</span>         
_________________________________________________________________
flatten_2 <span style="color:#808030; ">(</span>Flatten<span style="color:#808030; ">)</span>          <span style="color:#808030; ">(</span><span style="color:#074726; ">None</span><span style="color:#808030; ">,</span> <span style="color:#008c00; ">1024</span><span style="color:#808030; ">)</span>              <span style="color:#008c00; ">0</span>         
_________________________________________________________________
dense_4 <span style="color:#808030; ">(</span>Dense<span style="color:#808030; ">)</span>              <span style="color:#808030; ">(</span><span style="color:#074726; ">None</span><span style="color:#808030; ">,</span> <span style="color:#008c00; ">100</span><span style="color:#808030; ">)</span>               <span style="color:#008c00; ">102500</span>    
_________________________________________________________________
dense_5 <span style="color:#808030; ">(</span>Dense<span style="color:#808030; ">)</span>              <span style="color:#808030; ">(</span><span style="color:#074726; ">None</span><span style="color:#808030; ">,</span> <span style="color:#008c00; ">10</span><span style="color:#808030; ">)</span>                <span style="color:#008c00; ">1010</span>      
<span style="color:#44aadd; ">==</span><span style="color:#44aadd; ">==</span><span style="color:#44aadd; ">==</span><span style="color:#44aadd; ">==</span><span style="color:#44aadd; ">==</span><span style="color:#44aadd; ">==</span><span style="color:#44aadd; ">==</span><span style="color:#44aadd; ">==</span><span style="color:#44aadd; ">==</span><span style="color:#44aadd; ">==</span><span style="color:#44aadd; ">==</span><span style="color:#44aadd; ">==</span><span style="color:#44aadd; ">==</span><span style="color:#44aadd; ">==</span><span style="color:#44aadd; ">==</span><span style="color:#44aadd; ">==</span><span style="color:#44aadd; ">==</span><span style="color:#44aadd; ">==</span><span style="color:#44aadd; ">==</span><span style="color:#44aadd; ">==</span><span style="color:#44aadd; ">==</span><span style="color:#44aadd; ">==</span><span style="color:#44aadd; ">==</span><span style="color:#44aadd; ">==</span><span style="color:#44aadd; ">==</span><span style="color:#44aadd; ">==</span><span style="color:#44aadd; ">==</span><span style="color:#44aadd; ">==</span><span style="color:#44aadd; ">==</span><span style="color:#44aadd; ">==</span><span style="color:#44aadd; ">==</span><span style="color:#44aadd; ">==</span><span style="color:#808030; ">=</span>
Total params<span style="color:#808030; ">:</span> <span style="color:#008c00; ">159</span><span style="color:#808030; ">,</span><span style="color:#008c00; ">254</span>
Trainable params<span style="color:#808030; ">:</span> <span style="color:#008c00; ">159</span><span style="color:#808030; ">,</span><span style="color:#008c00; ">254</span>
Non<span style="color:#44aadd; ">-</span>trainable params<span style="color:#808030; ">:</span> <span style="color:#008c00; ">0</span>
</pre>

### 4.4. Part 4: Compile the CNN model

* Compile the CNN model, developed above

<pre style="color:#000000;background:#e6ffff;font-size:10px;line-height:1.5;"><span style="color:#696969; "># Compile the CNN model</span>
<span style="color:#696969; "># test the SGD optimizer</span>
<span style="color:#696969; "># opt = SGD(lr=0.01, momentum=0.9)</span>
model<span style="color:#808030; ">.</span><span style="color:#400000; ">compile</span><span style="color:#808030; ">(</span>optimizer<span style="color:#808030; ">=</span><span style="color:#0000e6; ">'adam'</span><span style="color:#808030; ">,</span> <span style="color:#696969; "># the optimizer: Gradient descent version (adam vs. SGD, etc.)</span>
              loss<span style="color:#808030; ">=</span><span style="color:#0000e6; ">'sparse_categorical_crossentropy'</span><span style="color:#808030; ">,</span> <span style="color:#696969; "># use for multi-class classification models</span>
              metrics<span style="color:#808030; ">=</span><span style="color:#808030; ">[</span><span style="color:#0000e6; ">'accuracy'</span><span style="color:#808030; ">]</span><span style="color:#808030; ">)</span> <span style="color:#696969; "># performance evaluation metric</span>
</pre>


### 4.5. Part 5: Train/Fit the model:

* Start training the compiled CNN model.


<pre style="color:#000000;background:#e6ffff;font-size:10px;line-height:1.5;"><span style="color:#696969; "># Train the model</span>
<span style="color:#696969; "># - Train for 25 epochs:</span>
result <span style="color:#808030; ">=</span> model<span style="color:#808030; ">.</span>fit<span style="color:#808030; ">(</span>x_train<span style="color:#808030; ">,</span> y_train<span style="color:#808030; ">,</span> 
                   validation_data<span style="color:#808030; ">=</span><span style="color:#808030; ">(</span>x_test<span style="color:#808030; ">,</span> y_test<span style="color:#808030; ">)</span><span style="color:#808030; ">,</span> 
                   epochs<span style="color:#808030; ">=</span><span style="color:#008c00; ">25</span><span style="color:#808030; ">)</span>

Epoch <span style="color:#008c00; ">1</span><span style="color:#44aadd; ">/</span><span style="color:#008c00; ">25</span>
<span style="color:#008c00; ">1875</span><span style="color:#44aadd; ">/</span><span style="color:#008c00; ">1875</span> <span style="color:#808030; ">[</span><span style="color:#44aadd; ">==</span><span style="color:#808030; ">=</span><span style="color:#808030; ">]</span> <span style="color:#44aadd; ">-</span> <span style="color:#008c00; ">77</span><span style="color:#ffffff; background:#dd0000; font-weight:bold; font-style:italic; ">s</span> <span style="color:#008c00; ">41</span><span style="color:#ffffff; background:#dd0000; font-weight:bold; font-style:italic; ">ms</span><span style="color:#44aadd; ">/</span>step <span style="color:#44aadd; ">-</span> loss<span style="color:#808030; ">:</span> <span style="color:#008000; ">0.2343</span> <span style="color:#44aadd; ">-</span> accuracy<span style="color:#808030; ">:</span> <span style="color:#008000; ">0.9269</span> <span style="color:#44aadd; ">-</span> val_loss<span style="color:#808030; ">:</span> <span style="color:#008000; ">0.0437</span> <span style="color:#44aadd; ">-</span> val_accuracy<span style="color:#808030; ">:</span> <span style="color:#008000; ">0.9865</span>
Epoch <span style="color:#008c00; ">2</span><span style="color:#44aadd; ">/</span><span style="color:#008c00; ">25</span>
<span style="color:#008c00; ">1875</span><span style="color:#44aadd; ">/</span><span style="color:#008c00; ">1875</span> <span style="color:#808030; ">[</span><span style="color:#44aadd; ">==</span><span style="color:#808030; ">=</span><span style="color:#808030; ">]</span> <span style="color:#44aadd; ">-</span> <span style="color:#008c00; ">77</span><span style="color:#ffffff; background:#dd0000; font-weight:bold; font-style:italic; ">s</span> <span style="color:#008c00; ">41</span><span style="color:#ffffff; background:#dd0000; font-weight:bold; font-style:italic; ">ms</span><span style="color:#44aadd; ">/</span>step <span style="color:#44aadd; ">-</span> loss<span style="color:#808030; ">:</span> <span style="color:#008000; ">0.0447</span> <span style="color:#44aadd; ">-</span> accuracy<span style="color:#808030; ">:</span> <span style="color:#008000; ">0.9852</span> <span style="color:#44aadd; ">-</span> val_loss<span style="color:#808030; ">:</span> <span style="color:#008000; ">0.0529</span> <span style="color:#44aadd; ">-</span> val_accuracy<span style="color:#808030; ">:</span> <span style="color:#008000; ">0.9837</span>
Epoch <span style="color:#008c00; ">3</span><span style="color:#44aadd; ">/</span><span style="color:#008c00; ">25</span>
<span style="color:#008c00; ">1875</span><span style="color:#44aadd; ">/</span><span style="color:#008c00; ">1875</span> <span style="color:#808030; ">[</span><span style="color:#44aadd; ">==</span><span style="color:#808030; ">=</span><span style="color:#808030; ">]</span> <span style="color:#44aadd; ">-</span> <span style="color:#008c00; ">77</span><span style="color:#ffffff; background:#dd0000; font-weight:bold; font-style:italic; ">s</span> <span style="color:#008c00; ">41</span><span style="color:#ffffff; background:#dd0000; font-weight:bold; font-style:italic; ">ms</span><span style="color:#44aadd; ">/</span>step <span style="color:#44aadd; ">-</span> loss<span style="color:#808030; ">:</span> <span style="color:#008000; ">0.0292</span> <span style="color:#44aadd; ">-</span> accuracy<span style="color:#808030; ">:</span> <span style="color:#008000; ">0.9908</span> <span style="color:#44aadd; ">-</span> val_loss<span style="color:#808030; ">:</span> <span style="color:#008000; ">0.0291</span> <span style="color:#44aadd; ">-</span> val_accuracy<span style="color:#808030; ">:</span> <span style="color:#008000; ">0.9912</span>
Epoch <span style="color:#008c00; ">4</span><span style="color:#44aadd; ">/</span><span style="color:#008c00; ">25</span>
<span style="color:#008c00; ">1875</span><span style="color:#44aadd; ">/</span><span style="color:#008c00; ">1875</span> <span style="color:#808030; ">[</span><span style="color:#44aadd; ">==</span><span style="color:#808030; ">=</span><span style="color:#808030; ">]</span> <span style="color:#44aadd; ">-</span> <span style="color:#008c00; ">77</span><span style="color:#ffffff; background:#dd0000; font-weight:bold; font-style:italic; ">s</span> <span style="color:#008c00; ">41</span><span style="color:#ffffff; background:#dd0000; font-weight:bold; font-style:italic; ">ms</span><span style="color:#44aadd; ">/</span>step <span style="color:#44aadd; ">-</span> loss<span style="color:#808030; ">:</span> <span style="color:#008000; ">0.0198</span> <span style="color:#44aadd; ">-</span> accuracy<span style="color:#808030; ">:</span> <span style="color:#008000; ">0.9937</span> <span style="color:#44aadd; ">-</span> val_loss<span style="color:#808030; ">:</span> <span style="color:#008000; ">0.0280</span> <span style="color:#44aadd; ">-</span> val_accuracy<span style="color:#808030; ">:</span> <span style="color:#008000; ">0.9921</span>
Epoch <span style="color:#008c00; ">5</span><span style="color:#44aadd; ">/</span><span style="color:#008c00; ">25</span>
<span style="color:#008c00; ">1875</span><span style="color:#44aadd; ">/</span><span style="color:#008c00; ">1875</span> <span style="color:#808030; ">[</span><span style="color:#44aadd; ">==</span><span style="color:#808030; ">=</span><span style="color:#808030; ">]</span> <span style="color:#44aadd; ">-</span> <span style="color:#008c00; ">77</span><span style="color:#ffffff; background:#dd0000; font-weight:bold; font-style:italic; ">s</span> <span style="color:#008c00; ">41</span><span style="color:#ffffff; background:#dd0000; font-weight:bold; font-style:italic; ">ms</span><span style="color:#44aadd; ">/</span>step <span style="color:#44aadd; ">-</span> loss<span style="color:#808030; ">:</span> <span style="color:#008000; ">0.0143</span> <span style="color:#44aadd; ">-</span> accuracy<span style="color:#808030; ">:</span> <span style="color:#008000; ">0.9956</span> <span style="color:#44aadd; ">-</span> val_loss<span style="color:#808030; ">:</span> <span style="color:#008000; ">0.0303</span> <span style="color:#44aadd; ">-</span> val_accuracy<span style="color:#808030; ">:</span> <span style="color:#008000; ">0.9897</span>
<span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span>
<span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span>
Epoch <span style="color:#008c00; ">20</span><span style="color:#44aadd; ">/</span><span style="color:#008c00; ">25</span>
<span style="color:#008c00; ">1875</span><span style="color:#44aadd; ">/</span><span style="color:#008c00; ">1875</span> <span style="color:#808030; ">[</span><span style="color:#44aadd; ">==</span><span style="color:#808030; ">=</span><span style="color:#808030; ">]</span> <span style="color:#44aadd; ">-</span> <span style="color:#008c00; ">78</span><span style="color:#ffffff; background:#dd0000; font-weight:bold; font-style:italic; ">s</span> <span style="color:#008c00; ">42</span><span style="color:#ffffff; background:#dd0000; font-weight:bold; font-style:italic; ">ms</span><span style="color:#44aadd; ">/</span>step <span style="color:#44aadd; ">-</span> loss<span style="color:#808030; ">:</span> <span style="color:#008000; ">0.0053</span> <span style="color:#44aadd; ">-</span> accuracy<span style="color:#808030; ">:</span> <span style="color:#008000; ">0.9988</span> <span style="color:#44aadd; ">-</span> val_loss<span style="color:#808030; ">:</span> <span style="color:#008000; ">0.0444</span> <span style="color:#44aadd; ">-</span> val_accuracy<span style="color:#808030; ">:</span> <span style="color:#008000; ">0.9934</span>
Epoch <span style="color:#008c00; ">21</span><span style="color:#44aadd; ">/</span><span style="color:#008c00; ">25</span>
<span style="color:#008c00; ">1875</span><span style="color:#44aadd; ">/</span><span style="color:#008c00; ">1875</span> <span style="color:#808030; ">[</span><span style="color:#44aadd; ">==</span><span style="color:#808030; ">=</span><span style="color:#808030; ">]</span> <span style="color:#44aadd; ">-</span> <span style="color:#008c00; ">78</span><span style="color:#ffffff; background:#dd0000; font-weight:bold; font-style:italic; ">s</span> <span style="color:#008c00; ">42</span><span style="color:#ffffff; background:#dd0000; font-weight:bold; font-style:italic; ">ms</span><span style="color:#44aadd; ">/</span>step <span style="color:#44aadd; ">-</span> loss<span style="color:#808030; ">:</span> <span style="color:#008000; ">0.0044</span> <span style="color:#44aadd; ">-</span> accuracy<span style="color:#808030; ">:</span> <span style="color:#008000; ">0.9987</span> <span style="color:#44aadd; ">-</span> val_loss<span style="color:#808030; ">:</span> <span style="color:#008000; ">0.0470</span> <span style="color:#44aadd; ">-</span> val_accuracy<span style="color:#808030; ">:</span> <span style="color:#008000; ">0.9933</span>
Epoch <span style="color:#008c00; ">22</span><span style="color:#44aadd; ">/</span><span style="color:#008c00; ">25</span>
<span style="color:#008c00; ">1875</span><span style="color:#44aadd; ">/</span><span style="color:#008c00; ">1875</span> <span style="color:#808030; ">[</span><span style="color:#44aadd; ">==</span><span style="color:#808030; ">=</span><span style="color:#808030; ">]</span> <span style="color:#44aadd; ">-</span> <span style="color:#008c00; ">79</span><span style="color:#ffffff; background:#dd0000; font-weight:bold; font-style:italic; ">s</span> <span style="color:#008c00; ">42</span><span style="color:#ffffff; background:#dd0000; font-weight:bold; font-style:italic; ">ms</span><span style="color:#44aadd; ">/</span>step <span style="color:#44aadd; ">-</span> loss<span style="color:#808030; ">:</span> <span style="color:#008000; ">0.0019</span> <span style="color:#44aadd; ">-</span> accuracy<span style="color:#808030; ">:</span> <span style="color:#008000; ">0.9994</span> <span style="color:#44aadd; ">-</span> val_loss<span style="color:#808030; ">:</span> <span style="color:#008000; ">0.0658</span> <span style="color:#44aadd; ">-</span> val_accuracy<span style="color:#808030; ">:</span> <span style="color:#008000; ">0.9893</span>
Epoch <span style="color:#008c00; ">23</span><span style="color:#44aadd; ">/</span><span style="color:#008c00; ">25</span>
<span style="color:#008c00; ">1875</span><span style="color:#44aadd; ">/</span><span style="color:#008c00; ">1875</span> <span style="color:#808030; ">[</span><span style="color:#44aadd; ">==</span><span style="color:#808030; ">=</span><span style="color:#808030; ">]</span> <span style="color:#44aadd; ">-</span> <span style="color:#008c00; ">79</span><span style="color:#ffffff; background:#dd0000; font-weight:bold; font-style:italic; ">s</span> <span style="color:#008c00; ">42</span><span style="color:#ffffff; background:#dd0000; font-weight:bold; font-style:italic; ">ms</span><span style="color:#44aadd; ">/</span>step <span style="color:#44aadd; ">-</span> loss<span style="color:#808030; ">:</span> <span style="color:#008000; ">0.0055</span> <span style="color:#44aadd; ">-</span> accuracy<span style="color:#808030; ">:</span> <span style="color:#008000; ">0.9986</span> <span style="color:#44aadd; ">-</span> val_loss<span style="color:#808030; ">:</span> <span style="color:#008000; ">0.0543</span> <span style="color:#44aadd; ">-</span> val_accuracy<span style="color:#808030; ">:</span> <span style="color:#008000; ">0.9914</span>
Epoch <span style="color:#008c00; ">24</span><span style="color:#44aadd; ">/</span><span style="color:#008c00; ">25</span>
<span style="color:#008c00; ">1875</span><span style="color:#44aadd; ">/</span><span style="color:#008c00; ">1875</span> <span style="color:#808030; ">[</span><span style="color:#44aadd; ">==</span><span style="color:#808030; ">=</span><span style="color:#808030; ">]</span> <span style="color:#44aadd; ">-</span> <span style="color:#008c00; ">79</span><span style="color:#ffffff; background:#dd0000; font-weight:bold; font-style:italic; ">s</span> <span style="color:#008c00; ">42</span><span style="color:#ffffff; background:#dd0000; font-weight:bold; font-style:italic; ">ms</span><span style="color:#44aadd; ">/</span>step <span style="color:#44aadd; ">-</span> loss<span style="color:#808030; ">:</span> <span style="color:#008000; ">0.0029</span> <span style="color:#44aadd; ">-</span> accuracy<span style="color:#808030; ">:</span> <span style="color:#008000; ">0.9991</span> <span style="color:#44aadd; ">-</span> val_loss<span style="color:#808030; ">:</span> <span style="color:#008000; ">0.0451</span> <span style="color:#44aadd; ">-</span> val_accuracy<span style="color:#808030; ">:</span> <span style="color:#008000; ">0.9918</span>
Epoch <span style="color:#008c00; ">25</span><span style="color:#44aadd; ">/</span><span style="color:#008c00; ">25</span>
</pre>


### 4.6. Part 6: Evaluate the model

* Evaluate the trained CNN model on the test data using different evaluation metrics:
  * Loss function
  * Accuracy
  * Confusion matrix.

#### 4.6.1. Loss function:

* Display the variations of the training and validation loss function with the number of epochs:

<pre style="color:#000000;background:#e6ffff;font-size:10px;line-height:1.5;"><span style="color:#696969; "># Plot loss per iteration</span>
<span style="color:#800000; font-weight:bold; ">import</span> matplotlib<span style="color:#808030; ">.</span>pyplot <span style="color:#800000; font-weight:bold; ">as</span> plt
plt<span style="color:#808030; ">.</span>plot<span style="color:#808030; ">(</span>result<span style="color:#808030; ">.</span>history<span style="color:#808030; ">[</span><span style="color:#0000e6; ">'loss'</span><span style="color:#808030; ">]</span><span style="color:#808030; ">,</span> label<span style="color:#808030; ">=</span><span style="color:#0000e6; ">'loss'</span><span style="color:#808030; ">)</span>
plt<span style="color:#808030; ">.</span>plot<span style="color:#808030; ">(</span>result<span style="color:#808030; ">.</span>history<span style="color:#808030; ">[</span><span style="color:#0000e6; ">'val_loss'</span><span style="color:#808030; ">]</span><span style="color:#808030; ">,</span> label<span style="color:#808030; ">=</span><span style="color:#0000e6; ">'val_loss'</span><span style="color:#808030; ">)</span><span style="color:#808030; ">;</span>
plt<span style="color:#808030; ">.</span>legend<span style="color:#808030; ">(</span><span style="color:#808030; ">)</span><span style="color:#808030; ">;</span>
plt<span style="color:#808030; ">.</span>xlabel<span style="color:#808030; ">(</span><span style="color:#0000e6; ">'Epoch Iteration'</span><span style="color:#808030; ">)</span><span style="color:#808030; ">;</span>
plt<span style="color:#808030; ">.</span>ylabel<span style="color:#808030; ">(</span><span style="color:#0000e6; ">'Loss'</span><span style="color:#808030; ">)</span><span style="color:#808030; ">;</span>
</pre>


<img src="images/cnn-evaluation-loss-function.png" width = 1000/>


#### 4.6.2. Accuracy:


* Display the variations of the training and validation accuracy with the number of epochs:


<pre style="color:#000000;background:#e6ffff;font-size:10px;line-height:1.5;"><span style="color:#696969; "># Plot accuracy per iteration</span>
plt<span style="color:#808030; ">.</span>plot<span style="color:#808030; ">(</span>result<span style="color:#808030; ">.</span>history<span style="color:#808030; ">[</span><span style="color:#0000e6; ">'accuracy'</span><span style="color:#808030; ">]</span><span style="color:#808030; ">,</span> label<span style="color:#808030; ">=</span><span style="color:#0000e6; ">'acc'</span><span style="color:#808030; ">)</span><span style="color:#808030; ">;</span>
plt<span style="color:#808030; ">.</span>plot<span style="color:#808030; ">(</span>result<span style="color:#808030; ">.</span>history<span style="color:#808030; ">[</span><span style="color:#0000e6; ">'val_accuracy'</span><span style="color:#808030; ">]</span><span style="color:#808030; ">,</span> label<span style="color:#808030; ">=</span><span style="color:#0000e6; ">'val_acc'</span><span style="color:#808030; ">)</span><span style="color:#808030; ">;</span>
plt<span style="color:#808030; ">.</span>legend<span style="color:#808030; ">(</span><span style="color:#808030; ">)</span><span style="color:#808030; ">;</span>
plt<span style="color:#808030; ">.</span>xlabel<span style="color:#808030; ">(</span><span style="color:#0000e6; ">'Epoch Iteration'</span><span style="color:#808030; ">)</span><span style="color:#808030; ">;</span>
plt<span style="color:#808030; ">.</span>ylabel<span style="color:#808030; ">(</span><span style="color:#0000e6; ">'Accuracy'</span><span style="color:#808030; ">)</span><span style="color:#808030; ">;</span>
</pre>

<img src="images/cnn-evaluation-accuracy.png" width = 1000/>


<pre style="color:#000000;background:#e6ffff;font-size:10px;line-height:1.5;"><span style="color:#696969; "># Compute the model accuracy on the test data</span>
<span style="color:#800000; font-weight:bold; ">print</span><span style="color:#808030; ">(</span>model<span style="color:#808030; ">.</span>evaluate<span style="color:#808030; ">(</span>x_test<span style="color:#808030; ">,</span> y_test<span style="color:#808030; ">)</span><span style="color:#808030; ">)</span>

<span style="color:#008c00; ">313</span><span style="color:#44aadd; ">/</span><span style="color:#008c00; ">313</span> <span style="color:#808030; ">[</span><span style="color:#44aadd; ">==</span><span style="color:#44aadd; ">==</span><span style="color:#44aadd; ">==</span><span style="color:#44aadd; ">==</span><span style="color:#44aadd; ">==</span><span style="color:#44aadd; ">==</span><span style="color:#44aadd; ">==</span><span style="color:#44aadd; ">==</span><span style="color:#44aadd; ">==</span><span style="color:#44aadd; ">==</span><span style="color:#44aadd; ">==</span><span style="color:#44aadd; ">==</span><span style="color:#44aadd; ">==</span><span style="color:#44aadd; ">==</span><span style="color:#44aadd; ">==</span><span style="color:#808030; ">]</span> <span style="color:#44aadd; ">-</span> <span style="color:#008c00; ">4</span><span style="color:#ffffff; background:#dd0000; font-weight:bold; font-style:italic; ">s</span> <span style="color:#008c00; ">13</span><span style="color:#ffffff; background:#dd0000; font-weight:bold; font-style:italic; ">ms</span><span style="color:#44aadd; ">/</span>step <span style="color:#44aadd; ">-</span> loss<span style="color:#808030; ">:</span> <span style="color:#008000; ">0.0583</span> <span style="color:#44aadd; ">-</span> accuracy<span style="color:#808030; ">:</span> <span style="color:#008000; ">0.9902</span>
<span style="color:#808030; ">[</span><span style="color:#008000; ">0.05827852711081505</span><span style="color:#808030; ">,</span> <span style="color:#008000; ">0.9901999831199646</span><span style="color:#808030; ">]</span>
</pre>


#### 4.6.3. Confusion Matrix Visualizations:

* Compute the confusion matrixl:

<pre style="color:#000000;background:#e6ffff;font-size:10px;line-height:1.5;"><span style="color:#696969; ">#-------------------------------------------------------------------------------</span>
<span style="color:#696969; "># Compute the confusion matrix</span>
<span style="color:#696969; ">#-------------------------------------------------------------------------------</span>
<span style="color:#800000; font-weight:bold; ">def</span> plot_confusion_matrix<span style="color:#808030; ">(</span>cm<span style="color:#808030; ">,</span> classes<span style="color:#808030; ">,</span>
                          normalize<span style="color:#808030; ">=</span><span style="color:#074726; ">False</span><span style="color:#808030; ">,</span>
                          title<span style="color:#808030; ">=</span><span style="color:#0000e6; ">'Confusion matrix'</span><span style="color:#808030; ">,</span>
                          cmap<span style="color:#808030; ">=</span>plt<span style="color:#808030; ">.</span>cm<span style="color:#808030; ">.</span>Blues<span style="color:#808030; ">)</span><span style="color:#808030; ">:</span>
  <span style="color:#696969; ">"""</span>
<span style="color:#696969; ">&nbsp;&nbsp;This function prints and plots the confusion matrix.</span>
<span style="color:#696969; ">&nbsp;&nbsp;Normalization can be applied by setting `normalize=True`.</span>
<span style="color:#696969; ">&nbsp;&nbsp;"""</span>
  <span style="color:#800000; font-weight:bold; ">if</span> normalize<span style="color:#808030; ">:</span>
      cm <span style="color:#808030; ">=</span> cm<span style="color:#808030; ">.</span>astype<span style="color:#808030; ">(</span><span style="color:#0000e6; ">'float'</span><span style="color:#808030; ">)</span> <span style="color:#44aadd; ">/</span> cm<span style="color:#808030; ">.</span><span style="color:#400000; ">sum</span><span style="color:#808030; ">(</span>axis<span style="color:#808030; ">=</span><span style="color:#008c00; ">1</span><span style="color:#808030; ">)</span><span style="color:#808030; ">[</span><span style="color:#808030; ">:</span><span style="color:#808030; ">,</span> np<span style="color:#808030; ">.</span>newaxis<span style="color:#808030; ">]</span>
      <span style="color:#800000; font-weight:bold; ">print</span><span style="color:#808030; ">(</span><span style="color:#0000e6; ">"Normalized confusion matrix"</span><span style="color:#808030; ">)</span>
  <span style="color:#800000; font-weight:bold; ">else</span><span style="color:#808030; ">:</span>
      <span style="color:#800000; font-weight:bold; ">print</span><span style="color:#808030; ">(</span><span style="color:#0000e6; ">'Confusion matrix, without normalization'</span><span style="color:#808030; ">)</span>

  <span style="color:#696969; "># Display the confusuon matrix</span>
  <span style="color:#800000; font-weight:bold; ">print</span><span style="color:#808030; ">(</span>cm<span style="color:#808030; ">)</span>
  <span style="color:#696969; "># display the confusion matrix</span>
  plt<span style="color:#808030; ">.</span>imshow<span style="color:#808030; ">(</span>cm<span style="color:#808030; ">,</span> interpolation<span style="color:#808030; ">=</span><span style="color:#0000e6; ">'nearest'</span><span style="color:#808030; ">,</span> cmap<span style="color:#808030; ">=</span>cmap<span style="color:#808030; ">)</span>
  plt<span style="color:#808030; ">.</span>title<span style="color:#808030; ">(</span>title<span style="color:#808030; ">)</span>
  plt<span style="color:#808030; ">.</span>colorbar<span style="color:#808030; ">(</span><span style="color:#808030; ">)</span>
  tick_marks <span style="color:#808030; ">=</span> np<span style="color:#808030; ">.</span>arange<span style="color:#808030; ">(</span><span style="color:#400000; ">len</span><span style="color:#808030; ">(</span>classes<span style="color:#808030; ">)</span><span style="color:#808030; ">)</span>
  plt<span style="color:#808030; ">.</span>xticks<span style="color:#808030; ">(</span>tick_marks<span style="color:#808030; ">,</span> classes<span style="color:#808030; ">,</span> rotation<span style="color:#808030; ">=</span><span style="color:#008c00; ">45</span><span style="color:#808030; ">)</span>
  plt<span style="color:#808030; ">.</span>yticks<span style="color:#808030; ">(</span>tick_marks<span style="color:#808030; ">,</span> classes<span style="color:#808030; ">)</span>
  
  fmt <span style="color:#808030; ">=</span> <span style="color:#0000e6; ">'.2f'</span> <span style="color:#800000; font-weight:bold; ">if</span> normalize <span style="color:#800000; font-weight:bold; ">else</span> <span style="color:#0000e6; ">'d'</span>
  thresh <span style="color:#808030; ">=</span> cm<span style="color:#808030; ">.</span><span style="color:#400000; ">max</span><span style="color:#808030; ">(</span><span style="color:#808030; ">)</span> <span style="color:#44aadd; ">/</span> <span style="color:#008000; ">2.</span>
  <span style="color:#800000; font-weight:bold; ">for</span> i<span style="color:#808030; ">,</span> j <span style="color:#800000; font-weight:bold; ">in</span> itertools<span style="color:#808030; ">.</span>product<span style="color:#808030; ">(</span><span style="color:#400000; ">range</span><span style="color:#808030; ">(</span>cm<span style="color:#808030; ">.</span>shape<span style="color:#808030; ">[</span><span style="color:#008c00; ">0</span><span style="color:#808030; ">]</span><span style="color:#808030; ">)</span><span style="color:#808030; ">,</span> <span style="color:#400000; ">range</span><span style="color:#808030; ">(</span>cm<span style="color:#808030; ">.</span>shape<span style="color:#808030; ">[</span><span style="color:#008c00; ">1</span><span style="color:#808030; ">]</span><span style="color:#808030; ">)</span><span style="color:#808030; ">)</span><span style="color:#808030; ">:</span>
      plt<span style="color:#808030; ">.</span>text<span style="color:#808030; ">(</span>j<span style="color:#808030; ">,</span> i<span style="color:#808030; ">,</span> format<span style="color:#808030; ">(</span>cm<span style="color:#808030; ">[</span>i<span style="color:#808030; ">,</span> j<span style="color:#808030; ">]</span><span style="color:#808030; ">,</span> fmt<span style="color:#808030; ">)</span><span style="color:#808030; ">,</span>
               horizontalalignment<span style="color:#808030; ">=</span><span style="color:#0000e6; ">"center"</span><span style="color:#808030; ">,</span>
               color<span style="color:#808030; ">=</span><span style="color:#0000e6; ">"white"</span> <span style="color:#800000; font-weight:bold; ">if</span> cm<span style="color:#808030; ">[</span>i<span style="color:#808030; ">,</span> j<span style="color:#808030; ">]</span> <span style="color:#44aadd; ">&gt;</span> thresh <span style="color:#800000; font-weight:bold; ">else</span> <span style="color:#0000e6; ">"black"</span><span style="color:#808030; ">)</span>

  plt<span style="color:#808030; ">.</span>tight_layout<span style="color:#808030; ">(</span><span style="color:#808030; ">)</span>
  plt<span style="color:#808030; ">.</span>ylabel<span style="color:#808030; ">(</span><span style="color:#0000e6; ">'True label'</span><span style="color:#808030; ">)</span>
  plt<span style="color:#808030; ">.</span>xlabel<span style="color:#808030; ">(</span><span style="color:#0000e6; ">'Predicted label'</span><span style="color:#808030; ">)</span>
  plt<span style="color:#808030; ">.</span>show<span style="color:#808030; ">(</span><span style="color:#808030; ">)</span>

<span style="color:#696969; ">#-------------------------------------------------------------------------------</span>
<span style="color:#696969; "># Predict the targets for the test data</span>
<span style="color:#696969; ">#-------------------------------------------------------------------------------</span>
p_test <span style="color:#808030; ">=</span> model<span style="color:#808030; ">.</span>predict<span style="color:#808030; ">(</span>x_test<span style="color:#808030; ">)</span><span style="color:#808030; ">.</span>argmax<span style="color:#808030; ">(</span>axis<span style="color:#808030; ">=</span><span style="color:#008c00; ">1</span><span style="color:#808030; ">)</span>
<span style="color:#696969; "># construct the confusion matrix</span>
cm <span style="color:#808030; ">=</span> confusion_matrix<span style="color:#808030; ">(</span>y_test<span style="color:#808030; ">,</span> p_test<span style="color:#808030; ">)</span>
<span style="color:#696969; "># plot the confusion matrix</span>
plot_confusion_matrix<span style="color:#808030; ">(</span>cm<span style="color:#808030; ">,</span> <span style="color:#400000; ">list</span><span style="color:#808030; ">(</span><span style="color:#400000; ">range</span><span style="color:#808030; ">(</span><span style="color:#008c00; ">10</span><span style="color:#808030; ">)</span><span style="color:#808030; ">)</span><span style="color:#808030; ">,</span> 
                      <span style="color:#074726; ">False</span><span style="color:#808030; ">,</span> 
                      <span style="color:#0000e6; ">'Confusion matrix'</span><span style="color:#808030; ">,</span> 
                      plt<span style="color:#808030; ">.</span>cm<span style="color:#808030; ">.</span>Greens<span style="color:#808030; ">)</span>
</pre>


<img src="images/Confusion-matrix.JPG" width = 1000/>

#### 4.6.4 Examine some of the misclassified digits:

* Display some of the misclassified digit:

<pre style="color:#000000;background:#e6ffff;font-size:10px;line-height:1.5;"><span style="color:#696969; "># - Find the indices of all the mis-classified examples</span>
misclassified_idx <span style="color:#808030; ">=</span> np<span style="color:#808030; ">.</span>where<span style="color:#808030; ">(</span>p_test <span style="color:#44aadd; ">!=</span> y_test<span style="color:#808030; ">)</span><span style="color:#808030; ">[</span><span style="color:#008c00; ">0</span><span style="color:#808030; ">]</span> <span style="color:#696969; "># select the index</span>
<span style="color:#696969; "># setup the subplot grid for the visualized images</span>
 <span style="color:#696969; "># the suplot grid shape</span>
num_rows <span style="color:#808030; ">=</span> <span style="color:#008c00; ">5</span>
<span style="color:#696969; "># the number of columns</span>
num_cols <span style="color:#808030; ">=</span> num_visualized_images <span style="color:#44aadd; ">//</span> num_rows
<span style="color:#696969; "># setup the subplots axes</span>
fig<span style="color:#808030; ">,</span> axes <span style="color:#808030; ">=</span> plt<span style="color:#808030; ">.</span>subplots<span style="color:#808030; ">(</span>nrows<span style="color:#808030; ">=</span>num_rows<span style="color:#808030; ">,</span> ncols<span style="color:#808030; ">=</span>num_cols<span style="color:#808030; ">,</span> figsize<span style="color:#808030; ">=</span><span style="color:#808030; ">(</span><span style="color:#008c00; ">6</span><span style="color:#808030; ">,</span> <span style="color:#008c00; ">8</span><span style="color:#808030; ">)</span><span style="color:#808030; ">)</span>
<span style="color:#696969; "># set a seed random number generator for reproducible results</span>
seed<span style="color:#808030; ">(</span>random_state_seed<span style="color:#808030; ">)</span>
<span style="color:#696969; "># iterate over the sub-plots</span>
<span style="color:#800000; font-weight:bold; ">for</span> row <span style="color:#800000; font-weight:bold; ">in</span> <span style="color:#400000; ">range</span><span style="color:#808030; ">(</span>num_rows<span style="color:#808030; ">)</span><span style="color:#808030; ">:</span>
  <span style="color:#800000; font-weight:bold; ">for</span> col <span style="color:#800000; font-weight:bold; ">in</span> <span style="color:#400000; ">range</span><span style="color:#808030; ">(</span>num_cols<span style="color:#808030; ">)</span><span style="color:#808030; ">:</span>
    <span style="color:#696969; "># get the next figure axis</span>
    ax <span style="color:#808030; ">=</span> axes<span style="color:#808030; ">[</span>row<span style="color:#808030; ">,</span> col<span style="color:#808030; ">]</span><span style="color:#808030; ">;</span>
    <span style="color:#696969; "># turn-off subplot axis</span>
    ax<span style="color:#808030; ">.</span>set_axis_off<span style="color:#808030; ">(</span><span style="color:#808030; ">)</span>
    <span style="color:#696969; "># select a random mis-classified example</span>
    counter <span style="color:#808030; ">=</span> np<span style="color:#808030; ">.</span>random<span style="color:#808030; ">.</span>choice<span style="color:#808030; ">(</span>misclassified_idx<span style="color:#808030; ">)</span>
    <span style="color:#696969; "># get test image </span>
    image <span style="color:#808030; ">=</span> np<span style="color:#808030; ">.</span>squeeze<span style="color:#808030; ">(</span>x_test<span style="color:#808030; ">[</span>counter<span style="color:#808030; ">,</span><span style="color:#808030; ">:</span><span style="color:#808030; ">]</span><span style="color:#808030; ">)</span>
    <span style="color:#696969; "># get the true labels of the selected image</span>
    label <span style="color:#808030; ">=</span> y_test<span style="color:#808030; ">[</span>counter<span style="color:#808030; ">]</span>
    <span style="color:#696969; "># get the predicted label of the test image</span>
    yhat <span style="color:#808030; ">=</span> p_test<span style="color:#808030; ">[</span>counter<span style="color:#808030; ">]</span>
    <span style="color:#696969; "># display the image </span>
    ax<span style="color:#808030; ">.</span>imshow<span style="color:#808030; ">(</span>image<span style="color:#808030; ">,</span> cmap<span style="color:#808030; ">=</span>plt<span style="color:#808030; ">.</span>cm<span style="color:#808030; ">.</span>gray_r<span style="color:#808030; ">,</span> interpolation<span style="color:#808030; ">=</span><span style="color:#0000e6; ">'nearest'</span><span style="color:#808030; ">)</span>
    <span style="color:#696969; "># display the true and predicted labels on the title of teh image</span>
    ax<span style="color:#808030; ">.</span>set_title<span style="color:#808030; ">(</span><span style="color:#0000e6; ">'Y = %i, $</span><span style="color:#0f69ff; ">\h</span><span style="color:#0000e6; ">at{Y}$ = %i'</span> <span style="color:#44aadd; ">%</span> <span style="color:#808030; ">(</span><span style="color:#400000; ">int</span><span style="color:#808030; ">(</span>label<span style="color:#808030; ">)</span><span style="color:#808030; ">,</span> <span style="color:#400000; ">int</span><span style="color:#808030; ">(</span>yhat<span style="color:#808030; ">)</span><span style="color:#808030; ">)</span><span style="color:#808030; ">,</span> size <span style="color:#808030; ">=</span> <span style="color:#008c00; ">8</span><span style="color:#808030; ">)</span>
</pre>


<img src="images/25-miclassified-test-images.png" width = 1000/>

### 4.7. Part 7: Display a final message after successful execution completion:


<pre style="color:#000000;background:#e6ffff;font-size:10px;line-height:1.5;"><span style="color:#696969; "># display a final message</span>
<span style="color:#696969; "># current time</span>
now <span style="color:#808030; ">=</span> datetime<span style="color:#808030; ">.</span>datetime<span style="color:#808030; ">.</span>now<span style="color:#808030; ">(</span><span style="color:#808030; ">)</span>
<span style="color:#696969; "># display a message</span>
<span style="color:#800000; font-weight:bold; ">print</span><span style="color:#808030; ">(</span><span style="color:#0000e6; ">'Program executed successfully on: '</span><span style="color:#44aadd; ">+</span> <span style="color:#400000; ">str</span><span style="color:#808030; ">(</span>now<span style="color:#808030; ">.</span>strftime<span style="color:#808030; ">(</span><span style="color:#0000e6; ">"%Y-%m-%d %H:%M:%S"</span><span style="color:#808030; ">)</span> <span style="color:#44aadd; ">+</span> <span style="color:#0000e6; ">"...Goodbye!</span><span style="color:#0f69ff; ">\n</span><span style="color:#0000e6; ">"</span><span style="color:#808030; ">)</span><span style="color:#808030; ">)</span>

Program executed successfully on<span style="color:#808030; ">:</span> <span style="color:#008c00; ">2021</span><span style="color:#44aadd; ">-</span><span style="color:#008c00; ">04</span><span style="color:#44aadd; ">-</span><span style="color:#008c00; ">05</span> <span style="color:#008c00; ">19</span><span style="color:#808030; ">:</span><span style="color:#008c00; ">42</span><span style="color:#808030; ">:</span><span style="color:#008000; ">07.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span>Goodbye!
</pre>


## 5. Analysis



In view of the presented results, we make the following observations:

* The simple designed CNN achieves nearly perfect accuracy (99%) of the MNIST data classification.
* The few misclassifications appear reasonable:
* It is reasonable to confuse 9 with 4, 9, 
* It is reasonable to confuse 9 with 7,
* It is reasonable to confuse 2 with 7, etc. 


## 6. Future Work

* We plan to explore the following related issues:

  * To explore ways of improving the performance of this simple CNN, including fine-tuning the following hyper-parameters:
  * The epochs should be increased until the validation loss and accuracy converge
  * The number of filters and layers
  * The dropout rate
  * The optimizer
  * The learning rate.


## 7. References

1. Yun Lecun, Corina Cortes, Christopher J.C. Buges. The MNIST database if handwritten digits. Retrieved from: http://yann.lecun.com/exdb/mnist/.
Prateek Goyal. MNIST dataset using Deep Leaning algorithm. Retrieved from: https://medium.com/@prtk13061992/mnist-dataset-using-deep-learning-algorithm-ann-c6f83aa594f5.
2. Jason Brownlee. How to Develop a CNN for MNIST Handwritten Digit Classification. Retrieved from: https://machinelearningmastery.com/how-to-develop-a-convolutional-neural-network-from-scratch-for-mnist-handwritten-digit-classification/.
3. Sambit Mahapatra. A simple 2D CNN for MNIST digit recognition. Retrieved from: https://towardsdatascience.com/a-simple-2d-cnn-for-mnist-digit-recognition-a998dbc1e79a.
4. Orhan G. Yalcin. Image Classification in 10 Minutes with MNIST Dataset Using Convolutional Neural Networks to Classify Handwritten Digits with TensorFlow and Keras | Supervised Deep Learning. Retrieved from: https://towardsdatascience.com/image-classification-in-10-minutes-with-mnist-dataset-54c35b77a38d.
5. Jay Gupta. Going beyond 99% — MNIST Handwritten Digits Recognition Learn how to optimize a neural network through various techniques to reach human-level performance. Retrieved from: https://towardsdatascience.com/going-beyond-99-mnist-handwritten-digits-recognition-cfff96337392.

