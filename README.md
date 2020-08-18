# Deep Neural Networks Implementation
  
An implementation of a Deep Neural Network using both:  
1. standard scientific computation libraries (**i.e. Building from scratch**) 
2. **TensorFlow Framework**  
  
## Overview
  
Starting from the very simple concept of **linear and logistic regression**, this set of jupyter Notebooks walks through the process of building **Deep Neural Networks** from scratch without using any **Machine Learning Frameworks** at first. Then we use the neural network we built in a simple application where we classify images as a **"cat"** or **"not cat"** and measure  the accuracy of our model. Then we expand our experience by building a deep neural network using **TensorFlow Framework** and use this neural network as an image classifier for hand signs.  
  
**NOTE:** These Notebooks are some of the programming assignments I successfully passed while attending both [Neural Networks and Deep Learning](https://www.coursera.org/learn/neural-networks-deep-learning) 
and [Improving Deep Neural Networks: Hyperparameter tuning, Regularization and Optimization](https://www.coursera.org/learn/deep-neural-network?specialization=deep-learning) 
Courses which are the **first** and **second** courses of [Deep Learning Specialization](https://www.coursera.org/specializations/deep-learning) 
offerd by [DeepLearning.ai](https://www.deeplearning.ai/) on **Coursera**.  
  
## OutLine
  
This repo contains 5 Juputer Notebooks, Let's have a quick look on each one of them.  
  
### 1. Logistic Regression with a Neural Network mindset
**Problem Statement:**  
  
We are given a dataset ("data.h5") containing:  

- a training set of m_train images labeled as cat (y=1) or non-cat (y=0)
- a test set of m_test images labeled as cat or non-cat
- each image is of shape (num_px, num_px, 3) where 3 is for the 3 channels (RGB). Thus, each image is square (height = num_px) and (width = num_px).  
  
We built a simple **image-recognition algorithm** that can correctly classify pictures as **cat or non-cat**.  
  
**Objectives:**  
  
- Build the general architecture of a learning algorithm, including:
  - Initializing parameters
  - Calculating the cost function and its gradient
  - Using an optimization algorithm (**gradient descent**)
- Gather all three functions above into a main model function, in the right order.  
  
### 2. Planar data classification with one hidden layer
**Problem Statement:**  
  
We have some data  that looks like a [flower](https://github.com/AbdullahBahi/Deep-Neural-Networks-Implementation-Applications/blob/master/2-%20Planar_data_classification_with_onehidden_layer/images/flower.PNG) when plotted as a scatter plot, with some red (label y=0) and some blue (y=1) points. our goal is to build a model to fit this data. In other words, we want the classifier to define regions as either red or blue.  
  
**Objectives:**  
  
- Implement a 2-class classification neural network with a single hidden layer
- Use units with a non-linear activation function, such as tanh
- Compute the cross entropy loss
- Implement forward and backward propagation
  
### 3. Building Deep Neural Network Step by Step
**Objectives:**  
  
To build the neural network, we implemented several "helper functions". These helper functions are used in the next NoteBook to build a two-layer neural network and an L-layer neural network. Here is an outline of this NoteBook:
- Initialize the parameters for a two-layer network and for an  L -layer neural network.
- Implement the forward propagation module (shown in purple in the figure below).
  - Complete the LINEAR part of a layer's forward propagation step (resulting in  Z[l] ).
  - The ACTIVATION function (relu/sigmoid) is given.
  - Combine the previous two steps into a new [LINEAR->ACTIVATION] forward function.
  - Stack the [LINEAR->RELU] forward function L-1 time (for layers 1 through L-1) and add a [LINEAR->SIGMOID] at the end (for the final layer  L ). This gives you a new L_model_forward function.
- Compute the loss.
- Implement the backward propagation module (denoted in red in this [figure](https://github.com/AbdullahBahi/Deep-Neural-Networks-Implementation-Applications/blob/master/3-%20Building_Deep_Neural_Network_Step_by_Step/images/final%20outline.png)).
  - Complete the LINEAR part of a layer's backward propagation step.
  - We give you the gradient of the ACTIVATE function (relu_backward/sigmoid_backward)
  - Combine the previous two steps into a new [LINEAR->ACTIVATION] backward function.
  - Stack [LINEAR->RELU] backward L-1 times and add [LINEAR->SIGMOID] backward in a new L_model_backward function
- Finally update the parameters.  
  
### 4. Deep Neural Network Application
**Problem Statement:**  
  
We will use the same "Cat vs non-Cat" dataset as in "Logistic Regression as a Neural Network". The model we had built had 70% test accuracy on classifying cats vs non-cats images. Hopefully, our new model will perform a better!
  
**Objectives:**  
  
We will use the functions we'd implemented in the previous NoteBook to build a deep network, and apply it to cat vs non-cat classification. 
- Build and apply a deep neural network to supervised learning.  
    
### 5. Building Deep Neural Network using TensorFlow
**Problem Statement:**  
  
One afternoon, with some friends we decided to teach our computers to decipher sign language. We spent a few hours taking pictures in front of a white wall and came up with the following dataset. **It's now our job to build an algorithm that would facilitate communications from a speech-impaired person to someone who doesn't understand sign language.**  
  
**Training set**: 1080 pictures (64 by 64 pixels) of signs representing numbers from 0 to 5 (180 pictures per number).  
**Test set**: 120 pictures (64 by 64 pixels) of signs representing numbers from 0 to 5 (20 pictures per number).  
  
**Note** that this is a subset of the **SIGNS** dataset. The complete dataset contains many more signs.  
  
**Objectives:**  
  
Until now, we've always used numpy to build neural networks. Now we step through a deep learning framework that will allow us to build neural networks more easily. In this Notebook, we do the following in **TensorFlow**:  
  
- Initialize variables
- Start your own session
- Train algorithms
- Implement a Neural Network  
  
