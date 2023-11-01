# Smile Detection using Tensorflow

## Introduction
 The main aim of this experiment is to develop a deep learning model, specifically a 
Convolutional neural network for smile detection. Also, the task is to run the model in real-time 
to detect if a person is smiling or not in a picture captured by the web camera. It is a binary 
classification problem to divide facial picture photos into smiley and non-smiley facial expression 
classes. On the evaluation dataset, we aim to attain an accuracy of at least 85%. 

## Dataset
 The GENKI-4k dataset is used for this task which is a subset of the MPLab GENKI 
dataset. It contains 4000 images of faces of people with corresponding labels of whether they 
are smiling or not which is 0 or 1. There are 2162 images with a smile (1) and 1838 images 
without a smile (0) in the dataset. 
 To train and test the performance of the model, the dataset is split into 80% 
of development set and 20% of the evaluation set using sklearn library.

Download the GENKI-4K Face dataset from https://inc.ucsd.edu/mplab/398.php

The directory structure should be as follows from the baseline root directory:

    GENKI4K/
     | - files/
     |   | - all images
     | - labels.txt


## Model Architecture
 Deep learning architecture contains 5 convolutional blocks. Each block has 2D.CNN 
layer followed by ReLU non-linear activation, a 2D max-pooling layer, and a dropout layer (used for 
overfitting). Convolution blocks are followed by 2 fully connected layers and a classification layer.
The fully connected layer combines the features learned by CNN layers to predict the output class.


## Running the baseline system

### Running an experiment

Experimental settings can be changed in the `main.py` file. 
Change the dataset path and the required parameters accordingly.
For training, use the parameter `train` as True.

To run an experiment, use the following command:

````shell script
$ python main.py
````

### Evaluation with pre-trained weights

For evaluation use the parameter `train` as False and run the following command:

````shell script
$ python main.py 
````

### Evaluation in real-time

The best model stored is used in the file `capture_video.py` and run the following command:

````shell script
$ python capture_video.py 
````

## Training
 The input to the model is the RGB face images from the GENKI-4k dataset, resized 
to 64x64 resolution. Sigmoid activation is used at the end of the model, which is interpreted
as the probability of a smile in each face image. So, we use Binary cross-entropy loss to train 
the model. After training the model for 100 iterations, the model that performs the 
best on the evaluation data is selected for use in real-time testing. Adam Optimizer is utilized with 
a default learning rate. Implementation is done in Keras. # smile_detection_Tensorflow
