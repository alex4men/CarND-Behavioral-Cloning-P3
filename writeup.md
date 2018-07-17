# **Behavioral Cloning**

## Writeup Template

### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/centerLaneDriving.jpg "Normal Image"
[image2]: ./examples/recovery1.jpg "Recovery Image1"
[image3]: ./examples/recovery2.jpg "Recovery Image2"
[image4]: ./examples/flip1.jpg "Non-Flipped Image"
[image5]: ./examples/flip2.jpg "Flipped Image"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network
* writeup_report.md summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My model is based on [the comma.ai convolution neural network](https://github.com/commaai/research/blob/master/train_steering_model.py) with 8x8 and 5x5 filter sizes and depths between 16 and 64 (model.py lines 79-92). It has 3 convolution layers and 2 fully connected layers.

The model includes ELU layers to introduce nonlinearity (code lines 82, 84, 88, 91), and the data is normalized in the model using a Keras lambda layer (code line 80) after the cropping the frame to remove excess information from the frame (trees, sky, car hood).

#### 2. Attempts to reduce overfitting in the model

The model contains dropout layers with keep_probability of 0.75 (which was chosen empirically) in order to reduce overfitting (model.py lines 87, 90).

The model was trained and validated on different data sets to ensure that the model was not overfitting (code line 61). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 94).

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I made my own dataset with a combination of center lane driving, recovering from the left and right sides of the road.

For details about how I created the training data, see the next section.

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to stick with a good existing architecture, tune some parameters and augment the data.

My first step was to use a convolution neural network model similar to the the [comma.ai convolution neural network](https://github.com/commaai/research/blob/master/train_steering_model.py) I thought this model might be appropriate because it was used by comma.ai for the same task, and it was quite similar to LeNet-5 and Nvidia's NN architectures.

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set.

To combat the overfitting, I used 2 dropout layers with keep_probabilities of 0.75. Original values were 0.2 and 0.5 which were quite small and the result was poor.

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture

The final model architecture (model.py lines 79-92) consisted of a convolution neural network with the following layers and layer sizes:

| Layer         		|     Description	        					|
|:---------------------:|:---------------------------------------------:|
| Input         		| 160x320x3 RGB image   							|
| Cropping         		| outputs 65x320x3 RGB image   							|
| Normalization (Lambda layer) | x / 127.5 - 1.0  |
| Convolution 8x8     	| 16 filters, 4x4 stride, same padding 	|
| ELU					|												|
| Convolution 5x5     	| 32 filters, 2x2 stride, same padding |
| ELU					|												|
| Convolution 5x5     	| 64 filters, 2x2 stride, same padding |
| Flatten		|       									|
| Dropout					|		keep probability 0.75							|
| ELU					|												|
| Fully connected		| 512 neurons      									|
| Dropout					|		keep probability 0.75							|
| ELU					|												|
| Fully connected		| 1 neuron      									|

#### 3. Creation of the Training Set & Training Process

To capture good driving behaviour, I first recorded two laps on track one using center lane driving. Here is an example image of center lane driving:

![alt text][image1]

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to recover from failures. These images show what a recovery looks like:

![alt text][image2]
![alt text][image3]

To augment the data set, I also flipped images and angles thinking that this would help to generalize better. For example, here is an image that has then been flipped:

![alt text][image4]
![alt text][image5]

After the collection process, I had 7565 number of data points. I then preprocessed this data by dividing by making special Python script (remove_path_beginning.py) which removes prepending path from each image path in the csv file and writes paths in the standard 'IMG/...' format.

I finally randomly shuffled the data set and put 20% of the data into a validation set.

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 1 because further training does not reduce validation error significantly. I used an adam optimizer so that manually training the learning rate wasn't necessary.
