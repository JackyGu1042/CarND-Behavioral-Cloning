# Behavioral Cloning

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/placeholder.png "Model Visualization"
[image2]: ./examples/placeholder.png "Grayscaling"
[image3]: ./examples/placeholder_small.png "Recovery Image"
[image4]: ./examples/placeholder_small.png "Recovery Image"
[image5]: ./examples/placeholder_small.png "Recovery Image"
[image6]: ./examples/placeholder_small.png "Normal Image"
[image7]: ./examples/placeholder_small.png "Flipped Image"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md or writeup_report.pdf summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model_multica.h5
```

#### 3. Submission code is usable and readable

The model_multica.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My model consists of below neural network table, which is very similar with NVIDIA architecture:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 160x320x3 Grayscacle image   					|
| Cropping         		| 70, 25 Symmetric cropping values	    		| 
| Convolution 5x5x24   	| 2x2 stride, Valid padding         			|
| RELU					|												|
| Convolution 5x5x36   	| 2x2 stride, Valid padding 					|
| RELU					|												|
| Convolution 5x5x48   	| 2x2 stride, Valid padding 					|
| RELU					|												|
| Convolution 3x3x64   	| 1x1 stride, Valid padding 					|
| Dropout				| 0.5 keep probility							|
| RELU					|												|
| Flaten        	    |   											|
| Dense					| 100 the output space 							|
| Dropout				| 0.5 keep probility							|
| Dense					| 50 the output space							|
| Dense					| 10  											|
| Dense					| 1	the output space							|
 

The model includes RELU layers to introduce nonlinearity, and the data is normalized in the model using a Keras lambda layer. 

#### 2. Attempts to reduce overfitting in the model

The model contains dropout layers in order to reduce overfitting. 

The model was trained and validated on different data sets to ensure that the model was not overfitting (code line 10-16). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, but I still tuned the learning rate which is 0.0001.

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road.

For details about how I created the training data, see the next section. 

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to optimize the NVIDIA architecture with reducing layer and add dropout.

My first step was to use a convolution neural network model similar to the NVIDIA. I thought this model might be appropriate, but it will cost a lot of training time. So I reduce the convolution layer to make the training prossess faster.

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set with validation_split=2. I found that my first model had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting. 

To combat the overfitting, I modified the model so that I add two dropout function and reduce the learning rate.
```sh
Adam = optimizers.Adam(lr=0.0001)
model.compile(loss='mse', optimizer=Adam)

history_object = model.fit(X_train,
                           y_train,
                           validation_split=0.2,
                           shuffle=True,
                           epochs=4,
                           verbose=1)
```

Then in order to increase the data set size, I use multi-camera's image recode, and create adjusted steering measurements for the side camera images.
```sh
	correction = 0.22 # this is a parameter to tune
	measurement_left = measurement_center + correction
	measurement_right = measurement_center - correction
```
Moreover, follow the lessons' guidance, I also flip the camera images to double the data size.
```sh
    augmented_images, augmented_measurements = [], []
    for image, measurement in zip(images, measurements):
        augmented_images.append(image)
        augmented_measurements.append(measurement)
        augmented_images.append(cv2.flip(image,1))
        augmented_measurements.append(measurement*(-1.0))
```
The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track. To improve the driving behavior in these cases, I record another two video folder:
```sh
data_huaxin_line1
data_huaxin_line2
```
In these two image folder, I record the spots where car fell off, with these two data set retraining, vehicle can drive better. And because the data size of these two folder is quit small, so I increase the learning rate to 0.0002 to let the network learn fast.

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture

The final model architecture consisted of a convolution neural network with the following layers and layer sizes ...

Here is a visualization of the architecture (note: visualizing the architecture is optional according to the project rubric)

![alt text][image1]

#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded one laps on track one using center lane driving. Here is an example image of center lane driving, in this record I try to keep the car in the center of road:

![alt text][image2]

I then recorded the vehicle recovering from the sides of the road back to center. And I also record some special area again. :

![alt text][image3]
![alt text][image4]
![alt text][image5]

Becasuse I add multi-camera and image flip, so I only need record these three video one time.

I finally randomly shuffled the data set and put 20% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 4. Although I used an adam optimizer, but I found manually training the learning rate is still necessary, different learning rate will get different perfomance.
