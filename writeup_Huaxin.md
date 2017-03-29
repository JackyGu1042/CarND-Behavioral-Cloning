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
[image1]: ./center.jpg "Normal Image"
[image2]: ./side.jpg "Recovery Image"
[image3]: ./felloffspots.jpg "Fell off area"
[image4]: ./track2.jpg "Track two"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model_multica.py containing the script to create and train the model for track one
* model_multica.h5 containing a trained convolution neural network for track one
* model_multica_map2.py containing the script to create and train the model for track two
* model_multica_map2.h5 containing a trained convolution neural network for track two
* drive.py for driving the car in autonomous mode
* writeup_report.md or writeup_report.pdf summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 

```sh
python drive.py model_multica.h5
python drive.py model_multica_map2.h5
```

#### 3. Submission code is usable and readable

The model_multica.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My model consists of below neural network table, which is very similar with NVIDIA architecture:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 160x320x3 RGB image   					|
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
| Dropout				| 0.5 keep probability							|
| Dense					| 50 the output space							|
| Dense					| 10  											|
| Dense					| 1	the output space							|
 

The model includes RELU layers to introduce nonlinearity, and the data is normalized in the model using a Keras lambda layer. 

#### 2. Attempts to reduce overfitting in the model

The model contains dropout layers in order to reduce overfitting. The model was trained and validated on different data sets to ensure that the model was not overfitting. The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, but I still tuned the learning rate which is 0.0001.(I found the different learning rate would effect the training performance a lot)

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the sides of the road and some special fell off area driving. For details about how I created the training data, see the next section. 

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to change the NVIDIA architecture with reducing layer and add dropout.

My first step was to use a convolution neural network model similar to the NVIDIA. I thought this model might be appropriate, but it will cost a lot of training time. So I reduce the convolution layer to make the training process faster. 

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set with validation_split=0.2. I found that my first model had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting. To combat the overfitting, I modified the model so that I add two dropout function and reduce the learning rate.
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
Moreover, follow the lessons' guidance, I flip the camera images to double the data size.
```sh
augmented_images, augmented_measurements = [], []
for image, measurement in zip(images, measurements):
augmented_images.append(image)
augmented_measurements.append(measurement)
augmented_images.append(cv2.flip(image,1))
augmented_measurements.append(measurement*(-1.0))
```
Then I also add Cropping2D function to reduce the unnecessary information of images, which can increase the effienicy of network training.
```sh
model.add(Cropping2D(cropping=((70,25), (0,0))))
```
The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track. To improve the driving behavior in these cases, I record another two video folder:
```sh
data_huaxin_line1
data_huaxin_line2
```
In these two video folder, I record the drving in the spots where car fell off and side road recovery. So in my code, for track one training, there is three times training, frist time is training with center driving data, and second time is training with fell off spots driving data, last one is training with side road recovery driving. And because the data size of the later two dataset is quite small, so I increase the learning rate to 0.0002 to let the network learn faster. In fact, I also prepare another training data about recovery driving on bridge, but I found it's not very neccessary(without this training vehicle already can pass, but sometime with this training vehicle cannot pass, so I comment it in my code)  
```sh
data_huaxin_bridge
```
Finally, vehicle has better performance with these three times training 

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded one laps on track one using center lane driving. Here is an example image of center lane driving, in this record I try to keep the car in the center of road:

![alt text][image1]

I then recorded the vehicle recovering from the sides of the road back to center. And I also record some special area again where the vehicle would fell off the road. :

![alt text][image2]
![alt text][image3]

Because I add multi-camera video and image flip, so I only need to record these three video one time enough.

I finally randomly shuffled the data set and put 20% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 4. Although I used an adam optimizer, but I found manually training the learning rate is still necessary, different learning rate will get different performance.

#### 4. Challenge for Track two

Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track two by executing 
```sh
python drive.py model_multica_map2.h5
```
This track two looks more difficult than track one(for human being). But I found there is a center white line in the road, vehicle can just follow the center line to drive.
So I just keep all the network structure, but small change the Cropping2D function's parameter to let the network more focus on center line:  
```sh
model.add(Cropping2D(cropping=((60,25), (0,0))))
```
In this track, I only record one lap which I try to keep the vehicle drive follow the central white.

![alt text][image4]

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.
