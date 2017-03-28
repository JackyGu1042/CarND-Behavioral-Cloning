import os
import csv
import cv2
import numpy as np

#Load the training data in different path
def load_training_data(folder_path):
    lines = []
    with open('./' + folder_path + '/driving_log.csv') as csvfile:
        reader = csv.reader(csvfile)
        for line in reader:
            lines.append(line)

    images = []
    #Car multi-camera image array
    image_center = []
    image_left = []
    image_right = []
    #Car steering measurements array
    measurements = []

    for line in lines:
        source_path = line[0]
        filename = source_path.split('/')[-1]
        center_path = './' + folder_path + '/IMG/' + filename
        #Center camera image
        image_center = cv2.imread(center_path) 
        
        source_path = line[1]
        filename = source_path.split('/')[-1]
        left_path = './' + folder_path + '/IMG/' + filename
        #Left camera image
        image_left = cv2.imread(left_path)
        
        source_path = line[2]
        filename = source_path.split('/')[-1]
        right_path = './' + folder_path + '/IMG/' + filename
        #Right camera image
        image_right = cv2.imread(right_path)

        
        measurement_center = float(line[3])
        # create adjusted steering measurements for the side camera images
        correction = 0.22 # this is a parameter to tune
        measurement_left = measurement_center + correction
        measurement_right = measurement_center - correction

        images.append(image_center)
        measurements.append(measurement_center)
        images.append(image_left)
        measurements.append(measurement_left)
        images.append(image_right)
        measurements.append(measurement_right)

    #augmented data, flip camera images to increase data size
    augmented_images, augmented_measurements = [], []
    for image, measurement in zip(images, measurements):
        augmented_images.append(image)
        augmented_measurements.append(measurement)
        augmented_images.append(cv2.flip(image,1))
        augmented_measurements.append(measurement*(-1.0))
    
    X_train = np.array(augmented_images)
    y_train = np.array(augmented_measurements)

    return X_train, y_train

#############################################################################

ch, row, col = 3, 160, 320  # Trimmed image format

#import Keras library
from keras.models import Sequential, Model, load_model
from keras.layers import Flatten, Dense, Lambda, Activation, Cropping2D, Dropout
from keras.layers.convolutional import Convolution2D, Conv2D
from keras.layers.pooling import MaxPooling2D
from keras import optimizers

import matplotlib.pyplot as plt

#NVIDIA Architecture with small modification
model = Sequential()
model.add(Lambda(lambda x: x/255.0 - 0.5, input_shape=(row,col,ch)))
model.add(Cropping2D(cropping=((60,25), (0,0))))

model.add(Conv2D(24, (5, 5), strides=(2,2)))
model.add(Activation('relu'))

model.add(Conv2D(36, (5, 5), strides=(2,2)))
model.add(Activation('relu'))

model.add(Conv2D(48, (5, 5), strides=(2,2)))
model.add(Activation('relu'))

model.add(Conv2D(64, (3, 3)))
model.add(Dropout(0.5)) #Add dropout function to avoid overfitting
model.add(Activation('relu'))

#Reduce one convolution layer to decrease training time
#model.add(Conv2D(64, (3, 3)))
#model.add(Dropout(0.5))

model.add(Activation('relu'))
model.add(Flatten())
model.add(Dense(100))
model.add(Dropout(0.5)) #Add dropout function to avoid overfitting
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))
model.summary()

##########################################
#Load normal drive image for 1st training#
##########################################
X_train, y_train = load_training_data("Map2/data_huaxin_center")

n_train = len(X_train)
X_train_shape = X_train.shape
image_shape = X_train[0].shape
y_train_shape = y_train.shape
class_shape = y_train[0].shape

print("Number of training examples =", n_train)
print("X_train data shape =", X_train_shape)
print("Image data shape =", image_shape)
print("y_train data shape =", y_train_shape)
print("class data shape =", class_shape)

#Learning rate is very important, it should be low with this structure
Adam = optimizers.Adam(lr=0.00005)
model.compile(loss='mse', optimizer=Adam)

history_object = model.fit(X_train,
                           y_train,
                           validation_split=0.2,
                           shuffle=True,
                           epochs=6,
                           verbose=1)

print("Finish normal drive training and save the model")
model.save('Model_multica_map2.h5')

####################################################
#Load side line1 back action image for 2nd training#
####################################################
X_train, y_train = load_training_data("Map2/data_huaxin_line1")

#Beacuse the data size is small, the learning rate should be higher
Adam = optimizers.Adam(lr=0.0002)
model.compile(loss='mse', optimizer=Adam)

history_object = model.fit(X_train,
                           y_train,
                           validation_split=0.2,
                           shuffle=True,
                           epochs=4,
                           verbose=1)

print("Finish curve line1 training and save the model")
model.save('Model_multica_map2.h5')
'''
####################################################
#Load side line2 back action image for 3id training#
####################################################
X_train, y_train = load_training_data("Map2/data_huaxin_line2")

#Beacuse the data size is small, the learning rate should be higher
Adam = optimizers.Adam(lr=0.0002)
model.compile(loss='mse', optimizer=Adam)

history_object = model.fit(X_train,
                           y_train,
                           validation_split=0.2,
                           shuffle=True,
                           epochs=4,
                           verbose=1)

print("Finish curve line2 training and save the model")
model.save('Model_multica.h5')

#Beacuse this bridge line back action will damage the car drive perfomance
#So comment it 

#############################################################
#Load side line back action on bridge image for 4th training#
#############################################################

X_train, y_train = load_training_data("Map2/data_huaxin_bridge")

Adam = optimizers.Adam(lr=0.0002)
model.compile(loss='mse', optimizer=Adam)


history_object = model.fit(X_train,
                           y_train,
                           validation_split=0.2,
                           shuffle=True,
                           epochs=4,
                           verbose=1)

print("Finish curve bridge training and save the model")
model.save('Model_multica.h5')
'''

'''
    #Use generator
history_object = model.fit_generator(train_generator,
                                     samples_per_epoch=len(train_samples),
                                     validation_data=validation_generator,
                                     nb_val_samples=len(validation_samples),
                                     nb_epoch=3,
                                     verbose=1)
'''
### print the keys contained in the history object
print(history_object.history.keys())

### plot the training and validation loss for each epoch
plt.plot(history_object.history['loss'])
plt.plot(history_object.history['val_loss'])
plt.title('model mean squared error loss')
plt.ylabel('mean squared error loss')
plt.xlabel('epoch')
plt.legend(['training set', 'validation set'], loc='upper right')
plt.show()

print("Finish all the training")
