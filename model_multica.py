import os
import csv
import cv2
import numpy as np

lines = []
with open('./data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        lines.append(line)


images = []
image_center = []
image_left = []
image_right = []

measurements = []

num = 0
for line in lines:
    #for i in range(3):#add three camerca images
    source_path = line[0]
    filename = source_path.split('/')[-1]
    center_path = './data/IMG/' + filename
    image_center = cv2.imread(center_path)
    
    source_path = line[1]
    filename = source_path.split('/')[-1]
    left_path = './data/IMG/' + filename
    image_left = cv2.imread(left_path)
    
    source_path = line[2]
    filename = source_path.split('/')[-1]
    right_path = './data/IMG/' + filename
    image_right = cv2.imread(right_path)

    
    measurement_center = float(line[3])
    # create adjusted steering measurements for the side camera images
    correction = 0.2 # this is a parameter to tune
    measurement_left = measurement_center + correction
    measurement_right = measurement_center - correction

    images.append(image_center)
    measurements.append(measurement_center)
    images.append(image_left)
    measurements.append(measurement_left)
    images.append(image_right)
    measurements.append(measurement_right)

#augmented data
augmented_images, augmented_measurements = [], []
for image, measurement in zip(images, measurements):
    augmented_images.append(image)
    augmented_measurements.append(measurement)
    #augmented_images.append(cv2.flip(image,1))
    #augmented_measurements.append(measurement*(-1.0))

X_train = np.array(augmented_images)
y_train = np.array(augmented_measurements)

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

ch, row, col = 3, 160, 320  # Trimmed image format

from keras.models import Sequential, Model
from keras.layers import Flatten, Dense, Lambda, Activation, Cropping2D, Dropout
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D
from keras import optimizers

import matplotlib.pyplot as plt


#NVIDIA Architecture, Huaxin add dropout
model = Sequential()
model.add(Lambda(lambda x: x/255.0 - 0.5, input_shape=(row,col,ch)))
model.add(Cropping2D(cropping=((70,25), (0,0))))

model.add(Convolution2D(24, 5, 5, subsample=(2,2)))
model.add(Activation('relu'))

model.add(Convolution2D(36, 5, 5, subsample=(2,2)))
model.add(Activation('relu'))

model.add(Convolution2D(48, 5, 5, subsample=(2,2)))
model.add(Activation('relu'))

model.add(Convolution2D(64, 3, 3))
model.add(Dropout(0.5))
model.add(Activation('relu'))

model.add(Convolution2D(64, 3, 3))
model.add(Dropout(0.5))
model.add(Activation('relu'))
model.add(Flatten())
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))

model.summary()
Adam = optimizers.Adam(lr=0.0008)
model.compile(loss='mse', optimizer=Adam)

history_object = model.fit(X_train,
                           y_train,
                           validation_split=0.2,
                           shuffle=True,
                           epochs=3,
                           verbose=1)
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

model.save('Model_multica.h5')
print("Model_multica.h5 saved")
