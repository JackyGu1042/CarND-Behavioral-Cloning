import csv
import cv2
import numpy as np

lines = []
with open('./data/data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        lines.append(line)

images = []
measurements = []
num = 0
for line in lines:
    #for i in range(3):#add three camerca images
    source_path = line[0]
    filename = source_path.split('/')[-1]
    current_path = './data/data/IMG/' + filename

    image = cv2.imread(current_path)
    images.append(image)
    #print("Image data shape =", image.shape)
    
    measurement = float(line[3])
    measurements.append(measurement)

#augmented data
augmented_images, augmented_measurements = [], []
for image, measurement in zip(images, measurements):
    augmented_images.append(image)
    augmented_measurements.append(measurement)
    augmented_images.append(cv2.flip(image,1))
    augmented_measurements.append(measurement*(-1.0))

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

from keras.models import Sequential, Model
from keras.layers import Flatten, Dense, Lambda, Activation, Cropping2D
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D
import matplotlib.pyplot as plt

'''
#LeNet
model = Sequential()
model.add(Lambda(lambda x: x/255.0 - 0.5, input_shape=(160,320,3)))
model.add(Cropping2D(cropping=((75,25), (0,0))))
model.add(Convolution2D(6, 5, 5, activation="relu"))
model.add(MaxPooling2D())
model.add(Convolution2D(6, 5, 5, activation="relu"))
model.add(MaxPooling2D())
model.add(Flatten())
model.add(Dense(120))
model.add(Dense(84))
model.add(Dense(1))
'''
#NVIDIA Architecture
model = Sequential()
model.add(Lambda(lambda x: x/255.0 - 0.5, input_shape=(160,320,3)))
model.add(Cropping2D(cropping=((70,25), (0,0))))
model.add(Convolution2D(24, 5, 5, subsample=(2,2), activation="relu"))
model.add(Convolution2D(36, 5, 5, subsample=(2,2), activation="relu"))
model.add(Convolution2D(48, 5, 5, subsample=(2,2), activation="relu"))
model.add(Convolution2D(64, 3, 3, activation="relu"))
model.add(Convolution2D(64, 3, 3, activation="relu"))
model.add(Flatten())
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))

model.summary()
model.compile(loss='mse', optimizer='adam')
history_object = model.fit(X_train,
                           y_train,
                           validation_split=0.2,
                           shuffle=True,
                           nb_epoch=7,
                           verbose=1)

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

model.save('model.h5')
print("Model.h5 saved")
