import csv
import numpy as np
from PIL import Image
from matplotlib import image as mpimg
from matplotlib import pyplot as plt


from sklearn.model_selection import train_test_split
from random import shuffle


def generator(samples, batch_size=32):
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []
            for batch_sample in batch_samples:
                
                steering_center = float(batch_sample[3])
                
                # create adjusted steering measurements for the side camera images
                correction = 0.2 # this is a parameter to tune
                steering_left = steering_center + correction
                steering_right = steering_center - correction
                
                # add images and angles to data set
                angles.extend([steering_center, steering_left, steering_right, 
                               -steering_center, -steering_left, -steering_right])

                # read in images from center, left and right cameras
                base_path = 'mydata/'
                img_center = np.asarray(Image.open(base_path + batch_sample[0]))
                img_left = np.asarray(Image.open(base_path + batch_sample[1]))
                img_right = np.asarray(Image.open(base_path + batch_sample[2]))
                # mirror images
                img_center_m = np.fliplr(img_center)
                img_left_m = np.fliplr(img_left)
                img_right_m = np.fliplr(img_right)
                # stack images
                images.extend([img_center, img_left, img_right, 
                               img_center_m, img_left_m, img_right_m])

            X_train = np.array(images)
            y_train = np.array(angles)
            yield X_train, y_train


# Reading CSV file
samples = []
with open('mydata/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        samples.append(line)


# Splitting to train and valiation datasets
train_samples, validation_samples = train_test_split(samples, test_size=0.2)


# Creating the model
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Lambda, ELU, Cropping2D, Conv2D


ch, row, col = 3, 160, 320  # camera format

model = Sequential()

# This architecture is based on the comma.ai architecture
model.add(Cropping2D(cropping=((70,25), (0,0)), input_shape=(row, col, ch)))
model.add(Lambda(lambda x: x/127.5 - 1.))
model.add(Conv2D(16, (8, 8), strides=(4, 4), padding="same"))
model.add(ELU())
model.add(Conv2D(32, (5, 5), strides=(2, 2), padding="same"))
model.add(ELU())
model.add(Conv2D(64, (5, 5), strides=(2, 2), padding="same"))
model.add(Flatten())
model.add(Dropout(.75))
model.add(ELU())
model.add(Dense(512))
model.add(Dropout(.75))
model.add(ELU())
model.add(Dense(1))


# Compile and train the model using the generator function
batch_size = 32
train_generator = generator(train_samples, batch_size)
validation_generator = generator(validation_samples, batch_size)

model.compile(optimizer="adam", loss="mse")

# Train the model
model.fit_generator(train_generator, steps_per_epoch=len(train_samples)/batch_size, 
                    validation_data=validation_generator, 
                    validation_steps=len(validation_samples)/batch_size, epochs=1)
model.save('model.h5')
