import csv
import cv2
import numpy as np
import h5py 
import scipy.misc

lines = []
with open('./datatrain_raw/driving_log.csv') as csvfile:
	reader = csv.reader(csvfile)
	for line in reader:
		lines.append(line)

images = []
measurements = []
for line in lines:
	for i in range(3):
		source_path = line[i]
		filename = source_path.split('/')[-1]
		current_path = './datatrain_raw/IMG/' + filename
		image = cv2.imread(current_path)
		image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
		correction = 0
		throttle = float(line[4])
		if i==1:
			throttle = throttle / 1.2
			correction = 0.20
		elif i==2:
			throttle = throttle / 1.2
			correction = -0.20
		measurement= [ float(line[3]) + correction, throttle]
		images.append(image)
		measurements.append(measurement)


X_train = np.array(images)
Y_train = np.array(measurements)

from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Cropping2D, Dropout
from keras.layers import Convolution2D
from keras.layers.pooling import MaxPooling2D
import tensorflow as tf
import PIL

model = Sequential()
model.add(Lambda(lambda x: x / 127.5 - 1, input_shape=(160,320,3)))
model.add(Cropping2D(cropping=((65,25),(0,0))))
model.add(Convolution2D(24, 5, 5,subsample=(2,2),activation="relu"))
model.add(Convolution2D(36, 5, 5,subsample=(2,2),activation="relu"))
model.add(Convolution2D(48, 5, 5,subsample=(2,2),activation="relu"))
model.add(Convolution2D(64, 3, 3,activation="relu"))
model.add(Convolution2D(64, 3, 3,activation="relu"))
model.add(Flatten())
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(20))
model.add(Dense(2))

model.compile(loss="mse", optimizer="adam")
history_object = model.fit(X_train, Y_train, validation_split=0.2, shuffle= True, nb_epoch=4, verbose=1)

model.save('model.h5')

import matplotlib.pyplot as plt

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
exit()