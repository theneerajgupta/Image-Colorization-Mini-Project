import os
import cv2
import pickle
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras import datasets, layers, models, utils
from tensorflow.keras.layers import *
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import to_categorical


DIMENSION = (1024, 512)
HEIGHT = 512
WIDTH = 1024
BATCH = 1
EPOCHS = 3


def load_dataset(file) :
	print("loading", file, "...")
	temp = []
	with open(file, "rb") as data :
		temp = pickle.load(data)
	return temp


X = []
Y = []
DS = load_dataset("datasets/combined.pickle")

for image in DS :
	b1, g1, r1, b2, g2, r2 = cv2.split(image)
	img1 = np.stack([b1, g1, r1], axis=2)
	X.append(cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY))
	Y.append(image)


# print("reshaping X and Y to (-1, height, width, x)...")
X = np.array(X)
Y = np.array(Y)
X = X.reshape(-1, HEIGHT, WIDTH, 1)
Y = Y.reshape(-1, HEIGHT, WIDTH, 6)


# print("normalizing X and Y...")
X = X * (1.0 / 255)
Y = Y * (1.0 / 255)


print("Shapes of X and Y >>>")
print("X :", X.shape)
print("Y :", Y.shape)


print("\n\nbuilding model...")
model = Sequential()
model.add(InputLayer(input_shape=(512, 1024, 1)))
model.add(Conv2D(100, (3, 3), activation="relu", padding="same"))
model.add(Conv2D(100, (3, 3), activation="relu", padding="same", strides=(2, 2)))
model.add(MaxPooling2D((2,2), strides=(2,2)))
model.add(Conv2D(200, (3, 3), activation="relu", padding="same"))
model.add(Conv2D(200, (3, 3), activation="relu", padding="same"))
model.add(Conv2D(400, (3, 3), activation="relu", padding="same", strides=(2, 2)))
model.add(MaxPooling2D((2,2), strides=(2,2)))
model.add(UpSampling2D((2, 2)))
model.add(Conv2D(200, (3, 3), activation='relu', padding='same'))
model.add(UpSampling2D((2, 2)))
model.add(Conv2D(100, (3, 3), activation='relu', padding='same'))
model.add(UpSampling2D((2, 2)))
model.add(Conv2D(50, (3, 3), activation='relu', padding='same'))
model.add(Conv2D(6, (3, 3), activation='tanh', padding='same'))
model.add(UpSampling2D((2, 2)))
model.summary()
model.compile(optimizer='rmsprop', loss='mse')


print("\n\ntraining model...")
model.fit(
		x = X,
		y = Y,
		batch_size = BATCH,
		epochs = EPOCHS,
	)


test = cv2.imread("test.jpg", cv2.IMREAD_GRAYSCALE)
test = cv2.resize(test, DIMENSION)
test = test * (1.0/255)
test = test.reshape(-1, 512, 1024, 1)
print(test.shape)


output = model.predict(test)
output = output.reshape(512, 1024, 6)
a, b, c, d, e, f = cv2.split(output)
output = np.stack([a, b, c], axis=2)
output = output * 255
output = output.astype(int)


cv2.imwrite("output.jpg", output)