import cv2
import tensorflow.keras.models
from tensorflow.keras.datasets import mnist

import mnist_models
from mnist_models import *
from mnist_utils import add_black, find_top_and_bottom_limits, find_left_and_right_limits
from visualize_mnist import peek_at_data
from tensorflow.keras.utils import to_categorical
import tensorflow as tf


import numpy as np
# import matplotlib.pyplot as plt
import os
import time
import random
import imgaug
# import cv2

join = os.path.join

# import pandas as pd
# df = pd.read_csv("CALIFORNIAN.csv")
# numbers = df.iloc[:, 12:].to_numpy().reshape(-1, 20,20).astype('float32')/255

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

def convert_keras_model_to_tflite(model: tf.keras.models.Model, model_path: str):
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()
    new_path = join("tflite_models", os.path.basename(model_path) + ".tflite")
    with open(new_path, 'wb') as f:
        f.write(tflite_model)


def convert_model_from_path_to_tflite(model_path: str):
    converter = tf.lite.TFLiteConverter.from_saved_model(model_path)
    tflite_model = converter.convert()
    new_path = join("tflite_models", os.path.basename(model_path) + ".tflite")
    with open(new_path, 'wb') as f:
        f.write(tflite_model)


saved_model_path = join(os.getcwd(), "tf_models/large2")
# tensorflow.keras.models.save_model(mnist_models.MNIST_large_model2(), saved_model_path)
# convert_model_from_path_to_tflite(saved_model_path)
# quit()
#

(X_train, y_train), (X_test, y_test) = mnist.load_data()
im_length = 34

print(X_train.shape)
print(X_test.shape)
# mask = (y_train == 1) | (y_train==5) | (y_train==6) | (y_train==7)
mask = y_train != 0
X_train = X_train[mask]
y_train = y_train[mask]
# mask = (y_test == 1) | (y_test == 5) | (y_test == 6) | (y_test == 7)
mask_test = y_test != 0
X_test = X_test[mask_test]
y_test = y_test[mask_test]
print(X_train.shape)
print(X_test.shape)

s = time.time()
print("Adding black padding to mnist images...")
X_train = add_black(X_train, d_size=im_length, old_size=28)
X_test = add_black(X_test, d_size=im_length, old_size=28)
print(f"Added black padding in {time.time() - s} seconds")
print(X_train.shape)
print(X_test.shape)


X_train = X_train.astype('float32')
X_test = X_test.astype('float32')

s = time.time()
print("Converting images to binary...")
for i in range(X_train.shape[0]):
    X_train[i] = cv2.threshold(X_train[i], random.gauss(150, 30), 1.0, cv2.CV_8U)[1].astype('float32')
for i in range(X_test.shape[0]):
    X_test[i] = cv2.threshold(X_test[i], random.gauss(150, 30), 1.0, cv2.CV_8U)[1].astype('float32')
print(f"Finished in {time.time() - s} seconds")

for i in range(40, 200, 9):
    for j in range(i, i+9):
        X_train[j] = cv2.threshold(X_train[j], random.gauss(160, 25), 1, cv2.CV_8U)[1].astype('float32')
    peek_at_data(X_train, [n for n in range(i,i+9)])

num_pixels = X_train.shape[1] * X_train.shape[2]
X_train = X_train.reshape(X_train.shape[0], im_length, im_length, 1).astype('float32')
X_test = X_test.reshape(X_test.shape[0], im_length, im_length, 1).astype('float32')


# one hot encode outputs
print(y_train[0])
# 0 is not included
y_train = to_categorical(y_train-1, num_classes=9)
y_test = to_categorical(y_test-1, num_classes=9)

print(y_train[0], y_train[0].shape)

# tensorflow.keras.callbacks.ModelCheckpoint(saved_model_path, 'val_acc', save_best_only=True, )

model = MNIST_large_model2(im_length)
# model: kerasModel = tensorflow.keras.models.load_model(saved_model_path)
new_model_path = join(os.getcwd(), "tf_models/large2a")
adam = tf.keras.optimizers.Adam(learning_rate=4e-4)
model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])
model.fit(X_train, y_train, validation_data=(X_test, X_test), epochs=20, batch_size=150, verbose=2,
          )

scores = model.evaluate(X_test, y_test, verbose=0)
print("CNN Error: %.2f%%" % (100 - scores[1] * 100))
tensorflow.keras.models.save_model(model, new_model_path)

print(f"Model saved in ${new_model_path}")
