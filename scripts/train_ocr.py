import cv2
import tensorflow.keras.models
from tensorflow.keras.datasets import mnist
from visualize_mnist import show_image
import mnist_models
from mnist_models import *
from mnist_utils import add_black, find_top_and_bottom_limits, find_left_and_right_limits
from visualize_mnist import peek_at_data
from tensorflow.keras.utils import to_categorical
import tensorflow as tf
import glob
import tqdm


import numpy as np
# import matplotlib.pyplot as plt
import os
import time
import random
import imgaug
# import cv2

join = os.path.join
dirname = os.path.dirname
basename = os.path.basename

# import pandas as pd
# df = pd.read_csv("CALIFORNIAN.csv")
# numbers = df.iloc[:, 12:].to_numpy().reshape(-1, 20,20).astype('float32')/255

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

def clamp(n, smallest, largest):
    return max(smallest, min(n, largest))

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

def get_label(txt_file):
    with open(txt_file, "r") as f:
        return int(f.readline())

def load_custom_data(data_folder):
    labels = []
    imgs = []
    txt_files = glob.glob(join(data_folder, "*.txt"))
    print(f"Loading data from {data_folder}...")
    for txt_file in tqdm.tqdm(txt_files):
        im_path = txt_file[:-4] + ".png"
        if not os.path.exists(im_path):
            continue
        im = cv2.imread(im_path, cv2.IMREAD_GRAYSCALE)
        label = get_label(txt_file)
        imgs.append(im)
        labels.append(label)
    return np.array(imgs), np.array(labels)

def write_model_report(model_version, folder, train_imgs, test_imgs, mnist_train_num, mnist_test_num, train_dirs, test_dirs, im_length, lr, epochs, batch_size, final_error):
    with open(join(folder, "model_report.txt"), "a+") as f:
        msg = f"Model version {str(model_version)}\n\n"
        msg += "Used ["
        for train_dir in train_dirs:
            msg += f"{train_dir}, "
        msg += f"]\nas training images, {train_imgs.shape[0]} in total, MNIST count: {mnist_train_num}\n\n"
        msg += "Used "
        for test_dir in test_dirs:
            msg += f"{test_dir}, "
        msg += f"\nas testing images, {test_imgs.shape[0]} in total, MNIST count: {mnist_test_num}\n"
        msg += f"\nLearning rate: {lr}, epochs: {epochs}, batch size: {batch_size}, error on test set: {round(final_error, 2)}%\n"
        msg += f"Image length: {im_length}\n"
        f.write(msg)

def reshape_and_normalize_data(X_train, X_test, train_imgs, test_imgs):
    X_train = X_train.reshape((X_train.shape[0], im_length, im_length, 1)).astype('float32')
    if np.max(X_train) == 255.0:
        print("Normalizing X_train")
        X_train /= 255.0
    X_test = X_test.reshape((X_test.shape[0], im_length, im_length, 1)).astype('float32')
    if np.max(X_test) == 255.0:
        print("Normalizing X_test")
        X_test /= 255.0
    train_imgs = train_imgs.reshape((train_imgs.shape[0], im_length, im_length, 1)).astype('float32')
    if len(train_imgs) > 0 and np.max(train_imgs) == 255.0:
        print("Normalizing train_imgs")
        train_imgs /= 255.0
    test_imgs = test_imgs.reshape((test_imgs.shape[0], im_length, im_length, 1)).astype('float32')
    if len(test_imgs) > 0 and np.max(test_imgs) == 255.0:
        print("Normalizing test_imgs")
        test_imgs /= 255.0
    return X_train, X_test, train_imgs, test_imgs

def get_specific_digits(imgs, labels, digits: list=None, digits_not: list=None):
    assert digits_not is None or digits is None
    mask = np.zeros(shape=labels.shape, dtype=bool)
    if digits is not None:
        for digit in digits:
            mask[labels == digit] = True
    elif digits_not is not None:
        for digit in digits_not:
            mask[labels != digit] = True
    else:
        raise NotImplementedError

    return imgs[mask], labels[mask]

im_length = 34

train_imgs = []
train_labels = []
test_imgs = []
test_labels = []

# LOAD CUSTOM DATA
train_dirs = [
    # join(dirname(os.getcwd()), "augmented_digit_images2"),
    # join(dirname(os.getcwd()), "augmented_training_set2"),
    # join(dirname(os.getcwd()), "training_set2"),
             ]
test_dirs = [
    join(dirname(os.getcwd()), "training_set2"),
             ]

for data_dir in train_dirs:
    imgs, labels = load_custom_data(data_dir)
    train_imgs.extend(imgs)
    train_labels.extend(labels)
train_imgs, train_labels = np.array(train_imgs), np.array(train_labels)

for test_dir in test_dirs:
    imgs, labels = load_custom_data(test_dir)
    test_imgs.extend(imgs)
    test_labels.extend(labels)
test_imgs, test_labels = np.array(test_imgs), np.array(test_labels)

(X_train, y_train), (X_test, y_test) = mnist.load_data()

# digits = [1,8]
# X_train, y_train = get_specific_digits(X_train, y_train, digits=digits)
# X_test, y_test = get_specific_digits(X_test, y_test, digits=digits)
# train_imgs, train_labels = get_specific_digits(train_imgs, train_labels, digits=digits)
# test_imgs, test_labels = get_specific_digits(test_imgs, test_labels, digits=digits)

X_train, y_train = get_specific_digits(X_train, y_train, digits_not=[0])
X_test, y_test = get_specific_digits(X_test, y_test, digits_not=[0])


# X_train, y_train = get_specific_digits(X_train, y_train, digits_not=[0])
# X_test, y_test = get_specific_digits(X_test, y_test, digits_not=[0])

print("X_train shape", X_train.shape)
print("X_test shape", X_test.shape)
print("Custom train images shape", train_imgs.shape)
print("Custom test images shape", test_imgs.shape)



s = time.time()
print("Adding black padding to mnist images...")
X_train = add_black(X_train, d_size=im_length, old_size=28)
X_test = add_black(X_test, d_size=im_length, old_size=28)
print(f"Added black padding in {time.time() - s} seconds")

print(f"Mnist train shape {X_train.shape}")
print(f"Mnist test shape {X_test.shape}")

s = time.time()
print("Converting images to binary...")
for i in range(X_train.shape[0]):
    X_train[i] = cv2.threshold(X_train[i], clamp(random.gauss(120, 50), 50, 200), 1.0, cv2.CV_8U)[1].astype('float32')
for i in range(X_test.shape[0]):
    X_test[i] = cv2.threshold(X_test[i], clamp(random.gauss(120, 50), 50, 200), 1.0, cv2.CV_8U)[1].astype('float32')
for i in range(train_imgs.shape[0]):
    train_imgs[i] = cv2.threshold(train_imgs[i], clamp(random.gauss(120, 50), 50, 200), 1.0, cv2.CV_8U)[1].astype('float32')
for i in range(test_imgs.shape[0]):
    test_imgs[i] = cv2.threshold(test_imgs[i], clamp(random.gauss(120, 50), 50, 200), 1.0, cv2.CV_8U)[1].astype('float32')
print(f"Finished in {time.time() - s} seconds")

# for i in range(34):
#     print(train_imgs[101][i])
# im_range = [100,101, 102, 103, 104, 105, 106, 107, 108]
# peek_at_data(X_train, im_range)
# peek_at_data(X_test, im_range)
# peek_at_data(train_imgs, im_range)
# peek_at_data(test_imgs, im_range)

X_train, X_test, train_imgs, test_imgs = reshape_and_normalize_data(X_train, X_test, train_imgs, test_imgs)

print(train_imgs.shape, X_train.shape, test_imgs.shape, X_test.shape)
split_train = X_train.shape[0]
# split_train = 5000
split_test = X_test.shape[0]
# split_test =
train_imgs = np.concatenate((train_imgs, X_train[:split_train]))
train_labels = np.concatenate((train_labels, y_train[:split_train]))
test_imgs = np.concatenate((test_imgs, X_test[:split_test]))
test_labels = np.concatenate((test_labels, y_test[:split_test]))

assert np.max(X_train) == 1.0, f"Max is {np.max(X_train)}, it must be 1.0"
assert np.max(X_test) == 1.0, f"Max is {np.max(X_test)}, it must be 1.0"
assert np.max(train_imgs) == 1.0, f"Max is {np.max(train_imgs)}, it must be 1.0"
assert np.max(test_imgs) == 1.0, f"Max is {np.max(test_imgs)}, it must be 1.0"

# for i in range(0, 3000, 9):
#     peek_at_data(train_imgs, [n for n in range(i,i+9)])
# quit()

# one hot encode outputs, 0 is not included
y_train = to_categorical(y_train-1, num_classes=9)
y_test = to_categorical(y_test-1, num_classes=9)

# custom_labels_encoded = to_categorical(custom_labels-1, num_classes=9)
train_labels_encoded = to_categorical(train_labels-1, num_classes=9)
test_labels_encoded = to_categorical(test_labels-1, num_classes=9)

print(f"Train images length {train_imgs.shape[0]}")
print(f"Test images length {test_imgs.shape[0]}")

# tensorflow.keras.callbacks.ModelCheckpoint(saved_model_path, 'val_acc', save_best_only=True, )

# model = MNIST_large_model2(im_length)
retrain = True
from_retrained_model = False
# train_number = ""
train_number = "0"
retrain_letter = "a"
model_version = MNIST_large_model
model_version_name = str(model_version).split("function ")[1].split(" at")[0]
# model = MNIST_model_example(im_length)

if retrain:
    if from_retrained_model:
        saved_model_path = join(f"/Users/jeffrey/Coding/sudoku_cam_py/tf_models/{model_version_name}_{train_number}{chr(ord(retrain_letter)-1)}")
    else:
        saved_model_path = join(f"/Users/jeffrey/Coding/sudoku_cam_py/tf_models/{model_version_name}_{train_number}")
    model: kerasModel = tensorflow.keras.models.load_model(saved_model_path)
else:
    model = model_version(im_length)

# print(model.weights)
# opt = SGD(learning_rate=4e-4, momentum=0.9)
# model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['accuracy'])
lr = 1e-4
epochs = 8
batch_size = 200
model.compile(optimizer=Adam(learning_rate=lr), loss='categorical_crossentropy', metrics=['accuracy'])
if retrain:
    new_model_path = f"/Users/jeffrey/Coding/sudoku_cam_py/tf_models/{model_version_name}_{train_number}{retrain_letter}"
else:
    new_model_path = f"/Users/jeffrey/Coding/sudoku_cam_py/tf_models/{model_version_name}_{train_number}"

# adam = tf.keras.optimizers.Adam(learning_rate=1e-3)
# model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])
# split_at_test = 1000
# model.fit(X_train[split_at_train:], y_train[split_at_train:], validation_data=(X_test[split_at_test:], y_test[split_at_test:]), epochs=20, batch_size=150, verbose=2,
#           )
# model.fit(train_imgs, train_labels_encoded,
#           validation_data=(X_test, y_test), epochs=10, batch_size=50, verbose=2,
#           )
# model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=10, batch_size=128, verbose=2,
#           )

assert not os.path.isdir(new_model_path), f"Model path {new_model_path} must not already exist"
model.fit(train_imgs, train_labels_encoded,
          validation_data=(test_imgs, test_labels_encoded), epochs=epochs, batch_size=batch_size, verbose=2,
          )

scores = model.evaluate(test_imgs, test_labels_encoded, verbose=0)
err = 100 - scores[1] * 100
print("CNN Error: %.2f%%" % (err))
tensorflow.keras.models.save_model(model, new_model_path)

print(f"Model saved in ${new_model_path}")
write_model_report(model_version_name, new_model_path, train_imgs, test_imgs, split_train, split_test, train_dirs, test_dirs, im_length, lr, epochs, batch_size, err)
