# import sys
# sys.path.append("/Users/jeffrey/Coding/sudoku_cam_py/mnist-competition")
from tensorflow.keras.models import Model as kerasModel
from tensorflow.keras.models import load_model
from tensorflow.keras.datasets import mnist


from board_finder import SudokuSolver
from mnist_utils import add_black
from visualize_mnist import peek_at_data, show_image
from mnist_utils import find_top_and_bottom_limits, find_left_and_right_limits

import os
import cv2
import numpy as np
import time

# tensorflow.lite.python.lite.TFLite

# path = "/Users/jeffrey/Coding/sudoku_cam_py/sudoku_dataset/mixed/image1005.jpg"
path = "/Users/jeffrey/Coding/sudoku_cam_py/sudoku_dataset/mixed/image1004.jpg"
# im = cv2.imread("/Users/jeffrey/Coding/sudoku_cam_py/sudoku_dataset/mixed/image1083.jpg")
im = cv2.imread(path)
# im = cv2.imread(r"/Users/jeffrey/Coding/sudoku_cam_py/sudoku_dataset/images/image1084.jpg")
# im = cv2.imread("/Users/jeffrey/Coding/sudoku_cam_py/sudoku_dataset/mixed/image1083.jpg")

# sample_board, (28x28) avg bounding box of digit (top, bottom, left, right): 6.3 20.94 8.88 17.3, (14.5, 8.5)
# sample_board2, (28x28) avg bounding box of digit (top, bottom, left, right): 8.62 18.4 10.22 16.02 (10, 6)
# sample_board3, (28x28) avg bounding box of digit (top, bottom, left, right): 5.44 22.97 8.91 19.16 (17.5, 10)
# sample_board4, (28x28) avg bounding box of digit (top, bottom, left, right): 5.59 18.71 9.18 16.35 (13, 7)
# mnist digits, (28x28) avg bounding box of digit (top, bottom, left, right): 4.68 23.43 6.73 21.15 (19, 14.5)

# Need to input sudoku digits such that they match up with the size of the trained mnist digits
# Struggles for 5s vs 6s and 7s vs 1s


# 8.0 29.0 11.0 24.0

# cv2.imshow("Image", im)
# cv2.waitKey(0)
join = os.path.join

s = time.time()
saved_model_path = "/Users/jeffrey/Coding/sudoku_cam_py/tf_models/MNIST_large_model_1a"
print(saved_model_path)
model : kerasModel = load_model(saved_model_path)
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
print("Time to load model: ", time.time()-s, "seconds")

# show_image(im)


s = time.time()
solver = SudokuSolver()
im_length = 34
images = solver.find_puzzle(im, im_length, debug=False, debug_only_get_individual_boxes=True)
board_image = np.zeros(shape=(9*im_length, 9*im_length))

debug = True
if debug:
    for i in range(9):
        for j in range(9):
            board_image[i*im_length:(i+1)*im_length, j*im_length:(j+1)*im_length] = images[i, j]
    cv2.imshow("Image", board_image)

images = np.reshape(images, (81, im_length, im_length)).astype("float32")
if np.max(images) == 255.0:
    images /= 255.0

mask = np.sum(images == 1.0, (1, 2)) >= 15
inds_w_digits = np.squeeze(np.argwhere(mask))
images_with_digits = images[mask]
# for i in range(len(images_with_digits)):
#     images_with_digits[i] = cv2.erode(images_with_digits[i], kernel=(2, 2))

# peek_at_data(images_with_digits, [i for i in range(10,19)])

# # print(images_with_digits.shape)
# (images, labels), (_, _) = mnist.load_data()
# images = add_black(images, d_size=im_length, old_size=28)
# print(images.shape)
# images = images[:81].reshape((81, im_length, im_length)).astype("float32") / 255.0

res = model.predict(
    np.expand_dims(images, 3).astype('float32'))

print("Time to find board and predict digits: ", time.time() - s, "seconds")
for i, r in enumerate(res):
    print(f"{i}: ", end="")
    # if i in [22, 28, 32, 40, 44, 52, 60, 68]:
    if np.max(r) >= .4 and i in inds_w_digits:
        print(np.argmax(r) + 1, end=" : ")
    else:
        print("No number", end=" : ")
    print("[", end="")
    for num in r:
        print(round(num, 2), end=", ")
    print("]")
    # show_image(images[i])

cv2.waitKey(0)

    # if i in [9, 11, 24, 37, 39, 42, 45, 77, 80]:
    #     cv2.imshow("Num", images[i])
    #     cv2.waitKey(0)


# with smaller model, 40x40 images 10 epochs 200 batch size
# [22, 28, 32, 40, 44, 52, 60, 68]
# with medium model, 40x40 images 30 epochs 200 batch size, .165 sec to load model .128 secs to find board and predict digits
# [40, 60, 80]
# with large model, 15 epochs, 75 batch size, 34x34 images, .255 sec to load model, .103 secs for inference
# [4, 40, 44], messed up on 5 and 6's
# large model2, 20 epochs batch size 150, 34x34 images, .255 sec to load, .107 sec for inference, no mistakes on sample_board.jpg
# however, many mistakes on sample board 2 and 3, if numbers are too small in board, it will fail, and it confuses 1s and 7s and 0s w 8s, 9s and 6s.
# Need to remove 0 from training set and need to scale all images to be the same size before going into the network



