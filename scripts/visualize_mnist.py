import matplotlib.pyplot as plt
import cv2
from tensorflow.keras.datasets import mnist
import numpy as np


(X_train, y_train), (X_test, y_test) = mnist.load_data()


def peek_at_data(data, inds_to_display: list):
    for i in range(len(inds_to_display)):
        subplot_string = "33" + str(i+1)
        plt.subplot(int(subplot_string))
        plt.imshow(data[inds_to_display[i]], cmap=plt.get_cmap('gray'))
        # show the plot
    plt.show()


def visualize_number(num):
    cv2.namedWindow(f"Number {num}", cv2.WINDOW_NORMAL)
    mask = y_train == num
    relevant_images = X_train[mask]
    for i in range(10):
        cv2.imshow(f"Number {num}", relevant_images[i])
        cv2.waitKey(0)


def show_image(image=None, windowName = "Image", images=None):
    assert image is not None or images is not None
    if images is not None:
        cv2.imshow("Images", np.hstack([im for im in images]))
    else:
        cv2.imshow(windowName, image)
    cv2.waitKey(0)
