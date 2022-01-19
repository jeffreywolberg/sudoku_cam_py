import matplotlib.pyplot as plt
import cv2
from tensorflow.keras.datasets import mnist
import numpy as np
import tensorflow


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
    return cv2.waitKey(0)


def visualize_augmented_MNIST_images(datagen: tensorflow.keras.preprocessing.image.ImageDataGenerator):
    # define number of rows & columns
    num_row = 4
    num_col = 8
    num = num_row * num_col

    # plot before
    print('BEFORE:\n')
    # plot images
    fig1, axes1 = plt.subplots(num_row, num_col, figsize=(1.5 * num_col, 2 * num_row))
    for i in range(num):
        ax = axes1[i // num_col, i % num_col]
        ax.imshow(255-X_train[i], cmap='gray_r')
        ax.set_title('Label: {}'.format(y_train[i]))
    fig1.canvas.manager.set_window_title('Before Augmentations')
    plt.tight_layout()
    plt.show()

    # plot after
    print('AFTER:\n')
    fig2, axes2 = plt.subplots(num_row, num_col, figsize=(1.5 * num_col, 2 * num_row))
    for X, Y in datagen.flow(255-X_train.reshape(X_train.shape[0], 28, 28, 1),
                             y_train.reshape(y_train.shape[0], 1),
                             batch_size=num,
                             shuffle=False):
        for i in range(0, num):
            ax = axes2[i // num_col, i % num_col]
            ax.imshow(X[i].reshape(28, 28), cmap='gray_r')
            ax.set_title('Label: {}'.format(int(Y[i])))
        break
    fig2.canvas.manager.set_window_title('After Augmentations')
    plt.tight_layout()
    plt.show()
