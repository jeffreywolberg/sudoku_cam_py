import cv2
import numpy as np

from visualize_mnist import show_image

im = cv2.imread("/Users/jeffrey/Coding/sudoku_cam_py/sudoku_dataset/mixed/image1078.jpg", cv2.IMREAD_GRAYSCALE)
show_image(im)
print(im.shape)
im = np.transpose(im)
show_image(im)