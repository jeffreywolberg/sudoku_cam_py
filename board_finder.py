import math

import cv2

from imutils.perspective import four_point_transform
from skimage.segmentation import clear_border
import numpy as np
import imutils
import os

join = os.path.join
basename = os.path.basename

class ImageProcessor(object):
    @staticmethod
    def get_binary_image(gray, debug=False):
        # gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (7, 7), 3)

        # apply adaptive thresholding and then invert the threshold map
        thresh = cv2.adaptiveThreshold(blurred, 255,
                                       cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
        binary = cv2.bitwise_not(thresh)

        if debug:
            cv2.imshow("Image Processing", np.hstack([gray, blurred, thresh, binary]))
            cv2.waitKey(0)

        return binary

    @staticmethod
    def find_board_vertices(binary_im, debug=False, colored_im=None):
        assert (debug and colored_im is not None) or (not debug)
        # determine contours (the distinct and continuous edges)
        cnts = cv2.findContours(binary_im.copy(), cv2.RETR_EXTERNAL,
                                cv2.CHAIN_APPROX_SIMPLE)
        cnts : list = imutils.grab_contours(cnts)

        
        # sort them by the area that they surround (in decreasing order)
        cnts = sorted(cnts, key=cv2.contourArea, reverse=True)

        board_vertices = None
        for c in cnts:
            # distill the contour into distinct vertices (the longer the contour the more vertices it will have)
            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, peri * .02, True)

            # if we have four vertices, we have probably found the outline of puzzle
            if len(approx) == 4:
                board_vertices = approx
                break

        if board_vertices is None:
            raise Exception(("Could not find Sudoku puzzle outline. "
                             "Try debugging your thresholding and contour steps."))
        if debug:
            output = image.copy()
            cv2.drawContours(output, [board_vertices], -1, (0, 255, 0), 2)
            cv2.imshow("Puzzle Outline", output)
            cv2.waitKey(0)
        
        return board_vertices

    @staticmethod
    def crop_and_warp(image, vertices, debug=False):
        image = four_point_transform(image, vertices.reshape(4,2))
        if debug:
            cv2.imshow("Warped Image", image)
            cv2.waitKey(0)
        return image

    @staticmethod
    def get_individual_boxes(image : np.ndarray, im_length=28, debug=False):
        # width, height = image.shape[0], image.shape[1]
        # print(image.shape)
        # print((width - width%9, height - height%9))
        # image = cv2.resize(image, (width - width%9, height - height%9))
        # cv2.imshow("Image", image)
        width, height = image.shape[0], image.shape[1]
        width_cell, height_cell = int(width / 9), int(height / 9)
        # print(width_cell, height_cell)
        # cells = np.zeros(shape=(9, 9, width_cell, height_cell))
        cells = np.zeros(shape=(9, 9, im_length, im_length))
        for i in range(9):
            for j in range(9):
                im = image[i*width_cell : (i+1)*width_cell, j*height_cell : (j+1)*height_cell]
                # cv2.imshow("Im before processing", cells[i, j]/255)
                # cv2.waitKey(0)
                digit_im = ImageProcessor.extract_digit(im.astype("uint8"), debug)
                cells[i][j] = cv2.resize(digit_im, (im_length, im_length))
        return cells


    @staticmethod
    def extract_digit(cell, debug=False):
        # apply automatic thresholding to the cell and then clear any
        # connected borders that touch the border of the cell

        # a higher threshold makes the digit in the image 'thicker/wider'
        thresh = cv2.threshold(cell, 170, 255,
                               cv2.THRESH_BINARY_INV
                               # | cv2.THRESH_OTSU
                               )[1]

        # thresh = cv2.adaptiveThreshold(cell, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)



        # print(thresh.shape)
        thresh = clear_border(thresh)
        if debug:
            cv2.imshow("Thresh after clearing borders", thresh)
            cv2.waitKey(0)
        return thresh

        # # check to see if we are visualizing the cell thresholding step
        # if debug:
        #     cv2.imshow("Cell Thresh", thresh)
        #     cv2.waitKey(0)
        #
        # # find contours in the thresholded cell
        # cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
        #                         cv2.CHAIN_APPROX_SIMPLE)
        # cnts = imutils.grab_contours(cnts)
        # if len(cnts) == 0:
        #     return None
        # # otherwise, find the largest contour in the cell and create a
        # # mask for the contour
        # c = max(cnts, key=cv2.contourArea)
        # mask = np.zeros(thresh.shape, dtype="uint8")
        # cv2.drawContours(mask, [c], -1, 255, -1)
        # if debug:
        #     cv2.imshow("Mask", mask)
        #     cv2.waitKey(0)
        #
        # (h, w) = thresh.shape
        # percentFilled = cv2.countNonZero(mask) / float(w * h)
        # # if less than 3% of the mask is filled then we are looking at
        # # noise and can safely ignore the contour
        # if percentFilled < 0.03:
        #     return None
        # # apply the mask to the thresholded cell
        # digit = cv2.bitwise_and(thresh, thresh, mask=mask)
        # # check to see if we should visualize the masking step
        # if debug:
        #     cv2.imshow("Digit", 255-digit)
        #     cv2.waitKey(0)
        # # return the digit to the calling function
        # return digit


class SudokuSolver(object):
    def find_puzzle(self, image : np.ndarray, im_length=28, debug=False):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        binary_image = ImageProcessor.get_binary_image(image, debug)
        vertices = ImageProcessor.find_board_vertices(binary_image, debug=debug, colored_im=image)
        board_image = ImageProcessor.crop_and_warp(image, vertices, debug)
        # board_image = cv2.resize(board_image, (im_length*9,im_length*9))
        print(board_image.shape)
        return ImageProcessor.get_individual_boxes(board_image, im_length, debug)

    def get_and_save_test_images(self, sudoku_img_paths : list, im_length=28):
        dir_name = "test_set"
        directory = join(os.getcwd(), dir_name)
        print(directory)
        if not os.path.isdir(directory):
            print(f"Making directory... {directory}")
            os.mkdir(directory)
        for im_path in sudoku_img_paths:
            imgs = self.find_puzzle(cv2.imread(im_path), im_length=im_length).reshape((81, im_length, im_length))
            for i in range(81):
                im = imgs[i]
                cv2.imwrite(join(dir_name, basename(im_path) + f"_im{i}.png"), im)
            print(f"Saved 81 images for sudoku board in {basename(im_path)}")



if __name__ == "__main__":
    # open_camera()
    solver = SudokuSolver()

    # im_paths = ["test.PNG", "sample_board.jpg", "sample_board2.jpg", "sample_board3.jpg", "sample_board4.jpg", "sample_board5.jpg"]
    # solver.get_and_save_test_images(im_paths, im_length=34)

    # im_path = "./test.PNG"
    # image = cv2.imread(im_path)
    # solver.find_puzzle(image, debug=True, im_length=34)
