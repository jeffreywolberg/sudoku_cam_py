import math

import cv2

from imutils.perspective import four_point_transform
from skimage.segmentation import clear_border
import numpy as np
import imutils
import os
from visualize_mnist import show_image
import random

join = os.path.join
basename = os.path.basename


class SudokuBoardNotFoundError(Exception):
    def __init__(self, message):
        self.message = message
        super().__init__(self.message)


class ImageProcessor(object):
    @staticmethod
    def get_binary_image(gray, debug=False):
        # gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (7, 7), 3)

        # apply adaptive thresholding and then invert the threshold map
        thresh = cv2.adaptiveThreshold(blurred, 255.0,
                                       cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 15, 2)
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
            raise SudokuBoardNotFoundError("Could not find Sudoku puzzle outline. "
                             "Try debugging your thresholding and contour steps.")
        if debug:
            output = colored_im.copy()
            cv2.drawContours(output, [board_vertices], -1, (0, 255, 0), 2)
            cv2.imshow("Puzzle Outline", output)
            cv2.waitKey(0)
            quit()
        
        return board_vertices

    @staticmethod
    def crop_and_warp(image, vertices, debug=False):
        image = four_point_transform(image, vertices.reshape(4,2))

        if debug:
            cv2.imshow("Warped Image", image)
            key = chr(cv2.waitKey(0))
            cv2.destroyWindow("Warped Image")
            if key.lower() == "q":
                raise SudokuBoardNotFoundError("Finder aborted, can't find board!")
        if image is None:
            raise SudokuBoardNotFoundError("Can't find board!")
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
        cells = np.zeros(shape=(9, 9, im_length, im_length), dtype='uint8')
        for i in range(9):
            for j in range(9):
                im = image[i*width_cell : (i+1)*width_cell, j*height_cell : (j+1)*height_cell]
                if im is None:
                    continue
                # show_image(im)
                # cv2.imshow("Im before processing", cells[i, j]/255)
                # cv2.waitKey(0)
                im = cv2.resize(im, (im_length, im_length))
                im = ImageProcessor.extract_digit(im, debug)
                # print(im[14])
                if im is None: continue
                cells[i][j] = im
        return cells


    @staticmethod
    def extract_digit(cell, debug=False):

        # apply automatic thresholding to the cell and then clear any
        # connected borders that touch the border of the cell

        # a higher threshold makes the digit in the image 'thicker/wider'
        if cell is None:
            return None

        # thresh = cv2.adaptiveThreshold(cell, 255.0, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 3, -2)
        # cv2.adaptiveThreshold(cell, 255)
        # show_image(cell)
        thresh = cv2.threshold(cell, 100, 255.0,
                               cv2.THRESH_BINARY_INV +
                               cv2.THRESH_OTSU
                               # | cv2.THRESH_OTSU
                               )[1]

        if thresh is None:
            return None

        # if debug:
        #     show_image(thresh, windowName="Before clearing border")

        thresh = clear_border(thresh)

        # if debug:
        #     show_image(thresh, windowName="After clearing border")

        if np.count_nonzero(thresh) / (thresh.shape[0] * thresh.shape[1]) < .015: return None

        else: return thresh

        cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
                                cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)
        # if no contours were found than this is an empty cell
        if len(cnts) == 0:
            return None
        # otherwise, find the largest contour in the cell and create a
        # mask for the contour
        c = max(cnts, key=cv2.contourArea)
        mask = np.zeros(thresh.shape, dtype="uint8")
        cv2.drawContours(mask, [c], -1, 255, -1)
        (h, w) = thresh.shape
        percentFilled = cv2.countNonZero(mask) / float(w * h)
        # if less than 3% of the mask is filled then we are looking at
        # noise and can safely ignore the contour
        if percentFilled < 0.015:
            # show_image(images=[mask, thresh, cell])
            return None
        # apply the mask to the thresholded cell
        digit = cv2.bitwise_and(thresh, thresh, mask=mask)
        # show_image(images=[digit, mask, thresh, cell])
        # check to see if we should visualize the masking step

        # return the digit to the calling function
        # print("Max", np.histogram(digit))
        return digit

        # thresh = clear_border(thresh)
        #
        # if debug:
        #     cv2.imshow("Cell Thresh", thresh)
        #     cv2.waitKey(0)
        #
        # return thresh



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
    def find_puzzle(self, image : np.ndarray, im_length=28, debug=False, debug_only_crop_and_warp=False, debug_only_get_individual_boxes=False):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        binary_image = ImageProcessor.get_binary_image(gray, debug)
        vertices = ImageProcessor.find_board_vertices(binary_image, debug=debug, colored_im=image)
        board_image = ImageProcessor.crop_and_warp(gray, vertices, debug or debug_only_crop_and_warp)

        # board_image = cv2.resize(board_image, (im_length*9,im_length*9))
        print(board_image.shape)
        return ImageProcessor.get_individual_boxes(board_image, im_length, debug or debug_only_get_individual_boxes)





if __name__ == "__main__":
    # open_camera()
    solver = SudokuSolver()

    # im_paths = ["test.PNG", "sample_board.jpg", "sample_board2.jpg", "sample_board3.jpg", "sample_board4.jpg", "sample_board5.jpg"]
    # solver.get_and_save_test_images(im_paths, im_length=34)

    im_path = "/Users/jeffrey/Coding/sudoku_cam_py/board_images/IMG-5300.jpg"
    # im_path = "Users/jeffrey/Coding/sudoku_cam_py/board_images/IMG-5318.jpg"
    image = cv2.imread(im_path)
    solver.find_puzzle(image, debug=True, im_length=34)

