import numpy as np
from tensorflow.keras.datasets import mnist

'''
Average BBox for each digit:
top, bottom, left, right
1: [ 4.33 23.29 10.34 17.65]
2: [ 4.01 22.21  5.7  22.42]
3: [ 4.67 23.5   5.81 20.48]
4: [ 5.01 23.77  6.69 21.09]
5: [ 5.22 23.2   6.43 22.61]
6: [ 2.53 21.43  7.8  21.37]
7: [ 6.77 25.52  6.26 20.18]
8: [ 4.86 23.75  7.58 21.21]
9: [ 6.24 25.19  7.23 20.03]
'''

def __add_black(im_slice : np.ndarray, padding):
    values = np.zeros(padding, np.int)
    im_slice = np.append(im_slice, values)
    im_slice = np.insert(im_slice, 0, values)
    return im_slice


def add_black(images : np.ndarray, padding=3):
    images = np.apply_along_axis(__add_black, 1, images, padding)
    return np.apply_along_axis(__add_black, 2, images, padding)


def find_top_and_bottom_limits(images : np.ndarray) -> np.ndarray:
    # find how many images have at least 20 white pixels, these are the cells with numbers
    images_with_digits = images[np.sum(images == 255, (1, 2)) >= 10]
    num_digits_on_board = images_with_digits.shape[0]
    out = np.zeros((num_digits_on_board, 2))
    for i, image in enumerate(images_with_digits):
        out[i] = find_boundaries_of_digit(image, row=True)
    return out


def find_left_and_right_limits(images: np.ndarray) -> np.ndarray:
    # find how many images have at least 20 white pixels, these are the cells with numbers
    images_with_digits = images[np.sum(images == 255, (1, 2)) >= 10]
    num_digits_on_board = images_with_digits.shape[0]
    out = np.zeros((num_digits_on_board, 2))
    for i, image in enumerate(images_with_digits):
        out[i] = find_boundaries_of_digit(image, col=True)
    return out


def find_boundaries_of_digit(image : np.ndarray, row=False, col=False) -> tuple:
    assert row or col
    s = image.sum(1) if row else image.sum(0)
    lines_w_pixels = np.squeeze(np.argwhere(s >= 255*1))
    if len(lines_w_pixels.shape) == 0:
        return lines_w_pixels, lines_w_pixels # these are numbers in this case
    if not np.any(lines_w_pixels):
        return np.nan, np.nan
    return lines_w_pixels[0], lines_w_pixels[len(lines_w_pixels)-1]

def compute_mnist_bounding_boxes_by_digit(num):
    # top, bottom, left, right
    relevant_images = np.concatenate((X_train[y_train == num],  X_test[y_test == num]))
    return np.round(np.average(np.concatenate((find_top_and_bottom_limits(relevant_images),
                                      find_left_and_right_limits(relevant_images)), axis=1),0), 2)

# def crop_and_scale_images():
#     # peek_at_data(images, [9, 11, 24, 37, 39, 42, 45, 77, 80])
#     top_and_bottom_limits = find_top_and_bottom_limits(images_with_digits)
#     left_and_right_limits = find_left_and_right_limits(images_with_digits)
#
#     # for i, (height_limits, width_limits) in enumerate(zip(top_and_bottom_limits, left_and_right_limits)):
#     #     top, bottom = height_limits
#     #     left, right = width_limits
#     #     print(top, bottom, left, right)
#     #     show_image(images_with_digits[i])
#
#     pct_height_occupied = (top_and_bottom_limits[:, 1] - top_and_bottom_limits[:, 0]) / im_length
#     pct_width_occupied = (left_and_right_limits[:, 1] - left_and_right_limits[:, 0]) / im_length
#     # avg_height, avg_width = np.average(pct_height_occupied), np.average(pct_width_occupied)
#
#     top_avg, bottom_avg = np.round(np.average(top_and_bottom_limits, 0), 2)
#     left_avg, right_avg = np.round(np.average(left_and_right_limits, 0), 2)
#     height_diff = bottom_avg - top_avg
#     width_diff = right_avg - left_avg
#
#     # scale_factor = 19/height_diff
#     # padding_height = int(5/scale_factor)
#     # padding_width = int(7/scale_factor)
#     padding_height = 5
#     padding_width = 7
#
#     print(inds_w_digits.shape, images_with_digits.shape, top_and_bottom_limits.shape, left_and_right_limits.shape)
#     assert inds_w_digits.shape[0] == images_with_digits.shape[0] == top_and_bottom_limits.shape[0] == \
#            left_and_right_limits.shape[0]
#     for i in range(len(inds_w_digits)):
#         top, bottom = int(top_and_bottom_limits[i][0]), int(top_and_bottom_limits[i][1])
#         left, right = int(left_and_right_limits[i][0]), int(left_and_right_limits[i][1])
#         if top >= padding_height and im_length - bottom >= padding_height \
#                 and left >= padding_width and im_length - right >= padding_width:
#             print("Condition is true!!!!!")
#             h_lim = (top - padding_height, bottom + padding_height)
#             w_lim = (left - padding_width, right + padding_width)
#             new_im = images_with_digits[i][h_lim[0]:h_lim[1], w_lim[0]:w_lim[1]]
#             print(h_lim, w_lim, new_im.shape)
#             new_im_scaled = cv2.resize(new_im, (im_length, im_length))
#             print(new_im.shape, new_im_scaled.shape)
#             show_image(images_with_digits[i], windowName="Before cropping")
#             show_image(new_im_scaled, windowName="After cropping")
#             images[inds_w_digits[i]] = new_im_scaled
#         else:
#             print("failed!")
#
#     # if width ratio of mnist to cell is 1.5
#     # and width and height padding_height are 6 and 4, respectively
#     # padding_height of cell must be 6/1.5 and 4/1.5, because when cell
#     # is expanded by 1.5 so that digit area matches mnist digit area,
#     # the paddings must also be similar




if __name__ == "__main__":
    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    bbox_stats = np.zeros((9,4))
    for i in range(9):
        bbox_stats[i] = compute_mnist_bounding_boxes_by_digit(i+1)
    print(bbox_stats)

