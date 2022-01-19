import re

import cv2
import imgaug.augmenters as iaa
import numpy as np
import glob
import os
from visualize_mnist import show_image, visualize_augmented_MNIST_images
import random
import tqdm
from tensorflow.keras.datasets import mnist
from train_ocr import get_specific_digits
import matplotlib.pyplot as plt

join = os.path.join
basename = os.path.basename

pad = iaa.Sequential(
    iaa.CropAndPad(px=(0, 5))
)
crop = iaa.Sequential(
    iaa.CropAndPad(px=(-5, 0))
)

def write_new_label_path(old_label_path, new_label_path):
    try:
        with open(old_label_path, "r") as old_f:
            label = int(old_f.readline().strip())
    except FileNotFoundError:
        return False
    with open(new_label_path, "w") as f:
        f.write(str(label))
    return True

class Augmenter(object):
    def __init__(self, data_folder, im_length):
        self.data_folder = data_folder
        self.new_data_folder = rf"/Users/jeffrey/Coding/sudoku_cam_py/augmented_{os.path.basename(self.data_folder)}"
        self.im_length = im_length


    def augment_images(self):
        total = 0
        images_glob = glob.glob(join(self.data_folder, "*.png"))
        images = [cv2.imread(file) for file in images_glob]
        image_paths = [file for file in images_glob]
        cropped_set = [crop(images=images), crop(images=images), crop(images=images), crop(images=images)]
        padded_set = [pad(images=images), pad(images=images), pad(images=images), pad(images=images)]
        print("Saving augmented images to", self.new_data_folder)
        if not os.path.isdir(self.new_data_folder): os.mkdir(self.new_data_folder)

        for set_num, cropped in enumerate(cropped_set):
            num_written = 0
            for i in range(len(cropped)):
                img_path = image_paths[i]
                file_path = join(self.new_data_folder, basename(img_path[:-4]) + f"_cropped_{set_num}.png")
                if os.path.exists(file_path): continue
                new_label_path = file_path[:-4] + ".txt"
                label_path = img_path[:-4] + ".txt"
                did_write = write_new_label_path(label_path, new_label_path)
                if not did_write: continue
                cv2.imwrite(file_path, cropped[i])

                num_written += 1
            print(f"Wrote {num_written} cropped images to augmented training set")
            total += num_written

        for set_num, padded in enumerate(padded_set):
            num_written = 0
            for i in range(len(padded)):
                img_path = image_paths[i]
                file_path = join(self.new_data_folder, basename(img_path[:-4]) + f"_padded_{set_num}.png")
                if os.path.exists(file_path): continue
                cv2.imwrite(file_path, padded[i])
                new_label_path = file_path[:-4] + ".txt"
                label_path = img_path[:-4] + ".txt"
                write_new_label_path(label_path, new_label_path)
                num_written += 1
            total += num_written
            print(f"Wrote {num_written} padded images to augmented training set")


        for set_num, cropped in enumerate(cropped_set):
            num_written = 0
            for i in range(len(cropped)):
                img_path = image_paths[i]
                file_path = join(self.new_data_folder, basename(img_path[:-4]) + f"_cropped_eroded_{set_num}.png")
                if os.path.exists(file_path): continue
                im = pad(image=cv2.cvtColor(cropped[i], cv2.COLOR_BGR2GRAY))
                out = cv2.erode(im, kernel=np.ones((2,2)))
                out = cv2.threshold(out, 20, 255, type=cv2.CV_8U)[1]
                cv2.imwrite(file_path, out)
                new_label_path = file_path[:-4] + ".txt"
                label_path = img_path[:-4] + ".txt"
                write_new_label_path(label_path, new_label_path)
                num_written += 1
            total += num_written
            print(f"Wrote {num_written} eroded cropped images to augmented training set")

        for set_num, padded in enumerate(padded_set):
            num_written = 0
            for i in range(len(padded)):
                img_path = image_paths[i]
                file_path = join(self.new_data_folder, basename(img_path[:-4]) + f"_padded_eroded_{set_num}.png")
                if os.path.exists(file_path): continue
                im = pad(image=cv2.cvtColor(padded[i], cv2.COLOR_BGR2GRAY))
                out = cv2.erode(im, kernel=np.ones((1,2)))
                out = cv2.threshold(out, 20, 255, type=cv2.CV_8U)[1]
                cv2.imwrite(file_path, out)
                new_label_path = file_path[:-4] + ".txt"
                label_path = img_path[:-4] + ".txt"
                write_new_label_path(label_path, new_label_path)
                num_written += 1
            total += num_written
            print(f"Wrote {num_written} eroded padded images to augmented training set")

        print(f"Saved a total of {total} augmented images")

    def show_diff_btwn_augmented_images(self, digit=None):
        im_paths = [glob.glob(join(self.new_data_folder, f"*{ext}")) for ext in [".png", ".jpg"]]
        im_paths = [p for paths in im_paths for p in paths]
        for im_p in im_paths:
            board_image = os.path.basename(re.split("_im\d", im_p)[0])
            for i in range(81):
                same_img_paths = [glob.glob(join(self.new_data_folder, board_image + f"_im{i}_*{ext}")) for ext in ["png", "jpg"]]
                same_img_paths =  [p for paths in same_img_paths for p in paths]
                if len(same_img_paths) == 0: continue
                # print(len(same_img_paths), same_img_paths)
                with open(same_img_paths[0][:-4]+".txt", 'r') as f:
                    label = int(f.readline())
                if digit is not None and label != digit: continue
                window_len = int(np.sqrt(len(same_img_paths)))
                imgs = np.zeros(shape=(window_len*self.im_length, window_len*self.im_length), dtype='uint8')
                for i in range(window_len):
                    for j in range(window_len):
                        im_num = window_len*i+j
                        imgs[i*self.im_length:(i+1)*self.im_length, j*self.im_length:(j+1)*self.im_length,] = cv2.imread(same_img_paths[im_num], cv2.IMREAD_GRAYSCALE)
                show_image(imgs)

    def save_salt_and_pepper_ims(self, im_length, num_images, folder):
        if not os.path.isdir(folder): os.mkdir(folder)
        for i in tqdm.tqdm(range(num_images)):
            im_path = join(folder, f"salt_pepper_im{i}.png")
            im = np.zeros((im_length, im_length))
            rand_range = random.randint(20, 35)
            for _ in range(rand_range):
                width = np.random.randint(0, 33)
                height = np.random.randint(0, 33)
                im[height, width] = 255
            with open(im_path[:-4] + ".txt", "w") as f:
                f.write("0")
            cv2.imwrite(im_path, im)
        print(f"Saved {num_images} salt and pepper ims in {folder}!")

    def save_ims_with_junk_around_border(self, im_length, num_images, folder):
        if not os.path.isdir(folder): os.mkdir(folder)
        for i in tqdm.tqdm(range(num_images)):
            im = np.zeros((im_length, im_length))
            location = random.choice(["top", "bottom", "left", "right"])
            im_path = join(folder, f"junk_around_border_{location}_im{i}.png")
            starting_range = (3,7)
            rand_range = random.randint(15, im_length-starting_range[1]-1)
            starting = random.randint(starting_range[0], starting_range[1])
            if location == "top":
                for i in range(rand_range):
                    if random.random() <= .75:
                        im[random.randint(2,4), starting + i] = 255
            elif location == "bottom":
                for i in range(rand_range):
                    if random.random() <= .75:
                        im[random.randint(im_length-4,im_length-2), starting + i] = 255
            elif location == "left":
                for i in range(rand_range):
                    if random.random() <= .75:
                        im[starting + i ,random.randint(2,4)] = 255
            elif location == "right":
                for i in range(rand_range):
                    if random.random() <= .75:
                        im[starting + i, random.randint(im_length-4,im_length-2)] = 255

            with open(im_path[:-4] + ".txt", "w") as f:
                f.write("0")
            cv2.imwrite(im_path, im)

        print(f"Saved {num_images} junk ims in {folder}!")

    def save_black_ims(self, folder, im_length, num_images):
        if not os.path.isdir(folder): os.mkdir(folder)
        for i in range(num_images):
            im_path = join(folder, f"black_im_{i}.png")
            cv2.imwrite(im_path, np.zeros((im_length, im_length)))
            with open(im_path[:-4] + ".txt", "w") as f:
                f.write("0")

    def augment_MNIST_images(self, save_folder, rotation_range=30, width_shift=.2, height_shift=.2, shear_range=40, zoom_range_low=1.0, zoom_range_high=1.0):
        if not os.path.isdir(save_folder): os.mkdir(save_folder)
        (X_train, y_train), (X_test, y_test) = mnist.load_data()
        X_train, y_train = get_specific_digits(X_train, y_train, digits_not=[0])

        from tensorflow.keras.preprocessing.image import ImageDataGenerator
        datagen = ImageDataGenerator(
            rotation_range=rotation_range,
                                     width_shift_range=width_shift,
                                     height_shift_range=height_shift,
                                     shear_range=shear_range,
                                     zoom_range=[zoom_range_low, zoom_range_high],
                                     )

        visualize_augmented_MNIST_images(datagen)
        quit()
        X_aug = np.zeros(shape=(X_train.shape[0], 28, 28, 1))
        y_aug = np.zeros(y_train.shape[0])
        b_size = 32
        for i, (X, y) in tqdm.tqdm(enumerate(datagen.flow(X_train.reshape(X_train.shape[0], 28, 28, 1),
                             y_train.reshape(y_train.shape[0], 1),
                             batch_size=b_size,
                             shuffle=False))):
            if (i+1)*b_size > X_aug.shape[0]:
                break
            X_aug[i*b_size:(i+1)*b_size] = X
            y_aug[i*b_size:(i+1)*b_size] = np.squeeze(y)
            for im_num in range(i*b_size, (i+1)*b_size):
                im_path = join(save_folder, f"im{im_num}_label{int(y_aug[im_num])}.png")
                cv2.imwrite(im_path, X_aug[im_num])
                with open(join(save_folder, im_path[:-4] + ".txt"), "a+") as f:
                    f.write(str(int(y_aug[im_num])))


        with open(join(save_folder, "augmentations_summary.txt"), "a+") as f:
            txt = "Augmentations used\n"\
            f"Rotation range: {datagen.rotation_range}\n" \
            f"width shift range: {datagen.width_shift_range}\n" \
            f"height shift range: {datagen.height_shift_range}\n" \
            f"shear range: {datagen.shear_range}\n" \
            f"zoom range: {datagen.zoom_range}\n" \

            f.write(txt)










#49057

if __name__ == "__main__":
    data_folder = "/Users/jeffrey/Coding/sudoku_cam_py/sudoku_dataset/digit_images2"
    augmenter = Augmenter(data_folder, 34)
    # augmenter.augment_images()
    # augmenter.show_diff_btwn_augmented_images(digit=1)
    # augmenter.save_salt_and_pepper_ims(34, 500, "/Users/jeffrey/Coding/sudoku_cam_py/junk_test_images")
    # augmenter.save_ims_with_junk_around_border(34, 500, "/Users/jeffrey/Coding/sudoku_cam_py/junk_test_images")
    # augmenter.save_black_ims("/Users/jeffrey/Coding/sudoku_cam_py/black_ims", 34, 5000)
    # augmenter.augment_MNIST_images(save_folder="/Users/jeffrey/Coding/sudoku_cam_py/augmented_MNIST_rotation_and_shear",
    #                                zoom_range_low=1, zoom_range_high=1, shear_range=30, rotation_range=30, width_shift=0, height_shift=0)
    augmenter.augment_MNIST_images(save_folder="/Users/jeffrey/Coding/sudoku_cam_py/augmented_MNIST_shear",
                                   zoom_range_low=1, zoom_range_high=1, shear_range=35, rotation_range=0, width_shift=.15, height_shift=.15)