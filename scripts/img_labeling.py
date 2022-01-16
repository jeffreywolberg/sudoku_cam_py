import cv2
import os
from board_finder import SudokuSolver, SudokuBoardNotFoundError
import glob
import numpy as np
from tensorflow.keras.datasets import mnist
from visualize_mnist import show_image
import shutil

basename = os.path.basename
join = os.path.join

class ImageLabeling(object):
    def __init__(self, im_folder):
        self.im_folder = im_folder

    def get_and_save_images(self, board_img_folder, save_directory="test_set", im_length=28):
        im_paths = [glob.glob(join(board_img_folder, f"*{ext}")) for ext in ["jpg", "png"]]
        im_paths = [p for paths in im_paths for p in paths]
        print(im_paths)
        sudoku_solver = SudokuSolver()
        print(save_directory)
        if not os.path.isdir(save_directory):
            print(f"Making directory... {save_directory}")
            os.mkdir(save_directory)
        for im_path in im_paths:
            # if images have already been processed for this board, skip
            if len(glob.glob(join(save_directory, basename(im_path) + "*"))) > 0:
                print(f"Board in image {im_path} has already been processed, skipping...")
                continue
            print(f"Finding board for {im_path}...")
            try:
                imgs = sudoku_solver.find_puzzle(cv2.imread(im_path), im_length=im_length, debug_only_crop_and_warp=True).reshape((81, im_length, im_length))
            except SudokuBoardNotFoundError:
                continue
            num_saved = 0
            for i in range(81):
                im = imgs[i]
                digit_path = join(save_directory, basename(im_path) + f"_im{i}.png")
                if os.path.exists(digit_path): continue
                if np.count_nonzero(im)/(im_length*im_length) <= .03: continue
                num_saved += 1
                cv2.imwrite(digit_path, im)
            print(f"Saved {num_saved} images for sudoku board in {basename(im_path)}")

    def run_labeling_pipeline(self):
        img_paths = glob.glob(join(self.im_folder, "*"))
        for im_path in img_paths:
            im = cv2.imread(im_path)
            im_name = im_path[:-4]
            if os.path.exists(im_name + '.txt'): continue
            cv2.imshow("Img", im)
            key = chr(cv2.waitKey(0))
            with open(join(self.im_folder, im_name+".txt"), "w") as f:
                f.write(key)


    def get_corresponding_label_for_im(self, im_path):
        file_path = im_path[:-4] + ".dat"
        print(f"Looking for corresponding labels in {file_path}")
        with open(file_path, "r") as f:
            lines = [line.strip("\n") for line in f.readlines()]
            print(lines)
            digits = []
            for line in lines:
                nums = line.split(" ")
                nums = list(filter("".__ne__, nums)) # remove any blank strings from list
                if len(nums) != 9:
                    continue
                digits.extend([n.strip("\n") for n in nums])
        if len(digits) != 81:
            print(digits)
            print(len(digits))
            raise FileNotFoundError
        return digits

    def add_to_bad_images_list(self, im_path, path_to_file="bad_images"):
        with open(path_to_file, "a+") as f:
            f.write(im_path + "\n")

    def is_im_good(self, im_path, path_to_file="bad_images"):
        with open(path_to_file, "r") as f:
            lines = [line.strip("\n") for line in f.readlines()]
            if im_path in lines:
                return False
            else: return True

    def remove_images_with_zero(self, folder):
        txt_files = glob.glob(join(folder, "*.txt"))
        print(len(txt_files))
        files_removed = 0
        for file_path in txt_files:
            should_remove = False
            with open(file_path, 'r') as f:
                line = f.readline()
                if str(line) == "0":
                    should_remove = True
            if should_remove:
                print(f"Removing {file_path} and {file_path[:-4] + '.png'}")
                os.remove(file_path)
                os.remove(file_path[:-4] + ".png")
                files_removed += 1
        print(f"Removed {files_removed} files!")


    def process_sudoku_dataset(self, folder_to_dataset, im_length=34, dir_name="digit_images"):
        bad_images_txt_file_path = join(folder_to_dataset, "bad_images_list.txt")
        print(bad_images_txt_file_path)
        if not os.path.exists(bad_images_txt_file_path):
            with open(bad_images_txt_file_path, "w") as f:
                f.write("")
        sudoku_solver = SudokuSolver()
        im_paths = [glob.glob(join(folder_to_dataset, f"*{ext}")) for ext in ["jpg", "png"]]
        im_paths = [p for paths in im_paths for p in paths]
        file_paths = [glob.glob(join(folder_to_dataset, f"*{ext}")) for ext in ["dat"]]
        file_paths = [p for paths in file_paths for p in paths]
        total_num_saved = 0
        for im_path in im_paths:
            im = cv2.imread(im_path)
            # check if image has been processed before
            if len(glob.glob(join(dir_name, basename(im_path[:-4]) + "*.txt"))) > 0:
                print("Sudoku board image has already been processed, continuing!")
                continue
            if not self.is_im_good(im_path, path_to_file=bad_images_txt_file_path):
                print("Sudoku board is in the list of bad images, continuing!")
                continue

            if im_path.endswith(".jpg"):
                cv2.imshow("Nonrotated", im)
                key = chr(cv2.waitKey(0))
                if key.lower() == "r":
                    print("Rotating image!")
                    im = cv2.rotate(im, cv2.ROTATE_90_COUNTERCLOCKWISE)
                    cv2.imshow("Rotated im", im)
                    cv2.waitKey(0)
            print(f"Finding board for {im_path}...")
            try:
                imgs = sudoku_solver.find_puzzle(im, im_length=im_length).reshape(
                (81, im_length, im_length))
            except SudokuBoardNotFoundError:
                print(f"Adding image {im_path} to bad images list")
                self.add_to_bad_images_list(im_path, path_to_file=bad_images_txt_file_path)
                print(f"Could not find board for {im_path}!")
                continue
            try:
                digits = self.get_corresponding_label_for_im(im_path)
            except FileNotFoundError:
                print("Couldn't find corresponding sudoku labels!")
                continue
            num_saved = 0
            for i in range(81):
                if digits[i] == 0:
                    continue
                im = imgs[i]
                eroded_img = cv2.erode(cv2.erode(im, (15,15)), (15,15))
                if np.count_nonzero(eroded_img) / (im_length * im_length) <= .04: continue
                # cv2.imshow("img", im)
                # cv2.waitKey(0)
                # cv2.imshow("img erode", eroded_img)
                # cv2.waitKey(0)
                num_saved += 1
                new_im_path = join(dir_name, basename(im_path[:-4]) + f"_im{i}.png")
                new_file_path = new_im_path[:-4]+".txt"
                cv2.imwrite(new_im_path, im)
                with open(new_file_path, "w") as f:
                    f.write(str(digits[i]))
                print(f"Saved image to {new_im_path}")
                print(f"Saved file to {new_file_path}")
            total_num_saved += num_saved
            print(f"Saved {num_saved} images for sudoku board in {basename(im_path)}")
        print(f"Labeled a total of {total_num_saved} digits!")



        print(im_paths)
        print(file_paths)


    def print_digit_distribution(self, folders : list):
        histo = np.zeros(9)
        total = 0
        for folder in folders:
            for file_path in glob.glob(join(folder, "*.txt")):
                with open(file_path, 'r') as f:
                    line = f.readline()
                    histo[int(line)-1] += 1
                    total += 1
        for i in range(len(histo)):
            print(f"{i+1}: {round(histo[i]/total*100,1)}%")





if __name__ == "__main__":
    folder = "/Users/jeffrey/Coding/sudoku_cam_py/training_set2"
    labeling = ImageLabeling(folder)
    digit_folder = "/Users/jeffrey/Coding/sudoku_cam_py/sudoku_dataset/digit_images"
    images_folder = "/Users/jeffrey/Coding/sudoku_cam_py/sudoku_dataset/mixed"
    # labeling.get_and_save_images(r"/Users/jeffrey/Coding/sudoku_cam_py/board_images", im_length=34, save_directory=r"/Users/jeffrey/Coding/sudoku_cam_py/training_set2")
    # labeling.run_labeling_pipeline()
    # labeling.run_labeling_pipeline()
    # labeling.process_sudoku_dataset(images_folder,
    #                                 dir_name=digit_folder)
    labeling.remove_images_with_zero(folder)
    # labeling.print_digit_distribution([digit_folder, folder])
    # labeling.remove_images_with_zero(folder)

