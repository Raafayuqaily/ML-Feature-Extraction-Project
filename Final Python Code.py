import cv2
import os
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt


def main():
    # path of folder containing raw images
    in_folder_path = "C:\\Users\\axelm\\Documents\\3 PET Images Original"  # change to your local folder pathway

    # path of folder to contain modified images
    out_folder_path = "C:\\Users\\axelm\\Documents\\4 PET Images new"  # change to your local folder pathway

    # create folders inside out_folder_path to store contrast, non-contrast, and pathways separately
    os.makedirs(out_folder_path + '\\(Pre) Contrast', exist_ok=True)
    os.makedirs(out_folder_path + '\\(Pre) Non Contrast', exist_ok=True)
    os.makedirs(out_folder_path + '\\(Post) Contrast', exist_ok=True)
    os.makedirs(out_folder_path + '\\(Post) Non Contrast', exist_ok=True)
    os.makedirs(out_folder_path + '\\Pathways', exist_ok=True)
    os.makedirs(out_folder_path + '\\Pathways', exist_ok=True)

    # create lists to save file pathways (for later use in DataFrame / CSV output)
    original_contrast_pathways = []
    original_non_contrast_pathways = []
    mask_contrast_pathways = []
    mask_non_contrast_pathways = []

    for file_name in os.listdir(in_folder_path):
        # file_name contains name of current file from in_folder_path

        # input_path appends the image's file name to in_folder_path to create a path to the image.
        input_path = os.path.join(in_folder_path, file_name)

        # load image and convert to grayscale
        img = cv2.imread(input_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Find middle pixel value (to determine whether CT scan is contrast or non-contrast)
        height, width = img.shape
        middle_pixel_value = img[height // 2, width // 2]

        # Re-save original image segregating contrast and non-contrast
        thresh = 200  # value to determine cutoff between contrast and non-contrast
        if middle_pixel_value >= thresh:
            # output_path contains the full path of the output (including modified file name)
            output_path = os.path.join(out_folder_path, '(Pre) Contrast', file_name)
            cv2.imwrite(output_path, img)

            # store output path to later put in data frame. (to print to Excel and CSV)
            original_contrast_pathways.append(output_path)

        else:
            # output_path contains the full path of the output (including modified file name)
            output_path = os.path.join(out_folder_path, '(Pre) Non Contrast', file_name)
            cv2.imwrite(output_path, img)

            # store output path to later put in data frame. (to print to Excel and CSV)
            original_non_contrast_pathways.append(output_path)

        # run processing method dependent on image type (Contrast vs Non-Contrast)
        if middle_pixel_value >= thresh:
            masked_image = contrast(img)

            # output_path contains the full path of the output (including modified file name)
            output_path = os.path.join(out_folder_path, '(Post) Contrast', '(mask) ' + file_name)

            # store output path to later put in data frame. (to print to Excel and CSV)
            mask_contrast_pathways.append(output_path)

        else:
            masked_image = non_contrast(img)

            # output_path contains the full path of the output (including modified file name)
            output_path = os.path.join(out_folder_path, '(Post) Non Contrast', '(mask) ' + file_name)

            # store output path to later put in data frame. (to print to Excel and CSV)
            mask_non_contrast_pathways.append(output_path)

        # Code below will print image to screen (displays final mask)
        # plt.imshow(masked_image, cmap="gray")
        # plt.show()

        # save the masked image
        cv2.imwrite(output_path, masked_image)

    # define lists of equal length to contrast and non-contrast pathways list. (repeats value 255 for label column)
    label_list_contrast = np.repeat(255, len(original_contrast_pathways))
    label_list_non_contrast = np.repeat(255, len(original_non_contrast_pathways))

    # define dataframes for export. One holds contrast pathways, the other non-contrast.
    df_contrast = pd.DataFrame(
        {'Image': original_contrast_pathways, 'Mask': mask_contrast_pathways, 'Label': label_list_contrast})
    df_non_contrast = pd.DataFrame(
        {'Image': original_non_contrast_pathways, 'Mask': mask_non_contrast_pathways, 'Label': label_list_non_contrast})

    # Save as Excel output (Contrast and Non-Contrast output on different sheets)
    with pd.ExcelWriter(out_folder_path + '\\Pathways\\Pathways Excel.xlsx', engine='openpyxl') as writer:
        df_contrast.to_excel(writer, sheet_name='Contrast', index=False)
        df_non_contrast.to_excel(writer, sheet_name='Non-Contrast', index=False)

    # Save as CSV output (Contrast and Non-Contrast output in different files)
    df_contrast.to_csv(out_folder_path + '\\Pathways\\(Contrast) Pathways CSV.csv', index=False)
    df_non_contrast.to_csv(out_folder_path + '\\Pathways\\(Non-Contrast) Pathways CSV.csv', index=False)


# Run Pyradiomics (need to change to local file pathway)
# Code below only runs pyradiomics for (Contrast) pathways CSV
os.system(
    "pyradiomics \"C:\\Users\\axelm\\Documents\\4 PET Images new\\Pathways\\(Contrast) Pathways CSV.csv\" -o "
    "output.csv -f csv")


def contrast(img):
    # modify raw images with methods below
    blur = cv2.GaussianBlur(img, (5, 5), 0)  # blur to averages out rapid changes in pixel intensities
    thresh_value, thresh_image = cv2.threshold(blur, 200, 255, cv2.THRESH_BINARY)  # binary thresh
    thresh_value, otsu_image = cv2.threshold(blur, 240, 255, cv2.THRESH_OTSU)  # otsu thresh
    otsu_image = cv2.bitwise_not(otsu_image)  # invert result of otsu thresh
    masked_image = thresh_image + otsu_image  # add results of thresh and otsu thresh
    masked_image = cv2.bitwise_not(masked_image)  # invert masked_image
    return masked_image


def non_contrast(img):
    # modify raw images with methods below
    increased_contrast = cv2.convertScaleAbs(img, alpha=6, beta=6)
    reduced_contrast = cv2.convertScaleAbs(img, alpha=0.4, beta=0)

    reduced_blur = cv2.GaussianBlur(reduced_contrast, (5, 5), 0)  # blur (same reason as above)
    increased_blur = cv2.GaussianBlur(increased_contrast, (5, 5), 0)  # blur (same reason as above)
    thresh_value, reduced_thresh_image = cv2.threshold(reduced_blur, 240, 255, cv2.THRESH_OTSU)  # otsu thresh
    thresh_value, increased_thresh_image = cv2.threshold(increased_blur, 240, 255, cv2.THRESH_OTSU)  # otsu thresh
    increased_thresh_image = cv2.bitwise_not(increased_thresh_image)  # invert result of increased_thresh_image
    masked_image = increased_thresh_image + reduced_thresh_image  # add results of otsu thresh images
    masked_image = cv2.bitwise_not(masked_image)  # invert masked_image
    return masked_image


# driver function
main()
