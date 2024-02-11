import os
import cv2
import numpy as np

def preprocess(input_image_path, output_folder_path, WL=350, WC=150, output_size=(364, 364)):
    """
    Normalizes a single image using Hounsfield unit scale windowing, crops a central window of specified size, 
    and saves the processed image to a specified output folder.

    Args:
    - input_image_path (str): Path to the input image.
    - output_folder_path (str): Path to the output folder where processed image will be saved.
    - WL (int): Window Length for Hounsfield unit normalization.
    - WC (int): Window Center for Hounsfield unit normalization.
    - output_size (tuple): Size of the output cropped image (width, height).
    """

    # Ensure the output directory exists
    if not os.path.exists(output_folder_path):
        os.makedirs(output_folder_path)

    # Extract the filename from the input image path
    filename = os.path.basename(input_image_path)

    # Read the image
    img = cv2.imread(input_image_path, cv2.IMREAD_GRAYSCALE)

    # Normalize using Hounsfield unit scale windowing
    min_hu = WC - WL // 2
    max_hu = WC + WL // 2
    img = np.clip(img, min_hu, max_hu)
    img = (img - min_hu) / (max_hu - min_hu)

    # Normalize between 0 and 1
    img = (img - np.min(img)) / (np.max(img) - np.min(img))

    # Crop the central part of the image
    center_x, center_y = img.shape[1] // 2, img.shape[0] // 2
    half_width, half_height = output_size[0] // 2, output_size[1] // 2
    cropped_img = img[center_y - half_height:center_y + half_height, center_x - half_width:center_x + half_width]

    # Save the processed image
    output_image_path = os.path.join(output_folder_path, filename)
    cv2.imwrite(output_image_path, cropped_img * 255)  # Multiply by 255 to convert back to 0-255 range

    return "Image processed and saved."
