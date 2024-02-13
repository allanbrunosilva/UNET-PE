
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import numpy as np
import cv2
import argparse
import pydicom as dicom
import pandas as pd
from glob import glob
from tqdm import tqdm
import tensorflow as tf
from tensorflow.keras.utils import CustomObjectScope
from sklearn.metrics import accuracy_score, f1_score, jaccard_score, precision_score, recall_score
from metrics import dice_loss, dice_coef, iou

""" Creating a directory """
def create_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process and augmentate image data.")
    parser.add_argument('--main_path', type=str, required=True, help="Path to repository.")  
    parser.add_argument('--dataset_name', type=str, required=True, help="Dataset name.")

    args = parser.parse_args()

    # Using the passed arguments
    main_path = args.main_path
    dataset_name = args.dataset_name

    """ Seeding """
    np.random.seed(42)
    tf.random.set_seed(42)

    """ Directory for storing files """
    create_dir(os.path.join(main_path, "data", "05_prediction", dataset_name))

    """ Loading model """
    with CustomObjectScope({'iou': iou, 'dice_coef': dice_coef, 'dice_loss': dice_loss}):
        model = tf.keras.models.load_model(os.path.join(main_path, "models", dataset_name,"model.h5"))

    """ Load the dataset """
    test_x = glob(os.path.join(main_path, "data/02_splited", dataset_name, "test/*/image/*.png"))
    print(f"Test: {len(test_x)}")

    """ Loop over the data """
    for x in tqdm(test_x):
        name = os.path.basename(x).split(".")[0]

        i = cv2.imread(x, cv2.IMREAD_GRAYSCALE)
        i = cv2.resize(i, (512, 512))
        i = np.expand_dims(i, axis=-1)
        i = i / np.max(i) * 255.0
        x = i / 255.0
        x = np.concatenate([x, x, x], axis=-1)
        x = np.expand_dims(x, axis=0)

        mask = model.predict(x)[0]
        mask = mask > 0.5
        mask = mask.astype(np.int32)
        mask = mask * 255

        cat_images = np.concatenate([i, mask], axis=1)
        cv2.imwrite(f"{main_path}/data/05_prediction/{dataset_name}/{name}.png", cat_images)
