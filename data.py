
import os
import argparse
import cv2
from glob import glob
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import albumentations as A

def create_dir(path):
    """ Create a directory. """
    if not os.path.exists(path):
        os.makedirs(path)

def load_data(path, split=0.8):
    """ Load the images and masks """
    images = sorted(glob(f"{path}/*/image/*.png"))
    masks = sorted(glob(f"{path}/*/mask/*.png"))

    """ Split the data """
    split_size = int(len(images) * split)
    train_x, valid_x = train_test_split(images, train_size=split_size, random_state=42, shuffle=False)
    train_y, valid_y = train_test_split(masks, train_size=split_size, random_state=42, shuffle=False)

    return (train_x, train_y), (valid_x, valid_y)

def augment_data(images, masks, save_path, augment=True):
    """ Performing data augmentation. """
    H, W = 364, 364

    aug = A.Compose([
        A.HorizontalFlip(p=0.5),  # 50% probability for horizontal flip
        A.VerticalFlip(p=0.5),  # 50% probability for vertical flip
        A.GaussNoise(p=0.2),  # 20% probability for Gaussian noise
    ])
    
    for idx, (x_path, y_path) in tqdm(enumerate(zip(images, masks)), total=len(images)):
        """ Extracting the patient ID and slice number from the path """
        parts = x_path.split(os.sep)
        patient_id, slice_name = parts[-3], parts[-1].split(".")[0]
        name = f"{patient_id}_{slice_name}"

        """ Read the image and mask """
        x = cv2.imread(x_path, cv2.IMREAD_COLOR)
        y = cv2.imread(y_path, cv2.IMREAD_COLOR)

        # Resize original images
        x_resized = cv2.resize(x, (W, H))
        y_resized = cv2.resize(y, (W, H))
        y_resized = y_resized / 255.0
        y_resized = (y_resized > 0.5) * 255

        # Save original images
        cv2.imwrite(os.path.join(save_path, "image", f"{name}.png"), x_resized)
        cv2.imwrite(os.path.join(save_path, "mask", f"{name}.png"), y_resized)

        # Apply augmentations if enabled
        if augment:
            augmented = aug(image=x, mask=y)
            x_aug = augmented['image']
            y_aug = augmented['mask']

            # Resize augmented images
            x_aug = cv2.resize(x_aug, (W, H))
            y_aug = cv2.resize(y_aug, (W, H))
            y_aug = y_aug / 255.0
            y_aug = (y_aug > 0.5) * 255

            # Save augmented images with a unique identifier
            cv2.imwrite(os.path.join(save_path, "image", f"{name}_aug_{idx}.png"), x_aug)
            cv2.imwrite(os.path.join(save_path, "mask", f"{name}_aug_{idx}.png"), y_aug)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process and augmentate image data.")
    parser.add_argument('--main_path', type=str, required=True, help="Path to repository.")  
    parser.add_argument('--dataset_name', type=str, required=True, help="Dataset name.")

    args = parser.parse_args()

    # Using the passed arguments
    main_path = args.main_path
    dataset_name = args.dataset_name

    """ Load the dataset """
    dataset_path = os.path.join(main_path, "data", "02_splited", dataset_name, "train")
    (train_x, train_y), (valid_x, valid_y) = load_data(dataset_path, split=0.8)

    print("Train: ", len(train_x))
    print("Valid: ", len(valid_x))

    create_dir(f"{main_path}/data/03_new_data/{dataset_name}/train/image/")
    create_dir(f"{main_path}/data/03_new_data/{dataset_name}/train/mask/")
    create_dir(f"{main_path}/data/03_new_data/{dataset_name}/valid/image/")
    create_dir(f"{main_path}/data/03_new_data/{dataset_name}/valid/mask/")

    augment_data(train_x, train_y, f"{main_path}/data/03_new_data/{dataset_name}/train/", augment=True)
    augment_data(valid_x, valid_y, f"{main_path}/data/03_new_data/{dataset_name}/valid/", augment=False)
