import os
import shutil
from sklearn.model_selection import train_test_split

def create_dir(path):
    """ Create a directory if it does not exist. """
    if not os.path.exists(path):
        os.makedirs(path)

def split(dataset_dir, output_dir, train_size=0.8):
    """ Split full sets into train and test sets and copy them to the output directory. """
    all_sets = os.listdir(dataset_dir)
    train_sets, test_sets = train_test_split(all_sets, train_size=train_size, random_state=42)

    # Function to copy a full set
    def copy_set(set_names, output_subdir):
        for set_name in set_names:
            src_dir = os.path.join(dataset_dir, set_name)
            dest_dir = os.path.join(output_dir, output_subdir, set_name)
            shutil.copytree(src_dir, dest_dir)

    # Creating output directories
    create_dir(os.path.join(output_dir, "train"))
    create_dir(os.path.join(output_dir, "test"))

    # Copying sets
    copy_set(train_sets, "train")
    copy_set(test_sets, "test")
