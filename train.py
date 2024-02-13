
import os
import numpy as np
import cv2
import argparse
from glob import glob
from sklearn.utils import shuffle
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger, ReduceLROnPlateau, EarlyStopping, TensorBoard
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import Recall, Precision
from model import build_unet
from metrics import dice_loss, dice_coef, iou

# Atualizando as dimensões de entrada para 364x364x5
H, W, C = 364, 364, 5

def create_dir(path):
    """ Create a directory. """
    if not os.path.exists(path):
        os.makedirs(path)

def shuffling(x, y):
    x, y = shuffle(x, y, random_state=42)
    return x, y

def load_data(path, num_slices=5):
    images = sorted(glob(os.path.join(path, "image", "*.png")))
    masks = sorted(glob(os.path.join(path, "mask", "*.png")))
    x, y = [], []
    # Garantindo que cada entrada terá cinco fatias vizinhas
    for i in range(num_slices // 2, len(images) - num_slices // 2):
        slices = images[i - num_slices // 2 : i + num_slices // 2 + 1]
        x.append(slices)
        y.append(masks[i])
    return x, y

def read_image(paths):
    images = [cv2.imread(path.decode('utf-8'), cv2.IMREAD_GRAYSCALE) / 255.0 for path in paths]
    images = [img.astype(np.float32) for img in images]
    x = np.stack(images, axis=-1)  # Empilhando as imagens ao longo do eixo do canal
    return x

def read_mask(path):
    x = cv2.imread(path.decode('utf-8'), cv2.IMREAD_GRAYSCALE)
    x = x / 255.0
    x = x > 0.5
    x = x.astype(np.float32)
    x = np.expand_dims(x, axis=-1)
    return x

def tf_parse(x, y):
    def _parse(x, y):
        x = read_image(x)
        y = read_mask(y)
        return x, y

    x, y = tf.numpy_function(_parse, [x, y], [tf.float32, tf.float32])
    x.set_shape([H, W, C])
    y.set_shape([H, W, 1])
    return x, y

# A maioria do restante do código permanece inalterada
def tf_dataset(x, y, batch=8):
    dataset = tf.data.Dataset.from_tensor_slices((x, y))
    dataset = dataset.map(tf_parse)
    dataset = dataset.batch(batch)
    dataset = dataset.prefetch(10)
    return dataset


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
    create_dir(os.path.join(main_path, "models", dataset_name))

    """ Hyperparameters """
    batch_size = 2
    lr = 1e-4
    num_epochs = 100
    model_path = os.path.join("models", dataset_name, "model.h5")
    csv_path = os.path.join("models", dataset_name, "data.csv")

    """ Dataset """
    dataset_path = os.path.join(main_path, "data", "03_new_data", dataset_name)
    train_path = os.path.join(dataset_path, "train")
    valid_path = os.path.join(dataset_path, "valid")

    train_x, train_y = load_data(train_path)
    train_x, train_y = shuffling(train_x, train_y)
    valid_x, valid_y = load_data(valid_path)

    print(f"Train: {len(train_x)} - {len(train_y)}")
    print(f"Valid: {len(valid_x)} - {len(valid_y)}")

    train_dataset = tf_dataset(train_x, train_y, batch=batch_size)
    valid_dataset = tf_dataset(valid_x, valid_y, batch=batch_size)

    """ Model """
    model = build_unet((H, W, C))
    metrics = [dice_coef, iou, Recall(), Precision()]
    model.compile(loss=dice_loss, optimizer=Adam(lr), metrics=metrics)

    callbacks = [
        ModelCheckpoint(model_path, verbose=1, save_best_only=True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=10, min_lr=1e-7, verbose=1),
        CSVLogger(csv_path),
        TensorBoard(),
        EarlyStopping(monitor='val_loss', patience=50, restore_best_weights=False),
    ]

    model.fit(
        train_dataset,
        epochs=num_epochs,
        validation_data=valid_dataset,
        callbacks=callbacks,
        shuffle=False
    )
