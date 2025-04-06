import sys
import zipfile
from pathlib import Path

import cv2 as cv
import numpy as np
import tensorflow as tf
import tqdm
from keras import callbacks, layers, models, optimizers

BATCH_SIZE = 50

def main():

    ## DATA EXTRACTION ---------------------------------------------------------
    
    print("Extracting data...")
    data_file = Path(__file__).parent / 'data.zip'
    if not data_file.exists():
        raise FileNotFoundError(f"Data file {data_file} not found.")
    
    with zipfile.ZipFile(data_file, 'r') as zip_ref:
        zip_ref.extractall(Path(__file__).parent)

    train_dirs = list(Path.cwd().glob('**/train/'))
    test_dirs = list(Path.cwd().glob('**/test/'))
    
    ## LOAD DATA ---------------------------------------------------------

    print("Loading data...")
    test_img, test_steer = load_all_data(test_dirs)
    train_img, train_steer = load_all_data(train_dirs)

    ## PREPROCESSING DATA ---------------------------------------------------------

    print("Preprocessing data...")
    train_data, test_data = [], []
    for image_path in tqdm.tqdm(train_img, desc="Processing training images"):
        train_data.append(process_image(image_path))

    for image_path in tqdm.tqdm(test_img, desc="Processing test images"):
        test_data.append(process_image(image_path))

    if len(train_data) != len(train_steer):
        raise ValueError("Mismatch between training data and steering angles.")
    if len(test_data) != len(test_steer):
        raise ValueError("Mismatch between test data and steering angles.")
    print(f"Training data size: {len(train_data)}", file=sys.stderr)

    train_ds = dataset_pipeline(train_data, train_steer, shuffle=True,
                                repeat=False, batch_size=BATCH_SIZE)
    test_ds = dataset_pipeline(test_data, test_steer, shuffle=False,
                               repeat=False, batch_size=BATCH_SIZE)

    print(f'{train_ds.cardinality()} training batches')

    def make_model(model_func, model_params, train_data, test_data):
        model = model_func()
        model.fit(
            train_ds,
            #steps_per_epoch=len(train_data) // BATCH_SIZE,
            validation_data=test_ds,
            #validation_steps=len(test_data) // BATCH_SIZE,
            **model_params,
        )
        return model
    
    ## TRAINING ---------------------------------------------------------
    print("Training model v3...")
    model_v3 = make_model(create_model_v3, model_v3_params, train_ds, test_ds)
    model_v3.save('model_v3.h5')

    print("Training model v19...")
    model_v19 = make_model(create_model_v19, model_v19_params, train_ds, test_ds)
    model_v19.save('model_v19.h5')

    print("Training model v27...")
    model_v27 = make_model(create_model_v27, model_v27_params, train_ds, test_ds)
    model_v27.save('model_v27.h5')

    print("Training model v35...")
    model_v35 = make_model(create_model_v35, model_v35_params, train_ds, test_ds)
    model_v35.save('model_v35.h5')
    
    print("Training complete.")

    ## EVALUATION ---------------------------------------------------------
    print("Evaluating models...")


def dataset_file_get_timestamp(file: Path):
    return int(file.stem.split('_')[0])


def dataset_file_get_steering_angle(file: Path):
    return float(file.stem.split('_')[1].replace('-', '.'))


def load_data(directory: Path):
    image_paths = []
    steering_angles = []
    # load image paths from the directory
    for file_path in directory.glob('*.png'):
        image_paths.append(file_path)

    # sort images by the timestamp in their filenames
    image_paths.sort(key=dataset_file_get_timestamp)

    # Extract steering angles from filenames
    for path in image_paths:
        steering_angles.append(dataset_file_get_steering_angle(path))
        
    return image_paths, steering_angles


def load_all_data(dirs: list[Path]):
    img, steer = [], []
    for directory in dirs:
        img_, steer_ = load_data(directory)
        img += img_
        steer += steer_
    return img, steer


def process_image(image_path):
    img = cv.imread(image_path) 
    hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
    mask = cv.inRange(hsv, (36, 0, 0), (70, 255,255))
    array = np.array(mask, dtype=np.float32)
    return array


def dataset_pipeline(features, labels, shuffle, *, batch_size=32,
                     repeat=False, rand_seed=42, shuffle_buffer_divider=10):
    dataset = tf.data.Dataset.from_tensor_slices((features, labels))
    if repeat:
        dataset = dataset.repeat()
    if shuffle:
        dataset = dataset.shuffle(
            buffer_size=len(features) // shuffle_buffer_divider,
            seed=rand_seed
        )
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(buffer_size=5)
    return dataset


def create_model_v3() -> models.Sequential:
    model = models.Sequential()
    model.add(layers.InputLayer(shape=(66, 100, 1)))
    model.add(layers.Conv2D(64, kernel_size=(3, 3), activation='relu'))
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(layers.Conv2D(128, kernel_size=(3, 3), activation='relu'))
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(layers.Flatten())
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(32, activation='relu'))
    model.add(layers.Dense(1, activation='linear'))
    
    model.compile(optimizer='adam', loss='mse')
    return model

def create_model_v19() -> models.Sequential:
    model = models.Sequential()
    model.add(layers.InputLayer(shape=(66, 100, 1)))
    model.add(layers.Conv2D(16, kernel_size=(3, 3), activation='relu'))
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(layers.Conv2D(32, kernel_size=(3, 3), activation='relu'))
    model.add(layers.Conv2D(32, kernel_size=(3, 3), activation='relu'))
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(layers.Dropout(0.2))
    model.add(layers.Conv2D(64, kernel_size=(3, 3), activation='relu'))
    model.add(layers.Conv2D(64, kernel_size=(3, 3), activation='relu'))
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(layers.Dropout(0.2))
    model.add(layers.Flatten())
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(32, activation='relu'))
    model.add(layers.Dense(16, activation='relu'))
    model.add(layers.Dense(1, activation='linear'))
    model.compile(optimizer='adam', loss='mse')
    return model

def conv_block(filters, thick, conv_params, model):
    for _ in range(thick):
        model.add(layers.SeparableConv2D(filters, **conv_params))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPool2D((2,2)))
    return model

def create_model_v27() -> models.Sequential:
    """Use ReduceLROnPlateau with this model"""
    conv_params = {
        'kernel_size': (3,3),
        'activation': 'relu',
        'padding': 'same',
        'use_bias': False
    }
    model = models.Sequential()
    model.add(layers.Input(shape=(66, 100, 1)))

    layer_widths = [8, 16, 32, 64, 128]
    for i in layer_widths:
        conv_block(i, 2, conv_params, model)
        
    model.add(layers.Dropout(0.5))
    model.add(layers.GlobalAvgPool2D())

    for i in layer_widths[::-1]:
        model.add(layers.Dense(i, activation='relu'))

    model.add(layers.Dense(1, activation='linear'))
    model.compile(optimizer=optimizers.Adam(0.002), loss='mse')
    return model

def create_model_v35() -> models.Sequential:
    """Use ReduceLROnPlateau with this model"""
    conv_params = {
        'kernel_size': (3,3),
        'activation': 'relu',
        'padding': 'same'
    }
    model = models.Sequential()
    model.add(layers.Input(shape=(66, 100, 1)))

    conv_block(8, 1, conv_params, model)
    conv_block(12, 1, conv_params, model)
    conv_block(16, 1, conv_params, model)
    conv_block(32, 2, conv_params, model)
    conv_block(48, 2, conv_params, model)
        
    model.add(layers.Dropout(0.5))
    model.add(layers.GlobalAvgPool2D())

    for i in [48, 32, 16, 8]:
        model.add(layers.Dense(i, activation='relu'))

    model.add(layers.Dense(1, activation='linear'))
    model.compile(optimizer=optimizers.RMSprop(0.002), loss='mse')
    return model


class EpochLogger(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        '''Logs training progress every 10 epochs.'''
        if (epoch + 1) % 10 == 0:  # Log every 10 epochs
            print(f"Epoch {epoch+1}: Loss = {logs['loss']:.4f},",
                  f"Val Loss = {logs.get('val_loss', 'N/A'):.4f}")


model_v3_params = {
    'epochs': 25,
    'batch_size': BATCH_SIZE
}

model_v19_params = {
    'epochs': 150,
    'batch_size': BATCH_SIZE,
    'verbose': 1,
    'callbacks': [
        EpochLogger()
    ]
}

model_v27_params = {
    'epochs': 200,
    'batch_size': BATCH_SIZE,
    'verbose': 0,
    'callbacks': [
        callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=10,
            verbose=0
        ),
        EpochLogger()
    ]
}

model_v35_params = {
    'epochs': 100,
    'batch_size': BATCH_SIZE,
    'verbose': 0,
    'callbacks': [
        callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=10,
            verbose=0
        ),
        EpochLogger()
    ]
}

if __name__ == '__main__':
    main()