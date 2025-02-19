import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, Callback
from sklearn.model_selection import train_test_split
from PIL import Image
import pandas as pd
import sys
import os
import argparse
import matplotlib.pyplot as plt
from dotenv import load_dotenv
import keras_tuner as kt
from datetime import datetime

load_dotenv()

from models import unet_model, pretrained_unet, unet_plus_plus
from utils import load_images, sensitivity, specificity, matthews_correlation

np.random.seed(45)
tf.random.set_seed(45)

strategy = tf.distribute.MirroredStrategy()
print('Number of devices: {}'.format(strategy.num_replicas_in_sync))

image_dir = os.getenv('IMAGE_DIR')
mask_dir = os.getenv('MASK_DIR')
csv_path = os.getenv('CSV_PATH')

val_image_dir = os.getenv('VAL_IMG_DIR')
val_mask_dir = os.getenv('VAL_MASK_DIR')
val_csv_path = os.getenv('VAL_CSV_PATH')

train_images, train_masks = load_images(image_dir, mask_dir, csv_path)
val_images, val_masks = load_images(val_image_dir, val_mask_dir, val_csv_path)

data_gen_args = dict(rotation_range=90.,
                     width_shift_range=0.1,
                     height_shift_range=0.1,
                     shear_range=0.2,
                     zoom_range=0.2,
                     horizontal_flip=True,
                     vertical_flip=True,
                     fill_mode='nearest')

image_datagen = ImageDataGenerator(**data_gen_args)
mask_datagen = ImageDataGenerator(**data_gen_args)

seed = 1
image_datagen.fit(train_images, augment=True, seed=seed)
mask_datagen.fit(train_masks, augment=True, seed=seed)

image_generator = image_datagen.flow(train_images, batch_size=16, seed=seed)
mask_generator = mask_datagen.flow(train_masks, batch_size=16, seed=seed)
train_generator = zip(image_generator, mask_generator)

def jaccard_index(y_true, y_pred):
    y_true_f = tf.cast(tf.reshape(y_true, [-1]), tf.float32)
    y_pred_f = tf.cast(tf.reshape(y_pred, [-1]), tf.float32)
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    sum_ = tf.reduce_sum(y_true_f + y_pred_f)
    jac = (intersection + 1e-7) / (sum_ - intersection + 1e-7)
    return jac

def build_model(hp):
    input_size = (128, 128, 3)
    # model_type = hp.Choice('model', ['base_unet', 'unet++', 'pretrained_unet' ])
    model_type = hp.Choice('model', ['base_unet', 'unet++' ])

    # backbone = hp.Choice('backbone', ['ConvNeXtBase', 'ConvNeXtLarge', 'ConvNeXtSmall', 'ConvNeXtTiny', 'ConvNeXtXLarge',
    #                              'DenseNet121', 'DenseNet169', 'DenseNet201', 'EfficientNetB0', 'EfficientNetB1', 
    #                              'EfficientNetB2', 'EfficientNetB3', 'EfficientNetB4', 'EfficientNetB5', 'EfficientNetB6', 
    #                              'EfficientNetB7', 'EfficientNetV2B0', 'EfficientNetV2B1', 'EfficientNetV2B2', 
    #                              'EfficientNetV2B3', 'EfficientNetV2L', 'EfficientNetV2M', 'EfficientNetV2S', 
    #                              'InceptionResNetV2', 'InceptionV3', 'MobileNet', 'MobileNetV2', 'MobileNetV3Large', 
    #                              'MobileNetV3Small', 'NASNetLarge', 'NASNetMobile', 'RegNetX002', 'RegNetX004', 'RegNetX006', 
    #                              'RegNetX008', 'RegNetX016', 'RegNetX032', 'RegNetX040', 'RegNetX064', 'RegNetX080', 
    #                              'RegNetX120', 'RegNetX160', 'RegNetX320', 'RegNetY002', 'RegNetY004', 'RegNetY006', 
    #                              'RegNetY008', 'RegNetY016', 'RegNetY032', 'RegNetY040', 'RegNetY064', 'RegNetY080', 
    #                              'RegNetY120', 'RegNetY160', 'RegNetY320', 'ResNet101', 'ResNet101V2', 'ResNet152', 
    #                              'ResNet152V2', 'ResNet50', 'ResNet50V2', 'ResNetRS101', 'ResNetRS152', 'ResNetRS200', 
    #                              'ResNetRS270', 'ResNetRS350', 'ResNetRS420', 'VGG16', 'VGG19', 'Xception'])
    learning_rate = hp.Float('learning_rate', min_value=1e-5, max_value=1e-2, sampling='LOG')
    batch_size = hp.Int('batch_size', min_value=8, max_value=32, step=8)
    epochs = hp.Int('epochs', min_value=10, max_value=100, step=10)
    
    if model_type == 'base_unet':
        model = unet_model(input_size=input_size)
    elif model_type == 'pretrained_unet':
        model = pretrained_unet(backbone=backbone, input_shape=input_size)
    elif model_type =='unet++':
        model = unet_plus_plus(input_size=input_size)
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate),
        loss='binary_crossentropy',
        metrics=['accuracy', sensitivity, specificity, matthews_correlation, jaccard_index]
    )
    return model

def main(args):
    tuner = kt.RandomSearch(
        build_model,
        objective='val_loss',
        max_trials=50,  # Increase the number of trials
        executions_per_trial=1,
        directory='/data/aravind/mycetoma_segmentation/tuner',
        project_name='tune_segmentation_model'
    )

    # current_time = datetime.now().strftime("%Y%m%d-%H%M%S")
    # checkpoint_filepath = f'/data/aravind/mycetoma_segmentation/checkpoints1/{current_time}-{}-best-model.h5'
    # checkpoint_callback = ModelCheckpoint(filepath=checkpoint_filepath,
    #                                       monitor='val_loss',
    #                                       verbose=1,
    #                                       save_best_only=True,
    #                                       mode='min')

    tuner.search(train_generator,
                 steps_per_epoch=len(train_images) // 16,
                 epochs=50,
                 validation_data=(val_images, val_masks),
                #  callbacks=[checkpoint_callback])
    )

    best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
    
    # Extracting the best hyperparameters
    best_model_type = best_hps.get('model')
    best_backbone = best_hps.get('backbone')
    best_learning_rate = best_hps.get('learning_rate')
    best_batch_size = best_hps.get('batch_size')
    best_epochs = best_hps.get('epochs')

    with strategy.scope():
        if best_model_type == 'unet':
            model = unet_model(input_size=(128, 128, 3))
        elif best_model_type == 'unet++':
            model = unet_plus_plus(input_size=(128,128,3))
        elif best_model_type == 'pretrained_unet':
            model = pretrained_unet(backbone=best_backbone, input_shape=(128, 128, 3))

        model.compile(
            optimizer=tf.keras.optimizers.Adam(best_learning_rate),
            loss='binary_crossentropy',
            metrics=['accuracy', sensitivity, specificity, matthews_correlation, jaccard_index]
        )

    history = model.fit(train_generator,
                        steps_per_epoch=len(train_images) // best_batch_size,
                        epochs=best_epochs,
                        validation_data=(val_images, val_masks),
                        callbacks=[checkpoint_callback])

    loss, accuracy, sens, spec, mcc, jac = model.evaluate(val_images, val_masks)
    print(f"Validation Loss: {loss}")
    print(f"Validation Accuracy: {accuracy}")
    print(f"Validation Sensitivity: {sens}")
    print(f"Validation Specificity: {spec}")
    print(f"Validation MCC: {mcc}")
    print(f"Validation Jaccard Index: {jac}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Hyperparameter tuning for a segmentation model.')
    args = parser.parse_args()
    main(args)
