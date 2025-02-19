import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import Callback, ModelCheckpoint
from sklearn.model_selection import train_test_split
from PIL import Image
import pandas as pd
import sys
import os
import argparse
import matplotlib.pyplot as plt
from dotenv import load_dotenv
import keras_tuner as kt

load_dotenv()

from models import unet_model, pretrained_unet, unet_plus_plus, trans_unet_plus_plus
from test import trans_unet
from utils import load_images, sensitivity, specificity, matthews_correlation, jaccard_index

parser = argparse.ArgumentParser(description='Train a segmentation model.')
parser.add_argument('--backbone', type=str, default='', 
                    choices=['ConvNeXtBase', 'ConvNeXtLarge', 'ConvNeXtSmall', 'ConvNeXtTiny', 'ConvNeXtXLarge',
                                'DenseNet121', 'DenseNet169', 'DenseNet201', 'EfficientNetB0', 'EfficientNetB1', 
                                'EfficientNetB2', 'EfficientNetB3', 'EfficientNetB4', 'EfficientNetB5', 'EfficientNetB6', 
                                'EfficientNetB7', 'EfficientNetV2B0', 'EfficientNetV2B1', 'EfficientNetV2B2', 
                                'EfficientNetV2B3', 'EfficientNetV2L', 'EfficientNetV2M', 'EfficientNetV2S', 
                                'InceptionResNetV2', 'InceptionV3', 'MobileNet', 'MobileNetV2', 'MobileNetV3Large', 
                                'MobileNetV3Small', 'NASNetLarge', 'NASNetMobile', 'RegNetX002', 'RegNetX004', 'RegNetX006', 
                                'RegNetX008', 'RegNetX016', 'RegNetX032', 'RegNetX040', 'RegNetX064', 'RegNetX080', 
                                'RegNetX120', 'RegNetX160', 'RegNetX320', 'RegNetY002', 'RegNetY004', 'RegNetY006', 
                                'RegNetY008', 'RegNetY016', 'RegNetY032', 'RegNetY040', 'RegNetY064', 'RegNetY080', 
                                'RegNetY120', 'RegNetY160', 'RegNetY320', 'ResNet101', 'ResNet101V2', 'ResNet152', 
                                'ResNet152V2', 'ResNet50', 'ResNet50V2', 'ResNetRS101', 'ResNetRS152', 'ResNetRS200', 
                                'ResNetRS270', 'ResNetRS350', 'ResNetRS420', 'VGG16', 'VGG19', 'Xception'],
                    help='The backbone model to use for the encoder.')
parser.add_argument('--model', type=str, default='base_unet', 
                    choices=['base_unet', 'pretrained_unet', 'unet++','trans_unet','trans_unet++'],
                    help='The type of model architecture to use.')
parser.add_argument('--loss', type=str, default='bce_dice_loss', 
                    choices=['dice_loss', 'bce_dice_loss', 'tversky_loss', 'focal_tversky', 'boundary_loss'],
                    help='The type of loss function to use.')
# parser.add_argument('--cuda', type=int, default=0, 
#                     help='the identifier of the gpu')
args = parser.parse_args()

# args.model='pretrained_unet'
# args.backbone='DenseNet201'
# args.loss='bce_dice_loss'
# os.environ["CUDA_VISIBLE_DEVICES"] = '3'

np.random.seed(45)
tf.random.set_seed(45)

strategy = tf.distribute.MirroredStrategy()
print('Number of devices: {}'.format(strategy.num_replicas_in_sync))

# Custom augmentation function
def custom_augment(image):
    image = tf.image.random_brightness(image, max_delta=0.2)
    image = tf.image.random_contrast(image, lower=0.8, upper=1.2)
    image = tf.image.random_saturation(image, lower=0.8, upper=1.2)
    image = tf.image.random_hue(image, max_delta=0.02)
    return image

# Custom preprocessing function
def custom_preprocessing(image):
    image = custom_augment(image)
    return image

# Load environment variables
image_dir = os.getenv('IMAGE_DIR')
mask_dir = os.getenv('MASK_DIR')
csv_path = os.getenv('CSV_PATH')

val_image_dir = os.getenv('VAL_IMG_DIR')
val_mask_dir = os.getenv('VAL_MASK_DIR')
val_csv_path = os.getenv('VAL_CSV_PATH')

train_images, train_masks = load_images(image_dir, mask_dir, csv_path)
val_images, val_masks = load_images(val_image_dir, val_mask_dir, val_csv_path)

# Data augmentation parameters
data_gen_args = dict(rotation_range=90.,
                     width_shift_range=0.1,
                     height_shift_range=0.1,
                     shear_range=0.2,
                     zoom_range=0.2,
                     horizontal_flip=True,
                     vertical_flip=True,
                     fill_mode='nearest')

# image_datagen = ImageDataGenerator(preprocessing_function=custom_preprocessing, **data_gen_args)
image_datagen = ImageDataGenerator(**data_gen_args)
mask_datagen = ImageDataGenerator(**data_gen_args)

seed = 1
image_datagen.fit(train_images, augment=True, seed=seed)
mask_datagen.fit(train_masks, augment=True, seed=seed)

image_generator = image_datagen.flow(train_images, batch_size=1, seed=seed)
mask_generator = mask_datagen.flow(train_masks, batch_size=1, seed=seed)
train_generator = zip(image_generator, mask_generator)

# Custom loss functions
epsilon = 1e-5
smooth = 1

def dsc(y_true, y_pred):
    y_true = tf.cast(y_true, tf.float32)
    y_true_f = tf.keras.backend.flatten(y_true)
    y_pred_f = tf.keras.backend.flatten(y_pred)
    intersection = tf.keras.backend.sum(y_true_f * y_pred_f)
    score = (2. * intersection + smooth) / (tf.keras.backend.sum(y_true_f) + tf.keras.backend.sum(y_pred_f) + smooth)
    return score

def dice_loss(y_true, y_pred):
    loss = 1 - dsc(y_true, y_pred)
    return loss

def bce_dice_loss(y_true, y_pred):
    loss = tf.keras.losses.binary_crossentropy(y_true, y_pred) + dice_loss(y_true, y_pred)
    return loss

def tversky(y_true, y_pred):
    y_true = tf.cast(y_true, tf.float32)
    y_true_pos = tf.keras.backend.flatten(y_true)
    y_pred_pos = tf.keras.backend.flatten(y_pred)
    true_pos = tf.keras.backend.sum(y_true_pos * y_pred_pos)
    false_neg = tf.keras.backend.sum(y_true_pos * (1 - y_pred_pos))
    false_pos = tf.keras.backend.sum((1 - y_true_pos) * y_pred_pos)
    alpha = 0.7
    return (true_pos + smooth) / (true_pos + alpha * false_neg + (1 - alpha) * false_pos + smooth)

def tversky_loss(y_true, y_pred):
    return 1 - tversky(y_true, y_pred)

def focal_tversky(y_true, y_pred):
    pt_1 = tversky(y_true, y_pred)
    gamma = 0.75
    return tf.keras.backend.pow((1 - pt_1), gamma)

def boundary_loss(y_true, y_pred):
    y_true = tf.cast(y_true, tf.float32)
    y_true_boundary = tf.image.sobel_edges(y_true)
    y_pred_boundary = tf.image.sobel_edges(y_pred)
    return tf.reduce_mean(tf.square(y_true_boundary - y_pred_boundary))

# Main function
def main(args):
    checkpoint_filepath = f'/data/aravind/mycetoma_segmentation/checkpoints/{args.backbone}-{args.model}.h5'
    # checkpoint_callback = CustomModelCheckpoint(filepath=checkpoint_filepath, backbone=args.backbone, model_type=args.model,
    #                                             monitor=args.loss,
    #                                             verbose=1,
    #                                             save_best_only=True,
    #                                             mode='min')

    checkpoint_callback = ModelCheckpoint(filepath=checkpoint_filepath,
                                          verbose=1,
                                          monitor='val_loss',
                                          save_best_only=True,
                                          mode='min')


    with strategy.scope():
        if args.model == 'base_unet':
            model = unet_model(input_size=(128, 128, 3))
        elif args.model == 'pretrained_unet':
            model = pretrained_unet(backbone=args.backbone, input_shape=(128, 128, 3))
        elif args.model == 'unet++':
            model = unet_plus_plus(input_size=(128, 128, 3))
        elif args.model == 'trans_unet':
            model = trans_unet(input_shape=(128, 128, 3))
        elif args.model == 'trans_unet++':
            model = trans_unet_plus_plus(input_shape=(128, 128, 3))
        else:
            raise ValueError(f"Unsupported model type: {args.model}")

        # Select the loss function
        if args.loss == 'dice_loss':
            loss_fn = dice_loss
        elif args.loss == 'bce_dice_loss':
            loss_fn = bce_dice_loss
        elif args.loss == 'tversky_loss':
            loss_fn = tversky_loss
        elif args.loss == 'focal_tversky':
            loss_fn = focal_tversky
        elif args.loss == 'boundary_loss':
            loss_fn = boundary_loss
        else:
            raise ValueError(f"Unsupported loss type: {args.loss}")

        model.summary()

        model.compile(optimizer='adam', 
                      loss=loss_fn, 
                      metrics=[sensitivity, specificity, matthews_correlation, jaccard_index])

    history = model.fit(train_generator,
                        steps_per_epoch=len(train_images) // 1,
                        epochs=150,
                        validation_data=(val_images, val_masks),
                        callbacks=[checkpoint_callback])

    model.evaluate(val_images, val_masks)

# Command-line argument parser
if __name__ == '__main__':

    main(args)
