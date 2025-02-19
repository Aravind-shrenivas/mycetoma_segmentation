import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint
from PIL import Image
import pandas as pd
import os
import argparse
from dotenv import load_dotenv

load_dotenv()

from models import unet_model, pretrained_unet
from utils import load_images, CustomModelCheckpoint

np.random.seed(45)
tf.random.set_seed(45)

strategy = tf.distribute.MirroredStrategy()
print('Number of devices: {}'.format(strategy.num_replicas_in_sync))

checkpoint_fine_tune = os.getenv('CHECKPOINT_FINE_TUNE')
val_image_dir = os.getenv('VAL_IMG_DIR')
val_mask_dir = os.getenv('VAL_MASK_DIR')
val_csv_path = os.getenv('VAL_CSV_PATH')
val_images, val_masks = load_images(val_image_dir, val_mask_dir, val_csv_path)

print("Validation Images shape: ", val_images.shape)
print("Validation Masks shape: ", val_masks.shape)

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
image_datagen.fit(val_images, augment=True, seed=seed)
mask_datagen.fit(val_masks, augment=True, seed=seed)

image_generator = image_datagen.flow(val_images, batch_size=16, seed=seed)
mask_generator = mask_datagen.flow(val_masks, batch_size=16, seed=seed)
val_generator = zip(image_generator, mask_generator)

def main(args):

    checkpoint_filepath = '/data/aravind/mycetoma_segmentation/checkpoints_fine_tuned/{backbone}-{model_type}-epoch:{epoch:02d}-val_loss:{val_loss:.5f}-val_acc:{val_accuracy:.5f}.h5'
    checkpoint_callback = CustomModelCheckpoint(filepath=checkpoint_filepath, backbone=args.backbone, model_type=args.model,
                                                monitor='val_loss',
                                                verbose=1,
                                                save_best_only=True,
                                                mode='min')

    with strategy.scope():
        if args.model == 'unet':
            model = unet_model(input_size=(128, 128, 3))
        elif args.model == 'pretrained_unet':
            model = pretrained_unet(backbone=args.backbone, input_shape=(128, 128, 3))
        else:
            raise ValueError(f"Unsupported model type: {args.model}")
        
        model.load_weights(args.checkpoint)
        
        model.compile(optimizer='adam', 
                      loss='binary_crossentropy', 
                      metrics=['accuracy'])

    history = model.fit(val_generator,
                        steps_per_epoch=len(val_images) // 16,
                        epochs=50,
                        validation_data=(val_images, val_masks),
                        callbacks=[checkpoint_callback])

    print("Fine-tuning complete. Weights saved to:", checkpoint_filepath)

# usage:
# python fine_tune.py --model pretrained_unet --backbone DenseNet201 --checkpoint path_to_checkpoint.h5
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Fine-tune a pre-trained segmentation model.')
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
    parser.add_argument('--model', type=str, default='unet', 
                        choices=['unet', 'pretrained_unet'],
                        help='The type of model architecture to use.')
    parser.add_argument('--checkpoint', type=str, required=True, 
                        help='Path to the checkpoint file of the pre-trained model.')
    args = parser.parse_args()
    main(args)
