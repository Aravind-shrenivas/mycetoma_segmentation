import numpy as np
import tensorflow as tf
from tensorflow.keras import models
import os
import argparse
from dotenv import load_dotenv
from utils import load_images, sensitivity, specificity, matthews_correlation, jaccard_index

load_dotenv()

def evaluate_model(weights_path, model_architecture, backbone, image_dir, mask_dir, csv_path):
    val_images, val_masks = load_images(image_dir, mask_dir, csv_path)

    if model_architecture == 'base_unet':
        from models import unet_model
        model = unet_model(input_size=(128, 128, 3))
    elif model_architecture == 'pretrained_unet':
        from models import pretrained_unet
        model = pretrained_unet(backbone=backbone, input_shape=(128, 128, 3))
    elif model_architecture == 'unet++':
        from models import unet_plus_plus
        model = unet_plus_plus(input_size=(128, 128, 3))
    elif model_architecture == 'trans_unet':
        from models import trans_unet
        model = trans_unet(input_size=(128, 128, 3))
    elif model_architecture == 'trans_unet++':
        from models import trans_unet_plus_plus
        model = trans_unet(input_size=(128, 128, 3))
    else:
        raise ValueError(f"Unsupported model type: {model_architecture}")

    model.load_weights(weights_path)
    model.compile(optimizer='adam', 
                  loss='binary_crossentropy', 
                  metrics=['accuracy', sensitivity, specificity, matthews_correlation, jaccard_index])

    loss, accuracy, sens, spec, mcc, jacc = model.evaluate(val_images, val_masks)
    print(f"Validation Loss: {loss}")
    print(f"Validation Accuracy: {accuracy}")
    print(f"Validation Sensitivity: {sens}")
    print(f"Validation Specificity: {spec}")
    print(f"Validation MCC: {mcc}")
    print(f"Validation Jaccard Index: {jacc}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate a trained segmentation model.')
    parser.add_argument('--weights', type=str, required=True, help='Path to the saved model weights.')
    parser.add_argument('--model', type=str, required=True, 
                        choices=['base_unet', 'pretrained_unet', 'unet++'], 
                        help='The model architecture type.')
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
                        help='The backbone model to use for the encoder (if using pretrained_unet).')
    parser.add_argument('--image_dir', type=str, required=True, help='Directory path for validation images.')
    parser.add_argument('--mask_dir', type=str, required=True, help='Directory path for validation masks.')
    parser.add_argument('--csv_path', type=str, required=True, help='Path to the CSV file for validation data.')
    args = parser.parse_args()

    evaluate_model(args.weights, args.model, args.backbone, args.image_dir, args.mask_dir, args.csv_path)
