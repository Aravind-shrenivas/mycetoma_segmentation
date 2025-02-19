import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint
from sklearn.model_selection import train_test_split
from PIL import Image
import pandas as pd
import sys
import os
import matplotlib.pyplot as plt
from transformers import TFViTModel

def unet_model(input_size=(128, 128, 3)):
    inputs = layers.Input(input_size)
    conv1 = layers.Conv2D(64, 3, activation='relu', padding='same')(inputs)
    conv1 = layers.Conv2D(64, 3, activation='relu', padding='same')(conv1)
    pool1 = layers.MaxPooling2D(pool_size=(2, 2))(conv1)
    
    conv2 = layers.Conv2D(128, 3, activation='relu', padding='same')(pool1)
    conv2 = layers.Conv2D(128, 3, activation='relu', padding='same')(conv2)
    pool2 = layers.MaxPooling2D(pool_size=(2, 2))(conv2)
    
    conv3 = layers.Conv2D(256, 3, activation='relu', padding='same')(pool2)
    conv3 = layers.Conv2D(256, 3, activation='relu', padding='same')(conv3)
    pool3 = layers.MaxPooling2D(pool_size=(2, 2))(conv3)
    
    conv4 = layers.Conv2D(512, 3, activation='relu', padding='same')(pool3)
    conv4 = layers.Conv2D(512, 3, activation='relu', padding='same')(conv4)
    pool4 = layers.MaxPooling2D(pool_size=(2, 2))(conv4)
    
    conv5 = layers.Conv2D(1024, 3, activation='relu', padding='same')(pool4)
    conv5 = layers.Conv2D(1024, 3, activation='relu', padding='same')(conv5)
    
    up6 = layers.Conv2D(512, 2, activation='relu', padding='same')(layers.UpSampling2D(size=(2, 2))(conv5))
    merge6 = layers.concatenate([conv4, up6], axis=3)
    conv6 = layers.Conv2D(512, 3, activation='relu', padding='same')(merge6)
    conv6 = layers.Conv2D(512, 3, activation='relu', padding='same')(conv6)
    
    up7 = layers.Conv2D(256, 2, activation='relu', padding='same')(layers.UpSampling2D(size=(2, 2))(conv6))
    merge7 = layers.concatenate([conv3, up7], axis=3)
    conv7 = layers.Conv2D(256, 3, activation='relu', padding='same')(merge7)
    conv7 = layers.Conv2D(256, 3, activation='relu', padding='same')(conv7)
    
    up8 = layers.Conv2D(128, 2, activation='relu', padding='same')(layers.UpSampling2D(size=(2, 2))(conv7))
    merge8 = layers.concatenate([conv2, up8], axis=3)
    conv8 = layers.Conv2D(128, 3, activation='relu', padding='same')(merge8)
    conv8 = layers.Conv2D(128, 3, activation='relu', padding='same')(conv8)
    
    up9 = layers.Conv2D(64, 2, activation='relu', padding='same')(layers.UpSampling2D(size=(2, 2))(conv8))
    merge9 = layers.concatenate([conv1, up9], axis=3)
    conv9 = layers.Conv2D(64, 3, activation='relu', padding='same')(merge9)
    conv9 = layers.Conv2D(64, 3, activation='relu', padding='same')(conv9)
    
    conv10 = layers.Conv2D(1, 1, activation='sigmoid')(conv9)
    
    model = models.Model(inputs, conv10)
    
    return model

def pretrained_unet(backbone='ResNet50', input_shape=(128, 128, 3)):
    if backbone == 'ResNet50':
        base_model = tf.keras.applications.ResNet50(input_shape=input_shape, include_top=False, weights='imagenet')
    elif backbone == 'ResNet101':
        base_model = tf.keras.applications.ResNet101(input_shape=input_shape, include_top=False, weights='imagenet')
    elif backbone == 'ResNet152':
        base_model = tf.keras.applications.ResNet152(input_shape=input_shape, include_top=False, weights='imagenet')
    elif backbone == 'DenseNet121':
        base_model = tf.keras.applications.DenseNet121(input_shape=input_shape, include_top=False, weights='imagenet')
    elif backbone == 'DenseNet169':
        base_model = tf.keras.applications.DenseNet169(input_shape=input_shape, include_top=False, weights='imagenet')
    elif backbone == 'DenseNet201':
        base_model = tf.keras.applications.DenseNet201(input_shape=input_shape, include_top=False, weights='imagenet')
    elif backbone == 'EfficientNetB0':
        base_model = tf.keras.applications.EfficientNetB0(input_shape=input_shape, include_top=False, weights='imagenet')
    elif backbone == 'EfficientNetB1':
        base_model = tf.keras.applications.EfficientNetB1(input_shape=input_shape, include_top=False, weights='imagenet')
    elif backbone == 'EfficientNetB2':
        base_model = tf.keras.applications.EfficientNetB2(input_shape=input_shape, include_top=False, weights='imagenet')
    elif backbone == 'EfficientNetB3':
        base_model = tf.keras.applications.EfficientNetB3(input_shape=input_shape, include_top=False, weights='imagenet')
    elif backbone == 'EfficientNetB4':
        base_model = tf.keras.applications.EfficientNetB4(input_shape=input_shape, include_top=False, weights='imagenet')
    elif backbone == 'EfficientNetB5':
        base_model = tf.keras.applications.EfficientNetB5(input_shape=input_shape, include_top=False, weights='imagenet')
    elif backbone == 'EfficientNetB6':
        base_model = tf.keras.applications.EfficientNetB6(input_shape=input_shape, include_top=False, weights='imagenet')
    elif backbone == 'EfficientNetB7':
        base_model = tf.keras.applications.EfficientNetB7(input_shape=input_shape, include_top=False, weights='imagenet')
    elif backbone == 'EfficientNetV2B0':
        base_model = tf.keras.applications.EfficientNetV2B0(input_shape=input_shape, include_top=False, weights='imagenet')
    elif backbone == 'EfficientNetV2B1':
        base_model = tf.keras.applications.EfficientNetV2B1(input_shape=input_shape, include_top=False, weights='imagenet')
    elif backbone == 'EfficientNetV2B2':
        base_model = tf.keras.applications.EfficientNetV2B2(input_shape=input_shape, include_top=False, weights='imagenet')
    elif backbone == 'EfficientNetV2B3':
        base_model = tf.keras.applications.EfficientNetV2B3(input_shape=input_shape, include_top=False, weights='imagenet')
    elif backbone == 'EfficientNetV2L':
        base_model = tf.keras.applications.EfficientNetV2L(input_shape=input_shape, include_top=False, weights='imagenet')
    elif backbone == 'EfficientNetV2M':
        base_model = tf.keras.applications.EfficientNetV2M(input_shape=input_shape, include_top=False, weights='imagenet')
    elif backbone == 'EfficientNetV2S':
        base_model = tf.keras.applications.EfficientNetV2S(input_shape=input_shape, include_top=False, weights='imagenet')
    elif backbone == 'InceptionResNetV2':
        base_model = tf.keras.applications.InceptionResNetV2(input_shape=input_shape, include_top=False, weights='imagenet')
    elif backbone == 'InceptionV3':
        base_model = tf.keras.applications.InceptionV3(input_shape=input_shape, include_top=False, weights='imagenet')
    elif backbone == 'MobileNet':
        base_model = tf.keras.applications.MobileNet(input_shape=input_shape, include_top=False, weights='imagenet')
    elif backbone == 'MobileNetV2':
        base_model = tf.keras.applications.MobileNetV2(input_shape=input_shape, include_top=False, weights='imagenet')
    elif backbone == 'MobileNetV3Large':
        base_model = tf.keras.applications.MobileNetV3Large(input_shape=input_shape, include_top=False, weights='imagenet')
    elif backbone == 'MobileNetV3Small':
        base_model = tf.keras.applications.MobileNetV3Small(input_shape=input_shape, include_top=False, weights='imagenet')
    elif backbone == 'NASNetLarge':
        base_model = tf.keras.applications.NASNetLarge(input_shape=input_shape, include_top=False, weights='imagenet')
    elif backbone == 'NASNetMobile':
        base_model = tf.keras.applications.NASNetMobile(input_shape=input_shape, include_top=False, weights='imagenet')
    elif backbone.startswith('RegNetX') or backbone.startswith('RegNetY'):
        base_model = getattr(tf.keras.applications, backbone)(input_shape=input_shape, include_top=False, weights='imagenet')
    elif backbone == 'ResNet101V2':
        base_model = tf.keras.applications.ResNet101V2(input_shape=input_shape, include_top=False, weights='imagenet')
    elif backbone == 'ResNet152V2':
        base_model = tf.keras.applications.ResNet152V2(input_shape=input_shape, include_top=False, weights='imagenet')
    elif backbone == 'ResNet50V2':
        base_model = tf.keras.applications.ResNet50V2(input_shape=input_shape, include_top=False, weights='imagenet')
    elif backbone == 'ResNetRS101':
        base_model = tf.keras.applications.ResNetRS101(input_shape=input_shape, include_top=False, weights='imagenet')
    elif backbone == 'ResNetRS152':
        base_model = tf.keras.applications.ResNetRS152(input_shape=input_shape, include_top=False, weights='imagenet')
    elif backbone == 'ResNetRS200':
        base_model = tf.keras.applications.ResNetRS200(input_shape=input_shape, include_top=False, weights='imagenet')
    elif backbone == 'ResNetRS270':
        base_model = tf.keras.applications.ResNetRS270(input_shape=input_shape, include_top=False, weights='imagenet')
    elif backbone == 'ResNetRS350':
        base_model = tf.keras.applications.ResNetRS350(input_shape=input_shape, include_top=False, weights='imagenet')
    elif backbone == 'ResNetRS420':
        base_model = tf.keras.applications.ResNetRS420(input_shape=input_shape, include_top=False, weights='imagenet')
    elif backbone == 'VGG16':
        base_model = tf.keras.applications.VGG16(input_shape=input_shape, include_top=False, weights='imagenet')
    elif backbone == 'VGG19':
        base_model = tf.keras.applications.VGG19(input_shape=input_shape, include_top=False, weights='imagenet')
    elif backbone == 'Xception':
        base_model = tf.keras.applications.Xception(input_shape=input_shape, include_top=False, weights='imagenet')
    elif backbone.startswith('ConvNeXt'):
        base_model = getattr(tf.keras.applications, backbone)(input_shape=input_shape, include_top=False, weights='imagenet')
    else:
        raise ValueError("Unsupported backbone: {}".format(backbone))

    base_model.trainable = False 

    inputs = tf.keras.Input(shape=input_shape)
    x = base_model(inputs, training=False)


    x = layers.Conv2D(512, (3, 3), activation='relu', padding='same')(x)
    x = layers.UpSampling2D((2, 2))(x)
    x = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(x)
    x = layers.UpSampling2D((2, 2))(x)
    x = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = layers.UpSampling2D((2, 2))(x)
    x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = layers.UpSampling2D((2, 2))(x)
    x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    x = layers.UpSampling2D((2, 2))(x)
    x = layers.Conv2D(1, (1, 1), activation='sigmoid', padding='same')(x)

    model = models.Model(inputs, x)
    return model

def conv_block(inputs, out_channels):
    x = layers.Conv2D(out_channels, 3, activation='relu', padding='same')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(out_channels, 3, activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    return x

def unet_plus_plus(input_size=(128, 128, 3), out_channels=1):
    inputs = layers.Input(input_size)
    
    # Encoder
    conv1 = conv_block(inputs, 64)
    skip1 = conv_block(conv1, 64)
    pool1 = layers.MaxPooling2D(pool_size=(2, 2))(skip1)
    
    conv2 = conv_block(pool1, 128)
    skip2 = conv_block(conv2, 128)
    pool2 = layers.MaxPooling2D(pool_size=(2, 2))(skip2)
    
    conv3 = conv_block(pool2, 256)
    skip3 = conv_block(conv3, 256)
    pool3 = layers.MaxPooling2D(pool_size=(2, 2))(skip3)
    
    # Center
    center = conv_block(pool3, 512)

    # Decoder
    up3 = layers.Conv2DTranspose(256, kernel_size=2, strides=2, padding='same')(center)
    up3 = layers.concatenate([up3, skip3], axis=3)
    up3 = conv_block(up3, 256)
    final_up3 = layers.Conv2D(out_channels, kernel_size=1)(up3)

    up2 = layers.Conv2DTranspose(128, kernel_size=2, strides=2, padding='same')(up3)
    up2 = layers.concatenate([up2, skip2], axis=3)
    up2 = conv_block(up2, 128)
    final_up2 = layers.Conv2D(out_channels, kernel_size=1)(up2)
    
    up1 = layers.Conv2DTranspose(64, kernel_size=2, strides=2, padding='same')(up2)
    up1 = layers.concatenate([up1, skip1], axis=3)
    up1 = conv_block(up1, 64)
    final_up1 = layers.Conv2D(out_channels, kernel_size=1)(up1)
    
    # Final output
    final = layers.Conv2D(out_channels, kernel_size=1, activation='sigmoid')(final_up1)
    
    model = models.Model(inputs, final)
    
    return model

#trans-unet
def transformer_encoder(inputs, num_layers, num_heads, dff, dropout_rate=0.1):
    for _ in range(num_layers):
        # Multi-head self-attention layer
        attn_output = layers.MultiHeadAttention(num_heads=num_heads, key_dim=dff)(inputs, inputs)
        attn_output = layers.Dropout(dropout_rate)(attn_output)
        out1 = layers.LayerNormalization(epsilon=1e-6)(inputs + attn_output)
        
        # Feed-forward network
        ffn_output = layers.Dense(dff, activation='relu')(out1)
        ffn_output = layers.Dense(inputs.shape[-1])(ffn_output)
        ffn_output = layers.Dropout(dropout_rate)(ffn_output)
        inputs = layers.LayerNormalization(epsilon=1e-6)(out1 + ffn_output)
        
    return inputs

def vit_encoder(input_shape, num_layers=12, num_heads=8, dff=512, patch_size=16, dropout_rate=0.1):
    inputs = layers.Input(shape=input_shape)
    num_patches = (input_shape[0] // patch_size) * (input_shape[1] // patch_size)
    
    # Linear projection of flattened patches
    x = layers.Conv2D(dff, patch_size, strides=patch_size, padding='valid')(inputs)
    x = layers.Reshape((num_patches, dff))(x)
    
    # Positional encoding
    positions = tf.range(start=0, limit=num_patches, delta=1)
    pos_embedding = layers.Embedding(input_dim=num_patches, output_dim=dff)(positions)
    x = x + pos_embedding
    
    # Transformer encoder
    x = transformer_encoder(x, num_layers, num_heads, dff, dropout_rate)
    
    return models.Model(inputs, x, name='vit_encoder')

def decoder_block(inputs, skip_features, num_filters):
    x = layers.UpSampling2D((2, 2))(inputs)
    x = layers.Concatenate()([x, skip_features])
    x = layers.Conv2D(num_filters, 3, padding='same', activation='relu')(x)
    x = layers.Conv2D(num_filters, 3, padding='same', activation='relu')(x)
    return x

def trans_unet(input_shape=(128, 128, 3)):
    inputs = layers.Input(shape=input_shape)
    
    # Encoder
    vit = vit_encoder(input_shape)
    enc_outputs = vit(inputs)
    
    # Reshaping encoder outputs to match the U-Net decoder inputs
    enc_outputs = layers.Reshape((input_shape[0] // 16, input_shape[1] // 16, -1))(enc_outputs)
    
    # Decoder
    conv1 = layers.Conv2D(64, 3, activation='relu', padding='same')(inputs)
    conv1 = layers.Conv2D(64, 3, activation='relu', padding='same')(conv1)
    pool1 = layers.MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = layers.Conv2D(128, 3, activation='relu', padding='same')(pool1)
    conv2 = layers.Conv2D(128, 3, activation='relu', padding='same')(conv2)
    pool2 = layers.MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = layers.Conv2D(256, 3, activation='relu', padding='same')(pool2)
    conv3 = layers.Conv2D(256, 3, activation='relu', padding='same')(conv3)
    pool3 = layers.MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = layers.Conv2D(512, 3, activation='relu', padding='same')(pool3)
    conv4 = layers.Conv2D(512, 3, activation='relu', padding='same')(conv4)
    pool4 = layers.MaxPooling2D(pool_size=(2, 2))(conv4)

    # Bridge
    bridge = layers.Conv2D(1024, 3, activation='relu', padding='same')(pool4)
    bridge = layers.Conv2D(1024, 3, activation='relu', padding='same')(bridge)

    # Decoder
    up6 = decoder_block(bridge, conv4, 512)
    up7 = decoder_block(up6, conv3, 256)
    up8 = decoder_block(up7, conv2, 128)
    up9 = decoder_block(up8, conv1, 64)

    # Output
    outputs = layers.Conv2D(1, 1, activation='sigmoid')(up9)
    
    return models.Model(inputs, outputs)

#trans_unet++ (ViT with 1 billion parameters)
def transformer_encoder_1(inputs, num_layers, num_heads, hidden_dim, mlp_dim, dropout_rate=0.1):
    for _ in range(num_layers):
        # Multi-head self-attention layer
        attn_output = layers.MultiHeadAttention(num_heads=num_heads, key_dim=hidden_dim // num_heads)(inputs, inputs)
        attn_output = layers.Dropout(dropout_rate)(attn_output)
        out1 = layers.LayerNormalization(epsilon=1e-6)(inputs + attn_output)
        
        # Feed-forward network
        ffn_output = layers.Dense(mlp_dim, activation='relu')(out1)
        ffn_output = layers.Dense(hidden_dim)(ffn_output)
        ffn_output = layers.Dropout(dropout_rate)(ffn_output)
        inputs = layers.LayerNormalization(epsilon=1e-6)(out1 + ffn_output)
        
    return inputs

def vit_encoder_1(input_shape, num_layers=40, num_heads=16, hidden_dim=1408, mlp_dim=6144, patch_size=14, dropout_rate=0.1):
    inputs = layers.Input(shape=input_shape)
    num_patches = (input_shape[0] // patch_size) * (input_shape[1] // patch_size)
    
    # Linear projection of flattened patches
    patches = layers.Conv2D(hidden_dim, patch_size, strides=patch_size, padding='valid')(inputs)
    patches = layers.Reshape((num_patches, hidden_dim))(patches)
    
    # Positional encoding
    positions = tf.range(start=0, limit=num_patches, delta=1)
    pos_embedding = layers.Embedding(input_dim=num_patches, output_dim=hidden_dim)(positions)
    encoded_patches = patches + pos_embedding
    
    # Transformer encoder
    encoded_patches = transformer_encoder_1(encoded_patches, num_layers, num_heads, hidden_dim, mlp_dim, dropout_rate)
    
    return models.Model(inputs, encoded_patches, name='vit_encoder')

def conv_block_1(inputs, out_channels):
    x = layers.Conv2D(out_channels, 3, activation='relu', padding='same')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(out_channels, 3, activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    return x

def unet_plus_plus_decoder_1(enc_outputs, input_shape, out_channels=1):
    # Reshape the encoder outputs to match the UNet++ decoder inputs
    enc_outputs = layers.Reshape((input_shape[0] // 14, input_shape[1] // 14, -1))(enc_outputs)

    # Decoder
    x20 = conv_block_1(enc_outputs, 256)  # Corresponding to the last encoder block

    x21 = conv_block_1(layers.concatenate([layers.Conv2DTranspose(256, 2, strides=2, padding='same')(x20)], axis=3), 256)
    x22 = conv_block_1(layers.concatenate([layers.Conv2DTranspose(128, 2, strides=2, padding='same')(x21)], axis=3), 128)
    x23 = conv_block_1(layers.concatenate([layers.Conv2DTranspose(64, 2, strides=2, padding='same')(x22)], axis=3), 64)

    x11 = conv_block_1(layers.concatenate([layers.Conv2DTranspose(128, 2, strides=2, padding='same')(x21)], axis=3), 128)
    x12 = conv_block_1(layers.concatenate([layers.Conv2DTranspose(64, 2, strides=2, padding='same')(x11)], axis=3), 64)

    x01 = conv_block_1(layers.concatenate([layers.Conv2DTranspose(64, 2, strides=2, padding='same')(x11)], axis=3), 64)

    # Final output
    output = layers.Conv2D(out_channels, 1, activation='sigmoid')(x23)  # Adjusted to the last decoder block

    return output

def trans_unet_plus_plus(input_shape=(128, 128, 3)):
    inputs = layers.Input(shape=input_shape)
    
    # Encoder
    vit = vit_encoder_1(input_shape)
    enc_outputs = vit(inputs)
    
    # UNet++ Decoder
    final_output = unet_plus_plus_decoder_1(enc_outputs, input_shape)
    
    model = models.Model(inputs, final_output)
    
    return model
