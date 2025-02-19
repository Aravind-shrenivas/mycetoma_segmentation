import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models

# Transformer Encoder
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

# ViT Encoder
def vit_encoder_1(input_shape, num_layers=40, num_heads=16, hidden_dim=1408, mlp_dim=6144, patch_size=16, dropout_rate=0.1):
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

# Decoder Block
def decoder_block(inputs, num_filters):
    x = layers.UpSampling2D((2, 2))(inputs)
    x = layers.Conv2D(num_filters, 3, padding='same', activation='relu')(x)
    x = layers.Conv2D(num_filters, 3, padding='same', activation='relu')(x)
    return x

# TransUNet Model
def trans_unet(input_shape=(128, 128, 3)):
    inputs = layers.Input(shape=input_shape)
    
    # Encoder
    vit = vit_encoder_1(input_shape)
    enc_outputs = vit(inputs)
    
    # Reshaping encoder outputs to match the U-Net decoder inputs
    reshaped_enc_outputs = layers.Reshape((input_shape[0] // 16, input_shape[1] // 16, 1408))(enc_outputs)
    
    # Bridge
    bridge = layers.Conv2D(1024, 3, activation='relu', padding='same')(reshaped_enc_outputs)
    bridge = layers.Conv2D(1024, 3, activation='relu', padding='same')(bridge)
    
    # Decoder
    up6 = decoder_block(bridge, 512)
    up7 = decoder_block(up6, 256)
    up8 = decoder_block(up7, 128)
    up9 = decoder_block(up8, 64)

    # Output
    outputs = layers.Conv2D(1, 1, activation='sigmoid')(up9)
    
    return models.Model(inputs, outputs)

# Example usage
model = trans_unet(input_shape=(128, 128, 3))
model.summary()
