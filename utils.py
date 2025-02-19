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
from dotenv import load_dotenv
from tensorflow.keras.callbacks import ModelCheckpoint, Callback
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

load_dotenv()

class CustomModelCheckpoint(Callback):
    def __init__(self, filepath, backbone, model_type, monitor, verbose=1, save_best_only=True, mode='min'):
        super(CustomModelCheckpoint, self).__init__()
        self.filepath = filepath
        self.backbone = backbone
        self.model_type = model_type
        self.monitor = monitor
        self.verbose = verbose
        self.save_best_only = save_best_only
        self.mode = mode
        self.best = np.Inf if mode == 'min' else -np.Inf

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        current = logs.get(self.monitor)
        val_accuracy = logs.get('val_accuracy')
        val_loss = logs.get('val_loss')
        if current is None:
            print(f"Warning: Can save best model only with {self.monitor} available, skipping.")
            return

        if self.save_best_only:
            if (self.mode == 'min' and current < self.best) or (self.mode == 'max' and current > self.best):
                self.best = current
                if self.verbose > 0:
                    print(f"\nEpoch {epoch + 1}: {self.monitor} improved to {current:.5f}, saving model to {self.filepath}")
                filepath = self.filepath.format(epoch=epoch + 1, val_loss=val_loss, val_accuracy=val_accuracy, backbone=self.backbone, model_type=self.model_type)
                self.model.save(filepath, overwrite=True)
                print(f'\nSaving model to: {filepath}\n')
            else:
                if self.verbose > 0:
                    print(f"\nEpoch {epoch + 1}: {self.monitor} did not improve from {self.best:.5f}")
        else:
            if self.verbose > 0:
                print(f"\nEpoch {epoch + 1}: saving model to {self.filepath}")
            filepath = self.filepath.format(epoch=epoch + 1, val_loss=val_loss, val_accuracy=val_accuracy, backbone=self.backbone, model_type=self.model_type)
            self.model.save(filepath, overwrite=True)

def load_images(image_dir, mask_dir, csv_path, img_size=(128, 128)):
    mappings = pd.read_csv(csv_path)
    
    images = []
    masks = []
    
    for _, row in mappings.iterrows():
        img_path = os.path.join(image_dir, row['image'])
        mask_path = os.path.join(mask_dir, row['mask'])
        
        img = Image.open(img_path).resize(img_size)
        mask = Image.open(mask_path).resize(img_size).convert('L')
        
        img_array = np.array(img) / 255.0
        mask_array = np.array(mask)

        images.append(img_array)
        masks.append(mask_array)
        
    images = np.array(images)
    masks = np.expand_dims(np.array(masks), axis=-1)
    
    return images, masks

def sensitivity(y_true, y_pred):
    true_positives = tf.reduce_sum(tf.round(tf.clip_by_value(y_true * y_pred, 0, 1)))
    possible_positives = tf.reduce_sum(tf.round(tf.clip_by_value(y_true, 0, 1)))
    return true_positives / (possible_positives + tf.keras.backend.epsilon())

def specificity(y_true, y_pred):
    true_negatives = tf.reduce_sum(tf.round(tf.clip_by_value((1 - y_true) * (1 - y_pred), 0, 1)))
    possible_negatives = tf.reduce_sum(tf.round(tf.clip_by_value(1 - y_true, 0, 1)))
    return true_negatives / (possible_negatives + tf.keras.backend.epsilon())

def matthews_correlation(y_true, y_pred):
    y_pred_pos = tf.round(tf.clip_by_value(y_pred, 0, 1))
    y_pred_neg = 1 - y_pred_pos
    
    y_true_pos = tf.round(tf.clip_by_value(y_true, 0, 1))
    y_true_neg = 1 - y_true_pos
    
    tp = tf.reduce_sum(y_true_pos * y_pred_pos)
    tn = tf.reduce_sum(y_true_neg * y_pred_neg)
    
    fp = tf.reduce_sum(y_true_neg * y_pred_pos)
    fn = tf.reduce_sum(y_true_pos * y_pred_neg)
    
    numerator = (tp * tn) - (fp * fn)
    denominator = tf.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    
    return numerator / (denominator + tf.keras.backend.epsilon())

def jaccard_index(y_true, y_pred):
    y_true_f = tf.cast(tf.reshape(y_true, [-1]), tf.float32)
    y_pred_f = tf.cast(tf.reshape(y_pred, [-1]), tf.float32)
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    sum_ = tf.reduce_sum(y_true_f + y_pred_f)
    jac = (intersection + 1e-7) / (sum_ - intersection + 1e-7)
    return jac

    