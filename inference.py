import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
import os
import matplotlib.pyplot as plt
from dotenv import load_dotenv

load_dotenv()

from utils import load_images, sensitivity, specificity, matthews_correlation, jaccard_index

image_dir = os.getenv('VAL_IMG_DIR')
mask_dir = os.getenv('VAL_MASK_DIR')
csv_path = os.getenv('VAL_CSV_PATH')
histogram_dir = os.getenv('HISTOGRAM_DIR')
result_dir = os.getenv('RESULTS_DIR')

best_model_path = os.path.join('/data/aravind/mycetoma_segmentation/checkpoints', '-base_unet.h5')  # Update this with the actual filename of the best checkpoint
custom_objects = {
    'sensitivity': sensitivity,
    'specificity': specificity,
    'matthews_correlation': matthews_correlation,
    'jaccard_index': jaccard_index
}
model = tf.keras.models.load_model(best_model_path, custom_objects=custom_objects)

images, masks = load_images(image_dir, mask_dir, csv_path)

X_train, X_val, y_train, y_val = train_test_split(images, masks, test_size=0.2, random_state=42)

sample_image = X_val[10]
sample_mask = y_val[10]

predicted_mask = model.predict(np.expand_dims(sample_image, axis=0))[0]

plt.figure(figsize=(12, 6))
plt.hist(predicted_mask.ravel(), bins=50, color='orange', alpha=0.75)
plt.title('Histogram of Predicted Mask Values')
plt.xlabel('Predicted Value')
plt.ylabel('Frequency')

output_path = os.path.join(histogram_dir, 'segmentation_histogram.png')
plt.savefig(output_path, bbox_inches='tight')
plt.close()

thresholds = [0.0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1, 0.11, 0.12, 0.13, 0.14, 0.15, 0.16, 0.17, 0.18, 0.19, 0.2, 0.21, 0.22, 0.23, 0.24, 0.25, 0.26, 0.27, 0.28, 0.29, 0.3, 0.31, 0.32, 0.33, 0.34, 0.35, 0.36, 0.37, 0.38, 0.39, 0.4, 0.41, 0.42, 0.43, 0.44, 0.45, 0.46, 0.47, 0.48, 0.49, 0.5, 0.51, 0.52, 0.53, 0.54, 0.55, 0.56, 0.57, 0.58, 0.59, 0.6]


for threshold_value in thresholds:
    binary_predicted_mask = (predicted_mask > threshold_value).astype(np.uint8)

    plt.figure(figsize=(24, 8))

    plt.subplot(1, 4, 1)
    plt.title('Original Image')
    plt.imshow(sample_image)
    plt.axis('off')

    plt.subplot(1, 4, 2)
    plt.title('Ground Truth Mask')
    plt.imshow(sample_mask[:, :, 0], cmap='gray')
    plt.axis('off')

    plt.subplot(1, 4, 3)
    plt.title('Predicted Mask')
    plt.imshow(predicted_mask[:, :, 0], cmap='gray')
    plt.axis('off')

    plt.subplot(1, 4, 4)
    plt.title('Binary Predicted Mask')
    plt.imshow(binary_predicted_mask[:, :, 0], cmap='gray')
    plt.axis('off')

    output_path2 = os.path.join(result_dir, f'segmentation_results_base_unet_2_{threshold_value}.png')
    plt.savefig(output_path2, bbox_inches='tight')
    plt.close()

print(f"Results saved to {output_path} and respective thresholded masks saved to {result_dir}")
