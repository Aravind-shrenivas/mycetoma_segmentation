# Mycetoma Image Classification & Segmentation  

## Overview  
This project focuses on **automated classification and segmentation of Mycetoma images**, using **deep learning models** to distinguish between different infection types. It was developed as part of the **Mycetoma Micro Image: Detect and Classify Challenge at MICCAI 2024**, where it was selected as a **top 5 finalist**.  

## Features  
- **Image Classification** using **ResNet50**, achieving **92% accuracy**.  
- **Image Segmentation** using **nnUNet**, achieving a **DICE score of 0.83**.  
- **Conditional Random Fields (CRF)** applied to refine segmentation accuracy.  
- **Automated preprocessing and augmentation** to improve model performance.  

## Methodology  
### **1. Data Preprocessing**  
- Images resized to **224x224 pixels** (classification) and **640x896 pixels** (segmentation).  
- Applied **Z-score normalization** for pixel intensity standardization.  
- **Data augmentation**: rotation, flipping, contrast adjustment.  

### **2. Model Development**  
#### **Classification (ResNet50)**  
- Used a **pre-trained ResNet50** for feature extraction and fine-tuning.  
- Applied **sigmoid activation** and **binary cross-entropy loss**.  
- Optimized using **Adam optimizer** with a learning rate of **1e-4**.  

#### **Segmentation (nnUNet)**  
- Implemented **PlainConvUNet** with **8 stages (32-512 feature maps per stage)**.  
- Used **Dice loss** as the primary loss function for better segmentation.  
- Post-processing with **Conditional Random Fields (CRF)** for refined boundaries.  

## Results  
| Task            | Metric        | Score  |  
|----------------|--------------|--------|  
| Classification | Accuracy      | **92%**  |  
| Segmentation   | Dice Score    | **0.83** |  
| Segmentation   | Jaccard Index | **0.75** |  

## Tools & Libraries  
- **TensorFlow & PyTorch** – Model training and inference.  
- **OpenCV & PIL** – Image preprocessing.  
- **scikit-learn & NumPy** – Data manipulation.  
- **Matplotlib & Seaborn** – Visualizations.  
- **Google Colab & AWS** – Model training and deployment.  

## Contributors
- Aravind Shrenivas Murali | University of Arizona
- Dr. Eung-Joo Lee | Research Advisor
