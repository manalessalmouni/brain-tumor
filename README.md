🧠 Brain Tumor MRI Classification – Deep Learning Project
📌 Overview

This project implements a Deep Learning pipeline for automatic brain tumor classification from MRI images.
It combines a custom Convolutional Neural Network (CNN) and Transfer Learning models to classify brain MRI scans into four categories:

Glioma

Meningioma

Pituitary Tumor

No Tumor

The project also explores model interpretability (Grad-CAM) and optional tumor detection using YOLOv8, making it closer to a real medical AI workflow.

🎯 Objectives

Build a CNN from scratch as a baseline

Apply Transfer Learning (DenseNet121, ResNet50, VGG16, EfficientNetB0)

Perform data preprocessing & augmentation

Compare model performances

Improve interpretability with Grad-CAM

Explore tumor localization with YOLOv8

📂 Dataset

Brain Tumor MRI Dataset – Kaggle
https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset

Details:

~7,000 MRI images

4 classes: Glioma, Meningioma, Pituitary, No Tumor

Balanced distribution

Axial T1 MRI slices

Images resized to 224×224

🖼️ Preprocessing

Resize to 224×224

Normalize pixels to [0,1]

Convert grayscale → RGB

One-hot encoding

Train / Validation / Test split

🔄 Data Augmentation

Rotation

Translation

Horizontal Flip

Zoom & Shear

Brightness / Contrast variation

🧠 Models
1. Custom CNN (Baseline)

4 Convolutional blocks

ReLU + MaxPooling

Dense(512) + Dropout

Softmax Output

~496K parameters

Accuracy: ~95%

2. Transfer Learning Models

DenseNet121

ResNet50

EfficientNetB0

VGG16

These models were fine-tuned using ImageNet pretrained weights.

📊 Results
Model	Accuracy
Custom CNN	95.4%
VGG16	98.1%
EfficientNetB0	98.1%
ResNet50	98.8%
DenseNet121	98.8%

Observations

Transfer Learning outperformed the custom CNN by ~3%

DenseNet121 & ResNet50 achieved the best accuracy

Most confusion occurred between Glioma and Meningioma

“No Tumor” class reached near-perfect precision

🔍 Interpretability

Grad-CAM heatmaps were used to visualize which brain regions influenced model predictions, increasing transparency and clinical trust.

📍 Optional – YOLOv8 Detection

YOLOv8 was tested for tumor localization using bounding boxes and mAP/IoU metrics.

🛠️ Tech Stack

Python

TensorFlow / Keras

Scikit-learn

NumPy

Matplotlib / Seaborn

Ultralytics YOLOv8

Google Colab GPU

🚀 Future Work

Vision Transformers (ViT)

Model Ensembles

Tumor Segmentation (U-Net)

External clinical validation

Web / Desktop deployment

⚠️ Disclaimer

This project is for educational and research purposes only and is not a medical diagnostic tool.
