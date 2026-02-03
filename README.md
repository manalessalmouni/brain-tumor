# 🧠 Brain Tumor Classification using Deep Learning (CNN)

## 📌 Project Overview

This project focuses on the **automatic classification of brain tumors from MRI images** using **Deep Learning** techniques.  
A **Convolutional Neural Network (CNN) built from scratch** is designed and trained to classify MRI scans into **four categories**:

- **Glioma**
- **Meningioma**
- **Pituitary Tumor**
- **No Tumor**

The project demonstrates the effectiveness of CNNs in **medical image analysis** and their potential as **decision-support tools** for radiologists.

---

## 🎯 Objectives

- Build a **CNN from scratch** for medical image classification  
- Apply **data preprocessing and augmentation** techniques  
- Achieve high classification accuracy on MRI images  
- Analyze model performance using quantitative and visual metrics  

---

## 📂 Dataset

The dataset used is the **Brain Tumor MRI Dataset**, organized as follows:

Dataset/
│
├── Training/
│ ├── glioma/
│ ├── meningioma/
│ ├── pituitary/
│ └── notumor/
│
└── Testing/
├── glioma/
├── meningioma/
├── pituitary/
└── notumor/

- Images are grayscale or RGB MRI scans
- Each folder name represents the corresponding class
- The dataset is relatively balanced across classes

---

## 🖼️ Data Preprocessing

- Image resizing to **150 × 150 pixels**
- Pixel normalization to the range **[0, 1]**
- One-hot encoding of class labels
- Train/validation/test split

---

## 🔄 Data Augmentation

- Random rotations  
- Width and height shifts  
- Shear transformations  
- Zoom operations  
- Horizontal flipping  

---

## 🧠 Model Architecture

- **4 Convolutional layers**: Filters 32 → 64 → 128 → 128, kernel size 4×4, ReLU activation  
- **MaxPooling layers** after each convolution (3×3)  
- **Fully connected layers**: Flatten → Dense(512, ReLU) → Dropout(0.5)  
- **Output layer**: Dense(4, Softmax)  

---

## 📉 Loss Function and Optimizer

- **Loss Function:** Categorical Crossentropy  
- **Optimizer:** Adam (learning rate 0.001)  
- **Evaluation Metric:** Accuracy  

---

## ⏱️ Training Strategy

- Batch size: **32**
- Maximum epochs: **40**
- Callbacks:
  - **EarlyStopping** to prevent overfitting
  - **ReduceLROnPlateau** to adjust learning rate dynamically

---

## 📊 Results

- **Test Accuracy:** ~95%  
- High precision, recall, and F1-score across all classes  
- Confusion matrix shows most misclassifications occur between **glioma and meningioma**, which is clinically reasonable.

---

## 📈 Evaluation and Visualization

- Training and validation accuracy/loss curves  
- Confusion matrix  
- Sample predictions with true vs predicted labels  

---

## 🛠️ Technologies Used

- Python  
- TensorFlow / Keras  
- NumPy  
- Matplotlib  
- Scikit-learn  

---

## 🚀 Future Improvements

- Apply **Transfer Learning** (VGG16, ResNet50, EfficientNet)  
- Add **model interpretability** using Grad-CAM  
- Train on larger and real clinical datasets  
- Deploy as a **web or desktop application**  

---

## ⚠️ Disclaimer

This project is intended **for educational and research purposes only**.  
It is **not a medical diagnostic tool** and should not be used as a substitute for professional medical advice.

---

## 👩‍🎓 Author

Master’s Degree – Artificial Intelligence  
 Deep Learning Project
