<h1 align="center"> Brain Tumor MRI Classification</h1>

<p align="center">
Deep Learning project for automatic brain tumor classification from MRI images.
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10-blue"/>
  <img src="https://img.shields.io/badge/TensorFlow-DeepLearning-orange"/>
  <img src="https://img.shields.io/badge/Status-Completed-green"/>
  <img src="https://img.shields.io/badge/License-Educational-lightgrey"/>
</p>

---

<h2>📌 Project Summary</h2>

<p>
This project builds an AI system that classifies brain MRI scans into <b>4 categories</b>:
</p>

<ul>
  <li>Glioma</li>
  <li>Meningioma</li>
  <li>Pituitary Tumor</li>
  <li>No Tumor</li>
</ul>

<p>
Includes a <b>Custom CNN baseline</b>, <b>Transfer Learning models</b>,
<b>Grad-CAM interpretability</b>, and optional <b>YOLOv8 detection</b>.
</p>

---

<h2>📂 Dataset</h2>

<p>
<b>Brain Tumor MRI Dataset – Kaggle</b><br/>
<a href="https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset">
https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset
</a>
</p>

<ul>
  <li>~7,000 MRI images</li>
  <li>4 balanced classes</li>
  <li>Axial T1 scans</li>
  <li>Resized to 224×224</li>
</ul>

---

<h2>⚙️ Pipeline</h2>

<b>Preprocessing</b>
<ul>
  <li>Resize → 224×224</li>
  <li>Normalize → [0,1]</li>
  <li>Grayscale → RGB</li>
  <li>One-Hot Encoding</li>
</ul>

<b>Data Augmentation</b>
<ul>
  <li>Rotation / Translation</li>
  <li>Flip</li>
  <li>Zoom & Shear</li>
  <li>Brightness & Contrast</li>
</ul>

---

<h2>🧠 Models</h2>

<b>Custom CNN</b>
<ul>
  <li>4 Conv Blocks + MaxPooling</li>
  <li>Dense(512) + Dropout</li>
  <li><b>Accuracy ≈ 95%</b></li>
</ul>

<b>Transfer Learning</b>
<ul>
  <li>DenseNet121</li>
  <li>ResNet50</li>
  <li>EfficientNetB0</li>
  <li>VGG16</li>
</ul>

---

<h2>📊 Results</h2>

<table>
<tr><th>Model</th><th>Accuracy</th></tr>
<tr><td>CNN</td><td>95.4%</td></tr>
<tr><td>VGG16</td><td>98.1%</td></tr>
<tr><td>EfficientNetB0</td><td>98.1%</td></tr>
<tr><td>ResNet50</td><td>98.8%</td></tr>
<tr><td><b>DenseNet121</b></td><td><b>98.8%</b></td></tr>
</table>

<p>
<b>Key Notes:</b><br/>
• Transfer Learning +3% vs CNN<br/>
• Best Models: DenseNet121 & ResNet50<br/>
• Main confusion: Glioma vs Meningioma
</p>

---

<h2>🔍 Interpretability</h2>
<p><b>Grad-CAM heatmaps</b> highlight tumor-relevant regions for transparency.</p>

---

<h2>📍 Optional</h2>
<p><b>YOLOv8</b> used for tumor detection & localization.</p>

---

<h2>🛠️ Tech Stack</h2>
<p>
Python • TensorFlow/Keras • Scikit-learn • NumPy • Matplotlib • YOLOv8 • Google Colab
</p>

---

<h2>🚀 Future Work</h2>
<ul>
  <li>Vision Transformers</li>
  <li>Segmentation (U-Net)</li>
  <li>Model Ensembles</li>
  <li>Web Deployment</li>
</ul>

---

<h2>⚠️ Disclaimer</h2>
<p>
Educational & research purposes only — <b>not a medical diagnostic tool</b>.
</p>
