# Neural Networks Project

**University:** University of Belgrade, School of Electrical Engineering
**Course:** Neural Networks
**Year:** 2024/25  

## Overview

Three independent tasks:

1. Fully connected neural network for binary classification  
2. Convolutional neural network for multi-class image classification
3. Fuzzy regulator (this task was prepared by a colleague)

---

### Task 1 – Fully Connected Network  

**Dataset:** `Star.csv`  
- 5852 samples  
- 8 features (Mean IP, Std IP, Excess kurtosis IP, Skewness IP, Mean DM-SNR, Std DM-SNR, Excess kurtosis DM-SNR, Skewness DM-SNR)  
- 2 classes (0 – not a star, 1 – star)  
- Binary classification of radio signals (pulsar candidates)  
- Imbalanced: 72% class 0 / 28% class 1

**Model**
- Loss: Binary cross-entropy  
- Hidden activation: ReLU  
- Output activation: Sigmoid  
- Optimizer: Adam  
- Regularization: Dropout + Early Stopping  

**Hyperparameter search**
- Hidden neurons: [16, 32, 64]
- Dropout rate: [0, 0.2, 0.25]
- Learning rate: [0.001, 0.01, 0.1]

Best found: `neurons=64`, `dropout=0.0`, `lr=0.1`

**Results**
- Class distribution histogram
- Training / validation accuracy &amp; loss curves
- Accuracy, Precision, Recall, F1-score, ROC AUC
- Confusion matrices (train &amp; test)

---

### Task 2 – Convolutional Neural Network  

**Dataset:** Brain Tumor MRI Multi-Class Dataset  
- 16 269 images  
- 4 classes: Glioma, Meningioma, Pituitary, Healthy  
- Noticeable class imbalance (Healthy is the largest class)

**Preprocessing**
- Resize to 240×240
- Pixel normalization
- Data augmentation

**Model**
- Loss: Categorical cross-entropy  
- Hidden activation: ReLU  
- Output activation: Softmax  
- Optimizer: Adam  
- Regularization: Dropout + Early Stopping  

**Results**
- Sample images from each class
- Training / validation curves
- Accuracy, Precision, Recall, F1-score
- Confusion matrices
- Examples of correctly and incorrectly classified images

