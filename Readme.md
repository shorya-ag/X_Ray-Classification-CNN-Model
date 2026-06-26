# Chest X-Ray Classification using CNN

## 📌 Project Overview

This project implements a **Convolutional Neural Network (CNN)** to classify **Chest X-Ray images** into two categories:

* **Normal**
* **Pneumonia**

The model is built using **TensorFlow/Keras** and follows a complete deep learning workflow, including image preprocessing, CNN model development, training, validation, and performance evaluation.

The objective of this project is to develop a CNN model capable of accurately distinguishing between **Normal** and **Pneumonia** chest X-ray images through an efficient deep learning pipeline.

To improve readability and maintainability, the project is organized into separate Python files, where each file is responsible for a specific stage of the pipeline.

---

## 📂 Project Structure

```text
Chest_XRay_Classification_CNN/
│
├── Libraries.py
├── Training_Data.py
├── Validation_Data.py
├── CNN_Model.py
├── Model_Evaluation.py

```

---

## 📁 File Description

### 📌 Libraries.py

This file contains all the required Python libraries and dependencies used throughout the project.

Some of the major libraries include:

* tensorFlow
* keras
* os
* OpenCV
* Matplotlib

---

### 📌 Training_Data.py

This script is responsible for preprocessing the training dataset.

The preprocessing pipeline includes:

* Loading training images
* Converting RGB images to grayscale (if required)
* Resizing every image to **100 × 100 pixels**
* Shuffling the dataset
* Feature scaling (pixel normalization)
* Preparing training features and labels

---

### 📌 Validation_Data.py

This file preprocesses the validation dataset using the same steps as the training data to ensure consistency.

The preprocessing steps include:

* Loading validation images
* Grayscale conversion (if required)
* Image resizing to **100 × 100 pixels**
* Dataset shuffling
* Feature scaling
* Preparing validation features and labels

---

### 📌 CNN_Model.py

This file contains the architecture of the Convolutional Neural Network used for binary image classification.

#### Model Architecture

* 3 Convolutional Layers
* 3 Max Pooling Layers
* Dropout (0.2) after each convolution block
* Flatten Layer
* Fully Connected (Dense) Hidden Layer
* Output Layer 

#### Activation Functions

* Convolutional Layers → **ReLu**
* Output Layer → **Sigmoid**

---

### 📌 Model_Evaluation.py

This file is responsible for evaluating the performance of the trained CNN model. It generates the following outputs:

- **Model Summary** of the CNN architecture
- **Training and Validation Accuracy** metrics
- **Accuracy Graph** showing the learning performance across training epochs

The evaluation results are saved in the repository as:

- 📄 **Model_Summary.jpeg** – Screenshot of the complete CNN model architecture and layer details.
- 📈 **Accuracy_Graph.png** – Visualization of the training and validation accuracy over the training epochs.

These outputs help in understanding the model architecture and analyzing its learning performance.

---

## 🧠 CNN Workflow

```
 Chest X-Ray Image
        │
        ▼
 Image Preprocessing
        │
        ▼
 Convolution Layer
        │
   Max Pooling
        │
        ▼
   Flatten Layer
        │
        ▼
Dense Hidden Layer (ReLU)
        │
        ▼
Output Layer (Sigmoid)
        │
        ▼
    Prediction
 (Normal / Pneumonia)
```

---



