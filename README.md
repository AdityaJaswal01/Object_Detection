# Object Detection using TensorFlow

This project implements a simple **object detection system** that performs **digit classification and bounding box regression** simultaneously using a custom Convolutional Neural Network trained on the MNIST dataset.

---

## 📌 Features

- Digit classification (0–9)
- Bounding box prediction
- Custom CNN architecture
- Training using TensorFlow Datasets (MNIST)
- IOU (Intersection over Union) evaluation
- Visualized predictions with bounding boxes

---

## 🧠 Tech Stack

- Python
- TensorFlow / Keras
- TensorFlow Datasets
- NumPy
- Matplotlib
- PIL

---

## 📂 Project Structure

├── ObjectDetection.py
├── README.md
└── .gitignore


---

## 📊 Dataset

- **MNIST** dataset loaded using `tensorflow_datasets`
- Digits are randomly placed inside a 75×75 image
- Model predicts:
  - Digit class (classification)
  - Bounding box coordinates (regression)

---

## 🧠 Model Architecture

- Convolution + Average Pooling layers
- Shared feature extractor
- Two output heads:
  - **Classification head** (Softmax)
  - **Bounding box head** (MSE loss)

Loss Functions:
- Classification: `categorical_crossentropy`
- Bounding Box: `mean squared error`

---

## 🚀 How to Run

### 1️⃣ Install dependencies
```bash
pip install tensorflow tensorflow-datasets numpy matplotlib pillow

2️⃣ Run the script
python ObjectDetection.py
