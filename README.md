# AI-Powered Medical Image Analysis System 🏥🔬

![Python](https://img.shields.io/badge/Python-3.14-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-orange.svg)
![TorchVision](https://img.shields.io/badge/TorchVision-0.15+-red.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

## Overview 🚀
This project is an **AI-Powered Medical Image Analysis System** designed to assist radiologists in diagnosing **Pneumonia from Chest X-Rays** with high accuracy using Deep Learning. It showcases how a Convolutional Neural Network (CNN) can process medical imaging data, extract vital features (like opacities), and predict illness in a fraction of a second.

This repository is built as an end-to-end industry-standard virtual simulation. It features robust preprocessing pipelines, custom PyTorch model architecture, real-time metrics tracking on CPU/GPU, and evaluation outputs. 

## Problem Statement 🩺
- **The Challenge:** Radiologists review hundreds of scans daily. Fatigue leads to a higher rate of false negatives (missing an illness), delaying critical patient care.
- **The Solution:** An AI binary classifier (Normal vs. Pneumonia) functioning as a "second pair of eyes," providing a preliminary diagnosis and confidence score instantly. 

## Tech Stack 💻
- **Language:** Python 3.14
- **Deep Learning:** PyTorch & TorchVision
- **Computer Vision:** OpenCV
- **Data Manipulation:** NumPy & Pandas
- **Visualization:** Matplotlib & Seaborn
- **Architecture:** Lightweight Sequential CNN (`nn.Module`) optimized for grayscale medical images.

## Architecture 🏗️
The images flow through our automated PyTorch pipeline:
1. **Input Stage:** Kaggle X-Ray JPEG -> **`ImageFolder` + `transforms`** -> **Grayscale (1 Channel)** -> **Resize (224x224)** -> **Normalized Tensor**.
2. **Feature Extraction:** 4 blocks of `nn.Conv2d` -> `nn.ReLU` -> `nn.MaxPool2d`.
3. **Classification:** `nn.Flatten` -> `nn.Linear (512)` -> `nn.Dropout (50%)` -> `nn.Linear (1)`.
4. **Output:** A percentage probability of Pneumonia via Sigmoid activation.

## Dataset Setup 📂
This project uses the publicly available **[Chest X-Ray Images (Pneumonia) Dataset](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)** from Kaggle. 

**Instructions to setup data:**
1. Download the dataset from Kaggle.
2. Extract the archive.
3. Move the `train`, `val`, and `test` folders directly into `data/chest_xray/`.
Your structure should look like this:
```
AI-Medical-Image-Analysis/
└── data/
    └── chest_xray/
        ├── train/
        │   ├── NORMAL/
        │   └── PNEUMONIA/
        ├── test/ ...
        └── val/ ...
```

## Installation 🛠️
```bash
# 1. Clone this repository
git clone https://github.com/YourUsername/AI-Medical-Image-Analysis.git
cd AI-Medical-Image-Analysis

# 2. Create a virtual environment
python -m venv venv

# For Windows
venv\Scripts\activate
# For Mac/Linux
source venv/bin/activate

# 3. Install required libraries
pip install -r requirements.txt
```

## Usage 🏃‍♂️

### 1. Train the Model
Ensure your data is in the `data/` folder and run:
```bash
python src/train.py
```
*This will output the training process, generate an accuracy curve to `/outputs/training_history.png`, and save the best weights `pneumonia_cnn_model.pth` into the `/models/` folder.*

### 2. Evaluate the System
Test the model on the unseen `test` dataset:
```bash
python src/predict.py
```
*This will print the Test Accuracy/Loss, generate a full Classification Report (Precision/Recall/F1), and save a Confusion Matrix diagram to `/outputs/confusion_matrix.png`.*

### 3. Predict a Single Image (Doctor Simulation)
```bash
python src/predict.py "data/chest_xray/val/PNEUMONIA/person1946_bacteria_4874.jpeg"
```
*Output:*
```
--- AI DIAGNOSIS ---
Result: PNEUMONIA
Confidence: 98.42%
```

## Results & Screenshots 📈
*(After running the code, place your screenshots in the `images/` folder and uncomment the links below!)*
- **Training Graph:** `![Training Graph](outputs/training_history.png)`
- **Confusion Matrix:** `![Confusion Matrix](outputs/confusion_matrix.png)`

## Learning Outcomes 🎓
- Gained hands-on experience structuring a Deep Learning repository.
- Built a custom Convolutional Neural Network from scratch in PyTorch.
- Mastered dynamic DataLoaders for unstructured image data.
- Implemented real-world model evaluation metrics (Confusion Matrix, F1-Score).
