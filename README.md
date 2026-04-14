# 🏥 AI-Powered Medical Image Analysis System  
### Deep Learning-Based Pneumonia Detection from Chest X-Rays

![Python](https://img.shields.io/badge/Python-3.14-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-orange.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.30+-red.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

---

## 🚀 Overview
This project is an **AI-Powered Medical Image Analysis System** that detects **Pneumonia from Chest X-Ray images** using Deep Learning.

It demonstrates how a **Convolutional Neural Network (CNN)** can:
- Process medical images  
- Extract meaningful features  
- Predict disease presence in real-time  

The system is built as a **complete end-to-end ML application**, including a **Streamlit-based web interface** for easy interaction.

---

## 🩺 Problem Statement
- Radiologists analyze hundreds of X-rays daily  
- Fatigue can lead to **false negatives (missed diagnoses)**  
- Delayed detection can critically impact patient health  

### ✅ Proposed Solution
An AI-based diagnostic assistant that:
- Classifies X-rays as **Normal or Pneumonia**  
- Provides **confidence scores**  
- Acts as a **second opinion system**  

---

## 🌍 Industry Relevance
- Healthcare AI is rapidly growing in **medical imaging diagnostics**  
- Used in:
  - Hospitals & radiology labs  
  - Telemedicine platforms  
  - AI-assisted diagnostic tools  

### 💡 Real-World Impact
- Reduces diagnostic workload  
- Improves accuracy  
- Enables faster decision-making  

---

## 💻 Tech Stack

### Core Technologies
- Python 3.14  
- PyTorch & TorchVision  
- Streamlit  

### Data Processing
- NumPy  
- Pandas  
- OpenCV  
- PIL  

### Visualization
- Matplotlib  
- Seaborn  

---

## 📂 Dataset
Dataset used:  
👉 https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia  

### Structure
```
data/chest_xray/
├── train/
├── val/
└── test/
```

---

## 🏗️ Architecture

### Pipeline Flow
```
X-Ray Image
   ↓
Grayscale Conversion
   ↓
Resize (224x224)
   ↓
Normalization
   ↓
CNN Feature Extraction
   ↓
Fully Connected Layers
   ↓
Sigmoid Output (Probability)
```

### Model Details
- 4 × Conv Blocks (`Conv2D + ReLU + MaxPool`)  
- Flatten Layer  
- Fully Connected Layer (512 neurons)  
- Dropout (0.5)  
- Output Layer (Sigmoid activation)  

---

## 🛠️ Installation

```bash
git clone https://github.com/Anupam-Santra/AI-Medical-Image-Analysis.git
cd AI-Medical-Image-Analysis

python -m venv venv

# Activate environment
venv\Scripts\activate   # Windows
source venv/bin/activate  # Mac/Linux

pip install -r requirements.txt
```

---

## 🚀 Usage

### 🔹 Run Web Application
```bash
streamlit run app.py
```

Open in browser:
```
http://localhost:8501
```

---

### 🔹 One-Click Launch (Windows)
Double-click:
```
start_app.bat
```

---

### 🔹 Train Model
```bash
python src/train.py
```

---

### 🔹 Evaluate Model
```bash
python src/predict.py
```

---

### 🔹 Predict Single Image
```bash
python src/predict.py "image_path.jpg"
```

---

## 📊 Results

- Model successfully classifies Pneumonia vs Normal  
- Provides **confidence-based predictions**  
- Evaluation metrics include:
  - Accuracy  
  - Precision  
  - Recall  
  - F1 Score  

---

## 📸 Screenshots

### 🎥 Live Demo
![Demo](images/demo.gif)

### 📈 Training Graph
![Training Graph](outputs/training_history.png)

### 📊 Confusion Matrix
![Confusion Matrix](outputs/confusion_matrix.png)

---

## 🎓 Learning Outcomes

- Built a **complete Deep Learning pipeline**  
- Developed a **custom CNN model using PyTorch**  
- Created a **user-friendly ML web application**  
- Implemented **real-world evaluation metrics**  
- Learned **project structuring for production-ready systems**  

---

# 🔥 Additional Enhancements

## 📂 Project Structure
```
AI-Medical-Image-Analysis/
├── app.py
├── start_app.bat
├── requirements.txt
├── data/
├── images/
├── models/
├── outputs/
├── src/
└── venv/
```

---

## 🚀 Future Improvements
- Cloud deployment (AWS / Streamlit Cloud)  
- Multi-disease classification  
- Model explainability (Grad-CAM)  
- REST API integration (FastAPI)  

---

## 📄 License
MIT License  

---

## ⭐ Support
If you found this useful:
- Star ⭐ the repo  
- Fork 🍴 it  
- Share 🚀  

---

## 👨‍💼 Author
**Your Name**  
GitHub: https://github.com/Anupam-Santra  
LinkedIn: https://www.linkedin.com/in/anupam-santra-615832277/

---