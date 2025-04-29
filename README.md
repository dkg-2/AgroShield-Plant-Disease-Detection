---
title: Agroshield Disease Prediction
emoji: 🔥
colorFrom: gray
colorTo: yellow
sdk: gradio
sdk_version: 5.25.2
app_file: app.py
pinned: false
license: mit
---

Check out the configuration reference at https://huggingface.co/docs/hub/spaces-config-reference

# 🌾 AgroShield - Plant Disease Detection Model

[![Hugging Face Spaces](https://img.shields.io/badge/Live%20Demo-Hugging%20Face-blue?logo=huggingface)](https://huggingface.co/spaces/dkg-2/Agroshield_disease_prediction)

AgroShield is a smart agriculture tool that helps farmers and agronomists identify plant diseases early using deep learning. This specific model detects diseases from leaf images and is deployed as an interactive Gradio app on Hugging Face Spaces.

---

## 🧠 Model Overview

This model is trained to classify plant leaves into **healthy** or **diseased** categories across multiple crops. It uses a **Convolutional Neural Network (CNN)** architecture and has been trained on the PlantVillage dataset.


## 📊 Dataset

- **Source**: [PlantVillage Dataset on Kaggle](https://www.kaggle.com/datasets/abdallahalidev/plantvillage-dataset)
- **Categories**: Includes several crop types and disease labels such as:
  - Tomato - Early Blight, Late Blight, Mosaic Virus, etc.
  - Potato - Early Blight, Late Blight
  - Apple - Apple Scab, Black Rot
  - ...and many more

---

## 🔧 Tech Stack

- **Framework**: TensorFlow / Keras
- **Web UI**: Gradio
- **Deployment**: Hugging Face Spaces
- **Language**: Python 3.x

---

## 🚀 Try It Out

- 📓 **Model Training Notebook (Colab)**:  
  👉 [Open in Colab](https://colab.research.google.com/drive/1AUMOm-TUhYP_vyrCBWuwXuO9gT5OIgoK?usp=sharing)

👉 **[Live Demo on Hugging Face Spaces](https://huggingface.co/spaces/dkg-2/Agroshield_disease_prediction)**

---

## 💻 Run Locally

### 1. Clone the repo
```bash
git clone https://github.com/dkg-2/AgroShield-Plant-Disease-Detection.git
cd AgroShield-Plant-Disease-Detection
pip install -r requirements.txt
python app/app.py

