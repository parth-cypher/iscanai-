# 👁️ iScanAI — AI-Powered Cataract Detection

[![Live Demo](https://img.shields.io/badge/🤗%20Hugging%20Face-Live%20Demo-blue)](https://huggingface.co/spaces/parth888/iscanai)
[![License](https://img.shields.io/badge/License-Apache%202.0-green)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.11-yellow)](https://python.org)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange)](https://tensorflow.org)
[![Accuracy](https://img.shields.io/badge/Accuracy-93.3%25-brightgreen)]()

> Upload an external eye photo and get instant AI-powered cataract screening.

---

## 🌐 Live Demo

**Try it now:** [https://huggingface.co/spaces/parth888/iscanai](https://huggingface.co/spaces/parth888/iscanai)

No installation needed. Just upload an eye photo and click Analyze.

---

## 📌 About

iScanAI is an end-to-end deep learning system that detects cataracts from external eye photographs. It is designed to make early cataract screening accessible without requiring specialized medical equipment like slit lamps or fundus cameras.

Cataracts are the leading cause of preventable blindness worldwide, responsible for over 51% of global blindness cases. Early detection is critical for timely surgical intervention — especially in low-resource settings.

---

## ✅ Results

| Class    | Precision | Recall | F1-Score | Support |
|----------|-----------|--------|----------|---------|
| Normal   | 0.964     | 0.900  | 0.931    | 30      |
| Cataract | 0.906     | 0.967  | 0.935    | 30      |
| **Overall Accuracy** | | | **93.3%** | 60 |

### Confusion Matrix

|               | Predicted Normal | Predicted Cataract |
|---------------|------------------|--------------------|
| Actual Normal | 27 ✅            | 3 ❌               |
| Actual Cataract | 1 ❌           | 29 ✅              |

---

## 🧠 How It Works

1. User uploads an external eye photograph
2. Image is preprocessed and resized to 224x224
3. MobileNetV2 model runs inference
4. System returns prediction, confidence score, and cataract probability

---

## 🏗️ Architecture

- **Base Model:** MobileNetV2 (pretrained on ImageNet)
- **Fine-tuned:** Last 10 layers unfrozen
- **Head:** GlobalAveragePooling2D → BatchNorm → Dense(64) → Dropout(0.6) → Dense(32) → Sigmoid
- **Regularization:** L2 (1e-3) + Dropout
- **Training:** Adam (lr=5e-5), Binary Cross-Entropy, Class Weights, EarlyStopping

---

## 🗂️ Project Structure

```
CataractDetectionAI/
├── app.py                          # Gradio web app (Hugging Face)
├── model_utils.py                  # Prediction logic and utilities
├── train_model.py                  # Model training pipeline
├── predict.py                      # CLI prediction script
├── augment_dataset.py              # Data augmentation tools
├── dataset_tools.py                # Dataset management utilities
├── split.py                        # Train/val split utility
├── cataract_external_model_metadata.json  # Model metadata
├── templates/                      # Flask HTML templates
├── static/                         # CSS and JS files
├── test_images/                    # Sample test images
└── requirements.txt                # Python dependencies
```

> ⚠️ The trained model file (`cataract_external_model.h5`) is not included in this repo due to GitHub file size limits. It is hosted on Hugging Face Spaces.

---

## 🚀 Run Locally

### 1. Clone the repo
```bash
git clone https://github.com/YOUR_USERNAME/iscanai.git
cd iscanai
```

### 2. Create virtual environment
```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Train the model
```bash
python train_model.py
```

### 5. Run the web app
```bash
python app.py
```

Then open `http://127.0.0.1:7860` in your browser.

---

## 📦 Requirements

```
gradio>=4.0.0
tensorflow>=2.12.0
Pillow>=9.0.0
numpy>=1.23.0
```

---

## 🔍 Key Improvements Made

- **Relaxed image filtering thresholds** — prevented 84% of valid images from being rejected
- **Augmentation capped at 5x** — stopped the model from memorizing duplicate images
- **Reduced unfrozen layers (30 → 10)** — fewer trainable parameters = less overfitting
- **Removed visual override rule** — eliminated brightness-based false positives
- **Added class weighting** — fixed imbalanced dataset bias
- **Added L2 regularization + BatchNorm** — better generalization

---

## ⚠️ Disclaimer

iScanAI is intended for **screening purposes only**. It does not replace clinical diagnosis by a qualified ophthalmologist. Always consult a medical professional for eye health concerns.

---

## 📄 License

This project is licensed under the [Apache 2.0 License](LICENSE).

---

## 👤 Author

**Parth Patil**
- Hugging Face: [parth888](https://huggingface.co/parth888)
- GitHub: [@parth-cypher](https://github.com/parth-cypher)

---

## 🌟 If you found this useful, give it a star!
