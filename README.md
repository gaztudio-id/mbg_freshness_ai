# MBG Freshness AI 🍎🥦

[![Python Version](https://img.shields.io/badge/python-3.8%20%7C%203.9%20%7C%203.10-blue.svg)](https://www.python.org/)
[![TensorFlow Version](https://img.shields.io/badge/tensorflow-2.11+-orange.svg)](https://www.tensorflow.org/)
[![Flask Version](https://img.shields.io/badge/flask-3.0+-green.svg)](https://flask.palletsprojects.com/)
[![License](https://img.shields.io/badge/license-MIT-lightgrey.svg)](LICENSE)

An editorial, real-time AI-powered freshness classification web application tailored for the **Makan Bergizi Gratis (MBG)** national program. Built using **MobileNetV2** transfer learning and served via a lightweight, high-performance Flask backend with a premium, responsive web interface.

Developed by **Kelompok 11 — Politeknik Caltex Riau**.

---

## 🌟 Key Features

### 1. Real-time Video Inference
* **Webcam Streaming:** Scans raw food ingredients via the device webcam.
* **Media Uploads:** Supports scanning via uploaded static images or pre-recorded videos.
* **Scan Viewport Effects:** Interactive neon scanning lines and color-coded bounding boxes representing prediction status (green for `segar`, red for `tidak_segar`).

### 2. Consumption Suitability Index
* **Piecewise Linear Scoring:** Calculates a precise percentage suitability score.
* **Actionable UX Recommendations:** Displays smart warnings based on safety thresholds (e.g., highly fresh, consume today, or discard/compost).

### 3. Interactive Floating "Dynamic Island" Navbar
* **Scroll-Responsive Capsule:** Smoothly morphs from a full header down to a compact floating capsule on scroll to maximize viewport estate.
* **Pulse & Glow States:** Border glows green dynamically during active AI inference.
* **Mobile-First Design:** Autocompacts on mobile viewports (< 768px) to fit all screen sizes perfectly.

### 4. Interactive Statistics & History
* **Live Session Metrics:** Displays total items scanned, average classification confidence, and count totals for fresh/stale items.
* **Scan Log:** Interactive time-stamped history list showing the 20 most recent prediction outcomes.

### 5. Training Report Dashboard
* **val_accuracy & val_loss metrics:** Shows key performance results from training logs.
* **Interactive Evaluation Charts:** Embeds training accuracy curves, training loss curves, and the evaluation confusion matrix.
* **Persisted Dark & Light Mode:** Switches themes smoothly with high-contrast neon stabilo highlights that persist across page reloads.

---

## 📊 Model & Training Overview

The underlying model is a customized deep neural network fine-tuned using transfer learning:

* **Base Model:** `MobileNetV2` (Pre-trained on ImageNet).
* **Custom Classification Head:** Global Average Pooling (GAP) ➔ Dense (128 units, ReLU) ➔ Dropout (0.5) ➔ Dense (1 unit, Sigmoid).
* **Optimizer:** Adam (learning rate = `0.0001`).
* **Loss Function:** Binary Crossentropy.
* **Data Augmentation:** Rotation, Width/Height Shift, Zoom, Horizontal Flip.
* **Dataset Size:** **64,627 images** (51,712 for training, 12,915 for validation) across two classes: `segar` (fresh) and `tidak_segar` (stale).
* **Early Stopping:** Triggered at **Epoch 28** based on validation loss patience.

### 📈 Metrics Summary
* **Validation Accuracy (`val_accuracy`):** `98.67%`
* **Validation Loss (`val_loss`):** `0.0372`
* **Average Training Time:** ~29 minutes (~1s/step, 1616 steps per epoch).

---

## 🛠️ Tech Stack

* **Backend:** Python 3.10+, Flask, TensorFlow 2.11+, OpenCV, NumPy.
* **Frontend:** HTML5 (Semantic Structure), Vanilla CSS3 (Synthesis Capital design tokens, HSL typography, CSS transitions), Javascript (ES6+, Fetch API, Canvas rendering).
* **Design Tokens:** Deep Indigo primary (`#120A59`) and vibrant neon lime accents (`#D8F500`).

---

## 🚀 Getting Started

### Prerequisites

Ensure you have Python 3.8+ installed on your local machine.

### Installation

1. **Clone the Repository:**
   ```bash
   git clone https://github.com/gaztudio-id/mbg_freshness_ai.git
   cd mbg_freshness_ai
   ```

2. **Set Up a Virtual Environment (Recommended):**
   ```bash
   python -m venv venv
   # On Windows:
   venv\Scripts\activate
   # On macOS/Linux:
   source venv/bin/activate
   ```

3. **Install Dependencies:**
   Ensure you install TensorFlow, OpenCV, Flask, and NumPy:
   ```bash
   pip install tensorflow flask opencv-python numpy
   ```

4. **Verify Model File:**
   Make sure the `mbg_freshness_model.h5` weights file is placed in the root directory.

### Running the App

1. **Start the Flask Server:**
   ```bash
   cd web
   python app.py
   ```
2. **Access the Web UI:**
   Open your browser and navigate to `http://127.0.0.1:8080`.

---

## 📂 Project Structure

```text
mbg_freshness_ai/
├── web/
│   ├── app.py                     # Flask server entrypoint & backend APIs
│   ├── static/                    # Training charts and static evaluation assets
│   │   ├── confusion_matrix.png
│   │   ├── grafik_akurasi.png
│   │   └── grafik_loss.png
│   └── templates/
│       └── index.html             # Main dashboard template (Synthesis Capital layout)
├── dataset_preparation.py         # Script to harmonize and split image datasets
├── mbg_freshness_model.h5         # Trained MobileNetV2 Keras model file
├── model_training.ipynb           # Notebook documenting model development & training
├── .gitignore                     # Configured gitignore for python & dataset files
└── README.md                      # Professional project documentation
```

---

## 👥 Contributors

* **Kelompok 11 — Politeknik Caltex Riau**
  * Ghaswul Fikri Fadhillah
  * *and team members*

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
