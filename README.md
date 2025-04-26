# 😰 Stress Detection with Emotions – Real-Time Emotion Recognition using CNN

A **Python-based deep learning project** that uses **Convolutional Neural Networks (CNN)** and **OpenCV** to recognize facial expressions in real-time and detect stress levels based on emotional cues.

> ⚠️ **Note:** This project uses your computer’s webcam for real-time facial emotion detection and must be run locally.

---
Download Dataset from kaggle : https://www.kaggle.com/datasets/msambare/fer2013
After downloading, you will have a zip file, extract it, and copy the location of that folder in section-2 of the code.

## 🚀 Features

- 🎥 **Real-time emotion detection** using OpenCV
- 🧠 **Deep Learning model (CNN)** trained on facial emotion datasets
- 📊 Classifies emotions: Angry, Disgust, Fear, Happy, Neutral, Sad, Surprise
- 😓 Maps emotions to **stress levels** (e.g., *Highly Stressed*, *Low Stressed*, *Not Stressed*)
- 📁 Support for **custom image dataset** with folder-based structure
- 🧪 Optional webcam-based emotion monitoring

---

## 🛠️ Tech Stack

- Python 3.9+ (✅ tested with **Python 3.13.2**)
- `tensorflow`, `keras`
- `opencv-python`
- `scikit-learn`
- `numpy`, `pandas`
- `matplotlib` (for visualization)
- `pywin32` (📍Windows only – required for webcam access)

---

## 📦 Installation

```bash
git clone https://github.com/yourusername/Stress-Detection-With-Emotions.git
cd Stress-Detection-With-Emotions
pip install -r requirements.txt
```

✅ Make sure:
- Your system has a working **webcam**
- You're using **Python 3.9–3.10** (TensorFlow not yet supported on 3.11+)

---

## 🧑‍💻 How to Run

### 🧠 Train the CNN Model:

```bash
python train_model.py
```

> This will train your model on the dataset inside the `train/` and `test/` folders and save the model as `model.h5`.

### 👁️ Start Real-Time Emotion & Stress Detection:

```bash
python detect_stress.py
```

> This opens your webcam, detects faces, classifies emotions, and displays stress level in real-time.

---

## 📁 Project Structure

```
📦 Stress-Detection-With-Emotions/
 ┣ 📁 aboutttttt/
 ┃ ┣ 📁 train/            # Emotion image folders: Angry, Happy, etc.
 ┃ ┗ 📁 test/             # Testing set with same folder structure
 ┣ 📜 train_model.py      # Script to train CNN model
 ┣ 📜 detect_stress.py    # Webcam-based emotion/stress detection
 ┣ 📜 model.h5            # Saved CNN model
 ┣ 📜 requirements.txt    # List of Python dependencies
 ┗ 📜 README.md           # Project documentation
```
