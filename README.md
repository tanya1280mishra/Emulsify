# 🎧 Emulsify: Emotion-Based Music Recommendation System

**Emulsify** is an intelligent system that integrates facial emotion recognition, heart rate variability analysis, and stress prediction to deliver personalized music recommendations that enhance emotional well-being and reduce stress.

---

## 🧠 Overview

Emulsify bridges the gap between **mental health support** and **technology** by leveraging computer vision, physiological signals, and machine learning for **context-aware music therapy**.

---

## 🗂️ Project Structure

### 1. **Stress Prediction Pipeline**

* **Data Collection**: Collect physiological signals and emotion-related data.
* **Preprocessing**: Clean and format data for modeling.
* **Feature Engineering**: Extract meaningful features from raw signals.
* **Label Encoding**: Convert categorical variables into numerical format.
* **Scaling**: Normalize features.
* **Model Design & Training**: Train stress prediction models.
* **Validation & Evaluation**: Assess model performance.

### 2. **Facial Emotion Recognition**

* **Face Detection**: Use Haar Cascade Classifier to detect faces in real-time.
* **Image Preprocessing**: Resize, grayscale, and normalize face images.
* **Emotion Classification**: CNN-based classifier to predict emotions from facial features.

### 3. **Song Recommendation System**

* **Data Collection**: Gather song metadata and user preferences.
* **Feature Engineering**: Extract acoustic and contextual song features.
* **Normalization & Clustering**: K-Means to cluster songs based on emotional profiles.
* **Classification**: LightGBM model for emotion-to-song mapping.
* **Ranking & Recommendation**: Rank songs using popularity and match scores.

---

## 🔄 Combined System Workflow

1. Real-time **Facial Emotion Detection**
2. **Heart Rate Variability Analysis**
3. **Stress Level Prediction**
4. **Emotion Classification**
5. **Music Recommendation**

---

## 🧰 Technological Stack

| Component         | Technology             |
| ----------------- | ---------------------- |
| Computer Vision   | OpenCV                 |
| Deep Learning     | Keras, TensorFlow      |
| ML Algorithms     | Scikit-learn, LightGBM |
| Data Manipulation | Pandas                 |

---

## 🎯 Applications

* 🎵 **Personalized Music Therapy**
* 💆 **Stress & Anxiety Management**
* 💡 **Emotional Awareness Tools**
* 🧘 **Well-being Enhancement Platforms**

---

## 🚧 Challenges Addressed

* Accurate **emotion recognition** from facial cues.
* Real-time **stress level prediction** using heart rate data.
* Adaptive and contextual **song recommendation** based on current emotional state.

---

## 📌 Key Highlights

* End-to-end integration of **vision**, **bio-signals**, and **recommendation engines**.
* Built with a modular architecture, enabling flexibility in model improvements and extension.
* Supports **real-time processing** for practical deployment in mobile or desktop environments.

---

## 📎 License

This project is for academic and research purposes. Licensing terms can be updated based on deployment requirements.

## 🚀 Getting Started

Follow these instructions to set up the project locally for development and testing purposes.

### ✅ Prerequisites

Make sure you have the following installed:

* Python 3.8+
* pip
* Git

Install required Python libraries:

```bash
pip install -r requirements.txt
```

> If `requirements.txt` is not present, install manually:

```bash
pip install opencv-python keras tensorflow scikit-learn pandas lightgbm
```

---

## 📂 Project Structure (Typical)

```
Emulsify/
├── face_recognition/
│   ├── detector.py
│   └── emotion_model.h5
├── stress_prediction/
│   ├── preprocess.py
│   ├── model.py
├── song_recommendation/
│   ├── recommender.py
│   ├── clustering_model.pkl
│   └── songs_dataset.csv
├── combined_system.py
├── requirements.txt
└── README.md
```
## 🧪 Usage

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/emulsify.git
cd emulsify
```
## 📊 Example Output

* Detected Emotion: `Sad`
* Predicted Stress Level: `High`
* 🎶 Recommended Songs:

  * “Let It Be” – The Beatles
  * “Weightless” – Marconi Union

---

## 💡 Tips

* For better real-time results, run on a GPU-enabled system.
* You can replace the song dataset (`songs_dataset.csv`) with your own music metadata.
* Use a spotify sensor API (like Polar or Fitbit) for live stress detection.
