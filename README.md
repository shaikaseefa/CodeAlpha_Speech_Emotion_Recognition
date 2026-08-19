# 🎤 Speech Emotion Recognition using Deep Learning (CNN)

## 📌 Project Overview

This project is developed as part of the **CodeAlpha Machine Learning Internship**.

The application recognizes human emotions from speech using a Convolutional Neural Network (CNN). Users can upload a `.wav` audio file through a Flask web application, and the trained model predicts the speaker's emotion with a confidence score.

---

## 🎯 Objectives

- Detect emotions from speech audio.
- Build a deep learning model using CNN.
- Extract MFCC audio features.
- Develop a Flask web application for emotion prediction.
- Display predicted emotion with confidence score.

---

## 😊 Supported Emotions

- Angry 😠
- Calm 😌
- Disgust 🤢
- Fearful 😨
- Happy 😊
- Neutral 😐
- Sad 😢
- Surprised 😲

---

## 🧠 Technologies Used

- Python
- TensorFlow / Keras
- Flask
- Librosa
- NumPy
- Scikit-learn
- Matplotlib
- HTML
- CSS

---

## 📂 Dataset

Dataset Used:

**RAVDESS (Ryerson Audio-Visual Database of Emotional Speech and Song)**

Number of Audio Files:
- 1440 Speech Samples

Audio Format:
- WAV

Sample Rate:
- 48 kHz

---

## 🏗 Project Structure

```text
EmotionRecognitionAI/
│
├── app.py
├── predict.py
├── requirements.txt
├── README.md
│
├── dataset/
├── saved_model/
├── src/
├── templates/
├── static/
├── uploads/
│
├── accuracy_graph.png
├── loss_graph.png
└── venv/
```

---

## ⚙️ Installation

Clone the repository:

```bash
git clone <repository-link>
```

Create Virtual Environment:

```bash
python -m venv venv
```

Activate Virtual Environment:

Windows

```bash
venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the Flask application:

```bash
python app.py
```

Open your browser:

```
http://127.0.0.1:5000
```

---

## 🚀 Features

- Upload WAV audio files
- Speech emotion prediction
- CNN-based deep learning model
- MFCC feature extraction
- Confidence score
- Clean and responsive interface

---

## 📊 Model Evaluation

Evaluation Metrics:

- Accuracy
- Precision
- Recall
- F1-Score
- Confusion Matrix

Training graphs:

- Accuracy Graph
- Loss Graph

---

## 🔮 Future Improvements

- Real-time microphone emotion detection
- Speech recording inside the application
- Support for multiple languages
- Improved CNN architecture
- Data augmentation for higher accuracy

---

## 👩‍💻 Author

**Shaik Aseefa**

B.Tech Computer Science Engineering

CodeAlpha Machine Learning Internship

---

## 📜 License

This project is developed for educational and internship purposes.