import os
import numpy as np
import librosa
import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

import tensorflow as tf

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Conv2D,
    MaxPooling2D,
    Dense,
    Dropout,
    Flatten,
    BatchNormalization
)

from tensorflow.keras.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    ReduceLROnPlateau
)

# =====================================================
# DATASET PATH
# =====================================================

DATASET_PATH = "dataset"

# =====================================================
# EMOTION LABELS
# =====================================================

emotion_dict = {
    "01": "neutral",
    "02": "calm",
    "03": "happy",
    "04": "sad",
    "05": "angry",
    "06": "fearful",
    "07": "surprised"
}

# =====================================================
# FEATURE EXTRACTION
# =====================================================

def extract_features(file_path):

    signal, sample_rate = librosa.load(file_path, sr=None)

    mfcc = librosa.feature.mfcc(
        y=signal,
        sr=sample_rate,
        n_mfcc=40
    )

    max_len = 174

    if mfcc.shape[1] < max_len:

        pad_width = max_len - mfcc.shape[1]

        mfcc = np.pad(
            mfcc,
            pad_width=((0, 0), (0, pad_width)),
            mode="constant"
        )

    else:

        mfcc = mfcc[:, :max_len]

    return mfcc

# =====================================================
# LOAD DATASET
# =====================================================

X = []
y = []

print("Loading dataset...")

for actor in os.listdir(DATASET_PATH):

    actor_path = os.path.join(DATASET_PATH, actor)

    if os.path.isdir(actor_path):

        for file in os.listdir(actor_path):

            if file.endswith(".wav"):

                emotion_code = file.split("-")[2]

                emotion = emotion_dict[emotion_code]

                file_path = os.path.join(actor_path, file)

                features = extract_features(file_path)

                X.append(features)

                y.append(emotion)

print("Dataset Loaded Successfully!")

# =====================================================
# CONVERT TO NUMPY
# =====================================================

X = np.array(X)

print("Original Shape :", X.shape)

# CNN expects 4D input

X = X[..., np.newaxis]

print("CNN Input Shape :", X.shape)

# =====================================================
# LABEL ENCODING
# =====================================================

encoder = LabelEncoder()

y = encoder.fit_transform(y)

print("Classes:")

print(encoder.classes_)

# =====================================================
# TRAIN TEST SPLIT
# =====================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print()

print("Training Shape :", X_train.shape)

print("Testing Shape :", X_test.shape)

print()

print("Preprocessing Completed Successfully!")