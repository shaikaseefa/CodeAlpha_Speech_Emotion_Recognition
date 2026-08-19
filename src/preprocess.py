import os
import librosa
import numpy as np

# Dataset path
DATASET_PATH = "dataset"

# Emotion labels
emotion_dict = {
    "01": "neutral",
    "02": "calm",
    "03": "happy",
    "04": "sad",
    "05": "angry",
    "06": "fearful",
    "07": "surprised"
}

# Lists to store data
X = []
y = []

# Function to extract MFCC features
def extract_features(file_path):
    signal, sample_rate = librosa.load(file_path, sr=None)

    mfcc = librosa.feature.mfcc(
        y=signal,
        sr=sample_rate,
        n_mfcc=40
    )

    # Convert variable-length MFCC to a fixed-length vector
    mfcc = np.mean(mfcc.T, axis=0)

    return mfcc

# Read dataset
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

print("Total Samples:", len(X))
print("First Feature Shape:", X[0].shape)
print("First Emotion:", y[0])