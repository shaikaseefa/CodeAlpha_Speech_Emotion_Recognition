import os
import numpy as np

from extract_features import extract_features

emotion_dict = {
    "01": "neutral",
    "02": "calm",
    "03": "happy",
    "04": "sad",
    "05": "angry",
    "06": "fearful",
    "07": "disgust",
    "08": "surprised"
}


def load_dataset(dataset_path="dataset"):

    X = []
    y = []

    print("Loading dataset...")

    for actor in os.listdir(dataset_path):

        actor_path = os.path.join(dataset_path, actor)

        if not os.path.isdir(actor_path):
            continue

        for file in os.listdir(actor_path):

            if file.endswith(".wav"):

                emotion_code = file.split("-")[2]

                emotion = emotion_dict[emotion_code]

                file_path = os.path.join(actor_path, file)

                features = extract_features(file_path)

                X.append(features)

                y.append(emotion)

    print("Dataset Loaded Successfully!")

    return np.array(X), np.array(y)