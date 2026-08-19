import os
import joblib
import numpy as np
import librosa
import matplotlib.pyplot as plt
from augmentation import add_noise, pitch_shift, stretch
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
import tensorflow as tf

from tensorflow.keras.layers import (
    Conv2D,
    MaxPooling2D,
    Dense,
    Dropout,
    BatchNormalization,
    GlobalAveragePooling2D
)

from tensorflow.keras.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    ReduceLROnPlateau
)

# =====================================================
# DATASET PATH
# =====================================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_PATH = os.path.join(BASE_DIR, "dataset")

print("Dataset Path:", DATASET_PATH)

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

count = 0

for actor in sorted(os.listdir(DATASET_PATH)):

    actor_path = os.path.join(DATASET_PATH, actor)

    if not os.path.isdir(actor_path):
        continue

    print("Reading:", actor)

    for file in sorted(os.listdir(actor_path)):

        if file.lower().endswith(".wav"):

            file_path = os.path.join(actor_path, file)

            try:
                emotion_code = file.split("-")[2]
                emotion = emotion_dict[emotion_code]

                features = extract_features(file_path)

                X.append(features)
                y.append(emotion)

                count += 1

            except Exception as e:
                print("Skipped:", file)
                print(e)

print("\nTotal Audio Loaded =", count)
# ============================
# Augmented Audio (Noise)
# ============================

from augmentation import add_noise

signal, sr = librosa.load(file_path, sr=None)

noise_signal = add_noise(signal)

noise_mfcc = librosa.feature.mfcc(
    y=noise_signal,
    sr=sr,
    n_mfcc=40
)

max_len = 174

if noise_mfcc.shape[1] < max_len:

    pad_width = max_len - noise_mfcc.shape[1]

    noise_mfcc = np.pad(
        noise_mfcc,
        ((0,0),(0,pad_width)),
        mode="constant"
    )

else:

    noise_mfcc = noise_mfcc[:, :max_len]

X.append(noise_mfcc)

y.append(emotion)
# ============================
# Augmented Audio (Pitch Shift)
# ============================

from augmentation import pitch_shift

pitch_signal = pitch_shift(signal, sr)

pitch_mfcc = librosa.feature.mfcc(
    y=pitch_signal,
    sr=sr,
    n_mfcc=40
)

if pitch_mfcc.shape[1] < max_len:

    pad_width = max_len - pitch_mfcc.shape[1]

    pitch_mfcc = np.pad(
        pitch_mfcc,
        ((0,0),(0,pad_width)),
        mode="constant"
    )

else:

    pitch_mfcc = pitch_mfcc[:, :max_len]

X.append(pitch_mfcc)
y.append(emotion)

print("Dataset Loaded Successfully!")
# ============================
# Augmented Audio (Time Stretch)
# ============================

from augmentation import stretch

stretch_signal = stretch(signal, rate=1.1)

stretch_mfcc = librosa.feature.mfcc(
    y=stretch_signal,
    sr=sr,
    n_mfcc=40
)

if stretch_mfcc.shape[1] < max_len:

    pad_width = max_len - stretch_mfcc.shape[1]

    stretch_mfcc = np.pad(
        stretch_mfcc,
        ((0,0),(0,pad_width)),
        mode="constant"
    )

else:

    stretch_mfcc = stretch_mfcc[:, :max_len]

X.append(stretch_mfcc)

y.append(emotion)

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

# Save Label Encoder
os.makedirs("models", exist_ok=True)

joblib.dump(
    encoder,
    "models/label_encoder.pkl"
)

print("Label Encoder Saved Successfully!") 

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

# =====================================================
# BUILD CNN MODEL
# =====================================================


from tensorflow.keras.layers import GlobalAveragePooling2D

model = Sequential()

# Block 1
model.add(Conv2D(
    32,
    (3,3),
    activation="relu",
    padding="same",
    input_shape=(40,174,1)
))
model.add(BatchNormalization())
model.add(MaxPooling2D((2,2)))
model.add(Dropout(0.25))

# Block 2
model.add(Conv2D(
    64,
    (3,3),
    activation="relu",
    padding="same"
))
model.add(BatchNormalization())
model.add(MaxPooling2D((2,2)))
model.add(Dropout(0.30))

# Block 3
model.add(Conv2D(
    128,
    (3,3),
    activation="relu",
    padding="same"
))
model.add(BatchNormalization())
model.add(MaxPooling2D((2,2)))
model.add(Dropout(0.30))

# Global Pooling
model.add(GlobalAveragePooling2D())

# Dense Layer
model.add(Dense(128, activation="relu"))
model.add(Dropout(0.50))

# Output Layer
model.add(Dense(8, activation="softmax"))

# =====================================================
# COMPILE MODEL
# =====================================================

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

print("\nCNN Model Summary\n")

model.summary()

# =====================================================
# CALLBACKS
# =====================================================

os.makedirs("saved_model", exist_ok=True)

checkpoint = ModelCheckpoint(
    "saved_model/best_emotion_model.keras",
    monitor="val_accuracy",
    save_best_only=True,
    verbose=1
)

early_stop = EarlyStopping(
    monitor="val_loss",
    patience=8,
    restore_best_weights=True
)

reduce_lr = ReduceLROnPlateau(
    monitor="val_loss",
    factor=0.5,
    patience=3,
    verbose=1
)

# =====================================================
# TRAIN MODEL
# =====================================================

history = model.fit(

    X_train,
    y_train,

    validation_data=(X_test, y_test),

    epochs=40,

    batch_size=32,

    callbacks=[
        checkpoint,
        early_stop,
        reduce_lr
    ],

    verbose=1
)

# =====================================================
# EVALUATE MODEL
# =====================================================

loss, accuracy = model.evaluate(
    X_test,
    y_test,
    verbose=0
)

print("\n===============================")
print("Test Accuracy :", round(accuracy*100,2),"%")
print("===============================\n")

# =====================================================
# CLASSIFICATION REPORT
# =====================================================

from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

y_pred = model.predict(X_test)

y_pred = np.argmax(y_pred, axis=1)

print(classification_report(
    y_test,
    y_pred,
    target_names=encoder.classes_
))

# =====================================================
# CONFUSION MATRIX
# =====================================================

cm = confusion_matrix(y_test, y_pred)

print("Confusion Matrix\n")

print(cm)

# =====================================================
# SAVE FINAL MODEL
# =====================================================

model.save("saved_model/emotion_cnn_final.keras")

print("\nModel Saved Successfully!")

# =====================================================
# ACCURACY GRAPH
# =====================================================

plt.figure(figsize=(8,5))

plt.plot(history.history["accuracy"], label="Training Accuracy")

plt.plot(history.history["val_accuracy"], label="Validation Accuracy")

plt.title("Training vs Validation Accuracy")

plt.xlabel("Epoch")

plt.ylabel("Accuracy")

plt.legend()

plt.grid(True)

plt.savefig("accuracy_graph.png")

plt.show()

# =====================================================
# LOSS GRAPH
# =====================================================

plt.figure(figsize=(8,5))

plt.plot(history.history["loss"], label="Training Loss")

plt.plot(history.history["val_loss"], label="Validation Loss")

plt.title("Training vs Validation Loss")

plt.xlabel("Epoch")

plt.ylabel("Loss")

plt.legend()

plt.grid(True)

plt.savefig("loss_graph.png")

plt.show()

print("\nProject Training Completed Successfully!")