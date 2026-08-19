import os
import numpy as np
import tensorflow as tf
from src.extract_features import extract_features  # Imports your source extraction module

# ==========================================================================
# 1. Model Initialization Environment Setup
# ==========================================================================
# Pointing directly to your verified file configuration
MODEL_PATH = "saved_model/emotion_cnn_final.keras"

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"CRITICAL ERROR: Model file not located at: {MODEL_PATH}")

print(f"\n[AI SYSTEM] Loading Neural Network Weights from: {MODEL_PATH}...")
model = tf.keras.models.load_model(MODEL_PATH)
print("[AI SYSTEM] Convolutional Architecture Loaded Successfully!\n")

# Label mappings corresponding to the RAVDESS/TESS classification boundaries
EMOTIONS = ["neutral", "calm", "happy", "sad", "angry", "fearful", "disgust", "surprised"]

# ==========================================================================
# 2. Prediction Pipeline Routine
# ==========================================================================
def predict_emotion(audio_path):
    """
    Extracts features from an audio file and processes predictions through the CNN model.
    Returns: (str: detected_emotion, float: confidence_score)
    """
    # #1. Extract acoustic features (MFCCs)
    features = extract_features(audio_path)
    
    if features is None:
        return "Error Processing Audio", 0.0

    # Ensure feature array is a NumPy array
    features = np.array(features)
    
    # #2. Reshape to match the input tensor dimensionality expected by your model
    # If your CNN architecture expects a 3D tensor (Batch, TimeSteps, Channels):
    if len(features.shape) == 1:
        input_vector = np.expand_dims(features, axis=0)      # Add batch dimension
        input_vector = np.expand_dims(input_vector, axis=-1)  # Add channel dimension
    elif len(features.shape) == 2:
        input_vector = np.expand_dims(features, axis=0)      # Add missing batch dimension
    else:
        input_vector = features

    # #3. Generate raw probability distributions
    predictions = model.predict(input_vector)
    
    # #4. Extract highest index metric and convert to standard types
    predicted_class_index = np.argmax(predictions, axis=1)[0]
    raw_confidence = predictions[0][predicted_class_index]
    
    # Handle percentage conversion safely
    if raw_confidence <= 1.0:
        confidence_score = round(float(raw_confidence) * 100, 2)
    else:
        confidence_score = round(float(raw_confidence), 2)

    # #5. Map class index to emotion string labels safely
    if predicted_class_index < len(EMOTIONS):
        detected_emotion = EMOTIONS[predicted_class_index]
    else:
        detected_emotion = "Unknown Classification"

    # Debugging console logs to trace input patterns
    print("\n" + "="*50)
    print(f"[DEBUG AI LOG] Processing Target File: {os.path.basename(audio_path)}")
    print(f"[DEBUG AI LOG] Input Tensor Matrix Shape: {input_vector.shape}")
    print(f"[DEBUG AI LOG] Full Prediction Distribution Array:\n{predictions[0]}")
    print(f"[DEBUG AI LOG] Chosen Index: {predicted_class_index} -> Label: {detected_emotion.upper()}")
    print(f"[DEBUG AI LOG] Confidence Readout Value: {confidence_score}%")
    print("="*50 + "\n")

    return detected_emotion, confidence_score
