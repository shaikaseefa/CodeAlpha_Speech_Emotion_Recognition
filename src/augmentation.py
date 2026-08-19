import numpy as np

# Add random noise
def add_noise(data):
    noise = 0.005 * np.random.randn(len(data))
    return data + noise

# Change speed slightly
def stretch(data, rate=1.1):
    import librosa
    return librosa.effects.time_stretch(data, rate=rate)

# Change pitch slightly
def pitch_shift(data, sr, steps=2):
    import librosa
    return librosa.effects.pitch_shift(
        data,
        sr=sr,
        n_steps=steps
    )