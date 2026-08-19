import numpy as np
import librosa


def extract_features(file_path):
    """
    Extract MFCC features from an audio file.

    Parameters:
        file_path (str): Path to the WAV file.

    Returns:
        numpy.ndarray: MFCC feature matrix of shape (40, 174)
    """

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