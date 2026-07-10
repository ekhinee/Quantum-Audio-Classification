import numpy as np
import librosa
from scipy.signal import hilbert



def phase_differencing_IF(x, sr=22050):
    z = hilbert(x)
    phi = np.unwrap(np.angle(z))
    if_d = (phi[2:] - phi[:-2]) / 2 
    if_hz = if_d * sr / (2 * np.pi)
    return if_hz # array 1D, len N-2


def spectral_moment_IF(x, sr=22050, n_fft=2048, hop_length=512):
    D = librosa.stft(x, n_fft=n_fft, hop_length=hop_length, window='hann')
    S = np.abs(D) ** 2  # (n_fft/2+1, T)

    freqs_hz = np.linspace(0, sr / 2, S.shape[0])

    goi = np.sum(freqs_hz[:, None] * S, axis=0)
    behe = np.sum(S, axis=0)
    if_hz = goi / (behe + 1e-10)
    return if_hz # array 1D , len T


def if_signal(x, classic=False, n_frames=16):
    if_features = phase_differencing_IF(x)
    if(classic):   
        return if_features
    
    T = len(if_features)
    idx = np.linspace(0, T, n_frames + 1, dtype=int)

    values = []

    for i in range(n_frames):
        block = if_features[idx[i]:idx[i+1]]

        if len(block) == 0:
            values.append(0.0)
        else:
            values.append(np.mean(block))
    
    return values


        

def preproc_if(X, classic=False, n_frames=16):  #changed

    phase_features = []

    for x in X:
        features = if_signal(x, classic=classic, n_frames=n_frames)
        phase_features.append(features)

    return phase_features