import numpy as np
import librosa
import numpy as np
from scipy.fftpack import dct
from scipy.fftpack import fft, ifft


def mfcc_phase(y,sr,n_mfcc,hop_length=512,n_mels=40):

    D = librosa.stft(
        y=y,
        n_fft=min(2048, len(y)),
        hop_length=hop_length
    )

    phase = np.angle(D)
    phase = np.unwrap(phase, axis=1)

    mel_basis = librosa.filters.mel(
        sr=sr,
        n_fft=min(2048, len(y)),
        n_mels=n_mels
    )

    # MAX en vez de suma ponderada
    # mel_phase = np.array([
    #     phase[np.argmax(mel_basis[m]), :]   # bin de frecuencia con mayor peso en ese filtro
    #     for m in range(n_mels)
    # ])
    dominant_bins = np.argmax(mel_basis, axis=1) 
    mel_phase = phase[dominant_bins, :]

    phase_ceps = dct(
        mel_phase,
        type=2,
        axis=0,
        norm='ortho'
    )

    return phase_ceps[:n_mfcc]

def preprocess_mfcc(X, sr, n_frames=16, n_mfcc=13, classic = False, type="mfcc"):

    X = np.asarray(X, dtype=float)

    if X.ndim == 1:
        X = X[None, :]

    X_out = []

    for signal in X:

        if(type=="phase"):
            mfcc = mfcc_phase(signal, sr, n_mfcc, hop_length=512, n_mels=40)
        else:
            mfcc = librosa.feature.mfcc(y=signal,sr=sr,n_mfcc=n_mfcc)

        if(classic):
            return mfcc
        
        T = mfcc.shape[1]

        # 2. dividir en 16 bloques temporales
        idx = np.linspace(0, T, n_frames + 1, dtype=int)

        values = []

        for i in range(n_frames):

            block = mfcc[:, idx[i]:idx[i+1]]  # (13, block_size)

            # 3. el blocke a 1 scalar
            if(type=="phase"): 
                scalar = np.angle(np.mean(np.exp(1j * block))) 
            else:
                scalar = np.mean(np.abs(block))  
            values.append(scalar)
        
        X_out.append(values)

    X_out = np.array(X_out)  # shape (N, 16)
    
    return X_out

def cepstral_smoothing(x_m, n_c=18, n_fft=512):
    X_m = fft(x_m, n_fft)
    abs_X_m = np.abs(X_m)
    abs_X_m = np.where(abs_X_m == 0, 1e-10, abs_X_m)
    log_X_m = np.log(abs_X_m)
    
    real_cepstrum = ifft(log_X_m, n_fft)
    
    w = np.zeros(n_fft)
    for n in range(n_fft):
        idx = n if n <= n_fft // 2 else n_fft - n
        if idx < n_c:
            w[n] = 1.0
        elif idx == n_c:
            w[n] = 0.5
            
    liftered_cepstrum = real_cepstrum * w
    Y_m = np.exp(np.real(fft(liftered_cepstrum, n_fft)))
    return Y_m

def MODGDF(subframe, gamma=0.9, alpha=0.3, N_c=16, n_fft=512):

    # Dado UN subframe, calcula sus características gd


    N = len(subframe)
    n = np.arange(N)
    y = n * subframe
    
    X = np.fft.fft(subframe, n_fft) 
    Y = np.fft.fft(y, n_fft)

    XR, XI = X.real, X.imag
    YR, YI = Y.real, Y.imag

    goi = XR * YR + XI * YI
    S = cepstral_smoothing(subframe, n_c=18, n_fft=n_fft)

    behe = (S**(2 * gamma) + 1e-8)
    gd = goi / behe

    # Escalar / Comprimir picos
    gd = np.sign(gd) * (np.abs(gd) ** alpha)

    # Pasar al dominio cepstral
    coeffs = dct(gd, type=2, norm='ortho')
    
    # Retornamos los coeficientes solicitados (N_c)
    return coeffs[1:N_c+1]

def MODGDF_signal(x, classic = False, gamma=0.9, alpha=0.3, N_c=16, subframe_len=512, hop_len=256):

    if(classic):
        x_len = len(x)
        if x_len < subframe_len:
            features_subframe = MODGDF(block, gamma=gamma, alpha=alpha, N_c=N_c, n_fft=subframe_len)
            subframe_gd = [np.mean(np.abs(features_subframe)**2)]
        else:
            subframe_gd = []
            for start in range(0, block_len - subframe_len + 1, hop_len):
                subframe = block[start : start + subframe_len]
                
                # MODGDF subframe bakoitzeko
                features_subframe = MODGDF(subframe, gamma=gamma, alpha=alpha, N_c=N_c, n_fft=subframe_len)
                mean = np.mean(np.abs(features_subframe)**2)
                subframe_gd.append(mean)
        return subframe_gd
    else:

        # 1. 16 bloketan banatu
        macro_blocks = np.array_split(x, 16)
        features = np.zeros(16)

        # 2. Bloke bakoitzeko balio bat lortu
        for i, block in enumerate(macro_blocks):

            block_len = len(block)
            if block_len < subframe_len:
    
                features_subframe = MODGDF(block, gamma=gamma, alpha=alpha, N_c=N_c, n_fft=subframe_len)
                subframe_means = [np.mean(np.abs(features_subframe)**2)]
            else:
                subframe_means = []
                for start in range(0, block_len - subframe_len + 1, hop_len):
                    subframe = block[start : start + subframe_len]
                    
                    # MODGDF subframe bakoitzeko
                    features_subframe = MODGDF(subframe, gamma=gamma, alpha=alpha, N_c=N_c, n_fft=subframe_len)
                    mean = np.mean(np.abs(features_subframe)**2)
                    subframe_means.append(mean)
                    
                if len(subframe_means) == 0:
                    subframe_means = [1e-8]

    
            features[i] = np.mean(subframe_means)

    
    return features


def preproc_gropup_delay(X,classic=False, gamma=0.9, l_w=128,alpha=0.3,N_c = 16):
   
    phase_features = []

    for x in X:
       features = MODGDF_signal(x, classic = classic)
       phase_features.append(features)
    
    # phase_features = np.array(phase_features)

    # f_min = phase_features.min()
    # f_max = phase_features.max()
    
    # denom = f_max - f_min if (f_max - f_min) > 0 else 1e-8

    # phase_features_scaled = 2 * np.pi * (phase_features - f_min) / denom - np.pi


    return phase_features



def preprocess(X, type, classic = False, sr = 22050):

    if(type=="mfcc" or type=="phase"):
        X_preprocessed = preprocess_mfcc(X, sr, classic=classic, type=type)
    elif(type=="gd"):
        X_preprocessed = preproc_gropup_delay(X, classic=classic)
    else:
        print("Aukeratutako type ez da zuzena")


    return X_preprocessed




import numpy as np

def normalize(X_train, X_val=None, X_test=None, binary=False, classic=False):

    xmin = np.min(X_train)
    xmax = np.max(X_train)

    if classic:
        new_min, new_max = 0.0, 1.0
    elif binary:
        new_min, new_max = 0, 15
    else:
        new_min, new_max = 0.0, 2*np.pi

    def norm(X):
        Xn = (X - xmin) / (xmax - xmin)
        Xn = Xn * (new_max - new_min) + new_min

        if binary:
            Xn = np.round(Xn).astype(int)

        return Xn

    X_train_n = norm(X_train)

    X_val_n = None
    if X_val is not None:
        X_val_n = norm(X_val)

    X_test_n = None
    if X_test is not None:
        X_test_n = norm(X_test)

    return X_train_n, X_val_n, X_test_n