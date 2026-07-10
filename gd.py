import numpy as np
import librosa
import numpy as np
from scipy.fftpack import dct
from scipy.fftpack import fft, ifft

def cepstral_smoothing(x_m, n_c=18, n_fft=512):
    X_m = fft(x_m, n_fft)
    abs_X_m = np.abs(X_m) ** 2
    abs_X_m = np.where(abs_X_m == 0, 1e-10, abs_X_m)
    log_X_m = np.log(abs_X_m)
    
    real_cepstrum = ifft(log_X_m, n_fft)
    
    w = np.zeros(n_fft)
    w[0] = 1.0
    for n in range(1, n_fft):
        idx = n if n <= n_fft // 2 else n_fft - n
        if idx < n_c:
            w[n] = 2.0
        elif idx == n_c:
            w[n] = 1.0
            
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

    if classic:
        x_len = len(x)
        if x_len < subframe_len:
            features_subframe = MODGDF(x, gamma=gamma, alpha=alpha, N_c=N_c, n_fft=subframe_len)
            subframe_gd = [features_subframe]
        else:
            subframe_gd = []
            for start in range(0, x_len - subframe_len + 1, hop_len):
                subframe = x[start : start + subframe_len]
                
                # MODGDF subframe bakoitzeko
                features_subframe = MODGDF(subframe, gamma=gamma, alpha=alpha, N_c=N_c, n_fft=subframe_len)
                subframe_gd.append(features_subframe)
        subframe_gd = np.array(subframe_gd)  #changed
        mean_coeffs = np.mean(subframe_gd, axis=0)
        return mean_coeffs
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


###JOSE

def cepstral_smoothing_v2(x_m, n_c=16, n_fft=512):
    X_m = fft(x_m, n_fft)
    power_X_m = np.abs(X_m) ** 2  #changed
    power_X_m = np.where(power_X_m == 0, 1e-10, power_X_m)  #changed
    log_power_X_m = np.log(power_X_m)  #changed

    real_cepstrum = ifft(log_power_X_m, n_fft)  #changed

    w = np.zeros(n_fft)
    w[0] = 1.0
    for n in range(1, n_fft):  #changed
        idx = n if n <= n_fft // 2 else n_fft - n
        if idx < n_c:
            w[n] = 2.0  #changed
        elif idx == n_c:
            w[n] = 1.0 #  changed

    liftered_cepstrum = real_cepstrum * w
    Y_m = np.exp(np.real(fft(liftered_cepstrum, n_fft)))
    return Y_m

def MODGDF_v2(subframe, gamma=0.9, alpha=0.3, N_c=16, n_fft=512):

    N = len(subframe)
    n = np.arange(N)
    y = n * subframe

    X = np.fft.fft(subframe, n_fft)
    Y = np.fft.fft(y, n_fft)

    XR, XI = X.real, X.imag
    YR, YI = Y.real, Y.imag

    goi = XR * YR + XI * YI
    S = cepstral_smoothing_v2(subframe, n_c=N_c, n_fft=n_fft)  #changed

    behe = (S ** (2 * gamma) + 1e-8)
    gd = goi / behe

    gd = np.sign(gd) * (np.abs(gd) ** alpha)

    coeffs = dct(gd, type=2, norm='ortho')

    return coeffs[1:N_c + 1]

def MODGDF_signal_v2(x, classic=False, gamma=0.9, alpha=0.3, N_c=16, subframe_len=512, hop_len=256, n_frames=16):

    if classic:
        x_len = len(x)  #changed ????

        if x_len < subframe_len:  #changed ?????
            coeffs = MODGDF_v2(x, gamma=gamma, alpha=alpha, N_c=N_c, n_fft=subframe_len)  #changed
            subframe_coeffs = [coeffs]  #changed
        else:
            subframe_coeffs = []  #changed
            for start in range(0, x_len - subframe_len + 1, hop_len):  #changed
                subframe = x[start: start + subframe_len]  #changed
                coeffs = MODGDF_v2(subframe, gamma=gamma, alpha=alpha, N_c=N_c, n_fft=subframe_len)  #changed
                subframe_coeffs.append(coeffs)  #changed

        subframe_coeffs = np.array(subframe_coeffs)  #changed
        mean_coeffs = np.mean(subframe_coeffs, axis=0)  #changed
        return mean_coeffs  #changed

    macro_blocks = np.array_split(x, n_frames)
    features = np.zeros(n_frames)

    for i, block in enumerate(macro_blocks):
        block_len = len(block)

        if block_len < subframe_len:
            coeffs = MODGDF_v2(block, gamma=gamma, alpha=alpha, N_c=N_c, n_fft=subframe_len)
            subframe_coeffs = [coeffs]  #changed
        else:
            subframe_coeffs = []  #changed
            for start in range(0, block_len - subframe_len + 1, hop_len):
                subframe = block[start: start + subframe_len]
                coeffs = MODGDF_v2(subframe, gamma=gamma, alpha=alpha, N_c=N_c, n_fft=subframe_len)
                subframe_coeffs.append(coeffs)  #changed

            if len(subframe_coeffs) == 0:
                subframe_coeffs = [np.full(N_c, 1e-8)]  #changed

        subframe_coeffs = np.array(subframe_coeffs)  #changed
        mean_coeffs = np.mean(subframe_coeffs, axis=0)  #changed
        features[i] = mean_coeffs[0]  #changed

    return features

def preproc_group_delay_v2(X, classic=False, gamma=0.9, alpha=0.3, N_c=16, subframe_len=512, hop_len=256, n_frames=16):  #changed

    phase_features = []

    for x in X:
        features = MODGDF_signal_v2(x, classic=classic, gamma=gamma, alpha=alpha, N_c=N_c, subframe_len=subframe_len, hop_len=hop_len, n_frames=n_frames)  #changed
        phase_features.append(features)

    return phase_features