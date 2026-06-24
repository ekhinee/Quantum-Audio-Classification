from pathlib import Path
import librosa
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split
import soundfile as sf

def save_audios_gtzan(input_path, output_path, n,sr=22050, duration=30.0):

    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    m = 0

    for file in Path(input_path).glob("*.wav"):
        try:
            audio, _ = librosa.load(file, sr=sr, duration=duration)
            sf.write(output_path / f"audio_{m}.wav",audio,sr)
            m = m + 1
            if(m==n):
                break
        except Exception as e:
            #print(f"Error en {file}: {e}")
            continue

    return 0


def prep_folds(path1,path2, output_path):

    X1 = list(Path(path1).glob("*.wav"))
    X2 = list(Path(path2).glob("*.wav"))

    X = np.array(X1 + X2)

    y1 = np.zeros(len(X1), dtype=int)
    y2 = np.ones(len(X2), dtype=int)

    y = np.concatenate([y1, y2])

    idx_all = np.arange(len(y))
    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

    folds = []
    for idx_trainval, idx_test in skf.split(idx_all, y):
        idx_train, idx_val = train_test_split(
            idx_trainval, test_size=0.20, random_state=42, stratify=y[idx_trainval]
        )
        folds.append({
            "train": idx_train,
            "val":   idx_val,
            "test":  idx_test
        })

    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    np.save(output_path / "X.npy", X)
    np.save(output_path / "y.npy", y)
    np.save(output_path / "folds.npy", folds, allow_pickle=True)

def load_audios(paths, sr=22050, duration=30.0):

    X = []
    length = int(sr * duration)

    for path in paths:
        audio, _ = librosa.load(path, sr=sr, duration=duration)

        if len(audio) < length:
            audio = np.pad(audio, (0, length - len(audio)))

        X.append(audio)

    return np.array(X)

def main():

    return 0



if __name__ == "__main__":
    main()


