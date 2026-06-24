
from exp import pipeline_cluster
from datuak_kargatu import prep_folds, load_audios
from preprozesaketa import preprocess

import numpy as np
import sys

def main():

    dataset_path="./data/prep/classical_jazz"
    classic = False
    preproc_mfcc = True
    preproc_phase = True
    phase_type = "gd" # "phase" edo "gd"
    encoding = "qtse_p3"

    train = True
    init_w = np.ones(4) + np.random.uniform(-0.3, 0.3, 4)
    init_b = np.random.uniform(-0.1, 0.1, 4)

    fold_i = int(sys.argv[1])

    pipeline_cluster(fold_i,dataset_path=dataset_path, classic=classic, preproc_mfcc=preproc_mfcc, train=train, preproc_phase=preproc_phase, phase_type=phase_type, encoding=encoding, init_w=init_w, init_b=init_b)

    return 0

if __name__ == "__main__":
    main()