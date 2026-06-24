
from exp import pipeline_cluster
from datuak_kargatu import prep_folds

import numpy as np
import sys

def main():

    dataset_path="./data/prep/classical_jazz"
    classic = False
    preproc_mfcc = True
    preproc_phase = True
    phase_type = "phase" # "phase" edo "gd"
    encoding = "qtse_p3"

    train = False
    init_w = None
    init_b = None

    fold_i = int(sys.argv[1])

    pipeline_cluster(fold_i,dataset_path=dataset_path, classic=classic, preproc_mfcc=preproc_mfcc, train=train, preproc_phase=preproc_phase, phase_type=phase_type, encoding=encoding, init_w=init_w, init_b=init_b)

    return 0

if __name__ == "__main__":
    main()