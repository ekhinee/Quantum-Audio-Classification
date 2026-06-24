from preprozesaketa import preprocess, normalize
from datuak_kargatu import load_audios
from kernel import get_kernel_matrix
from exp import pipeline_cluster

import numpy as np

def main():

    dataset_path="./data/prep/classical_jazz"
    classic = False
    preproc_mfcc = True
    preproc_phase = False
    phase_type = None
    encoding = "qtse"

    train = False
    init_w = None
    init_b = None

    pipeline_cluster(dataset_path=dataset_path, classic=classic, preproc_mfcc=preproc_mfcc, train=train, preproc_phase=preproc_phase, phase_type=phase_type, encoding=encoding, init_w=init_w, init_b=init_b)

    return 0

if __name__ == "__main__":
    main()