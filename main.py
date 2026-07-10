
from exp import pipeline_cluster
from datuak_kargatu import prep_folds, load_audios
from preprozesaketa import preprocess

import numpy as np
import sys

def main():

    

    # dataset_path="./data/prep/64_16_20/classical_rock"
    # classic = False
    # preproc_mfcc = True
    # preproc_phase = True
    # phase_type = "if" # "phase" edo "gd"
    # encoding = "qtse_p3" # "qtse", "qtse_p1" "qtse_p2" "qtse_p3" "ryrz_1"
    # train = True
    #init_w=None
    #init_b=None

    dataset_path = sys.argv[1]
    classic = sys.argv[2].lower() == "true"
    # preproc_mfcc = sys.argv[4].lower() == "true"
    # preproc_phase = sys.argv[5].lower() == "true"
    preproc_type = sys.argv[3] # "mfcc" or "phase". Just when it is classic or the quantum encoding is qtse (when we have just one type of value we need to specify how to preprocess)
    phase_type = sys.argv[4] # if it is classic or if the quantum encoding is qtse and the preprocessing is performed using mfcc -> is not necessary to specify
    encoding = sys.argv[5] # if it is classic -> its not necessary
    train = sys.argv[6].lower() == "true"

    SEED = int(sys.argv[7])
    np.random.seed(SEED)

    init_w = np.random.uniform(0.5, 1.5, 4)
    init_b = np.random.uniform(0, 2*np.pi, 4)


    print("Dataset: ",dataset_path)
    print("classic: ", classic)
    print("Encoding: ",encoding)
    print("Phase type: ", phase_type)
    print("SEED: ", SEED)

    print("----------------------")
    #adfspipeline_cluster(fold_i,dataset_path=dataset_path, classic=classic, preproc_mfcc=preproc_mfcc, train=train, preproc_phase=preproc_phase, phase_type=phase_type, encoding=encoding, init_w=init_w, init_b=init_b)
    
    acc_all = []
    auc_all = []
    f1_all = []

    for i in range(5):
        fold_i=i
        acc, auc, f1 = pipeline_cluster(fold_i,dataset_path=dataset_path, classic=classic, train=train, preproc_type=preproc_type, phase_type=phase_type, encoding=encoding, init_w=init_w, init_b=init_b)
        acc_all.append(acc)
        auc_all.append(auc)
        f1_all.append(f1)

    mean_acc = np.mean(acc_all)
    mean_auc = np.mean(auc_all)
    mean_f1 = np.mean(f1_all)

    std_acc = np.std(acc_all)
    std_auc = np.std(auc_all)
    std_f1 = np.std(f1_all)
    print("")
    print("------------------")
    print("")
    print("MEANs AND STDs: ")
    print("")
    print("MEAN ACC: ", mean_acc)
    print("MEAN AUC: ", mean_auc)
    print("MEAN F1: ", mean_f1)

    print("STD ACC: ", std_acc)
    print("STD AUC: ", std_auc)
    print("STD F1: ", std_f1)


    return 0

if __name__ == "__main__":
    main()