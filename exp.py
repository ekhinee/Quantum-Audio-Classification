import sys
from scipy.optimize import minimize
import numpy as np

from kernel import get_kernel_matrix
from svm import svm_classification
from preprozesaketa import preprocess, normalize
from datuak_kargatu import load_audios


def loop_train(steps,feature_map,init_w, init_b,X_train_t,y_train,X_test_t,y_test,X_train_p = None,X_test_p=None, ):
    

    best_acc = 0.0

    def objective(params):

        print("New iteration:")
        nonlocal best_acc

        mid = len(params) // 2

        w = params[:mid]
        b = params[mid:]
        
        # build kernels
        K_train_qtse_timbre_phase_trainable = get_kernel_matrix(X_train_t, feature_map=feature_map,w=w,b=b, X1_p=X_train_p)
        K_test_qtse_timbre_phase_trainable = get_kernel_matrix(X_test_t, feature_map=feature_map,w=w,b=b, X2=X_train_t, X1_p=X_test_p, X2_p=X_train_p)


        # classification
        acc, auc, f1 = svm_classification(K_train_qtse_timbre_phase_trainable,y_train,K_test_qtse_timbre_phase_trainable, y_test)

        if acc > best_acc:
            best_acc = acc
        
        print("Accuracy:", acc)
        print("Best accuracy: ", best_acc)

        print("Auc: ", auc)
        print("f1: ", f1)

        return -acc

    init_params = np.concatenate([init_w, init_b])

    #optimization
    result = minimize(objective, init_params ,method="COBYLA",options={"maxiter":steps})

    best_params = result.x

    return best_params, best_acc



def pipeline_cluster(dataset_path,classic, preproc_mfcc, train, preproc_phase=False, phase_type=None, encoding=None, init_w =None, init_b=None):
    '''
    
    '''
    # DATUAK KARGATU

    X = np.load(dataset_path + "/X.npy", allow_pickle=True)
    y = np.load(dataset_path + "/y.npy")
    folds = np.load(dataset_path + "/folds.npy", allow_pickle=True)

    fold_i = 0
    fold = folds[fold_i]

    print("fold: ",fold_i)

    X_train_paths = X[fold["train"]]
    X_val_paths   = X[fold["val"]]
    X_test_paths  = X[fold["test"]]

    y_train = y[fold["train"]]
    y_val   = y[fold["val"]]
    y_test  = y[fold["test"]]

    # cargar audios
    X_train = load_audios(X_train_paths)
    X_val   = load_audios(X_val_paths)
    X_test  = load_audios(X_test_paths)

    # -------------------------------------

    # DATUAK PREPROZESATU

    # binary moduan tratatuko dut, hala ez?
    if(encoding=="qtse" or encoding=="qtse_p1" or encoding=="qtse_p2" or encoding=="qtse_p3"):
        binary_mfcc = True
    else:
        binary_mfcc= False
    
    if(encoding=="qtse" or encoding=="qtse_p1" or encoding=="qtse_p2"):
        binary_phase = True
    else:
        binary_phase = False
    
    X_train_t_norm, X_val_t_norm, X_test_t_norm = None, None, None
    X_train_p_norm, X_val_p_norm, X_test_p_norm = None, None, None

    if(preproc_mfcc):
        X_train_t = preprocess(X_train,type="mfcc",classic=classic)
        X_val_t = preprocess(X_val,type="mfcc",classic=classic)
        X_test_t = preprocess(X_test,type="mfcc",classic=classic)

        X_train_t_norm, X_val_t_norm, X_test_t_norm = normalize(X_train_t,X_val=X_val_t, X_test=X_test_t,classic=classic, binary=binary_mfcc)
        

    if(preproc_phase):
        X_train_p = preprocess(X_train,type=phase_type)
        X_val_p = preprocess(X_val,type=phase_type)
        X_test_p = preprocess(X_test,type=phase_type)

        X_train_p_norm, X_val_p_norm, X_test_p_norm = normalize(X_train_p,X_val=X_val_p, X_test=X_test_p,classic=classic, binary=binary_phase)

    # ----------------------------------------

    # IF TRAIN
    w = None
    b = None

    if(train):
        best_params, best_acc = loop_train(steps=50,init_w=init_w, init_b=init_b,feature_map = "qtse_p3_trainable",X_train_t=X_train_t_norm,y_train=y_train,X_test_t=X_val_t_norm,y_test=y_val, X_train_p=X_train_p_norm, X_test_p=X_val_p_norm)

        print("Best params: ", best_params)
        print("Best acc: ", best_acc)

        mid = len(best_params) // 2

        w = best_params[:mid]
        b = best_params[mid:]

    # ----------------------------------------

    # KERNELAK KALKULATU

    K_train = get_kernel_matrix(X_train_t_norm, feature_map=encoding,w=w,b=b, X1_p=X_train_p_norm, train=train)
    K_test = get_kernel_matrix(X_test_t_norm, feature_map=encoding,w=w,b=b, X2=X_train_t_norm, X1_p=X_test_p_norm, X2_p=X_train_p_norm, train=train)


    # ----------------------------------------

    # SVM ENTRENAU ETA SCOREAK LORTU

    final_acc, final_auc, final_f1 = svm_classification(K_train,y_train,K_test,y_test)

    print("FINAL ACC: ", final_acc)
    print("FINAL AUC: ", final_auc)
    print("FINAL F1: ", final_f1)

    return 0

     


