from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import rbf_kernel
import sys

from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score
from sklearn.svm import SVC


from preprocess import preprocess
from kernel import get_kernel_matrix_trainable, kernel_alignment

def svm_classification(K_train, y_train, K_test, y_test):
    clf = SVC(kernel="precomputed")
    clf.fit(K_train, y_train)
    y_pred = clf.predict(K_test)
    acc = accuracy_score(y_test, y_pred)

    y_scores = clf.decision_function(K_test)
    auc = roc_auc_score(y_test, y_scores)

    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    return acc, auc, f1

def loop_train(steps,X_train_t,X_train_p,y_train,X_test_t,X_test_p,y_test, backend="statevector"):
    
    # # preprocess data
    # X_train_am = preprocess(X_train_raw, sr=16000)
    # X_test_am = preprocess(X_test_raw, sr=16000)

    # X_train_ph = preprocess(X_train_raw, sr=16000, phase=True)
    # X_test_ph = preprocess(X_test_raw, sr=16000, phase=True)

    best_acc = 0.0

    def objective(params):

        print("New iteration:")
        nonlocal best_acc

        # w_a = params[0:4]
        # b_a = params[4:8]

        # w_p = params[8:12]
        # b_p = params[12:16]

        w = params[0:4]
        b = params[4:8] 

        
        # build kernels
        K_train = get_kernel_matrix_trainable(X_train_t, feature_map="ry_rz_1_trainable2",w=w,b=b, backend=backend, X1_p=X_train_p)
        K_test = get_kernel_matrix_trainable(X_test_t, feature_map="ry_rz_1_trainable2",w=w,b=b, backend=backend, X2=X_train_t, X1_p=X_test_p, X2_p=X_train_p)


        # classification
        acc_qtse3, auc, f1 = svm_classification(K_train,y_train,K_test, y_test)

        if acc_qtse3 > best_acc:
            best_acc = acc_qtse3
        
        print("Accuracy:", acc_qtse3)
        print("Best accuracy: ", best_acc)

        print("Auc: ", auc)
        print("f1: ", f1)

        return -acc_qtse3
        
    init_params = np.random.randn(8)

    #optimization
    result = minimize(objective,init_params,method="COBYLA",options={"maxiter":steps})

    best_params = result.x

    return best_params, best_acc


def main():
    # Cargar datos
    X = np.load("data_gtzan/preparado/X.npy")
    y = np.load("data_gtzan/preparado/y.npy")
    folds = np.load("data_gtzan/preparado/folds.npy", allow_pickle=True)

    fold_i = int(sys.argv[1])
    fold = folds[fold_i]

    print("fold: ",fold_i)

    X_train, y_train = X[fold["train"]], y[fold["train"]]
    X_val,   y_val   = X[fold["val"]],   y[fold["val"]]
    X_test,  y_test  = X[fold["test"]],  y[fold["test"]]

    # print("X_train shape: ", X_train.shape)
    # print("X test shape: ", X_train.shape)

    X_train_t = preprocess(X_train,sr=22050, binary=False)
    X_test_t = preprocess(X_test,sr=22050, binary=False)

    X_val_t = preprocess(X_val,sr=22050, binary=False)
    X_val_p = preprocess(X_val,sr=22050,phase=True)

    X_train_p = preprocess(X_train,sr=22050,phase=True, binary=False)
    X_test_p = preprocess(X_test,sr=22050,phase=True, binary=False)

    # print("X train_t shape: ", X_train_t.shape)
    # print("X train_p shape: ", X_train_p.shape)
    # print("X_test_t shape: ", X_test_t.shape)
    # print("X_test_p shape: ", X_test_p.shape)


    # Train
    best_params, best_acc = loop_train(steps=10,X_train_t=X_train_t,X_train_p=X_train_p,y_train=y_train,X_test_t=X_val_t, X_test_p=X_val_p,y_test=y_val)

    print("Best params: ", best_params)
    print("Best acc: ", best_acc)


    w = best_params[0:4]
    b = best_params[4:8]

    # Calculate kernesl
    K_train = get_kernel_matrix_trainable(X_train_t, feature_map="ry_rz_1_trainable2",w=w,b=b, backend="statevector", X1_p=X_train_p)
    K_test = get_kernel_matrix_trainable(X_test_t, feature_map="ry_rz_1_trainable2",w=w,b=b, backend="statevector", X2=X_train_t, X1_p=X_test_p, X2_p=X_train_p)

    final_acc, final_auc, final_f1 = svm_classification(K_train,y_train,K_test,y_test)

    print("FINAL ACC: ", final_acc)
    print("FINAL AUC: ", final_auc)
    print("FINAL F1: ", final_f1)


if __name__ == "__main__":
    main()
