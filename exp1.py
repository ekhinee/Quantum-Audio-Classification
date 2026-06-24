import numpy as np
from scipy.optimize import minimize

from preprozesaketa import preprocess
from kernel import get_kernel_matrix, kernel_alignment, save_kernel_heatmap, get_kernel_matrix_trainable
from svm import svm_classification




def loop_train(steps,feature_map,X_train_t,y_train,X_test_t,y_test,X_train_p = None,X_test_p=None, backend="statevector"):
    
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
        K_train_qtse_timbre_phase_trainable = get_kernel_matrix_trainable(X_train_t, feature_map=feature_map,w=w,b=b, backend=backend, X1_p=X_train_p)
        K_test_qtse_timbre_phase_trainable = get_kernel_matrix_trainable(X_test_t, feature_map=feature_map,w=w,b=b, backend=backend, X2=X_train_t, X1_p=X_test_p, X2_p=X_train_p)


        # classification
        acc_qtse3, auc, f1 = svm_classification(K_train_qtse_timbre_phase_trainable,y_train,K_test_qtse_timbre_phase_trainable, y_test)

        if acc_qtse3 > best_acc:
            best_acc = acc_qtse3
        
        print("Accuracy:", acc_qtse3)
        print("Best accuracy: ", best_acc)

        print("Auc: ", auc)
        print("f1: ", f1)

        return -acc_qtse3
        
    # 16 parámetros
    init_params = np.random.randn(8)

    #optimization
    result = minimize(objective,init_params,method="COBYLA",options={"maxiter":steps})

    best_params = result.x

    return best_params, best_acc




def main():
    # # Cargar datos
    X = np.load("data_gtzan/preparado/X.npy")
    y = np.load("data_gtzan/preparado/y.npy")
    folds = np.load("data_gtzan/preparado/folds.npy", allow_pickle=True)


    for fold_i, fold in enumerate(folds):
        if(fold_i>1):
            a = 0

        else:

            print("fold: ",fold_i)

            X_train, y_train = X[fold["train"]], y[fold["train"]]
            X_val,   y_val   = X[fold["val"]],   y[fold["val"]]
            X_test,  y_test  = X[fold["test"]],  y[fold["test"]]

            X_train_t = preprocess(X_train,sr=16000)
            X_test_t = preprocess(X_test,sr=16000)

            X_val_t = preprocess(X_val,sr=16000)
            X_val_p = preprocess(X_val,sr=16000,phase=True)

            X_train_p = preprocess(X_train,sr=16000,phase=True)
            X_test_p = preprocess(X_test,sr=16000,phase=True)


            # Train
            best_params, best_acc = loop_train(steps=10,X_train_t=X_train_t,X_train_p=X_train_p,y_train=y_train,X_test_t=X_val_t,X_test_p=X_val_p,y_test=y_val)

            print("Best params: ", best_params)
            print("Best acc: ", best_acc)


            w = best_params[0:4]
            b = best_params[4:8]

            # Calculate kernesl
            K_train = get_kernel_matrix_trainable(X_train_t, feature_map="qtse_timbre_phase_trainable",w=w,b=b, backend="statevector", X1_p=X_train_p)
            K_test = get_kernel_matrix_trainable(X_test_t, feature_map="qtse_timbre_phase_trainable",w=w,b=b, backend="statevector", X2=X_train_t, X1_p=X_test_p, X2_p=X_train_p)

            final_acc = svm_classification(K_train,y_train,K_test,y_test)

            print("FINAL ACC: ", final_acc)




if __name__ == "__main__":
    main()
