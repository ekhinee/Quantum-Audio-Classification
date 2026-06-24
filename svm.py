from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score
from sklearn.svm import SVC



def svm_classic_kernel(X_train, y_train, X_test, y_test):
    
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

   
    K_train_cl = rbf_kernel(X_train)
    K_test_cl  = rbf_kernel(X_test, X_train)
    
    #ka = kernel_alignment(K_train_cl, y_train)
    
    clf = SVC(kernel="precomputed")
    clf.fit(K_train_cl, y_train)
    y_pred = clf.predict(K_test_cl)
    acc = accuracy_score(y_test, y_pred)

    y_scores = clf.decision_function(K_test_cl)
    auc = roc_auc_score(y_test, y_scores)

    f1 = f1_score(y_test, y_pred)
    
    return acc, auc, f1


def svm_classification(K_train, y_train, K_test, y_test):
    clf = SVC(kernel="precomputed")
    clf.fit(K_train, y_train)
    y_pred = clf.predict(K_test)
    acc = accuracy_score(y_test, y_pred)

    y_scores = clf.decision_function(K_test)
    auc = roc_auc_score(y_test, y_scores)

    f1 = f1_score(y_test, y_pred)

    print("Y pred: ", y_pred)

    return acc, auc, f1