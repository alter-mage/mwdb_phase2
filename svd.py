import numpy as np

def compute_svd(X):
    # Performing SVD with numpy
    U, D, VT = np.linalg.svd(X)
    svdListReturn = [U, D, VT]
    return svdListReturn

#Might be helpful?
def compute_svd_reverse(U, D, VT):
    X_remake = (U @ np.diag(D) @ VT)
    return X_remake