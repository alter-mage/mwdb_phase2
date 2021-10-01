from sklearn.decomposition import TruncatedSVD

def compute_svd(X_original, k):
    """
    parameters:
        X_original: The matrix of object*features
        k: Number of latent features

    returns:
        coefs:  Matrix of K latent features
    """
    trun_svd = TruncatedSVD(n_components=k)
    X = trun_svd.fit_transform(X_original)
    return X

#Might be helpful, not sure if correct?
def compute_svd_reverse(X, k):
    trun_svd = TruncatedSVD(n_components=k)
    X_original = trun_svd.fit_transform(X)
    return X_original