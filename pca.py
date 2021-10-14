from sklearn.decomposition import PCA
import numpy as np


class pca:
    """
    Represents PCA dimension technique
    ...
    Attributes:
        k: int
            Number of reduced features

        X: ndarray of shape (num_objects, num_features)
            Data matrix to be reduced

    Methods:
        transform(X)
            Transforms and returns X in the latent semantic space and the latent semantics
    """

    def __init__(self, k, X):
        """
        Parameters:
            k: int
                Number of reduced features

            X: ndarray of shape (num_objects, num_features)
                Data matrix to be reduced
        """

        self.x_ = np.array(X, dtype=np.float32)
        self.features_ = self.x_.shape[1]

        self.x_covariance_ = np.cov(self.x_.transpose())
        self.eigen_values_, self.eigen_vectors_ = np.linalg.eigh(self.x_covariance_)
        self.eigen_values_ = self.eigen_values_[::-1]
        self.eigen_vectors_ = self.eigen_vectors_.transpose()[::-1]

        self.u_, self.s_, self.u_transpose_ = self.eigen_vectors_[:][:k], \
                                              np.diag(self.eigen_values_[:k]), \
                                              self.eigen_vectors_.transpose()[:k]

        # self.pca_ = PCA(n_components=k)
        # self.pca_.fit(X)

    def transform(self, X):
        """
        Parameters:
            X: ndarray of shape (num_objects, num_features)
                Data matrix to be reduced

        Returns:
            Transforms and returns X in the latent semantic space and the latent semantics
        """
        return self.u_, self.u_transpose_


if __name__ == '__main__':
    dummy_data = [[1, 2, 3], [2, 4, 6]]
    pca_obj = pca(1, dummy_data)
    pca_obj.transform(X=dummy_data)
