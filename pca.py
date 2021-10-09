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
        self.pca_ = PCA(n_components=k)
        self.pca_.fit(X)

    def transform(self, X):
        """
        Parameters:
            X: ndarray of shape (num_objects, num_features)
                Data matrix to be reduced

        Returns:
            Transforms and returns X in the latent semantic space and the latent semantics
        """
        return self.pca_.transform(X), self.pca_.components_


if __name__ == '__main__':
    dummy_data = [[1, 2, 3], [2, 4, 6]]
    pca_obj = pca(1, dummy_data)
    pca_obj.transform(X=dummy_data)
