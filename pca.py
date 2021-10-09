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
        get_latent_semantics()
            Returns k reduced latent semantics of X

        transform(X)
            Transforms and returns X in the latent semantics space
    """

    def __init__(self, k, X):
        """
        Parameters:
            k: int
                Number of reduced features

            X: ndarray of shape (num_objects, num_features)
                Data matrix to be reduced
        """
        self.pca = PCA(n_components=k)
        self.pca.fit(X)

    def transform(self, X):
        """
        Parameters:
            X: ndarray of shape (num_objects, num_features)
                Data matrix to be reduced

        Returns:
            Transforms and returns X in the latent semantics space
        """
        return self.pca.transform(X), self.pca.components_


if __name__ == '__main__':
    pca_obj = pca(1, [1, 2, 3])
