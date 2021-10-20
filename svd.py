# coding=utf-8
import numpy as np


# it's not realizing the attributes, not sure why mine is having issues, might just be my IDE


class svd:
    """
        Represents SVD feature reduction class
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
                Transforms and returns X in the latent semantic space and latent semantics
        """

    def __init__(self, k, data_matrix):
        """
        Parameters:
            Datamatrix: ndarray of shape (num_objects, num_features)
                Data matrix to be reduced
            k: int
                Number of reduced features
        """

        self.matrix1 = data_matrix @ data_matrix.transpose()
        self.matrix2 = data_matrix.transpose() @ data_matrix

        self.eigen_values1, self.eigen_vectors1 = np.linalg.eig(self.matrix1)
        self.eigen_values2, self.eigen_vectors2 = np.linalg.eig(self.matrix2)

        self.idx1 = self.eigen_values1.argsort()[::-1]
        self.eigen_values1 = self.eigen_values1[self.idx1]
        self.eigen_vectors1 = self.eigen_vectors1[:, self.idx1]

        self.idx2 = self.eigen_values2.argsort()[::-1]
        self.eigen_values2 = self.eigen_values2[self.idx2]
        self.eigen_vectors2 = self.eigen_vectors2[:, self.idx2]

        self.left = self.eigen_vectors1[:, :k]
        self.S = np.diag(self.eigen_values1[:k] ** 0.5)
        self.right = self.eigen_vectors2[:, :k].transpose()

    def transform(self):
        """
        parameters:
            X: The matrix of object*features
            k: Number of latent features

        returns:
            Matrix of K latent features and latent semantics
        """

        # might want fit_transform, but should have been fitted already so ¯\_(ツ)_/¯
        return self.left, self.S, self.right


def get_transformation(data, right_matrix):
    return np.dot(np.array(data), np.array(right_matrix))

    # Might be helpful later
    # def compute_svd_reverse(X, k):
    #    trun_svd = TruncatedSVD(n_components=k)
    #    X_original = trun_svd.fit_transform(X)
    #    return X_original

    # components_ndarray of shape (n_components, n_features)
    # The right singular vectors of the input data.

    # explained_variance_ndarray of shape (n_components,)
    # The variance of the training samples transformed by a projection to each component.

    # explained_variance_ratio_ndarray of shape (n_components,)
    # Percentage of variance explained by each of the selected components.

    # singular_values_ndarray od shape (n_components,)
    # The singular values corresponding to each of the selected components. The singular values are equal to the 2-norms of the n_components variables in the lower-dimensional space.

    # n_features_in_int
    # Number of features seen during fit.
