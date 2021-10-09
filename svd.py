from sklearn.decomposition import TruncatedSVD
#it's not realizing the attributes, not sure why mine is having issues, might just be my IDE

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
        self.svd_ = TruncatedSVD(
            #I put in the default values according to documentation except for k
            n_components=k,
            algorithm='randomized',
            n_iter=5,
            random_state=None,
            tol=0.0,
        ).fit(X)

    def get_latent_semantics(self):
        """
        Returns:
            ndarray of shape (n_components, n_features)
        """
        return self.svd_.components_


    def transform(self, X):
        """
        parameters:
            X: The matrix of object*features
            k: Number of latent features

        returns:
            Matrix of K latent features
        """

        # might want fit_transform, but should have been fitted already so ¯\_(ツ)_/¯
        return self.svd_.transform(X)

    # Might be helpful later
    # def compute_svd_reverse(X, k):
    #    trun_svd = TruncatedSVD(n_components=k)
    #    X_original = trun_svd.fit_transform(X)
    #    return X_original

    #components_ndarray of shape (n_components, n_features)
    #The right singular vectors of the input data.

    #explained_variance_ndarray of shape (n_components,)
    #The variance of the training samples transformed by a projection to each component.

    #explained_variance_ratio_ndarray of shape (n_components,)
    #Percentage of variance explained by each of the selected components.

    #singular_values_ndarray od shape (n_components,)
    #The singular values corresponding to each of the selected components. The singular values are equal to the 2-norms of the n_components variables in the lower-dimensional space.

    #n_features_in_int
    #Number of features seen during fit.
