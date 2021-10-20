from sklearn.decomposition import LatentDirichletAllocation as LDA
import numpy as np


class lda:
    """
    Represents LDA feature reduction class 
    ...
    Attributes:

        Data_matrix: ndarray of shape (num_objects, num_features)
            Data matrix to be reduced

        k: int
            Number of reduced features
        
    Methods:
        
        compute_lda(X):
            Returns a Matrix of K latent features * N objects
            
    """
    def __init__(self, k, data_matrix):
        """
        parameters:
            data_matrix: ndarray of shape (num_objects, num_features)
                Data matrix to be reduced

            k: int
                Number of reduced features
        """
        self.data_matrix_ = data_matrix
        self.k_ = k
        self.lda_ = LDA(n_components=self.k_).fit(self.data_matrix_)
        self.u_, self.s_, self.v_ = self.lda_.transform(self.data_matrix_), [], self.lda_.components_.transpose()
    
    def transform(self):
        """
        Parameters:
            self: self object
        
        returns:
            self.coefs:
                matrix of K latent features * N Objects and transformed data matrix
        """
        return self.u_, self.s_, self.v_


def get_transformation(data, right_matrix):
    return np.dot(np.array(data), np.array(right_matrix))


"""
def compute_lda(Data_matrix , k):
    
    parameters:
        Data_matrix: The matrix of object*features
        k: Number of latent features
    
    returns:
        coefs:  Matrix of K latent features * N objects         
        lda_object = LDA(n_components = k)
    
    
    lda_object.fit(Data_matrix)

    latent_features = lda_object.components_()
    
    return latent_features
"""
