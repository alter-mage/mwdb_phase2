from sklearn.decomposition import LatentDirichletAllocation as LDA


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
    def __init__(self, data_matrix, k):
        """
        parameters:
            data_matrix: ndarray of shape (num_objects, num_features)
                Data matrix to be reduced

            k: int
                Number of reduced features
        """
        self.data_matrix = data_matrix
        self.k = k
        self.lda_ = LDA(n_components=self.k).fit(self.data_matrix)
    
    def transform(self, data_matrix):
        """
        Parameters:
            self: self object
            data_matrix: original data matrix
        
        returns:
            self.coefs:
                matrix of K latent features * N Objects and transformed data matrix
        """
        return self.lda_.transform(data_matrix), self.lda_.components_



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