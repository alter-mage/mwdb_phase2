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
    def __init__(self,Data_matrix,k):
        """
        parameters:
            Data_matrix: ndarray of shape (num_objects, num_features)
                Data matrix to be reduced

            k: int
                Number of reduced features
        """
        self.Data_matrix = Data_matrix
        self.k = k
        self.lda_object = LDA(n_components=self.k).fit(self.Data_matrix)
    
    def transform(self,Data_matrix):
        """
        Parameters:
            self: self object 
        
        returns:
            self.coefs:
                matrix of K latent features * N Objects 
        """
        self.coefs =self.lda_object.components_() 
        return self.coefs



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