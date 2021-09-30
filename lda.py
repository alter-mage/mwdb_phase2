from sklearn.decomposition import LatentDirichletAllocation as LDA

def compute_lda(Data_matrix , k):
    """
    parameters:
        Data_matrix: The matrix of object*features
        k: Number of latent features
    
    returns:
        coefs:  Matrix of K latent features * N objects         
    """
    lda_object = LDA(n_components = k)
    lda_object.fit(Data_matrix)

    latent_features = lda_object.components_()
    
    return latent_features

