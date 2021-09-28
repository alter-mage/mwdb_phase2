from sklearn.cluster import KMeans
import numpy as np

def compute_k_means(X):
    kmeans = KMeans(n_clusters=8, init='k-means++', n_init=10, max_iter=300, algorithm='auto', random_state=0).fit(X)
    return kmeans


# Note for other team members: I have kept the default values for all parameters for K-Means right now but we can change it as and when needed. If the k-means parameters are different for each task then I will change the function to keep all the values as parameters in the function itself.