import scipy.spatial


def get_similarity(x1, x2):
    return 1 - scipy.spatial.distance.cosine(x1.ravel(), x2.ravel())
