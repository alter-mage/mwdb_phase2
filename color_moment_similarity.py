import numpy as np


def get_similarity(x1, x2):
    return -1 * np.linalg.norm(
        np.subtract(x1, x2),
        ord=1
    )
