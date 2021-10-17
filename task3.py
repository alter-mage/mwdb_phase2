import pickle


import k_means
import pca
import svd
import lda
import utilities


def start_task3():
    with open('metadata.pickle', 'rb') as handle:
        metadata = pickle.load(handle)

    model = -1
    while not (
            0 <= model <= 2
    ):
        model = int(input('model number (0-2): '))

    k_upper_limit = len(metadata[next(iter(metadata))][utilities.feature_models[model]])
    k = -1
    while not (
            1 <= k <= k_upper_limit - 1
    ):
        k = int(input('value for k: '))


if __name__ == '__main__':
    start_task3()
