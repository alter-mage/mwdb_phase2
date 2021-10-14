import pickle

import color_moment_similarity
import elbp_similarity
import hog_similarity
import k_means
import pca
import svd
import lda

feature_models = ['color_moment', 'elbp', 'hog']
similarity_map = [color_moment_similarity.get_similarity, elbp_similarity.get_similarity, hog_similarity.get_similarity]
reduction_technique_map = [pca.pca, svd.svd, lda.lda, k_means.k_means]
valid_x = ['cc', 'con', 'detail', 'emboss', 'jitter', 'neg', 'noise1', 'noise2', 'original',
           'poster', 'rot', 'smooth', 'stipple']


def start_task4():
    with open('metadata.pickle', 'rb') as handle:
        metadata = pickle.load(handle)

    model = -1
    while not (
            0 <= model <= 2
    ):
        model = int(input('model number (0-2): '))

    k_upper_limit = len(metadata[next(iter(metadata))][feature_models[model]])
    k = -1
    while not (
            1 <= k <= k_upper_limit - 1
    ):
        k = int(input('value for k: '))


if __name__ == '__main__':
    start_task4()
