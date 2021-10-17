import pca
import svd
import lda
import k_means
import color_moment_similarity
import elbp_similarity
import hog_similarity

feature_models = ['color_moment', 'elbp', 'hog']
reduction_technique_map = [pca.pca, svd.svd, lda.lda, k_means.k_means]
valid_x = ['cc', 'con', 'detail', 'emboss', 'jitter', 'neg', 'noise1', 'noise2', 'original',
           'poster', 'rot', 'smooth', 'stipple']
similarity_map = [color_moment_similarity.get_similarity, elbp_similarity.get_similarity, hog_similarity.get_similarity]