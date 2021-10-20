import pca
import svd
import lda
import k_means
import color_moment_similarity
import elbp_similarity
import hog_similarity
import color_moment
import elbp
import hog

feature_models = ['color_moment', 'elbp', 'hog']
reduction_technique_map = [pca.pca, svd.svd, lda.lda, k_means.k_means]
valid_x = ['cc', 'con', 'detail', 'emboss', 'jitter', 'neg', 'noise1', 'noise2', 'original',
           'poster', 'rot', 'smooth', 'stipple']
similarity_measures = ['1/L1 distance', 'Cosine similarity', '1/Earth Mover\'s distance']
similarity_map = [color_moment_similarity.get_similarity, elbp_similarity.get_similarity, hog_similarity.get_similarity]
feature_extraction = [color_moment.get_cm_vector, elbp.get_elbp_vector, hog.get_hog_vector]
query_transformation = [pca.get_transformation, svd.get_transformation, lda.get_transformation, k_means.get_transformation]