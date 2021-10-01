import pickle

import k_means
import svd
import lda

feature_models = ['color_moment', 'elbp', 'hog']
reduction_technique_map = [None, svd.compute_svd, lda.compute_lda, k_means.k_means]

def start_task2():

    #moved up for upper limit validation of k value
    with open('metadata.pickle', 'rb') as handle:
        metadata = pickle.load(handle)

    #I thought we should include validation for all.
    model = -1
    while not (model >= 0 and model <= 2):
        model = int(input('model number (0-2): '))

    yinvalid = True
    while yinvalid:
        y = int(input('value for y: ')) #should be 1-40 for the Y value
        if 1 <= y <= 40:
            yinvalid = False

    #k measured starting from 1, not 0
    k_upper_limit = len(metadata[next(iter(metadata))][feature_models[model]])
    k = -1
    while not (1 <= k <= upperKLimit - 1): #STRIKE there should also be an upper limit validation, but that needs to be fetched from how much meta data,
        k = int(input('value for k: '))

    reduction_technique = -1
    while not (reduction_technique >= 0 and reduction_technique <= 3):
        reduction_technique = int(input('reduction technique (0-3): '))

    data_matrix, semantics_matrix = [], []
    for key in metadata:
        key_tokens = key.split('.')[0].split('-')
        if int(key_tokens[2]) == y: #I changed from 0 to 1 because Y would be 1
            data_matrix.append(metadata[key][feature_models[model]])

    try:
        reduction_obj = reduction_technique_map[reduction_technique](k, data_matrix)
        semantics_matrix = reduction_obj.get_latent_semantics()

        #TODO: return
    except:
        print('Something is invalid, exception thrown')
        return



# Seems like the image database is a little screwed up in terms of naming convention