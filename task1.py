import pickle

import k_means
import svd

model_map = ['color_moment', 'elbp', 'hog']
reduction_technique_map = [None, svd.compute_svd, None, k_means.k_means]
valid_x = ['cc', 'con', 'detail', 'emboss', 'jitter', 'neg', 'noise1', 'noise2', 'original',
           'poster', 'rot', 'smooth', 'stipple']

def start_task1():
    # moved up for upper limit validation of k value
    with open('metadata.pickle', 'rb') as handle:
        metadata = pickle.load(handle)

    # I thought we should include validation for all.
    model = -1
    while not (model >= 0 and model <= 2):
        model = int(input('model number (0-2): '))

    x = 'invalid x'
    while x not in valid_x:
        x = input('value for x: ')

    # k measured starting from 1, not 0
    upperKLimit = len(metadata)
    k = -1
    while not (
            k >= 1 and k <= upperKLimit - 1):  # STRIKE there should also be an upper limit validation, but that needs to be fetched from how much meta data,
        k = int(input('value for k: '))

    reduction_technique = -1
    while not (reduction_technique >= 0 and reduction_technique <= 3):
        reduction_technique = int(input('reduction technique (0-3): '))

    data_matrix, semantics_matrix = [], []
    for key in metadata:
        key_tokens = key.split('.')[0].split('-')
        if key_tokens[1] == x:
            data_matrix.append(metadata[key][model_map[model]])

    try:
        reduction_obj = reduction_technique_map[reduction_technique](k, data_matrix)
        semantics_matrix = reduction_obj.get_latent_semantics()

        # TODO: return
    except:
        print('Something is invalid, exception thrown')
        return

if __name__ == '__main__':
    start_task1()


