import pickle

import k_means

model_map = ['color_moment', 'elbp', 'hog']

def start_task1():
    model = int(input('model number: '))
    x = input('value for x: ')
    k = int(input('value for k: '))
    reduction_technique = int(input('reduction technique: '))

    with open('metadata.pickle', 'rb') as handle:
        metadata = pickle.load(handle)

    data_matrix, semantics_matrix = [], []
    for key in metadata:
        key_tokens = key.split('.')[0].split('-')
        if key_tokens[0] == x:
            data_matrix.append(metadata[key][model_map[model]])

    if reduction_technique == 3:
        semantics_matrix = k_means.compute_k_means(data_matrix)

    print('random')