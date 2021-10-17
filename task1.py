import pickle
import lda
import k_means
import pca
import svd
import numpy as np

feature_models = ['color_moment', 'elbp', 'hog']
reduction_technique_map = [pca.pca, svd.svd, lda.lda, k_means.k_means]
valid_x = ['cc', 'con', 'detail', 'emboss', 'jitter', 'neg', 'noise1', 'noise2', 'original',
           'poster', 'rot', 'smooth', 'stipple']


def start_task1():
    # moved up for upper limit validation of k value
    with open('metadata.pickle', 'rb') as handle:
        metadata = pickle.load(handle)

    # I thought we should include validation for all.
    model = -1
    while not (0 <= model <= 2):
        model = int(input('model number (0-2): '))

    x = 'invalid x'
    while x not in valid_x:
        x = input('value for x: ')

    # k measured starting from 1, not 0
    k_upper_limit = len(metadata)
    k = -1
    while not (
            1 <= k <= k_upper_limit - 1):  # STRIKE there should also be an upper limit validation, but that needs to be fetched from how much meta data,
        k = int(input('value for k: '))

    reduction_technique = -1
    while not (0 <= reduction_technique <= 3):
        reduction_technique = int(input('reduction technique (0-3): '))

    data_matrix, semantics_matrix = [], []
    for key in metadata:
        key_tokens = key.split('.')[0].split('-')
        if key_tokens[1] == x:
            data_matrix.append(metadata[key][feature_models[model]])
    
    #flattenign the data matrix to be used
    # Not sure if this flattened thing will be useful if number of images in all the folder of each type are not same
 
    flattened_dict = dict()
    for key in metadata:
        key_tokens = key.split('.')[0].split('-')
        y_value = key_tokens[2]
        if y_value not in flattened_dict:
            flattened_dict[y_value]= []
        flattened_dict[y_value].append( metadata[key][feature_models[model]])

    data_matrix_flattened = []
    for key in flattened_dict:
        data_matrix_flattened.append(flattened_dict[key])

    try:
        reduction_obj = reduction_technique_map[reduction_technique](k, data_matrix)
        semantics_matrix = reduction_obj.transform(data_matrix)
        
        ##assuming that pca return an np array
        obj1 = pca(data_matrix_flattened,k)
        left_flattened,right_flattened = obj1.transform(data_matrix_flattened)

        obj2 = pca(data_matrix,k)    
        left_matrix,right_matrix = obj2.transform(data_matrix)
        

        np.savetxt("task1_subject_weights.csv", left_flattened, delimiter=",")
        pickle.dump(right_matrix,open("task1_right.pkl","w"))

        # TODO: return
    except:
        print('Something is invalid, exception thrown')
        return


if __name__ == '__main__':
    start_task1()
