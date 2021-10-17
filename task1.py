import pickle
import lda
import k_means
import pca
import svd
import aggregation
import numpy as np
import utilities

def start_task1():
    # moved up for upper limit validation of k value
    with open('metadata.pickle', 'rb') as handle:
        metadata = pickle.load(handle)

    # I thought we should include validation for all.
    model = -1
    while not (0 <= model <= 2):
        model = int(input('model number (0-2): '))

    x = 'invalid x'
    while x not in utilities.valid_x:
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

    data_matrix = aggregation.group_by_subject(metadata, x, reduction_technique)
    
    #flattenign the data matrix to be used
    # Not sure if this flattened thing will be useful if number of images in all the folder of each type are not same

    try:
        reduction_obj_right = utilities.reduction_technique_map[reduction_technique](k, data_matrix)
        left_matrix, core_matrix, right_matrix = reduction_obj_right.transform()

        latent_out_file_path = '%s_%s_%s_%s_%s' % ('2', utilities.feature_models[model], str(x), str(k), str(reduction_technique))
        with open(latent_out_file_path, 'wb') as handle:
            pickle.dump({
                'left_matrix': left_matrix,
                'core_matrix': core_matrix,
                'right_matrix': right_matrix
            }, handle, protocol=pickle.HIGHEST_PROTOCOL)

        # TODO: return
    except:
        print('Something is invalid, exception thrown')
        return


if __name__ == '__main__':
    start_task1()
