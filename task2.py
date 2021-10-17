import pickle

import utilities


def start_task2():
    # moved up for upper limit validation of k value
    with open('metadata.pickle', 'rb') as handle:
        metadata = pickle.load(handle)

    # I thought we should include validation for all.
    model = -1
    while not (0 <= model <= 2):
        model = int(input('model number (0-2): '))

    yinvalid, y = True, -1
    while yinvalid:
        y = int(input('value for y: '))  # should be 1-40 for the Y value
        if 1 <= y <= 40:
            yinvalid = False

    # k measured starting from 1, not 0
    k_upper_limit = len(metadata[next(iter(metadata))][utilities.feature_models[model]])
    k = -1
    while not (
            1 <= k <= k_upper_limit - 1):  # STRIKE there should also be an upper limit validation, but that needs to be fetched from how much meta data,
        k = int(input('value for k: '))

    reduction_technique = -1
    while not (0 <= reduction_technique <= 3):
        reduction_technique = int(input('reduction technique (0-3): '))

    data_matrix = []
    for key in metadata:
        key_tokens = key.split('.')[0].split('-')
        if int(key_tokens[2]) == y:  # I changed from 0 to 1 because Y would be 1
            data_matrix.append(metadata[key][utilities.feature_models[model]])

    try:
        reduction_obj_right = utilities.reduction_technique_map[reduction_technique](k, data_matrix)
        left_matrix, core_matrix, right_matrix = reduction_obj_right.transform()

        latent_out_file_path = '%s_%s_%s_%s_%s' % ('2', utilities.feature_models[model], str(y), str(k), str(reduction_technique))
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

# Seems like the image database is a little screwed up in terms of naming convention
