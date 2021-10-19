import pickle
import utilities
import aggregation
import csv
import min_max_scaler

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
        y = int(input('value for Y: '))  # should be 1-40 for the Y value
        if 1 <= y <= 40:
            yinvalid = False

    # k measured starting from 1, not 0
    k_upper_limit = len(metadata[next(iter(metadata))][utilities.feature_models[model]])
    k = -1
    while not (1 <= k <= k_upper_limit - 1):  # STRIKE there should also be an upper limit validation, but that needs to be fetched from how much meta data,
        k = int(input('value for k: '))

    reduction_technique = -1
    while not (0 <= reduction_technique <= 3):
        reduction_technique = int(input('reduction technique (0-3): '))

    types, data_matrix = aggregation.group_by_type(metadata, y, model)
    data_matrix = min_max_scaler.transform(data_matrix)

    reduction_obj_right = utilities.reduction_technique_map[reduction_technique](k, data_matrix)
    left_matrix, core_matrix, right_matrix = reduction_obj_right.transform()

    latent_out_file_path = '%s_%s_%s_%s_%s' % ('2', utilities.feature_models[model], str(y), str(k), str(reduction_technique))
    with open(latent_out_file_path+'.pickle', 'wb') as handle:
        pickle.dump({
            'left_matrix': left_matrix,
            'core_matrix': core_matrix,
            'right_matrix': right_matrix
        }, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    fields = ['X']
    for i in range(1, k+1):
        fields.append('k_'+str(i))
    with open(latent_out_file_path+'.csv', 'w', newline='') as handle:
        write = csv.writer(handle)
        write.writerow(fields)
        for i, row in enumerate(left_matrix):
            r = row.tolist()
            r.insert(0, types[i])
            write.writerow(r)


if __name__ == '__main__':
    start_task2()
