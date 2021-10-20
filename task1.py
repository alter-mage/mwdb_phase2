import pickle
import aggregation
import utilities
import csv
import min_max_scaler

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
        x = input('value for X: ')

    # k measured starting from 1, not 0
    k_upper_limit = len(metadata[next(iter(metadata))][utilities.feature_models[model]])
    k = -1
    while not (1 <= k <= k_upper_limit - 1):
        k = int(input('value for k: '))

    reduction_technique = -1
    while not (0 <= reduction_technique <= 3):
        reduction_technique = int(input('reduction technique (0-3): '))

    data_matrix, subjects, data_matrix_index_map = aggregation.group_by_subject(metadata, x, model)

    reduction_obj_right = utilities.reduction_technique_map[reduction_technique](k, data_matrix)
    left_matrix, core_matrix, right_matrix = reduction_obj_right.transform()

    left_matrix_aggregated = aggregation.aggregate_by_mean(left_matrix, data_matrix_index_map)

    latent_out_file_path = '%s_%s_%s_%s_%s' % ('1', utilities.feature_models[model], str(x), str(k), utilities.reduction_technique_map_str[reduction_technique])
    with open(latent_out_file_path+'.pickle', 'wb') as handle:
        pickle.dump({
            'left_matrix': left_matrix,
            'core_matrix': core_matrix,
            'right_matrix': right_matrix
        }, handle, protocol=pickle.HIGHEST_PROTOCOL)

    fields = ['Y']
    for i in range(1, k+1):
        fields.append('k_'+str(i))
    with open(latent_out_file_path+'.csv', 'w', newline='') as handle:
        write = csv.writer(handle)
        write.writerow(fields)
        for i, row in enumerate(left_matrix_aggregated):
            r = row.tolist()
            r.insert(0, subjects[i])
            write.writerow(r)


if __name__ == '__main__':
    start_task1()
