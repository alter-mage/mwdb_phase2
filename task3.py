import pickle
import utilities
import csv

def start_task3():
    with open('metadata.pickle', 'rb') as handle:
        metadata = pickle.load(handle)
    
    with open('simp.pickle', 'rb') as handle:
        simp = pickle.load(handle)

    model = -1
    while not (0 <= model <= 2):
        model = int(input('model number (0-2): '))

    k_upper_limit = len(metadata[next(iter(metadata))][utilities.feature_models[model]])
    k = -1
    while not (1 <= k <= k_upper_limit - 1):
        k = int(input('value for k: '))
    
    reduction_technique = -1
    while not (0 <= reduction_technique <= 3):
        reduction_technique = int(input('reduction technique (0-3): '))

    Tsim = simp[utilities.feature_models[model]]['Tsim']
    types = simp[utilities.feature_models[model]]['types']
    with open('Tsim.csv', 'w', newline='') as handle:
        write = csv.writer(handle)
        write.writerow(types)
        write.writerows(Tsim)

    reduction_obj_right = utilities.reduction_technique_map[reduction_technique](k, Tsim)
    left_matrix, core_matrix, right_matrix = reduction_obj_right.transform()

    latent_out_file_path = '%s_%s_%s_%s' % ('3', utilities.feature_models[model], str(k), str(reduction_technique))
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
    start_task3()