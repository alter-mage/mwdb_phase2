import pickle
import utilities
import csv


def start_task3():
    # Reading metadata.pickle file, image representations
    with open('metadata.pickle', 'rb') as handle:
        metadata = pickle.load(handle)

    # Reading simp.pickle file, transformation matrices
    with open('simp.pickle', 'rb') as handle:
        simp = pickle.load(handle)

    # User Input: Model Number
    model = -1
    print()
    print("Data Models:")
    for index, value in enumerate(utilities.feature_models):
        print(index, value)
    while not (0 <= model <= 2):
        model = int(input('Enter Model Number (0-2): '))

    # Calculating upper limit of k
    # k measured starting from 1, not 0
    k_upper_limit = len(metadata[next(iter(metadata))][utilities.feature_models[model]])

    # User Input: k
    print()
    k = -1
    while not (1 <= k <= k_upper_limit - 1):
        k = int(input('Enter value of k (latent semantics): '))

    # User Input: reduction_technique
    reduction_technique = -1
    print()
    print("Reduction Techniques:")
    for index, value in enumerate(utilities.reduction_technique_map_str):
        print(index, value)
    while not (0 <= reduction_technique <= 3):
        reduction_technique = int(input('Enter reduction technique number (0-3): '))

    Tsim = simp[utilities.feature_models[model]]['Tsim']
    types = simp[utilities.feature_models[model]]['types']
    with open('Tsim.csv', 'w', newline='') as handle:
        write = csv.writer(handle)
        write.writerow(types)
        write.writerows(Tsim)

    reduction_obj_right = utilities.reduction_technique_map[reduction_technique](k, Tsim)
    left_matrix, core_matrix, right_matrix = reduction_obj_right.transform()

    latent_out_file_path = '%s_%s_%s_%s' % ('3', utilities.feature_models[model], str(k),
                                            utilities.reduction_technique_map_str[reduction_technique])
    with open(latent_out_file_path + '.pickle', 'wb') as handle:
        pickle.dump({
            'left_matrix': left_matrix,
            'core_matrix': core_matrix,
            'right_matrix': right_matrix,
            'simp': Tsim
        }, handle, protocol=pickle.HIGHEST_PROTOCOL)

    fields = ['X']
    for i in range(1, k + 1):
        fields.append('k_' + str(i))
    with open(latent_out_file_path + '.csv', 'w', newline='') as handle:
        write = csv.writer(handle)
        write.writerow(fields)
        for i, row in enumerate(left_matrix):
            r = row.tolist()
            r.insert(0, types[i])
            write.writerow(r)

    print()
    print("Output File Names: " + latent_out_file_path + ".csv/.pickle")


if __name__ == '__main__':
    start_task3()
