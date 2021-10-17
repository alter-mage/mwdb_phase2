import pickle
import utilities
import json
import codecs

def start_task4():
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

    Ssim = simp[utilities.feature_models[model]]['Ssim']
    json.dump(Ssim.tolist(), codecs.open('Ssim.json', 'w', encoding='utf-8'))

    reduction_obj_right = utilities.reduction_technique_map[reduction_technique](k, Ssim)
    left_matrix, core_matrix, right_matrix = reduction_obj_right.transform()

    latent_out_file_path = '%s_%s_%s_%s' % ('4', utilities.feature_models[model], str(k), str(reduction_technique))
    with open(latent_out_file_path+'.pickle', 'wb') as handle:
        pickle.dump({
            'left_matrix': left_matrix,
            'core_matrix': core_matrix,
            'right_matrix': right_matrix
        }, handle, protocol=pickle.HIGHEST_PROTOCOL)

    json.dump(left_matrix.tolist(), codecs.open(latent_out_file_path + '.json', 'w', encoding='utf-8'))


if __name__ == '__main__':
    start_task4()