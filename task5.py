import pickle
import aggregation
import utilities
import os
import cv2
import matplotlib.pyplot as plt
import numpy as np

def start_task5():
    with open('metadata.pickle', 'rb') as handle:
        metadata = pickle.load(handle)
    
    with open('simp.pickle', 'rb') as handle:
        simp = pickle.load(handle)

    query = 'query.png'
    while query not in os.listdir(os.getcwd()):
        query = input('Query image filename (\'query.png\' does not exist): ')

    latent_semantics_file = ''
    while latent_semantics_file+'.pickle' not in os.listdir(os.getcwd()):
        latent_semantics_file = input('Latent semantics filename: ')

    n_upper_limit = len(metadata)
    n = -1
    while not (1 <= n <= n_upper_limit):
        n = int(input('Value for n (how many similar images): '))
    
    with open(latent_semantics_file+'.pickle', 'rb') as handle:
        latent_semantics = pickle.load(handle)
    
    tokens = latent_semantics_file.split('_')
    task = int(tokens[0])
    if tokens[1] == 'color':
        tokens[1] = 'color_moment'
    feature_model = int(utilities.feature_models.index(tokens[1]))
    reduction_technique = int(utilities.reduction_technique_map_str.index(tokens[-1]))

    #query_image = cv2.imread(os.path.join('query', query), cv2.IMREAD_GRAYSCALE)
    query_image = cv2.imread(query, cv2.IMREAD_GRAYSCALE)
    query_features = utilities.feature_extraction[feature_model](query_image)

    right_matrix = latent_semantics['right_matrix']
    all_data_features = aggregation.all_data(metadata, query_features, feature_model)
    query_transform = all_data_features[-1]
    all_data_transform = all_data_features[:-1]
    
    if task > 2:
        sim = simp[utilities.feature_models[feature_model]]['T']
        if task == 4:
            sim = simp[utilities.feature_models[feature_model]]['S']
        query_transform = np.dot(np.array(query_transform), np.array(sim).T)
        all_data_transform = np.dot(np.array(all_data_transform), np.array(sim).T)
        
    all_data_k = utilities.query_transformation[reduction_technique](all_data_transform, right_matrix)
    query_k = utilities.query_transformation[reduction_technique](query_transform, right_matrix)
    similarity_scores = utilities.similarity_map[feature_model](query_k, all_data_k)
    similarities = {}
    for key, val in zip(sorted(metadata), similarity_scores):
        similarities[key] = val
    similarities = {k: v for k, v in sorted(similarities.items(), reverse=True, key=lambda item: item[1])}
    images = list(similarities.keys())[:n]
    scores = list(similarities.values())[:n]

    fig, axes = plt.subplots(n+1, 1)
    topk = []
    for i, axis in enumerate(axes):
        if i == 0:
            img = query_image
            axis.text(74, 25, query, size=9)
            axis.text(74, 45, 'Original image', size=9)
        else:
            img = metadata[images[i-1]]['image']
            axis.text(74, 25, images[i-1], size=9)
            axis.text(74, 45, str(scores[i-1]), size=9)
            topk.append(img)
        axis.imshow(img, cmap='gray')
        axis.tick_params(left = False, right = False, labelleft = False, labelbottom = False, bottom = False)
    fig.suptitle(str(n)+' Most Similar Images - '+utilities.similarity_measures[feature_model], size=10)
    plt.show()

if __name__ == '__main__':
    start_task5()