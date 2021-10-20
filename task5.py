import pickle
import aggregation
import utilities
import os
import cv2
import matplotlib.pyplot as plt

def start_task5():
    with open('metadata.pickle', 'rb') as handle:
        metadata = pickle.load(handle)
    
    with open('simp.pickle', 'rb') as handle:
        simp = pickle.load(handle)

    query = ''
    while query not in os.listdir('query'):
        query = input('query image filename: ')

    latent_semantics_file = ''
    while latent_semantics_file+'.pickle' not in os.listdir(os.getcwd()):
        latent_semantics_file = input('latent semantics filename: ')
    
    n_upper_limit = len(metadata)
    n = -1
    while not (1 <= n <= n_upper_limit):
        n = int(input('value for n: '))
    
    with open(latent_semantics_file+'.pickle', 'rb') as handle:
        latent_semantics = pickle.load(handle)
    
    tokens = latent_semantics_file.split('_')
    task = int(tokens[0])
    feature_model = int(tokens[1])
    reduction_technique = int(tokens[-1])
    query_image = cv2.imread(os.path.join('query', query), cv2.IMREAD_GRAYSCALE)
    query_features = utilities.feature_extraction[feature_model](query_image)
    
    if task <= 2:
        right_matrix = latent_semantics['right_matrix']
        all_data_matrix = latent_semantics['left_matrix']
        query_transform = query_features
    elif task == 3:
        query_transform = simp[utilities.feature_models[feature_model]]['T']
        right_matrix = latent_semantics['right_matrix']
        all_data_matrix = latent_semantics['left_matrix']
    query_k = utilities.query_transformation[reduction_technique](query_transform, right_matrix)
    similarity_scores = utilities.similarity_map[feature_model](query_k, all_data_matrix)
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
            axis.text(74, 35, 'Original image', size=9)
        else:
            img = metadata[images[i-1]]['image']
            axis.text(74, 35, str(scores[i-1]), size=9)
            topk.append(img)
        axis.imshow(img)#, cmap='gray')
        axis.tick_params(left = False, right = False, labelleft = False, labelbottom = False, bottom = False)
    fig.suptitle(str(n)+' Most Similar Images - '+utilities.similarity_measures[feature_model], size=10)
    plt.show()
    #all_data_features = aggregation.all_data(metadata, feature_model)
    #all_data_k = utilities.query_transformation[reduction_technique](all_data_features, right_matrix)

