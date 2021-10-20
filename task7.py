import pickle
import aggregation
import utilities
import os
import cv2
import matplotlib.pyplot as plt
import numpy as np

def start_task7():
    with open('metadata.pickle', 'rb') as handle:
        metadata = pickle.load(handle)
    
    with open('simp.pickle', 'rb') as handle:
        simp = pickle.load(handle)

    query = 'query.png'
    while query not in os.listdir(os.getcwd()):
        query = input('Query image filename (\'query.png\' does not exist): ')

    latent_semantics_file = ''
    while latent_semantics_file+'.pickle' not in os.listdir(os.getcwd()):
        latent_semantics_file = input('latent semantics filename: ')
    
    with open(latent_semantics_file+'.pickle', 'rb') as handle:
        latent_semantics = pickle.load(handle)
    
    tokens = latent_semantics_file.split('_') #makes 'color_moment' split, so just fixed it
    task = int(tokens[0])
    if tokens[1] == 'color':
        tokens[1] = 'color_moment'
    feature_model = int(utilities.feature_models.index(tokens[1]))
    reduction_technique = int(utilities.reduction_technique_map_str.index(tokens[-1]))

    # query_image = cv2.imread(os.path.join('query', query), cv2.IMREAD_GRAYSCALE)
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
    images = list(similarities.keys())[:41]

    subjects = {}
    for image in images:
        subject = image.split('.')[0].split('-')[2]
        if subject not in subjects:
            subjects[subject] = 1
        else:
            subjects[subject] += 1
    max_count = sorted(list(subjects.values()), reverse=True)[0]
    labels = []
    for subject in subjects:
        if subjects[subject] == max_count:
            labels.append(subject)
    label = labels[0]
    if len(labels) > 1:
        for image in images:
            subject = image.split('.')[0].split('-')[2]
            if subject in labels:
                label = subject
                break

    plt.title('Subject ID (Y) = '+label)
    plt.imshow(query_image, cmap='gray')
    plt.axis('off')
    plt.show()


if __name__ == '__main__':
    start_task7()