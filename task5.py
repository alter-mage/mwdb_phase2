import pickle
import aggregation
import utilities
import os
import cv2

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
    right_matrix = latent_semantics['right_matix']

    if task <= 2:
        query_image = cv2.imread(os.path.join('query', query), cv2.IMREAD_GRAYSCALE)
        query_features = utilities.feature_extraction[feature_model](query_image)
        query_k = utilities.query_transformation[reduction_technique](query_features, right_matrix)

        all_data_features = aggregation.all_data(metadata, feature_model)
        all_data_k = utilities.query_transformation[reduction_technique](all_data_features, right_matrix)

