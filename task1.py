import pickle

model_map = ['color_moment', 'elbp', 'hog']

def start_task1():
    model = input('model number: ')
    x = input('value for x: ')
    k = input('value for k: ')
    reduction_technique = input('reduction technique: ')

    with open('metadata.pickle', 'rb') as handle:
        metadata = pickle.load(handle)

    data_matrix = []
    for key in metadata:
        key_tokens = key.split('.')[0].split('-')
        if key_tokens[1] == x:
            data_matrix.append(metadata[key][model_map[model]])

    # Something with the reduction technique happend here