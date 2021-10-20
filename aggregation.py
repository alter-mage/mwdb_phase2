import numpy as np
import utilities
import min_max_scaler


def group_by_type(metadata, y, feature_model):
    type_image_map = {}
    for key in metadata:
        key_tokens = key.split('.')[0].split('-')
        if key_tokens[2] == str(y):
            if key_tokens[1] not in type_image_map:
                type_image_map[key_tokens[1]] = []
            type_image_map[key_tokens[1]].append(
                metadata[key][utilities.feature_models[feature_model]]
            )
    data_matrix = []
    for type_ in sorted(type_image_map):
        data_matrix.append(np.mean(type_image_map[type_], axis=0))
    return sorted(type_image_map), np.array(min_max_scaler.transform(data_matrix))


def group_by_type_all(metadata, feature_model):
    type_image_map = {}
    for key in metadata:
        key_tokens = key.split('.')[0].split('-')
        if key_tokens[1] not in type_image_map:
            type_image_map[key_tokens[1]] = []
        type_image_map[key_tokens[1]].append(
            metadata[key][utilities.feature_models[feature_model]]
        )
    data_matrix = []
    for type_ in sorted(type_image_map):
        data_matrix.append(np.mean(type_image_map[type_], axis=0))
    data_matrix = np.array(min_max_scaler.transform(data_matrix), dtype=np.float32)
    t_t_similarity = min_max_scaler.transform(np.dot(data_matrix, data_matrix.transpose()))
    return sorted(type_image_map), data_matrix, t_t_similarity


def group_by_subject(metadata, x, feature_model):
    subject_image_map = {}
    for key in metadata:
        key_tokens = key.split('.')[0].split('-')
        if key_tokens[1] == x:
            if int(key_tokens[2]) not in subject_image_map:
                subject_image_map[int(key_tokens[2])] = []
            subject_image_map[int(key_tokens[2])].append(
                metadata[key][utilities.feature_models[feature_model]])
    data_matrix = []
    for subject in sorted(subject_image_map):
        data_matrix.append(np.mean(subject_image_map[subject], axis=0))
    return sorted(subject_image_map), np.array(min_max_scaler.transform(data_matrix))


def group_by_subject_all(metadata, feature_model):
    subject_image_map = {}
    for key in metadata:
        key_tokens = key.split('.')[0].split('-')
        if int(key_tokens[2]) not in subject_image_map:
            subject_image_map[int(key_tokens[2])] = []
        subject_image_map[int(key_tokens[2])].append(
            metadata[key][utilities.feature_models[feature_model]])
    data_matrix = []
    for subject in sorted(subject_image_map):
        data_matrix.append(np.mean(subject_image_map[subject], axis=0))
    data_matrix = np.array(min_max_scaler.transform(data_matrix))
    s_s_similarity = min_max_scaler.transform(np.dot(data_matrix, data_matrix.transpose()))
    return sorted(subject_image_map), data_matrix, s_s_similarity


def all_data(metadata, feature_model):
    data_matrix = []
    for key in sorted(metadata):
        data_matrix.append(metadata[key][utilities.feature_models[feature_model]])
    return sorted(metadata), data_matrix


class aggregation:

    def __init__(self):
        pass
