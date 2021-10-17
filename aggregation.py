import numpy as np
import utilities


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
    return data_matrix


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
    data_matrix = np.array(data_matrix, dtype=np.float32)
    return data_matrix, np.dot(data_matrix, data_matrix.transpose())


def group_by_subject(metadata, x, feature_model):
    subject_image_map = {}
    for key in metadata:
        key_tokens = key.split('.')[0].split('-')
        if key_tokens[1] == x:
            if key_tokens[2] not in subject_image_map:
                subject_image_map[key_tokens[2]] = []
            subject_image_map[key_tokens[2]].append(
                metadata[key][utilities.feature_models[feature_model]])
    data_matrix = []
    for subject in sorted(subject_image_map):
        data_matrix.append(np.mean(subject_image_map[subject], axis=0))
    return data_matrix


def group_by_subject_all(metadata, feature_model):
    subject_image_map = {}
    for key in metadata:
        key_tokens = key.split('.')[0].split('-')
        if key_tokens[2] not in subject_image_map:
            subject_image_map[key_tokens[2]] = []
        subject_image_map[key_tokens[2]].append(
            metadata[key][utilities.feature_models[feature_model]])
    data_matrix = []
    for subject in sorted(subject_image_map):
        data_matrix.append(np.mean(subject_image_map[subject], axis=0))
    data_matrix = np.array(data_matrix)
    return data_matrix, np.dot(data_matrix, data_matrix.transpose())


class aggregation:

    def __init__(self):
        pass
