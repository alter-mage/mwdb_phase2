import numpy as np
import utilities

class aggregation:

    def group_by_subject(self, metadata, x, reduction_technique):
        subject_image_map = {}
        for key in metadata:
            key_tokens = key.split('.')[0].split('-')
            if key_tokens[1] == x:
                if key_tokens[2] not in subject_image_map:
                    subject_image_map[key_tokens[2]] = []
                subject_image_map[key_tokens[2]].append(metadata[key][utilities.reduction_technique_map[reduction_technique]])
        data_matrix = []
        for subject in sorted(subject_image_map):
            data_matrix.append(np.mean(subject_image_map[subject], axis=0))
        return data_matrix