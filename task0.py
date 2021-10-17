import sys
import os
import pickle
import cv2

import color_moment
import elbp
import hog
import aggregation
import utilities


def start_task0(metadata_file, simp_file):
    images_dir = os.path.join(os.getcwd(), 'sample_images')
    if not os.path.isdir(images_dir):
        print('download image dataset first')

    images = {}
    for filename in os.listdir(images_dir):
        img = cv2.imread(os.path.join(images_dir, filename), cv2.IMREAD_GRAYSCALE)
        if img is not None:
            images[filename] = {
                'image': img,
                'color_moment': color_moment.get_cm_vector(img),
                'elbp': elbp.get_elbp_vector(img),
                'hog': hog.get_hog_vector(img)
            }

    with open(metadata_file, 'wb') as handle:
        pickle.dump(images, handle, protocol=pickle.HIGHEST_PROTOCOL)

    similarity_map = {}
    for i in range(3):
        type_matrix, type_type_similarity = aggregation.group_by_type_all(images, i)
        subject_matrix, subject_subject_similarity = aggregation.group_by_subject_all(images, i)
        similarity_map[utilities.feature_models[i]] = {
            'T': type_matrix,
            'Tsim': type_type_similarity,
            'S': subject_matrix,
            'Ssim': subject_subject_similarity
        }

    with open(simp_file, 'wb') as handle:
        pickle.dump(similarity_map, handle, protocol=pickle.HIGHEST_PROTOCOL)

    print('dump successful')


if __name__ == '__main__':
    start_task0('metadata.pickle', 'simp.pickle')
