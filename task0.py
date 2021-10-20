import os
import pickle
import cv2
import aggregation
import utilities


def start_task0(metadata_file, simp_file):
    images_dir = os.path.join(os.getcwd(), 'all')
    if not os.path.isdir(images_dir):
        print('download image dataset first')

    images = {}
    for filename in os.listdir(images_dir):
        img = cv2.imread(os.path.join(images_dir, filename), cv2.IMREAD_GRAYSCALE)
        if img is not None:
            images[filename] = {'image': img}
            for i in range(3):
                images[filename][utilities.feature_models[i]] = utilities.feature_extraction[i](img)

    with open(metadata_file, 'wb') as handle:
        pickle.dump(images, handle, protocol=pickle.HIGHEST_PROTOCOL)

    similarity_map = {}
    for i in range(3):
        types, type_matrix, type_type_similarity = aggregation.group_by_type_all(images, i)
        subjects, subject_matrix, subject_subject_similarity = aggregation.group_by_subject_all(images, i)
        similarity_map[utilities.feature_models[i]] = {
            'T': type_matrix,
            'Tsim': type_type_similarity,
            'S': subject_matrix,
            'Ssim': subject_subject_similarity,
            'types': types,
            'subjects': subjects
        }

    with open(simp_file, 'wb') as handle:
        pickle.dump(similarity_map, handle, protocol=pickle.HIGHEST_PROTOCOL)

    print('dump successful')
x

if __name__ == '__main__':
    start_task0('metadata.pickle', 'simp.pickle')
