import sys
import os
import pickle
import cv2

import color_moment
import elbp
import hog

def start_task0(metadata_file):
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

    print('dump successful')
