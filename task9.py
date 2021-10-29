import pickle
import os
import graph_utilities
import csv
import numpy as np


def start_task9():
    with open('metadata.pickle', 'rb') as handle:
        metadata = pickle.load(handle)

    with open('simp.pickle', 'rb') as handle:
        simp = pickle.load(handle)

    latent_input_file = 'none'
    while not (
            os.path.isfile(os.path.join(os.getcwd(), latent_input_file + '.pickle')) and
            latent_input_file.startswith('4')
    ):
        latent_input_file = input(
            'Name of latent file from task 4 (subject-subject similarity matrix is contained within): ')

    with open(latent_input_file + '.pickle', 'rb') as handle:
        latent_input = pickle.load(handle)

    n = -1
    while not 1 <= n <= 40:
        n = int(input('Enter n (most similar subjects): '))

    m = -1
    while not 1 <= m <= n:
        m = int(input('Enter m (most significant m subjects): '))

    subject_seeds = []
    for i in range(3):
        subject_seed = -1
        while not 1 <= subject_seed <= 40:
            subject_seed = int(input('Enter a subject ID: '))
            if subject_seed in subject_seeds:
                subject_seed = -1
        subject_seeds.append(subject_seed - 1)

    s_s_simp = latent_input['simp']
    transformed_s_s_matrix = graph_utilities.transform_weights(s_s_simp)
    transition_matrix = graph_utilities.get_transition_matrix(transformed_s_s_matrix, n)
    ranks = graph_utilities.get_rank(transformed_s_s_matrix, len(transformed_s_s_matrix))
    teleportation_discount_matrix = graph_utilities.get_teleportation_discount_matrix(ranks, subject_seeds)
    robust_ranks = graph_utilities.get_robust_ranks(transition_matrix, teleportation_discount_matrix, m)

    out_file_path = '%s_%s_%s' % (str(9), str(n), str(m))
    fields = ['Subject', 'Score']
    with open(out_file_path + '.csv', 'w', newline='') as handle:
        write = csv.writer(handle)
        write.writerow(fields)
        writer = csv.writer(handle)
        writer.writerows(robust_ranks)


if __name__ == '__main__':
    start_task9()
