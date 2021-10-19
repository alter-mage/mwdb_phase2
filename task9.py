import pickle
import os
import graph_utilities
import csv


def start_task9():
    with open('metadata.pickle', 'rb') as handle:
        metadata = pickle.load(handle)

    with open('simp.pickle', 'rb') as handle:
        simp = pickle.load(handle)

    latent_input_file = 'none'
    while not os.path.isfile(os.path.join(os.getcwd(), latent_input_file+'.pickle')):
        latent_input_file = input('name of latent file: ')
    with open(latent_input_file+'.pickle', 'rb') as handle:
        latent_input = pickle.load(handle)

    n = -1
    while not 1 <= n <= 40:
        n = int(input('enter n: '))

    m = -1
    while not 1 <= m <= n:
        m = int(input('enter m: '))

    subject_seeds = []
    for i in range(3):
        subject_seed = -1
        while not 1 <= subject_seed <= 40:
            subject_seed = int(input('enter a subject seed: '))
        subject_seeds.append(subject_seed)

    s_s_simp = latent_input['simp']
    transition_matrix = graph_utilities.get_transition_matrix(s_s_simp, n)
    ranks = graph_utilities.get_rank_with_seeds(transition_matrix, m, subject_seeds)

    out_file_path = '%s_%s_%s' % (str(9), str(n), str(m))
    fields = ['Subject', 'Score']
    with open(out_file_path + '.csv', 'w', newline='') as handle:
        write = csv.writer(handle)
        write.writerow(fields)
        writer = csv.writer(handle)
        writer.writerows(ranks)
