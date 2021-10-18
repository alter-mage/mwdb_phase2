import heapq
import pickle
import os
import numpy as np
import graph_utilities
import csv


def start_task8():
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

    s_s_simp = latent_input['simp']
    transition_matrix = graph_utilities.get_transition_matrix(s_s_simp, n)
    ascos_similarity = graph_utilities.get_ascos_similarity(transition_matrix)
    ranks = graph_utilities.get_rank(ascos_similarity, m)

    out_file_path = '%s_%s_%s' % (str(8), str(n), str(m))
    fields = ['Subject', 'Score']
    with open(out_file_path + '.csv', 'w', newline='') as handle:
        write = csv.writer(handle)
        write.writerow(fields)
        writer = csv.writer(handle)
        writer.writerows(ranks)

    print('here')


if __name__ == '__main__':
    start_task8()