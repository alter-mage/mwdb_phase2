import heapq
import math
import min_max_scaler
import numpy as np


def get_rank(similarity_m, m):
    num_subjects = len(similarity_m)
    s_vector = np.full(num_subjects, fill_value=(1 / num_subjects), dtype=np.float32)
    pagerank = np.random.uniform(low=0, high=1, size=num_subjects)
    pagerank_error = np.zeros(num_subjects)
    c = 0.5

    convergence = False
    while not convergence:
        pagerank_new = (1 - c) * np.dot(similarity_m, pagerank) + c * s_vector
        pagerank_error_new = pagerank_new - pagerank

        convergence = True
        for i, row in enumerate(pagerank_error):
            if pagerank_error_new[i] - pagerank_error[i] > 0.01:
                # RuntimeWarning: invalid value encountered in double_scalars
                convergence = False
        pagerank = pagerank_new
        pagerank_error = pagerank_error_new

    pagerank_with_index = []
    for i, row in enumerate(pagerank):
        pagerank_with_index.append((i, row))
    pagerank_with_index.sort(reverse=True, key=lambda x: x[1])
    return pagerank_with_index[:m]


def get_ascos_similarity(transition_matrix):
    num_subjects = len(transition_matrix)
    ascos_s = np.random.uniform(-1, 1, (num_subjects, num_subjects))
    ascos_error = np.zeros((num_subjects, num_subjects))
    damping_factor = 0.3

    convergence = False
    while not convergence:
        ascos_s_next = np.zeros((num_subjects, num_subjects))
        ascos_error_next = np.zeros((num_subjects, num_subjects))
        for i, row in enumerate(transition_matrix):
            for j, score in enumerate(row):
                temp = 0
                if i == j:
                    ascos_s_next[i][j] = 1
                else:
                    row_sum = np.sum(transition_matrix[i])
                    for k, weight in enumerate(ascos_s[i]):
                        temp += transition_matrix[i][k] * (1 - math.exp(-transition_matrix[i][k])) * ascos_s[k][j]
                    if row_sum != 0:
                        temp *= damping_factor / row_sum
                ascos_s_next[i][j] = temp
                ascos_error[i][j] = ascos_s[i][j] - ascos_s_next[i][j]
        convergence = True
        for i, row in enumerate(ascos_error):
            for j, score in enumerate(row):
                if (ascos_error_next[i][j] - ascos_error[i][j]) / ascos_error[i][j] > 0.1:
                    convergence = False
                    break
        ascos_s = ascos_s_next
        ascos_error = ascos_error_next
    return ascos_s


def get_transition_matrix(s_s_simp, n):
    transition_matrix = np.zeros((40, 40), dtype=np.float32)
    for i, row in enumerate(s_s_simp):
        h = []
        heapq.heapify(h)
        for j, score in enumerate(row):
            if i != j:
                heapq.heappush(h, (-score, j))
            if len(h) > n:
                heapq.heappop(h)
        transition_matrix[i][i] = 1
        while h:
            curr = heapq.heappop(h)
            transition_matrix[i][curr[1]] = -curr[0]
    return transition_matrix.transpose()


def transform_weights(s_s_simp):
    num_subjects = len(s_s_simp)
    for i in range(num_subjects):
        curr_column_sum = sum(s_s_simp[:, i].tolist())
        for j in range(num_subjects):
            s_s_simp[j][i] /= curr_column_sum
    return s_s_simp


def get_teleportation_discount_matrix(ranks, subject_seeds):
    random_jump_probability = 0.5
    teleportation_discount_matrix = np.zeros((len(ranks), 1))
    for i, rank in enumerate(ranks):
        if i in subject_seeds:
            teleportation_discount_matrix[i][0] = rank[1] / (1-random_jump_probability)
        else:
            teleportation_discount_matrix[i][0] = (rank[1]-(random_jump_probability/3)) / (1-random_jump_probability)
    return teleportation_discount_matrix


def get_robust_ranks(transition_matrix, teleportation_discount_matrix, m):
    num_subjects = len(transition_matrix)
    robust_rank = np.random.uniform(low=0, high=1, size=(num_subjects, 1))
    robust_rank_error = np.zeros(num_subjects)
    random_teleportation_probability = 0.5

    convergence = False
    while not convergence:
        robust_rank_new = (1 - random_teleportation_probability) * np.dot(transition_matrix, robust_rank) +\
                       random_teleportation_probability * teleportation_discount_matrix
        robust_rank_error_new = robust_rank_new - robust_rank

        convergence = True
        for i, row in enumerate(robust_rank_error):
            if robust_rank_error_new[i] - robust_rank_error[i] > 0.001:
                # RuntimeWarning: invalid value encountered in double_scalars
                convergence = False
        robust_rank = robust_rank_new
        robust_rank_error = robust_rank_error_new

    robust_rank_with_index = []
    for i, row in enumerate(robust_rank):
        robust_rank_with_index.append((i, row[0]))
    robust_rank_with_index.sort(reverse=True, key=lambda x: x[1])
    robust_ranks_m = robust_rank_with_index[:m]
    return robust_ranks_m
