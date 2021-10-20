import heapq
import math

import numpy as np


def get_rank_with_seeds(transition_m, m, seeds_list):
    random_jump_factor = 0.5
    seeds_v = np.zeros((40, 1))
    for seed in seeds_list:
        seeds_v[seed-1] = 1/40

    ranks = np.dot((np.identity(40) - (1-random_jump_factor) * transition_m), random_jump_factor * seeds_v)
    ranks = ranks.reshape(40)

    teleportation_discount = np.zeros(40)
    for i, row in enumerate(ranks):
        if i+1 not in seeds_list:
            teleportation_discount[i] = ranks[i] / (1-random_jump_factor)
        else:
            teleportation_discount[i] = (ranks[i] - (random_jump_factor / 40)) / (1-random_jump_factor)

    ranks_with_discount = []
    for i, row in enumerate(teleportation_discount):
        ranks_with_discount.append((i+1, row))
    ranks_with_discount = sorted(ranks_with_discount, reverse=True, key=lambda x: x[1])
    return ranks_with_discount[:m]


def get_rank(similarity_m, m):
    s_vector = np.full(40, fill_value=(1/40), dtype=np.float32)
    pagerank = np.random.uniform(low=0, high=1, size=40)
    pagerank_error = np.zeros(40)
    c = 0.3

    convergence = False
    while not convergence:
        pagerank_new = (1-c) * np.dot(similarity_m, pagerank) + c*s_vector
        pagerank_error_new = pagerank_new - pagerank

        convergence = True
        for i, row in enumerate(pagerank_error):
            if pagerank_erro8r_new[i] - pagerank_error[i] / pagerank_error[i] > 0.1:
                #RuntimeWarning: invalid value encountered in double_scalars
                convergence = False
        pagerank = pagerank_new
        pagerank_error = pagerank_error_new

    pagerank_with_index = []
    for i, row in enumerate(pagerank):
        pagerank_with_index.append((i, row))
    pagerank_with_index.sort(reverse=True, key=lambda x: x[1])
    return pagerank_with_index[:m]


def get_ascos_similarity(transition_matrix):
    ascos_s = np.random.uniform(-1, 1, (40, 40))
    ascos_error = np.zeros((40, 40))
    damping_factor = 0.3

    convergence = False
    while not convergence:
        ascos_s_next = np.zeros((40, 40))
        ascos_error_next = np.zeros((40, 40))
        for i, row in enumerate(transition_matrix):
            for j, score in enumerate(row):
                temp = 0
                if i==j:
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
        while h:
            curr = heapq.heappop(h)
            transition_matrix[i][curr[1]] = -curr[0]
    return transition_matrix.transpose()
