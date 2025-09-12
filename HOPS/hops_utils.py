# This file is part of https://github.com/Turku-Quantum-Optics/hops
#
# Copyright (c) 2024-2025, Turku Quantum Optics
# 
# Licensed under the BSD 3-Clause License, see accompanying LICENSE,
# and README.md for further information.

import numpy as np
import scipy as sp

from . import math_utils
import more_itertools

def positive_neighbours(k_vector: np.ndarray, k_max: int) -> list[np.ndarray]:
    result = []
    for i in range(k_vector.size):
        candidate = k_vector.copy()
        candidate[i] += 1
        if np.sum(candidate) <= k_max:
            result.append(candidate)
    return result

def negative_neighbours(k_vector: np.ndarray) -> list[np.ndarray]:
    result = []
    for i in range(k_vector.size):
        candidate = k_vector.copy()
        candidate[i] -= 1
        if np.all(candidate >= 0):
            result.append(candidate)
    return result

def generate_k_vecs(components: int, k_max: int) -> list[np.ndarray]:
    result: list = []
    for sum in range(0, k_max + 1):
        partitions = math_utils.find_partitions(sum)
        for partition in partitions:
            while len(partition) < components:
                partition.append(0)
            while len(partition) > components:
                partition.pop()

            for permutation in more_itertools.distinct_permutations(partition):
                perm = list(permutation)
                result.append(perm)
    kVectors = sorted(list(map(lambda permutation: tuple(list(reversed(permutation))), result)), key=np.sum)
    return [np.array(list(kVec)) for kVec in list(dict.fromkeys(kVectors))]

def generate_positive_neighbour_indices(k_vectors: list[np.ndarray], k_max: int) -> list[list[tuple[int, int]]]:
    # Construct the cache for k-vector indices
    k_vector_dictionary = dict({(tuple(k_vector.tolist()), index) for index, k_vector in enumerate(k_vectors)})
    result: list[list[tuple[int, int]]] = []
    for k_vector in k_vectors:
        if np.sum(k_vector) == k_max:
            result.append([])
            continue
        positive_neighbour_indices: list[int] = []
        for i in range(len(k_vector)):
            k_vector_copy = k_vector.copy()
            k_vector_copy[i] = k_vector[i] + 1
            if np.sum(k_vector_copy) <= k_max:
                positive_neighbour_indices.append((i, k_vector_dictionary[tuple(k_vector_copy.tolist())]))
        result.append(positive_neighbour_indices)
    return result

def generate_negative_neighbour_indices(k_vectors: list[np.ndarray]) -> list[list[tuple[int, int]]]:
    # Construct the cache for k-vector indices
    k_vector_dictionary = dict({(tuple(k_vector.tolist()), index) for index, k_vector in enumerate(k_vectors)})
    result: list[list[tuple[int, int]]] = []
    for k_vector in k_vectors:
        negative_neighbour_indices: list[tuple[int, int]] = []
        for i in range(len(k_vector)):
            k_vector_copy = k_vector.copy()
            k_vector_copy[i] = k_vector[i] - 1
            if k_vector_copy[i] >= 0:
                negative_neighbour_indices.append((i, k_vector_dictionary[tuple(k_vector_copy.tolist())]))
        result.append(negative_neighbour_indices)
    return result

def generate_kW_array(k_vectors: list[np.ndarray], W: np.ndarray) -> np.ndarray:
    return np.array([-np.sum(k_vector * W) for k_vector in k_vectors], dtype=complex)

def generate_B_matrix(L: np.ndarray, k_vectors: list[np.ndarray], 
                      positive_neighbour_indices: list[list[int]], 
                      negative_neighbour_indices: list[list[tuple[int, int]]], G: np.ndarray) -> sp.sparse.csr_matrix:
    lil_matrix = sp.sparse.lil_matrix((L.shape[0] * len(k_vectors), L.shape[1] * len(k_vectors)), dtype=complex)
    L_dagger = L.T.conjugate()
    for index, k_vector in enumerate(k_vectors):
        row = index
        positive_neighbours = positive_neighbour_indices[index]
        negative_neighbours = negative_neighbour_indices[index]
        for _, positive_neighbour_index in positive_neighbours:
            column = positive_neighbour_index
            for i in range(L_dagger.shape[0]):
                for j in range(L_dagger.shape[1]):
                    if L_dagger[i, j] != 0:
                        lil_matrix[row * L.shape[0] + i, column * L.shape[1] + j] = -L_dagger[i, j]

        for k_index, negative_neighbour_index in negative_neighbours:
            column = negative_neighbour_index
            if k_vector[k_index] == 0 or G[k_index] == 0: continue
            kG = k_vector[k_index] * G[k_index]
            for i in range(L.shape[0]):
                for j in range(L.shape[1]):
                    if L[i, j] != 0:
                        lil_matrix[row * L.shape[0] + i, column * L.shape[1] + j] += kG * L[i, j]
    return lil_matrix.tocsr(copy=False)

def generate_P_matrix(dimension: int, positive_neighbour_indices: list[list[tuple[int, int]]]) -> sp.sparse.csr_matrix:
    lil_matrix = sp.sparse.lil_matrix((dimension * len(positive_neighbour_indices), dimension * len(positive_neighbour_indices)), dtype=complex)
    for index, positive_neighbours in enumerate(positive_neighbour_indices):
        row = index
        for _, positive_neighbour_index in positive_neighbours:
            column = positive_neighbour_index
            for i in range(dimension):
                lil_matrix[row * dimension + i, column * dimension + i] = 1 + 0j
    return lil_matrix.tocsr(copy=False)

# Multi-emitter functions
def generate_B_matrix_multi_emitter(L: list[np.ndarray], G: list[list[complex]],
                                    k_vectors: list[list[int]], 
                                    positive_neighbour_indices: list[list[tuple[int, int]]], 
                                    negative_neighbour_indices: list[list[tuple[int, int]]],
                                    k_vector_coordinate_map: list[list[int]]):
    B = sp.sparse.lil_matrix((L[0].shape[0] * len(k_vectors), L[0].shape[1] * len(k_vectors)), dtype=complex)
    for index, k_vector in enumerate(k_vectors):
        row = index
        positive_neighbours = positive_neighbour_indices[index]
        negative_neighbours = negative_neighbour_indices[index]
        for k_index, positive_neighbour_index in positive_neighbours:
            column = positive_neighbour_index
            (n, _, _) = k_vector_coordinate_map[k_index]
            Ln_dagger = L[n].T.conjugate()
            for i in range(Ln_dagger.shape[0]):
                for j in range(Ln_dagger.shape[1]):
                    if Ln_dagger[i, j] != 0:
                        B[row * Ln_dagger.shape[0] + i, column * Ln_dagger.shape[1] + j] = -Ln_dagger[i, j]
            
        for k_index, negative_neighbour_index in negative_neighbours:
            column = negative_neighbour_index
            if k_vector[k_index] == 0 or G[k_index] == 0: continue
            (_, m, _) = k_vector_coordinate_map[k_index]
            kG = k_vector[k_index] * G[k_index]
            Lm = L[m]
            for i in range(Lm.shape[0]):
                for j in range(Lm.shape[1]):
                    if Lm[i, j] != 0:
                        B[row * Lm.shape[0] + i, column * Lm.shape[1] + j] = kG
    return B.tocsr(copy=False)

def generate_P_matrices_multi_emitter(N: int, dimension: int, 
                                      positive_neighbour_indices: list[list[tuple[int, int]]], 
                                      k_vector_coordinate_map: list[list[int]]) -> list[sp.sparse.csr_matrix]:
    lil_matrices = [sp.sparse.lil_matrix((dimension * len(positive_neighbour_indices), dimension * len(positive_neighbour_indices)), dtype=complex) for _ in range(N)]
    for k_tuple_index, positive_neighbours in enumerate(positive_neighbour_indices):
        row = k_tuple_index
        for k_component_index, positive_neighbour_index in positive_neighbours:
            column = positive_neighbour_index
            (n, _, _) = k_vector_coordinate_map[k_component_index]
            for i in range(dimension):
                lil_matrices[n][row * dimension + i, column * dimension + i] = 1 + 0j
    return [lilMatrix.tocsr(copy=False) for lilMatrix in lil_matrices]

def generate_shift_matrix_multi_emitter(N: int, G: np.ndarray, k_vector_coordinate_map: list[list[int]]) -> sp.sparse.csr_matrix:
    result = sp.sparse.lil_matrix((len(G), N), dtype=complex)
    for index in range(len(G)):
        (_, m, _) = k_vector_coordinate_map[index]
        result[index, m] = G[index].conjugate()
    return result.tocsr(copy=False)

def create_k_tuple_component_index_map(N, G):
    result = []
    for n in range(N):
        for m in range(N):
            for mu in range(len(G[n][m])):
                result.append((n, m, mu))
    return result

def create_noise_shift_indices(N: int, k_vector_coordinate_map: list[list[int]]) -> list[tuple[int, int]]:
    result = []
    start = 0
    end = 0
    for n in range(N):
        while end < len(k_vector_coordinate_map) and k_vector_coordinate_map[end][0] == n:
            end += 1
        result.append((start, end))
        start = end
    return result