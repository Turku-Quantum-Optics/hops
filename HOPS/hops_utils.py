# This file is part of https://github.com/Turku-Quantum-Optics/hops
#
# Copyright (c) 2024, Turku Quantum Optics
# 
# Licensed under the BSD 3-Clause License, see accompanying LICENSE,
# and README.md for further information.

import numpy as np

import math_utils
import more_itertools

def positive_neighbours(kVector: np.ndarray, kMax: int) -> list[np.ndarray]:
    result = []
    for i in range(kVector.size):
        candidate = kVector.copy()
        candidate[i] += 1
        if np.sum(candidate) <= kMax:
            result.append(candidate)
    return result

def negative_neighbours(kVector: np.ndarray) -> list[np.ndarray]:
    result = []
    for i in range(kVector.size):
        candidate = kVector.copy()
        candidate[i] -= 1
        if np.all(candidate >= 0):
            result.append(candidate)
    return result

def generate_k_vecs(components: int, kMax: int) -> list[np.ndarray]:
    result: list = []
    for sum in range(0, kMax + 1):
        partitions = math_utils.findPartitions(sum)
        for partition in partitions:
            while len(partition) < components:
                partition.append(0)
            while len(partition) > components:
                partition.pop()

            for permutation in more_itertools.distinct_permutations(partition):
                perm = list(permutation)
                result.append(perm)
    
    #kVectors = sorted(list(map(lambda permutation: np.array(list(reversed(permutation))), result)), key=np.sum)
    kVectors = sorted(list(map(lambda permutation: tuple(list(reversed(permutation))), result)), key=np.sum)
    return [np.array(list(kVec)) for kVec in list(dict.fromkeys(kVectors))]

#TODO: This needs to be modified for different truncation conditions and e.g. making \phi^kMax+1 = something_constant
def generate_positive_neighbour_indices(kVecs: list[np.ndarray], kMax: int) -> list[list[int]]:
    # Construct the cache for k-vector indices
    kVecDict = dict({(tuple(kVec.tolist()), index) for index, kVec in enumerate(kVecs)})
    result: list[list[int]] = []
    for kVec in kVecs:
        if np.sum(kVec) == kMax:
            result.append([])
            continue
        positive_neighbour_indices: list[int] = []
        for i in range(len(kVec)):
            _kVec = kVec.copy()
            _kVec[i] = kVec[i] + 1
            if np.sum(_kVec) <= kMax:
                positive_neighbour_indices.append(kVecDict[tuple(_kVec.tolist())])
        result.append(positive_neighbour_indices)
    return result

def generate_negative_neighbour_indices(kVecs: list[np.ndarray]) -> list[list[tuple[int, int]]]:
    # Construct the cache for k-vector indices
    kVecDict = dict({(tuple(kVec.tolist()), index) for index, kVec in enumerate(kVecs)})
    result: list[list[tuple[int, int]]] = []
    for kVec in kVecs:
        negative_neighbour_indices: list[tuple[int, int]] = []
        for i in range(len(kVec)):
            _kVec = kVec.copy()
            _kVec[i] = kVec[i] - 1
            if _kVec[i] >= 0:
                negative_neighbour_indices.append((i, kVecDict[tuple(_kVec.tolist())]))
        result.append(negative_neighbour_indices)
    return result