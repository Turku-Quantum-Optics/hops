# This file is part of https://github.com/Turku-Quantum-Optics/hops
#
# Copyright (c) 2024-2025, Turku Quantum Optics
# 
# Licensed under the BSD 3-Clause License, see accompanying LICENSE,
# and README.md for further information.

import numpy as np
import scipy as sp

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

def generate_kW_array(kVecs: list[np.ndarray], W: np.ndarray) -> np.ndarray:
    resultArray = []
    for kTuple in kVecs:
        result = 0 + 0j
        for i in range(kTuple.size):
            result -= kTuple[i] * W[i]
        resultArray.append(result)
    return np.array(resultArray, dtype=complex)

def generate_B_Matrix(L: np.ndarray, kVecs: list[np.ndarray], 
                      positiveNeighbourIndices: list[list[int]], 
                      negativeNeighbourIndices: list[list[tuple[int, int]]], G: np.ndarray) -> sp.sparse.csr_matrix:
    lilMatrix = sp.sparse.lil_matrix((L.shape[0] * len(kVecs), L.shape[1] * len(kVecs)), dtype=complex)
    Ldagger = L.T.conjugate()
    for index, kTuple in enumerate(kVecs):
        row = index
        positiveNeighbours = positiveNeighbourIndices[index]
        negativeNeighbours = negativeNeighbourIndices[index]
        for positiveNeighbourIndex in positiveNeighbours:
            column = positiveNeighbourIndex
            for i in range(Ldagger.shape[0]):
                for j in range(Ldagger.shape[1]):
                    if Ldagger[i, j] != 0:
                        lilMatrix[row * L.shape[0] + i, column * L.shape[1] + j] = -Ldagger[i, j]

        for kIndex, negativeNeighbourIndex in negativeNeighbours:
            column = negativeNeighbourIndex
            if kTuple[kIndex] == 0 or G[kIndex] == 0: continue
            kG = kTuple[kIndex] * G[kIndex]
            for i in range(L.shape[0]):
                for j in range(L.shape[1]):
                    if L[i, j] != 0:
                        lilMatrix[row * L.shape[0] + i, column * L.shape[1] + j] += kG * L[i, j]
    return lilMatrix.tocsr(copy=False)

def generate_P_matrix(dimension: int, positiveNeighbourIndices: list[list[int]]) -> sp.sparse.csr_matrix:
    lilMatrix = sp.sparse.lil_matrix((dimension * len(positiveNeighbourIndices), dimension * len(positiveNeighbourIndices)), dtype=complex)
    for index, positiveNeighbours in enumerate(positiveNeighbourIndices):
        row = index
        for positiveNeighbourIndex in positiveNeighbours:
            column = positiveNeighbourIndex
            for i in range(dimension):
                lilMatrix[row * dimension + i, column * dimension + i] = 1 + 0j
    
    return lilMatrix.tocsr(copy=False)