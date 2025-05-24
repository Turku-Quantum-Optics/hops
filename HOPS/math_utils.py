# This file is part of https://github.com/Turku-Quantum-Optics/hops
#
# Copyright (c) 2024-2025, Turku Quantum Optics
# 
# Licensed under the BSD 3-Clause License, see accompanying LICENSE,
# and README.md for further information.

import numpy as np

def _findPartitions(array: list[int], index: int, number: int, reducedNum: int) -> list[list[int]]:
    result: list[list[int]] = list()
    if reducedNum < 0: return result
    if reducedNum == 0:
        retVal = list[int]()
        for i in reversed(range(index)):
            retVal.append(array[i])
        return [retVal]
    
    previous = 1 if index == 0 else array[index - 1]
    for k in range(previous, number + 1):
        array[index] = k
        for res in _findPartitions(array, index + 1, number, reducedNum - k):
            result.append(res)

    return result

def findPartitions(number: int) -> list[list[int]]:
    array = list(map(lambda _: 0, range(number)))
    return _findPartitions(array, 0, number, number)

def coth(x):
    return 1.0 / np.tanh(x)