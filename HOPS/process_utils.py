# This file is part of https://github.com/Turku-Quantum-Optics/hops
#
# Copyright (c) 2024, Turku Quantum Optics
# 
# Licensed under the BSD 3-Clause License, see accompanying LICENSE,
# and README.md for further information.

import psutil

def physicalCoreCount():
    return psutil.cpu_count(logical=False)

def logicalCoreCount():
    return psutil.cpu_count(logical=True)
