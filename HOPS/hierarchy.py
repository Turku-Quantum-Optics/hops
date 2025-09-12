# This file is part of https://github.com/Turku-Quantum-Optics/hops
#
# Copyright (c) 2024-2025, Turku Quantum Optics
# 
# Licensed under the BSD 3-Clause License, see accompanying LICENSE,
# and README.md for further information.

import numpy as np

# Base class for hierarchies
class HOPSHierarchy:
    def map_linear_trajectory(self, trajectories: np.ndarray):
        return np.einsum('ni,nj->nij', trajectories, trajectories.conjugate())

    def map_non_linear_trajectory(self, trajectories: np.ndarray):
        norms = np.linalg.norm(trajectories, axis=1, keepdims=True)
        normalized = trajectories / norms
        return np.einsum('ni,nj->nij', normalized, normalized.conjugate())
    
    def _propagate(self, dimension, solver):
        t_space = [solver.t]
        solutions = [np.array(solver.y[0:dimension], copy=True, dtype=complex)]
        status = None
        while status is None:
            solver.step()
            if solver.status == 'finished':
                status = 0
                break
            elif solver.status == 'failed':
                status = -1
                break
            if solver.step_size == 0: continue
            t = solver.t
            y = np.array(solver.y[0:dimension], copy=True, dtype=complex)
            t_space.append(t)
            solutions.append(y)
        if status == -1:
            print("Simulation failed!")
            return None
        t_space = np.array(t_space)
        solutions = np.array(solutions)
        return (t_space, solutions)