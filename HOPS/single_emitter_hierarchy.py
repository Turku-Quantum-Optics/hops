# This file is part of https://github.com/Turku-Quantum-Optics/hops
#
# Copyright (c) 2024-2025, Turku Quantum Optics
# 
# Licensed under the BSD 3-Clause License, see accompanying LICENSE,
# and README.md for further information.

import numpy as np

from typing import Callable

from .hierarchy import HOPSHierarchy
from . import hops_utils
from . import ODE_solvers

# TODO: Allow for a custom truncation condition of the auxiliary states (instead of the default triangular truncation condition)

# Single emitter implementation
class SingleParticleHierarchy(HOPSHierarchy):
    # Coupling operator
    L: np.ndarray
    # The system dimension
    dimension: int

    def __init__(self, dimension: int, L: np.ndarray, G: np.ndarray, W: np.ndarray, depth: int):
        assert(G.shape == W.shape)
        k_vectors = hops_utils.generate_k_vecs(G.size, depth)
        positive_neighbour_indices = hops_utils.generate_positive_neighbour_indices(k_vectors, depth)
        negative_neighbour_indices = hops_utils.generate_negative_neighbour_indices(k_vectors)
        self.k_W_array = hops_utils.generate_kW_array(k_vectors, W)
        self.B = hops_utils.generate_B_matrix(L, k_vectors, positive_neighbour_indices, negative_neighbour_indices, G)
        self.P = hops_utils.generate_P_matrix(dimension, positive_neighbour_indices)
        self.dimension = dimension
        self.L = L
        self.G = G
        self.W = W

    def solve_linear_HOPS(self, start: float, end: float, initial_state: np.ndarray, H: Callable[[float, np.ndarray], np.ndarray], z: Callable[[float], complex], custom_operators: Callable[[float, np.ndarray], np.ndarray] = [], step_size: float = 0.01) -> tuple[np.ndarray, np.ndarray]:
        calculation = _HOPSCalculationSingleEmitter(self, H, z, custom_operators)
        initial_states = np.concatenate((initial_state, np.zeros(self.B.shape[0] - self.dimension)))
        solver = ODE_solvers.FixedStepRK45(calculation._linear_step, start, initial_states, end, max_step=step_size)
        return self._propagate(self.dimension, solver)

    def solve_non_linear_HOPS(self, start: float, end: float, initial_state: np.ndarray, H: Callable[[float, np.ndarray], np.ndarray], z: Callable[[float], complex], custom_operators: Callable[[float, np.ndarray], np.ndarray] = [], step_size: float = 0.01) -> tuple[np.ndarray, np.ndarray]:
        calculation = _HOPSCalculationSingleEmitter(self, H, z, custom_operators)
        initial_states = np.concatenate((initial_state, np.zeros(self.B.shape[0] - self.dimension + self.G.shape[0])))
        solver = ODE_solvers.FixedStepRK45(calculation._non_linear_step, start, initial_states, end, max_step=step_size)
        return self._propagate(self.dimension, solver)

class _HOPSCalculationSingleEmitter:
    hierarchy: SingleParticleHierarchy
    H: Callable[[float, np.ndarray], np.ndarray]
    z: Callable[[float], complex]
    custom_operators: Callable[[float, np.ndarray], np.ndarray]

    def __init__(self, hierarchy, H, z, custom_operators):
        self.hierarchy = hierarchy
        self.H = H
        self.z = z
        self.custom_operators = custom_operators

    def _linear_step(self, t: float, current_states: np.ndarray):
        H_eff = -1j * self.H(t, current_states[0:self.hierarchy.dimension])
        H_eff += self.z(t).conjugate() * self.hierarchy.L
        for custom_operator in self.custom_operators:
            H_eff += custom_operator(t, current_states[0:self.hierarchy.dimension])

        vectors = current_states.reshape(-1, self.hierarchy.dimension)
        result = (vectors.dot(H_eff.T) + self.hierarchy.k_W_array[:, np.newaxis] * vectors).flatten()
        result += self.hierarchy.B.dot(current_states)
        return result
    
    def _non_linear_step(self, t: float, current_states: np.ndarray):
        shift_vector = current_states[-len(self.hierarchy.G):]
        current_states = current_states[0:-len(self.hierarchy.G)]
        current_state = current_states[0:self.hierarchy.dimension]
        norm_squared = current_state.conjugate().dot(current_state)
        L_dagger_expectation = current_state.conjugate().dot(self.hierarchy.L.conjugate().T.dot(current_state)) / (norm_squared + 1e-10)

        # Noise shift
        noise_shift = np.sum(shift_vector)
        shift_vector_result = L_dagger_expectation * self.hierarchy.G.conjugate() - self.hierarchy.W.conjugate() * shift_vector
        noise = self.z(t).conjugate() + noise_shift

        H_eff = -1j * self.H(t, current_states[0:self.hierarchy.dimension])
        H_eff += noise * self.hierarchy.L
        for custom_operator in self.custom_operators:
            H_eff += custom_operator(t, current_states[0:self.hierarchy.dimension])

        vectors = current_states.reshape(-1, self.hierarchy.dimension)
        result = (vectors.dot(H_eff.T) + self.hierarchy.k_W_array[:, np.newaxis] * vectors).flatten()
        result += self.hierarchy.B.dot(current_states)
        result += L_dagger_expectation * self.hierarchy.P.dot(current_states)
        return np.concatenate((result, shift_vector_result))