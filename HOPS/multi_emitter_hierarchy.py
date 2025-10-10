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

# TODO: White noise support
# TODO: Allow for a custom truncation condition of the auxiliary states (instead of the default triangular truncation condition)

# Multi emitter implementation    
class MultiParticleHierarchy(HOPSHierarchy):
    # Coupling operators
    L: list[np.ndarray]
    # The system dimension
    dimension: int

    def __init__(self, dimension: int, L: list[np.ndarray], G: list[list[complex]], W: list[list[complex]], depth: int):
        assert(len(L) > 0)
        for row in G:
            assert(len(L) == len(row))
        assert(len(G) == len(W))
        self.dimension = L[0].shape[0]
        for _L in L:
            if self.dimension != _L.shape[0]:
                assert(False)
        
        N = len(L)
        flattened_G = []
        flattened_W = []
        for i in range(N):
            for j in range(N):
                assert(len(G[i][j]) == len(W[i][j]))
                for g in G[i][j]: flattened_G.append(g)
                for w in W[i][j]: flattened_W.append(w)

        k_vector_dimension = len(flattened_G)

        k_vectors = hops_utils.generate_k_vecs(k_vector_dimension, depth)
        positive_neighbour_indices = hops_utils.generate_positive_neighbour_indices(k_vectors, depth)
        negative_neighbour_indices = hops_utils.generate_negative_neighbour_indices(k_vectors)
        k_vector_component_index_map = hops_utils.create_k_tuple_component_index_map(N, G)
        
        self.k_W_array = hops_utils.generate_kW_array(k_vectors, flattened_W)
        self.B = hops_utils.generate_B_matrix_multi_emitter(L, flattened_G, k_vectors, positive_neighbour_indices, negative_neighbour_indices, k_vector_component_index_map)
        self.P = hops_utils.generate_P_matrices_multi_emitter(N, dimension, k_vectors, flattened_G,  positive_neighbour_indices, k_vector_component_index_map)
        self.N = hops_utils.generate_N_matrices_multi_emitter(N, dimension, k_vectors, flattened_G, negative_neighbour_indices, k_vector_component_index_map)
        self.M = hops_utils.generate_shift_matrix_multi_emitter(N, flattened_G, k_vector_component_index_map)
        self.shift_indices = hops_utils.create_noise_shift_indices(N, k_vector_component_index_map)
        
        self.dimension = dimension
        self.L = L
        self.G = np.array(flattened_G, dtype=complex).conjugate()
        self.W = np.array(flattened_W, dtype=complex).conjugate()

    def solve_linear_HOPS(self, start: float, end: float, initial_state: np.ndarray, H: Callable[[float, np.ndarray], np.ndarray], z: list[Callable[[float], complex]], custom_operators: Callable[[float, np.ndarray], np.ndarray] = [], step_size: float = 0.01) -> tuple[np.ndarray, np.ndarray]:
        calculation = _HOPSCalculationMultiEmitter(self, H, z, None, custom_operators)
        initial_states = np.concatenate((initial_state, np.zeros(self.B.shape[0] - self.dimension)))
        solver = ODE_solvers.FixedStepRK45(calculation._linear_step, start, initial_states, end, max_step=step_size)
        return self._propagate(self.dimension, solver)

    def solve_non_linear_HOPS(self, start: float, end: float, initial_state: np.ndarray, H: Callable[[float, np.ndarray], np.ndarray], z: list[Callable[[float], complex]], shift_type: str | None = None, custom_operators: Callable[[float, np.ndarray], np.ndarray] = [], step_size: float = 0.01) -> tuple[np.ndarray, np.ndarray]:
        if shift_type is not None and shift_type != "mean-field":
            print(f"Unsupported shift type: {shift_type}")
            print("Supported shift types: None, \"mean-field\"")
            print("Running simulation with no shift")
        calculation = _HOPSCalculationMultiEmitter(self, H, z, shift_type, custom_operators)
        initial_states = np.concatenate((initial_state, np.zeros(self.B.shape[0] - self.dimension + self.G.shape[0])))
        solver = ODE_solvers.FixedStepRK45(calculation._non_linear_step, start, initial_states, end, max_step=step_size)
        return self._propagate(self.dimension, solver)

class _HOPSCalculationMultiEmitter: 
    hierarchy: MultiParticleHierarchy
    H: Callable[[float, np.ndarray], np.ndarray]
    z: list[Callable[[float], complex]]
    shift_type: str | None
    custom_operators: list[Callable[[float, np.ndarray], np.ndarray]]

    def __init__(self, hierarchy, H, z, shift_type, custom_operators):
        assert(len(z) == len(hierarchy.L))
        self.hierarchy = hierarchy
        self.H = H
        self.z = z
        self.shift_type = shift_type
        self.custom_operators = custom_operators

    def _linear_step(self, t: float, current_states: np.ndarray):
        H_eff = -1j * self.H(t, current_states[0:self.hierarchy.dimension])
        for i in range(len(self.z)):
            H_eff += self.z[i](t).conjugate() * self.hierarchy.L[i]
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
        #L_dagger_expectations = np.array([current_state.conjugate().dot(L.conjugate().T.dot(current_state)) / (norm_squared + 1e-10) for L in self.hierarchy.L], dtype=complex)
        L_expectations = np.array([current_state.conjugate().dot(L.dot(current_state)) / (norm_squared + 1e-10) for L in self.hierarchy.L], dtype=complex)
        L_dagger_expectations = L_expectations.conjugate()
        # Noise shifts
        noise_shifts = [np.sum(shift_vector[start:end]) for start, end in self.hierarchy.shift_indices]
        # G and W are already stored as conjugates
        shift_vector_result = self.hierarchy.M.dot(L_dagger_expectations) - self.hierarchy.W * shift_vector
        noises = [z(t).conjugate() + noise_shifts[i] for (i, z) in enumerate(self.z)]

        H_eff = -1j * self.H(t, current_state)
        for i, z in enumerate(noises):
            H_eff += z * self.hierarchy.L[i]
            if self.shift_type == "mean-field":
                H_eff -= noise_shifts[i].conjugate() * self.hierarchy.L[i].T.conjugate()
        for custom_operator in self.custom_operators:
            H_eff += custom_operator(t, current_state)

        vectors = current_states.reshape(-1, self.hierarchy.dimension)
        result = (vectors.dot(H_eff.T) + self.hierarchy.k_W_array[:, np.newaxis] * vectors).flatten()
        result += self.hierarchy.B.dot(current_states)
        #TODO: Construct hierarchy.N (for negative neighbour indices) and apply -L_exp * N.dot(current_states)
        for L_dagger_exp, P in zip(L_dagger_expectations, self.hierarchy.P):
            result += L_dagger_exp * P.dot(current_states)
        if self.shift_type == "mean-field":
            for L_exp, N in zip(L_expectations, self.hierarchy.N):
                result -= L_exp * N.dot(current_states)
        
        return np.concatenate((result, shift_vector_result))