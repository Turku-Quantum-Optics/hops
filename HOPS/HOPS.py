# This file is part of https://github.com/Turku-Quantum-Optics/hops
#
# Copyright (c) 2024-2025, Turku Quantum Optics
# 
# Licensed under the BSD 3-Clause License, see accompanying LICENSE,
# and README.md for further information.

import numpy as np
import scipy as sp

from typing import Callable

import hops_utils
import ODE_solvers

class SingleParticleHierarchy:
    # Coupling operator
    L: np.ndarray
    # The system dimension
    dimension: int

    def __init__(self, dimension: int, L: np.ndarray, G: np.ndarray, W: np.ndarray, depth: int):
        assert(G.shape == W.shape)
        kTuples = hops_utils.generate_k_vecs(G.size, depth)
        positiveNeighbourIndices = hops_utils.generate_positive_neighbour_indices(kTuples, depth)
        negativeNeighbourIndices = hops_utils.generate_negative_neighbour_indices(kTuples)
        self.kWArray = hops_utils.generate_kW_array(kTuples, W)
        self.B = hops_utils.generate_B_Matrix(L, kTuples, positiveNeighbourIndices, negativeNeighbourIndices, G)
        self.P = hops_utils.generate_P_matrix(dimension, positiveNeighbourIndices)
        self.dimension = dimension
        self.L = L
        self.G = G
        self.W = W

    def solveLinearHOPS(self, start: float, end: float, initialState: np.ndarray, H: Callable[[float, np.ndarray], np.ndarray], z: Callable[[float], complex], customOperators: Callable[[float, np.ndarray], np.ndarray] = [], stepSize: float = 0.01) -> tuple[np.ndarray, np.ndarray]:
        calculation = _HOPSCalculation(self, H, z, customOperators)
        initialStates = np.concatenate((initialState, np.zeros(self.B.shape[0] - self.dimension)))
        solver = ODE_solvers.FixedStepRK45(calculation._linearStep, start, initialStates, end, max_step=stepSize)
        return self._propagate(solver)

    def solveNonLinearHOPS(self, start: float, end: float, initialState: np.ndarray, H: Callable[[float, np.ndarray], np.ndarray], z: Callable[[float], complex], customOperators: Callable[[float, np.ndarray], np.ndarray] = [], stepSize: float = 0.01) -> tuple[np.ndarray, np.ndarray]:
        calculation = _HOPSCalculation(self, H, z, customOperators)
        initialStates = np.concatenate((initialState, np.zeros(self.B.shape[0] - self.dimension + self.G.shape[0])))
        solver = ODE_solvers.FixedStepRK45(calculation._nonLinearStep, start, initialStates, end, max_step=stepSize)
        return self._propagate(solver)

    def mapLinearTrajectory(self, trajectories: np.ndarray):
        return np.einsum('ni,nj->nij', trajectories, trajectories.conjugate())

    def mapNonLinearTrajectory(self, trajectories: np.ndarray):
        norms = np.linalg.norm(trajectories, axis=1, keepdims=True)
        normalized = trajectories / norms
        return np.einsum('ni,nj->nij', normalized, normalized.conjugate())

    def _propagate(self, solver):
        tSpace = [solver.t]
        solutions = [np.array(solver.y[0:self.dimension], copy=True, dtype=complex)]
        status = None
        while status is None:
            solver.step()
            if solver.status == 'finished':
                status = 0
            elif solver.status == 'failed':
                status = -1
                break
            if solver.step_size == 0: continue
            t = solver.t
            y = np.array(solver.y[0:self.dimension], copy=True, dtype=complex)
            tSpace.append(t)
            solutions.append(y)
                
        if status == -1:
            print("Simulation failed!")
            return None
        tSpace.append(solver.t)
        solutions.append(np.array(solver.y[0:self.dimension], copy=True, dtype=complex))
        tSpace = np.array(tSpace)
        solutions = np.array(solutions)
        return (tSpace, solutions)

class _HOPSCalculation:
    hierarchy: SingleParticleHierarchy
    H: Callable[[float, np.ndarray], np.ndarray]
    z: Callable[[float], complex]
    customOperators: Callable[[float, np.ndarray], np.ndarray]

    def __init__(self, hierarchy, H, z, customOperators):
        self.hierarchy = hierarchy
        self.H = H
        self.z = z
        self.customOperators = customOperators

    def _linearStep(self, t: float, currentStates: np.ndarray):
        Heff = -1j * self.H(t, currentStates[0:self.hierarchy.dimension])
        Heff += self.z(t).conjugate() * self.hierarchy.L
        for customOperator in self.customOperators:
            Heff += customOperator(t, currentStates[0:self.hierarchy.dimension])

        vectors = currentStates.reshape(-1, self.hierarchy.dimension)
        result = (vectors.dot(Heff.T) + self.hierarchy.kWArray[:, np.newaxis] * vectors).flatten()
        result += self.hierarchy.B.dot(currentStates)
        return result
    
    def _nonLinearStep(self, t: float, currentStates: np.ndarray):
        shiftVector = currentStates[-len(self.hierarchy.G):]
        currentStates = currentStates[0:-len(self.hierarchy.G)]
        currentState = currentStates[0:self.hierarchy.dimension]
        normSquared = currentState.conjugate().dot(currentState)
        LDaggerExpectation = currentState.conjugate().dot(self.hierarchy.L.conjugate().T.dot(currentState)) / (normSquared + 1e-10)

        # Noise shift
        noiseShift = np.sum(shiftVector)
        shiftVectorResult = LDaggerExpectation * self.hierarchy.G.conjugate() - self.hierarchy.W.conjugate() * shiftVector
        noise = self.z(t).conjugate() + noiseShift

        Heff = -1j * self.H(t, currentStates[0:self.hierarchy.dimension])
        Heff += noise * self.hierarchy.L
        for customOperator in self.customOperators:
            Heff += customOperator(t, currentStates[0:self.hierarchy.dimension])

        vectors = currentStates.reshape(-1, self.hierarchy.dimension)
        result = (vectors.dot(Heff.T) + self.hierarchy.kWArray[:, np.newaxis] * vectors).flatten()
        result += self.hierarchy.B.dot(currentStates)
        result += LDaggerExpectation * self.hierarchy.P.dot(currentStates)
        return np.concatenate((result, shiftVectorResult))