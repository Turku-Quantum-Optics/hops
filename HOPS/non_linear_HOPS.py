# This file is part of https://github.com/Turku-Quantum-Optics/hops
#
# Copyright (c) 2024, Turku Quantum Optics
# 
# Licensed under the BSD 3-Clause License, see accompanying LICENSE,
# and README.md for further information.

import numpy as np
import scipy as sp

from scipy.integrate import solve_ivp
from multiprocessing.pool import Pool

import hops_utils
import process_utils
import progress_management
import ODE_solvers
import SDE_solvers

from typing import Callable
import time
import copy

class SimulationState:
    # Hamiltonian H
    H: Callable[[float, np.ndarray], np.ndarray]
    #TODO: Don't assume two-level system
    HBaseMatrix00: sp.sparse.csr_matrix
    HBaseMatrix01: sp.sparse.csr_matrix
    HBaseMatrix10: sp.sparse.csr_matrix
    HBaseMatrix11: sp.sparse.csr_matrix

    # Coupling operator L
    L: np.ndarray
    LBaseMatrix: sp.sparse.csr_matrix

    # Constant base matrix
    baseMatrix: sp.sparse.csr_matrix

    # Neighbour index matrices
    positiveNeighbourMatrix: sp.sparse.csr_matrix

    # BCF exponential sum coefficients
    G: np.ndarray
    W: np.ndarray
    
    # Continuous noise processes
    z: Callable[[float], complex]
    customParts: list[(Callable[[float], complex], sp.sparse.csr_matrix)]

    # White noise processes
    # TODO: Support multiple white noise processes
    whiteNoise: Callable[[float], complex] = None
    whiteNoiseOperator: np.ndarray = None

    def __init__(self, t_span, HFunc, L, kVecs, G, W, posNeighbourIndices, negNeighbourIndices, max_step: float = 0.01):
        self.G = G
        self.W = W
        assert(len(posNeighbourIndices) == len(negNeighbourIndices))
        #TODO: Don't assume two-level system
        # Construct Hamiltonian sparse base matrices
        HBaseMatrix00 = sp.sparse.lil_matrix((len(kVecs) * 2, len(kVecs) * 2), dtype=complex)
        HBaseMatrix01 = sp.sparse.lil_matrix((len(kVecs) * 2, len(kVecs) * 2), dtype=complex)
        HBaseMatrix10 = sp.sparse.lil_matrix((len(kVecs) * 2, len(kVecs) * 2), dtype=complex)
        HBaseMatrix11 = sp.sparse.lil_matrix((len(kVecs) * 2, len(kVecs) * 2), dtype=complex)

        #TODO: Don't assume two-level system
        for index in range(len(kVecs)):
            HBaseMatrix00[2 * index, 2 * index] = 1
            HBaseMatrix01[2 * index, 2 * index + 1] = 1
            HBaseMatrix10[2 * index + 1, 2 * index] = 1
            HBaseMatrix11[2 * index + 1, 2 * index + 1] = 1

        self.HBaseMatrix00 = HBaseMatrix00.tocsr(copy=False)
        self.HBaseMatrix01 = HBaseMatrix01.tocsr(copy=False)
        self.HBaseMatrix10 = HBaseMatrix10.tocsr(copy=False)
        self.HBaseMatrix11 = HBaseMatrix11.tocsr(copy=False)

        self.H = HFunc
        
        # Coupling operator
        self.L = L
        #TODO: Don't assume two-level system
        LBaseMatrix = sp.sparse.lil_matrix((len(kVecs) * 2, len(kVecs) * 2), dtype=complex)
        for index in range(len(kVecs)):
            for i in range(0, 2):
                for j in range(0, 2):
                    if L[i, j] != 0:
                        LBaseMatrix[2 * index + i, 2 * index + j] = L[i, j]
        self.LBaseMatrix = LBaseMatrix.tocsr(copy=False)

        # Construct constant base matrix
        #TODO: Don't assume two-level system
        baseMatrix = sp.sparse.lil_matrix((len(kVecs) * 2, len(kVecs) * 2), dtype=complex)
        for index, kVec in enumerate(kVecs):
            baseMatrix[2*index, 2*index] = -np.sum(kVec * W)
            baseMatrix[2*index + 1, 2*index + 1] = -np.sum(kVec * W)
            for posIndex in posNeighbourIndices[index]:
                #TODO: Don't assume two-level system
                for i in range(0, 2):
                    for j in range(0, 2):
                        if L[i, j] != 0:
                            baseMatrix[2 * index + i, 2 * posIndex + j] = -L.T[i, j].conjugate()
            for mu, e_mu in negNeighbourIndices[index]:
                k_muG = kVec[mu] * G[mu]
                for i in range(0, 2):
                    for j in range(0, 2):
                        if k_muG * L[i, j] != 0:
                            baseMatrix[2 * index + i, 2 * e_mu + j] = k_muG * L[i, j]
        # Convert to CSR format to benefit from efficient matrix vector multiplication
        self.baseMatrix = baseMatrix.tocsr(copy=False)
        
        # Positive neighbour matrix
        #TODO: Don't assume two-level system
        positiveNeighbourMatrix = sp.sparse.lil_matrix((len(posNeighbourIndices) * 2, len(posNeighbourIndices) * 2), dtype=complex)
        for index, posIndices in enumerate(posNeighbourIndices):
            for posIndex in posIndices:
                positiveNeighbourMatrix[2 * index, 2 * posIndex] = 1
                positiveNeighbourMatrix[2 * index + 1, 2 * posIndex + 1] = 1
        self.positiveNeighbourMatrix = positiveNeighbourMatrix.tocsr(copy=False)

    def step(self, t: float, currentStates: np.ndarray) -> np.ndarray:
        shiftVector = currentStates[-len(self.G):]
        currentStates = currentStates[:-len(self.G)]
        currentState = currentStates[0:2]
        normSquared = currentState.conjugate().dot(currentState)
        if normSquared != 0.0:
            meanLDagger_t = currentState.conjugate().dot(self.L.conjugate().T.dot(currentState)) / normSquared
            if np.isnan(meanLDagger_t) or np.isinf(meanLDagger_t):
                print("<L\dagger>_t was", meanLDagger_t)
                meanLDagger_t = 0.0 + 0j
        else:
            print("Norm was zero?")
            meanLDagger_t = 0.0

        # Noise
        noise = self.z(t).conjugate()
        noiseShift = np.sum(shiftVector)
        noise += noiseShift
        shiftVectorResult = meanLDagger_t * self.G.conjugate() - self.W.conjugate() * shiftVector

        result = np.zeros(currentStates.shape, dtype=complex)
        result = self.baseMatrix.dot(currentStates)
        result += noise * self.LBaseMatrix.dot(currentStates)
        result += meanLDagger_t * self.positiveNeighbourMatrix.dot(currentStates)

        # Hamiltonian
        H = self.H(t, currentStates)
        #TODO: Don't assume two-level system
        h00 = H[0, 0]
        h01 = H[0, 1]
        h10 = H[1, 0]
        h11 = H[1, 1]
        if h00 != 0:
            result -= 1j * h00 * self.HBaseMatrix00.dot(currentStates)
        if h01 != 0:
            result -= 1j * h01 * self.HBaseMatrix01.dot(currentStates)
        if h10 != 0:
            result -= 1j * h10 * self.HBaseMatrix10.dot(currentStates)
        if h11 != 0:
            result -= 1j * h11 * self.HBaseMatrix11.dot(currentStates)

        # Custom operators
        for func, operator in self.customParts:
            result += func(t) * operator.dot(currentStates)

        return np.append(result, shiftVectorResult)

    def noise(self, t: float, currentStates: np.ndarray):
        if self.whiteNoiseOperator is None: return np.zeros(currentStates.shape, dtype=complex)
        currentStates = currentStates[:-len(self.G)]
        shiftVector = currentStates[-len(self.G):]
        result = self.whiteNoiseOperator.dot(currentStates)
        return np.append(result, shiftVector)

    def newCopy(self, z: Callable[[float], complex], customParts: list[(Callable[[float], complex], np.ndarray)], customWhiteNoise: Callable[[float], complex], customWhiteNoiseOperator: np.ndarray):
        _copy = copy.deepcopy(self)
        _copy.z = z
        _copy.customParts = [] 
        for func, operator in customParts:
            operatorBaseMatrix = sp.sparse.lil_matrix(_copy.baseMatrix.shape, dtype=complex)
            for index in range(int(operatorBaseMatrix.shape[0]/2)):
                for i in range(0, 2):
                    for j in range(0, 2):
                        if operator[i, j] != 0:
                            operatorBaseMatrix[2 * index + i, 2 * index + j] = operator[i, j]
            _copy.customParts.append((func, operatorBaseMatrix.tocsr()))

        # White noise
        if customWhiteNoise is not None and customWhiteNoiseOperator is not None:
            _copy.whiteNoise = customWhiteNoise
            _copy.whiteNoiseOperator = sp.sparse.lil_matrix(_copy.baseMatrix.shape, dtype=complex)
            #TODO: Don't assume two-level system
            for index in range(_copy.whiteNoiseOperator.shape[0] // 2):
                for i in range(0, 2):
                    for j in range(0, 2):
                        if customWhiteNoiseOperator[i, j] != 0:
                            _copy.whiteNoiseOperator[2 * index + i, 2 * index + j] = customWhiteNoiseOperator[i, j]
            _copy.whiteNoiseOperator = _copy.whiteNoiseOperator.tocsr()
        return _copy

def _nonLinearHOPSFuncWithAuxiliary(t_span: np.ndarray, initial_conditions, simulation_state: SimulationState, noise: Callable[[float], complex], customParts: list[(Callable[[float], complex], np.ndarray)]):
    simulationState = simulation_state.newCopy(noise, customParts)
    solution = solve_ivp(simulationState.step, (t_span[0], t_span[-1]), np.append(initial_conditions, np.zeros(simulationState.G.shape)), t_eval=t_span, max_step=0.01)
    return solution

def _nonLinearHOPSFunc(t_span, initial_conditions, max_step: float, simulation_state: SimulationState, noise: Callable[[float], complex], customParts: list[(Callable[[float], complex], np.ndarray)], progressBar: progress_management.ProgressBar, customWhiteNoise: Callable[[float], complex], customWhiteNoiseOperator: np.ndarray):
    simulationState = simulation_state.newCopy(noise, customParts, customWhiteNoise, customWhiteNoiseOperator)
    isODE = simulationState.whiteNoise is None or simulationState.whiteNoiseOperator is None
    initialConditions = np.append(initial_conditions, np.zeros(simulationState.G.shape))
    if isODE:
        if max_step is not None and max_step != np.inf:
            solver = ODE_solvers.FixedStepRK23(simulationState.step, t_span[0], initialConditions, t_span[-1], max_step=max_step)
        else:
            solver = ODE_solvers.AdaptiveStepRK45(simulationState.step, t_span[0], initialConditions, t_span[-1])
    else:
        # TODO: Support arbitrary many white noise processes
        #solver = SDE_solvers.SDESolver(t_span, initialConditions, simulationState.step, simulationState.noise, simulationState.whiteNoise)
        #solver = SDE_solvers.RK4SmallNoiseSDESolver(t_span, initialConditions, simulationState.step, simulationState.noise, simulationState.whiteNoise)
        solver = SDE_solvers.HeunSDESolver(t_span, initialConditions, simulationState.step, simulationState.noise, simulationState.whiteNoise)
        #solver = SDE_solvers.WeakSecondOrderAutonomousSDESolver(t_span, initialConditions, simulationState.step, simulationState.noise, simulationState.whiteNoise)
    tSpace = [0]
    solutions = [initialConditions[0:2]]
    status = None
    t = 0.0
    while status is None:
        if progressBar is not None:
            progressBar.update(t)
        solver.step()
        if solver.status == 'finished':
            status = 0
        elif solver.status == 'failed':
            status = -1
            break
        if solver.step_size == 0: continue
        t = solver.t
        y = np.array([solver.y[0], solver.y[1]], dtype=complex)
        tSpace.append(t)
        solutions.append(y)

    if status == -1:
        print("Simulation failed!")
        return None
    if progressBar is not None:
        progressBar.complete()
    tSpace.append(solver.t)
    solutions.append(solver.y[0:2])

    tSpace = np.array(tSpace)
    tSpace, indices = np.unique(tSpace, return_index=True)

    groundSolution = np.array([sol[0] for sol in solutions])[indices]
    excitedSolution = np.array([sol[1] for sol in solutions])[indices]
    if isODE:
        groundSolution = sp.interpolate.CubicSpline(tSpace, groundSolution)(t_span)
        excitedSolution = sp.interpolate.CubicSpline(tSpace, excitedSolution)(t_span)
    return (t_span, groundSolution, excitedSolution)

def _nonLinearHOPSWrapper(args): #(t_span, initial_conditions, max_step, simulation_state, noise: Callable[[float], complex], customParts, whiteNoise, whiteNoiseOperator):
    t_span, initial_conditions, max_step, simulation_state, noise, customParts, progressBar, whiteNoise, whiteNoiseOperator = args
    return _nonLinearHOPSFunc(t_span, initial_conditions, max_step, simulation_state, noise, customParts, progressBar, whiteNoise, whiteNoiseOperator)

#TODO: Implement function "densityMatrixFromTrajectories(trajectories: [NonLinearHOPSTrajectory])"
#TODO: Make this return a list of NonLinearHOPSTrajectory objects
def solve_non_linear_HOPS(t_span, 
                          initial_condition: tuple[complex, complex], 
                          H: Callable[[float], tuple[complex, complex, complex, complex]], 
                          L: np.ndarray,
                          noises: list[Callable[[float], complex]],
                          G: np.ndarray,
                          W: np.ndarray, 
                          customParts: list[(list[Callable[[float], complex]], np.ndarray)] = [],
                          whiteNoises: list[(Callable[[float], complex], np.ndarray)] = [],
                          kMax: int = 8,
                          max_step: float = np.inf,
                          parallelization: int | None = None,
                          logProgress: bool = False):
    coreCount = process_utils.logicalCoreCount()
    
    if parallelization is None:
        if len(noises) == 1:
            parallelization = 1
        else:
            parallelization = coreCount
    
    parallelization = min(coreCount, len(noises), parallelization)

    assert(len(W) == len(G))
    kVecComponentCount = W.shape[0]
    kVecs: list[np.ndarray] = hops_utils.generate_k_vecs(kVecComponentCount, kMax)
    posNeighbourIndices: list[list[int]] = hops_utils.generate_positive_neighbour_indices(kVecs, kMax)
    negNeighbourIndices: list[list[tuple[int, int]]] = hops_utils.generate_negative_neighbour_indices(kVecs)
    # Construct the common simulation state that will be copied for each trajectory
    simulation_state = SimulationState(t_span, H, L, kVecs, G, W, posNeighbourIndices, negNeighbourIndices)
    initial_conditions: list[complex] = []
    # TODO: We could also use some other kind of initial condition for the auxiliary states, see papers and their recommendations
    for _ in kVecs:
        initial_conditions.append(0)
        initial_conditions.append(0)
    for i in range(len(initial_condition)):
        initial_conditions[i] = initial_condition[i]

    print(f"Solving HOPS for {len(noises)} trajectories")
    start = time.time()
    solutions = []
    args = []
    if logProgress:
        with progress_management.ProgressManager() as manager:
            mainProgress = manager.addProgress("[green]Simulating:", total=len(noises))
            if len(whiteNoises) == 0:
                whiteNoises = [(None, None) for _ in noises]
            for index, noise in enumerate(noises):
                _customParts = [(customPart[0][index], customPart[1]) for customPart in customParts]
                progressBar = manager.addProgress(f"Trajectory {index + 1}", total=t_span[-1], visible=False)
                args.append((t_span, initial_conditions, max_step, simulation_state, noise, _customParts, progressBar, whiteNoises[index][0], whiteNoises[index][1]))

            if len(args) == 1:
                for index, solution in enumerate(map(_nonLinearHOPSWrapper, args)):
                    mainProgress.update(index + 1)
                    solutions.append(solution)
            else:
                with Pool(processes=parallelization) as p:
                    for index, solution in enumerate(p.imap(_nonLinearHOPSWrapper, args, 1)):
                        mainProgress.update(index + 1)
                        solutions.append(solution)
    else:
        if len(whiteNoises) == 0:
            whiteNoises = [(None, None) for _ in noises]
        for index, noise in enumerate(noises):
            _customParts = [(customPart[0][index], customPart[1]) for customPart in customParts]
            progressBar = None
            args.append((t_span, initial_conditions, max_step, simulation_state, noise, _customParts, progressBar, whiteNoises[index][0], whiteNoises[index][1]))

        start = time.time()
        if len(args) == 1:
            for index, solution in enumerate(map(_nonLinearHOPSWrapper, args)):
                solutions.append(solution)
        else:
            with Pool(processes=parallelization) as p:
                for index, solution in enumerate(p.imap(_nonLinearHOPSWrapper, args, 1)):
                    solutions.append(solution)
    end = time.time()
    print(f"Execution took {(end - start):.3f} seconds")
    
    tSpace = solutions[0][0]
    groundSolutions = list()
    excitedSolutions = list()
    for solution in solutions:
        groundSolutions.append(solution[1])
        excitedSolutions.append(solution[2])
    
    groundSolutions = np.array(groundSolutions)
    excitedSolutions = np.array(excitedSolutions)
    return (tSpace, groundSolutions, excitedSolutions)

def solve_non_linear_HOPS_with_auxiliary(t_span,
                                         initial_condition: tuple[complex, complex],
                                         H: Callable[[float], tuple[complex, complex, complex, complex]],
                                         L: np.ndarray,
                                         noise: Callable[[float], complex],
                                         W: np.ndarray,
                                         G: np.ndarray,
                                         customParts: list[(Callable[[float], complex], np.ndarray)] = [],
                                         whiteNoise: Callable[[float], complex] = None,
                                         whiteNoiseOperator: np.ndarray = None,
                                         kMax: int = 8):
    assert(len(W) == len(G))
    kVecComponentCount = W.shape[0]
    print("Generating k-vectors")
    kVecs: list[np.ndarray] = hops_utils.generate_k_vecs(kVecComponentCount, kMax)
    print("Indexing")
    posNeighbourIndices: list[list[int]] = hops_utils.generate_positive_neighbour_indices(kVecs, kMax)
    negNeighbourIndices: list[list[tuple[int, int]]] = hops_utils.generate_negative_neighbour_indices(kVecs)
    print("Constructing simulation state")
    # Construct the common simulation state that will be copied for each trajectory
    simulation_state = SimulationState(t_span, H, L, kVecs, G, W, posNeighbourIndices, negNeighbourIndices)
    initial_conditions: list[complex] = []
    # TODO: We could also use some other kind of initial condition, see papers and their recommendations
    for _ in kVecs:
        initial_conditions.append(0)
        initial_conditions.append(0)
    for i in range(len(initial_condition)):
        initial_conditions[i] = initial_condition[i]

    print(f"Solving HOPS with auxiliary states")
    start = time.time()
    solutions = _nonLinearHOPSFuncWithAuxiliary(t_span, initial_conditions, simulation_state, noise, customParts)
    end = time.time()
    print(f"Execution took {end - start} seconds")
    return  solutions
