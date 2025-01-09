import numpy as np
import scipy as sp

from scipy.integrate import solve_ivp
from multiprocessing.pool import Pool
from typing import Callable

import time
import copy

import hops_utils
import process_utils
import SDE_solvers
import progress_management

class SimulationState:
    # Hamiltonian H
    H: Callable[[float, np.ndarray], np.ndarray]
    #TODO: Don't assume two-level system
    # HMatrixParts: list[sp.sparse.csr_matrix] or similar
    HBaseMatrix00: sp.sparse.csr_matrix
    HBaseMatrix01: sp.sparse.csr_matrix
    HBaseMatrix10: sp.sparse.csr_matrix
    HBaseMatrix11: sp.sparse.csr_matrix

    # Coupling operator L
    L: np.ndarray
    LBaseMatrix: sp.sparse.csr_matrix

    # Constant matrix
    baseMatrix: sp.sparse.csr_matrix

    # Continuous noise processes
    z: Callable[[float], complex]
    customParts: list[(Callable[[float], complex], sp.sparse.csr_matrix)]

    # White noise processes
    # TODO: Support multiple white noise processes
    whiteNoise: Callable[[float], complex] = None
    whiteNoiseOperator: np.ndarray = None

    def __init__(self, t_span, HFunc, L, kVecs, G, W, posNeighbourIndices, negNeighbourIndices, max_step: float = 0.01):
        assert(len(posNeighbourIndices) == len(negNeighbourIndices))
        # TODO: Don't assume a two-level system
        # Construct Hamiltonian sparse base matrices
        HBaseMatrix00 = sp.sparse.lil_matrix((len(kVecs) * 2, len(kVecs) * 2), dtype=complex)
        HBaseMatrix01 = sp.sparse.lil_matrix((len(kVecs) * 2, len(kVecs) * 2), dtype=complex)
        HBaseMatrix10 = sp.sparse.lil_matrix((len(kVecs) * 2, len(kVecs) * 2), dtype=complex)
        HBaseMatrix11 = sp.sparse.lil_matrix((len(kVecs) * 2, len(kVecs) * 2), dtype=complex)

        #TODO: Don't assume two-level system
        for index in range(len(posNeighbourIndices)):
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
        # TODO: Don't assume two-level system
        dimension = 2
        LBaseMatrix = sp.sparse.lil_matrix((len(kVecs) * dimension, len(kVecs) * dimension), dtype=complex)
        for index in range(len(kVecs)):
            for i in range(0, dimension):
                for j in range(0, dimension):
                    if L[i, j] != 0:
                        LBaseMatrix[dimension * index + i, dimension * index + j] = L[i, j]
        self.LBaseMatrix = LBaseMatrix.tocsr(copy=False)

        # Construct constant base matrix
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
    
    def step(self, t: float, currentStates):
        # Noise
        noise = self.z(t).conjugate()
        result = self.baseMatrix.dot(currentStates) 
        result += noise * self.LBaseMatrix.dot(currentStates)
        
        # Hamiltonian
        # TODO: Do not assume a two-level system
        H = self.H(t, currentStates)
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

        return result
    
    def noise(self, t: float, currentStates: np.ndarray):
        return self.whiteNoiseOperator.dot(currentStates)

    def newCopy(self, noise, customParts, whiteNoise: Callable[[float], complex], whiteNoiseOperator: np.ndarray):
        _copy = copy.deepcopy(self)
        _copy.z = noise
        _copy.customParts = []
        #TODO: Don't assume two-level system
        dimension = 2
        for func, operator in customParts:
            operatorBaseMatrix = sp.sparse.lil_matrix(_copy.baseMatrix.shape, dtype=complex)
            for index in range(int(operatorBaseMatrix.shape[0]/2)):
                for i in range(0, 2):
                    for j in range(0, 2):
                        if operator[i, j] != 0:
                            operatorBaseMatrix[2 * index + i, 2 * index + j] = operator[i, j]
            _copy.customParts.append((func, operatorBaseMatrix.tocsr()))

        # White noise
        if whiteNoise is not None and whiteNoise is not None:
            _copy.whiteNoise = whiteNoise
            _copy.whiteNoiseOperator = sp.sparse.lil_matrix(_copy.baseMatrix.shape, dtype=complex)
            for index in range(_copy.whiteNoiseOperator.shape[0] // 2):
                for i in range(0, 2):
                    for j in range(0, 2):
                        if whiteNoiseOperator[i, j] != 0:
                            _copy.whiteNoiseOperator[2 * index + i, 2 * index + j] = whiteNoiseOperator[i, j]
            _copy.whiteNoiseOperator = _copy.whiteNoiseOperator.tocsr()

        return _copy

def _linearHOPSFuncWithAuxiliary(t_span, initial_conditions, simulation_state: SimulationState, noise: Callable[[float], complex], customParts: list[(Callable[[float], complex], np.ndarray)]):
    simulationState = simulation_state.newCopy(noise, customParts)
    solution = solve_ivp(simulationState.step, (t_span[0], t_span[-1]), np.array(initial_conditions), t_eval=t_span, max_step=0.001)
    return solution

def _linearHOPSFunc(t_span, initial_conditions, max_step: float, simulation_state: SimulationState, noise: Callable[[float], complex], customParts: list[(Callable[[float], complex], np.ndarray)], progressBar: progress_management.ProgressBar, customWhiteNoise: Callable[[float], complex], customWhiteNoiseOperator: np.ndarray):
    simulationState = simulation_state.newCopy(noise, customParts, customWhiteNoise, customWhiteNoiseOperator)
    isODE = simulationState.whiteNoise is None or simulationState.whiteNoiseOperator is None
    if isODE:
        solver = sp.integrate.RK45(simulationState.step, t_span[0], initial_conditions, t_span[-1], first_step=0.00001, max_step=max_step)
    else:
        # TODO: Support arbitrary many white noise processes
        solver = SDE_solvers.SDESolver(t_span, np.array(initial_conditions), simulationState.step, simulationState.noise, simulationState.whiteNoise)
        #solver = SDE_solvers.SDESolver(t_span, initial_conditions, simulationState.step, simulationState.noise, lambda t: 0)
    
    tSpace = [0]
    solutions = [np.array(initial_conditions)[0:2]]
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
        if isODE and max_step < np.inf:
            while t < solver.t:
                t0 = solver.t_old
                y0 = solver.y_old[0:2]
                y1 = solver.y[0:2]
                y = y0 + (t - t0) * (y1 - y0) / solver.step_size
                tSpace.append(t)
                solutions.append(y)
                t += max_step   
        else:
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

def _linearHOPSWrapper(args): # (t_span, initial_conditions, max_step, simulation_state: CustomSimulationState, noise: Callable[[float], complex], customParts, whiteNoise, whiteNoiseOperator, progressBar):
    t_span, initial_conditions, max_step, simulation_state, noise, customParts, progressBar, whiteNoise, whiteNoiseOperator = args
    return _linearHOPSFunc(t_span, initial_conditions, max_step, simulation_state, noise, customParts, progressBar, whiteNoise, whiteNoiseOperator)

#TODO: Implement function "densityMatrixFromTrajectories(trajectories: [LinearHOPSTrajectory])"
#TODO: Return a list of LinearHOPSTrajectory objects
#TODO: Swap order of W and G parameters
def solve_linear_HOPS(t_span, 
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
    # TODO: We could also use some other kind of initial condition, see papers and their recommendations
    for _ in kVecs:
        initial_conditions.append(0)
        initial_conditions.append(0)
        
    for i in range(len(initial_condition)):
        initial_conditions[i] = initial_condition[i]

    print(f"Solving HOPS for {len(noises)} trajectories")
    solutions = []
    with progress_management.ProgressManager() as manager:
        args = []
        if logProgress:
            mainProgress = manager.addProgress("[green]Simulating:", total=len(noises))
        if len(whiteNoises) == 0:
            whiteNoises = [(None, None) for _ in noises]
        for index, noise in enumerate(noises):
            _customParts = [(customPart[0][index], customPart[1]) for customPart in customParts]
            progressBar = manager.addProgress(f"Trajectory {index + 1}", total=t_span[-1], visible=False) if logProgress else None
            args.append((t_span, initial_conditions, max_step, simulation_state, noise, _customParts, progressBar, whiteNoises[index][0], whiteNoises[index][1]))
        #args = list(map(lambda noise: (t_span, initial_conditions, simulation_state, noise), noises))
        #bar = IncrementalBar("Simulating", max=len(args), suffix='%(percent).0f%% - ETA: %(eta)ds - Trajectories: %(index)d / %(max)d')
        start = time.time()
        if len(args) == 1:
            for index, solution in enumerate(map(_linearHOPSWrapper, args)):
                trajectoryIndex = index + 1
                if logProgress: mainProgress.update(trajectoryIndex)
                solutions.append(solution)
        else:
            with Pool(parallelization) as p:
                for index, solution in enumerate(p.imap(_linearHOPSWrapper, args, 1)):
                    trajectoryIndex = index + 1
                    if logProgress: mainProgress.update(trajectoryIndex)
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

def solve_linear_HOPS_with_auxiliary(t_span, 
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
    solutions = _linearHOPSFuncWithAuxiliary(t_span, initial_conditions, simulation_state, noise, customParts)
    end = time.time()
    print(f"Execution took {end - start} seconds")
    return solutions
