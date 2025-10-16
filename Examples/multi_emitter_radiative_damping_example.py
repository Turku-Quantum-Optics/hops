# This file is part of https://github.com/Turku-Quantum-Optics/hops
#
# Copyright (c) 2024-2025, Turku Quantum Optics
# 
# Licensed under the BSD 3-Clause License, see accompanying LICENSE,
# and README.md for further information.

import numpy as np
import HOPS
import scipy as sp

import matplotlib.pyplot as plt

def spectral_density(omega: np.ndarray, a: float, ksi: float) -> np.ndarray:
    return a * omega * omega * omega * np.exp(-omega * omega / (ksi * ksi))

def spectral_density_non_diag(omega: np.ndarray, a: float, gamma: float, omegaC: float) -> np.ndarray:
    omegaTilde = omega - omegaC
    return a * gamma / (omegaTilde * omegaTilde + gamma * gamma)

def spectral_density_array(omega: np.ndarray, a: float, ksi: float, A: float, gamma: float, omegaC: float) -> list[np.ndarray]:
    result = []
    for w in omega:
        Jdiag = np.diag(np.repeat(spectral_density(w, a, ksi), 3))
        JnonDiag = np.array([
            [0, 1, 1],
            [1, 0, 1],
            [1, 1, 0]
        ], dtype=complex) * spectral_density_non_diag(w, A, gamma, omegaC)
        result.append(Jdiag + JnonDiag)
    return result

def BCF_array(tau: np.ndarray, J, wmax=np.inf):
    result = np.zeros_like(tau, dtype=complex)
    for i, tau in enumerate(tau):
        integrand = lambda w: J(w) * np.exp(-1j * w * tau)
        result[i] = sp.integrate.quad(lambda w: integrand(w), 0, wmax, limit=500, complex_func=True)[0]
    return np.array(result)

def partial_trace(rho, keep, dims):
    N = len(dims)
    trace_over = [i for i in range(N) if i not in keep]

    reshaped = rho.reshape(dims + dims)

    for ax in sorted(trace_over, reverse=True):
        reshaped = np.trace(reshaped, axis1=ax, axis2=ax + reshaped.ndim//2)

    kept_dims = [dims[i] for i in keep]
    reshaped = reshaped.reshape(np.prod(kept_dims), np.prod(kept_dims))
    return reshaped

class Hamiltonian:
    def __init__(self, H1, H2, H3):
        self.H = H1 + H2 + H3

    def __call__(self, t, current_state):
        return self.H

class SpectralDensityArray:
    def __init__(self, a, ksi, A, gamma, omegaC):
        self.a = a
        self.ksi = ksi
        self.A = A
        self.gamma = gamma
        self.omegaC = omegaC

    def __call__(self, omega):
        return spectral_density_array(omega, self.a, self.ksi, self.A, self.gamma, self.omegaC)

class SigmaMinus:
    def __init__(self, dimension, subsystems, index):
        sigma_minus = np.array([[0, 1], [0, 0]], dtype=complex)
        identity = np.identity(dimension, dtype=complex)
        O = sigma_minus if index == 0 else identity
        if dimension <= 1:
            self.O = O
            return
        for i in range(1, subsystems):
            O = np.kron(O, sigma_minus) if i == index else np.kron(O, identity)
        self.O = O

    def __call__(self, t, state):
        return self.O

class CustomOperator:
    def __init__(self, gamma, O, normalize):
        self.gamma = gamma
        self.O = O
        self.ODagger = O.conjugate().T
        self.normalize = normalize
        
    def __call__(self, t, state):
        norm_squared = state.conjugate().dot(state)
        ODaggerExpectation = state.conjugate().dot(self.ODagger.dot(state)) / (norm_squared + 1e-12)
        if self.normalize:
            return -0.5 * self.gamma * self.ODagger.dot(self.O) + self.gamma * ODaggerExpectation * self.O
        else:
            return -0.5 * self.gamma * self.ODagger.dot(self.O)
        
def master_equation_solution(T, initial_rho, H, gamma, O):
    ODagger = O.conjugate().T
    ODaggerO = ODagger.dot(O)
    tSpace = [0]
    rho = [initial_rho]
    t = 0
    y = initial_rho
    dt = 0.01
    while t < T:
        t += dt
        y = y + dt * (-1j * (H.H.dot(y) - y.dot(H.H)) + gamma * (O.dot(y).dot(ODagger) - 0.5 * (ODaggerO.dot(y) + y.dot(ODaggerO))))
        tSpace.append(t)
        rho.append(y)
    return tSpace, np.array(rho, dtype=complex)

def _main():
    # Simulation time
    T = 50.0

    # Interaction parameters (BCF)
    A = 0.01
    gamma = 0.1
    omegaC = 1.0

    # Radiative damping
    gammaR = 0.0175

    # Number of trajectories to compute
    trajectories = 1 << 10

    # Spectral density of the phonon environment
    JDiag = lambda omega: spectral_density(omega, 0.0000027, 1.447)

    # Find the exponential sum form of the BCF
    tau = np.linspace(0, 20, 501)
    bcfDiag = BCF_array(tau, JDiag)
    #gDiag, wDiag = HOPS.fit_BCF(tau, bcfDiag, Kr=4, Ki=4)
    # Use the matrix pencil method since it gives better results in most cases
    gDiag, wDiag = HOPS.fit_BCF_matrix_pencil(bcfDiag, tau[1] - tau[0], K=3)

    G = [[gDiag, [], []],
         [[], gDiag, []],
         [[], [], gDiag]]

    W = [[wDiag, [], []],
         [[], wDiag, []],
         [[], [], wDiag]]

    L1 = L2 = L3 = np.array([[0, 0],
                             [0, 1]], dtype=complex)
    
    # Construct the array of coupling operators
    L = [np.kron(L1, np.kron(np.identity(2), np.identity(2))),
         np.kron(np.identity(2), np.kron(L2, np.identity(2))),
         np.kron(np.identity(2), np.kron(np.identity(2), L3))]

    # Construct the hierarchy
    hierarchy = HOPS.MultiParticleHierarchy(2**3, L, G, W, 4)

    # Initial state
    state = np.array([np.sqrt(0.5), np.sqrt(0.5)], dtype=complex)
    initial_state = np.kron(state, np.kron(state, state))

    # Hamiltonian
    _H = np.array([[0, 0],
                   [0, 1]], dtype=complex)
    H1 = 0.1 * np.kron(_H, np.kron(np.identity(2), np.identity(2)))
    H2 = 0.2 * np.kron(np.identity(2), np.kron(_H, np.identity(2)))
    H3 = 0.3 * np.kron(np.identity(2), np.kron(np.identity(2), _H))

    H = Hamiltonian(H1, H2, H3)

    # Create noise generator
    J_array = SpectralDensityArray(0.027, 1.447, A, gamma, omegaC)
    generator = HOPS.MultiGaussianNoiseProcessFFTGenerator(T, J_array)
    
    sigmaMinus = SigmaMinus(dimension=2, subsystems=3, index=0)
    wGenerator = HOPS.GaussianWhiteNoiseProcessGenerator(0, gammaR)
    wNoises = wGenerator.generate(trajectories)
    # Linear HOPS
    print("Simulating linear trajectories")
    rho = None
    linear_t_space = None
    customOperator = CustomOperator(gammaR, sigmaMinus.O, normalize=False)
    for i, w in zip(range(trajectories), wNoises):
        noises = generator.generate()
        print("Linear", i)
        _tSpace, trajectory = hierarchy.solve_linear_HOPS(0.0, T, initial_state, H, noises, 
                                                          custom_operators=[customOperator], 
                                                          diffusion_operators=[sigmaMinus],
                                                          white_noises=[w], step_size=0.1)
        if linear_t_space is None: linear_t_space = _tSpace
        _rho = hierarchy.map_linear_trajectory(trajectory) / trajectories
        if rho is None: 
            rho = _rho
        else: 
            rho += _rho
        
    linear_rho = np.array([partial_trace(rho[i], keep=[0], dims=[2,2,2]) for i in range(len(linear_t_space))])

    # Non-linear HOPS
    print("Simulating non-linear trajectories")
    rho = None
    non_linear_t_space = None
    customOperator = CustomOperator(gammaR, sigmaMinus.O, normalize=True)
    for i, w in zip(range(trajectories), wNoises):
        noises = generator.generate()
        print("Non-linear", i)
        _tSpace, trajectory = hierarchy.solve_non_linear_HOPS(0.0, T, initial_state, H, noises, 
                                                              custom_operators=[customOperator], 
                                                              diffusion_operators=[sigmaMinus],
                                                              white_noises=[w], step_size=0.1)
        if non_linear_t_space is None: non_linear_t_space = _tSpace
        _rho = hierarchy.map_non_linear_trajectory(trajectory) / trajectories
        if rho is None: 
            rho = _rho
        else: 
            rho += _rho
    non_linear_rho = np.array([partial_trace(rho[i], keep=[0], dims=[2,2,2]) for i in range(len(non_linear_t_space))])
    i = 0
    master_equation_t_space, master_equation_rho = master_equation_solution(T, np.outer(initial_state, initial_state), H, gammaR, sigmaMinus.O)
    master_equation_rho = np.array([partial_trace(master_equation_rho[i], keep=[0], dims=[2,2,2]) for i in range(len(master_equation_t_space))])

    # Plot population and coherences 
    linear_population = linear_rho[:, 0, 0].real - linear_rho[:, 1, 1].real
    linear_coherence = 2 * linear_rho[:, 0, 1].real
    linear_coherence_imag = 2 * linear_rho[:, 0, 1].imag

    non_linear_population = non_linear_rho[:, 0, 0].real - non_linear_rho[:, 1, 1].real
    non_linear_coherence = 2 * non_linear_rho[:, 0, 1].real
    non_linear_coherence_imag = 2 * non_linear_rho[:, 0, 1].imag

    master_equation_population = master_equation_rho[:, 0, 0].real - master_equation_rho[:, 1, 1].real
    master_equation_coherence = 2 * master_equation_rho[:, 0, 1].real
    master_equation_coherence_imag = 2 * master_equation_rho[:, 0, 1].imag
    
    # Plot
    plt.figure()
    plt.plot(linear_t_space, linear_population, label="HOPS: <z>")
    plt.plot(linear_t_space, linear_coherence, label="HOPS: <x>")
    plt.plot(linear_t_space, linear_coherence_imag, label="HOPS: <y>")

    plt.plot(master_equation_t_space, master_equation_population, label="Master Equation: <z>")
    plt.plot(master_equation_t_space, master_equation_coherence, label="Master Equation: <x>")
    plt.plot(master_equation_t_space, master_equation_coherence_imag, label="Master Equation: <y>")

    plt.title("Linear HOPS")
    plt.legend()
    plt.show()
    plt.close()

    plt.figure()
    plt.plot(non_linear_t_space, non_linear_population, label="HOPS: <z>")
    plt.plot(non_linear_t_space, non_linear_coherence, label="HOPS: <x>")
    plt.plot(non_linear_t_space, non_linear_coherence_imag, label="HOPS: <y>")

    plt.plot(master_equation_t_space, master_equation_population, label="Master Equation: <z>")
    plt.plot(master_equation_t_space, master_equation_coherence, label="Master Equation: <x>")
    plt.plot(master_equation_t_space, master_equation_coherence_imag, label="Master Equation: <y>")

    plt.title("Non-linear HOPS")
    plt.legend()
    plt.show()
    plt.close()

if __name__ == "__main__":
    _main()