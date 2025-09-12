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

from NMQSD import NMQSDDriver

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

def _main():
    # Simulation time
    T = 100.0

    # Interaction parameters (BCF)
    A = 0.01
    gamma = 0.1
    omegaC = 1.0

    # Number of trajectories to compute
    trajectories = 1 << 1

    # Spectral density of the phonon environment
    JDiag = lambda omega: spectral_density(omega, 0.027, 1.447)

    # Find the exponential sum form of the BCF
    tau = np.linspace(0, 20, 501)
    bcfDiag = BCF_array(tau, JDiag)
    #gDiag, wDiag = HOPS.fit_BCF(tau, bcfDiag, Kr=4, Ki=4)
    # Use the matrix pencil method since it gives better results in most cases
    gDiag, wDiag = HOPS.fit_BCF_matrix_pencil(bcfDiag, tau[1] - tau[0], K=3)
    gNonDiag = np.array([np.pi * A], dtype=complex)
    wNonDiag = np.array([1j * omegaC + gamma], dtype=complex)

    G = [[gDiag, gNonDiag, gNonDiag],
         [gNonDiag, gDiag, gNonDiag],
         [gNonDiag, gNonDiag, gDiag]]

    W = [[wDiag, wNonDiag, wNonDiag],
         [wNonDiag, wDiag, wNonDiag],
         [wNonDiag, wNonDiag, wDiag]]

    L1 = L2 = L3 = np.array([[0, 0],
                             [0, 1]], dtype=complex)
    
    # Construct the array of coupling operators
    L = [np.kron(L1, np.kron(np.identity(2), np.identity(2))),
         np.kron(np.identity(2), np.kron(L2, np.identity(2))),
         np.kron(np.identity(2), np.kron(np.identity(2), L3))]

    # Construct the hierarchy
    hierarchy = HOPS.MultiParticleHierarchy(2**3, L, G, W, 4)
    nmqsd_driver = NMQSDDriver(3, L, G, W)

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
    
    # Linear HOPS
    print("Simulating linear trajectories")
    rho = None
    nmqsd_rho = None
    linear_t_space = None
    for i in range(trajectories):
        noises = generator.generate()
        print("Linear", i)
        _tSpace, trajectory = hierarchy.solve_linear_HOPS(0.0, T, initial_state, H, noises, step_size=0.1)
        _, nmqsd_trajectory = nmqsd_driver.solve_linear_NMQSD(_tSpace, initial_state, H1 + H2 + H3, noises, max_step=0.1)
        if linear_t_space is None: linear_t_space = _tSpace
        _rho = hierarchy.map_linear_trajectory(trajectory) / trajectories
        _nmqsd_rho = nmqsd_driver.map_linear_trajectory(nmqsd_trajectory) / trajectories
        if rho is None: 
            rho = _rho
            nmqsd_rho = _nmqsd_rho
        else: 
            rho += _rho
            nmqsd_rho += _nmqsd_rho
        
    linear_rho = np.array([partial_trace(rho[i], keep=[0], dims=[2,2,2]) for i in range(len(linear_t_space))])
    linear_nmqsd_rho = np.array([partial_trace(nmqsd_rho[i], keep=[0], dims=[2,2,2]) for i in range(len(linear_t_space))])

    # Non-linear HOPS
    print("Simulating non-linear trajectories")
    rho = None
    nmqsd_rho = None
    non_linear_t_space = None
    for i in range(trajectories):
        noises = generator.generate()
        print("Non-linear", i)
        _tSpace, trajectory = hierarchy.solve_non_linear_HOPS(0.0, T, initial_state, H, noises, step_size=0.1)
        _, nmqsd_trajectory = nmqsd_driver.solve_non_linear_NMQSD(_tSpace, initial_state, H1 + H2 + H3, noises, max_step=0.1)
        if non_linear_t_space is None: non_linear_t_space = _tSpace
        _rho = hierarchy.map_non_linear_trajectory(trajectory) / trajectories
        _nmqsd_rho = nmqsd_driver.map_non_linear_trajectory(nmqsd_trajectory) / trajectories
        if rho is None: 
            rho = _rho
            nmqsd_rho = _nmqsd_rho
        else: 
            rho += _rho
            nmqsd_rho += _nmqsd_rho
    non_linear_rho = np.array([partial_trace(rho[i], keep=[0], dims=[2,2,2]) for i in range(len(non_linear_t_space))])
    non_linear_nmqsd_rho = np.array([partial_trace(nmqsd_rho[i], keep=[0], dims=[2,2,2]) for i in range(len(non_linear_t_space))])


    # Plot population and coherences 
    linear_population = linear_rho[:, 0, 0].real - linear_rho[:, 1, 1].real
    linear_coherence = 2 * linear_rho[:, 0, 1].real
    linear_coherence_imag = 2 * linear_rho[:, 0, 1].imag

    linear_nmqsd_population = linear_nmqsd_rho[:, 0, 0].real - linear_rho[:, 1, 1].real
    linear_nmqsd_coherence = 2 * linear_nmqsd_rho[:, 0, 1].real
    linear_nmqsd_coherence_imag = 2 * linear_nmqsd_rho[:, 0, 1].imag

    non_linear_population = non_linear_rho[:, 0, 0].real - non_linear_rho[:, 1, 1].real
    non_linear_coherence = 2 * non_linear_rho[:, 0, 1].real
    non_linear_coherence_imag = 2 * non_linear_rho[:, 0, 1].imag

    non_linear_nmqsd_population = non_linear_nmqsd_rho[:, 0, 0].real - non_linear_nmqsd_rho[:, 1, 1].real
    non_linear_nmqsd_coherence = 2 * non_linear_nmqsd_rho[:, 0, 1].real
    non_linear_nmqsd_coherence_imag = 2 * non_linear_nmqsd_rho[:, 0, 1].imag

    # Plot
    plt.figure()
    plt.plot(linear_t_space, linear_population, label="HOPS: <z>")
    plt.plot(linear_t_space, linear_coherence, label="HOPS: <x>")
    plt.plot(linear_t_space, linear_coherence_imag, label="HOPS: <y>")

    plt.plot(linear_t_space, linear_nmqsd_population, label="NMQSD: <z>", linestyle="--", color="black")
    plt.plot(linear_t_space, linear_nmqsd_coherence, label="NMQSD: <x>", linestyle="--", color="black")
    plt.plot(linear_t_space, linear_nmqsd_coherence_imag, label="NMQSD: <y>", linestyle="--", color="black")

    plt.title("Linear HOPS/NMQSD")
    plt.legend()
    plt.show()
    plt.close()

    plt.figure()
    plt.plot(non_linear_t_space, non_linear_population, label="HOPS: <z>")
    plt.plot(non_linear_t_space, non_linear_coherence, label="HOPS: <x>")
    plt.plot(non_linear_t_space, non_linear_coherence_imag, label="HOPS: <y>")

    plt.plot(non_linear_t_space, non_linear_nmqsd_population, label="NMQSD: <z>", linestyle="--", color="black")
    plt.plot(non_linear_t_space, non_linear_nmqsd_coherence, label="NMQSD: <x>", linestyle="--", color="black")
    plt.plot(non_linear_t_space, non_linear_nmqsd_coherence_imag, label="NMQSD: <y>", linestyle="--", color="black")

    plt.title("Non-linear HOPS/NMQSD")
    plt.legend()
    plt.show()
    plt.close()

if __name__ == "__main__":
    _main()