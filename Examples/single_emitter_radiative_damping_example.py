# This file is part of https://github.com/Turku-Quantum-Optics/hops
#
# Copyright (c) 2024-2025, Turku Quantum Optics
# 
# Licensed under the BSD 3-Clause License, see accompanying LICENSE,
# and README.md for further information.

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad, solve_ivp

import HOPS

class Hamiltonian:
    def __init__(self, omega: float):
        self.H = np.array([[0, 0.175 / 2],
                           [0.175 / 2, omega]], dtype=complex)

    def __call__(self, t, current_state):
        return self.H
    
class SpectralDensity:
    def __init__(self, a: float, ksi: float):
        self.a = a
        self.ksi = ksi

    def __call__(self, omega):
        return self.a * np.power(omega, 3) * np.exp(-np.power(omega / self.ksi, 2))    

def BCF_array(tau: np.ndarray, J, wmax=np.inf):
    result = np.zeros_like(tau, dtype=complex)
    for i, tau in enumerate(tau):
        integrand = lambda w: J(w) * np.exp(-1j * w * tau)
        result[i] = quad(lambda w: integrand(w), 0, wmax, limit=500, complex_func=True)[0]
    return np.array(result)

class SigmaMinus:
    def __init__(self):
        self.O = np.array([[0, 1], [0, 0]], dtype=complex)

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
    trajectories = 1024
    # Hamiltonian parameters

    # Total simulation time
    T = 50.0

    # Hamiltonian
    omegaX = 0.1 # Exciton transition frequency
    H = Hamiltonian(omegaX)

    # Spectral density of the phonon environment
    a = 0.027 # Coupling strength
    ksi = 1.447 # Cut-off frequency
    J = SpectralDensity(a, ksi)

    # Environment coupling operator (pure dephasing)
    L = np.array([[0, 0],
                  [0, 1]], dtype=complex)
    gamma = 0.0175

    # Find the exponential sum form of the BCF
    tau = np.linspace(0, 10, 501)
    bcf = BCF_array(tau, J)
    #G, W = HOPS.fit_BCF_tPFD(tau, bcf, Kr=4, Ki=4)
    # The matrix pencil method gives better results in most cases
    G, W = HOPS.fit_BCF_matrix_pencil(bcf, tau[1] - tau[0], K=3)

    # Initial condition
    initial_condition = np.array([np.sqrt(0.5), np.sqrt(0.5)], dtype=complex)
    
    # Generate noise processes
    print("Generating noises")
    generator = HOPS.GaussianNoiseProcessFFTGenerator(T, J, dt_max=0.01)
    white_noise_generator = HOPS.GaussianWhiteNoiseProcessGenerator(0, gamma / 2)
    noises = generator.generate(trajectories)
    white_noises = white_noise_generator.generate(trajectories)

    # Construct the hierarchy
    hierarchy = HOPS.SingleParticleHierarchy(2, L, G, W, 4)

    # Calculate linear trajectories
    print("Simulating linear HOPS")
    linear_rho = None
    linear_t_space = None
    custom_operator = CustomOperator(gamma, SigmaMinus().O, normalize=False)
    for i, (noise, white_noise) in enumerate(zip(noises, white_noises)):
        print(f"Linear {i}")
        _tSpace, trajectory = hierarchy.solve_linear_HOPS(0, T, initial_condition, H, noise, custom_operators=[custom_operator], diffusion_operators=[SigmaMinus()], white_noises=[white_noise], step_size=0.1)
        if linear_t_space is None:
            linear_t_space = _tSpace
        if linear_rho is None:
            linear_rho = hierarchy.map_linear_trajectory(trajectory) / trajectories
        else:
            linear_rho += hierarchy.map_linear_trajectory(trajectory) / trajectories
    
    # Calculate non-linear trajectories
    print("Simulating non-linear HOPS")
    non_linear_rho = None
    non_linear_t_space = None
    custom_operator = CustomOperator(gamma, SigmaMinus().O, normalize=True)
    for i, (noise, white_noise) in enumerate(zip(noises, white_noises)):
        print(f"Non-linear {i}")
        _tSpace, trajectory = hierarchy.solve_non_linear_HOPS(0, T, initial_condition, H, noise, custom_operators=[custom_operator], diffusion_operators=[SigmaMinus()], white_noises=[white_noise], step_size=0.1)
        if non_linear_t_space is None:
            non_linear_t_space = _tSpace
        if non_linear_rho is None:
            non_linear_rho = hierarchy.map_non_linear_trajectory(trajectory) / trajectories
        else:
            non_linear_rho += hierarchy.map_non_linear_trajectory(trajectory) / trajectories
    
    master_equation_t_space, master_equation_rho = master_equation_solution(T, np.outer(initial_condition, initial_condition), H, gamma, SigmaMinus().O)

    master_equation_population = master_equation_rho[:, 0, 0] - master_equation_rho[:, 1, 1]
    master_equation_coherence = 2 * master_equation_rho[:, 0, 1].real
    master_equation_coherence_imag = 2 * master_equation_rho[:, 0, 1].imag

    # Plot population and coherences 
    linear_population = linear_rho[:, 0, 0] - linear_rho[:, 1, 1]
    linear_coherence = 2 * linear_rho[:, 0, 1].real
    linear_coherence_imag = 2 * linear_rho[:, 0, 1].imag

    non_linear_population = non_linear_rho[:, 0, 0] - non_linear_rho[:, 1, 1]
    non_linear_coherence = 2 * non_linear_rho[:, 0, 1].real
    non_linear_coherence_imag = 2 * non_linear_rho[:, 0, 1].imag

    plt.figure()

    plt.plot(master_equation_t_space, master_equation_population.real, label="Population (ME)")
    plt.plot(master_equation_t_space, master_equation_coherence, label="Coherences (ME)")
    plt.plot(master_equation_t_space, master_equation_coherence_imag, label="Coherences imag (ME)")
    
    plt.plot(linear_t_space, linear_population.real, label="Population (linear)")
    plt.plot(linear_t_space, linear_coherence, label="Coherences (linear)")
    plt.plot(linear_t_space, linear_coherence_imag, label="Coherences imag (linear)")

    plt.plot(non_linear_t_space, non_linear_population.real, label="Population (non-linear)", linestyle="--")
    plt.plot(non_linear_t_space, non_linear_coherence, label="Coherences (non-linear)", linestyle="--")
    plt.plot(non_linear_t_space, non_linear_coherence_imag, label="Coherences imag (non-linear)", linestyle="--")
    
    plt.title("HOPS vs. ME")
    plt.legend()
    plt.xlabel("t")
    plt.ylabel("p/c")
    plt.show()
    plt.close()

if __name__ == "__main__":
    _main()