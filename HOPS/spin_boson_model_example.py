# This file is part of https://github.com/Turku-Quantum-Optics/hops
#
# Copyright (c) 2024, Turku Quantum Optics
# 
# Licensed under the BSD 3-Clause License, see accompanying LICENSE,
# and README.md for further information.

import numpy as np
import matplotlib.pyplot as plt

import linear_HOPS
import non_linear_HOPS
import noise_generation

class Hamiltonian:
    omega: float
    delta: float

    def __init__(self, omega: float, delta: float):
        self.omega = omega
        self.delta = delta

    def __call__(self, t, currentState):
        return np.array([[self.omega / 2, self.delta / 2],
                         [self.delta / 2, -self.omega / 2]], dtype=complex)
    
def spectralDensity(omega: np.ndarray, a: float, ksi: float) -> np.ndarray:
    return a * omega * omega * omega * np.exp(-omega * omega / (ksi * ksi))

def map_linear_trajectories_to_density_matrix(ground, excited):
    rho_t_00 = np.mean(ground * ground.conjugate(), axis=0).real
    rho_t_01 = np.mean(ground * excited.conjugate(), axis=0)
    rho_t_10 = rho_t_01.conjugate()
    rho_t_11 = np.mean(excited * excited.conjugate(), axis=0).real
    rho = np.array([np.array([[rho00, rho01], [rho10, rho11]], dtype=complex) for rho00, rho01, rho10, rho11 in zip(rho_t_00, rho_t_01, rho_t_10, rho_t_11)])
    return rho

def map_non_linear_trajectories_to_density_matrix(ground, excited):
    norms = np.sqrt(ground * ground.conjugate() + excited * excited.conjugate())
    groundNormalized = ground / norms 
    excitedNormalized = excited / norms
    return map_linear_trajectories_to_density_matrix(groundNormalized, excitedNormalized)

def linear_HOPS_independent_boson_model(trajectories: int):
    omega = 1
    delta = 1
    tSpace = np.linspace(0, 50, 10000)
    H = Hamiltonian(omega, delta)
    L = np.array([[0, 1],
                  [1, 0]], dtype=complex)
    J = lambda omega: spectralDensity(omega, 0.027, 1.447)
    generator = noise_generation.GaussianNoiseProcessFFTGenerator(0.0001, 9.0, 0.1, tSpace[-1], J)
    noises = generator.generate(trajectories)
    initialCondition = np.array([np.sqrt(0.5), np.sqrt(0.5)], dtype=complex)
    G = np.array([-0.00910165-0.03174624j, -0.00910165+0.03174624j, 0.03910214+0.0487802j, 0.03910214-0.0487802j, 
         0.00909684+0.0340994j, -0.00909684+0.0340994j, -0.01097494-0.03496594j, 0.01097494-0.03496594j], dtype=complex)
    W = np.array([1.23330001-2.72497136j, 1.23330001+2.72497136j, 1.02787572-1.33490999j, 1.02787572+1.33490999j,
        1.08606533+2.54695103j, 1.08606533-2.54695103j, 0.8075923 -1.15778801j, 0.8075923 +1.15778801j], dtype=complex)
    _, ground, excited = linear_HOPS.solve_linear_HOPS(tSpace, initialCondition, H, L, noises, G, W, kMax=5, max_step=0.05, logProgress=True)
    rho = map_linear_trajectories_to_density_matrix(ground, excited)

    population = rho[:, 0, 0] - rho[:, 1, 1]
    coherences = 2 * rho[:, 0, 1].real

    plt.figure()
    plt.plot(tSpace, population.real, label="Population")
    plt.plot(tSpace, coherences, label="Coherences")
    plt.legend()
    plt.xlabel("t")
    plt.ylabel("p/c")
    plt.show()
    plt.close()

def non_linear_HOPS_independent_boson_model(trajectories: int):
    omega = 1
    delta = 1
    tSpace = np.linspace(0, 50, 10000)
    H = Hamiltonian(omega, delta)
    L = np.array([[0, 1],
                  [1, 0]], dtype=complex)
    J = lambda omega: spectralDensity(omega, 0.027, 1.447)
    generator = noise_generation.GaussianNoiseProcessFFTGenerator(0.0001, 10.0, 0.1, tSpace[-1], J)
    noises = generator.generate(trajectories)
    initialCondition = np.array([np.sqrt(0.5), np.sqrt(0.5)], dtype=complex)
    G = np.array([-0.00910165-0.03174624j, -0.00910165+0.03174624j, 0.03910214+0.0487802j, 0.03910214-0.0487802j, 
         0.00909684+0.0340994j, -0.00909684+0.0340994j, -0.01097494-0.03496594j, 0.01097494-0.03496594j], dtype=complex)
    W = np.array([1.23330001-2.72497136j, 1.23330001+2.72497136j, 1.02787572-1.33490999j, 1.02787572+1.33490999j,
        1.08606533+2.54695103j, 1.08606533-2.54695103j, 0.8075923-1.15778801j, 0.8075923+1.15778801j], dtype=complex)
    _, ground, excited = non_linear_HOPS.solve_non_linear_HOPS(tSpace, initialCondition, H, L, noises, G, W, kMax=5, max_step=0.01, logProgress=True)
    rho = map_non_linear_trajectories_to_density_matrix(ground, excited)
    
    population = rho[:, 0, 0] - rho[:, 1, 1]
    coherences = 2 * rho[:, 0, 1].real

    plt.figure()
    plt.plot(tSpace, population.real, label="Population")
    plt.plot(tSpace, coherences, label="Coherences")
    plt.legend()
    plt.xlabel("t")
    plt.ylabel("p/c")
    plt.show()
    plt.close()

def _main():
    non_linear_HOPS_independent_boson_model(128)
    linear_HOPS_independent_boson_model(128)

if __name__ == "__main__":
    _main()