# This file is part of https://github.com/Turku-Quantum-Optics/hops
#
# Copyright (c) 2024-2025, Turku Quantum Optics
# 
# Licensed under the BSD 3-Clause License, see accompanying LICENSE,
# and README.md for further information.

import numpy as np
import matplotlib.pyplot as plt

import noise_generation
import HOPS

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

def _main():
    trajectories = 512
    omega = 1
    delta = 1
    T = 40
    # Hamiltonian
    H = Hamiltonian(omega, delta)
    # Environment coupling operator
    L = np.array([[0, 1],
                  [1, 1]], dtype=complex)
    # Spectral density of the phonon environment
    J = lambda omega: spectralDensity(omega, 0.027, 1.447)

    initialCondition = np.array([np.sqrt(0.5), np.sqrt(0.5)], dtype=complex)
    # Coefficients for the exponential series of the bath correlation function
    G = np.array([0.01687440926832053 + 0.03509995461639175j, 0.016874409268323128 - 0.03509995461638884j, 0.011593028595877951 - 0.03719991555698297j, 0.01159302859587723 + 0.03719991555697937j, -0.017728060308394755 + 0.05392982787285523j, 0.017728060308397728 + 0.05392982787285405j, 0.01596489084464861 - 0.05520023594463313j, -0.015964890844652396 - 0.05520023594462184j], dtype=complex)
    W = np.array([1.0876249188125862 + 2.3898514992153306j, 1.0876249188125748 - 2.38985149921527j, 0.8948770561682661 + 1.1269147511706916j, 0.8948770561681868 - 1.1269147511706867j, 1.2328133815567413 - 2.3575606242184284j, 1.2328133815567357 + 2.3575606242184333j, 1.0326864061396819 - 0.9290694964506778j, 1.0326864061395853 + 0.9290694964506567j], dtype=complex)
    
    # Generate noise processes
    generator = noise_generation.GaussianNoiseProcessFFTGenerator(0.0001, 10.0, 0.1, T, J)
    noises = generator.generate(trajectories)

    # Construct the hierarchy
    hierarchy = HOPS.SingleParticleHierarchy(2, L, G, W, 4)

    linearRho = None
    linearTSpace = None
     # Calculate the trajectories. In real simulations these would be parallelized
    for noise in noises:
        _tSpace, trajectory = hierarchy.solveLinearHOPS(0, T, initialCondition, H, noise, stepSize=0.05)
        if linearTSpace is None:
            linearTSpace = _tSpace
        if linearRho is None:
            linearRho = hierarchy.mapLinearTrajectory(trajectory) / trajectories
        else:
            linearRho += hierarchy.mapLinearTrajectory(trajectory) / trajectories
    
    nonLinearRho = None
    nonLinearTSpace = None
    # Calculate non-linear trajectories. In real simulations these would be parallelized
    for noise in noises:
        _tSpace, trajectory = hierarchy.solveNonLinearHOPS(0, T, initialCondition, H, noise, stepSize=0.05)
        if nonLinearTSpace is None:
            nonLinearTSpace = _tSpace
        if nonLinearRho is None:
            nonLinearRho = hierarchy.mapNonLinearTrajectory(trajectory) / trajectories
        else:
            nonLinearRho += hierarchy.mapNonLinearTrajectory(trajectory) / trajectories
    
    # Plot population and coherences 
    linearPopulation = linearRho[:, 0, 0] - linearRho[:, 1, 1]
    linearCoherence = 2 * linearRho[:, 0, 1].real

    nonLinearPopulation = nonLinearRho[:, 0, 0] - nonLinearRho[:, 1, 1]
    nonLinearCoherence = 2 * nonLinearRho[:, 0, 1].real

    plt.figure()
    plt.plot(linearTSpace, linearPopulation.real, label="Population (linear)")
    plt.plot(linearTSpace, linearCoherence, label="Coherences (linear)")
    plt.plot(nonLinearTSpace, nonLinearPopulation.real, label="Population (non-linear)", linestyle="--")
    plt.plot(nonLinearTSpace, nonLinearCoherence, label="Coherences (non-linear)", linestyle="--")
    plt.legend()
    plt.xlabel("t")
    plt.ylabel("p/c")
    plt.show()
    plt.close()

if __name__ == "__main__":
    _main()