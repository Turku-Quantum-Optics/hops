# This file is part of https://github.com/Turku-Quantum-Optics/hops
#
# Copyright (c) 2024-2025, Turku Quantum Optics
# 
# Licensed under the BSD 3-Clause License, see accompanying LICENSE,
# and README.md for further information.

import numpy as np
import scipy as sp

import matplotlib.pyplot as plt

import HOPS

def spectral_density(omega: np.ndarray, a: float, ksi: float) -> np.ndarray:
    return a * omega * omega * omega * np.exp(-omega * omega / (ksi * ksi))

def spectral_density_array(omega: np.ndarray, a: float, ksi: float) -> list[np.ndarray]:
    return [np.diag(np.repeat(spectral_density(w, a, ksi), 3)) for w in omega]

def BCF(spectralDensity, t, s):
    return sp.integrate.quad(lambda omega: spectralDensity(omega) * np.exp(-1j * omega * (t - s)), 0, np.inf, complex_func=True)[0]

def BCF_array(spectralDensity, t, s):
    return np.array([BCF(spectralDensity, u, s) for u in t], dtype=complex)

def test_noise_BCF(t: np.ndarray, s: float , bcf: np.ndarray, noises: list, title=""):
    mean_BCF = np.zeros(t.shape, dtype=complex)
    for z in noises:
        mean_BCF += z(t) * z(s).conjugate()
    mean_BCF /= len(noises)

    plt.figure()
    plt.plot(t, bcf.real, label="Re BCF", alpha=0.3, color="black")
    plt.plot(t, bcf.imag, label="Im BCF", alpha=0.3, color="black")
    plt.plot(t, mean_BCF.real, label="Re mBCF", linestyle="--")
    plt.plot(t, mean_BCF.imag, label="Im mBCF", linestyle="--")
    plt.legend()
    plt.title(title)
    plt.show()
    plt.close()

def _main():
    J_single = lambda omega: spectral_density(omega, 0.027, 1.447)
    J_multi = lambda omega: spectral_density_array(omega, 0.027, 1.447)

    trajectories = 5000
    T = 7.0
    t = np.linspace(0, T, 1000)
    s = 0
    bcf = BCF_array(J_single, t, s)

    print("Generating single noise")
    generator = HOPS.GaussianNoiseProcessFFTGenerator(T, J_single)
    single_noises = [generator.generate(3) for _ in range(trajectories)]
    for i in range(3):
        noises = [tupl[i] for tupl in single_noises]
        test_noise_BCF(t, s, bcf, noises, title=f"Single {i}: <z_t z_s^*> vs exact BCF")

    print("Generating multi noise (this might take a while)")
    generator = HOPS.MultiGaussianNoiseProcessFFTGenerator(T, J_multi)
    multi_noises = [generator.generate() for _ in range(trajectories)]
    for i in range(3):
        noises = [tupl[i] for tupl in multi_noises]
        test_noise_BCF(t, s, bcf, noises, title=f"Multi {i}: <z_t z_s^*> vs exact BCF")

if __name__ == "__main__":
    _main()
