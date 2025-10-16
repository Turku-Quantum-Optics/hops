# This file is part of https://github.com/Turku-Quantum-Optics/hops
#
# Copyright (c) 2024-2025, Turku Quantum Optics
# 
# Licensed under the BSD 3-Clause License, see accompanying LICENSE,
# and README.md for further information.

import numpy as np
import scipy as sp

from . import math_utils

from typing import Callable

def normal_distribution(mu: float, sigma: float, size: int = None) -> np.ndarray:
    return np.random.default_rng().normal(mu, sigma, size * 2).view(dtype=complex)

def bose_einstein(omega: float, temperature: float) -> float:
    if temperature == 0 or omega == 0: return 0
    beta = 1 / temperature
    return 1 / (np.exp(beta * omega) - 1)

def bose_einstein_array(omegaSpace, temperature: float):
    if temperature == 0: return np.zeros(omegaSpace.shape)
    if np.all(omegaSpace == 0): return np.zeros(omegaSpace.shape)
    beta = 1 / temperature
    return 1 / (np.exp(beta * omegaSpace) - 1)

def safe_eigh(J_all, tol=1e-14):
    # Symmetrize all matrices (ensure Hermitian)
    J_all = (J_all + np.swapaxes(J_all.conj(), -1, -2)) / 2

    # Compute norms (Frobenius norm along last two axes)
    norms = np.linalg.norm(J_all, axis=(1,2))

    # Identify "small" matrices
    mask_small = norms < tol

    # Allocate output arrays
    M, N, _ = J_all.shape
    eigen_values = np.zeros((M, N), dtype=float)
    eigen_vectors = np.zeros((M, N, N), dtype=complex)

    # Default eigenvectors = identity
    eigen_vectors[:] = np.eye(N, dtype=complex)

    # Process only non-small matrices
    idx = np.where(~mask_small)[0]
    if len(idx) > 0:
        evals, evecs = np.linalg.eigh(J_all[idx])
        eigen_values[idx] = np.clip(evals, 0, None)  # clip small negatives
        eigen_vectors[idx] = evecs

    return eigen_values, eigen_vectors

class NoiseProcess(Callable[[float], complex]):
    def __call__(self, t):
        NotImplementedError("You must implement __call__ method in subclass")

    def consuming_call(self, t: float):
        return self(t)

    def sample(self, tSpace: np.ndarray):
        NotImplementedError("You must implement sample method in subclass")

    def consuming_sample(self, tSpace: np.ndarray):
        return self.sample(tSpace)

    def create_spline(self, tSpace: np.ndarray) -> Callable:
        t_spaces = np.array_split(tSpace, 100)
        samples = np.array([], dtype=complex)
        for t_space in t_spaces:
            samples = np.concatenate((samples, self.sample(t_space)), dtype=complex)
        return sp.interpolate.CubicSpline(tSpace, samples)

class NoiseProcessGenerator():
    def __call__(self) -> NoiseProcess:
        raise NotImplementedError("Noise generator subclasses need to implement __call__ to generate new processes")

    def _generator(self, index):
        np.random.RandomState()
        return self()

    def generate(self, realisations: int) -> list[NoiseProcess]:
        return [self() for _ in range(realisations)]

class GaussianNoiseProcessFFT(NoiseProcess):
    def __init__(self, t_max: float, spectral_density: Callable[[np.ndarray], np.ndarray], dt_max: float = 0.1, delta_omega_max: float = 0.01, omega_max: float = None, spline = None, t_space = None):
        if spline is not None and t_space is not None:
            self.t_space = t_space
            self.spline = spline
            return
        assert(t_max > 0)
        assert(dt_max > 0)
        assert(delta_omega_max > 0)

        # 1. Set frequency resolution based on tMax
        delta_omega = min(delta_omega_max, np.pi / t_max)

        # 2. Compute minimum N so that dt <= dtMax and optionally omegaMax is covered
        N = 1024 if omega_max is None else math_utils.next_power_of_two(int(omega_max / delta_omega))
        N = max(1024, N)
        dt = 2 * np.pi / (N * delta_omega)
        while dt > dt_max:
            N <<= 1
            dt = 2 * np.pi / (N * delta_omega)

        omega_max = (N - 1) * delta_omega
        omega_space = np.linspace(0, omega_max, N)

        # 3. Generate complex Gaussian coefficients
        random_coefficients = normal_distribution(mu=0, sigma=np.sqrt(1/2), size=N)

        # 4. Evaluate J(omega) and build sqrt(J * deltaOmega) * xi
        J = spectral_density(omega_space)
        coefficients = np.sqrt(J * delta_omega) * random_coefficients

        # 5. FFT
        noise = sp.fft.fft(coefficients)

        # 6. Create valid time grid
        t_max_fft = 2 * np.pi / delta_omega
        t_space = np.linspace(0, t_max_fft, N)

        # 7. Trim noise up to desired tMax
        valid_indices = t_space <= t_max
        trimmed_time = t_space[valid_indices]
        trimmed_noise = noise[valid_indices]

        self.spline = sp.interpolate.CubicSpline(trimmed_time, trimmed_noise)

    def __call__(self, t: float) -> complex:
        return self.spline(t)

    def sample(self, t: np.ndarray) -> np.ndarray:
        return self.spline(t)
    
    def conjugate(self):
        samples = self.spline(self.t_space).conjugate()
        spline = sp.interpolate.CubicSpline(self.t_space, samples)
        process = GaussianNoiseProcessFFT(self.t_space[-1], lambda _: 0, spline=spline, t_space=self.t_space)
        return process

class GaussianNoiseProcessFFTGenerator(NoiseProcessGenerator):
    t_max: float
    dt_max: float
    delta_omega_max: float
    omega_max: float
    spectral_density: Callable[[np.ndarray], np.ndarray]
    
    def __init__(self, t_max, spectral_density: Callable[[np.ndarray], np.ndarray], dt_max: float = 0.1, delta_omega_max: float = 0.01, omega_max: float = None):
        self.t_max = t_max
        self.spectral_density = spectral_density
        self.dt_max = dt_max
        self.delta_omega_max = delta_omega_max
        self.omega_max = omega_max

    def __call__(self) -> GaussianNoiseProcessFFT:
        return GaussianNoiseProcessFFT(t_max=self.t_max, spectral_density=self.spectral_density, dt_max=self.dt_max, delta_omega_max=self.delta_omega_max, omega_max=self.omega_max)

class MultiNoiseProcessGenerator():
    def __call__(self) -> NoiseProcess:
        raise NotImplementedError("Noise generator subclasses need to implement __call__ to generate new processes")

    def _generator(self, index):
        np.random.RandomState()
        return self()

    def generate(self) -> list[NoiseProcess]:
        return self()

class MultiGaussianNoiseProcessFFTGenerator(MultiNoiseProcessGenerator):
    t_max: float
    dt_max: float
    delta_omega_max: float
    omega_max: float
    spectral_density: Callable[[np.ndarray], list[np.ndarray]]
    
    def __init__(self, t_max, spectral_density: Callable[[np.ndarray], list[np.ndarray]], dt_max: float = 0.1, delta_omega_max: float = 0.01, omega_max: float = None):
        assert(t_max > 0)
        assert(dt_max > 0)
        assert(delta_omega_max > 0)
        self.t_max = t_max
        self.spectral_density = spectral_density
        self.dt_max = dt_max
        self.delta_omega_max = delta_omega_max
        self.omega_max = omega_max

    def __call__(self) -> list[NoiseProcess]:
        delta_omega = min(self.delta_omega_max, np.pi / self.t_max)

        N = 1024 if self.omega_max is None else math_utils.next_power_of_two(int(self.omega_max / delta_omega))
        N = max(1024, N)
        dt = 2 * np.pi / (N * delta_omega)
        while dt > self.dt_max:
            N <<= 1
            dt = 2 * np.pi / (N * delta_omega)

        omega_max = (N - 1) * delta_omega
        omega_space = np.linspace(0, omega_max, N)

        J = np.array(self.spectral_density(omega_space))

        #eigen_values, eigen_vectors = np.linalg.eigh(J)
        eigen_values, eigen_vectors = safe_eigh(J)

        D = np.sqrt(delta_omega * eigen_values)
        U = eigen_vectors

        xi = np.random.normal(0, np.sqrt(1/2), size=(N, J[0].shape[0])) \
            + 1j * np.random.normal(0, np.sqrt(1/2), size=(N, J[0].shape[0]))

        A = D * xi

        # Equivalent of U.dot(sqrt(D)).dot(xi) for each batch
        coefficients = np.einsum("nij,nj->ni", U, A)

        noise = sp.fft.fft(coefficients, axis=0)

        t_max_fft = 2 * np.pi / delta_omega
        t_space = np.linspace(0, t_max_fft, N)

        valid_indices = t_space <= self.t_max
        trimmed_time = t_space[valid_indices]
        trimmed_noise = noise[valid_indices]
        
        t_space = trimmed_time

        noises = []
        for i in range(noise.shape[1]):
            spline = sp.interpolate.CubicSpline(t_space, trimmed_noise[:,i])
            z = GaussianNoiseProcessFFT(t_max=self.t_max, spectral_density=self.spectral_density, spline=spline, t_space=t_space)
            noises.append(z)
        return noises

class GaussianWhiteNoiseProcess(NoiseProcess):
    mu: float
    deviation: float
    samples: dict
    generator: np.random.Generator

    def __init__(self, mu, deviation):
        self.mu = mu
        self.deviation = np.sqrt(deviation)
        self.samples = dict()
        self.generator = np.random.default_rng()

    def __call__(self, t: float) -> complex:
        if t in self.samples:
            return self.samples[t]
        sample = normal_distribution(self.mu, self.deviation, 1)
        self.samples[t] = sample
        return sample
    
    def sample(self, t: np.ndarray) -> np.ndarray:
        _samples = normal_distribution(self.mu, self.deviation, t.shape[0])
        for i, _t in enumerate(t):
            if _t not in self.samples:
                self.samples[_t] = _samples[i]
        return np.array([self(_t) for _t in t], dtype=complex)
    
    def createSpline(self, tSpace):
        NotImplementedError("Cannot construct a spline for a white noise process.")

class GaussianWhiteNoiseProcessGenerator(NoiseProcessGenerator):
    mu: float
    deviation: float

    def __init__(self, mu, deviation):
        self.mu = mu
        self.deviation = deviation

    def __call__(self) -> GaussianWhiteNoiseProcess:
        return GaussianWhiteNoiseProcess(self.mu, self.deviation)
    
    def generate(self, realisations: int, parallelism: int | None = None):
        return [self() for _ in range(realisations)]
    
class ZeroNoise(NoiseProcess):
    def __call__(self, t):
        return 0.0
    
    def sample(self, t: np.ndarray) -> np.ndarray:
        return np.array([0.0 for _ in t], dtype=complex)
    
    def spline(self, tSpace: np.ndarray):
        return self

class ZeroNoiseProcessGenerator(NoiseProcessGenerator):
    def __call__(self) -> ZeroNoise:
        return ZeroNoise()