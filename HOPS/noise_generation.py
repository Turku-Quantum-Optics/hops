import numpy as np
import scipy as sp

from typing import Callable
from multiprocessing import Pool

def normalDistribution(mu: float, sigma: float, size: int = None) -> np.ndarray:
    return np.random.default_rng().normal(mu, sigma, size * 2).view(dtype=complex)

def boseEinstein(omega: float, temperature: float) -> float:
    if temperature == 0 or omega == 0: return 0
    beta = 1 / temperature
    return 1 / (np.exp(beta * omega) - 1)

def boseEinsteinArray(omegaSpace, temperature: float):
    if temperature == 0: return np.zeros(omegaSpace.shape)
    if np.all(omegaSpace == 0): return np.zeros(omegaSpace.shape)
    beta = 1 / temperature
    return 1 / (np.exp(beta * omegaSpace) - 1)

class NoiseProcess(Callable[[float], complex]):
    def __call__(self, t):
        NotImplementedError("You must implement __call__ method in subclass")

    def consumingCall(self, t: float):
        return self(t)

    def sample(self, tSpace: np.ndarray):
        NotImplementedError("You must implement sample method in subclass")

    def consumingSample(self, tSpace: np.ndarray):
        return self.sample(tSpace)

    def createSpline(self, tSpace: np.ndarray) -> Callable:
        tSpaces = np.array_split(tSpace, 100)
        samples = np.array([], dtype=complex)
        for _tSpace in tSpaces:
            samples = np.concatenate((samples, self.sample(_tSpace)), dtype=complex)
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
    tSpace: np.ndarray
    tMax: float

    def __init__(self, deltaOmega: float, omegaMax: float, dt: float, tMax: float, spectralDensity: Callable[[np.ndarray], np.ndarray], spline = None, tSpace = None):
        assert(tMax < np.pi / deltaOmega)
        self.tMax = tMax

        if spline is not None and tSpace is not None:
            self.tSpace = tSpace
            self.spline = spline
            return

        # 1. Determine omegaMax
        omegaMax = max(omegaMax, 2 * np.pi / dt)

        # 2. Calculate N (i.e. number of samples -> number of coefficients)
        N = int(omegaMax / deltaOmega)

        # 3. Calculate tSpace and omegaSpace
        omegaSpace = np.linspace(0, omegaMax, N)
        tSpace = np.linspace(0, np.pi / deltaOmega, N)

        # 4. Generate N normally distributed coefficients
        randomCoefficients = normalDistribution(mu=0, sigma=np.sqrt(1/2), size=N)

        # 5. Pre caculate coefficients
        sqrtSpectralDensity = np.sqrt(deltaOmega * spectralDensity(omegaSpace))
        coefficients = randomCoefficients * sqrtSpectralDensity

        # 6. Compute FFT
        noise_fft = sp.fft.fft(coefficients)[0:N//2]

        # 7. Create spline
        _tMax = tSpace[-1]
        _tSpace = np.linspace(0, _tMax, N//2, endpoint=True)
        indices = _tSpace <= tMax
        self.tSpace = _tSpace[indices]
        noise_fft = noise_fft[indices]
        self.spline =  sp.interpolate.CubicSpline(self.tSpace, noise_fft)

    def __call__(self, t: float) -> complex:
        return self.spline(t)

    def sample(self, t: np.ndarray) -> np.ndarray:
        return self.spline(t)
    
    def conjugate(self):
        samples = self.spline(self.tSpace).conjugate()
        spline = sp.interpolate.CubicSpline(self.tSpace, samples)
        process = GaussianNoiseProcessFFT(np.pi / (self.tSpace[-1] + 1), self.a, self.ksi, self.tMax, spline=spline, tSpace=self.tSpace)
        return process

class GaussianNoiseProcessFFTGenerator(NoiseProcessGenerator):
    deltaOmega: float
    omegaMax: float
    dt: float
    tMax: float
    spectralDensity: Callable[[np.ndarray], np.ndarray]

    def __init__(self, deltaOmega: float, omegaMax: float, dt: float, tMax: float, spectralDensity: Callable[[np.ndarray], np.ndarray]):
        self.deltaOmega = deltaOmega
        self.omegaMax = omegaMax
        self.dt = dt
        self.tMax = tMax
        self.spectralDensity = spectralDensity

    def __call__(self) -> GaussianNoiseProcessFFT:
        return GaussianNoiseProcessFFT(self.deltaOmega, self.omegaMax, self.dt, self.tMax, self.spectralDensity)

class GaussianWhiteNoiseProcess(NoiseProcess):
    mu: float
    deviation: float
    samples: dict
    generator: np.random.Generator

    def __init__(self, mu, deviation):
        self.mu = mu
        self.deviation = np.sqrt(deviation / 2)
        self.samples = dict()
        self.generator = np.random.default_rng()

    def __call__(self, t: float) -> complex:
        if t in self.samples:
            return self.samples[t]
        sample = normalDistribution(self.mu, self.deviation, 1)
        self.samples[t] = sample
        return sample
    
    def sample(self, t: np.ndarray) -> np.ndarray:
        _samples = normalDistribution(self.mu, self.deviation, t.shape[0])
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