# This file is part of https://github.com/Turku-Quantum-Optics/hops
#
# Copyright (c) 2024, Turku Quantum Optics
# 
# Licensed under the BSD 3-Clause License, see accompanying LICENSE,
# and README.md for further information.

import numpy as np
import scipy as sp
from scipy.integrate import RK45

from typing import Callable

class HeunSDESolver:
    """This SDE solver uses the Heun method for a Stratonovich SDE"""
    y: np.ndarray
    y_old: np.ndarray = None

    tIndex: int = 0
    t_old: float = None

    status: str = "running"

    # Drift
    alpha: Callable
    
    # Noise / diffusion
    beta: Callable

    # Wiener process
    dW: Callable

    tSpace: np.ndarray
    initialCondition: np.ndarray
    dt: float

    def __init__(self, tSpace: np.ndarray, initialCondition: np.ndarray, drift: Callable, noise: Callable, dW: Callable):
        self.tSpace = tSpace
        self.initialCondition = initialCondition
        self.alpha = drift
        self.beta = noise
        self.dW = dW

        # Pre sample
        if hasattr(dW, "sample"):
            _ = dW.sample(tSpace)

        self.y = initialCondition
        self.dt = tSpace[1] - tSpace[0]
    
    def step(self):
        if self.status == "finished":
            return
        t = self.t
        dt = self.dt
        self.tIndex += 1
        self.y_old = self.y
        self.t_old = t
        self.y = _heunMethodStartonovichStep(t, dt, self.y, self.alpha, self.beta, self.dW)
        if self.tIndex >= self.tSpace.size: 
            self.status = "finished"

    @property
    def t(self):
        if self.tIndex >= self.tSpace.size:
            return self.tSpace[-1]
        return self.tSpace[self.tIndex]
    
    @property
    def step_size(self):
        return self.t - self.t_old

class SDESolver:
    """This SDE solver uses a 3/2 order Runge-Kutta method which is only applicable to additive noise!"""
    y: np.ndarray
    y_old: np.ndarray = None

    tIndex: int = 0
    t_old: float = None

    status: str = "running"

    # Drift
    alpha: Callable
    
    # Noise / diffusion
    beta: Callable

    # Wiener process
    dW: Callable

    tSpace: np.ndarray
    initialCondition: np.ndarray
    dt: float

    def __init__(self, tSpace: np.ndarray, initialCondition: np.ndarray, drift: Callable, noise: Callable, dW: Callable):
        self.tSpace = tSpace
        self.initialCondition = initialCondition
        self.alpha = drift
        self.beta = noise
        self.dW = dW

        # Pre sample
        if hasattr(dW, "sample"):
            _ = dW.sample(tSpace)

        self.y = initialCondition
        self.dt = tSpace[1] - tSpace[0]
    
    def step(self):
        if self.status == "finished":
            return
        t = self.t
        dt = self.dt
        self.tIndex += 1
        self.y_old = self.y
        self.t_old = t
        self.y = _rungeKutta32OrderAdditiveNoiseItoStep(t, dt, self.y, self.alpha, self.beta, self.dW)
        if self.tIndex >= self.tSpace.size: 
            self.status = "finished"

    @property
    def t(self):
        if self.tIndex >= self.tSpace.size:
            return self.tSpace[-1]
        return self.tSpace[self.tIndex]
    
    @property
    def step_size(self):
        return self.t - self.t_old

class RK4SmallNoiseSDESolver:
    """This SDE solver (Ito) 4th order Runge-Kutta method for the drift but the noise is assumed to be small!"""
    y: np.ndarray
    y_old: np.ndarray = None

    tIndex: int = 0
    t_old: float = None

    status: str = "running"

    # Drift
    alpha: Callable
    
    # Noise / diffusion
    beta: Callable

    # Wiener process
    dW: Callable

    tSpace: np.ndarray
    initialCondition: np.ndarray
    dt: float

    def __init__(self, tSpace: np.ndarray, initialCondition: np.ndarray, drift: Callable, diffusion: Callable, dW: Callable):
        self.tSpace = tSpace
        self.initialCondition = initialCondition
        self.alpha = drift
        self.beta = diffusion
        self.dW = dW

        # Pre sample
        if hasattr(dW, "sample"):
            _ = dW.sample(tSpace)

        self.y = initialCondition
        self.dt = tSpace[1] - tSpace[0]
    
    def step(self):
        if self.status == "finished":
            return
        t = self.t
        y = self.y
        dt = self.dt
        self.tIndex += 1
        self.y_old = self.y
        self.t_old = t

        # RK4 step
        k1 = dt * self.alpha(t, y)
        k2 = dt * self.alpha(t + dt / 2, y + k1 / 2)
        k3 = dt * self.alpha(t + dt / 2, y + k2 / 2)
        k4 = dt * self.alpha(t + dt, y + k3)
        drift = (k1 + 2 * k2 + 2 * k3 + k4) / 6

        # Diffusion step. Note the noise is assumed to be very small
        diffusion = self.beta(t, y)

        self.y = y + drift + diffusion * np.sqrt(dt) * self.dW(t)

        if self.tIndex >= self.tSpace.size: 
            self.status = "finished"

    @property
    def t(self):
        if self.tIndex >= self.tSpace.size:
            return self.tSpace[-1]
        return self.tSpace[self.tIndex]
    
    @property
    def step_size(self):
        return self.t - self.t_old

class WeakSecondOrderAutonomousSDESolver:
    """This SDE solver (Ito) solves autonomous SDE in second order."""
    y: np.ndarray
    y_old: np.ndarray = None

    tIndex: int = 0
    t_old: float = None

    status: str = "running"

    # Drift
    alpha: Callable
    
    # Noise / diffusion
    beta: Callable

    # Wiener process
    dW: Callable

    tSpace: np.ndarray
    initialCondition: np.ndarray
    dt: float

    def __init__(self, tSpace: np.ndarray, initialCondition: np.ndarray, drift: Callable, diffusion: Callable, dW: Callable):
        self.tSpace = tSpace
        self.initialCondition = initialCondition
        self.alpha = drift
        self.beta = diffusion
        self.dW = dW

        # Pre sample
        if hasattr(dW, "sample"):
            _ = dW.sample(tSpace)

        self.y = initialCondition
        self.dt = tSpace[1] - tSpace[0]
    
    def step(self):
        if self.status == "finished":
            return
        t = self.t
        dt = self.dt
        self.tIndex += 1
        self.y_old = self.y
        self.t_old = t
        self.y = _weakSecondOrderAutonomousStep(t, dt, self.y, self.alpha, self.beta, self.dW)
        if self.tIndex >= self.tSpace.size: 
            self.status = "finished"

    @property
    def t(self):
        if self.tIndex >= self.tSpace.size:
            return self.tSpace[-1]
        return self.tSpace[self.tIndex]
    
    @property
    def step_size(self):
        return self.t - self.t_old

# Peter E. Kloeden, Eckhard Platen - Numerical solution of stochastic differential equations, p.485 eq. (15.1.1).
def _weakSecondOrderAutonomousStep(t: float, dt: float, y: np.ndarray, alpha: Callable, beta: Callable, dW: Callable):
    dtSqrt = np.sqrt(dt)
    dtSqrtInv = 1 / dtSqrt
    _dW = dtSqrt * dW(t)

    a = alpha(t, y)
    b = beta(t, y)

    yAux = y + a * dt + b * _dW
    yP = y + a * dt + b * dtSqrt
    yM = y + a * dt - b * dtSqrt

    aAux = alpha(t, yAux)
    bP = beta(t, yP)
    bM = beta(t, yM)

    return y + 0.5 * (aAux + a) * dt \
             + 0.25 * (bP + bM + 2 * b) * _dW \
             + 0.25 * (bP - bM) * (_dW * _dW - dt) * dtSqrtInv

# Peter E. Kloeden, Eckhard Platen - Numerical solution of stochastic differential equations, p.485 eq. (15.1.1).
def weakSecondOrderAutonomous(tSpace: np.ndarray, y0: np.ndarray, alpha: Callable, beta: Callable, wienerProcess: Callable):
    """Solves an Ito SDE of the form dy = \alpha(t, y)dt + \beta(t, y)dW where alpha and beta are autonomous, i.e. depends only on y"""
    result = [y0]
    dt = tSpace[1] - tSpace[0]
    for t in tSpace[1:]:
        y = result[-1]
        result.append(_weakSecondOrderAutonomousStep(t, dt, y, alpha, beta, wienerProcess))
    result = np.array(result)
    return tSpace, result

def _heunMethodStartonovichStep(t: float, dt: float, y: np.ndarray, alpha: Callable, beta: Callable, dW: Callable):
    aty = alpha(t, y)
    bty = beta(t, y)
    dtSqrt = np.sqrt(dt)
    xi = dtSqrt * dW(t)
    yBar = y + aty * dt + bty * xi
    yNew = y + 0.5 * (aty + alpha(t + dt, yBar)) * dt + 0.5 * (bty + beta(t + dt, yBar)) * xi
    return yNew

def heunMethodStratonovich(tSpace: np.ndarray, y0: np.ndarray, alpha: Callable, beta: Callable, wienerProcess: Callable):
    """Solves an Stratonovich SDE of the form dy = \alpha(t, y) dt + \beta(t, y)dW using Heun's method"""
    result = [y0]
    y = y0
    dt = tSpace[1] - tSpace[0]
    for t in tSpace[1:]:
        y = _heunMethodStartonovichStep(t, dt, y, alpha, beta, wienerProcess)
        result.append(y)
    result = np.array(result)
    return tSpace, result

def rk45EulerMaruyamaIto(tSpace: np.ndarray, y0: np.ndarray, alpha: Callable, beta: Callable, wienerProcess: Callable):
    #TODO: Doesn't give proper results...
    dt = tSpace[1] - tSpace[0]
    rk = RK45(alpha, tSpace[0], y0, tSpace[-1], first_step=dt, max_step=dt)
    _tSpace = []
    _solutions = []
    status = None
    t = 0.0
    dtSqrt = np.sqrt(dt)
    while status is None:
        rk.step()
        if rk.status == 'finished':
            status = 0
        elif rk.status == 'failed':
            status = -1
            break
        if rk.step_size == 0: continue
        rk.y += beta(rk.t_old, rk.y_old) * dtSqrt * wienerProcess(rk.t_old)
        while t < rk.t:
            t0 = rk.t_old
            y0 = rk.y_old
            y1 = rk.y
            y = y0 + (t - t0) * (y1 - y0) / rk.step_size
            _tSpace.append(t)
            _solutions.append(y)
            t += 0.005
    
    if status == -1:
        print("Simulation failed!")
        return None
    
    _tSpace.append(rk.t)
    _solutions.append(rk.y)

    _tSpace = np.array(_tSpace)
    _tSpace, indices = np.unique(_tSpace, return_index=True)
    _solutions = np.array(_solutions)[indices]
    _solutions = sp.interpolate.CubicSpline(_tSpace, _solutions)(tSpace)
    return tSpace, _solutions

def _eulerMaruyamaItoStep(t: float, dt: float, y: np.ndarray, alpha: Callable, beta: Callable, dW: Callable):
    dtSqrt = np.sqrt(dt)
    yNew = y + alpha(t, y) * dt + beta(t, y) * dtSqrt * dW(t)
    return yNew

def eulerMaruyamaIto(tSpace: np.ndarray, y0: np.ndarray, alpha: Callable, beta: Callable, wienerProcess: Callable):
    """Solves an Ito SDE of the form dy = \alpha(t, y)dt + \beta(t, y)dW using the Euler-Maruyama method"""
    result = [y0]
    y = y0
    dt = tSpace[1] - tSpace[0]
    for t in tSpace[1:]:
        y = _eulerMaruyamaItoStep(t, dt, y, alpha, beta, wienerProcess)
        result.append(y)
    result = np.array(result)
    return tSpace, result

def _rungeKuttaFirstOrderItoStep(t: float, dt: float, y: np.ndarray, alpha: Callable, beta: Callable, dW: Callable):
    dtSqrt = np.sqrt(dt)
    a = alpha(t, y)
    b = beta(t, y)
    _dW = dtSqrt * dW(t)
    tNew = t + dt
    d = y + a * dt + b * dtSqrt
    yNew = y + a * dt + b * _dW + 1 / (2 * dtSqrt) * (beta(tNew, d) - b) * (_dW * _dW - dt)
    return yNew

def rungeKuttaFirstOrderIto(tSpace: np.ndarray, y0: np.ndarray, alpha: Callable, beta: Callable, wienerProcess: Callable):
    """Solves an Ito SDE of the form dy = \alpha(t, y)dt + \beta(t, y)dW using the first order explicit stochastic Runge-Kutta method"""
    result = [y0]
    dt = tSpace[1] - tSpace[0]
    for t in tSpace[1:]:
        y = result[-1]
        result.append(_rungeKuttaFirstOrderItoStep(t, dt, y, alpha, beta, wienerProcess))
    result = np.array(result)
    return tSpace, result

# Peter E. Kloeden, Eckhard Platen - Numerical solution of stochastic differential equations, p.383 eq. (11.2.19). z1 (U1) and z2 (U2) defined in p. 352 eq. (10.4.2)
def _rungeKutta32OrderAdditiveNoiseItoStep(t: float, dt: float, y: np.ndarray, alpha: Callable, beta: Callable, dW: Callable):
    dtSqrt = np.sqrt(dt)
    dt32 = dt * dtSqrt
    tNew = t + dt
    #tOld = t - dt

    z1 = dW(t)
    _dW = dtSqrt * z1
    z2 = np.random.default_rng().normal(0, 1, size=2).view(dtype=complex)
    I10 = dt32 / 2 * (z1 + z2 / np.sqrt(3))

    #derivativeStart = time.time()
    a = alpha(t, y)
    b = beta(t, y)
    #derivativeTime = time.time() - derivativeStart

    d1Plus = y + a * dt + b * dtSqrt
    d1Minus = y + a * dt - b * dtSqrt

    #secondDerivativeStart = time.time()
    a_tNew_d1Plus = alpha(tNew, d1Plus)
    a_tNew_d1Minus = alpha(tNew, d1Minus)
    b_tNew = beta(tNew, y)
    #secondDerivativeTime = time.time() - secondDerivativeStart

    #arithmeticStart = time.time()
    yNew = y + a * dt + b * _dW \
            + 1 / 4 * (a_tNew_d1Plus - 2 * a + a_tNew_d1Minus) * dt \
            + 1 / (2 * dtSqrt) * (a_tNew_d1Plus - a_tNew_d1Minus) * I10 \
            + 1 / dt * (b_tNew - b) * (_dW * dt - I10)
    #arithmeticTime = time.time() - arithmeticStart
    #print(derivativeTime, secondDerivativeTime, arithmeticTime)
    return yNew
# Peter E. Kloeden, Eckhard Platen - Numerical solution of stochastic differential equations, p.383 eq. (11.2.19)
def rungeKutta32AdditiveNoiseOrderIto(tSpace: np.ndarray, y0: np.ndarray, alpha: Callable, beta: Callable, wienerProcess: Callable):
    """Solves an Ito SDE of the form dy = \alpha(t, y)dt + \beta(t, y)dW, for additive noise aka. \beta' = 0, using the 3/2 order explicit stochastic Runge-Kutta method"""
    result = [y0]
    dt = tSpace[1] - tSpace[0]
    for t in tSpace[1:]:
        y = result[-1]
        result.append(_rungeKutta32OrderAdditiveNoiseItoStep(t, dt, y, alpha, beta, wienerProcess))
    result = np.array(result)
    return tSpace, result

def _rungeKutta32OrderItoStep(t: float, dt: float, y: np.ndarray, alpha: Callable, beta: Callable, dW: Callable):
    dtSqrt = np.sqrt(dt)
    dt32 = dt * dtSqrt
    tNew = t + dt
    tOld = t - dt
    z1 = dW(t)
    _dW = dtSqrt * z1

    a = alpha(t, y)
    b = beta(t, y)

    z2 = np.random.normal(0, 1)
    I10 = dt32 / 2 * (z1 + z2 / np.sqrt(3))

    d1Plus = y + a * dt + b * dtSqrt
    d1Minus = y + a * dt - b * dtSqrt

    a_tNew_d1Plus = alpha(tNew, d1Plus)
    a_tOld_d1Minus = alpha(tOld, d1Minus)
    b_tNew_d1Plus = beta(tNew, d1Plus)
    b_tOld_d1Minus = beta(tOld, d1Minus)

    d2Plus = d1Plus + b_tNew_d1Plus * dtSqrt
    d2Minus = d1Plus - b_tNew_d1Plus * dtSqrt

    yNew = y + b * _dW + (1 / (2 * dtSqrt)) * (a_tNew_d1Plus - a_tOld_d1Minus) * I10 \
            + 1 / 4 * (a_tNew_d1Plus + 2 * a + a_tOld_d1Minus) * dt \
            + 1 / (4 * dtSqrt) * (b_tNew_d1Plus + b_tOld_d1Minus) * (_dW * _dW - dt) \
            + 1 / (2 * dt) * (b_tNew_d1Plus - 2 * b + b_tOld_d1Minus) * (_dW * dt - I10) \
            + 1 / (4 * dt) * (beta(tNew, d2Plus) - beta(tOld, d2Minus) - b_tNew_d1Plus + b_tOld_d1Minus) * ((_dW * _dW) / 3 - dt) * _dW
    return yNew

#TODO: Doesn't work at the moment
def rungeKutta32OrderIto(tSpace: np.ndarray, y0: np.ndarray, alpha: Callable, beta: Callable, wienerProcess: Callable):
    """Solves an Ito SDE of the form dy = \alpha(t, y)dt + \beta(t, y)dW using the 3/2 order explicit stochastic Runge-Kutta method"""
    result = [y0]
    dt = tSpace[1] - tSpace[0]
    for t in tSpace[1:]:
        y = result[-1]
        result.append(_rungeKutta32OrderItoStep(t, dt, y, alpha, beta, wienerProcess))
    result = np.array(result)
    return tSpace, result