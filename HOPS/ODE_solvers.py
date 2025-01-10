# This file is part of https://github.com/Turku-Quantum-Optics/hops
#
# Copyright (c) 2024, Turku Quantum Optics
# 
# Licensed under the BSD 3-Clause License, see accompanying LICENSE,
# and README.md for further information.

import numpy as np

from scipy.integrate._ivp.rk import RungeKutta, RK45, RK23, DOP853, RkDenseOutput, Dop853DenseOutput, dop853_coefficients

# Vendored copy of Scipy's RK-step function
def _rk_step(fun, t, y, f, h, A, B, C, K):
    K[0] = f
    for s, (a, c) in enumerate(zip(A[1:], C[1:]), start=1):
        dy = np.dot(K[:s].T, a[:s]) * h
        K[s] = fun(t + c * h, y + dy)

    y_new = y + h * np.dot(K[:-1].T, B)
    f_new = fun(t + h, y_new)

    K[-1] = f_new

    return y_new, f_new

# Adaptive step Runge-Kutta solvers. Basically the ones
# provided by Scipy
class AdaptiveStepRK23(RK23): ...

class AdaptiveStepRK45(RK45): ...

class AdaptiveStepDOP853(DOP853): ...

# Fixed step Runge-Kutta solvers. They give the same results
# as the Scipy Runge-Kutta solvers, with the exception
# that no error checking is performed and the time step
# stays fixed throughout the simulation.
class FixedStepRungeKutta(RungeKutta):
    def __init__(self, fun, t0, y0, t_bound, max_step,
                 rtol=1e-3, atol=1e-6, vectorized=False,
                 first_step=None, **extraneous):
        assert(max_step is not None)
        assert(max_step != np.inf)
        super().__init__(fun, t0, y0, t_bound, max_step, rtol, atol, vectorized, max_step)

    def _step_impl(self):
        t = self.t
        y = self.y
        max_step = self.max_step
        h_abs = np.abs(max_step)
        h = h_abs * self.direction
        t_new = t + h
        if self.direction * (t_new - self.t_bound) > 0:
            t_new = self.t_bound

        y_new, f_new = _rk_step(self.fun, t, y, self.f, h, self.A, self.B, self.C, self.K)
        
        self.h_previous = h
        self.y_old = y

        self.t = t_new
        self.y = y_new

        self.h_abs = h_abs
        self.f = f_new

        return True, None

    def _dense_output_impl(self):
        Q = self.K.T.dot(self.P)
        return RkDenseOutput(self.t_old, self.t, self.y_old, Q)

class FixedStepRK23(FixedStepRungeKutta):
    order = 3
    error_estimator_order = 2
    n_stages = 3
    C = np.array([0, 1/2, 3/4])
    A = np.array([
        [0, 0, 0],
        [1/2, 0, 0],
        [0, 3/4, 0]
    ])
    B = np.array([2/9, 1/3, 4/9])
    E = np.array([5/72, -1/12, -1/9, 1/8])
    P = np.array([[1, -4 / 3, 5 / 9],
                  [0, 1, -2/3],
                  [0, 4/3, -8/9],
                  [0, -1, 1]])
    
    def __init__(self, fun, t0, y0, t_bound, max_step, rtol=0.001, atol=0.000001, vectorized=False, first_step=None, **extraneous):
        super().__init__(fun, t0, y0, t_bound, max_step, rtol, atol, vectorized, first_step, **extraneous)

class FixedStepRK45(FixedStepRungeKutta):
    order = 5
    error_estimator_order = 4
    n_stages = 6
    C = np.array([0, 1/5, 3/10, 4/5, 8/9, 1])
    A = np.array([
        [0, 0, 0, 0, 0],
        [1/5, 0, 0, 0, 0],
        [3/40, 9/40, 0, 0, 0],
        [44/45, -56/15, 32/9, 0, 0],
        [19372/6561, -25360/2187, 64448/6561, -212/729, 0],
        [9017/3168, -355/33, 46732/5247, 49/176, -5103/18656]
    ])
    B = np.array([35/384, 0, 500/1113, 125/192, -2187/6784, 11/84])
    E = np.array([-71/57600, 0, 71/16695, -71/1920, 17253/339200, -22/525,
                  1/40])
    # Corresponds to the optimum value of c_6 from [2]_.
    P = np.array([
        [1, -8048581381/2820520608, 8663915743/2820520608,
         -12715105075/11282082432],
        [0, 0, 0, 0],
        [0, 131558114200/32700410799, -68118460800/10900136933,
         87487479700/32700410799],
        [0, -1754552775/470086768, 14199869525/1410260304,
         -10690763975/1880347072],
        [0, 127303824393/49829197408, -318862633887/49829197408,
         701980252875 / 199316789632],
        [0, -282668133/205662961, 2019193451/616988883, -1453857185/822651844],
        [0, 40617522/29380423, -110615467/29380423, 69997945/29380423]])
    
    def __init__(self, fun, t0, y0, t_bound, max_step, rtol=0.001, atol=0.000001, vectorized=False, first_step=None, **extraneous):
        super().__init__(fun, t0, y0, t_bound, max_step, rtol, atol, vectorized, first_step, **extraneous)

class FixedStepDOP853(FixedStepRungeKutta):
    n_stages = dop853_coefficients.N_STAGES
    order = 8
    error_estimator_order = 7
    A = dop853_coefficients.A[:n_stages, :n_stages]
    B = dop853_coefficients.B
    C = dop853_coefficients.C[:n_stages]
    E3 = dop853_coefficients.E3
    E5 = dop853_coefficients.E5
    D = dop853_coefficients.D

    A_EXTRA = dop853_coefficients.A[n_stages + 1:]
    C_EXTRA = dop853_coefficients.C[n_stages + 1:]

    def __init__(self, fun, t0, y0, t_bound, max_step, rtol=0.001, atol=0.000001, vectorized=False, first_step=None, **extraneous):
        super().__init__(fun, t0, y0, t_bound, max_step, rtol, atol, vectorized, first_step, **extraneous)
        self.K_extended = np.empty((dop853_coefficients.N_STAGES_EXTENDED, self.n), dtype=self.y.dtype)
        self.K = self.K_extended[:self.n_stages + 1]

    def _dense_output_impl(self):
        K = self.K_extended
        h = self.h_previous
        for s, (a, c) in enumerate(zip(self.A_EXTRA, self.C_EXTRA),
                                   start=self.n_stages + 1):
            dy = np.dot(K[:s].T, a[:s]) * h
            K[s] = self.fun(self.t_old + c * h, self.y_old + dy)

        F = np.empty((dop853_coefficients.INTERPOLATOR_POWER, self.n),
                     dtype=self.y_old.dtype)

        f_old = K[0]
        delta_y = self.y - self.y_old

        F[0] = delta_y
        F[1] = h * f_old - delta_y
        F[2] = 2 * delta_y - h * (self.f + f_old)
        F[3:] = h * np.dot(self.D, K)

        return Dop853DenseOutput(self.t_old, self.t, self.y_old, F)