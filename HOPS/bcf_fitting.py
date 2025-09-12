# This file is part of https://github.com/Turku-Quantum-Optics/hops
#
# Copyright (c) 2024-2025, Turku Quantum Optics
# 
# Licensed under the BSD 3-Clause License, see accompanying LICENSE,
# and README.md for further information.

import numpy as np
from numpy.polynomial import Polynomial as poly
from numpy.linalg import svd, solve, eig, lstsq

from scipy.linalg import hankel, eigh
from scipy.optimize import least_squares

# Implementation of the tPFD fitting scheme described in: 
# J. Chem. Phys. 156, 221102 (2022); https://doi.org/10.1063/5.0095961

# Works only for symmetric real matrices
def _takagi_symmetric(M):
    # Find the singular values (eigenvalues) of M and corresponding eigenvectors
    singular_values, Q = eigh(M)
    # Construct auxiliary matrix from the arguments / phases of the singular values
    P = np.diag(np.exp(-1j * np.angle(singular_values) / 2.0))
    # Store the absolute values of the singular values
    values = np.abs(singular_values)
    # Calculate QP
    QP = np.dot(Q, P)
    # Sort the absolute values in descending order
    sort_indices = np.flip(np.argsort(values))
    values = np.diag(values[sort_indices])
    # Reorder columns of QP to match the sorted singular absolute values
    QP = QP[:, sort_indices]
    return QP, values

def _residuals(x: np.ndarray, y: np.ndarray, w: np.ndarray) -> np.ndarray:
    j = np.repeat([np.arange(y.shape[0])], w.shape[0], axis=0).T
    w = np.repeat([w], y.shape[0], axis=0)
    return y - np.sum(x * np.power(w, j), axis=1)

# Residual function wrapper for least-squares fitting
def _res_wrap(x,y,w):
    n = w.shape[0]
    f = _residuals(x[:n] + 1j*x[n:], y, w)
    return np.concatenate((f.real, f.imag))

def _tPFD_fit(t: np.ndarray, y: np.ndarray, K: int):
    dt = t[1]-t[0]
    # Step (1) form the Hankel matrix
    H = hankel(y[:int(y.shape[0]/2 + 1)], y[int(y.shape[0]/2):])

    # Step (2) Takagi factorization
    U, Sigma = _takagi_symmetric(H)
    s = Sigma.diagonal()

    # Sort the c-eigenvalues in decending order (technically unnecessary here)
    sorted_indices = np.flip(np.argsort(np.abs(s)))
    s = s[sorted_indices]
    U = U[:, sorted_indices]

    # Take the c-eigenvector needed for the next step
    u = U[:,K]

    # Step (3) create a polynomial for the above chosen c-eigenvector
    p = poly(u)
    # Find roots
    w = p.roots()
    # Pick out the K roots that are inside the unit circle and sort them in ascending order
    w_lu = w[np.argsort(np.abs(w))][:K]
    # Calculate lambda coefficients
    W = -(1.0/dt) * (np.log(np.abs(w_lu)) + 1j*np.angle(w_lu))
    
    x0 = np.ones(2*w_lu.shape[0])
    # Step (4) least-squares fitting
    res = least_squares(_res_wrap,x0,args=(y,w_lu))
    G = res.x[:w_lu.shape[0]] + 1j*res.x[w_lu.shape[0]:]
    
    return G, W

# Implementation of the matrix pencil method
# TODO: Should we let the user pass the pencil parameter?
def _matrix_pencil_method_fit(bcf, dt, pencil_parameter=None, K=None):
    bcf = np.asarray(bcf, dtype=complex)
    M = len(bcf)
    if pencil_parameter is None:
        pencil_parameter = M // 2
    N = M - pencil_parameter

    # 1. Build Hankel matrices
    Y0 = np.column_stack([bcf[i:i+N] for i in range(pencil_parameter)])
    Y1 = np.column_stack([bcf[i+1:i+N+1] for i in range(pencil_parameter)])

    # 2. Singular value decomposition of Y0
    U, S, Vh = svd(Y0, full_matrices=False)

    if K is None:
        # Use significant singular values
        tol = np.max(S) * 1e-12 #TODO: Maybe let the user deside the tolerance?
        K = np.sum(S > tol)

    # 3. Use the K most significant singular values
    U_k = U[:, :K]
    S_k = np.diag(S[:K])
    V_k = Vh[:K, :].conjugate().T

    # 4. Reduced pencil matrix to obtain eigenvalues -> exponents W
    _V = U_k.T.conjugate().dot(Y1).dot(V_k)
    A = solve(S_k, _V)

    lambdas, _ = eig(A)
    W = -np.log(lambdas) / dt

    # 5. Solve amplitudes G by least squares
    V = np.exp(-np.outer(np.arange(M) * dt, W))
    G, _, _, _ = lstsq(V, bcf, rcond=None)

    return G, W

def fit_BCF_tPFD(t: np.ndarray, y: np.ndarray, Kr: int, Ki: int):
    G_real, W_real = (np.array([]), np.array([])) if Kr <= 0 else _tPFD_fit(t, y.real, Kr)
    G_imag, W_imag = (np.array([]), np.array([])) if Ki <= 0 else _tPFD_fit(t, y.imag, Ki)
    G = np.concatenate((G_real, 1j * G_imag))
    W = np.concatenate((W_real, W_imag))
    return G, W

def fit_BCF_matrix_pencil(bcf, dt, K=None):
    return _matrix_pencil_method_fit(bcf, dt, K=K)