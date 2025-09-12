import numpy as np

from scipy.integrate import solve_ivp


class NMQSDDriver:
    def __init__(self, N, L, G, W):
        self.N = N
        self.L = L
        self.G = G
        self.W = W
        K = 0
        for n in range(N):
            for m in range(N):
                K += len(G[n][m])
        row = 0
        WFlat = []
        M = np.zeros((K, N), dtype=complex)
        I = []
        for n in range(N):
            start = row
            for m in range(N):
                for mu in range(len(G[n][m])):
                    M[row, m] = G[n][m][mu].conjugate()
                    WFlat.append(W[n][m][mu])
                    row += 1
            end = row
            I.append((start, end))
        self.K = K
        self.M = M
        self.I = I
        self.WFlat = np.array(WFlat, dtype=complex)

    def solve_linear_NMQSD(self, t_eval, initial_state, H, noises, max_step=0.01):
        sol = solve_ivp(self._linearNMQSDStep, (t_eval[0], t_eval[-1]), initial_state, t_eval=t_eval, max_step=max_step, args=(self.N, H, noises, self.L, self.G, self.W,))
        trajectory = sol.y.T
        return sol.t, trajectory
    
    def solve_non_linear_NMQSD(self, t_eval, initial_state, H, noises, max_step=0.01):
        initial_state = np.concatenate((initial_state, np.zeros(self.K, dtype=complex)))
        sol = solve_ivp(self._nonLinearNMQSDStep, (t_eval[0], t_eval[-1]), initial_state, t_eval=t_eval, max_step=max_step, args=(self.N, H, noises, self.L, self.G, self.W, self.M, self.I, self.WFlat))
        dimension = H.shape[0]
        trajectory = sol.y.T[:, :dimension]
        return sol.t, trajectory
    
    def map_linear_trajectory(self, trajectories: np.ndarray):
            return np.einsum('ni,nj->nij', trajectories, trajectories.conjugate())

    def map_non_linear_trajectory(self, trajectories: np.ndarray):
        norms = np.linalg.norm(trajectories, axis=1, keepdims=True)
        normalized = trajectories / norms
        return np.einsum('ni,nj->nij', normalized, normalized.conjugate())

    def _linearNMQSDStep(self, t, y, N, H, z, L, G, W):
        # N = Number of emitters
        # H = System Hamiltonian
        # z = List of noises (N amount)
        # L = List of coupling operators (N amount)
        # G = Matrix of G coefficients (N x N matrix) corresponding to the BCF matrix
        # W = Matrix of W coefficients (N x N matrix) corresponding to the BCF matrix
        result = -1j * H.dot(y)
        for n in range(N):
            result += z[n](t).conjugate() * L[n].dot(y)
            for m in range(N):
                F_nm = np.sum(G[n][m] / W[n][m] * (1 - np.exp(-W[n][m] * t)))
                result -= F_nm * L[n].T.conjugate().dot(L[m].dot(y))
        return result

    def _nonLinearNMQSDStep(self, t, y, N, H, z, L, G, W, M, I, WFlat):
        # N = Number of emitters
        # H = System Hamiltonian
        # z = List of noises (N amount)
        # L = List of coupling operators (N amount)
        # G = Matrix of G coefficients (N x N matrix) corresponding to the BCF matrix
        # W = Matrix of W coefficients (N x N matrix) corresponding to the BCF matrix
        # M = Matrix for noise shift
        # I = Array of shift start and end indices (N amount)
        dimension = H.shape[0]
        identity = np.identity(dimension, dtype=complex)
        currentState = y[0:dimension]
        shiftVector = y[dimension:]
        LDaggerExp = np.array([currentState.conjugate().dot(L[n].T.conjugate()).dot(currentState) / (currentState.conjugate().dot(currentState) + 1e-12) for n in range(N)], dtype=complex)
        shiftedZ = [z[n](t).conjugate() + np.sum(shiftVector[I[n][0]:I[n][1]]) for n in range(N)]
        shiftVectorUpdate = M.dot(LDaggerExp) - WFlat.conjugate() * shiftVector

        stateUpdate = -1j * H.dot(currentState)
        for n in range(N):
            stateUpdate += shiftedZ[n] * L[n].dot(currentState)
            for m in range(N):
                F_nm = np.sum(G[n][m] / W[n][m] * (1 - np.exp(-W[n][m] * t)))
                stateUpdate -= F_nm * (L[n].T.conjugate() - LDaggerExp[n] * identity).dot(L[m].dot(currentState))
        return np.concatenate((stateUpdate, shiftVectorUpdate))