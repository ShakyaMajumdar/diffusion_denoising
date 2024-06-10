from typing import NamedTuple
import numpy as np
from sklearn.mixture import GaussianMixture
from scipy.stats import multivariate_normal
from utils import snr_db_to_np
from numba import njit, prange
import numpy.linalg as la


def get_patches(sig, P):
    """
    return list of all overlapping patches of length P each from sig
    """
    N = len(sig)
    patches = []
    sig2 = np.tile(sig, 2)
    for i in range(N):
        patches.append(sig2[i:i+P])
    return patches

@njit(fastmath=True)
def _G(u, alphas, Cov1inv, det_rts, means, Cs, K):
    """
    denoise a single patch
    """
    a_no_us = np.array([alphas[i] * mvn_pdf(means[i], Cov1inv[i], det_rts[i], u) for i in range(K)])
    M = np.zeros_like(Cs[0])
    for j in range(K):
        M += a_no_us[j]*Cs[j]
    return M @ u / a_no_us.sum()

@njit(parallel=True, fastmath=True)
def _D(z, P, K, N, Cs, alphas, Cov1inv, means, det_rts):
    """
    gmm-denoise the signal
    """
    res = np.zeros((N+P,))
    for i in prange(N):
        res[i:i+P] += _G(z[i:i+P], alphas, Cov1inv, det_rts, means, Cs, K)
    res[:P] += res[N:]
    return res[:N] / P

@njit(fastmath=True)
def mvn_pdf(mu, sigma_inv, det_rt, x):
    return det_rt * np.exp(-1/2 * (x - mu) @ sigma_inv @ (x - mu))

class _DenoiseCtx(NamedTuple):
    P: int  # size of each patch
    K: int  # no. of gmm components
    N: int  # size of signal
    sigma2: float  # noise power
    # Pis: list[np.ndarray]  # Pis[i] extracts [i:i+P] slice from vector of size N
    Cov1inv: list[np.ndarray]  # (gmm.covariances[j] + sigma2 * I)^-1
    Cs: list[np.ndarray]  # gmm.covariances[j] @ Cov1inv[j]
    alphas: list[np.ndarray]  # gmm component weights
    # Nos: list[np.ndarray]  # gmm component pdfs
    means: list[np.ndarray]
    det_rts: list[float]


class GMMDenoiser:
    def __init__(self, gmm_n_components, patch_size, train_signals):
        self.K = gmm_n_components
        self.P = patch_size
        self.gmm = GaussianMixture(gmm_n_components)
        self.train_signals = train_signals

        self._trained = False

    def n_params(self):
        return self.gmm.covariances_.size + self.gmm.means_.size + self.gmm.weights_.size

    def fit(self):
        self.gmm.fit(
            [
                patch
                for train_signal in self.train_signals
                for patch in get_patches(train_signal, self.P)
            ]
        )
        self._trained = True

    def compute_ctx(self, signal, snr_db_est):
        N = len(signal)
        sigma2 = snr_db_to_np(signal, snr_db_est)
        Cov1inv = np.array([
            la.inv(self.gmm.covariances_[j] + sigma2 * np.eye(self.P)) for j in range(self.K)
        ])
        Cs = np.array([self.gmm.covariances_[j] @ Cov1inv[j] for j in range(self.K)])
        alphas = self.gmm.weights_
        det_rts = np.array([(2*np.pi)**(-self.P/2) * la.det(Cov1inv[j])**(1/2) for j in range(self.K)])
        return _DenoiseCtx(self.P, self.K, N, sigma2, Cov1inv, Cs, alphas, self.gmm.means_, det_rts)

    def assert_trained(self):
        if not self._trained:
            raise RuntimeError("Model has not been fit")

    def denoise(self, signal, *, snr_db_est):
        self.assert_trained()
        ctx = self.compute_ctx(signal, snr_db_est)
        return _D(np.tile(signal,2), self.P, self.K, ctx.N, ctx.Cs, ctx.alphas, ctx.Cov1inv, ctx.means, ctx.det_rts)


