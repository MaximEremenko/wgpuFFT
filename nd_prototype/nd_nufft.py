"""Rank-generic ND NUFFT reference prototype (stage 1 of the ND NUFFT Math Design).

Standalone numpy implementation validating the per-dimension "translation-rank
placement" design: every dimension independently chooses how to pay its
translation rank r(eps):

  - Placement.SPREAD_S2   : ES-kernel spreading, sigma = 2.0   (small w, big grid)
  - Placement.SPREAD_S125 : ES-kernel spreading, sigma = 1.25  (big w, small grid)
  - Placement.RT          : Ruiz-Antolin/Townsend-style sigma = 1 factorization
                            (no fine grid, no kernel; K diagonally-scaled FFTs
                            via Chebyshev interpolation of the perturbation
                            factor exp(-2*pi*i*k*delta/N))

Conventions:
  type-1 (isign):  f[k]  = sum_j c[j] * exp(isign * 2*pi*i * k . x[j])
  type-2 (isign):  f[j]  = sum_k fhat[k] * exp(isign * 2*pi*i * k . x[j])
  x in [0,1)^d (torus), modes k_i in [-N_i/2, N_i/2).

This is prototype-grade code: clarity over speed; all heavy lifting is
vectorized over points but loops over kernel-offset tuples and RT node combos.
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass
from enum import Enum

import numpy as np


class Placement(Enum):
    SPREAD_S2 = "spread_s2"
    SPREAD_S125 = "spread_s125"
    RT = "rt"


# ---------------------------------------------------------------------------
# ES kernel (standard width/beta parameter rules)
# ---------------------------------------------------------------------------


def es_params(eps: float, sigma: float) -> tuple[int, float]:
    """Width/beta selection for the ES kernel."""
    if abs(sigma - 2.0) < 1e-12:
        w = int(math.ceil(math.log10(10.0 / eps)))
        w = max(2, w)
        beta_over_w = {2: 2.20, 3: 2.26, 4: 2.38}.get(w, 2.30)
        return w, beta_over_w * w
    w = int(math.ceil(math.log(1.0 / eps) / (math.pi * math.sqrt(1.0 - 1.0 / sigma))))
    w = max(2, min(16, w))
    beta = 0.97 * math.pi * w * (1.0 - 1.0 / (2.0 * sigma))
    return w, beta


def es_phi(z: np.ndarray, beta: float) -> np.ndarray:
    out = np.zeros_like(z, dtype=np.float64)
    m = np.abs(z) <= 1.0
    out[m] = np.exp(beta * (np.sqrt(1.0 - z[m] ** 2) - 1.0))
    return out


def es_phi_hat(nu: np.ndarray, w: int, beta: float, quad_points: int = 2000) -> np.ndarray:
    """Continuous Fourier transform phi_hat(nu) = int_{-w/2}^{w/2} phi(2t/w) e^{-2*pi*i*nu*t} dt.

    phi is even, so the transform is real:  2 * int_0^{w/2} phi cos(2*pi*nu*t) dt.
    Trapezoid quadrature; the integrand is smooth and compactly supported.
    """
    t = np.linspace(0.0, w / 2.0, quad_points)
    phi_t = es_phi(2.0 * t / w, beta)
    out = np.empty_like(nu, dtype=np.float64)
    for i, v in enumerate(np.atleast_1d(nu)):
        out.flat[i] = 2.0 * np.trapezoid(phi_t * np.cos(2.0 * math.pi * v * t), t)
    return out


# ---------------------------------------------------------------------------
# RT (sigma = 1) factorization via Chebyshev/Lagrange interpolation in delta
# ---------------------------------------------------------------------------


def rt_rank(eps: float, n: int, max_k: int = 40) -> int:
    """Smallest K such that Chebyshev interpolation (K nodes on delta in [-1/2,1/2])
    of exp(-2*pi*i*k*delta/n) meets eps for the worst mode |k| = n/2."""
    delta_test = np.linspace(-0.5, 0.5, 401)
    target = np.exp(-1j * 2.0 * math.pi * (0.5) * delta_test)  # k/n = 1/2 (worst case)
    for K in range(3, max_k + 1):
        nodes, weights = cheb_nodes_weights(K)
        L = lagrange_matrix(delta_test, nodes, weights)  # (401, K)
        approx = L @ np.exp(-1j * 2.0 * math.pi * 0.5 * nodes)
        if np.max(np.abs(approx - target)) < eps / 3.0:
            return K
    return max_k


def cheb_nodes_weights(K: int) -> tuple[np.ndarray, np.ndarray]:
    """Chebyshev points of the first kind on [-1/2, 1/2] + barycentric weights."""
    j = np.arange(K)
    theta = (2.0 * j + 1.0) * math.pi / (2.0 * K)
    nodes = 0.5 * np.cos(theta)
    weights = ((-1.0) ** j) * np.sin(theta)
    return nodes, weights


def lagrange_matrix(delta: np.ndarray, nodes: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Barycentric Lagrange basis values L_l(delta): shape (len(delta), K)."""
    d = delta[:, None] - nodes[None, :]
    exact = np.abs(d) < 1e-14
    d = np.where(exact, 1.0, d)
    terms = weights[None, :] / d
    L = terms / np.sum(terms, axis=1, keepdims=True)
    L = np.where(exact, 1.0, L)
    # rows with an exact hit: zero out the others
    hit_rows = exact.any(axis=1)
    if hit_rows.any():
        L[hit_rows] = exact[hit_rows].astype(np.float64)
    return L


# ---------------------------------------------------------------------------
# Plan
# ---------------------------------------------------------------------------


@dataclass
class DimPlan:
    placement: Placement
    n_modes: int
    fine: int  # fine-grid size (== n_modes for RT dims)
    w: int = 0
    beta: float = 0.0
    K: int = 0  # RT nodes


class NdNufftPlan:
    def __init__(self, n_modes: list[int], placements: list[Placement], eps: float):
        assert len(n_modes) == len(placements)
        self.eps = eps
        self.dims: list[DimPlan] = []
        for n, pl in zip(n_modes, placements):
            if pl is Placement.RT:
                self.dims.append(DimPlan(pl, n, n, K=rt_rank(eps, n)))
            else:
                sigma = 2.0 if pl is Placement.SPREAD_S2 else 1.25
                w, beta = es_params(eps, sigma)
                fine = int(math.ceil(sigma * n / 2.0) * 2)  # even fine size >= sigma*n
                self.dims.append(DimPlan(pl, n, fine, w=w, beta=beta))
        self.fine_shape = tuple(d.fine for d in self.dims)
        # per spread dim: deconvolution factors 1/phi_hat(k/fine) for k in [-N/2, N/2)
        self._deconv = []
        for d in self.dims:
            if d.placement is Placement.RT:
                self._deconv.append(None)
            else:
                k = np.arange(-(d.n_modes // 2), (d.n_modes + 1) // 2)
                self._deconv.append(1.0 / es_phi_hat(k / d.fine, d.w, d.beta))

    # -- geometry helpers ---------------------------------------------------

    def _spread_tables(self, x: np.ndarray):
        """Per dim: (cells, weights) for spread dims [(M,w) int, (M,w) float],
        (s, delta) for RT dims [(M,) int, (M,) float]."""
        tables = []
        for i, d in enumerate(self.dims):
            xi = np.mod(x[:, i], 1.0)
            if d.placement is Placement.RT:
                p = xi * d.n_modes
                s = np.round(p).astype(np.int64)
                delta = p - s  # in [-1/2, 1/2]
                tables.append((np.mod(s, d.n_modes), delta))
            else:
                p = xi * d.fine
                i0 = np.floor(p - d.w / 2.0).astype(np.int64) + 1
                a = np.arange(d.w)
                cells = np.mod(i0[:, None] + a[None, :], d.fine)
                z = (i0[:, None] + a[None, :] - p[:, None]) * (2.0 / d.w)
                weights = es_phi(z, d.beta)
                tables.append((cells, weights))
        return tables

    def _rt_dims(self):
        return [i for i, d in enumerate(self.dims) if d.placement is Placement.RT]

    def _spread_dims(self):
        return [i for i, d in enumerate(self.dims) if d.placement is not Placement.RT]

    def _mode_slices(self):
        """Slices/index arrays extracting the centered mode box from the fine grid
        after fftshift-style reindexing. We work in 'DFT index' space: mode k maps
        to index k mod fine."""
        idx = []
        for d in self.dims:
            k = np.arange(-(d.n_modes // 2), (d.n_modes + 1) // 2)
            idx.append(np.mod(k, d.fine))
        return idx

    # -- type 1: points -> modes --------------------------------------------

    def type1(self, x: np.ndarray, c: np.ndarray, isign: int = -1) -> np.ndarray:
        M, d = x.shape
        assert d == len(self.dims)
        tables = self._spread_tables(x)
        rt_dims = self._rt_dims()
        sp_dims = self._spread_dims()
        mode_idx = self._mode_slices()

        # RT: Lagrange basis values per RT dim: (M, K)
        rt_L = {}
        rt_nodes = {}
        for i in rt_dims:
            K = self.dims[i].K
            nodes, wts = cheb_nodes_weights(K)
            rt_L[i] = lagrange_matrix(tables[i][1], nodes, wts)
            rt_nodes[i] = nodes

        out_shape = tuple(dd.n_modes for dd in self.dims)
        f = np.zeros(out_shape, dtype=np.complex128)

        strides = np.array(
            [int(np.prod(self.fine_shape[i + 1:])) for i in range(len(self.dims))],
            dtype=np.int64,
        )
        rt_combos = list(itertools.product(*[range(self.dims[i].K) for i in rt_dims])) or [()]
        sp_offsets = list(itertools.product(*[range(self.dims[i].w) for i in sp_dims])) or [()]

        for combo in rt_combos:
            # per-point extra weight from RT Lagrange bases
            wextra = np.ones(M, dtype=np.float64)
            for i, l in zip(rt_dims, combo):
                wextra = wextra * rt_L[i][:, l]
            vals_base = c * wextra

            grid = np.zeros(int(np.prod(self.fine_shape)), dtype=np.complex128)
            # RT dims contribute fixed integer cells
            base_idx = np.zeros(M, dtype=np.int64)
            for i in rt_dims:
                base_idx += tables[i][0] * strides[i]
            for off in sp_offsets:
                idxs = base_idx.copy()
                wprod = np.ones(M, dtype=np.float64)
                for i, a in zip(sp_dims, off):
                    idxs += tables[i][0][:, a] * strides[i]
                    wprod = wprod * tables[i][1][:, a]
                np.add.at(grid, idxs, vals_base * wprod)

            grid = grid.reshape(self.fine_shape)
            # B_hat[k] = sum_m grid[m] e^{isign 2 pi i k m / n}
            if isign == -1:
                G = np.fft.fftn(grid)
            else:
                G = np.fft.ifftn(grid) * np.prod(self.fine_shape)
            # extract centered modes
            G = G[np.ix_(*mode_idx)]
            # RT node scaling: exp(isign*2*pi*i * k * delta_l / N) per RT dim
            for i, l in zip(rt_dims, combo):
                dd = self.dims[i]
                k = np.arange(-(dd.n_modes // 2), (dd.n_modes + 1) // 2)
                scale = np.exp(1j * isign * 2.0 * math.pi * k * rt_nodes[i][l] / dd.n_modes)
                sh = [1] * len(self.dims)
                sh[i] = dd.n_modes
                G = G * scale.reshape(sh)
            f += G

        # deconvolve spread dims
        for i in sp_dims:
            sh = [1] * len(self.dims)
            sh[i] = self.dims[i].n_modes
            f = f * self._deconv[i].reshape(sh)
        return f

    # -- type 2: modes -> points --------------------------------------------

    def type2(self, x: np.ndarray, fhat: np.ndarray, isign: int = -1) -> np.ndarray:
        M, d = x.shape
        assert d == len(self.dims)
        tables = self._spread_tables(x)
        rt_dims = self._rt_dims()
        sp_dims = self._spread_dims()
        mode_idx = self._mode_slices()

        rt_L = {}
        rt_nodes = {}
        for i in rt_dims:
            K = self.dims[i].K
            nodes, wts = cheb_nodes_weights(K)
            rt_L[i] = lagrange_matrix(tables[i][1], nodes, wts)
            rt_nodes[i] = nodes

        # pre-deconvolve spread dims
        g = fhat.astype(np.complex128)
        for i in sp_dims:
            sh = [1] * len(self.dims)
            sh[i] = self.dims[i].n_modes
            g = g * self._deconv[i].reshape(sh)

        strides = np.array(
            [int(np.prod(self.fine_shape[i + 1:])) for i in range(len(self.dims))],
            dtype=np.int64,
        )
        rt_combos = list(itertools.product(*[range(self.dims[i].K) for i in rt_dims])) or [()]
        sp_offsets = list(itertools.product(*[range(self.dims[i].w) for i in sp_dims])) or [()]

        f = np.zeros(M, dtype=np.complex128)
        for combo in rt_combos:
            gg = g.copy()
            for i, l in zip(rt_dims, combo):
                dd = self.dims[i]
                k = np.arange(-(dd.n_modes // 2), (dd.n_modes + 1) // 2)
                scale = np.exp(1j * isign * 2.0 * math.pi * k * rt_nodes[i][l] / dd.n_modes)
                sh = [1] * len(self.dims)
                sh[i] = dd.n_modes
                gg = gg * scale.reshape(sh)
            # pad into fine grid (DFT-index space); G[m] = sum_k g_k e^{isign 2 pi i k m / n}
            grid = np.zeros(self.fine_shape, dtype=np.complex128)
            grid[np.ix_(*mode_idx)] = gg
            if isign == -1:
                Gr = np.fft.fftn(grid)
            else:
                Gr = np.fft.ifftn(grid) * np.prod(self.fine_shape)
            Gflat = Gr.ravel()

            wextra = np.ones(M, dtype=np.float64)
            for i, l in zip(rt_dims, combo):
                wextra = wextra * rt_L[i][:, l]

            base_idx = np.zeros(M, dtype=np.int64)
            for i in rt_dims:
                base_idx += tables[i][0] * strides[i]
            acc = np.zeros(M, dtype=np.complex128)
            for off in sp_offsets:
                idxs = base_idx.copy()
                wprod = np.ones(M, dtype=np.float64)
                for i, a in zip(sp_dims, off):
                    idxs += tables[i][0][:, a] * strides[i]
                    wprod = wprod * tables[i][1][:, a]
                acc += Gflat[idxs] * wprod
            f += acc * wextra
        return f


# ---------------------------------------------------------------------------
# Direct NDFT oracle (f64)
# ---------------------------------------------------------------------------


def mode_grid(n_modes: list[int]) -> np.ndarray:
    axes = [np.arange(-(n // 2), (n + 1) // 2) for n in n_modes]
    mesh = np.meshgrid(*axes, indexing="ij")
    return np.stack([m.ravel() for m in mesh], axis=1)  # (Ntot, d)


def direct_type1(x, c, n_modes, isign=-1):
    k = mode_grid(n_modes)
    phase = np.exp(1j * isign * 2.0 * math.pi * (x @ k.T))  # (M, Ntot)
    return (phase.T @ c).reshape(n_modes)


def direct_type2(x, fhat, n_modes, isign=-1):
    k = mode_grid(n_modes)
    phase = np.exp(1j * isign * 2.0 * math.pi * (x @ k.T))  # (M, Ntot)
    return phase @ fhat.ravel()
