"""End-to-end (milestone 3b) validation: GPU type-1 spectrum and type-2 outputs
vs f64 direct NDFT oracles. Auto-detects the FFT sign convention and reports it."""

import itertools
import json
import sys

import numpy as np

with open("out/meta2.json", encoding="utf-8") as f:
    meta = json.load(f)

d, m, w, beta = meta["d"], meta["m"], meta["w"], meta["beta"]
fine, n_modes = meta["fine"], meta["n_modes"]

x = np.fromfile("out/x2.bin", dtype=np.float64).reshape(m, d)
cp = np.fromfile("out/c2.bin", dtype=np.float64).reshape(m, 2)
c = cp[:, 0] + 1j * cp[:, 1]
fp = np.fromfile("out/fhat.bin", dtype=np.float64).reshape(n_modes + [2])
fhat = fp[..., 0] + 1j * fp[..., 1]
spec_p = np.fromfile("out/t1_spec.bin", dtype=np.float32).reshape(fine + [2])
spec = (spec_p[..., 0] + 1j * spec_p[..., 1]).astype(np.complex128)
t2_p = np.fromfile("out/t2_out.bin", dtype=np.float32).reshape(m, 2)
t2 = (t2_p[:, 0] + 1j * t2_p[:, 1]).astype(np.complex128)


def es_phi(z, beta):
    out = np.zeros_like(z)
    mask = np.abs(z) <= 1.0
    out[mask] = np.exp(beta * (np.sqrt(1.0 - z[mask] ** 2) - 1.0))
    return out


def phi_hat(nu, w, beta):
    t = np.linspace(0.0, w / 2.0, 2000)
    ph = es_phi(2.0 * t / w, beta)
    return np.array([2.0 * np.trapezoid(ph * np.cos(2 * np.pi * v * t), t) for v in np.atleast_1d(nu)])


def mode_grid(nm):
    axes = [np.arange(-(n // 2), (n + 1) // 2) for n in nm]
    mesh = np.meshgrid(*axes, indexing="ij")
    return np.stack([q.ravel() for q in mesh], axis=1)


k = mode_grid(n_modes)

# --- type-1: extract centered modes from GPU fine spectrum, deconvolve ------
mode_idx = [np.mod(np.arange(-(n // 2), (n + 1) // 2), f) for n, f in zip(n_modes, fine)]
gpu_modes = spec[np.ix_(*mode_idx)]
for i in range(d):
    kk = np.arange(-(n_modes[i] // 2), (n_modes[i] + 1) // 2)
    dc = 1.0 / phi_hat(kk / fine[i], w, beta)
    sh = [1] * d
    sh[i] = n_modes[i]
    gpu_modes = gpu_modes * dc.reshape(sh)

best = None
for isign in (-1, +1):
    phase = np.exp(1j * isign * 2 * np.pi * (x @ k.T))
    ref1 = (phase.T @ c).reshape(n_modes)
    e = np.linalg.norm((gpu_modes - ref1).ravel()) / np.linalg.norm(ref1.ravel())
    if best is None or e < best[1]:
        best = (isign, e, ref1)
isign1, e1, ref1 = best

# sigma=1.25 conditioning: 1/phi_hat amplifies f32 FFT rounding noise at the
# mode-box edge (phi_hat(0)/phi_hat(edge) ~ 70x per dim at w=10). Interior
# modes must show clean f32 accuracy; the edge shell degrades by design.
interior = np.ones(n_modes, dtype=bool)
axes = [np.arange(-(n // 2), (n + 1) // 2) for n in n_modes]
for i in range(d):
    sh = [1] * d
    sh[i] = n_modes[i]
    interior &= np.abs(axes[i]).reshape(sh) <= n_modes[i] // 4
err = np.abs(gpu_modes - ref1)
rn = np.linalg.norm(ref1.ravel())
e1_int = np.linalg.norm(err[interior]) / rn
e1_edge = np.linalg.norm(err[~interior]) / rn
amp = None
for i in range(d):
    kk = np.arange(-(n_modes[i] // 2), (n_modes[i] + 1) // 2)
    ph = phi_hat(kk / fine[i], w, beta)
    a = ph.max() / ph.min()
    amp = a if amp is None else max(amp, a)
print(
    f"type-1 end-to-end: rel-l2 = {e1:.3e} (interior {e1_int:.3e}, edge {e1_edge:.3e}; "
    f"per-dim 1/phi_hat amplification {amp:.0f}x; FFT sign isign={isign1})"
)

# --- type-2: GPU outputs vs direct oracle -----------------------------------
best = None
for isign in (-1, +1):
    phase = np.exp(1j * isign * 2 * np.pi * (x @ k.T))
    ref2 = phase @ fhat.ravel()
    e = np.linalg.norm(t2 - ref2) / np.linalg.norm(ref2)
    if best is None or e < best[1]:
        best = (isign, e)
isign2, e2 = best
print(f"type-2 end-to-end: rel-l2 = {e2:.3e} (FFT sign matches isign={isign2})")

# Acceptance: interior modes at clean f32 accuracy; full box bounded by the
# documented sigma=1.25 edge-conditioning floor (f32 noise x ~70x/dim edge
# amplification). f64/df64 grids or sigma=2 recover the edge — design-note S2.
ok = e1_int < 1e-5 and e1 < 1e-3 and e2 < 1e-3 and isign1 == isign2
print("PASS" if ok else "FAIL")
sys.exit(0 if ok else 1)
