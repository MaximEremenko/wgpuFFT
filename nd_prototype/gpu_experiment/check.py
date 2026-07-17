"""Compare the GPU-produced spread grid (out/) against an f64 numpy reference."""

import json
import sys

import numpy as np

with open("out/meta.json", encoding="utf-8") as f:
    meta = json.load(f)

d, m, w, beta = meta["d"], meta["m"], meta["w"], meta["beta"]
fine = meta["fine"]

x = np.fromfile("out/x.bin", dtype=np.float64).reshape(m, d)
c_pairs = np.fromfile("out/c.bin", dtype=np.float64).reshape(m, 2)
c = c_pairs[:, 0] + 1j * c_pairs[:, 1]
gpu_pairs = np.fromfile("out/grid.bin", dtype=np.float32).reshape(fine + [2])
gpu = (gpu_pairs[..., 0] + 1j * gpu_pairs[..., 1]).astype(np.complex128)


def es_phi(z, beta):
    out = np.zeros_like(z)
    mask = np.abs(z) <= 1.0
    out[mask] = np.exp(beta * (np.sqrt(1.0 - z[mask] ** 2) - 1.0))
    return out


ref = np.zeros(fine, dtype=np.complex128)
import itertools

cells = []
weights = []
for i in range(d):
    p = np.mod(x[:, i], 1.0) * fine[i]
    i0 = np.floor(p - w / 2.0).astype(np.int64) + 1
    a = np.arange(w)
    cells.append(np.mod(i0[:, None] + a[None, :], fine[i]))
    z = (i0[:, None] + a[None, :] - p[:, None]) * (2.0 / w)
    weights.append(es_phi(z, beta))

flat = ref.ravel()
strides = [int(np.prod(fine[i + 1:])) for i in range(d)]
for off in itertools.product(range(w), repeat=d):
    idx = np.zeros(m, dtype=np.int64)
    wt = np.ones(m)
    for i, a in enumerate(off):
        idx += cells[i][:, a] * strides[i]
        wt = wt * weights[i][:, a]
    np.add.at(flat, idx, c * wt)
ref = flat.reshape(fine)

err = np.linalg.norm((gpu - ref).ravel()) / np.linalg.norm(ref.ravel())
print(f"GPU vs f64 reference spread grid: rel-l2 = {err:.3e}")
ok = err < 1e-4  # f32 kernel vs f64 reference
print("PASS" if ok else "FAIL")
sys.exit(0 if ok else 1)
