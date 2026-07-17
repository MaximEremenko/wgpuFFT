"""Validation matrix for the ND NUFFT prototype: accuracy vs direct NDFT oracle,
adjoint consistency, and clustered-point robustness, across dimensions and
per-dimension placements. Prints a table; exits nonzero on failure."""

from __future__ import annotations

import sys

import numpy as np

from nd_nufft import (
    NdNufftPlan,
    Placement,
    direct_type1,
    direct_type2,
)

S2 = Placement.SPREAD_S2
S125 = Placement.SPREAD_S125
RT = Placement.RT

rng = np.random.default_rng(20260715)


def rel_l2(a, b):
    return np.linalg.norm(a.ravel() - b.ravel()) / max(np.linalg.norm(b.ravel()), 1e-300)


def points(M, d, clustered=False):
    if not clustered:
        return rng.random((M, d))
    centers = rng.random((max(3, d), d))
    idx = rng.integers(0, len(centers), M)
    return np.mod(centers[idx] + 0.01 * rng.standard_normal((M, d)), 1.0)


def run_case(name, n_modes, placements, eps, M=1500, isign=-1, clustered=False):
    d = len(n_modes)
    x = points(M, d, clustered)
    c = rng.standard_normal(M) + 1j * rng.standard_normal(M)
    fhat = rng.standard_normal(n_modes) + 1j * rng.standard_normal(n_modes)

    plan = NdNufftPlan(list(n_modes), list(placements), eps)

    f1 = plan.type1(x, c, isign)
    f1_ref = direct_type1(x, c, list(n_modes), isign)
    e1 = rel_l2(f1, f1_ref)

    f2 = plan.type2(x, fhat, isign)
    f2_ref = direct_type2(x, fhat, list(n_modes), isign)
    e2 = rel_l2(f2, f2_ref)

    # adjoint: <type2(fhat), c> == <fhat, type1(c, -isign)>
    lhs = np.vdot(plan.type2(x, fhat, isign), c)
    rhs = np.vdot(fhat, plan.type1(x, c, -isign))
    ea = abs(lhs - rhs) / max(abs(lhs), 1e-300)

    tol = 60.0 * eps  # C*eps acceptance, FINUFFT-style
    ok = e1 < tol and e2 < tol and ea < tol
    knobs = ",".join(
        {S2: "s2", S125: "s1.25", RT: "rt"}[p] for p in placements
    )
    print(
        f"{'PASS' if ok else 'FAIL'}  {name:<26} eps={eps:.0e} [{knobs}] "
        f"t1={e1:.2e} t2={e2:.2e} adj={ea:.2e}"
    )
    return ok


def main():
    ok = True
    # d=1: every placement alone
    for pl, nm in [([S2], "1d spread s2"), ([S125], "1d spread s1.25"), ([RT], "1d rt")]:
        for eps in (1e-3, 1e-6, 1e-9):
            ok &= run_case(nm, [64], pl, eps)
    # d=2: mixes
    for pls, nm in [
        ([S2, S2], "2d s2+s2"),
        ([S125, RT], "2d s1.25+rt"),
        ([RT, RT], "2d rt+rt"),
    ]:
        for eps in (1e-6, 1e-9):
            ok &= run_case(nm, [32, 24], pls, eps)
    # d=3
    ok &= run_case("3d s1.25^2+rt", [16, 16, 12], [S125, S125, RT], 1e-6)
    ok &= run_case("3d s2^3", [16, 16, 12], [S2, S2, S2], 1e-6)
    # d=4: the design targets
    ok &= run_case("4d s1.25^4", [10, 10, 8, 8], [S125] * 4, 1e-6)
    ok &= run_case("4d s1.25^3+rt", [10, 10, 8, 8], [S125, S125, S125, RT], 1e-6)
    # d=5 small
    ok &= run_case("5d s1.25^4+rt", [6, 6, 6, 6, 6], [S125] * 4 + [RT], 1e-6)
    # isign=+1 and clustered robustness
    ok &= run_case("2d isign=+1", [32, 24], [S125, RT], 1e-6, isign=+1)
    ok &= run_case("3d clustered", [16, 16, 12], [S125, S125, RT], 1e-6, clustered=True)
    print("\nALL PASS" if ok else "\nFAILURES PRESENT")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
