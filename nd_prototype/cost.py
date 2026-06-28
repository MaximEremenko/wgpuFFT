"""Stage 2: cost-model calibration for the ND NUFFT prototype.

Two jobs:
1. CALIBRATE: analytic work counters (grid updates, fine-grid memory, FFT
   passes) derived from plan parameters, cross-checked against instrumented
   counts from an actual type-1 execution (they must match exactly).
2. POLICY: a placement chooser for (n_modes, eps, memory budget) and the
   resulting desktop-32GiB / browser-256MiB tables for the design note.

Key accounting fact this stage establishes (sharpening design-note section 5):
per-point work is the PRODUCT of per-dim placement costs,
    updates/point = prod_{spread dims} w_i  *  prod_{RT dims} K_i,
because every RT node combo re-spreads (equivalently: an RT dim stores K
coefficient channels per point). At eps=1e-6, K = 7 = w(sigma=2): an RT dim
costs the same per point as a sigma=2 spread dim. RT's advantage is therefore
PURELY memory (fine = N, no oversampling); its price is prod(K) FFT passes.
"""

from __future__ import annotations

import math

import numpy as np

from nd_nufft import NdNufftPlan, Placement, direct_type1

S2, S125, RT = Placement.SPREAD_S2, Placement.SPREAD_S125, Placement.RT

C64 = 8  # bytes, f32 complex (GPU projection)


def analytic_cost(plan: NdNufftPlan, M: int) -> dict:
    w_prod = 1
    k_prod = 1
    for d in plan.dims:
        if d.placement is RT:
            k_prod *= d.K
        else:
            w_prod *= d.w
    grid = int(np.prod(plan.fine_shape))
    return {
        "updates_per_point": w_prod * k_prod,
        "total_updates": M * w_prod * k_prod,
        "grid_elems": grid,
        "grid_bytes_f32c": grid * C64,
        "fft_passes": k_prod,
        "fft_flops": 5.0 * grid * max(1.0, math.log2(grid)) * k_prod,
    }


def instrumented_updates(plan: NdNufftPlan, x: np.ndarray, c: np.ndarray) -> int:
    """Count actual grid-update operations by monkey-patching np.add.at."""
    count = 0
    orig = np.add.at

    def counting_add_at(a, indices, b):
        nonlocal count
        count += np.size(indices)
        orig(a, indices, b)

    np.add.at = counting_add_at
    try:
        plan.type1(x, c, isign=-1)
    finally:
        np.add.at = orig
    return count


def fmt_bytes(b: float) -> str:
    for unit, s in [(1 << 40, "TiB"), (1 << 30, "GiB"), (1 << 20, "MiB"), (1 << 10, "KiB")]:
        if b >= unit:
            return f"{b / unit:.2f} {s}"
    return f"{b:.0f} B"


def choose_placements(n_modes: list[int], eps: float, budget_bytes: int,
                      max_rt: int = 2) -> list[Placement] | None:
    """Greedy policy: sigma=2 for d<=3 if it fits, else sigma=1.25 everywhere;
    then convert largest dims to RT (one at a time, up to max_rt) until the
    fine grid fits the budget. None = infeasible without streaming."""
    d = len(n_modes)

    def grid_bytes(placements):
        total = 1
        for n, p in zip(n_modes, placements):
            if p is RT:
                total *= n
            else:
                sigma = 2.0 if p is S2 else 1.25
                total *= int(math.ceil(sigma * n / 2.0) * 2)
        return total * C64

    if d <= 3:
        pl = [S2] * d
        if grid_bytes(pl) <= budget_bytes:
            return pl
    pl = [S125] * d
    rt_used = 0
    while grid_bytes(pl) > budget_bytes and rt_used < max_rt:
        # convert the largest still-spread dim (biggest memory win)
        cand = max(
            (i for i in range(d) if pl[i] is not RT),
            key=lambda i: n_modes[i],
            default=None,
        )
        if cand is None:
            break
        pl[cand] = RT
        rt_used += 1
    return pl if grid_bytes(pl) <= budget_bytes else None


def calibration():
    print("== calibration: analytic vs instrumented update counts ==")
    rng = np.random.default_rng(7)
    cases = [
        ([32, 24], [S2, S2]),
        ([32, 24], [S125, RT]),
        ([16, 16, 12], [S125, S125, RT]),
        ([10, 10, 8, 8], [S125, S125, S125, RT]),
    ]
    ok = True
    for n_modes, pls in cases:
        M = 500
        plan = NdNufftPlan(n_modes, pls, 1e-6)
        x = rng.random((M, len(n_modes)))
        c = rng.standard_normal(M) + 1j * rng.standard_normal(M)
        a = analytic_cost(plan, M)
        got = instrumented_updates(plan, x, c)
        match = got == a["total_updates"]
        ok &= match
        knobs = ",".join({S2: "s2", S125: "s1.25", RT: "rt"}[p] for p in pls)
        print(f"  [{knobs}] analytic={a['total_updates']} instrumented={got} "
              f"{'MATCH' if match else 'MISMATCH'}")
    # sanity: accuracy unaffected by instrumentation path
    plan = NdNufftPlan([16, 16], [S125, RT], 1e-6)
    x = rng.random((200, 2))
    c = rng.standard_normal(200) + 1j * rng.standard_normal(200)
    err = np.linalg.norm(plan.type1(x, c) - direct_type1(x, c, [16, 16])) / np.linalg.norm(
        direct_type1(x, c, [16, 16])
    )
    print(f"  accuracy spot check rel-l2 = {err:.2e}")
    return ok


def strategy_tables():
    print("\n== strategy comparison (eps=1e-6, M=N^d, f32-complex grid bytes) ==")
    header = (f"{'d':<2} {'N':<4} {'policy':<22} {'grid':>10} {'upd/pt':>8} "
              f"{'FFT passes':>10} {'fits 32GiB':>10} {'fits 256MiB':>11}")
    print(header)
    configs = [
        (2, 4096), (3, 256), (4, 64), (4, 96), (5, 32), (6, 16),
    ]
    for d, N in configs:
        n_modes = [N] * d
        for name, pls in [
            ("all s2", [S2] * d),
            ("all s1.25", [S125] * d),
            ("s1.25 + 1 RT", [S125] * (d - 1) + [RT]),
            ("s1.25 + 2 RT", [S125] * (d - 2) + [RT, RT]) if d >= 2 else None,
        ]:
            if pls is None:
                continue
            plan = NdNufftPlan(n_modes, pls, 1e-6)
            a = analytic_cost(plan, M=N ** d)
            print(f"{d:<2} {N:<4} {name:<22} {fmt_bytes(a['grid_bytes_f32c']):>10} "
                  f"{a['updates_per_point']:>8} {a['fft_passes']:>10} "
                  f"{'yes' if a['grid_bytes_f32c'] <= 32 * (1 << 30) else 'NO':>10} "
                  f"{'yes' if a['grid_bytes_f32c'] <= 256 * (1 << 20) else 'NO':>11}")
        print()


def policy_demo():
    print("== placement policy (greedy, max 2 RT dims) ==")
    for budget, bname in [(32 * (1 << 30), "desktop 32 GiB"), (256 * (1 << 20), "browser 256 MiB")]:
        print(f"-- budget: {bname}")
        for n_modes in [[4096, 4096], [256] * 3, [64] * 4, [96] * 4, [32] * 5, [16] * 6, [128] * 4]:
            pl = choose_placements(n_modes, 1e-6, budget)
            if pl is None:
                print(f"   {str(n_modes):<28} INFEASIBLE (needs streaming/segmented)")
                continue
            plan = NdNufftPlan(n_modes, pl, 1e-6)
            a = analytic_cost(plan, M=int(np.prod(n_modes)))
            knobs = ",".join({S2: "s2", S125: "s1.25", RT: "rt"}[p] for p in pl)
            print(f"   {str(n_modes):<28} [{knobs}] grid={fmt_bytes(a['grid_bytes_f32c'])} "
                  f"upd/pt={a['updates_per_point']} passes={a['fft_passes']}")


if __name__ == "__main__":
    ok = calibration()
    strategy_tables()
    policy_demo()
    print("\nCALIBRATION " + ("OK" if ok else "FAILED"))
    raise SystemExit(0 if ok else 1)
