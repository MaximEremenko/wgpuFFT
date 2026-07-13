#!/usr/bin/env python3
"""Small, reproducible FINUFFT CPU context benchmark for wgpu-nufft.

This intentionally uses FINUFFT's reusable Plan API and reports plan creation,
setpts, and execute timings separately.  The execute-only result is the useful
context for wgpu-nufft plan reuse; it is not a CPU/GPU equivalence claim.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import math
import os
import platform
import statistics
import sys
import time
from pathlib import Path
from typing import Iterable

import finufft
import numpy as np


DEFAULT_CASES = ((262_144, 262_144), (1_048_576, 1_048_576))
DEFAULT_SEED = 0x4E55_4646
DEFAULT_THREADS = 32
DEFAULT_SAMPLES = 10
LCG_MULTIPLIER = 1_664_525
LCG_INCREMENT = 1_013_904_223
U24_SCALE = np.float32(1.0 / (1 << 24))


def positive_int(value: str) -> int:
    parsed = int(value, 0)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def nonnegative_int(value: str) -> int:
    parsed = int(value, 0)
    if parsed < 0:
        raise argparse.ArgumentTypeError("value must be nonnegative")
    return parsed


def parse_case(value: str) -> tuple[int, int]:
    fields = value.replace("x", ":").split(":")
    if len(fields) == 1:
        n_modes = point_count = positive_int(fields[0])
    elif len(fields) == 2:
        n_modes, point_count = map(positive_int, fields)
    else:
        raise argparse.ArgumentTypeError("case must be N or N:M")
    return n_modes, point_count


def lcg_uniform(count: int, seed: int) -> np.ndarray:
    """Return prefix-stable float32 samples in [0, 1) from a u32 LCG."""

    state = seed & 0xFFFF_FFFF

    def samples() -> Iterable[int]:
        nonlocal state
        for _ in range(count):
            state = (state * LCG_MULTIPLIER + LCG_INCREMENT) & 0xFFFF_FFFF
            yield state >> 8

    values = np.fromiter(samples(), dtype=np.uint32, count=count)
    return values.astype(np.float32) * U24_SCALE


def make_inputs(n_modes: int, point_count: int, seed: int):
    """Generate field-specific, prefix-stable LCG streams for one case."""

    points_u = lcg_uniform(point_count, seed ^ 0xA341_316C)
    strength_re = lcg_uniform(point_count, seed ^ 0xC801_3EA4)
    strength_im = lcg_uniform(point_count, seed ^ 0xAD90_777D)
    mode_re = lcg_uniform(n_modes, seed ^ 0x7E95_761E)
    mode_im = lcg_uniform(n_modes, seed ^ 0x6C8E_9CF5)

    points = np.ascontiguousarray(
        points_u * np.float32(2.0 * math.pi) - np.float32(math.pi),
        dtype=np.float32,
    )
    strengths = np.empty(point_count, dtype=np.complex64)
    strengths.real = strength_re * np.float32(2.0) - np.float32(1.0)
    strengths.imag = strength_im * np.float32(2.0) - np.float32(1.0)
    modes = np.empty(n_modes, dtype=np.complex64)
    modes.real = mode_re * np.float32(2.0) - np.float32(1.0)
    modes.imag = mode_im * np.float32(2.0) - np.float32(1.0)
    return points, strengths, modes


def elapsed_ms(start_ns: int) -> float:
    return (time.perf_counter_ns() - start_ns) / 1_000_000.0


def stderr(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    return statistics.stdev(values) / math.sqrt(len(values))


def format_samples(values: list[float]) -> str:
    return "[" + ", ".join(f"{value:.6f}" for value in values) + "]"


def benchmark_kind(
    kind: int,
    n_modes: int,
    point_count: int,
    points: np.ndarray,
    input_values: np.ndarray,
    *,
    eps: float,
    threads: int,
    runs: int,
    samples: int,
    warmups: int,
) -> None:
    kind_name = f"type-{kind}"
    output_length = n_modes if kind == 1 else point_count
    output = np.empty(output_length, dtype=np.complex64)
    plan_samples: list[float] = []
    setpts_samples: list[float] = []
    execute_samples: list[list[float]] = []

    for _ in range(runs):
        start = time.perf_counter_ns()
        plan = finufft.Plan(
            kind,
            (n_modes,),
            eps=eps,
            isign=1,
            dtype="complex64",
            nthreads=threads,
            modeord=0,
            upsampfac=2.0,
        )
        plan_samples.append(elapsed_ms(start))

        start = time.perf_counter_ns()
        plan.setpts(points)
        setpts_samples.append(elapsed_ms(start))

        for _ in range(warmups):
            plan.execute(input_values, out=output)

        run_samples: list[float] = []
        gc_was_enabled = gc.isenabled()
        gc.disable()
        try:
            for _ in range(samples):
                start = time.perf_counter_ns()
                plan.execute(input_values, out=output)
                run_samples.append(elapsed_ms(start))
        finally:
            if gc_was_enabled:
                gc.enable()
        execute_samples.append(run_samples)
        del plan
        gc.collect()

    run_means = [statistics.mean(values) for values in execute_samples]
    execute_mean = statistics.mean(run_means)
    execute_stderr = stderr(run_means)
    execute_min = min(min(values) for values in execute_samples)
    setpts_mean = statistics.mean(setpts_samples)
    setpts_stderr = stderr(setpts_samples)
    setup_execute_means = [
        setpts_ms + execute_ms
        for setpts_ms, execute_ms in zip(setpts_samples, run_means)
    ]
    setup_execute_mean = statistics.mean(setup_execute_means)
    setup_execute_stderr = stderr(setup_execute_means)
    mpoints_per_second = point_count / (execute_mean * 1_000.0)

    print(f"kind={kind_name}")
    print(f"  plan_creation_ms={format_samples(plan_samples)}")
    print(f"  setpts_ms={format_samples(setpts_samples)}")
    for run_index, values in enumerate(execute_samples, start=1):
        print(f"  execute_run_{run_index}_ms={format_samples(values)}")
    print(f"  execute_run_means_ms={format_samples(run_means)}")
    print(
        "  execute_mean_ms="
        f"{execute_mean:.6f} +/- {execute_stderr:.6f} stderr; "
        f"min_ms={execute_min:.6f}; Mpoints/s={mpoints_per_second:.6f}"
    )
    print(
        f"  setpts_mean_ms={setpts_mean:.6f} +/- "
        f"{setpts_stderr:.6f} stderr"
    )
    print(
        "  setpts_plus_execute_mean_ms="
        f"{setup_execute_mean:.6f} +/- {setup_execute_stderr:.6f} stderr"
    )
    print(
        "RESULT,"
        f"kind={kind_name},N={n_modes},M={point_count},"
        f"execute_ms={execute_mean:.6f},execute_stderr_ms={execute_stderr:.6f},"
        f"execute_min_ms={execute_min:.6f},Mpoints_per_second={mpoints_per_second:.6f},"
        f"setpts_ms={setpts_mean:.6f},setpts_stderr_ms={setpts_stderr:.6f},"
        f"setpts_plus_execute_ms={setup_execute_mean:.6f},"
        f"setpts_plus_execute_stderr_ms={setup_execute_stderr:.6f}"
    )


def cpu_name() -> str:
    if sys.platform == "win32":
        try:
            import winreg

            with winreg.OpenKey(
                winreg.HKEY_LOCAL_MACHINE,
                r"HARDWARE\DESCRIPTION\System\CentralProcessor\0",
            ) as key:
                return str(winreg.QueryValueEx(key, "ProcessorNameString")[0]).strip()
        except OSError:
            pass
    return platform.processor() or "unknown"


def library_metadata() -> tuple[Path | None, str]:
    package_dir = Path(finufft.__file__).resolve().parent
    candidates = list(package_dir.glob("*finufft*.dll"))
    candidates += list(package_dir.glob("*finufft*.so"))
    candidates += list(package_dir.glob("*finufft*.dylib"))
    if not candidates:
        return None, "unavailable"
    library = candidates[0]
    digest = hashlib.sha256(library.read_bytes()).hexdigest()
    return library, digest


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--case",
        action="append",
        type=parse_case,
        dest="cases",
        metavar="N[:M]",
        help="mode and point counts; repeat for multiple cases",
    )
    parser.add_argument("--runs", type=positive_int, default=3)
    parser.add_argument("--samples", type=positive_int, default=DEFAULT_SAMPLES)
    parser.add_argument("--warmups", type=nonnegative_int, default=1)
    parser.add_argument(
        "--threads",
        type=nonnegative_int,
        default=DEFAULT_THREADS,
        help="explicit FINUFFT nthreads (default: 32 for the archive host)",
    )
    parser.add_argument("--eps", type=float, default=1.0e-6)
    parser.add_argument("--seed", type=lambda value: int(value, 0), default=DEFAULT_SEED)
    return parser


def main() -> int:
    args = make_parser().parse_args()
    if not math.isfinite(args.eps) or args.eps <= 0.0:
        raise SystemExit("--eps must be finite and positive")
    cases = args.cases or list(DEFAULT_CASES)
    library, library_sha256 = library_metadata()

    print("FINUFFT CPU context benchmark (not a CPU/GPU equivalence claim)")
    print(f"python={platform.python_version()} executable={sys.executable}")
    print(f"numpy={np.__version__}")
    print(f"finufft={finufft.__version__} package={Path(finufft.__file__).resolve()}")
    print(f"finufft_library={library or 'unavailable'} sha256={library_sha256}")
    print(f"platform={platform.platform()}")
    print(f"cpu={cpu_name()} logical_cpus={os.cpu_count()}")
    print(
        "method=complex64, eps="
        f"{args.eps:.9g}, sigma=2.0, isign=+1, modeord=0 (centered/CMCL), "
        f"nthreads={args.threads} (explicit), runs={args.runs}, "
        f"execute_samples_per_run={args.samples}, warmups_per_run={args.warmups}, "
        "preallocated output"
    )
    print(
        "data=u32 LCG state=1664525*state+1013904223 mod 2^32, "
        f"high 24 bits, seed=0x{args.seed & 0xFFFF_FFFF:08X}; "
        "field-specific streams are prefix-stable across cases"
    )
    print(
        "timing=plan creation, setpts, and execute reported separately; "
        "headline is execute-only; +/- is standard error across plan-recreated run means"
    )

    for n_modes, point_count in cases:
        print()
        print(f"case N={n_modes} M={point_count}")
        points, strengths, modes = make_inputs(n_modes, point_count, args.seed)
        benchmark_kind(
            1,
            n_modes,
            point_count,
            points,
            strengths,
            eps=args.eps,
            threads=args.threads,
            runs=args.runs,
            samples=args.samples,
            warmups=args.warmups,
        )
        benchmark_kind(
            2,
            n_modes,
            point_count,
            points,
            modes,
            eps=args.eps,
            threads=args.threads,
            runs=args.runs,
            samples=args.samples,
            warmups=args.warmups,
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
