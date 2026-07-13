#!/usr/bin/env python3
"""Same-silicon cuFINUFFT context benchmark for wgpu-nufft.

The harness uses reusable single-precision cuFINUFFT plans and GPU-resident
CuPy arrays.  It reports plan creation, setpts, execute, and setpts+execute
separately.  Every timed CUDA span is bounded by an explicit stream
synchronization so asynchronous enqueue time is never mistaken for execution.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import math
import os
import platform
import site
import statistics
import sys
import time
from pathlib import Path
from typing import Callable, Iterable

import numpy as np


DEFAULT_CASES = ((262_144, 262_144), (1_048_576, 1_048_576))
DEFAULT_SEED = 0x4E55_4646
DEFAULT_RUNS = 3
DEFAULT_SAMPLES = 10
DEFAULT_WARMUPS = 1
DEFAULT_TYPE2_BATCH = 32
DEFAULT_EPS = 1.0e-6
LCG_MULTIPLIER = 1_664_525
LCG_INCREMENT = 1_013_904_223
U24_SCALE = np.float32(1.0 / (1 << 24))


def configure_windows_dll_search() -> list[object]:
    """Retain Windows DLL-directory handles for CUDA and delved wheels."""

    if sys.platform != "win32" or not hasattr(os, "add_dll_directory"):
        return []
    candidates: list[Path] = []
    cuda_path = os.environ.get("CUDA_PATH")
    if cuda_path:
        candidates.append(Path(cuda_path) / "bin")
    for package_root in map(Path, site.getsitepackages()):
        candidates.append(package_root / "cufinufft.libs")

    handles = []
    for candidate in candidates:
        if candidate.is_dir():
            handles.append(os.add_dll_directory(str(candidate)))
    return handles


DLL_DIRECTORY_HANDLES = configure_windows_dll_search()

# Import CuPy first so its CUDA dependency loader is initialized before the
# ctypes-based cuFINUFFT loader.  The retained directories above also make the
# installed Windows wheel robust when CUDA is not in Python's safe DLL search.
import cupy as cp  # noqa: E402
import cufinufft  # noqa: E402


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
    """Return the same prefix-stable f32 LCG stream as the Rust harness."""

    state = seed & 0xFFFF_FFFF

    def samples() -> Iterable[int]:
        nonlocal state
        for _ in range(count):
            state = (state * LCG_MULTIPLIER + LCG_INCREMENT) & 0xFFFF_FFFF
            yield state >> 8

    values = np.fromiter(samples(), dtype=np.uint32, count=count)
    return values.astype(np.float32) * U24_SCALE


def make_inputs(n_modes: int, point_count: int, seed: int):
    """Generate the field-specific Phase D inputs without timed transfers."""

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


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def library_metadata() -> tuple[Path, str]:
    package_dir = Path(cufinufft.__file__).resolve().parent
    candidates = list(package_dir.glob("*cufinufft*.dll"))
    candidates += list(package_dir.glob("*cufinufft*.so"))
    candidates += list(package_dir.glob("*cufinufft*.dylib"))
    if not candidates:
        raise RuntimeError("could not locate the loaded cuFINUFFT library")
    library = candidates[0]
    return library, sha256_file(library)


def make_plan(
    kind: int,
    n_modes: int,
    eps: float,
    stream: cp.cuda.Stream,
    device_id: int,
):
    return cufinufft.Plan(
        kind,
        (n_modes,),
        eps=eps,
        isign=1,
        dtype="complex64",
        modeord=0,
        upsampfac=2.0,
        gpu_device_id=device_id,
        gpu_stream=stream.ptr,
    )


def relative_l2(actual: np.ndarray, reference: np.ndarray) -> float:
    numerator = np.linalg.norm(actual.astype(np.complex128) - reference)
    denominator = max(np.linalg.norm(reference), np.finfo(np.float64).tiny)
    return float(numerator / denominator)


def direct_type1(
    points: np.ndarray, strengths: np.ndarray, n_modes: int
) -> np.ndarray:
    modes = np.arange(-(n_modes // 2), (n_modes - 1) // 2 + 1, dtype=np.float64)
    phase = np.exp(1j * np.outer(modes, points.astype(np.float64)))
    return phase @ strengths.astype(np.complex128)


def direct_type2(points: np.ndarray, modes: np.ndarray) -> np.ndarray:
    n_modes = len(modes)
    mode_indices = np.arange(
        -(n_modes // 2), (n_modes - 1) // 2 + 1, dtype=np.float64
    )
    phase = np.exp(1j * np.outer(points.astype(np.float64), mode_indices))
    return phase @ modes.astype(np.complex128)


def run_correctness_smoke(device_id: int, eps: float, seed: int) -> None:
    n_modes = 32
    point_count = 41
    points, strengths, modes = make_inputs(n_modes, point_count, seed)
    points[:6] = np.array(
        [
            -np.float32(math.pi),
            np.nextafter(np.float32(math.pi), np.float32(-math.inf)),
            np.float32(0.0),
            np.float32(0.0),
            np.float32(0.125),
            np.float32(-0.75),
        ],
        dtype=np.float32,
    )

    device = cp.cuda.Device(device_id)
    device.use()
    stream = cp.cuda.Stream(non_blocking=True)
    gpu_points = cp.asarray(points)
    gpu_strengths = cp.asarray(strengths)
    gpu_modes = cp.asarray(modes)
    type1_output = cp.empty(n_modes, dtype=cp.complex64)
    type2_output = cp.empty(point_count, dtype=cp.complex64)
    cp.cuda.runtime.deviceSynchronize()

    type1 = make_plan(1, n_modes, eps, stream, device_id)
    type2 = make_plan(2, n_modes, eps, stream, device_id)
    type1.setpts(gpu_points)
    type2.setpts(gpu_points)
    type1.execute(gpu_strengths, out=type1_output)
    type2.execute(gpu_modes, out=type2_output)
    stream.synchronize()

    actual_type1 = cp.asnumpy(type1_output)
    actual_type2 = cp.asnumpy(type2_output)
    type1_error = relative_l2(
        actual_type1, direct_type1(points, strengths, n_modes)
    )
    type2_error = relative_l2(actual_type2, direct_type2(points, modes))

    zero_points = cp.zeros(point_count, dtype=cp.float32)
    cp.cuda.runtime.deviceSynchronize()
    type1.setpts(zero_points)
    type2.setpts(zero_points)
    type1.execute(gpu_strengths, out=type1_output)
    type2.execute(gpu_modes, out=type2_output)
    stream.synchronize()
    analytic_type1 = np.full(
        n_modes, strengths.astype(np.complex128).sum(), dtype=np.complex128
    )
    analytic_type2 = np.full(
        point_count, modes.astype(np.complex128).sum(), dtype=np.complex128
    )
    analytic_type1_error = relative_l2(cp.asnumpy(type1_output), analytic_type1)
    analytic_type2_error = relative_l2(cp.asnumpy(type2_output), analytic_type2)

    threshold = 20.0 * eps
    print(
        "CUFINUFFT_SMOKE "
        f"N={n_modes} M={point_count} eps={eps:.9g} "
        f"direct_type1_relative_l2={type1_error:.9e} "
        f"direct_type2_relative_l2={type2_error:.9e} "
        f"analytic_type1_relative_l2={analytic_type1_error:.9e} "
        f"analytic_type2_relative_l2={analytic_type2_error:.9e} "
        f"threshold={threshold:.9e}"
    )
    errors = (
        type1_error,
        type2_error,
        analytic_type1_error,
        analytic_type2_error,
    )
    worst = max(errors)
    if any(not math.isfinite(error) for error in errors) or worst > threshold:
        raise RuntimeError(
            f"cuFINUFFT correctness smoke failed: worst relative L2 {worst}"
        )

    del type1, type2
    gc.collect()
    stream.synchronize()


def elapsed_batch_ms(
    operation: Callable[[], None], repeats: int, stream: cp.cuda.Stream
) -> float:
    stream.synchronize()
    start_ns = time.perf_counter_ns()
    for _ in range(repeats):
        operation()
    stream.synchronize()
    return (time.perf_counter_ns() - start_ns) / 1_000_000.0 / repeats


def stderr(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    return statistics.stdev(values) / math.sqrt(len(values))


def format_samples(values: list[float]) -> str:
    return "[" + ", ".join(f"{value:.6f}" for value in values) + "]"


def flatten(values: list[list[float]]) -> list[float]:
    return [value for run in values for value in run]


def benchmark_kind(
    kind: int,
    n_modes: int,
    point_count: int,
    points,
    input_values,
    *,
    eps: float,
    runs: int,
    samples: int,
    warmups: int,
    type2_batch: int,
    device_id: int,
) -> None:
    kind_name = f"type-{kind}"
    transforms_per_sample = 1 if kind == 1 else type2_batch
    output_length = n_modes if kind == 1 else point_count
    output = cp.empty(output_length, dtype=cp.complex64)
    plan_samples: list[float] = []
    setpts_samples: list[list[float]] = []
    execute_samples: list[list[float]] = []
    combined_samples: list[list[float]] = []

    print(
        f"{kind_name}: starting transforms_per_sample={transforms_per_sample}"
    )
    for run_index in range(runs):
        stream = cp.cuda.Stream(non_blocking=True)
        cp.cuda.runtime.deviceSynchronize()
        start_ns = time.perf_counter_ns()
        plan = make_plan(kind, n_modes, eps, stream, device_id)
        stream.synchronize()
        plan_ms = (time.perf_counter_ns() - start_ns) / 1_000_000.0
        plan_samples.append(plan_ms)

        for _ in range(warmups):
            plan.setpts(points)
            plan.execute(input_values, out=output)
        stream.synchronize()

        run_setpts: list[float] = []
        run_execute: list[float] = []
        run_combined: list[float] = []
        for sample_index in range(samples):
            setpts_ms = elapsed_batch_ms(
                lambda: plan.setpts(points), transforms_per_sample, stream
            )
            execute_ms = elapsed_batch_ms(
                lambda: plan.execute(input_values, out=output),
                transforms_per_sample,
                stream,
            )

            def setpts_and_execute() -> None:
                plan.setpts(points)
                plan.execute(input_values, out=output)

            combined_ms = elapsed_batch_ms(
                setpts_and_execute, transforms_per_sample, stream
            )
            run_setpts.append(setpts_ms)
            run_execute.append(execute_ms)
            run_combined.append(combined_ms)
            print(
                f"run {run_index + 1}/{runs} sample {sample_index + 1}/{samples}: "
                f"setpts_ms={setpts_ms:.6f} execute_ms={execute_ms:.6f} "
                f"combined_ms={combined_ms:.6f}"
            )

        setpts_samples.append(run_setpts)
        execute_samples.append(run_execute)
        combined_samples.append(run_combined)
        print(
            f"run {run_index + 1}/{runs} summary: plan_ms={plan_ms:.6f} "
            f"setpts_ms={statistics.mean(run_setpts):.6f} "
            f"execute_ms={statistics.mean(run_execute):.6f} "
            f"combined_ms={statistics.mean(run_combined):.6f}"
        )

        stream.synchronize()
        del plan
        gc.collect()
        stream.synchronize()
        del stream

    setpts_run_means = [statistics.mean(values) for values in setpts_samples]
    execute_run_means = [statistics.mean(values) for values in execute_samples]
    combined_run_means = [statistics.mean(values) for values in combined_samples]
    setpts_mean = statistics.mean(setpts_run_means)
    execute_mean = statistics.mean(execute_run_means)
    combined_mean = statistics.mean(combined_run_means)
    paired_span = "setpts+execute" if kind == 1 else "execute"
    paired_mean = combined_mean if kind == 1 else execute_mean
    mpoints_per_second = point_count / (paired_mean * 1_000.0)

    print(f"  plan_creation_ms={format_samples(plan_samples)}")
    for run_index, values in enumerate(setpts_samples, start=1):
        print(f"  setpts_run_{run_index}_ms={format_samples(values)}")
    for run_index, values in enumerate(execute_samples, start=1):
        print(f"  execute_run_{run_index}_ms={format_samples(values)}")
    for run_index, values in enumerate(combined_samples, start=1):
        print(f"  combined_run_{run_index}_ms={format_samples(values)}")
    print(
        "RESULT,"
        f"kind={kind_name},N={n_modes},M={point_count},eps={eps:.9g},"
        f"runs={runs},samples_per_run={samples},"
        f"transforms_per_sample={transforms_per_sample},"
        f"plan_ms={statistics.mean(plan_samples):.6f},"
        f"plan_stderr_ms={stderr(plan_samples):.6f},"
        f"setpts_raw_ms={format_samples(flatten(setpts_samples))},"
        f"setpts_run_means_ms={format_samples(setpts_run_means)},"
        f"setpts_ms={setpts_mean:.6f},"
        f"setpts_stderr_ms={stderr(setpts_run_means):.6f},"
        f"setpts_min_ms={min(flatten(setpts_samples)):.6f},"
        f"execute_raw_ms={format_samples(flatten(execute_samples))},"
        f"execute_run_means_ms={format_samples(execute_run_means)},"
        f"execute_ms={execute_mean:.6f},"
        f"execute_stderr_ms={stderr(execute_run_means):.6f},"
        f"execute_min_ms={min(flatten(execute_samples)):.6f},"
        f"combined_raw_ms={format_samples(flatten(combined_samples))},"
        f"combined_run_means_ms={format_samples(combined_run_means)},"
        f"combined_ms={combined_mean:.6f},"
        f"combined_stderr_ms={stderr(combined_run_means):.6f},"
        f"combined_min_ms={min(flatten(combined_samples)):.6f},"
        f"paired_span={paired_span},paired_ms={paired_mean:.6f},"
        f"paired_million_points_per_second={mpoints_per_second:.6f},"
        "timing=host-wall-clock-with-explicit-stream-sync,"
        "transfers=excluded,outputs=preallocated"
    )


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
    parser.add_argument("--runs", type=positive_int, default=DEFAULT_RUNS)
    parser.add_argument("--samples", type=positive_int, default=DEFAULT_SAMPLES)
    parser.add_argument("--warmups", type=nonnegative_int, default=DEFAULT_WARMUPS)
    parser.add_argument(
        "--type2-batch", type=positive_int, default=DEFAULT_TYPE2_BATCH
    )
    parser.add_argument("--eps", type=float, default=DEFAULT_EPS)
    parser.add_argument(
        "--seed", type=lambda value: int(value, 0), default=DEFAULT_SEED
    )
    parser.add_argument("--device", type=nonnegative_int, default=0)
    parser.add_argument("--smoke-only", action="store_true")
    return parser


def main() -> int:
    args = make_parser().parse_args()
    if not math.isfinite(args.eps) or args.eps <= 0.0:
        raise SystemExit("--eps must be finite and positive")

    device = cp.cuda.Device(args.device)
    device.use()
    properties = cp.cuda.runtime.getDeviceProperties(args.device)
    device_name = properties["name"]
    if isinstance(device_name, bytes):
        device_name = device_name.decode(errors="replace")
    library, library_sha256 = library_metadata()
    cases = args.cases or list(DEFAULT_CASES)

    print("cuFINUFFT 1D GPU benchmark")
    print(f"python={platform.python_version()} executable={sys.executable}")
    print(f"numpy={np.__version__}")
    print(f"cupy={cp.__version__} package={Path(cp.__file__).resolve()}")
    print(
        f"cufinufft={cufinufft.__version__} "
        f"package={Path(cufinufft.__file__).resolve()}"
    )
    print(f"cufinufft_library={library} sha256={library_sha256}")
    print(
        f"device_id={args.device} name={device_name} "
        f"compute_capability={device.compute_capability} "
        f"pci_bus_id={device.pci_bus_id}"
    )
    print(
        f"cuda_driver={cp.cuda.runtime.driverGetVersion()} "
        f"cuda_runtime_linked={cp.cuda.runtime.runtimeGetVersion()} "
        f"cuda_path={os.environ.get('CUDA_PATH', 'unset')}"
    )
    print(f"platform={platform.platform()}")
    print(
        "method=complex64, eps="
        f"{args.eps:.9g}, sigma=2.0, isign=+1, modeord=0 "
        f"(centered/CMCL), runs={args.runs}, samples={args.samples}, "
        f"warmups={args.warmups}, type1_batch=1, "
        f"type2_batch={args.type2_batch}, preallocated outputs"
    )
    print(
        "timing=GPU-resident host wall clock around plan/setpts/execute spans; "
        "explicit stream synchronization before and after every timed span; "
        "H2D/D2H and input generation excluded"
    )
    print(
        "pairing=wgpu type-1 submit-wait versus cuFINUFFT setpts+execute; "
        "wgpu type-2 submit-wait versus cuFINUFFT execute; all cuFINUFFT "
        "spans are also reported"
    )
    print(
        "data=u32 LCG state=1664525*state+1013904223 mod 2^32, "
        f"high 24 bits, seed=0x{args.seed & 0xFFFF_FFFF:08X}; "
        "field-specific streams match the Rust benchmark"
    )

    run_correctness_smoke(args.device, args.eps, args.seed)
    if args.smoke_only:
        return 0

    for n_modes, point_count in cases:
        print()
        print(f"case N={n_modes} M={point_count}")
        points, strengths, modes = make_inputs(n_modes, point_count, args.seed)
        gpu_points = cp.asarray(points)
        gpu_strengths = cp.asarray(strengths)
        gpu_modes = cp.asarray(modes)
        cp.cuda.runtime.deviceSynchronize()
        print(
            f"device_inputs points_bytes={gpu_points.nbytes} "
            f"strengths_bytes={gpu_strengths.nbytes} modes_bytes={gpu_modes.nbytes}"
        )
        benchmark_kind(
            1,
            n_modes,
            point_count,
            gpu_points,
            gpu_strengths,
            eps=args.eps,
            runs=args.runs,
            samples=args.samples,
            warmups=args.warmups,
            type2_batch=args.type2_batch,
            device_id=args.device,
        )
        benchmark_kind(
            2,
            n_modes,
            point_count,
            gpu_points,
            gpu_modes,
            eps=args.eps,
            runs=args.runs,
            samples=args.samples,
            warmups=args.warmups,
            type2_batch=args.type2_batch,
            device_id=args.device,
        )
        cp.cuda.runtime.deviceSynchronize()
        del gpu_points, gpu_strengths, gpu_modes
        gc.collect()
        cp.get_default_memory_pool().free_all_blocks()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
