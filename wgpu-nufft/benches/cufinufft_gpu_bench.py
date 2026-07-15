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
DEFAULT_CASES_2D = (
    (512, 512, 512 * 512),
    (1_024, 1_024, 1_024 * 1_024),
)
DEFAULT_CASES_3D = (
    (64, 64, 64, 64 * 64 * 64),
    (128, 128, 128, 128 * 128 * 128),
)
DEFAULT_CASES_TYPE3 = ((65_536, 65_536),)
DEFAULT_SEED = 0x4E55_4646
DEFAULT_RUNS = 3
DEFAULT_SAMPLES = 10
DEFAULT_WARMUPS = 1
DEFAULT_TYPE2_BATCH = 32
DEFAULT_EPS = 1.0e-6
DEFAULT_TYPE3_SOURCE_HALFWIDTH = math.pi
DEFAULT_TYPE3_TARGET_HALFWIDTH = 16.0
DEFAULT_TYPE3_SOURCE_CENTER = math.pi / 4.0
DEFAULT_TYPE3_TARGET_CENTER = 4.0
UPSAMPFAC = 2.0
LCG_MULTIPLIER = 1_664_525
LCG_INCREMENT = 1_013_904_223
U24_SCALE = np.float32(1.0 / (1 << 24))
POINT_Y_SEED_MASK = 0xB7E1_5162
POINT_Z_SEED_MASK = 0x9E37_79B9
TYPE3_TARGET_SEED_MASKS = (0x243F_6A88, 0x85A3_08D3, 0x1319_8A2E)
SOURCE_POINT_SEED_MASKS = (0xA341_316C, POINT_Y_SEED_MASK, POINT_Z_SEED_MASK)


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


def parse_case_2d(value: str) -> tuple[int, int, int]:
    fields = value.split(":")
    if len(fields) > 2:
        raise argparse.ArgumentTypeError("2D case must be N0xN1 or N0xN1:M")
    dimensions = fields[0].lower().split("x")
    if len(dimensions) != 2:
        raise argparse.ArgumentTypeError("2D case must be N0xN1 or N0xN1:M")
    n0, n1 = map(positive_int, dimensions)
    mode_count = n0 * n1
    point_count = mode_count if len(fields) == 1 else positive_int(fields[1])
    return n0, n1, point_count


def parse_case_3d(value: str) -> tuple[int, int, int, int]:
    fields = value.split(":")
    if len(fields) > 2:
        raise argparse.ArgumentTypeError(
            "3D case must be N0xN1xN2 or N0xN1xN2:M"
        )
    dimensions = fields[0].lower().split("x")
    if len(dimensions) != 3:
        raise argparse.ArgumentTypeError(
            "3D case must be N0xN1xN2 or N0xN1xN2:M"
        )
    n0, n1, n2 = map(positive_int, dimensions)
    mode_count = n0 * n1 * n2
    point_count = mode_count if len(fields) == 1 else positive_int(fields[1])
    return n0, n1, n2, point_count


def parse_case_type3(value: str) -> tuple[int, int]:
    fields = value.split(":")
    if len(fields) == 1:
        source_count = target_count = positive_int(fields[0])
    elif len(fields) == 2:
        source_count, target_count = map(positive_int, fields)
    else:
        raise argparse.ArgumentTypeError("type-3 case must be M or M:K")
    return source_count, target_count


def normalized_mode_shape(n_modes: int | tuple[int, ...]) -> tuple[int, ...]:
    return (n_modes,) if isinstance(n_modes, int) else tuple(n_modes)


def cufinufft_mode_shape(n_modes: int | tuple[int, ...]) -> tuple[int, ...]:
    """Return the C/Python shape whose last (x) axis is wgpu axis zero."""

    return normalized_mode_shape(n_modes)[::-1]


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


def make_inputs_2d(n0: int, n1: int, point_count: int, seed: int):
    """Generate axis-zero-fast 2D inputs matching the Rust benchmark."""

    points_x_u = lcg_uniform(point_count, seed ^ 0xA341_316C)
    points_y_u = lcg_uniform(point_count, seed ^ POINT_Y_SEED_MASK)
    strength_re = lcg_uniform(point_count, seed ^ 0xC801_3EA4)
    strength_im = lcg_uniform(point_count, seed ^ 0xAD90_777D)
    mode_count = n0 * n1
    mode_re = lcg_uniform(mode_count, seed ^ 0x7E95_761E)
    mode_im = lcg_uniform(mode_count, seed ^ 0x6C8E_9CF5)

    def coordinates(values: np.ndarray) -> np.ndarray:
        return np.ascontiguousarray(
            values * np.float32(2.0 * math.pi) - np.float32(math.pi),
            dtype=np.float32,
        )

    strengths = np.empty(point_count, dtype=np.complex64)
    strengths.real = strength_re * np.float32(2.0) - np.float32(1.0)
    strengths.imag = strength_im * np.float32(2.0) - np.float32(1.0)
    modes = np.empty(mode_count, dtype=np.complex64)
    modes.real = mode_re * np.float32(2.0) - np.float32(1.0)
    modes.imag = mode_im * np.float32(2.0) - np.float32(1.0)
    # cuFINUFFT's Python arrays are C ordered `(n1, n0)`, so their last
    # dimension is the same axis-zero-fast layout used by wgpu-nufft.
    modes = modes.reshape((n1, n0))
    return (coordinates(points_x_u), coordinates(points_y_u)), strengths, modes


def make_inputs_3d(
    n0: int, n1: int, n2: int, point_count: int, seed: int
):
    """Generate axis-zero-fast 3D inputs matching the Rust benchmark."""

    point_streams = [
        lcg_uniform(point_count, seed ^ mask)
        for mask in SOURCE_POINT_SEED_MASKS
    ]
    strength_re = lcg_uniform(point_count, seed ^ 0xC801_3EA4)
    strength_im = lcg_uniform(point_count, seed ^ 0xAD90_777D)
    mode_count = n0 * n1 * n2
    mode_re = lcg_uniform(mode_count, seed ^ 0x7E95_761E)
    mode_im = lcg_uniform(mode_count, seed ^ 0x6C8E_9CF5)

    def coordinates(values: np.ndarray) -> np.ndarray:
        return np.ascontiguousarray(
            values * np.float32(2.0 * math.pi) - np.float32(math.pi),
            dtype=np.float32,
        )

    strengths = np.empty(point_count, dtype=np.complex64)
    strengths.real = strength_re * np.float32(2.0) - np.float32(1.0)
    strengths.imag = strength_im * np.float32(2.0) - np.float32(1.0)
    modes = np.empty(mode_count, dtype=np.complex64)
    modes.real = mode_re * np.float32(2.0) - np.float32(1.0)
    modes.imag = mode_im * np.float32(2.0) - np.float32(1.0)
    # C shape (N2,N1,N0) keeps logical axis zero contiguous/fastest.
    modes = modes.reshape((n2, n1, n0))
    return tuple(coordinates(values) for values in point_streams), strengths, modes


def stack_distinct_transforms(values: np.ndarray, n_trans: int) -> np.ndarray:
    """Return transform-major complex64 vectors with deterministic differences."""

    if n_trans == 1:
        return values
    factors = np.empty(n_trans, dtype=np.complex64)
    transform = np.arange(n_trans, dtype=np.float32)
    factors.real = np.float32(1.0) + transform * np.float32(1.0 / 32.0)
    factors.imag = transform * np.float32(-1.0 / 64.0)
    expanded = factors.reshape((n_trans,) + (1,) * values.ndim)
    return np.ascontiguousarray(expanded * values[np.newaxis, ...])


def affine_coordinates(
    values: np.ndarray, halfwidth: float, center: float
) -> np.ndarray:
    """Map unit samples to a fixed type-3 interval and pin both endpoints."""

    lower = np.float32(center - halfwidth)
    upper = np.float32(center + halfwidth)
    coordinates = np.ascontiguousarray(
        center
        + (values.astype(np.float64) * 2.0 - 1.0) * halfwidth,
        dtype=np.float32,
    )
    if len(coordinates) >= 1:
        coordinates[0] = lower
    if len(coordinates) >= 2:
        coordinates[1] = upper
    return coordinates


def make_inputs_type3(
    dimension: int,
    source_count: int,
    target_count: int,
    seed: int,
    source_halfwidth: float,
    target_halfwidth: float,
    source_center: float,
    target_center: float,
):
    """Generate deterministic unrestricted NU-to-NU points and strengths."""

    source_points = tuple(
        affine_coordinates(
            lcg_uniform(source_count, seed ^ SOURCE_POINT_SEED_MASKS[axis]),
            source_halfwidth,
            source_center,
        )
        for axis in range(dimension)
    )
    target_points = tuple(
        affine_coordinates(
            lcg_uniform(target_count, seed ^ TYPE3_TARGET_SEED_MASKS[axis]),
            target_halfwidth,
            target_center,
        )
        for axis in range(dimension)
    )
    strength_re = lcg_uniform(source_count, seed ^ 0xC801_3EA4)
    strength_im = lcg_uniform(source_count, seed ^ 0xAD90_777D)
    strengths = np.empty(source_count, dtype=np.complex64)
    strengths.real = strength_re * np.float32(2.0) - np.float32(1.0)
    strengths.imag = strength_im * np.float32(2.0) - np.float32(1.0)
    return source_points, target_points, strengths


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
    n_modes: int | tuple[int, ...],
    eps: float,
    stream: cp.cuda.Stream,
    device_id: int,
    n_trans: int = 1,
):
    native_batch_options = (
        {"gpu_maxbatchsize": min(n_trans, 8)} if n_trans > 1 else {}
    )
    return cufinufft.Plan(
        kind,
        cufinufft_mode_shape(n_modes),
        n_trans=n_trans,
        eps=eps,
        isign=1,
        dtype="complex64",
        modeord=0,
        upsampfac=UPSAMPFAC,
        gpu_device_id=device_id,
        gpu_stream=stream.ptr,
        **native_batch_options,
    )


def make_type3_plan(
    dimension: int,
    eps: float,
    isign: int,
    stream: cp.cuda.Stream,
    device_id: int,
):
    # A type-3 plan takes the dimension as an integer, not a mode shape.
    # modeord is intentionally omitted because FINUFFT defines it as a no-op
    # for NU-to-NU transforms.
    return cufinufft.Plan(
        3,
        dimension,
        eps=eps,
        isign=isign,
        dtype="complex64",
        upsampfac=UPSAMPFAC,
        gpu_device_id=device_id,
        gpu_stream=stream.ptr,
    )


def set_plan_points(plan, points) -> None:
    if isinstance(points, tuple):
        # The Python wrapper treats n_modes as C array shape `(N1, N0)` and
        # reverses both that shape and the setpts arguments for cuFINUFFT's
        # column-major C API.  Our tuple is in wgpu logical order `(x0, x1)`,
        # so present it in Python array-axis order `(x1, x0)` here.
        plan.setpts(*points[::-1])
    else:
        plan.setpts(points)


def set_type3_plan_points(plan, source_points, target_points) -> None:
    """Set logical axis-zero-fast type-3 source and target coordinates."""

    dimension = len(source_points)
    if len(target_points) != dimension:
        raise ValueError("source and target dimensions must match")
    # The Python wrapper reverses its coordinate axes for the column-major C
    # API. Present both tuples in Python array-axis order so low-level x is
    # logical axis zero, matching wgpu-nufft.
    plan.setpts(
        *source_points[::-1],
        *((None,) * (3 - dimension)),
        *target_points[::-1],
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


def centered_modes(length: int) -> np.ndarray:
    return np.arange(-(length // 2), (length - 1) // 2 + 1, dtype=np.float64)


def direct_type1_2d(
    points: tuple[np.ndarray, np.ndarray],
    strengths: np.ndarray,
    n_modes: tuple[int, int],
) -> np.ndarray:
    n0, n1 = n_modes
    k0 = np.tile(centered_modes(n0), n1)
    k1 = np.repeat(centered_modes(n1), n0)
    x, y = points
    phase = np.exp(
        1j
        * (
            np.outer(k0, x.astype(np.float64))
            + np.outer(k1, y.astype(np.float64))
        )
    )
    return (phase @ strengths.astype(np.complex128)).reshape((n1, n0))


def direct_type2_2d(
    points: tuple[np.ndarray, np.ndarray], modes: np.ndarray
) -> np.ndarray:
    n1, n0 = modes.shape
    k0 = np.tile(centered_modes(n0), n1)
    k1 = np.repeat(centered_modes(n1), n0)
    x, y = points
    phase = np.exp(
        1j
        * (
            np.outer(x.astype(np.float64), k0)
            + np.outer(y.astype(np.float64), k1)
        )
    )
    return phase @ modes.astype(np.complex128).reshape(-1)


def direct_type1_3d(
    points: tuple[np.ndarray, np.ndarray, np.ndarray],
    strengths: np.ndarray,
    n_modes: tuple[int, int, int],
) -> np.ndarray:
    n0, n1, n2 = n_modes
    k0 = np.tile(centered_modes(n0), n1 * n2)
    k1 = np.tile(np.repeat(centered_modes(n1), n0), n2)
    k2 = np.repeat(centered_modes(n2), n0 * n1)
    x, y, z = points
    phase = np.exp(
        1j
        * (
            np.outer(k0, x.astype(np.float64))
            + np.outer(k1, y.astype(np.float64))
            + np.outer(k2, z.astype(np.float64))
        )
    )
    return (phase @ strengths.astype(np.complex128)).reshape((n2, n1, n0))


def direct_type2_3d(
    points: tuple[np.ndarray, np.ndarray, np.ndarray], modes: np.ndarray
) -> np.ndarray:
    n2, n1, n0 = modes.shape
    k0 = np.tile(centered_modes(n0), n1 * n2)
    k1 = np.tile(np.repeat(centered_modes(n1), n0), n2)
    k2 = np.repeat(centered_modes(n2), n0 * n1)
    x, y, z = points
    phase = np.exp(
        1j
        * (
            np.outer(x.astype(np.float64), k0)
            + np.outer(y.astype(np.float64), k1)
            + np.outer(z.astype(np.float64), k2)
        )
    )
    return phase @ modes.astype(np.complex128).reshape(-1)


def direct_type3(
    source_points: tuple[np.ndarray, ...],
    strengths: np.ndarray,
    target_points: tuple[np.ndarray, ...],
    isign: int,
) -> np.ndarray:
    phase = sum(
        np.outer(target.astype(np.float64), source.astype(np.float64))
        for source, target in zip(source_points, target_points, strict=True)
    )
    return np.exp((1j if isign >= 0 else -1j) * phase) @ strengths.astype(
        np.complex128
    )


def point_bounds(points: tuple[np.ndarray, ...]) -> list[tuple[float, float]]:
    return [(float(axis.min()), float(axis.max())) for axis in points]


def format_axis_values(values: Iterable[float]) -> str:
    return "[" + "|".join(f"{value:.9g}" for value in values) + "]"


def format_axis_bounds(bounds: Iterable[tuple[float, float]]) -> str:
    return "[" + "|".join(f"{lower:.9g}:{upper:.9g}" for lower, upper in bounds) + "]"


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
    set_plan_points(type1, gpu_points)
    set_plan_points(type2, gpu_points)
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
    set_plan_points(type1, zero_points)
    set_plan_points(type2, zero_points)
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


def run_correctness_smoke_2d(device_id: int, eps: float, seed: int) -> None:
    n_modes = (8, 6)
    point_count = 41
    points, strengths, modes = make_inputs_2d(*n_modes, point_count, seed)
    points[0][:6] = np.array(
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
    points[1][:6] = np.array(
        [
            np.float32(math.pi),
            -np.float32(math.pi),
            np.float32(0.0),
            np.float32(0.0),
            np.float32(-0.375),
            np.float32(0.5),
        ],
        dtype=np.float32,
    )

    device = cp.cuda.Device(device_id)
    device.use()
    stream = cp.cuda.Stream(non_blocking=True)
    gpu_points = tuple(cp.asarray(axis) for axis in points)
    gpu_strengths = cp.asarray(strengths)
    gpu_modes = cp.asarray(modes)
    type1_output = cp.empty(cufinufft_mode_shape(n_modes), dtype=cp.complex64)
    type2_output = cp.empty(point_count, dtype=cp.complex64)
    cp.cuda.runtime.deviceSynchronize()

    type1 = make_plan(1, n_modes, eps, stream, device_id)
    type2 = make_plan(2, n_modes, eps, stream, device_id)
    set_plan_points(type1, gpu_points)
    set_plan_points(type2, gpu_points)
    type1.execute(gpu_strengths, out=type1_output)
    type2.execute(gpu_modes, out=type2_output)
    stream.synchronize()

    type1_error = relative_l2(
        cp.asnumpy(type1_output), direct_type1_2d(points, strengths, n_modes)
    )
    type2_error = relative_l2(
        cp.asnumpy(type2_output), direct_type2_2d(points, modes)
    )

    zero_points = tuple(cp.zeros(point_count, dtype=cp.float32) for _ in range(2))
    cp.cuda.runtime.deviceSynchronize()
    set_plan_points(type1, zero_points)
    set_plan_points(type2, zero_points)
    type1.execute(gpu_strengths, out=type1_output)
    type2.execute(gpu_modes, out=type2_output)
    stream.synchronize()
    analytic_type1 = np.full(
        cufinufft_mode_shape(n_modes),
        strengths.astype(np.complex128).sum(),
        dtype=np.complex128,
    )
    analytic_type2 = np.full(
        point_count, modes.astype(np.complex128).sum(), dtype=np.complex128
    )
    analytic_type1_error = relative_l2(cp.asnumpy(type1_output), analytic_type1)
    analytic_type2_error = relative_l2(cp.asnumpy(type2_output), analytic_type2)

    threshold = 20.0 * eps
    print(
        "CUFINUFFT_SMOKE "
        f"dimensions=2 N0={n_modes[0]} N1={n_modes[1]} M={point_count} "
        f"eps={eps:.9g} direct_type1_relative_l2={type1_error:.9e} "
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
            f"cuFINUFFT 2D correctness smoke failed: worst relative L2 {worst}"
        )

    del type1, type2
    gc.collect()
    stream.synchronize()


def run_correctness_smoke_3d(device_id: int, eps: float, seed: int) -> None:
    n_modes = (8, 6, 5)
    point_count = 41
    points, strengths, modes = make_inputs_3d(*n_modes, point_count, seed)
    points[0][:6] = np.array(
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
    points[1][:6] = np.array(
        [
            np.float32(math.pi),
            -np.float32(math.pi),
            np.float32(0.0),
            np.float32(0.0),
            np.float32(-0.375),
            np.float32(0.5),
        ],
        dtype=np.float32,
    )
    points[2][:6] = np.array(
        [
            np.nextafter(np.float32(math.pi), np.float32(-math.inf)),
            -np.float32(math.pi),
            np.float32(0.0),
            np.float32(0.0),
            np.float32(0.625),
            np.float32(-0.25),
        ],
        dtype=np.float32,
    )

    device = cp.cuda.Device(device_id)
    device.use()
    stream = cp.cuda.Stream(non_blocking=True)
    gpu_points = tuple(cp.asarray(axis) for axis in points)
    gpu_strengths = cp.asarray(strengths)
    gpu_modes = cp.asarray(modes)
    type1_output = cp.empty(cufinufft_mode_shape(n_modes), dtype=cp.complex64)
    type2_output = cp.empty(point_count, dtype=cp.complex64)
    cp.cuda.runtime.deviceSynchronize()

    type1 = make_plan(1, n_modes, eps, stream, device_id)
    type2 = make_plan(2, n_modes, eps, stream, device_id)
    set_plan_points(type1, gpu_points)
    set_plan_points(type2, gpu_points)
    type1.execute(gpu_strengths, out=type1_output)
    type2.execute(gpu_modes, out=type2_output)
    stream.synchronize()

    type1_error = relative_l2(
        cp.asnumpy(type1_output), direct_type1_3d(points, strengths, n_modes)
    )
    type2_error = relative_l2(
        cp.asnumpy(type2_output), direct_type2_3d(points, modes)
    )

    zero_points = tuple(cp.zeros(point_count, dtype=cp.float32) for _ in range(3))
    cp.cuda.runtime.deviceSynchronize()
    set_plan_points(type1, zero_points)
    set_plan_points(type2, zero_points)
    type1.execute(gpu_strengths, out=type1_output)
    type2.execute(gpu_modes, out=type2_output)
    stream.synchronize()
    analytic_type1 = np.full(
        cufinufft_mode_shape(n_modes),
        strengths.astype(np.complex128).sum(),
        dtype=np.complex128,
    )
    analytic_type2 = np.full(
        point_count, modes.astype(np.complex128).sum(), dtype=np.complex128
    )
    analytic_type1_error = relative_l2(cp.asnumpy(type1_output), analytic_type1)
    analytic_type2_error = relative_l2(cp.asnumpy(type2_output), analytic_type2)

    threshold = 30.0 * eps
    print(
        "CUFINUFFT_SMOKE "
        f"dimensions=3 N0={n_modes[0]} N1={n_modes[1]} N2={n_modes[2]} "
        f"M={point_count} eps={eps:.9g} "
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
            f"cuFINUFFT 3D correctness smoke failed: worst relative L2 {worst}"
        )

    del type1, type2
    gc.collect()
    stream.synchronize()


def run_native_batch_correctness_smoke(
    dimension: int, n_trans: int, device_id: int, eps: float, seed: int
) -> None:
    """Check cuFINUFFT's transform-major native many-vector interface."""

    if n_trans <= 1:
        return
    point_count = 17
    if dimension == 3:
        n_modes: int | tuple[int, ...] = (5, 4, 3)
        points, strengths, modes = make_inputs_3d(*n_modes, point_count, seed)
        type1_reference = direct_type1_3d(points, strengths, n_modes)
        type2_reference = direct_type2_3d(points, modes)
    elif dimension == 2:
        n_modes = (6, 5)
        points, strengths, modes = make_inputs_2d(*n_modes, point_count, seed)
        type1_reference = direct_type1_2d(points, strengths, n_modes)
        type2_reference = direct_type2_2d(points, modes)
    else:
        n_modes = 30
        points, strengths, modes = make_inputs(n_modes, point_count, seed)
        type1_reference = direct_type1(points, strengths, n_modes)
        type2_reference = direct_type2(points, modes)

    batched_strengths = stack_distinct_transforms(strengths, n_trans)
    batched_modes = stack_distinct_transforms(modes, n_trans)
    batched_type1_reference = stack_distinct_transforms(type1_reference, n_trans)
    batched_type2_reference = stack_distinct_transforms(type2_reference, n_trans)
    gpu_points = (
        tuple(cp.asarray(axis) for axis in points)
        if isinstance(points, tuple)
        else cp.asarray(points)
    )
    gpu_strengths = cp.asarray(batched_strengths)
    gpu_modes = cp.asarray(batched_modes)
    type1_output = cp.empty(
        (n_trans, *cufinufft_mode_shape(n_modes)), dtype=cp.complex64
    )
    type2_output = cp.empty((n_trans, point_count), dtype=cp.complex64)
    stream = cp.cuda.Stream(non_blocking=True)
    type1 = make_plan(1, n_modes, eps, stream, device_id, n_trans)
    type2 = make_plan(2, n_modes, eps, stream, device_id, n_trans)
    set_plan_points(type1, gpu_points)
    set_plan_points(type2, gpu_points)
    type1.execute(gpu_strengths, out=type1_output)
    type2.execute(gpu_modes, out=type2_output)
    stream.synchronize()

    type1_error = relative_l2(cp.asnumpy(type1_output), batched_type1_reference)
    type2_error = relative_l2(cp.asnumpy(type2_output), batched_type2_reference)
    threshold = (30.0 if dimension == 3 else 20.0) * eps
    print(
        "CUFINUFFT_NATIVE_BATCH_SMOKE "
        f"dimensions={dimension} ntrans={n_trans} M={point_count} "
        f"eps={eps:.9g} type1_relative_l2={type1_error:.9e} "
        f"type2_relative_l2={type2_error:.9e} threshold={threshold:.9e}"
    )
    errors = (type1_error, type2_error)
    if any(not math.isfinite(error) for error in errors) or max(errors) > threshold:
        raise RuntimeError(
            "cuFINUFFT native-batch correctness smoke failed: "
            f"worst relative L2 {max(errors)}"
        )

    del type1, type2
    gc.collect()
    stream.synchronize()


def run_correctness_smoke_type3(
    dimension: int,
    device_id: int,
    eps: float,
    seed: int,
    source_halfwidth: float,
    target_halfwidth: float,
    source_center: float,
    target_center: float,
) -> None:
    source_count = 7
    target_count = 9
    source_points, target_points, strengths = make_inputs_type3(
        dimension,
        source_count,
        target_count,
        seed,
        source_halfwidth,
        target_halfwidth,
        source_center,
        target_center,
    )

    device = cp.cuda.Device(device_id)
    device.use()
    stream = cp.cuda.Stream(non_blocking=True)
    gpu_source_points = tuple(cp.asarray(axis) for axis in source_points)
    gpu_target_points = tuple(cp.asarray(axis) for axis in target_points)
    gpu_strengths = cp.asarray(strengths)
    output = cp.empty(target_count, dtype=cp.complex64)
    cp.cuda.runtime.deviceSynchronize()

    errors: dict[int, float] = {}
    for isign in (1, -1):
        plan = make_type3_plan(dimension, eps, isign, stream, device_id)
        set_type3_plan_points(plan, gpu_source_points, gpu_target_points)
        plan.execute(gpu_strengths, out=output)
        stream.synchronize()
        errors[isign] = relative_l2(
            cp.asnumpy(output),
            direct_type3(source_points, strengths, target_points, isign),
        )
        # cuFINUFFT plan destruction uses its configured CUDA stream.
        del plan
        gc.collect()
        stream.synchronize()

    threshold = 100.0 * eps
    print(
        "CUFINUFFT_SMOKE "
        f"kind=type-3 dimensions={dimension} M_sources={source_count} "
        f"K_targets={target_count} eps={eps:.9g} "
        f"positive_relative_l2={errors[1]:.9e} "
        f"negative_relative_l2={errors[-1]:.9e} "
        f"source_bounds={format_axis_bounds(point_bounds(source_points))} "
        f"target_bounds={format_axis_bounds(point_bounds(target_points))} "
        f"threshold={threshold:.9e}"
    )
    worst = max(errors.values())
    if any(not math.isfinite(error) for error in errors.values()) or worst > threshold:
        raise RuntimeError(
            f"cuFINUFFT type-3 correctness smoke failed: worst relative L2 {worst}"
        )

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
    n_modes: int | tuple[int, ...],
    point_count: int,
    points,
    input_values,
    *,
    eps: float,
    runs: int,
    samples: int,
    warmups: int,
    type2_batch: int,
    n_trans: int,
    device_id: int,
) -> None:
    kind_name = f"type-{kind}"
    native_batch = n_trans > 1
    operation_repeats = 1 if native_batch or kind == 1 else type2_batch
    transforms_per_sample = n_trans if native_batch else operation_repeats
    single_output_shape = (
        cufinufft_mode_shape(n_modes) if kind == 1 else (point_count,)
    )
    output_shape = (
        (n_trans, *single_output_shape) if native_batch else single_output_shape
    )
    output = cp.empty(output_shape, dtype=cp.complex64)
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
        plan = make_plan(kind, n_modes, eps, stream, device_id, n_trans)
        stream.synchronize()
        plan_ms = (time.perf_counter_ns() - start_ns) / 1_000_000.0
        plan_samples.append(plan_ms)

        for _ in range(warmups):
            set_plan_points(plan, points)
            plan.execute(input_values, out=output)
        stream.synchronize()

        run_setpts: list[float] = []
        run_execute: list[float] = []
        run_combined: list[float] = []
        for sample_index in range(samples):
            setpts_ms = elapsed_batch_ms(
                lambda: set_plan_points(plan, points), operation_repeats, stream
            )
            execute_ms = elapsed_batch_ms(
                lambda: plan.execute(input_values, out=output),
                operation_repeats,
                stream,
            )

            def setpts_and_execute() -> None:
                set_plan_points(plan, points)
                plan.execute(input_values, out=output)

            combined_ms = elapsed_batch_ms(
                setpts_and_execute, operation_repeats, stream
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
    paired_per_vector = paired_mean / n_trans
    mpoints_per_second = point_count * n_trans / (paired_mean * 1_000.0)

    print(f"  plan_creation_ms={format_samples(plan_samples)}")
    for run_index, values in enumerate(setpts_samples, start=1):
        print(f"  setpts_run_{run_index}_ms={format_samples(values)}")
    for run_index, values in enumerate(execute_samples, start=1):
        print(f"  execute_run_{run_index}_ms={format_samples(values)}")
    for run_index, values in enumerate(combined_samples, start=1):
        print(f"  combined_run_{run_index}_ms={format_samples(values)}")
    mode_shape = normalized_mode_shape(n_modes)
    shape_fields = (
        f"N={mode_shape[0]}"
        if len(mode_shape) == 1
        else (
            f"dimensions=2,N0={mode_shape[0]},N1={mode_shape[1]},"
            f"N_total={math.prod(mode_shape)}"
            if len(mode_shape) == 2
            else (
                f"dimensions=3,N0={mode_shape[0]},N1={mode_shape[1]},"
                f"N2={mode_shape[2]},N_total={math.prod(mode_shape)}"
            )
        )
    )
    print(
        "RESULT,"
        f"kind={kind_name},{shape_fields},M={point_count},eps={eps:.9g},"
        f"runs={runs},samples_per_run={samples},"
        f"batch_mode={'native-ntrans' if native_batch else 'legacy-repeat'},"
        f"ntrans={n_trans},"
        f"transforms_per_sample={transforms_per_sample},"
        f"plan_ms={statistics.mean(plan_samples):.6f},"
        f"plan_stderr_ms={stderr(plan_samples):.6f},"
        f"setpts_raw_ms={format_samples(flatten(setpts_samples))},"
        f"setpts_run_means_ms={format_samples(setpts_run_means)},"
        f"setpts_ms={setpts_mean:.6f},"
        f"setpts_ms_per_vector={setpts_mean / n_trans:.6f},"
        f"setpts_stderr_ms={stderr(setpts_run_means):.6f},"
        f"setpts_min_ms={min(flatten(setpts_samples)):.6f},"
        f"execute_raw_ms={format_samples(flatten(execute_samples))},"
        f"execute_run_means_ms={format_samples(execute_run_means)},"
        f"execute_ms={execute_mean:.6f},"
        f"execute_ms_per_vector={execute_mean / n_trans:.6f},"
        f"execute_stderr_ms={stderr(execute_run_means):.6f},"
        f"execute_min_ms={min(flatten(execute_samples)):.6f},"
        f"combined_raw_ms={format_samples(flatten(combined_samples))},"
        f"combined_run_means_ms={format_samples(combined_run_means)},"
        f"combined_ms={combined_mean:.6f},"
        f"combined_ms_per_vector={combined_mean / n_trans:.6f},"
        f"combined_stderr_ms={stderr(combined_run_means):.6f},"
        f"combined_min_ms={min(flatten(combined_samples)):.6f},"
        f"paired_span={paired_span},paired_ms={paired_mean:.6f},"
        f"paired_ms_per_vector={paired_per_vector:.6f},"
        f"paired_million_points_per_second={mpoints_per_second:.6f},"
        "timing=host-wall-clock-with-explicit-stream-sync,"
        "transfers=excluded,outputs=preallocated"
    )


def benchmark_type3(
    dimension: int,
    source_points,
    target_points,
    input_values,
    *,
    source_count: int,
    target_count: int,
    eps: float,
    runs: int,
    samples: int,
    warmups: int,
    device_id: int,
) -> None:
    output = cp.empty(target_count, dtype=cp.complex64)
    plan_samples: list[float] = []
    setpts_samples: list[list[float]] = []
    execute_samples: list[list[float]] = []
    combined_samples: list[list[float]] = []

    print("type-3: starting transforms_per_sample=1")
    for run_index in range(runs):
        stream = cp.cuda.Stream(non_blocking=True)
        cp.cuda.runtime.deviceSynchronize()
        start_ns = time.perf_counter_ns()
        plan = make_type3_plan(dimension, eps, 1, stream, device_id)
        stream.synchronize()
        plan_ms = (time.perf_counter_ns() - start_ns) / 1_000_000.0
        plan_samples.append(plan_ms)

        for _ in range(warmups):
            set_type3_plan_points(plan, source_points, target_points)
            plan.execute(input_values, out=output)
        stream.synchronize()

        run_setpts: list[float] = []
        run_execute: list[float] = []
        run_combined: list[float] = []
        for sample_index in range(samples):
            setpts_ms = elapsed_batch_ms(
                lambda: set_type3_plan_points(
                    plan, source_points, target_points
                ),
                1,
                stream,
            )
            execute_ms = elapsed_batch_ms(
                lambda: plan.execute(input_values, out=output), 1, stream
            )

            def setpts_and_execute() -> None:
                set_type3_plan_points(plan, source_points, target_points)
                plan.execute(input_values, out=output)

            combined_ms = elapsed_batch_ms(setpts_and_execute, 1, stream)
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
    source_bounds = point_bounds(source_points)
    target_bounds = point_bounds(target_points)
    source_centers = [(lower + upper) * 0.5 for lower, upper in source_bounds]
    target_centers = [(lower + upper) * 0.5 for lower, upper in target_bounds]
    source_halfwidths = [
        (upper - lower) * 0.5 for lower, upper in source_bounds
    ]
    target_halfwidths = [
        (upper - lower) * 0.5 for lower, upper in target_bounds
    ]
    halfwidth_products = [
        source * target
        for source, target in zip(
            source_halfwidths, target_halfwidths, strict=True
        )
    ]
    fullwidth_products = [4.0 * product for product in halfwidth_products]
    million_points_per_second = (source_count + target_count) / (
        combined_mean * 1_000.0
    )

    print(f"  plan_creation_ms={format_samples(plan_samples)}")
    for run_index, values in enumerate(setpts_samples, start=1):
        print(f"  setpts_run_{run_index}_ms={format_samples(values)}")
    for run_index, values in enumerate(execute_samples, start=1):
        print(f"  execute_run_{run_index}_ms={format_samples(values)}")
    for run_index, values in enumerate(combined_samples, start=1):
        print(f"  combined_run_{run_index}_ms={format_samples(values)}")
    print(
        "RESULT,"
        f"kind=type-3,dimensions={dimension},M_sources={source_count},"
        f"K_targets={target_count},eps={eps:.9g},sigma={UPSAMPFAC:.9g},"
        "isign=1,modeord=not-applicable,"
        f"source_bounds={format_axis_bounds(source_bounds)},"
        f"target_bounds={format_axis_bounds(target_bounds)},"
        f"source_centers={format_axis_values(source_centers)},"
        f"target_centers={format_axis_values(target_centers)},"
        f"source_halfwidths={format_axis_values(source_halfwidths)},"
        f"target_halfwidths={format_axis_values(target_halfwidths)},"
        "halfwidth_space_bandwidth_products="
        f"{format_axis_values(halfwidth_products)},"
        "fullwidth_space_bandwidth_products="
        f"{format_axis_values(fullwidth_products)},"
        f"runs={runs},samples_per_run={samples},transforms_per_sample=1,"
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
        "paired_span=setpts+execute,"
        f"paired_ms={combined_mean:.6f},"
        "paired_million_source_plus_target_points_per_second="
        f"{million_points_per_second:.6f},"
        "timing=host-wall-clock-with-explicit-stream-sync,"
        "transfers=excluded,outputs=preallocated"
    )


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    dimension_group = parser.add_mutually_exclusive_group()
    dimension_group.add_argument(
        "--2d",
        action="store_true",
        dest="two_d",
        help="run the opt-in 2D matrix instead of the unchanged 1D matrix",
    )
    dimension_group.add_argument(
        "--3d",
        action="store_true",
        dest="three_d",
        help="run the opt-in 3D matrix instead of the unchanged 1D matrix",
    )
    parser.add_argument(
        "--type3",
        action="store_true",
        help="run NU-to-NU type-3 cases for the selected dimension",
    )
    parser.add_argument(
        "--case",
        action="append",
        type=parse_case,
        dest="cases",
        metavar="N[:M]",
        help="mode and point counts; repeat for multiple cases",
    )
    parser.add_argument(
        "--case-2d",
        action="append",
        type=parse_case_2d,
        dest="cases_2d",
        metavar="N0xN1[:M]",
        help="2D mode shape and point count; requires --2d",
    )
    parser.add_argument(
        "--case-3d",
        action="append",
        type=parse_case_3d,
        dest="cases_3d",
        metavar="N0xN1xN2[:M]",
        help="3D mode shape and point count; requires --3d",
    )
    parser.add_argument(
        "--case-type3",
        action="append",
        type=parse_case_type3,
        dest="cases_type3",
        metavar="M[:K]",
        help="type-3 source and target counts; requires --type3",
    )
    parser.add_argument(
        "--type3-source-halfwidth",
        type=float,
        default=DEFAULT_TYPE3_SOURCE_HALFWIDTH,
    )
    parser.add_argument(
        "--type3-target-halfwidth",
        type=float,
        default=DEFAULT_TYPE3_TARGET_HALFWIDTH,
    )
    parser.add_argument(
        "--type3-source-center",
        type=float,
        default=DEFAULT_TYPE3_SOURCE_CENTER,
    )
    parser.add_argument(
        "--type3-target-center",
        type=float,
        default=DEFAULT_TYPE3_TARGET_CENTER,
    )
    parser.add_argument("--runs", type=positive_int, default=DEFAULT_RUNS)
    parser.add_argument("--samples", type=positive_int, default=DEFAULT_SAMPLES)
    parser.add_argument("--warmups", type=nonnegative_int, default=DEFAULT_WARMUPS)
    parser.add_argument(
        "--type2-batch",
        type=positive_int,
        default=None,
        help=(
            "type-2 transforms per timed sample; defaults to 32 in 1D/2D "
            "and 1 in 3D to match the corresponding Rust harness"
        ),
    )
    parser.add_argument(
        "--ntrans",
        type=positive_int,
        default=1,
        help=(
            "native cuFINUFFT transforms sharing one point set; values above "
            "one use Plan(n_trans=...) and report batch-total plus per-vector "
            "timings (default: 1, preserving legacy behavior)"
        ),
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
    dimension = 3 if args.three_d else 2 if args.two_d else 1
    if args.ntrans > 1 and args.type2_batch is not None:
        raise SystemExit("--type2-batch cannot be combined with native --ntrans")
    if args.type3 and args.type2_batch is not None:
        raise SystemExit("--type2-batch does not apply to --type3")
    if args.type3 and args.ntrans != 1:
        raise SystemExit("native --ntrans currently covers type-1/type-2, not --type3")
    if args.type2_batch is None:
        args.type2_batch = 1 if dimension == 3 else DEFAULT_TYPE2_BATCH
    if args.type3:
        if args.cases or args.cases_2d or args.cases_3d:
            raise SystemExit("use --case-type3 with --type3")
    elif args.cases_type3:
        raise SystemExit("--case-type3 requires --type3")
    elif dimension == 1 and (args.cases_2d or args.cases_3d):
        raise SystemExit("--case-2d/--case-3d requires its dimension flag")
    elif dimension == 2 and (args.cases or args.cases_3d):
        raise SystemExit("use --case-2d, not --case/--case-3d, with --2d")
    elif dimension == 3 and (args.cases or args.cases_2d):
        raise SystemExit("use --case-3d, not --case/--case-2d, with --3d")

    type3_scalars = (
        ("--type3-source-halfwidth", args.type3_source_halfwidth, True),
        ("--type3-target-halfwidth", args.type3_target_halfwidth, True),
        ("--type3-source-center", args.type3_source_center, False),
        ("--type3-target-center", args.type3_target_center, False),
    )
    for field, value, positive in type3_scalars:
        if not math.isfinite(value) or (positive and value <= 0.0):
            qualifier = "finite and positive" if positive else "finite"
            raise SystemExit(f"{field} must be {qualifier}")
    type3_bounds = (
        args.type3_source_center - args.type3_source_halfwidth,
        args.type3_source_center + args.type3_source_halfwidth,
        args.type3_target_center - args.type3_target_halfwidth,
        args.type3_target_center + args.type3_target_halfwidth,
    )
    f32_max = float(np.finfo(np.float32).max)
    if any(
        not math.isfinite(bound) or abs(bound) > f32_max
        for bound in type3_bounds
    ):
        raise SystemExit("type-3 interval bounds must be representable as f32")

    device = cp.cuda.Device(args.device)
    device.use()
    properties = cp.cuda.runtime.getDeviceProperties(args.device)
    device_name = properties["name"]
    if isinstance(device_name, bytes):
        device_name = device_name.decode(errors="replace")
    library, library_sha256 = library_metadata()
    if args.type3:
        cases = args.cases_type3 or list(DEFAULT_CASES_TYPE3)
    elif dimension == 3:
        cases = args.cases_3d or list(DEFAULT_CASES_3D)
    elif dimension == 2:
        cases = args.cases_2d or list(DEFAULT_CASES_2D)
    else:
        cases = args.cases or list(DEFAULT_CASES)

    type_suffix = " type-3" if args.type3 else ""
    print(f"cuFINUFFT {dimension}D{type_suffix} GPU benchmark")
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
    dimension_method = f"dimensions={dimension}, " if dimension > 1 else ""
    if args.type3:
        print(
            "method=complex64, type=3, eps="
            f"{args.eps:.9g}, sigma={UPSAMPFAC:.1f}, isign=+1, "
            f"modeord=not-applicable, {dimension_method}"
            f"runs={args.runs}, samples={args.samples}, "
            f"warmups={args.warmups}, type3_batch=1, preallocated outputs"
        )
    else:
        native_batch_method = (
            f"native_ntrans={args.ntrans}, gpu_maxbatchsize={min(args.ntrans, 8)}, "
            "one setpts and one execute per timed sample"
            if args.ntrans > 1
            else (
                "native_ntrans=1, legacy repeat timing, "
                f"type2_batch={args.type2_batch}"
            )
        )
        print(
            "method=complex64, eps="
            f"{args.eps:.9g}, sigma={UPSAMPFAC:.1f}, isign=+1, modeord=0 "
            f"{dimension_method}"
            f"(centered/CMCL), runs={args.runs}, samples={args.samples}, "
            f"warmups={args.warmups}, {native_batch_method}, "
            "transform-major inputs, preallocated outputs"
        )
    print(
        "timing=GPU-resident host wall clock around plan/setpts/execute spans; "
        "explicit stream synchronization before and after every timed span; "
        "H2D/D2H and input generation excluded"
    )
    if args.type3:
        print(
            "pairing=wgpu type-3 submit-wait versus cuFINUFFT "
            "setpts+execute; plan, setpts, execute, and combined spans are "
            "all reported"
        )
    else:
        print(
            "pairing=wgpu type-1 submit-wait versus cuFINUFFT setpts+execute; "
            "wgpu type-2 submit-wait versus cuFINUFFT execute; all cuFINUFFT "
            "spans are also reported"
        )
    data_description = (
        "data=u32 LCG state=1664525*state+1013904223 mod 2^32, "
        f"high 24 bits, seed=0x{args.seed & 0xFFFF_FFFF:08X}; "
        "field-specific streams match the Rust benchmark"
    )
    if args.type3:
        source_seed_masks = (
            "["
            + "|".join(
                f"0x{mask:08X}"
                for mask in SOURCE_POINT_SEED_MASKS[:dimension]
            )
            + "]"
        )
        target_seed_masks = (
            "["
            + "|".join(
                f"0x{mask:08X}"
                for mask in TYPE3_TARGET_SEED_MASKS[:dimension]
            )
            + "]"
        )
        data_description += (
            f"; source_seed_masks={source_seed_masks}; "
            f"target_seed_masks={target_seed_masks}; "
            f"source_interval_center={args.type3_source_center:.9g} "
            f"source_interval_halfwidth={args.type3_source_halfwidth:.9g}; "
            f"target_interval_center={args.type3_target_center:.9g} "
            f"target_interval_halfwidth={args.type3_target_halfwidth:.9g}; "
            "interval endpoints are pinned exactly; source and target tuples "
            "are both reversed for the Python/column-major bridge"
        )
    elif dimension == 2:
        data_description += (
            f"; point_y_seed_mask=0x{POINT_Y_SEED_MASK:08X}; "
            "mode arrays use C shape (N1,N0) and setpts uses (x1,x0) so "
            "N0/x0 remains axis-zero-fast"
        )
    elif dimension == 3:
        data_description += (
            f"; point_y_seed_mask=0x{POINT_Y_SEED_MASK:08X}; "
            f"point_z_seed_mask=0x{POINT_Z_SEED_MASK:08X}; "
            "mode arrays use C shape (N2,N1,N0) and setpts uses "
            "(x2,x1,x0) so N0/x0 remains axis-zero-fast"
        )
    print(data_description)

    if args.type3:
        run_correctness_smoke_type3(
            dimension,
            args.device,
            args.eps,
            args.seed,
            args.type3_source_halfwidth,
            args.type3_target_halfwidth,
            args.type3_source_center,
            args.type3_target_center,
        )
    elif dimension == 3:
        run_correctness_smoke_3d(args.device, args.eps, args.seed)
    elif dimension == 2:
        run_correctness_smoke_2d(args.device, args.eps, args.seed)
    else:
        run_correctness_smoke(args.device, args.eps, args.seed)
    if not args.type3:
        run_native_batch_correctness_smoke(
            dimension, args.ntrans, args.device, args.eps, args.seed
        )
    if args.smoke_only:
        return 0

    if args.type3:
        for source_count, target_count in cases:
            source_points, target_points, strengths = make_inputs_type3(
                dimension,
                source_count,
                target_count,
                args.seed,
                args.type3_source_halfwidth,
                args.type3_target_halfwidth,
                args.type3_source_center,
                args.type3_target_center,
            )
            gpu_source_points = tuple(cp.asarray(axis) for axis in source_points)
            gpu_target_points = tuple(cp.asarray(axis) for axis in target_points)
            gpu_strengths = cp.asarray(strengths)
            cp.cuda.runtime.deviceSynchronize()
            source_bytes = sum(axis.nbytes for axis in gpu_source_points)
            target_bytes = sum(axis.nbytes for axis in gpu_target_points)
            print()
            print(
                f"case kind=type-3 dimensions={dimension} "
                f"M_sources={source_count} K_targets={target_count}"
            )
            print(
                f"source_bounds={format_axis_bounds(point_bounds(source_points))} "
                f"target_bounds={format_axis_bounds(point_bounds(target_points))}"
            )
            print(
                f"device_inputs source_points_bytes={source_bytes} "
                f"target_points_bytes={target_bytes} "
                f"strengths_bytes={gpu_strengths.nbytes}"
            )
            benchmark_type3(
                dimension,
                gpu_source_points,
                gpu_target_points,
                gpu_strengths,
                source_count=source_count,
                target_count=target_count,
                eps=args.eps,
                runs=args.runs,
                samples=args.samples,
                warmups=args.warmups,
                device_id=args.device,
            )
            cp.cuda.runtime.deviceSynchronize()
            del gpu_source_points, gpu_target_points, gpu_strengths
            gc.collect()
            cp.get_default_memory_pool().free_all_blocks()
        return 0

    for case in cases:
        if dimension == 3:
            n0, n1, n2, point_count = case
            n_modes: int | tuple[int, ...] = (n0, n1, n2)
            points, strengths, modes = make_inputs_3d(
                n0, n1, n2, point_count, args.seed
            )
            case_label = (
                f"dimensions=3 N0={n0} N1={n1} N2={n2} "
                f"N_total={n0 * n1 * n2} M={point_count}"
            )
            gpu_points = tuple(cp.asarray(axis) for axis in points)
        elif dimension == 2:
            n0, n1, point_count = case
            n_modes: int | tuple[int, ...] = (n0, n1)
            points, strengths, modes = make_inputs_2d(
                n0, n1, point_count, args.seed
            )
            case_label = (
                f"dimensions=2 N0={n0} N1={n1} "
                f"N_total={n0 * n1} M={point_count}"
            )
            gpu_points = tuple(cp.asarray(axis) for axis in points)
        else:
            n_modes, point_count = case
            points, strengths, modes = make_inputs(
                n_modes, point_count, args.seed
            )
            case_label = f"N={n_modes} M={point_count}"
            gpu_points = cp.asarray(points)
        print()
        print(f"case {case_label}")
        transform_strengths = stack_distinct_transforms(
            strengths, args.ntrans
        )
        transform_modes = stack_distinct_transforms(modes, args.ntrans)
        gpu_strengths = cp.asarray(transform_strengths)
        gpu_modes = cp.asarray(transform_modes)
        cp.cuda.runtime.deviceSynchronize()
        point_bytes = (
            sum(axis.nbytes for axis in gpu_points)
            if isinstance(gpu_points, tuple)
            else gpu_points.nbytes
        )
        print(
            f"device_inputs points_bytes={point_bytes} "
            f"strengths_bytes={gpu_strengths.nbytes} modes_bytes={gpu_modes.nbytes} "
            f"ntrans={args.ntrans} layout=transform-major"
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
            n_trans=args.ntrans,
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
            n_trans=args.ntrans,
            device_id=args.device,
        )
        cp.cuda.runtime.deviceSynchronize()
        del gpu_points, gpu_strengths, gpu_modes
        gc.collect()
        cp.get_default_memory_pool().free_all_blocks()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
