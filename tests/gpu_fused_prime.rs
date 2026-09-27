#![cfg(not(target_arch = "wasm32"))]

//! Focused correctness and routing coverage for fused Rader and Bluestein kernels.

use std::sync::mpsc;

use wgpu_fft::math::{reference_c2c_nd_f64, Complex64};
use wgpu_fft::{C2cRoute, FftConfig, FftPlan, FftTuning, Normalization};

const RADER_FUSED_LABEL: &str = "rader-fused-workgroup-stage";
const DIRECT_LABEL: &str = "rader-direct-dft-stage";
const BLUESTEIN_REGISTER_LABEL: &str = "bluestein-register-stage";
const BLUESTEIN_FUSED_LABEL: &str = "bluestein-fused-workgroup-stage";

#[test]
fn fused_prime_gpu_matches_cpu_and_storage_fallback() {
    if std::env::var_os("WGPU_FFT_RUN_GPU_TESTS").is_none() {
        eprintln!("skipping GPU test; set WGPU_FFT_RUN_GPU_TESTS=1 to run it");
        return;
    }

    pollster::block_on(run_fused_prime_cases());
}

async fn run_fused_prime_cases() {
    let Some(context) = wgpu_fft::device::request_default_device().await else {
        eprintln!("skipping GPU test; no suitable wgpu adapter was found");
        return;
    };
    eprintln!("adapter: {:?}", context.adapter.get_info());
    let storage_limit = context.device.limits().max_compute_workgroup_storage_size as u64;

    if storage_limit >= 48_008 {
        compare_rader_2999_with_16k_fallback(&context).await;
    } else {
        eprintln!(
            "skipping fused/fallback N=2999 comparison: device exposes only {storage_limit} bytes"
        );
    }

    run_direct_cases(&context);

    // Short primes take the direct DFT kernel by default; `direct_max_prime(0)`
    // keeps the fused Rader kernel covered for them.
    for (length, batch) in [(101, 3), (1009, 2)] {
        for inverse in [false, true] {
            run_fused_case(
                &context.device,
                &context.queue,
                config_1d(length, batch, inverse).with_tuning(rader_only()),
                C2cRoute::Rader,
                RADER_FUSED_LABEL,
                &["rader-permutation-helper", "rader-bfft-helper"],
                &format!("Rader N={length} batch={batch} inverse={inverse}"),
            );
        }
    }

    // Many lines: the fused Rader kernel packs several per workgroup,
    // contiguous and strided, cyclic and linear (zero-padded).
    for (label, config) in [
        (
            "Rader multi-line N=101 batch=1024",
            config_1d(101, 1024, false).with_tuning(rader_only()),
        ),
        (
            "Rader multi-line inverse N=101 batch=1024",
            config_1d(101, 1024, true).with_tuning(rader_only()),
        ),
        (
            "Rader multi-line strided 64x101 axis 1 batch=16",
            FftConfig::new_nd([64, 101])
                .with_axes([1])
                .with_batch(16)
                .with_normalization(Normalization::None)
                .with_tuning(rader_only()),
        ),
        (
            "Rader multi-line linear N=107 batch=1024",
            config_1d(107, 1024, false).with_tuning(
                FftTuning::default()
                    .with_direct_max_prime(0)
                    .with_force_rader_axes([0]),
            ),
        ),
    ] {
        run_fused_case(
            &context.device,
            &context.queue,
            config,
            C2cRoute::Rader,
            RADER_FUSED_LABEL,
            &["rader-permutation-helper", "rader-bfft-helper"],
            label,
        );
    }
    assert!(
        wgpu_fft::export_pipeline_cache_snapshot(&context.device)
            .pipeline_keys()
            .iter()
            .any(|key| key.contains("fused-prime:rader") && key.contains(":lines=")),
        "no multi-line fused Rader kernel was built"
    );

    // Strided lines too long for several per workgroup load and store two
    // (810-point convolutions) or four (2178-point) at a time and convolve
    // one after another; 769 and 1027 lines leave a partial last group.
    for (label, config, fragment) in [
        (
            "Rader serial strided 769x811 axis 1",
            FftConfig::new_nd([769, 811])
                .with_axes([1])
                .with_normalization(Normalization::None),
            ":n=811:stride=769:",
        ),
        (
            "Rader serial strided inverse 1027x1087 axis 1",
            FftConfig::inverse_nd([1027, 1087]).with_axes([1]),
            ":n=1087:stride=1027:",
        ),
    ] {
        run_fused_case(
            &context.device,
            &context.queue,
            config,
            C2cRoute::Rader,
            RADER_FUSED_LABEL,
            &["rader-permutation-helper", "rader-bfft-helper"],
            label,
        );
        let snapshot = wgpu_fft::export_pipeline_cache_snapshot(&context.device);
        assert!(
            snapshot
                .pipeline_keys()
                .iter()
                .any(|key| key.contains(fragment) && key.contains(":serial=")),
            "{label}: no serial-line Rader kernel"
        );
    }

    // Primes whose N - 1 has a prime factor from 17 to 61 convolve
    // cyclically over N - 1 points with radix-p stages where workgroup
    // storage holds them: 613 (612 = 36 * 17), strided lines of it four per
    // workgroup, 1381 (1380 = 60 * 23), 4093 (4092 = 132 * 31, just within
    // 32 KiB), 4241 (4240 = 80 * 53), and 5003 (5002 = 2 * 41 * 61).
    for (label, config, storage_bytes) in [
        (
            "medium-prime Rader N=613 batch=3",
            config_1d(613, 3, false),
            4_896u64,
        ),
        (
            "medium-prime Rader inverse N=613 batch=3",
            config_1d(613, 3, true),
            4_896,
        ),
        (
            "medium-prime Rader strided 64x613 axis 1 batch=16",
            FftConfig::new_nd([64, 613])
                .with_axes([1])
                .with_batch(16)
                .with_normalization(Normalization::None),
            4_896,
        ),
        (
            "medium-prime Rader inverse N=1381 batch=2",
            config_1d(1381, 2, true),
            11_040,
        ),
        (
            "medium-prime Rader N=4093",
            config_1d(4093, 1, false),
            32_736,
        ),
        (
            "medium-prime Rader N=4241 batch=2",
            config_1d(4241, 2, false),
            33_920,
        ),
        (
            "medium-prime Rader inverse N=5003",
            config_1d(5003, 1, true),
            40_016,
        ),
    ] {
        if storage_limit < storage_bytes {
            eprintln!("skipping {label}: device exposes only {storage_limit} bytes");
            continue;
        }
        run_fused_case(
            &context.device,
            &context.queue,
            config,
            C2cRoute::Rader,
            RADER_FUSED_LABEL,
            &["rader-permutation-helper", "rader-bfft-helper"],
            label,
        );
    }
    let snapshot = wgpu_fft::export_pipeline_cache_snapshot(&context.device);
    let keys = snapshot.pipeline_keys();
    let built = |fragment: &str| {
        keys.iter()
            .any(|key| key.contains("fused-prime:rader") && key.contains(fragment))
    };
    assert!(
        built(":m=612:factors=17x6x6"),
        "no radix-17 Rader kernel: {keys:?}"
    );
    if storage_limit >= 19_648 {
        assert!(
            keys.iter()
                .any(|key| key.contains(":n=613:stride=64:m=612:") && key.contains(":lines=4")),
            "no multi-line radix-17 Rader kernel: {keys:?}"
        );
    }
    if storage_limit >= 33_920 {
        assert!(
            built(":m=4240:factors=53x16x5"),
            "no radix-53 Rader kernel: {keys:?}"
        );
    }

    // Strided axes interleave several lines per register Bluestein
    // workgroup where the device allows it: eight 512-point convolutions of a
    // Rader axis (N=179), four 1024-point ones (N=419), and two 2048-point
    // ones of a Bluestein axis (N=986 = 2 * 17 * 29).
    for (label, config, route) in [
        (
            "register Bluestein strided 2048x179 axis 1",
            FftConfig::new_nd([2048, 179])
                .with_axes([1])
                .with_normalization(Normalization::None),
            C2cRoute::Rader,
        ),
        (
            "register Bluestein strided inverse 1024x419 axis 1",
            FftConfig::inverse_nd([1024, 419]).with_axes([1]),
            C2cRoute::Rader,
        ),
        (
            "register Bluestein strided 1024x986 axis 1",
            FftConfig::new_nd([1024, 986])
                .with_axes([1])
                .with_normalization(Normalization::None),
            C2cRoute::Bluestein,
        ),
    ] {
        let input = test_signal(config.total_complex_len().unwrap());
        let expected = reference_f64(&input, &config);
        let (actual, plan) = execute_c2c(&context.device, &context.queue, config, &input);
        assert_eq!(plan.route(), route, "{label}");
        assert_eq!(kernel_labels(&plan), [BLUESTEIN_REGISTER_LABEL], "{label}");
        assert_matches_reference(&actual, &expected, label);
        let (max_relative, rms_relative) = relative_error_metrics(&actual, &expected);
        eprintln!(
            "FUSED_PRIME_ACCURACY label={label:?} max_relative={max_relative:.9e} rms_relative={rms_relative:.9e}"
        );
        assert!(
            max_relative < 5.0e-7 && rms_relative < 5.0e-7,
            "{label}: max/rms relative error={max_relative}/{rms_relative}"
        );
    }
    if context
        .device
        .limits()
        .max_compute_invocations_per_workgroup
        >= 1024
    {
        let snapshot = wgpu_fft::export_pipeline_cache_snapshot(&context.device);
        let keys = snapshot.pipeline_keys();
        for (fragment, lines) in [
            (":n=179:stride=2048:m=512:", ":lines=8"),
            (":n=419:stride=1024:m=1024:", ":lines=4"),
            (":n=986:stride=1024:m=2048:", ":lines=2"),
        ] {
            assert!(
                keys.iter().any(|key| key.contains("fused-prime:bluestein")
                    && key.contains(fragment)
                    && key.contains(lines)),
                "no multi-line register Bluestein kernel {fragment}{lines}: {keys:?}"
            );
        }
    }

    // N=517 and N=2062 convolve over 1040 and 4125 points in workgroup
    // memory: 2048- and 8192-point register convolutions would be about
    // twice as long.
    for (length, batch, fused_storage_bytes) in [(517, 2, 8_320u64), (2062, 1, 33_000)] {
        for inverse in [false, true] {
            let config = config_1d(length, batch, inverse);
            let label = format!("Bluestein N={length} batch={batch} inverse={inverse}");
            if storage_limit >= fused_storage_bytes {
                run_fused_case(
                    &context.device,
                    &context.queue,
                    config,
                    C2cRoute::Bluestein,
                    BLUESTEIN_FUSED_LABEL,
                    &["bluestein-chirp-helper", "bluestein-bfft-helper"],
                    &label,
                );
            } else {
                run_bluestein_fallback_case(&context, config, &label);
            }
        }
    }

    for inverse in [false, true] {
        let config = if inverse {
            FftConfig::inverse_nd([2, 517, 3])
        } else {
            FftConfig::new_nd([2, 517, 3]).with_normalization(Normalization::None)
        }
        .with_axes([1])
        .with_batch(2);
        run_fused_case(
            &context.device,
            &context.queue,
            config,
            C2cRoute::Bluestein,
            BLUESTEIN_FUSED_LABEL,
            &["bluestein-chirp-helper", "bluestein-bfft-helper"],
            &format!("Bluestein ND axis shape=2x517x3 batch=2 inverse={inverse}"),
        );
    }

    // N=221 convolves over 512 points in registers rather than 441 in
    // workgroup memory: fewer stages and barriers. So does a prime whose
    // Rader convolution would be linear (N=179: 178 = 2 * 89).
    for (length, route) in [(221, C2cRoute::Bluestein), (179, C2cRoute::Rader)] {
        for inverse in [false, true] {
            let config = config_1d(length, 2, inverse);
            let label = format!("short register Bluestein N={length} inverse={inverse}");
            let input = test_signal(config.total_complex_len().unwrap());
            let expected = reference_f64(&input, &config);
            let (actual, plan) = execute_c2c(&context.device, &context.queue, config, &input);
            assert_eq!(plan.route(), route, "{label}");
            assert_eq!(kernel_labels(&plan), [BLUESTEIN_REGISTER_LABEL], "{label}");
            assert_matches_reference(&actual, &expected, &label);
        }
    }

    for (shape, fused_label, expected_helpers) in [
        (
            [2, 101],
            RADER_FUSED_LABEL,
            &["rader-permutation-helper", "rader-bfft-helper"][..],
        ),
        (
            [2, 517],
            BLUESTEIN_FUSED_LABEL,
            &["bluestein-chirp-helper", "bluestein-bfft-helper"][..],
        ),
    ] {
        for inverse in [false, true] {
            let config = if inverse {
                FftConfig::inverse_nd(shape)
            } else {
                FftConfig::new_nd(shape).with_normalization(Normalization::None)
            }
            .with_batch(2)
            .with_tuning(rader_only());
            run_fused_axis_sequence_case(
                &context.device,
                &context.queue,
                config,
                fused_label,
                expected_helpers,
                &format!("fused AxisSequence shape={shape:?} inverse={inverse}"),
            );
        }
    }

    assert_rader_fallback_plan(
        &FftPlan::c2c(
            &context.device,
            &context.queue,
            FftConfig::new(17)
                .with_normalization(Normalization::None)
                .with_tuning(rader_only()),
        )
        .unwrap(),
        "tiny N=17",
    );

    // Rader N=4099 (4098 = 2 * 3 * 683) uses M=8232, so its fused scratch
    // requires 8*M + 8 bytes.
    if storage_limit < 65_864 {
        run_rader_fallback_case(&context, 4099, "storage-limited N=4099");
    } else {
        eprintln!(
            "skipping N=4099 storage-fallback assertion: device exposes {storage_limit} bytes"
        );
    }

    #[cfg(windows)]
    std::mem::forget(context);
}

async fn compare_rader_2999_with_16k_fallback(context: &wgpu_fft::device::GpuContext) {
    let mut low_limits = context.adapter.limits();
    low_limits.max_compute_workgroup_storage_size = 16 * 1024;
    let (low_device, low_queue) = context
        .adapter
        .request_device(&wgpu::DeviceDescriptor {
            label: Some("wgpu_fft.test.fused_prime.low_storage_device"),
            required_features: wgpu::Features::empty(),
            required_limits: low_limits,
            experimental_features: wgpu::ExperimentalFeatures::disabled(),
            memory_hints: wgpu::MemoryHints::MemoryUsage,
            trace: wgpu::Trace::Off,
        })
        .await
        .expect("the adapter should allow a second device with a 16 KiB storage limit");
    assert_eq!(
        low_device.limits().max_compute_workgroup_storage_size,
        16 * 1024
    );

    let input = test_signal(2999);
    for inverse in [false, true] {
        let config = config_1d(2999, 1, inverse);
        let expected = reference_f64(&input, &config);
        let (fused, fused_plan) =
            execute_c2c(&context.device, &context.queue, config.clone(), &input);
        let (fallback, fallback_plan) =
            execute_c2c(&low_device, &low_queue, config.clone(), &input);

        assert_fused_prime_plan(
            &fused_plan,
            C2cRoute::Rader,
            RADER_FUSED_LABEL,
            &["rader-permutation-helper", "rader-bfft-helper"],
        );
        // The convolution no longer fits 16 KiB, so the prime runs Bluestein's
        // register-resident kernel.
        assert_eq!(
            kernel_labels(&fallback_plan),
            [BLUESTEIN_REGISTER_LABEL],
            "forced 16 KiB N=2999"
        );
        // Forcing Rader keeps the multi-pass Rader pipeline covered.
        let forced_rader = config
            .clone()
            .with_tuning(FftTuning::default().with_force_rader_axes([0]));
        let (staged, staged_plan) =
            execute_c2c(&low_device, &low_queue, forced_rader.clone(), &input);
        assert_rader_fallback_plan(&staged_plan, "forced 16 KiB N=2999");
        let staged_kernels = kernel_labels(&staged_plan);
        assert_eq!(
            staged_kernels.len(),
            9,
            "forced 16 KiB N=2999 should use five bridge kernels and two split fused passes per inner FFT: {staged_kernels:?}"
        );
        assert_matches_reference(&staged, &expected, "Rader N=2999 staged fallback");
        let unsplit = unsplit_fallback(&low_device, &low_queue, &forced_rader, &input);
        assert_eq!(
            unsplit.1.len(),
            17,
            "unsplit forced 16 KiB N=2999 should use five bridge kernels and twelve Stockham stages: {:?}",
            unsplit.1
        );
        assert_matches_reference(&unsplit.0, &expected, "Rader N=2999 unsplit fallback");

        let label = format!("Rader N=2999 inverse={inverse}");
        let (fused_max, fused_rms) = relative_error_metrics(&fused, &expected);
        let (fallback_max, fallback_rms) = relative_error_metrics(&fallback, &expected);
        eprintln!(
            "{label}: fused max/rms={fused_max:.9e}/{fused_rms:.9e}; fallback max/rms={fallback_max:.9e}/{fallback_rms:.9e}"
        );
        assert_matches_reference(&fused, &expected, &format!("{label} fused"));
        assert_matches_reference(&fallback, &expected, &format!("{label} fallback"));
        assert_close_f32(&fused, &fallback, &format!("{label} fused versus fallback"));
        assert!(fused_max < 5.0e-7 && fused_rms < 3.0e-7, "{label}");
        let fallback_reference = fallback
            .chunks_exact(2)
            .map(|pair| Complex64::new(f64::from(pair[0]), f64::from(pair[1])))
            .collect::<Vec<_>>();
        let (difference_max, difference_rms) = relative_error_metrics(&fused, &fallback_reference);
        assert!(
            difference_max < 3.0e-6 && difference_rms < 3.0e-6,
            "{label}: fused/fallback max/rms={difference_max}/{difference_rms}"
        );
    }

    let input = test_signal(2062);
    for inverse in [false, true] {
        let config = config_1d(2062, 1, inverse);
        let expected = reference_f64(&input, &config);
        let (fused, fused_plan) =
            execute_c2c(&context.device, &context.queue, config.clone(), &input);
        let (fallback, fallback_plan) =
            execute_c2c(&low_device, &low_queue, config.clone(), &input);

        assert_fused_prime_plan(
            &fused_plan,
            C2cRoute::Bluestein,
            BLUESTEIN_FUSED_LABEL,
            &["bluestein-chirp-helper", "bluestein-bfft-helper"],
        );
        // The 4125-point convolution does not fit 16 KiB of workgroup
        // memory, so it runs in registers over 8192 points.
        assert_eq!(
            kernel_labels(&fallback_plan),
            [BLUESTEIN_REGISTER_LABEL],
            "forced 16 KiB N=2062"
        );
        let unsplit = unsplit_fallback(&low_device, &low_queue, &config, &input);
        assert_eq!(
            unsplit.1.len(),
            13,
            "unsplit forced 16 KiB N=2062 should use three bridge kernels and ten Stockham stages: {:?}",
            unsplit.1
        );
        assert_matches_reference(&unsplit.0, &expected, "Bluestein N=2062 unsplit fallback");

        let label = format!("Bluestein N=2062 inverse={inverse}");
        let (fused_max, fused_rms) = relative_error_metrics(&fused, &expected);
        assert!(fused_max < 5.0e-7 && fused_rms < 5.0e-7, "{label}");
        assert_matches_reference(&fallback, &expected, &format!("{label} fallback"));
        assert_close_f32(&fused, &fallback, &format!("{label} fused versus fallback"));
        let fallback_reference = fallback
            .chunks_exact(2)
            .map(|pair| Complex64::new(f64::from(pair[0]), f64::from(pair[1])))
            .collect::<Vec<_>>();
        let (difference_max, difference_rms) = relative_error_metrics(&fused, &fallback_reference);
        assert!(
            difference_max < 3.0e-6 && difference_rms < 3.0e-6,
            "{label}: fused/fallback max/rms={difference_max}/{difference_rms}"
        );
    }

    #[cfg(windows)]
    {
        std::mem::forget(low_queue);
        std::mem::forget(low_device);
    }
}

/// Runs `config` with long inner FFTs left on Stockham stages, returning the
/// output and kernel labels.
fn unsplit_fallback(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    config: &FftConfig,
    input: &[f32],
) -> (Vec<f32>, Vec<String>) {
    let tuning = config.tuning().clone().with_fuse_long_axes(false);
    let (output, plan) = execute_c2c(device, queue, config.clone().with_tuning(tuning), input);
    let kernels = kernel_labels(&plan);
    assert!(
        kernels
            .iter()
            .any(|kernel| kernel.ends_with("-stockham-stage")),
        "unsplit fallback keeps Stockham stages: {kernels:?}"
    );
    (output, kernels)
}

fn rader_only() -> FftTuning {
    FftTuning::default().with_direct_max_prime(0)
}

/// Short primes transform with the direct DFT kernel: one kernel per axis,
/// many lines per workgroup, and f32 accuracy on par with the FFT routes.
fn run_direct_cases(context: &wgpu_fft::device::GpuContext) {
    // Primes whose Rader convolution would be linear (103, 107) or too short
    // for the fused kernel (17 to 61) take the direct kernel.
    for (length, batch) in [(17, 5), (31, 1), (61, 3), (103, 2), (107, 7)] {
        for inverse in [false, true] {
            run_fused_case(
                &context.device,
                &context.queue,
                config_1d(length, batch, inverse),
                C2cRoute::Rader,
                DIRECT_LABEL,
                &["rader-permutation-helper", "rader-bfft-helper"],
                &format!("direct N={length} batch={batch} inverse={inverse}"),
            );
        }
    }
    // Cyclic convolutions (N - 1 smooth) in the fused Rader kernel beat the
    // direct kernel.
    for (length, batch) in [(101, 2), (127, 7)] {
        run_fused_case(
            &context.device,
            &context.queue,
            config_1d(length, batch, false),
            C2cRoute::Rader,
            RADER_FUSED_LABEL,
            &["rader-permutation-helper", "rader-bfft-helper"],
            &format!("cyclic Rader N={length} batch={batch}"),
        );
    }
    // A long batch spreads over many workgroups, and a strided prime axis
    // loads several neighbouring lines together.
    run_fused_case(
        &context.device,
        &context.queue,
        FftConfig::new(17)
            .with_batch(4099)
            .with_normalization(Normalization::Forward),
        C2cRoute::Rader,
        DIRECT_LABEL,
        &["rader-permutation-helper", "rader-bfft-helper"],
        "direct N=17 batch=4099",
    );
    for inverse in [false, true] {
        let config = if inverse {
            FftConfig::inverse_nd([6, 43, 5])
        } else {
            FftConfig::new_nd([6, 43, 5]).with_normalization(Normalization::Orthogonal)
        }
        .with_axes([1])
        .with_batch(2);
        run_fused_case(
            &context.device,
            &context.queue,
            config,
            C2cRoute::Rader,
            DIRECT_LABEL,
            &["rader-permutation-helper", "rader-bfft-helper"],
            &format!("direct strided 6x43x5 axis 1 inverse={inverse}"),
        );
    }
    // All-prime ND shapes too large for one workgroup run one direct kernel
    // per axis.
    for shape in [vec![17, 17, 31], vec![19, 23, 29]] {
        let config = FftConfig::new_nd(shape.clone()).with_normalization(Normalization::None);
        let input = test_signal(config.total_complex_len().unwrap());
        let expected = reference_f64(&input, &config);
        let (actual, plan) = execute_c2c(&context.device, &context.queue, config, &input);
        let direct_kernels = kernel_labels(&plan)
            .iter()
            .filter(|label| label.ends_with("direct-dft-stage"))
            .count();
        assert_eq!(direct_kernels, shape.len(), "direct ND {shape:?}");
        let (max_relative, rms_relative) = relative_error_metrics(&actual, &expected);
        eprintln!(
            "FUSED_PRIME_ACCURACY label=\"direct ND {shape:?}\" max_relative={max_relative:.9e} rms_relative={rms_relative:.9e}"
        );
        assert!(
            max_relative < 5.0e-7 && rms_relative < 5.0e-7,
            "direct ND {shape:?}: max/rms relative error={max_relative}/{rms_relative}"
        );
    }
}

fn run_fused_case(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    config: FftConfig,
    route: C2cRoute,
    fused_label: &'static str,
    expected_helpers: &[&str],
    label: &str,
) {
    let input = test_signal(config.total_complex_len().unwrap());
    let expected = reference_f64(&input, &config);
    let (actual, plan) = execute_c2c(device, queue, config, &input);
    assert_fused_prime_plan(&plan, route, fused_label, expected_helpers);
    assert_matches_reference(&actual, &expected, label);
    let (max_relative, rms_relative) = relative_error_metrics(&actual, &expected);
    eprintln!(
        "FUSED_PRIME_ACCURACY label={label:?} max_relative={max_relative:.9e} rms_relative={rms_relative:.9e}"
    );
    assert!(
        max_relative < 5.0e-7 && rms_relative < 5.0e-7,
        "{label}: max/rms relative error={max_relative}/{rms_relative}"
    );
}

fn run_rader_fallback_case(context: &wgpu_fft::device::GpuContext, length: usize, label: &str) {
    let config = FftConfig::new(length).with_normalization(Normalization::None);
    let input = test_signal(length);
    let expected = reference_f64(&input, &config);
    let (actual, plan) = execute_c2c(&context.device, &context.queue, config, &input);
    // Primes whose convolution fits no workgroup run Bluestein's
    // register-resident kernel where the device supports one.
    if kernel_labels(&plan) != [BLUESTEIN_REGISTER_LABEL] {
        assert_rader_fallback_plan(&plan, label);
    }
    assert_matches_reference(&actual, &expected, label);
    let (max_relative, rms_relative) = relative_error_metrics(&actual, &expected);
    assert!(
        max_relative < 5.0e-6 && rms_relative < 5.0e-6,
        "{label}: max/rms relative error={max_relative}/{rms_relative}"
    );
}

fn run_bluestein_fallback_case(
    context: &wgpu_fft::device::GpuContext,
    config: FftConfig,
    label: &str,
) {
    let input = test_signal(config.total_complex_len().unwrap());
    let expected = reference_f64(&input, &config);
    let (actual, plan) = execute_c2c(&context.device, &context.queue, config, &input);
    if kernel_labels(&plan) != [BLUESTEIN_REGISTER_LABEL] {
        assert_bluestein_fallback_plan(&plan, label);
    }
    assert_matches_reference(&actual, &expected, label);
    let (max_relative, rms_relative) = relative_error_metrics(&actual, &expected);
    assert!(
        max_relative < 5.0e-6 && rms_relative < 5.0e-6,
        "{label}: max/rms relative error={max_relative}/{rms_relative}"
    );
}

fn run_fused_axis_sequence_case(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    config: FftConfig,
    fused_label: &str,
    expected_helpers: &[&str],
    label: &str,
) {
    let input = test_signal(config.total_complex_len().unwrap());
    let expected = reference_f64(&input, &config);
    let (actual, plan) = execute_c2c(device, queue, config, &input);
    assert_eq!(plan.route(), C2cRoute::AxisSequence, "{label}");
    let kernels = kernel_labels(&plan);
    assert_eq!(
        kernels,
        [
            "axis-sequence-mixed-fused-pow2-stage".to_string(),
            fused_label.to_string(),
        ],
        "{label}"
    );
    let mut expected_helper_labels = vec!["axis-sequence-workspace".to_string()];
    expected_helper_labels.extend(expected_helpers.iter().map(|helper| (*helper).to_string()));
    assert_eq!(helper_labels(&plan), expected_helper_labels, "{label}");
    assert_matches_reference(&actual, &expected, label);
    let (max_relative, rms_relative) = relative_error_metrics(&actual, &expected);
    assert!(
        max_relative < 5.0e-7 && rms_relative < 5.0e-7,
        "{label}: max/rms relative error={max_relative}/{rms_relative}"
    );
}

fn config_1d(length: usize, batch: usize, inverse: bool) -> FftConfig {
    if inverse {
        FftConfig::inverse(length)
    } else {
        FftConfig::new(length).with_normalization(Normalization::None)
    }
    .with_batch(batch)
}

fn assert_fused_prime_plan(
    plan: &FftPlan,
    route: C2cRoute,
    fused_label: &str,
    expected_helpers: &[&str],
) {
    assert_eq!(plan.route(), route);
    let diagnostics = plan.diagnostics();
    assert!(diagnostics.blockers().is_empty());
    assert_eq!(kernel_labels(plan), vec![fused_label.to_string()]);
    assert_eq!(
        helper_labels(plan),
        expected_helpers
            .iter()
            .map(|label| (*label).to_string())
            .collect::<Vec<_>>()
    );
}

fn assert_rader_fallback_plan(plan: &FftPlan, label: &str) {
    assert_eq!(plan.route(), C2cRoute::Rader, "{label}");
    let diagnostics = plan.diagnostics();
    assert!(diagnostics.blockers().is_empty(), "{label}");
    let kernels = kernel_labels(plan);
    assert!(
        !kernels.iter().any(|kernel| kernel == RADER_FUSED_LABEL),
        "{label}: {kernels:?}"
    );
    for required in [
        "rader-sum",
        "rader-pack",
        "rader-mul",
        "rader-write-y0",
        "rader-post",
    ] {
        assert!(
            kernels.iter().any(|kernel| kernel == required),
            "{label}: missing {required}: {kernels:?}"
        );
    }
    let helpers = helper_labels(plan);
    for required in [
        "rader-permutation-helper",
        "rader-bfft-helper",
        "rader-sum-helper",
        "rader-x0-helper",
        "rader-work-helper",
        "rader-fft-helper",
    ] {
        assert!(
            helpers.iter().any(|helper| helper == required),
            "{label}: missing {required}: {helpers:?}"
        );
    }
}

fn assert_bluestein_fallback_plan(plan: &FftPlan, label: &str) {
    assert_eq!(plan.route(), C2cRoute::Bluestein, "{label}");
    let kernels = kernel_labels(plan);
    assert!(
        !kernels.iter().any(|kernel| kernel == BLUESTEIN_FUSED_LABEL),
        "{label}: {kernels:?}"
    );
    for required in ["bluestein-pack", "bluestein-mul", "bluestein-post"] {
        assert!(
            kernels.iter().any(|kernel| kernel == required),
            "{label}: missing {required}: {kernels:?}"
        );
    }
    let helpers = helper_labels(plan);
    for required in [
        "bluestein-chirp-helper",
        "bluestein-bfft-helper",
        "bluestein-work-helper",
        "bluestein-fft-helper",
    ] {
        assert!(
            helpers.iter().any(|helper| helper == required),
            "{label}: missing {required}: {helpers:?}"
        );
    }
}

fn kernel_labels(plan: &FftPlan) -> Vec<String> {
    plan.diagnostics()
        .stages()
        .iter()
        .filter(|stage| stage.kind == "kernel")
        .map(|stage| stage.label.clone())
        .collect()
}

fn helper_labels(plan: &FftPlan) -> Vec<String> {
    plan.diagnostics()
        .stages()
        .iter()
        .filter(|stage| stage.kind == "helper-buffer-window")
        .map(|stage| stage.label.clone())
        .collect()
}

fn reference_f64(input: &[f32], config: &FftConfig) -> Vec<Complex64> {
    let values = input
        .chunks_exact(2)
        .map(|pair| Complex64::new(f64::from(pair[0]), f64::from(pair[1])))
        .collect::<Vec<_>>();
    reference_c2c_nd_f64(&values, config).unwrap()
}

fn execute_c2c(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    config: FftConfig,
    input: &[f32],
) -> (Vec<f32>, FftPlan) {
    let plan = FftPlan::c2c(device, queue, config).unwrap();
    let byte_len = std::mem::size_of_val(input) as u64;
    assert_eq!(plan.required_buffer_size_bytes(), byte_len);

    let input_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("wgpu_fft.test.fused_prime.input"),
        size: byte_len,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let output_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("wgpu_fft.test.fused_prime.output"),
        size: byte_len,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let readback = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("wgpu_fft.test.fused_prime.readback"),
        size: byte_len,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    queue.write_buffer(&input_buffer, 0, bytemuck::cast_slice(input));

    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("wgpu_fft.test.fused_prime.encoder"),
    });
    plan.execute_checked(device, &mut encoder, &input_buffer, &output_buffer)
        .unwrap();
    encoder.copy_buffer_to_buffer(&output_buffer, 0, &readback, 0, byte_len);
    queue.submit([encoder.finish()]);

    let actual = read_f32(device, &readback);
    (actual, plan)
}

fn test_signal(complex_len: usize) -> Vec<f32> {
    (0..complex_len)
        .flat_map(|index| {
            let x = index as f64 + 1.0;
            [
                ((x * 0.017).sin() * 0.7 + (x * 0.031).cos() * 0.2) as f32,
                ((x * 0.023).cos() * 0.5 - (x * 0.011).sin() * 0.3) as f32,
            ]
        })
        .collect()
}

fn assert_matches_reference(actual: &[f32], expected: &[Complex64], label: &str) {
    assert_eq!(actual.len(), expected.len() * 2, "{label}");
    for (index, (pair, expected)) in actual.chunks_exact(2).zip(expected).enumerate() {
        let dr = f64::from(pair[0]) - expected.re;
        let di = f64::from(pair[1]) - expected.im;
        let error = dr.hypot(di);
        let tolerance = 5.0e-2 + 2.0e-5 * expected.re.hypot(expected.im);
        assert!(
            error.is_finite() && error <= tolerance,
            "{label}: complex index {index}: actual=({}, {}), expected=({}, {}), error={error}, tolerance={tolerance}",
            pair[0],
            pair[1],
            expected.re,
            expected.im,
        );
    }
}

fn assert_close_f32(actual: &[f32], expected: &[f32], label: &str) {
    assert_eq!(actual.len(), expected.len(), "{label}");
    for (index, (&actual, &expected)) in actual.iter().zip(expected).enumerate() {
        let tolerance = 5.0e-2 + 2.0e-5 * expected.abs();
        assert!(
            actual.is_finite() && (actual - expected).abs() <= tolerance,
            "{label}: index {index}: actual={actual}, expected={expected}, tolerance={tolerance}"
        );
    }
}

fn relative_error_metrics(actual: &[f32], expected: &[Complex64]) -> (f64, f64) {
    let mut max_error = 0.0f64;
    let mut max_reference = 0.0f64;
    let mut error_energy = 0.0f64;
    let mut reference_energy = 0.0f64;
    for (pair, expected) in actual.chunks_exact(2).zip(expected) {
        let dr = f64::from(pair[0]) - expected.re;
        let di = f64::from(pair[1]) - expected.im;
        let error2 = dr * dr + di * di;
        let reference2 = expected.re * expected.re + expected.im * expected.im;
        max_error = max_error.max(error2.sqrt());
        max_reference = max_reference.max(reference2.sqrt());
        error_energy += error2;
        reference_energy += reference2;
    }
    (
        max_error / max_reference.max(f64::MIN_POSITIVE),
        (error_energy / reference_energy.max(f64::MIN_POSITIVE)).sqrt(),
    )
}

fn read_f32(device: &wgpu::Device, buffer: &wgpu::Buffer) -> Vec<f32> {
    let slice = buffer.slice(..);
    let (sender, receiver) = mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |result| {
        sender.send(result).expect("map result receiver is alive");
    });
    device
        .poll(wgpu::PollType::wait_indefinitely())
        .expect("device polling should succeed");
    receiver
        .recv()
        .expect("map callback should send a result")
        .expect("readback buffer should map");
    let mapped = slice
        .get_mapped_range()
        .expect("readback buffer should be mapped");
    let values = bytemuck::cast_slice(&mapped).to_vec();
    drop(mapped);
    buffer.unmap();
    values
}
