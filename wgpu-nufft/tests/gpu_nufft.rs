use wgpu_fft::{FftConfig, FftDirection, FftPlan, Normalization};
use wgpu_nufft::{NufftConfig, NufftPlan, NufftSign};

#[test]
fn nufft_fine_grid_builds_public_gpu_fft_plans() {
    if std::env::var_os("WGPU_FFT_RUN_GPU_TESTS").is_none() {
        eprintln!("skipping GPU test; set WGPU_FFT_RUN_GPU_TESTS=1 to run it");
        return;
    }
    pollster::block_on(run_gpu_plan_case());
}

async fn run_gpu_plan_case() {
    let Some(context) = wgpu_fft::device::request_default_device().await else {
        panic!("WGPU_FFT_RUN_GPU_TESTS was set but no suitable adapter was found");
    };
    let info = context.adapter.get_info();
    eprintln!(
        "gpu_nufft adapter: {} backend={:?} driver={} {}",
        info.name, info.backend, info.driver, info.driver_info
    );

    for (sign, make_plan) in [
        (
            NufftSign::Positive,
            NufftPlan::type1 as fn(NufftConfig) -> wgpu_nufft::Result<NufftPlan>,
        ),
        (NufftSign::Negative, NufftPlan::type2),
    ] {
        let nufft = make_plan(NufftConfig::new([17], 1.0e-6).with_sign(sign)).unwrap();
        assert_eq!(nufft.fine_grid_shape(), [36]);
        let direction = match sign {
            NufftSign::Positive => FftDirection::Inverse,
            NufftSign::Negative => FftDirection::Forward,
        };
        let fft_config = FftConfig::new(nufft.fine_grid_shape()[0])
            .with_direction(direction)
            .with_normalization(Normalization::None);
        let fft_plan = FftPlan::c2c(&context.device, &context.queue, fft_config).unwrap();
        assert_eq!(fft_plan.config().shape(), nufft.fine_grid_shape());
    }

    // Retain native wgpu objects until process teardown; this is the repository's
    // established workaround for intermittent Windows backend teardown stalls.
    std::mem::forget(context);
}
