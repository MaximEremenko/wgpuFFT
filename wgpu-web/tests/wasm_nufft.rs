#![cfg(target_arch = "wasm32")]

use js_sys::{Float64Array, Uint32Array};
use wasm_bindgen_test::*;
use wgpu_web::{
    WebFftPrecision, WebNufftModeOrder, WgpuFft, WgpuFftBuffer, WgpuNufftPlan, WgpuNufftType3Plan,
};

wasm_bindgen_test_configure!(run_in_browser);

fn u32_array(values: &[u32]) -> Uint32Array {
    Uint32Array::from(values)
}

fn f64_array(values: &[f64]) -> Float64Array {
    Float64Array::from(values)
}

fn f32_words(bytes: &[u8]) -> Vec<f32> {
    bytes
        .chunks_exact(4)
        .map(|word| f32::from_le_bytes(word.try_into().unwrap()))
        .collect()
}

fn assert_complex_constant(words: &[f32], values_per_batch: usize, expected: &[f32]) {
    assert_eq!(words.len(), values_per_batch * expected.len() * 2);
    for (batch, &expected_re) in expected.iter().enumerate() {
        let tolerance = 8.0e-3 * expected_re.abs().max(1.0);
        for value in 0..values_per_batch {
            let offset = 2 * (batch * values_per_batch + value);
            assert!(
                (words[offset] - expected_re).abs() <= tolerance,
                "batch {batch} value {value}: {} vs {expected_re}",
                words[offset]
            );
            assert!(words[offset + 1].abs() <= 8.0e-3);
        }
    }
}

async fn output_words(context: &WgpuFft, output: &WgpuFftBuffer) -> Vec<f32> {
    f32_words(&context.download(output).await.unwrap())
}

async fn execute_type1(context: &WgpuFft) {
    let plan: WgpuNufftPlan = context
        .create_nufft_type1_plan(
            u32_array(&[8]),
            1,
            2,
            1.0e-3,
            1,
            WebNufftModeOrder::Fft,
            2.0,
            WebFftPrecision::F32,
        )
        .await
        .unwrap();
    assert_eq!(plan.kind(), "type-1");
    assert_eq!(plan.dimensions(), 1);
    assert_eq!(plan.point_bytes(), 4.0);
    assert_eq!(plan.input_bytes(), 16.0);
    assert_eq!(plan.output_bytes(), 128.0);

    let points = context.upload(&[0.0]).unwrap();
    let strengths = context.upload(&[1.0, 0.0, 2.0, 0.0]).unwrap();
    let output = context.create_buffer(plan.output_bytes() as u32).unwrap();
    plan.execute(&points, &strengths, &output, 2).await.unwrap();
    assert_complex_constant(&output_words(context, &output).await, 8, &[1.0, 2.0]);
}

async fn execute_type2(context: &WgpuFft) {
    let plan: WgpuNufftPlan = context
        .create_nufft_type2_plan(
            u32_array(&[8]),
            1,
            2,
            1.0e-3,
            1,
            WebNufftModeOrder::Fft,
            2.0,
            WebFftPrecision::F32,
        )
        .await
        .unwrap();
    assert_eq!(plan.kind(), "type-2");
    assert_eq!(plan.point_bytes(), 4.0);
    assert_eq!(plan.input_bytes(), 128.0);
    assert_eq!(plan.output_bytes(), 16.0);

    let points = context.upload(&[0.0]).unwrap();
    let mut coefficients = vec![0.0f32; 2 * 8 * 2];
    coefficients[0] = 1.0;
    coefficients[2 * 8] = 2.0;
    let coefficients = context.upload(&coefficients).unwrap();
    let output = context.create_buffer(plan.output_bytes() as u32).unwrap();
    plan.execute(&points, &coefficients, &output, 2)
        .await
        .unwrap();
    assert_complex_constant(&output_words(context, &output).await, 1, &[1.0, 2.0]);
}

async fn execute_type3(context: &WgpuFft) {
    let plan: WgpuNufftType3Plan = context
        .create_nufft_type3_plan(
            f64_array(&[-1.0, 1.0]),
            f64_array(&[-1.0, 1.0]),
            1,
            2,
            2,
            1.0e-3,
            1,
            2.0,
            WebFftPrecision::F32,
        )
        .await
        .unwrap();
    assert_eq!(plan.dimensions(), 1);
    assert_eq!(plan.source_point_bytes(), 4.0);
    assert_eq!(plan.target_point_bytes(), 8.0);
    assert_eq!(plan.strength_bytes(), 16.0);
    assert_eq!(plan.output_bytes(), 32.0);

    let source_points = context.upload(&[0.0]).unwrap();
    let strengths = context.upload(&[1.0, 0.0, 2.0, 0.0]).unwrap();
    let target_points = context.upload(&[0.0, 0.5]).unwrap();
    let output = context.create_buffer(plan.output_bytes() as u32).unwrap();
    plan.execute(&source_points, &strengths, &target_points, &output, 2)
        .await
        .unwrap();
    assert_complex_constant(&output_words(context, &output).await, 2, &[1.0, 2.0]);
}

#[wasm_bindgen_test]
async fn default_limit_context_executes_all_three_nufft_types() {
    let context = WgpuFft::init_with_default_limits().await.unwrap();
    assert_eq!(
        context.max_storage_buffer_binding_size(),
        (128 * 1024 * 1024) as f64
    );
    assert_eq!(context.max_buffer_size(), (256 * 1024 * 1024) as f64);
    assert_eq!(context.max_compute_workgroup_storage_size(), 16 * 1024);
    assert_eq!(context.max_compute_invocations_per_workgroup(), 256);
    if context.df64_available() {
        assert_eq!(context.df64_canary_words(), 96);
    } else {
        assert!(context.df64_canary_error().is_some());
    }

    execute_type1(&context).await;
    execute_type2(&context).await;
    execute_type3(&context).await;

    let source = [1.0 / 3.0, -1.0 / 7.0];
    let split = context.upload_df64(f64_array(&source)).unwrap();
    let words = output_words(&context, &split).await;
    assert_eq!(words.len(), 4);
    for (index, &expected) in source.iter().enumerate() {
        let actual = f64::from(words[2 * index]) + f64::from(words[2 * index + 1]);
        assert!((actual - expected).abs() <= f64::from(f32::EPSILON).powi(2));
    }
}
