//! Wall-clock timings of the CPU backend.
//!
//! ```text
//! cargo bench --bench cpu_fft_bench -- [--runs N] [--f64]
//! ```

use std::time::{Duration, Instant};

use wgpu_fft::{CpuFftPlan, FftConfig, FftDirection, FftPrecision, Normalization};

fn main() {
    let mut runs = 5;
    let mut precision = FftPrecision::F32;
    // Cargo passes `--bench` to harness-free benchmarks.
    let mut arguments = std::env::args().skip(1).filter(|arg| arg != "--bench");
    while let Some(argument) = arguments.next() {
        match argument.as_str() {
            "--runs" => {
                runs = arguments
                    .next()
                    .and_then(|value| value.parse().ok())
                    .filter(|&runs| runs > 0)
                    .expect("--runs takes a positive count");
            }
            "--f64" => precision = FftPrecision::F64,
            "--help" | "-h" => {
                println!("usage: cargo bench --bench cpu_fft_bench -- [--runs N] [--f64]");
                return;
            }
            other => panic!("unknown argument {other:?}"),
        }
    }
    let shapes: [&[usize]; 6] = [
        &[2_000_000],
        &[1 << 21],
        &[2048, 2048],
        &[512, 512],
        &[256, 256, 256],
        &[64, 64, 64],
    ];
    for shape in shapes {
        let config = FftConfig::new_nd(shape.to_vec())
            .with_direction(FftDirection::Inverse)
            .with_normalization(Normalization::None)
            .with_precision(precision);
        let plan = CpuFftPlan::c2c(config).expect("plan");
        let median = match precision {
            FftPrecision::F64 => time(runs, plan.required_input_len(), |input, output| {
                plan.execute_f64(input, output).expect("execute")
            }),
            _ => time(runs, plan.required_input_len(), |input, output| {
                plan.execute(input, output).expect("execute")
            }),
        };
        let elements = shape.iter().product::<usize>();
        println!(
            "CPU_FFT shape={shape:?} {precision:?} median={:.2} ms ({:.0} Melem/s)",
            median.as_secs_f64() * 1.0e3,
            elements as f64 / median.as_secs_f64() / 1.0e6,
        );
    }
}

/// Median time of `runs` executions after one warm-up.
fn time<T: Copy + Default + From<u8>>(
    runs: usize,
    len: usize,
    mut execute: impl FnMut(&[T], &mut [T]),
) -> Duration {
    let input = (0..len)
        .map(|index| T::from((index % 7) as u8))
        .collect::<Vec<_>>();
    let mut output = vec![T::default(); len];
    execute(&input, &mut output);
    let mut times = (0..runs)
        .map(|_| {
            let start = Instant::now();
            execute(&input, &mut output);
            start.elapsed()
        })
        .collect::<Vec<_>>();
    times.sort();
    times[runs / 2]
}
