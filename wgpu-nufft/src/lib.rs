//! Portable GPU nonuniform FFT plans built on `wgpu-fft`.
//!
//! Planning and execution APIs arrive in the next implementation phase. This
//! crate is a separate workspace member so it can depend only on `wgpu-fft`'s
//! public surface.

#[cfg(test)]
mod tests {
    use wgpu_fft::FftConfig;

    #[test]
    fn workspace_path_dependency_exposes_the_public_fft_api() {
        FftConfig::new(8).validate().unwrap();
    }
}
