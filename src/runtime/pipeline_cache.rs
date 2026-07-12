use std::cell::RefCell;
use std::collections::hash_map::DefaultHasher;
use std::collections::HashMap;
use std::hash::{Hash, Hasher};

use crate::config::FftDirection;
use crate::runtime::axis_plan::AxisPrecision;

pub const PIPELINE_CACHE_SNAPSHOT_SCHEMA: &str = "wgpu-fft.pipeline-cache";
pub const PIPELINE_CACHE_SNAPSHOT_VERSION: u32 = 1;

thread_local! {
    static DEVICE_CACHES: RefCell<HashMap<u64, PipelineCache>> = RefCell::new(HashMap::new());
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PipelineCacheSnapshot {
    schema: &'static str,
    version: u32,
    shader_codes: Vec<String>,
    pipeline_keys: Vec<String>,
    shader_entries: Vec<SnapshotShaderEntry>,
    pipeline_entries: Vec<ComputePipelineCacheKey>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct SnapshotShaderEntry {
    key: ShaderCacheKey,
    code: String,
}

impl PipelineCacheSnapshot {
    pub fn empty() -> Self {
        Self::from_entries(Vec::new(), Vec::new())
    }

    pub fn schema(&self) -> &str {
        self.schema
    }

    pub fn version(&self) -> u32 {
        self.version
    }

    pub fn shader_codes(&self) -> &[String] {
        &self.shader_codes
    }

    pub fn pipeline_keys(&self) -> &[String] {
        &self.pipeline_keys
    }

    pub fn is_empty(&self) -> bool {
        self.shader_codes.is_empty() && self.pipeline_keys.is_empty()
    }

    fn from_entries(
        mut shader_entries: Vec<SnapshotShaderEntry>,
        mut pipeline_entries: Vec<ComputePipelineCacheKey>,
    ) -> Self {
        shader_entries.sort_by_key(|entry| entry.key.stable_key());
        pipeline_entries.sort_by_key(|key| key.stable_key());

        let shader_codes = shader_entries
            .iter()
            .map(|entry| entry.code.clone())
            .collect();
        let pipeline_keys = pipeline_entries
            .iter()
            .map(ComputePipelineCacheKey::stable_key)
            .collect();

        Self {
            schema: PIPELINE_CACHE_SNAPSHOT_SCHEMA,
            version: PIPELINE_CACHE_SNAPSHOT_VERSION,
            shader_codes,
            pipeline_keys,
            shader_entries,
            pipeline_entries,
        }
    }
}

impl Default for PipelineCacheSnapshot {
    fn default() -> Self {
        Self::empty()
    }
}

#[derive(Default)]
pub(crate) struct PipelineCache {
    bind_group_layouts: HashMap<PipelineLayoutCacheKey, wgpu::BindGroupLayout>,
    pipeline_layouts: HashMap<PipelineLayoutCacheKey, wgpu::PipelineLayout>,
    shader_modules: HashMap<ShaderCacheKey, wgpu::ShaderModule>,
    shader_sources: HashMap<ShaderCacheKey, String>,
    compute_pipelines: HashMap<ComputePipelineCacheKey, wgpu::ComputePipeline>,
}

pub(crate) fn with_device_pipeline_cache<R>(
    device: &wgpu::Device,
    f: impl FnOnce(&mut PipelineCache) -> R,
) -> R {
    let cache_id = device_cache_id(device);
    DEVICE_CACHES.with(|caches| {
        let mut caches = caches.borrow_mut();
        let cache = caches.entry(cache_id).or_default();
        f(cache)
    })
}

/// Drops cached shader modules, layouts, and compute pipelines for `device`
/// from the calling thread's cache.
///
/// Existing plans keep their own `wgpu` handles and remain usable. This is
/// primarily useful for long-running tools that create many shape-specialized
/// plans and no longer need earlier cache entries. The return value is `true`
/// only when an entry existed and was removed; it does not guarantee immediate
/// driver-level memory reclamation or affect entries on other threads.
pub fn clear_thread_local_pipeline_cache(device: &wgpu::Device) -> bool {
    let cache_id = device_cache_id(device);
    DEVICE_CACHES.with(|caches| caches.borrow_mut().remove(&cache_id).is_some())
}

pub fn export_pipeline_cache_snapshot(device: &wgpu::Device) -> PipelineCacheSnapshot {
    with_device_pipeline_cache(device, |cache| cache.export_snapshot())
}

pub fn import_pipeline_cache_snapshot(
    device: &wgpu::Device,
    snapshot: &PipelineCacheSnapshot,
) -> PipelineCacheSnapshot {
    with_device_pipeline_cache(device, |cache| {
        cache.import_snapshot(device, snapshot);
        cache.export_snapshot()
    })
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) enum PipelineLayoutCacheKey {
    AxisPlanInterleavedF32Lut,
    AxisPlanInterleavedF64Lut,
    BridgeReadWriteReadF32,
    BridgeReadWriteUniformF32,
    BridgeTwoWriteUniformF32,
    BridgeWriteReadUniformF32,
    BluesteinBridgePostF32,
    BluesteinPackInterleavedF32,
    BluesteinPackInterleavedF64,
    BluesteinMulInterleavedF32,
    BluesteinMulInterleavedF64,
    BluesteinPostInterleavedF32,
    BluesteinPostInterleavedF64,
    C2cSmoothBinaryF32,
    C2cSmoothTwiddleLutF32,
    C2cStridedBinaryF32,
    C2cStridedBinaryF64,
    DirectDftInterleavedF32Lut,
    DirectDftInterleavedF64Lut,
    FusedPrimeInterleavedF32,
    FusedPrimeInterleavedF64,
    FourStepUnaryF32,
    RealBinaryF32,
    RaderBridgePostF32,
    RaderSumInterleavedF32,
    RaderSumInterleavedF64,
    RaderPackInterleavedF32,
    RaderPackInterleavedF64,
    RaderMulInterleavedF32,
    RaderMulInterleavedF64,
    RaderWriteY0InterleavedF32,
    RaderWriteY0InterleavedF64,
    RaderPostInterleavedF32,
    RaderPostInterleavedF64,
}

impl PipelineLayoutCacheKey {
    fn stable_key(self) -> &'static str {
        match self {
            Self::AxisPlanInterleavedF32Lut => "axis-plan/interleaved-f32-lut",
            Self::AxisPlanInterleavedF64Lut => "axis-plan/interleaved-f64-lut",
            Self::BridgeReadWriteReadF32 => "bridge/read-write-read-f32",
            Self::BridgeReadWriteUniformF32 => "bridge/read-write-uniform-f32",
            Self::BridgeTwoWriteUniformF32 => "bridge/two-write-uniform-f32",
            Self::BridgeWriteReadUniformF32 => "bridge/write-read-uniform-f32",
            Self::BluesteinBridgePostF32 => "bridge/bluestein-post-f32",
            Self::BluesteinPackInterleavedF32 => "bluestein/pack/interleaved-f32",
            Self::BluesteinPackInterleavedF64 => "bluestein/pack/interleaved-f64",
            Self::BluesteinMulInterleavedF32 => "bluestein/mul/interleaved-f32",
            Self::BluesteinMulInterleavedF64 => "bluestein/mul/interleaved-f64",
            Self::BluesteinPostInterleavedF32 => "bluestein/post/interleaved-f32",
            Self::BluesteinPostInterleavedF64 => "bluestein/post/interleaved-f64",
            Self::C2cSmoothBinaryF32 => "c2c-smooth/binary-f32",
            Self::C2cSmoothTwiddleLutF32 => "c2c-smooth/twiddle-lut-f32",
            Self::C2cStridedBinaryF32 => "c2c-strided/binary-f32",
            Self::C2cStridedBinaryF64 => "c2c-strided/binary-f64",
            Self::DirectDftInterleavedF32Lut => "direct-dft/interleaved-f32-lut",
            Self::DirectDftInterleavedF64Lut => "direct-dft/interleaved-f64-lut",
            Self::FusedPrimeInterleavedF32 => "fused-prime/interleaved-f32",
            Self::FusedPrimeInterleavedF64 => "fused-prime/interleaved-f64",
            Self::FourStepUnaryF32 => "four-step/unary-f32",
            Self::RealBinaryF32 => "real/binary-f32",
            Self::RaderBridgePostF32 => "bridge/rader-post-f32",
            Self::RaderSumInterleavedF32 => "rader/sum/interleaved-f32",
            Self::RaderSumInterleavedF64 => "rader/sum/interleaved-f64",
            Self::RaderPackInterleavedF32 => "rader/pack/interleaved-f32",
            Self::RaderPackInterleavedF64 => "rader/pack/interleaved-f64",
            Self::RaderMulInterleavedF32 => "rader/mul/interleaved-f32",
            Self::RaderMulInterleavedF64 => "rader/mul/interleaved-f64",
            Self::RaderWriteY0InterleavedF32 => "rader/write-y0/interleaved-f32",
            Self::RaderWriteY0InterleavedF64 => "rader/write-y0/interleaved-f64",
            Self::RaderPostInterleavedF32 => "rader/post/interleaved-f32",
            Self::RaderPostInterleavedF64 => "rader/post/interleaved-f64",
        }
    }
}

impl PipelineCache {
    pub(crate) fn get_bind_group_layout(
        &mut self,
        device: &wgpu::Device,
        key: PipelineLayoutCacheKey,
    ) -> wgpu::BindGroupLayout {
        if let Some(layout) = self.bind_group_layouts.get(&key) {
            return layout.clone();
        }

        let label = format!(
            "wgpu_fft.pipeline_cache.bind_group_layout.{}",
            key.stable_key()
        );
        let entries = bind_group_layout_entries(key);
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some(&label),
            entries: &entries,
        });
        self.bind_group_layouts.insert(key, layout.clone());
        layout
    }

    pub(crate) fn get_pipeline_layout(
        &mut self,
        device: &wgpu::Device,
        key: PipelineLayoutCacheKey,
    ) -> wgpu::PipelineLayout {
        if let Some(layout) = self.pipeline_layouts.get(&key) {
            return layout.clone();
        }

        let bind_group_layout = self.get_bind_group_layout(device, key);
        let label = format!(
            "wgpu_fft.pipeline_cache.pipeline_layout.{}",
            key.stable_key()
        );
        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some(&label),
            bind_group_layouts: &[Some(&bind_group_layout)],
            immediate_size: 0,
        });
        self.pipeline_layouts.insert(key, layout.clone());
        layout
    }

    pub(crate) fn get_shader_module(
        &mut self,
        device: &wgpu::Device,
        key: &ShaderCacheKey,
        label: &str,
        source: impl FnOnce() -> String,
    ) -> wgpu::ShaderModule {
        if let Some(module) = self.shader_modules.get(key) {
            return module.clone();
        }

        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some(label),
            source: wgpu::ShaderSource::Wgsl({
                let source = source();
                self.shader_sources.insert(key.clone(), source.clone());
                source.into()
            }),
        });
        self.shader_modules.insert(key.clone(), module.clone());
        module
    }

    pub(crate) fn get_compute_pipeline(
        &mut self,
        device: &wgpu::Device,
        key: &ComputePipelineCacheKey,
        label: &str,
        shader_label: &str,
        shader_source: impl FnOnce() -> String,
    ) -> wgpu::ComputePipeline {
        if let Some(pipeline) = self.compute_pipelines.get(key) {
            return pipeline.clone();
        }

        let pipeline_layout = self.get_pipeline_layout(device, key.layout);
        let shader = self.get_shader_module(device, &key.shader, shader_label, shader_source);
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some(label),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some(&key.entry_point),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });
        self.compute_pipelines.insert(key.clone(), pipeline.clone());
        pipeline
    }

    fn export_snapshot(&self) -> PipelineCacheSnapshot {
        let shader_entries = self
            .shader_sources
            .iter()
            .map(|(key, code)| SnapshotShaderEntry {
                key: key.clone(),
                code: code.clone(),
            })
            .collect();
        let pipeline_entries = self.compute_pipelines.keys().cloned().collect();
        PipelineCacheSnapshot::from_entries(shader_entries, pipeline_entries)
    }

    fn import_snapshot(&mut self, device: &wgpu::Device, snapshot: &PipelineCacheSnapshot) {
        for entry in &snapshot.shader_entries {
            if !entry.key.is_supported_on_device(device) {
                continue;
            }
            let shader_label = format!(
                "wgpu_fft.pipeline_cache.import.shader.{}",
                entry.key.stable_key()
            );
            self.get_shader_module(device, &entry.key, &shader_label, || entry.code.clone());
        }

        let shader_sources = snapshot
            .shader_entries
            .iter()
            .map(|entry| (entry.key.clone(), entry.code.clone()))
            .collect::<HashMap<_, _>>();
        for key in &snapshot.pipeline_entries {
            if !key.shader.is_supported_on_device(device) {
                continue;
            }
            let pipeline_label = format!(
                "wgpu_fft.pipeline_cache.import.pipeline.{}",
                key.stable_key()
            );
            let shader_label = format!(
                "wgpu_fft.pipeline_cache.import.shader.{}",
                key.shader.stable_key()
            );
            let shader_source = shader_sources
                .get(&key.shader)
                .cloned()
                .or_else(|| self.shader_sources.get(&key.shader).cloned())
                .unwrap_or_else(|| key.shader.fallback_source());
            self.get_compute_pipeline(device, key, &pipeline_label, &shader_label, || {
                shader_source
            });
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub(crate) enum ShaderCacheKey {
    StockhamStage(StockhamStageKey),
    FusedPow2Stage(FusedPow2StageKey),
    FusedSmoothStage(FusedSmoothStageKey),
    FusedPrimeStage(FusedPrimeStageKey),
    FourStepStage(FourStepStageKey),
    BridgeStage(BridgeStageKey),
    RaderStage(RaderStageKey),
    BluesteinStage(BluesteinStageKey),
    RealStage(RealStageKey),
    C2cSmoothStage(C2cSmoothStageKey),
    C2cStridedStage(C2cStridedStageKey),
    DirectDftC2cLut(AxisPrecision),
}

impl ShaderCacheKey {
    pub(crate) fn stable_key(&self) -> String {
        match self {
            Self::StockhamStage(key) => key.stable_key(),
            Self::FusedPow2Stage(key) => key.stable_key(),
            Self::FusedSmoothStage(key) => key.stable_key(),
            Self::FusedPrimeStage(key) => key.stable_key(),
            Self::FourStepStage(key) => key.stable_key(),
            Self::BridgeStage(key) => key.stable_key(),
            Self::RaderStage(key) => key.stable_key(),
            Self::BluesteinStage(key) => key.stable_key(),
            Self::RealStage(key) => key.stable_key(),
            Self::C2cSmoothStage(key) => key.stable_key(),
            Self::C2cStridedStage(key) => key.stable_key(),
            Self::DirectDftC2cLut(precision) => format!(
                "shader:v3:direct-dft/c2c-{}:twiddle=host-f64-{}-v1",
                precision.as_str(),
                precision.as_str()
            ),
        }
    }

    fn fallback_source(&self) -> String {
        match self {
            Self::StockhamStage(key) => {
                crate::runtime::axis_plan::generate_stockham_radix_stage_wgsl_for_key(key)
            }
            Self::FusedPow2Stage(key) => {
                crate::runtime::axis_plan::generate_fused_pow2_stage_wgsl_for_key(key)
            }
            Self::FusedSmoothStage(key) => {
                crate::runtime::axis_plan::generate_fused_smooth_stage_wgsl_for_key(key)
            }
            Self::FusedPrimeStage(key) => match key.kind {
                FusedPrimeKind::Rader => {
                    crate::runtime::rader_axis::generate_fused_rader_wgsl_for_key(key)
                }
                FusedPrimeKind::Bluestein => {
                    crate::runtime::bluestein_axis::generate_fused_bluestein_wgsl_for_key(key)
                }
            },
            Self::FourStepStage(key) => {
                crate::runtime::four_step::generate_four_step_wgsl_for_key(key)
            }
            Self::BridgeStage(key) => crate::runtime::c2c::generate_bridge_wgsl_for_key(key),
            Self::RaderStage(key) => crate::runtime::rader_axis::generate_rader_wgsl_for_key(key),
            Self::BluesteinStage(key) => {
                crate::runtime::bluestein_axis::generate_bluestein_wgsl_for_key(key)
            }
            Self::RealStage(key) => crate::runtime::real::generate_real_wgsl_for_key(key),
            Self::C2cSmoothStage(key) => crate::runtime::c2c::generate_c2c_smooth_wgsl_for_key(key),
            Self::C2cStridedStage(key) => {
                crate::runtime::c2c::generate_c2c_strided_wgsl_for_key(key)
            }
            Self::DirectDftC2cLut(precision) => {
                crate::runtime::c2c::generate_direct_dft_wgsl(*precision)
            }
        }
    }

    fn is_supported_on_device(&self, device: &wgpu::Device) -> bool {
        if self.precision() == Some(AxisPrecision::F64)
            && !device.features().contains(wgpu::Features::SHADER_F64)
        {
            return false;
        }
        match self {
            Self::FusedPow2Stage(key) => {
                let limits = device.limits();
                key.is_supported_by_limits(
                    u64::from(limits.max_compute_workgroup_storage_size),
                    limits.max_compute_invocations_per_workgroup,
                    limits.max_compute_workgroup_size_x,
                )
            }
            Self::FusedSmoothStage(key) => {
                let limits = device.limits();
                key.is_supported_by_limits(
                    u64::from(limits.max_compute_workgroup_storage_size),
                    limits.max_compute_invocations_per_workgroup,
                    limits.max_compute_workgroup_size_x,
                )
            }
            Self::FusedPrimeStage(key) => {
                let limits = device.limits();
                key.is_supported_by_limits(
                    u64::from(limits.max_compute_workgroup_storage_size),
                    limits.max_compute_invocations_per_workgroup,
                    limits.max_compute_workgroup_size_x,
                )
            }
            _ => true,
        }
    }

    fn precision(&self) -> Option<AxisPrecision> {
        match self {
            Self::StockhamStage(key) => Some(key.precision),
            Self::FusedPow2Stage(key) => Some(key.precision),
            Self::FusedSmoothStage(key) => Some(key.precision),
            Self::DirectDftC2cLut(precision) => Some(*precision),
            Self::C2cStridedStage(key) => Some(key.precision),
            Self::RaderStage(key) => Some(key.precision),
            Self::BluesteinStage(key) => Some(key.precision),
            Self::FusedPrimeStage(key) => Some(key.precision),
            _ => None,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub(crate) struct ComputePipelineCacheKey {
    pub(crate) layout: PipelineLayoutCacheKey,
    pub(crate) entry_point: String,
    pub(crate) shader: ShaderCacheKey,
}

impl ComputePipelineCacheKey {
    pub(crate) fn stockham_stage(shader: StockhamStageKey) -> Self {
        let layout = axis_plan_layout_for_precision(shader.precision);
        Self {
            layout,
            entry_point: String::from("main"),
            shader: ShaderCacheKey::StockhamStage(shader),
        }
    }

    pub(crate) fn fused_pow2_stage(shader: FusedPow2StageKey) -> Self {
        let layout = axis_plan_layout_for_precision(shader.precision);
        Self {
            layout,
            entry_point: String::from("main"),
            shader: ShaderCacheKey::FusedPow2Stage(shader),
        }
    }

    pub(crate) fn fused_smooth_stage(shader: FusedSmoothStageKey) -> Self {
        let layout = axis_plan_layout_for_precision(shader.precision);
        Self {
            layout,
            entry_point: String::from("main"),
            shader: ShaderCacheKey::FusedSmoothStage(shader),
        }
    }

    pub(crate) fn fused_prime_stage(shader: FusedPrimeStageKey) -> Self {
        let layout = match shader.precision {
            AxisPrecision::F32 => PipelineLayoutCacheKey::FusedPrimeInterleavedF32,
            AxisPrecision::F64 => PipelineLayoutCacheKey::FusedPrimeInterleavedF64,
        };
        Self {
            layout,
            entry_point: String::from("main"),
            shader: ShaderCacheKey::FusedPrimeStage(shader),
        }
    }

    pub(crate) fn four_step_stage(shader: FourStepStageKey) -> Self {
        let layout = match shader.kind {
            FourStepKernelKind::StripeTranspose => PipelineLayoutCacheKey::C2cSmoothBinaryF32,
            FourStepKernelKind::Scale => PipelineLayoutCacheKey::FourStepUnaryF32,
        };
        Self {
            layout,
            entry_point: String::from("main"),
            shader: ShaderCacheKey::FourStepStage(shader),
        }
    }

    pub(crate) fn direct_dft_c2c(precision: AxisPrecision) -> Self {
        Self {
            layout: match precision {
                AxisPrecision::F32 => PipelineLayoutCacheKey::DirectDftInterleavedF32Lut,
                AxisPrecision::F64 => PipelineLayoutCacheKey::DirectDftInterleavedF64Lut,
            },
            entry_point: String::from("main"),
            shader: ShaderCacheKey::DirectDftC2cLut(precision),
        }
    }

    #[cfg(test)]
    pub(crate) fn direct_dft_c2c_f32() -> Self {
        Self::direct_dft_c2c(AxisPrecision::F32)
    }

    pub(crate) fn real_stage(shader: RealStageKey) -> Self {
        Self {
            layout: PipelineLayoutCacheKey::RealBinaryF32,
            entry_point: String::from("main"),
            shader: ShaderCacheKey::RealStage(shader),
        }
    }

    pub(crate) fn c2c_strided_stage(shader: C2cStridedStageKey) -> Self {
        let layout = match shader.precision {
            AxisPrecision::F32 => PipelineLayoutCacheKey::C2cStridedBinaryF32,
            AxisPrecision::F64 => PipelineLayoutCacheKey::C2cStridedBinaryF64,
        };
        Self {
            layout,
            entry_point: String::from("main"),
            shader: ShaderCacheKey::C2cStridedStage(shader),
        }
    }

    pub(crate) fn c2c_smooth_stage(shader: C2cSmoothStageKey) -> Self {
        let layout = match shader.kind {
            C2cSmoothKernelKind::TwiddleTranspose => PipelineLayoutCacheKey::C2cSmoothTwiddleLutF32,
            C2cSmoothKernelKind::GatherAxisLine
            | C2cSmoothKernelKind::ScatterAxisLine
            | C2cSmoothKernelKind::GatherSmoothPhase1
            | C2cSmoothKernelKind::ScatterSmoothPhase2 => {
                PipelineLayoutCacheKey::C2cSmoothBinaryF32
            }
        };
        Self {
            layout,
            entry_point: String::from("main"),
            shader: ShaderCacheKey::C2cSmoothStage(shader),
        }
    }

    pub(crate) fn bridge_stage(shader: BridgeStageKey) -> Self {
        let layout = match shader.kind {
            BridgeKernelKind::RaderSumInit => PipelineLayoutCacheKey::BridgeTwoWriteUniformF32,
            BridgeKernelKind::RaderSumAccumulate => PipelineLayoutCacheKey::RaderSumInterleavedF32,
            BridgeKernelKind::RaderPack | BridgeKernelKind::BluesteinPack => {
                PipelineLayoutCacheKey::BridgeReadWriteReadF32
            }
            BridgeKernelKind::RaderMul | BridgeKernelKind::BluesteinMul => {
                PipelineLayoutCacheKey::BridgeWriteReadUniformF32
            }
            BridgeKernelKind::RaderWriteY0 => PipelineLayoutCacheKey::BridgeReadWriteUniformF32,
            BridgeKernelKind::RaderPost => PipelineLayoutCacheKey::RaderBridgePostF32,
            BridgeKernelKind::BluesteinPost => PipelineLayoutCacheKey::BluesteinBridgePostF32,
        };
        Self {
            layout,
            entry_point: String::from("main"),
            shader: ShaderCacheKey::BridgeStage(shader),
        }
    }

    pub(crate) fn rader_stage(shader: RaderStageKey) -> Self {
        let layout = rader_layout_for(shader.kind, shader.precision);
        Self {
            layout,
            entry_point: String::from("main"),
            shader: ShaderCacheKey::RaderStage(shader),
        }
    }

    pub(crate) fn bluestein_stage(shader: BluesteinStageKey) -> Self {
        let layout = bluestein_layout_for(shader.kind, shader.precision);
        Self {
            layout,
            entry_point: String::from("main"),
            shader: ShaderCacheKey::BluesteinStage(shader),
        }
    }

    pub(crate) fn stable_key(&self) -> String {
        format!(
            "pipeline:v1:layout={}:entry={}:{}",
            self.layout.stable_key(),
            self.entry_point,
            self.shader.stable_key()
        )
    }
}

fn rader_layout_for(kind: RaderKernelKind, precision: AxisPrecision) -> PipelineLayoutCacheKey {
    match (kind, precision) {
        (RaderKernelKind::Sum, AxisPrecision::F32) => {
            PipelineLayoutCacheKey::RaderSumInterleavedF32
        }
        (RaderKernelKind::Sum, AxisPrecision::F64) => {
            PipelineLayoutCacheKey::RaderSumInterleavedF64
        }
        (RaderKernelKind::Pack, AxisPrecision::F32) => {
            PipelineLayoutCacheKey::RaderPackInterleavedF32
        }
        (RaderKernelKind::Pack, AxisPrecision::F64) => {
            PipelineLayoutCacheKey::RaderPackInterleavedF64
        }
        (RaderKernelKind::Mul, AxisPrecision::F32) => {
            PipelineLayoutCacheKey::RaderMulInterleavedF32
        }
        (RaderKernelKind::Mul, AxisPrecision::F64) => {
            PipelineLayoutCacheKey::RaderMulInterleavedF64
        }
        (RaderKernelKind::WriteY0, AxisPrecision::F32) => {
            PipelineLayoutCacheKey::RaderWriteY0InterleavedF32
        }
        (RaderKernelKind::WriteY0, AxisPrecision::F64) => {
            PipelineLayoutCacheKey::RaderWriteY0InterleavedF64
        }
        (RaderKernelKind::Post, AxisPrecision::F32) => {
            PipelineLayoutCacheKey::RaderPostInterleavedF32
        }
        (RaderKernelKind::Post, AxisPrecision::F64) => {
            PipelineLayoutCacheKey::RaderPostInterleavedF64
        }
    }
}

fn bluestein_layout_for(
    kind: BluesteinKernelKind,
    precision: AxisPrecision,
) -> PipelineLayoutCacheKey {
    match (kind, precision) {
        (BluesteinKernelKind::Pack, AxisPrecision::F32) => {
            PipelineLayoutCacheKey::BluesteinPackInterleavedF32
        }
        (BluesteinKernelKind::Pack, AxisPrecision::F64) => {
            PipelineLayoutCacheKey::BluesteinPackInterleavedF64
        }
        (BluesteinKernelKind::Mul, AxisPrecision::F32) => {
            PipelineLayoutCacheKey::BluesteinMulInterleavedF32
        }
        (BluesteinKernelKind::Mul, AxisPrecision::F64) => {
            PipelineLayoutCacheKey::BluesteinMulInterleavedF64
        }
        (BluesteinKernelKind::Post, AxisPrecision::F32) => {
            PipelineLayoutCacheKey::BluesteinPostInterleavedF32
        }
        (BluesteinKernelKind::Post, AxisPrecision::F64) => {
            PipelineLayoutCacheKey::BluesteinPostInterleavedF64
        }
    }
}

fn axis_plan_layout_for_precision(precision: AxisPrecision) -> PipelineLayoutCacheKey {
    match precision {
        AxisPrecision::F32 => PipelineLayoutCacheKey::AxisPlanInterleavedF32Lut,
        AxisPrecision::F64 => PipelineLayoutCacheKey::AxisPlanInterleavedF64Lut,
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) enum RaderKernelKind {
    Sum,
    Pack,
    Mul,
    WriteY0,
    Post,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) enum BluesteinKernelKind {
    Pack,
    Mul,
    Post,
}

impl BluesteinKernelKind {
    pub(crate) const fn as_str(self) -> &'static str {
        match self {
            Self::Pack => "pack",
            Self::Mul => "mul",
            Self::Post => "post",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) enum FusedPrimeKind {
    Rader,
    Bluestein,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) enum FourStepKernelKind {
    StripeTranspose,
    Scale,
}

impl FourStepKernelKind {
    pub(crate) const fn as_str(self) -> &'static str {
        match self {
            Self::StripeTranspose => "stripe-transpose",
            Self::Scale => "scale",
        }
    }
}

impl FusedPrimeKind {
    pub(crate) const fn as_str(self) -> &'static str {
        match self {
            Self::Rader => "rader",
            Self::Bluestein => "bluestein",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) enum BridgeKernelKind {
    RaderSumInit,
    RaderSumAccumulate,
    RaderPack,
    RaderMul,
    RaderWriteY0,
    RaderPost,
    BluesteinPack,
    BluesteinMul,
    BluesteinPost,
}

impl BridgeKernelKind {
    pub(crate) const fn as_str(self) -> &'static str {
        match self {
            Self::RaderSumInit => "rader-sum-init",
            Self::RaderSumAccumulate => "rader-sum-accumulate",
            Self::RaderPack => "rader-pack-windowed",
            Self::RaderMul => "rader-mul-windowed",
            Self::RaderWriteY0 => "rader-write-y0-windowed",
            Self::RaderPost => "rader-post-windowed",
            Self::BluesteinPack => "bluestein-pack-windowed",
            Self::BluesteinMul => "bluestein-mul-windowed",
            Self::BluesteinPost => "bluestein-post-windowed",
        }
    }
}

impl RaderKernelKind {
    pub(crate) const fn as_str(self) -> &'static str {
        match self {
            Self::Sum => "sum",
            Self::Pack => "pack",
            Self::Mul => "mul",
            Self::WriteY0 => "write-y0",
            Self::Post => "post",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) enum RealKernelKind {
    RealToComplex,
    RealToComplexWindowed,
    PackR2c,
    PackR2cWindowed,
    UnpackC2r,
    UnpackC2rWindowed,
    ComplexToReal,
    ComplexToRealWindowed,
    PackRealStrided,
    UnpackRealStrided,
    PackComplexStrided,
    UnpackComplexStrided,
}

impl RealKernelKind {
    pub(crate) const fn as_str(self) -> &'static str {
        match self {
            Self::RealToComplex => "real-to-complex",
            Self::RealToComplexWindowed => "real-to-complex-windowed",
            Self::PackR2c => "pack-r2c",
            Self::PackR2cWindowed => "pack-r2c-windowed",
            Self::UnpackC2r => "unpack-c2r",
            Self::UnpackC2rWindowed => "unpack-c2r-windowed",
            Self::ComplexToReal => "complex-to-real",
            Self::ComplexToRealWindowed => "complex-to-real-windowed",
            Self::PackRealStrided => "pack-real-strided",
            Self::UnpackRealStrided => "unpack-real-strided",
            Self::PackComplexStrided => "pack-complex-strided",
            Self::UnpackComplexStrided => "unpack-complex-strided",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) enum C2cStridedKernelKind {
    Pack,
    Unpack,
}

impl C2cStridedKernelKind {
    pub(crate) const fn as_str(self) -> &'static str {
        match self {
            Self::Pack => "pack-c2c-strided",
            Self::Unpack => "unpack-c2c-strided",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) enum C2cSmoothKernelKind {
    TwiddleTranspose,
    GatherAxisLine,
    ScatterAxisLine,
    GatherSmoothPhase1,
    ScatterSmoothPhase2,
}

impl C2cSmoothKernelKind {
    pub(crate) const fn as_str(self) -> &'static str {
        match self {
            Self::TwiddleTranspose => "twiddle-transpose",
            Self::GatherAxisLine => "gather-axis-line",
            Self::ScatterAxisLine => "scatter-axis-line",
            Self::GatherSmoothPhase1 => "gather-smooth-phase1",
            Self::ScatterSmoothPhase2 => "scatter-smooth-phase2",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub(crate) struct C2cSmoothStageKey {
    pub(crate) kind: C2cSmoothKernelKind,
    pub(crate) workgroup_size: u32,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub(crate) struct FourStepStageKey {
    pub(crate) kind: FourStepKernelKind,
    pub(crate) workgroup_size: u32,
}

impl FourStepStageKey {
    pub(crate) const fn new(kind: FourStepKernelKind, workgroup_size: u32) -> Self {
        Self {
            kind,
            workgroup_size,
        }
    }

    pub(crate) fn stable_key(&self) -> String {
        format!(
            "shader:v1:four-step:{}:workgroup={}",
            self.kind.as_str(),
            self.workgroup_size
        )
    }
}

impl C2cSmoothStageKey {
    pub(crate) const fn new(kind: C2cSmoothKernelKind, workgroup_size: u32) -> Self {
        Self {
            kind,
            workgroup_size,
        }
    }

    pub(crate) fn stable_key(&self) -> String {
        match self.kind {
            C2cSmoothKernelKind::TwiddleTranspose => format!(
                "shader:v2:c2c-smooth:{}:workgroup={}:twiddle=host-f64-two-level-f32-v1",
                self.kind.as_str(),
                self.workgroup_size
            ),
            _ => format!(
                "shader:v1:c2c-smooth:{}:workgroup={}",
                self.kind.as_str(),
                self.workgroup_size
            ),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub(crate) struct C2cStridedStageKey {
    pub(crate) kind: C2cStridedKernelKind,
    pub(crate) workgroup_size: u32,
    pub(crate) precision: AxisPrecision,
}

impl C2cStridedStageKey {
    pub(crate) const fn new(
        kind: C2cStridedKernelKind,
        workgroup_size: u32,
        precision: AxisPrecision,
    ) -> Self {
        Self {
            kind,
            workgroup_size,
            precision,
        }
    }

    pub(crate) fn stable_key(&self) -> String {
        format!(
            "shader:v2:c2c-strided:{}:precision={}:workgroup={}",
            self.kind.as_str(),
            self.precision.as_str(),
            self.workgroup_size
        )
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub(crate) struct RealStageKey {
    pub(crate) kind: RealKernelKind,
    pub(crate) rank: usize,
    pub(crate) dims: Vec<usize>,
    pub(crate) workgroup_size: u32,
}

impl RealStageKey {
    pub(crate) fn new(kind: RealKernelKind, dims: &[usize], workgroup_size: u32) -> Self {
        Self {
            kind,
            rank: dims.len(),
            dims: dims.to_vec(),
            workgroup_size,
        }
    }

    pub(crate) fn stable_key(&self) -> String {
        format!(
            "shader:v1:real:{}:rank={}:dims={}:workgroup={}",
            self.kind.as_str(),
            self.rank,
            dims_key(&self.dims),
            self.workgroup_size
        )
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub(crate) struct BridgeStageKey {
    pub(crate) kind: BridgeKernelKind,
    pub(crate) rank: usize,
    pub(crate) axis: usize,
    pub(crate) dims: Vec<usize>,
    pub(crate) axis_length: usize,
    pub(crate) stride_complex: usize,
    pub(crate) convolution_length: usize,
    pub(crate) workgroup_size: u32,
    pub(crate) apply_scale: bool,
    scale_bits: u32,
}

impl BridgeStageKey {
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn new(
        kind: BridgeKernelKind,
        rank: usize,
        axis: usize,
        dims: &[usize],
        axis_length: usize,
        stride_complex: usize,
        convolution_length: usize,
        workgroup_size: u32,
        apply_scale: bool,
        scale_factor: f32,
    ) -> Self {
        debug_assert_eq!(rank, dims.len());
        debug_assert!(axis < rank);
        debug_assert_eq!(axis_length, dims[axis]);
        debug_assert!(scale_factor.is_finite());

        let scale_bits = if apply_scale {
            scale_factor.to_bits()
        } else {
            1.0f32.to_bits()
        };

        Self {
            kind,
            rank,
            axis,
            dims: dims.to_vec(),
            axis_length,
            stride_complex,
            convolution_length,
            workgroup_size,
            apply_scale,
            scale_bits,
        }
    }

    pub(crate) fn scale_factor(&self) -> f32 {
        f32::from_bits(self.scale_bits)
    }

    pub(crate) fn stable_key(&self) -> String {
        format!(
            "shader:v1:bridge:{}:rank={}:axis={}:dims={}:n={}:stride={}:m={}:workgroup={}:scale={}:scale_bits=0x{:08x}",
            self.kind.as_str(),
            self.rank,
            self.axis,
            dims_key(&self.dims),
            self.axis_length,
            self.stride_complex,
            self.convolution_length,
            self.workgroup_size,
            self.apply_scale,
            self.scale_bits
        )
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub(crate) struct RaderStageKey {
    pub(crate) precision: AxisPrecision,
    pub(crate) kind: RaderKernelKind,
    pub(crate) rank: usize,
    pub(crate) axis: usize,
    pub(crate) dims: Vec<usize>,
    pub(crate) axis_length: usize,
    pub(crate) stride_complex: usize,
    pub(crate) convolution_length: usize,
    pub(crate) workgroup_size: u32,
    pub(crate) apply_scale: bool,
    scale_bits: u64,
}

impl RaderStageKey {
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn new(
        kind: RaderKernelKind,
        rank: usize,
        axis: usize,
        dims: &[usize],
        axis_length: usize,
        stride_complex: usize,
        convolution_length: usize,
        workgroup_size: u32,
        apply_scale: bool,
        scale_factor: f64,
        precision: AxisPrecision,
    ) -> Self {
        debug_assert_eq!(rank, dims.len());
        debug_assert!(axis < rank);
        debug_assert_eq!(axis_length, dims[axis]);
        debug_assert!(scale_factor.is_finite());

        let scale_bits = axis_scale_bits(precision, apply_scale, scale_factor);

        Self {
            precision,
            kind,
            rank,
            axis,
            dims: dims.to_vec(),
            axis_length,
            stride_complex,
            convolution_length,
            workgroup_size,
            apply_scale,
            scale_bits,
        }
    }

    pub(crate) fn scale_factor(&self) -> f64 {
        axis_scale_factor(self.precision, self.scale_bits)
    }

    pub(crate) fn stable_key(&self) -> String {
        format!(
            "shader:v2:rader:{}:precision={}:rank={}:axis={}:dims={}:n={}:stride={}:m={}:workgroup={}:scale={}:scale_bits={}",
            self.kind.as_str(),
            self.precision.as_str(),
            self.rank,
            self.axis,
            dims_key(&self.dims),
            self.axis_length,
            self.stride_complex,
            self.convolution_length,
            self.workgroup_size,
            self.apply_scale,
            axis_scale_bits_key(self.precision, self.scale_bits)
        )
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub(crate) struct BluesteinStageKey {
    pub(crate) kind: BluesteinKernelKind,
    pub(crate) precision: AxisPrecision,
    pub(crate) rank: usize,
    pub(crate) axis: usize,
    pub(crate) dims: Vec<usize>,
    pub(crate) axis_length: usize,
    pub(crate) stride_complex: usize,
    pub(crate) convolution_length: usize,
    pub(crate) workgroup_size: u32,
    pub(crate) apply_scale: bool,
    scale_bits: u64,
}

impl BluesteinStageKey {
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn new(
        kind: BluesteinKernelKind,
        rank: usize,
        axis: usize,
        dims: &[usize],
        axis_length: usize,
        stride_complex: usize,
        convolution_length: usize,
        workgroup_size: u32,
        apply_scale: bool,
        scale_factor: f64,
        precision: AxisPrecision,
    ) -> Self {
        debug_assert_eq!(rank, dims.len());
        debug_assert!(axis < rank);
        debug_assert_eq!(axis_length, dims[axis]);
        debug_assert!(scale_factor.is_finite());
        Self {
            kind,
            precision,
            rank,
            axis,
            dims: dims.to_vec(),
            axis_length,
            stride_complex,
            convolution_length,
            workgroup_size,
            apply_scale,
            scale_bits: axis_scale_bits(precision, apply_scale, scale_factor),
        }
    }

    pub(crate) fn scale_factor(&self) -> f64 {
        axis_scale_factor(self.precision, self.scale_bits)
    }

    pub(crate) fn stable_key(&self) -> String {
        format!(
            "shader:v1:bluestein:{}:precision={}:rank={}:axis={}:dims={}:n={}:stride={}:m={}:workgroup={}:scale={}:scale_bits={}",
            self.kind.as_str(),
            self.precision.as_str(),
            self.rank,
            self.axis,
            dims_key(&self.dims),
            self.axis_length,
            self.stride_complex,
            self.convolution_length,
            self.workgroup_size,
            self.apply_scale,
            axis_scale_bits_key(self.precision, self.scale_bits),
        )
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub(crate) struct StockhamStageKey {
    pub(crate) precision: AxisPrecision,
    pub(crate) rank: usize,
    pub(crate) axis: usize,
    pub(crate) dims: Vec<usize>,
    pub(crate) axis_length: usize,
    pub(crate) stride_complex: usize,
    pub(crate) radix: usize,
    pub(crate) ns: usize,
    pub(crate) direction: FftDirection,
    pub(crate) workgroup_size: u32,
    pub(crate) apply_scale: bool,
    scale_bits: u64,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub(crate) struct FusedPow2StageKey {
    pub(crate) precision: AxisPrecision,
    pub(crate) rank: usize,
    pub(crate) axis: usize,
    pub(crate) dims: Vec<usize>,
    pub(crate) axis_length: usize,
    pub(crate) stride_complex: usize,
    pub(crate) direction: FftDirection,
    pub(crate) workgroup_size: u32,
    pub(crate) apply_scale: bool,
    scale_bits: u64,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub(crate) struct FusedSmoothStageKey {
    pub(crate) precision: AxisPrecision,
    pub(crate) rank: usize,
    pub(crate) axis: usize,
    pub(crate) dims: Vec<usize>,
    pub(crate) axis_length: usize,
    pub(crate) stride_complex: usize,
    pub(crate) factors: Vec<usize>,
    pub(crate) direction: FftDirection,
    pub(crate) workgroup_size: u32,
    pub(crate) apply_scale: bool,
    scale_bits: u64,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub(crate) struct FusedPrimeStageKey {
    pub(crate) kind: FusedPrimeKind,
    pub(crate) precision: AxisPrecision,
    pub(crate) rank: usize,
    pub(crate) axis: usize,
    pub(crate) dims: Vec<usize>,
    pub(crate) axis_length: usize,
    pub(crate) stride_complex: usize,
    pub(crate) convolution_length: usize,
    pub(crate) factors: Vec<usize>,
    pub(crate) direction: FftDirection,
    pub(crate) workgroup_size: u32,
    pub(crate) apply_scale: bool,
    scale_bits: u64,
}

impl FusedPrimeStageKey {
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn new(
        kind: FusedPrimeKind,
        rank: usize,
        axis: usize,
        dims: &[usize],
        axis_length: usize,
        stride_complex: usize,
        convolution_length: usize,
        factors: &[usize],
        direction: FftDirection,
        workgroup_size: u32,
        apply_scale: bool,
        scale_factor: f64,
        precision: AxisPrecision,
    ) -> Self {
        debug_assert_eq!(rank, dims.len());
        debug_assert!(axis < rank);
        debug_assert_eq!(axis_length, dims[axis]);
        debug_assert_eq!(factors.iter().product::<usize>(), convolution_length);
        debug_assert!(scale_factor.is_finite());

        let scale_bits = axis_scale_bits(precision, apply_scale, scale_factor);

        Self {
            kind,
            precision,
            rank,
            axis,
            dims: dims.to_vec(),
            axis_length,
            stride_complex,
            convolution_length,
            factors: factors.to_vec(),
            direction,
            workgroup_size,
            apply_scale,
            scale_bits,
        }
    }

    pub(crate) fn scale_factor(&self) -> f64 {
        axis_scale_factor(self.precision, self.scale_bits)
    }

    pub(crate) fn stable_key(&self) -> String {
        format!(
            "shader:v2:fused-prime:{}:precision={}:rank={}:axis={}:dims={}:n={}:stride={}:m={}:factors={}:direction={}:workgroup={}:scale={}:scale_bits={}:twiddle=host-f64-{}-v1",
            self.kind.as_str(),
            self.precision.as_str(),
            self.rank,
            self.axis,
            dims_key(&self.dims),
            self.axis_length,
            self.stride_complex,
            self.convolution_length,
            dims_key(&self.factors),
            direction_key(self.direction),
            self.workgroup_size,
            self.apply_scale,
            axis_scale_bits_key(self.precision, self.scale_bits),
            self.precision.as_str(),
        )
    }

    pub(crate) fn is_supported_by_limits(
        &self,
        max_workgroup_storage_bytes: u64,
        max_invocations_per_workgroup: u32,
        max_workgroup_size_x: u32,
    ) -> bool {
        let complex_bytes = self.precision.complex_size_bytes() as usize;
        let Some(scratch_bytes) = self.convolution_length.checked_mul(complex_bytes) else {
            return false;
        };
        let extra_bytes = match self.kind {
            FusedPrimeKind::Rader => complex_bytes,
            FusedPrimeKind::Bluestein => 0usize,
        };
        let Some(workgroup_storage_bytes) = scratch_bytes.checked_add(extra_bytes) else {
            return false;
        };
        workgroup_storage_bytes as u64 <= max_workgroup_storage_bytes
            && self.workgroup_size <= max_invocations_per_workgroup
            && self.workgroup_size <= max_workgroup_size_x
    }
}

impl FusedSmoothStageKey {
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn new(
        rank: usize,
        axis: usize,
        dims: &[usize],
        axis_length: usize,
        stride_complex: usize,
        factors: &[usize],
        direction: FftDirection,
        workgroup_size: u32,
        apply_scale: bool,
        scale_factor: f64,
        precision: AxisPrecision,
    ) -> Self {
        debug_assert_eq!(rank, dims.len());
        debug_assert!(axis < rank);
        debug_assert_eq!(axis_length, dims[axis]);
        debug_assert!(!axis_length.is_power_of_two());
        debug_assert_eq!(factors.iter().product::<usize>(), axis_length);
        debug_assert!(scale_factor.is_finite());

        let scale_bits = axis_scale_bits(precision, apply_scale, scale_factor);

        Self {
            precision,
            rank,
            axis,
            dims: dims.to_vec(),
            axis_length,
            stride_complex,
            factors: factors.to_vec(),
            direction,
            workgroup_size,
            apply_scale,
            scale_bits,
        }
    }

    pub(crate) fn scale_factor(&self) -> f64 {
        axis_scale_factor(self.precision, self.scale_bits)
    }

    pub(crate) fn stable_key(&self) -> String {
        format!(
            "shader:v3:fused-smooth:precision={}:rank={}:axis={}:dims={}:n={}:stride={}:factors={}:direction={}:workgroup={}:scale={}:scale_bits={}:twiddle=host-f64-{}-v1",
            self.precision.as_str(),
            self.rank,
            self.axis,
            dims_key(&self.dims),
            self.axis_length,
            self.stride_complex,
            dims_key(&self.factors),
            direction_key(self.direction),
            self.workgroup_size,
            self.apply_scale,
            axis_scale_bits_key(self.precision, self.scale_bits),
            self.precision.as_str(),
        )
    }

    fn is_supported_by_limits(
        &self,
        max_workgroup_storage_bytes: u64,
        max_invocations_per_workgroup: u32,
        max_workgroup_size_x: u32,
    ) -> bool {
        let Some(scratch_bytes) = self
            .axis_length
            .checked_mul(self.precision.complex_size_bytes() as usize)
        else {
            return false;
        };
        scratch_bytes as u64 <= max_workgroup_storage_bytes
            && self.workgroup_size <= max_invocations_per_workgroup
            && self.workgroup_size <= max_workgroup_size_x
    }
}

impl FusedPow2StageKey {
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn new(
        rank: usize,
        axis: usize,
        dims: &[usize],
        axis_length: usize,
        stride_complex: usize,
        direction: FftDirection,
        workgroup_size: u32,
        apply_scale: bool,
        scale_factor: f64,
        precision: AxisPrecision,
    ) -> Self {
        debug_assert_eq!(rank, dims.len());
        debug_assert!(axis < rank);
        debug_assert_eq!(axis_length, dims[axis]);
        debug_assert!(axis_length.is_power_of_two());
        debug_assert!(scale_factor.is_finite());

        let scale_bits = axis_scale_bits(precision, apply_scale, scale_factor);

        Self {
            precision,
            rank,
            axis,
            dims: dims.to_vec(),
            axis_length,
            stride_complex,
            direction,
            workgroup_size,
            apply_scale,
            scale_bits,
        }
    }

    pub(crate) fn scale_factor(&self) -> f64 {
        axis_scale_factor(self.precision, self.scale_bits)
    }

    pub(crate) fn stable_key(&self) -> String {
        format!(
            "shader:v3:fused-pow2:precision={}:rank={}:axis={}:dims={}:n={}:stride={}:direction={}:workgroup={}:scale={}:scale_bits={}:twiddle=host-f64-{}-v1",
            self.precision.as_str(),
            self.rank,
            self.axis,
            dims_key(&self.dims),
            self.axis_length,
            self.stride_complex,
            direction_key(self.direction),
            self.workgroup_size,
            self.apply_scale,
            axis_scale_bits_key(self.precision, self.scale_bits),
            self.precision.as_str(),
        )
    }

    fn is_supported_by_limits(
        &self,
        max_workgroup_storage_bytes: u64,
        max_invocations_per_workgroup: u32,
        max_workgroup_size_x: u32,
    ) -> bool {
        let Some(scratch_bytes) = self
            .axis_length
            .checked_mul(self.precision.complex_size_bytes() as usize)
        else {
            return false;
        };
        scratch_bytes as u64 <= max_workgroup_storage_bytes
            && self.workgroup_size <= max_invocations_per_workgroup
            && self.workgroup_size <= max_workgroup_size_x
    }
}

impl StockhamStageKey {
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn new(
        rank: usize,
        axis: usize,
        dims: &[usize],
        axis_length: usize,
        stride_complex: usize,
        radix: usize,
        ns: usize,
        direction: FftDirection,
        workgroup_size: u32,
        apply_scale: bool,
        scale_factor: f64,
        precision: AxisPrecision,
    ) -> Self {
        debug_assert_eq!(rank, dims.len());
        debug_assert!(axis < rank);
        debug_assert_eq!(axis_length, dims[axis]);
        debug_assert!(scale_factor.is_finite());

        let scale_bits = axis_scale_bits(precision, apply_scale, scale_factor);

        Self {
            precision,
            rank,
            axis,
            dims: dims.to_vec(),
            axis_length,
            stride_complex,
            radix,
            ns,
            direction,
            workgroup_size,
            apply_scale,
            scale_bits,
        }
    }

    pub(crate) fn scale_factor(&self) -> f64 {
        axis_scale_factor(self.precision, self.scale_bits)
    }

    pub(crate) fn stable_key(&self) -> String {
        format!(
            "shader:v3:stockham:precision={}:rank={}:axis={}:dims={}:n={}:stride={}:radix={}:ns={}:direction={}:workgroup={}:scale={}:scale_bits={}:twiddle=host-f64-{}-v1",
            self.precision.as_str(),
            self.rank,
            self.axis,
            dims_key(&self.dims),
            self.axis_length,
            self.stride_complex,
            self.radix,
            self.ns,
            direction_key(self.direction),
            self.workgroup_size,
            self.apply_scale,
            axis_scale_bits_key(self.precision, self.scale_bits),
            self.precision.as_str(),
        )
    }
}

fn axis_scale_bits(precision: AxisPrecision, apply_scale: bool, scale_factor: f64) -> u64 {
    let value = if apply_scale { scale_factor } else { 1.0 };
    match precision {
        AxisPrecision::F32 => u64::from((value as f32).to_bits()),
        AxisPrecision::F64 => value.to_bits(),
    }
}

fn axis_scale_factor(precision: AxisPrecision, bits: u64) -> f64 {
    match precision {
        AxisPrecision::F32 => f64::from(f32::from_bits(bits as u32)),
        AxisPrecision::F64 => f64::from_bits(bits),
    }
}

fn axis_scale_bits_key(precision: AxisPrecision, bits: u64) -> String {
    match precision {
        AxisPrecision::F32 => format!("0x{:08x}", bits as u32),
        AxisPrecision::F64 => format!("0x{bits:016x}"),
    }
}

fn direction_key(direction: FftDirection) -> &'static str {
    match direction {
        FftDirection::Forward => "forward",
        FftDirection::Inverse => "inverse",
    }
}

fn dims_key(dims: &[usize]) -> String {
    let mut out = String::new();
    for (index, dim) in dims.iter().enumerate() {
        if index > 0 {
            out.push('x');
        }
        out.push_str(&dim.to_string());
    }
    out
}

fn device_cache_id(device: &wgpu::Device) -> u64 {
    let mut hasher = DefaultHasher::new();
    device.hash(&mut hasher);
    hasher.finish()
}

fn bind_group_layout_entries(key: PipelineLayoutCacheKey) -> Vec<wgpu::BindGroupLayoutEntry> {
    match key {
        PipelineLayoutCacheKey::C2cSmoothBinaryF32
        | PipelineLayoutCacheKey::C2cStridedBinaryF32
        | PipelineLayoutCacheKey::C2cStridedBinaryF64
        | PipelineLayoutCacheKey::RealBinaryF32
        | PipelineLayoutCacheKey::RaderWriteY0InterleavedF32
        | PipelineLayoutCacheKey::RaderWriteY0InterleavedF64 => {
            vec![
                storage_entry(0, true),
                storage_entry(1, false),
                uniform_entry(2),
            ]
        }
        PipelineLayoutCacheKey::AxisPlanInterleavedF32Lut
        | PipelineLayoutCacheKey::AxisPlanInterleavedF64Lut
        | PipelineLayoutCacheKey::DirectDftInterleavedF32Lut
        | PipelineLayoutCacheKey::DirectDftInterleavedF64Lut => vec![
            storage_entry(0, true),
            storage_entry(1, false),
            uniform_entry(2),
            storage_entry(3, true),
        ],
        PipelineLayoutCacheKey::C2cSmoothTwiddleLutF32 => vec![
            storage_entry(0, true),
            storage_entry(1, false),
            uniform_entry(2),
            storage_entry(3, true),
            storage_entry(4, true),
        ],
        PipelineLayoutCacheKey::FusedPrimeInterleavedF32
        | PipelineLayoutCacheKey::FusedPrimeInterleavedF64 => vec![
            storage_entry(0, true),
            storage_entry(1, false),
            storage_entry(2, true),
            storage_entry(3, true),
            storage_entry(4, true),
            uniform_entry(5),
        ],
        PipelineLayoutCacheKey::FourStepUnaryF32 => {
            vec![storage_entry(0, false), uniform_entry(1)]
        }
        PipelineLayoutCacheKey::BridgeTwoWriteUniformF32 => vec![
            storage_entry(0, false),
            storage_entry(1, false),
            uniform_entry(2),
        ],
        PipelineLayoutCacheKey::BridgeReadWriteUniformF32 => vec![
            storage_entry(0, true),
            storage_entry(1, false),
            uniform_entry(2),
        ],
        PipelineLayoutCacheKey::BridgeWriteReadUniformF32 => vec![
            storage_entry(0, false),
            storage_entry(1, true),
            uniform_entry(2),
        ],
        PipelineLayoutCacheKey::BridgeReadWriteReadF32 => vec![
            storage_entry(0, true),
            storage_entry(1, false),
            storage_entry(2, true),
            uniform_entry(3),
        ],
        PipelineLayoutCacheKey::BluesteinBridgePostF32
        | PipelineLayoutCacheKey::BluesteinPostInterleavedF32
        | PipelineLayoutCacheKey::BluesteinPostInterleavedF64 => vec![
            storage_entry(0, true),
            storage_entry(1, true),
            storage_entry(2, false),
            uniform_entry(3),
        ],
        PipelineLayoutCacheKey::RaderSumInterleavedF32
        | PipelineLayoutCacheKey::RaderSumInterleavedF64 => vec![
            storage_entry(0, true),
            storage_entry(1, false),
            storage_entry(2, false),
            uniform_entry(3),
        ],
        PipelineLayoutCacheKey::RaderPackInterleavedF32
        | PipelineLayoutCacheKey::RaderPackInterleavedF64
        | PipelineLayoutCacheKey::BluesteinPackInterleavedF32
        | PipelineLayoutCacheKey::BluesteinPackInterleavedF64 => vec![
            storage_entry(0, true),
            storage_entry(1, false),
            storage_entry(2, true),
            uniform_entry(3),
        ],
        PipelineLayoutCacheKey::RaderMulInterleavedF32
        | PipelineLayoutCacheKey::RaderMulInterleavedF64
        | PipelineLayoutCacheKey::BluesteinMulInterleavedF32
        | PipelineLayoutCacheKey::BluesteinMulInterleavedF64 => {
            vec![
                storage_entry(0, false),
                storage_entry(1, true),
                uniform_entry(2),
            ]
        }
        PipelineLayoutCacheKey::RaderPostInterleavedF32
        | PipelineLayoutCacheKey::RaderPostInterleavedF64 => vec![
            storage_entry(0, true),
            storage_entry(1, true),
            storage_entry(2, true),
            storage_entry(3, false),
            uniform_entry(4),
        ],
        PipelineLayoutCacheKey::RaderBridgePostF32 => vec![
            storage_entry(0, true),
            storage_entry(1, true),
            storage_entry(2, true),
            storage_entry(3, true),
            storage_entry(4, false),
            uniform_entry(5),
        ],
    }
}

fn storage_entry(binding: u32, read_only: bool) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only },
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

fn uniform_entry(binding: u32) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Uniform,
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stockham_shader_key_is_stable_and_includes_stage_constants() {
        let key = StockhamStageKey::new(
            2,
            1,
            &[4, 3],
            3,
            4,
            3,
            3,
            FftDirection::Forward,
            64,
            true,
            1.0 / 12.0,
            AxisPrecision::F32,
        );

        assert_eq!(key.scale_factor(), f64::from(1.0f32 / 12.0));
        assert_eq!(
            key.stable_key(),
            "shader:v3:stockham:precision=f32:rank=2:axis=1:dims=4x3:n=3:stride=4:radix=3:ns=3:direction=forward:workgroup=64:scale=true:scale_bits=0x3daaaaab:twiddle=host-f64-f32-v1"
        );
    }

    #[test]
    fn stockham_shader_key_ignores_scale_value_when_scale_is_not_applied() {
        let a = StockhamStageKey::new(
            1,
            0,
            &[8],
            8,
            1,
            8,
            8,
            FftDirection::Forward,
            64,
            false,
            1.0,
            AxisPrecision::F32,
        );
        let b = StockhamStageKey::new(
            1,
            0,
            &[8],
            8,
            1,
            8,
            8,
            FftDirection::Forward,
            64,
            false,
            0.125,
            AxisPrecision::F32,
        );

        assert_eq!(a, b);
        assert_eq!(a.scale_factor(), 1.0);
    }

    #[test]
    fn fused_pow2_shader_key_is_stable_and_includes_plan_constants() {
        let key = FusedPow2StageKey::new(
            2,
            1,
            &[4, 256],
            256,
            4,
            FftDirection::Inverse,
            256,
            true,
            1.0 / 1024.0,
            AxisPrecision::F32,
        );

        assert_eq!(key.scale_factor(), 1.0 / 1024.0);
        assert_eq!(
            key.stable_key(),
            "shader:v3:fused-pow2:precision=f32:rank=2:axis=1:dims=4x256:n=256:stride=4:direction=inverse:workgroup=256:scale=true:scale_bits=0x3a800000:twiddle=host-f64-f32-v1"
        );
        let pipeline = ComputePipelineCacheKey::fused_pow2_stage(key);
        assert!(pipeline.stable_key().starts_with(
            "pipeline:v1:layout=axis-plan/interleaved-f32-lut:entry=main:shader:v3:fused-pow2:"
        ));
    }

    #[test]
    fn fused_pow2_shader_key_ignores_unused_scale_value() {
        let a = FusedPow2StageKey::new(
            1,
            0,
            &[4096],
            4096,
            1,
            FftDirection::Forward,
            256,
            false,
            1.0,
            AxisPrecision::F32,
        );
        let b = FusedPow2StageKey::new(
            1,
            0,
            &[4096],
            4096,
            1,
            FftDirection::Forward,
            256,
            false,
            1.0 / 4096.0,
            AxisPrecision::F32,
        );
        assert_eq!(a, b);
        assert_eq!(a.scale_factor(), 1.0);
    }

    #[test]
    fn fused_pow2_shader_key_checks_target_device_compute_limits() {
        let key = FusedPow2StageKey::new(
            1,
            0,
            &[4096],
            4096,
            1,
            FftDirection::Forward,
            256,
            false,
            1.0,
            AxisPrecision::F32,
        );
        assert!(key.is_supported_by_limits(32 * 1024, 256, 256));
        assert!(!key.is_supported_by_limits(16 * 1024, 256, 256));
        assert!(!key.is_supported_by_limits(32 * 1024, 255, 256));
        assert!(!key.is_supported_by_limits(32 * 1024, 256, 255));
    }

    #[test]
    fn native_f64_axis_direct_and_strided_keys_are_precision_distinct() {
        let make_pow2 = |precision| {
            FusedPow2StageKey::new(
                1,
                0,
                &[2048],
                2048,
                1,
                FftDirection::Inverse,
                256,
                true,
                1.0 / 2048.0,
                precision,
            )
        };
        let f32_key = make_pow2(AxisPrecision::F32);
        let f64_key = make_pow2(AxisPrecision::F64);
        assert_ne!(f32_key, f64_key);
        assert!(f64_key.stable_key().contains("precision=f64"));
        assert!(f64_key.stable_key().contains("scale_bits=0x"));
        assert!(f64_key.is_supported_by_limits(32 * 1024, 256, 256));
        assert!(!FusedPow2StageKey::new(
            1,
            0,
            &[4096],
            4096,
            1,
            FftDirection::Forward,
            256,
            false,
            1.0,
            AxisPrecision::F64,
        )
        .is_supported_by_limits(48 * 1024, 256, 256));

        let f32_pipeline = ComputePipelineCacheKey::fused_pow2_stage(f32_key);
        let f64_pipeline = ComputePipelineCacheKey::fused_pow2_stage(f64_key);
        assert_eq!(
            f32_pipeline.layout,
            PipelineLayoutCacheKey::AxisPlanInterleavedF32Lut
        );
        assert_eq!(
            f64_pipeline.layout,
            PipelineLayoutCacheKey::AxisPlanInterleavedF64Lut
        );
        assert_ne!(f32_pipeline.stable_key(), f64_pipeline.stable_key());

        assert_eq!(
            ComputePipelineCacheKey::direct_dft_c2c(AxisPrecision::F64).layout,
            PipelineLayoutCacheKey::DirectDftInterleavedF64Lut
        );
        assert_eq!(
            ComputePipelineCacheKey::c2c_strided_stage(C2cStridedStageKey::new(
                C2cStridedKernelKind::Pack,
                64,
                AxisPrecision::F64,
            ))
            .layout,
            PipelineLayoutCacheKey::C2cStridedBinaryF64
        );
    }

    #[test]
    fn fused_smooth_shader_key_is_distinct_and_includes_factor_schedule() {
        let key = FusedSmoothStageKey::new(
            2,
            1,
            &[4, 1001],
            1001,
            4,
            &[13, 11, 7],
            FftDirection::Inverse,
            256,
            true,
            1.0 / 4096.0,
            AxisPrecision::F32,
        );

        assert_eq!(key.scale_factor(), 1.0 / 4096.0);
        assert_eq!(
            key.stable_key(),
            "shader:v3:fused-smooth:precision=f32:rank=2:axis=1:dims=4x1001:n=1001:stride=4:factors=13x11x7:direction=inverse:workgroup=256:scale=true:scale_bits=0x39800000:twiddle=host-f64-f32-v1"
        );
        let pipeline = ComputePipelineCacheKey::fused_smooth_stage(key);
        assert!(pipeline.stable_key().contains("shader:v3:fused-smooth:"));
        assert!(!pipeline.stable_key().contains("shader:v3:fused-pow2:"));
    }

    #[test]
    fn fused_smooth_shader_key_canonicalizes_scale_and_checks_limits() {
        let a = FusedSmoothStageKey::new(
            1,
            0,
            &[3000],
            3000,
            1,
            &[8, 5, 5, 5, 3],
            FftDirection::Forward,
            256,
            false,
            1.0,
            AxisPrecision::F32,
        );
        let b = FusedSmoothStageKey::new(
            1,
            0,
            &[3000],
            3000,
            1,
            &[8, 5, 5, 5, 3],
            FftDirection::Forward,
            256,
            false,
            1.0 / 3000.0,
            AxisPrecision::F32,
        );
        assert_eq!(a, b);
        assert_eq!(a.scale_factor(), 1.0);
        assert!(a.is_supported_by_limits(24_000, 256, 256));
        assert!(!a.is_supported_by_limits(23_999, 256, 256));
        assert!(!a.is_supported_by_limits(24_000, 255, 256));
        assert!(!a.is_supported_by_limits(24_000, 256, 255));
    }

    #[test]
    fn fused_prime_shader_key_is_stable_and_uses_typed_layout() {
        let key = FusedPrimeStageKey::new(
            FusedPrimeKind::Rader,
            2,
            1,
            &[4, 2999],
            2999,
            4,
            6000,
            &[8, 5, 5, 5, 3, 2],
            FftDirection::Inverse,
            256,
            true,
            0.5,
            AxisPrecision::F32,
        );

        assert_eq!(key.scale_factor(), 0.5);
        assert_eq!(
            key.stable_key(),
            "shader:v2:fused-prime:rader:precision=f32:rank=2:axis=1:dims=4x2999:n=2999:stride=4:m=6000:factors=8x5x5x5x3x2:direction=inverse:workgroup=256:scale=true:scale_bits=0x3f000000:twiddle=host-f64-f32-v1"
        );
        let pipeline = ComputePipelineCacheKey::fused_prime_stage(key);
        assert_eq!(
            pipeline.layout,
            PipelineLayoutCacheKey::FusedPrimeInterleavedF32
        );
        assert!(pipeline.stable_key().starts_with(
            "pipeline:v1:layout=fused-prime/interleaved-f32:entry=main:shader:v2:fused-prime:rader:"
        ));
    }

    #[test]
    fn fused_prime_shader_key_canonicalizes_scale_and_checks_kind_storage() {
        let key = |kind, apply_scale, scale_factor| {
            FusedPrimeStageKey::new(
                kind,
                1,
                0,
                &[2999],
                2999,
                1,
                6000,
                &[8, 5, 5, 5, 3, 2],
                FftDirection::Forward,
                256,
                apply_scale,
                scale_factor,
                AxisPrecision::F32,
            )
        };

        let rader = key(FusedPrimeKind::Rader, false, 1.0);
        let rader_unused_scale = key(FusedPrimeKind::Rader, false, 1.0 / 2999.0);
        assert_eq!(rader, rader_unused_scale);
        assert_eq!(rader.scale_factor(), 1.0);
        assert!(rader.is_supported_by_limits(48_008, 256, 256));
        assert!(!rader.is_supported_by_limits(48_007, 256, 256));
        assert!(!rader.is_supported_by_limits(48_008, 255, 256));
        assert!(!rader.is_supported_by_limits(48_008, 256, 255));

        let bluestein = key(FusedPrimeKind::Bluestein, false, 1.0);
        assert!(bluestein.is_supported_by_limits(48_000, 256, 256));
        assert!(!bluestein.is_supported_by_limits(47_999, 256, 256));
        assert_ne!(rader, bluestein);
        assert!(bluestein.stable_key().contains("fused-prime:bluestein"));
    }

    #[test]
    fn compute_pipeline_key_wraps_layout_entry_point_and_shader_key() {
        let shader = StockhamStageKey::new(
            1,
            0,
            &[8],
            8,
            1,
            8,
            8,
            FftDirection::Inverse,
            64,
            false,
            1.0,
            AxisPrecision::F32,
        );
        let pipeline = ComputePipelineCacheKey::stockham_stage(shader);

        assert!(pipeline
            .stable_key()
            .starts_with("pipeline:v1:layout=axis-plan/interleaved-f32-lut:entry=main:"));
        assert!(pipeline.stable_key().contains("direction=inverse"));

        assert_eq!(
            ComputePipelineCacheKey::direct_dft_c2c_f32().stable_key(),
            "pipeline:v1:layout=direct-dft/interleaved-f32-lut:entry=main:shader:v3:direct-dft/c2c-f32:twiddle=host-f64-f32-v1"
        );
    }

    #[test]
    fn snapshot_exposes_stable_schema_codes_and_pipeline_keys() {
        let shader = ShaderCacheKey::StockhamStage(StockhamStageKey::new(
            1,
            0,
            &[8],
            8,
            1,
            8,
            8,
            FftDirection::Forward,
            64,
            false,
            1.0,
            AxisPrecision::F32,
        ));
        let pipeline = ComputePipelineCacheKey::stockham_stage(match &shader {
            ShaderCacheKey::StockhamStage(key) => key.clone(),
            ShaderCacheKey::FusedPow2Stage(_) => unreachable!(),
            ShaderCacheKey::FusedSmoothStage(_) => unreachable!(),
            ShaderCacheKey::FusedPrimeStage(_) => unreachable!(),
            ShaderCacheKey::FourStepStage(_) => unreachable!(),
            ShaderCacheKey::BridgeStage(_) => unreachable!(),
            ShaderCacheKey::RaderStage(_) => unreachable!(),
            ShaderCacheKey::BluesteinStage(_) => unreachable!(),
            ShaderCacheKey::RealStage(_) => unreachable!(),
            ShaderCacheKey::C2cSmoothStage(_) => unreachable!(),
            ShaderCacheKey::C2cStridedStage(_) => unreachable!(),
            ShaderCacheKey::DirectDftC2cLut(_) => unreachable!(),
        });
        let snapshot = PipelineCacheSnapshot::from_entries(
            vec![SnapshotShaderEntry {
                key: shader,
                code: String::from("wgsl-a"),
            }],
            vec![pipeline],
        );

        assert_eq!(snapshot.schema(), PIPELINE_CACHE_SNAPSHOT_SCHEMA);
        assert_eq!(snapshot.version(), PIPELINE_CACHE_SNAPSHOT_VERSION);
        assert_eq!(snapshot.shader_codes(), &[String::from("wgsl-a")]);
        assert_eq!(snapshot.pipeline_keys().len(), 1);
        assert!(!snapshot.is_empty());
    }

    #[test]
    fn rader_pipeline_key_uses_typed_helper_layout_and_shader_key() {
        let shader = RaderStageKey::new(
            RaderKernelKind::Pack,
            2,
            1,
            &[4, 17],
            17,
            4,
            32,
            64,
            false,
            1.0,
            AxisPrecision::F32,
        );
        let pipeline = ComputePipelineCacheKey::rader_stage(shader.clone());

        assert_eq!(
            pipeline.layout,
            PipelineLayoutCacheKey::RaderPackInterleavedF32
        );
        assert!(pipeline.stable_key().contains("shader:v2:rader:pack"));
        assert!(pipeline.stable_key().contains("dims=4x17"));
        assert_eq!(shader.scale_factor(), 1.0);
    }

    #[test]
    fn real_pipeline_key_uses_typed_helper_layout_and_shader_key() {
        let shader = RealStageKey::new(RealKernelKind::PackR2c, &[17, 4], 64);
        let pipeline = ComputePipelineCacheKey::real_stage(shader);

        assert_eq!(pipeline.layout, PipelineLayoutCacheKey::RealBinaryF32);
        assert_eq!(
            pipeline.stable_key(),
            "pipeline:v1:layout=real/binary-f32:entry=main:shader:v1:real:pack-r2c:rank=2:dims=17x4:workgroup=64"
        );
    }

    #[test]
    fn real_windowed_pipeline_keys_are_stable_and_snapshot_visible() {
        for (kind, expected) in [
            (
                RealKernelKind::RealToComplexWindowed,
                "real-to-complex-windowed",
            ),
            (RealKernelKind::PackR2cWindowed, "pack-r2c-windowed"),
            (RealKernelKind::UnpackC2rWindowed, "unpack-c2r-windowed"),
            (
                RealKernelKind::ComplexToRealWindowed,
                "complex-to-real-windowed",
            ),
        ] {
            let shader = RealStageKey::new(kind, &[64], 64);
            let pipeline = ComputePipelineCacheKey::real_stage(shader.clone());
            assert_eq!(pipeline.layout, PipelineLayoutCacheKey::RealBinaryF32);
            assert!(pipeline.stable_key().contains(expected));
            let snapshot = PipelineCacheSnapshot::from_entries(
                vec![SnapshotShaderEntry {
                    key: ShaderCacheKey::RealStage(shader),
                    code: String::from("wgsl"),
                }],
                vec![pipeline],
            );
            assert!(snapshot
                .pipeline_keys()
                .iter()
                .any(|key| key.contains(expected)));
        }
    }

    #[test]
    fn c2c_strided_pipeline_key_uses_typed_helper_layout_and_shader_key() {
        let shader = C2cStridedStageKey::new(C2cStridedKernelKind::Pack, 64, AxisPrecision::F32);
        let pipeline = ComputePipelineCacheKey::c2c_strided_stage(shader);

        assert_eq!(pipeline.layout, PipelineLayoutCacheKey::C2cStridedBinaryF32);
        assert_eq!(
            pipeline.stable_key(),
            "pipeline:v1:layout=c2c-strided/binary-f32:entry=main:shader:v2:c2c-strided:pack-c2c-strided:precision=f32:workgroup=64"
        );
    }

    #[test]
    fn c2c_smooth_pipeline_key_uses_typed_helper_layout_and_shader_key() {
        let shader = C2cSmoothStageKey::new(C2cSmoothKernelKind::TwiddleTranspose, 64);
        let pipeline = ComputePipelineCacheKey::c2c_smooth_stage(shader);

        assert_eq!(
            pipeline.layout,
            PipelineLayoutCacheKey::C2cSmoothTwiddleLutF32
        );
        assert_eq!(
            pipeline.stable_key(),
            "pipeline:v1:layout=c2c-smooth/twiddle-lut-f32:entry=main:shader:v2:c2c-smooth:twiddle-transpose:workgroup=64:twiddle=host-f64-two-level-f32-v1"
        );

        for (kind, expected) in [
            (C2cSmoothKernelKind::GatherAxisLine, "gather-axis-line"),
            (C2cSmoothKernelKind::ScatterAxisLine, "scatter-axis-line"),
            (
                C2cSmoothKernelKind::GatherSmoothPhase1,
                "gather-smooth-phase1",
            ),
            (
                C2cSmoothKernelKind::ScatterSmoothPhase2,
                "scatter-smooth-phase2",
            ),
        ] {
            let shader = C2cSmoothStageKey::new(kind, 64);
            let pipeline = ComputePipelineCacheKey::c2c_smooth_stage(shader);
            assert_eq!(pipeline.layout, PipelineLayoutCacheKey::C2cSmoothBinaryF32);
            assert!(pipeline.stable_key().contains(expected));
        }
    }

    #[test]
    fn four_step_pipeline_keys_are_stable_and_use_typed_layouts() {
        let transpose = ComputePipelineCacheKey::four_step_stage(FourStepStageKey::new(
            FourStepKernelKind::StripeTranspose,
            256,
        ));
        assert_eq!(transpose.layout, PipelineLayoutCacheKey::C2cSmoothBinaryF32);
        assert_eq!(
            transpose.stable_key(),
            "pipeline:v1:layout=c2c-smooth/binary-f32:entry=main:shader:v1:four-step:stripe-transpose:workgroup=256"
        );

        let scale = ComputePipelineCacheKey::four_step_stage(FourStepStageKey::new(
            FourStepKernelKind::Scale,
            64,
        ));
        assert_eq!(scale.layout, PipelineLayoutCacheKey::FourStepUnaryF32);
        assert_eq!(
            scale.stable_key(),
            "pipeline:v1:layout=four-step/unary-f32:entry=main:shader:v1:four-step:scale:workgroup=64"
        );
    }

    #[test]
    fn four_step_unary_layout_is_read_write_storage_plus_uniform() {
        let entries = bind_group_layout_entries(PipelineLayoutCacheKey::FourStepUnaryF32);

        assert_eq!(entries.len(), 2);
        assert_eq!(entries[0].binding, 0);
        assert!(matches!(
            &entries[0].ty,
            wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Storage { read_only: false },
                ..
            }
        ));
        assert_eq!(entries[1].binding, 1);
        assert!(matches!(
            &entries[1].ty,
            wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Uniform,
                ..
            }
        ));
    }

    #[test]
    fn bridge_pipeline_keys_are_stable_and_snapshot_visible() {
        for (kind, expected) in [
            (BridgeKernelKind::RaderSumInit, "rader-sum-init"),
            (BridgeKernelKind::RaderSumAccumulate, "rader-sum-accumulate"),
            (BridgeKernelKind::RaderPack, "rader-pack-windowed"),
            (BridgeKernelKind::RaderMul, "rader-mul-windowed"),
            (BridgeKernelKind::RaderWriteY0, "rader-write-y0-windowed"),
            (BridgeKernelKind::RaderPost, "rader-post-windowed"),
            (BridgeKernelKind::BluesteinPack, "bluestein-pack-windowed"),
            (BridgeKernelKind::BluesteinMul, "bluestein-mul-windowed"),
            (BridgeKernelKind::BluesteinPost, "bluestein-post-windowed"),
        ] {
            let shader = BridgeStageKey::new(kind, 1, 0, &[17], 17, 1, 32, 64, false, 1.0);
            let pipeline = ComputePipelineCacheKey::bridge_stage(shader.clone());
            assert!(pipeline.stable_key().contains(expected));
            let snapshot = PipelineCacheSnapshot::from_entries(
                vec![SnapshotShaderEntry {
                    key: ShaderCacheKey::BridgeStage(shader),
                    code: String::from("wgsl"),
                }],
                vec![pipeline],
            );
            assert!(snapshot
                .pipeline_keys()
                .iter()
                .any(|key| key.contains(expected)));
        }
    }

    #[test]
    fn empty_snapshot_has_no_codes_or_pipeline_keys() {
        let snapshot = PipelineCacheSnapshot::empty();
        assert_eq!(snapshot.schema(), PIPELINE_CACHE_SNAPSHOT_SCHEMA);
        assert_eq!(snapshot.version(), PIPELINE_CACHE_SNAPSHOT_VERSION);
        assert!(snapshot.shader_codes().is_empty());
        assert!(snapshot.pipeline_keys().is_empty());
        assert!(snapshot.is_empty());
    }
}
