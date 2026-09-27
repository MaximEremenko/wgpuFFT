use std::cell::RefCell;
use std::collections::hash_map::DefaultHasher;
use std::collections::HashMap;
#[cfg(feature = "serde")]
use std::collections::HashSet;
#[cfg(feature = "serde")]
use std::fmt;
use std::hash::{Hash, Hasher};

use crate::config::FftDirection;
use crate::math::DoubleFloat;
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

#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Debug, Clone, PartialEq, Eq)]
struct SnapshotShaderEntry {
    key: ShaderCacheKey,
    code: String,
}

#[cfg(feature = "serde")]
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
struct PipelineCacheSnapshotJson {
    schema: String,
    version: u32,
    shader_codes: Vec<String>,
    pipeline_keys: Vec<String>,
    shader_entries: Vec<SnapshotShaderEntry>,
    pipeline_entries: Vec<ComputePipelineCacheKey>,
}

/// Failure while encoding or decoding a persistent pipeline-cache snapshot.
///
/// Snapshots are versioned, typed descriptions of WGSL and pipeline keys. They
/// are not backend driver binaries. Decoding rejects stale or modified source
/// rather than compiling a shader that no longer matches its typed key.
#[cfg(feature = "serde")]
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum PipelineCacheSnapshotError {
    Json {
        message: String,
    },
    SchemaMismatch {
        expected: &'static str,
        actual: String,
    },
    VersionMismatch {
        expected: u32,
        actual: u32,
    },
    Integrity {
        reason: String,
    },
}

#[cfg(feature = "serde")]
impl fmt::Display for PipelineCacheSnapshotError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Json { message } => write!(f, "invalid pipeline-cache snapshot JSON: {message}"),
            Self::SchemaMismatch { expected, actual } => write!(
                f,
                "pipeline-cache snapshot schema mismatch: expected {expected:?}, got {actual:?}"
            ),
            Self::VersionMismatch { expected, actual } => write!(
                f,
                "pipeline-cache snapshot version mismatch: expected {expected}, got {actual}"
            ),
            Self::Integrity { reason } => {
                write!(
                    f,
                    "pipeline-cache snapshot integrity check failed: {reason}"
                )
            }
        }
    }
}

#[cfg(feature = "serde")]
impl std::error::Error for PipelineCacheSnapshotError {}

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

    /// Serializes this snapshot as versioned JSON.
    ///
    /// The snapshot is validated before serialization, including regeneration
    /// of every WGSL source from its typed key.
    #[cfg(feature = "serde")]
    pub fn to_json(&self) -> Result<String, PipelineCacheSnapshotError> {
        let canonical = Self::validated_json_parts(PipelineCacheSnapshotJson {
            schema: self.schema.to_owned(),
            version: self.version,
            shader_codes: self.shader_codes.clone(),
            pipeline_keys: self.pipeline_keys.clone(),
            shader_entries: self.shader_entries.clone(),
            pipeline_entries: self.pipeline_entries.clone(),
        })?;
        serde_json::to_string(&canonical).map_err(|error| PipelineCacheSnapshotError::Json {
            message: error.to_string(),
        })
    }

    /// Decodes and validates a versioned JSON pipeline-cache snapshot.
    ///
    /// Stable key projections, pipeline layouts, entry points, and WGSL source
    /// must all agree with the deserialized typed keys.
    #[cfg(feature = "serde")]
    pub fn from_json(json: &str) -> Result<Self, PipelineCacheSnapshotError> {
        let parts =
            serde_json::from_str(json).map_err(|error| PipelineCacheSnapshotError::Json {
                message: error.to_string(),
            })?;
        let canonical = Self::validated_json_parts(parts)?;
        Ok(Self::from_entries(
            canonical.shader_entries,
            canonical.pipeline_entries,
        ))
    }

    #[cfg(feature = "serde")]
    fn validated_json_parts(
        parts: PipelineCacheSnapshotJson,
    ) -> Result<PipelineCacheSnapshotJson, PipelineCacheSnapshotError> {
        if parts.schema != PIPELINE_CACHE_SNAPSHOT_SCHEMA {
            return Err(PipelineCacheSnapshotError::SchemaMismatch {
                expected: PIPELINE_CACHE_SNAPSHOT_SCHEMA,
                actual: parts.schema,
            });
        }
        if parts.version != PIPELINE_CACHE_SNAPSHOT_VERSION {
            return Err(PipelineCacheSnapshotError::VersionMismatch {
                expected: PIPELINE_CACHE_SNAPSHOT_VERSION,
                actual: parts.version,
            });
        }

        let mut shader_keys = HashSet::with_capacity(parts.shader_entries.len());
        for entry in &parts.shader_entries {
            entry.key.validate_snapshot_shape()?;
            if !shader_keys.insert(entry.key.clone()) {
                return Err(PipelineCacheSnapshotError::Integrity {
                    reason: format!("duplicate shader key {}", entry.key.stable_key()),
                });
            }
            let expected_source = entry.key.fallback_source();
            if entry.code != expected_source {
                return Err(PipelineCacheSnapshotError::Integrity {
                    reason: format!(
                        "WGSL source does not match typed shader key {}",
                        entry.key.stable_key()
                    ),
                });
            }
        }

        let mut pipeline_keys = HashSet::with_capacity(parts.pipeline_entries.len());
        for pipeline in &parts.pipeline_entries {
            let expected = ComputePipelineCacheKey::from_shader_key(pipeline.shader.clone());
            if pipeline != &expected {
                return Err(PipelineCacheSnapshotError::Integrity {
                    reason: format!(
                        "pipeline layout or entry point does not match typed shader key {}",
                        pipeline.shader.stable_key()
                    ),
                });
            }
            if !pipeline_keys.insert(pipeline.clone()) {
                return Err(PipelineCacheSnapshotError::Integrity {
                    reason: format!("duplicate pipeline key {}", pipeline.stable_key()),
                });
            }
            if !shader_keys.contains(&pipeline.shader) {
                return Err(PipelineCacheSnapshotError::Integrity {
                    reason: format!(
                        "pipeline {} has no matching WGSL source entry",
                        pipeline.stable_key()
                    ),
                });
            }
        }

        let canonical =
            Self::from_entries(parts.shader_entries.clone(), parts.pipeline_entries.clone());
        if parts.shader_codes != canonical.shader_codes {
            return Err(PipelineCacheSnapshotError::Integrity {
                reason: String::from(
                    "shader_codes do not match the canonical typed shader entries",
                ),
            });
        }
        if parts.pipeline_keys != canonical.pipeline_keys {
            return Err(PipelineCacheSnapshotError::Integrity {
                reason: String::from(
                    "pipeline_keys do not match the canonical typed pipeline entries",
                ),
            });
        }

        Ok(PipelineCacheSnapshotJson {
            schema: PIPELINE_CACHE_SNAPSHOT_SCHEMA.to_owned(),
            version: PIPELINE_CACHE_SNAPSHOT_VERSION,
            shader_codes: canonical.shader_codes,
            pipeline_keys: canonical.pipeline_keys,
            shader_entries: canonical.shader_entries,
            pipeline_entries: canonical.pipeline_entries,
        })
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

#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) enum PipelineLayoutCacheKey {
    AxisPlanInterleavedF32Lut,
    AxisPlanInterleavedF64Lut,
    AxisPlanInterleavedDf64Lut,
    /// An axis-plan kernel reading and writing one buffer in place.
    AxisPlanInPlaceF32Lut,
    AxisPlanInPlaceF64Lut,
    AxisPlanInPlaceDf64Lut,
    BridgeReadWriteReadF32,
    BridgeReadWriteUniformF32,
    BridgeTwoWriteUniformF32,
    BridgeWriteReadUniformF32,
    BluesteinBridgePostF32,
    BluesteinPackInterleavedF32,
    BluesteinPackInterleavedF64,
    BluesteinPackInterleavedDf64,
    BluesteinMulInterleavedF32,
    BluesteinMulInterleavedF64,
    BluesteinMulInterleavedDf64,
    BluesteinPostInterleavedF32,
    BluesteinPostInterleavedF64,
    BluesteinPostInterleavedDf64,
    C2cSmoothBinaryF32,
    C2cSmoothTwiddleLutF32,
    C2cStridedBinaryF32,
    C2cStridedBinaryF64,
    C2cStridedBinaryDf64,
    DirectDftInterleavedF32Lut,
    DirectDftInterleavedF64Lut,
    DirectDftInterleavedDf64Lut,
    FusedPrimeInterleavedF32,
    FusedPrimeInterleavedF64,
    FusedPrimeInterleavedDf64,
    FourStepUnaryF32,
    RealBinaryF32,
    RaderBridgePostF32,
    RaderSumInterleavedF32,
    RaderSumInterleavedF64,
    RaderSumInterleavedDf64,
    RaderPackInterleavedF32,
    RaderPackInterleavedF64,
    RaderPackInterleavedDf64,
    RaderMulInterleavedF32,
    RaderMulInterleavedF64,
    RaderMulInterleavedDf64,
    RaderWriteY0InterleavedF32,
    RaderWriteY0InterleavedF64,
    RaderWriteY0InterleavedDf64,
    RaderPostInterleavedF32,
    RaderPostInterleavedF64,
    RaderPostInterleavedDf64,
}

impl PipelineLayoutCacheKey {
    fn stable_key(self) -> &'static str {
        match self {
            Self::AxisPlanInterleavedF32Lut => "axis-plan/interleaved-f32-lut",
            Self::AxisPlanInterleavedF64Lut => "axis-plan/interleaved-f64-lut",
            Self::AxisPlanInterleavedDf64Lut => "axis-plan/interleaved-df64-lut",
            Self::AxisPlanInPlaceF32Lut => "axis-plan/in-place-f32-lut",
            Self::AxisPlanInPlaceF64Lut => "axis-plan/in-place-f64-lut",
            Self::AxisPlanInPlaceDf64Lut => "axis-plan/in-place-df64-lut",
            Self::BridgeReadWriteReadF32 => "bridge/read-write-read-f32",
            Self::BridgeReadWriteUniformF32 => "bridge/read-write-uniform-f32",
            Self::BridgeTwoWriteUniformF32 => "bridge/two-write-uniform-f32",
            Self::BridgeWriteReadUniformF32 => "bridge/write-read-uniform-f32",
            Self::BluesteinBridgePostF32 => "bridge/bluestein-post-f32",
            Self::BluesteinPackInterleavedF32 => "bluestein/pack/interleaved-f32",
            Self::BluesteinPackInterleavedF64 => "bluestein/pack/interleaved-f64",
            Self::BluesteinPackInterleavedDf64 => "bluestein/pack/interleaved-df64",
            Self::BluesteinMulInterleavedF32 => "bluestein/mul/interleaved-f32",
            Self::BluesteinMulInterleavedF64 => "bluestein/mul/interleaved-f64",
            Self::BluesteinMulInterleavedDf64 => "bluestein/mul/interleaved-df64",
            Self::BluesteinPostInterleavedF32 => "bluestein/post/interleaved-f32",
            Self::BluesteinPostInterleavedF64 => "bluestein/post/interleaved-f64",
            Self::BluesteinPostInterleavedDf64 => "bluestein/post/interleaved-df64",
            Self::C2cSmoothBinaryF32 => "c2c-smooth/binary-f32",
            Self::C2cSmoothTwiddleLutF32 => "c2c-smooth/twiddle-lut-f32",
            Self::C2cStridedBinaryF32 => "c2c-strided/binary-f32",
            Self::C2cStridedBinaryF64 => "c2c-strided/binary-f64",
            Self::C2cStridedBinaryDf64 => "c2c-strided/binary-df64",
            Self::DirectDftInterleavedF32Lut => "direct-dft/interleaved-f32-lut",
            Self::DirectDftInterleavedF64Lut => "direct-dft/interleaved-f64-lut",
            Self::DirectDftInterleavedDf64Lut => "direct-dft/interleaved-df64-lut",
            Self::FusedPrimeInterleavedF32 => "fused-prime/interleaved-f32",
            Self::FusedPrimeInterleavedF64 => "fused-prime/interleaved-f64",
            Self::FusedPrimeInterleavedDf64 => "fused-prime/interleaved-df64",
            Self::FourStepUnaryF32 => "four-step/unary-f32",
            Self::RealBinaryF32 => "real/binary-f32",
            Self::RaderBridgePostF32 => "bridge/rader-post-f32",
            Self::RaderSumInterleavedF32 => "rader/sum/interleaved-f32",
            Self::RaderSumInterleavedF64 => "rader/sum/interleaved-f64",
            Self::RaderSumInterleavedDf64 => "rader/sum/interleaved-df64",
            Self::RaderPackInterleavedF32 => "rader/pack/interleaved-f32",
            Self::RaderPackInterleavedF64 => "rader/pack/interleaved-f64",
            Self::RaderPackInterleavedDf64 => "rader/pack/interleaved-df64",
            Self::RaderMulInterleavedF32 => "rader/mul/interleaved-f32",
            Self::RaderMulInterleavedF64 => "rader/mul/interleaved-f64",
            Self::RaderMulInterleavedDf64 => "rader/mul/interleaved-df64",
            Self::RaderWriteY0InterleavedF32 => "rader/write-y0/interleaved-f32",
            Self::RaderWriteY0InterleavedF64 => "rader/write-y0/interleaved-f64",
            Self::RaderWriteY0InterleavedDf64 => "rader/write-y0/interleaved-df64",
            Self::RaderPostInterleavedF32 => "rader/post/interleaved-f32",
            Self::RaderPostInterleavedF64 => "rader/post/interleaved-f64",
            Self::RaderPostInterleavedDf64 => "rader/post/interleaved-df64",
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
            label: Some(&shader_module_label(label)),
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
            compilation_options: wgpu::PipelineCompilationOptions {
                // Every generated kernel writes the workgroup memory it reads
                // before its first barrier (Bluestein and Rader store explicit
                // zeros in their padding), so wgpu's zero fill is redundant.
                // naga lowers that fill to a serial store loop on one
                // invocation, which made FXC take minutes per fused kernel and
                // DXC seconds. Browsers always zero-fill regardless.
                zero_initialize_workgroup_memory: false,
                ..Default::default()
            },
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

#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
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
    SmallVolume(SmallVolumeKey),
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
            Self::SmallVolume(key) => key.stable_key(),
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
                FusedPrimeKind::Bluestein => match &key.registers {
                    Some(registers) => {
                        crate::runtime::register_fft::generate_register_bluestein_wgsl(
                            key, registers,
                        )
                    }
                    None => {
                        crate::runtime::bluestein_axis::generate_fused_bluestein_wgsl_for_key(key)
                    }
                },
                FusedPrimeKind::Direct => {
                    crate::runtime::direct_prime::generate_direct_prime_wgsl_for_key(key)
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
            Self::SmallVolume(key) => {
                crate::runtime::small_volume::generate_small_volume_wgsl_for_key(key)
            }
        }
    }

    fn is_supported_on_device(&self, device: &wgpu::Device) -> bool {
        if self.precision() == Some(AxisPrecision::F64)
            && !device.features().contains(wgpu::Features::SHADER_F64)
        {
            return false;
        }
        let limits = device.limits();
        if !self.is_supported_by_1d_workgroup_limits(&limits) {
            return false;
        }
        match self {
            Self::FusedPow2Stage(key) => key.is_supported_by_limits(
                u64::from(limits.max_compute_workgroup_storage_size),
                limits.max_compute_invocations_per_workgroup,
                limits.max_compute_workgroup_size_x,
            ),
            Self::FusedSmoothStage(key) => key.is_supported_by_limits(
                u64::from(limits.max_compute_workgroup_storage_size),
                limits.max_compute_invocations_per_workgroup,
                limits.max_compute_workgroup_size_x,
            ),
            Self::FusedPrimeStage(key) => key.is_supported_by_limits(
                u64::from(limits.max_compute_workgroup_storage_size),
                limits.max_compute_invocations_per_workgroup,
                limits.max_compute_workgroup_size_x,
            ),
            Self::SmallVolume(key) => key.supported_by_device_limits(&limits),
            _ => true,
        }
    }

    fn is_supported_by_1d_workgroup_limits(&self, limits: &wgpu::Limits) -> bool {
        if let Self::FourStepStage(key) = self {
            if key.kind == FourStepKernelKind::StripeTranspose {
                const TILE: u32 = 16;
                return key.workgroup_size <= limits.max_compute_invocations_per_workgroup
                    && TILE <= limits.max_compute_workgroup_size_x
                    && TILE <= limits.max_compute_workgroup_size_y;
            }
        }
        let workgroup_size = match self {
            Self::StockhamStage(key) => key.workgroup_size,
            Self::FusedPow2Stage(key) => key.workgroup_size,
            Self::FusedSmoothStage(key) => key.workgroup_size,
            Self::FusedPrimeStage(key) => key.workgroup_size,
            Self::BridgeStage(key) => key.workgroup_size,
            Self::RaderStage(key) => key.workgroup_size,
            Self::BluesteinStage(key) => key.workgroup_size,
            Self::RealStage(key) => key.workgroup_size,
            Self::C2cSmoothStage(key) => key.workgroup_size,
            Self::C2cStridedStage(key) => key.workgroup_size,
            Self::SmallVolume(key) => key.workgroup_size,
            Self::FourStepStage(key) if key.kind == FourStepKernelKind::Scale => key.workgroup_size,
            // Direct DFT has a fixed 64-lane shader. Four-step transpose is
            // 16x16 even though its key records 256 total invocations, so it
            // must not be validated as a 256x1 kernel here.
            Self::DirectDftC2cLut(_) | Self::FourStepStage(_) => return true,
        };
        if workgroup_size == 0
            || workgroup_size > limits.max_compute_invocations_per_workgroup
            || workgroup_size > limits.max_compute_workgroup_size_x
        {
            return false;
        }

        let scratch_bytes = match self {
            Self::RaderStage(key) if key.kind == RaderKernelKind::Sum => {
                u64::from(workgroup_size).checked_mul(key.precision.complex_size_bytes())
            }
            Self::BridgeStage(key) if key.kind == BridgeKernelKind::RaderSumAccumulate => {
                u64::from(workgroup_size).checked_mul(8)
            }
            _ => Some(0),
        };
        scratch_bytes
            .is_some_and(|bytes| bytes <= u64::from(limits.max_compute_workgroup_storage_size))
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
            Self::SmallVolume(_) => Some(AxisPrecision::F32),
            _ => None,
        }
    }

    #[cfg(feature = "serde")]
    fn validate_snapshot_shape(&self) -> Result<(), PipelineCacheSnapshotError> {
        match self {
            Self::StockhamStage(key) => {
                validate_snapshot_axis_shape(
                    key.rank,
                    key.axis,
                    &key.dims,
                    key.axis_length,
                    key.stride_complex,
                    key.workgroup_size,
                )?;
                if !is_supported_snapshot_radix(key.radix)
                    || key.ns == 0
                    || key.ns > key.axis_length
                    || key.ns % key.radix != 0
                    || key.axis_length % key.ns != 0
                {
                    return snapshot_integrity("invalid Stockham radix/stage geometry");
                }
                validate_snapshot_axis_scale(key.precision, key.apply_scale, key.scale_bits)
            }
            Self::FusedPow2Stage(key) => {
                validate_snapshot_axis_shape(
                    key.rank,
                    key.axis,
                    &key.dims,
                    key.axis_length,
                    key.stride_complex,
                    key.workgroup_size,
                )?;
                if key.axis_length < 2 || !key.axis_length.is_power_of_two() {
                    return snapshot_integrity("invalid fused power-of-two axis length");
                }
                if key.registers.as_ref().is_some_and(|registers| {
                    key.lines_per_workgroup == 0
                        || !registers.is_consistent(
                            key.axis_length,
                            key.workgroup_size / key.lines_per_workgroup,
                        )
                }) {
                    return snapshot_integrity("invalid register-resident radix schedule");
                }
                validate_snapshot_axis_scale(key.precision, key.apply_scale, key.scale_bits)
            }
            Self::FusedSmoothStage(key) => {
                validate_snapshot_axis_shape(
                    key.rank,
                    key.axis,
                    &key.dims,
                    key.axis_length,
                    key.stride_complex,
                    key.workgroup_size,
                )?;
                if key.axis_length < 2
                    || key.axis_length.is_power_of_two()
                    || !validate_snapshot_factors(&key.factors, key.axis_length)
                {
                    return snapshot_integrity("invalid fused smooth-radix geometry");
                }
                validate_snapshot_axis_scale(key.precision, key.apply_scale, key.scale_bits)
            }
            Self::FusedPrimeStage(key) => {
                validate_snapshot_axis_shape(
                    key.rank,
                    key.axis,
                    &key.dims,
                    key.axis_length,
                    key.stride_complex,
                    key.workgroup_size,
                )?;
                validate_snapshot_convolution(key.kind, key.axis_length, key.convolution_length)?;
                let factors_valid = match (key.kind, &key.registers) {
                    (FusedPrimeKind::Bluestein, Some(registers)) => {
                        key.factors == registers.radices
                            && key.lines_per_workgroup > 0
                            && registers.is_consistent(
                                key.convolution_length,
                                key.workgroup_size / key.lines_per_workgroup,
                            )
                    }
                    (FusedPrimeKind::Direct, None) => key.factors == [key.axis_length],
                    (_, None) => validate_snapshot_factors(&key.factors, key.convolution_length),
                    (_, Some(_)) => false,
                };
                if !factors_valid {
                    return snapshot_integrity("invalid fused-prime convolution factors");
                }
                validate_snapshot_axis_scale(key.precision, key.apply_scale, key.scale_bits)
            }
            Self::FourStepStage(key) => {
                if key.workgroup_size == 0
                    || (key.kind == FourStepKernelKind::StripeTranspose
                        && key.workgroup_size != 256)
                {
                    return snapshot_integrity("invalid four-step workgroup geometry");
                }
                Ok(())
            }
            Self::BridgeStage(key) => {
                validate_snapshot_axis_shape(
                    key.rank,
                    key.axis,
                    &key.dims,
                    key.axis_length,
                    key.stride_complex,
                    key.workgroup_size,
                )?;
                let kind = match key.kind {
                    BridgeKernelKind::RaderSumInit
                    | BridgeKernelKind::RaderSumAccumulate
                    | BridgeKernelKind::RaderPack
                    | BridgeKernelKind::RaderMul
                    | BridgeKernelKind::RaderWriteY0
                    | BridgeKernelKind::RaderPost => FusedPrimeKind::Rader,
                    BridgeKernelKind::BluesteinPack
                    | BridgeKernelKind::BluesteinMul
                    | BridgeKernelKind::BluesteinPost => FusedPrimeKind::Bluestein,
                };
                validate_snapshot_convolution(kind, key.axis_length, key.convolution_length)?;
                validate_snapshot_f32_scale(key.apply_scale, key.scale_bits)
            }
            Self::RaderStage(key) => {
                validate_snapshot_axis_shape(
                    key.rank,
                    key.axis,
                    &key.dims,
                    key.axis_length,
                    key.stride_complex,
                    key.workgroup_size,
                )?;
                validate_snapshot_convolution(
                    FusedPrimeKind::Rader,
                    key.axis_length,
                    key.convolution_length,
                )?;
                validate_snapshot_axis_scale(key.precision, key.apply_scale, key.scale_bits)
            }
            Self::BluesteinStage(key) => {
                validate_snapshot_axis_shape(
                    key.rank,
                    key.axis,
                    &key.dims,
                    key.axis_length,
                    key.stride_complex,
                    key.workgroup_size,
                )?;
                validate_snapshot_convolution(
                    FusedPrimeKind::Bluestein,
                    key.axis_length,
                    key.convolution_length,
                )?;
                validate_snapshot_axis_scale(key.precision, key.apply_scale, key.scale_bits)
            }
            Self::RealStage(key) => {
                if key.rank != key.dims.len() {
                    return snapshot_integrity("real shader key rank does not match dimensions");
                }
                if key.workgroup_size == 0 {
                    return snapshot_integrity("real-stage workgroup size must be nonzero");
                }
                match key.kind {
                    RealKernelKind::PackR2c
                    | RealKernelKind::PackR2cWindowed
                    | RealKernelKind::UnpackC2r
                    | RealKernelKind::UnpackC2rWindowed => {
                        validate_snapshot_dims(key.rank, &key.dims)
                    }
                    RealKernelKind::PackRealStrided
                    | RealKernelKind::UnpackRealStrided
                    | RealKernelKind::PackComplexStrided
                    | RealKernelKind::UnpackComplexStrided => {
                        if key.rank == 0 && key.dims.is_empty() {
                            Ok(())
                        } else {
                            snapshot_integrity(
                                "real strided shader key must not carry transform dimensions",
                            )
                        }
                    }
                    RealKernelKind::RealToComplex
                    | RealKernelKind::RealToComplexWindowed
                    | RealKernelKind::ComplexToReal
                    | RealKernelKind::ComplexToRealWindowed => {
                        if key.dims.is_empty() {
                            Ok(())
                        } else {
                            validate_snapshot_dims(key.rank, &key.dims)
                        }
                    }
                }
            }
            Self::C2cSmoothStage(key) => {
                if key.workgroup_size == 0 {
                    return snapshot_integrity("smooth-stage workgroup size must be nonzero");
                }
                Ok(())
            }
            Self::C2cStridedStage(key) => {
                if key.workgroup_size == 0 {
                    return snapshot_integrity("strided-stage workgroup size must be nonzero");
                }
                Ok(())
            }
            Self::DirectDftC2cLut(_) => Ok(()),
            Self::SmallVolume(key) => {
                validate_snapshot_dims(key.dims.len(), &key.dims)?;
                let consistent = key.workgroup_size > 0
                    && key.dims.len() == key.axes.len()
                    && key.dims.iter().zip(&key.axes).all(|(&n, axis)| match axis {
                        SmallVolumeAxis::Stages(radices) => {
                            radices.iter().product::<usize>() == n
                                && radices.iter().all(|&radix| (2..=16).contains(&radix))
                        }
                        SmallVolumeAxis::Direct => n % 2 == 1,
                    })
                    && key
                        .dims
                        .iter()
                        .all(|&n| key.twiddle_length.is_multiple_of(n));
                if !consistent {
                    return snapshot_integrity("invalid small-volume geometry");
                }
                validate_snapshot_axis_scale(AxisPrecision::F32, key.apply_scale, key.scale_bits)
            }
        }
    }
}

#[cfg(feature = "serde")]
fn snapshot_integrity<T>(reason: impl Into<String>) -> Result<T, PipelineCacheSnapshotError> {
    Err(PipelineCacheSnapshotError::Integrity {
        reason: reason.into(),
    })
}

#[cfg(feature = "serde")]
fn validate_snapshot_dims(rank: usize, dims: &[usize]) -> Result<(), PipelineCacheSnapshotError> {
    if rank == 0 || rank != dims.len() || dims.contains(&0) {
        return snapshot_integrity("rank and dimensions are inconsistent");
    }
    let Some(total) = dims
        .iter()
        .try_fold(1usize, |total, &dim| total.checked_mul(dim))
    else {
        return snapshot_integrity("dimension product overflows usize");
    };
    if total > u32::MAX as usize || dims.iter().any(|&dim| dim > u32::MAX as usize) {
        return snapshot_integrity("shader dimensions exceed the WGSL u32 index domain");
    }
    Ok(())
}

#[cfg(feature = "serde")]
fn validate_snapshot_axis_shape(
    rank: usize,
    axis: usize,
    dims: &[usize],
    axis_length: usize,
    stride_complex: usize,
    workgroup_size: u32,
) -> Result<(), PipelineCacheSnapshotError> {
    validate_snapshot_dims(rank, dims)?;
    if axis >= rank
        || dims[axis] != axis_length
        || axis_length > u32::MAX as usize
        || stride_complex == 0
        || stride_complex > u32::MAX as usize
        || workgroup_size == 0
    {
        return snapshot_integrity("axis-stage geometry is inconsistent");
    }
    Ok(())
}

#[cfg(feature = "serde")]
fn validate_snapshot_scale(scale: f64) -> Result<(), PipelineCacheSnapshotError> {
    if !scale.is_finite() {
        return snapshot_integrity("shader scale is not finite");
    }
    Ok(())
}

#[cfg(feature = "serde")]
fn validate_snapshot_axis_scale(
    precision: AxisPrecision,
    apply_scale: bool,
    bits: u64,
) -> Result<(), PipelineCacheSnapshotError> {
    let scale = axis_scale_factor(precision, bits);
    validate_snapshot_scale(scale)?;
    if bits != axis_scale_bits(precision, apply_scale, scale) {
        return snapshot_integrity("shader scale bits are not canonically encoded");
    }
    Ok(())
}

#[cfg(feature = "serde")]
fn validate_snapshot_f32_scale(
    apply_scale: bool,
    bits: u32,
) -> Result<(), PipelineCacheSnapshotError> {
    let scale = f32::from_bits(bits);
    validate_snapshot_scale(f64::from(scale))?;
    let expected = if apply_scale { scale } else { 1.0 };
    if bits != expected.to_bits() {
        return snapshot_integrity("shader f32 scale bits are not canonically encoded");
    }
    Ok(())
}

#[cfg(feature = "serde")]
fn is_supported_snapshot_radix(radix: usize) -> bool {
    matches!(radix, 2 | 3 | 4 | 5 | 7 | 8 | 11 | 13)
}

#[cfg(feature = "serde")]
fn validate_snapshot_factors(factors: &[usize], expected: usize) -> bool {
    !factors.is_empty()
        && factors.iter().copied().all(is_supported_snapshot_radix)
        && factors
            .iter()
            .try_fold(1usize, |product, &factor| product.checked_mul(factor))
            == Some(expected)
}

#[cfg(feature = "serde")]
fn validate_snapshot_convolution(
    kind: FusedPrimeKind,
    axis_length: usize,
    convolution_length: usize,
) -> Result<(), PipelineCacheSnapshotError> {
    if axis_length < 2 || convolution_length == 0 || convolution_length > u32::MAX as usize {
        return snapshot_integrity("invalid prime-axis convolution length");
    }
    let minimum = match kind {
        FusedPrimeKind::Rader => axis_length
            .checked_sub(1)
            .and_then(|length| length.checked_mul(2))
            .and_then(|length| length.checked_sub(1)),
        FusedPrimeKind::Bluestein => axis_length
            .checked_mul(2)
            .and_then(|length| length.checked_sub(1)),
        FusedPrimeKind::Direct => {
            return if convolution_length == axis_length {
                Ok(())
            } else {
                snapshot_integrity("a direct prime kernel has no convolution")
            };
        }
    };
    // Rader may also convolve cyclically over exactly axis_length - 1.
    let cyclic = kind == FusedPrimeKind::Rader && convolution_length + 1 == axis_length;
    if !cyclic
        && (minimum.is_none_or(|minimum| convolution_length < minimum)
            || (kind == FusedPrimeKind::Rader && convolution_length < axis_length))
    {
        return snapshot_integrity("prime-axis convolution is too short");
    }
    Ok(())
}

#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub(crate) struct ComputePipelineCacheKey {
    pub(crate) layout: PipelineLayoutCacheKey,
    pub(crate) entry_point: String,
    pub(crate) shader: ShaderCacheKey,
}

impl ComputePipelineCacheKey {
    /// Whether this axis-plan kernel may also run in place: a fused kernel
    /// that stores each line where it read it.
    pub(crate) fn supports_in_place(&self) -> bool {
        match &self.shader {
            ShaderCacheKey::FusedPow2Stage(key) => key.split_pass.is_none(),
            ShaderCacheKey::FusedSmoothStage(key) => key.split_pass.is_none(),
            _ => false,
        }
    }

    /// The in-place variant of a kernel for which [`Self::supports_in_place`].
    pub(crate) fn in_place(&self) -> Self {
        match &self.shader {
            ShaderCacheKey::FusedPow2Stage(key) => {
                Self::fused_pow2_stage(key.clone().with_in_place())
            }
            ShaderCacheKey::FusedSmoothStage(key) => {
                Self::fused_smooth_stage(key.clone().with_in_place())
            }
            _ => unreachable!("only fused axis kernels run in place"),
        }
    }

    #[cfg(feature = "serde")]
    fn from_shader_key(shader: ShaderCacheKey) -> Self {
        match shader {
            ShaderCacheKey::StockhamStage(key) => Self::stockham_stage(key),
            ShaderCacheKey::FusedPow2Stage(key) => Self::fused_pow2_stage(key),
            ShaderCacheKey::FusedSmoothStage(key) => Self::fused_smooth_stage(key),
            ShaderCacheKey::FusedPrimeStage(key) => Self::fused_prime_stage(key),
            ShaderCacheKey::FourStepStage(key) => Self::four_step_stage(key),
            ShaderCacheKey::BridgeStage(key) => Self::bridge_stage(key),
            ShaderCacheKey::RaderStage(key) => Self::rader_stage(key),
            ShaderCacheKey::BluesteinStage(key) => Self::bluestein_stage(key),
            ShaderCacheKey::RealStage(key) => Self::real_stage(key),
            ShaderCacheKey::C2cSmoothStage(key) => Self::c2c_smooth_stage(key),
            ShaderCacheKey::C2cStridedStage(key) => Self::c2c_strided_stage(key),
            ShaderCacheKey::DirectDftC2cLut(precision) => Self::direct_dft_c2c(precision),
            ShaderCacheKey::SmallVolume(key) => Self::small_volume(key),
        }
    }

    pub(crate) fn stockham_stage(shader: StockhamStageKey) -> Self {
        let layout = axis_plan_layout_for_precision(shader.precision);
        Self {
            layout,
            entry_point: String::from("main"),
            shader: ShaderCacheKey::StockhamStage(shader),
        }
    }

    pub(crate) fn fused_pow2_stage(shader: FusedPow2StageKey) -> Self {
        let layout = axis_plan_layout(shader.precision, shader.in_place);
        Self {
            layout,
            entry_point: String::from("main"),
            shader: ShaderCacheKey::FusedPow2Stage(shader),
        }
    }

    pub(crate) fn fused_smooth_stage(shader: FusedSmoothStageKey) -> Self {
        let layout = axis_plan_layout(shader.precision, shader.in_place);
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
            AxisPrecision::Df64 => PipelineLayoutCacheKey::FusedPrimeInterleavedDf64,
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
                AxisPrecision::Df64 => PipelineLayoutCacheKey::DirectDftInterleavedDf64Lut,
            },
            entry_point: String::from("main"),
            shader: ShaderCacheKey::DirectDftC2cLut(precision),
        }
    }

    pub(crate) fn small_volume(shader: SmallVolumeKey) -> Self {
        Self {
            layout: PipelineLayoutCacheKey::AxisPlanInterleavedF32Lut,
            entry_point: String::from("main"),
            shader: ShaderCacheKey::SmallVolume(shader),
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
            AxisPrecision::Df64 => PipelineLayoutCacheKey::C2cStridedBinaryDf64,
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
        (RaderKernelKind::Sum, AxisPrecision::Df64) => {
            PipelineLayoutCacheKey::RaderSumInterleavedDf64
        }
        (RaderKernelKind::Pack, AxisPrecision::Df64) => {
            PipelineLayoutCacheKey::RaderPackInterleavedDf64
        }
        (RaderKernelKind::Mul, AxisPrecision::Df64) => {
            PipelineLayoutCacheKey::RaderMulInterleavedDf64
        }
        (RaderKernelKind::WriteY0, AxisPrecision::Df64) => {
            PipelineLayoutCacheKey::RaderWriteY0InterleavedDf64
        }
        (RaderKernelKind::Post, AxisPrecision::Df64) => {
            PipelineLayoutCacheKey::RaderPostInterleavedDf64
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
        (BluesteinKernelKind::Pack, AxisPrecision::Df64) => {
            PipelineLayoutCacheKey::BluesteinPackInterleavedDf64
        }
        (BluesteinKernelKind::Mul, AxisPrecision::Df64) => {
            PipelineLayoutCacheKey::BluesteinMulInterleavedDf64
        }
        (BluesteinKernelKind::Post, AxisPrecision::Df64) => {
            PipelineLayoutCacheKey::BluesteinPostInterleavedDf64
        }
    }
}

fn axis_plan_layout_for_precision(precision: AxisPrecision) -> PipelineLayoutCacheKey {
    match precision {
        AxisPrecision::F32 => PipelineLayoutCacheKey::AxisPlanInterleavedF32Lut,
        AxisPrecision::F64 => PipelineLayoutCacheKey::AxisPlanInterleavedF64Lut,
        AxisPrecision::Df64 => PipelineLayoutCacheKey::AxisPlanInterleavedDf64Lut,
    }
}

/// Layout of an axis-plan kernel, in place or out of place.
pub(crate) fn axis_plan_layout(precision: AxisPrecision, in_place: bool) -> PipelineLayoutCacheKey {
    match (precision, in_place) {
        (_, false) => axis_plan_layout_for_precision(precision),
        (AxisPrecision::F32, true) => PipelineLayoutCacheKey::AxisPlanInPlaceF32Lut,
        (AxisPrecision::F64, true) => PipelineLayoutCacheKey::AxisPlanInPlaceF64Lut,
        (AxisPrecision::Df64, true) => PipelineLayoutCacheKey::AxisPlanInPlaceDf64Lut,
    }
}

#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) enum RaderKernelKind {
    Sum,
    Pack,
    Mul,
    WriteY0,
    Post,
}

#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
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

#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) enum FusedPrimeKind {
    Rader,
    Bluestein,
    /// A direct DFT of a short prime axis (see `runtime::direct_prime`).
    Direct,
}

#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
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
            Self::Direct => "direct",
        }
    }
}

#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
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

#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
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

#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
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

#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
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

#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub(crate) struct C2cSmoothStageKey {
    pub(crate) kind: C2cSmoothKernelKind,
    pub(crate) workgroup_size: u32,
}

#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
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

#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
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

#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
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

#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
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

#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
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

#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
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

#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
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

#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
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
    /// Lines transformed by one workgroup; 1 keeps the original kernel.
    #[cfg_attr(feature = "serde", serde(default = "one_line_per_workgroup"))]
    pub(crate) lines_per_workgroup: u32,
    /// Set when this kernel is one pass of a split long axis.
    #[cfg_attr(feature = "serde", serde(default))]
    pub(crate) split_pass: Option<SplitPass>,
    /// Set when the line is kept in registers instead of workgroup memory.
    #[cfg_attr(feature = "serde", serde(default))]
    pub(crate) registers: Option<RegisterSchedule>,
    /// Reads and writes one buffer in place (see [`Self::with_in_place`]).
    #[cfg_attr(feature = "serde", serde(default))]
    pub(crate) in_place: bool,
}

/// Radix schedule of a register-resident fused kernel (see
/// `runtime::register_fft`); the workgroup size fixes the elements each
/// invocation holds.
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub(crate) struct RegisterSchedule {
    /// Stockham radices, first stage first.
    pub(crate) radices: Vec<usize>,
    /// Elements of the workgroup exchange buffer, a power of two.
    pub(crate) exchange_len: usize,
}

impl RegisterSchedule {
    fn stable_key_suffix(&self) -> String {
        format!(
            ":registers=r{}.exchange{}",
            dims_key(&self.radices),
            self.exchange_len
        )
    }

    /// Whether the schedule is a valid factorization for `workgroup_size`
    /// invocations over a line of `axis_length`.
    #[cfg(feature = "serde")]
    fn is_consistent(&self, axis_length: usize, workgroup_size: u32) -> bool {
        let workgroup_size = workgroup_size as usize;
        workgroup_size > 0
            && axis_length.is_multiple_of(workgroup_size)
            && self.exchange_len.is_power_of_two()
            && self.exchange_len <= axis_length
            && axis_length.is_multiple_of(self.exchange_len)
            && self.radices.iter().product::<usize>() == axis_length
            && self.radices.iter().all(|&radix| {
                matches!(radix, 2 | 4 | 8 | 16)
                    && (axis_length / workgroup_size).is_multiple_of(radix)
            })
    }
}

#[cfg(feature = "serde")]
fn one_line_per_workgroup() -> u32 {
    1
}

/// One pass of an axis split into two fused passes, `N = N1 * N2`.
///
/// Viewing the axis as `n = n2 + N2 * n1`, the first pass transforms `n1`
/// and multiplies by `W_N^(n2 * k1)` on store; the second transforms `n2` and
/// stores element `k2` of line `k1` at `k1 + N1 * k2`.
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub(crate) struct SplitPass {
    /// Full axis length `N`; the bound twiddle table holds `N` entries.
    pub(crate) full_length: usize,
    /// Multiplier for this pass's own twiddle indices (`N` / pass length).
    pub(crate) twiddle_scale: usize,
    /// First pass: `(rows, row_stride_lines)`, where the row index `n2` of a
    /// line is `(line / row_stride_lines) % rows`.
    pub(crate) row_twiddle: Option<(usize, usize)>,
    /// Second pass: the `(dims, axis)` the transform stores through.
    pub(crate) output: Option<(Vec<usize>, usize)>,
}

impl SplitPass {
    fn stable_key_suffix(&self) -> String {
        let mut suffix = format!(":split=n{}.scale{}", self.full_length, self.twiddle_scale);
        if let Some((rows, row_stride_lines)) = self.row_twiddle {
            suffix.push_str(&format!(".rows{rows}.rowstride{row_stride_lines}"));
        }
        if let Some((dims, axis)) = &self.output {
            suffix.push_str(&format!(".out{}.axis{axis}", dims_key(dims)));
        }
        suffix
    }
}

#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
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
    /// Lines transformed by one workgroup; 1 keeps the original kernel.
    #[cfg_attr(feature = "serde", serde(default = "one_line_per_workgroup"))]
    pub(crate) lines_per_workgroup: u32,
    /// Set when this kernel is one pass of a split long axis.
    #[cfg_attr(feature = "serde", serde(default))]
    pub(crate) split_pass: Option<SplitPass>,
    /// Reads and writes one buffer in place (see
    /// [`FusedPow2StageKey::with_in_place`]).
    #[cfg_attr(feature = "serde", serde(default))]
    pub(crate) in_place: bool,
    /// Pads workgroup-memory indices by one element per 16 (see
    /// `fused_smooth_pads_indices`).
    #[cfg_attr(feature = "serde", serde(default))]
    pub(crate) padded_indices: bool,
}

#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
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
    /// Lines per workgroup of a direct, multi-line Rader, or strided register
    /// Bluestein kernel; 1 otherwise.
    #[cfg_attr(feature = "serde", serde(default = "one_line_per_workgroup"))]
    pub(crate) lines_per_workgroup: u32,
    /// Set when a Bluestein convolution runs in registers.
    #[cfg_attr(feature = "serde", serde(default))]
    pub(crate) registers: Option<RegisterSchedule>,
    /// Output pairs per invocation of a direct kernel; 1 otherwise.
    #[cfg_attr(feature = "serde", serde(default = "one_line_per_workgroup"))]
    pub(crate) pairs_per_invocation: u32,
    /// Pads the workgroup-memory indices of a fused Rader or Bluestein
    /// convolution by one element per 16.
    #[cfg_attr(feature = "serde", serde(default))]
    pub(crate) padded_indices: bool,
    /// Strided lines a one-line fused Rader kernel loads and stores together
    /// and convolves one after another; 1 otherwise.
    #[cfg_attr(feature = "serde", serde(default = "one_line_per_workgroup"))]
    pub(crate) serial_lines: u32,
}

/// How one axis of a small volume is transformed (see
/// `runtime::small_volume`).
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub(crate) enum SmallVolumeAxis {
    /// Stockham stages of these radices, first stage first.
    Stages(Vec<usize>),
    /// A direct DFT of an odd length over symmetric pairs.
    Direct,
}

/// An `f32` kernel that transforms every axis of a volume small enough for
/// one workgroup's memory (see `runtime::small_volume`).
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub(crate) struct SmallVolumeKey {
    /// Axis lengths, the first contiguous.
    pub(crate) dims: Vec<usize>,
    pub(crate) axes: Vec<SmallVolumeAxis>,
    /// Points of the twiddle table, a multiple of every axis length.
    pub(crate) twiddle_length: usize,
    pub(crate) direction: FftDirection,
    pub(crate) workgroup_size: u32,
    pub(crate) apply_scale: bool,
    scale_bits: u64,
}

impl SmallVolumeKey {
    pub(crate) fn new(
        dims: &[usize],
        axes: Vec<SmallVolumeAxis>,
        twiddle_length: usize,
        direction: FftDirection,
        workgroup_size: u32,
        apply_scale: bool,
        scale_factor: f64,
    ) -> Self {
        debug_assert_eq!(dims.len(), axes.len());
        Self {
            dims: dims.to_vec(),
            axes,
            twiddle_length,
            direction,
            workgroup_size,
            apply_scale,
            scale_bits: axis_scale_bits(AxisPrecision::F32, apply_scale, scale_factor),
        }
    }

    pub(crate) fn scale_factor(&self) -> f64 {
        axis_scale_factor(AxisPrecision::F32, self.scale_bits)
    }

    /// Workgroup memory the kernel needs: the volume and each direct axis's
    /// table of roots.
    pub(crate) fn workgroup_storage_bytes(&self) -> Option<u64> {
        let elements = self
            .dims
            .iter()
            .try_fold(1usize, |product, &n| product.checked_mul(n))?;
        let roots = self
            .dims
            .iter()
            .zip(&self.axes)
            .filter(|(_, axis)| **axis == SmallVolumeAxis::Direct)
            .map(|(&n, _)| n)
            .sum::<usize>();
        let complex = elements.checked_add(roots)?;
        u64::try_from(complex).ok()?.checked_mul(8)
    }

    /// Whether `limits` allow this kernel.
    pub(crate) fn supported_by_device_limits(&self, limits: &wgpu::Limits) -> bool {
        self.workgroup_storage_bytes()
            .is_some_and(|bytes| bytes <= u64::from(limits.max_compute_workgroup_storage_size))
            && self.workgroup_size > 0
            && self.workgroup_size <= limits.max_compute_invocations_per_workgroup
            && self.workgroup_size <= limits.max_compute_workgroup_size_x
    }

    pub(crate) fn stable_key(&self) -> String {
        let axes = self
            .axes
            .iter()
            .map(|axis| match axis {
                SmallVolumeAxis::Stages(radices) => format!("r{}", dims_key(radices)),
                SmallVolumeAxis::Direct => String::from("direct"),
            })
            .collect::<Vec<_>>()
            .join(",");
        format!(
            "shader:v1:small-volume:precision=f32:dims={}:axes={axes}:twiddles={}:direction={}:workgroup={}:scale={}:scale_bits={}:twiddle=host-f64-f32-v1",
            dims_key(&self.dims),
            self.twiddle_length,
            direction_key(self.direction),
            self.workgroup_size,
            self.apply_scale,
            axis_scale_bits_key(AxisPrecision::F32, self.scale_bits),
        )
    }
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
            lines_per_workgroup: 1,
            registers: None,
            pairs_per_invocation: 1,
            padded_indices: false,
            serial_lines: 1,
        }
    }

    /// Loads and stores `lines` strided lines together and convolves them one
    /// after another (one-line fused Rader kernels only).
    pub(crate) fn with_serial_lines(mut self, lines: u32) -> Self {
        self.serial_lines = lines.max(1);
        self
    }

    /// Pads the convolution's workgroup-memory indices by one element per 16.
    pub(crate) fn with_padded_indices(mut self) -> Self {
        self.padded_indices = true;
        self
    }

    /// Whether `limits` allow this kernel.
    pub(crate) fn supported_by_device_limits(&self, limits: &wgpu::Limits) -> bool {
        self.is_supported_by_limits(
            u64::from(limits.max_compute_workgroup_storage_size),
            limits.max_compute_invocations_per_workgroup,
            limits.max_compute_workgroup_size_x,
        )
    }

    /// Runs the convolution's FFTs in registers with `schedule`.
    pub(crate) fn with_registers(mut self, schedule: RegisterSchedule) -> Self {
        self.registers = Some(schedule);
        self
    }

    /// Transforms `lines` lines per workgroup (direct kernels only).
    pub(crate) fn with_lines_per_workgroup(mut self, lines: u32) -> Self {
        self.lines_per_workgroup = lines.max(1);
        self
    }

    /// Produces `pairs` output pairs per invocation (direct kernels only).
    pub(crate) fn with_pairs_per_invocation(mut self, pairs: u32) -> Self {
        self.pairs_per_invocation = pairs.max(1);
        self
    }

    pub(crate) fn scale_factor(&self) -> f64 {
        axis_scale_factor(self.precision, self.scale_bits)
    }

    pub(crate) fn stable_key(&self) -> String {
        let mut key = format!(
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
        );
        if self.lines_per_workgroup > 1 {
            key.push_str(&format!(":lines={}", self.lines_per_workgroup));
        }
        if self.pairs_per_invocation > 1 {
            key.push_str(&format!(":pairs={}", self.pairs_per_invocation));
        }
        if self.padded_indices {
            key.push_str(":pad16");
        }
        if self.serial_lines > 1 {
            key.push_str(&format!(":serial={}", self.serial_lines));
        }
        if let Some(registers) = &self.registers {
            key.push_str(&registers.stable_key_suffix());
        }
        key
    }

    pub(crate) fn is_supported_by_limits(
        &self,
        max_workgroup_storage_bytes: u64,
        max_invocations_per_workgroup: u32,
        max_workgroup_size_x: u32,
    ) -> bool {
        let complex_bytes = self.precision.complex_size_bytes() as usize;
        let scratch_elements = match (self.kind, &self.registers) {
            (_, Some(registers)) => registers
                .exchange_len
                .checked_mul(self.lines_per_workgroup as usize),
            (FusedPrimeKind::Direct, None) => self
                .axis_length
                .checked_mul(self.lines_per_workgroup as usize),
            // Several lines of the convolution, and each line's x[0].
            (FusedPrimeKind::Rader, None) if self.lines_per_workgroup > 1 => {
                let lines = self.lines_per_workgroup as usize;
                crate::runtime::axis_plan::multiline_line_stride(
                    self.convolution_length,
                    lines,
                    self.stride_complex != 1,
                )
                .checked_mul(lines)
                .map(|elements| {
                    if self.padded_indices {
                        crate::runtime::axis_plan::padded_workgroup_len(elements)
                    } else {
                        elements
                    }
                })
                .and_then(|elements| elements.checked_add(lines))
            }
            _ if self.padded_indices => Some(crate::runtime::axis_plan::padded_workgroup_len(
                self.convolution_length,
            )),
            _ => Some(self.convolution_length),
        };
        let Some(scratch_bytes) =
            scratch_elements.and_then(|elements| elements.checked_mul(complex_bytes))
        else {
            return false;
        };
        let extra_bytes = match self.kind {
            // Each serial line's x[0] and output bin 0.
            FusedPrimeKind::Rader if self.serial_lines > 1 => {
                2 * self.serial_lines as usize * complex_bytes
            }
            FusedPrimeKind::Rader => 0usize,
            FusedPrimeKind::Bluestein => 0usize,
            // The roots.
            FusedPrimeKind::Direct => self.axis_length * complex_bytes,
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
            lines_per_workgroup: 1,
            split_pass: None,
            in_place: false,
            padded_indices: false,
        }
    }

    /// Pads workgroup-memory indices by one element per 16.
    pub(crate) fn with_padded_indices(mut self) -> Self {
        self.padded_indices = true;
        self
    }

    /// Whether `limits` allow this kernel.
    pub(crate) fn supported_by_device_limits(&self, limits: &wgpu::Limits) -> bool {
        self.is_supported_by_limits(
            u64::from(limits.max_compute_workgroup_storage_size),
            limits.max_compute_invocations_per_workgroup,
            limits.max_compute_workgroup_size_x,
        )
    }

    /// See [`FusedPow2StageKey::with_in_place`].
    pub(crate) fn with_in_place(mut self) -> Self {
        debug_assert!(self.split_pass.is_none());
        self.in_place = true;
        self
    }

    /// Transforms `lines` lines per workgroup instead of one.
    pub(crate) fn with_lines_per_workgroup(mut self, lines: u32) -> Self {
        self.lines_per_workgroup = lines.max(1);
        self
    }

    /// Makes this kernel one pass of a split long axis.
    pub(crate) fn with_split_pass(mut self, split_pass: SplitPass) -> Self {
        self.split_pass = Some(split_pass);
        self
    }

    pub(crate) fn scale_factor(&self) -> f64 {
        axis_scale_factor(self.precision, self.scale_bits)
    }

    pub(crate) fn stable_key(&self) -> String {
        let mut key = format!(
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
        );
        if self.lines_per_workgroup > 1 {
            key.push_str(&format!(":lines={}", self.lines_per_workgroup));
        }
        if let Some(split_pass) = &self.split_pass {
            key.push_str(&split_pass.stable_key_suffix());
        }
        if self.in_place {
            key.push_str(":in_place");
        }
        if self.padded_indices {
            key.push_str(":pad16");
        }
        key
    }

    fn is_supported_by_limits(
        &self,
        max_workgroup_storage_bytes: u64,
        max_invocations_per_workgroup: u32,
        max_workgroup_size_x: u32,
    ) -> bool {
        let Some(scratch_bytes) = crate::runtime::axis_plan::multiline_line_stride(
            self.axis_length,
            self.lines_per_workgroup as usize,
            crate::runtime::axis_plan::multiline_element_major(
                self.stride_complex,
                self.split_pass.as_ref(),
            ),
        )
        .checked_mul(self.lines_per_workgroup as usize)
        .map(|elements| {
            if self.padded_indices {
                crate::runtime::axis_plan::padded_workgroup_len(elements)
            } else {
                elements
            }
        })
        .and_then(|elements| elements.checked_mul(self.precision.complex_size_bytes() as usize)) else {
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
            lines_per_workgroup: 1,
            split_pass: None,
            registers: None,
            in_place: false,
        }
    }

    /// Reads and writes one buffer in place. Safe because every workgroup
    /// reads its whole lines before writing them and no line crosses
    /// workgroups; not for split passes, which store transposed.
    pub(crate) fn with_in_place(mut self) -> Self {
        debug_assert!(self.split_pass.is_none());
        self.in_place = true;
        self
    }

    /// Transforms `lines` lines per workgroup instead of one.
    pub(crate) fn with_lines_per_workgroup(mut self, lines: u32) -> Self {
        self.lines_per_workgroup = lines.max(1);
        self
    }

    /// Makes this kernel one pass of a split long axis.
    pub(crate) fn with_split_pass(mut self, split_pass: SplitPass) -> Self {
        self.split_pass = Some(split_pass);
        self
    }

    /// Keeps the line in registers with `schedule`.
    pub(crate) fn with_registers(mut self, schedule: RegisterSchedule) -> Self {
        self.registers = Some(schedule);
        self
    }

    pub(crate) fn scale_factor(&self) -> f64 {
        axis_scale_factor(self.precision, self.scale_bits)
    }

    pub(crate) fn stable_key(&self) -> String {
        let mut key = format!(
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
        );
        if self.lines_per_workgroup > 1 {
            key.push_str(&format!(":lines={}", self.lines_per_workgroup));
        }
        if let Some(split_pass) = &self.split_pass {
            key.push_str(&split_pass.stable_key_suffix());
        }
        if let Some(registers) = &self.registers {
            key.push_str(&registers.stable_key_suffix());
        }
        if self.in_place {
            key.push_str(":in_place");
        }
        key
    }

    fn is_supported_by_limits(
        &self,
        max_workgroup_storage_bytes: u64,
        max_invocations_per_workgroup: u32,
        max_workgroup_size_x: u32,
    ) -> bool {
        let scratch_elements = match &self.registers {
            Some(registers) => registers
                .exchange_len
                .checked_mul(self.lines_per_workgroup as usize),
            None => crate::runtime::axis_plan::multiline_line_stride(
                self.axis_length,
                self.lines_per_workgroup as usize,
                crate::runtime::axis_plan::multiline_element_major(
                    self.stride_complex,
                    self.split_pass.as_ref(),
                ),
            )
            .checked_mul(self.lines_per_workgroup as usize),
        };
        let Some(scratch_bytes) = scratch_elements.and_then(|elements| {
            elements.checked_mul(self.precision.complex_size_bytes() as usize)
        }) else {
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
        AxisPrecision::Df64 => {
            let value = DoubleFloat::from_f64(value);
            u64::from(value.hi.to_bits()) | (u64::from(value.lo.to_bits()) << 32)
        }
    }
}

fn axis_scale_factor(precision: AxisPrecision, bits: u64) -> f64 {
    match precision {
        AxisPrecision::F32 => f64::from(f32::from_bits(bits as u32)),
        AxisPrecision::F64 => f64::from_bits(bits),
        AxisPrecision::Df64 => {
            f64::from(f32::from_bits(bits as u32)) + f64::from(f32::from_bits((bits >> 32) as u32))
        }
    }
}

fn axis_scale_bits_key(precision: AxisPrecision, bits: u64) -> String {
    match precision {
        AxisPrecision::F32 => format!("0x{:08x}", bits as u32),
        AxisPrecision::F64 => format!("0x{bits:016x}"),
        AxisPrecision::Df64 => format!("hi=0x{:08x},lo=0x{:08x}", bits as u32, (bits >> 32) as u32),
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

/// Debug label passed to wgpu for a shader module.
///
/// wgpu hands the label to DXC as the source file name, and DXC fails to
/// "read" names with some colon patterns (such as `a:b=c`), which stable
/// cache keys contain. Cache keys themselves are unchanged.
fn shader_module_label(label: &str) -> String {
    label.replace(':', ".")
}

fn bind_group_layout_entries(key: PipelineLayoutCacheKey) -> Vec<wgpu::BindGroupLayoutEntry> {
    match key {
        PipelineLayoutCacheKey::C2cSmoothBinaryF32
        | PipelineLayoutCacheKey::C2cStridedBinaryF32
        | PipelineLayoutCacheKey::C2cStridedBinaryF64
        | PipelineLayoutCacheKey::C2cStridedBinaryDf64
        | PipelineLayoutCacheKey::RealBinaryF32
        | PipelineLayoutCacheKey::RaderWriteY0InterleavedF32
        | PipelineLayoutCacheKey::RaderWriteY0InterleavedF64
        | PipelineLayoutCacheKey::RaderWriteY0InterleavedDf64 => {
            vec![
                storage_entry(0, true),
                storage_entry(1, false),
                uniform_entry(2),
            ]
        }
        PipelineLayoutCacheKey::AxisPlanInterleavedF32Lut
        | PipelineLayoutCacheKey::AxisPlanInterleavedF64Lut
        | PipelineLayoutCacheKey::AxisPlanInterleavedDf64Lut
        | PipelineLayoutCacheKey::DirectDftInterleavedF32Lut
        | PipelineLayoutCacheKey::DirectDftInterleavedF64Lut
        | PipelineLayoutCacheKey::DirectDftInterleavedDf64Lut => vec![
            storage_entry(0, true),
            storage_entry(1, false),
            uniform_entry(2),
            storage_entry(3, true),
        ],
        PipelineLayoutCacheKey::AxisPlanInPlaceF32Lut
        | PipelineLayoutCacheKey::AxisPlanInPlaceF64Lut
        | PipelineLayoutCacheKey::AxisPlanInPlaceDf64Lut => vec![
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
        | PipelineLayoutCacheKey::FusedPrimeInterleavedF64
        | PipelineLayoutCacheKey::FusedPrimeInterleavedDf64 => vec![
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
        | PipelineLayoutCacheKey::BluesteinPostInterleavedF64
        | PipelineLayoutCacheKey::BluesteinPostInterleavedDf64 => vec![
            storage_entry(0, true),
            storage_entry(1, true),
            storage_entry(2, false),
            uniform_entry(3),
        ],
        PipelineLayoutCacheKey::RaderSumInterleavedF32
        | PipelineLayoutCacheKey::RaderSumInterleavedF64
        | PipelineLayoutCacheKey::RaderSumInterleavedDf64 => vec![
            storage_entry(0, true),
            storage_entry(1, false),
            storage_entry(2, false),
            uniform_entry(3),
        ],
        PipelineLayoutCacheKey::RaderPackInterleavedF32
        | PipelineLayoutCacheKey::RaderPackInterleavedF64
        | PipelineLayoutCacheKey::RaderPackInterleavedDf64
        | PipelineLayoutCacheKey::BluesteinPackInterleavedF32
        | PipelineLayoutCacheKey::BluesteinPackInterleavedF64
        | PipelineLayoutCacheKey::BluesteinPackInterleavedDf64 => vec![
            storage_entry(0, true),
            storage_entry(1, false),
            storage_entry(2, true),
            uniform_entry(3),
        ],
        PipelineLayoutCacheKey::RaderMulInterleavedF32
        | PipelineLayoutCacheKey::RaderMulInterleavedF64
        | PipelineLayoutCacheKey::RaderMulInterleavedDf64
        | PipelineLayoutCacheKey::BluesteinMulInterleavedF32
        | PipelineLayoutCacheKey::BluesteinMulInterleavedF64
        | PipelineLayoutCacheKey::BluesteinMulInterleavedDf64 => {
            vec![
                storage_entry(0, false),
                storage_entry(1, true),
                uniform_entry(2),
            ]
        }
        PipelineLayoutCacheKey::RaderPostInterleavedF32
        | PipelineLayoutCacheKey::RaderPostInterleavedF64
        | PipelineLayoutCacheKey::RaderPostInterleavedDf64 => vec![
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
    fn shader_module_labels_have_no_colons() {
        assert_eq!(
            shader_module_label(
                "wgpu_fft.c2c.strided.shader.shader:v2:c2c-strided:precision=f32:workgroup=64"
            ),
            "wgpu_fft.c2c.strided.shader.shader.v2.c2c-strided.precision=f32.workgroup=64"
        );
    }

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
    fn dynamic_1d_keys_check_lane_and_reduction_storage_limits_without_flattening_tiles() {
        let mut limits = wgpu::Limits {
            max_compute_invocations_per_workgroup: 64,
            max_compute_workgroup_size_x: 64,
            max_compute_workgroup_storage_size: 1024,
            ..wgpu::Limits::default()
        };

        let stockham = ShaderCacheKey::StockhamStage(StockhamStageKey::new(
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
        assert!(stockham.is_supported_by_1d_workgroup_limits(&limits));
        let mut oversized = stockham.clone();
        if let ShaderCacheKey::StockhamStage(key) = &mut oversized {
            key.workgroup_size = 128;
        }
        assert!(!oversized.is_supported_by_1d_workgroup_limits(&limits));

        let rader_sum = ShaderCacheKey::RaderStage(RaderStageKey::new(
            RaderKernelKind::Sum,
            1,
            0,
            &[17],
            17,
            1,
            32,
            64,
            false,
            1.0,
            AxisPrecision::Df64,
        ));
        assert!(rader_sum.is_supported_by_1d_workgroup_limits(&limits));
        limits.max_compute_workgroup_storage_size = 1023;
        assert!(!rader_sum.is_supported_by_1d_workgroup_limits(&limits));

        let transpose = ShaderCacheKey::FourStepStage(FourStepStageKey::new(
            FourStepKernelKind::StripeTranspose,
            256,
        ));
        assert!(!transpose.is_supported_by_1d_workgroup_limits(&limits));
        limits.max_compute_invocations_per_workgroup = 256;
        limits.max_compute_workgroup_size_x = 16;
        limits.max_compute_workgroup_size_y = 16;
        assert!(transpose.is_supported_by_1d_workgroup_limits(&limits));
        limits.max_compute_workgroup_size_y = 8;
        assert!(!transpose.is_supported_by_1d_workgroup_limits(&limits));
        limits.max_compute_workgroup_size_y = 16;
        limits.max_compute_workgroup_size_x = 64;
        let scale =
            ShaderCacheKey::FourStepStage(FourStepStageKey::new(FourStepKernelKind::Scale, 64));
        assert!(scale.is_supported_by_1d_workgroup_limits(&limits));
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
        let df64_key = make_pow2(AxisPrecision::Df64);
        assert_ne!(f32_key, f64_key);
        assert_ne!(f64_key, df64_key);
        assert!(df64_key.stable_key().contains("precision=df64"));
        assert!(df64_key.stable_key().contains("scale_bits=hi="));
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
        let df64_pipeline = ComputePipelineCacheKey::fused_pow2_stage(df64_key);
        assert_eq!(
            f32_pipeline.layout,
            PipelineLayoutCacheKey::AxisPlanInterleavedF32Lut
        );
        assert_eq!(
            f64_pipeline.layout,
            PipelineLayoutCacheKey::AxisPlanInterleavedF64Lut
        );
        assert_eq!(
            df64_pipeline.layout,
            PipelineLayoutCacheKey::AxisPlanInterleavedDf64Lut
        );
        assert_ne!(f32_pipeline.stable_key(), f64_pipeline.stable_key());

        assert_eq!(
            ComputePipelineCacheKey::direct_dft_c2c(AxisPrecision::F64).layout,
            PipelineLayoutCacheKey::DirectDftInterleavedF64Lut
        );
        assert_eq!(
            ComputePipelineCacheKey::direct_dft_c2c(AxisPrecision::Df64).layout,
            PipelineLayoutCacheKey::DirectDftInterleavedDf64Lut
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
        assert_eq!(
            ComputePipelineCacheKey::c2c_strided_stage(C2cStridedStageKey::new(
                C2cStridedKernelKind::Unpack,
                64,
                AxisPrecision::Df64,
            ))
            .layout,
            PipelineLayoutCacheKey::C2cStridedBinaryDf64
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
    fn df64_prime_pipeline_keys_use_distinct_typed_layouts() {
        let factors = [8, 5, 5];
        let fused = ComputePipelineCacheKey::fused_prime_stage(FusedPrimeStageKey::new(
            FusedPrimeKind::Rader,
            1,
            0,
            &[101],
            101,
            1,
            200,
            &factors,
            FftDirection::Inverse,
            256,
            true,
            1.0 / 101.0,
            AxisPrecision::Df64,
        ));
        assert_eq!(
            fused.layout,
            PipelineLayoutCacheKey::FusedPrimeInterleavedDf64
        );
        assert!(fused.stable_key().contains("precision=df64"));
        assert!(fused.stable_key().contains("scale_bits=hi="));

        for (kind, expected) in [
            (
                RaderKernelKind::Sum,
                PipelineLayoutCacheKey::RaderSumInterleavedDf64,
            ),
            (
                RaderKernelKind::Pack,
                PipelineLayoutCacheKey::RaderPackInterleavedDf64,
            ),
            (
                RaderKernelKind::Mul,
                PipelineLayoutCacheKey::RaderMulInterleavedDf64,
            ),
            (
                RaderKernelKind::WriteY0,
                PipelineLayoutCacheKey::RaderWriteY0InterleavedDf64,
            ),
            (
                RaderKernelKind::Post,
                PipelineLayoutCacheKey::RaderPostInterleavedDf64,
            ),
        ] {
            let pipeline = ComputePipelineCacheKey::rader_stage(RaderStageKey::new(
                kind,
                1,
                0,
                &[17],
                17,
                1,
                32,
                64,
                true,
                1.0 / 17.0,
                AxisPrecision::Df64,
            ));
            assert_eq!(pipeline.layout, expected);
        }

        for (kind, expected) in [
            (
                BluesteinKernelKind::Pack,
                PipelineLayoutCacheKey::BluesteinPackInterleavedDf64,
            ),
            (
                BluesteinKernelKind::Mul,
                PipelineLayoutCacheKey::BluesteinMulInterleavedDf64,
            ),
            (
                BluesteinKernelKind::Post,
                PipelineLayoutCacheKey::BluesteinPostInterleavedDf64,
            ),
        ] {
            let pipeline = ComputePipelineCacheKey::bluestein_stage(BluesteinStageKey::new(
                kind,
                1,
                0,
                &[34],
                34,
                1,
                70,
                64,
                true,
                1.0 / 34.0,
                AxisPrecision::Df64,
            ));
            assert_eq!(pipeline.layout, expected);
        }
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
        // Rader and Bluestein both need exactly the convolution.
        assert!(rader.is_supported_by_limits(48_000, 256, 256));
        assert!(!rader.is_supported_by_limits(47_999, 256, 256));
        assert!(!rader.is_supported_by_limits(48_000, 255, 256));
        assert!(!rader.is_supported_by_limits(48_000, 256, 255));

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
            ShaderCacheKey::SmallVolume(_) => unreachable!(),
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

        let df64 = ComputePipelineCacheKey::c2c_strided_stage(C2cStridedStageKey::new(
            C2cStridedKernelKind::Unpack,
            64,
            AxisPrecision::Df64,
        ));
        assert_eq!(df64.layout, PipelineLayoutCacheKey::C2cStridedBinaryDf64);
        assert_eq!(
            df64.stable_key(),
            "pipeline:v1:layout=c2c-strided/binary-df64:entry=main:shader:v2:c2c-strided:unpack-c2c-strided:precision=df64:workgroup=64"
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

    #[cfg(feature = "serde")]
    fn serializable_stockham_snapshot() -> PipelineCacheSnapshot {
        let shader = ShaderCacheKey::StockhamStage(StockhamStageKey::new(
            2,
            1,
            &[3, 8],
            8,
            3,
            8,
            8,
            FftDirection::Inverse,
            64,
            true,
            1.0 / 24.0,
            AxisPrecision::F32,
        ));
        let pipeline = ComputePipelineCacheKey::from_shader_key(shader.clone());
        PipelineCacheSnapshot::from_entries(
            vec![SnapshotShaderEntry {
                code: shader.fallback_source(),
                key: shader,
            }],
            vec![pipeline],
        )
    }

    #[cfg(feature = "serde")]
    #[test]
    fn snapshot_json_round_trip_preserves_typed_entries() {
        let snapshot = serializable_stockham_snapshot();
        let json = snapshot.to_json().expect("snapshot should serialize");
        let decoded = PipelineCacheSnapshot::from_json(&json).expect("snapshot should decode");

        assert_eq!(decoded, snapshot);
        assert!(json.contains(PIPELINE_CACHE_SNAPSHOT_SCHEMA));
        assert!(json.contains("StockhamStage"));
    }

    #[cfg(feature = "serde")]
    #[test]
    fn snapshot_json_round_trip_accepts_dimensionless_real_strided_keys() {
        let shader =
            ShaderCacheKey::RealStage(RealStageKey::new(RealKernelKind::PackRealStrided, &[], 64));
        let snapshot = PipelineCacheSnapshot::from_entries(
            vec![SnapshotShaderEntry {
                code: shader.fallback_source(),
                key: shader.clone(),
            }],
            vec![ComputePipelineCacheKey::from_shader_key(shader)],
        );

        let json = snapshot.to_json().expect("snapshot should serialize");
        let decoded = PipelineCacheSnapshot::from_json(&json).expect("snapshot should decode");
        assert_eq!(decoded, snapshot);
    }

    #[cfg(feature = "serde")]
    #[test]
    fn snapshot_json_rejects_schema_and_version_drift() {
        let json = serializable_stockham_snapshot()
            .to_json()
            .expect("snapshot should serialize");
        let mut value: serde_json::Value = serde_json::from_str(&json).unwrap();
        value["schema"] = serde_json::Value::String(String::from("other.pipeline-cache"));
        let error = PipelineCacheSnapshot::from_json(&serde_json::to_string(&value).unwrap())
            .expect_err("wrong schema must fail");
        assert!(matches!(
            error,
            PipelineCacheSnapshotError::SchemaMismatch { .. }
        ));

        let mut value: serde_json::Value = serde_json::from_str(&json).unwrap();
        value["version"] = serde_json::Value::from(PIPELINE_CACHE_SNAPSHOT_VERSION + 1);
        let error = PipelineCacheSnapshot::from_json(&serde_json::to_string(&value).unwrap())
            .expect_err("wrong version must fail");
        assert!(matches!(
            error,
            PipelineCacheSnapshotError::VersionMismatch { .. }
        ));
    }

    #[cfg(feature = "serde")]
    #[test]
    fn snapshot_json_rejects_modified_wgsl_and_stable_keys() {
        let json = serializable_stockham_snapshot()
            .to_json()
            .expect("snapshot should serialize");
        let mut value: serde_json::Value = serde_json::from_str(&json).unwrap();
        value["shader_entries"][0]["code"] =
            serde_json::Value::String(String::from("@compute @workgroup_size(1) fn main() {}"));
        value["shader_codes"][0] = value["shader_entries"][0]["code"].clone();
        let error = PipelineCacheSnapshot::from_json(&serde_json::to_string(&value).unwrap())
            .expect_err("modified WGSL must fail");
        assert!(matches!(
            error,
            PipelineCacheSnapshotError::Integrity { .. }
        ));

        let mut value: serde_json::Value = serde_json::from_str(&json).unwrap();
        let forged_bits = (1u64 << 32) | u64::from((1.0f32 / 24.0).to_bits());
        value["shader_entries"][0]["key"]["StockhamStage"]["scale_bits"] =
            serde_json::Value::from(forged_bits);
        value["pipeline_entries"][0]["shader"]["StockhamStage"]["scale_bits"] =
            serde_json::Value::from(forged_bits);
        let error = PipelineCacheSnapshot::from_json(&serde_json::to_string(&value).unwrap())
            .expect_err("noncanonical high f32 scale bits must fail");
        assert!(matches!(
            error,
            PipelineCacheSnapshotError::Integrity { .. }
        ));

        let mut value: serde_json::Value = serde_json::from_str(&json).unwrap();
        value["pipeline_keys"][0] = serde_json::Value::String(String::from("forged-key"));
        let error = PipelineCacheSnapshot::from_json(&serde_json::to_string(&value).unwrap())
            .expect_err("modified stable key must fail");
        assert!(matches!(
            error,
            PipelineCacheSnapshotError::Integrity { .. }
        ));
    }

    #[cfg(feature = "serde")]
    #[test]
    fn snapshot_json_rejects_pipeline_layout_mismatch_and_missing_source() {
        let json = serializable_stockham_snapshot()
            .to_json()
            .expect("snapshot should serialize");
        let mut value: serde_json::Value = serde_json::from_str(&json).unwrap();
        value["pipeline_entries"][0]["layout"] =
            serde_json::Value::String(String::from("RealBinaryF32"));
        let error = PipelineCacheSnapshot::from_json(&serde_json::to_string(&value).unwrap())
            .expect_err("layout mismatch must fail");
        assert!(matches!(
            error,
            PipelineCacheSnapshotError::Integrity { .. }
        ));

        let mut value: serde_json::Value = serde_json::from_str(&json).unwrap();
        value["shader_entries"] = serde_json::Value::Array(Vec::new());
        value["shader_codes"] = serde_json::Value::Array(Vec::new());
        let error = PipelineCacheSnapshot::from_json(&serde_json::to_string(&value).unwrap())
            .expect_err("missing shader source must fail");
        assert!(matches!(
            error,
            PipelineCacheSnapshotError::Integrity { .. }
        ));
    }
}
