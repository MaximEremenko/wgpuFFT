use std::num::NonZeroU64;

use crate::error::{NufftError, Result};
use crate::gpu_dispatch::split_workgroups;

const WORKGROUP_SIZE: u32 = 256;
const ITEMS_PER_INVOCATION: u32 = 8;
pub(crate) const BLOCK_ELEMENTS: usize = WORKGROUP_SIZE as usize * ITEMS_PER_INVOCATION as usize;
const BANK_WIDTH: usize = 32;
const PADDED_ELEMENTS: usize = BLOCK_ELEMENTS + BLOCK_ELEMENTS / BANK_WIDTH;
const WORKGROUP_STORAGE_BYTES: u32 = (PADDED_ELEMENTS * size_of::<u32>()) as u32;
const U32_BYTES: u64 = size_of::<u32>() as u64;
const _: () = assert!(WORKGROUP_STORAGE_BYTES <= 16 * 1024);

struct ScanLevel {
    length: usize,
    block_count: usize,
    block_dispatch: (u32, u32, u32),
    fixup_dispatch: (u32, u32, u32),
}

/// Fixed-length, reusable, out-of-place exclusive scan over `u32` values.
///
/// Arithmetic follows WGSL `u32` wrapping semantics. Each workgroup scans
/// 2,048 values in an 8.25 KiB bank-padded shared array, block sums are scanned
/// recursively, and offsets are added from the deepest level back to level 0.
pub(crate) struct GpuExclusiveScanU32 {
    length: usize,
    levels: Vec<ScanLevel>,
    block_sums: Vec<wgpu::Buffer>,
    recursive_outputs: Vec<wgpu::Buffer>,
    block_pipeline: wgpu::ComputePipeline,
    block_layout: wgpu::BindGroupLayout,
    fixup_pipeline: wgpu::ComputePipeline,
    fixup_layout: wgpu::BindGroupLayout,
    max_storage_binding_bytes: u64,
}

impl GpuExclusiveScanU32 {
    pub(crate) fn new(device: &wgpu::Device, length: usize) -> Result<Self> {
        let limits = device.limits();
        let maximum_workgroup_size = limits
            .max_compute_invocations_per_workgroup
            .min(limits.max_compute_workgroup_size_x);
        if maximum_workgroup_size < WORKGROUP_SIZE {
            return Err(NufftError::GpuWorkgroupSizeUnsupported {
                requested: WORKGROUP_SIZE,
                maximum: maximum_workgroup_size,
            });
        }
        if limits.max_compute_workgroup_storage_size < WORKGROUP_STORAGE_BYTES {
            return Err(NufftError::GpuWorkgroupStorageUnsupported {
                requested_bytes: WORKGROUP_STORAGE_BYTES,
                maximum_bytes: limits.max_compute_workgroup_storage_size,
            });
        }
        let _ = u32::try_from(length).map_err(|_| NufftError::LengthOverflow {
            context: "GPU exclusive scan length",
        })?;

        let value_bytes = checked_bytes("GPU exclusive scan values", length)?;
        validate_planned_buffer_size(
            "GPU exclusive scan values",
            value_bytes,
            limits.max_storage_buffer_binding_size,
            limits.max_buffer_size,
        )?;

        let block_pipeline =
            create_compute_pipeline(device, "wgpu_nufft.scan.block", &generate_block_scan_wgsl());
        let block_layout = block_pipeline.get_bind_group_layout(0);
        let fixup_pipeline =
            create_compute_pipeline(device, "wgpu_nufft.scan.fixup", &generate_fixup_wgsl());
        let fixup_layout = fixup_pipeline.get_bind_group_layout(0);

        let mut levels = Vec::new();
        if length > 0 {
            let mut level_length = length;
            loop {
                let block_count = level_length.div_ceil(BLOCK_ELEMENTS);
                let block_workgroups =
                    u32::try_from(block_count).map_err(|_| NufftError::LengthOverflow {
                        context: "GPU exclusive scan block count",
                    })?;
                let fixup_workgroups = u32::try_from(
                    level_length.div_ceil(WORKGROUP_SIZE as usize),
                )
                .map_err(|_| NufftError::LengthOverflow {
                    context: "GPU exclusive scan fixup workgroup count",
                })?;
                levels.push(ScanLevel {
                    length: level_length,
                    block_count,
                    block_dispatch: split_workgroups(
                        block_workgroups,
                        limits.max_compute_workgroups_per_dimension,
                    )?,
                    fixup_dispatch: split_workgroups(
                        fixup_workgroups,
                        limits.max_compute_workgroups_per_dimension,
                    )?,
                });
                if block_count == 1 {
                    break;
                }
                level_length = block_count;
            }
        }

        let mut block_sums = Vec::with_capacity(levels.len());
        for (index, level) in levels.iter().enumerate() {
            let bytes = checked_bytes("GPU exclusive scan block sums", level.block_count)?;
            validate_planned_buffer_size(
                "GPU exclusive scan block sums",
                bytes,
                limits.max_storage_buffer_binding_size,
                limits.max_buffer_size,
            )?;
            let label = format!("wgpu_nufft.scan.level_{index}.block_sums");
            block_sums.push(device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(&label),
                size: bytes,
                usage: wgpu::BufferUsages::STORAGE,
                mapped_at_creation: false,
            }));
        }

        let mut recursive_outputs = Vec::with_capacity(levels.len().saturating_sub(1));
        for (index, level) in levels.iter().enumerate().skip(1) {
            let bytes = checked_bytes("GPU exclusive scan recursive output", level.length)?;
            validate_planned_buffer_size(
                "GPU exclusive scan recursive output",
                bytes,
                limits.max_storage_buffer_binding_size,
                limits.max_buffer_size,
            )?;
            let label = format!("wgpu_nufft.scan.level_{index}.output");
            recursive_outputs.push(device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(&label),
                size: bytes,
                usage: wgpu::BufferUsages::STORAGE,
                mapped_at_creation: false,
            }));
        }

        Ok(Self {
            length,
            levels,
            block_sums,
            recursive_outputs,
            block_pipeline,
            block_layout,
            fixup_pipeline,
            fixup_layout,
            max_storage_binding_bytes: limits.max_storage_buffer_binding_size,
        })
    }

    /// Encodes `output[i] = sum(input[0..i])` for the plan's fixed length.
    pub(crate) fn encode(
        &self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        input: &wgpu::Buffer,
        output: &wgpu::Buffer,
    ) -> Result<()> {
        if self.length == 0 {
            return Ok(());
        }
        let value_bytes = checked_bytes("GPU exclusive scan values", self.length)?;
        validate_storage_buffer(
            "exclusive scan input",
            input,
            value_bytes,
            self.max_storage_binding_bytes,
        )?;
        validate_storage_buffer(
            "exclusive scan output",
            output,
            value_bytes,
            self.max_storage_binding_bytes,
        )?;
        if input == output {
            return Err(NufftError::GpuBufferAliasUnsupported {
                first: "exclusive scan input",
                second: "exclusive scan output",
            });
        }

        for (level_index, level) in self.levels.iter().enumerate() {
            let level_input = if level_index == 0 {
                input
            } else {
                &self.block_sums[level_index - 1]
            };
            let level_output = if level_index == 0 {
                output
            } else {
                &self.recursive_outputs[level_index - 1]
            };
            let input_bytes = checked_bytes("GPU exclusive scan level input", level.length)?;
            let block_sum_bytes =
                checked_bytes("GPU exclusive scan level block sums", level.block_count)?;
            let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("wgpu_nufft.scan.block.bind_group"),
                layout: &self.block_layout,
                entries: &[
                    binding_entry(0, level_input, input_bytes),
                    binding_entry(1, level_output, input_bytes),
                    binding_entry(2, &self.block_sums[level_index], block_sum_bytes),
                ],
            });
            encode_pass(
                encoder,
                "wgpu_nufft.scan.block.pass",
                &self.block_pipeline,
                &bind_group,
                level.block_dispatch,
            );
        }

        for level_index in (0..self.levels.len().saturating_sub(1)).rev() {
            let level = &self.levels[level_index];
            let values = if level_index == 0 {
                output
            } else {
                &self.recursive_outputs[level_index - 1]
            };
            let offsets = &self.recursive_outputs[level_index];
            let value_bytes = checked_bytes("GPU exclusive scan fixup values", level.length)?;
            let offset_bytes =
                checked_bytes("GPU exclusive scan fixup offsets", level.block_count)?;
            let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("wgpu_nufft.scan.fixup.bind_group"),
                layout: &self.fixup_layout,
                entries: &[
                    binding_entry(0, values, value_bytes),
                    binding_entry(1, offsets, offset_bytes),
                ],
            });
            encode_pass(
                encoder,
                "wgpu_nufft.scan.fixup.pass",
                &self.fixup_pipeline,
                &bind_group,
                level.fixup_dispatch,
            );
        }
        Ok(())
    }

    #[cfg(test)]
    fn level_lengths(&self) -> Vec<usize> {
        self.levels.iter().map(|level| level.length).collect()
    }
}

fn create_compute_pipeline(
    device: &wgpu::Device,
    label: &str,
    source: &str,
) -> wgpu::ComputePipeline {
    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some(label),
        source: wgpu::ShaderSource::Wgsl(source.into()),
    });
    device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some(label),
        layout: None,
        module: &shader,
        entry_point: Some("main"),
        compilation_options: wgpu::PipelineCompilationOptions::default(),
        cache: None,
    })
}

fn encode_pass(
    encoder: &mut wgpu::CommandEncoder,
    label: &str,
    pipeline: &wgpu::ComputePipeline,
    bind_group: &wgpu::BindGroup,
    dispatch: (u32, u32, u32),
) {
    let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
        label: Some(label),
        timestamp_writes: None,
    });
    pass.set_pipeline(pipeline);
    pass.set_bind_group(0, bind_group, &[]);
    pass.dispatch_workgroups(dispatch.0, dispatch.1, dispatch.2);
}

fn binding_entry<'a>(
    binding: u32,
    buffer: &'a wgpu::Buffer,
    size: u64,
) -> wgpu::BindGroupEntry<'a> {
    wgpu::BindGroupEntry {
        binding,
        resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
            buffer,
            offset: 0,
            size: NonZeroU64::new(size),
        }),
    }
}

fn checked_bytes(context: &'static str, elements: usize) -> Result<u64> {
    u64::try_from(elements)
        .ok()
        .and_then(|count| count.checked_mul(U32_BYTES))
        .ok_or(NufftError::LengthOverflow { context })
}

fn validate_planned_buffer_size(
    buffer: &'static str,
    bytes: u64,
    binding_limit: u64,
    buffer_limit: u64,
) -> Result<()> {
    let limit = binding_limit.min(buffer_limit);
    if bytes > limit {
        Err(NufftError::GpuBufferBindingTooLarge {
            buffer,
            required_bytes: bytes,
            limit_bytes: limit,
        })
    } else {
        Ok(())
    }
}

fn validate_storage_buffer(
    label: &'static str,
    buffer: &wgpu::Buffer,
    required_bytes: u64,
    binding_limit: u64,
) -> Result<()> {
    if !buffer.usage().contains(wgpu::BufferUsages::STORAGE) {
        return Err(NufftError::GpuBufferMissingUsage {
            buffer: label,
            required_usage: "STORAGE",
        });
    }
    if buffer.size() < required_bytes {
        return Err(NufftError::GpuBufferTooSmall {
            buffer: label,
            required_bytes,
            actual_bytes: buffer.size(),
        });
    }
    if required_bytes > binding_limit {
        return Err(NufftError::GpuBufferBindingTooLarge {
            buffer: label,
            required_bytes,
            limit_bytes: binding_limit,
        });
    }
    Ok(())
}

fn generate_block_scan_wgsl() -> String {
    format!(
        r#"const WORKGROUP_SIZE: u32 = {WORKGROUP_SIZE}u;
const ITEMS_PER_INVOCATION: u32 = {ITEMS_PER_INVOCATION}u;
const BLOCK_ELEMENTS: u32 = {BLOCK_ELEMENTS}u;

@group(0) @binding(0) var<storage, read> input_values: array<u32>;
@group(0) @binding(1) var<storage, read_write> output_values: array<u32>;
@group(0) @binding(2) var<storage, read_write> block_sums: array<u32>;
var<workgroup> scratch: array<u32, {PADDED_ELEMENTS}>;

fn physical_index(logical: u32) -> u32 {{
    return logical + (logical >> 5u);
}}

@compute @workgroup_size({WORKGROUP_SIZE})
fn main(
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(num_workgroups) nwg: vec3<u32>,
) {{
    let wg_flat = (wid.z * nwg.y + wid.y) * nwg.x + wid.x;
    let block_count = arrayLength(&block_sums);
    if (wg_flat >= block_count) {{ return; }}

    let length = arrayLength(&input_values);
    let block_base = wg_flat * BLOCK_ELEMENTS;
    let remaining = length - block_base;
    for (var item = 0u; item < ITEMS_PER_INVOCATION; item = item + 1u) {{
        let logical = item * WORKGROUP_SIZE + lid.x;
        var value = 0u;
        if (logical < remaining) {{
            value = input_values[block_base + logical];
        }}
        scratch[physical_index(logical)] = value;
    }}
    workgroupBarrier();

    var stride = 1u;
    loop {{
        let node_count = BLOCK_ELEMENTS / (2u * stride);
        for (var node = lid.x; node < node_count; node = node + WORKGROUP_SIZE) {{
            let left = stride * (2u * node + 1u) - 1u;
            let right = stride * (2u * node + 2u) - 1u;
            scratch[physical_index(right)] =
                scratch[physical_index(right)] + scratch[physical_index(left)];
        }}
        workgroupBarrier();
        if (stride == BLOCK_ELEMENTS / 2u) {{ break; }}
        stride = stride * 2u;
    }}

    if (lid.x == 0u) {{
        block_sums[wg_flat] = scratch[physical_index(BLOCK_ELEMENTS - 1u)];
        scratch[physical_index(BLOCK_ELEMENTS - 1u)] = 0u;
    }}
    workgroupBarrier();

    stride = BLOCK_ELEMENTS / 2u;
    loop {{
        let node_count = BLOCK_ELEMENTS / (2u * stride);
        for (var node = lid.x; node < node_count; node = node + WORKGROUP_SIZE) {{
            let left = stride * (2u * node + 1u) - 1u;
            let right = stride * (2u * node + 2u) - 1u;
            let temporary = scratch[physical_index(left)];
            scratch[physical_index(left)] = scratch[physical_index(right)];
            scratch[physical_index(right)] =
                scratch[physical_index(right)] + temporary;
        }}
        workgroupBarrier();
        if (stride == 1u) {{ break; }}
        stride = stride / 2u;
    }}

    for (var item = 0u; item < ITEMS_PER_INVOCATION; item = item + 1u) {{
        let logical = item * WORKGROUP_SIZE + lid.x;
        if (logical < remaining) {{
            output_values[block_base + logical] = scratch[physical_index(logical)];
        }}
    }}
}}
"#,
    )
}

fn generate_fixup_wgsl() -> String {
    format!(
        r#"const WORKGROUP_SIZE: u32 = {WORKGROUP_SIZE}u;
const BLOCK_ELEMENTS: u32 = {BLOCK_ELEMENTS}u;

@group(0) @binding(0) var<storage, read_write> values: array<u32>;
@group(0) @binding(1) var<storage, read> block_offsets: array<u32>;

@compute @workgroup_size({WORKGROUP_SIZE})
fn main(
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(num_workgroups) nwg: vec3<u32>,
) {{
    let total = arrayLength(&values);
    if (total == 0u) {{ return; }}
    let wg_flat = (wid.z * nwg.y + wid.y) * nwg.x + wid.x;
    let last_workgroup = (total - 1u) / WORKGROUP_SIZE;
    if (wg_flat > last_workgroup) {{ return; }}
    let workgroup_base = wg_flat * WORKGROUP_SIZE;
    if (lid.x >= total - workgroup_base) {{ return; }}
    let index = workgroup_base + lid.x;
    values[index] = values[index] + block_offsets[index / BLOCK_ELEMENTS];
}}
"#,
    )
}

#[cfg(test)]
#[cfg_attr(target_arch = "wasm32", allow(dead_code))]
mod tests {
    use std::sync::mpsc;
    use std::time::Duration;

    use wgpu::util::DeviceExt;

    use super::*;

    #[test]
    fn hierarchy_matches_large_scan_geometry() {
        assert_eq!(level_lengths_for(0), Vec::<usize>::new());
        assert_eq!(level_lengths_for(1), vec![1]);
        assert_eq!(level_lengths_for(1 << 21), vec![1 << 21, 1024]);
        assert_eq!(level_lengths_for(1 << 27), vec![1 << 27, 65_536, 32]);
    }

    #[test]
    fn block_shader_keeps_tail_guards_inside_uniform_barrier_flow() {
        let source = generate_block_scan_wgsl();
        assert!(source.contains("var<workgroup> scratch: array<u32, 2112>;"));
        assert!(source.contains("if (wg_flat >= block_count) { return; }"));
        assert!(source.contains("if (logical < remaining)"));
        assert!(!source.contains("select("));
        assert!(source.matches("workgroupBarrier();").count() >= 4);
        assert_eq!(physical_index(BLOCK_ELEMENTS - 1), 2110);
    }

    #[test]
    fn cpu_reference_uses_wrapping_u32_semantics() {
        assert_eq!(
            exclusive_reference(&[u32::MAX, 1, 2, u32::MAX]),
            [0, u32::MAX, 0, 2]
        );
    }

    #[test]
    #[cfg(not(target_arch = "wasm32"))]
    fn gpu_exclusive_scan_matches_cpu_bit_exactly() {
        if std::env::var_os("WGPU_FFT_RUN_GPU_TESTS").is_none() {
            eprintln!("skipping GPU scan test; set WGPU_FFT_RUN_GPU_TESTS=1 to run it");
            return;
        }
        pollster::block_on(run_gpu_scan_cases());
    }

    fn level_lengths_for(length: usize) -> Vec<usize> {
        if length == 0 {
            return Vec::new();
        }
        let mut lengths = Vec::new();
        let mut current = length;
        loop {
            lengths.push(current);
            let blocks = current.div_ceil(BLOCK_ELEMENTS);
            if blocks == 1 {
                break;
            }
            current = blocks;
        }
        lengths
    }

    fn physical_index(logical: usize) -> usize {
        logical + logical / BANK_WIDTH
    }

    fn exclusive_reference(input: &[u32]) -> Vec<u32> {
        let mut running = 0u32;
        input
            .iter()
            .map(|&value| {
                let prefix = running;
                running = running.wrapping_add(value);
                prefix
            })
            .collect()
    }

    async fn run_gpu_scan_cases() {
        let Some(context) = wgpu_fft::device::request_default_device().await else {
            panic!("WGPU_FFT_RUN_GPU_TESTS was set but no suitable adapter was found");
        };
        let info = context.adapter.get_info();
        eprintln!(
            "gpu_scan adapter: {} backend={:?} driver={} {}",
            info.name, info.backend, info.driver, info.driver_info
        );

        let required_lengths = [
            0usize,
            1,
            255,
            256,
            257,
            65_535,
            1 << 21,
            (1 << 21) + 13,
            BLOCK_ELEMENTS - 1,
            BLOCK_ELEMENTS,
            BLOCK_ELEMENTS + 1,
            BLOCK_ELEMENTS * BLOCK_ELEMENTS + 13,
        ];
        for (case_index, length) in required_lengths.into_iter().enumerate() {
            let input = random_values(length, 0x51A7_0000u32 ^ case_index as u32);
            assert_gpu_case(
                &context.device,
                &context.queue,
                &input,
                &format!("length-{length}"),
            );
        }
        assert_gpu_case(
            &context.device,
            &context.queue,
            &vec![0; BLOCK_ELEMENTS + 17],
            "all-zero",
        );
        let large = (0..4097)
            .map(|index| match index % 4 {
                0 => u32::MAX,
                1 => 1,
                2 => u32::MAX - 7,
                _ => 19,
            })
            .collect::<Vec<_>>();
        assert_gpu_case(
            &context.device,
            &context.queue,
            &large,
            "wrapping-large-counts",
        );
        for (case_index, length) in [3usize, 17, 513, 4093, 12_345].into_iter().enumerate() {
            let input = random_values(length, 0xC001_D00Du32 ^ case_index as u32);
            assert_gpu_case(
                &context.device,
                &context.queue,
                &input,
                &format!("property-{case_index}-length-{length}"),
            );
        }

        #[cfg(windows)]
        std::mem::forget(context);
    }

    fn random_values(length: usize, mut state: u32) -> Vec<u32> {
        (0..length)
            .map(|_| {
                state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                state
            })
            .collect()
    }

    fn assert_gpu_case(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        input_values: &[u32],
        label: &str,
    ) {
        let length = input_values.len();
        let input_storage = if input_values.is_empty() {
            vec![0xA5A5_A5A5]
        } else {
            input_values.to_vec()
        };
        let output_words = length.max(1);
        let output_initial = vec![0xDEAD_BEEFu32; output_words];
        let input = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("wgpu_nufft.scan.test.input"),
            contents: bytemuck::cast_slice(&input_storage),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let output = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("wgpu_nufft.scan.test.output"),
            contents: bytemuck::cast_slice(&output_initial),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        });
        let output_bytes = (output_words * size_of::<u32>()) as u64;
        let readback = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("wgpu_nufft.scan.test.readback"),
            size: output_bytes,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let scan = GpuExclusiveScanU32::new(device, length).unwrap();
        assert_eq!(scan.level_lengths(), level_lengths_for(length));
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("wgpu_nufft.scan.test.encoder"),
        });
        scan.encode(device, &mut encoder, &input, &output).unwrap();
        encoder.copy_buffer_to_buffer(&output, 0, &readback, 0, output_bytes);
        let submission = queue.submit([encoder.finish()]);
        let slice = readback.slice(..);
        let (sender, receiver) = mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |result| {
            sender.send(result).unwrap();
        });
        device
            .poll(wgpu::PollType::Wait {
                submission_index: Some(submission),
                timeout: Some(Duration::from_secs(120)),
            })
            .unwrap();
        receiver.recv().unwrap().unwrap();
        let mapped = slice.get_mapped_range();
        let actual = bytemuck::cast_slice::<u8, u32>(&mapped).to_vec();
        drop(mapped);
        readback.unmap();

        if length == 0 {
            assert_eq!(
                actual, output_initial,
                "{label}: zero-length scan must be a no-op"
            );
        } else {
            let expected = exclusive_reference(input_values);
            assert_eq!(actual, expected, "{label}: exact exclusive scan mismatch");
        }
        eprintln!("GPU_SCAN_EXACT label={label} length={length} ok");
    }
}
