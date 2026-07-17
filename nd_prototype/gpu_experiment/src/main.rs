//! Stage 3 of the ND NUFFT prototype: rank-generic WGSL spreading experiment.
//!
//! Question under test: does the S1 gather formulation (one workgroup per
//! output tile, one register accumulator per owned cell, shared-memory point
//! batches, weights = product of per-dim ES kernel evals, periodic wrap) work
//! as *generated* WGSL for arbitrary dimension, and what throughput does it
//! reach at d=4 on real hardware?
//!
//! The kernel is generated for any d; dimension only appears as unrolled code.
//! Binning happens on the host (points duplicated into every tile whose halo
//! their support touches — a scratch simplification; production would read
//! neighbor bins instead of duplicating).
//!
//! Outputs (out/ directory): grid.bin (f32 interleaved complex), inputs as
//! x.bin/c.bin (f64), and meta.json — consumed by check.py which recomputes
//! the reference spread grid in f64 numpy and compares.

use std::fmt::Write as _;
use std::fs;
use std::path::Path;
use std::time::Instant;

// ---------------------------------------------------------------------------
// ES kernel parameters (FINUFFT rules, same as nd_nufft.py)
// ---------------------------------------------------------------------------

fn es_params(eps: f64, sigma: f64) -> (usize, f64) {
    if (sigma - 2.0).abs() < 1e-12 {
        let w = (10.0 / eps).log10().ceil() as usize;
        let w = w.max(2);
        let bow = match w {
            2 => 2.20,
            3 => 2.26,
            4 => 2.38,
            _ => 2.30,
        };
        (w, bow * w as f64)
    } else {
        let w = ((1.0 / eps).ln() / (std::f64::consts::PI * (1.0 - 1.0 / sigma).sqrt())).ceil()
            as usize;
        let w = w.clamp(2, 16);
        let beta = 0.97 * std::f64::consts::PI * w as f64 * (1.0 - 1.0 / (2.0 * sigma));
        (w, beta)
    }
}

#[allow(dead_code)] // host-side reference eval, used by the upcoming end-to-end step
fn es_phi(z: f64, beta: f64) -> f64 {
    if z.abs() <= 1.0 {
        (beta * ((1.0 - z * z).sqrt() - 1.0)).exp()
    } else {
        0.0
    }
}

// ---------------------------------------------------------------------------
// WGSL generation (rank-generic)
// ---------------------------------------------------------------------------

struct KernelCfg {
    d: usize,
    fine: Vec<usize>,
    tile: Vec<usize>,
    w: usize,
    beta: f64,
    batch: usize,
    threads: usize,
}

impl KernelCfg {
    fn tiles_per_dim(&self) -> Vec<usize> {
        self.fine
            .iter()
            .zip(&self.tile)
            .map(|(f, t)| f.div_ceil(*t))
            .collect()
    }
    fn tile_vol(&self) -> usize {
        self.tile.iter().product()
    }
    fn cells_per_thread(&self) -> usize {
        self.tile_vol().div_ceil(self.threads)
    }
}

fn generate_wgsl(cfg: &KernelCfg) -> String {
    let d = cfg.d;
    let tpd = cfg.tiles_per_dim();
    let cpt = cfg.cells_per_thread();
    let mut s = String::new();

    let _ = writeln!(
        s,
        "// generated rank-generic spreading kernel: d={d}, w={}, tile={:?}, fine={:?}",
        cfg.w, cfg.tile, cfg.fine
    );
    s.push_str(
        "@group(0) @binding(0) var<storage, read> pts: array<f32>;\n\
         @group(0) @binding(1) var<storage, read> strengths: array<vec2<f32>>;\n\
         @group(0) @binding(2) var<storage, read> tile_offsets: array<u32>;\n\
         @group(0) @binding(3) var<storage, read> point_ids: array<u32>;\n\
         @group(0) @binding(4) var<storage, read_write> grid: array<vec2<f32>>;\n\n",
    );
    let _ = writeln!(s, "const BETA: f32 = {:.9};", cfg.beta);
    let _ = writeln!(s, "const INV_HALF_W: f32 = {:.9};", 2.0 / cfg.w as f64);
    let _ = writeln!(s, "const BATCH: u32 = {}u;", cfg.batch);
    let _ = writeln!(s, "const THREADS: u32 = {}u;", cfg.threads);
    let _ = writeln!(s, "const CPT: u32 = {}u;", cpt);
    for i in 0..d {
        let _ = writeln!(s, "const FINE{i}: u32 = {}u;", cfg.fine[i]);
        let _ = writeln!(s, "const TILE{i}: u32 = {}u;", cfg.tile[i]);
        let _ = writeln!(s, "const TPD{i}: u32 = {}u;", tpd[i]);
    }
    let _ = writeln!(s, "const TILE_VOL: u32 = {}u;", cfg.tile_vol());
    s.push_str(&format!(
        "\nvar<workgroup> s_pts: array<f32, {}>;\n\
         var<workgroup> s_str: array<vec2<f32>, {}>;\n\n",
        cfg.batch * d,
        cfg.batch
    ));
    s.push_str(
        "fn phi(dm: f32) -> f32 {\n\
         \x20 let z = dm * INV_HALF_W;\n\
         \x20 if (abs(z) > 1.0) { return 0.0; }\n\
         \x20 return exp(BETA * (sqrt(1.0 - z * z) - 1.0));\n}\n\n",
    );

    s.push_str("@compute @workgroup_size(THREADS, 1, 1)\n");
    s.push_str(
        "fn main(@builtin(local_invocation_id) lid: vec3<u32>, \
         @builtin(workgroup_id) wid: vec3<u32>, \
         @builtin(num_workgroups) nwg: vec3<u32>) {\n",
    );
    s.push_str("  let tile_id: u32 = (wid.z * nwg.y + wid.y) * nwg.x + wid.x;\n");
    // tile coordinate decomposition (row-major, dim 0 slowest)
    s.push_str("  var rem: u32 = tile_id;\n");
    for i in (0..d).rev() {
        let _ = writeln!(s, "  let t{i}: u32 = rem % TPD{i}; rem = rem / TPD{i};");
    }
    // per-owned-cell global coordinates + in-bounds mask, precomputed
    let _ = writeln!(s, "  var acc: array<vec2<f32>, CPT>;");
    let _ = writeln!(s, "  var cell_ok: array<bool, CPT>;");
    for i in 0..d {
        let _ = writeln!(s, "  var cg{i}: array<f32, CPT>;");
    }
    s.push_str("  for (var slot: u32 = 0u; slot < CPT; slot = slot + 1u) {\n");
    s.push_str("    let local: u32 = lid.x + slot * THREADS;\n");
    s.push_str("    var ok: bool = local < TILE_VOL;\n");
    s.push_str("    var lrem: u32 = min(local, TILE_VOL - 1u);\n");
    for i in (0..d).rev() {
        let _ = writeln!(
            s,
            "    let l{i}: u32 = lrem % TILE{i}; lrem = lrem / TILE{i};\n    \
             let g{i}: u32 = t{i} * TILE{i} + l{i};\n    \
             ok = ok && (g{i} < FINE{i});\n    \
             cg{i}[slot] = f32(g{i});"
        );
    }
    s.push_str("    cell_ok[slot] = ok;\n    acc[slot] = vec2<f32>(0.0, 0.0);\n  }\n");

    // point range for this tile (uniform per workgroup)
    s.push_str(
        "  let p_begin: u32 = tile_offsets[tile_id];\n  \
         let p_end: u32 = tile_offsets[tile_id + 1u];\n",
    );
    s.push_str("  var base: u32 = p_begin;\n");
    s.push_str("  loop {\n    if (base >= p_end) { break; }\n");
    s.push_str("    let count: u32 = min(BATCH, p_end - base);\n");
    // cooperative load
    s.push_str("    for (var i: u32 = lid.x; i < count; i = i + THREADS) {\n");
    s.push_str("      let pid: u32 = point_ids[base + i];\n");
    for i in 0..d {
        let _ = writeln!(
            s,
            "      s_pts[i * {d}u + {i}u] = pts[pid * {d}u + {i}u];"
        );
    }
    s.push_str("      s_str[i] = strengths[pid];\n    }\n");
    s.push_str("    workgroupBarrier();\n");
    // accumulate
    s.push_str("    for (var j: u32 = 0u; j < count; j = j + 1u) {\n");
    s.push_str("      let cj: vec2<f32> = s_str[j];\n");
    for i in 0..d {
        let _ = writeln!(s, "      let p{i}: f32 = s_pts[j * {d}u + {i}u];");
    }
    s.push_str("      for (var slot: u32 = 0u; slot < CPT; slot = slot + 1u) {\n");
    s.push_str("        if (!cell_ok[slot]) { continue; }\n");
    s.push_str("        var wgt: f32 = 1.0;\n");
    for i in 0..d {
        let _ = writeln!(
            s,
            "        var dm{i}: f32 = cg{i}[slot] - p{i};\n        \
             dm{i} = dm{i} - round(dm{i} / f32(FINE{i})) * f32(FINE{i});\n        \
             wgt = wgt * phi(dm{i});"
        );
    }
    s.push_str("        acc[slot] = acc[slot] + cj * wgt;\n      }\n    }\n");
    s.push_str("    workgroupBarrier();\n    base = base + count;\n  }\n");
    // write out
    s.push_str("  for (var slot: u32 = 0u; slot < CPT; slot = slot + 1u) {\n");
    s.push_str("    if (!cell_ok[slot]) { continue; }\n");
    s.push_str("    var idx: u32 = 0u;\n");
    s.push_str("    let local2: u32 = lid.x + slot * THREADS;\n");
    s.push_str("    var lrem2: u32 = local2;\n");
    for i in (0..d).rev() {
        let _ = writeln!(s, "    let m{i}: u32 = lrem2 % TILE{i}; lrem2 = lrem2 / TILE{i};");
    }
    for i in 0..d {
        let _ = writeln!(s, "    idx = idx * FINE{i} + (t{i} * TILE{i} + m{i});");
    }
    s.push_str("    grid[idx] = acc[slot];\n  }\n}\n");
    s
}

// ---------------------------------------------------------------------------
// Host binning: duplicate each point into every tile its support touches
// ---------------------------------------------------------------------------

fn bin_points(x: &[f64], m: usize, cfg: &KernelCfg) -> (Vec<u32>, Vec<u32>) {
    let d = cfg.d;
    let tpd = cfg.tiles_per_dim();
    let n_tiles: usize = tpd.iter().product();
    let half_w = cfg.w as f64 / 2.0;
    let mut lists: Vec<Vec<u32>> = vec![Vec::new(); n_tiles];
    let mut tiles_touched: Vec<Vec<usize>> = vec![Vec::new(); d];
    for j in 0..m {
        for tt in tiles_touched.iter_mut() {
            tt.clear();
        }
        for i in 0..d {
            let fine = cfg.fine[i] as f64;
            let p = (x[j * d + i].rem_euclid(1.0)) * fine;
            let lo = (p - half_w).floor() as i64;
            let hi = (p + half_w).ceil() as i64;
            let t_lo = lo.div_euclid(cfg.tile[i] as i64);
            let t_hi = hi.div_euclid(cfg.tile[i] as i64);
            let mut seen = Vec::new();
            for t in t_lo..=t_hi {
                let tw = t.rem_euclid(tpd[i] as i64) as usize;
                if !seen.contains(&tw) {
                    seen.push(tw);
                }
            }
            tiles_touched[i] = seen;
        }
        // cartesian product of touched tile indices
        let mut combos: Vec<usize> = vec![0];
        for i in 0..d {
            let mut next = Vec::with_capacity(combos.len() * tiles_touched[i].len());
            for &c in &combos {
                for &t in &tiles_touched[i] {
                    next.push(c * tpd[i] + t);
                }
            }
            combos = next;
        }
        for tile in combos {
            lists[tile].push(j as u32);
        }
    }
    let mut offsets = Vec::with_capacity(n_tiles + 1);
    let mut ids = Vec::new();
    offsets.push(0u32);
    for l in &lists {
        ids.extend_from_slice(l);
        offsets.push(ids.len() as u32);
    }
    (offsets, ids)
}

// ---------------------------------------------------------------------------
// LCG data generation (deterministic, matches nothing — inputs are exported)
// ---------------------------------------------------------------------------

struct Lcg(u64);
impl Lcg {
    fn next_f64(&mut self) -> f64 {
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        ((self.0 >> 11) as f64) / ((1u64 << 53) as f64)
    }
}

// ---------------------------------------------------------------------------
// GPU runner
// ---------------------------------------------------------------------------

struct Gpu {
    device: wgpu::Device,
    queue: wgpu::Queue,
    adapter_name: String,
}

fn init_gpu() -> Gpu {
    pollster::block_on(async {
        let mut idesc = wgpu::InstanceDescriptor::new_without_display_handle();
        idesc.backends = wgpu::Backends::VULKAN | wgpu::Backends::DX12 | wgpu::Backends::METAL;
        let instance = wgpu::Instance::new(idesc.with_env());
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                ..Default::default()
            })
            .await
            .expect("no adapter");
        let name = adapter.get_info().name.clone();
        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                required_limits: adapter.limits(),
                ..Default::default()
            })
            .await
            .expect("no device");
        Gpu {
            device,
            queue,
            adapter_name: name,
        }
    })
}

#[allow(clippy::too_many_arguments)]
fn run_case(
    gpu: &Gpu,
    label: &str,
    n_modes: &[usize],
    sigma: f64,
    eps: f64,
    tile: &[usize],
    m_points: usize,
    export_dir: Option<&Path>,
    timed_runs: usize,
) -> f64 {
    let d = n_modes.len();
    let (w, beta) = es_params(eps, sigma);
    let fine: Vec<usize> = n_modes
        .iter()
        .map(|n| ((sigma * *n as f64 / 2.0).ceil() as usize) * 2)
        .collect();
    let cfg = KernelCfg {
        d,
        fine: fine.clone(),
        tile: tile.to_vec(),
        w,
        beta,
        batch: 64,
        threads: 256,
    };
    let grid_len: usize = fine.iter().product();

    let mut lcg = Lcg(0x5EED_1234_ABCD_0001);
    let x: Vec<f64> = (0..m_points * d).map(|_| lcg.next_f64()).collect();
    let c: Vec<f64> = (0..m_points * 2).map(|_| lcg.next_f64() * 2.0 - 1.0).collect();

    let (offsets, ids) = bin_points(&x, m_points, &cfg);
    let dup = ids.len() as f64 / m_points as f64;

    let x_ref = &x;
    let fine_ref = &fine;
    let pts_f32: Vec<f32> = (0..m_points)
        .flat_map(|j| {
            (0..d).map(move |i| ((x_ref[j * d + i].rem_euclid(1.0)) * fine_ref[i] as f64) as f32)
        })
        .collect();
    let str_f32: Vec<f32> = c.iter().map(|&v| v as f32).collect();

    let wgsl = generate_wgsl(&cfg);
    let module = gpu
        .device
        .create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("nd_spread"),
            source: wgpu::ShaderSource::Wgsl(wgsl.clone().into()),
        });
    let pipeline = gpu
        .device
        .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("nd_spread"),
            layout: None,
            module: &module,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });

    let mk_storage = |data: &[u8], usage: wgpu::BufferUsages| {
        use wgpu::util::DeviceExt;
        gpu.device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: None,
                contents: data,
                usage,
            })
    };
    use wgpu::BufferUsages as BU;
    let b_pts = mk_storage(bytemuck::cast_slice(&pts_f32), BU::STORAGE);
    let b_str = mk_storage(bytemuck::cast_slice(&str_f32), BU::STORAGE);
    let b_off = mk_storage(bytemuck::cast_slice(&offsets), BU::STORAGE);
    let b_ids = mk_storage(bytemuck::cast_slice(&ids), BU::STORAGE);
    let b_grid = gpu.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("grid"),
        size: (grid_len * 8) as u64,
        usage: BU::STORAGE | BU::COPY_SRC | BU::COPY_DST,
        mapped_at_creation: false,
    });
    let bind = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None,
        layout: &pipeline.get_bind_group_layout(0),
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: b_pts.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 1, resource: b_str.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 2, resource: b_off.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 3, resource: b_ids.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 4, resource: b_grid.as_entire_binding() },
        ],
    });

    let n_tiles: usize = cfg.tiles_per_dim().iter().product();
    let max_dim = gpu.device.limits().max_compute_workgroups_per_dimension as usize;
    let gx = n_tiles.min(max_dim);
    let gy = n_tiles.div_ceil(gx);
    assert!(gy <= max_dim, "grid too large: {n_tiles} tiles");

    let dispatch = |gpu: &Gpu| {
        let mut enc = gpu
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        enc.clear_buffer(&b_grid, 0, None);
        {
            let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: None,
                timestamp_writes: None,
            });
            pass.set_pipeline(&pipeline);
            pass.set_bind_group(0, &bind, &[]);
            pass.dispatch_workgroups(gx as u32, gy as u32, 1);
        }
        gpu.queue.submit([enc.finish()]);
        gpu.device
            .poll(wgpu::PollType::wait_indefinitely())
            .expect("poll");
    };

    // warm-up + timed
    dispatch(gpu);
    dispatch(gpu);
    let mut times = Vec::new();
    for _ in 0..timed_runs {
        let t0 = Instant::now();
        dispatch(gpu);
        times.push(t0.elapsed().as_secs_f64());
    }
    times.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let t_med = times[times.len() / 2];

    let updates = m_points as f64 * (w as f64).powi(d as i32);
    println!(
        "{label}: d={d} fine={fine:?} w={w} tiles={n_tiles} dup={dup:.1}x  \
         median {:.3} ms  ({:.3e} upd/s, {:.1} GiB grid)",
        t_med * 1e3,
        updates / t_med,
        (grid_len * 8) as f64 / (1u64 << 30) as f64,
    );

    if let Some(dir) = export_dir {
        fs::create_dir_all(dir).unwrap();
        // read back grid
        let staging = gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: (grid_len * 8) as u64,
            usage: BU::COPY_DST | BU::MAP_READ,
            mapped_at_creation: false,
        });
        let mut enc = gpu
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        enc.copy_buffer_to_buffer(&b_grid, 0, &staging, 0, (grid_len * 8) as u64);
        gpu.queue.submit([enc.finish()]);
        let slice = staging.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |r| tx.send(r).unwrap());
        gpu.device
            .poll(wgpu::PollType::wait_indefinitely())
            .expect("poll");
        rx.recv().unwrap().unwrap();
        let grid_bytes = slice.get_mapped_range().to_vec();
        staging.unmap();

        fs::write(dir.join("grid.bin"), &grid_bytes).unwrap();
        fs::write(dir.join("x.bin"), bytemuck::cast_slice::<f64, u8>(&x)).unwrap();
        fs::write(dir.join("c.bin"), bytemuck::cast_slice::<f64, u8>(&c)).unwrap();
        let meta = format!(
            "{{\"d\":{d},\"m\":{m},\"w\":{w},\"beta\":{beta},\"fine\":{fine:?},\"sigma\":{sigma}}}",
            m = m_points
        );
        fs::write(dir.join("meta.json"), meta).unwrap();
        println!("  exported correctness artifacts to {}", dir.display());
    }
    t_med
}

fn main() {
    let gpu = init_gpu();
    println!("adapter: {}", gpu.adapter_name);

    // correctness case: small d=4, exported for check.py
    run_case(
        &gpu,
        "correctness d=4",
        &[12, 12, 12, 12],
        1.25,
        1e-6,
        &[4, 4, 4, 4],
        500,
        Some(Path::new("out")),
        1,
    );

    // perf cases (tile shapes chosen to keep host-binning duplication sane)
    run_case(&gpu, "perf d=3 (256^3 s1.25 eps=1e-6)", &[256, 256, 256], 1.25, 1e-6, &[8, 8, 4], 1_000_000, None, 3);
    run_case(&gpu, "perf d=4 (48^4 s1.25 eps=1e-6)", &[48, 48, 48, 48], 1.25, 1e-6, &[8, 8, 4, 4], 500_000, None, 3);
    run_case(&gpu, "perf d=4 (64^4 s1.25 eps=1e-6)", &[64, 64, 64, 64], 1.25, 1e-6, &[8, 8, 4, 4], 500_000, None, 3);
    run_case(&gpu, "perf d=5 (24^5 s1.25 eps=1e-3)", &[24, 24, 24, 24, 24], 1.25, 1e-3, &[4, 4, 4, 3, 3], 200_000, None, 3);
}
