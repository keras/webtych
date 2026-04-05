//! GPU pipeline creation — bind-group layouts, compute pipelines, and the
//! per-frame dispatch logic.
//!
//! The six compute passes (in order) are:
//!
//! 1. **inject** — inject pressure + colour-density at event locations.
//! 2. **collide** — BGK relaxation (in-place on src buffer).
//! 3. **stream**  — propagate distributions from src to dst (ping-pong).
//! 4. **boundary** — moving bounce-back at obstacle cells.
//! 5. **extract** — compute macroscopic ρ and u from distributions.
//! 6. **advect**  — semi-Lagrangian advection of colour-density fields.

use crate::config::{CollisionMode, SimConfig};
use crate::grid::GpuGrid;
use crate::types::{LbmUniforms, ObstacleTexel};

const WG_SIZE: u32 = 8; // workgroup tile edge; dispatch is ceil(W/8) × ceil(H/8)

/// Embedded WGSL source — included at compile time so the crate is self-contained.
macro_rules! shader {
    ($name:literal) => {
        include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/shaders/", $name))
    };
}

/// All compute pipelines for one simulation instance.
pub struct GpuPipelines {
    pub inject: wgpu::ComputePipeline,
    pub collide_bgk: wgpu::ComputePipeline,
    pub collide_trt: wgpu::ComputePipeline,
    pub collide_mrt: wgpu::ComputePipeline,
    pub stream: wgpu::ComputePipeline,
    pub boundary: wgpu::ComputePipeline,
    pub extract: wgpu::ComputePipeline,
    pub advect: wgpu::ComputePipeline,

    /// Sampler used by the advect pass for bilinear colour-density sampling.
    pub linear_sampler: wgpu::Sampler,

    /// Layout shared by all compute passes (same bind-group slot assignments).
    pub bind_group_layout: wgpu::BindGroupLayout,
}

impl GpuPipelines {
    pub fn new(device: &wgpu::Device, _config: &SimConfig) -> Self {
        // ── Bind-group layout ────────────────────────────────────────────────
        //
        // Slot assignments (shared across all passes):
        //  0  — uniform buffer (LbmUniforms)
        //  1  — distributions src (r/w storage)
        //  2  — distributions dst (w storage)
        //  3  — macroscopic buffer (r/w storage)
        //  4  — obstacle texture (sampled)
        //  5  — event ring buffer (read storage)
        //  6   — packed color density buffer (r/w storage)
        //  7   — injection stamp texture (sampled)
        //
        // Not all passes use every binding, but using the same layout
        // everywhere means we only need one bind group.

        let mut entries: Vec<wgpu::BindGroupLayoutEntry> = vec![
            // 0: uniforms
            wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            // 1: distributions src
            wgpu::BindGroupLayoutEntry {
                binding: 1,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            // 2: distributions dst
            wgpu::BindGroupLayoutEntry {
                binding: 2,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            // 3: macroscopic
            wgpu::BindGroupLayoutEntry {
                binding: 3,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            // 4: obstacle texture (sampled f32)
            wgpu::BindGroupLayoutEntry {
                binding: 4,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Texture {
                    sample_type: wgpu::TextureSampleType::Float { filterable: false },
                    view_dimension: wgpu::TextureViewDimension::D2,
                    multisampled: false,
                },
                count: None,
            },
            // 5: event ring buffer (read-only)
            wgpu::BindGroupLayoutEntry {
                binding: 5,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
        ];

        // Colour density (single packed buffer, binding 6).
        entries.push(wgpu::BindGroupLayoutEntry {
            binding: 6,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Storage { read_only: false },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        });

        entries.push(wgpu::BindGroupLayoutEntry {
            binding: 7,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Texture {
                sample_type: wgpu::TextureSampleType::Float { filterable: false },
                view_dimension: wgpu::TextureViewDimension::D2,
                multisampled: false,
            },
            count: None,
        });

        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("lbm::bind_group_layout"),
            entries: &entries,
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("lbm::pipeline_layout"),
            bind_group_layouts: &[Some(&bind_group_layout)],
            immediate_size: 0,
        });

        // ── Shader modules ───────────────────────────────────────────────────
        let inject_src = shader!("lbm_inject.wgsl");
        let collide_bgk_src = shader!("lbm_collide_bgk.wgsl");
        let collide_trt_src = shader!("lbm_collide_trt.wgsl");
        let collide_mrt_src = shader!("lbm_collide_mrt.wgsl");
        let stream_src = shader!("lbm_stream.wgsl");
        let boundary_src = shader!("lbm_boundary.wgsl");
        let extract_src = shader!("lbm_extract.wgsl");
        let advect_src = shader!("lbm_advect_color.wgsl");

        let make_shader = |label: &str, src: &str| {
            device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some(label),
                source: wgpu::ShaderSource::Wgsl(src.into()),
            })
        };

        let sm_inject = make_shader("lbm_inject", inject_src);
        let sm_collide_bgk = make_shader("lbm_collide_bgk", collide_bgk_src);
        let sm_collide_trt = make_shader("lbm_collide_trt", collide_trt_src);
        let sm_collide_mrt = make_shader("lbm_collide_mrt", collide_mrt_src);
        let sm_stream = make_shader("lbm_stream", stream_src);
        let sm_boundary = make_shader("lbm_boundary", boundary_src);
        let sm_extract = make_shader("lbm_extract", extract_src);
        let sm_advect = make_shader("lbm_advect_color", advect_src);

        let make_pipeline = |label: &str, sm: &wgpu::ShaderModule| {
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(label),
                layout: Some(&pipeline_layout),
                module: sm,
                entry_point: Some("main"),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                cache: None,
            })
        };

        let inject = make_pipeline("lbm::inject", &sm_inject);
        let collide_bgk = make_pipeline("lbm::collide_bgk", &sm_collide_bgk);
        let collide_trt = make_pipeline("lbm::collide_trt", &sm_collide_trt);
        let collide_mrt = make_pipeline("lbm::collide_mrt", &sm_collide_mrt);
        let stream = make_pipeline("lbm::stream", &sm_stream);
        let boundary = make_pipeline("lbm::boundary", &sm_boundary);
        let extract = make_pipeline("lbm::extract", &sm_extract);
        let advect = make_pipeline("lbm::advect", &sm_advect);

        let linear_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("lbm::linear_sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });

        Self {
            inject,
            collide_bgk,
            collide_trt,
            collide_mrt,
            stream,
            boundary,
            extract,
            advect,
            linear_sampler,
            bind_group_layout,
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Per-frame dispatch helpers
// ─────────────────────────────────────────────────────────────────────────────

/// Build a bind group from the current grid state.
///
/// As the source/destination buffers swap every frame (ping-pong), we rebuild
/// the bind group each frame.  This is cheap — bind groups are reference-counted
/// handles with no GPU allocation.
pub fn build_bind_group(
    device: &wgpu::Device,
    layout: &wgpu::BindGroupLayout,
    grid: &GpuGrid,
) -> wgpu::BindGroup {
    let mut entries: Vec<wgpu::BindGroupEntry> = vec![
        wgpu::BindGroupEntry {
            binding: 0,
            resource: grid.uniform_buffer.as_entire_binding(),
        },
        wgpu::BindGroupEntry {
            binding: 1,
            resource: grid.src_distributions().as_entire_binding(),
        },
        wgpu::BindGroupEntry {
            binding: 2,
            resource: grid.dst_distributions().as_entire_binding(),
        },
        wgpu::BindGroupEntry {
            binding: 3,
            resource: grid.macroscopic.as_entire_binding(),
        },
        wgpu::BindGroupEntry {
            binding: 4,
            resource: wgpu::BindingResource::TextureView(&grid.obstacle_texture_view),
        },
        wgpu::BindGroupEntry {
            binding: 5,
            resource: grid.event_buffer.as_entire_binding(),
        },
    ];

    entries.push(wgpu::BindGroupEntry {
        binding: 6,
        resource: grid.color_densities.as_entire_binding(),
    });
    // grid.color_densities is a single packed buffer: [cell * MAX_COLORS + channel].

    entries.push(wgpu::BindGroupEntry {
        binding: 7,
        resource: wgpu::BindingResource::TextureView(&grid.injection_texture_view),
    });

    device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("lbm::bind_group"),
        layout,
        entries: &entries,
    })
}

/// Encode all six compute passes into `encoder`.
///
/// Call once per frame after uploading the uniform buffer, obstacle texture,
/// and event ring buffer.
pub fn encode_lbm_passes(
    encoder: &mut wgpu::CommandEncoder,
    pipelines: &GpuPipelines,
    bind_group: &wgpu::BindGroup,
    grid_width: u32,
    grid_height: u32,
    collision_mode: CollisionMode,
) {
    let wg_x = grid_width.div_ceil(WG_SIZE);
    let wg_y = grid_height.div_ceil(WG_SIZE);

    let mut dispatch = |label: &str, pipeline: &wgpu::ComputePipeline| {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some(label),
            timestamp_writes: None,
        });
        pass.set_pipeline(pipeline);
        pass.set_bind_group(0, bind_group, &[]);
        pass.dispatch_workgroups(wg_x, wg_y, 1);
    };

    let collide_pipeline = match collision_mode {
        CollisionMode::Bgk => &pipelines.collide_bgk,
        CollisionMode::Trt => &pipelines.collide_trt,
        CollisionMode::Mrt => &pipelines.collide_mrt,
    };

    dispatch("lbm::inject", &pipelines.inject);
    dispatch("lbm::collide", collide_pipeline);
    dispatch("lbm::stream", &pipelines.stream);
    dispatch("lbm::boundary", &pipelines.boundary);
    dispatch("lbm::extract", &pipelines.extract);
    dispatch("lbm::advect", &pipelines.advect);
}

// ─────────────────────────────────────────────────────────────────────────────
// Obstacle rasterisation (CPU)
// ─────────────────────────────────────────────────────────────────────────────

/// Rasterise a slice of [`crate::types::ObstaclePatch`] into a flat `ObstacleTexel` grid.
///
/// The result is ready to be passed to [`GpuGrid::upload_obstacle_texture`].
pub fn rasterise_obstacles(
    patches: &[crate::types::ObstaclePatch],
    config: &SimConfig,
    out: &mut Vec<ObstacleTexel>,
) {
    let w = config.grid_width as usize;
    let h = config.grid_height as usize;

    // Clear to empty.
    out.clear();
    out.resize(
        w * h,
        ObstacleTexel {
            mask: 0.0,
            vel_x: 0.0,
            vel_y: 0.0,
            open_boundary: 0.0,
        },
    );

    // World-space size of one grid cell — needed for the PSM area-overlap fraction.
    let cell_w = config.world_width / config.grid_width as f32;
    let cell_h = config.world_height / config.grid_height as f32;

    // Each sub-step covers 1/substeps of the frame interval, so the per-step
    // wall velocity must be scaled down accordingly.
    let vel_scale = 1.0 / config.substeps.max(1) as f32;

    for patch in patches {
        // OBB centre and half-extents in world-space.
        let cx = (patch.x_min + patch.x_max) * 0.5;
        let cy = (patch.y_min + patch.y_max) * 0.5;
        let hx = (patch.x_max - patch.x_min) * 0.5;
        let hy = (patch.y_max - patch.y_min) * 0.5;

        if patch.rotation == 0.0 {
            // ── Fast AABB path ──────────────────────────────────────────────
            let gx_min = ((patch.x_min / config.world_width) * config.grid_width as f32).floor() as i32;
            let gx_max = ((patch.x_max / config.world_width) * config.grid_width as f32).ceil() as i32;
            let gy_min = ((patch.y_min / config.world_height) * config.grid_height as f32).floor() as i32;
            let gy_max = ((patch.y_max / config.world_height) * config.grid_height as f32).ceil() as i32;

            let gx_min = gx_min.max(0) as usize;
            let gx_max = gx_max.min(config.grid_width as i32) as usize;
            let gy_min = gy_min.max(0) as usize;
            let gy_max = gy_max.min(config.grid_height as i32) as usize;

            for gy in gy_min..gy_max {
                for gx in gx_min..gx_max {
                    let cell_x_lo = gx as f32 * cell_w;
                    let cell_x_hi = cell_x_lo + cell_w;
                    let cell_y_lo = gy as f32 * cell_h;
                    let cell_y_hi = cell_y_lo + cell_h;
                    let ox = (patch.x_max.min(cell_x_hi) - patch.x_min.max(cell_x_lo)).max(0.0);
                    let oy = (patch.y_max.min(cell_y_hi) - patch.y_min.max(cell_y_lo)).max(0.0);
                    let fraction = (ox / cell_w) * (oy / cell_h);
                    // Inset mode: skip cells that aren't fully inside the geometry.
                    if patch.inset && fraction < 1.0 { continue; }
                    if fraction > out[gy * w + gx].mask {
                        let ob = out[gy * w + gx].open_boundary;
                        out[gy * w + gx] = ObstacleTexel {
                            mask: fraction,
                            vel_x: patch.vel_x * vel_scale,
                            vel_y: patch.vel_y * vel_scale,
                            open_boundary: ob,
                        };
                    }
                }
            }
        } else {
            // ── OBB path: multi-sample coverage ────────────────────────────
            // Precompute trig for rotating sample points into OBB local space.
            let (sin_r, cos_r) = patch.rotation.sin_cos();

            // AABB of the OBB — used to limit candidate cells.
            let aabb_hw = hx * cos_r.abs() + hy * sin_r.abs();
            let aabb_hh = hx * sin_r.abs() + hy * cos_r.abs();
            let aabb_x_min = cx - aabb_hw;
            let aabb_x_max = cx + aabb_hw;
            let aabb_y_min = cy - aabb_hh;
            let aabb_y_max = cy + aabb_hh;

            let gx_min = ((aabb_x_min / config.world_width) * config.grid_width as f32).floor() as i32;
            let gx_max = ((aabb_x_max / config.world_width) * config.grid_width as f32).ceil() as i32;
            let gy_min = ((aabb_y_min / config.world_height) * config.grid_height as f32).floor() as i32;
            let gy_max = ((aabb_y_max / config.world_height) * config.grid_height as f32).ceil() as i32;

            let gx_min = gx_min.max(0) as usize;
            let gx_max = gx_max.min(config.grid_width as i32) as usize;
            let gy_min = gy_min.max(0) as usize;
            let gy_max = gy_max.min(config.grid_height as i32) as usize;

            // 4×4 stratified sample grid inside each cell.
            const N: usize = 4;
            const INV_N: f32 = 1.0 / N as f32;
            const N_SAMPLES: u32 = (N * N) as u32;

            for gy in gy_min..gy_max {
                for gx in gx_min..gx_max {
                    let cell_x_lo = gx as f32 * cell_w;
                    let cell_y_lo = gy as f32 * cell_h;
                    let mut hits = 0u32;
                    for sy in 0..N {
                        let py = cell_y_lo + (sy as f32 + 0.5) * cell_h * INV_N;
                        for sx in 0..N {
                            let px = cell_x_lo + (sx as f32 + 0.5) * cell_w * INV_N;
                            // Rotate sample point into OBB local frame.
                            let dx = px - cx;
                            let dy = py - cy;
                            let lx = dx * cos_r + dy * sin_r;
                            let ly = -dx * sin_r + dy * cos_r;
                            if lx.abs() <= hx && ly.abs() <= hy {
                                hits += 1;
                            }
                        }
                    }
                    // Inset mode: skip cells that aren't fully inside the geometry.
                    if patch.inset && hits < N_SAMPLES { continue; }
                    let fraction = hits as f32 / N_SAMPLES as f32;
                    if fraction > out[gy * w + gx].mask {
                        let ob = out[gy * w + gx].open_boundary;
                        out[gy * w + gx] = ObstacleTexel {
                            mask: fraction,
                            vel_x: patch.vel_x * vel_scale,
                            vel_y: patch.vel_y * vel_scale,
                            open_boundary: ob,
                        };
                    }
                }
            }
        }
    }
}

/// Build the [`LbmUniforms`] struct from config + per-frame event count.
pub fn build_uniforms(
    config: &SimConfig,
    event_count: u32,
    additive_injection: bool,
) -> LbmUniforms {
    let mut inject_densities = [[0f32; 4]; 2];
    let mut inject_color_densities = [[0f32; 4]; 2];
    let mut dissipations = [[0f32; 4]; 2];

    for (i, p) in config.effect_profiles.iter().enumerate().take(8) {
        inject_densities[i / 4][i % 4] = p.inject_density;
        inject_color_densities[i / 4][i % 4] = p.inject_color_density;
        dissipations[i / 4][i % 4] = p.dissipation;
    }

    // MRT relaxation rates (Lallemand & Luo 2000).
    let s_nu = 1.0 / config.tau;                   // shear/normal stress → kinematic viscosity
    let s_e  = config.mrt_s_e;                     // energy and ghost modes
    let s_q  = config.mrt_s_q.unwrap_or_else(|| {
        // Recommended: keeps viscosity well-defined across resolutions.
        8.0 * (2.0 - s_nu) / (8.0 - s_nu)
    });

    // Pack as 3 vec4s: [s0..s3], [s4..s7], [s8, 0, 0, 0]
    // Conserved-moment slots (s0, s3, s5) are 0.0; their non-equilibrium is
    // always zero so the value never contributes to the collision.
    let mrt_s = [
        [0.0_f32, s_e,   s_e,  0.0],   // s0(unused), s1(e), s2(ε), s3(unused)
        [s_q,     0.0,   s_q,  s_nu],  // s4(qx), s5(unused), s6(qy), s7(Pxx)
        [s_nu,    0.0,   0.0,  0.0],   // s8(Pxy), padding
    ];

    LbmUniforms {
        grid_width: config.grid_width,
        grid_height: config.grid_height,
        tau: config.tau,
        _pad0: 0.0,
        world_width: config.world_width,
        world_height: config.world_height,
        event_count,
        color_count: config.color_count,
        inject_densities,
        inject_color_densities,
        dissipations,
        gravity_x: config.gravity_x,
        gravity_y: config.gravity_y,
        injection_mode: if additive_injection { 1 } else { 0 },
        _pad1: 0,
        mrt_s,
    }
}
