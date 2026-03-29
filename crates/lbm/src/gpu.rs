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

use crate::config::SimConfig;
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
    pub collide: wgpu::ComputePipeline,
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
        let collide_src = shader!("lbm_collide.wgsl");
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
        let sm_collide = make_shader("lbm_collide", collide_src);
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
        let collide = make_pipeline("lbm::collide", &sm_collide);
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
            collide,
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

    dispatch("lbm::inject", &pipelines.inject);
    dispatch("lbm::collide", &pipelines.collide);
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

    for patch in patches {
        // Convert world-space bounds to grid-space integer cells.
        let gx_min = ((patch.x_min / config.world_width) * config.grid_width as f32).floor() as i32;
        let gx_max = ((patch.x_max / config.world_width) * config.grid_width as f32).ceil() as i32;
        let gy_min =
            ((patch.y_min / config.world_height) * config.grid_height as f32).floor() as i32;
        let gy_max =
            ((patch.y_max / config.world_height) * config.grid_height as f32).ceil() as i32;

        // Clamp to grid bounds.
        let gx_min = gx_min.max(0) as usize;
        let gx_max = gx_max.min(config.grid_width as i32) as usize;
        let gy_min = gy_min.max(0) as usize;
        let gy_max = gy_max.min(config.grid_height as i32) as usize;

        for gy in gy_min..gy_max {
            for gx in gx_min..gx_max {
                // Compute the area-overlap fraction (PSM fill fraction).
                // This is the proportion of this grid cell covered by the patch’s rectangle.
                let cx_lo = gx as f32 * cell_w;
                let cx_hi = cx_lo + cell_w;
                let cy_lo = gy as f32 * cell_h;
                let cy_hi = cy_lo + cell_h;
                let ox = (patch.x_max.min(cx_hi) - patch.x_min.max(cx_lo)).max(0.0);
                let oy = (patch.y_max.min(cy_hi) - patch.y_min.max(cy_lo)).max(0.0);
                let fraction = (ox / cell_w) * (oy / cell_h);
                // Keep the velocity of whichever patch has the highest coverage.
                if fraction > out[gy * w + gx].mask {
                    // Preserve open_boundary flag — it is set independently.
                    let ob = out[gy * w + gx].open_boundary;
                    out[gy * w + gx] = ObstacleTexel {
                        mask: fraction,
                        vel_x: patch.vel_x,
                        vel_y: patch.vel_y,
                        open_boundary: ob,
                    };
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

    LbmUniforms {
        grid_width: config.grid_width,
        grid_height: config.grid_height,
        tau: config.tau,
        inv_tau: 1.0 / config.tau,
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
        _pad: 0,
    }
}
