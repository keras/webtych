use std::sync::Arc;
use wgpu::util::DeviceExt;
use winit::window::Window;

use webtych_game::physics::BlockInstance;
use webtych_game::ColorPalette;
use webtych_lbm::{InjectionEvent, ObstaclePatch, OpenBoundaryPatch, SimConfig, Simulation};

// ── WGSL shader ───────────────────────────────────────────────────────────────

const SHADER: &str = r#"
struct Camera {
    projection: mat4x4<f32>,
}
@group(0) @binding(0) var<uniform> camera: Camera;

struct Palette {
    count: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
    colors: array<vec4<f32>, 16>,
}
@group(0) @binding(1) var<uniform> palette: Palette;

struct VOut {
    @builtin(position) clip_pos: vec4<f32>,
    @location(0) color: vec4<f32>,
}

@vertex
fn vs_main(
    // Per-vertex: unit quad corner
    @location(0) vert_pos: vec2<f32>,
    // Per-instance attributes
    @location(1) inst_pos: vec2<f32>,
    @location(2) inst_rot: f32,
    @location(3) inst_color_id: u32,
    @location(4) inst_alive: u32,
) -> VOut {
    // Rotate the quad corner by the instance rotation.
    let c = cos(inst_rot);
    let s = sin(inst_rot);
    let rotated = vec2<f32>(
        vert_pos.x * c - vert_pos.y * s,
        vert_pos.x * s + vert_pos.y * c,
    );

    // Scale to cell size (0.95 for a small gap between cells) and translate.
    let cell_size = 0.95;
    let world_pos = rotated * cell_size * 0.5 + inst_pos;

    var out: VOut;
    out.clip_pos = camera.projection * vec4<f32>(world_pos, 0.0, 1.0);

    let idx = min(inst_color_id, palette.count - 1u);
    var base_color = palette.colors[idx];

    // Dim the block slightly if it's being destroyed.
    if inst_alive == 0u {
        base_color = base_color * 0.4;
    }

    out.color = base_color;
    return out;
}

@fragment
fn fs_main(in: VOut) -> @location(0) vec4<f32> {
    return in.color;
}
"#;

// ── Board background shader ───────────────────────────────────────────────────

const BOARD_SHADER: &str = r#"
struct Camera {
    projection: mat4x4<f32>,
}
@group(0) @binding(0) var<uniform> camera: Camera;

struct VOut {
    @builtin(position) clip_pos: vec4<f32>,
}

@vertex
fn vs_main(@location(0) vert_pos: vec2<f32>) -> VOut {
    var out: VOut;
    out.clip_pos = camera.projection * vec4<f32>(vert_pos, 0.0, 1.0);
    return out;
}

@fragment
fn fs_main(in: VOut) -> @location(0) vec4<f32> {
    return vec4<f32>(0.08, 0.08, 0.12, 1.0);
}
"#;

// ── Smoke bake compute shader ─────────────────────────────────────────────────

const SMOKE_BAKE_SHADER: &str = r#"
struct SmokeBakeUniforms {
    grid_width:  u32,
    grid_height: u32,
    color_count: u32,
    _pad:        u32,
    colors: array<vec4<f32>, 16>,
}

@group(0) @binding(0) var<uniform> u: SmokeBakeUniforms;
// macroscopic buffer: [rho, ux, uy] per cell
@group(0) @binding(1) var<storage, read> macro_buf: array<f32>;
@group(0) @binding(2) var smoke_out: texture_storage_2d<rgba8unorm, write>;

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    if gid.x >= u.grid_width || gid.y >= u.grid_height {
        return;
    }
    let cell = gid.y * u.grid_width + gid.x;

    let rho = macro_buf[cell * 3u];
    let ux  = macro_buf[cell * 3u + 1u];
    let uy  = macro_buf[cell * 3u + 2u];

    // Overpressure: red. Underpressure: blue. Velocity: green.
    let scale = 8.0;
    let r = clamp((rho - 1.0) * scale, 0.0, 1.0);
    let b = clamp((1.0 - rho) * scale, 0.0, 1.0);
    let g = clamp(sqrt(ux * ux + uy * uy) * 40.0, 0.0, 1.0);
    let a = clamp(r + g + b, 0.0, 1.0);

    textureStore(smoke_out, vec2<i32>(gid.xy), vec4<f32>(r, g, b, a));
}
"#;

// ── Smoke render shader ──────────────────────────────────────────────────────

const SMOKE_RENDER_SHADER: &str = r#"
struct Camera {
    projection: mat4x4<f32>,
}
@group(0) @binding(0) var<uniform> camera: Camera;
@group(0) @binding(1) var smoke_tex:  texture_2d<f32>;
@group(0) @binding(2) var smoke_samp: sampler;

struct VOut {
    @builtin(position) clip_pos: vec4<f32>,
    @location(0)       uv:       vec2<f32>,
}

@vertex
fn vs_main(@location(0) vert_pos: vec2<f32>) -> VOut {
    var out: VOut;
    out.clip_pos = camera.projection * vec4<f32>(vert_pos, 0.0, 1.0);
    out.uv = vert_pos / vec2<f32>(10.0, 20.0);
    return out;
}

@fragment
fn fs_main(in: VOut) -> @location(0) vec4<f32> {
    let smoke = textureSample(smoke_tex, smoke_samp, in.uv);
    return vec4<f32>(smoke.rgb, smoke.a * 0.85);
}
"#;

// ── Error type ────────────────────────────────────────────────────────────────

#[derive(Debug)]
pub enum RenderError {
    /// Surface is lost or outdated — caller should call resize().
    Reconfigure,
}

// ── Quad vertex data ──────────────────────────────────────────────────────────

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct Vertex {
    position: [f32; 2],
}

const QUAD_VERTICES: &[Vertex] = &[
    Vertex { position: [-1.0, -1.0] },
    Vertex { position: [ 1.0, -1.0] },
    Vertex { position: [ 1.0,  1.0] },
    Vertex { position: [-1.0,  1.0] },
];

const QUAD_INDICES: &[u16] = &[0, 1, 2, 0, 2, 3];

// ── Palette GPU layout ──────────────────────────────────────────────────────

/// Maximum number of colors in the GPU palette uniform.
const MAX_PALETTE_COLORS: usize = 16;

/// GPU-side palette struct: count (u32) + padding + 16 × vec4<f32>.
/// Total: 16 bytes header + 16 × 16 bytes = 272 bytes.
#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct GpuPalette {
    count: u32,
    _pad: [u32; 3],
    colors: [[f32; 4]; MAX_PALETTE_COLORS],
}

impl GpuPalette {
    fn from_palette(palette: &ColorPalette) -> Self {
        let mut gpu = Self {
            count: palette.len().min(MAX_PALETTE_COLORS as u32),
            _pad: [0; 3],
            colors: [[0.0; 4]; MAX_PALETTE_COLORS],
        };
        for (i, rgba) in palette.as_slice().iter().take(MAX_PALETTE_COLORS).enumerate() {
            gpu.colors[i] = *rgba;
        }
        gpu
    }
}

// ── Smoke bake uniforms ─────────────────────────────────────────────────────

/// GPU uniform for the smoke bake compute pass.
/// Must match the WGSL `SmokeBakeUniforms` struct exactly.
#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct SmokeBakeUniforms {
    grid_width: u32,
    grid_height: u32,
    color_count: u32,
    _pad: u32,
    colors: [[f32; 4]; MAX_PALETTE_COLORS],
}

// ── Renderer ──────────────────────────────────────────────────────────────────

pub struct Renderer {
    surface: wgpu::Surface<'static>,
    device: wgpu::Device,
    queue: wgpu::Queue,
    config: wgpu::SurfaceConfiguration,
    // Block rendering
    block_pipeline: wgpu::RenderPipeline,
    vertex_buf: wgpu::Buffer,
    index_buf: wgpu::Buffer,
    instance_buf: wgpu::Buffer,
    instance_count: u32,
    camera_buf: wgpu::Buffer,
    #[allow(dead_code)]
    palette_buf: wgpu::Buffer,
    bind_group: wgpu::BindGroup,
    // Board background
    board_pipeline: wgpu::RenderPipeline,
    board_vertex_buf: wgpu::Buffer,
    board_index_buf: wgpu::Buffer,
    // LBM fluid simulation
    simulation: Simulation,
    // Smoke bake compute pass (texture + uniform kept alive for GPU references)
    #[allow(dead_code)]
    smoke_texture: wgpu::Texture,
    #[allow(dead_code)]
    smoke_texture_view: wgpu::TextureView,
    smoke_bake_pipeline: wgpu::ComputePipeline,
    smoke_bake_bind_group: wgpu::BindGroup,
    #[allow(dead_code)]
    smoke_bake_uniform_buf: wgpu::Buffer,
    // Smoke render pass
    smoke_render_pipeline: wgpu::RenderPipeline,
    smoke_render_bind_group: wgpu::BindGroup,
}

/// Orthographic projection matrix for a 2D view.
/// Maps world coordinates to clip space [-1, 1].
fn ortho_projection(left: f32, right: f32, bottom: f32, top: f32) -> [f32; 16] {
    let w = right - left;
    let h = top - bottom;
    [
        2.0 / w, 0.0,     0.0, 0.0,
        0.0,     2.0 / h, 0.0, 0.0,
        0.0,     0.0,     1.0, 0.0,
        -(right + left) / w, -(top + bottom) / h, 0.0, 1.0,
    ]
}

impl Renderer {
    pub async fn new(window: Arc<Window>, palette: &ColorPalette) -> Self {
        let size   = window.inner_size();
        let width  = size.width.max(1);
        let height = size.height.max(1);

        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends:                 wgpu::Backends::all(),
            flags:                    wgpu::InstanceFlags::default(),
            memory_budget_thresholds: wgpu::MemoryBudgetThresholds::default(),
            backend_options:          wgpu::BackendOptions::default(),
            display:                  None,
        });

        let surface = instance.create_surface(window).expect("create surface");

        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference:       wgpu::PowerPreference::default(),
                compatible_surface:     Some(&surface),
                force_fallback_adapter: false,
            })
            .await
            .expect(
                "No WebGPU adapter found. \
                 Please use Chrome 113+, Edge 113+, or Safari 18+.",
            );

        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                label:                 None,
                required_features:     wgpu::Features::empty(),
                required_limits:       wgpu::Limits::default(),
                experimental_features: Default::default(),
                memory_hints:          wgpu::MemoryHints::default(),
                trace:                 Default::default(),
            })
            .await
            .expect("request device");

        // ── surface configuration ────────────────────────────────────────
        let caps   = surface.get_capabilities(&adapter);
        let format = caps
            .formats
            .iter()
            .copied()
            .find(|f| f.is_srgb())
            .unwrap_or(caps.formats[0]);

        let config = wgpu::SurfaceConfiguration {
            usage:                         wgpu::TextureUsages::RENDER_ATTACHMENT,
            format,
            width,
            height,
            present_mode:                  wgpu::PresentMode::AutoVsync,
            alpha_mode:                    caps.alpha_modes[0],
            view_formats:                  vec![],
            desired_maximum_frame_latency: 2,
        };
        surface.configure(&device, &config);

        // ── quad geometry ────────────────────────────────────────────────
        let vertex_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label:    Some("quad vertices"),
            contents: bytemuck::cast_slice(QUAD_VERTICES),
            usage:    wgpu::BufferUsages::VERTEX,
        });

        let index_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label:    Some("quad indices"),
            contents: bytemuck::cast_slice(QUAD_INDICES),
            usage:    wgpu::BufferUsages::INDEX,
        });

        // ── instance buffer (initially empty, resized as needed) ─────────
        let instance_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label:              Some("block instances"),
            size:               std::mem::size_of::<BlockInstance>() as u64 * 256,
            usage:              wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // ── camera uniform ───────────────────────────────────────────────
        // View the board with some margin. Board is 10 wide × 20 tall.
        let margin = 2.0;
        let proj = ortho_projection(
            -margin,               // left
            10.0 + margin,         // right
            -margin,               // bottom
            20.0 + margin,         // top
        );
        let camera_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label:    Some("camera"),
            contents: bytemuck::cast_slice(&proj),
            usage:    wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        // ── palette uniform ─────────────────────────────────────────────
        let gpu_palette = GpuPalette::from_palette(palette);
        let palette_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label:    Some("palette"),
            contents: bytemuck::cast_slice(&[gpu_palette]),
            usage:    wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label:   Some("camera+palette bgl"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding:    0,
                    visibility: wgpu::ShaderStages::VERTEX,
                    ty: wgpu::BindingType::Buffer {
                        ty:                 wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size:   None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding:    1,
                    visibility: wgpu::ShaderStages::VERTEX,
                    ty: wgpu::BindingType::Buffer {
                        ty:                 wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size:   None,
                    },
                    count: None,
                },
            ],
        });

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label:   Some("camera+palette bg"),
            layout:  &bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding:  0,
                    resource: camera_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding:  1,
                    resource: palette_buf.as_entire_binding(),
                },
            ],
        });

        // ── block render pipeline ────────────────────────────────────────
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label:  Some("block shader"),
            source: wgpu::ShaderSource::Wgsl(SHADER.into()),
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label:              Some("block layout"),
            bind_group_layouts: &[Some(&bgl)],
            immediate_size:     0,
        });

        let vertex_layout = wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<Vertex>() as u64,
            step_mode:    wgpu::VertexStepMode::Vertex,
            attributes:   &[
                // location(0): vert_pos
                wgpu::VertexAttribute {
                    format: wgpu::VertexFormat::Float32x2,
                    offset: 0,
                    shader_location: 0,
                },
            ],
        };

        let instance_layout = wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<BlockInstance>() as u64,
            step_mode:    wgpu::VertexStepMode::Instance,
            attributes:   &[
                // location(1): inst_pos (position: [f32; 2])
                wgpu::VertexAttribute {
                    format: wgpu::VertexFormat::Float32x2,
                    offset: 0,
                    shader_location: 1,
                },
                // location(2): inst_rot (rotation: f32)
                wgpu::VertexAttribute {
                    format: wgpu::VertexFormat::Float32,
                    offset: 8,
                    shader_location: 2,
                },
                // location(3): inst_color_id (color_id: u32)
                wgpu::VertexAttribute {
                    format: wgpu::VertexFormat::Uint32,
                    offset: 12,
                    shader_location: 3,
                },
                // location(4): inst_alive (alive: u32)
                wgpu::VertexAttribute {
                    format: wgpu::VertexFormat::Uint32,
                    offset: 16,
                    shader_location: 4,
                },
            ],
        };

        let block_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label:  Some("block pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module:              &shader,
                entry_point:         Some("vs_main"),
                buffers:             &[vertex_layout, instance_layout],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module:              &shader,
                entry_point:         Some("fs_main"),
                targets: &[Some(wgpu::ColorTargetState {
                    format,
                    blend:      Some(wgpu::BlendState::REPLACE),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                ..Default::default()
            },
            depth_stencil: None,
            multisample:   wgpu::MultisampleState::default(),
            multiview_mask: None,
            cache:         None,
        });

        // ── board background pipeline ────────────────────────────────────
        let board_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label:  Some("board shader"),
            source: wgpu::ShaderSource::Wgsl(BOARD_SHADER.into()),
        });

        let board_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label:              Some("board layout"),
            bind_group_layouts: &[Some(&bgl)],
            immediate_size:     0,
        });

        let board_vertex_layout = wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<Vertex>() as u64,
            step_mode:    wgpu::VertexStepMode::Vertex,
            attributes:   &[
                wgpu::VertexAttribute {
                    format: wgpu::VertexFormat::Float32x2,
                    offset: 0,
                    shader_location: 0,
                },
            ],
        };

        let board_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label:  Some("board pipeline"),
            layout: Some(&board_pipeline_layout),
            vertex: wgpu::VertexState {
                module:              &board_shader,
                entry_point:         Some("vs_main"),
                buffers:             &[board_vertex_layout],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module:              &board_shader,
                entry_point:         Some("fs_main"),
                targets: &[Some(wgpu::ColorTargetState {
                    format,
                    blend:      Some(wgpu::BlendState::REPLACE),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                ..Default::default()
            },
            depth_stencil: None,
            multisample:   wgpu::MultisampleState::default(),
            multiview_mask: None,
            cache:         None,
        });

        // Board quad: rectangle from (0,0) to (10,20) in world coords.
        let board_verts: &[Vertex] = &[
            Vertex { position: [0.0,  0.0] },
            Vertex { position: [10.0, 0.0] },
            Vertex { position: [10.0, 20.0] },
            Vertex { position: [0.0,  20.0] },
        ];

        let board_vertex_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label:    Some("board vertices"),
            contents: bytemuck::cast_slice(board_verts),
            usage:    wgpu::BufferUsages::VERTEX,
        });

        let board_index_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label:    Some("board indices"),
            contents: bytemuck::cast_slice(QUAD_INDICES),
            usage:    wgpu::BufferUsages::INDEX,
        });

        // ── LBM simulation ──────────────────────────────────────────────
        let color_count = palette.len();
        let sim_config = SimConfig::for_game_board(10.0, 20.0, color_count);
        let grid_width = sim_config.grid_width;
        let grid_height = sim_config.grid_height;
        let mut simulation = Simulation::new(&device, sim_config);

        // Open boundaries on all four edges — pressure/smoke can drain out
        // rather than piling up and driving the simulation unstable.
        let b = 20.0 / 256.0 * 2.0; // ~2 grid cells
        simulation.set_open_boundaries(&[
            OpenBoundaryPatch { x_min: 0.0,      y_min: 20.0 - b, x_max: 10.0,     y_max: 20.0 }, // top
            OpenBoundaryPatch { x_min: 0.0,      y_min: 0.0,      x_max: 10.0,     y_max: b    }, // bottom
            OpenBoundaryPatch { x_min: 0.0,      y_min: 0.0,      x_max: b,        y_max: 20.0 }, // left
            OpenBoundaryPatch { x_min: 10.0 - b, y_min: 0.0,      x_max: 10.0,     y_max: 20.0 }, // right
        ]);

        // ── Smoke texture (output of bake, input to render) ─────────────
        let smoke_texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("smoke_texture"),
            size: wgpu::Extent3d {
                width: grid_width,
                height: grid_height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage: wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let smoke_texture_view =
            smoke_texture.create_view(&wgpu::TextureViewDescriptor::default());

        // ── Smoke bake compute pipeline ─────────────────────────────────
        let smoke_bake_uniforms = SmokeBakeUniforms {
            grid_width,
            grid_height,
            color_count,
            _pad: 0,
            colors: gpu_palette.colors,
        };
        let smoke_bake_uniform_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label:    Some("smoke_bake_uniforms"),
            contents: bytemuck::cast_slice(&[smoke_bake_uniforms]),
            usage:    wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        let smoke_bake_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label:   Some("smoke_bake bgl"),
            entries: &[
                // 0: uniforms
                wgpu::BindGroupLayoutEntry {
                    binding:    0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty:                 wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size:   None,
                    },
                    count: None,
                },
                // 1: color_densities (read-only storage)
                wgpu::BindGroupLayoutEntry {
                    binding:    1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty:                 wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size:   None,
                    },
                    count: None,
                },
                // 2: smoke output texture (write-only storage)
                wgpu::BindGroupLayoutEntry {
                    binding:    2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::StorageTexture {
                        access:         wgpu::StorageTextureAccess::WriteOnly,
                        format:         wgpu::TextureFormat::Rgba8Unorm,
                        view_dimension: wgpu::TextureViewDimension::D2,
                    },
                    count: None,
                },
            ],
        });

        let smoke_bake_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label:  Some("smoke_bake bg"),
            layout: &smoke_bake_bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding:  0,
                    resource: smoke_bake_uniform_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding:  1,
                    resource: simulation.macroscopic_buffer().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding:  2,
                    resource: wgpu::BindingResource::TextureView(&smoke_texture_view),
                },
            ],
        });

        let smoke_bake_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label:  Some("smoke bake shader"),
            source: wgpu::ShaderSource::Wgsl(SMOKE_BAKE_SHADER.into()),
        });

        let smoke_bake_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label:              Some("smoke_bake layout"),
                bind_group_layouts: &[Some(&smoke_bake_bgl)],
                immediate_size:     0,
            });

        let smoke_bake_pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label:              Some("smoke_bake pipeline"),
                layout:             Some(&smoke_bake_pipeline_layout),
                module:             &smoke_bake_shader,
                entry_point:        Some("main"),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                cache:              None,
            });

        // ── Smoke render pipeline ───────────────────────────────────────
        let smoke_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label:           Some("smoke_sampler"),
            address_mode_u:  wgpu::AddressMode::ClampToEdge,
            address_mode_v:  wgpu::AddressMode::ClampToEdge,
            mag_filter:      wgpu::FilterMode::Linear,
            min_filter:      wgpu::FilterMode::Linear,
            ..Default::default()
        });

        let smoke_render_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label:   Some("smoke_render bgl"),
            entries: &[
                // 0: camera
                wgpu::BindGroupLayoutEntry {
                    binding:    0,
                    visibility: wgpu::ShaderStages::VERTEX,
                    ty: wgpu::BindingType::Buffer {
                        ty:                 wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size:   None,
                    },
                    count: None,
                },
                // 1: smoke texture
                wgpu::BindGroupLayoutEntry {
                    binding:    1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type:    wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled:   false,
                    },
                    count: None,
                },
                // 2: sampler
                wgpu::BindGroupLayoutEntry {
                    binding:    2,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
            ],
        });

        let smoke_render_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label:  Some("smoke_render bg"),
            layout: &smoke_render_bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding:  0,
                    resource: camera_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding:  1,
                    resource: wgpu::BindingResource::TextureView(&smoke_texture_view),
                },
                wgpu::BindGroupEntry {
                    binding:  2,
                    resource: wgpu::BindingResource::Sampler(&smoke_sampler),
                },
            ],
        });

        let smoke_render_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label:  Some("smoke render shader"),
            source: wgpu::ShaderSource::Wgsl(SMOKE_RENDER_SHADER.into()),
        });

        let smoke_render_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label:              Some("smoke_render layout"),
                bind_group_layouts: &[Some(&smoke_render_bgl)],
                immediate_size:     0,
            });

        let smoke_render_vertex_layout = wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<Vertex>() as u64,
            step_mode:    wgpu::VertexStepMode::Vertex,
            attributes:   &[wgpu::VertexAttribute {
                format:          wgpu::VertexFormat::Float32x2,
                offset:          0,
                shader_location: 0,
            }],
        };

        let smoke_render_pipeline =
            device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label:  Some("smoke_render pipeline"),
                layout: Some(&smoke_render_pipeline_layout),
                vertex: wgpu::VertexState {
                    module:              &smoke_render_shader,
                    entry_point:         Some("vs_main"),
                    buffers:             &[smoke_render_vertex_layout],
                    compilation_options: Default::default(),
                },
                fragment: Some(wgpu::FragmentState {
                    module:              &smoke_render_shader,
                    entry_point:         Some("fs_main"),
                    targets: &[Some(wgpu::ColorTargetState {
                        format,
                        blend:      Some(wgpu::BlendState::ALPHA_BLENDING),
                        write_mask: wgpu::ColorWrites::ALL,
                    })],
                    compilation_options: Default::default(),
                }),
                primitive: wgpu::PrimitiveState {
                    topology: wgpu::PrimitiveTopology::TriangleList,
                    ..Default::default()
                },
                depth_stencil: None,
                multisample:   wgpu::MultisampleState::default(),
                multiview_mask: None,
                cache:         None,
            });

        Self {
            surface,
            device,
            queue,
            config,
            block_pipeline,
            vertex_buf,
            index_buf,
            instance_buf,
            instance_count: 0,
            camera_buf,
            palette_buf,
            bind_group,
            board_pipeline,
            board_vertex_buf,
            board_index_buf,
            simulation,
            smoke_texture,
            smoke_texture_view,
            smoke_bake_pipeline,
            smoke_bake_bind_group,
            smoke_bake_uniform_buf,
            smoke_render_pipeline,
            smoke_render_bind_group,
        }
    }

    pub fn resize(&mut self, size: winit::dpi::PhysicalSize<u32>) {
        let w = size.width.max(1);
        let h = size.height.max(1);
        self.config.width  = w;
        self.config.height = h;
        self.surface.configure(&self.device, &self.config);

        // Update projection to maintain correct aspect ratio.
        let board_w: f32 = 10.0;
        let board_h: f32 = 20.0;
        let margin: f32 = 2.0;

        let view_w = board_w + margin * 2.0;
        let view_h = board_h + margin * 2.0;

        let aspect = w as f32 / h as f32;
        let view_aspect = view_w / view_h;

        let (proj_w, proj_h) = if aspect > view_aspect {
            // Window is wider than view — expand horizontal range.
            (view_h * aspect, view_h)
        } else {
            // Window is taller than view — expand vertical range.
            (view_w, view_w / aspect)
        };

        let cx = board_w / 2.0;
        let cy = board_h / 2.0;
        let proj = ortho_projection(
            cx - proj_w / 2.0,
            cx + proj_w / 2.0,
            cy - proj_h / 2.0,
            cy + proj_h / 2.0,
        );
        self.queue.write_buffer(&self.camera_buf, 0, bytemuck::cast_slice(&proj));
    }

    /// Upload a new color palette to the GPU.
    #[allow(dead_code)]
    pub fn update_palette(&mut self, palette: &ColorPalette) {
        let gpu_palette = GpuPalette::from_palette(palette);
        self.queue.write_buffer(&self.palette_buf, 0, bytemuck::cast_slice(&[gpu_palette]));
    }

    /// Upload new block instance data for this frame.
    pub fn update_instances(&mut self, instances: &[BlockInstance]) {
        self.instance_count = instances.len() as u32;

        if instances.is_empty() {
            return;
        }

        let data = bytemuck::cast_slice(instances);
        let required_size = data.len() as u64;

        // Grow the instance buffer if needed.
        if required_size > self.instance_buf.size() {
            self.instance_buf = self.device.create_buffer(&wgpu::BufferDescriptor {
                label:              Some("block instances"),
                size:               required_size,
                usage:              wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
        }

        self.queue.write_buffer(&self.instance_buf, 0, data);
    }

    /// Feed obstacle patches and injection events to the fluid simulation.
    /// Call once per frame after game.update(), before render().
    pub fn update_fluid(
        &mut self,
        obstacles: &[ObstaclePatch],
        events: Vec<InjectionEvent>,
    ) {
        self.simulation.set_obstacles(&self.queue, obstacles);
        for event in events {
            self.simulation.push_event(event);
        }
        self.simulation.step(&self.device, &self.queue);
    }

    pub fn render(&self) -> Result<(), RenderError> {
        let output = match self.surface.get_current_texture() {
            wgpu::CurrentSurfaceTexture::Success(t)
            | wgpu::CurrentSurfaceTexture::Suboptimal(t) => t,
            wgpu::CurrentSurfaceTexture::Outdated
            | wgpu::CurrentSurfaceTexture::Lost => return Err(RenderError::Reconfigure),
            // Transient conditions — skip this frame and try again next tick.
            wgpu::CurrentSurfaceTexture::Timeout
            | wgpu::CurrentSurfaceTexture::Occluded
            | wgpu::CurrentSurfaceTexture::Validation => return Ok(()),
        };

        let view = output
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());

        let mut enc = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("frame"),
        });

        // ── Smoke bake compute pass ─────────────────────────────────────
        {
            let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("smoke bake"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.smoke_bake_pipeline);
            pass.set_bind_group(0, &self.smoke_bake_bind_group, &[]);
            let wg_x = self.simulation.config.grid_width.div_ceil(16);
            let wg_y = self.simulation.config.grid_height.div_ceil(16);
            pass.dispatch_workgroups(wg_x, wg_y, 1);
        }

        // ── Main render pass ────────────────────────────────────────────
        {
            let mut pass = enc.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("main pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view:           &view,
                    resolve_target: None,
                    depth_slice:    None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color {
                            r: 0.02,
                            g: 0.02,
                            b: 0.04,
                            a: 1.0,
                        }),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                occlusion_query_set:      None,
                timestamp_writes:         None,
                multiview_mask:           None,
            });

            // Draw board background.
            pass.set_pipeline(&self.board_pipeline);
            pass.set_bind_group(0, &self.bind_group, &[]);
            pass.set_vertex_buffer(0, self.board_vertex_buf.slice(..));
            pass.set_index_buffer(self.board_index_buf.slice(..), wgpu::IndexFormat::Uint16);
            pass.draw_indexed(0..6, 0, 0..1);

            // Draw smoke (alpha blended over the board).
            pass.set_pipeline(&self.smoke_render_pipeline);
            pass.set_bind_group(0, &self.smoke_render_bind_group, &[]);
            pass.set_vertex_buffer(0, self.board_vertex_buf.slice(..));
            pass.set_index_buffer(self.board_index_buf.slice(..), wgpu::IndexFormat::Uint16);
            pass.draw_indexed(0..6, 0, 0..1);

            // Draw block instances.
            if self.instance_count > 0 {
                pass.set_pipeline(&self.block_pipeline);
                pass.set_bind_group(0, &self.bind_group, &[]);
                pass.set_vertex_buffer(0, self.vertex_buf.slice(..));
                pass.set_vertex_buffer(1, self.instance_buf.slice(..));
                pass.set_index_buffer(self.index_buf.slice(..), wgpu::IndexFormat::Uint16);
                pass.draw_indexed(0..6, 0, 0..self.instance_count);
            }
        }

        self.queue.submit(std::iter::once(enc.finish()));
        output.present();
        Ok(())
    }
}
