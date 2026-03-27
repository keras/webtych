use std::sync::Arc;
use wgpu::util::DeviceExt;
use winit::window::Window;

use webtych_game::physics::BlockInstance;

// ── WGSL shader ───────────────────────────────────────────────────────────────

const SHADER: &str = r#"
struct Camera {
    projection: mat4x4<f32>,
}
@group(0) @binding(0) var<uniform> camera: Camera;

struct VOut {
    @builtin(position) clip_pos: vec4<f32>,
    @location(0) color: vec4<f32>,
}

const COLORS = array<vec4<f32>, 4>(
    vec4<f32>(0.90, 0.22, 0.22, 1.0),   // red
    vec4<f32>(0.22, 0.42, 0.90, 1.0),   // blue
    vec4<f32>(0.22, 0.80, 0.35, 1.0),   // green
    vec4<f32>(0.92, 0.82, 0.12, 1.0),   // yellow
);

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

    let idx = min(inst_color_id, 3u);
    var base_color = COLORS[idx];

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
    bind_group: wgpu::BindGroup,
    // Board background
    board_pipeline: wgpu::RenderPipeline,
    board_vertex_buf: wgpu::Buffer,
    board_index_buf: wgpu::Buffer,
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
    pub async fn new(window: Arc<Window>) -> Self {
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

        let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label:   Some("camera bgl"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding:    0,
                visibility: wgpu::ShaderStages::VERTEX,
                ty: wgpu::BindingType::Buffer {
                    ty:                 wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size:   None,
                },
                count: None,
            }],
        });

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label:   Some("camera bg"),
            layout:  &bgl,
            entries: &[wgpu::BindGroupEntry {
                binding:  0,
                resource: camera_buf.as_entire_binding(),
            }],
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
            bind_group,
            board_pipeline,
            board_vertex_buf,
            board_index_buf,
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
