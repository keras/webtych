use std::sync::Arc;
use wgpu::util::DeviceExt;
use winit::window::Window;

// ── WGSL shader ───────────────────────────────────────────────────────────────

const SHADER: &str = r#"
struct Uniforms {
    time: f32,
}
@group(0) @binding(0) var<uniform> u: Uniforms;

struct VOut {
    @builtin(position) pos:   vec4<f32>,
    @location(0)       color: vec3<f32>,
}

// Equilateral triangle, rotated by u.time.
@vertex
fn vs_main(@builtin(vertex_index) vi: u32) -> VOut {
    const PI2: f32 = 6.2831853;
    let base_angle = f32(vi) * (PI2 / 3.0) + PI2 / 4.0;
    let t = u.time;
    let angle = base_angle + t;
    let r = 0.55;
    let pos = vec2<f32>(cos(angle) * r, sin(angle) * r);

    var colors = array<vec3<f32>, 3>(
        vec3<f32>(1.0, 0.25, 0.25),
        vec3<f32>(0.25, 1.0, 0.45),
        vec3<f32>(0.25, 0.55, 1.0),
    );

    var out: VOut;
    out.pos   = vec4<f32>(pos, 0.0, 1.0);
    out.color = colors[vi];
    return out;
}

@fragment
fn fs_main(in: VOut) -> @location(0) vec4<f32> {
    return vec4<f32>(in.color, 1.0);
}
"#;

// ── Renderer ──────────────────────────────────────────────────────────────────

pub struct Renderer {
    surface:         wgpu::Surface<'static>,
    device:          wgpu::Device,
    queue:           wgpu::Queue,
    config:          wgpu::SurfaceConfiguration,
    pipeline:        wgpu::RenderPipeline,
    uniform_buf:     wgpu::Buffer,
    bind_group:      wgpu::BindGroup,
    frame:           u64,
}

impl Renderer {
    pub async fn new(window: Arc<Window>) -> Self {
        let size = window.inner_size();
        let width  = size.width.max(1);
        let height = size.height.max(1);

        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
            backends: wgpu::Backends::all(),
            ..Default::default()
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
                 Please use Chrome 113+ or another WebGPU-capable browser.",
            );

        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    label:             None,
                    required_features: wgpu::Features::empty(),
                    required_limits:   wgpu::Limits::default(),
                    memory_hints:      wgpu::MemoryHints::default(),
                },
                None,
            )
            .await
            .expect("request device");

        // ── surface configuration ────────────────────────────────────────────
        let caps   = surface.get_capabilities(&adapter);
        let format = caps
            .formats
            .iter()
            .copied()
            .find(|f| f.is_srgb())
            .unwrap_or(caps.formats[0]);

        let config = wgpu::SurfaceConfiguration {
            usage:                        wgpu::TextureUsages::RENDER_ATTACHMENT,
            format,
            width,
            height,
            present_mode:                 wgpu::PresentMode::AutoVsync,
            alpha_mode:                   caps.alpha_modes[0],
            view_formats:                 vec![],
            desired_maximum_frame_latency: 2,
        };
        surface.configure(&device, &config);

        // ── uniform buffer (f32 time) ────────────────────────────────────────
        let uniform_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label:    Some("uniforms"),
            contents: bytemuck::bytes_of(&0.0f32),
            usage:    wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label:   Some("bgl"),
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
            label:   Some("bg"),
            layout:  &bgl,
            entries: &[wgpu::BindGroupEntry {
                binding:  0,
                resource: uniform_buf.as_entire_binding(),
            }],
        });

        // ── render pipeline ──────────────────────────────────────────────────
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label:  Some("shader"),
            source: wgpu::ShaderSource::Wgsl(SHADER.into()),
        });

        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label:                Some("layout"),
            bind_group_layouts:   &[&bgl],
            push_constant_ranges: &[],
        });

        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label:  Some("pipeline"),
            layout: Some(&layout),
            vertex: wgpu::VertexState {
                module:               &shader,
                entry_point:          Some("vs_main"),
                buffers:              &[],
                compilation_options:  Default::default(),
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
            multiview:     None,
            cache:         None,
        });

        Self {
            surface,
            device,
            queue,
            config,
            pipeline,
            uniform_buf,
            bind_group,
            frame: 0,
        }
    }

    pub fn resize(&mut self, size: winit::dpi::PhysicalSize<u32>) {
        let w = size.width.max(1);
        let h = size.height.max(1);
        self.config.width  = w;
        self.config.height = h;
        self.surface.configure(&self.device, &self.config);
    }

    pub fn update(&mut self) {
        self.frame += 1;
        // ~60 fps assumed; gives one full revolution per ~6 seconds.
        let time: f32 = self.frame as f32 / 60.0;
        self.queue.write_buffer(&self.uniform_buf, 0, bytemuck::bytes_of(&time));
    }

    pub fn render(&self) -> Result<(), wgpu::SurfaceError> {
        let output = self.surface.get_current_texture()?;
        let view   = output
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
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color {
                            r: 0.04,
                            g: 0.04,
                            b: 0.08,
                            a: 1.0,
                        }),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                occlusion_query_set:      None,
                timestamp_writes:         None,
            });

            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &self.bind_group, &[]);
            pass.draw(0..3, 0..1);
        }

        self.queue.submit(std::iter::once(enc.finish()));
        output.present();
        Ok(())
    }
}
