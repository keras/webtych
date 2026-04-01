//! Interactive LBM debug visualizer.
//!
//! Opens a window showing the density field in real-time, with an egui
//! sidebar for tweaking simulation parameters.
//!
//! Controls:
//!   - Left-click on the field  → inject a pressure event at that position
//!   - egui panel on the right  → τ, intensity, pause/play, reset, stats
//!
//! Run with:
//!   cargo run --example visualizer -p webtych-lbm

use std::sync::Arc;

use egui_wgpu::ScreenDescriptor;
use wgpu::util::DeviceExt;
use winit::{
    application::ApplicationHandler,
    dpi::PhysicalSize,
    event::{ElementState, MouseButton, WindowEvent},
    event_loop::{ActiveEventLoop, EventLoop},
    window::{Window, WindowId},
};

use webtych_lbm::{
    config::{EffectProfile, SimConfig},
    types::{EventKind, InjectionEvent, ObstaclePatch, OpenBoundaryPatch},
    Simulation,
};

// ── Grid constants ────────────────────────────────────────────────────────────

const GRID_W: u32 = 128;
const GRID_H: u32 = 192;

// ── WGSL: fullscreen density visualisation ────────────────────────────────────

const VIS_SHADER: &str = r#"
// Vertex stage: emit a fullscreen triangle.
@vertex
fn vs_main(@builtin(vertex_index) vi: u32) -> @builtin(position) vec4<f32> {
    // Three vertices that cover the whole clip space.
    var pos = array<vec2<f32>, 3>(
        vec2<f32>(-1.0, -1.0),
        vec2<f32>( 3.0, -1.0),
        vec2<f32>(-1.0,  3.0),
    );
    return vec4<f32>(pos[vi], 0.0, 1.0);
}

// uniforms
struct Vis {
    grid_w:   u32,
    grid_h:   u32,
    win_w:    u32,   // surface width  in physical pixels
    win_h:    u32,   // surface height in physical pixels
    peak_rho: f32,
    _pad0:    u32,
    _pad1:    u32,
    _pad2:    u32,
}
@group(0) @binding(0) var<uniform> vis: Vis;

// macroscopic buffer: [rho, ux, uy] per cell
@group(0) @binding(1) var<storage, read> macro_buf: array<f32>;

// obstacle texture: R channel = 1.0 for solid cells
@group(0) @binding(2) var obstacle_tex: texture_2d<f32>;

// ── Oklch → linear sRGB ───────────────────────────────────────────────────────
// Oklch: L = lightness [0,1], C = chroma [0,~0.37], H = hue angle (radians)
// Hue rotation at constant L and C is perceptually uniform (no brightness shift).
fn oklch_to_linear_srgb(L: f32, C: f32, H: f32) -> vec3<f32> {
    let a = C * cos(H);
    let b = C * sin(H);

    // Oklab → LMS (cube roots)
    let l_ = L + 0.3963377774 * a + 0.2158037573 * b;
    let m_ = L - 0.1055613458 * a - 0.0638541728 * b;
    let s_ = L - 0.0894841775 * a - 1.2914855480 * b;

    let l = l_ * l_ * l_;
    let m = m_ * m_ * m_;
    let s = s_ * s_ * s_;

    // LMS → linear sRGB
    return vec3<f32>(
         4.0767416621 * l - 3.3077115913 * m + 0.2309699292 * s,
        -1.2684380046 * l + 2.6097574011 * m - 0.3413193965 * s,
        -0.0041960863 * l - 0.7034186147 * m + 1.7076147010 * s,
    );
}

fn linear_to_srgb(c: f32) -> f32 {
    if c <= 0.0031308 {
        return 12.92 * c;
    }
    return 1.055 * pow(max(c, 0.0), 1.0 / 2.4) - 0.055;
}

// Fragment stage: encode fluid state in Oklch.
//   Hue       → velocity direction (atan2)
//   Chroma    → speed  (0 = grey, saturates at ~0.04 LBM units)
//   Lightness → density (low rho = dark, high rho = bright)
@fragment
fn fs_main(@builtin(position) frag: vec4<f32>) -> @location(0) vec4<f32> {
    // Map physical pixel → grid cell using actual framebuffer dimensions.
    let cx = min(u32(frag.x) * vis.grid_w / vis.win_w, vis.grid_w - 1u);
    let cy = min(u32(frag.y) * vis.grid_h / vis.win_h, vis.grid_h - 1u);
    let cell = cy * vis.grid_w + cx;

    // Sample obstacle fill fraction (PSM); fully solid cells render as flat grey.
    let obs = textureLoad(obstacle_tex, vec2<i32>(i32(cx), i32(cy)), 0);
    let solid_frac = obs.r;
    if solid_frac > 0.999 {
        return vec4<f32>(0.8, obs.g + 0.5, obs.b + 0.5, 1.0);
    }

    // Open-boundary (sink) cells render as green.
    if obs.a > 0.5 {
        return vec4<f32>(0.1, 0.8, 0.2, 1.0);
    }

    let rho = macro_buf[cell * 3u];
    let ux  = macro_buf[cell * 3u + 1u];
    let uy  = macro_buf[cell * 3u + 2u];

    // Lightness: density mapped to [0.25, 0.90].
    let hi = max(vis.peak_rho, 1.001);
    let t = clamp((rho - 0.95) / (hi - 0.95), 0.0, 1.0);
    let L = 0 + 0.65 * t;

    // Chroma: speed, saturates around 0.04 LBM units.
    let speed = sqrt(ux * ux + uy * uy);
    let C = clamp(speed * 25.0, 0.0, 0.33);

    // Hue: direction of velocity (atan2 gives [-π, π], that's fine for cos/sin).
    let H = atan2(uy, ux);

    let linear_rgb = oklch_to_linear_srgb(L, C, H);

    // Gamma-encode and clamp (Oklch can produce small out-of-gamut values).
    var rgb = vec3<f32>(
        clamp(linear_to_srgb(linear_rgb.x), 0.0, 1.0),
        clamp(linear_to_srgb(linear_rgb.y), 0.0, 1.0),
        clamp(linear_to_srgb(linear_rgb.z), 0.0, 1.0),
    );

    // Blend fluid colour with obstacle grey by PSM fill fraction.
    let grey = vec3<f32>(0.8, 0.8, 0.85);
    rgb = mix(rgb, grey, solid_frac);
    return vec4<f32>(rgb, 1.0);
}
"#;

// ── App state ─────────────────────────────────────────────────────────────────

struct AppState {
    // wgpu
    device: Arc<wgpu::Device>,
    queue: Arc<wgpu::Queue>,
    surface: wgpu::Surface<'static>,
    surface_config: wgpu::SurfaceConfiguration,

    // visualisation pipeline
    vis_pipeline: wgpu::RenderPipeline,
    vis_uniform_buf: wgpu::Buffer,
    vis_bind_group: wgpu::BindGroup,
    vis_bind_layout: wgpu::BindGroupLayout,

    staging_buf: wgpu::Buffer,      // MAP_READ copy of macroscopic
    dist_staging_buf: wgpu::Buffer, // MAP_READ copy of distributions

    // egui
    egui_ctx: egui::Context,
    egui_renderer: egui_wgpu::Renderer,
    egui_winit: egui_winit::State,

    // simulation
    sim: Simulation,
    paused: bool,
    pending_step: bool,
    step_count: u64,
    tau_ui: f32, // τ as edited in egui (applied on change)
    intensity_ui: f32,
    peak_rho: f32,
    nonambient: u32,
    mouse_pos: Option<(f32, f32)>, // normalised 0-1 in window space
    block_phase: f32,              // oscillation angle (radians)
    block_speed_ui: f32,           // phase increment per step
    gravity_ui: f32,               // gravity_y fed to sim config
    additive_injection_ui: bool,
    substeps_ui: u32,
    macroscopic_data: Vec<f32>, // last readback of [rho, ux, uy] per cell
    dist_data: Vec<f32>,        // last readback of 9 distribution values per cell
    max_speed: f32,             // maximum fluid speed observed in the last readback

    window: Arc<Window>,
}

// ── Visualiser uniform (matches WGSL) ─────────────────────────────────────────

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct VisUniforms {
    grid_w: u32,
    grid_h: u32,
    win_w: u32,
    win_h: u32,
    peak_rho: f32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

// ── Application ──────────────────────────────────────────────────────────────

struct App {
    state: Option<AppState>,
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.state.is_some() {
            return;
        }

        let window = Arc::new(
            event_loop
                .create_window(
                    winit::window::WindowAttributes::default()
                        .with_title("LBM Visualizer")
                        .with_inner_size(PhysicalSize::new(600u32, 900u32)),
                )
                .expect("create window"),
        );

        let state = pollster::block_on(init_gpu(Arc::clone(&window)));
        self.state = Some(state);
    }

    fn window_event(&mut self, event_loop: &ActiveEventLoop, _id: WindowId, event: WindowEvent) {
        let Some(state) = self.state.as_mut() else {
            return;
        };

        // Forward to egui first.
        let egui_resp = state.egui_winit.on_window_event(&state.window, &event);

        // Always track cursor position — egui often marks CursorMoved as consumed
        // even over the field area, which would leave mouse_pos stale.
        // Normalise x relative to the field area width (not the full window) so
        // that all downstream consumers get 0..1 in field-space directly.
        if let WindowEvent::CursorMoved { position, .. } = &event {
            let size = state.window.inner_size();
            let panel_phys = (200.0 * state.window.scale_factor()) as u32;
            let field_w = size.width.saturating_sub(panel_phys).max(1);
            state.mouse_pos = Some((
                position.x as f32 / field_w as f32,
                position.y as f32 / size.height.max(1) as f32,
            ));
        }

        // Inject if the click is in the field area (nx < 1.0 means left of panel).
        if let WindowEvent::MouseInput {
            state: ElementState::Pressed,
            button: MouseButton::Left,
            ..
        } = &event
        {
            if let Some((nx, ny)) = state.mouse_pos {
                if nx < 1.0 {
                    let x = nx * state.sim.config.world_width;
                    let y = ny * state.sim.config.world_height;
                    state.sim.push_event(InjectionEvent {
                        x,
                        y,
                        intensity: state.intensity_ui,
                        stamp_radius: 0.6,
                        color_id: 0,
                        kind: EventKind::Destroy,
                        velocity_scale: 0.06,
                        base_vel_x: 0.0,
                        base_vel_y: 0.0,
                    });
                }
            }
        }

        if egui_resp.consumed {
            return;
        }

        match &event {
            WindowEvent::CloseRequested => event_loop.exit(),

            WindowEvent::Resized(size) => {
                state.surface_config.width = size.width.max(1);
                state.surface_config.height = size.height.max(1);
                state
                    .surface
                    .configure(&state.device, &state.surface_config);
            }

            WindowEvent::RedrawRequested => {
                update_and_render(state);
                state.window.request_redraw(); // continuous loop
            }

            _ => {}
        }
    }
}

// ── GPU Initialisation ────────────────────────────────────────────────────────

async fn init_gpu(window: Arc<Window>) -> AppState {
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
        backends: wgpu::Backends::all(),
        flags: wgpu::InstanceFlags::default(),
        memory_budget_thresholds: wgpu::MemoryBudgetThresholds::default(),
        backend_options: wgpu::BackendOptions::default(),
        display: None,
    });

    let surface = instance
        .create_surface(Arc::clone(&window))
        .expect("create surface");

    let adapter = instance
        .request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: Some(&surface),
            force_fallback_adapter: false,
        })
        .await
        .expect("no adapter found");

    println!("Adapter : {}", adapter.get_info().name);
    println!("Backend : {:?}", adapter.get_info().backend);

    let (device, queue) = adapter
        .request_device(&wgpu::DeviceDescriptor {
            label: Some("lbm-vis"),
            required_features: wgpu::Features::empty(),
            required_limits: wgpu::Limits::default(),
            experimental_features: Default::default(),
            memory_hints: wgpu::MemoryHints::default(),
            trace: Default::default(),
        })
        .await
        .expect("device creation failed");

    let device = Arc::new(device);
    let queue = Arc::new(queue);

    // ── Surface config ───────────────────────────────────────────────────
    let phys = window.inner_size();
    let surface_caps = surface.get_capabilities(&adapter);
    let surface_fmt = surface_caps
        .formats
        .iter()
        .copied()
        .find(|f| f.is_srgb())
        .unwrap_or(surface_caps.formats[0]);

    let surface_config = wgpu::SurfaceConfiguration {
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
        format: surface_fmt,
        width: phys.width,
        height: phys.height,
        present_mode: wgpu::PresentMode::AutoVsync,
        alpha_mode: surface_caps.alpha_modes[0],
        view_formats: vec![],
        desired_maximum_frame_latency: 2,
    };
    surface.configure(&device, &surface_config);

    // ── Simulation ────────────────────────────────────────────────────────
    let tau = 0.7_f32;
    let config = SimConfig {
        grid_width: GRID_W,
        grid_height: GRID_H,
        tau,
        mrt_s_e: 1.0,
        mrt_s_q: None,
        world_width: 10.0,
        world_height: 15.0, // portrait 2:3 world matching the 128×192 grid
        color_count: 1,
        substeps: 1,
        gravity_x: 0.0,
        gravity_y: 0.0003, // default: gentle downward pull
        effect_profiles: vec![EffectProfile {
            inject_density: 0.5, // mild injection — avoids Mach-limit overshoot
            inject_color_density: 0.3,
            dissipation: 0.995,
        }],
    };
    let mut sim = Simulation::new(&device, config);

    // Mark the top row as an open outflow boundary so injected density can drain
    // upward under gravity instead of accumulating indefinitely.
    let top_cell_h = sim.config.world_height / sim.config.grid_height as f32;
    sim.set_open_boundaries(&[OpenBoundaryPatch {
        x_min: 0.0,
        y_min: 0.0,
        x_max: sim.config.world_width,
        y_max: top_cell_h,
    }]);

    // ── Staging buffer for macroscopic readback ───────────────────────────
    let macro_size = sim.macroscopic_buffer().size();
    let staging_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("vis-staging"),
        size: macro_size,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    // ── Staging buffer for distribution readback ──────────────────────────
    let dist_size = sim.grid.src_distributions().size();
    let dist_staging_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("vis-dist-staging"),
        size: dist_size,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    // ── Visualisation pipeline ────────────────────────────────────────────
    let vis_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("vis-shader"),
        source: wgpu::ShaderSource::Wgsl(VIS_SHADER.into()),
    });

    let vis_uniform_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("vis-uniforms"),
        contents: bytemuck::bytes_of(&VisUniforms {
            grid_w: GRID_W,
            grid_h: GRID_H,
            win_w: phys.width,
            win_h: phys.height,
            peak_rho: 1.05,
            _pad0: 0,
            _pad1: 0,
            _pad2: 0,
        }),
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
    });

    let vis_bind_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("vis-bgl"),
        entries: &[
            wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 1,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 2,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Texture {
                    sample_type: wgpu::TextureSampleType::Float { filterable: false },
                    view_dimension: wgpu::TextureViewDimension::D2,
                    multisampled: false,
                },
                count: None,
            },
        ],
    });

    let vis_bind_group = make_vis_bind_group(
        &device,
        &vis_bind_layout,
        &vis_uniform_buf,
        sim.macroscopic_buffer(),
        &sim.grid.obstacle_texture_view,
    );

    let vis_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("vis-pl"),
        bind_group_layouts: &[Some(&vis_bind_layout)],
        immediate_size: 0,
    });

    let vis_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("vis-rp"),
        layout: Some(&vis_pipeline_layout),
        vertex: wgpu::VertexState {
            module: &vis_shader,
            entry_point: Some("vs_main"),
            buffers: &[],
            compilation_options: Default::default(),
        },
        fragment: Some(wgpu::FragmentState {
            module: &vis_shader,
            entry_point: Some("fs_main"),
            targets: &[Some(wgpu::ColorTargetState {
                format: surface_fmt,
                blend: Some(wgpu::BlendState::REPLACE),
                write_mask: wgpu::ColorWrites::ALL,
            })],
            compilation_options: Default::default(),
        }),
        primitive: wgpu::PrimitiveState::default(),
        depth_stencil: None,
        multisample: wgpu::MultisampleState::default(),
        multiview_mask: None,
        cache: None,
    });

    // ── egui ──────────────────────────────────────────────────────────────
    let egui_ctx = egui::Context::default();
    let egui_winit = egui_winit::State::new(
        egui_ctx.clone(),
        egui::ViewportId::ROOT,
        &window,
        None,
        None,
        None,
    );
    let egui_renderer =
        egui_wgpu::Renderer::new(&device, surface_fmt, egui_wgpu::RendererOptions::default());

    AppState {
        device,
        queue,
        surface,
        surface_config,
        vis_pipeline,
        vis_uniform_buf,
        vis_bind_group,
        vis_bind_layout,
        staging_buf,
        dist_staging_buf,
        egui_ctx,
        egui_renderer,
        egui_winit,
        sim,
        paused: false,
        pending_step: false,
        step_count: 0,
        tau_ui: tau,
        intensity_ui: 1.0,
        peak_rho: 1.1, // start above ambient so colormap isn't blown-out on frame 1
        nonambient: 0,
        mouse_pos: None,
        block_phase: 0.0,
        block_speed_ui: 0.005,
        gravity_ui: 0.0003,
        additive_injection_ui: true,
        substeps_ui: 1,
        macroscopic_data: vec![1.0f32; (GRID_W * GRID_H * 3) as usize],
        dist_data: vec![0.0f32; (GRID_W * GRID_H * 9) as usize],
        max_speed: 0.0,
        window,
    }
}

fn make_vis_bind_group(
    device: &wgpu::Device,
    layout: &wgpu::BindGroupLayout,
    uniform_buf: &wgpu::Buffer,
    macro_buf: &wgpu::Buffer,
    obstacle_view: &wgpu::TextureView,
) -> wgpu::BindGroup {
    device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("vis-bg"),
        layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: uniform_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: macro_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: wgpu::BindingResource::TextureView(obstacle_view),
            },
        ],
    })
}

// ── Per-frame update + render ─────────────────────────────────────────────────

fn update_and_render(state: &mut AppState) {
    // ── Apply config changes from UI ──────────────────────────────────────
    if (state.tau_ui - state.sim.config.tau).abs() > 1e-4 {
        state.sim.config.tau = state.tau_ui;
    }
    state.sim.config.gravity_y = state.gravity_ui;
    state.sim.config.substeps = state.substeps_ui;
    state
        .sim
        .set_additive_injection(state.additive_injection_ui);

    // ── Determine whether to advance the simulation this frame ───────────
    let was_pending_step = state.pending_step;
    let should_step = !state.paused || was_pending_step;
    state.pending_step = false;

    // ── Advance oscillating block (narrow vertical bar near the bottom) ───
    // The block moves left-right sinusoidally.  Its world velocity is
    // converted to lattice units and stored in the obstacle texel so that
    // the boundary pass applies moving bounce-back momentum transfer.
    if should_step {
        state.block_phase += state.block_speed_ui;
    }
    let block_cx = 5.0_f32 + 2.5 * state.block_phase.sin();
    let block_vel_w = if state.paused && !was_pending_step {
        0.0_f32
    } else {
        2.5 * state.block_speed_ui * state.block_phase.cos()
    };
    // world-units/step → lattice-units/step
    let block_vel_l = block_vel_w * GRID_W as f32 / 10.0;
    state.sim.set_obstacles(
        &state.queue,
        &[
            ObstaclePatch {
                // oscillating bar — narrow (0.5 wu) and tall (2 wu)
                x_min: block_cx - 0.25,
                y_min: 12.5,
                x_max: block_cx + 0.25,
                y_max: 14.5,
                vel_x: block_vel_l,
                vel_y: 0.0,
                rotation: 0.0,
                inset: false,
            },
            ObstaclePatch {
                // solid floor so injected density pools visibly
                x_min: 0.0,
                y_min: 14.7,
                x_max: 10.0,
                y_max: 15.0,
                vel_x: 0.0,
                vel_y: 0.0,
                rotation: 0.0,
                inset: false,
            },
        ],
    );

    // ── Continuous top-right injector ─────────────────────────────────────
    if should_step && state.step_count % 4 == 0 {
        state.sim.push_event(InjectionEvent {
            x: 7.5,
            y: 1.5,
            intensity: state.intensity_ui,
            stamp_radius: 0.85,
            color_id: 0,
            kind: EventKind::Destroy,
            velocity_scale: 0.08,
            base_vel_x: 0.0,
            base_vel_y: 0.02,
        });
    }

    // ── Advance simulation ────────────────────────────────────────────────
    if should_step {
        state.sim.step(&state.device, &state.queue);
        state.step_count += 1;
    }

    // ── Non-blocking readback for stats (every 10 steps or so) ───────────
    if state.step_count % 10 == 1 || state.paused {
        do_readback(state);
    }

    // ── Update vis uniform (window size may have changed on resize) ─────────
    let phys = state.window.inner_size();
    let panel_phys = (200.0 * state.window.scale_factor()) as u32;
    let field_w = phys.width.saturating_sub(panel_phys).max(1);
    state.queue.write_buffer(
        &state.vis_uniform_buf,
        0,
        bytemuck::bytes_of(&VisUniforms {
            grid_w: GRID_W,
            grid_h: GRID_H,
            win_w: field_w,
            win_h: phys.height.max(1),
            peak_rho: (state.peak_rho * 1.1).max(1.01),
            _pad0: 0,
            _pad1: 0,
            _pad2: 0,
        }),
    );

    // ── Acquire surface texture FIRST ─────────────────────────────────────
    // Do this before building the egui frame so we never process egui
    // texture deltas without actually uploading them (causes a panic on
    // the next frame when egui tries to free a never-allocated texture).
    let output = match state.surface.get_current_texture() {
        wgpu::CurrentSurfaceTexture::Success(t) | wgpu::CurrentSurfaceTexture::Suboptimal(t) => t,
        _ => {
            // Window is occluded / not ready yet — skip this frame entirely.
            return;
        }
    };
    let view = output
        .texture
        .create_view(&wgpu::TextureViewDescriptor::default());

    // ── egui: build UI ────────────────────────────────────────────────────
    let raw_input = state.egui_winit.take_egui_input(&state.window);

    let tau_ref = &mut state.tau_ui;
    let intensity_ref = &mut state.intensity_ui;
    let paused_ref = &mut state.paused;
    let gravity_ref = &mut state.gravity_ui;
    let block_speed_ref = &mut state.block_speed_ui;
    let additive_injection_ref = &mut state.additive_injection_ui;
    let substeps_ref = &mut state.substeps_ui;
    let step_count_ref = state.step_count;
    let peak_rho_ref = state.peak_rho;
    let nonambient_ref = state.nonambient;
    let max_speed_ref = state.max_speed;
    let sim_tau_ref = state.sim.config.tau;
    let mut do_reset = false;
    let mut do_step = false;

    let full_output = state.egui_ctx.run_ui(raw_input, |ui| {
        egui::Panel::right("controls")
            .exact_size(200.0)
            .show_inside(ui, |ui| {
                ui.heading("LBM Visualizer");
                ui.separator();

                ui.label("Relaxation τ");
                ui.add(egui::Slider::new(tau_ref, 0.51..=2.0).step_by(0.01));

                ui.label("Inject intensity");
                ui.add(egui::Slider::new(intensity_ref, 0.1..=5.0).step_by(0.1));

                ui.label("Gravity");
                ui.add(egui::Slider::new(gravity_ref, 0.0..=0.002).step_by(0.00005));

                ui.label("Block speed");
                ui.add(egui::Slider::new(block_speed_ref, 0.0..=0.025).step_by(0.001));

                ui.label("Substeps");
                ui.add(egui::Slider::new(substeps_ref, 1..=8).step_by(1.0));

                ui.checkbox(additive_injection_ref, "Additive injection");

                ui.separator();
                if ui
                    .button(if *paused_ref { "▶ Play" } else { "⏸ Pause" })
                    .clicked()
                {
                    *paused_ref = !*paused_ref;
                }
                if ui.button("⏭ Step").clicked() {
                    do_step = true;
                }
                if ui.button("⟳ Reset").clicked() {
                    do_reset = true;
                }

                ui.separator();
                ui.label(format!("Step : {step_count_ref}"));
                ui.label(format!("Peak ρ : {peak_rho_ref:.4}"));
                ui.label(format!("Max speed : {max_speed_ref:.5}"));
                ui.label(format!("Non-ambient cells : {nonambient_ref}"));
                ui.label(format!("Grid : {}×{}", GRID_W, GRID_H));
                ui.label(format!("τ (active) : {sim_tau_ref:.3}"));

                ui.separator();
                ui.label("Cell under cursor:");
                if let Some((nx, ny)) = state.mouse_pos.filter(|(nx, _)| *nx < 1.0) {
                    let gx = ((nx * GRID_W as f32).floor() as u32).min(GRID_W - 1);
                    let gy = ((ny * GRID_H as f32).floor() as u32).min(GRID_H - 1);
                    let cell_idx = (gy * GRID_W + gx) as usize;
                    let rho = state.macroscopic_data[cell_idx * 3];
                    let ux = state.macroscopic_data[cell_idx * 3 + 1];
                    let uy = state.macroscopic_data[cell_idx * 3 + 2];
                    ui.label(format!("  ({}, {})", gx, gy));
                    ui.label(format!("  ρ : {:.4}", rho));
                    ui.label(format!("  u_x : {:.4}", ux));
                    ui.label(format!("  u_y : {:.4}", uy));

                    // D2Q9 distribution: 3×3 grid
                    // directions: 0=(0,0) 1=(+1,0) 2=(0,+1) 3=(-1,0) 4=(0,-1)
                    //             5=(+1,+1) 6=(-1,+1) 7=(-1,-1) 8=(+1,-1)
                    // display layout (row 0=top):
                    //   [6] [2] [5]   NW  N  NE
                    //   [3] [0] [1]   W   C  E
                    //   [7] [4] [8]   SW  S  SE
                    ui.add_space(4.0);
                    ui.label("D2Q9 fᵢ:");
                    let base = cell_idx * 9;
                    let f = |i: usize| -> f32 {
                        let v = state.dist_data.get(base + i).copied().unwrap_or(0.0);
                        (v * 10000.0).round() / 10000.0
                    };
                    let display_order: [[usize; 3]; 3] = [[6, 2, 5], [3, 0, 1], [7, 4, 8]];
                    egui::Grid::new("d2q9_grid")
                        .num_columns(3)
                        .spacing([2.0, 2.0])
                        .show(ui, |ui| {
                            for row in &display_order {
                                for &dir in row {
                                    ui.label(
                                        egui::RichText::new(format!("{:.4}", f(dir)))
                                            .monospace()
                                            .size(9.5),
                                    );
                                }
                                ui.end_row();
                            }
                        });
                } else {
                    ui.label("  (move cursor over field)");
                }

                ui.separator();
                ui.label("Left-click on field to inject");
            });
    });

    // Apply deferred actions.
    if do_step {
        state.pending_step = true;
    }
    if do_reset {
        let cfg = state.sim.config.clone();
        state.sim = Simulation::new(&state.device, cfg);
        state
            .sim
            .set_additive_injection(state.additive_injection_ui);
        let top_cell_h = state.sim.config.world_height / state.sim.config.grid_height as f32;
        state.sim.set_open_boundaries(&[OpenBoundaryPatch {
            x_min: 0.0,
            y_min: 0.0,
            x_max: state.sim.config.world_width,
            y_max: top_cell_h,
        }]);
        state.vis_bind_group = make_vis_bind_group(
            &state.device,
            &state.vis_bind_layout,
            &state.vis_uniform_buf,
            state.sim.macroscopic_buffer(),
            &state.sim.grid.obstacle_texture_view,
        );
        state.step_count = 0;
        state.peak_rho = 1.0;
        state.nonambient = 0;
        state.max_speed = 0.0;
        state.block_phase = 0.0;
    }

    // ── Build encoder ─────────────────────────────────────────────────────
    let mut encoder = state
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("vis-frame"),
        });

    // ── Visualisation render pass ─────────────────────────────────────────
    {
        let mut rp = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("vis-pass"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: &view,
                resolve_target: None,
                depth_slice: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                    store: wgpu::StoreOp::Store,
                },
            })],
            depth_stencil_attachment: None,
            ..Default::default()
        });
        rp.set_scissor_rect(0, 0, field_w, phys.height.max(1));
        rp.set_pipeline(&state.vis_pipeline);
        rp.set_bind_group(0, &state.vis_bind_group, &[]);
        rp.draw(0..3, 0..1); // fullscreen triangle (clipped to field area)
    }

    // ── egui render pass ─────────────────────────────────────────────────
    let phys = state.window.inner_size();
    let screen_desc = ScreenDescriptor {
        size_in_pixels: [phys.width, phys.height],
        pixels_per_point: state.window.scale_factor() as f32,
    };

    state
        .egui_winit
        .handle_platform_output(&state.window, full_output.platform_output);

    let egui_tris = state
        .egui_ctx
        .tessellate(full_output.shapes, full_output.pixels_per_point);

    for (id, img) in &full_output.textures_delta.set {
        state
            .egui_renderer
            .update_texture(&state.device, &state.queue, *id, img);
    }
    state.egui_renderer.update_buffers(
        &state.device,
        &state.queue,
        &mut encoder,
        &egui_tris,
        &screen_desc,
    );

    {
        let rp = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("egui-pass"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: &view,
                resolve_target: None,
                depth_slice: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Load, // composite over the vis
                    store: wgpu::StoreOp::Store,
                },
            })],
            depth_stencil_attachment: None,
            ..Default::default()
        });
        // forget_lifetime() converts RenderPass<'encoder> → RenderPass<'static>
        // which is required by egui_renderer.render().
        let mut rp = rp.forget_lifetime();
        state
            .egui_renderer
            .render(&mut rp, &egui_tris, &screen_desc);
    }

    for id in &full_output.textures_delta.free {
        state.egui_renderer.free_texture(id);
    }

    state.queue.submit(std::iter::once(encoder.finish()));
    output.present();
}

// ── Non-blocking GPU → CPU readback for stats ────────────────────────────────

fn do_readback(state: &mut AppState) {
    let macro_buf = state.sim.macroscopic_buffer();
    let buf_size = macro_buf.size();
    let dist_buf = state.sim.grid.src_distributions();
    let dist_size = dist_buf.size();

    // Copy macroscopic + distributions → staging buffers.
    let mut enc = state
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("readback-enc"),
        });
    enc.copy_buffer_to_buffer(macro_buf, 0, &state.staging_buf, 0, buf_size);
    enc.copy_buffer_to_buffer(dist_buf, 0, &state.dist_staging_buf, 0, dist_size);
    state.queue.submit(std::iter::once(enc.finish()));

    // Blocking poll (fast — Metal/Vulkan, no window stall).
    let _ = state.device.poll(wgpu::PollType::wait_indefinitely());

    let (tx, rx) = std::sync::mpsc::channel();
    state
        .staging_buf
        .slice(..)
        .map_async(wgpu::MapMode::Read, move |r| {
            let _ = tx.send(r);
        });
    let _ = state.device.poll(wgpu::PollType::wait_indefinitely());

    if rx.recv().map(|r| r.is_ok()).unwrap_or(false) {
        let data: Vec<f32> = {
            let view = state.staging_buf.slice(..).get_mapped_range();
            bytemuck::cast_slice::<u8, f32>(&view).to_vec()
        };
        state.staging_buf.unmap();

        let n = (GRID_W * GRID_H) as usize;
        let mut peak = 0.0_f32;
        let mut na = 0u32;
        let mut max_speed = 0.0_f32;
        for cell in 0..n {
            let rho = data[cell * 3];
            let ux = data[cell * 3 + 1];
            let uy = data[cell * 3 + 2];
            if rho > peak {
                peak = rho;
            }
            if (rho - 1.0).abs() > 0.01 {
                na += 1;
            }
            let speed = (ux * ux + uy * uy).sqrt();
            if speed > max_speed {
                max_speed = speed;
            }
        }
        state.peak_rho = peak;
        state.nonambient = na;
        state.max_speed = max_speed;
        state.macroscopic_data = data;
    }

    // Read distribution buffer.
    let (dtx, drx) = std::sync::mpsc::channel();
    state
        .dist_staging_buf
        .slice(..)
        .map_async(wgpu::MapMode::Read, move |r| {
            let _ = dtx.send(r);
        });
    let _ = state.device.poll(wgpu::PollType::wait_indefinitely());

    if drx.recv().map(|r| r.is_ok()).unwrap_or(false) {
        let data: Vec<f32> = {
            let view = state.dist_staging_buf.slice(..).get_mapped_range();
            bytemuck::cast_slice::<u8, f32>(&view).to_vec()
        };
        state.dist_staging_buf.unmap();
        state.dist_data = data;
    }
}

// ── Entry point ───────────────────────────────────────────────────────────────

fn main() {
    env_logger::init();

    let event_loop = EventLoop::new().expect("create event loop");
    let mut app = App { state: None };
    event_loop.run_app(&mut app).expect("run");
}
