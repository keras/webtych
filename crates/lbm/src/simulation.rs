//! Top-level [`Simulation`] struct — owns all GPU resources and orchestrates
//! the per-frame update sequence.
//!
//! # Typical usage
//!
//! ```no_run
//! // --- Setup (once) ---
//! let config = SimConfig::for_game_board(10.0, 20.0, 5);
//! let mut sim = Simulation::new(&device, config);
//!
//! // --- Per-frame ---
//! // 1. Tell the simulation where the solid obstacles are.
//! sim.set_obstacles(&queue, &obstacle_patches);
//!
//! // 2. Enqueue any destruction / impact events from game logic.
//! sim.push_event(InjectionEvent {
//!     x: 5.0,
//!     y: 10.0,
//!     intensity: 1.0,
//!     stamp_radius: 0.6,
//!     color_id: 2,
//!     kind: EventKind::Destroy,
//!     velocity_scale: 0.05,
//!     base_vel_x: 0.0,
//!     base_vel_y: 0.0,
//! });
//!
//! // 3. Advance the simulation one step.
//! sim.step(&device, &queue);
//!
//! // 4. The `macroscopic` and `color_densities` buffers on `sim.grid` are now
//! //    ready to be bound in render / particle passes.
//! ```

use crate::config::SimConfig;
use crate::gpu::{
    build_bind_group, build_uniforms, encode_lbm_passes, rasterise_obstacles, GpuPipelines,
};
use crate::grid::{build_default_injection_stamp, GpuGrid};
use crate::types::{
    GpuEvent, InjectionEvent, LbmUniforms, ObstaclePatch, ObstacleTexel, OpenBoundaryPatch,
};

/// The main entry point for the LBM simulation.
///
/// Create with [`Simulation::new`], call [`Simulation::step`] once per frame.
pub struct Simulation {
    /// Public access to the raw GPU grid (buffers + textures).
    /// Downstream render/particle passes can bind these directly.
    pub grid: GpuGrid,

    /// Cached configuration.
    pub config: SimConfig,

    pipelines: GpuPipelines,

    /// Obstacle texel scratch buffer re-used every frame.
    obstacle_scratch: Vec<ObstacleTexel>,

    /// Previous frame's obstacle map — used to detect solid→fluid transitions.
    prev_obstacle_scratch: Vec<ObstacleTexel>,

    /// Events queued since the last [`step`] call.
    pending_events: Vec<InjectionEvent>,

    /// Per-cell open-boundary flags (A channel of the obstacle texture).
    /// Cells with value 1.0 are forced to equilibrium at ambient density each step.
    /// Populated by [`set_open_boundaries`]; merged into the obstacle texture upload.
    open_boundary_flags: Vec<f32>,

    /// Whether the default injection stamp texture has been uploaded.
    injection_stamp_uploaded: bool,

    /// Event injection mode.
    /// true = additive delta, false = replacement overwrite.
    additive_injection: bool,
}

impl Simulation {
    /// Create a new simulation.
    ///
    /// Allocates all GPU buffers and compiles all six compute shader pipelines.
    /// Safe to call once at startup; pipeline compilation may take a few hundred
    /// milliseconds on first run.
    pub fn new(device: &wgpu::Device, config: SimConfig) -> Self {
        let grid = GpuGrid::new(device, &config);
        let pipelines = GpuPipelines::new(device, &config);

        let obstacle_scratch = vec![
            ObstacleTexel {
                mask: 0.0,
                vel_x: 0.0,
                vel_y: 0.0,
                open_boundary: 0.0,
            };
            (config.grid_width * config.grid_height) as usize
        ];
        let prev_obstacle_scratch = obstacle_scratch.clone();
        let open_boundary_flags = vec![0.0f32; (config.grid_width * config.grid_height) as usize];

        Self {
            grid,
            config,
            pipelines,
            obstacle_scratch,
            prev_obstacle_scratch,
            pending_events: Vec::new(),
            open_boundary_flags,
            injection_stamp_uploaded: false,
            additive_injection: true,
        }
    }

    // ── Per-frame API ───────────────────────────────────────────────────────

    /// Upload an obstacle map for this frame.
    ///
    /// Call this before [`step`] with the current physics body transforms.
    /// Internally rasterises the patches into a flat texel grid and uploads to GPU.
    pub fn set_obstacles(&mut self, queue: &wgpu::Queue, patches: &[ObstaclePatch]) {
        rasterise_obstacles(patches, &self.config, &mut self.obstacle_scratch);

        // Stamp open-boundary flags into the A channel before upload.
        for (texel, &flag) in self
            .obstacle_scratch
            .iter_mut()
            .zip(self.open_boundary_flags.iter())
        {
            texel.open_boundary = flag;
        }

        self.grid
            .upload_obstacle_texture(queue, &self.obstacle_scratch);
        std::mem::swap(&mut self.prev_obstacle_scratch, &mut self.obstacle_scratch);
    }

    /// Mark world-space rectangles as open-boundary (Zou-He outflow) cells.
    ///
    /// Calling this **replaces** the current open-boundary map; pass an empty slice
    /// to clear all boundaries.  The new map takes effect on the next
    /// [`set_obstacles`] call, which merges the flags into the obstacle texture.
    pub fn set_open_boundaries(&mut self, patches: &[OpenBoundaryPatch]) {
        // Clear previous flags.
        for v in &mut self.open_boundary_flags {
            *v = 0.0;
        }

        let w = self.config.grid_width as usize;
        let h = self.config.grid_height as usize;

        for patch in patches {
            let gx_min = ((patch.x_min / self.config.world_width) * self.config.grid_width as f32)
                .floor() as i32;
            let gx_max = ((patch.x_max / self.config.world_width) * self.config.grid_width as f32)
                .ceil() as i32;
            let gy_min = ((patch.y_min / self.config.world_height) * self.config.grid_height as f32)
                .floor() as i32;
            let gy_max = ((patch.y_max / self.config.world_height) * self.config.grid_height as f32)
                .ceil() as i32;

            let gx_min = gx_min.max(0) as usize;
            let gx_max = gx_max.min(w as i32) as usize;
            let gy_min = gy_min.max(0) as usize;
            let gy_max = gy_max.min(h as i32) as usize;

            for gy in gy_min..gy_max {
                for gx in gx_min..gx_max {
                    self.open_boundary_flags[gy * w + gx] = 1.0;
                }
            }
        }
    }

    /// Queue an injection event to be processed in the next [`step`] call.
    ///
    /// Events are silently dropped if the queue exceeds
    /// [`crate::config::MAX_EVENTS`] to prevent buffer overflows.
    pub fn push_event(&mut self, event: InjectionEvent) {
        if self.pending_events.len() < crate::config::MAX_EVENTS {
            self.pending_events.push(event);
        }
    }

    /// Select whether event injection is additive or replacement-style.
    pub fn set_additive_injection(&mut self, additive: bool) {
        self.additive_injection = additive;
    }

    /// Advance the simulation by one logical step.
    ///
    /// Internally executes `config.substeps` LBM passes per call.  Events are
    /// injected only on the first sub-step; subsequent sub-steps run with zero
    /// events so pressure waves propagate without re-triggering injection.
    ///
    /// 1. Uploads pending events and uniforms.
    /// 2. Encodes all six compute passes into a command encoder (× substeps).
    /// 3. Submits to the queue.
    /// 4. Swaps ping-pong buffers.
    pub fn step(&mut self, device: &wgpu::Device, queue: &wgpu::Queue) {
        if !self.injection_stamp_uploaded {
            let stamp = build_default_injection_stamp();
            self.grid.upload_injection_texture(queue, &stamp);
            self.injection_stamp_uploaded = true;
        }

        let substeps = self.config.substeps.max(1);

        // Upload events to GPU — used only by the first sub-step.
        let event_count = self.pending_events.len() as u32;
        if event_count > 0 {
            let gpu_events: Vec<GpuEvent> =
                self.pending_events.iter().map(GpuEvent::from).collect();
            queue.write_buffer(
                &self.grid.event_buffer,
                0,
                bytemuck::cast_slice(&gpu_events),
            );
            self.pending_events.clear();
        }

        for sub in 0..substeps {
            // After the first sub-step, suppress event injection.
            let active_events = if sub == 0 { event_count } else { 0 };

            let uniforms: LbmUniforms =
                build_uniforms(&self.config, active_events, self.additive_injection);
            // write_buffer must be followed by a submit before the next write_buffer
            // so each sub-step's GPU passes see the correct uniform values.
            queue.write_buffer(&self.grid.uniform_buffer, 0, bytemuck::bytes_of(&uniforms));

            let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("lbm::step"),
            });

            // Build bind group (ping-pong state changes each sub-step).
            let bind_group =
                build_bind_group(device, &self.pipelines.bind_group_layout, &self.grid);

            encode_lbm_passes(
                &mut encoder,
                &self.pipelines,
                &bind_group,
                self.config.grid_width,
                self.config.grid_height,
                self.config.collision_mode,
            );

            queue.submit(std::iter::once(encoder.finish()));

            // After streaming, the dst buffer is the authoritative state.
            self.grid.swap();
        }
    }

    // ── Accessors for renderer integration ─────────────────────────────────

    /// Reference to the macroscopic quantities buffer (`[ρ, u_x, u_y]` per cell).
    ///
    /// Bind this in particle-update compute passes and smoke render passes.
    pub fn macroscopic_buffer(&self) -> &wgpu::Buffer {
        &self.grid.macroscopic
    }

    /// Reference to the packed colour-density storage buffer.
    ///
    /// Layout: `color_densities[cell_index * MAX_COLORS + channel]`
    /// where `MAX_COLORS = 8` and `cell_index = row * grid_width + col`.
    ///
    /// Bind this in smoke render passes; sample by indexing with the target channel.
    pub fn color_density_buffer(&self) -> &wgpu::Buffer {
        &self.grid.color_densities
    }
}
