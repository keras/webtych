//! CPU-side types that are uploaded to the GPU.
//!
//! All types implement [`bytemuck::Pod`] + [`bytemuck::Zeroable`] so they can be
//! cast directly to `&[u8]` for `queue.write_buffer` / `queue.write_texture`.

use bytemuck::{Pod, Zeroable};

// ─────────────────────────────────────────────────────────────────────────────
// Obstacle
// ─────────────────────────────────────────────────────────────────────────────

/// An axis-aligned rectangular obstacle patch to be rasterised into the
/// obstacle texture before each simulation step.
///
/// Match the game's coordinate system: origin bottom-left, Y-up.
/// The simulation will convert these to grid coordinates via [`SimConfig::world_to_grid`].
#[derive(Debug, Clone, Copy)]
pub struct ObstaclePatch {
    /// Left edge in world-space.
    pub x_min: f32,
    /// Bottom edge in world-space.
    pub y_min: f32,
    /// Right edge in world-space.
    pub x_max: f32,
    /// Top edge in world-space.
    pub y_max: f32,
    /// Obstacle velocity (world-space units / s) for moving bounce-back.
    pub vel_x: f32,
    pub vel_y: f32,
}

/// One texel of the obstacle texture uploaded to the GPU each frame.
///
/// Layout: `Rgba32Float`
/// * R — solid mask: 0.0 = fluid, 1.0 = solid obstacle.
/// * G — obstacle velocity X (world units / step).
/// * B — obstacle velocity Y.
/// * A — open boundary flag: 1.0 = Zou-He outflow cell (force ρ→1, preserve u).
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct ObstacleTexel {
    pub mask: f32,
    pub vel_x: f32,
    pub vel_y: f32,
    /// Open boundary flag stored in the A channel.
    /// Set via [`crate::simulation::Simulation::set_open_boundaries`].
    pub open_boundary: f32,
}

/// An axis-aligned rectangle marking cells as open-boundary (Zou-He outflow).
///
/// Open-boundary cells have their distributions forced to the D2Q9 equilibrium
/// at ambient density (ρ = 1.0) each step, while preserving the local velocity.
/// This acts as a pressure sink: excess density drains through these cells.
///
/// Coordinates use the same world-space convention as [`ObstaclePatch`].
#[derive(Debug, Clone, Copy)]
pub struct OpenBoundaryPatch {
    pub x_min: f32,
    pub y_min: f32,
    pub x_max: f32,
    pub y_max: f32,
}

// ─────────────────────────────────────────────────────────────────────────────
// Events
// ─────────────────────────────────────────────────────────────────────────────

/// Event type discriminant written into [`GpuEvent::event_type`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u32)]
pub enum EventKind {
    /// Block-destroy: inject pressure + coloured smoke.
    Destroy = 0,
    /// Block impact: inject a mild pressure pulse, no smoke.
    Impact = 1,
}

/// A single event to be injected into the fluid each frame.
///
/// Events are written to a ring buffer that the `lbm_inject` compute shader reads.
#[derive(Debug, Clone, Copy)]
pub struct InjectionEvent {
    /// World-space position of the event.
    pub x: f32,
    pub y: f32,
    /// Strength multiplier applied on top of the per-colour effect profile.
    pub intensity: f32,
    /// Which colour channel to inject density into (index into colour buffer array).
    pub color_id: u32,
    /// Destroy or impact.
    pub kind: EventKind,
}

/// GPU-mapped representation of [`InjectionEvent`].
///
/// Must stay in sync with the `GpuEvent` struct in `lbm_inject.wgsl`.
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct GpuEvent {
    pub position: [f32; 2],
    pub intensity: f32,
    pub color_id: u32,
    pub event_type: u32,
    pub _pad: [u32; 3],
}

impl From<&InjectionEvent> for GpuEvent {
    fn from(e: &InjectionEvent) -> Self {
        Self {
            position: [e.x, e.y],
            intensity: e.intensity,
            color_id: e.color_id,
            event_type: e.kind as u32,
            _pad: [0; 3],
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Uniforms
// ─────────────────────────────────────────────────────────────────────────────

/// Per-frame uniform block shared by all LBM compute passes.
///
/// Must stay in sync with `LbmUniforms` in the WGSL shaders.
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct LbmUniforms {
    pub grid_width: u32,
    pub grid_height: u32,
    pub tau: f32,
    pub inv_tau: f32,

    pub world_width: f32,
    pub world_height: f32,
    pub event_count: u32,
    pub color_count: u32,

    /// Per-colour inject_density values packed as 2×vec4 (indexed by color_id).
    /// MAX_COLORS = 8, so 2 vec4s cover all slots.
    pub inject_densities: [[f32; 4]; 2],
    /// Per-colour inject_color_density values.
    pub inject_color_densities: [[f32; 4]; 2],
    /// Per-colour dissipation values.
    pub dissipations: [[f32; 4]; 2],

    /// Gravity body-force acceleration (lattice units / step).
    /// Positive Y = toward bottom of screen (increasing pixel Y).
    pub gravity_x: f32,
    pub gravity_y: f32,
    pub _pad: [u32; 2],
}
