//! CPU-side types that are uploaded to the GPU.
//!
//! All types implement [`bytemuck::Pod`] + [`bytemuck::Zeroable`] so they can be
//! cast directly to `&[u8]` for `queue.write_buffer` / `queue.write_texture`.

use bytemuck::{Pod, Zeroable};

// ─────────────────────────────────────────────────────────────────────────────
// Obstacle
// ─────────────────────────────────────────────────────────────────────────────

/// A rectangular obstacle patch (optionally rotated) to be rasterised into the
/// obstacle texture before each simulation step.
///
/// `x_min/y_min/x_max/y_max` are the bounds of the **unrotated** rectangle in
/// world-space (origin bottom-left, Y-up).  `rotation` is the clockwise angle in
/// radians applied around the rectangle's centre.  Pass `rotation: 0.0` for
/// axis-aligned patches (uses a fast AABB path internally).
#[derive(Debug, Clone, Copy)]
pub struct ObstaclePatch {
    /// Left edge of the unrotated rectangle in world-space.
    pub x_min: f32,
    /// Bottom edge of the unrotated rectangle in world-space.
    pub y_min: f32,
    /// Right edge of the unrotated rectangle in world-space.
    pub x_max: f32,
    /// Top edge of the unrotated rectangle in world-space.
    pub y_max: f32,
    /// Obstacle velocity (world-space units / s) for moving bounce-back.
    pub vel_x: f32,
    pub vel_y: f32,
    /// Clockwise rotation in radians around the rectangle centre.  Use `0.0` for
    /// axis-aligned patches.
    pub rotation: f32,
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
    /// Injection stamp radius in world-space units.
    ///
    /// The `lbm_inject` pass samples a stamp texture in this radius around
    /// (`x`, `y`) and uses the sampled mask to spread injection over an area.
    pub stamp_radius: f32,
    /// Which colour channel to inject density into (index into colour buffer array).
    pub color_id: u32,
    /// Destroy or impact.
    pub kind: EventKind,
    /// Multiplier applied to the velocity profile sampled from the injection stamp.
    pub velocity_scale: f32,
    /// Constant velocity bias added to injected fluid (lattice units / step).
    pub base_vel_x: f32,
    /// Constant velocity bias added to injected fluid (lattice units / step).
    pub base_vel_y: f32,
}

/// GPU-mapped representation of [`InjectionEvent`].
///
/// Must stay in sync with the `GpuEvent` struct in `lbm_inject.wgsl`.
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct GpuEvent {
    pub position: [f32; 2],
    pub intensity: f32,
    pub stamp_radius: f32,
    pub color_id: u32,
    pub event_type: u32,
    pub velocity_scale: f32,
    pub base_vel_x: f32,
    pub base_vel_y: f32,
}

impl From<&InjectionEvent> for GpuEvent {
    fn from(e: &InjectionEvent) -> Self {
        Self {
            position: [e.x, e.y],
            intensity: e.intensity,
            stamp_radius: e.stamp_radius,
            color_id: e.color_id,
            event_type: e.kind as u32,
            velocity_scale: e.velocity_scale,
            base_vel_x: e.base_vel_x,
            base_vel_y: e.base_vel_y,
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
    /// τ = 1/s_ν.  Kept in uniforms for the gravity body-force shift (u_eff = u + g·τ).
    pub tau: f32,
    pub _pad0: f32,

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
    /// Injection write mode.
    /// 0 = replacement (overwrite cell state), 1 = additive (delta onto existing state).
    pub injection_mode: u32,
    pub _pad1: u32,

    /// MRT relaxation rates packed as 3×vec4 (12 slots, 9 used).
    /// Layout: [s0..s3], [s4..s7], [s8, 0, 0, 0]
    /// s0/s3/s5 are conserved-moment placeholders (zero non-equilibrium, value unused).
    /// s1=s_e, s2=s_ε, s4=s6=s_q, s7=s8=s_ν=1/τ.
    pub mrt_s: [[f32; 4]; 3],
}
