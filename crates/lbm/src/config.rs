//! Simulation configuration — grid dimensions, physics parameters, and per-colour effect profiles.

/// Maximum number of simultaneous colour-density channels.
///
/// This is a compile-time ceiling; the actual number used is `SimConfig::color_count`.
pub const MAX_COLORS: usize = 8;

/// Maximum number of injection events per frame.
pub const MAX_EVENTS: usize = 256;

/// LBM collision operator.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CollisionMode {
    /// Single-relaxation-time (Bhatnagar-Gross-Krook).
    /// Simple, fast, but viscosity-dependent wall placement.
    Bgk,
    /// Two-relaxation-time.  Separates symmetric (viscosity) and
    /// antisymmetric (magic parameter Λ = 3/16) relaxation.  Eliminates
    /// viscosity-dependent wall errors at minimal extra cost.
    Trt,
    /// Multiple-relaxation-time (Lallemand & Luo 2000).
    /// Separate rates for energy, flux, and stress moments — most stable.
    Mrt,
}

/// Configuration for the LBM simulation.
///
/// Create once and pass to [`Simulation::new`](crate::Simulation::new).
/// Most fields can also be changed at runtime by recreating the simulation —
/// buffer sizes are derived from this struct.
#[derive(Debug, Clone)]
pub struct SimConfig {
    /// Width of the simulation grid in cells.
    pub grid_width: u32,
    /// Height of the simulation grid in cells.
    pub grid_height: u32,

    /// MRT shear relaxation time τ (= 1/s_ν).  Controls kinematic viscosity:
    ///   ν = (1/3) × (τ − 0.5)
    ///
    /// Must be > 0.5 for numerical stability.
    /// * 0.6  → low viscosity, turbulent swirling smoke.
    /// * 0.8  → moderate viscosity, smooth billowing.
    /// * 1.0+ → high viscosity, thick flow.
    pub tau: f32,

    /// MRT energy relaxation rate s_e (= s_ε).
    ///
    /// Controls how fast the energy and ghost-energy moments relax.
    /// Must be in (0, 2) for stability.  Default 1.0 is a safe choice;
    /// values closer to 2.0 damp energy modes more aggressively.
    pub mrt_s_e: f32,

    /// MRT energy-flux relaxation rate s_q.
    ///
    /// `None` (default) uses the Lallemand & Luo recommended formula:
    ///   s_q = 8 × (2 − s_ν) / (8 − s_ν)
    /// which keeps the effective viscosity well-defined at all resolutions.
    pub mrt_s_q: Option<f32>,

    /// Physical world width that the grid covers (in game-world units).
    /// Used to convert world-space obstacle positions to grid coordinates.
    pub world_width: f32,

    /// Physical world height that the grid covers.
    pub world_height: f32,

    /// Number of passive colour-density channels (≤ [`MAX_COLORS`]).
    pub color_count: u32,

    /// Per-colour effect profiles (length must equal `color_count`).
    pub effect_profiles: Vec<EffectProfile>,

    /// Number of LBM steps executed per [`Simulation::step`] call.
    ///
    /// Each extra sub-step effectively multiplies the speed of sound (and all
    /// advection speeds) by `substeps` in physical / visual terms, at the cost
    /// of `substeps × GPU time per frame`.  Default is 1 (no sub-stepping).
    pub substeps: u32,

    /// Gravity body-force acceleration (lattice units / step).
    /// Positive values pull toward +X / +Y (right / bottom of screen).
    pub gravity_x: f32,
    pub gravity_y: f32,

    /// Which collision operator to use.
    pub collision_mode: CollisionMode,
}

impl SimConfig {
    /// Convenience constructor matching the Tetris-style board defaults from
    /// the game crate (10×20 cells, 1.0 cell-size).
    ///
    /// Grid resolution defaults to 256×256 on all targets; bump to 512×512 on
    /// native if adapter limits allow.
    pub fn for_game_board(world_width: f32, world_height: f32, color_count: u32) -> Self {
        assert!(
            color_count as usize <= MAX_COLORS,
            "color_count {color_count} exceeds MAX_COLORS {MAX_COLORS}"
        );
        Self {
            grid_width: 256,
            grid_height: 256,
            tau: 0.7,
            mrt_s_e: 1.0,
            mrt_s_q: None,
            world_width,
            world_height,
            color_count,
            effect_profiles: (0..color_count).map(|_| EffectProfile::default()).collect(),
            substeps: 1,
            gravity_x: 0.0,
            gravity_y: 0.0,
            collision_mode: CollisionMode::Mrt,
        }
    }

    /// Total number of cells in the grid.
    pub fn cell_count(&self) -> u32 {
        self.grid_width * self.grid_height
    }

    /// Convert a world-space position to a (fractional) grid cell coordinate.
    /// Returns `None` if the point is outside the simulation domain.
    pub fn world_to_grid(&self, x: f32, y: f32) -> Option<(f32, f32)> {
        let gx = x / self.world_width * self.grid_width as f32;
        let gy = y / self.world_height * self.grid_height as f32;
        if gx < 0.0 || gy < 0.0 || gx >= self.grid_width as f32 || gy >= self.grid_height as f32 {
            None
        } else {
            Some((gx, gy))
        }
    }
}

/// Per-colour parameters controlling how a destruction/impact event affects
/// the fluid and colour-density field.
#[derive(Debug, Clone)]
pub struct EffectProfile {
    /// Overpressure injected when blocks of this colour are destroyed.
    /// Expressed as a multiple of ambient density ρ₀ = 1.0.
    /// Higher values produce stronger pressure blasts.
    pub inject_density: f32,

    /// How much colour-density smoke to inject on destruction.
    pub inject_color_density: f32,

    /// Per-frame dissipation factor for colour-density (0.0 = instant, 1.0 = never).
    /// Typically 0.99–0.999 for slowly fading smoke trails.
    pub dissipation: f32,
}

impl Default for EffectProfile {
    fn default() -> Self {
        Self {
            inject_density: 3.0,
            inject_color_density: 0.1,
            dissipation: 0.995,
        }
    }
}
