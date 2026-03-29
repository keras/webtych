//! `webtych-lbm` — GPU-accelerated D2Q9 Lattice Boltzmann fluid simulator.
//!
//! # Overview
//!
//! This crate implements a 2-D compressible Lattice Boltzmann Method (LBM)
//! solver with the following characteristics:
//!
//! * **D2Q9 lattice** — 9 velocity directions per cell.
//! * **BGK (single-relaxation-time) collision** operator.
//! * **Moving bounce-back boundaries** — obstacles upload their velocity so
//!   momentum is correctly transferred to the fluid.
//! * **Passive colour-density advection** — N independent scalar fields
//!   advected through the velocity field (semi-Lagrangian).
//! * **Runs entirely on the GPU** via wgpu compute shaders (WGSL).
//!
//! # Integration contract
//!
//! The crate is intentionally decoupled from game logic. The integration
//! surface is:
//!
//! * [`SimConfig`] — grid size, relaxation time, colour count, etc.
//! * [`ObstaclePatch`] — a rectangle of obstacle cells with a velocity; one or
//!   more patches are uploaded per frame from CPU-side physics state.
//! * [`InjectionEvent`] — a pressure/density injection at a world position
//!   (e.g. block-destroy events from the game).
//! * [`Simulation`] — owns all GPU resources; call [`Simulation::step`] once
//!   per frame after uploading patches and events.
//!
//! # Crate layout
//!
//! ```
//! lbm/
//! ├── src/
//! │   ├── lib.rs         ← this file, re-exports public API
//! │   ├── config.rs      ← SimConfig + per-colour EffectProfile
//! │   ├── grid.rs        ← GPU buffer / texture allocation helpers
//! │   ├── gpu.rs         ← bind-group layouts, pipelines, per-frame upload
//! │   ├── simulation.rs  ← top-level Simulation struct + step()
//! │   └── types.rs       ← Pod/Zeroable GPU-mapped types
//! └── shaders/
//!     ├── lbm_inject.wgsl
//!     ├── lbm_collide.wgsl
//!     ├── lbm_stream.wgsl
//!     ├── lbm_boundary.wgsl
//!     ├── lbm_extract.wgsl
//!     └── lbm_advect_color.wgsl
//! ```

pub mod config;
pub mod gpu;
pub mod grid;
pub mod simulation;
pub mod types;

pub use config::{EffectProfile, SimConfig};
pub use simulation::Simulation;
pub use types::{EventKind, InjectionEvent, ObstaclePatch, OpenBoundaryPatch};
