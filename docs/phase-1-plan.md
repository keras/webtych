# Phase 1 Implementation Plan: Core Game Loop

## Overview

Phase 1 transforms the spinning-triangle demo into a playable falling-block game. Triminos fall under physics, stack on walls and each other, same-color groups of 3+ clear, score increases, and the game ends when blocks reach the top. All blocks are flat-colored instanced quads -- no soft-body deformation, no fluid, no particles.

The work splits into **5 PRs (milestones)**, ordered by dependency. Each PR is independently mergeable and testable.

---

## PR 1: Workspace Restructure -- `game` and `gpu` Crates

**Why first**: Every subsequent PR depends on the crate boundaries being established. This is a pure structural refactor with no new features.

### Step 1.1: Create `crates/game/` crate

Create `crates/game/Cargo.toml` with dependencies:
- `rapier2d` (added but not yet used -- just compiling proves WASM compat)
- `glam` (f32 math library, used everywhere)
- `bytemuck` with `derive` (for GPU-transmittable types)
- `log`
- `rand` with `small_rng` feature (WASM-safe, no OS entropy needed)

For rapier2d, the Cargo.toml needs conditional features:
```toml
[dependencies]
rapier2d = { version = "0.22", features = ["enhanced-determinism"] }

[target.'cfg(target_arch = "wasm32")'.dependencies]
rapier2d = { version = "0.22", features = ["enhanced-determinism", "wasm-bindgen"] }
```

Create stub `crates/game/src/lib.rs` with module declarations for `board`, `trimino`, `physics`, `scoring`, `events`, `state` -- all as empty files with placeholder types.

### Step 1.2: Create `crates/gpu/` crate

Create `crates/gpu/Cargo.toml` depending on `wgpu`, `bytemuck`, `glam`, and `webtych-game` (the game crate).

Extract the GPU initialization code from the current `renderer.rs` into `crates/gpu/src/context.rs`. This becomes a `GpuContext` struct holding `device`, `queue`, `surface`, `config`, `adapter`. The `new()` async constructor moves here verbatim.

Create `crates/gpu/src/renderer.rs` that owns a `GpuContext` and orchestrates frame rendering. For now it just re-implements the current triangle (will be replaced in PR 5).

Create `crates/gpu/src/blocks/mod.rs` as an empty placeholder.

### Step 1.3: Rewire `crates/app/`

Update `crates/app/Cargo.toml` to depend on `webtych-game` and `webtych-gpu`. Remove direct `wgpu`/`bytemuck` dependencies (they come transitively through `gpu`).

Update `lib.rs` to import `Renderer` from the `gpu` crate instead of the local `renderer` module. Delete `crates/app/src/renderer.rs`.

### Step 1.4: Update workspace `Cargo.toml`

Add `crates/game` and `crates/gpu` to `members`.

### Step 1.5: Update CI

The CI workflow needs to check/clippy all workspace members. Change to check the entire workspace instead of `-p webtych`. The WASM check should apply to game and gpu crates too (skip the app binary target on WASM, keep `--lib`).

**Files created/modified**:
- `crates/game/Cargo.toml` (new)
- `crates/game/src/lib.rs` (new)
- `crates/game/src/board.rs` (new, stub)
- `crates/game/src/trimino.rs` (new, stub)
- `crates/game/src/physics.rs` (new, stub)
- `crates/game/src/scoring.rs` (new, stub)
- `crates/game/src/events.rs` (new, stub)
- `crates/game/src/state.rs` (new, stub)
- `crates/gpu/Cargo.toml` (new)
- `crates/gpu/src/lib.rs` (new)
- `crates/gpu/src/context.rs` (new, extracted from renderer.rs)
- `crates/gpu/src/renderer.rs` (new, thin wrapper)
- `crates/gpu/src/blocks/mod.rs` (new, stub)
- `crates/app/Cargo.toml` (modified)
- `crates/app/src/lib.rs` (modified)
- `crates/app/src/renderer.rs` (deleted)
- `Cargo.toml` (modified -- workspace members)
- `.github/workflows/ci.yml` (modified)

**Testable**: `cargo check`, `cargo check --target wasm32-unknown-unknown`, and the existing spinning triangle still renders in both native and WASM. The e2e smoke test still passes.

---

## PR 2: Physics World, Board, and Trimino Definitions

**Why second**: Establishes the core domain types that input handling and rendering depend on. Can be tested entirely headlessly with unit tests.

### Step 2.1: `events.rs` -- GameEvent enum

```rust
pub enum GameEvent {
    Spawn { piece_id: u32 },
    Destroy { cells: Vec<CellId>, color: ColorId, chain_level: u32 },
    Impact { position: glam::Vec2, velocity: f32 },
    GameOver,
}
```

Also define `CellId` (a newtype around rapier's `RigidBodyHandle`) and `ColorId` (a `u8` or small enum).

### Step 2.2: `trimino.rs` -- Piece definitions

Define `ColorId` enum with 4 colors (Red, Blue, Green, Yellow -- matches the plan's "start with 4" decision).

Define trimino shapes. The original Triptych uses triminos (3-cell pieces). Define the standard set:
- **I-shape**: 3 cells in a line
- **L-shape**: 2 cells in a line + 1 cell perpendicular (and its mirror)
- **Triangle**: 3 cells in an L

Each shape is defined as a list of `(i8, i8)` offsets from the piece's pivot cell. Store shapes as a `const` array.

Implement a **random bag** system: shuffle all shape variants, deal them in order, reshuffle when exhausted. Use `rand::rngs::SmallRng` seeded from a u64 (WASM-safe -- no system entropy required).

### Step 2.3: `board.rs` -- Play field geometry

Define board constants:
- `BOARD_WIDTH`: physical width in world units (e.g., 10.0)
- `BOARD_HEIGHT`: physical height (e.g., 20.0)
- `CELL_SIZE`: size of one cell (e.g., 1.0)
- `SPAWN_X`, `SPAWN_Y`: spawn position at top center

Define a `Board` struct that holds the static geometry data (wall positions). Used to create the static colliders in the physics world.

### Step 2.4: `physics.rs` -- Rapier world setup

Create a `PhysicsWorld` struct wrapping:
- `RigidBodySet`, `ImpulseJointSet`, `ColliderSet`
- `PhysicsPipeline`, `QueryPipeline`
- `IntegrationParameters`, `IslandManager`
- `BroadPhaseMultiSap`, `NarrowPhase`, `CCDSolver`

Initialize with:
- Gravity: `(0.0, -9.81 * GRAVITY_SCALE)` -- start with `GRAVITY_SCALE = 2.0` and tune
- Restitution `0.3`, friction `0.5` set on colliders

Methods:
- `new(board: &Board) -> Self` -- creates the world and adds static wall/floor colliders
- `step()` -- advances physics one fixed timestep
- `spawn_trimino(shape, color, position, rotation) -> Vec<RigidBodyHandle>` -- creates rigid bodies for each cell, connects them with fixed joints
- `apply_force(handle, force)` -- for input-driven movement
- `body_position(handle) -> (Vec2, f32)` -- get position and rotation
- `contact_pairs()` -- expose narrow phase contacts for match detection

Each cell is a `cuboid` collider of size `CELL_SIZE / 2.0` (half-extents). Fixed joints between cells in a piece keep the trimino rigid until a cell is destroyed.

### Step 2.5: Unit tests

- Test that `PhysicsWorld::new()` creates the expected number of static colliders
- Test that `spawn_trimino()` creates the correct number of bodies and joints
- Test that a single physics step doesn't panic
- Test random bag produces all shapes before repeating

**Files**: `crates/game/src/{events,trimino,board,physics,lib}.rs`

**Testable**: `cargo test -p webtych-game` passes. Physics world initializes, triminos can be spawned, gravity pulls them down over multiple steps.

---

## PR 3: Game State Machine, Input, and Fixed Timestep

**Why third**: Connects the physics world to the app's event loop. After this PR, blocks fall and respond to keyboard input.

### Step 3.1: `state.rs` -- GameState and phase machine

```rust
pub enum Phase {
    Spawning,       // Generating next piece
    Falling,        // Active piece under player control
    Settling,       // Piece landed, waiting for physics to stabilize
    Matching,       // Checking for and destroying matches
    GameOver,
}
```

`GameState` struct holds: phase, physics, board, random bag, active piece, placed cells, score, level, lines_cleared, events queue, accumulator, drop timer.

### Step 3.2: Input mapping

Define a platform-agnostic `Input` struct with: `move_left`, `move_right`, `soft_drop`, `hard_drop`, `rotate_cw`, `rotate_ccw`, `pause`.

In `crates/app/src/lib.rs`, translate `winit::event::KeyEvent` into this struct:
- Arrow keys for movement/soft drop
- Space/Up for hard drop
- Z/A for rotate CCW, X/D for rotate CW
- Escape for pause

### Step 3.3: Fixed timestep in the game loop

At `RedrawRequested`:
1. Compute `frame_dt` from wall clock (`web_sys::performance().now()` on WASM, `std::time::Instant` on native). Cap at 100ms.
2. Run fixed timestep accumulator at 60 Hz physics rate.

### Step 3.4: Movement and rotation

- **Horizontal movement**: Apply lateral impulse to all bodies in the active piece
- **Soft drop**: Apply downward impulse (gravity multiplier)
- **Hard drop**: Set high downward velocity on all bodies
- **Rotation**: Set positions directly around the pivot with collision checks via `query_pipeline.intersection_with_shape()`. This gives precise, grid-aligned rotation. Check for wall kicks if rotation fails in place.

### Step 3.5: Piece settling detection

Check if all cells in the active piece have velocity below `SETTLE_THRESHOLD` for `SETTLE_FRAMES` consecutive frames (~10). Also check for floor/placed-cell contacts. When settled, move cells to `placed_cells` and transition to `Matching`.

### Step 3.6: Automatic drop

Timer-based downward impulse: `drop_interval = DROP_INTERVAL_BASE * (0.8 ^ level)`.

**Files**: `crates/game/src/state.rs`, `crates/game/src/lib.rs`, `crates/app/src/lib.rs`

**Testable**: Native binary shows triminos falling and responding to keyboard input (even if rendering is still the triangle -- verify via logs or debug output).

---

## PR 4: Match Detection, Destruction, Scoring, and Game Over

**Why fourth**: Completes the game logic. After this PR, the game is fully playable.

### Step 4.1: `scoring.rs` -- Match-3 detection

1. Build a contact graph from `narrow_phase.contact_pairs()` -- edges between same-color cells with active contacts
2. Run connected components (flood fill / union-find)
3. Any component with 3+ cells is a match

Important: contacts are between colliders; map `ColliderHandle -> RigidBodyHandle` via `collider_set[handle].parent()`. Only count active contacts (non-zero contact force or penetration depth > 0).

### Step 4.2: Destruction

For each matched cell:
1. Remove rigid body and collider from physics world
2. Remove joints connected to the destroyed cell (remaining cells become independent and may fall)
3. Push `GameEvent::Destroy`
4. Transition to `Settling` again (remaining cells may shift and form new matches)

State machine loop:
```
Spawning -> Falling -> Settling -> Matching
                                    |
                         (matches) -> Destroy -> Settling -> Matching
                                    |
                         (no matches) -> Spawning
```

### Step 4.3: Scoring

- `points = cells_cleared * 100 * (chain_level + 1)`
- Every 10 lines cleared, increment level (increases drop speed)

### Step 4.4: Game over detection

After spawning, check if new piece cells overlap with existing placed cells via `query_pipeline.intersection_with_shape()`. If overlap exists, transition to `Phase::GameOver`.

### Step 4.5: Unit tests

- Test 3 same-color cells in contact are detected as a match
- Test 2 cells don't trigger a match
- Test chain detection
- Test scoring formula
- Test game over condition

**Files**: `crates/game/src/{scoring,state,physics}.rs`

**Testable**: Full game logic works in unit tests. Can programmatically play a full game.

---

## PR 5: Instanced Block Rendering

**Why last**: Requires stable game state API from PRs 2-4. Replaces the spinning triangle with actual gameplay visuals.

### Step 5.1: Block instance data

```rust
#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub struct BlockInstance {
    pub position: [f32; 2],
    pub rotation: f32,
    pub scale: f32,
    pub color: [f32; 4],
}
```

Simpler than the project plan's corner-based format -- that's for Phase 2 soft-body. Phase 1 uses position + rotation per rigid body.

### Step 5.2: WGSL block shader

`crates/gpu/shaders/block.wgsl`:
- Vertex shader: unit quad + per-instance position/rotation/scale/color, orthographic projection
- Fragment shader: flat color with slight edge darkening for visual separation

### Step 5.3: Orthographic projection

`glam::Mat4::orthographic_rh()` mapping board bounds to clip space. Update on resize with letterboxing to maintain aspect ratio.

### Step 5.4: Renderer integration

- `prepare_frame(game_state)`: extract position/rotation/color for all cells, fill instance buffer
- `render()`: clear screen, draw instanced quads

Pre-allocate instance buffer for ~512 cells max.

### Step 5.5: Wall rendering

Render walls as static quads (left, right, floor) in neutral gray. Same instance buffer.

### Step 5.6: Wire up in app

`RedrawRequested`: `game_state.update(dt, &input)` -> `renderer.prepare_frame(&game_state)` -> `renderer.render()`

Game state exposes `fn cells() -> impl Iterator<Item = CellRenderData>` for the renderer.

### Step 5.7: Simple UI overlay

On WASM: HTML overlay `<div>` for score/level, updated via `web_sys`. On native: log to console (text rendering deferred to Phase 6).

### Step 5.8: Update e2e test

Verify smoke test still passes (canvas renders, no panics).

**Files**: `crates/gpu/src/blocks/mod.rs`, `crates/gpu/shaders/block.wgsl` (new), `crates/gpu/src/renderer.rs`, `crates/app/src/lib.rs`, `crates/game/src/state.rs`

**Testable**: Full game is visually playable. Colored quads fall, keyboard controls work, matches clear, score increases, game over triggers. Both native and WASM work.

---

## Key Architectural Decisions

### 1. Separate rigid bodies + fixed joints (not compound colliders)
Individual cells can be destroyed independently in Phase 4. When a middle cell is cleared, the joint is removed and remaining cells become independent bodies. Compound colliders would require complex decomposition at destruction time.

### 2. Movement via impulse, rotation via position-set
Impulse-based horizontal movement gives physics-natural feel. Direct rotation with collision checks gives precise, grid-aligned rotation that feels like a puzzle game. Wall kick logic shifts the piece if rotation fails in place.

### 3. Contact-based matching (not grid-based)
Triptych's uniqueness: pieces settle wherever physics puts them, matching depends on physical contact. Flood-fill runs on an adjacency graph built from `narrow_phase.contact_pairs()`.

### 4. Flat quads for Phase 1
The project plan's corner-based `BlockInstance` is for Phase 2 soft-body. Phase 1 uses simpler position/rotation per rigid body, avoiding premature complexity.

### 5. Game state owns physics, renderer borrows read-only
Strict `Input -> GameState -> Renderer` data flow. The renderer never mutates game state. Prepares for the project plan's "CPU physics is authoritative, GPU visuals are cosmetic" rule.

---

## Potential Challenges

1. **Rapier2d WASM binary size**: Expect 500KB-1MB added. Monitor with `wasm-opt -O3` and LTO.

2. **Contact flickering**: Physics contacts can flicker as bodies micro-settle. Match detection should only run after settling is confirmed, not every frame.

3. **Rotation edge cases**: Wall kick logic needed when rotating near walls/other pieces.

4. **Fixed joint stability**: Many bodies connected by joints can cause solver jitter in tall stacks. May need to increase solver iterations.

5. **Rapier feature gating**: `wasm-bindgen` and `parallel` features are mutually exclusive. Must use `[target.'cfg(...)'.dependencies]` for conditional compilation.

---

## Dependency Graph

```
PR 1 (Workspace Restructure)
  |
  v
PR 2 (Physics, Board, Triminos)
  |
  v
PR 3 (State Machine, Input, Timestep)
  |
  v
PR 4 (Matching, Destruction, Scoring)
  |
  v
PR 5 (Instanced Block Rendering)
```

Each PR depends on the one above. Within each PR, sub-steps can often be developed independently.
