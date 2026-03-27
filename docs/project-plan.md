# Triptych Remake — Project Plan

A visual-effect-heavy reimagining of Chronic Logic's *Triptych* (2002), built in Rust with wgpu, targeting WebAssembly as the primary distribution platform.

---

## 1. Vision

The original Triptych was a physics-based falling-block puzzler: trimino pieces bounce, squish, and stack under real physics instead of snapping to a grid. Three or more same-color cells in contact clear. The remake preserves this core loop but wraps it in a GPU-driven visual spectacle — fluid smoke dynamics, per-color destruction effects, pressure-driven gas simulation, and particle systems that interact with the fluid field. The game runs in any modern browser via WebGPU, with a native desktop target as a secondary build.

---

## 2. Target Platform & Constraints

### 2.1 Primary: Browser via WASM + WebGPU

- **Toolchain**: `wasm32-unknown-unknown` via `wasm-pack` or `trunk`.
- **GPU API**: wgpu compiles to WASM and delegates to the browser's native WebGPU implementation. All shaders must be WGSL — wgpu handles this automatically but no SPIR-V or GLSL extensions are available on the web target.
- **Compute shaders**: Fully supported in WebGPU. This is critical — the entire fluid simulation and particle system run in compute passes.
- **Threading**: Web workers are available but shared memory (`SharedArrayBuffer`) requires cross-origin isolation headers (`COOP`/`COEP`). The physics step should be designed to run single-threaded by default, with optional parallelism on native. Rapier's `parallel` feature (rayon-based) cannot be used on WASM; use `wasm-bindgen` feature instead.
- **Memory**: WASM linear memory defaults to ~256 MB max in most browsers. The fluid grid and particle buffers live on the GPU, so CPU-side pressure is modest. Budget ~64 MB CPU-side for physics state, asset data, and application logic.
- **Storage budget**: Shader modules, textures, and GPU buffers. WebGPU has per-device limits; query `maxStorageBufferBindingSize` (typically 128–256 MB) and `maxComputeInvocationsPerWorkgroup` (typically 256) at init and adapt grid resolution accordingly.

### 2.2 Secondary: Native Desktop

- **Backends**: Vulkan (Linux/Windows), Metal (macOS), DX12 (Windows) — all handled transparently by wgpu.
- **Extras**: SIMD-accelerated Rapier (`simd-stable` feature), rayon parallelism, higher fluid grid resolution, more particles.

### 2.3 Browser Compatibility

WebGPU is available in Chrome 113+, Edge 113+, Firefox (behind flag, full support rolling out), and Safari 18+. For browsers without WebGPU, the project does **not** fall back to WebGL — the compute-heavy architecture requires compute shaders. Display a clear message directing users to a supported browser.

---

## 3. Architecture Overview

```
┌──────────────────────────────────────────────────────┐
│                    Game Loop (CPU)                    │
│                                                      │
│  ┌──────────────┐   ┌───────────────┐                │
│  │   Input       │──▶│  Game State   │                │
│  │  (keyboard)   │   │  (score,      │                │
│  └──────────────┘   │   level,      │                │
│                      │   spawn)      │                │
│                      └──────┬────────┘                │
│                             │                         │
│                             ▼                         │
│                   ┌──────────────────┐                │
│                   │  Physics Step    │                │
│                   │  (Rapier2D)      │                │
│                   │                  │                │
│                   │  • rigid bodies  │                │
│                   │  • contacts      │                │
│                   │  • soft body*    │                │
│                   └──────┬───────────┘                │
│                          │                            │
│                ┌─────────┴──────────┐                 │
│                │                    │                  │
│                ▼                    ▼                  │
│     ┌────────────────┐     ┌──────────────┐          │
│     │ Obstacle        │     │ Destroy      │          │
│     │ Texture Upload  │     │ Events       │          │
│     │ (mask +         │     │ → Ring       │          │
│     │  velocity +     │     │   Buffer     │          │
│     │  sweep fill)    │     │              │          │
│     └───────┬────────┘     └──────┬───────┘          │
│             │                     │                   │
└─────────────┼─────────────────────┼───────────────────┘
              │                     │
              ▼                     ▼
┌──────────────────────────────────────────────────────┐
│                   GPU Pipeline                        │
│                                                      │
│  ┌────────────────────────────────────────────┐      │
│  │  Compute: Fluid Simulation                  │      │
│  │  1. Apply forces + event injections         │      │
│  │  2. Apply boundary (obstacle velocity)      │      │
│  │  3. Advect velocity (semi-Lagrangian)       │      │
│  │  4. Pressure solve (Jacobi, N iterations)   │      │
│  │  5. Pressure projection + boundary enforce  │      │
│  │  6. Advect density fields                   │      │
│  └────────────────────────────────────────────┘      │
│                                                      │
│  ┌────────────────────────────────────────────┐      │
│  │  Compute: Particle Update                   │      │
│  │  • Read fluid velocity field                │      │
│  │  • Integrate positions                      │      │
│  │  • Lifetime / fade                          │      │
│  └────────────────────────────────────────────┘      │
│                                                      │
│  ┌────────────────────────────────────────────┐      │
│  │  Render Passes                              │      │
│  │  1. Block geometry (instanced quads)        │      │
│  │  2. Fluid density → screen-space smoke      │      │
│  │  3. Particle billboard sprites              │      │
│  │  4. Post-process (bloom, distortion, shake) │      │
│  │  5. UI overlay                              │      │
│  └────────────────────────────────────────────┘      │
│                                                      │
└──────────────────────────────────────────────────────┘
```

The critical design rule: **the CPU physics simulation is authoritative and the GPU visual simulation is purely cosmetic**. Data flows CPU → GPU only. No GPU readback. The fluid simulation, particle systems, and all visual effects never influence gameplay.

---

## 4. Crate & Dependency Map

| Crate | Purpose | WASM-safe | Notes |
|---|---|---|---|
| `wgpu` | GPU abstraction (render + compute) | Yes | Use `webgpu` feature on WASM, native backends on desktop |
| `winit` | Window + event loop | Yes | `web-sys` integration for WASM canvas |
| `rapier2d` | 2D physics engine | Yes | Use `wasm-bindgen` feature; disable `parallel` on WASM |
| `glam` | Linear algebra | Yes | `f32` throughout for GPU compat |
| `bytemuck` | Safe GPU buffer casting | Yes | `derive` feature for `Pod`/`Zeroable` |
| `wasm-bindgen` | JS interop | N/A | WASM target only |
| `web-sys` | Browser API bindings | N/A | WASM target only — canvas, performance.now, etc. |
| `log` + `console_log` | Logging | Yes | `console_log` for WASM, `env_logger` for native |
| `trunk` or `wasm-pack` | Build/bundle tool | N/A | Build tooling |
| `cfg-if` | Conditional compilation | Yes | Gate platform-specific code |

### Notable exclusions

- **No game engine / ECS framework.** The game is simple enough that a manual game loop is clearer and avoids framework lock-in.
- **No audio crate in this plan.** Audio is deferred to a later phase (likely `cpal` native + Web Audio API via `web-sys`).

---

## 5. Module Structure

```
triptych/
├── Cargo.toml                    # workspace root
├── crates/
│   ├── game/                     # core game logic (platform-agnostic)
│   │   ├── src/
│   │   │   ├── lib.rs
│   │   │   ├── board.rs          # play field bounds, spawn logic
│   │   │   ├── trimino.rs        # piece definitions, color enum
│   │   │   ├── physics.rs        # rapier2d setup, step, contact extraction
│   │   │   ├── scoring.rs        # match detection, chain combos, level progression
│   │   │   ├── events.rs         # GameEvent enum: Destroy, Impact, Spawn
│   │   │   └── state.rs          # top-level GameState, phase machine
│   │   └── Cargo.toml
│   │
│   ├── gpu/                      # all wgpu rendering and compute
│   │   ├── src/
│   │   │   ├── lib.rs
│   │   │   ├── context.rs        # device, queue, surface, adapter init
│   │   │   ├── fluid/
│   │   │   │   ├── mod.rs
│   │   │   │   ├── grid.rs       # FluidGrid: textures, bind groups, dispatch
│   │   │   │   ├── solver.rs     # advection, pressure, projection, event injection
│   │   │   │   └── boundary.rs   # obstacle texture: mask + velocity, sweep fill
│   │   │   ├── particles/
│   │   │   │   ├── mod.rs
│   │   │   │   ├── emitter.rs    # per-color emission profiles
│   │   │   │   ├── update.rs     # compute pass: integrate, sample fluid, lifetime
│   │   │   │   └── render.rs     # billboard quad instancing
│   │   │   ├── blocks/
│   │   │   │   ├── mod.rs
│   │   │   │   ├── mesh.rs       # trimino quad geometry, instance buffer
│   │   │   │   └── shader.rs     # per-block procedural deformation shader
│   │   │   ├── postprocess/
│   │   │   │   ├── mod.rs
│   │   │   │   ├── bloom.rs      # dual-filter Kawase bloom
│   │   │   │   ├── distortion.rs # screen-space warp from impact events
│   │   │   │   └── composite.rs  # final tone map + gamma
│   │   │   ├── effects.rs        # ColorEffect trait, per-color effect registry
│   │   │   └── renderer.rs       # orchestrates full frame: compute → render → post
│   │   ├── shaders/              # .wgsl source files
│   │   │   ├── fluid_advect.wgsl
│   │   │   ├── fluid_pressure.wgsl
│   │   │   ├── fluid_project.wgsl
│   │   │   ├── fluid_inject.wgsl
│   │   │   ├── fluid_boundary.wgsl
│   │   │   ├── particle_update.wgsl
│   │   │   ├── particle_render.wgsl
│   │   │   ├── block_vert.wgsl
│   │   │   ├── block_frag.wgsl
│   │   │   ├── bloom.wgsl
│   │   │   ├── distortion.wgsl
│   │   │   ├── composite.wgsl
│   │   │   └── smoke_render.wgsl
│   │   └── Cargo.toml
│   │
│   └── app/                      # platform entry points
│       ├── src/
│       │   ├── main.rs           # native entry (winit event loop)
│       │   └── web.rs            # WASM entry (#[wasm_bindgen(start)])
│       ├── index.html            # WASM host page
│       └── Cargo.toml
│
├── assets/                       # textures, fonts (if any)
├── Trunk.toml                    # trunk build config for WASM
└── README.md
```

---

## 6. Detailed Subsystem Design

### 6.1 Physics Layer (CPU)

**Engine**: Rapier2D with `wasm-bindgen` + `enhanced-determinism` features.

**Rigid bodies**: Each trimino cell is a separate rigid body connected by fixed joints to its neighbors within the same piece. This allows individual cells to detach during destruction. Restitution ~0.3 (bouncy but not crazy), friction ~0.5.

**Soft-body simulation**: Triptych's signature squishiness. There are two viable approaches:

1. **Spring-damper lattice**: Each trimino cell is a cluster of 4 rigid body particles connected by stiff spring joints. The cell deforms as the particles shift under load. Visually, the cell's rendered quad interpolates its corners from the particle positions. This is cheap on Rapier and gives good visual deformation.

2. **Pressure soft body**: Model each cell as a closed polygon with edge particles. Internal pressure pushes outward; external forces compress. More complex but more physically accurate jelly behavior.

**Recommended**: Approach 1 (spring-damper) for initial implementation. It's simpler, Rapier handles it natively with impulse joints, and the visual deformation feeds naturally into the GPU obstacle texture.

**Contact extraction**: Every physics step, iterate `narrow_phase.contact_pairs()`. For each active contact:
- Extract contact normal, penetration depth, relative velocity.
- Feed destruction events (match-3 clears) to the GPU event ring buffer.
- Impact velocity is used for screen effects (shake, bloom) but does not drive fluid injection — the moving-boundary obstacle texture handles gas displacement naturally.

**Body transform history**: Store each body's previous-frame position and velocity. These are used during obstacle texture rasterization to perform sweep-filling (see §6.2A), preventing fast-moving bodies from tunneling through fluid cells.

### 6.2 CPU → GPU Boundary

Three data channels cross the boundary each frame:

**A. Obstacle texture upload (with velocity)**

A 2D grid (matching the fluid grid resolution) carrying both a solid/empty mask and the obstacle's velocity at each solid cell. Format: `Rgba16Float` — R channel for the solid mask (0.0 or 1.0), GB channels for the obstacle's X/Y velocity at that cell. Uploaded via `queue.write_texture()` each frame.

Generation (CPU-side):
1. For each physics body, retrieve its current position, rotation, and linear velocity.
2. Rasterize the body's collider shape onto the grid, filling covered cells with `(1.0, vel.x, vel.y, 0.0)`.
3. **Sweep fill**: For each body, also fill the cells between last frame's position and this frame's position. This prevents fast-moving obstacles from tunneling through fluid cells in a single timestep. Sweep by translating the shape along its frame-to-frame displacement vector and filling all intermediate cells. Tag swept cells with the body's velocity.
4. For the spring-damper soft bodies, use the convex hull of each cell's 4 particles as the rasterization shape.

Empty cells are `(0.0, 0.0, 0.0, 0.0)`. The fluid solver uses this texture in its boundary enforcement step: at any cell where the mask is 1.0, the fluid velocity is set to the obstacle velocity (the standard moving-boundary no-penetration condition in CFD). The pressure solver then naturally generates the pressure gradients needed to push fluid laterally out of narrowing gaps — no explicit squeeze injection needed.

**B. Event ring buffer**

A GPU storage buffer written each frame with a variable number of events:

```rust
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct GpuEvent {
    position: [f32; 2],       // world-space location
    intensity: f32,           // magnitude
    color_id: u32,            // which color (maps to effect profile)
    event_type: u32,          // 0=destroy, 1=impact
    _pad: [u32; 3],
}
```

A separate uniform holds the event count for this frame. The injection compute pass reads events [0..count] and applies them.

**C. Block instance buffer**

Per-block instance data for the geometry render pass:

```rust
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct BlockInstance {
    // corners from the 4 spring-damper particles
    corners: [[f32; 2]; 4],
    color_id: u32,
    alive: u32,               // 0 = being destroyed (trigger dissolve shader)
    destroy_progress: f32,    // 0.0 → 1.0 animation
    _pad: f32,
}
```

### 6.3 Fluid Simulation (GPU Compute)

An Eulerian 2D grid-based fluid solver running entirely in compute shaders. Based on Jos Stam's stable fluids method, adapted for GPU.

**Grid resolution**: 256×256 as default on WASM. 512×512 on native. Configurable at init based on adapter limits.

**Textures** (all `Rgba32Float` or `Rg32Float`):

| Texture | Format | Purpose |
|---|---|---|
| `velocity_0` | `Rg32Float` | Velocity field (ping) |
| `velocity_1` | `Rg32Float` | Velocity field (pong) |
| `pressure_0` | `R32Float` | Pressure field (ping) |
| `pressure_1` | `R32Float` | Pressure field (pong) |
| `divergence` | `R32Float` | Divergence of velocity |
| `density_rgba` | `Rgba32Float` | Density per color channel (up to 4 colors packed) |
| `density_rgba_2` | `Rgba32Float` | Second density texture (if >4 colors) |
| `obstacle` | `Rgba16Float` | Boundary mask (R) + obstacle velocity (GB), CPU-uploaded |

**Compute passes per frame** (in order):

1. **Force injection** (`fluid_inject.wgsl`): Read event buffer. For each destroy event, apply radial velocity burst + density injection at the event position, scaled by intensity and parameterized by `color_id`. For impact events, apply a smaller burst scaled by impact velocity.

2. **Boundary enforcement** (`fluid_boundary.wgsl`): For every grid cell, sample the obstacle texture. If the mask channel (R) is ≥ 0.5, set the velocity field at that cell to the obstacle's velocity (from G and B channels). This is the standard moving-boundary no-penetration condition. The pressure solver in the next steps will generate the pressure gradients that push fluid laterally out of narrowing gaps — gas compression emerges naturally from incompressibility, with no explicit squeeze logic.

3. **Advect velocity** (`fluid_advect.wgsl`): Semi-Lagrangian advection. For each cell, trace backward along velocity, bilinearly sample the previous velocity field. Write to ping-pong target. Apply vorticity confinement here (curl → cross product force → add to velocity).

4. **Compute divergence** (`fluid_pressure.wgsl`, pass 1): Finite differences on the velocity field.

5. **Pressure solve** (`fluid_pressure.wgsl`, passes 2..N): Jacobi iteration. 20–40 iterations is typical; on WASM, 20 is a good perf/quality tradeoff. Each iteration dispatches the full grid once, ping-ponging between `pressure_0` and `pressure_1`. The obstacle texture enforces Neumann boundary (zero pressure gradient at solid walls).

6. **Pressure projection** (`fluid_project.wgsl`): Subtract pressure gradient from velocity to enforce incompressibility. Re-enforce obstacle boundary: at any cell where the obstacle mask is solid, reset velocity to the obstacle's stored velocity. This ensures the moving-boundary condition is maintained after projection.

7. **Advect density** (`fluid_advect.wgsl`, reused with different bindings): Advect each density channel through the now-divergence-free velocity field. Apply slight diffusion (controlled dissipation rate per color).

**Workgroup size**: 8×8 = 64 threads per workgroup (WebGPU-friendly default). Dispatch `ceil(grid_w/8) × ceil(grid_h/8)` workgroups per pass.

### 6.4 Particle System (GPU Compute)

A storage-buffer-based particle pool. Each particle:

```wgsl
struct Particle {
    pos: vec2<f32>,
    vel: vec2<f32>,
    color: vec4<f32>,
    life: f32,          // 0.0 → 1.0, decreasing
    size: f32,
    effect_type: u32,   // maps to render style
    flags: u32,
}
```

**Pool size**: 64K particles (WASM), 256K (native). Pre-allocated storage buffer.

**Emission**: When a destroy or impact event fires, the CPU writes an emission command to a small emission buffer: `{ position, count, color_id, effect_type, velocity_spread }`. The particle update compute pass reads these, atomically allocates from a free-list (atomic counter on the pool), and initializes new particles.

**Update pass** (`particle_update.wgsl`):
1. For each active particle, sample the fluid velocity texture at the particle's grid-space position.
2. Blend the fluid velocity into the particle velocity (configurable coupling strength — full coupling makes particles drift with smoke, low coupling keeps them ballistic with just a nudge).
3. Apply gravity, drag.
4. Integrate position (Euler is fine for visual particles).
5. Decrement lifetime. When lifetime ≤ 0, return to free-list.

**Render**: Instanced billboard quads. Vertex shader reads particle buffer, emits a screen-aligned quad. Fragment shader applies per-`effect_type` texturing: soft circle for smoke-like particles, sharp star for spark particles, stretched trail for fast-moving debris.

### 6.5 Per-Color Effect Profiles

Each block color maps to a destruction effect profile, defined as a data-driven struct:

```rust
struct ColorEffect {
    // Fluid injection
    smoke_density: f32,           // how much density to inject
    smoke_color: [f32; 4],        // RGBA of injected density
    velocity_burst_strength: f32, // radial velocity magnitude
    vorticity_boost: f32,         // extra curl at injection site

    // Particles
    particle_count: u32,          // how many to emit on destroy
    particle_speed_range: [f32; 2],
    particle_size_range: [f32; 2],
    particle_lifetime_range: [f32; 2],
    particle_color_start: [f32; 4],
    particle_color_end: [f32; 4],
    particle_effect_type: u32,    // render style index

    // Screen effects
    screen_shake_intensity: f32,
    bloom_boost: f32,
    distortion_radius: f32,
    distortion_strength: f32,
}
```

Example profiles:

- **Red**: Heavy smoke, slow billowing, deep crimson density, moderate particles with ember trails, strong bloom.
- **Blue**: Minimal smoke, sharp velocity burst (water-splash feel), many small fast particles, screen ripple distortion.
- **Green**: Swirling vortex (high `vorticity_boost`), spiral particle emission, green-tinted wispy density.
- **Yellow**: Bright flash (high `bloom_boost`), explosive radial particles, minimal smoke, strong screen shake.
- **Purple**: Dense lingering smoke cloud, slow fade, large soft particles, chromatic aberration distortion.

These profiles are loaded as a uniform buffer, indexed by `color_id` in the shaders.

### 6.6 Moving Boundary Fluid Interaction

Gas compression effects (e.g. air being squeezed out between two closing blocks) emerge naturally from the fluid solver rather than being driven by an explicit system. The mechanism:

1. The obstacle texture carries each solid cell's velocity (see §6.2A).
2. The boundary enforcement pass writes obstacle velocity into the fluid velocity field at solid cells.
3. Two blocks closing toward each other impose inward velocities on the fluid between them.
4. The pressure solver enforces incompressibility (divergence-free velocity field). Since the fluid between the blocks is being driven inward from both sides but cannot compress, the solver generates a pressure field that redirects flow laterally out the open sides of the gap.
5. As the gap narrows across successive frames, fewer fluid cells remain between the blocks, but the pressure gradient per cell increases — so the lateral outflow velocity accelerates naturally.
6. When the gap closes completely, both bodies become adjacent solid regions in the obstacle texture and the fluid simply flows around them.

This replaces any need for a CPU-side gap tracker or explicit squeeze event injection. The only CPU work is the obstacle texture rasterization with sweep fill (already required for correct boundary handling).

**Tuning**: If the effect is too subtle visually, scale up the obstacle velocities written to the texture by a factor (e.g. 1.5×). This exaggerates the compression effect without breaking the simulation — the pressure solver still produces a physically plausible flow pattern, just amplified. If the effect is too violent (fluid blowing out too fast), reduce the multiplier or increase fluid viscosity in the gap region.

**Sweep fill is critical here**: Without it, a block that moves 3 grid cells in one frame would teleport past the intervening fluid rather than pushing through it. The sweep fill ensures all intermediate cells are tagged as solid with the correct velocity, giving the pressure solver a continuous wall to work with.

### 6.7 Render Pipeline

**Pass 1 — Block geometry** (render pass):
- Instanced quads with per-instance corner positions from the spring-damper particles.
- Vertex shader deforms the quad mesh to match the soft body shape.
- Fragment shader: procedural jelly shading — slight internal refraction, specular highlight that shifts with deformation, color from `color_id`.
- Blocks being destroyed (`alive == 0`) run a dissolve shader keyed on `destroy_progress`: noise-based erosion from edges inward.

**Pass 2 — Fluid smoke** (render pass):
- Full-screen quad.
- Fragment shader samples `density_rgba` (and optionally `density_rgba_2`), maps each channel to its color profile's smoke color, composites additively.
- Apply exponential fog falloff based on density magnitude for depth-like effect.

**Pass 3 — Particles** (render pass):
- Instanced billboard quads.
- Depth-sorted is unnecessary for additive blending (most particle effects).

**Pass 4 — Post-process chain** (render passes):
1. **Bloom**: Threshold bright pixels → downscale chain (4 levels) → Kawase blur at each level → upscale and composite.
2. **Screen distortion**: Accumulate active distortion sources (from recent events) into a small distortion buffer. Fragment shader displaces UV lookup. Distortions fade out over time.
3. **Composite**: Tone mapping (ACES or similar), gamma correction, optional vignette, output to surface.

---

## 7. WASM-Specific Considerations

### 7.1 Build Pipeline

Use **Trunk** for the WASM build:
- Compiles to `wasm32-unknown-unknown`.
- Bundles the `.wasm` binary with an HTML host page.
- Handles `wasm-bindgen` glue generation.
- Serves locally with live reload during development.

`Trunk.toml`:
```toml
[build]
target = "crates/app/index.html"
dist = "dist"

[watch]
watch = ["crates", "assets"]
```

### 7.2 Async Initialization

WebGPU adapter and device requests are async. On native, `pollster::block_on` works. On WASM, use `wasm_bindgen_futures::spawn_local`. The init sequence:

1. Request adapter (`wgpu::Instance::request_adapter`).
2. Request device with required features and limits.
3. Query actual limits (grid resolution, particle count budgets).
4. Create all GPU resources.
5. Start the game loop via `winit`'s event loop (or `request_animation_frame` loop on web).

### 7.3 Frame Timing

`winit` on WASM uses `requestAnimationFrame` internally. For physics stability, use a fixed timestep accumulator:

```rust
const PHYSICS_DT: f32 = 1.0 / 60.0;
let mut accumulator: f32 = 0.0;

// Each frame:
let frame_dt = /* wall time delta, capped at 0.1s */;
accumulator += frame_dt;
while accumulator >= PHYSICS_DT {
    physics_step(PHYSICS_DT);
    accumulator -= PHYSICS_DT;
}
let alpha = accumulator / PHYSICS_DT; // interpolation factor for rendering
```

### 7.4 Input

Keyboard input via `winit` events (works on both native and WASM). Map:
- Arrow keys: move piece left/right, soft drop.
- A/D (or Z/X): rotate piece.
- Space: hard drop.
- Escape: pause.

On WASM, `winit` translates browser keyboard events. Ensure the canvas has focus — add a click-to-focus handler in the host HTML.

### 7.5 Performance Budgets (WASM @ 60 FPS)

| Subsystem | Target Budget |
|---|---|
| Physics step (Rapier) | ≤ 3 ms |
| CPU → GPU uploads (obstacle tex w/ velocity, events, instances) | ≤ 1.5 ms |
| Fluid sim (6 compute passes × 256² grid) | ≤ 4 ms GPU |
| Particle update (64K) | ≤ 1 ms GPU |
| Render passes (blocks + smoke + particles + post) | ≤ 4 ms GPU |
| **Total** | **≤ 13.5 ms** (headroom for 16.6 ms frame) |

If pressure solve iterations dominate, reduce from 20 to 12 or drop grid to 128×128. Profile early on target hardware.

### 7.6 Asset Loading

No heavy asset pipeline — the game is procedurally rendered. If particle sprite textures are needed, embed them as `include_bytes!` in the binary or load via `web-sys` fetch. Keep the WASM binary + assets under 5 MB for fast load times.

---

## 8. Implementation Phases

### Phase 0 — Scaffolding (1–2 weeks)

**Goal**: Window opens, wgpu renders a colored triangle, builds and runs in both native and WASM.

- [x] Workspace setup: `game`, `gpu`, `app` crates.
- [x] `winit` event loop with platform-conditional init (native vs WASM).
- [x] `wgpu` context creation, surface configuration.
- [x] Basic render pass: clear screen + hardcoded triangle.
- [x] Trunk build config, verify WASM runs in Chrome.
- [x] CI: build both targets (native + wasm).

### Phase 1 — Core Game Loop (2–3 weeks)

**Goal**: Triminos fall, stack, and clear. No visual effects. Blocks are flat colored quads.

- [ ] Rapier2D integration: world setup, gravity, walls.
- [ ] Trimino definitions: shapes, colors, random bag.
- [ ] Piece spawning and input-controlled movement/rotation.
- [ ] Contact-based stacking (restitution, friction tuning).
- [ ] Match-3+ detection: flood fill on contacting same-color cells.
- [ ] Destruction: remove matched cells, fire `DestroyEvent`.
- [ ] Scoring, level progression, increasing drop speed.
- [ ] Game over detection (stack reaches top).
- [ ] Instanced block rendering on GPU — flat color, no deformation yet.

### Phase 2 — Soft Body Deformation (1–2 weeks)

**Goal**: Blocks visually squish and bounce. The "jello" feel.

- [ ] Replace single rigid body per cell with 4-particle spring-damper cluster.
- [ ] Tune spring stiffness and damping for satisfying squish.
- [ ] Upload corner positions as instance data.
- [ ] Block vertex shader: deform quad from corner positions.
- [ ] Block fragment shader: procedural jelly material (fake refraction, specular).

### Phase 3 — Fluid Simulation Foundation (2–3 weeks)

**Goal**: Smoke appears and flows around blocks. Moving boundaries push fluid naturally.

- [ ] Allocate fluid grid textures and bind groups.
- [ ] Implement advection compute shader (semi-Lagrangian).
- [ ] Implement pressure solve (Jacobi iterations, ping-pong).
- [ ] Implement projection pass.
- [ ] Obstacle texture with velocity: rasterize physics bodies to `Rgba16Float` grid (mask + velocity).
- [ ] Sweep fill: fill intermediate cells between previous and current body positions.
- [ ] Boundary enforcement pass: set fluid velocity to obstacle velocity at solid cells.
- [ ] Test: manually inject density, watch it flow around blocks and get pushed by moving blocks.
- [ ] Smoke render pass: full-screen quad sampling density.
- [ ] Performance profiling on WASM — tune grid size and iteration count.

### Phase 4 — Destruction Effects (2 weeks)

**Goal**: Clearing blocks produces per-color smoke and particle explosions.

- [ ] Event ring buffer: CPU writes destroy events, GPU reads.
- [ ] Fluid injection pass: radial velocity burst + density at event positions.
- [ ] Particle system: storage buffer pool, emission from events, lifetime management.
- [ ] Particle compute update: gravity, drag, basic integration.
- [ ] Particle render: instanced billboards, additive blending.
- [ ] Per-color effect profiles: define 4–6 color schemes.
- [ ] Block dissolve shader: noise erosion keyed on `destroy_progress`.

### Phase 5 — Fluid–Physics Integration & Particle Coupling (1–2 weeks)

**Goal**: Gas visibly squeezes between closing blocks via the moving boundary system. Particles drift in smoke.

- [ ] Validate gas compression: drop block onto stack, verify fluid is pushed laterally by moving boundaries.
- [ ] Tune obstacle velocity multiplier for visual punch (start at 1.0×, try 1.5×).
- [ ] Verify sweep fill prevents tunneling at high drop speeds.
- [ ] Particle–fluid coupling: sample fluid velocity in particle update pass.
- [ ] Tune coupling strength per effect type (smoke particles: high coupling, sparks: low coupling).
- [ ] Stress test: rapid chain clears + fast drops, verify fluid stays stable and performance holds.

### Phase 6 — Post-Processing & Polish (2 weeks)

**Goal**: Bloom, distortion, screen shake. The game looks finished.

- [ ] Bloom: bright-pass threshold, downscale chain, Kawase blur, composite.
- [ ] Screen distortion buffer: accumulate per-event distortions, decay over time.
- [ ] Screen shake: sinusoidal offset in the camera uniform, triggered by impacts/clears.
- [ ] Chromatic aberration (optional, tied to specific color effects).
- [ ] Vignette, tone mapping, gamma.
- [ ] Ambient background: subtle parallax grid or gradient.
- [ ] UI: score display, level, next piece preview (simple text or quads).

### Phase 7 — Optimization & Shipping (2 weeks)

**Goal**: Stable 60 FPS on mid-range hardware in Chrome. Polished release.

- [ ] GPU profiling: `wgpu` timestamp queries (where supported), Chrome GPU profiler.
- [ ] Reduce pressure solve iterations if needed.
- [ ] Particle LOD: reduce max particles on low-end detected via adapter info.
- [ ] WASM binary size optimization: `wasm-opt -O3`, `lto = true`, `opt-level = 'z'`.
- [ ] Touch controls (stretch goal for mobile browsers).
- [ ] Hosting: static site deployment (Netlify, Vercel, or GitHub Pages).
- [ ] Cross-browser testing: Chrome, Edge, Safari 18+.

---

## 9. Risk Register

| Risk | Impact | Likelihood | Mitigation |
|---|---|---|---|
| WebGPU compute shaders unavailable in target browser | Blocks shipping | Low (Chrome/Edge stable, Safari 18+) | Display clear error message; no WebGL fallback |
| Fluid sim too slow at 256² on integrated GPUs | Visual quality degradation | Medium | Dynamic grid scaling: query adapter, start at 128² on weak hardware |
| Rapier's spring joints produce unstable soft bodies | Gameplay feel broken | Medium | Tune spring constants carefully; fall back to rigid bodies with vertex deformation shader only |
| WASM binary too large (>10 MB) | Slow page load | Low | No heavy dependencies; `wasm-opt`, LTO, no debug symbols in release |
| Pressure Jacobi solver doesn't converge in 20 iterations | Visual artifacts in fluid | Medium | Pre-condition with Red-Black Gauss-Seidel ordering if Jacobi is insufficient; or increase to 30 and accept perf cost |
| GPU buffer limits hit on low-end devices | Crash on init | Low | Query `maxStorageBufferBindingSize`, scale particle pool and grid accordingly |
| Moving boundary sweep fill too coarse at low grid resolution | Gas compression effect invisible — fluid teleports through obstacles | Medium | Increase grid resolution in the gap region or use sub-step obstacle rasterization (rasterize at 2× physics rate). Also scale obstacle velocity multiplier to compensate |
| Cross-platform determinism needed for replays | Feature scope creep | Low | Use Rapier `enhanced-determinism` from day one; defer replay system |

---

## 10. Open Design Questions

1. **How many colors?** The original had 5. More colors = more density channels = more GPU memory. With RGBA packing, 4 fit in one texture. 5–8 need two textures. Decision: start with 4 (one texture), add a second if more are needed.

2. **Chain reactions**: When cleared blocks cause others to fall and form new matches, should each chain level escalate the visual effects? Proposed: yes — each chain level multiplies `velocity_burst_strength` and `particle_count` by a scaling factor, creating escalating spectacle.

3. **Persistent ambient smoke**: Should there always be a low-level ambient haze in the play field, or only event-driven smoke? Proposed: thin ambient density injection at the bottom of the screen (rising heat-haze effect) so the fluid sim always has something visible, even before any clears happen.

4. **Piece preview**: The original showed the next piece. Should the preview also have fluid effects? Proposed: no — keep the preview clean as a UI element. Focus GPU budget on the main board.

5. **Multiplayer**: Out of scope for this plan. The architecture supports it (game state is deterministic and serializable) but the networking layer, matchmaking, and split-screen rendering are separate projects.