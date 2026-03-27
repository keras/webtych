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
│                   │  • spring joints │                │
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
│  │  Compute: Fluid Simulation (LBM D2Q9)      │      │
│  │  1. Event injection (density + body force)  │      │
│  │  2. Collision (BGK relaxation + forces)     │      │
│  │  3. Streaming (propagate distributions)     │      │
│  │  4. Boundary (bounce-back w/ obstacle vel)  │      │
│  │  5. Extract macroscopic velocity + density  │      │
│  │  6. Advect color density fields             │      │
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
│   │   │   │   ├── solver.rs     # LBM D2Q9: collision, streaming, macroscopic extraction
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
│   │   │   ├── lbm_collide.wgsl
│   │   │   ├── lbm_stream.wgsl
│   │   │   ├── lbm_boundary.wgsl
│   │   │   ├── lbm_inject.wgsl
│   │   │   ├── lbm_extract.wgsl
│   │   │   ├── density_advect.wgsl
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

**Rigid bodies**: Each trimino cell is a separate rigid body with a rectangular collider. Cells within the same piece are connected by **spring-damper joints** at their shared edges. The cells themselves are rigid — they don't deform — but the springs let the trimino flex, bounce, and wobble as a whole. On impact, cells separate slightly and oscillate, then pull back together. This produces the bouncy "jello" feel at the trimino level without any per-cell deformation.

Spring parameters (tune to taste):
- **Stiffness**: High enough that cells don't visibly separate during normal stacking, but low enough that impacts produce visible flex. Start around 500–1000 N/m.
- **Damping**: Moderate — enough to settle oscillations within ~0.5s but not so much that the wobble is killed instantly. Start around 10–20 Ns/m.
- **Rest length**: Zero (cells want to be flush against each other).

Restitution ~0.3 (bouncy but not crazy), friction ~0.5.

When cells are destroyed (match-3 clear), the spring joints connecting them to surviving neighbors are removed, and the surviving cells become independent or form smaller connected groups.

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
2. Rasterize the body's rectangular collider onto the grid, filling covered cells with `(1.0, vel.x, vel.y, 0.0)`.
3. **Sweep fill**: For each body, also fill the cells between last frame's position and this frame's position. This prevents fast-moving obstacles from tunneling through fluid cells in a single timestep. Sweep by translating the shape along its frame-to-frame displacement vector and filling all intermediate cells. Tag swept cells with the body's velocity.

Empty cells are `(0.0, 0.0, 0.0, 0.0)`. The LBM solver uses this texture in its boundary pass: at solid cells, the moving bounce-back scheme reflects distributions with a velocity correction from the obstacle velocity. This transfers momentum from moving obstacles into the fluid, causing gas to be pushed out of narrowing gaps naturally — no explicit squeeze injection needed.

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
    position: [f32; 2],       // center position from rigid body
    rotation: f32,            // angle in radians from rigid body
    color_id: u32,
    alive: u32,               // 0 = being destroyed (trigger dissolve shader)
    destroy_progress: f32,    // 0.0 → 1.0 animation
    _pad: [f32; 2],
}
```

### 6.3 Fluid Simulation (GPU Compute) — Lattice Boltzmann Method

A D2Q9 Lattice Boltzmann solver running entirely in compute shaders. Unlike the Stam stable-fluids approach (incompressible, iterative pressure solve), LBM is inherently compressible and non-iterative — pressure waves, gas expansion, and shock-like effects emerge naturally from the simulation. Each cell update is purely local (read neighbors, compute, write self), making it an ideal fit for GPU compute.

**Grid resolution**: 256×256 as default on WASM. 512×512 on native. Configurable at init based on adapter limits.

#### D2Q9 Lattice

Each cell stores 9 distribution functions `f_i` corresponding to the 9 lattice velocities:

```
6  2  5
 \ | /
3--0--1
 / | \
7  4  8
```

Velocities: `e_0 = (0,0)`, `e_1 = (1,0)`, `e_2 = (0,1)`, `e_3 = (-1,0)`, `e_4 = (0,-1)`, `e_5 = (1,1)`, `e_6 = (-1,1)`, `e_7 = (-1,-1)`, `e_8 = (1,-1)`.

Weights: `w_0 = 4/9`, `w_{1-4} = 1/9`, `w_{5-8} = 1/36`.

Macroscopic quantities extracted from distributions:
- Fluid density: `ρ = Σ f_i`
- Fluid velocity: `u = (1/ρ) Σ f_i × e_i`
- Pressure: `p = ρ × c_s²` where `c_s² = 1/3` (lattice speed of sound squared)

#### GPU Buffers

| Buffer | Format | Size (256² grid) | Purpose |
|---|---|---|---|
| `distributions_0` | Storage buffer, `f32 × 9` per cell | ~2.4 MB | Distribution functions (ping) |
| `distributions_1` | Storage buffer, `f32 × 9` per cell | ~2.4 MB | Distribution functions (pong) |
| `macroscopic` | Storage buffer, `f32 × 3` per cell | ~768 KB | Extracted ρ, u.x, u.y per cell |
| `obstacle` | `Rgba16Float` texture | ~512 KB | Boundary mask (R) + obstacle velocity (GB), CPU-uploaded |
| `density_colors` | Storage buffer, `f32 × N` per cell | ~256 KB per color | Passive scalar density per color |

Total GPU memory at 256²: ~7 MB. At 512²: ~28 MB. Well within WebGPU limits.

#### Compute Passes Per Frame (in order)

1. **Event injection** (`lbm_inject.wgsl`): Read the event ring buffer. For each destroy event, modify the distributions at the event cells to represent elevated fluid density (high ρ). This is done by setting `f_i = w_i × ρ_inject` at those cells, where `ρ_inject` is the desired overpressure. The resulting density imbalance with surrounding cells creates a pressure gradient that the collision and streaming steps will naturally expand outward. Also inject colored smoke density into the corresponding color channel. The `color_id` and intensity from the event parameterize both the LBM density injection strength and the smoke color density amount, controlled by the per-color effect profile.

2. **Collision** (`lbm_collide.wgsl`): For each non-obstacle cell, compute the BGK (Bhatnagar-Gross-Krook) relaxation:

   ```
   f_i_new = f_i - (1/τ) × (f_i - f_i_eq) + F_i
   ```

   Where `τ` is the relaxation time (controls viscosity: `ν = c_s² × (τ - 0.5)`), `f_i_eq` is the equilibrium distribution computed from local ρ and u, and `F_i` is an optional body force term (e.g. gravity on smoke, buoyancy). The equilibrium is:

   ```
   f_i_eq = w_i × ρ × (1 + (e_i · u)/c_s² + (e_i · u)²/(2×c_s⁴) - (u · u)/(2×c_s²))
   ```

   This is the most ALU-heavy pass but still a single dispatch over the full grid.

3. **Streaming** (`lbm_stream.wgsl`): Propagate each distribution to its neighbor cell along its lattice direction. `f_i` at cell `(x, y)` moves to cell `(x + e_i.x, y + e_i.y)`. This is a pure data-shuffle — each thread reads from its neighbor and writes to itself (pull scheme) or reads from itself and writes to its neighbor (push scheme). Pull scheme is simpler for boundary handling. Ping-pong between `distributions_0` and `distributions_1`.

4. **Boundary enforcement** (`lbm_boundary.wgsl`): For cells marked as solid in the obstacle texture, apply the **moving bounce-back** scheme. Distributions arriving at a solid cell are reflected back the way they came, with a velocity correction for the obstacle's motion:

   ```
   f_opposite(x, t+1) = f_i(x, t_post_collision) - 2 × w_i × ρ_wall × (e_i · u_wall) / c_s²
   ```

   Where `u_wall` is the obstacle velocity from the obstacle texture's GB channels, and `opposite` is the index of the reverse direction. This correctly transfers momentum from moving obstacles to the fluid — a falling block pushes fluid downward, two closing blocks squeeze fluid laterally — all emerging from the same bounce-back rule.

5. **Extract macroscopic quantities** (`lbm_extract.wgsl`): For each cell, sum the distributions to get ρ and u (see formulas above). Write to the `macroscopic` buffer. This velocity field is what particles sample for fluid coupling, and it can also be used in the smoke render pass if needed.

6. **Advect color densities** (`density_advect.wgsl`): Semi-Lagrangian advection of each passive color density field through the extracted velocity field. For each cell, trace backward along u, bilinearly sample the previous color density, write the result. Apply dissipation (multiply by a decay factor per color, per frame). This is the only pass that uses semi-Lagrangian advection — the LBM itself doesn't need it.

**Workgroup size**: 8×8 = 64 threads per workgroup. Dispatch `ceil(grid_w/8) × ceil(grid_h/8)` workgroups per pass.

#### Relaxation Time (τ) and Viscosity

The relaxation time τ controls how quickly the fluid relaxes toward equilibrium, which maps to kinematic viscosity: `ν = (1/3) × (τ - 0.5)`. For visual smoke:

- `τ = 0.6` → low viscosity, fast swirling, turbulent look.
- `τ = 0.8` → moderate viscosity, smooth billowing.
- `τ = 1.0+` → high viscosity, thick syrupy flow.

Start with `τ ≈ 0.7` and tune for visual feel. Values too close to 0.5 cause instability. This is a single uniform value — trivial to tweak at runtime.

#### Destruction as Pressure Injection

When blocks are destroyed, the cleared grid cells transition from obstacle (solid) to fluid (empty) and simultaneously receive a high-density distribution injection. The mechanism:

1. Obstacle texture is updated: destroyed block cells go from mask=1 to mask=0.
2. The injection pass sets distributions at those cells to equilibrium at elevated density `ρ_inject` (e.g. 3–5× ambient ρ₀=1.0) with zero velocity.
3. The collision pass sees these cells as high-density surrounded by ambient-density neighbors.
4. Streaming naturally propagates the excess density outward. The density gradient drives outward velocity — this is a compressible pressure wave.
5. Over several frames, the wave expands, reflects off obstacles, and dissipates.

Different colors inject different `ρ_inject` values (from the effect profile), producing different expansion strengths. A heavy color produces a powerful blast; a light color produces a gentle puff. The geometry of surrounding blocks shapes the expansion — a block destroyed in a tight corner produces a directed jet through the available opening, while one in open space produces a symmetric expansion. All of this is emergent, not authored.

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
1. For each active particle, sample the macroscopic velocity buffer at the particle's grid-space position.
2. Blend the fluid velocity into the particle velocity (configurable coupling strength — full coupling makes particles drift with smoke, low coupling keeps them ballistic with just a nudge).
3. Apply gravity, drag.
4. Integrate position (Euler is fine for visual particles).
5. Decrement lifetime. When lifetime ≤ 0, return to free-list.

**Render**: Instanced billboard quads. Vertex shader reads particle buffer, emits a screen-aligned quad. Fragment shader applies per-`effect_type` texturing: soft circle for smoke-like particles, sharp star for spark particles, stretched trail for fast-moving debris.

### 6.5 Per-Color Effect Profiles

Each block color maps to a destruction effect profile, defined as a data-driven struct:

```rust
struct ColorEffect {
    // Fluid injection (LBM)
    lbm_density_inject: f32,      // ρ_inject value (e.g. 2.0–5.0× ambient)
    smoke_density: f32,           // how much colored smoke density to inject
    smoke_color: [f32; 4],        // RGBA of injected smoke
    smoke_dissipation: f32,       // per-frame decay rate for this color's smoke

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

- **Red**: High `lbm_density_inject` (4.0×), heavy smoke, slow dissipation — a powerful billowing explosion with deep crimson density, moderate particles with ember trails, strong bloom.
- **Blue**: Moderate `lbm_density_inject` (2.5×), minimal smoke, fast dissipation — a sharp pressure pop (water-splash feel), many small fast particles, screen ripple distortion.
- **Green**: Moderate `lbm_density_inject` (3.0×), medium smoke — the expansion interacts with surrounding geometry to create swirling patterns, spiral particle emission, green-tinted wispy density.
- **Yellow**: High `lbm_density_inject` (5.0×), minimal smoke, fast dissipation — a powerful pressure wave with bright flash (`bloom_boost`), explosive radial particles, strong screen shake.
- **Purple**: Low `lbm_density_inject` (1.5×), very heavy smoke, very slow dissipation — a gentle expansion that leaves a dense lingering cloud, large soft particles, chromatic aberration distortion.

These profiles are loaded as a uniform buffer, indexed by `color_id` in the shaders.

### 6.6 Moving Boundary Fluid Interaction

Gas compression effects (e.g. air being squeezed out between two closing blocks) emerge naturally from the LBM solver rather than being driven by an explicit system. The mechanism:

1. The obstacle texture carries each solid cell's velocity (see §6.2A).
2. The boundary pass applies moving bounce-back: distributions arriving at solid cells are reflected with a velocity correction proportional to the obstacle's velocity. This transfers momentum from the moving obstacle into the fluid.
3. Two blocks closing toward each other impose inward momentum on the fluid between them via bounce-back at both surfaces.
4. Because LBM is inherently compressible, the fluid between the blocks actually compresses — density (and therefore pressure) rises in the narrowing gap.
5. The elevated pressure drives lateral outflow through the open sides of the gap. As the gap narrows, pressure increases and outflow accelerates.
6. When the gap closes completely, both bodies become adjacent solid regions in the obstacle texture and the fluid flows around them.

Compared to the Stam solver approach, the LBM version is more physically accurate here — the gas genuinely compresses before being expelled, rather than relying on an incompressibility constraint to redirect flow. This means you can see a brief pressure buildup before the lateral jet, which looks more natural.

**Tuning**: Scale obstacle velocities in the texture by a multiplier (e.g. 1.5×) to exaggerate the effect. Adjusting τ (relaxation time) also affects how the compression behaves — lower τ produces sharper, more turbulent jets, higher τ produces smoother expulsion.

**Sweep fill remains critical**: Without it, fast-moving blocks tunnel through fluid cells. The sweep fill ensures all intermediate cells receive bounce-back boundary conditions, giving the LBM a continuous moving wall to interact with.

### 6.7 Render Pipeline

**Pass 1 — Block geometry** (render pass):
- Instanced quads with per-instance position and rotation from the rigid body transforms.
- Vertex shader applies a simple 2D rotation + translation.
- Blocks being destroyed (`alive == 0`) run a dissolve shader keyed on `destroy_progress`: noise-based erosion from edges inward.

**Pass 2 — Fluid smoke** (render pass):
- Full-screen quad.
- Fragment shader samples each color's density field from the `density_colors` storage buffer, multiplies by the color profile's smoke tint, composites additively.
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
| LBM sim (inject + collide + stream + boundary + extract, 256² grid) | ≤ 3 ms GPU |
| Color density advection (per-color passive scalar) | ≤ 1 ms GPU |
| Particle update (64K) | ≤ 1 ms GPU |
| Render passes (blocks + smoke + particles + post) | ≤ 4 ms GPU |
| **Total** | **≤ 13.5 ms** (headroom for 16.6 ms frame) |

The LBM passes are fewer than the Stam solver (no iterative pressure solve), so the fluid sim budget is potentially tighter. The collision pass is ALU-heavier per cell but bandwidth-lighter overall. Profile early on target hardware.

### 7.6 Asset Loading

No heavy asset pipeline — the game is procedurally rendered. If particle sprite textures are needed, embed them as `include_bytes!` in the binary or load via `web-sys` fetch. Keep the WASM binary + assets under 5 MB for fast load times.

---

## 8. Implementation Phases

### Phase 0 — Scaffolding

**Goal**: Window opens, wgpu renders a colored triangle, builds and runs in both native and WASM.

- [x] Workspace setup: `game`, `gpu`, `app` crates.
- [x] `winit` event loop with platform-conditional init (native vs WASM).
- [x] `wgpu` context creation, surface configuration.
- [x] Basic render pass: clear screen + hardcoded triangle.
- [x] Trunk build config, verify WASM runs in Chrome.
- [x] CI: build both targets (native + wasm).

### Phase 1 — Core Game Loop

**Goal**: Triminos fall, stack, and clear. No visual effects. Blocks are flat colored quads.

- [x] Rapier2D integration: world setup, gravity, walls.
- [x] Trimino definitions: shapes, colors, random bag.
- [x] Piece spawning and input-controlled movement/rotation.
- [x] Contact-based stacking (restitution, friction tuning).
- [x] Match-3+ detection: flood fill on contacting same-color cells.
- [x] Destruction: remove matched cells, fire `DestroyEvent`.
- [x] Scoring, level progression, increasing drop speed.
- [x] Game over detection (stack reaches top).
- [x] Instanced block rendering on GPU — flat color, no deformation yet.

### Phase 2 — Spring-Connected Blocks

**Goal**: Blocks wobble and flex on impact. The trimino feels bouncy and alive.

- [x] Connect trimino cells with spring-damper joints at shared edges.
- [x] Tune spring stiffness and damping for satisfying wobble on impact.
- [x] Remove spring joints on cell destruction, let surviving cells separate or regroup.
- [x] Upload per-block position + rotation as instance data.
- [x] Block vertex shader: transform quad by position and rotation.

### Phase 3 — Fluid Simulation Foundation

**Goal**: LBM fluid sim runs on GPU. Smoke appears and flows around blocks. Moving boundaries push fluid naturally.

- [ ] Allocate LBM distribution buffers (2 × 9 floats per cell, ping-pong) and macroscopic buffer.
- [ ] Implement collision compute shader (BGK relaxation with equilibrium calculation).
- [ ] Implement streaming compute shader (pull scheme, ping-pong).
- [ ] Implement macroscopic extraction pass (ρ, u from distributions).
- [ ] Obstacle texture with velocity: rasterize physics bodies to `Rgba16Float` grid (mask + velocity).
- [ ] Sweep fill: fill intermediate cells between previous and current body positions.
- [ ] Implement moving bounce-back boundary pass using obstacle texture.
- [ ] Allocate color density buffers, implement semi-Lagrangian advection for passive scalars.
- [ ] Test: manually inject high-density region, watch pressure wave expand around blocks.
- [ ] Smoke render pass: full-screen quad sampling color density fields.
- [ ] Tune τ (relaxation time) for visual feel — start at 0.7.
- [ ] Performance profiling on WASM — tune grid size.

### Phase 4 — Destruction Effects

**Goal**: Clearing blocks produces per-color pressure explosions and particle effects.

- [ ] Event ring buffer: CPU writes destroy events, GPU reads.
- [ ] LBM injection pass: set elevated ρ in distributions at destroyed cells + inject colored smoke density.
- [ ] Verify pressure wave expands outward and is shaped by surrounding geometry.
- [ ] Particle system: storage buffer pool, emission from events, lifetime management.
- [ ] Particle compute update: gravity, drag, basic integration.
- [ ] Particle render: instanced billboards, additive blending.
- [ ] Per-color effect profiles: define 5+ color schemes with different `lbm_density_inject` values.
- [ ] Block dissolve shader: noise erosion keyed on `destroy_progress`.

### Phase 5 — Fluid–Physics Integration & Particle Coupling

**Goal**: Gas visibly compresses and squeezes between closing blocks via LBM bounce-back. Particles drift in smoke.

- [ ] Validate gas compression: drop block onto stack, verify fluid density rises in gap and expels laterally.
- [ ] Tune obstacle velocity multiplier for visual punch (start at 1.0×, try 1.5×).
- [ ] Tune τ for different visual qualities (turbulent vs smooth expulsion).
- [ ] Verify sweep fill prevents tunneling at high drop speeds.
- [ ] Particle–fluid coupling: sample extracted velocity field in particle update pass.
- [ ] Tune coupling strength per effect type (smoke particles: high coupling, sparks: low coupling).
- [ ] Stress test: rapid chain clears + fast drops, verify LBM stays stable and performance holds.

### Phase 6 — Post-Processing & Polish

**Goal**: Bloom, distortion, screen shake. The game looks finished.

- [ ] Bloom: bright-pass threshold, downscale chain, Kawase blur, composite.
- [ ] Screen distortion buffer: accumulate per-event distortions, decay over time.
- [ ] Screen shake: sinusoidal offset in the camera uniform, triggered by impacts/clears.
- [ ] Chromatic aberration (optional, tied to specific color effects).
- [ ] Vignette, tone mapping, gamma.
- [ ] Ambient background: subtle parallax grid or gradient.
- [ ] UI: score display, level, next piece preview (simple text or quads).

### Phase 7 — Optimization & Shipping

**Goal**: Stable 60 FPS on mid-range hardware in Chrome. Polished release.

- [ ] GPU profiling: `wgpu` timestamp queries (where supported), Chrome GPU profiler.
- [ ] LBM dispatch optimization: verify collision + streaming fit within budget; reduce grid if needed.
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
| Rapier's spring joints between cells produce jittery or unstable behavior | Gameplay feel broken | Low | Spring-connected rigid blocks are simpler than deformable soft bodies and well within Rapier's comfort zone. If needed, increase damping or switch to fixed joints (losing the wobble but keeping stability) |
| WASM binary too large (>10 MB) | Slow page load | Low | No heavy dependencies; `wasm-opt`, LTO, no debug symbols in release |
| Pressure Jacobi solver doesn't converge in 20 iterations | Visual artifacts in fluid | Medium | N/A — replaced by LBM which has no iterative solve. LBM stability depends on τ staying above 0.5 and Mach number staying low (velocity << c_s). Clamp injection density to prevent supersonic flow |
| GPU buffer limits hit on low-end devices | Crash on init | Low | Query `maxStorageBufferBindingSize`, scale particle pool and grid accordingly |
| Moving boundary sweep fill too coarse at low grid resolution | Gas compression effect invisible — fluid teleports through obstacles | Medium | Increase grid resolution in the gap region or use sub-step obstacle rasterization (rasterize at 2× physics rate). Also scale obstacle velocity multiplier to compensate |
| Cross-platform determinism needed for replays | Feature scope creep | Low | Use Rapier `enhanced-determinism` from day one; defer replay system |

---

## 10. Resolved Design Decisions

1. **Color count**: Minimum 5 (matching the original), configurable by difficulty setting. Higher difficulty adds more colors. The LBM architecture imposes no limit — each additional color is just one passive scalar field (~256 KB at 256²) and one advection dispatch per frame.

2. **Chain reactions**: Yes. Each chain level multiplies `lbm_density_inject` and `particle_count` by a scaling factor, creating escalating pressure waves and spectacle.

3. **Persistent ambient smoke**: Configurable. Implement thin ambient density injection (rising heat-haze) as a toggle. When enabled, the fluid sim always has something visible even before any clears happen. When disabled, smoke only appears from destruction events.

## 11. Open Design Questions

1. **Piece preview**: The original showed the next piece. Should the preview also have fluid effects? Proposed: no — keep the preview clean as a UI element. Focus GPU budget on the main board.

2. **Multiplayer**: Out of scope for this plan. The architecture supports it (game state is deterministic and serializable) but the networking layer, matchmaking, and split-screen rendering are separate projects.

---

## 12. Glossary

**Advection** — The transport of a quantity (like smoke density) by a velocity field. "Advect the density" means moving the smoke according to how the fluid is flowing.

**BGK (Bhatnagar-Gross-Krook)** — The collision model used in the LBM solver. Each cell's distributions relax toward an equilibrium state at a rate controlled by the relaxation time τ. Named after the three physicists who proposed it.

**Bind group** — A wgpu concept: a collection of GPU resources (buffers, textures, samplers) bound together and made accessible to a shader at a specific slot. Defined by a bind group layout.

**Bounce-back** — The boundary condition scheme used in LBM for solid walls. Distributions hitting a solid cell are reflected back in the opposite direction. "Moving bounce-back" adds a velocity correction so that moving walls transfer momentum to the fluid.

**Collision (LBM)** — The step where each cell's distributions are relaxed toward equilibrium. This is where viscosity, forces, and pressure behavior are determined. Purely local — each cell only reads its own data.

**Compute shader** — A GPU program that runs general-purpose computation (not rendering). Used here for the LBM solver, particle updates, and density advection. Written in WGSL for WebGPU.

**D2Q9** — The lattice configuration for 2D LBM: 2 dimensions, 9 velocity directions (center + 4 cardinal + 4 diagonal). Each cell stores 9 distribution values.

**Dispatch** — Launching a compute shader on the GPU. You specify how many workgroups to run; each workgroup contains a fixed number of threads.

**Distribution function (f_i)** — In LBM, the probability of finding a particle moving in direction `i` at a given cell. The 9 distributions per cell encode the fluid's density, velocity, and stress state. Not directly visible — macroscopic quantities are extracted from them.

**Equilibrium distribution (f_i_eq)** — The distribution state a cell would have if the fluid were locally at rest (in thermodynamic equilibrium) at its current density and velocity. The collision step drives distributions toward this state.

**Eulerian** — A simulation approach where the grid is fixed in space and the fluid moves through it. Both LBM and Stam's method are Eulerian. The alternative (Lagrangian) tracks individual fluid particles.

**Instanced rendering** — Drawing many copies of the same mesh (e.g. a quad) in one draw call, with per-instance data (position, color, etc.) varying each copy. Used for blocks and particles.

**Lattice Boltzmann Method (LBM)** — A fluid simulation technique based on kinetic theory rather than solving the Navier-Stokes equations directly. Fluid is modeled as distributions of fictitious particles on a discrete lattice. Collision and streaming steps recover correct macroscopic fluid behavior. Inherently compressible, non-iterative, and highly parallelizable.

**Macroscopic quantities** — Density (ρ) and velocity (u) — the human-scale fluid properties extracted from the LBM distributions by summing them. These are what you actually see and what particles interact with.

**Passive scalar** — A quantity carried by the fluid that doesn't affect the fluid's motion. The colored smoke densities are passive scalars — they ride the velocity field but don't change it. Contrast with the LBM fluid density (ρ), which directly determines pressure and drives flow.

**Ping-pong** — A double-buffering technique where you alternate reading from buffer A and writing to buffer B, then swap roles next frame. Avoids read-write hazards on the GPU. Used for the LBM distribution buffers.

**Pull scheme** — A streaming implementation where each cell reads distributions from its neighbors (pulling data inward). The alternative "push scheme" has each cell write to its neighbors. Pull is simpler for boundary handling.

**Relaxation time (τ)** — The LBM parameter controlling how fast distributions approach equilibrium. Directly determines kinematic viscosity: `ν = (1/3)(τ - 0.5)`. Lower τ → less viscous, more turbulent. Must stay above 0.5 for stability.

**Semi-Lagrangian advection** — A technique for moving a field through a velocity field: for each cell, trace backward along the velocity to find where the material came from, then interpolate. Used here only for the passive color densities, not the LBM itself.

**Spring-damper joint** — A physics joint that connects two rigid bodies with a spring force (pulls them toward a rest distance) and a damping force (resists relative velocity). Used here to connect trimino cells — the spring provides the wobble, the damper prevents endless oscillation.

**Storage buffer** — A GPU buffer that compute shaders can read from and write to. Unlike uniform buffers, storage buffers can be large and support random access. Used for LBM distributions, particles, and color densities.

**Streaming (LBM)** — The step where each distribution is shifted to its neighboring cell along its lattice direction. This propagates information (pressure waves, momentum) through the grid. Combined with collision, it produces correct fluid dynamics.

**Sweep fill** — When rasterizing the obstacle texture, filling not just the body's current position but all cells between the previous and current positions. Prevents fast-moving objects from tunneling through the fluid grid between frames.

**Trimino** — A game piece made of three connected cells (the Triptych equivalent of a Tetris tetromino). Each cell is a separate rigid body connected to its neighbors by spring-damper joints, allowing the piece to flex and wobble. Each cell has a color and can be individually destroyed when matched.

**Workgroup** — A group of GPU threads that execute together and can share local memory. WebGPU limits apply per-workgroup. This project uses 8×8 = 64 threads per workgroup, the recommended default for WebGPU portability.

**WGSL (WebGPU Shading Language)** — The shader language for WebGPU. Syntactically similar to Rust. All shaders in this project are WGSL — it's the only option when targeting the browser via wgpu.

**wgpu** — A Rust library that provides a safe, cross-platform GPU abstraction based on the WebGPU standard. Runs natively on Vulkan/Metal/DX12 and in the browser via WebAssembly + WebGPU.
