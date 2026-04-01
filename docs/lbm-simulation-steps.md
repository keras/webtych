# LBM Simulation — Step Overview

The simulation uses a **D2Q9 lattice** and runs **6 GPU compute passes per substep**.
Each call to `Simulation::step()` dispatches them in this order.

---

## Pass 1 — INJECT (`lbm_inject.wgsl`)

Processes queued game events. For each event (block destroy / impact), stamps fluid
density and colour-smoke onto the grid at the event's position. Uses a stamp texture
to spread the injection over an area.

Two modes:
- **Additive (`add_injection_delta`)** — adds density while preserving momentum conservation.
- **Replacement (`set_injection_equilibrium`)** — overwrites with a full equilibrium state.

Event type 0 (Destroy) injects both pressure and colour; type 1 (Impact) injects mild
pressure only (25%).

---

## Pass 2 — COLLIDE (`lbm_collide.wgsl`)

BGK single-relaxation-time collision:

```
f_i ← f_i − (1/τ) × (f_i − f_i_eq)
```

Each non-obstacle cell relaxes its 9 distribution functions toward the local equilibrium
(`f_eq`). Gravity is applied as a velocity shift inside the equilibrium computation.
Velocity is clamped at `max_u = 0.25` to prevent Mach instability.

---

## Pass 3 — STREAM (`lbm_stream.wgsl`)

Pull-scheme propagation: each cell pulls its incoming distributions from neighbours.

```
dst[cell, i] = src[cell − e_i, i]
```

Writes to the ping-pong destination buffer. After this pass the buffers are swapped
(`Grid::swap()`) so the destination becomes the source for the next substep.

---

## Pass 4 — BOUNDARY (`lbm_boundary.wgsl`)

Two boundary condition types:

- **Open boundary / Zou-He outflow**: cells marked with A=1.0 are forced to equilibrium
  at ambient density (ρ=1.0), draining excess pressure while preserving local velocity.
  Configured via `Simulation::set_open_boundaries()`.
- **Moving bounce-back / PSM**: solid obstacle cells reflect distributions back into the
  fluid, accounting for obstacle velocity. Blended by solid fill fraction
  (Partial Smooth Mover).

---

## Pass 5 — EXTRACT (`lbm_extract.wgsl`)

Sums the 9 post-stream distributions to recover macroscopic quantities per cell:

```
ρ    = Σ f_i
u_x  = Σ (f_i × e_ix) / ρ
u_y  = Σ (f_i × e_iy) / ρ
```

Writes to the macroscopic buffer (`[ρ, u_x, u_y]` per cell). This buffer is read by
the CPU for particle integration and by the advect pass.

---

## Pass 6 — ADVECT (`lbm_advect_color.wgsl`)

Semi-Lagrangian advection of up to 8 passive colour-density fields (one per block colour).
For each cell:

1. Look up fluid velocity `u` from the macroscopic buffer.
2. Back-trace: `sample_pos = position − u × dt` (dt = 1 lattice step).
3. Bilinearly sample the previous colour value (`sample_color()`).
4. Apply per-colour dissipation: `new_color = old_color × dissipation[c]`.

No ping-pong for the colour buffer — the back-tracing and row-major sweep order make
in-place writes safe.

---

## Ping-Pong Buffer Management

Distributions use two GPU storage buffers (A and B). The `pong` flag tracks which is
the current source. After each substep `Grid::swap()` flips source and destination.

| Buffer | Read as | Written as |
|--------|---------|------------|
| Binding 1 | `dist_src` | — |
| Binding 2 | — | `dist_dst` |

---

## D2Q9 Lattice Constants

| Index | Direction | e_i     | Weight  |
|-------|-----------|---------|---------|
| 0     | rest      | (0, 0)  | 4/9     |
| 1–4   | cardinal  | ±x, ±y  | 1/9     |
| 5–8   | diagonal  | ±x ±y   | 1/36    |

Speed of sound squared: `c_s² = 1/3`

---

## Key Files

| File | Role |
|------|------|
| `simulation.rs` | Top-level orchestration (`Simulation::step()`), event upload, substep loop |
| `gpu.rs` | Pipeline dispatch (`encode_lbm_passes`) |
| `grid.rs` | Ping-pong buffer management (`Grid::swap()`) |
| `lbm_inject.wgsl` | Event injection |
| `lbm_collide.wgsl` | BGK collision |
| `lbm_stream.wgsl` | Streaming / propagation |
| `lbm_boundary.wgsl` | Open boundary + moving bounce-back |
| `lbm_extract.wgsl` | Macroscopic extraction |
| `lbm_advect_color.wgsl` | Colour-smoke advection |
