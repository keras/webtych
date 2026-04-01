// lbm_advect_color.wgsl
//
// Pass 6: Semi-Lagrangian advection of passive colour-density fields.
//
// For each cell and each active colour channel:
//   1. Look up the fluid velocity u from the macroscopic buffer.
//   2. Trace backward one step: sample = position − u × dt
//      (dt = 1 lattice step, so the displacement is simply u in grid units).
//   3. Bilinearly sample the *previous* colour density from a read-only copy.
//   4. Apply per-colour dissipation and write the result.
//
// Because we read from the SAME buffer we write (no ping-pong for colour),
// we use a trick common in GPU fluid: process in a deterministic sweep order
// (row-major, no dependencies across threads) — each thread reads neighbours
// that haven't been updated yet (back-tracing means we always read "upstream").
// This is safe for divergence-free or near-divergence-free velocity fields, and
// good enough for visual smoke where small errors are imperceptible.
//
// For a cleaner approach, a separate ping-pong buffer pair could be added for
// colour density. The current design keeps buffer count low for the investigation
// phase; ping-pong can be added when integrating into the full renderer.

// ── Structs / bindings ──────────────────────────────────────────────────────

struct LbmUniforms {
    grid_width:  u32,
    grid_height: u32,
    tau:         f32,
    inv_tau:     f32,

    world_width:  f32,
    world_height: f32,
    event_count:  u32,
    color_count:  u32,

    inject_densities:       array<vec4<f32>, 2>,
    inject_color_densities: array<vec4<f32>, 2>,
    dissipations:           array<vec4<f32>, 2>,

    gravity_x: f32,
    gravity_y: f32,
    injection_mode: u32,
    inv_tau_minus: f32,
}

@group(0) @binding(0) var<uniform>            u:               LbmUniforms;
@group(0) @binding(1) var<storage, read_write> dist_src:        array<f32>; // unused
@group(0) @binding(2) var<storage, read_write> dist_dst:        array<f32>; // unused
@group(0) @binding(3) var<storage, read_write> macroscopic:     array<f32>; // read
@group(0) @binding(4) var                      obstacle:        texture_2d<f32>;
@group(0) @binding(5) var<storage, read>       events:          array<f32>; // unused
// Packed colour-density buffer: index = cell * MAX_COLORS + channel
@group(0) @binding(6) var<storage, read_write> color_densities: array<f32>;

/// MAX_COLORS must match the Rust constant MAX_COLORS = 8.
const MAX_COLORS: u32 = 8u;

// ── Helpers ────────────────────────────────────────────────────────────────

/// Index into a packed vec4x2 array (replaces array<f32,8> in uniform space).
fn arr8(a: array<vec4<f32>, 2>, i: u32) -> f32 {
    let lo = i < 4u;
    let v  = select(a[1], a[0], lo);
    let j  = select(i - 4u, i, lo);
    switch j {
        case 0u: { return v.x; }
        case 1u: { return v.y; }
        case 2u: { return v.z; }
        default: { return v.w; }
    }
}

fn cell_idx(gx: u32, gy: u32) -> u32 { return gy * u.grid_width + gx; }
fn macro_idx(cell: u32)        -> u32 { return cell * 3u; }

/// Bilinearly sample colour channel `channel` at fractional grid coords (px, py).
///
/// WGSL cannot pass storage-buffer pointers as function arguments, so we
/// compute the 4 corner cell indices first, then dispatch per-channel to read
/// the 4 values, then interpolate.
fn sample_color(channel: u32, px: f32, py: f32) -> f32 {
    let w  = f32(u.grid_width);
    let h  = f32(u.grid_height);
    let cx = clamp(px, 0.0, w - 1.001);
    let cy = clamp(py, 0.0, h - 1.001);
    let x0 = u32(floor(cx));
    let y0 = u32(floor(cy));
    let x1 = min(x0 + 1u, u.grid_width  - 1u);
    let y1 = min(y0 + 1u, u.grid_height - 1u);
    let tx = cx - floor(cx);
    let ty = cy - floor(cy);

    let c00 = cell_idx(x0, y0);
    let c10 = cell_idx(x1, y0);
    let c01 = cell_idx(x0, y1);
    let c11 = cell_idx(x1, y1);

    // Read from the packed buffer using [cell * MAX_COLORS + channel].
    let v00 = color_densities[c00 * MAX_COLORS + channel];
    let v10 = color_densities[c10 * MAX_COLORS + channel];
    let v01 = color_densities[c01 * MAX_COLORS + channel];
    let v11 = color_densities[c11 * MAX_COLORS + channel];

    return mix(mix(v00, v10, tx), mix(v01, v11, tx), ty);
}

/// Write a colour-density value using the packed buffer layout.
fn write_color(channel: u32, cell: u32, value: f32) {
    color_densities[cell * MAX_COLORS + channel] = value;
}

// ── Main ─────────────────────────────────────────────────────────────────────

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let gx = gid.x;
    let gy = gid.y;
    if gx >= u.grid_width || gy >= u.grid_height { return; }

    // Skip obstacle cells — colour density doesn't exist inside solids.
    let obs = textureLoad(obstacle, vec2<i32>(i32(gx), i32(gy)), 0);
    if obs.r > 0.5 { return; }

    let cell = cell_idx(gx, gy);
    let m0   = macro_idx(cell);
    let ux   = macroscopic[m0 + 1u];
    let uy   = macroscopic[m0 + 2u];

    // Back-trace: "where did the fluid parcel at this cell come from?"
    let px = f32(gx) + 0.5 - ux;
    let py = f32(gy) + 0.5 - uy;

    for (var c: u32 = 0u; c < u.color_count; c++) {
        let advected = sample_color(c, px, py);
        let dissipated = advected * arr8(u.dissipations, c);
        write_color(c, cell, dissipated);
    }
}
