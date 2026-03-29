// lbm_inject.wgsl
//
// Pass 1: Event injection.
//
// For each GpuEvent in the ring buffer, inject elevated fluid density and
// colour-density smoke at the event's grid position.  Events of type 0
// (Destroy) inject both pressure and colour density; type 1 (Impact) injects
// only a mild pressure pulse.

// ── Shared uniform ──────────────────────────────────────────────────────────

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
    _pad0: u32,
    _pad1: u32,
}

struct GpuEvent {
    position:   vec2<f32>,
    intensity:  f32,
    color_id:   u32,
    event_type: u32,
    _pad:       vec3<u32>,
}

// ── Bindings ─────────────────────────────────────────────────────────────────

@group(0) @binding(0) var<uniform>             u:               LbmUniforms;
@group(0) @binding(1) var<storage, read_write> dist_src:        array<f32>;  // 9×f32 per cell
@group(0) @binding(2) var<storage, read_write> dist_dst:        array<f32>;  // unused
@group(0) @binding(3) var<storage, read_write> macroscopic:     array<f32>;  // unused
@group(0) @binding(4) var                      obstacle:        texture_2d<f32>;
@group(0) @binding(5) var<storage, read>       events:          array<GpuEvent>;
// Packed colour-density buffer: index = cell * MAX_COLORS + channel
@group(0) @binding(6) var<storage, read_write> color_densities: array<f32>;

// ── D2Q9 constants ────────────────────────────────────────────────────────────

const W0: f32 = 4.0 / 9.0;
const W1: f32 = 1.0 / 9.0;
const W2: f32 = 1.0 / 36.0;
/// MAX_COLORS must match the Rust constant MAX_COLORS = 8.
const MAX_COLORS: u32 = 8u;

fn weight(i: u32) -> f32 {
    if i == 0u { return W0; }
    if i < 5u  { return W1; }
    return W2;
}

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

fn cell_index(gx: u32, gy: u32) -> u32 {
    return gy * u.grid_width + gx;
}

fn dist_index(cell: u32, i: u32) -> u32 {
    return cell * 9u + i;
}

/// Set all 9 distributions at `cell` to the D2Q9 equilibrium at rest
/// (u = 0) with density `rho`:  f_i = w_i * rho.
fn inject_equilibrium(cell: u32, rho: f32) {
    for (var i: u32 = 0u; i < 9u; i++) {
        dist_src[dist_index(cell, i)] = weight(i) * rho;
    }
}

/// Accumulate colour density at the packed buffer slot for (cell, channel).
fn inject_color(cell: u32, channel: u32, amount: f32) {
    let idx = cell * MAX_COLORS + channel;
    color_densities[idx] += amount;
}

// ── Main ───────────────────────────────────────────────────────────────────

// This pass is dispatched over the full grid (one thread per cell) but each
// thread iterates over all events.  For a small number of events (≤256) and a
// moderate grid (256²), this is faster than one-thread-per-event because it
// avoids a scatter pattern on the distribution buffer.

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let gx = gid.x;
    let gy = gid.y;
    if gx >= u.grid_width || gy >= u.grid_height { return; }

    let cell = cell_index(gx, gy);

    for (var e: u32 = 0u; e < u.event_count; e++) {
        let ev = events[e];

        // Convert world-space position to grid coordinates.
        let ex = (ev.position.x / u.world_width)  * f32(u.grid_width);
        let ey = (ev.position.y / u.world_height) * f32(u.grid_height);

        // Apply within a 3×3 cell radius (soft injection zone).
        let dx = f32(gx) + 0.5 - ex;
        let dy = f32(gy) + 0.5 - ey;
        let dist2 = dx * dx + dy * dy;
        let radius = 3.0;
        if dist2 > radius * radius { continue; }

        let falloff = 1.0 - sqrt(dist2) / radius;
        let cid = min(ev.color_id, u.color_count - 1u);

        // Pressure injection (Destroy at full strength, Impact at 25%).
        var rho_inject = arr8(u.inject_densities, cid) * ev.intensity * falloff;
        if ev.event_type == 1u { rho_inject *= 0.25; }
        if rho_inject > 0.001 {
            inject_equilibrium(cell, 1.0 + rho_inject);
        }

        // Colour density injection (Destroy only).
        if ev.event_type == 0u {
            let color_amount = arr8(u.inject_color_densities, cid) * ev.intensity * falloff;
            inject_color(cell, cid, color_amount);
        }
    }
}
