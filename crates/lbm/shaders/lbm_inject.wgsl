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
    _pad0:       f32,

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
    _pad1: u32,

    mrt_s: array<vec4<f32>, 3>,
}

struct GpuEvent {
    position:   vec2<f32>,
    intensity:  f32,
    stamp_radius: f32,
    color_id:   u32,
    event_type: u32,
    velocity_scale: f32,
    base_vel_x: f32,
    base_vel_y: f32,
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
@group(0) @binding(7) var injection_stamp: texture_2d<f32>;

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

// D2Q9 lattice velocity vectors (float) for equilibrium calculations.
const LEX = array<f32, 9>(0.0,  1.0, 0.0, -1.0,  0.0,  1.0, -1.0, -1.0,  1.0);
const LEY = array<f32, 9>(0.0,  0.0, 1.0,  0.0, -1.0,  1.0,  1.0, -1.0, -1.0);
const CS2: f32 = 1.0 / 3.0;

fn f_eq(i: u32, rho: f32, ux: f32, uy: f32) -> f32 {
    let eu = LEX[i] * ux + LEY[i] * uy;
    let uu = ux * ux + uy * uy;
    return weight(i) * rho * (1.0 + eu / CS2 + eu * eu / (2.0 * CS2 * CS2) - uu / (2.0 * CS2));
}

/// Add density and momentum to a cell as an additive source term.
///
/// Reads the cell's current macroscopic state and computes the equilibrium
/// delta relative to it, so individual f_i stay physical and never go negative
/// from the injector alone.
fn add_injection_delta(cell: u32, delta_rho: f32, ux_inject: f32, uy_inject: f32) {
    // Read current macroscopic state from the distributions.
    var rho_cur = 0.0;
    var mx_cur  = 0.0;
    var my_cur  = 0.0;
    for (var i: u32 = 0u; i < 9u; i++) {
        let fi = dist_src[dist_index(cell, i)];
        rho_cur += fi;
        mx_cur  += fi * LEX[i];
        my_cur  += fi * LEY[i];
    }
    if rho_cur < 1e-6 { rho_cur = 1e-6; }
    let ux_cur = mx_cur / rho_cur;
    let uy_cur = my_cur / rho_cur;

    // Target: add density; blend velocity by momentum conservation.
    let rho_new = rho_cur + delta_rho;
    let ux_new  = (rho_cur * ux_cur + delta_rho * ux_inject) / rho_new;
    let uy_new  = (rho_cur * uy_cur + delta_rho * uy_inject) / rho_new;

    // Delta = f_eq(new state) − f_eq(current state).
    // Sum is exactly +delta_rho and individual deltas are small perturbations.
    for (var i: u32 = 0u; i < 9u; i++) {
        let idx = dist_index(cell, i);
        dist_src[idx] += f_eq(i, rho_new, ux_new, uy_new)
                       - f_eq(i, rho_cur, ux_cur, uy_cur);
    }
}

/// Replace cell distributions with a full equilibrium state at injected density.
fn set_injection_equilibrium(cell: u32, rho: f32, ux: f32, uy: f32) {
    for (var i: u32 = 0u; i < 9u; i++) {
        dist_src[dist_index(cell, i)] = f_eq(i, rho, ux, uy);
    }
}

/// Accumulate colour density at the packed buffer slot for (cell, channel).
fn inject_color(cell: u32, channel: u32, amount: f32) {
    let idx = cell * MAX_COLORS + channel;
    color_densities[idx] += amount;
}

/// Sample the injection stamp texture at UV in [0,1].
fn sample_stamp(uv: vec2<f32>) -> vec4<f32> {
    let dims = textureDimensions(injection_stamp);
    let sx = min(dims.x - 1u, u32(clamp(uv.x, 0.0, 0.999999) * f32(dims.x)));
    let sy = min(dims.y - 1u, u32(clamp(uv.y, 0.0, 0.999999) * f32(dims.y)));
    return textureLoad(injection_stamp, vec2<i32>(i32(sx), i32(sy)), 0);
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

        let stamp_radius = max(ev.stamp_radius, 1e-4);
        let wx = (f32(gx) + 0.5) * u.world_width / f32(u.grid_width);
        let wy = (f32(gy) + 0.5) * u.world_height / f32(u.grid_height);

        let uv = vec2<f32>(
            (wx - ev.position.x) / (2.0 * stamp_radius) + 0.5,
            (wy - ev.position.y) / (2.0 * stamp_radius) + 0.5,
        );
        if uv.x < 0.0 || uv.x > 1.0 || uv.y < 0.0 || uv.y > 1.0 { continue; }

        let stamp = sample_stamp(uv);
        let mask = stamp.r;
        if mask <= 1e-4 { continue; }

        let cid = min(ev.color_id, u.color_count - 1u);

        // Pressure injection (Destroy at full strength, Impact at 25%).
        var rho_inject = arr8(u.inject_densities, cid) * ev.intensity * mask;
        if ev.event_type == 1u { rho_inject *= 0.25; }
        if rho_inject > 0.001 {
            let ux = ev.base_vel_x + stamp.g * ev.velocity_scale;
            let uy = ev.base_vel_y + stamp.b * ev.velocity_scale;
            if u.injection_mode == 1u {
                add_injection_delta(cell, rho_inject, ux, uy);
            } else {
                set_injection_equilibrium(cell, 1.0 + rho_inject, ux, uy);
            }
        }

        // Colour density injection (Destroy only).
        if ev.event_type == 0u {
            let color_amount = arr8(u.inject_color_densities, cid) * ev.intensity * mask;
            inject_color(cell, cid, color_amount);
        }
    }
}
