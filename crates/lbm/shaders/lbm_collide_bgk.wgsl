// lbm_collide_bgk.wgsl
//
// Pass 2: BGK (single-relaxation-time) collision.
//
// For each non-obstacle fluid cell, compute the equilibrium distribution
// from the local macroscopic rho and u, then relax toward it:
//   f_i <- f_i - (1/tau) * (f_i - f_i_eq)
//
// The boundary pass will later overwrite solid cells, so we skip them here
// to avoid wasted work (checking the obstacle texture is cheap but the
// arithmetic is not).

// ── Structs / bindings ───────────────────────────────────────────────────────

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

    mrt_s: array<vec4<f32>, 3>,  // unused by BGK
}

@group(0) @binding(0) var<uniform>            u:         LbmUniforms;
@group(0) @binding(1) var<storage, read_write> dist_src:  array<f32>;
@group(0) @binding(2) var<storage, read_write> dist_dst:  array<f32>;  // unused
@group(0) @binding(3) var<storage, read_write> macroscopic: array<f32>;
@group(0) @binding(4) var obstacle: texture_2d<f32>;
@group(0) @binding(5) var<storage, read>       events:    array<f32>; // unused
@group(0) @binding(6) var<storage, read_write> color_densities: array<f32>; // unused

// ── D2Q9 constants ─────────────────────────────────────────────────────────

const EX = array<f32, 9>(0.0,  1.0, 0.0, -1.0,  0.0,  1.0, -1.0, -1.0,  1.0);
const EY = array<f32, 9>(0.0,  0.0, 1.0,  0.0, -1.0,  1.0,  1.0, -1.0, -1.0);
const W  = array<f32, 9>(
    4.0/9.0,
    1.0/9.0, 1.0/9.0, 1.0/9.0, 1.0/9.0,
    1.0/36.0, 1.0/36.0, 1.0/36.0, 1.0/36.0
);

const CS2: f32 = 1.0 / 3.0;

// ── Equilibrium ──────────────────────────────────────────────────────────────

fn f_eq(i: u32, rho: f32, ux: f32, uy: f32) -> f32 {
    let eu  = EX[i] * ux + EY[i] * uy;
    let uu  = ux * ux + uy * uy;
    return W[i] * rho * (1.0 + eu / CS2 + eu * eu / (2.0 * CS2 * CS2) - uu / (2.0 * CS2));
}

// ── Helpers ──────────────────────────────────────────────────────────────────

fn cell_index(gx: u32, gy: u32) -> u32 { return gy * u.grid_width + gx; }
fn dist_idx(cell: u32, i: u32) -> u32 { return cell * 9u + i; }
fn macro_idx(cell: u32) -> u32         { return cell * 3u; }

// ── Main ─────────────────────────────────────────────────────────────────────

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let gx = gid.x;
    let gy = gid.y;
    if gx >= u.grid_width || gy >= u.grid_height { return; }

    let obs = textureLoad(obstacle, vec2<i32>(i32(gx), i32(gy)), 0);
    if obs.r > 0.999 { return; }

    let cell = cell_index(gx, gy);
    let d0 = dist_idx(cell, 0u);

    var f: array<f32, 9>;
    for (var i: u32 = 0u; i < 9u; i++) { f[i] = dist_src[d0 + i]; }

    var rho = 0.0;
    var ux  = 0.0;
    var uy  = 0.0;
    for (var i: u32 = 0u; i < 9u; i++) {
        rho += f[i];
        ux  += f[i] * EX[i];
        uy  += f[i] * EY[i];
    }
    if rho < 1e-6 { rho = 1e-6; }
    ux /= rho;
    uy /= rho;

    let max_u = 0.25;
    ux = clamp(ux, -max_u, max_u);
    uy = clamp(uy, -max_u, max_u);

    let ux_g = ux + u.gravity_x * u.tau;
    let uy_g = uy + u.gravity_y * u.tau;

    // BGK relaxation: f_i <- f_i - (1/tau) * (f_i - f_eq_i)
    let inv_tau = 1.0 / u.tau;
    for (var i: u32 = 0u; i < 9u; i++) {
        f[i] = f[i] - inv_tau * (f[i] - f_eq(i, rho, ux_g, uy_g));
    }

    for (var i: u32 = 0u; i < 9u; i++) { dist_src[d0 + i] = f[i]; }

    let m0 = macro_idx(cell);
    macroscopic[m0]      = rho;
    macroscopic[m0 + 1u] = ux;
    macroscopic[m0 + 2u] = uy;
}
