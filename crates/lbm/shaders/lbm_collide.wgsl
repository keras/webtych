// lbm_collide.wgsl
//
// Pass 2: TRT (Two-Relaxation-Time) collision.
//
// For each non-obstacle fluid cell, split f into symmetric (+) and
// antisymmetric (−) parts relative to opposite-direction pairs, then
// relax each independently:
//
//   f_i ← f_i − s⁺ × (f⁺_i − f⁺_i_eq) − s⁻ × (f⁻_i − f⁻_i_eq)
//
// where  f⁺_i = ½(f_i + f_ī),  f⁻_i = ½(f_i − f_ī)
// and    s⁺ = 1/τ⁺  (controls kinematic viscosity)
//        s⁻ = 1/τ⁻  (set via magic number Λ = (τ⁺−½)(τ⁻−½) = 3/16
//                     to eliminate wall-location errors)
//
// The boundary pass will later overwrite solid cells, so we skip them here
// to avoid wasted work (checking the obstacle texture is cheap but the
// arithmetic is not).

// ── Structs / bindings (same layout as inject pass) ──────────────────────────

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

@group(0) @binding(0) var<uniform>            u:         LbmUniforms;
@group(0) @binding(1) var<storage, read_write> dist_src:  array<f32>;
@group(0) @binding(2) var<storage, read_write> dist_dst:  array<f32>;  // unused
@group(0) @binding(3) var<storage, read_write> macroscopic: array<f32>;
@group(0) @binding(4) var obstacle: texture_2d<f32>;
@group(0) @binding(5) var<storage, read>       events:    array<f32>; // unused (opaque)
@group(0) @binding(6) var<storage, read_write> color_densities: array<f32>; // unused

// ── D2Q9 constants ─────────────────────────────────────────────────────────

// Lattice velocity vectors [i] → (ex, ey)
// 0=(0,0), 1=(1,0), 2=(0,1), 3=(-1,0), 4=(0,-1),
// 5=(1,1), 6=(-1,1), 7=(-1,-1), 8=(1,-1)
const EX = array<f32, 9>(0.0,  1.0, 0.0, -1.0,  0.0,  1.0, -1.0, -1.0,  1.0);
const EY = array<f32, 9>(0.0,  0.0, 1.0,  0.0, -1.0,  1.0,  1.0, -1.0, -1.0);
const W  = array<f32, 9>(
    4.0/9.0,
    1.0/9.0, 1.0/9.0, 1.0/9.0, 1.0/9.0,
    1.0/36.0, 1.0/36.0, 1.0/36.0, 1.0/36.0
);

const CS2: f32 = 1.0 / 3.0;   // lattice speed-of-sound squared

// Opposite-direction index: 0↔0, 1↔3, 2↔4, 5↔7, 6↔8
const OPP = array<u32, 9>(0u, 3u, 4u, 1u, 2u, 7u, 8u, 5u, 6u);

// ── Equilibrium ──────────────────────────────────────────────────────────────

fn f_eq(i: u32, rho: f32, ux: f32, uy: f32) -> f32 {
    let eu  = EX[i] * ux + EY[i] * uy;          // e_i · u
    let uu  = ux * ux + uy * uy;                  // |u|²
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

    // Skip only fully solid obstacle cells — partial PSM cells still need BGK collision.
    let obs = textureLoad(obstacle, vec2<i32>(i32(gx), i32(gy)), 0);
    if obs.r > 0.999 { return; }

    let cell = cell_index(gx, gy);
    let d0 = dist_idx(cell, 0u);

    // Read distributions.
    var f: array<f32, 9>;
    for (var i: u32 = 0u; i < 9u; i++) { f[i] = dist_src[d0 + i]; }

    // Extract macroscopic quantities.
    var rho = 0.0;
    var ux  = 0.0;
    var uy  = 0.0;
    for (var i: u32 = 0u; i < 9u; i++) {
        rho += f[i];
        ux  += f[i] * EX[i];
        uy  += f[i] * EY[i];
    }
    // Guard against zero density (can happen at initialisation edges).
    if rho < 1e-6 { rho = 1e-6; }
    ux /= rho;
    uy /= rho;

    // Clamp velocity to keep Mach number well below 1 (cs ≈ 0.577 in lattice units).
    // Without this, large density gradients can drive |u| → ∞ and produce NaN.
    let max_u = 0.25;
    ux = clamp(ux, -max_u, max_u);
    uy = clamp(uy, -max_u, max_u);

    // Apply gravity body-force via equilibrium velocity shift.
    // u_eff = u + g × τ drives the fluid toward a net drift at terminal velocity g.
    let ux_g = ux + u.gravity_x * u.tau;
    let uy_g = uy + u.gravity_y * u.tau;

    // TRT relaxation: relax symmetric and antisymmetric parts independently.
    for (var i: u32 = 0u; i < 9u; i++) {
        let j        = OPP[i];
        let eq_i     = f_eq(i, rho, ux_g, uy_g);
        let eq_j     = f_eq(j, rho, ux_g, uy_g);
        let sym_neq  = 0.5 * ((f[i] + f[j]) - (eq_i + eq_j));
        let asym_neq = 0.5 * ((f[i] - f[j]) - (eq_i - eq_j));
        f[i] = f[i] - u.inv_tau * sym_neq - u.inv_tau_minus * asym_neq;
    }

    // Write back.
    for (var i: u32 = 0u; i < 9u; i++) { dist_src[d0 + i] = f[i]; }

    // Cache macroscopic quantities for the extract pass (avoids recomputing).
    // The extract pass will overwrite with the post-stream values, but having
    // something reasonable here helps the advect pass on the first frame.
    let m0 = macro_idx(cell);
    macroscopic[m0]      = rho;
    macroscopic[m0 + 1u] = ux;
    macroscopic[m0 + 2u] = uy;
}
