// lbm_collide.wgsl
//
// Pass 2: MRT (Multiple Relaxation Time) collision.
//
// Transform f to moment space via M, relax each moment independently with
// its own rate s_i, then transform back:
//
//   m     = M f
//   m_neq = s ⊙ (m − m_eq)
//   f'    = f − M⁻¹ m_neq
//
// Moment ordering (D2Q9, Lallemand & Luo 2000):
//   0: ρ   (density)        — conserved, always zero non-eq
//   1: e   (energy)         — relaxes at s_e  (bulk viscosity)
//   2: ε   (ghost energy)   — relaxes at s_e
//   3: jx  (x-momentum)     — conserved, always zero non-eq
//   4: qx  (x-energy flux)  — relaxes at s_q
//   5: jy  (y-momentum)     — conserved, always zero non-eq
//   6: qy  (y-energy flux)  — relaxes at s_q
//   7: Pxx (normal stress)  — relaxes at s_ν = 1/τ  (kinematic viscosity)
//   8: Pxy (shear stress)   — relaxes at s_ν = 1/τ
//
// Packed in uniforms as mrt_s (3 vec4f, 12 slots, 9 used):
//   mrt_s[0] = (s0, s1, s2, s3)   s0/s3 unused (conserved)
//   mrt_s[1] = (s4, s5, s6, s7)   s5 unused (conserved)
//   mrt_s[2] = (s8, _, _, _)
//
// The boundary pass will later overwrite solid cells, so we skip them here
// to avoid wasted work (checking the obstacle texture is cheap but the
// arithmetic is not).

// ── Structs / bindings ───────────────────────────────────────────────────────

struct LbmUniforms {
    grid_width:  u32,
    grid_height: u32,
    tau:         f32,
    _pad0:       f32,   // unused

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
    solid_push_fluid: u32,

    mrt_s: array<vec4<f32>, 3>,  // [s0..s3], [s4..s7], [s8, _, _, _]
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

fn push_fluid_toward(d0: u32, rho: f32, vel_x: f32, vel_y: f32) {
    let vel  = vec2<f32>(vel_x, vel_y);
    let vlen = length(vel);
    let dir  = select(vec2<f32>(0.0, 0.0), vel / vlen, vlen > 1e-6);
    var w: array<f32, 9>;
    var wsum = 0.0;
    for (var i: u32 = 0u; i < 9u; i++) {
        let proj = EX[i] * dir.x + EY[i] * dir.y;
        w[i] = max(0.0, proj);
        wsum += w[i];
    }
    if wsum > 1e-6 {
        for (var i: u32 = 0u; i < 9u; i++) {
            dist_src[d0 + i] = rho * w[i] / wsum;
        }
    } else {
        dist_src[d0] = rho;
        for (var i: u32 = 1u; i < 9u; i++) { dist_src[d0 + i] = 0.0; }
    }
}

// ── Main ─────────────────────────────────────────────────────────────────────

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let gx = gid.x;
    let gy = gid.y;
    if gx >= u.grid_width || gy >= u.grid_height { return; }

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

    // Skip only fully solid obstacle cells — partial PSM cells still need collision.
    let obs = textureLoad(obstacle, vec2<i32>(i32(gx), i32(gy)), 0);
    if obs.r > 0.999 {
        if u.solid_push_fluid != 0u {
            push_fluid_toward(d0, rho, obs.g, obs.b);
        }
        return;
    }

    // Clamp velocity to keep Mach number well below 1 (cs ≈ 0.577 in lattice units).
    // Without this, large density gradients can drive |u| → ∞ and produce NaN.
    let max_u = 0.25;
    ux = clamp(ux, -max_u, max_u);
    uy = clamp(uy, -max_u, max_u);

    // Apply gravity body-force via equilibrium velocity shift.
    // u_eff = u + g × τ drives the fluid toward a net drift at terminal velocity g.
    let ux_g = ux + u.gravity_x * u.tau;
    let uy_g = uy + u.gravity_y * u.tau;

    // ── Transform to moment space: m = M f ───────────────────────────────────
    //
    // M rows (Lallemand & Luo 2000 D2Q9 basis), velocity ordering 0..8:
    //   ρ:   [ 1, 1, 1, 1, 1, 1, 1, 1, 1]
    //   e:   [-4,-1,-1,-1,-1, 2, 2, 2, 2]
    //   ε:   [ 4,-2,-2,-2,-2, 1, 1, 1, 1]
    //   jx:  [ 0, 1, 0,-1, 0, 1,-1,-1, 1]
    //   qx:  [ 0,-2, 0, 2, 0, 1,-1,-1, 1]
    //   jy:  [ 0, 0, 1, 0,-1, 1, 1,-1,-1]
    //   qy:  [ 0, 0,-2, 0, 2, 1, 1,-1,-1]
    //   Pxx: [ 0, 1,-1, 1,-1, 0, 0, 0, 0]
    //   Pxy: [ 0, 0, 0, 0, 0, 1,-1, 1,-1]

    let m1 = -4.0*f[0] - f[1]   - f[2]   - f[3]   - f[4]   + 2.0*f[5] + 2.0*f[6] + 2.0*f[7] + 2.0*f[8];
    let m2 =  4.0*f[0] - 2.0*f[1] - 2.0*f[2] - 2.0*f[3] - 2.0*f[4] +     f[5] +     f[6] +     f[7] +     f[8];
    let m4 = -2.0*f[1]             + 2.0*f[3]             +     f[5] -     f[6] -     f[7] +     f[8];
    let m6 =           - 2.0*f[2]             + 2.0*f[4]  +     f[5] +     f[6] -     f[7] -     f[8];
    let m7 =      f[1] -     f[2]  +     f[3] -     f[4];
    let m8 =                                                    f[5] -     f[6] +     f[7] -     f[8];
    // m0 = rho, m3 = rho*ux, m5 = rho*uy (already known from macroscopic extraction)

    // ── Equilibrium moments ───────────────────────────────────────────────────
    let u2     = ux_g * ux_g + uy_g * uy_g;
    let meq1   = rho * (-2.0 + 3.0 * u2);
    let meq2   = rho * ( 1.0 - 3.0 * u2);
    let meq4   = -rho * ux_g;
    let meq6   = -rho * uy_g;
    let meq7   =  rho * (ux_g * ux_g - uy_g * uy_g);
    let meq8   =  rho * ux_g * uy_g;

    // ── Relaxation: v[i] = s[i] × (m[i] − m_eq[i]) ──────────────────────────
    // Conserved moments (0, 3, 5) have zero non-equilibrium by definition.
    let s1 = u.mrt_s[0].y;   // s_e  (energy)
    let s2 = u.mrt_s[0].z;   // s_ε  (ghost energy)
    let s4 = u.mrt_s[1].x;   // s_q  (x-energy flux)
    let s6 = u.mrt_s[1].z;   // s_q  (y-energy flux)
    let s7 = u.mrt_s[1].w;   // s_ν  (normal stress = 1/τ)
    let s8 = u.mrt_s[2].x;   // s_ν  (shear stress  = 1/τ)

    let v1 = s1 * (m1 - meq1);
    let v2 = s2 * (m2 - meq2);
    let v4 = s4 * (m4 - meq4);
    let v6 = s6 * (m6 - meq6);
    let v7 = s7 * (m7 - meq7);
    let v8 = s8 * (m8 - meq8);

    // ── Back-transform: f -= M⁻¹ v ───────────────────────────────────────────
    //
    // M⁻¹ = M^T diag(1/w), w = [9,36,36,6,12,6,12,4,4].
    // Rows for each direction j (conserved v0/v3/v5 = 0, omitted):
    //   j=0:  -v1/9  + v2/9
    //   j=1:  -v1/36 - v2/18         - v4/6         + v7/4
    //   j=2:  -v1/36 - v2/18                - v6/6  - v7/4
    //   j=3:  -v1/36 - v2/18         + v4/6         + v7/4
    //   j=4:  -v1/36 - v2/18                + v6/6  - v7/4
    //   j=5:  +v1/18 + v2/36 + v4/12 + v6/12        + v8/4
    //   j=6:  +v1/18 + v2/36 - v4/12 + v6/12        - v8/4
    //   j=7:  +v1/18 + v2/36 - v4/12 - v6/12        + v8/4
    //   j=8:  +v1/18 + v2/36 + v4/12 - v6/12        - v8/4

    let c12 = -v1 / 36.0 - v2 / 18.0;   // shared by j=1..4
    let c58 =  v1 / 18.0 + v2 / 36.0;   // shared by j=5..8

    f[0] -= -v1 / 9.0  + v2 / 9.0;
    f[1] -= c12 - v4 / 6.0  + v7 / 4.0;
    f[2] -= c12 - v6 / 6.0  - v7 / 4.0;
    f[3] -= c12 + v4 / 6.0  + v7 / 4.0;
    f[4] -= c12 + v6 / 6.0  - v7 / 4.0;
    f[5] -= c58 + v4 / 12.0 + v6 / 12.0 + v8 / 4.0;
    f[6] -= c58 - v4 / 12.0 + v6 / 12.0 - v8 / 4.0;
    f[7] -= c58 - v4 / 12.0 - v6 / 12.0 + v8 / 4.0;
    f[8] -= c58 + v4 / 12.0 - v6 / 12.0 - v8 / 4.0;

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
