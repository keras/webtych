// lbm_boundary.wgsl
//
// Pass 4: Moving bounce-back boundary conditions (pull-gather scheme).
//
// For each non-solid cell we apply the moving bounce-back scheme by *pulling*
// contributions from solid neighbours rather than having solid cells push to
// fluid neighbours.  The formula is the same:
//
//   f_opp(x_fluid, t+1) = f_j(x_solid, t) − 2 × w_j × ρ_wall × (e_j · u_wall) / c_s²
//
// where x_solid is the solid neighbour in direction e_j from x_fluid, and
// f_j(x_solid) is the distribution that streamed FROM x_fluid INTO x_solid
// in direction j (i.e. dist_dst[solid_cell, j] after the stream pass).
//
// Pull-gather eliminates write-write races: each thread writes exclusively to
// dist_dst[this_fluid_cell, ...] — no other thread touches those slots.

// ── Structs / bindings ──────────────────────────────────────────────────────

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

@group(0) @binding(0) var<uniform>            u:         LbmUniforms;
@group(0) @binding(1) var<storage, read_write> dist_src: array<f32>; // unused
@group(0) @binding(2) var<storage, read_write> dist_dst: array<f32>; // post-stream: r/w
@group(0) @binding(3) var<storage, read_write> macroscopic: array<f32>; // unused
@group(0) @binding(4) var obstacle: texture_2d<f32>;
@group(0) @binding(5) var<storage, read>       events:   array<f32>; // unused
@group(0) @binding(6) var<storage, read_write> color_densities: array<f32>; // unused

// ── D2Q9 constants ────────────────────────────────────────────────────────────

// Lattice velocity vectors e_i.
const EX = array<i32, 9>( 0,  1,  0, -1,  0,  1, -1, -1,  1);
const EY = array<i32, 9>( 0,  0,  1,  0, -1,  1,  1, -1, -1);

// Weights w_i.
const W = array<f32, 9>(
    4.0/9.0,
    1.0/9.0, 1.0/9.0, 1.0/9.0, 1.0/9.0,
    1.0/36.0, 1.0/36.0, 1.0/36.0, 1.0/36.0
);

// Opposite direction index: OPPOSITE[i] gives the index j such that e_j == -e_i.
const OPPOSITE = array<u32, 9>(0u, 3u, 4u, 1u, 2u, 7u, 8u, 5u, 6u);

const CS2: f32 = 1.0 / 3.0;
const RHO_WALL: f32 = 1.0; // reference density used in the correction term

// ── Helpers ───────────────────────────────────────────────────────────────────

fn in_bounds(gx: i32, gy: i32) -> bool {
    return gx >= 0 && gy >= 0 && u32(gx) < u.grid_width && u32(gy) < u.grid_height;
}

fn cell_idx(gx: u32, gy: u32) -> u32 { return gy * u.grid_width + gx; }
fn dist_idx(cell: u32, i: u32) -> u32 { return cell * 9u + i; }

// D2Q9 equilibrium — reused for the open-boundary Zou-He condition.
fn f_eq(i: u32, rho: f32, ux: f32, uy: f32) -> f32 {
    let ex = f32(EX[i]);
    let ey = f32(EY[i]);
    let eu = ex * ux + ey * uy;
    let uu = ux * ux + uy * uy;
    return W[i] * rho * (1.0 + eu / CS2 + eu * eu / (2.0 * CS2 * CS2) - uu / (2.0 * CS2));
}

// ── Main ─────────────────────────────────────────────────────────────────────

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let gx = gid.x;
    let gy = gid.y;
    if gx >= u.grid_width || gy >= u.grid_height { return; }

    let obs = textureLoad(obstacle, vec2<i32>(i32(gx), i32(gy)), 0);

    // ── Open-boundary (Zou-He outflow) ────────────────────────────────────────
    // Cells with A channel = 1.0 are forced to equilibrium at ambient density
    // (ρ = 1.0) while preserving the local velocity.  This drains excess pressure
    // without artificially injecting or removing momentum.
    if obs.a > 0.5 {
        let cell = cell_idx(gx, gy);
        let d0   = dist_idx(cell, 0u);
        var rho  = 0.0;
        var ux   = 0.0;
        var uy   = 0.0;
        for (var i: u32 = 0u; i < 9u; i++) {
            let fi = dist_dst[d0 + i];
            rho += fi;
            ux  += fi * f32(EX[i]);
            uy  += fi * f32(EY[i]);
        }
        if rho < 1e-6 { rho = 1e-6; }
        ux /= rho;
        uy /= rho;
        for (var i: u32 = 0u; i < 9u; i++) {
            dist_dst[d0 + i] = f_eq(i, 1.0, ux, uy);
        }
        return;
    }

    // ── Pull-gather bounce-back ───────────────────────────────────────────────
    // Only non-fully-solid cells receive bounce-back contributions.
    if obs.r > 0.999 { return; }

    let fluid_cell = cell_idx(gx, gy);

    // For each direction j, check if the neighbour in that direction is solid.
    // If so, pull the bounce-back contribution back to this cell.
    //
    // Key: this thread only ever writes to dist_dst[fluid_cell, ...].  No other
    // thread shares this cell's index, so there are no write-write races.
    //
    // The pulled value is dist_dst[solid_neighbour, j].  By the pull-stream
    // invariant (dst[x, j] = src[x − e_j, j]), after streaming this slot holds
    // the distribution that originated at (solid − e_j) = this fluid cell, i.e.
    // exactly the f_incoming that the solid cell received from us in direction j.
    for (var j: u32 = 1u; j < 9u; j++) { // skip j=0 (rest)
        let nx = i32(gx) + EX[j];
        let ny = i32(gy) + EY[j];
        if !in_bounds(nx, ny) { continue; }

        let obs_nb = textureLoad(obstacle, vec2<i32>(nx, ny), 0);
        let s = obs_nb.r; // solid fill fraction of the neighbour
        if s < 0.001 { continue; } // fluid neighbour — no bounce-back

        let uw = obs_nb.g; // obstacle vel_x
        let vw = obs_nb.b; // obstacle vel_y

        // Distribution that streamed FROM this fluid cell INTO the solid neighbour.
        let solid_cell  = cell_idx(u32(nx), u32(ny));
        let f_incoming  = dist_dst[dist_idx(solid_cell, j)];

        // Moving bounce-back correction: e_j · u_wall
        // (e_j points from this fluid cell toward the solid cell).
        let ej_dot_uw  = f32(EX[j]) * uw + f32(EY[j]) * vw;
        let correction = 2.0 * W[j] * RHO_WALL * ej_dot_uw / CS2;

        // PSM blend: mix existing streamed value with full bounce-back by fill fraction.
        // s=1 → full moving bounce-back; s→0 → no boundary effect.
        let f_bb  = max(f_incoming - correction, 0.0);
        let opp_j = OPPOSITE[j];
        dist_dst[dist_idx(fluid_cell, opp_j)] = mix(
            dist_dst[dist_idx(fluid_cell, opp_j)],
            f_bb,
            s
        );
    }
}
