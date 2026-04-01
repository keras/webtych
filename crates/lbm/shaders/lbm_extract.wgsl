// lbm_extract.wgsl
//
// Pass 5: Extract macroscopic quantities.
//
// For each non-obstacle cell, sum the post-stream/bounce-back distributions
// to obtain the macroscopic density ρ and velocity u:
//
//   ρ   = Σ f_i
//   u_x = (Σ f_i × e_ix) / ρ
//   u_y = (Σ f_i × e_iy) / ρ
//
// Results are written to the `macroscopic` storage buffer: [ρ, u_x, u_y] per cell.
// The velocity field is read by the advect pass and exposed to CPU for particles.

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
@group(0) @binding(2) var<storage, read_write> dist_dst: array<f32>; // post-stream read
@group(0) @binding(3) var<storage, read_write> macroscopic: array<f32>; // write
@group(0) @binding(4) var obstacle: texture_2d<f32>;
@group(0) @binding(5) var<storage, read>       events:   array<f32>; // unused
@group(0) @binding(6) var<storage, read_write> color_densities: array<f32>; // unused

// ── D2Q9 constants ────────────────────────────────────────────────────────────

const EX = array<f32, 9>(0.0,  1.0, 0.0, -1.0,  0.0,  1.0, -1.0, -1.0,  1.0);
const EY = array<f32, 9>(0.0,  0.0, 1.0,  0.0, -1.0,  1.0,  1.0, -1.0, -1.0);

// ── Helpers ───────────────────────────────────────────────────────────────────

fn cell_idx(gx: u32, gy: u32) -> u32 { return gy * u.grid_width + gx; }
fn dist_idx(cell: u32, i: u32) -> u32 { return cell * 9u + i; }
fn macro_idx(cell: u32)        -> u32 { return cell * 3u; }

// ── Main ─────────────────────────────────────────────────────────────────────

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let gx = gid.x;
    let gy = gid.y;
    if gx >= u.grid_width || gy >= u.grid_height { return; }

    let cell = cell_idx(gx, gy);
    let m0   = macro_idx(cell);

    // Fully solid cells: report obstacle velocity, skip distribution extraction.
    // Partial PSM cells fall through and compute their actual macro quantities.
    let obs = textureLoad(obstacle, vec2<i32>(i32(gx), i32(gy)), 0);
    if obs.r > 0.999 {
        macroscopic[m0]      = 1.0;
        macroscopic[m0 + 1u] = obs.g; // expose obstacle velocity (useful for particles)
        macroscopic[m0 + 2u] = obs.b;
        return;
    }

    let d0 = dist_idx(cell, 0u);
    var rho = 0.0;
    var ux  = 0.0;
    var uy  = 0.0;
    for (var i: u32 = 0u; i < 9u; i++) {
        let fi = dist_dst[d0 + i];
        rho += fi;
        ux  += fi * EX[i];
        uy  += fi * EY[i];
    }
    if rho < 1e-6 { rho = 1e-6; }
    ux /= rho;
    uy /= rho;

    macroscopic[m0]      = rho;
    macroscopic[m0 + 1u] = ux;
    macroscopic[m0 + 2u] = uy;
}
