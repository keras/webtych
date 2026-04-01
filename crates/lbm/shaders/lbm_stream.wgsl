// lbm_stream.wgsl
//
// Pass 3: Streaming (propagation).
//
// Pull scheme: each thread reads the post-collision distribution from its
// *incoming* neighbour direction and writes to the dst buffer at its own cell.
//
//   dst[cell, i] = src[cell − e_i, i]
//
// Boundary handling: cells that would read outside the grid clamp to the edge
// (zero-flux / no-slip for now; the boundary pass applies the proper
// bounce-back for obstacle cells).

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
@group(0) @binding(1) var<storage, read_write> dist_src: array<f32>;
@group(0) @binding(2) var<storage, read_write> dist_dst: array<f32>;
@group(0) @binding(3) var<storage, read_write> macroscopic: array<f32>; // unused
@group(0) @binding(4) var obstacle: texture_2d<f32>;                    // unused
@group(0) @binding(5) var<storage, read>       events:   array<f32>;    // unused
@group(0) @binding(6) var<storage, read_write> color_densities: array<f32>; // unused

// ── D2Q9 lattice velocity vectors ────────────────────────────────────────────

const EX = array<i32, 9>( 0,  1,  0, -1,  0,  1, -1, -1,  1);
const EY = array<i32, 9>( 0,  0,  1,  0, -1,  1,  1, -1, -1);

// ── Helpers ──────────────────────────────────────────────────────────────────

fn clamp_coord(c: i32, max_val: u32) -> i32 {
    return clamp(c, 0, i32(max_val) - 1);
}

fn cell_at(gx: i32, gy: i32) -> u32 {
    let cx = clamp_coord(gx, u.grid_width);
    let cy = clamp_coord(gy, u.grid_height);
    return u32(cy) * u.grid_width + u32(cx);
}

fn dist_idx(cell: u32, i: u32) -> u32 { return cell * 9u + i; }

// ── Main ─────────────────────────────────────────────────────────────────────

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let gx = gid.x;
    let gy = gid.y;
    if gx >= u.grid_width || gy >= u.grid_height { return; }

    let dst_cell = gy * u.grid_width + gx;

    for (var i: u32 = 0u; i < 9u; i++) {
        // Pull from the cell that was "upstream" in direction i.
        let src_gx = i32(gx) - EX[i];
        let src_gy = i32(gy) - EY[i];
        let src_cell = cell_at(src_gx, src_gy);
        dist_dst[dist_idx(dst_cell, i)] = dist_src[dist_idx(src_cell, i)];
    }
}
