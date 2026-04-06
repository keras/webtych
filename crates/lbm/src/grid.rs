//! GPU buffer and texture allocation helpers.
//!
//! [`GpuGrid`] owns every wgpu resource the simulation needs: distribution
//! ping-pong buffers, macroscopic-quantities buffer, obstacle texture, colour-
//! density buffers, event ring-buffer, and the uniform buffer.

use wgpu::util::DeviceExt;

use crate::config::{SimConfig, MAX_COLORS, MAX_EVENTS};
use crate::types::{GpuEvent, LbmUniforms, ObstacleTexel};

/// Side length of the injection stamp texture used by `lbm_inject`.
pub const INJECTION_STAMP_SIZE: u32 = 64;

/// All GPU-side resources for one simulation instance.
pub struct GpuGrid {
    // ── Distribution function buffers (ping-pong) ────────────────────────────
    /// Current-frame distributions (read in streaming, write in collision).
    pub distributions_a: wgpu::Buffer,
    /// Next-frame distributions (ping-pong target / result of streaming).
    pub distributions_b: wgpu::Buffer,
    /// Which ping-pong buffer is currently the "source" (false=A, true=B).
    pub pong: bool,

    // ── Macroscopic quantities ───────────────────────────────────────────────
    /// `[rho, u.x, u.y]` per cell — written by extract pass, read by advect + particles.
    pub macroscopic: wgpu::Buffer,

    // ── Obstacle texture ─────────────────────────────────────────────────────
    /// CPU-uploadable texture: R=mask, GB=obstacle velocity, A=unused.
    pub obstacle_texture: wgpu::Texture,
    pub obstacle_texture_view: wgpu::TextureView,

    /// Injection stamp texture sampled by the inject pass.
    ///
    /// Layout: R=mask, G=velocity_x profile, B=velocity_y profile, A=unused.
    pub injection_texture: wgpu::Texture,
    pub injection_texture_view: wgpu::TextureView,

    // ── Colour density buffer (packed) ───────────────────────────────────────
    /// Single packed buffer: `color_densities[cell * MAX_COLORS + channel]`.
    /// Size: `cell_count * MAX_COLORS * sizeof(f32)` bytes.
    pub color_densities: wgpu::Buffer,

    // ── Event ring buffer ────────────────────────────────────────────────────
    /// Holds up to MAX_EVENTS `GpuEvent` structs, written from CPU each frame.
    pub event_buffer: wgpu::Buffer,

    // ── Uniform buffer ───────────────────────────────────────────────────────
    pub uniform_buffer: wgpu::Buffer,

    // ── Cached config ────────────────────────────────────────────────────────
    pub grid_width: u32,
    pub grid_height: u32,
}

impl GpuGrid {
    /// Allocate all GPU resources for the given configuration.
    pub fn new(device: &wgpu::Device, config: &SimConfig) -> Self {
        let cell_count = (config.grid_width * config.grid_height) as u64;

        // 9 f32 distribution functions per cell (D2Q9).

        // Initialise distributions to equilibrium: f_i = w_i * rho0, rho0=1.
        // D2Q9 weights: w0=4/9, w1-4=1/9, w5-8=1/36
        let weights: [f32; 9] = [
            4.0 / 9.0,
            1.0 / 9.0,
            1.0 / 9.0,
            1.0 / 9.0,
            1.0 / 9.0,
            1.0 / 36.0,
            1.0 / 36.0,
            1.0 / 36.0,
            1.0 / 36.0,
        ];
        let init_data: Vec<f32> = (0..cell_count)
            .flat_map(|_| weights.iter().copied())
            .collect();
        let init_bytes: &[u8] = bytemuck::cast_slice(&init_data);

        let distributions_a = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("lbm::distributions_a"),
            contents: init_bytes,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC,
        });
        let distributions_b = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("lbm::distributions_b"),
            contents: init_bytes,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC,
        });

        // Macroscopic: [rho, ux, uy] per cell, initialised to rho=1, u=(0,0).
        let mut macro_init: Vec<f32> = Vec::with_capacity(cell_count as usize * 3);
        for _ in 0..cell_count {
            macro_init.push(1.0); // rho
            macro_init.push(0.0); // ux
            macro_init.push(0.0); // uy
        }
        let macroscopic = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("lbm::macroscopic"),
            contents: bytemuck::cast_slice(&macro_init),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        });

        // Obstacle texture: Rgba32Float (R=mask, G=vel_x, B=vel_y, A=0).
        let obstacle_texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("lbm::obstacle_texture"),
            size: wgpu::Extent3d {
                width: config.grid_width,
                height: config.grid_height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba32Float,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        });
        let obstacle_texture_view =
            obstacle_texture.create_view(&wgpu::TextureViewDescriptor::default());

        let injection_texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("lbm::injection_texture"),
            size: wgpu::Extent3d {
                width: INJECTION_STAMP_SIZE,
                height: INJECTION_STAMP_SIZE,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba32Float,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        let injection_texture_view =
            injection_texture.create_view(&wgpu::TextureViewDescriptor::default());

        // Packed colour density buffer: [cell * MAX_COLORS + channel].
        // Always MAX_COLORS slots per cell; unused channels simply stay zero.
        let zero_color: Vec<f32> = vec![0.0f32; cell_count as usize * MAX_COLORS];
        let color_densities = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("lbm::color_densities"),
            contents: bytemuck::cast_slice(&zero_color),
            usage: wgpu::BufferUsages::STORAGE,
        });

        // Event ring buffer.
        let event_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("lbm::event_buffer"),
            size: (MAX_EVENTS * std::mem::size_of::<GpuEvent>()) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Uniform buffer — zero-initialised; populated each frame in step().
        let uniform_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("lbm::uniform_buffer"),
            size: std::mem::size_of::<LbmUniforms>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        Self {
            distributions_a,
            distributions_b,
            pong: false,
            macroscopic,
            obstacle_texture,
            obstacle_texture_view,
            injection_texture,
            injection_texture_view,
            color_densities,
            event_buffer,
            uniform_buffer,
            grid_width: config.grid_width,
            grid_height: config.grid_height,
        }
    }

    /// Return the storage buffer currently holding the "current" distributions
    /// (the one the collision/inject passes read from).
    pub fn src_distributions(&self) -> &wgpu::Buffer {
        if self.pong {
            &self.distributions_b
        } else {
            &self.distributions_a
        }
    }

    /// Return the storage buffer to write the post-streaming distributions into.
    pub fn dst_distributions(&self) -> &wgpu::Buffer {
        if self.pong {
            &self.distributions_a
        } else {
            &self.distributions_b
        }
    }

    /// Swap ping-pong buffers.
    pub fn swap(&mut self) {
        self.pong = !self.pong;
    }

    /// Upload a rasterised obstacle texture from CPU.
    ///
    /// `data` must have exactly `grid_width * grid_height` [`ObstacleTexel`] elements.
    pub fn upload_obstacle_texture(&self, queue: &wgpu::Queue, data: &[ObstacleTexel]) {
        assert_eq!(
            data.len(),
            (self.grid_width * self.grid_height) as usize,
            "obstacle texture data length mismatch"
        );
        queue.write_texture(
            self.obstacle_texture.as_image_copy(),
            bytemuck::cast_slice(data),
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(self.grid_width * std::mem::size_of::<ObstacleTexel>() as u32),
                rows_per_image: Some(self.grid_height),
            },
            wgpu::Extent3d {
                width: self.grid_width,
                height: self.grid_height,
                depth_or_array_layers: 1,
            },
        );
    }

    /// Upload the injection stamp texture used by the event inject pass.
    pub fn upload_injection_texture(&self, queue: &wgpu::Queue, data: &[[f32; 4]]) {
        assert_eq!(
            data.len(),
            (INJECTION_STAMP_SIZE * INJECTION_STAMP_SIZE) as usize,
            "injection texture data length mismatch"
        );

        queue.write_texture(
            self.injection_texture.as_image_copy(),
            bytemuck::cast_slice(data),
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(INJECTION_STAMP_SIZE * std::mem::size_of::<[f32; 4]>() as u32),
                rows_per_image: Some(INJECTION_STAMP_SIZE),
            },
            wgpu::Extent3d {
                width: INJECTION_STAMP_SIZE,
                height: INJECTION_STAMP_SIZE,
                depth_or_array_layers: 1,
            },
        );
    }
}

/// Build a default radial injection stamp texture.
///
/// R = density mask (soft circular falloff)
/// G,B = normalized outward velocity profile in [-1, 1]
pub fn build_default_injection_stamp() -> Vec<[f32; 4]> {
    let n = INJECTION_STAMP_SIZE as usize;
    let mut out = vec![[0.0f32; 4]; n * n];

    for y in 0..n {
        for x in 0..n {
            let u = (x as f32 + 0.5) / n as f32;
            let v = (y as f32 + 0.5) / n as f32;
            let dx = 2.0 * (u - 0.5);
            let dy = 2.0 * (v - 0.5);
            let r = (dx * dx + dy * dy).sqrt();

            let mask = if r < 1.0 {
                let t = 1.0 - r;
                t * t
            } else {
                0.0
            };

            let norm = (dx * dx + dy * dy).sqrt().max(1e-6);
            let vx = if mask > 0.0 { dx / norm } else { 0.0 };
            let vy = if mask > 0.0 { dy / norm } else { 0.0 };
            out[y * n + x] = [mask, vx, vy, 0.0];
        }
    }

    out
}
