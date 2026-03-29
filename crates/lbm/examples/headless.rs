//! Headless LBM smoke-test.
//!
//! Requests a wgpu device (no window needed), creates a small 64×64 simulation,
//! injects a pressure event at the centre, runs 60 steps, and reads back the
//! macroscopic buffer to print the peak density — confirming the GPU compute
//! passes actually fired.
//!
//! Run with:
//!   cargo run --example headless -p webtych-lbm

use webtych_lbm::{
    config::{EffectProfile, SimConfig},
    types::{EventKind, InjectionEvent},
    Simulation,
};

fn main() {
    env_logger::init();

    pollster::block_on(async {
        // ── device init ──────────────────────────────────────────────────
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends:                 wgpu::Backends::all(),
            flags:                    wgpu::InstanceFlags::default(),
            memory_budget_thresholds: wgpu::MemoryBudgetThresholds::default(),
            backend_options:          wgpu::BackendOptions::default(),
            display:                  None,
        });

        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference:       wgpu::PowerPreference::HighPerformance,
                compatible_surface:     None, // headless — no surface needed
                force_fallback_adapter: false,
            })
            .await
            .expect("no WebGPU adapter found");

        println!("Adapter: {}", adapter.get_info().name);
        println!("Backend: {:?}", adapter.get_info().backend);

        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                label:                 Some("lbm-headless"),
                required_features:     wgpu::Features::empty(),
                required_limits:       wgpu::Limits::default(),
                experimental_features: Default::default(),
                memory_hints:          wgpu::MemoryHints::default(),
                trace:                 Default::default(),
            })
            .await
            .expect("failed to create device");

        // ── simulation setup ─────────────────────────────────────────────
        // Small grid for fast iteration: 64×64 cells, 1 colour channel.
        let world_w = 10.0_f32;
        let world_h = 10.0_f32;

        let config = SimConfig {
            grid_width:      64,
            grid_height:     64,
            tau:             0.7,
            world_width:     world_w,
            world_height:    world_h,
            color_count:     1,
            effect_profiles: vec![EffectProfile {
                inject_density:       4.0,
                inject_color_density: 1.0,
                dissipation:          0.99,
            }],
        };

        let mut sim = Simulation::new(&device, config);

        // ── inject one destroy event at the centre ───────────────────────
        sim.push_event(InjectionEvent {
            x:         world_w / 2.0,
            y:         world_h / 2.0,
            intensity: 1.0,
            color_id:  0,
            kind:      EventKind::Destroy,
        });

        // ── run 60 steps ─────────────────────────────────────────────────
        const STEPS: u32 = 60;
        for step in 0..STEPS {
            sim.step(&device, &queue);
            if step % 10 == 0 {
                println!("Step {step}…");
            }
        }

        // ── read back the macroscopic buffer ─────────────────────────────
        // Copy macroscopic (storage) → a MAP_READ staging buffer.
        let macro_buf  = sim.macroscopic_buffer();
        let buf_size   = macro_buf.size();

        let staging = device.create_buffer(&wgpu::BufferDescriptor {
            label:              Some("staging"),
            size:               buf_size,
            usage:              wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("readback"),
        });
        encoder.copy_buffer_to_buffer(macro_buf, 0, &staging, 0, buf_size);
        queue.submit(std::iter::once(encoder.finish()));

        // Map and read.
        let (tx, rx) = std::sync::mpsc::channel();
        staging
            .slice(..)
            .map_async(wgpu::MapMode::Read, move |r| tx.send(r).unwrap());

        device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
        rx.recv().unwrap().expect("map failed");

        let data: Vec<f32> = {
            let view = staging.slice(..).get_mapped_range();
            bytemuck::cast_slice::<u8, f32>(&view).to_vec()
        };
        staging.unmap();

        // data layout: [rho, ux, uy] per cell in row-major order.
        let n_cells = (64 * 64) as usize;
        let mut peak_rho: f32 = 0.0;
        let mut peak_cell = 0;
        let mut nonambient_count = 0u32;

        for cell in 0..n_cells {
            let rho = data[cell * 3];
            if rho > peak_rho {
                peak_rho = rho;
                peak_cell = cell;
            }
            if (rho - 1.0).abs() > 0.01 {
                nonambient_count += 1;
            }
        }

        let px = peak_cell % 64;
        let py = peak_cell / 64;

        println!();
        println!("── Results after {STEPS} steps ────────────────────────────────");
        println!("  Peak density : {peak_rho:.4} at grid cell ({px}, {py})");
        println!("  Grid cells with ρ ≠ 1 (> ±0.01): {nonambient_count}");

        // Sanity check: the wave should have spread — cells with non-ambient
        // density shouldn't be zero (event fired) and shouldn't be the entire
        // grid (dissipation hasn't erased everything in 60 steps).
        assert!(nonambient_count > 0,   "No cells were affected — event injection didn't fire.");
        assert!(nonambient_count < 64 * 64, "Every cell is non-ambient — something is wrong.");
        assert!(peak_rho > 1.001,       "Peak density hasn't risen — collision/stream broken.");

        println!();
        println!("All assertions passed. LBM simulation is working correctly.");
    });
}
