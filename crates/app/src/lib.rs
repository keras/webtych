mod renderer;

use std::sync::Arc;
use winit::{
    application::ApplicationHandler,
    event::WindowEvent,
    event_loop::{ActiveEventLoop, EventLoop},
    keyboard::{KeyCode, PhysicalKey},
    window::{Window, WindowId},
};

use renderer::Renderer;
use webtych_game::{GameState, InputAction};

// On WASM the GPU init is async; we hand a shared slot to spawn_local
// and poll it from the event handler until it's ready.
#[cfg(target_arch = "wasm32")]
use std::{cell::RefCell, rc::Rc};

struct App {
    window: Option<Arc<Window>>,
    renderer: Option<Renderer>,
    game_state: Option<GameState>,
    input_buffer: Vec<InputAction>,
    #[cfg(not(target_arch = "wasm32"))]
    last_instant: Option<std::time::Instant>,
    #[cfg(target_arch = "wasm32")]
    last_time_ms: Option<f64>,
    #[cfg(target_arch = "wasm32")]
    pending: Rc<RefCell<Option<Renderer>>>,
}

impl App {
    fn new() -> Self {
        Self {
            window: None,
            renderer: None,
            game_state: None,
            input_buffer: Vec::new(),
            #[cfg(not(target_arch = "wasm32"))]
            last_instant: None,
            #[cfg(target_arch = "wasm32")]
            last_time_ms: None,
            #[cfg(target_arch = "wasm32")]
            pending: Rc::new(RefCell::new(None)),
        }
    }

    /// Compute delta time since last frame.
    fn delta_time(&mut self) -> f32 {
        #[cfg(not(target_arch = "wasm32"))]
        {
            let now = std::time::Instant::now();
            let dt = self
                .last_instant
                .map(|prev| now.duration_since(prev).as_secs_f32())
                .unwrap_or(1.0 / 60.0);
            self.last_instant = Some(now);
            dt
        }
        #[cfg(target_arch = "wasm32")]
        {
            let now = web_sys::window()
                .and_then(|w| w.performance())
                .map(|p| p.now())
                .unwrap_or(0.0);
            let dt = self
                .last_time_ms
                .map(|prev| ((now - prev) / 1000.0) as f32)
                .unwrap_or(1.0 / 60.0);
            self.last_time_ms = Some(now);
            dt
        }
    }
}

/// Map winit key codes to game input actions.
fn map_key(key: KeyCode) -> Option<InputAction> {
    match key {
        KeyCode::ArrowLeft | KeyCode::KeyA => Some(InputAction::MoveLeft),
        KeyCode::ArrowRight | KeyCode::KeyD => Some(InputAction::MoveRight),
        KeyCode::ArrowUp | KeyCode::KeyW => Some(InputAction::RotateCW),
        KeyCode::KeyZ => Some(InputAction::RotateCCW),
        KeyCode::ArrowDown | KeyCode::KeyS => Some(InputAction::SoftDrop),
        KeyCode::Space => Some(InputAction::HardDrop),
        _ => None,
    }
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        #[cfg(not(target_arch = "wasm32"))]
        let attrs = Window::default_attributes().with_title("Webtych");

        #[cfg(target_arch = "wasm32")]
        let attrs = {
            use wasm_bindgen::JsCast;
            use winit::platform::web::WindowAttributesExtWebSys;

            let canvas = web_sys::window()
                .expect("no window")
                .document()
                .expect("no document")
                .get_element_by_id("canvas")
                .expect("no #canvas element")
                .dyn_into::<web_sys::HtmlCanvasElement>()
                .expect("#canvas is not a canvas");

            // Let CSS (100vw × 100vh !important) control the display size.
            // winit fires a Resized event that drives the wgpu surface config.
            Window::default_attributes()
                .with_title("Webtych")
                .with_canvas(Some(canvas))
        };

        let window = Arc::new(event_loop.create_window(attrs).unwrap());

        let game_state = GameState::new();
        let palette = game_state.palette.clone();

        #[cfg(not(target_arch = "wasm32"))]
        {
            self.renderer = Some(pollster::block_on(Renderer::new(window.clone(), &palette)));
        }

        #[cfg(target_arch = "wasm32")]
        {
            let pending = self.pending.clone();
            let win = window.clone();
            wasm_bindgen_futures::spawn_local(async move {
                let r = Renderer::new(win, &palette).await;
                *pending.borrow_mut() = Some(r);
            });
        }

        self.game_state = Some(game_state);
        self.window = Some(window);
    }

    fn window_event(
        &mut self,
        event_loop: &ActiveEventLoop,
        _id: WindowId,
        event: WindowEvent,
    ) {
        // Promote async-initialised renderer once it's ready.
        #[cfg(target_arch = "wasm32")]
        if self.renderer.is_none() {
            if let Some(r) = self.pending.borrow_mut().take() {
                self.renderer = Some(r);
            }
        }

        match event {
            WindowEvent::Resized(_)
            | WindowEvent::RedrawRequested
            | WindowEvent::CloseRequested => {}
            WindowEvent::KeyboardInput { event, .. } => {
                if event.state == winit::event::ElementState::Pressed && !event.repeat {
                    if let PhysicalKey::Code(code) = event.physical_key {
                        if let Some(action) = map_key(code) {
                            self.input_buffer.push(action);
                        }
                    }
                }
                return;
            }
            _ => return,
        }

        let Some(window) = self.window.clone() else {
            return;
        };

        // Compute dt and drain inputs before borrowing renderer.
        if matches!(event, WindowEvent::RedrawRequested) {
            let dt = self.delta_time();
            let inputs: Vec<InputAction> = self.input_buffer.drain(..).collect();
            if let Some(game) = self.game_state.as_mut() {
                game.update(dt, &inputs);
            }
        }

        let Some(renderer) = self.renderer.as_mut() else {
            if matches!(event, WindowEvent::RedrawRequested) {
                window.request_redraw();
            }
            return;
        };

        match event {
            WindowEvent::Resized(size) => {
                renderer.resize(size);
            }
            WindowEvent::RedrawRequested => {
                if let Some(game) = self.game_state.as_ref() {
                    let instances = game.block_instances();
                    renderer.update_instances(&instances);
                }

                match renderer.render() {
                    Ok(_) => {}
                    Err(renderer::RenderError::Reconfigure) => {
                        renderer.resize(window.inner_size());
                    }
                }
                window.request_redraw();
            }
            WindowEvent::CloseRequested => event_loop.exit(),
            _ => {}
        }
    }
}

pub fn run() {
    let event_loop = EventLoop::new().unwrap();

    #[cfg(not(target_arch = "wasm32"))]
    {
        let mut app = App::new();
        event_loop.run_app(&mut app).unwrap();
    }

    #[cfg(target_arch = "wasm32")]
    {
        use winit::platform::web::EventLoopExtWebSys;
        event_loop.spawn_app(App::new());
    }
}

// ── WASM entry ────────────────────────────────────────────────────────────────

#[cfg(target_arch = "wasm32")]
use wasm_bindgen::prelude::*;

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen(start)]
pub fn wasm_main() {
    std::panic::set_hook(Box::new(console_error_panic_hook::hook));
    console_log::init_with_level(log::Level::Info).ok();
    run();
}
