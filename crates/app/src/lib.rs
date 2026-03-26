mod renderer;

use std::sync::Arc;
use winit::{
    application::ApplicationHandler,
    event::WindowEvent,
    event_loop::{ActiveEventLoop, EventLoop},
    window::{Window, WindowId},
};

use renderer::Renderer;

// On WASM the GPU init is async; we hand a shared slot to spawn_local
// and poll it from the event handler until it's ready.
#[cfg(target_arch = "wasm32")]
use std::{cell::RefCell, rc::Rc};

struct App {
    window: Option<Arc<Window>>,
    renderer: Option<Renderer>,
    #[cfg(target_arch = "wasm32")]
    pending: Rc<RefCell<Option<Renderer>>>,
}

impl App {
    fn new() -> Self {
        Self {
            window: None,
            renderer: None,
            #[cfg(target_arch = "wasm32")]
            pending: Rc::new(RefCell::new(None)),
        }
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

            let web_window = web_sys::window().expect("no window");
            let document = web_window.document().expect("no document");
            let canvas = document
                .get_element_by_id("canvas")
                .expect("no #canvas element")
                .dyn_into::<web_sys::HtmlCanvasElement>()
                .expect("#canvas is not a canvas");

            // Size the canvas to fill the viewport in physical pixels.
            let w = web_window.inner_width().unwrap().as_f64().unwrap() as u32;
            let h = web_window.inner_height().unwrap().as_f64().unwrap() as u32;
            canvas.set_width(w);
            canvas.set_height(h);

            Window::default_attributes()
                .with_title("Webtych")
                .with_canvas(Some(canvas))
                .with_inner_size(winit::dpi::PhysicalSize::new(w, h))
        };

        let window = Arc::new(event_loop.create_window(attrs).unwrap());

        #[cfg(not(target_arch = "wasm32"))]
        {
            self.renderer = Some(pollster::block_on(Renderer::new(window.clone())));
        }

        #[cfg(target_arch = "wasm32")]
        {
            let pending = self.pending.clone();
            let win = window.clone();
            wasm_bindgen_futures::spawn_local(async move {
                let r = Renderer::new(win).await;
                *pending.borrow_mut() = Some(r);
            });
        }

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

        let (Some(window), Some(renderer)) =
            (self.window.as_ref(), self.renderer.as_mut())
        else {
            // Request another frame while waiting for GPU init.
            if let WindowEvent::RedrawRequested = event {
                if let Some(w) = &self.window {
                    w.request_redraw();
                }
            }
            return;
        };

        match event {
            WindowEvent::Resized(size) => {
                renderer.resize(size);
            }
            WindowEvent::RedrawRequested => {
                renderer.update();
                match renderer.render() {
                    Ok(_) => {}
                    Err(wgpu::SurfaceError::Lost | wgpu::SurfaceError::Outdated) => {
                        renderer.resize(window.inner_size());
                    }
                    Err(wgpu::SurfaceError::OutOfMemory) => {
                        log::error!("Out of GPU memory — exiting");
                        event_loop.exit();
                    }
                    Err(e) => log::warn!("Surface error: {e:?}"),
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
