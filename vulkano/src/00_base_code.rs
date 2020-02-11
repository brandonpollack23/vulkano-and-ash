use vulkano;
use winit::{
  dpi::LogicalSize,
  event::{ElementState, Event, VirtualKeyCode, WindowEvent},
  event_loop::{ControlFlow, EventLoop, EventLoopWindowTarget},
  window::{Window, WindowBuilder},
};

const WIDTH: u32 = 800;
const HEIGHT: u32 = 600;

struct HelloTriangleApplication {
  window: HelloTriangleWindow,
}

struct HelloTriangleWindow {
  event_loop: EventLoop<()>,
  winit_window: Window,
}

impl HelloTriangleWindow {
  pub fn init_window() -> HelloTriangleWindow {
    let event_loop = EventLoop::new();
    let winit_window = WindowBuilder::new()
      .with_title("Vulkano Vulkan Tutorial")
      .with_inner_size(LogicalSize::new(f64::from(WIDTH), f64::from(HEIGHT)))
      .build(&event_loop)
      .expect("Failed to create winit window");

    HelloTriangleWindow {
      event_loop,
      winit_window,
    }
  }

  fn run<F>(self, event_handler: F)
  where
    F: 'static + FnMut(Event<()>, &EventLoopWindowTarget<()>, &mut ControlFlow),
  {
    self.event_loop.run(event_handler);
  }
}

impl HelloTriangleApplication {
  pub fn initialize() -> Self {
    Self {
      window: HelloTriangleWindow::init_window(),
    }
  }

  fn main_loop(self) {
    self.window.run(move |window_event, _, control_flow| {
      match window_event {
        // When the window system requests a close, signal to winit that we'd like to close the window.
        Event::WindowEvent {
          event: WindowEvent::CloseRequested,
          ..
        } => *control_flow = ControlFlow::Exit,

        // When the keyboard input is a press on the escape key, exit and print the line.
        Event::WindowEvent {
          event: WindowEvent::KeyboardInput { input, .. },
          ..
        } => {
          if let (Some(VirtualKeyCode::Escape), ElementState::Pressed) =
            (input.virtual_keycode, input.state)
          {
            dbg!();
            *control_flow = ControlFlow::Exit
          }
        }
        _ => (),
      }
    });
  }
}

fn main() {
  let mut app = HelloTriangleApplication::initialize();
  app.main_loop();
}
