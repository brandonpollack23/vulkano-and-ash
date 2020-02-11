use std::sync::Arc;
use vulkano::{
  self,
  instance::{ApplicationInfo, Instance, InstanceExtensions, Version},
};
use winit::{
  dpi::LogicalSize,
  event::{ElementState, Event, VirtualKeyCode, WindowEvent},
  event_loop::{ControlFlow, EventLoop, EventLoopWindowTarget},
  window::{Window, WindowBuilder},
};

const WIDTH: u32 = 800;
const HEIGHT: u32 = 600;

/// Struct representing the window to draw the triangle in.
struct HelloTriangleWindow {
  event_loop: EventLoop<()>,
  // allow dead_code because the window is kept around for its destructor to kill the window.
  #[allow(dead_code)]
  winit_window: Window,
}

impl HelloTriangleWindow {
  /// Create a window for the application.
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

/// Struct representing the application to display the triangle.
pub struct HelloTriangleApplication {
  window: HelloTriangleWindow,
  instance: Arc<Instance>,
}

impl HelloTriangleApplication {
  pub fn initialize() -> Self {
    Self {
      window: HelloTriangleWindow::init_window(),
      instance: Self::init_vulkan_instance(),
    }
  }

  /// Initializes vulkan instance.
  fn init_vulkan_instance() -> Arc<Instance> {
    let supported_extensions =
      InstanceExtensions::supported_by_core().expect("failed to retrieve supported extensions");
    println!("Supported extensions: {:?}", supported_extensions);

    let app_info = ApplicationInfo {
      application_name: Some("Hello Triangle".into()),
      application_version: Some(Version {
        major: 0,
        minor: 1,
        patch: 0,
      }),
      engine_name: Some("No Engine".into()),
      engine_version: None,
    };

    // In vulkano we use "new" static factory methods to construct vkInstance and other vulkan objects instead of passing all the params in a create_info struct.
    Instance::new(Some(&app_info), &vulkano_win::required_extensions(), None)
      .expect("Failed to create Vulkan instance")
  }

  /// Takes full control of the executing thread and runs the event loop for it.
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
  let app = HelloTriangleApplication::initialize();
  app.main_loop();
}
