use std::sync::Arc;
use vulkano::{
  self,
  instance::{
    debug::{DebugCallback, MessageSeverity, MessageType},
    layers_list, ApplicationInfo, Instance, InstanceExtensions, Version,
  },
};
use winit::{
  dpi::LogicalSize,
  event::{ElementState, Event, VirtualKeyCode, WindowEvent},
  event_loop::{ControlFlow, EventLoop, EventLoopWindowTarget},
  window::{Window, WindowBuilder},
};

const VALIDATION_LAYERS: &[&str] = &["VK_LAYER_LUNARG_standard_validation"];

const WIDTH: u32 = 800;
const HEIGHT: u32 = 600;

#[cfg(all(debug_assertions))]
const ENABLE_VALIDATION_LAYERS: bool = true;
#[cfg(not(debug_assertions))]
const ENABLE_VALIDATION_LAYERS: bool = false;

/// Struct representing the application to display the triangle.
pub struct HelloTriangleApplication {
  window: HelloTriangleWindow,
  #[allow(dead_code)]
  instance: Arc<Instance>,
  #[allow(dead_code)]
  debug_callback: Option<DebugCallback>,
}

impl HelloTriangleApplication {
  pub fn initialize() -> Self {
    let window = HelloTriangleWindow::init_window();
    let instance = Self::create_vulkan_instance();
    let debug_callback = Self::setup_debug_callback_if_enabled(&instance);

    Self {
      window,
      instance,
      debug_callback,
    }
  }

  /// Initializes vulkan instance.
  fn create_vulkan_instance() -> Arc<Instance> {
    if ENABLE_VALIDATION_LAYERS && !Self::check_and_print_validation_layer_support() {
      panic!("Validation layers requested, but not available!");
    }

    Self::print_supported_extensions();

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

    // In vulkano we use "new" static factory methods to construct vkInstance and
    // other vulkan objects instead of passing all the params in a create_info
    // struct.
    if ENABLE_VALIDATION_LAYERS {
      Instance::new(
        Some(&app_info),
        &Self::get_required_extensions(),
        VALIDATION_LAYERS.iter().cloned(),
      )
      .expect("Failed to create Vulkan instance")
    } else {
      Instance::new(Some(&app_info), &vulkano_win::required_extensions(), None)
        .expect("Failed to create Vulkan instance")
    }
  }

  fn check_and_print_validation_layer_support() -> bool {
    let layers: Vec<_> = layers_list()
      .expect("Could not get available layers")
      .map(|l| l.name().to_owned())
      .collect();

    println!(
      "Supported Validation Layers: \n\t{:?}\nRequested Validation Layers \n\t{:?}\n",
      layers, VALIDATION_LAYERS
    );

    // Ensure all the Validation layers we require
    VALIDATION_LAYERS
      .iter()
      .all(|layer_name| layers.contains(&layer_name.to_string()))
  }

  fn get_required_extensions() -> InstanceExtensions {
    let mut extensions = vulkano_win::required_extensions();

    if ENABLE_VALIDATION_LAYERS {
      // No need to check for the existence of this extension because the validation
      // layers being present already confirms it.
      extensions.ext_debug_utils = true;
    }

    extensions
  }

  fn print_supported_extensions() {
    let supported_extensions =
      InstanceExtensions::supported_by_core().expect("failed to retrieve supported extensions");
    println!("Supported Extensions:\n\t{:?}\n", supported_extensions);
  }

  fn setup_debug_callback_if_enabled(instance: &Arc<Instance>) -> Option<DebugCallback> {
    if !ENABLE_VALIDATION_LAYERS {
      return None;
    }

    let msg_severity = MessageSeverity {
      error: true,
      warning: true,
      information: false,
      verbose: true,
    };

    let msg_types = MessageType {
      general: true,
      performance: true,
      validation: true,
    };

    // The extension function (vkCreateDebugUtilsMessenger) needs to be loaded
    // (since it's a part of the debug_utils extension). This would be done in
    // c-like languages by calling vkGetInstanceProdAccr and passing a string with
    // the name of the extension function (see [the vulkan tutorial](https://vulkan-tutorial.com/Drawing_a_triangle/Setup/Validation_layers)).
    //  Instead, vulkano (and via that vk-sys) already handles that and we just need
    // to call the convenient creation function, whose destructor will also call the
    // corresponding destroy function.
    DebugCallback::new(instance, msg_severity, msg_types, |msg| {
      let message_prefix = if msg.severity.error {
        "VALIDATION ERROR:"
      } else if msg.severity.warning {
        "VALIDATION WARNING:"
      } else if msg.severity.information {
        "VALIDATION WARNING:"
      } else if msg.severity.verbose {
        "VALIDATION VERBOSE MESSAGE:"
      } else {
        ""
      };

      println!(
        "{} validation layer {}; {}",
        message_prefix, msg.layer_prefix, msg.description
      );
    })
    .ok()
  }

  /// Takes full control of the executing thread and runs the event loop for it.
  fn main_loop(self) {
    self.window.run(move |window_event, _, control_flow| {
      match window_event {
        // When the window system requests a close, signal to winit that we'd like to close the
        // window.
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

fn main() {
  let app = HelloTriangleApplication::initialize();
  app.main_loop();
}
