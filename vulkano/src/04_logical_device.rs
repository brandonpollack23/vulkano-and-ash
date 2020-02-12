use std::sync::Arc;
use vulkano::{
  self,
  device::{Device, DeviceExtensions, Features, Queue},
  instance::{
    debug::{DebugCallback, MessageSeverity, MessageType},
    layers_list, ApplicationInfo, Instance, InstanceExtensions, PhysicalDevice, Version,
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

struct QueueFamilyIndices {
  graphics_queue_family_index: Option<usize>,
}
impl QueueFamilyIndices {
  fn new() -> QueueFamilyIndices {
    QueueFamilyIndices {
      graphics_queue_family_index: None,
    }
  }

  fn is_complete(&self) -> bool {
    self.graphics_queue_family_index.is_some()
  }
}

/// Struct representing the application to display the triangle.
pub struct HelloTriangleApplication {
  window: HelloTriangleWindow,
  #[allow(dead_code)]
  instance: Arc<Instance>,
  #[allow(dead_code)]
  debug_callback: Option<DebugCallback>,
  #[allow(dead_code)]
  physical_device_index: usize,
  #[allow(dead_code)]
  logical_device: Arc<Device>,

  // This field, along with all the queues, might have more distinct names in a more complex
  // system, to signify what each is really for
  #[allow(dead_code)]
  graphics_queue: Arc<Queue>,
}
impl HelloTriangleApplication {
  pub fn initialize() -> Self {
    let window = HelloTriangleWindow::init_window();
    let instance = Self::create_vulkan_instance();
    let debug_callback = Self::setup_debug_callback_if_enabled(&instance);
    let physical_device_index = Self::pick_physical_device(&instance);
    let (logical_device, graphics_queue) =
      Self::create_logical_device_and_queues(&instance, physical_device_index);

    Self {
      window,
      instance,
      debug_callback,
      physical_device_index,
      logical_device,
      graphics_queue,
    }
  }

  /// Initializes vulkan instance.
  fn create_vulkan_instance() -> Arc<Instance> {
    if ENABLE_VALIDATION_LAYERS && !Self::check_and_print_validation_layer_support() {
      panic!("Validation layers requested, but not available!");
    }

    Self::print_supported_extensions();
    let extensions = if ENABLE_VALIDATION_LAYERS {
      Self::get_required_extensions()
    } else {
      vulkano_win::required_extensions()
    };
    println!("Required Extensions:\n\t{:?}\n", extensions);

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
    Instance::new(
      Some(&app_info),
      &extensions,
      VALIDATION_LAYERS.iter().cloned(),
    )
    .expect("Failed to create Vulkan instance")
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
    println!("Supported Extensions:\n\t{:?}", supported_extensions);
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
    //
    // Unfortunately, vulkano doesn't support setting a value for pNext in
    // vkInstanceCreateInfo so we can't get debug callbacks for create/destroy
    // instance, but vulkano is supposed to validate and cover those errors anyway,
    // right?
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

  /// For now we're going to just select the first enumerated physical device.
  /// A real game engine could:
  /// * Allow users to select via a dropdown and reinitialize from physical
  ///   devices onward
  /// * Check for most wanted features and pick the best possible GPU with a
  ///   priority or score.
  ///
  /// For now we'll select the first device that supports all the queue families
  /// we need.
  fn pick_physical_device(instance: &Arc<Instance>) -> usize {
    PhysicalDevice::enumerate(instance)
      .position(|physical_device| Self::is_device_suitable(&physical_device))
      .expect("Failed to find a suitable Physical Device!")
  }

  fn is_device_suitable(physical_device: &PhysicalDevice) -> bool {
    let feature_indices = Self::find_queue_families(physical_device);
    feature_indices.is_complete()
  }

  /// Returns the [QueueFamilyIndices](struct.QueueFamilyIndices.html) supported
  /// by this physical device.
  /// TODO QueueFamilyIndices member
  fn find_queue_families(physical_device: &PhysicalDevice) -> QueueFamilyIndices {
    let mut indices_of_queue_families_that_support_feature = QueueFamilyIndices::new();

    for (i, queue_family) in physical_device.queue_families().enumerate() {
      if queue_family.supports_graphics() {
        indices_of_queue_families_that_support_feature.graphics_queue_family_index = Some(i);
      }

      if indices_of_queue_families_that_support_feature.is_complete() {
        break;
      }
    }

    indices_of_queue_families_that_support_feature
  }

  /// Creates the logical device from the instance and physical device index.
  /// Returns the logical device and the graphics queue.
  fn create_logical_device_and_queues(
    instance: &Arc<Instance>, physical_device_index: usize,
  ) -> (Arc<Device>, Arc<Queue>) {
    let physical_device = PhysicalDevice::from_index(instance, physical_device_index)
      .expect("Unable to get physical device at the given index");

    let queue_indices = Self::find_queue_families(&physical_device);

    let graphics_queue_family_index = queue_indices
      .graphics_queue_family_index
      .expect("No Graphics Family Queue Index!");
    let graphics_queue_family = physical_device
      .queue_families()
      .nth(graphics_queue_family_index)
      .unwrap();
    let graphics_queue_priority = 1.0f32;

    // Right now no specific features are needed, but later we'll likely need some
    // features for pipelines. When we actually start drawing, we'll also need
    // extensions like VK_KHR_swapchain.
    let (device, mut queues) = Device::new(
      physical_device,
      &Features::none(),
      &DeviceExtensions::none(),
      [(graphics_queue_family, graphics_queue_priority)]
        .iter()
        .cloned(),
    )
    .expect("Unable to create logical device!");

    (device, queues.next().unwrap())
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
