use std::{collections::HashSet, iter::FromIterator, sync::Arc};
use vulkano::{
  self,
  device::{Device, DeviceExtensions, Features, Queue},
  format::Format,
  framebuffer::{RenderPassAbstract, Subpass},
  image::{ImageUsage, SwapchainImage},
  instance::{
    debug::{DebugCallback, MessageSeverity, MessageType},
    layers_list, ApplicationInfo, Instance, InstanceExtensions, PhysicalDevice, Version,
  },
  pipeline::{
    vertex::BufferlessDefinition, viewport::Viewport, GraphicsPipeline, GraphicsPipelineAbstract,
  },
  single_pass_renderpass,
  swapchain::{
    Capabilities, ColorSpace, CompositeAlpha, FullscreenExclusive, PresentMode,
    SupportedPresentModes, Surface, Swapchain,
  },
  sync::SharingMode,
};
use vulkano_win::VkSurfaceBuild;
use winit::{
  dpi::LogicalSize,
  event::{ElementState, Event, VirtualKeyCode, WindowEvent},
  event_loop::{ControlFlow, EventLoop, EventLoopWindowTarget},
  window::{Window, WindowBuilder},
};

const VALIDATION_LAYERS: &[&str] = &["VK_LAYER_LUNARG_standard_validation"];

const WIDTH: u32 = 800;
const HEIGHT: u32 = 600;

fn required_device_extensions() -> DeviceExtensions {
  DeviceExtensions {
    khr_swapchain: true,
    ..DeviceExtensions::none()
  }
}

#[cfg(all(debug_assertions))]
const ENABLE_VALIDATION_LAYERS: bool = true;
#[cfg(not(debug_assertions))]
const ENABLE_VALIDATION_LAYERS: bool = false;

struct QueueFamilyIndices {
  graphics_queue_family_index: Option<usize>,
  present_queue_family_index: Option<usize>,
}
impl QueueFamilyIndices {
  fn new() -> QueueFamilyIndices {
    QueueFamilyIndices {
      graphics_queue_family_index: None,
      present_queue_family_index: None,
    }
  }

  fn is_complete(&self) -> bool {
    self.graphics_queue_family_index.is_some() && self.present_queue_family_index.is_some()
  }
}

/// Struct representing the window to draw the triangle in.
struct HelloTriangleWindow {
  event_loop: EventLoop<()>,
  winit_window_surface: Arc<Surface<Window>>,
}
impl HelloTriangleWindow {
  /// Create a window for the application.
  pub fn create_vksurface_and_window(instance: &Arc<Instance>) -> HelloTriangleWindow {
    let event_loop = EventLoop::new();
    let winit_window_surface = WindowBuilder::new()
      .with_title("Vulkano Vulkan Tutorial")
      .with_inner_size(LogicalSize::new(f64::from(WIDTH), f64::from(HEIGHT)))
      .build_vk_surface(&event_loop, instance.clone())
      .expect("Failed to create winit window");

    HelloTriangleWindow {
      event_loop,
      winit_window_surface,
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
#[allow(dead_code)]
pub struct HelloTriangleApplication {
  instance: Arc<Instance>,
  debug_callback: Option<DebugCallback>,
  window_surface: HelloTriangleWindow,
  physical_device_index: usize,
  logical_device: Arc<Device>,

  // These fields might have more distinct names in a more complex system, to signify what each is
  // really for.
  graphics_queue: Arc<Queue>,
  presentation_queue: Arc<Queue>,

  swap_chain: Arc<Swapchain<Window>>,
  // vkImageViews are handled by Vulkano and already set up within it's SwapchainImage type:
  // swapchain/swapchain.rs:603 -> image/swapchain.rs:59.  Vulkano configures it as a 2d image with
  // no swizzling, matches the format, VK_IMAGE_ASPECT_COLOR_BIT (color target) no mipmapping and 1
  // layer.
  swap_chain_images: Vec<Arc<SwapchainImage<Window>>>,

  render_pass: Arc<dyn RenderPassAbstract + Send + Sync>,
  graphics_pipeline: Arc<dyn GraphicsPipelineAbstract + Send + Sync>,
}
impl HelloTriangleApplication {
  fn initialize() -> Self {
    let instance = Self::create_vulkan_instance();
    let debug_callback = Self::setup_debug_callback_if_enabled(&instance);
    let window_surface = HelloTriangleWindow::create_vksurface_and_window(&instance);
    let physical_device_index =
      Self::pick_physical_device(&instance, &window_surface.winit_window_surface);
    let (logical_device, graphics_queue, presentation_queue) =
      Self::create_logical_device_and_queues(
        &instance,
        &window_surface.winit_window_surface,
        physical_device_index,
      );
    let (swap_chain, swap_chain_images) = Self::create_swap_chain(
      &instance,
      &window_surface.winit_window_surface,
      physical_device_index,
      &logical_device,
      &graphics_queue,
      &presentation_queue,
    );

    let render_pass = Self::create_render_pass(&logical_device, swap_chain.format());

    // In a real implementation, we may have more than one pipeline for different
    // passes or processes, but in vulkan-tutorial only one.
    let graphics_pipeline =
      Self::create_graphics_pipeline(&logical_device, swap_chain.dimensions(), &render_pass);

    Self {
      instance,
      debug_callback,
      window_surface,
      physical_device_index,
      logical_device,
      graphics_queue,
      presentation_queue,
      swap_chain,
      swap_chain_images,
      render_pass,
      graphics_pipeline,
    }
  }

  /// Initializes vulkan instance.
  fn create_vulkan_instance() -> Arc<Instance> {
    if ENABLE_VALIDATION_LAYERS && !Self::check_and_print_validation_layer_support() {
      panic!("Validation layers requested, but not available!");
    }

    Self::print_supported_instance_extensions();
    let extensions = if ENABLE_VALIDATION_LAYERS {
      Self::get_required_instance_extensions()
    } else {
      vulkano_win::required_extensions()
    };
    println!("Required Instance Extensions:\n\t{:?}\n", extensions);

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
  /// For now we'll select the first device that supports all the queue
  /// families, extensions, and surface capabilities we need.
  fn pick_physical_device(instance: &Arc<Instance>, surface: &Arc<Surface<Window>>) -> usize {
    PhysicalDevice::enumerate(instance)
      .position(|physical_device| Self::is_device_suitable(surface, &physical_device))
      .expect("Failed to find a suitable Physical Device!")
  }

  /// Creates the logical device from the instance and physical device index.
  /// Returns a tuple of (Device, (graphics) Queue, (presentation) Queue).
  fn create_logical_device_and_queues(
    instance: &Arc<Instance>, surface: &Arc<Surface<Window>>, physical_device_index: usize,
  ) -> (Arc<Device>, Arc<Queue>, Arc<Queue>) {
    let physical_device = PhysicalDevice::from_index(instance, physical_device_index)
      .expect("Unable to get physical device at the given index");

    let queue_indices = Self::find_queue_families(surface, &physical_device);

    // Unique queue family indices.
    let queue_family_indices: HashSet<usize> = HashSet::from_iter(
      [
        queue_indices
          .graphics_queue_family_index
          .expect("No graphics queue family index"),
        queue_indices
          .present_queue_family_index
          .expect("No present queue family index"),
      ]
      .iter()
      .map(|i| *i),
    );
    // All queues have the same priority.
    let queue_priority = 1.0f32;
    let queue_families = queue_family_indices.iter().map(|&index| {
      (
        physical_device.queue_families().nth(index).unwrap(),
        queue_priority,
      )
    });

    // Right now no specific features are needed, but later we'll likely need some
    // features for pipelines. When we actually start drawing, we'll also need
    // extensions like VK_KHR_swapchain.
    let (device, mut queues) = Device::new(
      physical_device,
      &Features::none(),
      &required_device_extensions(),
      queue_families,
    )
    .expect("Unable to create logical device!");

    let graphics_queue = queues.next().unwrap();
    let present_queue = queues.next().unwrap_or(graphics_queue.clone());

    (device, graphics_queue, present_queue)
  }

  /// Create the swap chain and it's images for the surface.
  fn create_swap_chain(
    instance: &Arc<Instance>, surface: &Arc<Surface<Window>>, physical_device_index: usize,
    logical_device: &Arc<Device>, graphics_queue: &Arc<Queue>, present_queue: &Arc<Queue>,
  ) -> (Arc<Swapchain<Window>>, Vec<Arc<SwapchainImage<Window>>>) {
    let physical_device = PhysicalDevice::from_index(instance, physical_device_index).unwrap();
    let capabilities = surface
      .capabilities(physical_device)
      .expect("Failed to get surface capabilities");

    let surface_format = Self::choose_swap_surface_format(&capabilities.supported_formats);
    let present_mode = Self::choose_swap_present_mode(capabilities.present_modes);
    let extent = Self::choose_swap_extent(&capabilities);

    // For now use the minimum number of images required for the surface to function
    // plus one (so we don't have to wait) but only if it is less than the max.
    let image_count = capabilities
      .max_image_count
      .map(|max_count| max_count.min(capabilities.min_image_count + 1))
      .unwrap_or(capabilities.min_image_count + 1);

    // We're rendering directly to images so the color attachment is used.
    let image_usage = ImageUsage {
      color_attachment: true,
      ..ImageUsage::none()
    };

    // Next we need to decide how queues will be shared across families (if they
    // are separate families). Exclusive offers the best performance, but if we
    // have separate families for present and graphics queues that are going to be
    // used by the swapchain, we need to use Concurrent mode.
    let queue_family_indices = Self::find_queue_families(surface, &physical_device);
    let sharing_mode = if queue_family_indices.present_queue_family_index.unwrap()
      == queue_family_indices.graphics_queue_family_index.unwrap()
    {
      SharingMode::Exclusive
    } else {
      // SharingMode implements From (and derives Into) for &[&Arc<Queue>], so we can
      // just use this to derive VK_SHARING_MODE_CONCURRENT as well as the
      // queueFamilyIndexCount/Indices
      vec![present_queue, graphics_queue].as_slice().into()
    };

    println!(
      "Creating Swapchain:\n\timage_count: {}\n\textent: {:?}\n\tpresent_mode \
       {:?}\n\tsurface_format: {:?}",
      image_count, extent, present_mode, surface_format
    );

    Swapchain::new(
      logical_device.clone(),
      surface.clone(),
      image_count,
      surface_format.0, // Pixel Data Format
      extent,
      1,           // 1 layer, more would be for images with multiple layers like stereoscopic 3d.
      image_usage, // Using the images for drawing to.
      sharing_mode, /* Whether or not we are sharing queue families (concurrent) or not
                    * (exclusive). */
      capabilities.current_transform, /* A swapchain can apply an overall transform to an image,
                                       * like rotation or flip.  No need to do that so just use
                                       * the current transform of the capabilities. */
      CompositeAlpha::Opaque, // Dont blend with other windows in the window system.
      present_mode,           /* What type of present mode we're doing (immediate, fifo, fifo
                               * messagebox etc). */
      FullscreenExclusive::Default, // TODO what does this do?
      true,                         /* clipped, no need to draw pixels that are obscured (by
                                     * another window or off the
                                     * screen) */
      surface_format.1, // ColorSpace
    )
    .expect("Unable to create swapchain!")
  }

  /// Create the graphics pipeline by compiling all the shaders (vertex,
  /// fragment, geometry?), this at compile time in this example, but
  /// [can be done at runtime](https://github.com/vulkano-rs/vulkano/blob/master/examples/src/bin/runtime-shader/main.rs),
  /// and setting up all the fixed stages (IA, R, etc)
  fn create_graphics_pipeline(
    logical_device: &Arc<Device>, swap_chain_extent: [u32; 2],
    render_pass: &Arc<dyn RenderPassAbstract + Send + Sync>,
  ) -> Arc<dyn GraphicsPipelineAbstract + Send + Sync> {
    // I will be compiling the shaders at compile time in rust using macros
    // provided by Vulkano. As in vulkan-tutorial, this can be done at runtime
    // (see vs and fs initialization [here](https://github.com/vulkano-rs/vulkano/blob/master/examples/src/bin/runtime-shader/main.rs).

    // The vulkano_shaders::shader! macro generates a bunch of structs, etc with
    // common names so we need to wrap them in their own namespace to reference
    // here.  These structs contain the compiled spirv shader code we can load.
    //
    // To use it, we just call the NAMESPACE_NAME::Shader::load(logical_device)
    //
    // Once pipeline creation is finished, we can destroy the Shader (and contained
    // vulkano::ShaderModule/vk::ShaderModule structs) since the spirv will now be
    // copied into the driver.
    mod vertex_shader {
      vulkano_shaders::shader! {
        ty: "vertex",
        path: "../shaders/09_shader_base.vert"
      }
    }

    mod fragment_shader {
      vulkano_shaders::shader! {
        ty: "fragment",
        path: "../shaders/09_shader_base.frag"
      }
    }

    let vert_shader_module = vertex_shader::Shader::load(logical_device.clone())
      .expect("Failed to create vertex shader module");

    let frag_shader_module = fragment_shader::Shader::load(logical_device.clone())
      .expect("Failed to create fragment shader module");

    // In vulkan-tutorial (and in c++ in general) you need to assign shaders to
    // a specific stage via VkPipelineShadersStageCreateInfo, but vulkano
    // handles that for us later on in the
    // [pipeline_builder](https://github.com/vulkano-rs/vulkano/blob/e09588bebfe328e6f984bd885ae9311eaa909d41/vulkano/src/pipeline/graphics_pipeline/builder.rs#L436)

    // As vulkan-tutorial says, we need to be explicit about everything, but rather
    // than make a huge create info for each type, Vulkano provides some sane
    // defaults it seems.
    //
    // Input assembly is set to a default of triangle topology with no restart by
    // Vulkano.  Primitive restart is for _STRIP type primitives (to "restart" the
    // strip with a value of 0xFFFFFFFF, ie have multiple strips in one buffer).
    // PipelineLayout is done when the builder is complete (calling
    // with_pipeline_layout and passing it in separate.
    Arc::new(
      GraphicsPipeline::start()
          // Begin VertexInputCreateInfo fields.
          .vertex_input(BufferlessDefinition {}) /* I'm not using any buffers for vertex input, so 
                                                    no need to specify vertex attribute descriptions--type of attributes passed in vertex 
                                                    shader, which binding to load them from and at which offset--or vertex binding 
                                                    descriptions--spacing between data and whether data is per vertex or per instance 
                                                    (vkPipelineVertexInputStateCreateInfo fields).*/
          // Triangle_list and primitive restart may already be defined by the zeroing out
          // of the rest of the vertex input create info, but its better to be explicit in
          // case VK_FALSE changes somehow.
          .triangle_list()
          .primitive_restart(false)
          // End VertexInput fields

          // Begin PipelineViewportStateCreateInfo fields.
          .viewports(vec![Viewport {
            origin: [0.0, 0.0],
            dimensions: [swap_chain_extent[0] as f32, swap_chain_extent[1] as f32],
            depth_range: 0.0..1.0,
          }]) /* Region of the framebuffer that will be rendered to--almost always 0,0 to
                * width, height.  A viewport acts as a transformation, so it'll actually shrink
                * the final image to fit in the viewport if thats what you do.  Scissors just
                * clip it. This also sets scissors to match the viewport, so no need to also
                * specify that. */
          // End Viewport/Scissor fields

          // Begin PipelineRasterizationStateCreateInfo fields.
          .depth_clamp(false) /* If this is true, any fragments beyond near and far planes of the 
                                 view frustum will be clamped to n and f instead of discarded.*/
          .polygon_mode_fill() // Fill polygons, not only edges, not only vertices
          //.rasterizer_discard(false) // this is off by default but just being explicit for the tutorial.
          .line_width(1.0) // Lines connecting vertices in terms of number of frags, any more than 1 requires wideLines GPU feature.
          .cull_mode_back() // cull back faces of primitives
          .front_face_clockwise() // points in clockwise order are the fronts of primitives
          // No depth biasing...something useful for shadow mapping, default is disabled.
          // End Rasterization fields

          // Begin PipelineMultisampleStateCreateInfo fields.
          // disabled by default, lets keep it that way.
          // End PipelineMultisampleStateCreateInfo fields.

          // Begin PipelineDepthStencilStateCreateInfo fields.
          // disabled by default, lets keep it that way.
          // End PipelineDepthStencilStateCreateInfo fields.

          // Dont want to have to set any dynamic states (Viewport, linewidth, etc) at drawing time so leave
          // PipelineDynamicStateCreateInfo empty.

          // No uniforms so no need to make a PipelineLayout

          // Begin PipelineColorBlend fields. How to combine with the color already in the framebuffer.
          .blend_pass_through() // Disable blending.
          // End PipelineColorBlend fields.

          // Begin shader fields.
          .vertex_shader(vert_shader_module.main_entry_point(), ()) /* Second fields in shaders are
                                                                       constants*/
          .fragment_shader(frag_shader_module.main_entry_point(), ())
          // Now just add the render_pass and build.
          .render_pass(Subpass::from(render_pass.clone(), 0).expect("Could not create subpass from render pass"))
          .build(logical_device.clone())
          .expect("Could not build graphics pipeline"),
    )
  }

  fn create_render_pass(
    logical_device: &Arc<Device>, color_format: Format,
  ) -> Arc<dyn RenderPassAbstract + Send + Sync> {
    // This macro does all the work of building up the render pass, there is a
    // multiplass equivalent as well.
    // Vulkano does all this because there is a type template param for RenderPass
    // that is hard to make yourself that is used for compile time safety checks.
    // see [the deferred rendering example](https://github.com/vulkano-rs/vulkano/blob/master/examples/src/bin/deferred/frame/system.rs)
    // for a more in depth description.
    Arc::new(
      single_pass_renderpass!(logical_device.clone(),
        attachments: {
          color: { // The 0th attachment (directly referenced by layout = 0 output in the frag shader).
            load: Clear, // Clear the screen before rendering.
            store: Store, // Store the rendered contents to memory
            format: color_format, // Color format of the attachment
            samples: 1, // No multisampling
            // Stencil store and load are dont care since we aren't using a stencil
            // Initial layout undefined
            // Final layout is set by the single pass macro as VK_IMAGE_LAYOUT_RESENT_SRC_KHR
          }
        },
        pass: { // Only one subpass, the color attachment output defined above.
          color: [color],
          depth_stencil: {}
        }
      )
      .expect("Error building render pass"),
    )
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

  fn get_required_instance_extensions() -> InstanceExtensions {
    let mut extensions = vulkano_win::required_extensions();

    if ENABLE_VALIDATION_LAYERS {
      // No need to check for the existence of this extension because the validation
      // layers being present already confirms it.
      extensions.ext_debug_utils = true;
    }

    extensions
  }

  fn print_supported_instance_extensions() {
    let supported_extensions =
      InstanceExtensions::supported_by_core().expect("failed to retrieve supported extensions");
    println!(
      "Supported Instance Extensions:\n\t{:?}",
      supported_extensions
    );
  }

  /// Checks if the device has all the features and queue families needed.
  fn is_device_suitable(surface: &Arc<Surface<Window>>, physical_device: &PhysicalDevice) -> bool {
    // Supports all queue families we need.
    let queue_family_indices = Self::find_queue_families(surface, physical_device);
    // Supports all the extensions we need (such as VK_KHR_swapchain)
    let device_extensions_supported = Self::are_all_required_extensions_supported(physical_device);

    // Swap chain is adequate for vulkan-tutorial if it has at least one supported
    // image format and one supported presentation mode for the given surface.
    let swap_chain_adequate = if device_extensions_supported {
      // This capabilities struct contains everything laid out in
      // SwapChainSupportDetails (and more) from the vulkan-tutorial.
      let capabilities = surface
        .capabilities(*physical_device)
        .expect("Could not query surface capabilities");
      !capabilities.supported_formats.is_empty()
        && capabilities.present_modes.iter().next().is_some()
    } else {
      false
    };

    queue_family_indices.is_complete() && device_extensions_supported && swap_chain_adequate
  }

  /// Returns the [QueueFamilyIndices](struct.QueueFamilyIndices.html) supported
  /// by this physical device.
  /// Right now graphics/present queues are treated seperately, but we could try
  /// to optimize by looking for a queue family that supports both instead of
  /// the first of each.
  /// TODO QueueFamilyIndices member
  fn find_queue_families(
    surface: &Arc<Surface<Window>>, physical_device: &PhysicalDevice,
  ) -> QueueFamilyIndices {
    let mut indices_of_queue_families_that_support_feature = QueueFamilyIndices::new();

    for (i, queue_family) in physical_device.queue_families().enumerate() {
      if queue_family.supports_graphics() {
        indices_of_queue_families_that_support_feature.graphics_queue_family_index = Some(i);
      }
      if surface.is_supported(queue_family).unwrap() {
        indices_of_queue_families_that_support_feature.present_queue_family_index = Some(i);
      }

      if indices_of_queue_families_that_support_feature.is_complete() {
        break;
      }
    }

    indices_of_queue_families_that_support_feature
  }

  /// Check all the extensions defined by required_device_extensions() are
  /// supported by the device.
  fn are_all_required_extensions_supported(device: &PhysicalDevice) -> bool {
    let available_device_extensions = DeviceExtensions::supported_by_device(*device);
    let required_device_extensions = required_device_extensions();
    println!(
      "Supported Device Extensions:\n\t{:?}\nRequired Device Extensions:\n\t{:?}\n",
      available_device_extensions, required_device_extensions
    );

    available_device_extensions.intersection(&required_device_extensions)
      == required_device_extensions
  }

  /// Choose the [format](https://www.khronos.org/registry/vulkan/specs/1.2-khr-extensions/html/chap33.html#VkFormat)
  /// and [color space](https://www.khronos.org/registry/vulkan/specs/1.2-khr-extensions/html/chap29.html#VkColorSpaceKHR)
  /// we want to draw to the surface.  A format is the layout of the pixels for
  /// a color (eg R8G8B8 etc).  A ColorSpace is a form of sRGB to determine
  /// the true color from the digital color.
  ///
  /// For the purposes of vulkan-tutorial, we want B8G8R8A8_SRGB format and
  /// VK_COLOR_SPACE_NONLINEAR_KHR for colorspace.
  fn choose_swap_surface_format(
    available_formats: &[(Format, ColorSpace)],
  ) -> (Format, ColorSpace) {
    *available_formats
      .iter()
      .find(|(format, colorspace)| {
        *format == Format::B8G8R8A8Srgb && *colorspace == ColorSpace::SrgbNonLinear
      })
      .unwrap_or(&available_formats[0])
  }

  /// Selects mailbox if available (double buffer with replacement aka triple
  /// buffer) otherwise just regular FIFO (double buffer).
  fn choose_swap_present_mode(available_present_modes: SupportedPresentModes) -> PresentMode {
    if available_present_modes.mailbox {
      PresentMode::Mailbox
    } else {
      // Fifo is guaranteed to be available.
      PresentMode::Fifo
    }
  }

  fn choose_swap_extent(capabilities: &Capabilities) -> [u32; 2] {
    if let Some(current_extent) = capabilities.current_extent {
      return current_extent;
    }

    // The window manager indicates we can set the extent to our liking, lets set it
    // to the window dimensions clamped to the min and max supported by the surface.
    [
      capabilities.min_image_extent[0].max(capabilities.max_image_extent[0].min(WIDTH)),
      capabilities.min_image_extent[1].max(capabilities.max_image_extent[1].min(HEIGHT)),
    ]
  }

  /// Takes full control of the executing thread and runs the event loop for it.
  fn main_loop(self) {
    self
      .window_surface
      .run(move |window_event, _, control_flow| {
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

fn main() {
  let app = HelloTriangleApplication::initialize();
  app.main_loop();
}
