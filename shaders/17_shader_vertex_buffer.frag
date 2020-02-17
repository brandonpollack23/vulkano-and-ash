#version 450
// This extension allows for multiple multiple to use this shader at a time.  Since we already kind of keep things seperate in Vulkan I feel like this was already implied.
#extension GL_ARB_seperate_shader_objects : enable

// Specify the same input location index (0) as the output of the vert shader.
// The value here will be automagically interpolated by the GPU in all dimensions
// for each fragment's position.
layout(location = 0) in vec3 fragColor;

// Unlike gl_Position in vertex shaders, there is no default output for fragment
// shaders, so we create one called outColor
// location = 0 specifies the index of the (only) framebuffer to write to.
layout(location = 0) out vec4 outColor;

void main() {
  // Sometimes you'll see shaders using gl_FragColor for output instead of
  // specifying.  This is deprecated, you should specify yourself.
  outColor = vec4(fragColor, 1.0);
}
