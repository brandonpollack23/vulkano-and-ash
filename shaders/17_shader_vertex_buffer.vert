#version 450
// This extension allows for multiple multiple to use this shader at a time.
// Since we already kind of keep things seperate in Vulkan I feel like this was already implied.
#extension GL_ARB_separate_shader_objects : enable

layout(location = 0) in vec2 inPosition;
layout(location = 1) in vec3 inColor;

// The location layout here is actually what links to the fragment shader (0)
layout(location = 0) out vec3 fragColor;

void main() {
  // Output directly in clip coordinates, with a homogeneous coordinate of
  // 1 so the perspective divide does nothing.
  // gl_VertexIndex is a built in specifying what the current value of the index buffer is.
  gl_Position = vec4(inPosition, 0.0, 1.0);
  fragColor = inColor;
}