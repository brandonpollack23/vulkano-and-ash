#version 450
// This extension allows for multiple multiple to use this shader at a time.
// Since we already kind of keep things seperate in Vulkan I feel like this was already implied.
#extension GL_ARB_seperate_shader_objects : enable

layout(location = 0) in vec2 inPosition;
layout(location = 1) in vec3 inColor;

// The location layout here is actually what links to the fragment shader (0)
layout(location = 0) out vec3 fragColor;

// Normally we would never do this constant array thing, but for the sake of
// tutorials...
vec2 positions[3] = vec2[](
  vec2(0.0, -0.5), // center
  vec2(0.5, 0.5),  // bottom right
  vec2(-0.5, 0.5)  // bottom left
);

vec3 colors[3] = vec3[](
  vec3(1.0, 0.0, 0.0),
  vec3(0.0, 1.0, 0.0),
  vec3(0.0, 0.0, 1.0)
);

void main() {
  // Output directly in clip coordinates, with a homogeneous coordinate of
  // 1 so the perspective divide does nothing.
  // gl_VertexIndex is a built in specifying what the current value of the index buffer is.
  gl_Position = vec4(positions[gl_VertexIndex], 0.0, 1.0);
  fragColor = colors[gl_VertexIndex];
}