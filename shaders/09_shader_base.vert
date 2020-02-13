#version 450

// The location layout here is actually what links to the fragment shader (0)
layout(location = 0) out vec3 fragColor;

// Normally we would never do this constant array thing, but for the sake of
// tutorials...
vec2 positions[3] = vec2[](
vec2(0.0, -0.5), // center
vec2(0.5, 0.5), // bottom right
vec2(-0.5, 0.5)// bottom left
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