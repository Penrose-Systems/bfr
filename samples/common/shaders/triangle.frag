// Writes the color the push constants hold.
#version 460

layout(push_constant) uniform Push {
    uint positions;
    uint pad0;
    uint pad1;
    uint pad2;
    vec4 color;
} push;

layout(location = 0) out vec4 color;

void main() {
    color = push.color;
}
