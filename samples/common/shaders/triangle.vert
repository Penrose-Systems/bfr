// Pulls each vertex's position from the storage buffer the push constants
// name, through BFR's storage buffer table (api/bfr.yaml, bfr_pipeline).
#version 460
#extension GL_EXT_nonuniform_qualifier : require

layout(set = 0, binding = 2) readonly buffer Positions {
    vec4 positions[];
} buffers[];

layout(push_constant) uniform Push {
    uint positions;
    uint pad0;
    uint pad1;
    uint pad2;
    vec4 color;
} push;

void main() {
    gl_Position = buffers[nonuniformEXT(push.positions)].positions[gl_VertexIndex];
}
