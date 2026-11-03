// A vertex shader that pulls positions through BFR's storage buffer table
// (api/bfr.yaml, bfr_pipeline). For the pipeline tests; spirv.h holds it
// compiled, see compile.sh.
#version 460
#extension GL_EXT_nonuniform_qualifier : require

layout(set = 0, binding = 2) readonly buffer Positions {
    vec4 positions[];
} buffers[];

layout(push_constant) uniform Push {
    uint positions;
    uint textureIndex;
    uint samplerIndex;
} push;

void main() {
    gl_Position = buffers[nonuniformEXT(push.positions)].positions[gl_VertexIndex];
}
