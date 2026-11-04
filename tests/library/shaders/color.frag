// A fragment shader that writes the color the push constants hold, declared
// as BFR's layout (api/bfr.yaml, bfr_pipeline) states. For the draw tests;
// spirv.h holds it compiled, see compile.sh.
#version 460

layout(push_constant) uniform Push {
    uint positions;
    uint textureIndex;
    uint samplerIndex;
    uint pad;
    vec4 color;
} push;

layout(location = 0) out vec4 color;

void main() {
    color = push.color;
}
