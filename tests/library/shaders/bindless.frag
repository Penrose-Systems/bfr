// A fragment shader that samples through BFR's texture and sampler tables
// (api/bfr.yaml, bfr_pipeline). For the pipeline tests; spirv.h holds it
// compiled, see compile.sh.
#version 460
#extension GL_EXT_nonuniform_qualifier : require

layout(set = 0, binding = 0) uniform texture2D textures[];
layout(set = 0, binding = 3) uniform sampler samplers[];

layout(push_constant) uniform Push {
    uint positions;
    uint textureIndex;
    uint samplerIndex;
} push;

layout(location = 0) out vec4 color;

void main() {
    color = texture(sampler2D(textures[nonuniformEXT(push.textureIndex)], samplers[nonuniformEXT(push.samplerIndex)]),
                    vec2(0.5));
}
