// A miss shader that leaves all ones in trace.rgen's payload.
#version 460
#extension GL_EXT_ray_tracing : require

layout(location = 0) rayPayloadInEXT uint hit;

void main() {
    hit = 0xFFFFFFFFu;
}
