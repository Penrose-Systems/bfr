// A closest-hit shader that leaves the instance's custom index in
// trace.rgen's payload.
#version 460
#extension GL_EXT_ray_tracing : require

layout(location = 0) rayPayloadInEXT uint hit;

void main() {
    hit = gl_InstanceCustomIndexEXT;
}
