#pragma once

#include <DirectXMath.h>
#include <cstdint>
#include <vector>

namespace ocean {

// GPU layout must match GerstnerWave in shaders/basic.hlsl (32-byte stride).
struct GerstnerWaveGPU {
    DirectX::XMFLOAT3 k{};
    float omega{};
    float amplitude{};
    float _pad[3]{};
};

// Discrete Pierson-Moskowitz spectrum samples -> Gerstner wave parameters.
void GenerateWavesFromPiersonMoskowitz(
    std::vector<GerstnerWaveGPU>& out_waves,
    uint32_t wave_count,
    float wind_speed_u19_5,
    uint32_t seed = 1);

}  // namespace ocean
