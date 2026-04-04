#include "wave_spectrum.h"

#include <cmath>
#include <random>

namespace ocean {

namespace {

constexpr float kG = 9.81f;
constexpr float kAlpha = 8.1e-3f;
constexpr float kBeta = 0.74f;

float PM_S(float omega, float omega0) {
    if (omega <= 1e-6f) {
        return 0.f;
    }
    const float w5 = omega * omega * omega * omega * omega;
    const float r = omega0 / omega;
    const float exp_term = std::exp(-kBeta * (r * r * r * r));
    return kAlpha * kG * kG * exp_term / w5;
}

}  // namespace

void GenerateWavesFromPiersonMoskowitz(
    std::vector<GerstnerWaveGPU>& out_waves,
    uint32_t wave_count,
    float wind_speed_u19_5,
    uint32_t seed) {
    out_waves.clear();
    if (wave_count == 0 || wind_speed_u19_5 < 0.1f) {
        return;
    }

    const float omega0 = kG / wind_speed_u19_5;
    const float omega_min = 0.15f * omega0;
    const float omega_max = 6.f * omega0;

    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> u01(0.f, 1.f);

    out_waves.reserve(wave_count);
    for (uint32_t i = 0; i < wave_count; ++i) {
        const float t = u01(rng);
        const float log_w_min = std::log(omega_min);
        const float log_w_max = std::log(omega_max);
        const float omega = std::exp(log_w_min + t * (log_w_max - log_w_min));

        const float S = PM_S(omega, omega0);
        const float delta_omega = (omega_max - omega_min) / static_cast<float>(wave_count);
        float A = std::sqrt(std::max(S * delta_omega, 1e-12f));
        A = std::min(A, 2.5f);

        const float angle = u01(rng) * 6.2831853f;
        const float k_len = (omega * omega) / kG;
        GerstnerWaveGPU w{};
        w.k.x = k_len * std::cos(angle);
        w.k.z = k_len * std::sin(angle);
        w.k.y = 0.f;
        w.omega = omega;
        w.amplitude = A;
        w._pad[0] = w._pad[1] = w._pad[2] = 0.f;
        out_waves.push_back(w);
    }
}

}  // namespace ocean
