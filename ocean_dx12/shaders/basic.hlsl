cbuffer FrameCB : register(b0) {
    row_major float4x4 View;
    row_major float4x4 Proj;
    row_major float4x4 ViewNoRoll;
    row_major float4x4 ViewWithRoll;
    float4 CameraPos;
    float4 Params;
    float4 Params2;
};

struct GerstnerWave {
    float3 k;
    float omega;
    float amplitude;
    float3 _pad;
};

StructuredBuffer<GerstnerWave> gWaves : register(t0);

struct VSInput {
    float3 pos : POSITION;
};

struct VSOutput {
    float4 clip : SV_POSITION;
    float3 worldPos : TEXCOORD0;
};

VSOutput VSMain(VSInput vin) {
    VSOutput o;
    float4 wp = float4(vin.pos, 1.0f);
    o.worldPos = wp.xyz;
    o.clip = mul(mul(wp, ViewWithRoll), Proj);
    return o;
}

float4 PSMain(VSOutput pin) : SV_TARGET {
    uint wc = (uint) Params2.y;
    float h = 0.0f;
    float t = Params.z;
    float g = 9.81f;

    for (uint i = 0; i < wc && i < 256u; ++i) {
        GerstnerWave w = gWaves[i];
        float kxz_len = length(w.k.xz);
        if (kxz_len < 1e-6f)
            continue;
        float phase = dot(w.k.xz, pin.worldPos.xz) - w.omega * t;
        h += w.amplitude * cos(phase);
    }

    float3 base = float3(0.05f, 0.18f, 0.32f);
    float3 tint = float3(0.12f, 0.35f, 0.45f) * saturate(h * 0.25f + 0.5f);
    return float4(base + tint, 1.0f);
}
