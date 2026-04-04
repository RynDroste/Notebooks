cbuffer FrameCB : register(b0) {
    row_major float4x4 View;
    row_major float4x4 Proj;
    row_major float4x4 ViewNoRoll;
    row_major float4x4 InvViewNoRoll;
    row_major float4x4 InvViewProjNoRoll;
    float4 CameraPos;
    float4 Params;
    float4 Params2;
    float4 ScreenHorizon;
};

struct GerstnerWave {
    float3 k;
    float omega;
    float amplitude;
    float3 _pad;
};

StructuredBuffer<GerstnerWave> gWaves : register(t0);

struct VS_OUT {
    float4 ndcHint : POSITION;
};

struct HS_PATCH {
    float edge[4] : SV_TessFactor;
    float inside[2] : SV_InsideTessFactor;
};

struct DS_OUT {
    float4 clip : SV_POSITION;
    float3 worldPos : WORLDPOS;
    float3 normalW : NORMAL;
};

VS_OUT VSMain(uint vid : SV_VertexID) {
    VS_OUT o;
    float H = clamp(ScreenHorizon.z, -1.0, 1.0);
    float2 p;
    if (vid == 0)
        p = float2(-1.0, -1.0);
    else if (vid == 1)
        p = float2(1.0, -1.0);
    else if (vid == 2)
        p = float2(-1.0, H);
    else
        p = float2(1.0, H);
    o.ndcHint = float4(p, 0, 1);
    return o;
}

HS_PATCH CalcPatchFactors(float Hndc, float N) {
    HS_PATCH pt;
    float h = saturate(Hndc * 0.5 + 0.5);
    float inner1 = max(1.0, floor(h * N + 0.5));
    float inner0 = max(1.0, floor(N));
    pt.edge[0] = inner1;
    pt.edge[1] = inner1;
    pt.edge[2] = inner0;
    pt.edge[3] = inner0;
    pt.inside[0] = inner0;
    pt.inside[1] = inner1;
    return pt;
}

HS_PATCH HSPatch(InputPatch<VS_OUT, 4> patch) {
    float H = ScreenHorizon.z;
    float N = max(1.0, Params.w);
    return CalcPatchFactors(H, N);
}

[domain("quad")]
[partitioning("integer")]
[outputtopology("triangle_cw")]
[outputcontrolpoints(4)]
[patchconstantfunc("HSPatch")]
VS_OUT HSMain(InputPatch<VS_OUT, 4> patch, uint id : SV_OutputControlPointID) {
    return patch[id];
}

float3 WorldPosOnSeaPlane(float2 ndc) {
    float3 o = CameraPos.xyz;
    float4 wNear = mul(float4(ndc.x, ndc.y, 0.0, 1.0), InvViewProjNoRoll);
    float4 wFar = mul(float4(ndc.x, ndc.y, 1.0, 1.0), InvViewProjNoRoll);
    wNear.xyz /= max(abs(wNear.w), 1e-8);
    wFar.xyz /= max(abs(wFar.w), 1e-8);
    float3 dir = wFar.xyz - wNear.xyz;
    float dlen = length(dir);
    if (dlen < 1e-8)
        return float3(o.x, 0.0, o.z);
    dir /= dlen;
    if (dir.y >= -1e-4) {
        float3 horiz = float3(dir.x, 0.0, dir.z);
        float hl = length(horiz);
        if (hl < 1e-6)
            horiz = float3(1.0, 0.0, 0.0);
        else
            horiz /= hl;
        dir = normalize(horiz + float3(0.0, -0.02, 0.0));
    }
    dir.y = min(dir.y, -1e-4);
    float t = -o.y / dir.y;
    t = clamp(t, 0.0, 5.0e5);
    return o + t * dir;
}

float EstimateLFallback(float3 worldOnPlane, float tessAlongV) {
    float3 o = CameraPos.xyz;
    float dist = length(worldOnPlane - o);
    float halfFov = Params.y;
    float sh = max(ScreenHorizon.y, 1.0);
    float mu = dist * tan(halfFov) / sh;
    return max(mu / max(tessAlongV, 1.0), 0.05);
}

float CalcFilterThresholdL_Algorithm1(float3 dWorld, float nPlane, float halfFov, float Nsubdiv) {
    float N = max(Nsubdiv, 1.0);
    float mu = nPlane * tan(halfFov) / N;
    float4 t = float4(0, 1, 0, 0);
    float4 fVec = float4(0, 0, 1, 0);
    float4 dView = mul(float4(dWorld, 0.0), ViewNoRoll);
    float4 dPerp = float4(0, dView.y, dView.z, 0);
    float denom = dot(dPerp, fVec);
    if (abs(denom) < 1e-8)
        return -1.0;
    float4 qPerp = (nPlane / denom) * dPerp + mu * t;
    float4 dW = mul(dPerp, InvViewNoRoll);
    float4 qW = mul(qPerp, InvViewNoRoll);
    float lenD = length(dW.xyz);
    float lenQ = length(qW.xyz);
    if (lenD < 1e-10 || lenQ < 1e-10)
        return -1.0;
    float3 dPrime = dW.xyz / lenD;
    float3 qPrime = qW.xyz / lenQ;
    float3 o = CameraPos.xyz;
    if (abs(dPrime.y) < 1e-8 || abs(qPrime.y) < 1e-8)
        return -1.0;
    float alpha = o.y / dPrime.y;
    float beta = o.y / qPrime.y;
    float3 diff = beta * qPrime - alpha * dPrime;
    return max(length(diff), 1e-4);
}

void GerstnerAddDisp(GerstnerWave w, float2 xz, float t, inout float3 disp) {
    float kxz = length(w.k.xz);
    if (kxz < 1e-6)
        return;
    float phase = dot(w.k.xz, xz) - w.omega * t;
    float s = sin(phase);
    float c = cos(phase);
    disp.x += -(w.k.x / kxz) * w.amplitude * s;
    disp.y += w.amplitude * c;
    disp.z += -(w.k.z / kxz) * w.amplitude * s;
}

float3 SumWavesDisp(float3 baseWorld, float L, float t) {
    uint wc = (uint) (Params2.y + 0.5);
    float g = 9.81;
    float kNyq = Params2.x;
    float3 disp = 0;
    float2 xz = baseWorld.xz;
    for (uint i = 0; i < wc && i < 256u; ++i) {
        GerstnerWave w = gWaves[i];
        float klen = length(w.k.xz);
        if (klen < 1e-6)
            continue;
        float lambda = 2.0 * 3.14159265 * g / klen;
        if (abs(lambda * w.omega) < kNyq * L)
            continue;
        GerstnerAddDisp(w, xz, t, disp);
    }
    return baseWorld + disp;
}

[domain("quad")]
DS_OUT DSMain(HS_PATCH factors, float2 uv : SV_DomainLocation, const OutputPatch<VS_OUT, 4> patch) {
    DS_OUT o;
    float4 b = lerp(lerp(patch[0].ndcHint, patch[1].ndcHint, uv.x), lerp(patch[2].ndcHint, patch[3].ndcHint, uv.x), uv.y);
    float2 ndc = b.xy;
    float3 worldFlat = WorldPosOnSeaPlane(ndc);
    float tessV = max(factors.inside[1], 1.0);
    float3 o = CameraPos.xyz;
    float3 dW = worldFlat - o;
    float dLen = length(dW);
    float3 dWorld = (dLen > 1e-8) ? (dW / dLen) : float3(0, -1, 0);
    float L = CalcFilterThresholdL_Algorithm1(dWorld, Params.x, Params.y, Params.w);
    if (L < 0.0)
        L = EstimateLFallback(worldFlat, tessV);
    float3 worldDisp = SumWavesDisp(worldFlat, L, Params.z);
    float4 clip = mul(mul(float4(worldDisp, 1.0), View), Proj);
    o.clip = clip;
    o.worldPos = worldDisp;
    o.normalW = float3(0, 1, 0);
    return o;
}

float4 PSMain(DS_OUT pin) : SV_TARGET {
    float3 dn = cross(ddx(pin.worldPos), ddy(pin.worldPos));
    float dl = length(dn);
    float3 nDeriv = (dl > 1e-10) ? (dn / dl) : float3(0, 1, 0);
    if (dot(nDeriv, float3(0, 1, 0)) < 0.0)
        nDeriv = -nDeriv;
    float3 nwp = normalize(lerp(float3(0, 1, 0), nDeriv, 0.22));
    float3 Ld = normalize(float3(0.35, 0.9, 0.25));
    float ndl = saturate(dot(nwp, Ld));
    float3 base = float3(0.05, 0.15, 0.28);
    float3 lit = base + float3(0.2, 0.45, 0.55) * ndl;
    return float4(lit, 1.0);
}
