#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>

#include <d3d12.h>
#include <dxgi1_6.h>
#include <d3dcompiler.h>
#include <DirectXMath.h>
#include <wrl/client.h>

#include "wave_spectrum.h"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#pragma comment(lib, "d3d12.lib")
#pragma comment(lib, "dxgi.lib")
#pragma comment(lib, "d3dcompiler.lib")

using Microsoft::WRL::ComPtr;
using namespace DirectX;

namespace {

constexpr UINT kFrameCount = 2;
constexpr UINT kWidth = 1280;
constexpr UINT kHeight = 800;
constexpr UINT kWaveCount = 60;
constexpr float kWindU19_5 = 12.f;
constexpr float kNearPlane = 0.1f;
constexpr float kFarPlane = 500.f;
constexpr float kMaxTessN = 64.f;
constexpr float kNyquistK = 4.f;
constexpr float kDemoRollAmplitudeRad = 0.45f;

LRESULT CALLBACK WndProc(HWND hwnd, UINT msg, WPARAM wParam, LPARAM lParam);

inline void BuildCameraViewsWithRollSplit(
    FXMVECTOR eye,
    FXMVECTOR focus,
    float rollRad,
    XMMATRIX& outViewNoRoll,
    XMMATRIX& outViewWithRoll) {
    const XMVECTOR f = XMVector3Normalize(XMVectorSubtract(focus, eye));
    const XMVECTOR worldUp = XMVectorSet(0.f, 1.f, 0.f, 0.f);
    XMVECTOR r0 = XMVector3Cross(worldUp, f);
    const float rl = XMVectorGetX(XMVector3LengthSq(r0));
    if (rl < 1e-10f) {
        r0 = XMVector3Cross(XMVectorSet(1.f, 0.f, 0.f, 0.f), f);
    }
    r0 = XMVector3Normalize(r0);
    const XMVECTOR u0 = XMVector3Normalize(XMVector3Cross(f, r0));

    const float c = std::cos(rollRad);
    const float s = std::sin(rollRad);
    const XMVECTOR r1 = XMVectorAdd(XMVectorScale(r0, c), XMVectorScale(u0, s));
    const XMVECTOR u1 = XMVectorSubtract(XMVectorScale(u0, c), XMVectorScale(r0, s));

    outViewNoRoll = XMMatrixLookToLH(eye, f, u0);
    outViewWithRoll = XMMatrixLookToLH(eye, f, u1);
}

struct alignas(256) FrameConstants {
    XMFLOAT4X4 View{};
    XMFLOAT4X4 Proj{};
    XMFLOAT4X4 ViewNoRoll{};
    XMFLOAT4X4 ViewWithRoll{};
    XMFLOAT4 CameraPos{};
    XMFLOAT4 Params{};   // near, halfFovY, time, maxTessN
    XMFLOAT4 Params2{};  // nyquistK, waveCount, windU19_5, _
    uint8_t _padding[512 - (sizeof(XMFLOAT4X4) * 4 + sizeof(XMFLOAT4) * 3)]{};
};

static_assert(sizeof(FrameConstants) == 512, "FrameConstants must be 512 bytes for CB alignment");

struct Vertex {
    XMFLOAT3 position{};
};

ComPtr<ID3D12Device> g_device;
ComPtr<ID3D12CommandQueue> g_cmdQueue;
ComPtr<IDXGISwapChain3> g_swapChain;
ComPtr<ID3D12DescriptorHeap> g_rtvHeap;
ComPtr<ID3D12DescriptorHeap> g_dsvHeap;
ComPtr<ID3D12Resource> g_renderTargets[kFrameCount];
ComPtr<ID3D12Resource> g_depthStencil;
ComPtr<ID3D12CommandAllocator> g_cmdAlloc[kFrameCount];
ComPtr<ID3D12GraphicsCommandList> g_cmdList;
ComPtr<ID3D12Fence> g_fence;
UINT64 g_fenceValue = 0;
HANDLE g_fenceEvent = nullptr;
UINT g_rtvDescriptorSize = 0;
UINT g_dsvDescriptorSize = 0;
UINT g_frameIndex = 0;
HWND g_hwnd = nullptr;

ComPtr<ID3D12RootSignature> g_rootSig;
ComPtr<ID3D12PipelineState> g_pso;
ComPtr<ID3D12Resource> g_vb;
ComPtr<ID3D12Resource> g_ib;
ComPtr<ID3D12Resource> g_vbUpload;
ComPtr<ID3D12Resource> g_ibUpload;
UINT g_indexCount = 0;
D3D12_VERTEX_BUFFER_VIEW g_vbv{};
D3D12_INDEX_BUFFER_VIEW g_ibv{};

ComPtr<ID3D12Resource> g_frameCB[kFrameCount];
void* g_frameCBMapped[kFrameCount]{};
ComPtr<ID3D12Resource> g_wavesBuffer;
UINT g_srvDescriptorSize = 0;
ComPtr<ID3D12DescriptorHeap> g_srvHeap;

ComPtr<ID3DBlob> g_vsBlob;
ComPtr<ID3DBlob> g_psBlob;

float g_timeSec = 0.f;

bool GetShaderPath(wchar_t* out, size_t outChars) {
    wchar_t exe[MAX_PATH];
    if (!GetModuleFileNameW(nullptr, exe, MAX_PATH)) {
        return false;
    }
    wchar_t* last = wcsrchr(exe, L'\\');
    if (!last) {
        return false;
    }
    *last = L'\0';
    swprintf_s(out, outChars, L"%s\\shaders\\basic.hlsl", exe);
    return true;
}

void WaitForGpu() {
    const UINT64 v = ++g_fenceValue;
    if (FAILED(g_cmdQueue->Signal(g_fence.Get(), v))) {
        return;
    }
    if (g_fence->GetCompletedValue() < v) {
        g_fence->SetEventOnCompletion(v, g_fenceEvent);
        WaitForSingleObject(g_fenceEvent, INFINITE);
    }
}

bool CompileShaderFile(const wchar_t* path, const char* entry, const char* target, ComPtr<ID3DBlob>& outBlob) {
    ComPtr<ID3DBlob> err;
    const UINT flags = D3DCOMPILE_ENABLE_STRICTNESS | D3DCOMPILE_PACK_MATRIX_ROW_MAJOR;
    const HRESULT hr = D3DCompileFromFile(
        path,
        nullptr,
        D3D_COMPILE_STANDARD_FILE_INCLUDE,
        entry,
        target,
        flags,
        0,
        &outBlob,
        &err);
    if (FAILED(hr)) {
        if (err) {
            OutputDebugStringA(static_cast<const char*>(err->GetBufferPointer()));
        }
        return false;
    }
    return true;
}

bool CreateDeviceAndSwapChain(HWND hwnd) {
    UINT dxgiFactoryFlags = 0;
#if defined(_DEBUG)
    {
        ComPtr<ID3D12Debug> dbg;
        if (SUCCEEDED(D3D12GetDebugInterface(IID_PPV_ARGS(&dbg)))) {
            dbg->EnableDebugLayer();
            dxgiFactoryFlags |= DXGI_CREATE_FACTORY_DEBUG;
        }
    }
#endif

    ComPtr<IDXGIFactory4> factory;
    if (FAILED(CreateDXGIFactory2(dxgiFactoryFlags, IID_PPV_ARGS(&factory)))) {
        return false;
    }

    ComPtr<IDXGIAdapter1> adapter;
    for (UINT i = 0; DXGI_ERROR_NOT_FOUND != factory->EnumAdapters1(i, &adapter); ++i, adapter.Reset()) {
        DXGI_ADAPTER_DESC1 desc{};
        adapter->GetDesc1(&desc);
        if (desc.Flags & DXGI_ADAPTER_FLAG_SOFTWARE) {
            continue;
        }
        if (SUCCEEDED(D3D12CreateDevice(adapter.Get(), D3D_FEATURE_LEVEL_11_0, IID_PPV_ARGS(&g_device)))) {
            break;
        }
    }
    if (!g_device) {
        if (FAILED(D3D12CreateDevice(nullptr, D3D_FEATURE_LEVEL_11_0, IID_PPV_ARGS(&g_device)))) {
            return false;
        }
    }

    D3D12_COMMAND_QUEUE_DESC qd{};
    qd.Type = D3D12_COMMAND_LIST_TYPE_DIRECT;
    if (FAILED(g_device->CreateCommandQueue(&qd, IID_PPV_ARGS(&g_cmdQueue)))) {
        return false;
    }

    DXGI_SWAP_CHAIN_DESC1 scd{};
    scd.Width = kWidth;
    scd.Height = kHeight;
    scd.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
    scd.SampleDesc.Count = 1;
    scd.BufferUsage = DXGI_USAGE_RENDER_TARGET_OUTPUT;
    scd.BufferCount = kFrameCount;
    scd.SwapEffect = DXGI_SWAP_EFFECT_FLIP_DISCARD;
    scd.AlphaMode = DXGI_ALPHA_MODE_UNSPECIFIED;
    scd.Scaling = DXGI_SCALING_STRETCH;

    ComPtr<IDXGISwapChain1> sc1;
    if (FAILED(factory->CreateSwapChainForHwnd(g_cmdQueue.Get(), hwnd, &scd, nullptr, nullptr, &sc1))) {
        return false;
    }
    if (FAILED(sc1.As(&g_swapChain))) {
        return false;
    }
    factory->MakeWindowAssociation(hwnd, DXGI_MWA_NO_ALT_ENTER);

    g_frameIndex = g_swapChain->GetCurrentBackBufferIndex();

    D3D12_DESCRIPTOR_HEAP_DESC rtvHd{};
    rtvHd.NumDescriptors = kFrameCount;
    rtvHd.Type = D3D12_DESCRIPTOR_HEAP_TYPE_RTV;
    if (FAILED(g_device->CreateDescriptorHeap(&rtvHd, IID_PPV_ARGS(&g_rtvHeap)))) {
        return false;
    }
    g_rtvDescriptorSize = g_device->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_RTV);

    D3D12_CPU_DESCRIPTOR_HANDLE rtvHandle = g_rtvHeap->GetCPUDescriptorHandleForHeapStart();
    for (UINT n = 0; n < kFrameCount; ++n) {
        if (FAILED(g_swapChain->GetBuffer(n, IID_PPV_ARGS(&g_renderTargets[n])))) {
            return false;
        }
        g_device->CreateRenderTargetView(g_renderTargets[n].Get(), nullptr, rtvHandle);
        rtvHandle.ptr += static_cast<SIZE_T>(g_rtvDescriptorSize);
    }

    D3D12_DESCRIPTOR_HEAP_DESC dsvHd{};
    dsvHd.NumDescriptors = 1;
    dsvHd.Type = D3D12_DESCRIPTOR_HEAP_TYPE_DSV;
    if (FAILED(g_device->CreateDescriptorHeap(&dsvHd, IID_PPV_ARGS(&g_dsvHeap)))) {
        return false;
    }
    g_dsvDescriptorSize = g_device->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_DSV);

    D3D12_HEAP_PROPERTIES hp{};
    hp.Type = D3D12_HEAP_TYPE_DEFAULT;
    D3D12_RESOURCE_DESC ds{};
    ds.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
    ds.Width = kWidth;
    ds.Height = kHeight;
    ds.DepthOrArraySize = 1;
    ds.MipLevels = 1;
    ds.Format = DXGI_FORMAT_D32_FLOAT;
    ds.SampleDesc.Count = 1;
    ds.Flags = D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL;
    D3D12_CLEAR_VALUE clear{};
    clear.Format = DXGI_FORMAT_D32_FLOAT;
    clear.DepthStencil.Depth = 1.f;
    if (FAILED(g_device->CreateCommittedResource(
            &hp,
            D3D12_HEAP_FLAG_NONE,
            &ds,
            D3D12_RESOURCE_STATE_DEPTH_WRITE,
            &clear,
            IID_PPV_ARGS(&g_depthStencil)))) {
        return false;
    }
    g_device->CreateDepthStencilView(g_depthStencil.Get(), nullptr, g_dsvHeap->GetCPUDescriptorHandleForHeapStart());

    for (UINT i = 0; i < kFrameCount; ++i) {
        if (FAILED(g_device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&g_cmdAlloc[i])))) {
            return false;
        }
    }
    if (FAILED(g_device->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, g_cmdAlloc[0].Get(), nullptr, IID_PPV_ARGS(&g_cmdList)))) {
        return false;
    }
    g_cmdList->Close();

    if (FAILED(g_device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&g_fence)))) {
        return false;
    }
    g_fenceValue = 1;
    g_fenceEvent = CreateEvent(nullptr, FALSE, FALSE, nullptr);
    if (!g_fenceEvent) {
        return false;
    }

    return true;
}

bool CreateRootSignatureAndPSO() {
    wchar_t shaderPath[MAX_PATH];
    if (!GetShaderPath(shaderPath, MAX_PATH)) {
        return false;
    }
    if (!CompileShaderFile(shaderPath, "VSMain", "vs_5_0", g_vsBlob)) {
        return false;
    }
    if (!CompileShaderFile(shaderPath, "PSMain", "ps_5_0", g_psBlob)) {
        return false;
    }

    D3D12_DESCRIPTOR_RANGE ranges[1]{};
    ranges[0].RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_SRV;
    ranges[0].NumDescriptors = 1;
    ranges[0].BaseShaderRegister = 0;
    ranges[0].RegisterSpace = 0;
    ranges[0].OffsetInDescriptorsFromTableStart = D3D12_DESCRIPTOR_RANGE_OFFSET_APPEND;

    D3D12_ROOT_PARAMETER rootParams[2]{};
    rootParams[0].ParameterType = D3D12_ROOT_PARAMETER_TYPE_CBV;
    rootParams[0].Descriptor.ShaderRegister = 0;
    rootParams[0].Descriptor.RegisterSpace = 0;
    rootParams[0].ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;

    rootParams[1].ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
    rootParams[1].DescriptorTable.NumDescriptorRanges = 1;
    rootParams[1].DescriptorTable.pDescriptorRanges = ranges;
    rootParams[1].ShaderVisibility = D3D12_SHADER_VISIBILITY_PIXEL;

    D3D12_ROOT_SIGNATURE_DESC rs{};
    rs.NumParameters = 2;
    rs.pParameters = rootParams;
    rs.NumStaticSamplers = 0;
    rs.pStaticSamplers = nullptr;
    rs.Flags = D3D12_ROOT_SIGNATURE_FLAG_ALLOW_INPUT_ASSEMBLER_INPUT_LAYOUT;

    ComPtr<ID3DBlob> rsBlob;
    ComPtr<ID3DBlob> rsErrors;
    if (FAILED(D3D12SerializeRootSignature(&rs, D3D_ROOT_SIGNATURE_VERSION_1, &rsBlob, &rsErrors))) {
        return false;
    }
    if (FAILED(g_device->CreateRootSignature(0, rsBlob->GetBufferPointer(), rsBlob->GetBufferSize(), IID_PPV_ARGS(&g_rootSig)))) {
        return false;
    }

    D3D12_INPUT_ELEMENT_DESC il[] = {
        {"POSITION", 0, DXGI_FORMAT_R32G32B32_FLOAT, 0, 0, D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA, 0},
    };

    D3D12_GRAPHICS_PIPELINE_STATE_DESC pso{};
    pso.pRootSignature = g_rootSig.Get();
    pso.VS = {g_vsBlob->GetBufferPointer(), g_vsBlob->GetBufferSize()};
    pso.PS = {g_psBlob->GetBufferPointer(), g_psBlob->GetBufferSize()};
    pso.BlendState.RenderTarget[0].RenderTargetWriteMask = D3D12_COLOR_WRITE_ENABLE_ALL;
    pso.SampleMask = UINT_MAX;
    pso.RasterizerState.FillMode = D3D12_FILL_MODE_SOLID;
    pso.RasterizerState.CullMode = D3D12_CULL_MODE_NONE;
    pso.RasterizerState.FrontCounterClockwise = FALSE;
    pso.DepthStencilState.DepthEnable = TRUE;
    pso.DepthStencilState.DepthWriteMask = D3D12_DEPTH_WRITE_MASK_ALL;
    pso.DepthStencilState.DepthFunc = D3D12_COMPARISON_FUNC_LESS;
    pso.InputLayout = {il, _countof(il)};
    pso.PrimitiveTopologyType = D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE;
    pso.NumRenderTargets = 1;
    pso.RTVFormats[0] = DXGI_FORMAT_R8G8B8A8_UNORM;
    pso.DSVFormat = DXGI_FORMAT_D32_FLOAT;
    pso.SampleDesc.Count = 1;

    if (FAILED(g_device->CreateGraphicsPipelineState(&pso, IID_PPV_ARGS(&g_pso)))) {
        return false;
    }

    return true;
}

bool CreateSrvHeap() {
    D3D12_DESCRIPTOR_HEAP_DESC hd{};
    hd.NumDescriptors = 1;
    hd.Type = D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV;
    hd.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE;
    if (FAILED(g_device->CreateDescriptorHeap(&hd, IID_PPV_ARGS(&g_srvHeap)))) {
        return false;
    }
    g_srvDescriptorSize = g_device->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
    (void)g_srvDescriptorSize;
    return true;
}

void UploadMesh() {
    const float half = 80.f;
    const Vertex verts[] = {
        {{-half, 0.f, -half}},
        {{half, 0.f, -half}},
        {{half, 0.f, half}},
        {{-half, 0.f, half}},
    };
    const uint16_t idx[] = {0, 1, 2, 0, 2, 3};
    g_indexCount = 6;

    const UINT vbSize = static_cast<UINT>(sizeof(verts));
    const UINT ibSize = static_cast<UINT>(sizeof(idx));

    D3D12_HEAP_PROPERTIES uploadHp{};
    uploadHp.Type = D3D12_HEAP_TYPE_UPLOAD;
    D3D12_RESOURCE_DESC ub{};
    ub.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
    ub.Width = vbSize;
    ub.Height = 1;
    ub.DepthOrArraySize = 1;
    ub.MipLevels = 1;
    ub.SampleDesc.Count = 1;
    ub.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;

    g_device->CreateCommittedResource(&uploadHp, D3D12_HEAP_FLAG_NONE, &ub, D3D12_RESOURCE_STATE_GENERIC_READ, nullptr, IID_PPV_ARGS(&g_vbUpload));
    ub.Width = ibSize;
    g_device->CreateCommittedResource(&uploadHp, D3D12_HEAP_FLAG_NONE, &ub, D3D12_RESOURCE_STATE_GENERIC_READ, nullptr, IID_PPV_ARGS(&g_ibUpload));

    void* map = nullptr;
    g_vbUpload->Map(0, nullptr, &map);
    std::memcpy(map, verts, vbSize);
    g_vbUpload->Unmap(0, nullptr);

    g_ibUpload->Map(0, nullptr, &map);
    std::memcpy(map, idx, ibSize);
    g_ibUpload->Unmap(0, nullptr);

    D3D12_HEAP_PROPERTIES defHp{};
    defHp.Type = D3D12_HEAP_TYPE_DEFAULT;
    ub.Width = vbSize;
    g_device->CreateCommittedResource(&defHp, D3D12_HEAP_FLAG_NONE, &ub, D3D12_RESOURCE_STATE_COPY_DEST, nullptr, IID_PPV_ARGS(&g_vb));
    ub.Width = ibSize;
    g_device->CreateCommittedResource(&defHp, D3D12_HEAP_FLAG_NONE, &ub, D3D12_RESOURCE_STATE_COPY_DEST, nullptr, IID_PPV_ARGS(&g_ib));

    g_cmdAlloc[0]->Reset();
    g_cmdList->Reset(g_cmdAlloc[0].Get(), nullptr);
    g_cmdList->CopyResource(g_vb.Get(), g_vbUpload.Get());
    g_cmdList->CopyResource(g_ib.Get(), g_ibUpload.Get());
    D3D12_RESOURCE_BARRIER b[2]{};
    b[0].Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
    b[0].Transition.pResource = g_vb.Get();
    b[0].Transition.StateBefore = D3D12_RESOURCE_STATE_COPY_DEST;
    b[0].Transition.StateAfter = D3D12_RESOURCE_STATE_VERTEX_AND_CONSTANT_BUFFER;
    b[0].Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
    b[1] = b[0];
    b[1].Transition.pResource = g_ib.Get();
    b[1].Transition.StateAfter = D3D12_RESOURCE_STATE_INDEX_BUFFER;
    g_cmdList->ResourceBarrier(2, b);
    g_cmdList->Close();
    ID3D12CommandList* lists[] = {g_cmdList.Get()};
    g_cmdQueue->ExecuteCommandLists(1, lists);
    WaitForGpu();

    g_vbv.BufferLocation = g_vb->GetGPUVirtualAddress();
    g_vbv.SizeInBytes = vbSize;
    g_vbv.StrideInBytes = sizeof(Vertex);

    g_ibv.BufferLocation = g_ib->GetGPUVirtualAddress();
    g_ibv.SizeInBytes = ibSize;
    g_ibv.Format = DXGI_FORMAT_R16_UINT;
}

bool CreateFrameConstantBuffers() {
    D3D12_HEAP_PROPERTIES hp{};
    hp.Type = D3D12_HEAP_TYPE_UPLOAD;
    D3D12_RESOURCE_DESC rd{};
    rd.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
    rd.Width = sizeof(FrameConstants);
    rd.Height = 1;
    rd.DepthOrArraySize = 1;
    rd.MipLevels = 1;
    rd.SampleDesc.Count = 1;
    rd.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;

    for (UINT i = 0; i < kFrameCount; ++i) {
        if (FAILED(g_device->CreateCommittedResource(
                &hp,
                D3D12_HEAP_FLAG_NONE,
                &rd,
                D3D12_RESOURCE_STATE_GENERIC_READ,
                nullptr,
                IID_PPV_ARGS(&g_frameCB[i])))) {
            return false;
        }
        D3D12_RANGE readRange{0, 0};
        g_frameCB[i]->Map(0, &readRange, &g_frameCBMapped[i]);
    }
    return true;
}

bool CreateWavesBuffer(const std::vector<ocean::GerstnerWaveGPU>& waves) {
    const UINT64 byteSize = static_cast<UINT64>(sizeof(ocean::GerstnerWaveGPU)) * waves.size();
    if (byteSize == 0) {
        return false;
    }

    D3D12_HEAP_PROPERTIES uploadHp{};
    uploadHp.Type = D3D12_HEAP_TYPE_UPLOAD;
    D3D12_RESOURCE_DESC uploadRd{};
    uploadRd.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
    uploadRd.Width = byteSize;
    uploadRd.Height = 1;
    uploadRd.DepthOrArraySize = 1;
    uploadRd.MipLevels = 1;
    uploadRd.SampleDesc.Count = 1;
    uploadRd.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;

    ComPtr<ID3D12Resource> upload;
    if (FAILED(g_device->CreateCommittedResource(
            &uploadHp,
            D3D12_HEAP_FLAG_NONE,
            &uploadRd,
            D3D12_RESOURCE_STATE_GENERIC_READ,
            nullptr,
            IID_PPV_ARGS(&upload)))) {
        return false;
    }

    void* map = nullptr;
    upload->Map(0, nullptr, &map);
    std::memcpy(map, waves.data(), static_cast<size_t>(byteSize));
    upload->Unmap(0, nullptr);

    D3D12_HEAP_PROPERTIES defHp{};
    defHp.Type = D3D12_HEAP_TYPE_DEFAULT;
    if (FAILED(g_device->CreateCommittedResource(
            &defHp,
            D3D12_HEAP_FLAG_NONE,
            &uploadRd,
            D3D12_RESOURCE_STATE_COPY_DEST,
            nullptr,
            IID_PPV_ARGS(&g_wavesBuffer)))) {
        return false;
    }

    g_cmdAlloc[0]->Reset();
    g_cmdList->Reset(g_cmdAlloc[0].Get(), nullptr);
    g_cmdList->CopyResource(g_wavesBuffer.Get(), upload.Get());
    D3D12_RESOURCE_BARRIER bar{};
    bar.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
    bar.Transition.pResource = g_wavesBuffer.Get();
    bar.Transition.StateBefore = D3D12_RESOURCE_STATE_COPY_DEST;
    bar.Transition.StateAfter = D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE;
    bar.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
    g_cmdList->ResourceBarrier(1, &bar);
    g_cmdList->Close();
    ID3D12CommandList* lists[] = {g_cmdList.Get()};
    g_cmdQueue->ExecuteCommandLists(1, lists);
    WaitForGpu();

    D3D12_SHADER_RESOURCE_VIEW_DESC srv{};
    srv.Format = DXGI_FORMAT_UNKNOWN;
    srv.ViewDimension = D3D12_SRV_DIMENSION_BUFFER;
    srv.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
    srv.Buffer.FirstElement = 0;
    srv.Buffer.NumElements = static_cast<UINT>(waves.size());
    srv.Buffer.StructureByteStride = sizeof(ocean::GerstnerWaveGPU);
    srv.Buffer.Flags = D3D12_BUFFER_SRV_FLAG_NONE;

    g_device->CreateShaderResourceView(g_wavesBuffer.Get(), &srv, g_srvHeap->GetCPUDescriptorHandleForHeapStart());
    return true;
}

void UpdateFrameConstants(float dt) {
    g_timeSec += dt;

    const float aspect = static_cast<float>(kWidth) / static_cast<float>(kHeight);
    const float fovY = XMConvertToRadians(55.f);
    XMMATRIX proj = XMMatrixPerspectiveFovLH(fovY, aspect, kNearPlane, kFarPlane);

    const float orbitR = 35.f;
    const float camY = 6.f;
    const float ang = g_timeSec * 0.15f;
    const XMVECTOR eye = XMVectorSet(std::cos(ang) * orbitR, camY, std::sin(ang) * orbitR, 0.f);
    const XMVECTOR focus = XMVectorSet(0.f, 0.f, 0.f, 0.f);
    const float rollRad = kDemoRollAmplitudeRad * std::sin(g_timeSec * 0.75f);

    XMMATRIX viewNoRoll;
    XMMATRIX viewWithRoll;
    BuildCameraViewsWithRollSplit(eye, focus, rollRad, viewNoRoll, viewWithRoll);

    FrameConstants cb{};
    XMStoreFloat4x4(&cb.View, viewWithRoll);
    XMStoreFloat4x4(&cb.Proj, proj);
    XMStoreFloat4x4(&cb.ViewNoRoll, viewNoRoll);
    XMStoreFloat4x4(&cb.ViewWithRoll, viewWithRoll);
    XMStoreFloat4(&cb.CameraPos, eye);
    cb.Params = XMFLOAT4(kNearPlane, fovY * 0.5f, g_timeSec, kMaxTessN);
    cb.Params2 = XMFLOAT4(kNyquistK, static_cast<float>(kWaveCount), kWindU19_5, 0.f);

    std::memcpy(g_frameCBMapped[g_frameIndex], &cb, sizeof(FrameConstants));
}

void RenderFrame(float dt) {
    g_cmdAlloc[g_frameIndex]->Reset();
    g_cmdList->Reset(g_cmdAlloc[g_frameIndex].Get(), g_pso.Get());

    UpdateFrameConstants(dt);

    D3D12_RESOURCE_BARRIER toRT{};
    toRT.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
    toRT.Transition.pResource = g_renderTargets[g_frameIndex].Get();
    toRT.Transition.StateBefore = D3D12_RESOURCE_STATE_PRESENT;
    toRT.Transition.StateAfter = D3D12_RESOURCE_STATE_RENDER_TARGET;
    toRT.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
    g_cmdList->ResourceBarrier(1, &toRT);

    const float clearColor[] = {0.02f, 0.04f, 0.08f, 1.f};
    D3D12_CPU_DESCRIPTOR_HANDLE rtv = g_rtvHeap->GetCPUDescriptorHandleForHeapStart();
    rtv.ptr += static_cast<SIZE_T>(g_frameIndex) * g_rtvDescriptorSize;
    D3D12_CPU_DESCRIPTOR_HANDLE dsv = g_dsvHeap->GetCPUDescriptorHandleForHeapStart();
    g_cmdList->OMSetRenderTargets(1, &rtv, FALSE, &dsv);
    g_cmdList->ClearRenderTargetView(rtv, clearColor, 0, nullptr);
    g_cmdList->ClearDepthStencilView(dsv, D3D12_CLEAR_FLAG_DEPTH, 1.f, 0, 0, nullptr);

    D3D12_VIEWPORT vp{};
    vp.Width = static_cast<float>(kWidth);
    vp.Height = static_cast<float>(kHeight);
    vp.MinDepth = 0.f;
    vp.MaxDepth = 1.f;
    D3D12_RECT sr{};
    sr.left = 0;
    sr.top = 0;
    sr.right = static_cast<LONG>(kWidth);
    sr.bottom = static_cast<LONG>(kHeight);
    g_cmdList->RSSetViewports(1, &vp);
    g_cmdList->RSSetScissorRects(1, &sr);

    ID3D12DescriptorHeap* heaps[] = {g_srvHeap.Get()};
    g_cmdList->SetDescriptorHeaps(1, heaps);
    g_cmdList->SetGraphicsRootSignature(g_rootSig.Get());
    g_cmdList->SetPipelineState(g_pso.Get());

    const D3D12_GPU_VIRTUAL_ADDRESS cbAddr = g_frameCB[g_frameIndex]->GetGPUVirtualAddress();
    g_cmdList->SetGraphicsRootConstantBufferView(0, cbAddr);
    g_cmdList->SetGraphicsRootDescriptorTable(1, g_srvHeap->GetGPUDescriptorHandleForHeapStart());

    g_cmdList->IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
    g_cmdList->IASetVertexBuffers(0, 1, &g_vbv);
    g_cmdList->IASetIndexBuffer(&g_ibv);
    g_cmdList->DrawIndexedInstanced(g_indexCount, 1, 0, 0, 0);

    toRT.Transition.StateBefore = D3D12_RESOURCE_STATE_RENDER_TARGET;
    toRT.Transition.StateAfter = D3D12_RESOURCE_STATE_PRESENT;
    g_cmdList->ResourceBarrier(1, &toRT);

    g_cmdList->Close();
    ID3D12CommandList* lists[] = {g_cmdList.Get()};
    g_cmdQueue->ExecuteCommandLists(1, lists);

    g_swapChain->Present(1, 0);

    const UINT64 fv = g_fenceValue;
    g_cmdQueue->Signal(g_fence.Get(), fv);
    g_fenceValue++;

    g_frameIndex = g_swapChain->GetCurrentBackBufferIndex();
    if (g_fence->GetCompletedValue() < fv) {
        g_fence->SetEventOnCompletion(fv, g_fenceEvent);
        WaitForSingleObject(g_fenceEvent, INFINITE);
    }
}

void Shutdown() {
    WaitForGpu();
    for (UINT i = 0; i < kFrameCount; ++i) {
        if (g_frameCB[i]) {
            g_frameCB[i]->Unmap(0, nullptr);
            g_frameCBMapped[i] = nullptr;
        }
    }
    if (g_fenceEvent) {
        CloseHandle(g_fenceEvent);
        g_fenceEvent = nullptr;
    }
}

LRESULT CALLBACK WndProc(HWND hwnd, UINT msg, WPARAM wParam, LPARAM lParam) {
    switch (msg) {
    case WM_DESTROY:
        PostQuitMessage(0);
        return 0;
    case WM_KEYDOWN:
        if (wParam == VK_ESCAPE) {
            PostQuitMessage(0);
        }
        return 0;
    default:
        return DefWindowProcW(hwnd, msg, wParam, lParam);
    }
}

}  // namespace

int WINAPI wWinMain(HINSTANCE hInst, HINSTANCE, PWSTR, int) {
    WNDCLASSEXW wc{};
    wc.cbSize = sizeof(wc);
    wc.style = CS_HREDRAW | CS_VREDRAW;
    wc.lpfnWndProc = WndProc;
    wc.hInstance = hInst;
    wc.hCursor = LoadCursor(nullptr, IDC_ARROW);
    wc.lpszClassName = L"OceanDX12PhaseA";
    RegisterClassExW(&wc);

    RECT rc{0, 0, static_cast<LONG>(kWidth), static_cast<LONG>(kHeight)};
    AdjustWindowRect(&rc, WS_OVERLAPPEDWINDOW, FALSE);
    g_hwnd = CreateWindowExW(
        0,
        wc.lpszClassName,
        L"Ocean DX12 — Phase A",
        WS_OVERLAPPEDWINDOW,
        CW_USEDEFAULT,
        CW_USEDEFAULT,
        rc.right - rc.left,
        rc.bottom - rc.top,
        nullptr,
        nullptr,
        hInst,
        nullptr);

    if (!g_hwnd) {
        return -1;
    }

    if (!CreateDeviceAndSwapChain(g_hwnd)) {
        MessageBoxW(g_hwnd, L"D3D12 / swap chain init failed.", L"Error", MB_OK);
        return -1;
    }
    if (!CreateRootSignatureAndPSO()) {
        MessageBoxW(
            g_hwnd,
            L"Shader compile or PSO failed. Ensure shaders\\basic.hlsl is next to the exe.",
            L"Error",
            MB_OK);
        return -1;
    }
    if (!CreateSrvHeap()) {
        return -1;
    }

    std::vector<ocean::GerstnerWaveGPU> waves;
    ocean::GenerateWavesFromPiersonMoskowitz(waves, kWaveCount, kWindU19_5, 42);
    if (!CreateFrameConstantBuffers() || !CreateWavesBuffer(waves)) {
        MessageBoxW(g_hwnd, L"Constant buffer or wave upload failed.", L"Error", MB_OK);
        return -1;
    }

    UploadMesh();

    ShowWindow(g_hwnd, SW_SHOW);

    LARGE_INTEGER freq{};
    LARGE_INTEGER t0{};
    QueryPerformanceFrequency(&freq);
    QueryPerformanceCounter(&t0);

    MSG msg{};
    while (msg.message != WM_QUIT) {
        while (PeekMessageW(&msg, nullptr, 0, 0, PM_REMOVE)) {
            TranslateMessage(&msg);
            DispatchMessageW(&msg);
        }
        LARGE_INTEGER t1{};
        QueryPerformanceCounter(&t1);
        const float dt = static_cast<float>(t1.QuadPart - t0.QuadPart) / static_cast<float>(freq.QuadPart);
        t0 = t1;
        RenderFrame(dt);
    }

    Shutdown();
    return static_cast<int>(msg.wParam);
}
