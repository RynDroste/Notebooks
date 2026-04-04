# VS 2026: CMake >= 4.2, e.g. cmake -B build -G "Visual Studio 18 2026" -A x64
# Or: cmake --preset vs2026-x64 && cmake --build --preset release
cmake_minimum_required(VERSION 3.20)
project(ocean_dx12 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_executable(ocean_dx12 WIN32
    src/main.cpp
    src/wave_spectrum.cpp
)

target_include_directories(ocean_dx12 PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/src")

target_link_libraries(ocean_dx12 PRIVATE
    d3d12
    dxgi
    d3dcompiler
    user32
    gdi32
)

if(MSVC)
    target_compile_options(ocean_dx12 PRIVATE /utf-8)
endif()

# Copy shaders next to the exe for D3DCompileFromFile at runtime.
add_custom_command(TARGET ocean_dx12 POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_directory
        "${CMAKE_CURRENT_SOURCE_DIR}/shaders"
        "$<TARGET_FILE_DIR:ocean_dx12>/shaders"
    VERBATIM
)
