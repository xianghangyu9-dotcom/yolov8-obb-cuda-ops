cmake_minimum_required(VERSION 3.24)

project(yolov8_obb_cuda_ops LANGUAGES CXX CUDA)

find_package(CUDAToolkit REQUIRED)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 120)
endif()

add_executable(cuda_smoke
    tests/cuda_smoke.cu
)

add_library(yolo_obb_cuda_v0 STATIC
    cuda/src/fused_obb_decode_v0.cu
)

target_include_directories(yolo_obb_cuda_v0
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/cuda/include
)

target_link_libraries(yolo_obb_cuda_v0
    PUBLIC
        CUDA::cudart
)

target_compile_options(yolo_obb_cuda_v0
    PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo>
)

set_target_properties(yolo_obb_cuda_v0
    PROPERTIES
        POSITION_INDEPENDENT_CODE ON
)

add_executable(run_decode_v0
    cuda/tests/run_decode_v0.cpp
)

target_link_libraries(run_decode_v0
    PRIVATE
        yolo_obb_cuda_v0
        CUDA::cudart
)

add_executable(benchmark_decode_v0
    cuda/tests/benchmark_decode_v0.cpp
)

target_link_libraries(benchmark_decode_v0
    PRIVATE
        yolo_obb_cuda_v0
        CUDA::cudart
)