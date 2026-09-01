cmake_minimum_required(VERSION 3.24)

project(yolov8_obb_cuda_ops LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 120)
endif()

add_executable(cuda_smoke
    tests/cuda_smoke.cu
)