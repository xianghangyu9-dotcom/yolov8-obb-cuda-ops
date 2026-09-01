#include <cuda_runtime.h>

#include <algorithm>
#include <cstdlib>
#include <iostream>
#include <vector>

#define CUDA_CHECK(call)                                                     \
    do {                                                                     \
        cudaError_t error = (call);                                          \
        if (error != cudaSuccess) {                                          \
            std::cerr << "CUDA error: " << cudaGetErrorString(error)         \
                      << " at " << __FILE__ << ":" << __LINE__ << '\n';      \
            std::exit(EXIT_FAILURE);                                         \
        }                                                                    \
    } while (0)

__global__ void vector_add(
    const float* a,
    const float* b,
    float* c,
    int n) {

    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index < n) {
        c[index] = a[index] + b[index];
    }
}

int main() {
    constexpr int n = 1 << 20;
    const std::size_t bytes = n * sizeof(float);

    std::vector<float> host_a(n, 1.5f);
    std::vector<float> host_b(n, 2.5f);
    std::vector<float> host_c(n, 0.0f);

    float* device_a = nullptr;
    float* device_b = nullptr;
    float* device_c = nullptr;

    CUDA_CHECK(cudaMalloc(&device_a, bytes));
    CUDA_CHECK(cudaMalloc(&device_b, bytes));
    CUDA_CHECK(cudaMalloc(&device_c, bytes));

    CUDA_CHECK(cudaMemcpy(
        device_a, host_a.data(), bytes, cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaMemcpy(
        device_b, host_b.data(), bytes, cudaMemcpyHostToDevice));

    constexpr int threads = 256;
    const int blocks = (n + threads - 1) / threads;

    vector_add<<<blocks, threads>>>(device_a, device_b, device_c, n);

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(
        host_c.data(), device_c, bytes, cudaMemcpyDeviceToHost));

    float max_error = 0.0f;

    for (int i = 0; i < n; ++i) {
        max_error = std::max(
            max_error,
            std::abs(host_c[i] - 4.0f));
    }

    std::cout << "max_error = " << max_error << '\n';

    CUDA_CHECK(cudaFree(device_a));
    CUDA_CHECK(cudaFree(device_b));
    CUDA_CHECK(cudaFree(device_c));

    return max_error == 0.0f ? EXIT_SUCCESS : EXIT_FAILURE;
}