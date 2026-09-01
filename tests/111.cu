#include<iostream>
#include<cuda_runtime.h>
#include<algorithm>
#include<vector>
#include<cstdlib>

#define CUDACHECK(expr)                                           \
    do{                                                           \
        cudaError_t error=(expr);                                 \
        if(error!=cudaSuccess)                                    \
        {                                                         \
            std::cerr<<"error is "<<cudaGetErrorString(error)     \
                <<"at"<<__FILE__<<":"<<__LINE__<<'\n';                  \
            std::exit(EXIT_FAILURE);                              \
        }                                                         \
    }while(0)                                                     

__global__ void vectoradd(const float* a,const float* b,float* c,int n)
{
    int index = blockDim.x * blockIdx.x + threadIdx.x;
    if(index < n)
    {
        c[index] = a[index] + b[index];
    }
}

int main()
{
    constexpr int n=1 << 20;
    std::vector<float> host_a(n,1.5f);
    std::vector<float> host_b(n,2.5f);
    std::vector<float> host_c(n,0.0f);
    const std::size_t bytes=n*sizeof(float);

    float* device_a=nullptr;
    float* device_b=nullptr;
    float* device_c=nullptr;

    CUDACHECK(cudaMalloc(&device_a,bytes));
    CUDACHECK(cudaMalloc(&device_b,bytes));
    CUDACHECK(cudaMalloc(&device_c,bytes));

    CUDACHECK(cudaMemcpy(device_a,host_a.data(),bytes,cudaMemcpyHostToDevice));
    CUDACHECK(cudaMemcpy(device_b,host_b.data(),bytes,cudaMemcpyHostToDevice));

    constexpr int threads = 256;
    const int blocks = (n + threads - 1) / threads;

    vectoradd<<<blocks , threads>>>(device_a, device_b, device_c, n);

    CUDACHECK(cudaGetLastError());
    CUDACHECK(cudaDeviceSynchronize());

    CUDACHECK(cudaMemcpy(host_c.data(),device_c,bytes,cudaMemcpyDeviceToHost));

    float max_error=0.0f;
    for(int i=0;i<n;i++)
    {
        max_error=std::max(std::abs(host_c[i]-4.0f),max_error);
    }
    std::cout<<"max_error = "<<max_error<<'\n';

    CUDACHECK(cudaFree(device_a));
    CUDACHECK(cudaFree(device_b));
    CUDACHECK(cudaFree(device_c));

    return max_error==0.0f ? EXIT_SUCCESS : EXIT_FAILURE;
}