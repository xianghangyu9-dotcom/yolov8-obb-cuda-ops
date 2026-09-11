#include"fused_obb_decode_v0.h"

#include <cuda_runtime.h>

#include <fstream>
#include <iostream>
#include <string>
#include <vector>

using namespace std;

#define CUDA_CHECK(expr)                    \
    do{                                     \
        cudaError_t error =  (expr);        \
        if(error != cudaSuccess){           \
            cerr<<"cuda error is "           \
            <<cudaGetErrorString(error)     \
            <<" on "                          \
            <<__FILE__                      \
            <<" : "                           \
            <<__LINE__                      \
            <<endl;                         \
            return 1;                       \
        }                                   \
    }while(0)

bool ReadFloatFile(
    const std::string& path,
    std::vector<float>& data) {

    std::ifstream file(
        path,
        std::ios::binary);

    if (!file.is_open()) {
        std::cerr
            << "Cannot open file: "
            << path
            << std::endl;

        return false;
    }

    file.read(
        reinterpret_cast<char*>(data.data()),
        data.size() * sizeof(float));

    if (!file) {
        std::cerr
            << "Cannot read enough data from: "
            << path
            << std::endl;

        return false;
    }

    return true;
}

bool WriteFloatFile(
    const std::string& path,
    const std::vector<float>& data) {

    std::ofstream file(
        path,
        std::ios::binary);

    if (!file.is_open()) {
        std::cerr
            << "Cannot create file: "
            << path
            << std::endl;

        return false;
    }

    file.write(
        reinterpret_cast<const char*>(data.data()),
        data.size() * sizeof(float));

    return static_cast<bool>(file);
}

bool WriteIntFile(
    const std::string& path,
    const std::vector<int>& data) {

    std::ofstream file(
        path,
        std::ios::binary);

    if (!file.is_open()) {
        std::cerr
            << "Cannot create file: "
            << path
            << std::endl;

        return false;
    }

    file.write(
        reinterpret_cast<const char*>(data.data()),
        data.size() * sizeof(int));

    return static_cast<bool>(file);
}

int main(
    int argc,
    char** argv) {
    std::string case_directory =
        "artifacts/cuda_v0/real";

    if (argc >= 2) {
        case_directory = argv[1];
    }

    std::cout
        << "Case directory: "
        << case_directory
        << std::endl;
    
    constexpr int BatchSize = 1;
    constexpr int Regmax = 16;
    constexpr int NumClasses = 15;

    constexpr int Channels = 4 * Regmax + NumClasses + 1;

    constexpr int p3_height = 80;
    constexpr int p3_width  = 80;
    constexpr int p3_stride = 8;

    constexpr int p4_height = 40;
    constexpr int p4_width  = 40;
    constexpr int P4_stride = 16;

    constexpr int p5_height = 20;
    constexpr int p5_width  = 20;
    constexpr int p5_stride = 32;

    constexpr int p3_position = p3_height * p3_width;
    constexpr int p4_position = p4_height * p4_width;
    constexpr int p5_position = p5_height * p5_width;
    
    constexpr int Capacity = p3_position + p4_position + p5_position;
    
    constexpr float conf_threshold = 0.25f;

    const int p3_element_count =
        BatchSize
        * Channels
        * p3_height
        * p3_width;

    const int p4_element_count =
        BatchSize
        * Channels
        * p4_height
        * p4_width;

    const int p5_element_count =
        BatchSize
        * Channels
        * p5_height
        * p5_width;
    
     std::vector<float> host_p3(
        p3_element_count);

    std::vector<float> host_p4(
        p4_element_count);

    std::vector<float> host_p5(
        p5_element_count);
    
    if (!ReadFloatFile(
            case_directory + "/p3.bin",
            host_p3)) {
        return 1;
    }

    if (!ReadFloatFile(
            case_directory + "/p4.bin",
            host_p4)) {
        return 1;
    }

    if (!ReadFloatFile(
            case_directory + "/p5.bin",
            host_p5)) {
        return 1;
    }

    std::cout
        << "Input files loaded successfully"
        << std::endl;
    
    float* device_p3 = nullptr;
    float* device_p4 = nullptr;
    float* device_p5 = nullptr;

    float* device_boxes = nullptr;
    float* device_scores = nullptr;

    int* device_labels = nullptr;
    int* device_indices = nullptr;
    int* device_count = nullptr;
    int* device_overflow = nullptr;

    //输入
    CUDA_CHECK(
        cudaMalloc(reinterpret_cast<void**>(&device_p3),
        p3_element_count * sizeof(float)));
    CUDA_CHECK(
        cudaMalloc(reinterpret_cast<void**>(&device_p4),
        p4_element_count * sizeof(float)));
    CUDA_CHECK(
        cudaMalloc(reinterpret_cast<void**>(&device_p5),
        p5_element_count * sizeof(float)));
    
    //输出
    CUDA_CHECK(
        cudaMalloc(reinterpret_cast<void**>(&device_boxes),
        Capacity * 5 * sizeof(float)));
    CUDA_CHECK(
        cudaMalloc(reinterpret_cast<void**>(&device_labels),
        Capacity * sizeof(int)));
    CUDA_CHECK(
        cudaMalloc(reinterpret_cast<void**>(&device_scores),
        Capacity * sizeof(float)));
    CUDA_CHECK(
        cudaMalloc(reinterpret_cast<void**>(&device_count),
        sizeof(int)));
    CUDA_CHECK(
        cudaMalloc(reinterpret_cast<void**>(&device_indices),
        Capacity * sizeof(int)));
    CUDA_CHECK(
        cudaMalloc(reinterpret_cast<void**>(&device_overflow),
        sizeof(int)));
    
    //CPU数据搬运至GPU
    CUDA_CHECK(
        cudaMemcpy(
            device_p3,
            host_p3.data(),
            p3_element_count * sizeof(float),
            cudaMemcpyHostToDevice
        )
    );
    CUDA_CHECK(
        cudaMemcpy(
            device_p4,
            host_p4.data(),
            p4_element_count * sizeof(float),
            cudaMemcpyHostToDevice
        )
    );
    CUDA_CHECK(
        cudaMemcpy(
            device_p5,
            host_p5.data(),
            p5_element_count * sizeof(float),
            cudaMemcpyHostToDevice
        )
    );
    
    std::cout
        << "Input copied from CPU to GPU"
        <<std::endl;
    
    //三个特征层
    yolo_obb::ObbLevelDesc levels[3];

    levels[0].data = device_p3;
    levels[0].height = p3_height;
    levels[0].width = p3_width;
    levels[0].stride = 8;
    levels[0].level_offset = 0;

    levels[1].data = device_p4;
    levels[1].height = p4_height;
    levels[1].width = p4_width;
    levels[1].stride = 16;
    levels[1].level_offset = p3_position;

    levels[2].data = device_p5;
    levels[2].height = p5_height;
    levels[2].width = p5_width;
    levels[2].stride = 32;
    levels[2].level_offset = p3_position + p4_position;

    yolo_obb::ObbDecodeOutput output;

    output.boxes = device_boxes;
    output.scores = device_scores;
    output.labels = device_labels;
    output.indices = device_indices;
    output.counts = device_count;
    output.overflow = device_overflow;
    output.capacity_per_batch = Capacity;
    
    CUDA_CHECK(
        yolo_obb::LaunchFusedObbDecodeV0(
            levels,
            3,
            BatchSize,
            Channels,
            NumClasses,
            Regmax,
            conf_threshold,
            output,
            nullptr));
    
    CUDA_CHECK(
        cudaDeviceSynchronize());
    
    std::cout
        << "CUDA Kernel completed"
        << std::endl;
    
    int host_count = 0;
    int host_overflow = 0;

    CUDA_CHECK(
        cudaMemcpy(
            &host_count,
            output.counts,
            sizeof(int),
            cudaMemcpyDeviceToHost));
    CUDA_CHECK(
        cudaMemcpy(
            &host_overflow,
            output.overflow,
            sizeof(int),
            cudaMemcpyDeviceToHost));

    std::cout
        << "Candidate count: "
        << host_count
        << std::endl;

    std::cout
        << "Overflow: "
        << host_overflow
        << std::endl;
    
    if(host_overflow != 0){
        std::cerr
            <<"Capacity is not enough"
            <<std::endl;
        return 1;
    }
    
    if(host_count > Capacity){
        std::cerr
            <<"count is invalid"
            <<std::endl;
        return 1;
    }

    std::vector<float> host_boxes (5 * Capacity);
    std::vector<float> host_scores (Capacity);
    std::vector<int> host_labels (Capacity);
    std::vector<int> host_indices (Capacity);

    CUDA_CHECK(
        cudaMemcpy(
            host_boxes.data(),
            output.boxes,
            5 * Capacity * sizeof(float),
            cudaMemcpyDeviceToHost
        ));
    CUDA_CHECK(
        cudaMemcpy(
            host_scores.data(),
            output.scores,
            Capacity * sizeof(float),
            cudaMemcpyDeviceToHost
        ));
    CUDA_CHECK(
        cudaMemcpy(
            host_labels.data(),
            output.labels,
            Capacity * sizeof(int),
            cudaMemcpyDeviceToHost
        ));
    CUDA_CHECK(
        cudaMemcpy(
            host_indices.data(),
            output.indices,
            Capacity * sizeof(int),
            cudaMemcpyDeviceToHost
        ));
    
        if (!WriteFloatFile(
            case_directory + "/cuda_boxes.bin",
            host_boxes)) {
        return 1;
    }

    //保存输出
    if (!WriteFloatFile(
            case_directory + "/cuda_scores.bin",
            host_scores)) {
        return 1;
    }
    
    if (!WriteIntFile(
            case_directory + "/cuda_labels.bin",
            host_labels)) {
        return 1;
    }

    if (!WriteIntFile(
            case_directory + "/cuda_indices.bin",
            host_indices)) {
        return 1;
    }

    if (!WriteIntFile(
            case_directory + "/cuda_count.bin",
            std::vector<int>{host_count})) {
        return 1;
    }

    if (!WriteIntFile(
            case_directory + "/cuda_overflow.bin",
            std::vector<int>{host_overflow})) {
        return 1;
    }

    int preview_num = 3 < host_count ? 3 : host_count;
    for(int i = 0; i < preview_num; i++){
        std::cout
            <<"num : "<<i
            <<" position is "<<host_indices[i]
            <<" label is "<<host_labels[i]
            <<" conf is "<<host_scores[i]
            <<" boxes is "
            <<"( "
            <<host_boxes[i * 5 + 0]<<", "
            <<host_boxes[i * 5 + 1]<<", "
            <<host_boxes[i * 5 + 2]<<", "
            <<host_boxes[i * 5 + 3]<<", "
            <<host_boxes[i * 5 + 4]<<", "
            <<" )"<<std::endl;
    };
    
    CUDA_CHECK(cudaFree(device_p3));
    CUDA_CHECK(cudaFree(device_p4));
    CUDA_CHECK(cudaFree(device_p5));

    CUDA_CHECK(cudaFree(device_boxes));
    CUDA_CHECK(cudaFree(device_labels));
    CUDA_CHECK(cudaFree(device_scores));
    CUDA_CHECK(cudaFree(device_count));
    CUDA_CHECK(cudaFree(device_indices));
    CUDA_CHECK(cudaFree(device_overflow));

    std::cout<< "run_decode_v0 is successful"<<std::endl;
    
    return 0;
}
