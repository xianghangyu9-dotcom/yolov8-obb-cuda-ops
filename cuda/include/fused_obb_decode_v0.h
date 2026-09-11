#pragma once

#include <cuda_runtime.h>

namespace yolo_obb {

struct ObbLevelDesc {
    // 指向 GPU 上的 NCHW Raw Tensor。
    const float* data = nullptr;

    int height = 0;
    int width = 0;

    // 当前特征层相对于网络输入的步长。
    int stride = 0;

    // 当前层在全局 8400 个位置中的起始索引。
    int level_offset = 0;
};

struct ObbDecodeOutput {
    // [batch, capacity_per_batch, 5]
    float* boxes = nullptr;

    // [batch, capacity_per_batch]
    float* scores = nullptr;

    // [batch, capacity_per_batch]
    int* labels = nullptr;

    // 原始 P3/P4/P5 全局索引
    int* indices = nullptr;

    // [batch]
    int* counts = nullptr;

    // [batch]
    // 输出容量不足时置为 1
    int* overflow = nullptr;

    int capacity_per_batch = 0;
};

cudaError_t LaunchFusedObbDecodeV0(
    const ObbLevelDesc* levels,
    int num_levels,
    int batch_size,
    int channels,
    int num_classes,
    int reg_max,
    float conf_threshold,
    const ObbDecodeOutput& output,
    cudaStream_t stream);

}  // namespace yolo_obb