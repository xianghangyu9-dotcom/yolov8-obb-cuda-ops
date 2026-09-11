#include "fused_obb_decode_v0.h"
#include <cmath>
#include <cfloat>
#include <cstddef>

namespace yolo_obb{
namespace{

constexpr float Pi = 3.14159265358979323846f;

__device__ __forceinline__
float loadNCHW(
    const float *raw,
    int batch_index,
    int channel_index,
    int spatial_index,
    int channels,
    int spatial_size){

    const std::size_t offset=(
        (
            static_cast<std::size_t>(batch_index)
            * channels 
            + channel_index 
        )
        * spatial_size
        + spatial_index
    );

    return raw[offset];
}

__device__ __forceinline__
float StableSigmoid(float value){
    if(value >= 0.0f){
        const float exp_value = expf(-value);
        return 1.0f/(1.0+exp_value);
    }
    const float exp_value = expf(value);
    return exp_value/(1.0+exp_value);
}

__global__
void FusedObbDecodeV0Kernel(
    ObbLevelDesc level,
    int batch_size,
    int channels,
    int num_classes,
    int reg_max,
    float conf_threshold,
    ObbDecodeOutput output
){
    const int spatial_index = blockDim.x * blockIdx.x + threadIdx.x;
    const int spatial_size = level.height * level.width;
    const int batch_index = blockIdx.y;
    
    if( spatial_index >= spatial_size || batch_index >= batch_size ){
        return;
    }

    //1、DFL,先找到最大值
    float distance[4];

    for(int i = 0; i < 4; i++){
        const int channel_begin = i * reg_max;
        float max_logit = -FLT_MAX;

        for(int reg_index = 0; reg_index < reg_max; reg_index++){
            const float logit = loadNCHW(
                level.data,
                batch_index,
                channel_begin + reg_index,
                spatial_index,
                channels,
                spatial_size
            );

            max_logit = max(max_logit , logit);
        }

        //2、二次循环，使用sigmoid计算概率
        float exp_sum = 0.0f;
        float weighted_sum = 0.0f;

        for(int reg_index = 0; reg_index < reg_max; reg_index++){
            const float logit = loadNCHW(
                level.data,
                batch_index,
                channel_begin + reg_index,
                spatial_index,
                channels,
                spatial_size
            );
            const float exp_value = expf(logit - max_logit);

            exp_sum += exp_value;
            weighted_sum += (
                exp_value * static_cast<float>(reg_index)
            );
        }
        distance[i] = weighted_sum / exp_sum;
    }
    float best_conf = -1.0f;
    int best_class = 0;
    const int class_begin = 4 * reg_max;

    //3、类别解码
    for(int class_index = 0; class_index < num_classes; class_index++)
    {
        const float class_logit = loadNCHW(
            level.data,
            batch_index,
            class_begin + class_index,
            spatial_index,
            channels,
            spatial_size
        );

        float class_conf = StableSigmoid(class_logit);

        if(class_conf > best_conf){
            best_conf = max(class_conf, best_conf);
            best_class = class_index;
        }
    }
    //4、置信度过滤
    if(!(best_conf > conf_threshold)) return;

    //5、角度解码
    const int angle_begin = class_begin + num_classes;

    float angle_logit = loadNCHW(
        level.data,
        batch_index,
        angle_begin,
        spatial_index,
        channels,
        spatial_size
    );
    
    const float angle = (StableSigmoid(angle_logit)-0.25f) * Pi;

    //6、还原为二维的位置坐标
    const int y = spatial_index / level.width;
    const int x = spatial_index % level.width;

    const float anchor_x = static_cast<float>(x) + 0.5f;
    const float anchor_y = static_cast<float>(y) + 0.5f;

    const float left   = distance[0];
    const float top    = distance[1];
    const float right  = distance[2];
    const float bottom = distance[3];

    const float x_offset = (right  - left) * 0.5f;
    const float y_offset = (bottom - top ) * 0.5f;

    const float cos_angle = cos(angle);
    const float sin_angle = sin(angle);

    const float rotate_x = x_offset * cos_angle - y_offset * sin_angle;
    const float rotate_y = x_offset * sin_angle + y_offset * cos_angle;
    
    const float stride   = static_cast<float>(level.stride);
    const float center_x = (anchor_x + rotate_x) * stride;
    const float center_y = (anchor_y + rotate_y) * stride;
    const float height   = (top + bottom) * stride;
    const float width    = (left + right) * stride;

    //申请输出位置，有效框相邻输出，不在原位置
    const int output_slot = atomicAdd(
        output.counts + batch_index,
        1);

    if (output_slot >= output.capacity_per_batch) {
        atomicExch(
            output.overflow + batch_index,
            1);

        return;
    }

    const std::size_t output_index =
        static_cast<std::size_t>(batch_index)
        * output.capacity_per_batch
        + output_slot;

    float* output_box =
        output.boxes + output_index * 5;

    output_box[0] = center_x;
    output_box[1] = center_y;
    output_box[2] = width;
    output_box[3] = height;
    output_box[4] = angle;

    output.scores[output_index] = best_conf;
    output.labels[output_index] = best_class;

    output.indices[output_index] =
    level.level_offset + spatial_index; 
}

} // namespace

cudaError_t LaunchFusedObbDecodeV0(
    const ObbLevelDesc* levels,
    int num_levels,
    int batch_size,
    int channels,
    int num_classes,
    int reg_max,
    float conf_threshold,
    const ObbDecodeOutput& output,
    cudaStream_t stream) {

    // Host 端参数检查。
     
    if (
        levels == nullptr
        || num_levels <= 0
        || batch_size <= 0
        || num_classes <= 0
        || reg_max <= 0
        || output.capacity_per_batch <= 0
        || output.boxes == nullptr
        || output.scores == nullptr
        || output.labels == nullptr
        || output.indices == nullptr
        || output.counts == nullptr
        || output.overflow == nullptr
    ) {
        return cudaErrorInvalidValue;
    }

    const int expected_channels =
        4 * reg_max + num_classes + 1;

    if (channels != expected_channels) {
        return cudaErrorInvalidValue;
    }

    for (
        int level_index = 0;
        level_index < num_levels;
        ++level_index
    ) {
        const ObbLevelDesc& level =
            levels[level_index];

        if (
            level.data == nullptr
            || level.height <= 0
            || level.width <= 0
            || level.stride <= 0
            || level.level_offset < 0
        ) {
            return cudaErrorInvalidValue;
        }
    }
    // 初始化输出
    const std::size_t total_slots =
        static_cast<std::size_t>(batch_size)
        * output.capacity_per_batch;

    cudaError_t error = cudaSuccess;

    error = cudaMemsetAsync(
        output.boxes,
        0,
        total_slots * 5 * sizeof(float),
        stream);

    if (error != cudaSuccess) {
        return error;
    }

    error = cudaMemsetAsync(
        output.scores,
        0,
        total_slots * sizeof(float),
        stream);

    if (error != cudaSuccess) {
        return error;
    }
    // int 的全部字节设为 0xFF，即 -1。
    error = cudaMemsetAsync(
        output.labels,
        0xFF,
        total_slots * sizeof(int),
        stream);

    if (error != cudaSuccess) {
        return error;
    }

    error = cudaMemsetAsync(
        output.indices,
        0xFF,
        total_slots * sizeof(int),
        stream);

    if (error != cudaSuccess) {
        return error;
    }

    error = cudaMemsetAsync(
        output.counts,
        0,
        batch_size * sizeof(int),
        stream);

    if (error != cudaSuccess) {
        return error;
    }

    error = cudaMemsetAsync(
        output.overflow,
        0,
        batch_size * sizeof(int),
        stream);

    if (error != cudaSuccess) {
        return error;
    }

    constexpr int threads_per_block = 256;

    for (
        int level_index = 0;
        level_index < num_levels;
        ++level_index
    ) {
        const ObbLevelDesc level =
            levels[level_index];

        const int spatial_size =
            level.height * level.width;

        const int blocks_x =
            (
                spatial_size
                + threads_per_block
                - 1
            )
            / threads_per_block;

        const dim3 block(
            threads_per_block,
            1,
            1);

        const dim3 grid(
            blocks_x,
            batch_size,
            1);

        FusedObbDecodeV0Kernel<<<
            grid,
            block,
            0,
            stream
        >>>(
            level,
            batch_size,
            channels,
            num_classes,
            reg_max,
            conf_threshold,
            output);

        error = cudaGetLastError();

        if (error != cudaSuccess) {
            return error;
        }
    }

    return cudaSuccess;

    }
} // namespace yolo_obb