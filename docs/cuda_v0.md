# FusedObbDecode CUDA V0

## 目标

建立正确性优先的 CUDA 基线。

## 输入

P3/P4/P5 FP32 NCHW Raw Tensor。

## 线程映射

- 一个 Thread 处理一个候选位置；
- blockDim.x=256；
- grid.x 覆盖当前特征层 H×W；
- grid.y 对应 Batch；
- P3/P4/P5 各启动一次 Kernel。

## 单线程工作

1. 读取 4×16 个 DFL Logit；
2. Stable Softmax；
3. 计算 l,t,r,b 期望；
4. 遍历类别并计算最大 Score；
5. 执行严格大于阈值判断；
6. 解码 Angle；
7. 执行 dist2rbox；
8. atomicAdd 分配输出槽位；
9. 写入 box/score/label/index。

## 输出顺序

atomicAdd 输出顺序不稳定。
正确性比较前按照原始 index 排序。

## 当前限制

- 仅支持 FP32；
- P0 只测试 Batch=1；
- reg_max=16；
- 不包含排序、TopK 和 NMS；
- 未使用 Warp Ballot；
- 未使用 Shared Memory；
- 未使用 Fast Math。

## 正确性结果

run_decode_v0与check_cuda_v0运行结果如下:（只打印前三个有效结果）

real:

Candidate count: 12
Overflow: 0
num : 0 position is 3609 label is 10 conf is 0.265886 boxes is ( 81.6692, 361.152, 8.55681, 18.2118, 1.40745,  )
num : 1 position is 3419 label is 10 conf is 0.409638 boxes is ( 478.671, 340.783, 17.2348, 8.78945, 0.0250854,  )
num : 2 position is 3420 label is 10 conf is 0.514297 boxes is ( 478.597, 340.921, 16.8894, 8.52011, 0.0193206,  )

box max abs error: 3.0517578125e-05
box mean abs error: 9.6075234523596e-07
score max abs error: 2.9802322387695312e-08
count: 12
overflow: 0
expected_count: 12
cuda_count: 12

all_zero_dfl:

Candidate count: 8400
Overflow: 0
num : 0 position is 3840 label is 0 conf is 1 boxes is ( 4, 388, 120, 120, 0.785398,  )
num : 1 position is 3841 label is 0 conf is 1 boxes is ( 12, 388, 120, 120, 0.785398,  )
num : 2 position is 3842 label is 0 conf is 1 boxes is ( 20, 388, 120, 120, 0.785398,  )

box max abs error: 0.0
box mean abs error: 0.0
score max abs error: 0.0
count: 8400
overflow: 0
expected_count: 8400
cuda_count: 8400

none:

Candidate count: 0
Overflow: 0

count: 0
overflow: 0
CUDA V0 check passed
case_dir: /home/xhy/projects/yolo_obb-cuda/artifacts/cuda_v0/none
expected_count: 0
cuda_count: 0

extrme:

Candidate count: 8400
Overflow: 0
num : 0 position is 5696 label is 0 conf is 1 boxes is ( 132, 572, 238.805, 238.805, 2.35619,  )
num : 1 position is 5697 label is 0 conf is 1 boxes is ( 140, 572, 238.805, 238.805, 2.35619,  )
num : 2 position is 5698 label is 0 conf is 1 boxes is ( 148, 572, 238.805, 238.805, 2.35619,  )

box max abs error: 6.103515625e-05
box mean abs error: 8.138021257764194e-06
score max abs error: 0.0
count: 8400
overflow: 0
expected_count: 8400
cuda_count: 8400

## 性能

real：

V0 total: 74.2369 ms
V0 average: 0.0742369 ms / inference

all_zero_dfl:

V0 total: 70.804 ms
V0 average: 0.070804 ms / inference

none:

V0 total: 73.2632 ms
V0 average: 0.0732632 ms / inference

extreme:

V0 total: 69.9323 ms
V0 average: 0.0699323 ms / inference