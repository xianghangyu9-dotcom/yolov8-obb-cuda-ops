# FusedObbDecode 算子规格

## 1. 输入

输入为三个连续 NCHW Tensor：

- P3: [B, 4R+C+1, H3, W3]
- P4: [B, 4R+C+1, H4, W4]
- P5: [B, 4R+C+1, H5, W5]

支持输入类型：

- FP32
- FP16

P0 只要求：

- B=1
- R=16
- 输入尺寸 640×640
- strides=[8,16,32]

## 2. 通道布局

- [0, 4R): DFL Box Logits
- [4R, 4R+C): Class Logits
- [4R+C, 4R+C+1): Angle Logit

Box Logit 顺序：

- left:   [0, R)
- top:    [R, 2R)
- right:  [2R, 3R)
- bottom: [3R, 4R)

## 3. 位置顺序

每层内部：

global_index = level_offset + y * width + x

层顺序：

P3 → P4 → P5

## 4. 数值语义

- DFL 使用 FP32 Stable Softmax；
- DFL bin 值为 0～R-1；
- FP16 输入使用 FP32 累加；
- Class Score = sigmoid(class_logit)；
- 每个位置只保留最大类别；
- 类别分数相同则保留较小类别编号；
- Angle = (sigmoid(angle_logit)-0.25)*pi；
- Angle 单位为弧度；
- 输出 xywh 为 640×640 网络输入空间的像素坐标；
- 本算子不执行逆 Letterbox；
- 本算子不裁剪越界框。

## 5. 候选筛选

- 使用 score > conf_threshold；
- 默认 conf_threshold=0.25；
- 按 score 降序稳定排序；
- Score 相同时，原始 global_index 较小者优先；
- 默认 pre_topk=1024。

## 6. Fast-NMS

- 使用 ProbIoU；
- 只抑制相同类别；
- 使用 Fast-NMS，而不是 Greedy NMS；
- ProbIoU >= iou_threshold 时抑制；
- 默认 iou_threshold=0.45；
- 最终最多保留 max_det=300。

## 7. 输出

Decode/PreTopK：

- boxes:   [B, pre_topk, 5], FP32
- scores:  [B, pre_topk], FP32
- labels:  [B, pre_topk], INT32
- indices: [B, pre_topk], INT32
- count:   [B], INT32

NMS：

- boxes:   [B, max_det, 5], FP32
- scores:  [B, max_det], FP32
- labels:  [B, max_det], INT32
- indices: [B, max_det], INT32
- count:   [B], INT32

有效数据位于 [0,count)。

填充值：

- boxes=0
- scores=0
- labels=-1
- indices=-1

## 8. 非法输入

要求输入：

- Shape 合法；
- NCHW 连续；
- 三层 Batch 和通道数一致；
- 不含 NaN 和 Inf。

Python Reference 遇到非法输入时抛出异常。