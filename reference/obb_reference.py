from __future__ import annotations

from dataclasses import dataclass

import torch

from reference.obb_math import (
    decode_angle,
    decode_xywhr,
    dfl_expectation,
    fast_nms_sorted,
    make_anchor_points,
)


@dataclass(frozen=True)
class OperatorConfig:
    reg_max: int = 16
    strides: tuple[int, ...] = (8, 16, 32)

    conf_threshold: float = 0.25
    iou_threshold: float = 0.45

    pre_topk: int = 1024
    max_det: int = 300


def flatten_raw_levels(
    raw_levels: list[torch.Tensor],
) -> tuple[torch.Tensor, list[tuple[int, int]]]:
    """
    [B,D,H,W] × L
        ↓
    [B,N,D]
    """
    if not raw_levels:
        raise ValueError("raw_levels 不能为空")

    batch_size = raw_levels[0].shape[0]
    channel_count = raw_levels[0].shape[1]

    flattened_levels = []
    level_shapes = []

    for level_index, raw in enumerate(raw_levels):
        if raw.ndim != 4:
            raise ValueError(
                f"Level {level_index} 不是 NCHW Tensor"
            )

        batch, channels, height, width = raw.shape

        if batch != batch_size:
            raise ValueError("各层 Batch 必须一致")

        if channels != channel_count:
            raise ValueError("各层通道数必须一致")

        if not torch.isfinite(raw).all():
            raise ValueError(
                f"Level {level_index} 存在 NaN 或 Inf"
            )

        level_flat = (
            raw.permute(0, 2, 3, 1)
            .reshape(batch, height * width, channels)
        )

        flattened_levels.append(level_flat)
        level_shapes.append((height, width))

    return (
        torch.cat(flattened_levels, dim=1),
        level_shapes,
    )


def decode_dense(
    raw_levels: list[torch.Tensor],
    config: OperatorConfig,
) -> dict[str, torch.Tensor]:
    """
    解码所有位置，但暂不做阈值过滤。

    输出数量仍然是 8400。
    """
    flat_raw, level_shapes = flatten_raw_levels(
        raw_levels
    )

    # FP16 输入也统一转成 FP32 参考计算。
    flat_raw = flat_raw.float()

    batch, position_count, raw_channels = (
        flat_raw.shape
    )

    box_channels = 4 * config.reg_max
    angle_channels = 1

    class_count = (
        raw_channels
        - box_channels
        - angle_channels
    )

    if class_count <= 0:
        raise ValueError(
            f"无法推断类别数：raw_channels={raw_channels}"
        )

    box_logits = flat_raw[
        ..., :box_channels
    ]

    class_logits = flat_raw[
        ...,
        box_channels:box_channels + class_count,
    ]

    angle_logits = flat_raw[
        ...,
        box_channels + class_count:,
    ]

    distances = dfl_expectation(
        box_logits,
        config.reg_max,
    )

    angles = decode_angle(angle_logits)

    anchor_points, stride_tensor = (
        make_anchor_points(
            raw_levels,
            config.strides,
        )
    )

    if anchor_points.shape[0] != position_count:
        raise RuntimeError(
            "Anchor 数量与预测位置数不一致"
        )

    boxes = decode_xywhr(
        distances,
        angles,
        anchor_points,
        stride_tensor,
    )

    class_probabilities = torch.sigmoid(
        class_logits
    )

    scores, labels = class_probabilities.max(
        dim=-1
    )

    original_indices = torch.arange(
        position_count,
        device=flat_raw.device,
        dtype=torch.int64,
    ).unsqueeze(0).expand(batch, -1)

    return {
        "boxes": boxes,
        "scores": scores,
        "labels": labels,
        "indices": original_indices,
        "class_probabilities": class_probabilities,
        "distances": distances,
        "angles": angles,
        "level_shapes": level_shapes,
    }


def select_candidates(
    dense: dict[str, torch.Tensor],
    config: OperatorConfig,
) -> dict[str, torch.Tensor]:
    """
    置信度过滤 + 稳定排序 + Pre-TopK。

    输出使用固定长度，并通过 count 表示有效数量。
    """
    dense_boxes = dense["boxes"]
    dense_scores = dense["scores"]
    dense_labels = dense["labels"]
    dense_indices = dense["indices"]

    batch_size = dense_boxes.shape[0]
    topk = config.pre_topk
    device = dense_boxes.device

    output_boxes = torch.zeros(
        batch_size,
        topk,
        5,
        dtype=torch.float32,
        device=device,
    )

    output_scores = torch.zeros(
        batch_size,
        topk,
        dtype=torch.float32,
        device=device,
    )

    output_labels = torch.full(
        (batch_size, topk),
        fill_value=-1,
        dtype=torch.int32,
        device=device,
    )

    output_indices = torch.full(
        (batch_size, topk),
        fill_value=-1,
        dtype=torch.int32,
        device=device,
    )

    output_count = torch.zeros(
        batch_size,
        dtype=torch.int32,
        device=device,
    )

    for batch_index in range(batch_size):
        # 约定使用严格大于，不是大于等于。
        valid_mask = (
            dense_scores[batch_index]
            > config.conf_threshold
        )

        valid_positions = torch.nonzero(
            valid_mask,
            as_tuple=False,
        ).squeeze(1)

        if valid_positions.numel() == 0:
            continue

        valid_scores = dense_scores[
            batch_index,
            valid_positions,
        ]

        # stable=True：
        # 分数相同时保留原始位置的先后顺序。
        sorted_order = torch.argsort(
            valid_scores,
            descending=True,
            stable=True,
        )

        selected_positions = valid_positions[
            sorted_order[:topk]
        ]

        selected_count = selected_positions.numel()

        output_boxes[
            batch_index,
            :selected_count,
        ] = dense_boxes[
            batch_index,
            selected_positions,
        ]

        output_scores[
            batch_index,
            :selected_count,
        ] = dense_scores[
            batch_index,
            selected_positions,
        ]

        output_labels[
            batch_index,
            :selected_count,
        ] = dense_labels[
            batch_index,
            selected_positions,
        ].to(torch.int32)

        output_indices[
            batch_index,
            :selected_count,
        ] = dense_indices[
            batch_index,
            selected_positions,
        ].to(torch.int32)

        output_count[batch_index] = selected_count

    return {
        "boxes": output_boxes,
        "scores": output_scores,
        "labels": output_labels,
        "indices": output_indices,
        "count": output_count,
    }


def apply_fast_nms(
    candidates: dict[str, torch.Tensor],
    config: OperatorConfig,
) -> dict[str, torch.Tensor]:
    """
    对 Pre-TopK 候选执行 Class-Aware ProbIoU Fast-NMS。
    """
    candidate_boxes = candidates["boxes"]
    candidate_scores = candidates["scores"]
    candidate_labels = candidates["labels"]
    candidate_indices = candidates["indices"]
    candidate_count = candidates["count"]

    batch_size = candidate_boxes.shape[0]
    max_det = config.max_det
    device = candidate_boxes.device

    output_boxes = torch.zeros(
        batch_size,
        max_det,
        5,
        dtype=torch.float32,
        device=device,
    )

    output_scores = torch.zeros(
        batch_size,
        max_det,
        dtype=torch.float32,
        device=device,
    )

    output_labels = torch.full(
        (batch_size, max_det),
        -1,
        dtype=torch.int32,
        device=device,
    )

    output_indices = torch.full(
        (batch_size, max_det),
        -1,
        dtype=torch.int32,
        device=device,
    )

    output_count = torch.zeros(
        batch_size,
        dtype=torch.int32,
        device=device,
    )

    for batch_index in range(batch_size):
        count = int(candidate_count[batch_index])

        if count == 0:
            continue

        boxes = candidate_boxes[
            batch_index,
            :count,
        ]

        scores = candidate_scores[
            batch_index,
            :count,
        ]

        labels = candidate_labels[
            batch_index,
            :count,
        ]

        keep = fast_nms_sorted(
            boxes,
            scores,
            labels,
            config.iou_threshold,
        )

        keep = keep[:max_det]
        kept_count = keep.numel()

        output_boxes[
            batch_index,
            :kept_count,
        ] = boxes[keep]

        output_scores[
            batch_index,
            :kept_count,
        ] = scores[keep]

        output_labels[
            batch_index,
            :kept_count,
        ] = labels[keep]

        output_indices[
            batch_index,
            :kept_count,
        ] = candidate_indices[
            batch_index,
            :count,
        ][keep]

        output_count[batch_index] = kept_count

    return {
        "boxes": output_boxes,
        "scores": output_scores,
        "labels": output_labels,
        "indices": output_indices,
        "count": output_count,
    }


def obb_postprocess_reference(
    raw_levels: list[torch.Tensor],
    config: OperatorConfig,
) -> dict[str, dict[str, torch.Tensor]]:
    dense = decode_dense(raw_levels, config)

    candidates = select_candidates(
        dense,
        config,
    )

    detections = apply_fast_nms(
        candidates,
        config,
    )

    return {
        "dense": dense,
        "candidates": candidates,
        "detections": detections,
    }