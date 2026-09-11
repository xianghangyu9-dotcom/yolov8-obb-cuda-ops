from __future__ import annotations

import math

import torch


def stable_softmax(
    logits: torch.Tensor,
    dim: int = -1,
) -> torch.Tensor:
    """
    数值稳定 Softmax。

    无论输入是 FP16 还是 FP32，内部均使用 FP32 计算。
    """
    logits_fp32 = logits.float()

    max_value = logits_fp32.amax(
        dim=dim,
        keepdim=True,
    )

    exp_value = torch.exp(logits_fp32 - max_value)

    return exp_value / exp_value.sum(
        dim=dim,
        keepdim=True,
    )


def dfl_expectation(
    box_logits: torch.Tensor,
    reg_max: int,
) -> torch.Tensor:
    """
    输入:
        box_logits: [B, N, 4*reg_max]

    输出:
        distances: [B, N, 4]
        最后一维依次为 l, t, r, b。
    """
    if box_logits.ndim != 3:
        raise ValueError(
            f"box_logits 必须是三维，实际为 {box_logits.shape}"
        )

    batch, positions, channels = box_logits.shape

    expected_channels = 4 * reg_max

    if channels != expected_channels:
        raise ValueError(
            f"DFL 通道数错误：得到 {channels}，"
            f"预期 {expected_channels}"
        )

    logits = box_logits.reshape(
        batch,
        positions,
        4,
        reg_max,
    )

    probabilities = stable_softmax(
        logits,
        dim=-1,
    )

    projection = torch.arange(
        reg_max,
        dtype=torch.float32,
        device=box_logits.device,
    )

    distances = (
        probabilities * projection
    ).sum(dim=-1)

    return distances


def decode_angle(
    angle_logits: torch.Tensor,
) -> torch.Tensor:
    """
    YOLOv8-OBB 角度解码。

    输入:
        [B, N, 1] 原始角度 Logit

    输出:
        [B, N, 1] 弧度
    """
    return (
        torch.sigmoid(angle_logits.float()) - 0.25
    ) * math.pi


def make_anchor_points(
    raw_levels: list[torch.Tensor],
    strides: tuple[int, ...],
    grid_offset: float = 0.5,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    根据每层 H、W 生成 Anchor Center。

    返回:
        anchor_points: [N, 2]，网格坐标
        stride_tensor: [N, 1]
    """
    if len(raw_levels) != len(strides):
        raise ValueError(
            "Raw Tensor 数量必须与 stride 数量一致"
        )

    anchor_points = []
    stride_values = []

    device = raw_levels[0].device

    for raw, stride in zip(raw_levels, strides):
        _, _, height, width = raw.shape

        y = (
            torch.arange(
                height,
                device=device,
                dtype=torch.float32,
            )
            + grid_offset
        )

        x = (
            torch.arange(
                width,
                device=device,
                dtype=torch.float32,
            )
            + grid_offset
        )

        grid_y, grid_x = torch.meshgrid(
            y,
            x,
            indexing="ij",
        )

        level_anchors = torch.stack(
            [grid_x, grid_y],
            dim=-1,
        ).reshape(-1, 2)

        level_strides = torch.full(
            size=(height * width, 1),
            fill_value=float(stride),
            dtype=torch.float32,
            device=device,
        )

        anchor_points.append(level_anchors)
        stride_values.append(level_strides)

    return (
        torch.cat(anchor_points, dim=0),
        torch.cat(stride_values, dim=0),
    )


def dist2rbox_grid(
    distances: torch.Tensor,
    angles: torch.Tensor,
    anchor_points: torch.Tensor,
) -> torch.Tensor:
    """
    在网格坐标系中把 l,t,r,b 解码为 cx,cy,w,h。

    输入:
        distances:     [B, N, 4]
        angles:        [B, N, 1]
        anchor_points: [N, 2]

    输出:
        boxes_grid: [B, N, 4]
    """
    left_top, right_bottom = distances.split(
        2,
        dim=-1,
    )

    cos_angle = torch.cos(angles)
    sin_angle = torch.sin(angles)

    offset_x, offset_y = (
        (right_bottom - left_top) / 2
    ).split(1, dim=-1)

    rotated_x = (
        offset_x * cos_angle
        - offset_y * sin_angle
    )

    rotated_y = (
        offset_x * sin_angle
        + offset_y * cos_angle
    )

    center = torch.cat(
        [rotated_x, rotated_y],
        dim=-1,
    ) + anchor_points.unsqueeze(0)

    width_height = left_top + right_bottom

    return torch.cat(
        [center, width_height],
        dim=-1,
    )


def decode_xywhr(
    distances: torch.Tensor,
    angles: torch.Tensor,
    anchor_points: torch.Tensor,
    stride_tensor: torch.Tensor,
) -> torch.Tensor:
    """
    输出像素坐标下的 cx,cy,w,h,angle。
    """
    boxes_grid = dist2rbox_grid(
        distances,
        angles,
        anchor_points,
    )

    boxes_pixel = (
        boxes_grid
        * stride_tensor.unsqueeze(0)
    )

    return torch.cat(
        [boxes_pixel, angles],
        dim=-1,
    )


def _covariance_components(
    boxes: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    把 OBB 转换为二维高斯分布的协方差分量。

    boxes: [N, 5]，格式为 cx,cy,w,h,angle
    """
    width_variance = (
        boxes[..., 2:3].float().square() / 12.0
    )

    height_variance = (
        boxes[..., 3:4].float().square() / 12.0
    )

    angle = boxes[..., 4:5].float()

    cos_angle = torch.cos(angle)
    sin_angle = torch.sin(angle)

    cos_square = cos_angle.square()
    sin_square = sin_angle.square()

    covariance_a = (
        width_variance * cos_square
        + height_variance * sin_square
    )

    covariance_b = (
        width_variance * sin_square
        + height_variance * cos_square
    )

    covariance_c = (
        width_variance - height_variance
    ) * cos_angle * sin_angle

    return covariance_a, covariance_b, covariance_c


def pairwise_probiou(
    boxes1: torch.Tensor,
    boxes2: torch.Tensor,
    eps: float = 1e-7,
) -> torch.Tensor:
    """
    两组旋转框之间的 ProbIoU。

    输入:
        boxes1: [N, 5]
        boxes2: [M, 5]

    输出:
        similarity: [N, M]
    """
    if boxes1.numel() == 0 or boxes2.numel() == 0:
        return torch.empty(
            (boxes1.shape[0], boxes2.shape[0]),
            dtype=torch.float32,
            device=boxes1.device,
        )

    boxes1 = boxes1.float()
    boxes2 = boxes2.float()

    x1 = boxes1[:, 0:1]
    y1 = boxes1[:, 1:2]

    x2 = boxes2[:, 0].unsqueeze(0)
    y2 = boxes2[:, 1].unsqueeze(0)

    a1, b1, c1 = _covariance_components(boxes1)
    a2, b2, c2 = _covariance_components(boxes2)

    a2 = a2.squeeze(-1).unsqueeze(0)
    b2 = b2.squeeze(-1).unsqueeze(0)
    c2 = c2.squeeze(-1).unsqueeze(0)

    denominator = (
        (a1 + a2) * (b1 + b2)
        - (c1 + c2).square()
    )

    t1 = (
        (
            (a1 + a2) * (y1 - y2).square()
            + (b1 + b2) * (x1 - x2).square()
        )
        / (denominator + eps)
    ) * 0.25

    t2 = (
        (
            (c1 + c2)
            * (x2 - x1)
            * (y1 - y2)
        )
        / (denominator + eps)
    ) * 0.5

    determinant1 = (
        a1 * b1 - c1.square()
    ).clamp_min(0.0)

    determinant2 = (
        a2 * b2 - c2.square()
    ).clamp_min(0.0)

    t3 = torch.log(
        denominator
        / (
            4.0
            * torch.sqrt(
                determinant1 * determinant2
            )
            + eps
        )
        + eps
    ) * 0.5

    bhattacharyya_distance = (
        t1 + t2 + t3
    ).clamp(min=eps, max=100.0)

    hellinger_distance = torch.sqrt(
        1.0
        - torch.exp(-bhattacharyya_distance)
        + eps
    )

    return 1.0 - hellinger_distance


def fast_nms_sorted(
    boxes: torch.Tensor,
    scores: torch.Tensor,
    labels: torch.Tensor,
    iou_threshold: float,
) -> torch.Tensor:
    """
    输入必须已经按 Score 降序排列。

    返回值是输入数组中需要保留的位置。
    """
    del scores  # 排序已在调用前完成

    candidate_count = boxes.shape[0]

    if candidate_count == 0:
        return torch.empty(
            (0,),
            dtype=torch.int64,
            device=boxes.device,
        )

    similarities = pairwise_probiou(
        boxes,
        boxes,
    )

    same_class = (
        labels[:, None] == labels[None, :]
    )

    # row < column：
    # 行是更高分框，列是更低分框。
    upper_triangle = torch.triu(
        torch.ones(
            candidate_count,
            candidate_count,
            dtype=torch.bool,
            device=boxes.device,
        ),
        diagonal=1,
    )

    suppressed = (
        (similarities >= iou_threshold)
        & same_class
        & upper_triangle
    ).any(dim=0)

    return torch.nonzero(
        ~suppressed,
        as_tuple=False,
    ).squeeze(1)