from pathlib import Path

import numpy as np
import torch

from reference.obb_reference import (
    OperatorConfig,
    decode_dense,
)


ROOT = Path(__file__).resolve().parents[1]
GOLDEN_DIR = ROOT / "artifacts" / "golden"
OUTPUT_ROOT = ROOT / "artifacts" / "cuda_v0"

CONFIG = OperatorConfig(
    reg_max=16,
    strides=(8, 16, 32),
    conf_threshold=0.25,
    iou_threshold=0.45,
    pre_topk=1024,
    max_det=300,
)


def load_raw(name: str) -> torch.Tensor:
    data = torch.load(
        GOLDEN_DIR / name,
        map_location="cpu",
        weights_only=True,
    )

    if isinstance(data, dict):
        data = data["raw"]

    return data.contiguous().float()


def save_array(
    path: Path,
    tensor: torch.Tensor,
    dtype,
) -> None:
    array = (
        tensor.detach()
        .cpu()
        .numpy()
        .astype(dtype, copy=False)
    )

    array.tofile(path)


def export_case(
    case_name: str,
    raw_levels: list[torch.Tensor],
) -> None:
    case_dir = OUTPUT_ROOT / case_name
    case_dir.mkdir(parents=True, exist_ok=True)

    for level_name, raw in zip(
        ["p3", "p4", "p5"],
        raw_levels,
    ):
        save_array(
            case_dir / f"{level_name}.bin",
            raw,
            np.float32,
        )

    dense = decode_dense(
        raw_levels,
        CONFIG,
    )

    # nonzero 返回的索引是升序，提取有效的置信度作为标准答案。
    keep = torch.nonzero(
        dense["scores"][0] > CONFIG.conf_threshold,
        as_tuple=False,
    ).squeeze(1)

    expected_boxes = dense["boxes"][0, keep]
    expected_scores = dense["scores"][0, keep]
    expected_labels = dense["labels"][0, keep]
    expected_indices = dense["indices"][0, keep]

    save_array(
        case_dir / "expected_boxes.bin",
        expected_boxes,
        np.float32,
    )

    save_array(
        case_dir / "expected_scores.bin",
        expected_scores,
        np.float32,
    )

    save_array(
        case_dir / "expected_labels.bin",
        expected_labels,
        np.int32,
    )

    save_array(
        case_dir / "expected_indices.bin",
        expected_indices,
        np.int32,
    )

    np.array(
        [keep.numel()],
        dtype=np.int32,
    ).tofile(case_dir / "expected_count.bin")

    print(
        f"{case_name}: expected_count={keep.numel()}"
    )


def main():
    real_levels = [
        load_raw("raw_p3.pt"),
        load_raw("raw_p4.pt"),
        load_raw("raw_p5.pt"),
    ]

    export_case("real", real_levels)

    # Case 1：所有类别 Logit 很小，没有候选通过。
    none_levels = [
        raw.clone()
        for raw in real_levels
    ]

    class_begin = 4 * CONFIG.reg_max
    class_end = none_levels[0].shape[1] - 1

    for raw in none_levels:
        raw[:, class_begin:class_end] = -20.0

    export_case("none", none_levels)

    # Case 2：
    # Box DFL Logit 全为 0，因此每条边的期望都是 7.5。
    # 类别 0 分数接近 1，全部 8400 个位置通过。
    all_levels = []

    for raw in real_levels:
        test_raw = torch.zeros_like(raw)

        test_raw[:, class_begin:class_end] = -20.0
        test_raw[:, class_begin] = 20.0

        # angle_logit=0，对应 angle=pi/4。
        test_raw[:, -1:] = 0.0

        all_levels.append(test_raw)

    export_case("all_zero_dfl", all_levels)

    # Case 3：测试极端 DFL 和角度 Logit。
    extreme_levels = []

    dfl_pattern = torch.linspace(
        -20.0,
        20.0,
        steps=CONFIG.reg_max,
    )

    for raw in real_levels:
        test_raw = torch.zeros_like(raw)

        for side in range(4):
            begin = side * CONFIG.reg_max
            end = begin + CONFIG.reg_max

            test_raw[:, begin:end] = (
                dfl_pattern.view(1, -1, 1, 1)
            )

        test_raw[:, class_begin:class_end] = -20.0
        test_raw[:, class_begin] = 20.0
        test_raw[:, -1:] = 20.0

        extreme_levels.append(test_raw)

    export_case("extreme", extreme_levels)


if __name__ == "__main__":
    main()