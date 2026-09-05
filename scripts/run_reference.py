from pathlib import Path

import torch

from reference.obb_reference import (
    OperatorConfig,
    obb_postprocess_reference,
)


ROOT = Path(__file__).resolve().parents[1]
GOLDEN_DIR = ROOT / "artifacts" / "golden"
OUTPUT_PATH = (
    ROOT
    / "artifacts"
    / "golden"
    / "reference_postprocess.pt"
)


def load_raw_level(path: Path) -> torch.Tensor:
    data = torch.load(
        path,
        map_location="cpu",
        weights_only=True,
    )

    # 兼容第一阶段保存为字典或直接保存 Tensor。
    if isinstance(data, dict):
        return data["raw"]

    return data


def main():
    raw_levels = [
        load_raw_level(GOLDEN_DIR / "raw_p3.pt"),
        load_raw_level(GOLDEN_DIR / "raw_p4.pt"),
        load_raw_level(GOLDEN_DIR / "raw_p5.pt"),
    ]

    config = OperatorConfig(
        reg_max=16,
        strides=(8, 16, 32),
        conf_threshold=0.25,
        iou_threshold=0.45,
        pre_topk=1024,
        max_det=300,
    )

    results = obb_postprocess_reference(
        raw_levels,
        config,
    )

    dense = results["dense"]
    candidates = results["candidates"]
    detections = results["detections"]

    print("Dense boxes:", dense["boxes"].shape)
    print("Dense scores:", dense["scores"].shape)

    print(
        "Candidate count:",
        candidates["count"].tolist(),
    )

    print(
        "Detection count:",
        detections["count"].tolist(),
    )

    if int(detections["count"][0]) > 0:
        count = int(detections["count"][0])

        print(
            "Top 5 scores:",
            detections["scores"][0, :min(5, count)],
        )

        print(
            "Top 5 labels:",
            detections["labels"][0, :min(5, count)],
        )

        print(
            "Top 5 original indices:",
            detections["indices"][0, :min(5, count)],
        )

        print(
            "Top box:",
            detections["boxes"][0, 0],
        )
    for name in ["boxes", "scores", "angles", "distances"]:
        tensor = dense[name]

        print(
            name,
            "shape =", tuple(tensor.shape),
            "dtype =", tensor.dtype,
            "min =", float(tensor.min()),
            "max =", float(tensor.max()),
            "finite =", bool(torch.isfinite(tensor).all()),
        )

    torch.save(results, OUTPUT_PATH)

    print("参考结果已保存到：", OUTPUT_PATH)


if __name__ == "__main__":
    main()