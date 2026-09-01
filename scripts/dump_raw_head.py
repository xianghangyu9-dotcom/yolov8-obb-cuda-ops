import json
from pathlib import Path

import torch
from ultralytics import YOLO


ROOT = Path(__file__).resolve().parents[1]
WEIGHTS = ROOT / "weights" / "yolov8n-obb.pt"
OUTPUT_DIR = ROOT / "artifacts" / "golden"


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # 使用 CPU + FP32 制作参考结果，减少 GPU 非确定性。
    yolo = YOLO(str(WEIGHTS))
    network = yolo.model.cpu().float().eval()

    # Ultralytics 模型内部的最后一个模块通常就是 OBB Head。
    head = network.model[-1]

    print("Head class:", type(head).__name__)
    print("Number of classes:", head.nc)
    print("reg_max:", head.reg_max)
    print("Number of angle channels:", head.ne)
    print("Stride:", head.stride.tolist())

    captured = {}

    def capture_head_inputs(module, args):
        feature_maps = args[0]

        # 必须 clone。
        # 一些版本的 Head Forward 会原地修改传入的列表。
        captured["features"] = [
            feature.detach().clone()
            for feature in feature_maps
        ]

    hook_handle = head.register_forward_pre_hook(capture_head_inputs)

    # 固定随机种子，让每次生成相同输入。
    generator = torch.Generator(device="cpu")
    generator.manual_seed(20260830)

    input_tensor = torch.rand(
        size=(1, 3, 640, 640),
        generator=generator,
        dtype=torch.float32,
    )

    with torch.inference_mode():
        _ = network(input_tensor)

    hook_handle.remove()

    if "features" not in captured:
        raise RuntimeError("没有捕获到 OBB Head 输入")

    feature_maps = captured["features"]

    if len(feature_maps) != 3:
        raise RuntimeError(
            f"预期 3 个尺度，实际得到 {len(feature_maps)} 个"
        )

    reg_max = int(head.reg_max)
    class_count = int(head.nc)
    angle_channels = int(head.ne)

    expected_channels = (
        4 * reg_max
        + class_count
        + angle_channels
    )

    metadata = {
        "input_shape": list(input_tensor.shape),
        "input_dtype": str(input_tensor.dtype),
        "reg_max": reg_max,
        "class_count": class_count,
        "angle_channels": angle_channels,
        "expected_raw_channels": expected_channels,
        "stride": [float(value) for value in head.stride],
        "levels": [],
    }

    torch.save(
        input_tensor,
        OUTPUT_DIR / "input_fp32.pt",
    )

    with torch.inference_mode():
        for level_index, feature in enumerate(feature_maps):
            # 4 × reg_max 个边框分布 Logit。
            box_logits = head.cv2[level_index](feature)

            # C 个类别 Logit，尚未执行 Sigmoid。
            class_logits = head.cv3[level_index](feature)

            # 原始角度 Logit，尚未执行 Sigmoid 和角度映射。
            angle_logits = head.cv4[level_index](feature)

            raw = torch.cat(
                [box_logits, class_logits, angle_logits],
                dim=1,
            )

            if raw.shape[1] != expected_channels:
                raise RuntimeError(
                    f"Level {level_index} 通道数错误："
                    f"得到 {raw.shape[1]}，"
                    f"预期 {expected_channels}"
                )

            level_name = f"p{level_index + 3}"

            torch.save(
                {
                    "box_logits": box_logits.cpu(),
                    "class_logits": class_logits.cpu(),
                    "angle_logits": angle_logits.cpu(),
                    "raw": raw.cpu(),
                },
                OUTPUT_DIR / f"raw_{level_name}.pt",
            )

            metadata["levels"].append(
                {
                    "name": level_name,
                    "feature_shape": list(feature.shape),
                    "box_shape": list(box_logits.shape),
                    "class_shape": list(class_logits.shape),
                    "angle_shape": list(angle_logits.shape),
                    "raw_shape": list(raw.shape),
                }
            )

            print(
                level_name,
                "feature =", tuple(feature.shape),
                "raw =", tuple(raw.shape),
            )

    with open(
        OUTPUT_DIR / "metadata.json",
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(metadata, file, indent=2, ensure_ascii=False)

    print("Golden Tensor 已保存到：", OUTPUT_DIR)


if __name__ == "__main__":
    main()