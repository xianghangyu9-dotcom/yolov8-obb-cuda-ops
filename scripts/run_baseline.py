from pathlib import Path

import torch
from ultralytics import YOLO


ROOT = Path(__file__).resolve().parents[1]
WEIGHTS = ROOT / "weights" / "yolov8n-obb.pt"
IMAGE = ROOT / "assets" / "test.jpg"
OUTPUT = ROOT / "artifacts"


def main():
    if not IMAGE.is_file():
        raise FileNotFoundError(
            f"没有找到测试图片：{IMAGE}\n"
            "请把测试图片保存为 assets/test.jpg"
        )

    if not torch.cuda.is_available():
        raise RuntimeError(
            "PyTorch 无法使用 CUDA，请先检查驱动、WSL 和 PyTorch 环境"
        )

    print("PyTorch:", torch.__version__)
    print("PyTorch CUDA:", torch.version.cuda)
    print("GPU:", torch.cuda.get_device_name(0))
    print("Input image:", IMAGE)

    model = YOLO(str(WEIGHTS))

    if model.task != "obb":
        raise RuntimeError(
            f"预期加载 OBB 模型，实际任务类型为：{model.task}"
        )

    results = model.predict(
        source=str(IMAGE),
        imgsz=640,
        device=0,
        conf=0.25,
        save=True,
        project=str(OUTPUT),
        name="baseline",
        exist_ok=True,
    )

    if len(results) == 0:
        raise RuntimeError("模型没有返回图片结果")

    result = results[0]

    detection_count = (
        0
        if result.obb is None
        else len(result.obb)
    )

    print("Task:", model.task)
    print("Detection count:", detection_count)
    print("Timing:", result.speed)
    print("Saved directory:", OUTPUT / "baseline")

    if detection_count == 0:
        print("模型运行成功，但当前图片没有旋转框通过置信度筛选")
        return

    print("xywhr shape:", result.obb.xywhr.shape)
    print("confidence shape:", result.obb.conf.shape)
    print("class shape:", result.obb.cls.shape)

    print("First box xywhr:", result.obb.xywhr[0])
    print("First confidence:", result.obb.conf[0])
    print("First class ID:", result.obb.cls[0].int())


if __name__ == "__main__":
    main()