import types
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch
from ultralytics import YOLO


ROOT = Path(__file__).resolve().parents[1]
WEIGHTS = ROOT / "weights" / "yolov8n-obb.pt"
GOLDEN_DIR = ROOT / "artifacts" / "golden"
ONNX_DIR = ROOT / "artifacts" / "onnx"
ONNX_PATH = ONNX_DIR / "yolov8n_obb_raw_head.onnx"


def raw_obb_forward(self, features):
    """直接输出三个尺度的原始 Head Tensor。"""

    outputs = []

    for level_index, feature in enumerate(features):
        box_logits = self.cv2[level_index](feature)
        class_logits = self.cv3[level_index](feature)
        angle_logits = self.cv4[level_index](feature)

        raw = torch.cat(
            [box_logits, class_logits, angle_logits],
            dim=1,
        )

        outputs.append(raw)

    return tuple(outputs)


def main():
    ONNX_DIR.mkdir(parents=True, exist_ok=True)

    yolo = YOLO(str(WEIGHTS))
    network = yolo.model.cpu().float().eval()
    head = network.model[-1]

    input_tensor = torch.load(
        GOLDEN_DIR / "input_fp32.pt",
        map_location="cpu",
        weights_only=True,
    )

    # 保存原来的 forward，导出完成后恢复。
    original_forward = head.forward

    # 把普通 Python 函数绑定成 head 对象的方法。
    head.forward = types.MethodType(
        raw_obb_forward,
        head,
    )

    try:
        with torch.inference_mode():
            pytorch_outputs = network(input_tensor)

        torch.onnx.export(
            network,
            (input_tensor,),
            str(ONNX_PATH),
            input_names=["images"],
            output_names=["p3", "p4", "p5"],
            opset_version=18,
            do_constant_folding=True,

            # 使用传统 Trace 导出路径。
            # 对这种临时替换 forward 的方式更直接。
            dynamo=False,
        )

    finally:
        head.forward = original_forward

    # 检查 ONNX 文件结构是否合法。
    onnx_model = onnx.load(str(ONNX_PATH))
    onnx.checker.check_model(onnx_model)

    print("ONNX 文件检查通过：", ONNX_PATH)

    # 使用 ONNX Runtime CPU 执行。
    session = ort.InferenceSession(
        str(ONNX_PATH),
        providers=["CPUExecutionProvider"],
    )

    ort_outputs = session.run(
        ["p3", "p4", "p5"],
        {"images": input_tensor.numpy()},
    )

    for level_index, (torch_output, ort_output) in enumerate(
        zip(pytorch_outputs, ort_outputs)
    ):
        torch_array = torch_output.detach().cpu().numpy()

        max_abs_error = float(
            np.max(np.abs(torch_array - ort_output))
        )

        print(
            f"P{level_index + 3}:",
            f"shape={ort_output.shape},",
            f"max_abs_error={max_abs_error:.8e}",
        )

        np.testing.assert_allclose(
            torch_array,
            ort_output,
            rtol=1e-4,
            atol=1e-4,
        )

    print("PyTorch 与 ONNX Runtime 输出一致")


if __name__ == "__main__":
    main()