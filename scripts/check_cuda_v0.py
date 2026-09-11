import argparse
from pathlib import Path

import numpy as np


CAPACITY = 8400


def load_int32(path: Path) -> np.ndarray:
    return np.fromfile(path, dtype=np.int32)


def load_float32(path: Path) -> np.ndarray:
    return np.fromfile(path, dtype=np.float32)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "case_dir",
        type=Path,
    )

    args = parser.parse_args()
    case_dir = args.case_dir

    expected_count = int(
        load_int32(
            case_dir / "expected_count.bin"
        )[0]
    )

    cuda_count = int(
        load_int32(
            case_dir / "cuda_count.bin"
        )[0]
    )

    overflow = int(
        load_int32(
            case_dir / "cuda_overflow.bin"
        )[0]
    )

    assert overflow == 0
    assert cuda_count == expected_count
    assert 0 <= cuda_count <= CAPACITY

    cuda_boxes = load_float32(
        case_dir / "cuda_boxes.bin"
    ).reshape(CAPACITY, 5)[:cuda_count]

    cuda_scores = load_float32(
        case_dir / "cuda_scores.bin"
    )[:cuda_count]

    cuda_labels = load_int32(
        case_dir / "cuda_labels.bin"
    )[:cuda_count]

    cuda_indices = load_int32(
        case_dir / "cuda_indices.bin"
    )[:cuda_count]

    # atomicAdd 输出顺序不固定，所以按原始位置排序。
    order = np.argsort(
        cuda_indices,
        kind="stable",
    )

    cuda_boxes = cuda_boxes[order]
    cuda_scores = cuda_scores[order]
    cuda_labels = cuda_labels[order]
    cuda_indices = cuda_indices[order]

    expected_boxes = load_float32(
        case_dir / "expected_boxes.bin"
    ).reshape(expected_count, 5)

    expected_scores = load_float32(
        case_dir / "expected_scores.bin"
    )

    expected_labels = load_int32(
        case_dir / "expected_labels.bin"
    )

    expected_indices = load_int32(
        case_dir / "expected_indices.bin"
    )

    if cuda_count > 0:
        assert np.unique(cuda_indices).size == cuda_count

        np.testing.assert_array_equal(
            cuda_indices,
            expected_indices,
        )

        np.testing.assert_array_equal(
            cuda_labels,
            expected_labels,
        )

        np.testing.assert_allclose(
            cuda_scores,
            expected_scores,
            rtol=1e-5,
            atol=1e-6,
        )

        np.testing.assert_allclose(
            cuda_boxes,
            expected_boxes,
            rtol=1e-4,
            atol=1e-4,
        )

        box_abs_error = np.abs(
            cuda_boxes - expected_boxes
        )

        score_abs_error = np.abs(
            cuda_scores - expected_scores
        )

        print(
            "box max abs error:",
            float(box_abs_error.max()),
        )

        print(
            "box mean abs error:",
            float(box_abs_error.mean()),
        )

        print(
            "score max abs error:",
            float(score_abs_error.max()),
        )

    print("count:", cuda_count)
    print("overflow:", overflow)
    print("CUDA V0 check passed")

    expected_count = int(
    np.fromfile(
        case_dir / "expected_count.bin",
        dtype=np.int32,
    )[0]
    )

    cuda_count = int(
        np.fromfile(
            case_dir / "cuda_count.bin",
            dtype=np.int32,
        )[0]
    )

    print("case_dir:", case_dir.resolve())
    print("expected_count:", expected_count)
    print("cuda_count:", cuda_count)

    assert cuda_count == expected_count, (
        f"Count mismatch: CUDA={cuda_count}, "
        f"Reference={expected_count}"
    )

if __name__ == "__main__":
    main()