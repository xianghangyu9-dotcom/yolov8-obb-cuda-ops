import math

import torch

from reference.obb_math import (
    decode_angle,
    dfl_expectation,
    dist2rbox_grid,
    fast_nms_sorted,
    make_anchor_points,
    pairwise_probiou,
)
from reference.obb_reference import (
    OperatorConfig,
    decode_dense,
    select_candidates,
)


def test_zero_dfl_logits_give_7_5():
    reg_max = 16

    logits = torch.zeros(
        1,
        3,
        4 * reg_max,
    )

    distances = dfl_expectation(
        logits,
        reg_max,
    )

    expected = torch.full(
        (1, 3, 4),
        7.5,
    )

    torch.testing.assert_close(
        distances,
        expected,
    )


def test_extreme_dfl_logits_are_finite():
    reg_max = 16

    logits = torch.linspace(
        -20.0,
        20.0,
        steps=4 * reg_max,
    ).reshape(1, 1, -1)

    distances = dfl_expectation(
        logits,
        reg_max,
    )

    assert torch.isfinite(distances).all()
    assert (distances >= 0).all()
    assert (distances <= reg_max - 1).all()


def test_angle_boundaries():
    logits = torch.tensor(
        [[[-100.0], [0.0], [100.0]]]
    )

    angles = decode_angle(logits)

    torch.testing.assert_close(
        angles[0, 0, 0],
        torch.tensor(-math.pi / 4),
        atol=1e-5,
        rtol=0,
    )

    torch.testing.assert_close(
        angles[0, 1, 0],
        torch.tensor(math.pi / 4),
        atol=1e-6,
        rtol=0,
    )

    torch.testing.assert_close(
        angles[0, 2, 0],
        torch.tensor(3 * math.pi / 4),
        atol=1e-5,
        rtol=0,
    )


def test_anchor_order():
    raw = torch.empty(
        1,
        10,
        2,
        3,
    )

    anchors, strides = make_anchor_points(
        [raw],
        (8,),
    )

    expected = torch.tensor(
        [
            [0.5, 0.5],
            [1.5, 0.5],
            [2.5, 0.5],
            [0.5, 1.5],
            [1.5, 1.5],
            [2.5, 1.5],
        ]
    )

    torch.testing.assert_close(
        anchors,
        expected,
    )

    torch.testing.assert_close(
        strides,
        torch.full((6, 1), 8.0),
    )


def test_dist2rbox_angle_zero():
    distances = torch.tensor(
        [[[2.0, 4.0, 6.0, 8.0]]]
    )

    angles = torch.zeros(1, 1, 1)

    anchors = torch.tensor(
        [[10.0, 20.0]]
    )

    decoded = dist2rbox_grid(
        distances,
        angles,
        anchors,
    )

    # offset_x = (6-2)/2 = 2
    # offset_y = (8-4)/2 = 2
    # center = (12,22)
    # width = 2+6 = 8
    # height = 4+8 = 12
    expected = torch.tensor(
        [[[12.0, 22.0, 8.0, 12.0]]]
    )

    torch.testing.assert_close(
        decoded,
        expected,
    )


def test_probiou_identity_and_symmetry():
    boxes = torch.tensor(
        [
            [100.0, 100.0, 40.0, 20.0, 0.0],
            [120.0, 105.0, 30.0, 10.0, 0.3],
        ]
    )

    similarities = pairwise_probiou(
        boxes,
        boxes,
    )

    assert similarities[0, 0] > 0.999
    assert similarities[1, 1] > 0.999

    torch.testing.assert_close(
        similarities,
        similarities.T,
        atol=1e-6,
        rtol=1e-6,
    )


def test_fast_nms_is_class_aware():
    boxes = torch.tensor(
        [
            [100.0, 100.0, 40.0, 20.0, 0.0],
            [100.0, 100.0, 40.0, 20.0, 0.0],
            [100.0, 100.0, 40.0, 20.0, 0.0],
        ]
    )

    scores = torch.tensor(
        [0.9, 0.8, 0.7]
    )

    labels = torch.tensor(
        [0, 0, 1]
    )

    keep = fast_nms_sorted(
        boxes,
        scores,
        labels,
        iou_threshold=0.45,
    )

    # 第二个框与第一个框同类且完全重叠，被抑制。
    # 第三个框虽然重叠，但类别不同，应保留。
    assert keep.tolist() == [0, 2]


def test_confidence_threshold_is_strict():
    reg_max = 16
    class_count = 2

    # 三个位置，类别 Logit 都为 0，Sigmoid 后都是 0.5。
    raw = torch.zeros(
        1,
        4 * reg_max + class_count + 1,
        1,
        3,
    )

    config = OperatorConfig(
        reg_max=reg_max,
        strides=(8,),
        conf_threshold=0.5,
        pre_topk=3,
    )

    dense = decode_dense(
        [raw],
        config,
    )

    candidates = select_candidates(
        dense,
        config,
    )

    # 使用 score > threshold，因此等于 0.5 不通过。
    assert candidates["count"].tolist() == [0]


def test_equal_scores_keep_original_order():
    reg_max = 16
    class_count = 2

    raw = torch.zeros(
        1,
        4 * reg_max + class_count + 1,
        1,
        3,
    )

    config = OperatorConfig(
        reg_max=reg_max,
        strides=(8,),
        conf_threshold=0.49,
        pre_topk=3,
    )

    dense = decode_dense(
        [raw],
        config,
    )

    candidates = select_candidates(
        dense,
        config,
    )

    assert candidates["indices"][0].tolist() == [
        0,
        1,
        2,
    ]


def test_fp16_input_produces_fp32_output():
    reg_max = 16
    class_count = 2

    raw = torch.zeros(
        1,
        4 * reg_max + class_count + 1,
        2,
        2,
        dtype=torch.float16,
    )

    config = OperatorConfig(
        reg_max=reg_max,
        strides=(8,),
    )

    dense = decode_dense(
        [raw],
        config,
    )

    assert dense["boxes"].dtype == torch.float32
    assert dense["scores"].dtype == torch.float32