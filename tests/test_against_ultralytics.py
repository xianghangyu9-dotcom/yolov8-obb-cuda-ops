import torch

from ultralytics.nn.modules.block import DFL
from ultralytics.utils.metrics import batch_probiou
from ultralytics.utils.tal import (
    dist2rbox as ultralytics_dist2rbox,
)
from ultralytics.utils.tal import (
    make_anchors as ultralytics_make_anchors,
)

from reference.obb_math import (
    dfl_expectation,
    dist2rbox_grid,
    make_anchor_points,
    pairwise_probiou,
)


def test_dfl_matches_ultralytics():
    torch.manual_seed(123)

    batch = 2
    anchors = 17
    reg_max = 16

    # Ultralytics DFL 输入是 [B,4R,N]。
    logits_ultralytics = torch.randn(
        batch,
        4 * reg_max,
        anchors,
    )

    logits_ours = logits_ultralytics.transpose(
        1,
        2,
    )

    ours = dfl_expectation(
        logits_ours,
        reg_max,
    )

    official_module = DFL(reg_max)

    official = official_module(
        logits_ultralytics
    ).transpose(1, 2)

    torch.testing.assert_close(
        ours,
        official,
        atol=1e-6,
        rtol=1e-5,
    )


def test_anchor_generation_matches_ultralytics():
    raw_levels = [
        torch.empty(1, 80, 8, 8),
        torch.empty(1, 80, 4, 4),
        torch.empty(1, 80, 2, 2),
    ]

    strides = (8, 16, 32)

    ours_anchor, ours_stride = make_anchor_points(
        raw_levels,
        strides,
    )

    official_anchor, official_stride = (
        ultralytics_make_anchors(
            raw_levels,
            strides,
            grid_cell_offset=0.5,
        )
    )

    torch.testing.assert_close(
        ours_anchor,
        official_anchor.float(),
    )

    torch.testing.assert_close(
        ours_stride,
        official_stride.float(),
    )


def test_dist2rbox_matches_ultralytics():
    torch.manual_seed(456)

    batch = 2
    positions = 20

    distances = torch.rand(
        batch,
        positions,
        4,
    ) * 15.0

    angles = (
        torch.rand(batch, positions, 1)
        * torch.pi
        - torch.pi / 4
    )

    anchors = torch.rand(
        positions,
        2,
    ) * 80.0

    ours = dist2rbox_grid(
        distances,
        angles,
        anchors,
    )

    # 官方函数使用 [B,4,N] 和 dim=1。
    official = ultralytics_dist2rbox(
        distances.transpose(1, 2),
        angles.transpose(1, 2),
        anchors.T.unsqueeze(0),
        dim=1,
    ).transpose(1, 2)

    torch.testing.assert_close(
        ours,
        official,
        atol=1e-5,
        rtol=1e-5,
    )


def test_probiou_matches_ultralytics():
    boxes1 = torch.tensor(
        [
            [100.0, 100.0, 40.0, 20.0, 0.0],
            [150.0, 120.0, 30.0, 60.0, 0.4],
        ]
    )

    boxes2 = torch.tensor(
        [
            [105.0, 100.0, 40.0, 20.0, 0.1],
            [300.0, 300.0, 50.0, 50.0, 0.0],
            [150.0, 120.0, 30.0, 60.0, 0.4],
        ]
    )

    ours = pairwise_probiou(
        boxes1,
        boxes2,
    )

    official = batch_probiou(
        boxes1,
        boxes2,
    )

    torch.testing.assert_close(
        ours,
        official,
        atol=1e-6,
        rtol=1e-5,
    )