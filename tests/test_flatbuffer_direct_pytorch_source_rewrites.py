from __future__ import annotations

import pytest

from onnx2tf.tflite_builder.pytorch_source_rewrites import (
    _fold_channel_first_gap_conv_bridges,
    _rewrite_channel_first_gap_outputs_to_explicit_channel_last,
    _rewrite_channel_first_se_scale_binary_bridges,
)


def test_gap_conv_rewrite_removes_redundant_nhwc_to_nchw_bridge() -> None:
    assert _fold_channel_first_gap_conv_bridges(
        [
            "gap = torch.mean(input=x, dim=[2, 3], keepdim=True)",
            "y = self.conv(gap.permute(0, 3, 1, 2).contiguous())",
        ]
    ) == [
        "gap = torch.mean(input=x, dim=[2, 3], keepdim=True)",
        "y = self.conv(gap)",
    ]


def test_gap_output_rewrite_emits_explicit_channel_last_mean() -> None:
    assert _rewrite_channel_first_gap_outputs_to_explicit_channel_last(
        [
            "gap = torch.mean(input=x, dim=(2, 3), keepdim=True)",
            "gap_nhwc = gap.permute(0, 2, 3, 1).contiguous()",
        ]
    ) == [
        "gap_nhwc = torch.mean("
        "x.permute(0, 2, 3, 1).contiguous(), dim=[1, 2], keepdim=True)"
    ]


@pytest.mark.parametrize(
    "rewrite",
    [
        _fold_channel_first_gap_conv_bridges,
        _rewrite_channel_first_gap_outputs_to_explicit_channel_last,
        _rewrite_channel_first_se_scale_binary_bridges,
    ],
)
def test_gap_se_source_rewrites_preserve_unmatched_source(rewrite) -> None:
    lines = ["y = torch.relu(x)", "return y"]

    assert rewrite(lines) == lines
