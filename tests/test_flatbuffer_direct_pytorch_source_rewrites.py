from __future__ import annotations

import pytest

from onnx2tf.tflite_builder.pytorch_source_rewrites import (
    _fold_channel_first_gap_conv_bridges,
    _fold_channel_last_affine_conv_bridges,
    _rewrite_channel_first_gap_outputs_to_explicit_channel_last,
    _rewrite_channel_first_se_scale_binary_bridges,
    _rewrite_channel_last_gap_means_to_reduce_mean,
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


def test_affine_conv_rewrite_handles_compact_assignments() -> None:
    assert _fold_channel_last_affine_conv_bridges(
        [
            "x_nhwc=_align_tensor_to_target_shape(x_cf.permute(0, 2, 3, 1).contiguous(), [1, 8, 8, 4])",
            "_binary_lhs_0, _binary_rhs_0=_align_binary_inputs(x_nhwc, self.const_mul_any, [1, 8, 8, 4])",
            "mul0=_align_tensor_to_target_shape(torch.mul(_binary_lhs_0, _binary_rhs_0), [1, 8, 8, 4])",
            "_binary_lhs_1, _binary_rhs_1=_align_binary_inputs(mul0, self.const_add_any, [1, 8, 8, 4])",
            "add0=_align_tensor_to_target_shape(torch.add(_binary_lhs_1, _binary_rhs_1), [1, 8, 8, 4])",
            "y=self.conv_block_0(add0.permute(0, 3, 1, 2).contiguous())",
        ],
        derive_local_var_name=lambda base_name: f"{base_name}_tmp",
        channel_first_constant_expr_for_buffer_attr=lambda buffer_expr, target_shape: (
            "self.const_mul_cf"
            if buffer_expr == "self.const_mul_any"
            and list(target_shape) == [1, 4, 1, 1]
            else "self.const_add_cf"
            if buffer_expr == "self.const_add_any"
            and list(target_shape) == [1, 4, 1, 1]
            else None
        ),
    ) == [
        "mul0_cf_tmp = torch.mul(x_cf, self.const_mul_cf)",
        "add0_cf_tmp = torch.add(mul0_cf_tmp, self.const_add_cf)",
        "y = self.conv_block_0(add0_cf_tmp)",
    ]


def test_channel_last_gap_rewrite_handles_rank3_and_rank4_permute_forms() -> None:
    first_line = (
        "t_1780 = _align_tensor_to_target_shape("
        "input=torch.permute(input=t1780_cf, dims=(0, 2, 1)).contiguous(), "
        "target_shape=(1, 4096, 180))"
    )

    assert _rewrite_channel_last_gap_means_to_reduce_mean(
        [
            first_line,
            "t1781_cf = torch.mean(input=t_1780, dim=(2,), keepdim=True)",
            "gap_nhwc = torch.mean(input=_torch_permute(input=features_cf, perm=(0, 2, 3, 1)), dim=(1, 2), keepdim=True)",
        ]
    ) == [
        first_line,
        "t1781_cf = torch.mean(t1780_cf, dim=1, keepdim=True)",
        "gap_nhwc = _reduce_mean(features_cf.permute(0, 2, 3, 1).contiguous(), "
        "_normalize_axes([1, 2], "
        "features_cf.permute(0, 2, 3, 1).contiguous().ndim), keepdims=True)",
    ]


@pytest.mark.parametrize(
    "rewrite",
    [
        _fold_channel_first_gap_conv_bridges,
        _rewrite_channel_last_gap_means_to_reduce_mean,
        _rewrite_channel_first_gap_outputs_to_explicit_channel_last,
        _rewrite_channel_first_se_scale_binary_bridges,
    ],
)
def test_gap_se_source_rewrites_preserve_unmatched_source(rewrite) -> None:
    lines = ["y = torch.relu(x)", "return y"]

    assert rewrite(lines) == lines


def test_affine_conv_rewrite_preserves_unmatched_source() -> None:
    lines = ["y = torch.relu(x)", "return y"]

    assert (
        _fold_channel_last_affine_conv_bridges(
            lines,
            derive_local_var_name=lambda base_name: base_name,
            channel_first_constant_expr_for_buffer_attr=lambda _buffer, _shape: None,
        )
        == lines
    )
