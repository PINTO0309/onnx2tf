from __future__ import annotations

import pytest

from onnx2tf.tflite_builder.pytorch_source_rewrites import (
    _collapse_redundant_torch_permute_chains,
    _fold_boundary_transpose_pad_conv_bridges,
    _fold_channel_first_gap_conv_bridges,
    _fold_channel_first_hardsigmoid_gate_conv_bridges,
    _fold_channel_last_affine_conv_bridges,
    _fold_channel_last_prelu_bridges,
    _fold_rank4_reshape_permute_conv_bridges,
    _inline_trivial_public_layout_bridge_aliases,
    _prune_dead_forward_lines,
    _repair_channel_last_gap_conv_inputs,
    _rewrite_channel_first_gap_outputs_to_explicit_channel_last,
    _rewrite_channel_first_se_scale_binary_bridges,
    _rewrite_channel_last_binary_bridge_chains,
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


def test_boundary_transpose_conv_rewrite_removes_redundant_input_bridge() -> None:
    assert _fold_boundary_transpose_pad_conv_bridges(
        [
            "conv0=self.conv_block_0(x_nhwc.permute(0, 2, 3, 1).contiguous())",
            "y=_torch_permute(conv0, [0, 2, 3, 1])",
        ]
    ) == [
        "conv0 = self.conv_block_0(x_nhwc)",
        "y=_torch_permute(conv0, [0, 2, 3, 1])",
    ]


def test_redundant_permute_rewrite_collapses_matching_chain() -> None:
    assert _collapse_redundant_torch_permute_chains(
        ["out=_torch_permute(x, [0, 2, 3, 1]).permute(0, 2, 3, 1).contiguous()"]
    ) == ["out=_torch_permute(x, [0, 2, 3, 1])"]


def test_public_layout_bridge_rewrite_inlines_alias() -> None:
    assert _inline_trivial_public_layout_bridge_aliases(
        [
            "foo_public_layout_bridge=foo_cf",
            "bar = torch.add(foo_public_layout_bridge, other)",
            "baz = _torch_permute(foo_public_layout_bridge, [0, 3, 1, 2])",
        ]
    ) == [
        "bar = torch.add(foo_cf, other)",
        "baz = _torch_permute(foo_cf, [0, 3, 1, 2])",
    ]


def test_channel_last_prelu_rewrite_folds_round_trip_bridges() -> None:
    assert _fold_channel_last_prelu_bridges(
        [
            "x_cf=_torch_permute(x, [0, 3, 1, 2])",
            "x_prelu=self.prelu_0(x_cf)",
            "y=_torch_permute(x_prelu, [0, 2, 3, 1])",
        ]
    ) == [
        "y = self.prelu_0(x.permute(0, 3, 1, 2).contiguous()).permute(0, 2, 3, 1).contiguous()"
    ]


def test_rank4_reshape_permute_conv_rewrite_folds_layout_round_trip() -> None:
    assert _fold_rank4_reshape_permute_conv_bridges(
        [
            "x_reshaped = torch.reshape(input=x_cf, shape=(1, 16, 4, 4))",
            "x_nhwc = torch.reshape(input=x_reshaped.permute(0, 2, 3, 1).contiguous(), shape=(1, 4, 4, 16))",
            "y = _align_tensor_to_target_shape(self.conv_block_0(x_nhwc.permute(0, 3, 1, 2).contiguous()), (1, 32, 4, 4))",
        ]
    ) == [
        "x_reshaped = torch.reshape(input=x_cf, shape=(1, 16, 4, 4))",
        "y = _align_tensor_to_target_shape(self.conv_block_0(x_reshaped), [1, 32, 4, 4])",
    ]


def test_channel_first_hardsigmoid_gate_rewrite_folds_classifier_block() -> None:
    assert _fold_channel_first_hardsigmoid_gate_conv_bridges(
        [
            "        gate_mul = torch.mul(features_cf, 0.1666666716337204)",
            "        gate_add = _align_tensor_to_target_shape(torch.add(gate_mul, 0.5), [1, 8, 16, 960])",
            "        gate_sig = torch.clamp(gate_add, min=0.0, max=1.0)",
            "        _binary_rhs_0, _binary_lhs_0 = _align_binary_inputs_to_anchor(gate_sig, features_cf, [1, 8, 16, 960])",
            "        gated_nhwc = _align_tensor_to_target_shape(torch.mul(_binary_lhs_0, _binary_rhs_0), [1, 8, 16, 960])",
            "        y0 = self.conv_block_62(gated_nhwc.permute(0, 3, 1, 2).contiguous())",
            "        gap = torch.mean(gated_nhwc, dim=[1, 2], keepdim=True)",
            "        ygap = self.conv_block_66(gap.permute(0, 3, 1, 2).contiguous())",
        ]
    ) == [
        "        gate_sig = torch.nn.functional.hardsigmoid(features_cf)",
        "        gated_nhwc = torch.mul(features_cf, gate_sig)",
        "        y0 = self.conv_block_62(gated_nhwc)",
        "        gap = torch.mean(gated_nhwc, dim=[2, 3], keepdim=True)",
        "        ygap = self.conv_block_66(gap)",
    ]


def test_channel_last_binary_bridge_rewrite_folds_conv_input_chain() -> None:
    assert _rewrite_channel_last_binary_bridge_chains(
        [
            "in_public_layout_bridge = _torch_permute(in_t, [0, 2, 3, 1])",
            "_binary_lhs_1, _binary_rhs_1 = _align_binary_inputs(in_public_layout_bridge, self.const_tensor_623_nhwc, [1, 3, 64, 64])",
            "cv65_in = _align_tensor_to_target_shape(torch.sub(_binary_lhs_1, _binary_rhs_1), [1, 3, 64, 64])",
            "cv65_out_nhwc_cf = self.conv_block_0(cv65_in.permute(0, 3, 1, 2).contiguous())",
        ],
        derive_local_var_name=lambda base_name: f"{base_name}_tmp",
        channel_first_constant_expr_for_buffer_attr=lambda buffer_expr, target_shape: (
            "self.const_tensor623_nhwc_ch_first1_x3_x1_x1"
            if buffer_expr == "self.const_tensor_623_nhwc"
            and list(target_shape) == [1, 3, 1, 1]
            else None
        ),
    ) == [
        "cv65_in_cf_tmp = torch.sub(in_t, self.const_tensor623_nhwc_ch_first1_x3_x1_x1)",
        "cv65_out_nhwc_cf = self.conv_block_0(cv65_in_cf_tmp)",
    ]


def test_channel_last_gap_conv_repair_inserts_channel_first_input_bridge() -> None:
    assert _repair_channel_last_gap_conv_inputs(
        [
            "        gap = torch.mean(features, dim=[1, 2], keepdim=True)",
            "        y = self.conv_block_66(gap)",
        ]
    ) == [
        "        gap = torch.mean(features, dim=[1, 2], keepdim=True)",
        "        y = self.conv_block_66(gap.permute(0, 3, 1, 2).contiguous())",
    ]


def test_channel_last_gap_conv_repair_ignores_scalar_dim_mean() -> None:
    lines = [
        "        gap = torch.mean(features, dim=1, keepdim=True)",
        "        y = self.conv_block_66(gap)",
    ]

    assert _repair_channel_last_gap_conv_inputs(lines) == lines


def test_dead_forward_line_pruning_removes_unreachable_assignment() -> None:
    assert _prune_dead_forward_lines(
        [
            "x = torch.relu(input0)",
            "dead = torch.neg(input0)",
            "y = torch.add(x, input0)",
        ],
        input_var_names=["input0"],
        output_var_names=["y"],
    ) == [
        "x = torch.relu(input0)",
        "y = torch.add(x, input0)",
    ]


def test_dead_forward_line_pruning_keeps_multi_output_dependencies() -> None:
    lines = [
        "pair = torch.chunk(input0, 2)",
        "left, right = pair",
        "y = torch.add(left, right)",
    ]

    assert (
        _prune_dead_forward_lines(
            lines,
            input_var_names=["input0"],
            output_var_names=["y"],
        )
        == lines
    )


@pytest.mark.parametrize(
    "rewrite",
    [
        _collapse_redundant_torch_permute_chains,
        _fold_boundary_transpose_pad_conv_bridges,
        _fold_channel_first_gap_conv_bridges,
        _fold_channel_first_hardsigmoid_gate_conv_bridges,
        _fold_channel_last_prelu_bridges,
        _fold_rank4_reshape_permute_conv_bridges,
        _inline_trivial_public_layout_bridge_aliases,
        _repair_channel_last_gap_conv_inputs,
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


def test_channel_last_binary_bridge_rewrite_preserves_unmatched_source() -> None:
    lines = ["y = torch.relu(x)", "return y"]

    assert (
        _rewrite_channel_last_binary_bridge_chains(
            lines,
            derive_local_var_name=lambda base_name: base_name,
            channel_first_constant_expr_for_buffer_attr=lambda _buffer, _shape: None,
        )
        == lines
    )
