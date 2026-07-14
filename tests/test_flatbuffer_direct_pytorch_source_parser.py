from __future__ import annotations

from onnx2tf.tflite_builder.pytorch_source_parser import (
    _any_line_matches,
    _count_lines_matching,
    _extract_prefixed_call_exprs,
    _model_source_lines,
    _normalize_permute_dims_expr,
    _parse_aligned_binary_assign_with_shape,
    _parse_apply_resize_assign,
    _parse_apply_softmax_assign,
    _parse_dynamic_apply_pool2d_assign,
    _parse_local_response_norm_assign,
    _parse_reduce_max_assign,
    _parse_simple_return_identifier,
    _parse_align_binary_inputs_to_anchor_assign_with_shape,
    _parse_align_tensor_target_shape_assign,
    _parse_align_tensor_target_shape_expr,
    _parse_aligned_rank4_assign,
    _parse_apply_concat_inputs_axis_and_shape,
    _parse_apply_pool2d_assign_with_shape,
    _parse_apply_pool2d_input_and_channel_last,
    _parse_apply_pool2d_input_expr,
    _parse_apply_resize_input_and_channel_last,
    _parse_apply_resize_input_size_shape_and_channel_last,
    _parse_apply_softmax_input_and_axis,
    _parse_apply_softmax_input_axis_and_shape,
    _parse_binary_add_args,
    _parse_binary_mul_args,
    _parse_binary_sub_args,
    _parse_channel_last_gather_slice_assign,
    _parse_constant_pad_assign,
    _parse_copy_call_expr,
    _parse_dynamic_binary_add_align_assign,
    _parse_dynamic_binary_align_assign,
    _parse_local_response_norm_input_expr,
    _parse_permuted_conv_input_assign,
    _parse_rank4_shape_expr,
    _parse_rank4_shape_literal,
    _parse_simple_assignment_line,
    _parse_static_binary_add_align_assign,
    _parse_tensor_split_assign,
    _parse_torch_cat_inputs_and_dim,
    _parse_torch_permute_assign,
    _resolve_nhwc_to_nchw_bridge_source,
    _split_top_level_csv_exprs,
    _strip_outer_parentheses,
)


def test_source_parser_preserves_nested_csv_expressions() -> None:
    assert _split_top_level_csv_exprs(
        "x, torch.add(y, 1), [1, 2], {'axis': (0, 3)}"
    ) == ["x", "torch.add(y, 1)", "[1, 2]", "{'axis': (0, 3)}"]


def test_source_parser_decodes_binary_and_alignment_arguments() -> None:
    assert _parse_binary_add_args("x, torch.relu(y)") == (
        "x",
        "torch.relu(y)",
    )
    assert _parse_binary_mul_args("input=x, other=y") == ("x", "y")
    assert _parse_binary_sub_args("1.0, torch.sigmoid(mask)") == (
        "1.0",
        "torch.sigmoid(mask)",
    )
    assert _parse_binary_sub_args("input=1.0, other=mask") == ("1.0", "mask")
    assert _parse_binary_sub_args("input=1.0") is None
    assert _parse_align_tensor_target_shape_expr(
        "_align_tensor_to_target_shape(input=x, target_shape=[1, 2, 3, 4])"
    ) == ("x", "[1, 2, 3, 4]")


def test_source_parser_decodes_assignments_shapes_and_concat() -> None:
    assert _parse_simple_assignment_line("    value: torch.Tensor = torch.relu(x)") == (
        "    ",
        "value",
        "torch.relu(x)",
    )
    assert _parse_rank4_shape_literal("(1, 2, 3, 4)") == (1, 2, 3, 4)
    assert _parse_apply_concat_inputs_axis_and_shape(
        "_apply_concat([x, torch.relu(y)], axis=3, target_shape=[1, 2, 3, 4])"
    ) == (["x", "torch.relu(y)"], 3, [1, 2, 3, 4])
    assert _parse_apply_concat_inputs_axis_and_shape(
        "_apply_concat([boxes, scores], axis=2, target_shape=[1, -1, +6])"
    ) == (["boxes", "scores"], 2, [1, -1, 6])
    assert _parse_torch_cat_inputs_and_dim("torch.cat(tensors=[x, y], dim=1)") == (
        ["x", "y"],
        1,
    )
    assert _parse_permuted_conv_input_assign(
        "    y = self.conv_block_0(x.permute(0, 3, 1, 2).contiguous())"
    ) == ("    ", "y", "conv_block_0", "x")
    assert (
        _parse_permuted_conv_input_assign(
            "    y: torch.Tensor = "
            "self.conv_block_0(x.permute(0, 3, 1, 2).contiguous())"
        )
        is None
    )
    assert _parse_aligned_rank4_assign(
        "    y = _align_tensor_to_target_shape(torch.add(x, z), [1, 2, 3, 4])"
    ) == ("    ", "y", "torch.add(x, z)", [1, 2, 3, 4])


def test_source_parser_normalizes_only_complete_outer_syntax() -> None:
    assert _strip_outer_parentheses("(((x + y)))") == "x + y"
    assert _strip_outer_parentheses("(x) + (y)") == "(x) + (y)"
    assert _normalize_permute_dims_expr("(0, 2, 3, 1)") == "0,2,3,1"


def test_source_parser_decodes_resize_softmax_and_dynamic_batch() -> None:
    dynamic_batch = "_shape_list(x)[0]"
    assert _parse_rank4_shape_expr(f"[{dynamic_batch}, 8, 9, 3]") == (
        dynamic_batch,
        8,
        9,
        3,
    )
    assert _parse_apply_resize_input_size_shape_and_channel_last(
        f"_apply_resize(x, size=[8, 9], "
        f"target_shape=[{dynamic_batch}, 8, 9, 3], channel_last=True)"
    ) == ("x", (8, 9), (dynamic_batch, 8, 9, 3), True)
    assert _parse_apply_softmax_input_axis_and_shape(
        "_apply_softmax(input=x, axis=3, target_shape=[1, 2, 3, 4])"
    ) == ("x", 3, ("1", 2, 3, 4))


def test_source_parser_decodes_pool_and_tensor_split_assignments() -> None:
    assert _parse_apply_pool2d_assign_with_shape(
        "    y = _apply_pool2d(input=x, kernel_size=[2, 2], "
        "stride=[2, 2], padding=[0, 0], is_max_pool=False, "
        "channel_last=True, target_shape=[1, 4, 4, 3])"
    ) == (
        "    ",
        "y",
        "x",
        "kernel_size=[2, 2], stride=[2, 2], padding=[0, 0]",
        [1, 4, 4, 3],
        False,
        True,
    )
    assert _parse_tensor_split_assign(
        "a, b = list(torch.tensor_split(x, 2, dim=_normalize_dim(-1, x.ndim)))"
    ) == ("", ["a", "b"], "x", 2, -1)


def test_source_parser_decodes_gather_slice_and_nchw_bridge() -> None:
    assert _parse_channel_last_gather_slice_assign("y = x[:, :, :, [0, 2]]") == (
        "y",
        "x",
        "0, 2",
    )
    assert (
        _resolve_nhwc_to_nchw_bridge_source(
            "_torch_permute(input=x, dims=[0, 3, 1, 2]).contiguous()"
        )
        == "x"
    )
    assert (
        _resolve_nhwc_to_nchw_bridge_source("x.permute(0, 2, 3, 1).contiguous()")
        is None
    )


def test_source_parser_decodes_copy_alignment_and_permute_assignments() -> None:
    assert _parse_copy_call_expr(
        "    self.buf.copy_(torch.relu(x), non_blocking=True)"
    ) == (
        "    ",
        "self.buf",
        "buf",
        "torch.relu(x)",
        ", non_blocking=True",
    )
    assert _parse_align_tensor_target_shape_assign(
        "y = _align_tensor_to_target_shape(x, [1,2,3,4])"
    ) == ("x", "[1,2,3,4]")
    assert _parse_torch_permute_assign(
        "    y = _torch_permute(input=x, dims=[0, 2, 3, 1]).contiguous()"
    ) == ("    ", "y", "x", [0, 2, 3, 1])


def test_source_parser_decodes_runtime_helper_inputs() -> None:
    assert (
        _parse_local_response_norm_input_expr("F.local_response_norm(input=x, size=5)")
        == "x"
    )
    assert (
        _parse_apply_pool2d_input_expr("_apply_pool2d(input=x, channel_last=True)")
        == "x"
    )
    assert _parse_apply_resize_input_and_channel_last(
        "_apply_resize(input=x, channel_last=False)"
    ) == ("x", False)
    assert _parse_apply_pool2d_input_and_channel_last(
        "_apply_pool2d(x, channel_last=True)"
    ) == ("x", True)
    assert _parse_apply_softmax_input_and_axis("_apply_softmax(input=x, axis=-1)") == (
        "x",
        -1,
    )


def test_source_parser_decodes_constant_pad_and_binary_alignment() -> None:
    assert _parse_constant_pad_assign(
        "y = F.pad(x, [1, 2, 3, 4], 'constant', 0.0)"
    ) == ("", "y", "x", [1, 2, 3, 4], "0.0")
    assert _parse_dynamic_binary_add_align_assign(
        "y = _align_tensor_to_target_shape(torch.add(a, b), "
        "[int(ref.shape[0]), 8, int(ref.shape[2]), int(ref.shape[3])])"
    ) == ("", "y", "a", "b", 8)
    assert _parse_dynamic_binary_align_assign(
        "y = _align_tensor_to_target_shape(torch.mul(a, b), "
        "[int(ref.shape[0]), 8, int(ref.shape[2]), int(ref.shape[3])])"
    ) == ("", "y", "mul", "a", "b", 8)
    assert _parse_static_binary_add_align_assign(
        "y = _align_tensor_to_target_shape(torch.add(a, b), [1, 8, 4, 4])"
    ) == ("", "y", "a", "b", [1, 8, 4, 4])
    assert _parse_align_binary_inputs_to_anchor_assign_with_shape(
        "a2, b2 = _align_binary_inputs_to_anchor(a, b, [1, 8, 4, 4])"
    ) == ("", "a2", "b2", "a", "b", [1, 8, 4, 4])


def test_source_parser_scans_lines_and_balanced_calls() -> None:
    lines = _model_source_lines("x = 1\ny = torch.add(x, 2)\nz = torch.add(y, 3)\n")
    assert lines == [
        "x = 1",
        "y = torch.add(x, 2)",
        "z = torch.add(y, 3)",
    ]
    assert _any_line_matches(lines, r"torch\.add")
    assert _count_lines_matching(lines, r"torch\.add") == 2
    assert _extract_prefixed_call_exprs(
        "lhs + _apply_resize(x, size=(4, 5)) + "
        "_apply_resize(torch.add(y, 1), size=(8, 9))",
        "_apply_resize(",
    ) == [
        "_apply_resize(x, size=(4, 5))",
        "_apply_resize(torch.add(y, 1), size=(8, 9))",
    ]


def test_source_parser_decodes_precanonicalize_repair_statements() -> None:
    assert _parse_aligned_binary_assign_with_shape(
        "y = _align_tensor_to_target_shape(torch.add(a, b), [1, 8, 4, 4])"
    ) == ("y", "add", "a", "b", [1, 8, 4, 4])
    assert _parse_simple_return_identifier("return (y)") == "y"
    assert _parse_dynamic_apply_pool2d_assign(
        "y = _apply_pool2d(input=x, filter_height=2, filter_width=2, "
        "target_shape=_tensor_shape_list(ref), is_max_pool=True, channel_last=False)"
    ) == ("", "y", "x", "filter_height=2, filter_width=2", "ref", True)
    assert _parse_local_response_norm_assign(
        "y = F.local_response_norm(input=x, size=5, alpha=0.0001, beta=0.75, k=1.0)"
    ) == ("", "y", "x", "5", "0.0001", "0.75", "1.0")
    assert _parse_apply_softmax_assign(
        "y = _apply_softmax(input=x, axis=3, beta=1.0, target_shape=[1, 2, 3, 4])"
    ) == ("", "y", "x", 3, "1.0", [1, 2, 3, 4])
    assert _parse_apply_resize_assign(
        "y = _apply_resize(input=x, size=[8, 9], method='bilinear', "
        "target_shape=[1, 8, 9, 3], align_corners=False, "
        "half_pixel_centers=True, channel_last=True)"
    ) == ("", "y", "x", 8, 9, "bilinear", [1, 8, 9, 3], False, True, True)
    assert _parse_reduce_max_assign(
        "y = _reduce_max(input=x, axes=_normalize_axes([3], x.ndim), keepdims=True)"
    ) == ("", "y", "x", 3, "True")
