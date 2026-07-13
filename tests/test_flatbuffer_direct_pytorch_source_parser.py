from __future__ import annotations

from onnx2tf.tflite_builder.pytorch_source_parser import (
    _normalize_permute_dims_expr,
    _parse_align_tensor_target_shape_expr,
    _parse_apply_concat_inputs_axis_and_shape,
    _parse_apply_pool2d_assign_with_shape,
    _parse_apply_resize_input_size_shape_and_channel_last,
    _parse_apply_softmax_input_axis_and_shape,
    _parse_binary_add_args,
    _parse_binary_mul_args,
    _parse_channel_last_gather_slice_assign,
    _parse_rank4_shape_expr,
    _parse_rank4_shape_literal,
    _parse_simple_assignment_line,
    _parse_tensor_split_assign,
    _parse_torch_cat_inputs_and_dim,
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
    assert _parse_torch_cat_inputs_and_dim("torch.cat(tensors=[x, y], dim=1)") == (
        ["x", "y"],
        1,
    )


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
