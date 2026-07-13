from __future__ import annotations

from onnx2tf.tflite_builder.pytorch_source_parser import (
    _normalize_permute_dims_expr,
    _parse_align_tensor_target_shape_expr,
    _parse_apply_concat_inputs_axis_and_shape,
    _parse_binary_add_args,
    _parse_binary_mul_args,
    _parse_rank4_shape_literal,
    _parse_simple_assignment_line,
    _parse_torch_cat_inputs_and_dim,
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
