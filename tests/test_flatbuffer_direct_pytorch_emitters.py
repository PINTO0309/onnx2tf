from __future__ import annotations

from onnx2tf.tflite_builder.ir import ModelIR, OperatorIR, TensorIR
from onnx2tf.tflite_builder.pytorch_emitters import (
    _emit_native_shape_transform_misc_op_for_codegen,
    _emit_native_unary_op_for_codegen,
)


def _unary_model_ir(*, output_layout: str) -> ModelIR:
    model_ir = ModelIR(name="unary_emitter")
    output_shape = (
        [1, 2, 4, 3] if output_layout == "NHWC" else [1, 3, 2, 4]
    )
    model_ir.tensors = {
        "x": TensorIR(
            "x",
            "FLOAT32",
            [1, 3, 2, 4],
            [1, 3, 2, 4],
            logical_layout="NCHW",
        ),
        "y": TensorIR(
            "y",
            "FLOAT32",
            output_shape,
            output_shape,
            logical_layout=output_layout,
        ),
    }
    return model_ir


def _emit_unary(
    *,
    model_ir: ModelIR,
    op: OperatorIR,
    aliases: dict[str, str],
    runtime_imports: set[str],
    forward_lines: list[str],
    can_emit_channel_first: bool = True,
    channel_first_input_expr: str | None = "x_cf",
    skip_alignment: bool = True,
) -> bool:
    return _emit_native_unary_op_for_codegen(
        model_ir=model_ir,
        op=op,
        outputs=["y"],
        output_vars=["y_var"],
        channel_first_tensor_expr_aliases=aliases,
        runtime_imports=runtime_imports,
        forward_lines=forward_lines,
        tensor_expr_fn=lambda name: f"tensor_{name}",
        channel_first_passthrough_input_expr_fn=(
            lambda _name: channel_first_input_expr
        ),
        can_emit_channel_first_shape_preserving_unary_op_fn=(
            lambda _op: can_emit_channel_first
        ),
        derived_local_var_name_fn=lambda _name, _prefix: "y_var_cf_0",
        can_omit_materialized_channel_last_alias_fn=lambda _name: False,
        target_shape_literal_fn=lambda _name: "[1, 2, 4, 3]",
        tensor_shape_list_fn=lambda _name: [1, 3, 2, 4],
        should_skip_align_for_shape_preserving_unary_fn=(
            lambda _input, _output: skip_alignment
        ),
        emit_maybe_aligned_expr_fn=(
            lambda *, output_name, expr, inferred_shape: (
                f"aligned({output_name}, {expr}, {inferred_shape})"
            )
        ),
    )


def test_unary_emitter_preserves_channel_first_expression() -> None:
    model_ir = _unary_model_ir(output_layout="NCHW")
    aliases = {"y": "stale_alias"}
    runtime_imports: set[str] = set()
    forward_lines: list[str] = []

    emitted = _emit_unary(
        model_ir=model_ir,
        op=OperatorIR("RELU", ["x"], ["y"]),
        aliases=aliases,
        runtime_imports=runtime_imports,
        forward_lines=forward_lines,
    )

    assert emitted is True
    assert forward_lines == ["y_var = torch.relu(x_cf)"]
    assert aliases == {}
    assert runtime_imports == set()


def test_unary_emitter_materializes_channel_last_layout_bridge() -> None:
    model_ir = _unary_model_ir(output_layout="NHWC")
    aliases: dict[str, str] = {}
    runtime_imports: set[str] = set()
    forward_lines: list[str] = []

    emitted = _emit_unary(
        model_ir=model_ir,
        op=OperatorIR("RELU", ["x"], ["y"]),
        aliases=aliases,
        runtime_imports=runtime_imports,
        forward_lines=forward_lines,
    )

    assert emitted is True
    assert forward_lines == [
        "y_var_cf_0 = torch.relu(x_cf)",
        "y_var = _align_tensor_to_target_shape("
        "y_var_cf_0.permute(0, 2, 3, 1).contiguous(), [1, 2, 4, 3])",
    ]
    assert aliases == {"y": "y_var_cf_0"}
    assert runtime_imports == {"_align_tensor_to_target_shape"}


def test_unary_emitter_uses_fallback_alignment_callback() -> None:
    model_ir = _unary_model_ir(output_layout="NCHW")
    aliases = {"y": "stale_alias"}
    runtime_imports: set[str] = set()
    forward_lines: list[str] = []

    emitted = _emit_unary(
        model_ir=model_ir,
        op=OperatorIR("LEAKY_RELU", ["x"], ["y"], {"alpha": 0.125}),
        aliases=aliases,
        runtime_imports=runtime_imports,
        forward_lines=forward_lines,
        can_emit_channel_first=False,
        channel_first_input_expr=None,
        skip_alignment=False,
    )

    assert emitted is True
    assert forward_lines == [
        "y_var = aligned(y, F.leaky_relu(tensor_x, negative_slope=0.125), "
        "[1, 3, 2, 4])"
    ]
    assert aliases == {}
    assert runtime_imports == set()


def test_unary_emitter_rejects_unsupported_op_without_mutation() -> None:
    model_ir = _unary_model_ir(output_layout="NCHW")
    aliases = {"existing": "alias"}
    runtime_imports = {"existing_runtime_helper"}
    forward_lines = ["existing_line"]

    emitted = _emit_unary(
        model_ir=model_ir,
        op=OperatorIR("CUSTOM", ["x"], ["y"]),
        aliases=aliases,
        runtime_imports=runtime_imports,
        forward_lines=forward_lines,
    )

    assert emitted is False
    assert aliases == {"existing": "alias"}
    assert runtime_imports == {"existing_runtime_helper"}
    assert forward_lines == ["existing_line"]


def _emit_shape_transform(
    *,
    model_ir: ModelIR,
    op: OperatorIR,
    output_vars: list[str],
    runtime_imports: set[str],
    forward_lines: list[str],
) -> bool:
    return _emit_native_shape_transform_misc_op_for_codegen(
        model_ir=model_ir,
        op=op,
        op_type=str(op.op_type),
        outputs=[str(name) for name in op.outputs],
        output_vars=output_vars,
        runtime_imports=runtime_imports,
        forward_lines=forward_lines,
        tensor_expr_fn=lambda name: f"tensor_{name}",
        axis_expr_from_input_fn=(
            lambda name, *, device_expr: f"axis({name}, {device_expr})"
        ),
    )


def test_shape_transform_emitter_normalizes_constant_reverse_axes() -> None:
    import numpy as np

    model_ir = ModelIR(name="reverse_emitter")
    model_ir.tensors = {
        "x": TensorIR("x", "FLOAT32", [1, 2, 3, 4]),
        "axes": TensorIR(
            "axes",
            "INT64",
            [2],
            data=np.asarray([-1, 1], dtype=np.int64),
        ),
    }
    runtime_imports: set[str] = set()
    forward_lines: list[str] = []

    emitted = _emit_shape_transform(
        model_ir=model_ir,
        op=OperatorIR("REVERSE_V2", ["x", "axes"], ["y"]),
        output_vars=["y_var"],
        runtime_imports=runtime_imports,
        forward_lines=forward_lines,
    )

    assert emitted is True
    assert forward_lines == ["y_var = torch.flip(tensor_x, dims=[3, 1])"]
    assert runtime_imports == set()


def test_shape_transform_emitter_preserves_axis_and_output_contracts() -> None:
    model_ir = ModelIR(name="shape_transform_emitter")
    model_ir.tensors = {
        "x": TensorIR("x", "FLOAT32", [1, 2, 3]),
        "axis": TensorIR("axis", "INT32", [1]),
    }
    cases = [
        (
            OperatorIR("EXPAND_DIMS", ["x", "axis"], ["expanded"]),
            ["expanded_var"],
            [
                "expanded_var = torch.unsqueeze("
                "tensor_x, dim=axis(axis, tensor_x))"
            ],
            set(),
        ),
        (
            OperatorIR(
                "SQUEEZE",
                ["x"],
                ["squeezed"],
                {"squeezeDims": [0, -1]},
            ),
            ["squeezed_var"],
            [
                "squeezed_var = tensor_x",
                "squeezed_var = torch.squeeze(squeezed_var, "
                "dim=_normalize_dim(0, squeezed_var.ndim))",
                "squeezed_var = torch.squeeze(squeezed_var, "
                "dim=_normalize_dim(-1, squeezed_var.ndim))",
            ],
            {"_normalize_dim"},
        ),
        (
            OperatorIR("PACK", ["x", "x"], ["packed"], {"axis": 1}),
            ["packed_var"],
            ["packed_var = torch.stack([tensor_x, tensor_x], dim=1)"],
            set(),
        ),
        (
            OperatorIR("UNPACK", ["x"], ["a", "b"], {"axis": -1}),
            ["a_var", "b_var"],
            [
                "a_var, b_var = list(torch.unbind(tensor_x, "
                "dim=_normalize_dim(-1, tensor_x.ndim)))"
            ],
            {"_normalize_dim"},
        ),
        (
            OperatorIR(
                "SPLIT",
                ["axis", "x"],
                ["a", "b"],
                {"numSplits": 2},
            ),
            ["a_var", "b_var"],
            [
                "a_var, b_var = list(torch.tensor_split(tensor_x, 2, "
                "dim=_normalize_dim(axis(axis, tensor_x), tensor_x.ndim)))"
            ],
            {"_normalize_dim"},
        ),
    ]

    for op, output_vars, expected_lines, expected_imports in cases:
        runtime_imports: set[str] = set()
        forward_lines: list[str] = []
        emitted = _emit_shape_transform(
            model_ir=model_ir,
            op=op,
            output_vars=output_vars,
            runtime_imports=runtime_imports,
            forward_lines=forward_lines,
        )
        assert emitted is True
        assert forward_lines == expected_lines
        assert runtime_imports == expected_imports
