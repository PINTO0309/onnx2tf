from __future__ import annotations

from onnx2tf.tflite_builder.ir import ModelIR, OperatorIR, TensorIR
from onnx2tf.tflite_builder.pytorch_emitters import (
    _emit_native_binary_op_for_codegen_impl,
    _emit_native_concat_op_for_codegen,
    _emit_native_conv2d_module_op_for_codegen,
    _emit_native_conv3d_module_op_for_codegen,
    _emit_native_fully_connected_module_op_for_codegen,
    _emit_native_prelu_module_op_for_codegen,
    _emit_native_recurrent_module_op_for_codegen,
    _emit_native_shape_transform_misc_op_for_codegen,
    _emit_native_transpose_op_for_codegen,
    _emit_native_transpose_conv2d_module_op_for_codegen,
    _emit_native_transpose_conv3d_module_op_for_codegen,
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


def _emit_binary(
    *,
    model_ir: ModelIR,
    op: OperatorIR,
    aliases: dict[str, str],
    runtime_imports: set[str],
    forward_lines: list[str],
    dtypes: dict[str, str] | None = None,
    scalar_literals: dict[str, str] | None = None,
    can_emit_channel_first: bool = False,
    runtime_passthrough: str | None = None,
    requires_runtime_alignment: bool = False,
    uncertain_tensors: set[str] | None = None,
    preferred_anchor: str | None = None,
) -> bool:
    return _emit_native_binary_op_for_codegen_impl(
        model_ir=model_ir,
        op=op,
        op_index=7,
        outputs=["y"],
        output_vars=["y_var"],
        output_target_shape="[1, 2, 4, 3]",
        channel_first_tensor_expr_aliases=aliases,
        runtime_imports=runtime_imports,
        forward_lines=forward_lines,
        runtime_shape_uncertain_tensors=uncertain_tensors or set(),
        tensor_dtype_name_fn=lambda name: (dtypes or {}).get(name),
        binary_operand_expr_fn=lambda name, other: f"expr_{name}_vs_{other}",
        scalar_literal_expr_fn=lambda name: (scalar_literals or {}).get(name),
        can_emit_channel_first_binary_op_fn=(
            lambda _op: can_emit_channel_first
        ),
        channel_first_binary_input_expr_fn=(
            lambda name, _other: f"cf_{name}"
        ),
        derived_local_var_name_fn=lambda _name, _prefix: "y_var_cf_0",
        can_omit_materialized_channel_last_alias_fn=lambda _name: False,
        target_shape_literal_fn=lambda _name: "[1, 2, 4, 3]",
        emit_maybe_aligned_expr_fn=(
            lambda *, output_name, expr, inferred_shape: (
                f"aligned({output_name}, {expr}, {inferred_shape})"
            )
        ),
        binary_runtime_shape_passthrough_operand_fn=(
            lambda _lhs, _rhs: runtime_passthrough
        ),
        binary_requires_runtime_alignment_fn=(
            lambda _lhs, _rhs, _output: requires_runtime_alignment
        ),
        preferred_binary_alignment_anchor_fn=(
            lambda _lhs, _rhs, _output: preferred_anchor
        ),
        activation_lines_fn=(
            lambda name, fused: []
            if fused == "NONE"
            else [f"activate({name}, {fused})"]
        ),
        binary_output_target_shape_literal_fn=(
            lambda **_kwargs: "[9, 8]"
        ),
    )


def _binary_model_ir(*, output_layout: str = "UNKNOWN") -> ModelIR:
    model_ir = ModelIR(name="binary_emitter")
    output_shape = (
        [1, 2, 4, 3] if output_layout == "NHWC" else [1, 3, 2, 4]
    )
    model_ir.tensors = {
        "lhs": TensorIR("lhs", "FLOAT32", [1, 3, 2, 4]),
        "rhs": TensorIR("rhs", "FLOAT32", [1, 3, 2, 4]),
        "y": TensorIR(
            "y",
            "FLOAT32",
            output_shape,
            logical_layout=output_layout,
        ),
    }
    return model_ir


def test_binary_emitter_preserves_integer_division_and_bool_scalar_rules() -> None:
    model_ir = _binary_model_ir()
    cases = [
        (
            OperatorIR("DIV", ["lhs", "rhs"], ["y"]),
            {"lhs": "INT32", "rhs": "INT64"},
            {},
            "y_var = aligned(y, torch.div(expr_lhs_vs_rhs, expr_rhs_vs_lhs, "
            "rounding_mode='trunc'), None)",
        ),
        (
            OperatorIR("EQUAL", ["lhs", "rhs"], ["y"]),
            {},
            {"rhs": "True"},
            "y_var = aligned(y, torch.eq(expr_lhs_vs_rhs, "
            "torch.as_tensor(True, dtype=expr_lhs_vs_rhs.dtype, "
            "device=expr_lhs_vs_rhs.device)), None)",
        ),
    ]

    for op, dtypes, scalar_literals, expected_line in cases:
        aliases = {"y": "stale_alias"}
        runtime_imports: set[str] = set()
        forward_lines: list[str] = []
        emitted = _emit_binary(
            model_ir=model_ir,
            op=op,
            aliases=aliases,
            runtime_imports=runtime_imports,
            forward_lines=forward_lines,
            dtypes=dtypes,
            scalar_literals=scalar_literals,
        )
        assert emitted is True
        assert forward_lines == [expected_line]
        assert aliases == {}
        assert runtime_imports == set()


def test_binary_emitter_materializes_channel_last_bridge_after_activation() -> None:
    model_ir = _binary_model_ir(output_layout="NHWC")
    aliases: dict[str, str] = {}
    runtime_imports: set[str] = set()
    forward_lines: list[str] = []

    emitted = _emit_binary(
        model_ir=model_ir,
        op=OperatorIR(
            "ADD",
            ["lhs", "rhs"],
            ["y"],
            {"fusedActivationFunction": "RELU6"},
        ),
        aliases=aliases,
        runtime_imports=runtime_imports,
        forward_lines=forward_lines,
        can_emit_channel_first=True,
    )

    assert emitted is True
    assert forward_lines == [
        "y_var_cf_0 = torch.add(cf_lhs, cf_rhs)",
        "activate(y_var_cf_0, RELU6)",
        "y_var = _align_tensor_to_target_shape("
        "y_var_cf_0.permute(0, 2, 3, 1).contiguous(), [1, 2, 4, 3])",
    ]
    assert aliases == {"y": "y_var_cf_0"}
    assert runtime_imports == {"_align_tensor_to_target_shape"}


def test_binary_emitter_uses_uncertain_operand_as_alignment_anchor() -> None:
    model_ir = _binary_model_ir()
    aliases = {"y": "stale_alias"}
    runtime_imports: set[str] = set()
    forward_lines: list[str] = []

    emitted = _emit_binary(
        model_ir=model_ir,
        op=OperatorIR("MUL", ["lhs", "rhs"], ["y"]),
        aliases=aliases,
        runtime_imports=runtime_imports,
        forward_lines=forward_lines,
        requires_runtime_alignment=True,
        uncertain_tensors={"lhs"},
    )

    assert emitted is True
    assert forward_lines == [
        "_binary_lhs_7, _binary_rhs_7 = _align_binary_inputs_to_anchor("
        "expr_lhs_vs_rhs, expr_rhs_vs_lhs, [9, 8])",
        "y_var = aligned(y, torch.mul(_binary_lhs_7, _binary_rhs_7), None)",
    ]
    assert aliases == {}
    assert runtime_imports == {"_align_binary_inputs_to_anchor"}


def _emit_transpose(
    *,
    model_ir: ModelIR,
    op: OperatorIR,
    aliases: dict[str, str],
    runtime_imports: set[str],
    forward_lines: list[str],
    preserve_names: set[str] | None = None,
    folded_expr: str | None = None,
    stale_hint: bool = False,
    binary_consumers: bool = False,
) -> bool:
    return _emit_native_transpose_op_for_codegen(
        model_ir=model_ir,
        op=op,
        outputs=[str(name) for name in op.outputs],
        output_vars=["y_var"],
        preserve_channel_last_tensor_names=preserve_names or set(),
        consumer_index={},
        producer_index={},
        channel_first_tensor_expr_aliases=aliases,
        runtime_imports=runtime_imports,
        forward_lines=forward_lines,
        tensor_expr_fn=lambda name: f"tensor_{name}",
        tensor_expr_for_channel_first_bridge_fn=(
            lambda _name, _perm: folded_expr
        ),
        can_fold_channel_last_alias_slice_consumer_fn=(
            lambda _op, *, expected_input_name: False
        ),
        all_consumers_are_channel_first_binary_ops_fn=(
            lambda _name: binary_consumers
        ),
        can_omit_materialized_channel_last_alias_fn=lambda _name: False,
        has_channel_last_consumer_hint_for_same_shape_transpose_fn=(
            lambda _op: stale_hint
        ),
        is_batchless_rank3_public_output_transpose_fn=lambda _op: False,
        target_shape_literal_fn=lambda _name: "[1, 2, 4, 3]",
    )


def _transpose_model_ir(
    *,
    input_layout: str = "UNKNOWN",
    output_layout: str = "UNKNOWN",
) -> ModelIR:
    model_ir = ModelIR(name="transpose_emitter")
    model_ir.tensors = {
        "x": TensorIR(
            "x",
            "FLOAT32",
            [1, 3, 2, 4],
            logical_layout=input_layout,
        ),
        "y": TensorIR(
            "y",
            "FLOAT32",
            [1, 2, 4, 3],
            logical_layout=output_layout,
        ),
    }
    return model_ir


def test_transpose_emitter_preserves_elision_and_folded_alias_paths() -> None:
    model_ir = _transpose_model_ir()
    op = OperatorIR("TRANSPOSE", ["x"], ["y"], {"perm": [0, 2, 3, 1]})

    runtime_imports: set[str] = set()
    forward_lines: list[str] = []
    aliases: dict[str, str] = {}
    emitted = _emit_transpose(
        model_ir=model_ir,
        op=op,
        aliases=aliases,
        runtime_imports=runtime_imports,
        forward_lines=forward_lines,
        preserve_names={"x"},
        stale_hint=True,
    )
    assert emitted is True
    assert forward_lines == ["y_var = tensor_x"]
    assert aliases == {}
    assert runtime_imports == set()

    forward_lines.clear()
    emitted = _emit_transpose(
        model_ir=model_ir,
        op=op,
        aliases=aliases,
        runtime_imports=runtime_imports,
        forward_lines=forward_lines,
        preserve_names={"x"},
        folded_expr="folded_x_cf",
    )
    assert emitted is True
    assert forward_lines == ["y_var = folded_x_cf"]
    assert aliases == {"y": "y_var"}
    assert runtime_imports == set()


def test_transpose_emitter_preserves_alias_only_and_runtime_paths() -> None:
    channel_model_ir = _transpose_model_ir(
        input_layout="NCHW",
        output_layout="NHWC",
    )
    op = OperatorIR("TRANSPOSE", ["x"], ["y"], {"perm": [0, 2, 3, 1]})
    runtime_imports: set[str] = set()
    forward_lines: list[str] = []
    aliases: dict[str, str] = {}

    emitted = _emit_transpose(
        model_ir=channel_model_ir,
        op=op,
        aliases=aliases,
        runtime_imports=runtime_imports,
        forward_lines=forward_lines,
        preserve_names={"x"},
        binary_consumers=True,
    )
    assert emitted is True
    assert forward_lines == []
    assert aliases == {"y": "tensor_x"}
    assert runtime_imports == set()

    runtime_model_ir = _transpose_model_ir()
    aliases.clear()
    emitted = _emit_transpose(
        model_ir=runtime_model_ir,
        op=op,
        aliases=aliases,
        runtime_imports=runtime_imports,
        forward_lines=forward_lines,
        preserve_names={"x"},
    )
    assert emitted is True
    assert forward_lines == [
        "y_var = _torch_permute(tensor_x, [0, 2, 3, 1])"
    ]
    assert aliases == {}
    assert runtime_imports == {"_shape_list", "_torch_permute"}


def _emit_concat(
    *,
    model_ir: ModelIR,
    op: OperatorIR,
    aliases: dict[str, str],
    runtime_imports: set[str],
    forward_lines: list[str],
    channel_first_spec: tuple[int, list[int], list[int]] | None,
    stored_shape: list[int] | None,
    exact_shape: list[int] | None = None,
    target_shape: list[int] | None = None,
) -> bool:
    return _emit_native_concat_op_for_codegen(
        model_ir=model_ir,
        op=op,
        op_index=5,
        outputs=["y"],
        output_vars=["y_var"],
        output_target_shape="[1, 2, 4, 6]",
        channel_first_tensor_expr_aliases=aliases,
        runtime_imports=runtime_imports,
        forward_lines=forward_lines,
        tensor_expr_fn=lambda name: f"tensor_{name}",
        derived_local_var_name_fn=lambda _name, _prefix: "y_var_cf_0",
        activation_lines_fn=(
            lambda name, fused: []
            if fused == "NONE"
            else [f"activate({name}, {fused})"]
        ),
        resolve_concat_axis_for_channel_first_fn=(
            lambda _op: channel_first_spec
        ),
        channel_first_concat_input_expr_fn=lambda name: f"cf_{name}",
        tensor_shape_list_fn=lambda _name: stored_shape,
        can_omit_materialized_channel_last_alias_fn=lambda _name: False,
        target_shape_literal_fn=lambda _name: "[1, 2, 4, 6]",
        tensor_exact_static_shape_list_fn=lambda _name: exact_shape,
        target_shape_values_fn=lambda _name: target_shape,
    )


def _concat_model_ir(*, output_layout: str) -> tuple[ModelIR, OperatorIR]:
    model_ir = ModelIR(name="concat_emitter")
    model_ir.tensors = {
        "lhs": TensorIR("lhs", "FLOAT32", [1, 2, 4, 3]),
        "rhs": TensorIR("rhs", "FLOAT32", [1, 2, 4, 3]),
        "y": TensorIR(
            "y",
            "FLOAT32",
            [1, 2, 4, 6],
            logical_layout=output_layout,
        ),
    }
    op = OperatorIR("CONCATENATION", ["lhs", "rhs"], ["y"], {"axis": 3})
    model_ir.operators = [op]
    return model_ir, op


def test_concat_emitter_materializes_channel_first_output_bridge() -> None:
    model_ir, op = _concat_model_ir(output_layout="NHWC")
    aliases: dict[str, str] = {}
    runtime_imports: set[str] = set()
    forward_lines: list[str] = []

    emitted = _emit_concat(
        model_ir=model_ir,
        op=op,
        aliases=aliases,
        runtime_imports=runtime_imports,
        forward_lines=forward_lines,
        channel_first_spec=(1, [1, 6, 2, 4], [0, 2, 3, 1]),
        stored_shape=[1, 2, 4, 6],
    )

    assert emitted is True
    assert forward_lines == [
        "y_var_cf_0 = torch.cat([cf_lhs, cf_rhs], dim=1)",
        "y_var = _align_tensor_to_target_shape("
        "y_var_cf_0.permute(0, 2, 3, 1).contiguous(), [1, 2, 4, 6])",
    ]
    assert aliases == {"y": "y_var_cf_0"}
    assert runtime_imports == {"_align_tensor_to_target_shape"}


def test_concat_emitter_keeps_channel_axis_sensitive_consumer_layout() -> None:
    model_ir, op = _concat_model_ir(output_layout="NHWC")
    model_ir.operators.append(
        OperatorIR("GATHER", ["y", "indices"], ["gathered"], {"axis": 3})
    )
    aliases = {"y": "stale_alias"}
    runtime_imports: set[str] = set()
    forward_lines: list[str] = []

    emitted = _emit_concat(
        model_ir=model_ir,
        op=op,
        aliases=aliases,
        runtime_imports=runtime_imports,
        forward_lines=forward_lines,
        channel_first_spec=(1, [1, 6, 2, 4], [0, 2, 3, 1]),
        stored_shape=[1, 2, 4, 6],
    )

    assert emitted is True
    assert forward_lines == [
        "y_var = _apply_concat([tensor_lhs, tensor_rhs], axis=3, "
        "target_shape=[1, 2, 4, 6], fused='NONE')"
    ]
    assert aliases == {}
    assert runtime_imports == {"_apply_concat"}


def test_concat_emitter_preserves_exact_shape_fallback() -> None:
    model_ir, op = _concat_model_ir(output_layout="UNKNOWN")
    aliases = {"y": "stale_alias"}
    runtime_imports: set[str] = set()
    forward_lines: list[str] = []

    emitted = _emit_concat(
        model_ir=model_ir,
        op=op,
        aliases=aliases,
        runtime_imports=runtime_imports,
        forward_lines=forward_lines,
        channel_first_spec=None,
        stored_shape=None,
        exact_shape=[1, 48],
        target_shape=[1, 2, 4, 6],
    )

    assert emitted is True
    assert forward_lines == [
        "y_var = torch.reshape(_apply_concat([tensor_lhs, tensor_rhs], "
        "axis=3, target_shape=[1, 2, 4, 6], fused='NONE'), [1, 48])"
    ]
    assert aliases == {}
    assert runtime_imports == {"_apply_concat"}


def test_recurrent_module_emitter_preserves_state_argument_contracts() -> None:
    cases = [
        (
            OperatorIR(
                "UNIDIRECTIONAL_SEQUENCE_RNN",
                [f"rnn_i{index}" for index in range(5)],
                ["y"],
            ),
            "self.recurrent_0(expr_rnn_i0, expr_rnn_i4)",
        ),
        (
            OperatorIR(
                "UNIDIRECTIONAL_SEQUENCE_LSTM",
                [f"lstm_i{index}" for index in range(15)],
                ["y"],
            ),
            "self.recurrent_0(expr_lstm_i0, expr_lstm_i13, expr_lstm_i14)",
        ),
        (
            OperatorIR(
                "BIDIRECTIONAL_SEQUENCE_LSTM",
                [f"bilstm_i{index}" for index in range(29)],
                ["y"],
            ),
            "self.recurrent_0(expr_bilstm_i0, expr_bilstm_i25, "
            "expr_bilstm_i26, expr_bilstm_i27, expr_bilstm_i28)",
        ),
    ]

    for op, expected_call in cases:
        runtime_imports: set[str] = set()
        forward_lines: list[str] = []
        emitted = _emit_native_recurrent_module_op_for_codegen(
            op=op,
            op_type=str(op.op_type),
            attr_name="recurrent_0",
            output_vars=["y_var"],
            output_target_shape="[1, 2, 3]",
            runtime_imports=runtime_imports,
            forward_lines=forward_lines,
            tensor_expr_fn=lambda name: f"expr_{name}",
        )
        assert emitted is True
        assert forward_lines == [
            f"y_var = _align_tensor_to_target_shape({expected_call}, [1, 2, 3])"
        ]
        assert runtime_imports == {"_align_tensor_to_target_shape"}


def test_recurrent_module_emitter_rejects_unrelated_op_without_mutation() -> None:
    runtime_imports = {"existing_helper"}
    forward_lines = ["existing_line"]
    op = OperatorIR("FULLY_CONNECTED", ["x"], ["y"])

    emitted = _emit_native_recurrent_module_op_for_codegen(
        op=op,
        op_type=str(op.op_type),
        attr_name="linear_0",
        output_vars=["y_var"],
        output_target_shape="[1, 2]",
        runtime_imports=runtime_imports,
        forward_lines=forward_lines,
        tensor_expr_fn=lambda name: f"expr_{name}",
    )

    assert emitted is False
    assert runtime_imports == {"existing_helper"}
    assert forward_lines == ["existing_line"]


def test_fully_connected_module_emitter_preserves_fused_activation_order() -> None:
    forward_lines: list[str] = []
    op = OperatorIR(
        "FULLY_CONNECTED",
        ["x", "weight", "bias"],
        ["y"],
        {"fusedActivationFunction": "RELU6"},
    )

    emitted = _emit_native_fully_connected_module_op_for_codegen(
        op=op,
        op_type=str(op.op_type),
        attr_name="linear_0",
        output_vars=["y_var"],
        forward_lines=forward_lines,
        tensor_expr_fn=lambda name: f"expr_{name}",
        activation_lines_fn=lambda name, fused: [
            f"activate({name}, {fused})"
        ],
    )

    assert emitted is True
    assert forward_lines == [
        "y_var = self.linear_0(expr_x)",
        "activate(y_var, RELU6)",
    ]


def test_prelu_module_emitter_bridges_channel_last_parameter_axis() -> None:
    import numpy as np

    model_ir = ModelIR(name="prelu_emitter")
    model_ir.tensors = {
        "x": TensorIR(
            "x",
            "FLOAT32",
            [1, 2, 4, 3],
            logical_layout="NHWC",
        ),
        "slope": TensorIR(
            "slope",
            "FLOAT32",
            [3],
            data=np.asarray([0.1, 0.2, 0.3], dtype=np.float32),
        ),
        "y": TensorIR(
            "y",
            "FLOAT32",
            [1, 2, 4, 3],
            logical_layout="NHWC",
        ),
    }
    op = OperatorIR("PRELU", ["x", "slope"], ["y"])
    forward_lines: list[str] = []

    emitted = _emit_native_prelu_module_op_for_codegen(
        model_ir=model_ir,
        op=op,
        op_type=str(op.op_type),
        attr_name="prelu_0",
        outputs=["y"],
        output_vars=["y_var"],
        forward_lines=forward_lines,
        tensor_expr_fn=lambda name: f"expr_{name}",
        emit_maybe_aligned_expr_fn=lambda **_kwargs: "unused",
        tensor_shape_list_fn=lambda _name: [1, 2, 4, 3],
        should_skip_align_for_shape_preserving_unary_fn=(
            lambda _input, _output: True
        ),
    )

    assert emitted is True
    assert forward_lines == [
        "y_var = self.prelu_0("
        "expr_x.permute(0, 3, 1, 2).contiguous())"
        ".permute(0, 2, 3, 1).contiguous()"
    ]


def test_prelu_module_emitter_uses_alignment_policy_fallback() -> None:
    model_ir = ModelIR(name="prelu_alignment")
    model_ir.tensors = {
        "x": TensorIR("x", "FLOAT32", [1, 4]),
        "slope": TensorIR("slope", "FLOAT32", [1]),
        "y": TensorIR("y", "FLOAT32", [1, 4]),
    }
    op = OperatorIR("PRELU", ["x", "slope"], ["y"])
    forward_lines: list[str] = []

    emitted = _emit_native_prelu_module_op_for_codegen(
        model_ir=model_ir,
        op=op,
        op_type=str(op.op_type),
        attr_name="prelu_0",
        outputs=["y"],
        output_vars=["y_var"],
        forward_lines=forward_lines,
        tensor_expr_fn=lambda name: f"expr_{name}",
        emit_maybe_aligned_expr_fn=(
            lambda *, output_name, expr, inferred_shape: (
                f"aligned({output_name}, {expr}, {inferred_shape})"
            )
        ),
        tensor_shape_list_fn=lambda _name: [1, 4],
        should_skip_align_for_shape_preserving_unary_fn=(
            lambda _input, _output: False
        ),
    )

    assert emitted is True
    assert forward_lines == [
        "y_var = aligned(y, self.prelu_0(expr_x), [1, 4])"
    ]


def test_transpose_conv2d_module_emitter_preserves_named_nhwc_override() -> None:
    import numpy as np

    model_ir = ModelIR(name="transpose_conv2d_emitter")
    model_ir.tensors = {
        "output_shape": TensorIR(
            "output_shape",
            "INT32",
            [4],
            data=np.asarray([1, 8, 8, 4], dtype=np.int32),
        ),
        "x": TensorIR("x", "FLOAT32", [1, 3, 4, 4]),
        "y_nhwc": TensorIR(
            "y_nhwc",
            "FLOAT32",
            [1, 4, 8, 8],
            logical_layout="NCHW",
        ),
    }
    op = OperatorIR(
        "TRANSPOSE_CONV",
        ["output_shape", "weight", "x"],
        ["y_nhwc"],
        {"fusedActivationFunction": "RELU"},
    )
    runtime_imports: set[str] = set()
    forward_lines: list[str] = []

    emitted = _emit_native_transpose_conv2d_module_op_for_codegen(
        model_ir=model_ir,
        op=op,
        op_type=str(op.op_type),
        attr_name="transpose_conv2d_0",
        outputs=["y_nhwc"],
        output_vars=["y_var"],
        output_target_shape="[1, 8, 8, 4]",
        runtime_imports=runtime_imports,
        forward_lines=forward_lines,
        tensor_expr_fn=lambda name: f"expr_{name}",
        activation_lines_fn=lambda name, fused: [
            f"activate({name}, {fused})"
        ],
    )

    assert emitted is True
    assert forward_lines == [
        "y_var = _apply_module_transpose_conv2d("
        "expr_x, self.transpose_conv2d_0.weight, "
        "self.transpose_conv2d_0.bias, "
        "list(self.transpose_conv2d_0.stride), "
        "list(self.transpose_conv2d_0.padding), "
        "list(self.transpose_conv2d_0.dilation), "
        "list(self.transpose_conv2d_0.output_padding), "
        "self.transpose_conv2d_0.groups, target_shape=[1, 4, 8, 8], "
        "fallback_shape=[1, 8, 8, 4], target_logical_layout='NHWC', "
        "fused='NONE')",
        "activate(y_var, RELU)",
    ]
    assert runtime_imports == {"_apply_module_transpose_conv2d"}


def test_transpose_conv3d_module_emitter_preserves_shape_fallback() -> None:
    model_ir = ModelIR(name="transpose_conv3d_emitter")
    model_ir.tensors = {
        "output_shape": TensorIR("output_shape", "INT32", [5]),
        "x": TensorIR("x", "FLOAT32", [1, 3, 2, 4, 4]),
        "y": TensorIR(
            "y",
            "FLOAT32",
            [1, 2, 8, 8, 5],
            logical_layout="NDHWC",
        ),
    }
    op = OperatorIR(
        "CONV_3D_TRANSPOSE",
        ["output_shape", "weight", "x"],
        ["y"],
    )
    runtime_imports: set[str] = set()
    forward_lines: list[str] = []

    emitted = _emit_native_transpose_conv3d_module_op_for_codegen(
        model_ir=model_ir,
        op=op,
        op_type=str(op.op_type),
        attr_name="transpose_conv3d_0",
        outputs=["y"],
        output_vars=["y_var"],
        output_target_shape="[1, 2, 8, 8, 5]",
        runtime_imports=runtime_imports,
        forward_lines=forward_lines,
        tensor_expr_fn=lambda name: f"expr_{name}",
        activation_lines_fn=lambda _name, _fused: [],
    )

    assert emitted is True
    assert forward_lines == [
        "y_var = _apply_module_transpose_conv3d("
        "expr_x, self.transpose_conv3d_0.weight, "
        "self.transpose_conv3d_0.bias, "
        "list(self.transpose_conv3d_0.stride), "
        "list(self.transpose_conv3d_0.padding), "
        "list(self.transpose_conv3d_0.dilation), "
        "list(self.transpose_conv3d_0.output_padding), "
        "self.transpose_conv3d_0.groups, target_shape=[1, 2, 8, 8, 5], "
        "fallback_shape=[1, 2, 8, 8, 5], "
        "target_logical_layout='NDHWC', fused='NONE')"
    ]
    assert runtime_imports == {"_apply_module_transpose_conv3d"}


def test_conv3d_module_emitter_materializes_channel_last_output() -> None:
    model_ir = ModelIR(name="conv3d_emitter")
    model_ir.tensors = {
        "x": TensorIR("x", "FLOAT32", [1, 3, 2, 4, 4]),
        "y": TensorIR(
            "y",
            "FLOAT32",
            [1, 2, 4, 4, 5],
            logical_layout="NDHWC",
        ),
    }
    op = OperatorIR(
        "CONV_3D",
        ["x", "weight", "bias"],
        ["y"],
        {"fusedActivationFunction": "RELU"},
    )
    aliases: dict[str, str] = {}
    runtime_imports: set[str] = set()
    forward_lines: list[str] = []

    emitted = _emit_native_conv3d_module_op_for_codegen(
        model_ir=model_ir,
        op=op,
        op_type=str(op.op_type),
        attr_name="conv3d_0",
        outputs=["y"],
        output_vars=["y_var"],
        output_target_shape="[1, 2, 4, 4, 5]",
        channel_first_tensor_expr_aliases=aliases,
        runtime_imports=runtime_imports,
        forward_lines=forward_lines,
        tensor_expr_fn=lambda name: f"expr_{name}",
        derived_local_var_name_fn=lambda _name, _prefix: "y_var_cf_0",
        emit_module_output_expr_fn=(
            lambda *, output_name, expr, raw_output_layout: (
                f"bridge({output_name}, {expr}, {raw_output_layout})"
            )
        ),
        can_emit_direct_module_call_fn=lambda _op: True,
        activation_lines_fn=lambda name, fused: [
            f"activate({name}, {fused})"
        ],
    )

    assert emitted is True
    assert forward_lines == [
        "y_var_cf_0 = self.conv3d_0(expr_x)",
        "y_var = bridge(y, y_var_cf_0, NCDHW)",
        "activate(y_var, RELU)",
    ]
    assert aliases == {"y": "y_var_cf_0"}
    assert runtime_imports == set()


def test_conv3d_module_emitter_preserves_runtime_helper_fallback() -> None:
    model_ir = ModelIR(name="conv3d_runtime_emitter")
    model_ir.tensors = {
        "x": TensorIR("x", "FLOAT32", [1, 3, 2, 4, 4]),
        "y": TensorIR(
            "y",
            "FLOAT32",
            [1, 5, 2, 4, 4],
            logical_layout="NCDHW",
        ),
    }
    op = OperatorIR("CONV_3D", ["x", "weight", "bias"], ["y"])
    aliases = {"y": "stale_alias"}
    runtime_imports: set[str] = set()
    forward_lines: list[str] = []

    emitted = _emit_native_conv3d_module_op_for_codegen(
        model_ir=model_ir,
        op=op,
        op_type=str(op.op_type),
        attr_name="conv3d_0",
        outputs=["y"],
        output_vars=["y_var"],
        output_target_shape="[1, 5, 2, 4, 4]",
        channel_first_tensor_expr_aliases=aliases,
        runtime_imports=runtime_imports,
        forward_lines=forward_lines,
        tensor_expr_fn=lambda name: f"expr_{name}",
        derived_local_var_name_fn=lambda _name, _prefix: "unused",
        emit_module_output_expr_fn=lambda **_kwargs: "unused",
        can_emit_direct_module_call_fn=lambda _op: False,
        activation_lines_fn=lambda _name, _fused: [],
    )

    assert emitted is True
    assert forward_lines == [
        "y_var = _apply_module_conv3d(self.conv3d_0, expr_x, "
        "target_shape=[1, 5, 2, 4, 4], "
        "target_logical_layout='NCDHW', fused='NONE')"
    ]
    assert aliases == {}
    assert runtime_imports == {"_apply_module_conv3d"}


def _emit_conv2d(
    *,
    model_ir: ModelIR,
    op: OperatorIR,
    aliases: dict[str, str],
    runtime_imports: set[str],
    forward_lines: list[str],
    direct: bool,
    folded_input_expr: str | None = None,
    pad_spec: list[int] | None = None,
) -> bool:
    return _emit_native_conv2d_module_op_for_codegen(
        model_ir=model_ir,
        op=op,
        op_type=str(op.op_type),
        op_index=3,
        attr_name="conv2d_0",
        outputs=["y"],
        output_vars=["y_var"],
        output_target_shape=repr(model_ir.tensors["y"].shape),
        conv_module_pad_specs=({3: pad_spec} if pad_spec is not None else {}),
        channel_first_tensor_expr_aliases=aliases,
        runtime_imports=runtime_imports,
        forward_lines=forward_lines,
        tensor_expr_fn=lambda name: f"expr_{name}",
        tensor_expr_for_channel_first_bridge_fn=(
            lambda _name, _perm: folded_input_expr
        ),
        derived_local_var_name_fn=lambda _name, _prefix: "y_var_cf_0",
        emit_module_output_expr_fn=(
            lambda *, output_name, expr, raw_output_layout: (
                f"bridge({output_name}, {expr}, {raw_output_layout})"
            )
        ),
        target_shape_literal_fn=lambda _name: repr(model_ir.tensors["y"].shape),
        conv2d_input_pre_permute_fn=lambda *_args, **_kwargs: [0, 3, 1, 2],
        can_emit_direct_module_call_fn=lambda _op: direct,
        activation_lines_fn=(
            lambda name, fused: []
            if fused == "NONE"
            else [f"activate({name}, {fused})"]
        ),
        tensor_shape_list_fn=lambda name: list(model_ir.tensors[name].shape),
    )


def test_conv2d_module_emitter_reuses_folded_channel_first_input() -> None:
    model_ir = ModelIR(name="conv2d_emitter")
    model_ir.tensors = {
        "x": TensorIR(
            "x",
            "FLOAT32",
            [1, 4, 4, 3],
            logical_layout="NHWC",
        ),
        "weight": TensorIR("weight", "FLOAT32", [5, 3, 3, 3]),
        "y": TensorIR(
            "y",
            "FLOAT32",
            [1, 4, 4, 5],
            logical_layout="NHWC",
        ),
    }
    op = OperatorIR("CONV_2D", ["x", "weight"], ["y"])
    aliases: dict[str, str] = {}
    runtime_imports: set[str] = set()
    forward_lines: list[str] = []

    emitted = _emit_conv2d(
        model_ir=model_ir,
        op=op,
        aliases=aliases,
        runtime_imports=runtime_imports,
        forward_lines=forward_lines,
        direct=True,
        folded_input_expr="folded_x_cf",
    )

    assert emitted is True
    assert forward_lines == [
        "y_var_cf_0 = self.conv2d_0(folded_x_cf)",
        "y_var = bridge(y, y_var_cf_0, NCHW)",
    ]
    assert aliases == {"y": "y_var_cf_0"}
    assert runtime_imports == set()


def test_conv2d_module_emitter_preserves_padded_runtime_fallback() -> None:
    model_ir = ModelIR(name="conv2d_runtime_emitter")
    model_ir.tensors = {
        "x": TensorIR(
            "x",
            "FLOAT32",
            [1, 3, 4, 4],
            logical_layout="NCHW",
        ),
        "weight": TensorIR("weight", "FLOAT32", [5, 3, 3, 3]),
        "y": TensorIR(
            "y",
            "FLOAT32",
            [1, 5, 4, 4],
            logical_layout="NCHW",
        ),
    }
    op = OperatorIR(
        "CONV_2D",
        ["x", "weight"],
        ["y"],
        {"fusedActivationFunction": "RELU"},
    )
    aliases = {"y": "stale_alias"}
    runtime_imports: set[str] = set()
    forward_lines: list[str] = []

    emitted = _emit_conv2d(
        model_ir=model_ir,
        op=op,
        aliases=aliases,
        runtime_imports=runtime_imports,
        forward_lines=forward_lines,
        direct=False,
        pad_spec=[1, 1, 2, 2],
    )

    assert emitted is True
    assert forward_lines == [
        "y_var = _apply_module_conv2d(self.conv2d_0, "
        "F.pad(expr_x, [1, 1, 2, 2], mode='constant', value=0.0), "
        "target_shape=[1, 5, 4, 4], target_logical_layout='NCHW', "
        "fused='NONE')",
        "activate(y_var, RELU)",
    ]
    assert aliases == {}
    assert runtime_imports == {"_apply_module_conv2d"}
