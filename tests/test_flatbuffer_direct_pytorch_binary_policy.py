from __future__ import annotations

from onnx2tf.tflite_builder.ir import ModelIR, OperatorIR, TensorIR
from onnx2tf.tflite_builder.pytorch_binary_policy import (
    _all_consumers_are_channel_first_binary_ops_for_codegen,
    _binary_requires_runtime_alignment_for_codegen,
    _binary_runtime_shape_passthrough_operand_for_codegen,
    _can_emit_channel_first_binary_op_for_codegen,
    _can_omit_materialized_channel_last_alias_recursive_for_codegen,
    _channel_first_binary_input_expr_for_codegen,
    _preferred_binary_alignment_anchor_for_codegen,
)


def _tensor(
    name: str,
    shape: list[int],
    *,
    signature: list[int] | None = None,
) -> TensorIR:
    return TensorIR(
        name=name,
        dtype="FLOAT32",
        shape=shape,
        shape_signature=signature,
    )


def test_binary_runtime_shape_passthrough_uses_all_ones_peer() -> None:
    model_ir = ModelIR(
        name="binary_passthrough",
        tensors={
            "lhs": _tensor("lhs", [2, 3]),
            "rhs": _tensor("rhs", [1, 1]),
        },
    )

    assert _binary_runtime_shape_passthrough_operand_for_codegen(
        model_ir=model_ir,
        runtime_shape_uncertain_tensors={"lhs"},
        lhs_name="lhs",
        rhs_name="rhs",
    ) == "lhs"


def test_binary_alignment_uses_dynamic_signature_anchor() -> None:
    model_ir = ModelIR(
        name="binary_alignment",
        tensors={
            "lhs": _tensor("lhs", [2, 3], signature=[-1, 3]),
            "rhs": _tensor("rhs", [2, 3], signature=[2, 3]),
            "output": _tensor("output", [2, 3], signature=[-1, 3]),
        },
    )

    assert _binary_requires_runtime_alignment_for_codegen(
        model_ir=model_ir,
        runtime_shape_uncertain_tensors=set(),
        lhs_name="lhs",
        rhs_name="rhs",
        output_name="output",
    )
    assert _preferred_binary_alignment_anchor_for_codegen(
        model_ir=model_ir,
        lhs_name="lhs",
        rhs_name="rhs",
        output_name="output",
    ) == "lhs"


def test_all_consumers_require_supported_channel_first_binary_ops() -> None:
    model_ir = ModelIR(
        name="binary_consumers",
        operators=[OperatorIR(op_type="ADD", inputs=["source", "peer"], outputs=["out"])],
    )

    assert _all_consumers_are_channel_first_binary_ops_for_codegen(
        model_ir=model_ir,
        consumer_index={"source": [0]},
        direct_codegen_binary_functions={"ADD"},
        can_emit_channel_first_binary_op_fn=lambda op: op.op_type == "ADD",
        output_name="source",
    )
    assert not _all_consumers_are_channel_first_binary_ops_for_codegen(
        model_ir=model_ir,
        consumer_index={"source": [0]},
        direct_codegen_binary_functions={"ADD"},
        can_emit_channel_first_binary_op_fn=lambda _op: False,
        output_name="source",
    )


def test_alias_elision_preserves_public_output_boundary() -> None:
    tensor = _tensor("source", [1, 8, 8, 3])
    tensor.logical_layout = "NHWC"

    def can_omit(model_ir: ModelIR) -> bool:
        return _can_omit_materialized_channel_last_alias_recursive_for_codegen(
            model_ir=model_ir,
            consumer_index={},
            direct_codegen_unary_expressions=set(),
            tensor_shape_list_fn=lambda name: list(model_ir.tensors[name].shape),
            channel_first_reduction_plan_fn=lambda _op, _name: None,
            can_emit_channel_first_shape_preserving_unary_op_fn=lambda _op: False,
            can_emit_channel_first_binary_op_fn=lambda _op: False,
            can_resolve_channel_first_expr_statically_fn=lambda _name: False,
            conv2d_input_pre_permute_fn=lambda *_args, **_kwargs: None,
            output_name="source",
            seen_names=set(),
        )

    assert can_omit(ModelIR(name="internal", tensors={"source": tensor}))
    assert not can_omit(
        ModelIR(name="public", tensors={"source": tensor}, outputs=["source"])
    )


def test_channel_first_binary_input_expr_uses_layout_bridge() -> None:
    source = _tensor("source", [1, 4, 5, 3])
    source.logical_layout = "NHWC"
    peer = _tensor("peer", [1, 3, 4, 5])
    peer.logical_layout = "NCHW"
    model_ir = ModelIR(
        name="binary_input_expr",
        tensors={"source": source, "peer": peer},
    )

    expr = _channel_first_binary_input_expr_for_codegen(
        model_ir=model_ir,
        channel_first_tensor_expr_aliases={},
        channel_first_constant_buffer_alias_exprs={},
        permuted_constant_buffer_alias_exprs={},
        scalar_literal_expr_fn=lambda _name: None,
        tensor_shape_list_fn=lambda name: list(model_ir.tensors[name].shape),
        tensor_expr_fn=lambda name: f"value_{name}",
        channel_first_passthrough_input_expr_fn=lambda _name: None,
        tensor_name="source",
        other_tensor_name="peer",
    )

    assert expr == "value_source.permute(0, 3, 1, 2).contiguous()"


def test_channel_first_binary_capability_requires_complete_broadcast() -> None:
    tensors = {
        name: _tensor(name, [1, 3, 4, 5])
        for name in ("lhs", "rhs", "output")
    }
    for tensor in tensors.values():
        tensor.logical_layout = "NCHW"
    op = OperatorIR(op_type="ADD", inputs=["lhs", "rhs"], outputs=["output"])
    model_ir = ModelIR(name="binary_capability", tensors=tensors, operators=[op])

    assert _can_emit_channel_first_binary_op_for_codegen(
        model_ir=model_ir,
        tensor_shape_list_fn=lambda name: list(model_ir.tensors[name].shape),
        channel_first_shape_for_tensor_fn=lambda name: list(
            model_ir.tensors[name].shape
        ),
        scalar_literal_expr_fn=lambda _name: None,
        can_omit_materialized_channel_last_alias_fn=lambda _name: False,
        channel_first_binary_input_expr_fn=lambda name, _other: f"value_{name}",
        op=op,
    )
