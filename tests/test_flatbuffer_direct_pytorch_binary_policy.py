from __future__ import annotations

from onnx2tf.tflite_builder.ir import ModelIR, OperatorIR, TensorIR
from onnx2tf.tflite_builder.pytorch_binary_policy import (
    _all_consumers_are_channel_first_binary_ops_for_codegen,
    _binary_requires_runtime_alignment_for_codegen,
    _binary_runtime_shape_passthrough_operand_for_codegen,
    _can_omit_materialized_channel_last_alias_recursive_for_codegen,
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
