from __future__ import annotations

from copy import deepcopy

import pytest

from onnx2tf.tflite_builder.core.graph import ModelIRGraphIndex
from onnx2tf.tflite_builder.core.layout import LayoutState
from onnx2tf.tflite_builder.ir import ModelIR, OperatorIR, TensorIR
from onnx2tf.tflite_builder.lower_from_onnx2tf import (
    _optimize_fuse_conv_activation_chains,
)
from onnx2tf.tflite_builder.passes.activation_fusion import (
    optimize_fuse_activation_chains,
)


def _tensor(name: str, *, dtype: str = "FLOAT32") -> TensorIR:
    return TensorIR(
        name=name,
        dtype=dtype,
        shape=[1, 2],
        shape_signature=[1, 2],
    )


def _make_chain(
    producer_type: str = "CONV_2D",
    activation_type: str = "RELU",
) -> ModelIR:
    model_ir = ModelIR("activation_fusion_pass")
    binary = producer_type in {"ADD", "SUB", "MUL", "DIV"}
    model_ir.inputs = ["x0", "x1"] if binary else ["x0"]
    model_ir.outputs = ["z"]
    model_ir.tensors = {
        "x0": _tensor("x0"),
        "y": _tensor("y"),
        "z": _tensor("z"),
    }
    if binary:
        model_ir.tensors["x1"] = _tensor("x1")
    options = {
        "fusedActivationFunction": "NONE",
        "__skip_add_activation_fuse__": True,
    }
    model_ir.operators = [
        OperatorIR(
            op_type=producer_type,
            inputs=list(model_ir.inputs),
            outputs=["y"],
            options=options,
        ),
        OperatorIR(
            op_type=activation_type,
            inputs=["y"],
            outputs=["z"],
        ),
    ]
    return model_ir


def _stats(
    *,
    conv: int = 0,
    add: int = 0,
    sub: int = 0,
    mul: int = 0,
    div: int = 0,
) -> dict[str, int]:
    binary = int(add + sub + mul + div)
    return {
        "fused_conv_activation_chains": int(conv),
        "fused_add_activation_chains": int(add),
        "fused_sub_activation_chains": int(sub),
        "fused_mul_activation_chains": int(mul),
        "fused_div_activation_chains": int(div),
        "fused_binary_activation_chains": binary,
        "fused_activation_chains_total": int(conv + binary),
    }


def _fingerprint(model_ir: ModelIR) -> tuple[object, ...]:
    return (
        tuple(model_ir.inputs),
        tuple(model_ir.outputs),
        tuple(
            (
                operator.op_type,
                tuple(operator.inputs),
                tuple(operator.outputs),
                repr(operator.options),
            )
            for operator in model_ir.operators
        ),
        tuple(
            (
                name,
                tensor.dtype,
                tuple(tensor.shape),
                (
                    None
                    if tensor.shape_signature is None
                    else tuple(tensor.shape_signature)
                ),
            )
            for name, tensor in sorted(model_ir.tensors.items())
        ),
        repr(model_ir.metadata),
    )


@pytest.mark.parametrize(
    ("producer_type", "activation_type", "expected", "lineage_kind"),
    [
        (
            "CONV_2D",
            "RELU_N1_TO_1",
            _stats(conv=1),
            "fuse_conv_activation",
        ),
        (
            "DEPTHWISE_CONV_2D",
            "RELU6",
            _stats(conv=1),
            "fuse_conv_activation",
        ),
        ("ADD", "RELU", _stats(add=1), "fuse_add_activation"),
        ("SUB", "RELU6", _stats(sub=1), "fuse_binary_activation"),
        ("MUL", "RELU", _stats(mul=1), "fuse_binary_activation"),
        ("DIV", "RELU6", _stats(div=1), "fuse_binary_activation"),
    ],
)
def test_activation_fusion_preserves_counters_lineage_index_and_layout(
    producer_type: str,
    activation_type: str,
    expected: dict[str, int],
    lineage_kind: str,
) -> None:
    model_ir = _make_chain(producer_type, activation_type)
    graph_index = ModelIRGraphIndex(model_ir)
    layout_state = LayoutState.from_model_ir(model_ir)

    assert optimize_fuse_activation_chains(
        model_ir,
        graph_index=graph_index,
        layout_state=layout_state,
    ) == expected

    assert [operator.op_type for operator in model_ir.operators] == [
        producer_type
    ]
    producer = model_ir.operators[0]
    assert producer.outputs == ["z"]
    assert producer.options["fusedActivationFunction"] == activation_type
    if producer_type == "ADD":
        assert "__skip_add_activation_fuse__" not in producer.options
    else:
        assert producer.options["__skip_add_activation_fuse__"] is True
    assert "y" not in model_ir.tensors
    assert layout_state.validate_against_model_ir(model_ir) == []
    assert [
        event["kind"] for event in model_ir.metadata["tensor_lineage_events"]
    ] == ["rename_tensor", lineage_kind, "prune_unused_tensors"]

    refreshed = ModelIRGraphIndex(model_ir)
    assert graph_index.producers == refreshed.producers
    assert graph_index.consumers == refreshed.consumers
    assert graph_index.duplicate_producers == refreshed.duplicate_producers
    before_second = _fingerprint(model_ir)
    assert optimize_fuse_activation_chains(
        model_ir,
        graph_index=graph_index,
        layout_state=layout_state,
    ) == _stats()
    assert _fingerprint(model_ir) == before_second


@pytest.mark.parametrize(
    "guard",
    [
        "fanout",
        "already_fused",
        "dtype_mismatch",
        "protected_producer_output",
        "protected_activation_output",
        "public_output_bridge",
        "unsupported_activation",
    ],
)
def test_activation_fusion_guards_are_atomic_noops(guard: str) -> None:
    model_ir = _make_chain()
    if guard == "fanout":
        model_ir.tensors["side"] = _tensor("side")
        model_ir.outputs.append("side")
        model_ir.operators.append(
            OperatorIR(op_type="IDENTITY", inputs=["y"], outputs=["side"])
        )
    elif guard == "already_fused":
        model_ir.operators[0].options["fusedActivationFunction"] = "RELU"
    elif guard == "dtype_mismatch":
        model_ir.tensors["z"].dtype = "INT32"
    elif guard == "protected_producer_output":
        model_ir.metadata["protected_boundary_tensor_names"] = ["y"]
    elif guard == "protected_activation_output":
        model_ir.metadata["protected_boundary_tensor_names"] = ["z"]
    elif guard == "public_output_bridge":
        model_ir.tensors["tail"] = _tensor("tail")
        model_ir.outputs.append("tail")
        model_ir.operators.append(
            OperatorIR(op_type="IDENTITY", inputs=["z"], outputs=["tail"])
        )
    elif guard == "unsupported_activation":
        model_ir.operators[1].op_type = "LOGISTIC"
    before = _fingerprint(model_ir)

    assert optimize_fuse_activation_chains(model_ir) == _stats()
    assert _fingerprint(model_ir) == before


def test_compatibility_wrapper_matches_module_and_reconciles_layout() -> None:
    direct = _make_chain("ADD", "RELU6")
    wrapped = deepcopy(direct)
    direct_layout = LayoutState.from_model_ir(direct)
    wrapped_layout = LayoutState.from_model_ir(wrapped)

    direct_stats = optimize_fuse_activation_chains(
        direct,
        layout_state=direct_layout,
    )
    wrapped_stats = _optimize_fuse_conv_activation_chains(
        wrapped,
        layout_state=wrapped_layout,
    )

    assert direct_stats == wrapped_stats == _stats(add=1)
    assert _fingerprint(direct) == _fingerprint(wrapped)
    assert direct_layout == wrapped_layout
    assert wrapped_layout.validate_against_model_ir(wrapped) == []


def test_foreign_graph_index_is_ignored_deterministically() -> None:
    first = _make_chain("CONV_2D", "RELU")
    second = deepcopy(first)
    foreign = ModelIRGraphIndex(_make_chain("ADD", "RELU"))

    assert optimize_fuse_activation_chains(
        first,
        graph_index=foreign,
    ) == _stats(conv=1)
    assert optimize_fuse_activation_chains(second) == _stats(conv=1)
    assert _fingerprint(first) == _fingerprint(second)
