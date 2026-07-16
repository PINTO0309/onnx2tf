from __future__ import annotations

from copy import deepcopy

import numpy as np

from onnx2tf.tflite_builder.core.graph import ModelIRGraphIndex
from onnx2tf.tflite_builder.core.layout import LayoutState
from onnx2tf.tflite_builder.ir import ModelIR, OperatorIR, TensorIR
from onnx2tf.tflite_builder.lower_from_onnx2tf import (
    _optimize_fold_conv_mul_add_affine_chains,
)
from onnx2tf.tflite_builder.passes.conv_mul_affine_fold import (
    _apply_plan,
    _plan_signature,
    _resolve_candidate,
    optimize_conv_mul_affine_mul_only_chains,
)


_TOTAL_KEY = "folded_conv_mul_add_affine_chains"
_ADD_ONLY_KEY = "folded_conv_add_only_affine_chains"
_MUL_ONLY_KEY = "folded_conv_mul_only_affine_chains"
_MUL_ADD_KEY = "folded_conv_mul_add_only_affine_chains"


def _tensor(
    name: str,
    shape: list[int],
    *,
    dtype: str = "FLOAT32",
    data: np.ndarray | None = None,
) -> TensorIR:
    return TensorIR(
        name=name,
        dtype=dtype,
        shape=list(shape),
        shape_signature=list(shape),
        data=data,
        is_variable=False,
    )


def _make_conv_mul_ir() -> ModelIR:
    model_ir = ModelIR("indexed_conv_mul_affine_fold")
    model_ir.inputs = ["x"]
    model_ir.outputs = ["y"]
    weights = np.asarray(
        [
            [[[1.0, 2.0]]],
            [[[3.0, 4.0]]],
            [[[5.0, 6.0]]],
        ],
        dtype=np.float32,
    )
    model_ir.tensors = {
        "x": _tensor("x", [1, 4, 4, 2]),
        "w": _tensor("w", [3, 1, 1, 2], data=weights),
        "b": _tensor(
            "b",
            [3],
            data=np.asarray([0.1, 0.2, 0.3], dtype=np.float32),
        ),
        "conv_out": _tensor("conv_out", [1, 4, 4, 3]),
        "scale": _tensor(
            "scale",
            [1, 1, 1, 3],
            data=np.asarray([2.0, 3.0, 4.0], dtype=np.float32).reshape(
                1, 1, 1, 3
            ),
        ),
        "mul_out": _tensor("mul_out", [1, 4, 4, 3]),
        "y": _tensor("y", [1, 4, 4, 3]),
    }
    model_ir.operators = [
        OperatorIR(
            op_type="CONV_2D",
            inputs=["x", "w", "b"],
            outputs=["conv_out"],
            options={
                "padding": "SAME",
                "strideH": 1,
                "strideW": 1,
                "dilationHFactor": 1,
                "dilationWFactor": 1,
                "fusedActivationFunction": "NONE",
            },
        ),
        OperatorIR(
            op_type="MUL",
            inputs=["conv_out", "scale"],
            outputs=["mul_out"],
            options={"fusedActivationFunction": "NONE"},
        ),
        OperatorIR(op_type="RELU", inputs=["mul_out"], outputs=["y"]),
    ]
    return model_ir


def _stats(total: int, *, mul_only: int, mul_add: int = 0) -> dict[str, int]:
    return {
        _TOTAL_KEY: total,
        _ADD_ONLY_KEY: 0,
        _MUL_ONLY_KEY: mul_only,
        _MUL_ADD_KEY: mul_add,
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
                tensor.is_variable,
                repr(tensor.quantization),
                tensor.logical_layout,
                tensor.physical_layout,
                (
                    None
                    if tensor.data is None
                    else (
                        str(np.asarray(tensor.data).dtype),
                        tuple(np.asarray(tensor.data).shape),
                        tuple(np.asarray(tensor.data).reshape(-1).tolist()),
                    )
                ),
            )
            for name, tensor in sorted(model_ir.tensors.items())
        ),
        repr(model_ir.metadata),
    )


def test_indexed_conv_mul_fold_preserves_index_layout_and_lineage() -> None:
    model_ir = _make_conv_mul_ir()
    graph_index = ModelIRGraphIndex(model_ir)
    layout_state = LayoutState.from_model_ir(model_ir)
    original_weights = np.asarray(model_ir.tensors["w"].data).copy()
    original_bias = np.asarray(model_ir.tensors["b"].data).copy()
    coefficient = np.asarray([2.0, 3.0, 4.0], dtype=np.float32)

    assert optimize_conv_mul_affine_mul_only_chains(
        model_ir,
        graph_index=graph_index,
        layout_state=layout_state,
    ) == _stats(1, mul_only=1)

    assert [operator.op_type for operator in model_ir.operators] == [
        "CONV_2D",
        "RELU",
    ]
    assert model_ir.operators[0].outputs == ["mul_out"]
    assert model_ir.operators[1].inputs == ["mul_out"]
    np.testing.assert_array_equal(
        model_ir.tensors["w"].data,
        original_weights * coefficient.reshape(3, 1, 1, 1),
    )
    np.testing.assert_array_equal(
        model_ir.tensors["b"].data,
        original_bias * coefficient,
    )
    assert [
        event["source"] for event in model_ir.metadata["tensor_lineage_events"]
    ] == ["set_operator_outputs"]

    refreshed = ModelIRGraphIndex(model_ir)
    assert graph_index.producers == refreshed.producers
    assert graph_index.consumers == refreshed.consumers
    assert graph_index.duplicate_producers == refreshed.duplicate_producers
    assert layout_state.validate_against_model_ir(model_ir) == []
    assert optimize_conv_mul_affine_mul_only_chains(
        model_ir,
        graph_index=graph_index,
        layout_state=layout_state,
    ) == _stats(0, mul_only=0)


def test_indexed_conv_mul_fold_candidate_and_bound_are_strict() -> None:
    model_ir = _make_conv_mul_ir()
    graph_index = ModelIRGraphIndex(model_ir)
    conv = model_ir.operators[0]

    assert optimize_conv_mul_affine_mul_only_chains(
        model_ir,
        graph_index=graph_index,
        candidate=model_ir.operators[1],
    ) == _stats(0, mul_only=0)
    assert optimize_conv_mul_affine_mul_only_chains(
        model_ir,
        graph_index=graph_index,
        candidate=conv,
        max_rewrites=0,
    ) == _stats(0, mul_only=0)
    assert optimize_conv_mul_affine_mul_only_chains(
        model_ir,
        graph_index=graph_index,
        candidate=conv,
        max_rewrites=1,
    ) == _stats(1, mul_only=1)


def test_indexed_conv_mul_fold_rejects_stale_plan_atomically() -> None:
    model_ir = _make_conv_mul_ir()
    graph_index = ModelIRGraphIndex(model_ir)
    layout_state = LayoutState.from_model_ir(model_ir)
    plan = _resolve_candidate(
        model_ir,
        graph_index,
        model_ir.operators[0],
        layout_state=layout_state,
    )
    assert plan is not None
    signature = _plan_signature(plan)

    model_ir.tensors["scale"].data[0, 0, 0, 0] += 1.0
    before_apply = _fingerprint(model_ir)
    assert _plan_signature(plan) == signature
    assert not _apply_plan(
        model_ir,
        graph_index,
        plan,
        layout_state=layout_state,
    )
    assert _fingerprint(model_ir) == before_apply


def test_indexed_conv_mul_fold_is_deterministic() -> None:
    first = _make_conv_mul_ir()
    second = deepcopy(first)

    assert optimize_conv_mul_affine_mul_only_chains(first) == _stats(
        1, mul_only=1
    )
    assert optimize_conv_mul_affine_mul_only_chains(second) == _stats(
        1, mul_only=1
    )
    assert _fingerprint(first) == _fingerprint(second)


def test_indexed_conv_mul_fold_preserves_legacy_signed_zero_bits() -> None:
    model_ir = _make_conv_mul_ir()
    bias = np.asarray([-0.0, 0.0, -0.0], dtype=np.float32)
    coefficient = np.asarray([-2.0, -3.0, 4.0], dtype=np.float32)
    model_ir.tensors["b"].data = bias.copy()
    model_ir.tensors["scale"].data = coefficient.reshape(1, 1, 1, 3)

    assert optimize_conv_mul_affine_mul_only_chains(model_ir) == _stats(
        1, mul_only=1
    )

    expected = bias * coefficient + np.zeros((3,), dtype=np.float32)
    np.testing.assert_array_equal(
        np.asarray(model_ir.tensors["b"].data).view(np.uint32),
        expected.view(np.uint32),
    )
    np.testing.assert_array_equal(
        np.asarray(model_ir.tensors["b"].data).view(np.uint32),
        np.zeros((3,), dtype=np.float32).view(np.uint32),
    )


def test_indexed_conv_mul_wrapper_keeps_one_prune_and_layout_boundary() -> None:
    model_ir = _make_conv_mul_ir()
    layout_state = LayoutState.from_model_ir(model_ir)

    assert _optimize_fold_conv_mul_add_affine_chains(
        model_ir,
        layout_state=layout_state,
    ) == _stats(1, mul_only=1)
    assert set(model_ir.tensors) == {"x", "w", "b", "mul_out", "y"}
    assert layout_state.validate_against_model_ir(model_ir) == []


def test_relaxed_coefficients_and_dynamic_signature_stay_on_fallback() -> None:
    scalar = _make_conv_mul_ir()
    scalar.tensors["scale"].shape = [1]
    scalar.tensors["scale"].shape_signature = [1]
    scalar.tensors["scale"].data = np.asarray([2.0], dtype=np.float32)
    dynamic = _make_conv_mul_ir()
    for name in ("x", "conv_out", "mul_out"):
        dynamic.tensors[name].shape_signature[0] = -1

    for model_ir in (scalar, dynamic):
        assert optimize_conv_mul_affine_mul_only_chains(model_ir) == _stats(
            0, mul_only=0
        )
        assert _optimize_fold_conv_mul_add_affine_chains(model_ir) == _stats(
            1, mul_only=1
        )


def test_missing_bias_and_relu_variants_stay_on_fallback() -> None:
    missing_bias = _make_conv_mul_ir()
    missing_bias.operators[0].inputs = ["x", "w"]
    missing_bias.tensors.pop("b")
    relu = _make_conv_mul_ir()
    relu.operators[0].options["fusedActivationFunction"] = "RELU"

    for model_ir in (missing_bias, relu):
        assert optimize_conv_mul_affine_mul_only_chains(model_ir) == _stats(
            0, mul_only=0
        )
        assert _optimize_fold_conv_mul_add_affine_chains(model_ir) == _stats(
            1, mul_only=1
        )


def test_constant_add_suffix_stays_on_mul_add_fallback() -> None:
    model_ir = _make_conv_mul_ir()
    model_ir.tensors["add_c"] = _tensor(
        "add_c",
        [3],
        data=np.asarray([0.5, 0.25, 0.125], dtype=np.float32),
    )
    model_ir.tensors["add_out"] = _tensor("add_out", [1, 4, 4, 3])
    model_ir.operators.insert(
        2,
        OperatorIR(
            op_type="ADD",
            inputs=["mul_out", "add_c"],
            outputs=["add_out"],
            options={"fusedActivationFunction": "NONE"},
        ),
    )
    model_ir.operators[3].inputs = ["add_out"]

    assert optimize_conv_mul_affine_mul_only_chains(model_ir) == _stats(
        0, mul_only=0
    )
    assert _optimize_fold_conv_mul_add_affine_chains(model_ir) == _stats(
        1,
        mul_only=0,
        mul_add=1,
    )


def test_shared_side_constant_is_rejected_by_owner_but_kept_by_fallback() -> None:
    model_ir = _make_conv_mul_ir()
    model_ir.tensors["scale_alias"] = _tensor("scale_alias", [1, 1, 1, 3])
    model_ir.outputs.append("scale_alias")
    model_ir.operators.append(
        OperatorIR(
            op_type="IDENTITY",
            inputs=["scale"],
            outputs=["scale_alias"],
        )
    )

    assert optimize_conv_mul_affine_mul_only_chains(model_ir) == _stats(
        0, mul_only=0
    )
    assert _optimize_fold_conv_mul_add_affine_chains(model_ir) == _stats(
        1, mul_only=1
    )


def test_malformed_options_are_rejected_without_mutation() -> None:
    model_ir = _make_conv_mul_ir()
    model_ir.operators[0].options["strideH"] = object()
    before = _fingerprint(model_ir)

    assert optimize_conv_mul_affine_mul_only_chains(model_ir) == _stats(
        0, mul_only=0
    )
    assert _fingerprint(model_ir) == before
