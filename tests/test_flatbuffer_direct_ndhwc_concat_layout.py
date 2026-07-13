from __future__ import annotations

from copy import deepcopy

import numpy as np
import pytest

from onnx2tf.tflite_builder.core.graph import ModelIRGraphIndex
from onnx2tf.tflite_builder.core.model_ir_pass_state import ModelIRPassState
from onnx2tf.tflite_builder.ir import (
    ModelIR,
    OperatorIR,
    QuantParamIR,
    TensorIR,
)
from onnx2tf.tflite_builder.lower_from_onnx2tf import (
    _optimize_transpose_pre_concat_ndhwc_chains,
)
from onnx2tf.tflite_builder.passes.ndhwc_concat_layout import (
    run_ndhwc_concat_layout_cleanup,
)


def _tensor(name: str, shape: list[int]) -> TensorIR:
    return TensorIR(
        name=name,
        dtype="FLOAT32",
        shape=list(shape),
        shape_signature=list(shape),
    )


def _model(*, boundary: str | None = None) -> ModelIR:
    model_ir = ModelIR("generic_ndhwc_pre_concat")
    model_ir.inputs = ["x0_ndhwc", "x1_ndhwc"]
    model_ir.outputs = ["y0", "y1"]
    pre_perm = (
        [0, 1, 2, 3, 4]
        if boundary == "invalid_pre_permutation"
        else [0, 4, 1, 2, 3]
    )
    model_ir.tensors = {
        "x0_ndhwc": _tensor("x0_ndhwc", [1, 2, 3, 4, 5]),
        "x1_ndhwc": _tensor(
            "x1_ndhwc",
            [1, 9 if boundary == "spatial_shape_mismatch" else 2, 3, 4, 6],
        ),
        "pre_perm": TensorIR(
            "pre_perm",
            "INT32",
            [5],
            [5],
            data=np.asarray(pre_perm, dtype=np.int32),
        ),
        "post_perm": TensorIR(
            "post_perm",
            "INT32",
            [5],
            [5],
            data=np.asarray([0, 2, 3, 4, 1], dtype=np.int32),
        ),
        "x0_ncdhw": _tensor("x0_ncdhw", [1, 5, 2, 3, 4]),
        "x1_ncdhw": _tensor("x1_ncdhw", [1, 6, 2, 3, 4]),
        "x1_unary_ncdhw": _tensor(
            "x1_unary_ncdhw",
            [1, 6, 9 if boundary == "spatial_shape_mismatch" else 2, 3, 4],
        ),
        "concat_ncdhw": _tensor("concat_ncdhw", [1, 11, 2, 3, 4]),
        "post0_ndhwc": _tensor("post0_ndhwc", [1, 2, 3, 4, 11]),
        "post1_ndhwc": _tensor("post1_ndhwc", [1, 2, 3, 4, 11]),
        "y0": _tensor("y0", [1, 2, 3, 4, 11]),
        "y1": _tensor("y1", [1, 2, 3, 4, 11]),
    }
    if boundary == "invalid_direct_rank":
        model_ir.tensors["x0_ndhwc"].shape = [1, 2, 3, 5]
        model_ir.tensors["x0_ndhwc"].shape_signature = [1, 2, 3, 5]
    if boundary == "invalid_post_permutation":
        model_ir.tensors["bad_post_perm"] = TensorIR(
            "bad_post_perm",
            "INT32",
            [5],
            [5],
            data=np.asarray([0, 1, 2, 3, 4], dtype=np.int32),
        )
    model_ir.operators = [
        OperatorIR("TRANSPOSE", ["x0_ndhwc", "pre_perm"], ["x0_ncdhw"]),
        OperatorIR("TRANSPOSE", ["x1_ndhwc", "pre_perm"], ["x1_ncdhw"]),
        OperatorIR(
            "IDENTITY" if boundary == "unsupported_unary" else "LEAKY_RELU",
            ["x1_ncdhw"],
            ["x1_unary_ncdhw"],
        ),
        OperatorIR(
            "CONCATENATION",
            ["x0_ncdhw", "x1_unary_ncdhw"],
            ["concat_ncdhw"],
            options={"axis": 2 if boundary == "concat_axis" else 1},
        ),
        OperatorIR(
            "TRANSPOSE",
            [
                "concat_ncdhw",
                "bad_post_perm"
                if boundary == "invalid_post_permutation"
                else "post_perm",
            ],
            ["post0_ndhwc"],
        ),
        OperatorIR("RELU", ["post0_ndhwc"], ["y0"]),
        OperatorIR(
            "TRANSPOSE",
            ["concat_ncdhw", "post_perm"],
            ["post1_ndhwc"],
        ),
        OperatorIR("RELU", ["post1_ndhwc"], ["y1"]),
    ]

    side_sources = {
        "direct_adapter_fanout": "x0_ncdhw",
        "unary_adapter_fanout": "x1_ncdhw",
        "unary_output_fanout": "x1_unary_ncdhw",
        "concat_nontranspose_fanout": "concat_ncdhw",
    }
    if boundary in side_sources:
        source = side_sources[boundary]
        model_ir.tensors["side"] = _tensor(
            "side",
            list(model_ir.tensors[source].shape),
        )
        model_ir.outputs.append("side")
        model_ir.operators.append(OperatorIR("IDENTITY", [source], ["side"]))
    public_sources = {
        "public_direct_adapter": "x0_ncdhw",
        "public_unary_adapter": "x1_ncdhw",
        "public_unary_output": "x1_unary_ncdhw",
        "public_concat": "concat_ncdhw",
        "public_post": "post0_ndhwc",
    }
    if boundary in public_sources:
        model_ir.outputs.append(public_sources[boundary])
    return model_ir


def _assert_model_equal(actual: ModelIR, expected: ModelIR) -> None:
    assert actual.inputs == expected.inputs
    assert actual.outputs == expected.outputs
    assert [
        (op.op_type, op.inputs, op.outputs, op.options)
        for op in actual.operators
    ] == [
        (op.op_type, op.inputs, op.outputs, op.options)
        for op in expected.operators
    ]
    assert actual.tensors.keys() == expected.tensors.keys()
    for name, tensor in actual.tensors.items():
        expected_tensor = expected.tensors[name]
        assert tensor.dtype == expected_tensor.dtype
        assert tensor.shape == expected_tensor.shape
        assert tensor.shape_signature == expected_tensor.shape_signature
        if tensor.data is None or expected_tensor.data is None:
            assert tensor.data is expected_tensor.data
        else:
            np.testing.assert_array_equal(tensor.data, expected_tensor.data)


def test_ndhwc_pre_concat_direct_unary_multi_post_characterization() -> None:
    model_ir = _model()
    model_ir.tensors["concat_ncdhw"].quantization = QuantParamIR(
        scale=[0.25] * 11,
        zero_point=[0] * 11,
        quantized_dimension=1,
    )
    model_ir.tensors["x1_unary_ncdhw"].quantization = {
        "scale": [0.5] * 6,
        "zero_point": [0] * 6,
        "quantized_dimension": 1,
    }

    stats = _optimize_transpose_pre_concat_ndhwc_chains(model_ir)

    assert stats["optimized_transpose_pre_concat_ndhwc_chains"] == 1
    assert all(op.op_type != "TRANSPOSE" for op in model_ir.operators)
    unary_op = next(op for op in model_ir.operators if op.op_type == "LEAKY_RELU")
    assert unary_op.inputs == ["x1_ndhwc"]
    assert model_ir.tensors["x1_unary_ncdhw"].shape == [1, 2, 3, 4, 6]
    unary_quantization = model_ir.tensors["x1_unary_ncdhw"].quantization
    assert isinstance(unary_quantization, dict)
    assert unary_quantization["quantized_dimension"] == 4
    concat_op = next(
        op for op in model_ir.operators if op.op_type == "CONCATENATION"
    )
    assert concat_op.inputs == ["x0_ndhwc", "x1_unary_ncdhw"]
    assert concat_op.outputs == ["post0_ndhwc"]
    assert concat_op.options["axis"] == 4
    assert model_ir.tensors["post0_ndhwc"].shape == [1, 2, 3, 4, 11]
    quantization = model_ir.tensors["post0_ndhwc"].quantization
    assert isinstance(quantization, QuantParamIR)
    assert quantization.scale == [0.25] * 11
    assert quantization.zero_point == [0] * 11
    assert quantization.quantized_dimension == 4
    relu_ops = [op for op in model_ir.operators if op.op_type == "RELU"]
    assert [op.inputs for op in relu_ops] == [
        ["post0_ndhwc"],
        ["post0_ndhwc"],
    ]


def test_ndhwc_pre_concat_runner_uses_one_index_and_transaction(
    monkeypatch,
) -> None:
    model_ir = _model()
    calls = {"refresh": 0, "snapshot": 0, "fingerprint": 0}
    original_refresh = ModelIRGraphIndex.refresh
    original_snapshot = ModelIRPassState.snapshot
    original_fingerprint = ModelIRPassState.fingerprint

    def counted_refresh(index: ModelIRGraphIndex) -> None:
        calls["refresh"] += 1
        original_refresh(index)

    def counted_snapshot(state: ModelIRPassState) -> ModelIR:
        calls["snapshot"] += 1
        return original_snapshot(state)

    def counted_fingerprint(state: ModelIRPassState) -> bytes:
        calls["fingerprint"] += 1
        return original_fingerprint(state)

    monkeypatch.setattr(ModelIRGraphIndex, "refresh", counted_refresh)
    monkeypatch.setattr(ModelIRPassState, "snapshot", counted_snapshot)
    monkeypatch.setattr(ModelIRPassState, "fingerprint", counted_fingerprint)
    diagnostics: list[dict] = []

    stats = run_ndhwc_concat_layout_cleanup(
        model_ir,
        diagnostics=diagnostics,
    )

    assert stats == {"optimized_transpose_pre_concat_ndhwc_chains": 1}
    assert calls == {"refresh": 1, "snapshot": 1, "fingerprint": 0}
    assert len(diagnostics) == 1
    assert diagnostics[0]["code"] == "layout.ndhwc_pre_concat"
    assert diagnostics[0]["status"] == "changed"
    assert diagnostics[0]["metrics"] == {
        "preflight_operators_visited": 4,
        "state_built": True,
        "snapshot_count": 1,
        "fingerprint_count": 0,
    }


@pytest.mark.parametrize(
    "boundary",
    [
        "direct_adapter_fanout",
        "unary_adapter_fanout",
        "unary_output_fanout",
        "concat_nontranspose_fanout",
        "public_direct_adapter",
        "public_unary_adapter",
        "public_unary_output",
        "public_concat",
        "public_post",
        "invalid_pre_permutation",
        "invalid_post_permutation",
        "concat_axis",
        "unsupported_unary",
        "invalid_direct_rank",
        "spatial_shape_mismatch",
    ],
)
def test_ndhwc_pre_concat_rejects_unsafe_boundary(boundary: str) -> None:
    model_ir = _model(boundary=boundary)
    original = deepcopy(model_ir)

    stats = _optimize_transpose_pre_concat_ndhwc_chains(model_ir)

    assert stats["optimized_transpose_pre_concat_ndhwc_chains"] == 0
    _assert_model_equal(model_ir, original)
