from __future__ import annotations

from copy import deepcopy

import numpy as np
import pytest

import onnx2tf.tflite_builder.passes.nhwc_concat_layout as nhwc_concat_layout
from onnx2tf.tflite_builder.ir import (
    ModelIR,
    OperatorIR,
    QuantParamIR,
    TensorIR,
)
from onnx2tf.tflite_builder.lower_from_onnx2tf import (
    _optimize_transpose_pre_concat_nhwc_chains,
)


def _tensor(name: str, shape: list[int]) -> TensorIR:
    return TensorIR(
        name=name,
        dtype="FLOAT32",
        shape=list(shape),
        shape_signature=list(shape),
    )


def _model(*, boundary: str | None = None) -> ModelIR:
    model_ir = ModelIR("nhwc_direct_pre_concat")
    model_ir.inputs = ["x0_nhwc", "x1_nhwc"]
    model_ir.outputs = ["y0", "y1"]
    model_ir.tensors = {
        "x0_nhwc": _tensor("x0_nhwc", [1, 5, 7, 2]),
        "x1_nhwc": _tensor(
            "x1_nhwc",
            [1, 9 if boundary == "stale_spatial_metadata" else 5, 7, 3],
        ),
        "pre0_perm": TensorIR(
            "pre0_perm",
            "INT32",
            [4],
            [4],
            data=np.asarray(
                [0, 1, 2, 3]
                if boundary == "invalid_pre_permutation"
                else [0, 3, 1, 2],
                dtype=np.int32,
            ),
        ),
        "pre1_perm": TensorIR(
            "pre1_perm",
            "INT32",
            [4],
            [4],
            data=np.asarray([0, 3, 1, 2], dtype=np.int32),
        ),
        "post0_perm": TensorIR(
            "post0_perm",
            "INT32",
            [4],
            [4],
            data=np.asarray(
                [0, 1, 2, 3]
                if boundary == "invalid_post_permutation"
                else [0, 2, 3, 1],
                dtype=np.int32,
            ),
        ),
        "post1_perm": TensorIR(
            "post1_perm",
            "INT32",
            [4],
            [4],
            data=np.asarray([0, 2, 3, 1], dtype=np.int32),
        ),
        "x0_nchw": _tensor("x0_nchw", [1, 2, 5, 7]),
        "x1_nchw": _tensor("x1_nchw", [1, 3, 5, 7]),
        "concat_nchw": _tensor("concat_nchw", [1, 5, 5, 7]),
        "post0_nhwc": _tensor("post0_nhwc", [1, 5, 7, 5]),
        "post1_nhwc": _tensor("post1_nhwc", [1, 5, 7, 5]),
        "y0": _tensor("y0", [1, 5, 7, 5]),
        "y1": _tensor("y1", [1, 5, 7, 5]),
    }
    if boundary == "invalid_source_rank":
        model_ir.tensors["x0_nhwc"].shape = [1, 5, 2]
        model_ir.tensors["x0_nhwc"].shape_signature = [1, 5, 2]
    model_ir.operators = [
        OperatorIR("TRANSPOSE", ["x0_nhwc", "pre0_perm"], ["x0_nchw"]),
        OperatorIR("TRANSPOSE", ["x1_nhwc", "pre1_perm"], ["x1_nchw"]),
        OperatorIR(
            "CONCATENATION",
            ["x0_nchw", "x1_nchw"],
            ["concat_nchw"],
            options={"axis": 2 if boundary == "concat_axis" else 1},
        ),
        OperatorIR(
            "TRANSPOSE",
            ["concat_nchw", "post0_perm"],
            ["post0_nhwc"],
        ),
        OperatorIR("RELU", ["post0_nhwc"], ["y0"]),
        OperatorIR(
            "TRANSPOSE",
            ["concat_nchw", "post1_perm"],
            ["post1_nhwc"],
        ),
        OperatorIR("RELU", ["post1_nhwc"], ["y1"]),
    ]
    if boundary == "concat_nontranspose_fanout":
        model_ir.tensors["side"] = _tensor("side", [1, 5, 5, 7])
        model_ir.outputs.append("side")
        model_ir.operators.append(
            OperatorIR("IDENTITY", ["concat_nchw"], ["side"])
        )
    if boundary == "public_concat":
        model_ir.outputs.append("concat_nchw")
    if boundary == "public_post":
        model_ir.outputs.append("post0_nhwc")
    return model_ir


def _unary_model(
    *,
    unary_op: str = "RELU",
    boundary: str | None = None,
) -> ModelIR:
    model_ir = _model()
    if boundary == "spatial_shape_mismatch":
        model_ir.tensors["x1_nhwc"].shape = [1, 9, 7, 3]
        model_ir.tensors["x1_nhwc"].shape_signature = [1, 9, 7, 3]
        model_ir.tensors["x1_nchw"].shape = [1, 3, 9, 7]
        model_ir.tensors["x1_nchw"].shape_signature = [1, 3, 9, 7]
    unary_shape = (
        [1, 3, 9, 7]
        if boundary == "spatial_shape_mismatch"
        else [1, 3, 5, 7]
    )
    if boundary == "invalid_unary_rank":
        unary_shape = [1, 3, 5]
    model_ir.tensors["x1_unary"] = _tensor("x1_unary", unary_shape)
    model_ir.operators.insert(
        2,
        OperatorIR(unary_op, ["x1_nchw"], ["x1_unary"]),
    )
    concat_op = next(
        op for op in model_ir.operators if op.op_type == "CONCATENATION"
    )
    concat_op.inputs = ["x0_nchw", "x1_unary"]

    if boundary == "unary_output_fanout":
        model_ir.tensors["unary_side"] = _tensor(
            "unary_side",
            list(unary_shape),
        )
        model_ir.outputs.append("unary_side")
        model_ir.operators.append(
            OperatorIR("IDENTITY", ["x1_unary"], ["unary_side"])
        )
    if boundary == "public_unary_output":
        model_ir.outputs.append("x1_unary")
    if boundary == "unary_adapter_fanout":
        model_ir.tensors["adapter_side"] = _tensor(
            "adapter_side",
            list(model_ir.tensors["x1_nchw"].shape),
        )
        model_ir.outputs.append("adapter_side")
        model_ir.operators.append(
            OperatorIR("IDENTITY", ["x1_nchw"], ["adapter_side"])
        )
    if boundary == "public_unary_adapter":
        model_ir.outputs.append("x1_nchw")
    return model_ir


def _all_unary_model(*, fanout_input: int | None = None) -> ModelIR:
    model_ir = _unary_model(unary_op="GELU")
    model_ir.tensors["x0_unary"] = _tensor(
        "x0_unary",
        [1, 2, 5, 7],
    )
    model_ir.operators.insert(
        1,
        OperatorIR("TANH", ["x0_nchw"], ["x0_unary"]),
    )
    concat_op = next(
        op for op in model_ir.operators if op.op_type == "CONCATENATION"
    )
    concat_op.inputs = ["x0_unary", "x1_unary"]
    if fanout_input is not None:
        source_name = "x0_unary" if fanout_input == 0 else "x1_unary"
        source_shape = list(model_ir.tensors[source_name].shape)
        side_name = f"unary{fanout_input}_side"
        model_ir.tensors[side_name] = _tensor(side_name, source_shape)
        model_ir.outputs.append(side_name)
        model_ir.operators.append(
            OperatorIR("IDENTITY", [source_name], [side_name])
        )
    return model_ir


def _pad_model(
    *,
    boundary: str | None = None,
    shared_pads: bool = False,
    shared_adapter: bool = False,
) -> ModelIR:
    model_ir = _model()
    model_ir.tensors["x1_nhwc"].shape = [1, 4, 7, 3]
    model_ir.tensors["x1_nhwc"].shape_signature = [1, 4, 7, 3]
    model_ir.tensors["x1_nchw"].shape = [1, 3, 4, 7]
    model_ir.tensors["x1_nchw"].shape_signature = [1, 3, 4, 7]
    pad_output_shape = (
        [1, 3, 9, 7]
        if boundary == "spatial_shape_mismatch"
        else [1, 3, 5, 7]
    )
    if boundary == "invalid_pad_output_rank":
        pad_output_shape = [1, 3, 5]
    pads_data = np.asarray(
        [[0, 0], [0, 0], [0, 1], [0, 0]],
        dtype=np.int32,
    )
    if boundary == "invalid_pads_shape":
        pads_data = np.asarray(
            [[0, 0], [0, 0], [0, 1]],
            dtype=np.int32,
        )
    model_ir.tensors["pads_nchw"] = TensorIR(
        "pads_nchw",
        "INT32",
        list(pads_data.shape),
        list(pads_data.shape),
        data=None if boundary == "missing_pads_data" else pads_data,
    )
    model_ir.tensors["pad_value"] = TensorIR(
        "pad_value",
        "FLOAT32",
        [1],
        [1],
        data=np.asarray([0.25], dtype=np.float32),
    )
    model_ir.tensors["x1_pad"] = _tensor("x1_pad", pad_output_shape)
    model_ir.operators.insert(
        2,
        OperatorIR(
            "MIRROR_PAD" if boundary == "unsupported_pad" else "PAD",
            ["x1_nchw", "pads_nchw", "pad_value"],
            ["x1_pad"],
        ),
    )
    concat_op = next(
        op for op in model_ir.operators if op.op_type == "CONCATENATION"
    )
    concat_op.inputs = ["x0_nchw", "x1_pad"]

    if boundary == "pad_output_fanout":
        model_ir.tensors["pad_side"] = _tensor(
            "pad_side",
            list(pad_output_shape),
        )
        model_ir.outputs.append("pad_side")
        model_ir.operators.append(
            OperatorIR("IDENTITY", ["x1_pad"], ["pad_side"])
        )
    if boundary == "public_pad_output":
        model_ir.outputs.append("x1_pad")
    if boundary == "public_pad_adapter":
        model_ir.outputs.append("x1_nchw")
    if shared_adapter:
        model_ir.tensors["adapter_side"] = _tensor(
            "adapter_side",
            [1, 3, 4, 7],
        )
        model_ir.outputs.append("adapter_side")
        model_ir.operators.append(
            OperatorIR("IDENTITY", ["x1_nchw"], ["adapter_side"])
        )
    if shared_pads:
        model_ir.tensors["outside_pad"] = _tensor(
            "outside_pad",
            [1, 2, 6, 7],
        )
        model_ir.outputs.append("outside_pad")
        model_ir.operators.append(
            OperatorIR(
                "PAD",
                ["x0_nchw", "pads_nchw", "pad_value"],
                ["outside_pad"],
            )
        )
    return model_ir


def _dequantize_model(
    *,
    all_dequantize: bool = False,
    boundary: str | None = None,
    shared_adapter: bool = False,
) -> ModelIR:
    model_ir = _model()
    model_ir.tensors["x1_nhwc"].dtype = "INT8"
    model_ir.tensors["x1_nhwc"].quantization = QuantParamIR(
        scale=[0.25] * 3,
        zero_point=[0] * 3,
        quantized_dimension=3,
    )
    model_ir.tensors["x1_nchw"].dtype = "INT8"
    model_ir.tensors["x1_nchw"].quantization = QuantParamIR(
        scale=[0.25] * 3,
        zero_point=[0] * 3,
        quantized_dimension=1,
    )
    dequantize_shape = (
        [1, 3, 9, 7]
        if boundary == "spatial_shape_mismatch"
        else [1, 3, 5, 7]
    )
    if boundary == "invalid_dequantize_output_rank":
        dequantize_shape = [1, 3, 5]
    model_ir.tensors["x1_dequantized"] = _tensor(
        "x1_dequantized",
        dequantize_shape,
    )
    model_ir.tensors["x1_dequantized"].quantization = {
        "scale": [0.5] * 3,
        "zero_point": [0] * 3,
        "quantized_dimension": 1,
    }
    dequantize_inputs = ["x1_nchw"]
    if boundary == "invalid_dequantize_arity":
        model_ir.tensors["extra"] = _tensor("extra", [1])
        dequantize_inputs.append("extra")
    model_ir.operators.insert(
        2,
        OperatorIR(
            "CAST" if boundary == "unsupported_dequantize" else "DEQUANTIZE",
            dequantize_inputs,
            ["x1_dequantized"],
        ),
    )
    concat_op = next(
        op for op in model_ir.operators if op.op_type == "CONCATENATION"
    )
    concat_op.inputs = ["x0_nchw", "x1_dequantized"]

    if all_dequantize:
        model_ir.tensors["x0_nhwc"].dtype = "INT8"
        model_ir.tensors["x0_nhwc"].quantization = QuantParamIR(
            scale=[0.125] * 2,
            zero_point=[0] * 2,
            quantized_dimension=3,
        )
        model_ir.tensors["x0_nchw"].dtype = "INT8"
        model_ir.tensors["x0_nchw"].quantization = QuantParamIR(
            scale=[0.125] * 2,
            zero_point=[0] * 2,
            quantized_dimension=1,
        )
        model_ir.tensors["x0_dequantized"] = _tensor(
            "x0_dequantized",
            [1, 2, 5, 7],
        )
        model_ir.operators.insert(
            1,
            OperatorIR(
                "DEQUANTIZE",
                ["x0_nchw"],
                ["x0_dequantized"],
            ),
        )
        concat_op.inputs = ["x0_dequantized", "x1_dequantized"]

    if boundary == "dequantize_output_fanout":
        model_ir.tensors["dequantize_side"] = _tensor(
            "dequantize_side",
            list(dequantize_shape),
        )
        model_ir.outputs.append("dequantize_side")
        model_ir.operators.append(
            OperatorIR(
                "IDENTITY",
                ["x1_dequantized"],
                ["dequantize_side"],
            )
        )
    if boundary == "public_dequantize_output":
        model_ir.outputs.append("x1_dequantized")
    if boundary == "public_dequantize_adapter":
        model_ir.outputs.append("x1_nchw")
    if shared_adapter:
        model_ir.tensors["dequantize_adapter_side"] = _tensor(
            "dequantize_adapter_side",
            [1, 3, 5, 7],
        )
        model_ir.outputs.append("dequantize_adapter_side")
        model_ir.operators.append(
            OperatorIR(
                "IDENTITY",
                ["x1_nchw"],
                ["dequantize_adapter_side"],
            )
        )
    return model_ir


def _assert_model_equal(actual: ModelIR, expected: ModelIR) -> None:
    assert actual.inputs == expected.inputs
    assert actual.outputs == expected.outputs
    assert actual.description == expected.description
    assert actual.metadata == expected.metadata
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
        assert tensor.quantization == expected_tensor.quantization
        if tensor.data is None or expected_tensor.data is None:
            assert tensor.data is expected_tensor.data
        else:
            np.testing.assert_array_equal(tensor.data, expected_tensor.data)


@pytest.mark.parametrize("boundary", [None, "stale_spatial_metadata"])
def test_nhwc_direct_pre_concat_multi_post_is_transactionally_optimized(
    boundary: str | None,
) -> None:
    model_ir = _model(boundary=boundary)
    model_ir.tensors["concat_nchw"].quantization = QuantParamIR(
        scale=[0.25] * 5,
        zero_point=[0] * 5,
        quantized_dimension=1,
    )
    model_ir.tensors["side0"] = _tensor("side0", [1, 2, 5, 7])
    model_ir.outputs.extend(["side0", "x1_nchw"])
    model_ir.operators.append(
        OperatorIR("IDENTITY", ["x0_nchw"], ["side0"])
    )
    diagnostics: list[dict] = []

    stats = _optimize_transpose_pre_concat_nhwc_chains(
        model_ir,
        diagnostics=diagnostics,
    )

    assert stats == {"optimized_transpose_pre_concat_nhwc_chains": 1}
    assert [op.op_type for op in model_ir.operators].count("TRANSPOSE") == 2
    concat_op = next(
        op for op in model_ir.operators if op.op_type == "CONCATENATION"
    )
    assert concat_op.inputs == ["x0_nhwc", "x1_nhwc"]
    assert concat_op.outputs == ["post0_nhwc"]
    assert concat_op.options["axis"] == 3
    assert model_ir.tensors["post0_nhwc"].shape == [1, 5, 7, 5]
    quantization = model_ir.tensors["post0_nhwc"].quantization
    assert isinstance(quantization, QuantParamIR)
    assert quantization.quantized_dimension == 3
    assert [
        op.inputs for op in model_ir.operators if op.op_type == "RELU"
    ] == [["post0_nhwc"], ["post0_nhwc"]]
    assert diagnostics[0]["code"] == "layout.nhwc_pre_concat_direct"
    assert diagnostics[0]["status"] == "changed"
    assert diagnostics[0]["metrics"] == {
        "preflight_operators_visited": 3,
        "state_built": True,
        "snapshot_count": 1,
        "fingerprint_count": 0,
    }


def test_nhwc_direct_precondition_reuses_prepared_candidate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model_ir = _model()
    calls: dict[str, int] = {}
    original_resolver = nhwc_concat_layout._resolve_nhwc_concat_candidate

    def counted_resolver(*args, family: str, **kwargs):
        calls[family] = int(calls.get(family, 0)) + 1
        return original_resolver(*args, family=family, **kwargs)

    monkeypatch.setattr(
        nhwc_concat_layout,
        "_resolve_nhwc_concat_candidate",
        counted_resolver,
    )

    stats = nhwc_concat_layout.run_nhwc_concat_layout_cleanup(model_ir)

    assert stats[nhwc_concat_layout._DIRECT_STATS_KEY] == 1
    assert calls["direct"] == 2
    assert sum(calls.values()) == len(nhwc_concat_layout._NHWC_FAMILY_SPECS) + 1


@pytest.mark.parametrize(
    "boundary",
    [
        "invalid_pre_permutation",
        "invalid_post_permutation",
        "concat_axis",
        "invalid_source_rank",
        "concat_nontranspose_fanout",
        "public_concat",
        "public_post",
    ],
)
def test_nhwc_direct_pre_concat_rejects_unsafe_boundary(boundary: str) -> None:
    model_ir = _model(boundary=boundary)
    original = deepcopy(model_ir)

    stats = _optimize_transpose_pre_concat_nhwc_chains(model_ir)

    assert stats == {"optimized_transpose_pre_concat_nhwc_chains": 0}
    _assert_model_equal(model_ir, original)


@pytest.mark.parametrize(
    "unary_op",
    ["RELU", "RELU6", "LOGISTIC", "TANH", "GELU"],
)
def test_nhwc_one_unary_plus_direct_pre_concat_is_indexed(
    unary_op: str,
) -> None:
    model_ir = _unary_model(unary_op=unary_op)
    model_ir.tensors["x1_unary"].quantization = {
        "scale": [0.5] * 3,
        "zero_point": [0] * 3,
        "quantized_dimension": 1,
    }
    diagnostics: list[dict] = []

    stats = _optimize_transpose_pre_concat_nhwc_chains(
        model_ir,
        diagnostics=diagnostics,
    )

    assert stats == {"optimized_transpose_pre_concat_nhwc_chains": 1}
    unary = next(op for op in model_ir.operators if op.op_type == unary_op)
    assert unary.inputs == ["x1_nhwc"]
    assert model_ir.tensors["x1_unary"].shape == [1, 5, 7, 3]
    quantization = model_ir.tensors["x1_unary"].quantization
    assert isinstance(quantization, dict)
    assert quantization["quantized_dimension"] == 3
    concat_op = next(
        op for op in model_ir.operators if op.op_type == "CONCATENATION"
    )
    assert concat_op.inputs == ["x0_nhwc", "x1_unary"]
    assert concat_op.outputs == ["post0_nhwc"]
    assert concat_op.options["axis"] == 3
    assert all(op.op_type != "TRANSPOSE" for op in model_ir.operators)
    unary_event = next(
        event
        for event in diagnostics
        if event["code"] == "layout.nhwc_pre_concat_unary"
    )
    assert unary_event["status"] == "changed"
    assert unary_event["metrics"]["snapshot_count"] == 1
    assert unary_event["metrics"]["fingerprint_count"] == 0


@pytest.mark.parametrize(
    ("boundary", "unary_op"),
    [
        ("unsupported_unary", "ABS"),
        ("unary_output_fanout", "RELU"),
        ("public_unary_output", "RELU"),
        ("unary_adapter_fanout", "RELU"),
        ("public_unary_adapter", "RELU"),
        ("spatial_shape_mismatch", "RELU"),
        ("invalid_unary_rank", "RELU"),
    ],
)
def test_nhwc_one_unary_plus_direct_rejects_unsafe_boundary(
    boundary: str,
    unary_op: str,
) -> None:
    model_ir = _unary_model(unary_op=unary_op, boundary=boundary)
    original = deepcopy(model_ir)

    stats = _optimize_transpose_pre_concat_nhwc_chains(model_ir)

    assert stats == {"optimized_transpose_pre_concat_nhwc_chains": 0}
    _assert_model_equal(model_ir, original)


def test_nhwc_all_unary_pre_concat_uses_same_indexed_family() -> None:
    model_ir = _all_unary_model()
    model_ir.tensors["x0_unary"].quantization = QuantParamIR(
        scale=[0.125] * 2,
        zero_point=[0] * 2,
        quantized_dimension=1,
    )
    diagnostics: list[dict] = []

    stats = _optimize_transpose_pre_concat_nhwc_chains(
        model_ir,
        diagnostics=diagnostics,
    )

    assert stats == {"optimized_transpose_pre_concat_nhwc_chains": 1}
    tanh_op = next(op for op in model_ir.operators if op.op_type == "TANH")
    gelu_op = next(op for op in model_ir.operators if op.op_type == "GELU")
    assert tanh_op.inputs == ["x0_nhwc"]
    assert gelu_op.inputs == ["x1_nhwc"]
    assert model_ir.tensors["x0_unary"].shape == [1, 5, 7, 2]
    assert model_ir.tensors["x1_unary"].shape == [1, 5, 7, 3]
    quantization = model_ir.tensors["x0_unary"].quantization
    assert isinstance(quantization, QuantParamIR)
    assert quantization.quantized_dimension == 3
    concat_op = next(
        op for op in model_ir.operators if op.op_type == "CONCATENATION"
    )
    assert concat_op.inputs == ["x0_unary", "x1_unary"]
    assert concat_op.options["axis"] == 3
    assert all(op.op_type != "TRANSPOSE" for op in model_ir.operators)
    unary_event = next(
        event
        for event in diagnostics
        if event["code"] == "layout.nhwc_pre_concat_unary"
    )
    assert unary_event["status"] == "changed"


@pytest.mark.parametrize("fanout_input", [0, 1])
def test_nhwc_all_unary_rejects_fanout_without_partial_mutation(
    fanout_input: int,
) -> None:
    model_ir = _all_unary_model(fanout_input=fanout_input)
    original = deepcopy(model_ir)

    stats = _optimize_transpose_pre_concat_nhwc_chains(model_ir)

    assert stats == {"optimized_transpose_pre_concat_nhwc_chains": 0}
    _assert_model_equal(model_ir, original)


def test_nhwc_pad_plus_direct_pre_concat_is_indexed() -> None:
    model_ir = _pad_model(shared_adapter=True)
    model_ir.tensors["x1_pad"].quantization = QuantParamIR(
        scale=[0.25] * 3,
        zero_point=[0] * 3,
        quantized_dimension=1,
    )
    diagnostics: list[dict] = []

    stats = _optimize_transpose_pre_concat_nhwc_chains(
        model_ir,
        diagnostics=diagnostics,
    )

    assert stats == {"optimized_transpose_pre_concat_nhwc_chains": 1}
    pad_op = next(op for op in model_ir.operators if op.op_type == "PAD")
    assert pad_op.inputs == ["x1_nhwc", "pads_nchw", "pad_value"]
    np.testing.assert_array_equal(
        model_ir.tensors["pads_nchw"].data,
        np.asarray(
            [[0, 0], [0, 1], [0, 0], [0, 0]],
            dtype=np.int32,
        ),
    )
    assert model_ir.tensors["x1_pad"].shape == [1, 5, 7, 3]
    quantization = model_ir.tensors["x1_pad"].quantization
    assert isinstance(quantization, QuantParamIR)
    assert quantization.quantized_dimension == 3
    concat_op = next(
        op for op in model_ir.operators if op.op_type == "CONCATENATION"
    )
    assert concat_op.inputs == ["x0_nhwc", "x1_pad"]
    assert concat_op.options["axis"] == 3
    remaining_transposes = [
        op for op in model_ir.operators if op.op_type == "TRANSPOSE"
    ]
    assert [op.outputs for op in remaining_transposes] == [["x1_nchw"]]
    pad_event = next(
        event
        for event in diagnostics
        if event["code"] == "layout.nhwc_pre_concat_pad"
    )
    assert pad_event["status"] == "changed"
    assert pad_event["metrics"]["snapshot_count"] == 1
    assert pad_event["metrics"]["fingerprint_count"] == 0


def test_nhwc_pad_clones_shared_pads_constant() -> None:
    model_ir = _pad_model(shared_pads=True)
    original_pads = np.array(model_ir.tensors["pads_nchw"].data, copy=True)

    stats = _optimize_transpose_pre_concat_nhwc_chains(model_ir)

    assert stats == {"optimized_transpose_pre_concat_nhwc_chains": 1}
    pad_ops = [op for op in model_ir.operators if op.op_type == "PAD"]
    rewritten_pad = next(op for op in pad_ops if op.outputs == ["x1_pad"])
    outside_pad = next(op for op in pad_ops if op.outputs == ["outside_pad"])
    assert outside_pad.inputs[1] == "pads_nchw"
    np.testing.assert_array_equal(
        model_ir.tensors["pads_nchw"].data,
        original_pads,
    )
    rewritten_pads_name = rewritten_pad.inputs[1]
    assert rewritten_pads_name != "pads_nchw"
    np.testing.assert_array_equal(
        model_ir.tensors[rewritten_pads_name].data,
        np.asarray(
            [[0, 0], [0, 1], [0, 0], [0, 0]],
            dtype=np.int32,
        ),
    )


@pytest.mark.parametrize(
    "boundary",
    [
        "unsupported_pad",
        "pad_output_fanout",
        "public_pad_output",
        "public_pad_adapter",
        "missing_pads_data",
        "invalid_pads_shape",
        "invalid_pad_output_rank",
        "spatial_shape_mismatch",
    ],
)
def test_nhwc_pad_plus_direct_rejects_unsafe_boundary(
    boundary: str,
) -> None:
    model_ir = _pad_model(boundary=boundary)
    original = deepcopy(model_ir)

    stats = _optimize_transpose_pre_concat_nhwc_chains(model_ir)

    assert stats == {"optimized_transpose_pre_concat_nhwc_chains": 0}
    _assert_model_equal(model_ir, original)


def test_nhwc_dequantize_plus_direct_preserves_quantization_provenance() -> None:
    model_ir = _dequantize_model(shared_adapter=True)
    source_quantization = deepcopy(model_ir.tensors["x1_nhwc"].quantization)
    diagnostics: list[dict] = []

    stats = _optimize_transpose_pre_concat_nhwc_chains(
        model_ir,
        diagnostics=diagnostics,
    )

    assert stats == {"optimized_transpose_pre_concat_nhwc_chains": 1}
    dequantize_op = next(
        op for op in model_ir.operators if op.op_type == "DEQUANTIZE"
    )
    assert dequantize_op.inputs == ["x1_nhwc"]
    assert model_ir.tensors["x1_nhwc"].quantization == source_quantization
    assert model_ir.tensors["x1_dequantized"].shape == [1, 5, 7, 3]
    output_quantization = model_ir.tensors["x1_dequantized"].quantization
    assert isinstance(output_quantization, dict)
    assert output_quantization["quantized_dimension"] == 3
    concat_op = next(
        op for op in model_ir.operators if op.op_type == "CONCATENATION"
    )
    assert concat_op.inputs == ["x0_nhwc", "x1_dequantized"]
    assert concat_op.options["axis"] == 3
    remaining_transposes = [
        op for op in model_ir.operators if op.op_type == "TRANSPOSE"
    ]
    assert [op.outputs for op in remaining_transposes] == [["x1_nchw"]]
    event = next(
        event
        for event in diagnostics
        if event["code"] == "layout.nhwc_pre_concat_dequantize"
    )
    assert event["status"] == "changed"
    assert event["metrics"]["snapshot_count"] == 1
    assert event["metrics"]["fingerprint_count"] == 0


def test_nhwc_all_dequantize_inputs_use_indexed_family() -> None:
    model_ir = _dequantize_model(all_dequantize=True)

    stats = _optimize_transpose_pre_concat_nhwc_chains(model_ir)

    assert stats == {"optimized_transpose_pre_concat_nhwc_chains": 1}
    dequantize_ops = [
        op for op in model_ir.operators if op.op_type == "DEQUANTIZE"
    ]
    assert [op.inputs for op in dequantize_ops] == [
        ["x0_nhwc"],
        ["x1_nhwc"],
    ]
    assert all(op.op_type != "TRANSPOSE" for op in model_ir.operators)
    concat_op = next(
        op for op in model_ir.operators if op.op_type == "CONCATENATION"
    )
    assert concat_op.inputs == ["x0_dequantized", "x1_dequantized"]
    assert concat_op.options["axis"] == 3


@pytest.mark.parametrize(
    "boundary",
    [
        "unsupported_dequantize",
        "invalid_dequantize_arity",
        "dequantize_output_fanout",
        "public_dequantize_output",
        "public_dequantize_adapter",
        "invalid_dequantize_output_rank",
        "spatial_shape_mismatch",
    ],
)
def test_nhwc_dequantize_rejects_unsafe_boundary(boundary: str) -> None:
    model_ir = _dequantize_model(boundary=boundary)
    original = deepcopy(model_ir)

    stats = _optimize_transpose_pre_concat_nhwc_chains(model_ir)

    assert stats == {"optimized_transpose_pre_concat_nhwc_chains": 0}
    _assert_model_equal(model_ir, original)
