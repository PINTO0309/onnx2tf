from __future__ import annotations

from copy import deepcopy

import numpy as np
import pytest

from onnx2tf.tflite_builder.core.graph import ModelIRGraphIndex
from onnx2tf.tflite_builder.ir import (
    ModelIR,
    OperatorIR,
    QuantParamIR,
    TensorIR,
)
from onnx2tf.tflite_builder.lower_from_onnx2tf import (
    _optimize_transpose_pre_dequant_concat_quantize_post_nhwc_chains,
)
from onnx2tf.tflite_builder.passes.dequant_concat_quantize_layout import (
    run_dequant_concat_quantize_layout_cleanup,
)


def _tensor(
    name: str,
    dtype: str,
    shape: list[int],
    *,
    quantization: QuantParamIR | None = None,
    data: np.ndarray | None = None,
) -> TensorIR:
    return TensorIR(
        name=name,
        dtype=dtype,
        shape=list(shape),
        shape_signature=list(shape),
        data=None if data is None else np.asarray(data),
        is_variable=False,
        quantization=quantization,
    )


def _quantization() -> QuantParamIR:
    return QuantParamIR(
        scale=[0.125],
        zero_point=[-3],
        quantized_dimension=0,
        min=[-15.625],
        max=[16.25],
    )


def _model(
    *,
    post_count: int = 1,
    boundary: str | None = None,
) -> ModelIR:
    model_ir = ModelIR("dequant_concat_quantize_nhwc")
    model_ir.inputs = ["q0_nhwc", "q1_nhwc"]
    model_ir.outputs = [f"y{index}" for index in range(post_count)]
    pre_perm = [0, 1, 2, 3] if boundary == "pre_permutation" else [0, 3, 1, 2]
    post_perm = [0, 1, 2, 3] if boundary == "post_permutation" else [0, 2, 3, 1]
    quantization = _quantization()
    model_ir.tensors = {
        "q0_nhwc": _tensor(
            "q0_nhwc",
            "INT8",
            [1, 2, 3, 2],
            quantization=deepcopy(quantization),
        ),
        "q1_nhwc": _tensor(
            "q1_nhwc",
            "INT8",
            [1, 2, 3, 2],
            quantization=deepcopy(quantization),
        ),
        "q0_nchw": _tensor(
            "q0_nchw",
            "INT8",
            [1, 2, 2, 3],
            quantization=deepcopy(quantization),
        ),
        "q1_nchw": _tensor(
            "q1_nchw",
            "INT8",
            [1, 2, 2, 3],
            quantization=deepcopy(quantization),
        ),
        "f0_nchw": _tensor("f0_nchw", "FLOAT32", [1, 2, 2, 3]),
        "f1_nchw": _tensor("f1_nchw", "FLOAT32", [1, 2, 2, 3]),
        "cat_nchw": _tensor("cat_nchw", "FLOAT32", [1, 4, 2, 3]),
        "qcat_nchw": _tensor(
            "qcat_nchw",
            "INT8",
            [1, 4, 2, 3],
            quantization=deepcopy(quantization),
        ),
        "pre_perm": _tensor(
            "pre_perm",
            "INT32",
            [4],
            data=np.asarray(pre_perm, dtype=np.int32),
        ),
        "post_perm": _tensor(
            "post_perm",
            "INT32",
            [4],
            data=np.asarray(post_perm, dtype=np.int32),
        ),
    }
    model_ir.operators = [
        OperatorIR("TRANSPOSE", ["q0_nhwc", "pre_perm"], ["q0_nchw"]),
        OperatorIR("DEQUANTIZE", ["q0_nchw"], ["f0_nchw"]),
        OperatorIR("TRANSPOSE", ["q1_nhwc", "pre_perm"], ["q1_nchw"]),
        OperatorIR("DEQUANTIZE", ["q1_nchw"], ["f1_nchw"]),
        OperatorIR(
            "CONCATENATION",
            ["f0_nchw", "f1_nchw"],
            ["cat_nchw"],
            options={"axis": 2 if boundary == "concat_axis" else 1},
        ),
        OperatorIR("QUANTIZE", ["cat_nchw"], ["qcat_nchw"]),
    ]
    for index in range(post_count):
        post_name = f"post{index}_nhwc"
        output_name = f"y{index}"
        model_ir.tensors[post_name] = _tensor(
            post_name,
            "INT8",
            [1, 2, 3, 4],
            quantization=deepcopy(quantization),
        )
        model_ir.tensors[output_name] = _tensor(
            output_name,
            "FLOAT32",
            [1, 2, 3, 4],
        )
        model_ir.operators.extend(
            [
                OperatorIR(
                    "TRANSPOSE",
                    ["qcat_nchw", "post_perm"],
                    [post_name],
                ),
                OperatorIR("DEQUANTIZE", [post_name], [output_name]),
            ]
        )

    side_sources = {
        "pre_fanout": "q0_nchw",
        "dequant_fanout": "f0_nchw",
        "concat_fanout": "cat_nchw",
        "quantized_fanout": "qcat_nchw",
    }
    if boundary in side_sources:
        source = side_sources[boundary]
        source_tensor = model_ir.tensors[source]
        model_ir.tensors["side"] = _tensor(
            "side",
            source_tensor.dtype,
            list(source_tensor.shape),
            quantization=deepcopy(source_tensor.quantization),
        )
        model_ir.outputs.append("side")
        model_ir.operators.append(OperatorIR("IDENTITY", [source], ["side"]))
    public_sources = {
        "public_pre": "q0_nchw",
        "public_dequant": "f0_nchw",
        "public_concat": "cat_nchw",
        "public_quantized": "qcat_nchw",
        "public_post": "post0_nhwc",
    }
    if boundary in public_sources:
        model_ir.outputs.append(public_sources[boundary])
    if boundary == "non_dequant_input":
        model_ir.operators[1].op_type = "IDENTITY"
    elif boundary == "missing_quantization":
        model_ir.tensors["qcat_nchw"].quantization = None
    elif boundary == "missing_source_quantization":
        model_ir.tensors["q0_nhwc"].quantization = None
    elif boundary == "invalid_rank":
        model_ir.tensors["cat_nchw"].shape = [1, 4, 6]
        model_ir.tensors["cat_nchw"].shape_signature = [1, 4, 6]
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
        assert tensor.quantization == expected_tensor.quantization
        if tensor.data is None or expected_tensor.data is None:
            assert tensor.data is expected_tensor.data
        else:
            np.testing.assert_array_equal(tensor.data, expected_tensor.data)


def _assert_common_nhwc_rewrite(model_ir: ModelIR) -> None:
    assert all(
        op.op_type != "TRANSPOSE"
        or op.outputs == ["q0_nchw"]
        for op in model_ir.operators
    )
    dq0 = next(op for op in model_ir.operators if op.outputs == ["f0_nchw"])
    dq1 = next(op for op in model_ir.operators if op.outputs == ["f1_nchw"])
    assert dq0.inputs == ["q0_nhwc"]
    assert dq1.inputs == ["q1_nhwc"]
    concat = next(
        op for op in model_ir.operators if op.op_type == "CONCATENATION"
    )
    assert concat.options["axis"] == 3
    assert model_ir.tensors["f0_nchw"].shape == [1, 2, 3, 2]
    assert model_ir.tensors["f1_nchw"].shape == [1, 2, 3, 2]
    assert model_ir.tensors["cat_nchw"].shape == [1, 2, 3, 4]
    quantize = next(op for op in model_ir.operators if op.op_type == "QUANTIZE")
    assert quantize.outputs == ["post0_nhwc"]
    canonical = model_ir.tensors["post0_nhwc"]
    assert canonical.dtype == "INT8"
    assert canonical.shape == [1, 2, 3, 4]
    assert canonical.quantization == _quantization()


def test_dequant_concat_quantize_layout_characterization() -> None:
    model_ir = _model()

    stats = _optimize_transpose_pre_dequant_concat_quantize_post_nhwc_chains(
        model_ir
    )

    assert stats[
        "optimized_transpose_pre_dequant_concat_quantize_post_nhwc_chains"
    ] == 1
    _assert_common_nhwc_rewrite(model_ir)
    assert all(op.op_type != "TRANSPOSE" for op in model_ir.operators)
    output = next(op for op in model_ir.operators if op.outputs == ["y0"])
    assert output.inputs == ["post0_nhwc"]


def test_dequant_concat_quantize_layout_merges_post_fanout() -> None:
    model_ir = _model(post_count=2)

    stats = _optimize_transpose_pre_dequant_concat_quantize_post_nhwc_chains(
        model_ir
    )

    assert stats[
        "optimized_transpose_pre_dequant_concat_quantize_post_nhwc_chains"
    ] == 1
    _assert_common_nhwc_rewrite(model_ir)
    assert all(op.op_type != "TRANSPOSE" for op in model_ir.operators)
    for output_name in ["y0", "y1"]:
        output = next(
            op for op in model_ir.operators if op.outputs == [output_name]
        )
        assert output.inputs == ["post0_nhwc"]


def test_dequant_concat_quantize_layout_retains_shared_pre_adapter() -> None:
    model_ir = _model(boundary="pre_fanout")

    stats = _optimize_transpose_pre_dequant_concat_quantize_post_nhwc_chains(
        model_ir
    )

    assert stats[
        "optimized_transpose_pre_dequant_concat_quantize_post_nhwc_chains"
    ] == 1
    _assert_common_nhwc_rewrite(model_ir)
    transposes = [op for op in model_ir.operators if op.op_type == "TRANSPOSE"]
    assert len(transposes) == 1
    assert transposes[0].outputs == ["q0_nchw"]
    side = next(op for op in model_ir.operators if op.outputs == ["side"])
    assert side.inputs == ["q0_nchw"]


@pytest.mark.parametrize(
    "boundary",
    [
        "dequant_fanout",
        "concat_fanout",
        "quantized_fanout",
        "public_pre",
        "public_dequant",
        "public_concat",
        "public_quantized",
        "public_post",
        "pre_permutation",
        "post_permutation",
        "concat_axis",
        "non_dequant_input",
        "missing_quantization",
        "missing_source_quantization",
        "invalid_rank",
    ],
)
def test_dequant_concat_quantize_layout_rejects_unsafe_boundary(
    boundary: str,
) -> None:
    model_ir = _model(boundary=boundary)
    original = deepcopy(model_ir)

    stats = _optimize_transpose_pre_dequant_concat_quantize_post_nhwc_chains(
        model_ir
    )

    assert stats[
        "optimized_transpose_pre_dequant_concat_quantize_post_nhwc_chains"
    ] == 0
    _assert_model_equal(model_ir, original)


def test_dequant_concat_quantize_layout_runner_reuses_one_index(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model_ir = _model(post_count=2)
    diagnostics: list[dict[str, object]] = []
    refresh_count = 0
    original_refresh = ModelIRGraphIndex.refresh

    def counted_refresh(graph_index: ModelIRGraphIndex) -> None:
        nonlocal refresh_count
        refresh_count += 1
        original_refresh(graph_index)

    monkeypatch.setattr(ModelIRGraphIndex, "refresh", counted_refresh)

    stats = run_dequant_concat_quantize_layout_cleanup(
        model_ir,
        diagnostics=diagnostics,
    )

    stats_key = (
        "optimized_transpose_pre_dequant_concat_quantize_post_nhwc_chains"
    )
    assert stats[stats_key] == 1
    assert refresh_count == 1
    assert len(diagnostics) == 1
    assert diagnostics[0]["code"] == "layout.dequant_concat_quantize_nhwc"
    assert diagnostics[0]["changed"] is True
    assert diagnostics[0]["metrics"]["snapshot_count"] == 1


@pytest.mark.parametrize(
    "boundary",
    [
        "dequant_fanout",
        "concat_fanout",
        "quantized_fanout",
        "public_pre",
        "public_dequant",
        "public_concat",
        "public_quantized",
        "public_post",
        "pre_permutation",
        "post_permutation",
        "concat_axis",
        "non_dequant_input",
        "missing_quantization",
        "missing_source_quantization",
        "invalid_rank",
    ],
)
def test_dequant_concat_quantize_layout_runner_rejects_before_snapshot(
    boundary: str,
) -> None:
    model_ir = _model(boundary=boundary)
    original = deepcopy(model_ir)
    diagnostics: list[dict[str, object]] = []

    stats = run_dequant_concat_quantize_layout_cleanup(
        model_ir,
        diagnostics=diagnostics,
    )

    stats_key = (
        "optimized_transpose_pre_dequant_concat_quantize_post_nhwc_chains"
    )
    assert stats[stats_key] == 0
    assert len(diagnostics) == 1
    assert diagnostics[0]["changed"] is False
    assert diagnostics[0]["skipped_by_precondition"] is True
    assert diagnostics[0]["metrics"]["snapshot_count"] == 0
    _assert_model_equal(model_ir, original)
