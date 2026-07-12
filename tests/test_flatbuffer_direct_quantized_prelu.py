from __future__ import annotations

import numpy as np

from onnx2tf.tflite_builder.core.graph import ModelIRGraphIndex
from onnx2tf.tflite_builder.ir import ModelIR, OperatorIR, QuantParamIR, TensorIR
from onnx2tf.tflite_builder.passes.quantized_prelu import run_quantized_prelu_cleanup


def _qparams(scale: float = 0.1) -> QuantParamIR:
    return QuantParamIR(scale=[float(scale)], zero_point=[0], quantized_dimension=0)


def _tensor(
    name: str,
    dtype: str,
    shape: list[int],
    *,
    data: np.ndarray | None = None,
    quantized: bool = False,
) -> TensorIR:
    return TensorIR(
        name=name,
        dtype=dtype,
        shape=list(shape),
        shape_signature=list(shape),
        data=data,
        is_variable=data is None,
        quantization=_qparams() if quantized else None,
    )


def _transpose_prelu_model(*, quantized_tail: bool) -> ModelIR:
    model_ir = ModelIR("transpose_quantized_prelu")
    model_ir.inputs = ["xq"]
    model_ir.outputs = ["yq" if quantized_tail else "z"]
    model_ir.tensors = {
        "xq": _tensor("xq", "INT8", [1, 2, 2, 3], quantized=True),
        "pre_perm": _tensor(
            "pre_perm",
            "INT32",
            [4],
            data=np.asarray([0, 3, 1, 2], dtype=np.int32),
        ),
        "aq": _tensor("aq", "INT8", [1, 3, 2, 2], quantized=True),
        "xf": _tensor("xf", "FLOAT32", [1, 3, 2, 2]),
        "alpha": _tensor(
            "alpha",
            "FLOAT32",
            [1, 3, 1, 1],
            data=np.asarray([[[[0.1]], [[0.2]], [[0.3]]]], dtype=np.float32),
        ),
        "yf": _tensor("yf", "FLOAT32", [1, 3, 2, 2]),
        "post_perm": _tensor(
            "post_perm",
            "INT32",
            [4],
            data=np.asarray([0, 2, 3, 1], dtype=np.int32),
        ),
        "y": _tensor("y", "FLOAT32", [1, 2, 2, 3]),
    }
    operators = [
        OperatorIR("TRANSPOSE", ["xq", "pre_perm"], ["aq"]),
        OperatorIR("DEQUANTIZE", ["aq"], ["xf"]),
        OperatorIR("PRELU", ["xf", "alpha"], ["yf"]),
    ]
    if quantized_tail:
        model_ir.tensors["yfq"] = _tensor(
            "yfq", "INT8", [1, 3, 2, 2], quantized=True
        )
        model_ir.tensors["yq"] = _tensor(
            "yq", "INT8", [1, 2, 2, 3], quantized=True
        )
        operators.extend(
            [
                OperatorIR("QUANTIZE", ["yf"], ["yfq"]),
                OperatorIR("TRANSPOSE", ["yfq", "post_perm"], ["yq"]),
            ]
        )
    else:
        model_ir.tensors["z"] = _tensor("z", "FLOAT32", [1, 2, 2, 3])
        operators.extend(
            [
                OperatorIR("TRANSPOSE", ["yf", "post_perm"], ["y"]),
                OperatorIR("IDENTITY", ["y"], ["z"]),
            ]
        )
    model_ir.operators = operators
    return model_ir


def test_transpose_dequant_prelu_quantize_bridge_characterization(monkeypatch) -> None:
    model_ir = _transpose_prelu_model(quantized_tail=True)
    refresh_count = 0
    original_refresh = ModelIRGraphIndex.refresh

    def counted_refresh(graph_index: ModelIRGraphIndex) -> None:
        nonlocal refresh_count
        refresh_count += 1
        original_refresh(graph_index)

    monkeypatch.setattr(ModelIRGraphIndex, "refresh", counted_refresh)
    diagnostics: list[dict] = []
    stats = run_quantized_prelu_cleanup(
        model_ir,
        include_transpose_bridge=False,
        include_quantize_fusion=False,
        include_depthwise_fusion=False,
        diagnostics=diagnostics,
    )

    assert stats["removed_transpose_dequant_prelu_quantize_bridges"] == 1
    assert sum(stats.values()) == 1
    assert [op.op_type for op in model_ir.operators] == [
        "DEQUANTIZE",
        "PRELU",
        "QUANTIZE",
    ]
    assert model_ir.operators[0].inputs == ["xq"]
    assert model_ir.operators[-1].outputs == ["yq"]
    assert model_ir.tensors["alpha"].shape == [1, 1, 1, 3]
    assert refresh_count == 1
    assert len(diagnostics) == 1
    assert diagnostics[0]["code"] == "layout.transpose_dequant_prelu_quantize"
    assert diagnostics[0]["status"] == "changed"
    assert diagnostics[0]["metrics"]["snapshot_count"] == 1


def test_transpose_dequant_prelu_transpose_bridge_characterization() -> None:
    model_ir = _transpose_prelu_model(quantized_tail=False)

    stats = run_quantized_prelu_cleanup(
        model_ir,
        include_transpose_quantize_bridge=False,
        include_quantize_fusion=False,
        include_depthwise_fusion=False,
    )

    assert stats["removed_transpose_dequant_prelu_transpose_bridges"] == 1
    assert sum(stats.values()) == 1
    assert [op.op_type for op in model_ir.operators] == [
        "DEQUANTIZE",
        "PRELU",
        "IDENTITY",
    ]
    assert model_ir.operators[0].inputs == ["xq"]
    assert model_ir.operators[1].outputs == ["y"]
    assert model_ir.tensors["alpha"].shape == [1, 1, 1, 3]


def test_dequant_prelu_quantize_characterization() -> None:
    model_ir = ModelIR("quantized_prelu")
    model_ir.inputs = ["xq"]
    model_ir.outputs = ["yq"]
    model_ir.tensors = {
        "xq": _tensor("xq", "INT8", [1, 2, 2, 3], quantized=True),
        "xf": _tensor("xf", "FLOAT32", [1, 2, 2, 3]),
        "alpha": _tensor(
            "alpha",
            "FLOAT32",
            [3],
            data=np.asarray([0.1, 0.2, 0.3], dtype=np.float32),
        ),
        "yf": _tensor("yf", "FLOAT32", [1, 2, 2, 3]),
        "yq": _tensor("yq", "INT8", [1, 2, 2, 3], quantized=True),
    }
    model_ir.operators = [
        OperatorIR("DEQUANTIZE", ["xq"], ["xf"]),
        OperatorIR("PRELU", ["xf", "alpha"], ["yf"]),
        OperatorIR("QUANTIZE", ["yf"], ["yq"]),
    ]

    stats = run_quantized_prelu_cleanup(
        model_ir,
        include_transpose_quantize_bridge=False,
        include_transpose_bridge=False,
        include_depthwise_fusion=False,
    )

    assert stats["folded_dequant_prelu_quantize_chains"] == 1
    assert sum(stats.values()) == 1
    assert [op.op_type for op in model_ir.operators] == ["PRELU"]
    assert model_ir.operators[0].inputs == ["xq", "alpha"]
    assert model_ir.operators[0].outputs == ["yq"]
    assert model_ir.tensors["alpha"].dtype == "INT8"
    assert model_ir.tensors["alpha"].quantization is not None


def test_dequant_prelu_depthwise_quantize_characterization() -> None:
    model_ir = ModelIR("quantized_prelu_depthwise")
    model_ir.inputs = ["xq"]
    model_ir.outputs = ["yq"]
    model_ir.tensors = {
        "xq": _tensor("xq", "INT8", [1, 2, 2, 2], quantized=True),
        "xf": _tensor("xf", "FLOAT32", [1, 2, 2, 2]),
        "alpha": _tensor(
            "alpha",
            "FLOAT32",
            [2],
            data=np.asarray([0.1, 0.2], dtype=np.float32),
        ),
        "pf": _tensor("pf", "FLOAT32", [1, 2, 2, 2]),
        "weights": _tensor(
            "weights",
            "FLOAT32",
            [1, 1, 1, 2],
            data=np.asarray([[[[0.5, -0.25]]]], dtype=np.float32),
        ),
        "bias": _tensor(
            "bias",
            "FLOAT32",
            [2],
            data=np.asarray([0.1, -0.1], dtype=np.float32),
        ),
        "yf": _tensor("yf", "FLOAT32", [1, 2, 2, 2]),
        "yq": _tensor("yq", "INT8", [1, 2, 2, 2], quantized=True),
    }
    model_ir.operators = [
        OperatorIR("DEQUANTIZE", ["xq"], ["xf"]),
        OperatorIR("PRELU", ["xf", "alpha"], ["pf"]),
        OperatorIR(
            "DEPTHWISE_CONV_2D",
            ["pf", "weights", "bias"],
            ["yf"],
        ),
        OperatorIR("QUANTIZE", ["yf"], ["yq"]),
    ]

    stats = run_quantized_prelu_cleanup(
        model_ir,
        include_transpose_quantize_bridge=False,
        include_transpose_bridge=False,
        include_quantize_fusion=False,
    )

    assert stats["folded_dequant_prelu_depthwise_quantize_chains"] == 1
    assert sum(stats.values()) == 1
    assert [op.op_type for op in model_ir.operators] == [
        "PRELU",
        "DEPTHWISE_CONV_2D",
    ]
    prelu_op, depthwise_op = model_ir.operators
    assert prelu_op.inputs[0] == "xq"
    assert depthwise_op.outputs == ["yq"]
    assert model_ir.tensors[prelu_op.inputs[1]].dtype == "INT8"
    assert model_ir.tensors[depthwise_op.inputs[1]].dtype == "INT8"
    assert model_ir.tensors[depthwise_op.inputs[2]].dtype == "INT32"


def test_quantized_prelu_runner_rejects_incomplete_chain_before_snapshot(
    monkeypatch,
) -> None:
    model_ir = ModelIR("incomplete_quantized_prelu")
    model_ir.inputs = ["xq"]
    model_ir.outputs = ["z"]
    model_ir.tensors = {
        "xq": _tensor("xq", "INT8", [1, 2], quantized=True),
        "xf": _tensor("xf", "FLOAT32", [1, 2]),
        "alpha": _tensor(
            "alpha",
            "FLOAT32",
            [2],
            data=np.asarray([0.1, 0.2], dtype=np.float32),
        ),
        "y": _tensor("y", "FLOAT32", [1, 2]),
        "z": _tensor("z", "FLOAT32", [1, 2]),
    }
    model_ir.operators = [
        OperatorIR("DEQUANTIZE", ["xq"], ["xf"]),
        OperatorIR("PRELU", ["xf", "alpha"], ["y"]),
        OperatorIR("IDENTITY", ["y"], ["z"]),
    ]
    refresh_count = 0
    original_refresh = ModelIRGraphIndex.refresh

    def counted_refresh(graph_index: ModelIRGraphIndex) -> None:
        nonlocal refresh_count
        refresh_count += 1
        original_refresh(graph_index)

    monkeypatch.setattr(ModelIRGraphIndex, "refresh", counted_refresh)
    diagnostics: list[dict] = []

    stats = run_quantized_prelu_cleanup(model_ir, diagnostics=diagnostics)

    assert sum(stats.values()) == 0
    assert refresh_count == 1
    assert len(diagnostics) == 4
    assert all(event["status"] == "skipped" for event in diagnostics)
    assert all(event["metrics"]["state_built"] is True for event in diagnostics)
    assert all(event["metrics"]["snapshot_count"] == 0 for event in diagnostics)
