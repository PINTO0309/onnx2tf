from __future__ import annotations

from onnx2tf.tflite_builder.ir import ModelIR, OperatorIR, TensorIR
from onnx2tf.tflite_builder.pytorch_source_graph_rewrites import (
    _bridge_boundary_metadata_gather_nd_inputs,
    _infer_gather_nd_shape_for_codegen,
)


def _make_boundary_gather_model(*, output_shape: list[int]) -> ModelIR:
    model_ir = ModelIR(name="boundary_metadata_gather_nd")
    model_ir.inputs = ["data", "indices"]
    model_ir.outputs = ["output"]
    model_ir.tensors["data"] = TensorIR(
        name="data",
        dtype="FLOAT32",
        shape=[1, 17, 48, 64],
        shape_signature=[1, 17, 48, 64],
        logical_layout="NCHW",
    )
    model_ir.tensors["indices"] = TensorIR(
        name="indices",
        dtype="INT32",
        shape=[6, 3],
        shape_signature=[6, 3],
        logical_layout="UNKNOWN",
    )
    model_ir.tensors["output"] = TensorIR(
        name="output",
        dtype="FLOAT32",
        shape=list(output_shape),
        shape_signature=list(output_shape),
        logical_layout="UNKNOWN",
    )
    model_ir.operators.append(
        OperatorIR(
            op_type="GATHER_ND",
            inputs=["data", "indices"],
            outputs=["output"],
            options={},
        )
    )
    return model_ir


def test_infer_gather_nd_shape_uses_index_depth() -> None:
    model_ir = _make_boundary_gather_model(output_shape=[6, 17])

    assert _infer_gather_nd_shape_for_codegen(
        model_ir=model_ir,
        params_shape=[1, 17, 48, 64],
        indices_tensor_name="indices",
    ) == [6, 64]
    assert _infer_gather_nd_shape_for_codegen(
        model_ir=model_ir,
        params_shape=[1, 48, 64, 17],
        indices_tensor_name="indices",
    ) == [6, 17]


def test_boundary_gather_rewrite_inserts_required_layout_bridge() -> None:
    model_ir = _make_boundary_gather_model(output_shape=[6, 17])

    assert _bridge_boundary_metadata_gather_nd_inputs(
        ["output=_apply_gather_nd(data, indices, target_shape=[6, 17])"],
        model_ir=model_ir,
        tensor_var_names={"data": "data", "indices": "indices", "output": "output"},
    ) == [
        "output = _apply_gather_nd(_torch_permute(data, [0, 2, 3, 1]), "
        "indices, target_shape=[6, 17])"
    ]


def test_boundary_gather_rewrite_collapses_duplicate_permute() -> None:
    model_ir = _make_boundary_gather_model(output_shape=[6, 17])

    assert _bridge_boundary_metadata_gather_nd_inputs(
        [
            "output=_apply_gather_nd(_torch_permute(data, [0, 2, 3, 1]).permute(0, 2, 3, 1).contiguous(), indices, target_shape=[6, 17])"
        ],
        model_ir=model_ir,
        tensor_var_names={"data": "data", "indices": "indices", "output": "output"},
    ) == [
        "output = _apply_gather_nd(_torch_permute(data, [0, 2, 3, 1]), "
        "indices, target_shape=[6, 17])"
    ]


def test_boundary_gather_rewrite_is_noop_when_original_layout_matches() -> None:
    model_ir = _make_boundary_gather_model(output_shape=[6, 64])
    lines = ["output=_apply_gather_nd(data, indices, target_shape=[6, 64])"]

    assert (
        _bridge_boundary_metadata_gather_nd_inputs(
            lines,
            model_ir=model_ir,
            tensor_var_names={
                "data": "data",
                "indices": "indices",
                "output": "output",
            },
        )
        == lines
    )
