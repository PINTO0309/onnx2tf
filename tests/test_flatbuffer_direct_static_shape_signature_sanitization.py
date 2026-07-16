from __future__ import annotations

import copy

import numpy as np
import pytest

from onnx2tf.tflite_builder.ir import ModelIR, OperatorIR, TensorIR
from onnx2tf.tflite_builder.lower_from_onnx2tf import (
    _sanitize_static_shape_signature_consistency,
)
from onnx2tf.tflite_builder.passes.static_shape_signature_sanitization import (
    sanitize_static_shape_signature_consistency,
)


def _signatures(model_ir: ModelIR) -> dict[str, list[int] | None]:
    return {
        name: (
            None
            if tensor.shape_signature is None
            else [int(value) for value in tensor.shape_signature]
        )
        for name, tensor in model_ir.tensors.items()
    }


def _tensor(
    name: str,
    shape: list[int],
    signature: list[int] | None,
    *,
    data: np.ndarray | None = None,
) -> TensorIR:
    return TensorIR(
        name=name,
        dtype="FLOAT32",
        shape=list(shape),
        shape_signature=(
            None if signature is None else list(signature)
        ),
        data=data,
    )


def test_static_scalar_missing_rank_and_stale_signatures_are_repaired() -> None:
    model_ir = ModelIR("static_signature_repairs")
    model_ir.tensors = {
        "scalar": _tensor("scalar", [], None),
        "missing": _tensor("missing", [2, 3], None),
        "rank_mismatch": _tensor("rank_mismatch", [2, 3], [2]),
        "stale": _tensor("stale", [2, 3], [2, 4]),
        "dynamic_runtime": _tensor(
            "dynamic_runtime", [-1, 3], [-1, 3]
        ),
    }

    stats = sanitize_static_shape_signature_consistency(model_ir)

    assert stats == {
        "sanitized_static_shape_signature_consistency": 4,
        "preserved_dynamic_boundary_shape_signature": 0,
        "preserved_dynamic_leading_axis_shape_signature": 0,
        "preserved_dynamic_lineage_shape_signature": 0,
    }
    assert _signatures(model_ir) == {
        "scalar": [],
        "missing": [2, 3],
        "rank_mismatch": [2, 3],
        "stale": [2, 3],
        "dynamic_runtime": [-1, 3],
    }
    before = _signatures(model_ir)
    assert sanitize_static_shape_signature_consistency(model_ir) == {
        "sanitized_static_shape_signature_consistency": 0,
        "preserved_dynamic_boundary_shape_signature": 0,
        "preserved_dynamic_leading_axis_shape_signature": 0,
        "preserved_dynamic_lineage_shape_signature": 0,
    }
    assert _signatures(model_ir) == before


def test_boundary_signature_map_restores_dynamic_contract() -> None:
    model_ir = ModelIR("boundary_signature_contract")
    model_ir.inputs = ["input"]
    model_ir.metadata["dynamic_boundary_shape_signature_map"] = {
        "input": [-1, 99]
    }
    model_ir.tensors["input"] = _tensor("input", [2, 3], [2, 99])

    stats = sanitize_static_shape_signature_consistency(model_ir)

    assert stats["sanitized_static_shape_signature_consistency"] == 1
    assert stats["preserved_dynamic_boundary_shape_signature"] == 1
    assert model_ir.tensors["input"].shape_signature == [-1, 3]


@pytest.mark.parametrize(
    ("op_type", "signature", "options", "shape_data"),
    [
        ("WHERE", [-1, 4], {}, None),
        ("RANGE", [-1], {}, None),
        ("RESHAPE", [-1, 4], {"newShape": [-1, 4]}, None),
        (
            "RESHAPE",
            [-1, 4],
            {"newShape": []},
            np.asarray([-1, 4], dtype=np.int32),
        ),
        ("TOPK_V2", [2, -1], {}, None),
    ],
)
def test_runtime_extent_producers_become_dynamic_lineage_roots(
    op_type: str,
    signature: list[int],
    options: dict,
    shape_data: np.ndarray | None,
) -> None:
    model_ir = ModelIR(f"{op_type.lower()}_dynamic_root")
    runtime_shape = [2 if value < 0 else value for value in signature]
    model_ir.tensors["source"] = _tensor("source", [2, 4], [2, 4])
    inputs = ["source"]
    if shape_data is not None:
        model_ir.tensors["shape"] = TensorIR(
            name="shape",
            dtype="INT32",
            shape=[2],
            shape_signature=[2],
            data=shape_data,
        )
        inputs.append("shape")
    model_ir.tensors["root"] = _tensor(
        "root", runtime_shape, signature
    )
    model_ir.operators = [
        OperatorIR(
            op_type=op_type,
            inputs=inputs,
            outputs=["root"],
            options=options,
        )
    ]

    stats = sanitize_static_shape_signature_consistency(model_ir)

    assert stats["sanitized_static_shape_signature_consistency"] == 0
    assert stats["preserved_dynamic_boundary_shape_signature"] == 1
    assert model_ir.tensors["root"].shape_signature == signature


def _dynamic_lineage_model() -> ModelIR:
    model_ir = ModelIR("recursive_dynamic_lineage")
    model_ir.inputs = ["input"]
    model_ir.metadata["onnx_dynamic_input_tensor_names"] = ["input"]
    model_ir.metadata["dynamic_boundary_shape_signature_map"] = {
        "input": [-1, 4]
    }
    model_ir.tensors = {
        "input": _tensor("input", [2, 4], [-1, 4]),
        "leading": _tensor("leading", [2, 4], [-1, 4]),
        "multi_axis": _tensor("multi_axis", [2, 4], [-1, -1]),
        "constant_payload": _tensor(
            "constant_payload",
            [2, 4],
            [-1, -1],
            data=np.zeros((2, 4), dtype=np.float32),
        ),
    }
    model_ir.operators = [
        OperatorIR(
            op_type="IDENTITY", inputs=["input"], outputs=["leading"]
        ),
        OperatorIR(
            op_type="IDENTITY", inputs=["leading"], outputs=["multi_axis"]
        ),
        OperatorIR(
            op_type="IDENTITY",
            inputs=["multi_axis"],
            outputs=["constant_payload"],
        ),
    ]
    return model_ir


def test_recursive_lineage_preserves_dynamic_axes_but_stops_at_constants() -> None:
    model_ir = _dynamic_lineage_model()

    stats = sanitize_static_shape_signature_consistency(model_ir)

    assert stats == {
        "sanitized_static_shape_signature_consistency": 1,
        "preserved_dynamic_boundary_shape_signature": 1,
        "preserved_dynamic_leading_axis_shape_signature": 1,
        "preserved_dynamic_lineage_shape_signature": 1,
    }
    assert model_ir.tensors["input"].shape_signature == [-1, 4]
    assert model_ir.tensors["leading"].shape_signature == [-1, 4]
    assert model_ir.tensors["multi_axis"].shape_signature == [-1, -1]
    assert model_ir.tensors["constant_payload"].shape_signature == [2, 4]


def test_graph_output_preserves_dynamic_leading_axis_without_lineage() -> None:
    model_ir = ModelIR("dynamic_graph_output")
    model_ir.outputs = ["output"]
    model_ir.tensors["output"] = _tensor("output", [2, 4], [-1, 4])

    stats = sanitize_static_shape_signature_consistency(model_ir)

    assert stats["preserved_dynamic_leading_axis_shape_signature"] == 1
    assert stats["sanitized_static_shape_signature_consistency"] == 0
    assert model_ir.tensors["output"].shape_signature == [-1, 4]


def test_cyclic_internal_lineage_terminates_and_is_sanitized() -> None:
    model_ir = ModelIR("cyclic_internal_lineage")
    model_ir.tensors = {
        "a": _tensor("a", [2, 3], [-1, -1]),
        "b": _tensor("b", [2, 3], [-1, -1]),
    }
    model_ir.operators = [
        OperatorIR(op_type="IDENTITY", inputs=["b"], outputs=["a"]),
        OperatorIR(op_type="IDENTITY", inputs=["a"], outputs=["b"]),
    ]

    stats = sanitize_static_shape_signature_consistency(model_ir)

    assert stats["sanitized_static_shape_signature_consistency"] == 2
    assert model_ir.tensors["a"].shape_signature == [2, 3]
    assert model_ir.tensors["b"].shape_signature == [2, 3]


def test_compatibility_wrapper_matches_module_owner() -> None:
    direct_model = _dynamic_lineage_model()
    wrapper_model = copy.deepcopy(direct_model)

    direct_stats = sanitize_static_shape_signature_consistency(direct_model)
    wrapper_stats = _sanitize_static_shape_signature_consistency(wrapper_model)

    assert wrapper_stats == direct_stats
    assert _signatures(wrapper_model) == _signatures(direct_model)
