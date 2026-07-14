from __future__ import annotations

import ast
from pathlib import Path

import numpy as np
import pytest

from onnx2tf.tflite_builder.artifact_preparation import (
    isolate_float32_model_ir_for_tflite_write,
)
from onnx2tf.tflite_builder.core.model_ir_pass_state import ModelIRPassState
from onnx2tf.tflite_builder.ir import (
    ModelIR,
    OperatorIR,
    QuantParamIR,
    TensorIR,
    clone_model_ir_with_float16,
    clone_model_ir_with_float32,
)


REPO_ROOT = Path(__file__).resolve().parents[1]


def _make_model_ir() -> ModelIR:
    model_ir = ModelIR(
        name="artifact_preparation",
        metadata={"artifact": {"version": 1}},
    )
    model_ir.inputs = ["x"]
    model_ir.outputs = ["y"]
    model_ir.tensors["x"] = TensorIR(
        name="x",
        dtype="FLOAT32",
        shape=[1],
        data=np.asarray([1.0], dtype=np.float32),
    )
    model_ir.tensors["scale"] = TensorIR(
        name="scale",
        dtype="FLOAT16",
        shape=[1],
        data=np.asarray([0.5], dtype=np.float16),
        quantization=QuantParamIR(
            scale=[0.25],
            zero_point=[0],
            quantized_dimension=0,
            min=[-1.0],
            max=[1.0],
        ),
    )
    model_ir.tensors["y"] = TensorIR(
        name="y",
        dtype="FLOAT32",
        shape=[1],
    )
    model_ir.operators = [
        OperatorIR(
            op_type="RELU",
            inputs=["x"],
            outputs=["y"],
            options={"inDataType": "FLOAT16", "nested": ["FLOAT16"]},
        ),
    ]
    subgraph = ModelIR(name="body")
    subgraph.tensors["body_value"] = TensorIR(
        name="body_value",
        dtype="FLOAT32",
        shape=[1],
        data=np.asarray([3.0], dtype=np.float32),
    )
    model_ir.subgraphs = [subgraph]
    return model_ir


@pytest.mark.parametrize(
    (
        "split_manifest_path",
        "output_saved_model_from_model_ir",
        "output_pytorch_from_model_ir",
        "expect_isolated",
    ),
    [
        (None, False, False, False),
        (None, True, False, True),
        (None, False, True, True),
        (None, True, True, True),
        ("model_split_manifest.json", False, False, False),
        ("model_split_manifest.json", True, False, False),
        ("model_split_manifest.json", False, True, False),
        ("model_split_manifest.json", True, True, False),
    ],
)
def test_float32_tflite_write_isolates_only_ir_needed_by_later_exporters(
    split_manifest_path: str | None,
    output_saved_model_from_model_ir: bool,
    output_pytorch_from_model_ir: bool,
    expect_isolated: bool,
) -> None:
    model_ir = _make_model_ir()

    prepared = isolate_float32_model_ir_for_tflite_write(
        model_ir,
        split_manifest_path=split_manifest_path,
        output_saved_model_from_model_ir=output_saved_model_from_model_ir,
        output_pytorch_from_model_ir=output_pytorch_from_model_ir,
    )

    assert (prepared is not model_ir) is expect_isolated
    if expect_isolated:
        assert prepared.operators[0] is not model_ir.operators[0]
        assert prepared.tensors["x"].data is not model_ir.tensors["x"].data
        prepared.operators[0].op_type = "ABS"
        prepared.tensors["x"].data[0] = 2.0
        assert model_ir.operators[0].op_type == "RELU"
        np.testing.assert_array_equal(
            model_ir.tensors["x"].data,
            np.asarray([1.0], dtype=np.float32),
        )


@pytest.mark.parametrize(
    "clone_precision_ir",
    [clone_model_ir_with_float32, clone_model_ir_with_float16],
)
def test_reused_terminal_precision_ir_matches_legacy_second_clone(
    clone_precision_ir,
) -> None:
    terminal_precision_ir = clone_precision_ir(_make_model_ir())
    legacy_second_clone = clone_precision_ir(terminal_precision_ir)

    assert (
        ModelIRPassState(terminal_precision_ir).fingerprint()
        == ModelIRPassState(legacy_second_clone).fingerprint()
    )
    assert terminal_precision_ir.metadata == legacy_second_clone.metadata


def test_float16_tflite_write_reuses_its_terminal_precision_ir() -> None:
    builder_tree = ast.parse(
        (REPO_ROOT / "onnx2tf" / "tflite_builder" / "__init__.py").read_text(
            encoding="utf-8"
        )
    )
    export_function = next(
        node
        for node in builder_tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "export_tflite_model_flatbuffer_direct"
    )
    assignments = [
        node
        for node in ast.walk(export_function)
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == "model_ir_fp16_tflite"
            for target in node.targets
        )
    ]

    assert len(assignments) == 1
    assert isinstance(assignments[0].value, ast.Name)
    assert assignments[0].value.id == "model_ir_fp16"


def test_precision_validation_reuses_write_graph_indexes() -> None:
    builder_tree = ast.parse(
        (REPO_ROOT / "onnx2tf" / "tflite_builder" / "__init__.py").read_text(
            encoding="utf-8"
        )
    )
    export_function = next(
        node
        for node in builder_tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "export_tflite_model_flatbuffer_direct"
    )
    validation_index_by_model = {}
    for node in ast.walk(export_function):
        if not (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "run_model_ir_validation_pipeline"
            and node.args
            and isinstance(node.args[0], ast.Name)
        ):
            continue
        graph_index_keyword = next(
            (
                keyword.value
                for keyword in node.keywords
                if keyword.arg == "graph_index"
            ),
            None,
        )
        if isinstance(graph_index_keyword, ast.Name):
            validation_index_by_model[node.args[0].id] = graph_index_keyword.id

    assert validation_index_by_model == {
        "model_ir_fp32_tflite": "fp32_write_graph_index",
        "model_ir_fp16_tflite": "fp16_write_graph_index",
    }


def test_terminal_artifact_state_is_released_after_its_last_use() -> None:
    builder_tree = ast.parse(
        (REPO_ROOT / "onnx2tf" / "tflite_builder" / "__init__.py").read_text(
            encoding="utf-8"
        )
    )
    export_function = next(
        node
        for node in builder_tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "export_tflite_model_flatbuffer_direct"
    )

    terminal_names = {
        "fp32_write_graph_index",
        "model_ir_fp32_tflite",
        "model_ir_fp32",
        "fp16_write_graph_index",
        "model_ir_fp16_tflite",
        "model_ir_fp16",
        "dynamic_model_ir",
        "integer_model_ir",
        "integer_result",
        "full_integer_model_ir",
        "full_integer_result",
        "integer_quant_with_int16_act_model_ir",
        "full_integer_quant_with_int16_act_model_ir",
        "calibration_samples",
        "calibration_ranges",
        "calibration_report",
    }
    release_lines = {
        name: [
            node.lineno
            for node in ast.walk(export_function)
            if isinstance(node, ast.Delete)
            for target in node.targets
            if isinstance(target, ast.Name) and target.id == name
        ]
        for name in terminal_names
    }
    for name in terminal_names:
        release_lines[name].extend(
            node.lineno
            for node in ast.walk(export_function)
            if isinstance(node, ast.Assign)
            and isinstance(node.value, ast.Constant)
            and node.value.value is None
            for target in node.targets
            if isinstance(target, ast.Name) and target.id == name
        )
    released_at = {
        name: max(lines)
        for name, lines in release_lines.items()
        if lines
    }

    assert set(released_at) == terminal_names
    for name in sorted(terminal_names):
        load_lines = [
            node.lineno
            for node in ast.walk(export_function)
            if isinstance(node, ast.Name)
            and node.id == name
            and isinstance(node.ctx, ast.Load)
        ]
        assert load_lines
        assert max(load_lines) < released_at[name]
