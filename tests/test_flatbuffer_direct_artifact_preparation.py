from __future__ import annotations

import ast
from pathlib import Path

import numpy as np
import pytest

from onnx2tf.tflite_builder.artifact_preparation import (
    isolate_float32_model_ir_for_tflite_write,
    resolve_requested_artifact_controls,
    resolve_requested_exporter_controls,
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


def test_unrequested_artifact_controls_do_not_read_related_options(
    monkeypatch,
) -> None:
    class _RejectingOptions(dict):
        def get(self, key, default=None):
            raise AssertionError(f"unrequested option was read: {key}")

    monkeypatch.setenv(
        "ONNX2TF_FLATBUFFER_DIRECT_SPLIT_MAX_BYTES",
        "invalid-unrequested-value",
    )
    monkeypatch.setenv(
        "ONNX2TF_FLATBUFFER_DIRECT_CALIBRATION_PERCENTILE",
        "invalid-unrequested-value",
    )

    controls = resolve_requested_artifact_controls(
        _RejectingOptions(),
        split_plan_requested=False,
        quantization_requested=False,
        default_split_max_bytes=1024,
        default_split_target_bytes=768,
    )

    assert controls.split_max_bytes is None
    assert controls.split_target_bytes is None
    assert controls.quantization is None


def test_requested_artifact_controls_preserve_existing_option_values() -> None:
    controls = resolve_requested_artifact_controls(
        {
            "tflite_split_max_bytes": "2048",
            "tflite_split_target_bytes": 1536,
            "flatbuffer_direct_calibration_method": "percentile",
            "flatbuffer_direct_calibration_percentile": "98.5",
            "flatbuffer_direct_quant_min_numel": "17",
            "flatbuffer_direct_quant_min_abs_max": "0.25",
            "flatbuffer_direct_quant_scale_floor": "1e-6",
        },
        split_plan_requested=True,
        quantization_requested=True,
        default_split_max_bytes=1024,
        default_split_target_bytes=768,
    )

    assert controls.split_max_bytes == 2048
    assert controls.split_target_bytes == 1536
    assert dict(controls.quantization or {}) == {
        "calibration_method": "percentile",
        "calibration_percentile": 98.5,
        "min_numel": 17,
        "min_abs_max": 0.25,
        "scale_floor": 1e-6,
    }
    with pytest.raises(TypeError):
        controls.quantization["min_numel"] = 1  # type: ignore[index]


def test_unrequested_exporter_controls_do_not_read_related_options() -> None:
    class _RejectingOptions(dict):
        def get(self, key, default=None):
            raise AssertionError(f"unrequested option was read: {key}")

    controls = resolve_requested_exporter_controls(
        _RejectingOptions(),
        output_folder_path="artifacts",
        output_file_name="model",
        saved_model_requested=False,
        pytorch_requested=False,
        calibration_inputs_requested=False,
    )

    assert controls.saved_model_output_folder_path == "artifacts"
    assert controls.persist_saved_model_output is False
    assert controls.pytorch_output_folder_path == "artifacts/model_pytorch"
    assert controls.native_pytorch_generation_timeout_sec == 0
    assert controls.custom_input_op_name_np_data_path is None
    assert controls.shape_hints is None
    assert controls.test_data_nhwc_path is None


def test_requested_exporter_controls_preserve_values_and_dependencies() -> None:
    controls = resolve_requested_exporter_controls(
        {
            "saved_model_output_folder_path": "saved",
            "persist_saved_model_output": False,
            "pytorch_output_folder_path": "torch",
            "native_pytorch_generation_timeout_sec": "37",
            "custom_input_op_name_np_data_path": [["input", "sample.npy"]],
            "shape_hints": {"input": [1, 3, 8, 8]},
            "test_data_nhwc_path": "sample.npy",
        },
        output_folder_path="artifacts",
        output_file_name="model",
        saved_model_requested=True,
        pytorch_requested=True,
        calibration_inputs_requested=False,
    )

    assert controls.saved_model_output_folder_path == "saved"
    assert controls.persist_saved_model_output is False
    assert controls.pytorch_output_folder_path == "torch"
    assert controls.native_pytorch_generation_timeout_sec == 37
    assert controls.custom_input_op_name_np_data_path == [["input", "sample.npy"]]
    assert controls.shape_hints == {"input": [1, 3, 8, 8]}
    assert controls.test_data_nhwc_path == "sample.npy"

    calibration_only = resolve_requested_exporter_controls(
        {"custom_input_op_name_np_data_path": "calibration.npy"},
        output_folder_path="artifacts",
        output_file_name="model",
        saved_model_requested=False,
        pytorch_requested=False,
        calibration_inputs_requested=True,
    )
    assert calibration_only.custom_input_op_name_np_data_path == "calibration.npy"
    assert calibration_only.native_pytorch_generation_timeout_sec == 0

    with pytest.raises(ValueError):
        resolve_requested_exporter_controls(
            {"native_pytorch_generation_timeout_sec": "invalid"},
            output_folder_path="artifacts",
            output_file_name="model",
            saved_model_requested=False,
            pytorch_requested=True,
            calibration_inputs_requested=False,
        )


def test_artifact_control_resolution_has_one_policy_owner() -> None:
    builder_source = (
        REPO_ROOT / "onnx2tf" / "tflite_builder" / "__init__.py"
    ).read_text(encoding="utf-8")

    assert builder_source.count("resolve_requested_artifact_controls(") == 1
    assert builder_source.count("resolve_requested_exporter_controls(") == 1
    assert "def _resolve_quantization_controls" not in builder_source
    assert "ONNX2TF_FLATBUFFER_DIRECT_QUANT_MIN_NUMEL" not in builder_source
    assert "ONNX2TF_FLATBUFFER_DIRECT_SPLIT_MAX_BYTES" not in builder_source
    assert '"native_pytorch_generation_timeout_sec"' not in builder_source


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
        if (
            isinstance(graph_index_keyword, ast.Name)
            and node.args[0].id
            in {"model_ir_fp32_tflite", "model_ir_fp16_tflite"}
        ):
            validation_index_by_model[node.args[0].id] = graph_index_keyword.id

    assert validation_index_by_model == {
        "model_ir_fp32_tflite": "fp32_write_graph_index",
        "model_ir_fp16_tflite": "fp16_write_graph_index",
    }


def test_split_artifact_stages_reuse_source_graph_index() -> None:
    source = (
        REPO_ROOT / "onnx2tf" / "tflite_builder" / "__init__.py"
    ).read_text(encoding="utf-8")
    builder_tree = ast.parse(source)
    export_function = next(
        node
        for node in builder_tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "export_tflite_model_flatbuffer_direct"
    )
    expected_calls = {
        "run_model_ir_validation_pipeline": "model_ir",
        "plan_contiguous_partitions_by_size": "model_ir",
        "write_split_model_files_and_manifest": "model_ir",
    }
    matched_calls = set()
    for node in ast.walk(export_function):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Name):
            continue
        expected_model_name = expected_calls.get(node.func.id)
        if expected_model_name is None:
            continue
        model_argument = (
            node.args[0]
            if node.args
            else next(
                (
                    keyword.value
                    for keyword in node.keywords
                    if keyword.arg == "model_ir"
                ),
                None,
            )
        )
        if not (
            isinstance(model_argument, ast.Name)
            and model_argument.id == expected_model_name
        ):
            continue
        index_argument = next(
            (
                keyword.value
                for keyword in node.keywords
                if keyword.arg == "graph_index"
            ),
            None,
        )
        if index_argument is None:
            continue
        assert isinstance(index_argument, ast.Name)
        assert index_argument.id == "split_graph_index"
        matched_calls.add(node.func.id)

    assert matched_calls == set(expected_calls)


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
        "split_graph_index",
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
