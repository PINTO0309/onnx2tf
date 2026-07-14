from __future__ import annotations

import json

import onnx
import pytest
from onnx import TensorProto, helper

from onnx2tf.tflite_builder.ir import ModelIR, TensorIR
from onnx2tf.tflite_builder.pytorch_export_errors import ModelIRPyTorchExportError
from onnx2tf.tflite_builder.pytorch_string_normalizer_exporter import (
    _extract_string_normalizer_config_from_onnx_graph,
    export_pytorch_package_from_string_normalizer_onnx,
)


def _string_normalizer_model() -> onnx.ModelProto:
    node = helper.make_node(
        "StringNormalizer",
        ["text"],
        ["normalized"],
        domain="ai.onnx.ml",
        case_change_action="LOWER",
        is_case_sensitive=0,
        locale="en_US",
        stopwords=["A", "THE"],
    )
    graph = helper.make_graph(
        [node],
        "string_normalizer",
        [helper.make_tensor_value_info("text", TensorProto.STRING, [1])],
        [helper.make_tensor_value_info("normalized", TensorProto.STRING, [1])],
    )
    return helper.make_model(
        graph,
        opset_imports=[helper.make_opsetid("ai.onnx.ml", 3)],
    )


def test_string_normalizer_config_decodes_attributes() -> None:
    assert _extract_string_normalizer_config_from_onnx_graph(
        _string_normalizer_model()
    ) == {
        "input_name": "text",
        "output_name": "normalized",
        "case_change_action": "LOWER",
        "is_case_sensitive": False,
        "locale": "en_US",
        "stopwords": ["A", "THE"],
    }


@pytest.mark.parametrize(
    "invalid_graph", [None, helper.make_model(helper.make_graph([], "empty", [], []))]
)
def test_string_normalizer_config_rejects_invalid_graph(invalid_graph) -> None:
    assert _extract_string_normalizer_config_from_onnx_graph(invalid_graph) is None


def test_string_normalizer_package_uses_shared_scaffolding_and_metadata(
    tmp_path,
) -> None:
    model_ir = ModelIR(
        name="string_normalizer",
        inputs=["text"],
        outputs=["normalized"],
    )
    model_ir.tensors = {
        "text": TensorIR(name="text", dtype="STRING", shape=[1]),
        "normalized": TensorIR(name="normalized", dtype="STRING", shape=[1]),
    }

    result = export_pytorch_package_from_string_normalizer_onnx(
        model_ir=model_ir,
        output_folder_path=str(tmp_path),
        onnx_graph=_string_normalizer_model(),
    )

    assert result == str(tmp_path)
    assert (tmp_path / "__init__.py").is_file()
    assert (tmp_path / "runtime.py").is_file()
    assert (tmp_path / "model.py").is_file()
    metadata = json.loads((tmp_path / "metadata.json").read_text(encoding="utf-8"))
    assert metadata["execution_backend"] == "string_normalizer"
    assert metadata["inputs"] == ["text"]
    assert metadata["outputs"] == ["normalized"]
    assert metadata["string_normalizer"]["stopwords"] == ["A", "THE"]


def test_string_normalizer_package_rejects_invalid_graph_before_writing(
    tmp_path,
) -> None:
    output_path = tmp_path / "rejected"

    with pytest.raises(ModelIRPyTorchExportError, match="single-op StringNormalizer"):
        export_pytorch_package_from_string_normalizer_onnx(
            model_ir=ModelIR(name="invalid"),
            output_folder_path=str(output_path),
            onnx_graph=None,
        )

    assert not output_path.exists()
