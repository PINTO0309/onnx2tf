from __future__ import annotations

from pathlib import Path

from onnx import TensorProto, helper, save

from onnx2tf.utils.flatbuffer_direct_corpus_manifest import (
    build_corpus_manifest,
    node_count_tier,
)


def _write_model(path: Path, node_count: int) -> None:
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1])
    previous = "x"
    nodes = []
    for index in range(node_count):
        output = f"v{index}"
        nodes.append(helper.make_node("Relu", [previous], [output], name=f"relu{index}"))
        previous = output
    y = helper.make_tensor_value_info(previous, TensorProto.FLOAT, [1])
    model = helper.make_model(
        helper.make_graph(nodes, "g", [x], [y]),
        opset_imports=[helper.make_opsetid("", 13)],
    )
    save(model, path)


def test_node_count_tier_boundaries() -> None:
    assert [node_count_tier(value) for value in [1, 49, 50, 199, 200, 499]] == [0, 0, 1, 1, 2, 2]
    assert [node_count_tier(value) for value in [500, 999, 1000, 1999, 2000]] == [3, 3, 4, 4, 5]


def test_manifest_is_deterministic_and_classifies_invalid_models(tmp_path: Path) -> None:
    _write_model(tmp_path / "small.onnx", 2)
    (tmp_path / "invalid.onnx").write_bytes(b"not an onnx model")
    manifest = build_corpus_manifest(root_dir=str(tmp_path), hash_contents=True)
    assert manifest["model_count"] == 2
    assert manifest["loadable_count"] == 1
    assert manifest["invalid_count"] == 1
    assert manifest["tier_counts"]["0"] == 1
    by_name = {item["path"]: item for item in manifest["models"]}
    assert by_name["small.onnx"]["node_count"] == 2
    assert by_name["small.onnx"]["sha256"]
    assert by_name["invalid.onnx"]["status"] == "invalid_onnx"
