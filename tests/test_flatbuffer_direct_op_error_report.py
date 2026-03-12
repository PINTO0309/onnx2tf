import onnx
from onnx import TensorProto, helper

from onnx2tf.tflite_builder.accuracy_evaluator import (
    _collect_nondeterministic_onnx_tensor_reasons,
)
from onnx2tf.utils.flatbuffer_direct_op_error_report import (
    _sanitize_input_name_for_filename,
)


def test_sanitize_input_name_for_filename_replaces_path_separators() -> None:
    assert _sanitize_input_name_for_filename("gpu_0/data_0") == "gpu_0_data_0"
    assert _sanitize_input_name_for_filename(r"gpu_0\\data_0") == "gpu_0_data_0"


def test_sanitize_input_name_for_filename_handles_empty_like_name() -> None:
    assert _sanitize_input_name_for_filename("///") == "input"


def test_collect_nondeterministic_onnx_tensor_reasons_skips_only_unseeded_random_paths() -> None:
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [2, 3])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [2, 3])
    graph = helper.make_graph(
        [
            helper.make_node("Abs", ["x"], ["det"], name="DetAbsNode"),
            helper.make_node("RandomNormalLike", ["x"], ["rand"], name="UnseededRandomNode"),
            helper.make_node("Add", ["rand", "x"], ["y"], name="RandomAddNode"),
            helper.make_node(
                "RandomUniformLike",
                ["x"],
                ["seeded_rand"],
                name="SeededRandomNode",
                seed=13.0,
            ),
        ],
        "nondeterministic_random_skip_graph",
        [x],
        [y],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_operatorsetid("", 19)])
    model.ir_version = 10

    reasons = _collect_nondeterministic_onnx_tensor_reasons(model)

    assert reasons["rand"].startswith("depends_on_unseeded_random_op: UnseededRandomNode(RandomNormalLike)")
    assert reasons["y"].startswith("depends_on_unseeded_random_op: UnseededRandomNode(RandomNormalLike)")
    assert "det" not in reasons
    assert "seeded_rand" not in reasons
