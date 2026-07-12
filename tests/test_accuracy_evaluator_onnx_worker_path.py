from __future__ import annotations

import queue
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper

from onnx2tf.tflite_builder.accuracy_evaluator import (
    _all_input_shapes_are_static,
    _onnx_inference_worker,
)


def _make_add_model() -> onnx.ModelProto:
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 3])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 3])
    z = helper.make_tensor_value_info("z", TensorProto.FLOAT, [1, 3])
    model = helper.make_model(
        helper.make_graph(
            [helper.make_node("Add", ["x", "y"], ["z"])],
            "worker_path_add",
            [x, y],
            [z],
        ),
        opset_imports=[helper.make_operatorsetid("", 13)],
    )
    model.ir_version = 10
    return model


def test_onnx_inference_worker_loads_graph_from_path(tmp_path: Path) -> None:
    model_path = tmp_path / "add.onnx"
    onnx.save(_make_add_model(), model_path)
    result_queue: queue.Queue = queue.Queue()
    x = np.asarray([[1.0, 2.0, 3.0]], dtype=np.float32)
    y = np.asarray([[4.0, 5.0, 6.0]], dtype=np.float32)

    _onnx_inference_worker(
        {
            "onnx_graph_path": str(model_path),
            "onnx_output_names": ["z"],
            "onnx_inputs": {"x": x, "y": y},
            "use_memmap_outputs": False,
        },
        result_queue,
    )

    result = result_queue.get_nowait()
    assert result["ok"] is True
    np.testing.assert_array_equal(
        result["onnx_outputs"]["z"],
        np.asarray([[5.0, 7.0, 9.0]], dtype=np.float32),
    )


def test_default_delegates_require_fully_static_input_shapes() -> None:
    assert _all_input_shapes_are_static(
        [("image", np.dtype(np.float32), [1, 3, 1024, 1024])]
    )
    assert not _all_input_shapes_are_static(
        [("image", np.dtype(np.float32), [1, 3, -1, -1])]
    )
