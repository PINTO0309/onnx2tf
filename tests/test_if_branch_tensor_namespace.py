from __future__ import annotations

import numpy as np
from onnx import TensorProto, helper, numpy_helper

from onnx2tf.tflite_builder.core import (
    ModelIRGraphIndex,
    validate_model_ir_invariants,
)
from onnx2tf.tflite_builder.lower_from_onnx2tf import lower_onnx_to_ir


def _make_if_with_colliding_branch_local_names():
    cond = helper.make_tensor_value_info("cond", TensorProto.BOOL, [])
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1])

    def _branch(*, graph_name: str, output_name: str, bias: float):
        initializer = numpy_helper.from_array(
            np.asarray([bias], dtype=np.float32),
            name="shared_initializer",
        )
        nodes = [
            helper.make_node(
                "Add",
                ["x", "shared_initializer"],
                ["shared_internal"],
                name="SharedAdd",
            ),
            helper.make_node(
                "Identity",
                ["shared_internal"],
                [output_name],
                name="BranchIdentity",
            ),
        ]
        return helper.make_graph(
            nodes,
            graph_name,
            [],
            [helper.make_tensor_value_info(output_name, TensorProto.FLOAT, [1])],
            initializer=[initializer],
        )

    if_node = helper.make_node(
        "If",
        ["cond"],
        ["y"],
        name="NamespacedIf",
        then_branch=_branch(
            graph_name="then_graph",
            output_name="then_output",
            bias=1.0,
        ),
        else_branch=_branch(
            graph_name="else_graph",
            output_name="else_output",
            bias=2.0,
        ),
    )
    graph = helper.make_graph([if_node], "if_namespace", [cond, x], [y])
    return helper.make_model(
        graph,
        opset_imports=[helper.make_operatorsetid("", 13)],
    )


def test_if_branches_namespace_internal_tensors_and_initializers() -> None:
    model_ir = lower_onnx_to_ir(
        _make_if_with_colliding_branch_local_names(),
        output_file_name="if_branch_tensor_namespace",
    )

    assert ModelIRGraphIndex(model_ir).duplicate_producers == {}
    assert validate_model_ir_invariants(model_ir) == []
    assert "shared_internal" not in model_ir.tensors
    assert "shared_initializer" not in model_ir.tensors

    then_initializer = model_ir.tensors["NamespacedIf_if_then_initializer_0"]
    else_initializer = model_ir.tensors["NamespacedIf_if_else_initializer_0"]
    np.testing.assert_array_equal(
        np.asarray(then_initializer.data),
        np.asarray([1.0], dtype=np.float32),
    )
    np.testing.assert_array_equal(
        np.asarray(else_initializer.data),
        np.asarray([2.0], dtype=np.float32),
    )
