import numpy as np
import pytest

from onnx2tf.tflite_builder.ir import ModelIR, OperatorIR, TensorIR
from onnx2tf.tflite_builder.split_planner import (
    build_partition_model_ir,
    find_dependency_safe_split_points,
    plan_contiguous_partitions_by_size,
    validate_partition_ranges,
)


def _make_chain_model_ir(op_count: int = 6) -> ModelIR:
    model = ModelIR(name="chain")
    model.inputs = ["x"]
    model.outputs = [f"t{op_count - 1}"]
    model.tensors["x"] = TensorIR(name="x", dtype="FLOAT32", shape=[1])
    for idx in range(op_count):
        out_name = f"t{idx}"
        in_name = "x" if idx == 0 else f"t{idx - 1}"
        model.operators.append(
            OperatorIR(
                op_type="DUMMY",
                inputs=[in_name],
                outputs=[out_name],
                options={},
                version=1,
            )
        )
        model.tensors[out_name] = TensorIR(name=out_name, dtype="FLOAT32", shape=[1])
    return model


def test_split_planner_converges_to_target() -> None:
    model_ir = _make_chain_model_ir(op_count=6)
    report = plan_contiguous_partitions_by_size(
        model_ir=model_ir,
        target_max_bytes=250,
        hard_max_bytes=350,
        size_estimator=lambda part: len(part.operators) * 100,
    )
    assert report["plan_valid"] is True
    ranges = [
        (part["start_op_index"], part["end_op_index"])
        for part in report["partitions"]
    ]
    assert ranges == [(0, 2), (2, 4), (4, 6)]
    assert all(part["estimated_bytes"] <= 250 for part in report["partitions"])
    assert len(report["edges"]) == 2


def test_validate_partition_ranges_rejects_gap() -> None:
    model_ir = _make_chain_model_ir(op_count=6)
    with pytest.raises(ValueError):
        validate_partition_ranges(
            model_ir=model_ir,
            partition_ranges=[(0, 2), (3, 6)],
        )


def test_dependency_safe_split_points_chain() -> None:
    model_ir = _make_chain_model_ir(op_count=6)
    points = find_dependency_safe_split_points(model_ir)
    assert [point["index"] for point in points] == [1, 2, 3, 4, 5]


def test_build_partition_model_ir_prunes_dead_branch_ops_and_inputs() -> None:
    model_ir = ModelIR(name="branch_prune")
    model_ir.inputs = ["x", "y"]
    model_ir.outputs = ["b"]
    model_ir.tensors["x"] = TensorIR(name="x", dtype="FLOAT32", shape=[1])
    model_ir.tensors["y"] = TensorIR(name="y", dtype="FLOAT32", shape=[1])
    model_ir.tensors["a"] = TensorIR(name="a", dtype="FLOAT32", shape=[1])
    model_ir.tensors["b"] = TensorIR(name="b", dtype="FLOAT32", shape=[1])
    model_ir.tensors["dead"] = TensorIR(name="dead", dtype="FLOAT32", shape=[1])
    model_ir.operators.extend(
        [
            OperatorIR(
                op_type="DUMMY",
                inputs=["x"],
                outputs=["a"],
                options={},
                version=1,
            ),
            OperatorIR(
                op_type="DUMMY",
                inputs=["a"],
                outputs=["b"],
                options={},
                version=1,
            ),
            OperatorIR(
                op_type="DUMMY",
                inputs=["y"],
                outputs=["dead"],
                options={},
                version=1,
            ),
        ]
    )

    part_model = build_partition_model_ir(
        model_ir=model_ir,
        start_op_index=0,
        end_op_index=3,
        partition_id=1,
    )

    assert [op.outputs[0] for op in part_model.operators] == ["a", "b"]
    assert part_model.inputs == ["x"]
    assert part_model.outputs == ["b"]
    assert "dead" not in part_model.tensors
    assert "y" not in part_model.tensors


def test_build_partition_model_ir_excludes_embedded_constants_from_partition_inputs() -> None:
    model_ir = ModelIR(name="constant_boundary_inputs")
    model_ir.inputs = ["x"]
    model_ir.outputs = ["z"]
    model_ir.tensors["x"] = TensorIR(name="x", dtype="FLOAT32", shape=[1, 1, 4, 4])
    model_ir.tensors["weight"] = TensorIR(
        name="weight",
        dtype="FLOAT32",
        shape=[1, 1, 3, 3],
        data=np.ones((1, 1, 3, 3), dtype=np.float32),
    )
    model_ir.tensors["bias"] = TensorIR(
        name="bias",
        dtype="FLOAT32",
        shape=[1],
        data=np.zeros((1,), dtype=np.float32),
    )
    model_ir.tensors["conv_out"] = TensorIR(name="conv_out", dtype="FLOAT32", shape=[1, 1, 2, 2])
    model_ir.tensors["z"] = TensorIR(name="z", dtype="FLOAT32", shape=[1, 1, 2, 2])
    model_ir.operators.extend(
        [
            OperatorIR(
                op_type="CONV_2D",
                inputs=["x", "weight", "bias"],
                outputs=["conv_out"],
                options={"padding": "VALID"},
                version=1,
            ),
            OperatorIR(
                op_type="RELU",
                inputs=["conv_out"],
                outputs=["z"],
                options={},
                version=1,
            ),
        ]
    )

    part_model = build_partition_model_ir(
        model_ir=model_ir,
        start_op_index=0,
        end_op_index=2,
        partition_id=1,
    )

    assert part_model.inputs == ["x"]
    assert part_model.outputs == ["z"]
    assert "weight" in part_model.tensors
    assert "bias" in part_model.tensors
    assert part_model.tensors["weight"].data is not None
    assert part_model.tensors["bias"].data is not None


def test_split_planner_rejects_oversized_single_partition() -> None:
    model_ir = _make_chain_model_ir(op_count=3)
    with pytest.raises(ValueError):
        plan_contiguous_partitions_by_size(
            model_ir=model_ir,
            target_max_bytes=300,
            hard_max_bytes=400,
            size_estimator=lambda _part: 500,
        )
