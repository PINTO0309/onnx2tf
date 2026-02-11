import pytest

from onnx2tf.tflite_builder.ir import ModelIR, OperatorIR, TensorIR
from onnx2tf.tflite_builder.split_planner import (
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


def test_split_planner_rejects_oversized_single_partition() -> None:
    model_ir = _make_chain_model_ir(op_count=3)
    with pytest.raises(ValueError):
        plan_contiguous_partitions_by_size(
            model_ir=model_ir,
            target_max_bytes=300,
            hard_max_bytes=400,
            size_estimator=lambda _part: 500,
        )
