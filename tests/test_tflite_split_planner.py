import random

import numpy as np
import pytest

import onnx2tf.tflite_builder.split_planner as split_planner_module
from onnx2tf.tflite_builder.ir import ModelIR, OperatorIR, TensorIR
from onnx2tf.tflite_builder.split_planner import (
    _collect_inputs,
    _collect_outputs,
    build_partition_model_ir,
    find_dependency_safe_split_points,
    plan_contiguous_partitions_by_size,
    validate_partition_ranges,
)


def test_partition_tensor_collection_preserves_first_seen_order() -> None:
    operators = [
        OperatorIR(
            op_type="DUMMY",
            inputs=["x", "shared", ""],
            outputs=["a", "shared_out"],
        ),
        OperatorIR(
            op_type="DUMMY",
            inputs=["shared", "a", "x"],
            outputs=["shared_out", "b", "a"],
        ),
    ]

    assert _collect_inputs(operators) == ["x", "shared", "a"]
    assert _collect_outputs(operators) == ["a", "shared_out", "b"]


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


def test_split_planner_reuses_one_model_ir_graph_index(monkeypatch) -> None:
    index_build_count = 0
    original_graph_index = split_planner_module.ModelIRGraphIndex

    class _CountingGraphIndex(original_graph_index):
        def __post_init__(self) -> None:
            nonlocal index_build_count
            index_build_count += 1
            super().__post_init__()

    monkeypatch.setattr(
        split_planner_module,
        "ModelIRGraphIndex",
        _CountingGraphIndex,
    )

    report = plan_contiguous_partitions_by_size(
        model_ir=_make_chain_model_ir(op_count=6),
        target_max_bytes=250,
        hard_max_bytes=350,
        size_estimator=lambda part: len(part.operators) * 100,
    )

    assert report["plan_valid"] is True
    assert index_build_count == 1


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
    assert [point["crossing_tensors"] for point in points] == [
        ["t0"],
        ["t1"],
        ["t2"],
        ["t3"],
        ["t4"],
    ]


def _reference_dependency_safe_split_points(
    model_ir: ModelIR,
) -> list[dict[str, object]]:
    op_count = len(model_ir.operators)
    if op_count <= 1:
        return []
    producer_index: dict[str, int] = {}
    for op_idx, op in enumerate(model_ir.operators):
        for output_name in op.outputs:
            if output_name and output_name not in producer_index:
                producer_index[output_name] = op_idx

    points: list[dict[str, object]] = []
    for boundary in range(1, op_count):
        valid = True
        crossing_tensors: set[str] = set()
        for op_idx, op in enumerate(model_ir.operators):
            for input_name in op.inputs:
                if not input_name:
                    continue
                producer = producer_index.get(input_name)
                if producer is None:
                    continue
                if op_idx < boundary and producer >= boundary:
                    valid = False
                    break
                if op_idx >= boundary and producer < boundary:
                    crossing_tensors.add(input_name)
            if not valid:
                break
        if valid:
            points.append(
                {
                    "index": boundary,
                    "crossing_count": len(crossing_tensors),
                    "crossing_tensors": sorted(crossing_tensors),
                }
            )
    return points


def test_dependency_safe_split_points_matches_legacy_randomized_graphs() -> None:
    rng = random.Random(20260714)
    for case_index in range(200):
        op_count = rng.randint(0, 40)
        tensor_pool = [f"t{idx}" for idx in range(max(1, op_count // 2))]
        model_ir = ModelIR(name=f"random_split_graph_{case_index}")
        for op_idx in range(op_count):
            inputs = [
                rng.choice(tensor_pool + ["external", ""])
                for _ in range(rng.randint(0, 4))
            ]
            output_name = rng.choice(tensor_pool)
            model_ir.operators.append(
                OperatorIR(
                    op_type="DUMMY",
                    inputs=inputs,
                    outputs=[output_name],
                )
            )

        assert find_dependency_safe_split_points(
            model_ir
        ) == _reference_dependency_safe_split_points(model_ir)


def test_dependency_safe_split_points_reads_each_operator_edge_list_once() -> None:
    access_counts = {"inputs": 0, "outputs": 0}

    class _CountingOperator:
        def __init__(self, index: int) -> None:
            self.index = int(index)
            self.op_type = "DUMMY"

        @property
        def inputs(self):
            access_counts["inputs"] += 1
            return ["x" if self.index == 0 else f"t{self.index - 1}"]

        @property
        def outputs(self):
            access_counts["outputs"] += 1
            return [f"t{self.index}"]

    op_count = 256
    model_ir = ModelIR(name="linear_split_scan")
    model_ir.operators = [_CountingOperator(index) for index in range(op_count)]

    points = find_dependency_safe_split_points(model_ir)

    assert len(points) == op_count - 1
    assert access_counts == {"inputs": op_count, "outputs": op_count}


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
                options={"axis": 1},
                axis_semantics={"axis": "channel"},
                version=2,
                onnx_node_name="first_node",
                onnx_op_type="Identity",
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
    first_op = part_model.operators[0]
    assert first_op.options == {"axis": 1}
    assert first_op.axis_semantics == {"axis": "channel"}
    assert first_op.version == 2
    assert first_op.onnx_node_name == "first_node"
    assert first_op.onnx_op_type == "Identity"


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
