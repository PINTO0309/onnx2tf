import inspect
import random

import numpy as np
import pytest

import onnx2tf.tflite_builder.split_planner as split_planner_module
from onnx2tf.tflite_builder.core.graph import ModelIRGraphIndex
from onnx2tf.tflite_builder.ir import ModelIR, OperatorIR, QuantParamIR, TensorIR
from onnx2tf.tflite_builder.split_planner import (
    _collect_inputs,
    _collect_outputs,
    build_partition_model_ir,
    crop_model_ir_by_boundary_tensors,
    find_dependency_safe_split_points,
    plan_contiguous_partitions_by_size,
    validate_partition_ranges,
    write_split_model_files_and_manifest,
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


def test_graph_index_answers_consumer_suffix_queries_from_sorted_tail() -> None:
    model_ir = _make_chain_model_ir(op_count=4)
    model_ir.operators[3].inputs.append("t0")
    graph_index = ModelIRGraphIndex(model_ir)

    assert graph_index.has_consumer_at_or_after("t0", 3) is True
    assert graph_index.has_consumer_at_or_after("t0", 4) is False
    assert graph_index.has_consumer_at_or_after("missing", 0) is False


def test_crop_model_ir_uses_intermediate_boundaries_without_extra_operators() -> None:
    model_ir = _make_chain_model_ir(op_count=5)

    cropped = crop_model_ir_by_boundary_tensors(
        model_ir=model_ir,
        requested_inputs=["t1"],
        requested_outputs=["t3"],
    )

    assert cropped.inputs == ["t1"]
    assert cropped.outputs == ["t3"]
    assert [op.outputs for op in cropped.operators] == [["t2"], ["t3"]]
    assert set(cropped.tensors) == {"t1", "t2", "t3"}


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


def test_split_planner_materializes_first_producer_map_once(monkeypatch) -> None:
    producer_map_build_count = 0
    original_first_producer_indices = split_planner_module._first_producer_indices

    def counted_first_producer_indices(graph_index):
        nonlocal producer_map_build_count
        producer_map_build_count += 1
        return original_first_producer_indices(graph_index)

    monkeypatch.setattr(
        split_planner_module,
        "_first_producer_indices",
        counted_first_producer_indices,
    )

    report = plan_contiguous_partitions_by_size(
        model_ir=_make_chain_model_ir(op_count=16),
        target_max_bytes=400,
        hard_max_bytes=500,
        size_estimator=lambda part: len(part.operators) * 100,
    )

    assert report["plan_valid"] is True
    assert producer_map_build_count == 1


def test_split_size_candidates_borrow_constant_buffers_read_only() -> None:
    constant_data = np.arange(1024, dtype=np.float32)
    model_ir = ModelIR(name="borrowed_split_constants")
    model_ir.inputs = ["x"]
    model_ir.outputs = ["y"]
    model_ir.tensors = {
        "x": TensorIR(name="x", dtype="FLOAT32", shape=[1024]),
        "constant": TensorIR(
            name="constant",
            dtype="FLOAT32",
            shape=[1024],
            data=constant_data,
        ),
        "y": TensorIR(name="y", dtype="FLOAT32", shape=[1024]),
    }
    model_ir.operators = [
        OperatorIR(
            op_type="ADD",
            inputs=["x", "constant"],
            outputs=["y"],
        )
    ]
    borrowed_flags = []

    plan_contiguous_partitions_by_size(
        model_ir=model_ir,
        target_max_bytes=100,
        hard_max_bytes=200,
        size_estimator=lambda part: (
            borrowed_flags.append(part.tensors["constant"].data is constant_data)
            or len(part.operators)
        ),
    )

    assert borrowed_flags
    assert all(borrowed_flags)
    np.testing.assert_array_equal(model_ir.tensors["constant"].data, constant_data)


def test_split_writer_reuses_one_model_ir_graph_index(monkeypatch, tmp_path) -> None:
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
    monkeypatch.setattr(
        split_planner_module,
        "run_model_ir_validation_pipeline",
        lambda _model_ir: None,
    )
    monkeypatch.setattr(
        split_planner_module,
        "write_model_file",
        lambda **kwargs: None,
    )

    result = write_split_model_files_and_manifest(
        schema_tflite={},
        model_ir=_make_chain_model_ir(op_count=4),
        plan_report={
            "target_max_bytes": 250,
            "hard_max_bytes": 350,
            "total_estimated_bytes": 400,
            "partitions": [
                {
                    "partition_id": 1,
                    "start_op_index": 0,
                    "end_op_index": 2,
                    "estimated_bytes": 200,
                },
                {
                    "partition_id": 2,
                    "start_op_index": 2,
                    "end_op_index": 4,
                    "estimated_bytes": 200,
                },
            ],
            "edges": [
                {
                    "from_partition": 0,
                    "to_partition": 1,
                    "tensors": ["t1"],
                }
            ],
        },
        output_folder_path=str(tmp_path),
        output_file_name="shared_index",
    )

    assert index_build_count == 1
    assert result["split_partition_count"] == 2
    assert (tmp_path / "shared_index_split_manifest.json").is_file()


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


def test_build_partition_reuses_boundary_scan_when_all_ops_are_required(
    monkeypatch,
) -> None:
    call_counts = {"inputs": 0, "outputs": 0}
    original_collect_inputs = split_planner_module._collect_inputs
    original_collect_outputs = split_planner_module._collect_outputs

    def counted_inputs(operators):
        call_counts["inputs"] += 1
        return original_collect_inputs(operators)

    def counted_outputs(operators):
        call_counts["outputs"] += 1
        return original_collect_outputs(operators)

    monkeypatch.setattr(split_planner_module, "_collect_inputs", counted_inputs)
    monkeypatch.setattr(split_planner_module, "_collect_outputs", counted_outputs)

    part_model = build_partition_model_ir(
        model_ir=_make_chain_model_ir(op_count=16),
        start_op_index=0,
        end_op_index=16,
        partition_id=1,
    )

    assert len(part_model.operators) == 16
    assert call_counts == {"inputs": 1, "outputs": 1}


def test_partition_builder_uses_shared_operator_clone_contract() -> None:
    source = inspect.getsource(split_planner_module.build_partition_model_ir)

    assert "clone_operator_ir(op, options=dict(op.options))" in source
    assert "clone_tensor_ir(" in source
    assert "copy_data=copy_tensor_data" in source
    assert "clone_quantization=False" in source
    assert "onnx_node_name=op.onnx_node_name" not in source


def test_partition_tensor_clone_preserves_buffer_and_quantization_policies() -> None:
    model_ir = _make_chain_model_ir(op_count=1)
    data = np.asarray([1.0], dtype=np.float32)
    quantization = QuantParamIR(scale=[0.5], zero_point=[0])
    model_ir.tensors["t0"].data = data
    model_ir.tensors["t0"].quantization = quantization

    borrowed = build_partition_model_ir(
        model_ir=model_ir,
        start_op_index=0,
        end_op_index=1,
        partition_id=1,
        copy_tensor_data=False,
    )
    copied = build_partition_model_ir(
        model_ir=model_ir,
        start_op_index=0,
        end_op_index=1,
        partition_id=1,
        copy_tensor_data=True,
    )

    assert borrowed.tensors["t0"].data is data
    assert copied.tensors["t0"].data is not data
    assert borrowed.tensors["t0"].quantization is quantization
    assert copied.tensors["t0"].quantization is quantization


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
    assert part_model.tensors["weight"].data is not model_ir.tensors["weight"].data
    assert part_model.tensors["bias"].data is not model_ir.tensors["bias"].data


def test_split_planner_rejects_oversized_single_partition() -> None:
    model_ir = _make_chain_model_ir(op_count=3)
    with pytest.raises(ValueError):
        plan_contiguous_partitions_by_size(
            model_ir=model_ir,
            target_max_bytes=300,
            hard_max_bytes=400,
            size_estimator=lambda _part: 500,
        )
