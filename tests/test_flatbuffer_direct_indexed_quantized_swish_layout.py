from __future__ import annotations

import copy

import numpy as np
import onnx2tf.tflite_builder.lower_from_onnx2tf as lowering_module
import onnx2tf.tflite_builder.passes.quantized_swish_layout as swish_module

from onnx2tf.tflite_builder.core.graph import ModelIRGraphIndex
from onnx2tf.tflite_builder.ir import ModelIR, OperatorIR, TensorIR
from onnx2tf.tflite_builder.passes.quantized_swish_layout import (
    rewrite_transpose_swish_qdq_nhwc_branches,
)


def _tensor(
    name: str,
    shape: list[int],
    *,
    dtype: str = "FLOAT32",
    data: np.ndarray | None = None,
) -> TensorIR:
    return TensorIR(
        name=name,
        dtype=dtype,
        shape=list(shape),
        shape_signature=list(shape),
        data=data,
        is_variable=data is None,
    )


def _add_swish_branch(
    model_ir: ModelIR,
    *,
    prefix: str,
    pre_output: str,
    spatial: int,
    quantized_output: bool,
    post_transpose: bool,
) -> str:
    nchw_shape = [1, 2, int(spatial), int(spatial)]
    nhwc_shape = [1, int(spatial), int(spatial), 2]
    names = {
        "dq_log": f"{prefix}_dq_log",
        "sig": f"{prefix}_sig",
        "sig_q": f"{prefix}_sig_q",
        "sig_dq": f"{prefix}_sig_dq",
        "dq_data": f"{prefix}_dq_data",
        "mul": f"{prefix}_mul",
    }
    for name in [
        names["dq_log"],
        names["sig"],
        names["sig_dq"],
        names["dq_data"],
        names["mul"],
    ]:
        model_ir.tensors[name] = _tensor(name, nchw_shape)
    model_ir.tensors[names["sig_q"]] = _tensor(
        names["sig_q"],
        nchw_shape,
        dtype="INT8",
    )
    model_ir.operators.extend(
        [
            OperatorIR("DEQUANTIZE", [pre_output], [names["dq_log"]]),
            OperatorIR("LOGISTIC", [names["dq_log"]], [names["sig"]]),
            OperatorIR("QUANTIZE", [names["sig"]], [names["sig_q"]]),
            OperatorIR("DEQUANTIZE", [names["sig_q"]], [names["sig_dq"]]),
            OperatorIR("DEQUANTIZE", [pre_output], [names["dq_data"]]),
            OperatorIR(
                "MUL",
                [names["dq_data"], names["sig_dq"]],
                [names["mul"]],
            ),
        ]
    )

    branch_output = names["mul"]
    branch_dtype = "FLOAT32"
    if quantized_output:
        branch_output = f"{prefix}_q"
        branch_dtype = "INT8"
        model_ir.tensors[branch_output] = _tensor(
            branch_output,
            nchw_shape,
            dtype=branch_dtype,
        )
        model_ir.operators.append(
            OperatorIR("QUANTIZE", [names["mul"]], [branch_output])
        )

    if post_transpose:
        post_output = f"{prefix}_post"
        tap_output = f"{prefix}_tap"
        model_ir.tensors[post_output] = _tensor(
            post_output,
            nhwc_shape,
            dtype=branch_dtype,
        )
        model_ir.tensors[tap_output] = _tensor(
            tap_output,
            nhwc_shape,
            dtype=branch_dtype,
        )
        model_ir.operators.extend(
            [
                OperatorIR(
                    "TRANSPOSE",
                    [branch_output, "perm_post"],
                    [post_output],
                ),
                OperatorIR("RELU", [post_output], [tap_output]),
            ]
        )
        model_ir.outputs.append(tap_output)

    return branch_output


def _make_shared_multibranch_model_ir() -> tuple[ModelIR, set[str]]:
    model_ir = ModelIR("indexed_quantized_swish_shared_multibranch")
    spatial = 160
    nhwc_shape = [1, spatial, spatial, 2]
    nchw_shape = [1, 2, spatial, spatial]
    model_ir.tensors["perm_pre"] = _tensor(
        "perm_pre",
        [4],
        dtype="INT32",
        data=np.asarray([0, 3, 1, 2], dtype=np.int32),
    )
    model_ir.tensors["perm_post"] = _tensor(
        "perm_post",
        [4],
        dtype="INT32",
        data=np.asarray([0, 2, 3, 1], dtype=np.int32),
    )
    for prefix in ["a", "b"]:
        source = f"{prefix}_source"
        pre_output = f"{prefix}_nchw"
        model_ir.inputs.append(source)
        model_ir.tensors[source] = _tensor(source, nhwc_shape, dtype="INT8")
        model_ir.tensors[pre_output] = _tensor(
            pre_output,
            nchw_shape,
            dtype="INT8",
        )
        model_ir.operators.append(
            OperatorIR("TRANSPOSE", [source, "perm_pre"], [pre_output])
        )

    a1_output = _add_swish_branch(
        model_ir,
        prefix="a1",
        pre_output="a_nchw",
        spatial=spatial,
        quantized_output=True,
        post_transpose=True,
    )
    _add_swish_branch(
        model_ir,
        prefix="a2",
        pre_output="a_nchw",
        spatial=spatial,
        quantized_output=False,
        post_transpose=True,
    )
    b_output = _add_swish_branch(
        model_ir,
        prefix="b",
        pre_output="b_nchw",
        spatial=spatial,
        quantized_output=True,
        post_transpose=False,
    )
    model_ir.tensors["sum"] = _tensor("sum", nchw_shape, dtype="INT8")
    model_ir.operators.append(OperatorIR("ADD", [a1_output, b_output], ["sum"]))
    model_ir.outputs.append("sum")

    rewritten = {
        f"{prefix}_{suffix}"
        for prefix, quantized in [("a1", True), ("a2", False), ("b", True)]
        for suffix in [
            "dq_log",
            "sig",
            "sig_q",
            "sig_dq",
            "dq_data",
            "mul",
            *(["q"] if quantized else []),
        ]
    }
    return model_ir, rewritten


def _make_concat_closure_model_ir() -> ModelIR:
    model_ir = ModelIR("indexed_quantized_swish_concat_closure")
    spatial = 80
    nhwc_shape = [1, spatial, spatial, 2]
    nchw_shape = [1, 2, spatial, spatial]
    model_ir.inputs = ["source"]
    model_ir.tensors["source"] = _tensor("source", nhwc_shape, dtype="INT8")
    model_ir.tensors["nchw"] = _tensor("nchw", nchw_shape, dtype="INT8")
    model_ir.tensors["perm_pre"] = _tensor(
        "perm_pre",
        [4],
        dtype="INT32",
        data=np.asarray([0, 3, 1, 2], dtype=np.int32),
    )
    model_ir.tensors["perm_post"] = _tensor(
        "perm_post",
        [4],
        dtype="INT32",
        data=np.asarray([0, 2, 3, 1], dtype=np.int32),
    )
    model_ir.operators = [OperatorIR("TRANSPOSE", ["source", "perm_pre"], ["nchw"])]
    branch_output = _add_swish_branch(
        model_ir,
        prefix="branch",
        pre_output="nchw",
        spatial=spatial,
        quantized_output=True,
        post_transpose=False,
    )
    model_ir.tensors["branch_tail_dq"] = _tensor(
        "branch_tail_dq",
        nchw_shape,
    )
    model_ir.tensors["peer"] = _tensor(
        "peer",
        nchw_shape,
        data=np.ones(nchw_shape, dtype=np.float32),
    )
    model_ir.tensors["concat"] = _tensor(
        "concat",
        [1, 4, spatial, spatial],
    )
    model_ir.tensors["concat_q"] = _tensor(
        "concat_q",
        [1, 4, spatial, spatial],
        dtype="INT8",
    )
    model_ir.tensors["concat_post"] = _tensor(
        "concat_post",
        [1, spatial, spatial, 4],
        dtype="INT8",
    )
    model_ir.tensors["y"] = _tensor(
        "y",
        [1, spatial, spatial, 4],
        dtype="INT8",
    )
    model_ir.operators.extend(
        [
            OperatorIR("DEQUANTIZE", [branch_output], ["branch_tail_dq"]),
            OperatorIR(
                "CONCATENATION",
                ["branch_tail_dq", "peer"],
                ["concat"],
                options={"axis": 1},
            ),
            OperatorIR("QUANTIZE", ["concat"], ["concat_q"]),
            OperatorIR(
                "TRANSPOSE",
                ["concat_q", "perm_post"],
                ["concat_post"],
            ),
            OperatorIR("RELU", ["concat_post"], ["y"]),
        ]
    )
    model_ir.outputs = ["y"]
    return model_ir


def _assert_index_matches_fresh(
    model_ir: ModelIR,
    graph_index: ModelIRGraphIndex,
) -> None:
    fresh = ModelIRGraphIndex(model_ir)
    assert graph_index.producers == fresh.producers
    assert graph_index.consumers == fresh.consumers
    assert graph_index.duplicate_producers == fresh.duplicate_producers
    assert graph_index._operator_indices_by_id == fresh._operator_indices_by_id
    assert graph_index._operator_indices_by_type == fresh._operator_indices_by_type


def test_indexed_swish_primary_rewrites_shared_multibranch_and_keeps_index(
    monkeypatch,
) -> None:
    model_ir, expected_rewritten = _make_shared_multibranch_model_ir()
    graph_index = ModelIRGraphIndex(model_ir)

    def unexpected_map_rebuild(*args, **kwargs):
        raise AssertionError("unexpected compatibility graph-map rebuild")

    monkeypatch.setattr(
        lowering_module,
        "_build_tensor_consumer_map",
        unexpected_map_rebuild,
    )
    monkeypatch.setattr(
        lowering_module,
        "_build_tensor_producer_map",
        unexpected_map_rebuild,
    )

    result = rewrite_transpose_swish_qdq_nhwc_branches(
        model_ir,
        graph_index=graph_index,
    )

    assert result.rewritten_branches == 3
    assert result.removed_pre_transposes == 2
    assert set(result.rewritten_tensors) == expected_rewritten
    assert not any(
        str(op.op_type) == "TRANSPOSE" and str(op.outputs[0]) in {"a_nchw", "b_nchw"}
        for op in model_ir.operators
    )
    dequantize_sources = {
        str(op.outputs[0]): str(op.inputs[0])
        for op in model_ir.operators
        if str(op.op_type) == "DEQUANTIZE"
        and str(op.outputs[0]).endswith(("_dq_log", "_dq_data"))
    }
    assert dequantize_sources == {
        "a1_dq_log": "a_source",
        "a1_dq_data": "a_source",
        "a2_dq_log": "a_source",
        "a2_dq_data": "a_source",
        "b_dq_log": "b_source",
        "b_dq_data": "b_source",
    }
    assert all(
        list(model_ir.tensors[name].shape) == [1, 160, 160, 2]
        for name in expected_rewritten
    )
    _assert_index_matches_fresh(model_ir, graph_index)


def test_indexed_swish_primary_constructs_one_index(monkeypatch) -> None:
    model_ir, _ = _make_shared_multibranch_model_ir()
    refresh_count = 0
    original_refresh = ModelIRGraphIndex.refresh

    def counted_refresh(graph_index: ModelIRGraphIndex) -> None:
        nonlocal refresh_count
        refresh_count += 1
        original_refresh(graph_index)

    monkeypatch.setattr(ModelIRGraphIndex, "refresh", counted_refresh)

    result = rewrite_transpose_swish_qdq_nhwc_branches(model_ir)

    assert result.rewritten_branches == 3
    assert refresh_count == 1


def test_indexed_swish_primary_preserves_public_and_fanout_guards() -> None:
    public_model, _ = _make_shared_multibranch_model_ir()
    public_model.outputs.append("a1_sig")
    public_result = rewrite_transpose_swish_qdq_nhwc_branches(public_model)
    assert public_result.rewritten_branches == 2
    assert "a1_sig" not in public_result.rewritten_tensors
    assert any(
        str(op.op_type) == "TRANSPOSE" and str(op.outputs[0]) == "a_nchw"
        for op in public_model.operators
    )

    fanout_model, _ = _make_shared_multibranch_model_ir()
    fanout_model.tensors["a1_data_tap"] = _tensor(
        "a1_data_tap",
        [1, 2, 160, 160],
    )
    fanout_model.operators.append(OperatorIR("RELU", ["a1_dq_data"], ["a1_data_tap"]))
    fanout_model.outputs.append("a1_data_tap")
    fanout_result = rewrite_transpose_swish_qdq_nhwc_branches(fanout_model)
    assert fanout_result.rewritten_branches == 2
    assert "a1_dq_data" not in fanout_result.rewritten_tensors

    post_output_model, _ = _make_shared_multibranch_model_ir()
    post_output_model.outputs.append("a1_post")
    post_output_result = rewrite_transpose_swish_qdq_nhwc_branches(post_output_model)
    assert post_output_result.rewritten_branches == 2
    assert "a1_q" not in post_output_result.rewritten_tensors


def test_indexed_swish_primary_requires_explicit_small_spatial_concat_closure() -> None:
    model_ir = _make_concat_closure_model_ir()

    spatial_guard_result = rewrite_transpose_swish_qdq_nhwc_branches(
        copy.deepcopy(model_ir),
    )
    assert spatial_guard_result.rewritten_branches == 0

    unsafe_tail_result = rewrite_transpose_swish_qdq_nhwc_branches(
        copy.deepcopy(model_ir),
        min_spatial_stage=0,
    )
    assert unsafe_tail_result.rewritten_branches == 0

    closure_model = copy.deepcopy(model_ir)
    closure_result = rewrite_transpose_swish_qdq_nhwc_branches(
        closure_model,
        min_spatial_stage=0,
        require_concat_closure=True,
    )
    assert closure_result.rewritten_branches == 1
    assert closure_result.removed_pre_transposes == 1
    assert list(closure_model.tensors["branch_q"].shape) == [1, 80, 80, 2]

    wrong_axis_model = copy.deepcopy(model_ir)
    concat = next(
        op for op in wrong_axis_model.operators if str(op.op_type) == "CONCATENATION"
    )
    concat.options["axis"] = 3
    wrong_axis_result = rewrite_transpose_swish_qdq_nhwc_branches(
        wrong_axis_model,
        min_spatial_stage=0,
        require_concat_closure=True,
    )
    assert wrong_axis_result.rewritten_branches == 0


def test_indexed_swish_primary_skips_index_without_transpose(monkeypatch) -> None:
    model_ir = ModelIR("indexed_quantized_swish_no_transpose")
    model_ir.inputs = ["x"]
    model_ir.outputs = ["y"]
    model_ir.tensors["x"] = _tensor("x", [1, 4])
    model_ir.tensors["y"] = _tensor("y", [1, 4])
    model_ir.operators = [OperatorIR("RELU", ["x"], ["y"])]

    def unexpected_index(*args, **kwargs):
        raise AssertionError("unexpected graph index allocation")

    monkeypatch.setattr(swish_module, "ModelIRGraphIndex", unexpected_index)

    result = rewrite_transpose_swish_qdq_nhwc_branches(model_ir)

    assert result.rewritten_branches == 0
    assert result.removed_pre_transposes == 0
    assert result.rewritten_tensors == frozenset()
