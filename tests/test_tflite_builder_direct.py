import glob
import json
import math
import os
import shutil
import tempfile
from contextlib import contextmanager

import numpy as np
import onnx
import onnx2tf
import pytest
from onnx import TensorProto, helper, numpy_helper
from onnx2tf.tflite_builder.ir import ModelIR, OperatorIR, TensorIR
from onnx2tf.tflite_builder.lower_from_onnx2tf import (
    _dtype_from_onnx_elem_type,
    _optimize_asin_transpose_passthrough_chains,
    _optimize_erf_transpose_passthrough_chains,
    _optimize_boundary_input_transpose_channel_slice_blocks,
    _optimize_center_size_offset_terminal_transpose_chains,
    _optimize_duplicate_transpose_fanout,
    _optimize_fuse_conv_activation_chains,
    _optimize_fold_conv_mul_add_affine_chains,
    _optimize_hardsigmoid_mul_transpose_passthrough_chains,
    _optimize_layout_transpose_chains,
    _optimize_maximum_with_zero_input2_to_relu,
    _optimize_maximum_minimum_relu0to1_chains,
    _optimize_singleton_channel_layout_transpose_to_reshape,
    _optimize_consecutive_inverse_singleton_layout_reshapes,
    _optimize_consecutive_reshape_passthrough_chains,
    _optimize_flatten_concat_expanddims_to_nhwc_concat,
    _optimize_singleton_layout_reshape_unary_passthrough_chains,
    _optimize_squeeze_reshape_identity_chains,
    _optimize_singleton_spatial_nhwc_transpose_reshape_flatten,
    _optimize_singleton_reshape_concat_post_transpose_nhwc_chains,
    _optimize_transpose_csp_attention_nhwc_chains,
    _optimize_transpose_conv_attention_nhwc_propagation_chains,
    _optimize_transpose_conv3d_leaky_mul_unsqueeze_ndhwc_chains,
    _optimize_transpose_cost_volume_scatter_ndhwc_chains,
    _optimize_transpose_3d_leaky_logistic_muladd_ndhwc_chains,
    _optimize_transpose_axis3_const_concat_bridge_nhwc_chains,
    _optimize_transpose_elementwise_concat_conv_nhwc_groups,
    _optimize_transpose_elementwise_roundtrip_nhwc_nchw_fanout_chains,
    _optimize_convpool_output_transpose_nhwc_passthrough_chains,
    _optimize_transpose_slice_logistic_concat_reshape_tail_nhwc_chains,
    _optimize_shufflenet_reshape_transpose_shuffle_nhwc_chains,
    _optimize_shufflenet_transpose_shuffle_chains,
    _optimize_sinet_concat_resize_affine_transpose_chains,
    _optimize_sinet_dual_resize_affine_transpose_chains,
    _optimize_transposeconv_output_nhwc_passthrough_chains,
    _optimize_batchmatmul_affine_transpose_input_chains,
    _optimize_batchmatmul_reshape_se_nhwc_chains,
    _optimize_transpose_mean_prepost_nhwc_passthrough_chains,
    _optimize_transpose_layernorm_stats_nhwc_propagation_chains,
    _optimize_layernorm_stats_via_existing_post_transpose_nhwc_chains,
    _optimize_transpose_se_fc_mul_prepost_nhwc_chains,
    _optimize_transpose_se_conv_mul_prepost_nhwc_chains,
    _optimize_transpose_input_chains_pre_concat_to_single_post_adapter,
    _optimize_transpose_logistic_muladd_prepost_nhwc_chains,
    _optimize_transpose_nested_weighted_add_swish_prepost_nhwc_chains,
    _optimize_transpose_pad_prepost_nhwc_chains,
    _optimize_transpose_weighted_add_swish_prepost_nhwc_chains,
    _optimize_transpose_pre_add_nhwc_chains,
    _optimize_transpose_pre_add_mul_add_prelu_nhwc_chains,
    _optimize_transpose_pre_unary_reshape_transpose_suffix_nhwc_chains,
    _optimize_transpose_reshape_transpose_to_expanddims_nhwc_chains,
    _optimize_transpose_pre_concat_ndhwc_chains,
    _optimize_transpose_pre_concat_nhwc_chains,
    _optimize_transpose_pre_unary_mean_terminal_nhwc_chains,
    _optimize_transpose_mul_add_const_prepost_nhwc_chains,
    _optimize_transpose_mul_add_const_prelu_prepost_nhwc_terminal_chains,
    _topologically_sort_operators,
    _reconcile_static_tensor_shapes,
    _resolve_dynamic_reshape_shapes,
    lower_onnx_to_ir,
)
from onnx2tf.tflite_builder.model_writer import serialize_model
from onnx2tf.tflite_builder.preprocess import (
    register_default_preprocess_rules,
    run_preprocess_pipeline,
)
from onnx2tf.tflite_builder.schema_loader import load_schema_module

Interpreter = pytest.importorskip("ai_edge_litert.interpreter").Interpreter
pytest.importorskip("onnxruntime")


def _save_model(tmpdir: str, name: str, model: onnx.ModelProto) -> str:
    model_path = os.path.join(tmpdir, f"{name}.onnx")
    onnx.save(model, model_path)
    return model_path


def _convert(
    model_path: str,
    output_dir: str,
    backend: str,
    output_dynamic_range_quantized_tflite: bool = False,
    output_integer_quantized_tflite: bool = False,
    quant_type: str = "per-channel",
    input_quant_dtype: str = "int8",
    output_quant_dtype: str = "int8",
    eval_with_onnx: bool = False,
    eval_num_samples: int = 10,
    eval_rtol: float = 0.0,
    eval_atol: float = 1e-4,
    eval_fail_on_threshold: bool = False,
    eval_target_tflite: str = "float32",
    eval_compare_mode: str = "auto",
    eval_split_models: bool = False,
    eval_split_reference: str = "unsplit_tflite",
    eval_split_fail_on_threshold: bool = False,
    auto_split_tflite_by_size: bool = False,
    report_op_coverage: bool = False,
    flatbuffer_direct_fallback_to_tf_converter: bool = False,
    flatbuffer_direct_allow_custom_ops: bool = False,
    flatbuffer_direct_custom_op_allowlist: list[str] | None = None,
    auto_split_max_size: str | int | None = None,
    tflite_split_max_bytes: int = 1073741824,
    tflite_split_target_bytes: int = 1060000000,
) -> str:
    onnx2tf.convert(
        input_onnx_file_path=model_path,
        output_folder_path=output_dir,
        disable_strict_mode=True,
        verbosity="error",
        tflite_backend=backend,
        output_dynamic_range_quantized_tflite=output_dynamic_range_quantized_tflite,
        output_integer_quantized_tflite=output_integer_quantized_tflite,
        quant_type=quant_type,
        input_quant_dtype=input_quant_dtype,
        output_quant_dtype=output_quant_dtype,
        eval_with_onnx=eval_with_onnx,
        eval_num_samples=eval_num_samples,
        eval_rtol=eval_rtol,
        eval_atol=eval_atol,
        eval_fail_on_threshold=eval_fail_on_threshold,
        eval_target_tflite=eval_target_tflite,
        eval_compare_mode=eval_compare_mode,
        eval_split_models=eval_split_models,
        eval_split_reference=eval_split_reference,
        eval_split_fail_on_threshold=eval_split_fail_on_threshold,
        auto_split_tflite_by_size=auto_split_tflite_by_size,
        report_op_coverage=report_op_coverage,
        flatbuffer_direct_fallback_to_tf_converter=flatbuffer_direct_fallback_to_tf_converter,
        flatbuffer_direct_allow_custom_ops=flatbuffer_direct_allow_custom_ops,
        flatbuffer_direct_custom_op_allowlist=flatbuffer_direct_custom_op_allowlist,
        auto_split_max_size=auto_split_max_size,
        tflite_split_max_bytes=tflite_split_max_bytes,
        tflite_split_target_bytes=tflite_split_target_bytes,
    )
    model_name = os.path.splitext(os.path.basename(model_path))[0]
    return os.path.join(output_dir, f"{model_name}_float32.tflite")


def _run_add_inference(tflite_path: str) -> np.ndarray:
    interpreter = Interpreter(model_path=tflite_path)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    x = np.array([[1.0, 2.0, 3.0]], dtype=np.float32)
    y = np.array([[4.0, 5.0, 6.0]], dtype=np.float32)
    by_name = {detail["name"]: detail for detail in input_details}
    if "x" in by_name and "y" in by_name:
        interpreter.set_tensor(by_name["x"]["index"], x)
        interpreter.set_tensor(by_name["y"]["index"], y)
    else:
        interpreter.set_tensor(input_details[0]["index"], x)
        interpreter.set_tensor(input_details[1]["index"], y)
    interpreter.invoke()
    return interpreter.get_tensor(output_details[0]["index"])


def _make_add_model() -> onnx.ModelProto:
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 3])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 3])
    z = helper.make_tensor_value_info("z", TensorProto.FLOAT, [1, 3])
    node = helper.make_node("Add", ["x", "y"], ["z"], name="AddNode")
    graph = helper.make_graph([node], "add_graph", [x, y], [z])
    return helper.make_model(graph, opset_imports=[helper.make_operatorsetid("", 13)])


def test_flatbuffer_direct_topological_sort_reorders_producer_before_consumer() -> None:
    model_ir = ModelIR(name="topological_sort_test")
    model_ir.inputs = ["x"]
    model_ir.outputs = ["y"]
    model_ir.tensors["x"] = TensorIR(
        name="x",
        dtype="FLOAT32",
        shape=[1, 1, 2, 3],
        shape_signature=[1, 1, 2, 3],
    )
    model_ir.tensors["perm"] = TensorIR(
        name="perm",
        dtype="INT32",
        shape=[4],
        shape_signature=[4],
        data=np.asarray([0, 2, 3, 1], dtype=np.int32),
        is_variable=False,
    )
    model_ir.tensors["x_nhwc"] = TensorIR(
        name="x_nhwc",
        dtype="FLOAT32",
        shape=[1, 2, 3, 1],
        shape_signature=[1, 2, 3, 1],
    )
    model_ir.tensors["bias"] = TensorIR(
        name="bias",
        dtype="FLOAT32",
        shape=[1, 2, 3, 1],
        shape_signature=[1, 2, 3, 1],
        data=np.zeros((1, 2, 3, 1), dtype=np.float32),
        is_variable=False,
    )
    model_ir.tensors["y"] = TensorIR(
        name="y",
        dtype="FLOAT32",
        shape=[1, 2, 3, 1],
        shape_signature=[1, 2, 3, 1],
    )
    model_ir.operators = [
        OperatorIR(
            op_type="ADD",
            inputs=["x_nhwc", "bias"],
            outputs=["y"],
            options={},
        ),
        OperatorIR(
            op_type="TRANSPOSE",
            inputs=["x", "perm"],
            outputs=["x_nhwc"],
            options={},
        ),
    ]

    stats = _topologically_sort_operators(model_ir)
    assert stats["cycle_detected"] == 0
    assert stats["reordered_operators"] > 0
    assert [str(op.op_type) for op in model_ir.operators] == ["TRANSPOSE", "ADD"]


def test_flatbuffer_direct_transpose_layernorm_stats_nhwc_propagation() -> None:
    model_ir = ModelIR(name="transpose_layernorm_stats_nhwc_propagation_test")
    model_ir.inputs = ["x_nhwc"]
    model_ir.outputs = ["y"]

    model_ir.tensors["x_nhwc"] = TensorIR(
        name="x_nhwc",
        dtype="FLOAT32",
        shape=[1, 1, 5, 3],
        shape_signature=[1, 1, 5, 3],
    )
    model_ir.tensors["perm_nhwc_to_nchw"] = TensorIR(
        name="perm_nhwc_to_nchw",
        dtype="INT32",
        shape=[4],
        shape_signature=[4],
        data=np.asarray([0, 3, 1, 2], dtype=np.int32),
        is_variable=False,
    )
    model_ir.tensors["x_nchw"] = TensorIR(
        name="x_nchw",
        dtype="FLOAT32",
        shape=[1, 3, 1, 5],
        shape_signature=[1, 3, 1, 5],
    )
    model_ir.tensors["mean_axes"] = TensorIR(
        name="mean_axes",
        dtype="INT32",
        shape=[1],
        shape_signature=[1],
        data=np.asarray([1], dtype=np.int32),
        is_variable=False,
    )
    model_ir.tensors["var_axes"] = TensorIR(
        name="var_axes",
        dtype="INT32",
        shape=[1],
        shape_signature=[1],
        data=np.asarray([1], dtype=np.int32),
        is_variable=False,
    )
    model_ir.tensors["m"] = TensorIR(
        name="m",
        dtype="FLOAT32",
        shape=[1, 1, 1, 5],
        shape_signature=[1, 1, 1, 5],
    )
    model_ir.tensors["n"] = TensorIR(
        name="n",
        dtype="FLOAT32",
        shape=[1, 1, 1, 5],
        shape_signature=[1, 1, 1, 5],
    )
    model_ir.tensors["d"] = TensorIR(
        name="d",
        dtype="FLOAT32",
        shape=[1, 3, 1, 5],
        shape_signature=[1, 3, 1, 5],
    )
    model_ir.tensors["sq"] = TensorIR(
        name="sq",
        dtype="FLOAT32",
        shape=[1, 3, 1, 5],
        shape_signature=[1, 3, 1, 5],
    )
    model_ir.tensors["v"] = TensorIR(
        name="v",
        dtype="FLOAT32",
        shape=[1, 1, 1, 5],
        shape_signature=[1, 1, 1, 5],
    )
    model_ir.tensors["eps"] = TensorIR(
        name="eps",
        dtype="FLOAT32",
        shape=[],
        shape_signature=[],
        data=np.asarray(1e-6, dtype=np.float32),
        is_variable=False,
    )
    model_ir.tensors["y"] = TensorIR(
        name="y",
        dtype="FLOAT32",
        shape=[1, 1, 1, 5],
        shape_signature=[1, 1, 1, 5],
    )
    model_ir.tensors["keep_nchw_bias"] = TensorIR(
        name="keep_nchw_bias",
        dtype="FLOAT32",
        shape=[1, 3, 1, 5],
        shape_signature=[1, 3, 1, 5],
        data=np.zeros((1, 3, 1, 5), dtype=np.float32),
        is_variable=False,
    )
    model_ir.tensors["keep_nchw"] = TensorIR(
        name="keep_nchw",
        dtype="FLOAT32",
        shape=[1, 3, 1, 5],
        shape_signature=[1, 3, 1, 5],
    )

    model_ir.operators = [
        OperatorIR(op_type="TRANSPOSE", inputs=["x_nhwc", "perm_nhwc_to_nchw"], outputs=["x_nchw"]),
        OperatorIR(
            op_type="MEAN",
            inputs=["x_nchw", "mean_axes"],
            outputs=["m"],
            options={"keepDims": True},
        ),
        OperatorIR(op_type="NEG", inputs=["m"], outputs=["n"]),
        OperatorIR(op_type="SUB", inputs=["x_nchw", "m"], outputs=["d"]),
        OperatorIR(op_type="MUL", inputs=["d", "d"], outputs=["sq"]),
        OperatorIR(
            op_type="MEAN",
            inputs=["sq", "var_axes"],
            outputs=["v"],
            options={"keepDims": True},
        ),
        OperatorIR(op_type="ADD", inputs=["v", "eps"], outputs=["y"]),
        # Keep one extra NCHW consumer so transpose must remain.
        OperatorIR(op_type="ADD", inputs=["x_nchw", "keep_nchw_bias"], outputs=["keep_nchw"]),
    ]

    stats = _optimize_transpose_layernorm_stats_nhwc_propagation_chains(model_ir)
    assert stats["optimized_transpose_layernorm_stats_nhwc_propagation_chains"] == 1

    mean0 = model_ir.operators[1]
    sub = model_ir.operators[3]
    mean1 = model_ir.operators[5]
    assert list(mean0.inputs) == ["x_nhwc", "mean_axes"]
    assert list(sub.inputs) == ["x_nhwc", "m"]
    assert list(mean1.inputs) == ["sq", "var_axes"]
    assert np.array_equal(np.asarray(model_ir.tensors["mean_axes"].data), np.asarray([3], dtype=np.int32))
    assert np.array_equal(np.asarray(model_ir.tensors["var_axes"].data), np.asarray([3], dtype=np.int32))

    # The transpose remains because it still feeds keep_nchw.
    assert [str(op.op_type) for op in model_ir.operators].count("TRANSPOSE") == 1
    assert list(model_ir.tensors["m"].shape) == [1, 1, 5, 1]
    assert list(model_ir.tensors["d"].shape) == [1, 1, 5, 3]
    assert list(model_ir.tensors["sq"].shape) == [1, 1, 5, 3]
    assert list(model_ir.tensors["v"].shape) == [1, 1, 5, 1]


def test_flatbuffer_direct_layernorm_stats_via_existing_post_transpose_nhwc() -> None:
    model_ir = ModelIR(name="layernorm_stats_via_post_transpose_nhwc_test")
    model_ir.inputs = ["x_nchw"]
    model_ir.outputs = ["y"]

    model_ir.tensors["x_nchw"] = TensorIR(
        name="x_nchw",
        dtype="FLOAT32",
        shape=[1, 3, 1, 5],
        shape_signature=[1, 3, 1, 5],
    )
    model_ir.tensors["perm_nchw_to_nhwc"] = TensorIR(
        name="perm_nchw_to_nhwc",
        dtype="INT32",
        shape=[4],
        shape_signature=[4],
        data=np.asarray([0, 2, 3, 1], dtype=np.int32),
        is_variable=False,
    )
    model_ir.tensors["x_nhwc"] = TensorIR(
        name="x_nhwc",
        dtype="FLOAT32",
        shape=[1, 1, 5, 3],
        shape_signature=[1, 1, 5, 3],
    )
    model_ir.tensors["mean_axes"] = TensorIR(
        name="mean_axes",
        dtype="INT32",
        shape=[1],
        shape_signature=[1],
        data=np.asarray([1], dtype=np.int32),
        is_variable=False,
    )
    model_ir.tensors["var_axes"] = TensorIR(
        name="var_axes",
        dtype="INT32",
        shape=[1],
        shape_signature=[1],
        data=np.asarray([1], dtype=np.int32),
        is_variable=False,
    )
    model_ir.tensors["m"] = TensorIR(
        name="m",
        dtype="FLOAT32",
        shape=[1, 1, 1, 5],
        shape_signature=[1, 1, 1, 5],
    )
    model_ir.tensors["d"] = TensorIR(
        name="d",
        dtype="FLOAT32",
        shape=[1, 3, 1, 5],
        shape_signature=[1, 3, 1, 5],
    )
    model_ir.tensors["sq"] = TensorIR(
        name="sq",
        dtype="FLOAT32",
        shape=[1, 3, 1, 5],
        shape_signature=[1, 3, 1, 5],
    )
    model_ir.tensors["v"] = TensorIR(
        name="v",
        dtype="FLOAT32",
        shape=[1, 1, 1, 5],
        shape_signature=[1, 1, 1, 5],
    )
    model_ir.tensors["eps"] = TensorIR(
        name="eps",
        dtype="FLOAT32",
        shape=[],
        shape_signature=[],
        data=np.asarray(1e-6, dtype=np.float32),
        is_variable=False,
    )
    model_ir.tensors["y"] = TensorIR(
        name="y",
        dtype="FLOAT32",
        shape=[1, 1, 1, 5],
        shape_signature=[1, 1, 1, 5],
    )

    model_ir.operators = [
        OperatorIR(op_type="TRANSPOSE", inputs=["x_nchw", "perm_nchw_to_nhwc"], outputs=["x_nhwc"]),
        OperatorIR(
            op_type="MEAN",
            inputs=["x_nchw", "mean_axes"],
            outputs=["m"],
            options={"keepDims": True},
        ),
        OperatorIR(op_type="SUB", inputs=["x_nchw", "m"], outputs=["d"]),
        OperatorIR(op_type="MUL", inputs=["d", "d"], outputs=["sq"]),
        OperatorIR(
            op_type="MEAN",
            inputs=["sq", "var_axes"],
            outputs=["v"],
            options={"keepDims": True},
        ),
        OperatorIR(op_type="ADD", inputs=["v", "eps"], outputs=["y"]),
    ]

    stats = _optimize_layernorm_stats_via_existing_post_transpose_nhwc_chains(model_ir)
    assert stats["optimized_layernorm_stats_via_existing_post_transpose_nhwc_chains"] == 1

    assert list(model_ir.operators[1].inputs) == ["x_nhwc", "mean_axes"]
    assert list(model_ir.operators[2].inputs) == ["x_nhwc", "m"]
    assert list(model_ir.operators[4].inputs) == ["sq", "var_axes"]
    assert np.array_equal(np.asarray(model_ir.tensors["mean_axes"].data), np.asarray([3], dtype=np.int32))
    assert np.array_equal(np.asarray(model_ir.tensors["var_axes"].data), np.asarray([3], dtype=np.int32))
    assert list(model_ir.tensors["m"].shape) == [1, 1, 5, 1]
    assert list(model_ir.tensors["d"].shape) == [1, 1, 5, 3]
    assert list(model_ir.tensors["sq"].shape) == [1, 1, 5, 3]
    assert list(model_ir.tensors["v"].shape) == [1, 1, 5, 1]


def test_flatbuffer_direct_onnx_string_dtype_mapping() -> None:
    assert _dtype_from_onnx_elem_type(int(TensorProto.STRING)) == "STRING"


def _make_string_normalizer_model(
    *,
    case_change_action: str = "NONE",
    is_case_sensitive: bool = True,
    locale: str = "en_US",
    stopwords: list[str] | None = None,
    input_shape: list[int] | None = None,
    output_shape: list[int] | None = None,
) -> onnx.ModelProto:
    x = helper.make_tensor_value_info(
        "input",
        TensorProto.STRING,
        list(input_shape) if input_shape is not None else [1],
    )
    y = helper.make_tensor_value_info(
        "output",
        TensorProto.STRING,
        list(output_shape) if output_shape is not None else [1],
    )
    node_kwargs = {
        "case_change_action": str(case_change_action),
        "is_case_sensitive": int(bool(is_case_sensitive)),
        "locale": str(locale),
    }
    if stopwords is not None and len(stopwords) > 0:
        node_kwargs["stopwords"] = [str(v) for v in stopwords]
    node = helper.make_node(
        "StringNormalizer",
        ["input"],
        ["output"],
        name="StringNormalizerNode",
        **node_kwargs,
    )
    graph = helper.make_graph([node], "string_normalizer_graph", [x], [y])
    return helper.make_model(graph, opset_imports=[helper.make_operatorsetid("", 18)])


def _make_string_normalizer_constant_input_model(
    *,
    values: list[str],
    case_change_action: str = "NONE",
    is_case_sensitive: bool = True,
    locale: str = "en_US",
    stopwords: list[str] | None = None,
) -> onnx.ModelProto:
    y = helper.make_tensor_value_info("output", TensorProto.STRING, [1])
    node_kwargs = {
        "case_change_action": str(case_change_action),
        "is_case_sensitive": int(bool(is_case_sensitive)),
        "locale": str(locale),
    }
    if stopwords is not None and len(stopwords) > 0:
        node_kwargs["stopwords"] = [str(v) for v in stopwords]
    node = helper.make_node(
        "StringNormalizer",
        ["input_const"],
        ["output"],
        name="StringNormalizerConstNode",
        **node_kwargs,
    )
    init = helper.make_tensor(
        name="input_const",
        data_type=TensorProto.STRING,
        dims=[len(values)],
        vals=[str(v) for v in values],
    )
    graph = helper.make_graph([node], "string_normalizer_const_graph", [], [y], initializer=[init])
    return helper.make_model(graph, opset_imports=[helper.make_operatorsetid("", 18)])


def test_flatbuffer_direct_string_normalizer_builtin_noop_lowering() -> None:
    model = _make_string_normalizer_model(
        case_change_action="NONE",
        stopwords=[],
        input_shape=[1],
        output_shape=[1, 1],
    )
    model_ir = lower_onnx_to_ir(
        model,
        "string_normalizer_builtin_noop",
        allow_custom_ops=False,
        optimize_layout_transpose_chains=False,
    )
    assert len(model_ir.operators) == 1
    op = model_ir.operators[0]
    assert op.op_type == "RESHAPE"
    assert op.outputs == ["output"]


def test_flatbuffer_direct_string_normalizer_builtin_stopwords_case_sensitive() -> None:
    model = _make_string_normalizer_model(
        case_change_action="NONE",
        is_case_sensitive=True,
        stopwords=["aaa", "bbb"],
        input_shape=[4],
        output_shape=[4],
    )
    model_ir = lower_onnx_to_ir(
        model,
        "string_normalizer_builtin_stopwords",
        allow_custom_ops=False,
        optimize_layout_transpose_chains=False,
    )
    op_types = [str(op.op_type) for op in model_ir.operators]
    assert "WHERE" in op_types
    assert "GATHER" in op_types
    assert "CUSTOM" not in op_types


def test_flatbuffer_direct_string_normalizer_constant_input_folds() -> None:
    model = _make_string_normalizer_constant_input_model(
        values=["AAA", "bbb", "ccc"],
        case_change_action="LOWER",
        is_case_sensitive=False,
        stopwords=["aaa", "bbb"],
    )
    model_ir = lower_onnx_to_ir(
        model,
        "string_normalizer_const_fold",
        allow_custom_ops=False,
        optimize_layout_transpose_chains=False,
    )
    assert len(model_ir.operators) == 0
    output_tensor = model_ir.tensors["output"]
    assert str(output_tensor.dtype) == "STRING"
    assert np.asarray(output_tensor.data, dtype=object).reshape(-1).tolist() == ["ccc"]


def test_flatbuffer_direct_string_normalizer_nontrivial_attrs_custom_fallback() -> None:
    model = _make_string_normalizer_model(
        case_change_action="LOWER",
        is_case_sensitive=False,
        stopwords=["aaa", "bbb"],
        input_shape=[1],
        output_shape=[1, 1],
    )
    model_ir = lower_onnx_to_ir(
        model,
        "string_normalizer_custom_fallback",
        allow_custom_ops=True,
        optimize_layout_transpose_chains=False,
    )
    assert len(model_ir.operators) == 1
    op = model_ir.operators[0]
    assert op.op_type == "CUSTOM"
    assert str(op.options.get("customCode")) == "ONNX_STRINGNORMALIZER"


def _make_optional_has_element_model() -> onnx.ModelProto:
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1])
    h = helper.make_tensor_value_info("/OptionalHasElement_output_0", TensorProto.BOOL, [])
    n = helper.make_tensor_value_info("/Not_output_0", TensorProto.BOOL, [])
    y = helper.make_tensor_value_info("y", TensorProto.BOOL, [])
    node_has = helper.make_node(
        "OptionalHasElement",
        ["x"],
        ["/OptionalHasElement_output_0"],
        name="/OptionalHasElement",
    )
    node_not0 = helper.make_node(
        "Not",
        ["/OptionalHasElement_output_0"],
        ["/Not_output_0"],
        name="/Not",
    )
    node_not1 = helper.make_node(
        "Not",
        ["/Not_output_0"],
        ["y"],
        name="/Not_1",
    )
    graph = helper.make_graph(
        [node_has, node_not0, node_not1],
        "optional_has_element_graph",
        [x],
        [y],
        value_info=[h, n],
    )
    return helper.make_model(graph, opset_imports=[helper.make_operatorsetid("", 15)])


def test_flatbuffer_direct_optional_has_element_builtin_lowering() -> None:
    model = _make_optional_has_element_model()
    model_ir = lower_onnx_to_ir(
        model,
        "optional_has_element_builtin",
        allow_custom_ops=False,
        optimize_layout_transpose_chains=False,
    )
    op_types = [str(op.op_type) for op in model_ir.operators]
    assert "CUSTOM" not in op_types

    has_element_tensor = model_ir.tensors["/OptionalHasElement_output_0"]
    assert str(has_element_tensor.dtype) == "BOOL"
    assert bool(np.asarray(has_element_tensor.data, dtype=np.bool_).reshape(()))


def _make_dropout_model(*, with_mask_output: bool = False) -> onnx.ModelProto:
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 3, 4, 4])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 3, 4, 4])
    outputs = ["y"]
    graph_outputs = [y]
    if with_mask_output:
        m = helper.make_tensor_value_info("m", TensorProto.BOOL, [1, 3, 4, 4])
        outputs.append("m")
        graph_outputs.append(m)

    node = helper.make_node(
        "Dropout",
        ["x"],
        outputs,
        name="DropoutNode",
    )
    graph = helper.make_graph([node], "dropout_graph", [x], graph_outputs)
    return helper.make_model(graph, opset_imports=[helper.make_operatorsetid("", 13)])


def _make_input_slice_model() -> onnx.ModelProto:
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 3, 4, 4])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 1, 4, 4])
    starts = numpy_helper.from_array(np.asarray([0], dtype=np.int64), name="slice_starts")
    ends = numpy_helper.from_array(np.asarray([1], dtype=np.int64), name="slice_ends")
    axes = numpy_helper.from_array(np.asarray([1], dtype=np.int64), name="slice_axes")
    steps = numpy_helper.from_array(np.asarray([1], dtype=np.int64), name="slice_steps")
    node = helper.make_node(
        "Slice",
        ["x", "slice_starts", "slice_ends", "slice_axes", "slice_steps"],
        ["y"],
        name="InputSliceNode",
    )
    graph = helper.make_graph(
        [node],
        "input_slice_graph",
        [x],
        [y],
        initializer=[starts, ends, axes, steps],
    )
    return helper.make_model(graph, opset_imports=[helper.make_operatorsetid("", 13)])


def _make_erf_tile_scatternd_model() -> onnx.ModelProto:
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [2, 3])
    updates = helper.make_tensor_value_info("updates", TensorProto.FLOAT, [2])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [2, 6])

    repeats = numpy_helper.from_array(np.asarray([1, 2], dtype=np.int64), name="repeats")
    indices = numpy_helper.from_array(
        np.asarray([[0, 1], [1, 4]], dtype=np.int64),
        name="indices",
    )
    nodes = [
        helper.make_node("Erf", ["x"], ["x_erf"], name="ErfNode"),
        helper.make_node("Tile", ["x_erf", "repeats"], ["x_tiled"], name="TileNode"),
        helper.make_node(
            "ScatterND",
            ["x_tiled", "indices", "updates"],
            ["y"],
            name="ScatterNDNode",
        ),
    ]
    graph = helper.make_graph(
        nodes,
        "erf_tile_scatternd_graph",
        [x, updates],
        [y],
        initializer=[repeats, indices],
    )
    return helper.make_model(graph, opset_imports=[helper.make_operatorsetid("", 16)])


def _make_add_const_model() -> onnx.ModelProto:
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 3])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 3])
    c = numpy_helper.from_array(
        np.array([[1.0, 2.0, 3.0]], dtype=np.float32),
        name="c",
    )
    node = helper.make_node("Add", ["x", "c"], ["y"], name="AddConstNode")
    graph = helper.make_graph([node], "add_const_graph", [x], [y], initializer=[c])
    return helper.make_model(graph, opset_imports=[helper.make_operatorsetid("", 13)])


def _make_where_neg_inf_broadcast_model() -> onnx.ModelProto:
    cond = helper.make_tensor_value_info("cond", TensorProto.BOOL, [1, 1, 1, 8])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 1, 1, 8])
    z = helper.make_tensor_value_info("z", TensorProto.FLOAT, [1, 1, 1, 8])
    x_const = numpy_helper.from_array(
        np.asarray([-np.inf], dtype=np.float32),
        name="x_const",
    )
    node = helper.make_node(
        "Where",
        ["cond", "x_const", "y"],
        ["z"],
        name="WhereNegInfBroadcastNode",
    )
    graph = helper.make_graph(
        [node],
        "where_neg_inf_broadcast_graph",
        [cond, y],
        [z],
        initializer=[x_const],
    )
    return helper.make_model(graph, opset_imports=[helper.make_operatorsetid("", 13)])


def _make_conv_model() -> onnx.ModelProto:
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 1, 4, 4])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 1, 2, 2])
    w = numpy_helper.from_array(np.ones((1, 1, 3, 3), dtype=np.float32), name="W")
    b = numpy_helper.from_array(np.zeros((1,), dtype=np.float32), name="B")
    node = helper.make_node(
        "Conv",
        ["x", "W", "B"],
        ["y"],
        name="ConvNode",
        pads=[0, 0, 0, 0],
        strides=[1, 1],
    )
    graph = helper.make_graph([node], "conv_graph", [x], [y], initializer=[w, b])
    return helper.make_model(graph, opset_imports=[helper.make_operatorsetid("", 13)])


def _make_conv_fp16_model() -> onnx.ModelProto:
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT16, [1, 1, 4, 4])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT16, [1, 1, 2, 2])
    w = numpy_helper.from_array(np.ones((1, 1, 3, 3), dtype=np.float16), name="W")
    b = numpy_helper.from_array(np.zeros((1,), dtype=np.float16), name="B")
    node = helper.make_node(
        "Conv",
        ["x", "W", "B"],
        ["y"],
        name="ConvFp16Node",
        pads=[0, 0, 0, 0],
        strides=[1, 1],
    )
    graph = helper.make_graph([node], "conv_fp16_graph", [x], [y], initializer=[w, b])
    return helper.make_model(graph, opset_imports=[helper.make_operatorsetid("", 13)])


def _make_conv_clip_model(*, clip_min: float, clip_max: float) -> onnx.ModelProto:
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 1, 4, 4])
    z = helper.make_tensor_value_info("z", TensorProto.FLOAT, [1, 1, 2, 2])
    w = numpy_helper.from_array(np.ones((1, 1, 3, 3), dtype=np.float32), name="W")
    b = numpy_helper.from_array(np.zeros((1,), dtype=np.float32), name="B")
    conv = helper.make_node(
        "Conv",
        ["x", "W", "B"],
        ["y"],
        name="ConvNode",
        pads=[0, 0, 0, 0],
        strides=[1, 1],
    )
    clip = helper.make_node(
        "Clip",
        ["y"],
        ["z"],
        name="ConvClipNode",
        min=float(clip_min),
        max=float(clip_max),
    )
    graph = helper.make_graph([conv, clip], "conv_clip_graph", [x], [z], initializer=[w, b])
    return helper.make_model(graph, opset_imports=[helper.make_operatorsetid("", 13)])


def _make_depthwise_conv_clip_model(*, clip_min: float, clip_max: float) -> onnx.ModelProto:
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 2, 4, 4])
    z = helper.make_tensor_value_info("z", TensorProto.FLOAT, [1, 2, 2, 2])
    # ONNX depthwise: group==in_channels, filter shape=[out_channels, in_channels/group, kh, kw]
    w = numpy_helper.from_array(np.ones((2, 1, 3, 3), dtype=np.float32), name="W")
    b = numpy_helper.from_array(np.zeros((2,), dtype=np.float32), name="B")
    conv = helper.make_node(
        "Conv",
        ["x", "W", "B"],
        ["y"],
        name="DepthwiseConvNode",
        pads=[0, 0, 0, 0],
        strides=[1, 1],
        group=2,
    )
    clip = helper.make_node(
        "Clip",
        ["y"],
        ["z"],
        name="DepthwiseConvClipNode",
        min=float(clip_min),
        max=float(clip_max),
    )
    graph = helper.make_graph([conv, clip], "depthwise_conv_clip_graph", [x], [z], initializer=[w, b])
    return helper.make_model(graph, opset_imports=[helper.make_operatorsetid("", 13)])


def _make_fused_conv_model(
    *,
    activation: str,
    activation_params: list[float] | None = None,
) -> onnx.ModelProto:
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 3, 8, 8])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 4, 8, 8])
    w = numpy_helper.from_array(np.ones((4, 3, 3, 3), dtype=np.float32), name="W")
    b = numpy_helper.from_array(np.zeros((4,), dtype=np.float32), name="B")
    attrs: dict[str, object] = {
        "pads": [1, 1, 1, 1],
        "strides": [1, 1],
        "dilations": [1, 1],
        "group": 1,
        "activation": str(activation),
    }
    if activation_params is not None:
        attrs["activation_params"] = [float(v) for v in list(activation_params)]
    node = helper.make_node(
        "FusedConv",
        ["x", "W", "B"],
        ["y"],
        name=f"FusedConv{activation}Node",
        domain="com.microsoft",
        **attrs,
    )
    graph = helper.make_graph([node], f"fused_conv_{activation.lower()}_graph", [x], [y], initializer=[w, b])
    return helper.make_model(
        graph,
        opset_imports=[
            helper.make_operatorsetid("", 13),
            helper.make_operatorsetid("com.microsoft", 1),
        ],
    )


def _make_fused_conv_clip_chain_unknown_intermediate_model() -> onnx.ModelProto:
    x = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 3, 640, 640])
    y = helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 28, 320, 320])
    w1 = numpy_helper.from_array(np.ones((28, 3, 3, 3), dtype=np.float32), name="fc1w")
    b1 = numpy_helper.from_array(np.zeros((28,), dtype=np.float32), name="fc1b")
    w2 = numpy_helper.from_array(np.ones((28, 28, 3, 3), dtype=np.float32), name="fc2w")
    b2 = numpy_helper.from_array(np.zeros((28,), dtype=np.float32), name="fc2b")
    n0 = helper.make_node(
        "FusedConv",
        ["input", "fc1w", "fc1b"],
        ["fc1_output"],
        name="FusedConv_0",
        domain="com.microsoft",
        pads=[1, 1, 1, 1],
        strides=[2, 2],
        activation="Clip",
        activation_params=[0.0, 6.0],
    )
    n1 = helper.make_node(
        "FusedConv",
        ["fc1_output", "fc2w", "fc2b"],
        ["output"],
        name="FusedConv_1",
        domain="com.microsoft",
        pads=[1, 1, 1, 1],
        strides=[1, 1],
        activation="Clip",
        activation_params=[0.0, 6.0],
    )
    graph = helper.make_graph([n0, n1], "fused_conv_clip_chain_graph", [x], [y], initializer=[w1, b1, w2, b2])
    return helper.make_model(
        graph,
        opset_imports=[
            helper.make_operatorsetid("", 13),
            helper.make_operatorsetid("com.microsoft", 2),
        ],
    )


def _make_fused_conv_leakyrelu_chain_unknown_intermediate_model() -> onnx.ModelProto:
    x = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 3, 640, 640])
    y = helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 28, 320, 320])
    w1 = numpy_helper.from_array(np.ones((28, 3, 3, 3), dtype=np.float32), name="fc1w")
    b1 = numpy_helper.from_array(np.zeros((28,), dtype=np.float32), name="fc1b")
    w2 = numpy_helper.from_array(np.ones((28, 28, 3, 3), dtype=np.float32), name="fc2w")
    b2 = numpy_helper.from_array(np.zeros((28,), dtype=np.float32), name="fc2b")
    n0 = helper.make_node(
        "FusedConv",
        ["input", "fc1w", "fc1b"],
        ["fc1_output"],
        name="FusedConv_0",
        domain="com.microsoft",
        pads=[1, 1, 1, 1],
        strides=[2, 2],
        activation="LeakyRelu",
        activation_params=[0.01],
    )
    n1 = helper.make_node(
        "FusedConv",
        ["fc1_output", "fc2w", "fc2b"],
        ["output"],
        name="FusedConv_1",
        domain="com.microsoft",
        pads=[1, 1, 1, 1],
        strides=[1, 1],
        activation="LeakyRelu",
        activation_params=[0.01],
    )
    graph = helper.make_graph(
        [n0, n1],
        "fused_conv_leakyrelu_chain_graph",
        [x],
        [y],
        initializer=[w1, b1, w2, b2],
    )
    return helper.make_model(
        graph,
        opset_imports=[
            helper.make_operatorsetid("", 13),
            helper.make_operatorsetid("com.microsoft", 2),
        ],
    )


def _make_fused_conv_tanh_chain_unknown_intermediate_model() -> onnx.ModelProto:
    x = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 3, 640, 640])
    y = helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 28, 320, 320])
    w1 = numpy_helper.from_array(np.ones((28, 3, 3, 3), dtype=np.float32), name="fc1w")
    b1 = numpy_helper.from_array(np.zeros((28,), dtype=np.float32), name="fc1b")
    w2 = numpy_helper.from_array(np.ones((28, 28, 3, 3), dtype=np.float32), name="fc2w")
    b2 = numpy_helper.from_array(np.zeros((28,), dtype=np.float32), name="fc2b")
    n0 = helper.make_node(
        "FusedConv",
        ["input", "fc1w", "fc1b"],
        ["fc1_output"],
        name="FusedConv_0",
        domain="com.microsoft",
        pads=[1, 1, 1, 1],
        strides=[2, 2],
        activation="Tanh",
    )
    n1 = helper.make_node(
        "FusedConv",
        ["fc1_output", "fc2w", "fc2b"],
        ["output"],
        name="FusedConv_1",
        domain="com.microsoft",
        pads=[1, 1, 1, 1],
        strides=[1, 1],
        activation="Tanh",
    )
    graph = helper.make_graph(
        [n0, n1],
        "fused_conv_tanh_chain_graph",
        [x],
        [y],
        initializer=[w1, b1, w2, b2],
    )
    return helper.make_model(
        graph,
        opset_imports=[
            helper.make_operatorsetid("", 13),
            helper.make_operatorsetid("com.microsoft", 2),
        ],
    )


def _make_conv1d_model() -> onnx.ModelProto:
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 1, 8])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 2, 8])
    w = numpy_helper.from_array(np.ones((2, 1, 3), dtype=np.float32), name="W")
    b = numpy_helper.from_array(np.zeros((2,), dtype=np.float32), name="B")
    node = helper.make_node(
        "Conv",
        ["x", "W", "B"],
        ["y"],
        name="Conv1DNode",
        pads=[1, 1],
        strides=[1],
        dilations=[1],
    )
    graph = helper.make_graph([node], "conv1d_graph", [x], [y], initializer=[w, b])
    return helper.make_model(graph, opset_imports=[helper.make_operatorsetid("", 13)])


def _make_convtranspose1d_model() -> onnx.ModelProto:
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 2, 8])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 1, 8])
    w = numpy_helper.from_array(np.ones((2, 1, 3), dtype=np.float32), name="W")
    b = numpy_helper.from_array(np.zeros((1,), dtype=np.float32), name="B")
    node = helper.make_node(
        "ConvTranspose",
        ["x", "W", "B"],
        ["y"],
        name="ConvTranspose1DNode",
        pads=[1, 1],
        strides=[1],
        dilations=[1],
    )
    graph = helper.make_graph([node], "convtranspose1d_graph", [x], [y], initializer=[w, b])
    return helper.make_model(graph, opset_imports=[helper.make_operatorsetid("", 13)])


def _make_convtranspose2d_output_padding_model() -> onnx.ModelProto:
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 2, 4, 4])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 3, 8, 8])
    w = numpy_helper.from_array(np.ones((2, 3, 3, 3), dtype=np.float32), name="W")
    b = numpy_helper.from_array(np.zeros((3,), dtype=np.float32), name="B")
    node = helper.make_node(
        "ConvTranspose",
        ["x", "W", "B"],
        ["y"],
        name="ConvTranspose2DOutputPaddingNode",
        pads=[1, 1, 1, 1],
        strides=[2, 2],
        dilations=[1, 1],
        output_padding=[1, 1],
    )
    graph = helper.make_graph(
        [node],
        "convtranspose2d_output_padding_graph",
        [x],
        [y],
        initializer=[w, b],
    )
    return helper.make_model(graph, opset_imports=[helper.make_operatorsetid("", 13)])


def _make_conv3d_model() -> onnx.ModelProto:
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 2, 5, 6, 7])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 3, 5, 6, 7])
    w = numpy_helper.from_array(np.ones((3, 2, 3, 3, 3), dtype=np.float32), name="W")
    b = numpy_helper.from_array(np.zeros((3,), dtype=np.float32), name="B")
    node = helper.make_node(
        "Conv",
        ["x", "W", "B"],
        ["y"],
        name="Conv3DNode",
        pads=[1, 1, 1, 1, 1, 1],
        strides=[1, 1, 1],
        dilations=[1, 1, 1],
    )
    graph = helper.make_graph([node], "conv3d_graph", [x], [y], initializer=[w, b])
    return helper.make_model(graph, opset_imports=[helper.make_operatorsetid("", 13)])


def _make_convtranspose3d_model() -> onnx.ModelProto:
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 3, 3, 3, 3])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 2, 6, 6, 6])
    w = numpy_helper.from_array(np.ones((3, 2, 4, 4, 4), dtype=np.float32), name="W")
    b = numpy_helper.from_array(np.zeros((2,), dtype=np.float32), name="B")
    node = helper.make_node(
        "ConvTranspose",
        ["x", "W", "B"],
        ["y"],
        name="ConvTranspose3DNode",
        pads=[1, 1, 1, 1, 1, 1],
        strides=[2, 2, 2],
        dilations=[1, 1, 1],
        output_padding=[0, 0, 0],
    )
    graph = helper.make_graph(
        [node],
        "convtranspose3d_graph",
        [x],
        [y],
        initializer=[w, b],
    )
    return helper.make_model(graph, opset_imports=[helper.make_operatorsetid("", 13)])


def _make_conv_dynamic_batch_model() -> onnx.ModelProto:
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, ["N", 1, 4, 4])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, ["N", 1, 2, 2])
    w = numpy_helper.from_array(np.ones((1, 1, 3, 3), dtype=np.float32), name="W")
    b = numpy_helper.from_array(np.zeros((1,), dtype=np.float32), name="B")
    node = helper.make_node(
        "Conv",
        ["x", "W", "B"],
        ["y"],
        name="ConvNodeDynamicBatch",
        pads=[0, 0, 0, 0],
        strides=[1, 1],
    )
    graph = helper.make_graph([node], "conv_dynamic_batch_graph", [x], [y], initializer=[w, b])
    return helper.make_model(graph, opset_imports=[helper.make_operatorsetid("", 13)])


def _make_conv_stride2_symmetric_pad_model() -> onnx.ModelProto:
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 3, 8, 8])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 4, 4, 4])
    w = numpy_helper.from_array(np.ones((4, 3, 3, 3), dtype=np.float32), name="W")
    b = numpy_helper.from_array(np.zeros((4,), dtype=np.float32), name="B")
    node = helper.make_node(
        "Conv",
        ["x", "W", "B"],
        ["y"],
        name="ConvStride2SymPadNode",
        pads=[1, 1, 1, 1],
        strides=[2, 2],
        dilations=[1, 1],
    )
    graph = helper.make_graph(
        [node],
        "conv_stride2_symmetric_pad_graph",
        [x],
        [y],
        initializer=[w, b],
    )
    return helper.make_model(graph, opset_imports=[helper.make_operatorsetid("", 13)])


def _make_grouped_conv_model() -> onnx.ModelProto:
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 4, 5, 5])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 6, 5, 5])
    w = numpy_helper.from_array(np.ones((6, 2, 3, 3), dtype=np.float32), name="W")
    b = numpy_helper.from_array(np.zeros((6,), dtype=np.float32), name="B")
    node = helper.make_node(
        "Conv",
        ["x", "W", "B"],
        ["y"],
        name="GroupedConvNode",
        group=2,
        pads=[1, 1, 1, 1],
        strides=[1, 1],
    )
    graph = helper.make_graph([node], "grouped_conv_graph", [x], [y], initializer=[w, b])
    return helper.make_model(graph, opset_imports=[helper.make_operatorsetid("", 13)])


def _make_pool_model() -> onnx.ModelProto:
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 1, 4, 4])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 1, 2, 2])
    node = helper.make_node(
        "AveragePool",
        ["x"],
        ["y"],
        name="PoolNode",
        kernel_shape=[2, 2],
        strides=[2, 2],
        pads=[0, 0, 0, 0],
    )
    graph = helper.make_graph([node], "pool_graph", [x], [y])
    return helper.make_model(graph, opset_imports=[helper.make_operatorsetid("", 13)])


def _make_global_max_pool_model() -> onnx.ModelProto:
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 3, 4, 5])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 3, 1, 1])
    node = helper.make_node(
        "GlobalMaxPool",
        ["x"],
        ["y"],
        name="GlobalMaxPoolNode",
    )
    graph = helper.make_graph([node], "global_max_pool_graph", [x], [y])
    return helper.make_model(graph, opset_imports=[helper.make_operatorsetid("", 13)])


def _make_reduce_l2_model() -> onnx.ModelProto:
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 3, 4, 5])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 1, 4, 5])
    node = helper.make_node(
        "ReduceL2",
        ["x"],
        ["y"],
        name="ReduceL2Node",
        axes=[1],
        keepdims=1,
    )
    graph = helper.make_graph([node], "reduce_l2_graph", [x], [y])
    return helper.make_model(graph, opset_imports=[helper.make_operatorsetid("", 13)])


def _make_average_pool_exclude_pad_model() -> onnx.ModelProto:
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 1, 4, 4])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 1, 2, 2])
    node = helper.make_node(
        "AveragePool",
        ["x"],
        ["y"],
        name="AveragePoolExcludePadNode",
        kernel_shape=[3, 3],
        strides=[2, 2],
        pads=[1, 1, 1, 1],
        count_include_pad=0,
    )
    graph = helper.make_graph([node], "average_pool_exclude_pad_graph", [x], [y])
    return helper.make_model(graph, opset_imports=[helper.make_operatorsetid("", 13)])


def _make_maxpool_dynamic_batch_model() -> onnx.ModelProto:
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, ["N", 1, 4, 4])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, ["N", 1, 2, 2])
    node = helper.make_node(
        "MaxPool",
        ["x"],
        ["y"],
        name="MaxPoolNodeDynamicBatch",
        kernel_shape=[2, 2],
        strides=[2, 2],
        pads=[0, 0, 0, 0],
    )
    graph = helper.make_graph([node], "maxpool_dynamic_batch_graph", [x], [y])
    return helper.make_model(graph, opset_imports=[helper.make_operatorsetid("", 13)])


def _make_gemm_model() -> onnx.ModelProto:
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 4])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 3])
    w = numpy_helper.from_array(np.ones((3, 4), dtype=np.float32), name="W")
    b = numpy_helper.from_array(np.zeros((3,), dtype=np.float32), name="B")
    node = helper.make_node("Gemm", ["x", "W", "B"], ["y"], name="GemmNode", transB=1)
    graph = helper.make_graph([node], "gemm_graph", [x], [y], initializer=[w, b])
    return helper.make_model(graph, opset_imports=[helper.make_operatorsetid("", 13)])


def _make_add_chain_model() -> onnx.ModelProto:
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 3])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 3])
    z = helper.make_tensor_value_info("z", TensorProto.FLOAT, [1, 3])
    n0 = helper.make_node("Add", ["x", "y"], ["a0"], name="AddChain0")
    n1 = helper.make_node("Add", ["a0", "y"], ["a1"], name="AddChain1")
    n2 = helper.make_node("Add", ["a1", "y"], ["z"], name="AddChain2")
    graph = helper.make_graph([n0, n1, n2], "add_chain_graph", [x, y], [z])
    return helper.make_model(graph, opset_imports=[helper.make_operatorsetid("", 13)])


def _make_add_relu_model() -> onnx.ModelProto:
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 3])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 3])
    z = helper.make_tensor_value_info("z", TensorProto.FLOAT, [1, 3])
    n0 = helper.make_node("Add", ["x", "y"], ["a0"], name="AddReluAdd")
    n1 = helper.make_node("Relu", ["a0"], ["z"], name="AddReluRelu")
    graph = helper.make_graph([n0, n1], "add_relu_graph", [x, y], [z])
    return helper.make_model(graph, opset_imports=[helper.make_operatorsetid("", 13)])


def _make_binary_activation_model(binary_op: str, activation_op: str) -> onnx.ModelProto:
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 3])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 3])
    z = helper.make_tensor_value_info("z", TensorProto.FLOAT, [1, 3])
    n0 = helper.make_node(binary_op, ["x", "y"], ["a0"], name=f"{binary_op}ActivationBinary")
    if activation_op == "Relu6":
        n1 = helper.make_node(
            "Clip",
            ["a0"],
            ["z"],
            name=f"{binary_op}ActivationRelu6",
            min=0.0,
            max=6.0,
        )
    else:
        n1 = helper.make_node(activation_op, ["a0"], ["z"], name=f"{binary_op}Activation{activation_op}")
    graph = helper.make_graph(
        [n0, n1],
        f"{binary_op.lower()}_{activation_op.lower()}_graph",
        [x, y],
        [z],
    )
    return helper.make_model(graph, opset_imports=[helper.make_operatorsetid("", 13)])


def _make_forward_lstm_model() -> onnx.ModelProto:
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [3, 1, 2])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [3, 1, 1, 2])
    W = numpy_helper.from_array(
        np.asarray(
            [
                [
                    [0.20, -0.10],
                    [0.05, 0.15],
                    [0.30, -0.25],
                    [0.10, 0.12],
                    [-0.18, 0.08],
                    [0.11, -0.09],
                    [0.04, 0.07],
                    [-0.06, 0.03],
                ]
            ],
            dtype=np.float32,
        ),
        name="W",
    )
    R = numpy_helper.from_array(
        np.asarray(
            [
                [
                    [0.05, -0.02],
                    [0.03, 0.01],
                    [0.02, -0.04],
                    [0.01, 0.03],
                    [-0.02, 0.02],
                    [0.04, -0.01],
                    [0.03, 0.02],
                    [-0.01, 0.01],
                ]
            ],
            dtype=np.float32,
        ),
        name="R",
    )
    B = numpy_helper.from_array(
        np.asarray(
            [[0.01, -0.02, 0.03, 0.01, -0.01, 0.00, 0.02, -0.03, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]],
            dtype=np.float32,
        ),
        name="B",
    )
    initial_h = numpy_helper.from_array(np.zeros((1, 1, 2), dtype=np.float32), name="initial_h")
    initial_c = numpy_helper.from_array(np.zeros((1, 1, 2), dtype=np.float32), name="initial_c")
    lstm = helper.make_node(
        "LSTM",
        ["x", "W", "R", "B", "", "initial_h", "initial_c"],
        ["y"],
        name="ForwardLSTMNode",
        hidden_size=2,
        direction="forward",
    )
    graph = helper.make_graph(
        [lstm],
        "forward_lstm_graph",
        [x],
        [y],
        initializer=[W, R, B, initial_h, initial_c],
    )
    return helper.make_model(graph, opset_imports=[helper.make_operatorsetid("", 13)])


def _make_reverse_lstm_model() -> onnx.ModelProto:
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [3, 1, 2])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [3, 1, 1, 2])
    W = numpy_helper.from_array(
        np.asarray(
            [
                [
                    [0.20, -0.10],
                    [0.05, 0.15],
                    [0.30, -0.25],
                    [0.10, 0.12],
                    [-0.18, 0.08],
                    [0.11, -0.09],
                    [0.04, 0.07],
                    [-0.06, 0.03],
                ]
            ],
            dtype=np.float32,
        ),
        name="W",
    )
    R = numpy_helper.from_array(
        np.asarray(
            [
                [
                    [0.05, -0.02],
                    [0.03, 0.01],
                    [0.02, -0.04],
                    [0.01, 0.03],
                    [-0.02, 0.02],
                    [0.04, -0.01],
                    [0.03, 0.02],
                    [-0.01, 0.01],
                ]
            ],
            dtype=np.float32,
        ),
        name="R",
    )
    B = numpy_helper.from_array(
        np.asarray(
            [[0.01, -0.02, 0.03, 0.01, -0.01, 0.00, 0.02, -0.03, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]],
            dtype=np.float32,
        ),
        name="B",
    )
    initial_h = numpy_helper.from_array(np.zeros((1, 1, 2), dtype=np.float32), name="initial_h")
    initial_c = numpy_helper.from_array(np.zeros((1, 1, 2), dtype=np.float32), name="initial_c")
    lstm = helper.make_node(
        "LSTM",
        ["x", "W", "R", "B", "", "initial_h", "initial_c"],
        ["y"],
        name="ReverseLSTMNode",
        hidden_size=2,
        direction="reverse",
    )
    graph = helper.make_graph(
        [lstm],
        "reverse_lstm_graph",
        [x],
        [y],
        initializer=[W, R, B, initial_h, initial_c],
    )
    return helper.make_model(graph, opset_imports=[helper.make_operatorsetid("", 13)])


def _make_forward_lstm_with_state_io_model() -> onnx.ModelProto:
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [4, 1, 2])
    h_in = helper.make_tensor_value_info("h_in", TensorProto.FLOAT, [1, 1, 2])
    c_in = helper.make_tensor_value_info("c_in", TensorProto.FLOAT, [1, 1, 2])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [4, 1, 1, 2])
    h_out = helper.make_tensor_value_info("h_out", TensorProto.FLOAT, [1, 1, 2])
    c_out = helper.make_tensor_value_info("c_out", TensorProto.FLOAT, [1, 1, 2])
    W = numpy_helper.from_array(
        np.asarray(
            [
                [
                    [0.20, -0.10],
                    [0.05, 0.15],
                    [0.30, -0.25],
                    [0.10, 0.12],
                    [-0.18, 0.08],
                    [0.11, -0.09],
                    [0.04, 0.07],
                    [-0.06, 0.03],
                ]
            ],
            dtype=np.float32,
        ),
        name="W",
    )
    R = numpy_helper.from_array(
        np.asarray(
            [
                [
                    [0.05, -0.02],
                    [0.03, 0.01],
                    [0.02, -0.04],
                    [0.01, 0.03],
                    [-0.02, 0.02],
                    [0.04, -0.01],
                    [0.03, 0.02],
                    [-0.01, 0.01],
                ]
            ],
            dtype=np.float32,
        ),
        name="R",
    )
    B = numpy_helper.from_array(
        np.asarray(
            [[0.01, -0.02, 0.03, 0.01, -0.01, 0.00, 0.02, -0.03, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]],
            dtype=np.float32,
        ),
        name="B",
    )
    lstm = helper.make_node(
        "LSTM",
        ["x", "W", "R", "B", "", "h_in", "c_in"],
        ["y", "h_out", "c_out"],
        name="ForwardLSTMStateIONode",
        hidden_size=2,
        direction="forward",
    )
    graph = helper.make_graph(
        [lstm],
        "forward_lstm_state_io_graph",
        [x, h_in, c_in],
        [y, h_out, c_out],
        initializer=[W, R, B],
    )
    return helper.make_model(graph, opset_imports=[helper.make_operatorsetid("", 13)])


def _make_reverse_rnn_model() -> onnx.ModelProto:
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [4, 1, 2])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [4, 1, 1, 3])
    y_h = helper.make_tensor_value_info("y_h", TensorProto.FLOAT, [1, 1, 3])
    W = numpy_helper.from_array(
        np.asarray(
            [
                [
                    [0.20, -0.10],
                    [0.03, 0.07],
                    [-0.04, 0.12],
                ]
            ],
            dtype=np.float32,
        ),
        name="W",
    )
    R = numpy_helper.from_array(
        np.asarray(
            [
                [
                    [0.05, -0.02, 0.01],
                    [0.02, 0.04, -0.03],
                    [-0.01, 0.03, 0.02],
                ]
            ],
            dtype=np.float32,
        ),
        name="R",
    )
    B = numpy_helper.from_array(
        np.asarray([[0.01, -0.02, 0.03, 0.00, 0.00, 0.00]], dtype=np.float32),
        name="B",
    )
    initial_h = numpy_helper.from_array(np.zeros((1, 1, 3), dtype=np.float32), name="initial_h")
    rnn = helper.make_node(
        "RNN",
        ["x", "W", "R", "B", "", "initial_h"],
        ["y", "y_h"],
        name="ReverseRNNNode",
        hidden_size=3,
        direction="reverse",
        activations=["Relu"],
    )
    graph = helper.make_graph(
        [rnn],
        "reverse_rnn_graph",
        [x],
        [y, y_h],
        initializer=[W, R, B, initial_h],
    )
    return helper.make_model(graph, opset_imports=[helper.make_operatorsetid("", 13)])


def _make_bidirectional_rnn_model() -> onnx.ModelProto:
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [4, 1, 2])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [4, 2, 1, 3])
    y_h = helper.make_tensor_value_info("y_h", TensorProto.FLOAT, [2, 1, 3])
    W = numpy_helper.from_array(
        np.asarray(
            [
                [
                    [0.20, -0.10],
                    [0.03, 0.07],
                    [-0.04, 0.12],
                ],
                [
                    [0.11, -0.05],
                    [0.06, 0.02],
                    [-0.03, 0.09],
                ],
            ],
            dtype=np.float32,
        ),
        name="W",
    )
    R = numpy_helper.from_array(
        np.asarray(
            [
                [
                    [0.05, -0.02, 0.01],
                    [0.02, 0.04, -0.03],
                    [-0.01, 0.03, 0.02],
                ],
                [
                    [0.03, -0.01, 0.02],
                    [0.01, 0.02, -0.02],
                    [-0.02, 0.02, 0.01],
                ],
            ],
            dtype=np.float32,
        ),
        name="R",
    )
    B = numpy_helper.from_array(
        np.asarray(
            [
                [0.01, -0.02, 0.03, 0.00, 0.00, 0.00],
                [0.02, -0.01, 0.01, 0.00, 0.00, 0.00],
            ],
            dtype=np.float32,
        ),
        name="B",
    )
    initial_h = numpy_helper.from_array(np.zeros((2, 1, 3), dtype=np.float32), name="initial_h")
    rnn = helper.make_node(
        "RNN",
        ["x", "W", "R", "B", "", "initial_h"],
        ["y", "y_h"],
        name="BidirectionalRNNNode",
        hidden_size=3,
        direction="bidirectional",
        activations=["Tanh", "Relu"],
    )
    graph = helper.make_graph(
        [rnn],
        "bidirectional_rnn_graph",
        [x],
        [y, y_h],
        initializer=[W, R, B, initial_h],
    )
    return helper.make_model(graph, opset_imports=[helper.make_operatorsetid("", 13)])


def _make_transpose_wrapped_add_relu_model() -> onnx.ModelProto:
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 2, 3, 4])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 2, 3, 4])
    z = helper.make_tensor_value_info("z", TensorProto.FLOAT, [1, 2, 3, 4])
    nodes = [
        helper.make_node("Transpose", ["x"], ["x_nchw"], name="TWAR_X", perm=[0, 3, 1, 2]),
        helper.make_node("Transpose", ["y"], ["y_nchw"], name="TWAR_Y", perm=[0, 3, 1, 2]),
        helper.make_node("Add", ["x_nchw", "y_nchw"], ["sum_nchw"], name="TWAR_Add"),
        helper.make_node("Relu", ["sum_nchw"], ["sum_relu_nchw"], name="TWAR_Relu0"),
        helper.make_node(
            "Transpose",
            ["sum_relu_nchw"],
            ["sum_nhwc"],
            name="TWAR_Out",
            perm=[0, 2, 3, 1],
        ),
        helper.make_node("Relu", ["sum_nhwc"], ["z"], name="TWAR_Relu1"),
    ]
    graph = helper.make_graph(nodes, "transpose_wrapped_add_relu_graph", [x, y], [z])
    return helper.make_model(graph, opset_imports=[helper.make_operatorsetid("", 13)])


def _make_transpose_mul_const_add_transpose_model() -> onnx.ModelProto:
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 2, 3, 4])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 2, 3, 4])
    z = helper.make_tensor_value_info("z", TensorProto.FLOAT, [1, 2, 3, 4])
    scale = numpy_helper.from_array(
        np.asarray([[[[1.0]], [[0.5]], [[1.5]], [[2.0]]]], dtype=np.float32),
        name="tmcat_scale_nchw",
    )
    nodes = [
        helper.make_node("Transpose", ["x"], ["x_nchw"], name="TMCAT_X", perm=[0, 3, 1, 2]),
        helper.make_node("Transpose", ["y"], ["y_nchw"], name="TMCAT_Y", perm=[0, 3, 1, 2]),
        helper.make_node("Mul", ["y_nchw", "tmcat_scale_nchw"], ["y_scaled_nchw"], name="TMCAT_Mul"),
        helper.make_node("Add", ["x_nchw", "y_scaled_nchw"], ["sum_nchw"], name="TMCAT_Add"),
        helper.make_node("Transpose", ["sum_nchw"], ["sum_nhwc"], name="TMCAT_Out", perm=[0, 2, 3, 1]),
        helper.make_node("Relu", ["sum_nhwc"], ["z"], name="TMCAT_Relu"),
    ]
    graph = helper.make_graph(
        nodes,
        "transpose_mul_const_add_transpose_graph",
        [x, y],
        [z],
        initializer=[scale],
    )
    return helper.make_model(graph, opset_imports=[helper.make_operatorsetid("", 13)])


def _make_non_max_suppression_model() -> onnx.ModelProto:
    boxes = helper.make_tensor_value_info("boxes", TensorProto.FLOAT, [1, 5, 4])
    scores = helper.make_tensor_value_info("scores", TensorProto.FLOAT, [1, 1, 5])
    selected_indices = helper.make_tensor_value_info("selected_indices", TensorProto.INT64, [20, 3])

    max_output = numpy_helper.from_array(np.asarray([20], dtype=np.int64), name="max_output")
    iou_threshold = numpy_helper.from_array(np.asarray([0.5], dtype=np.float32), name="iou_threshold")
    score_threshold = numpy_helper.from_array(np.asarray([0.4], dtype=np.float32), name="score_threshold")

    node = helper.make_node(
        "NonMaxSuppression",
        ["boxes", "scores", "max_output", "iou_threshold", "score_threshold"],
        ["selected_indices"],
        name="NMSNode",
        center_point_box=0,
    )
    graph = helper.make_graph(
        [node],
        "nms_graph",
        [boxes, scores],
        [selected_indices],
        initializer=[max_output, iou_threshold, score_threshold],
    )
    return helper.make_model(graph, opset_imports=[helper.make_operatorsetid("", 13)])


def _make_non_max_suppression_multiclass_model() -> onnx.ModelProto:
    boxes = helper.make_tensor_value_info("boxes", TensorProto.FLOAT, [1, 5, 4])
    scores = helper.make_tensor_value_info("scores", TensorProto.FLOAT, [1, 3, 5])
    selected_indices = helper.make_tensor_value_info("selected_indices", TensorProto.INT64, [20, 3])

    max_output = numpy_helper.from_array(np.asarray([20], dtype=np.int64), name="max_output")
    iou_threshold = numpy_helper.from_array(np.asarray([0.5], dtype=np.float32), name="iou_threshold")
    score_threshold = numpy_helper.from_array(np.asarray([0.4], dtype=np.float32), name="score_threshold")

    node = helper.make_node(
        "NonMaxSuppression",
        ["boxes", "scores", "max_output", "iou_threshold", "score_threshold"],
        ["selected_indices"],
        name="NMSNodeMultiClass",
        center_point_box=0,
    )
    graph = helper.make_graph(
        [node],
        "nms_multiclass_graph",
        [boxes, scores],
        [selected_indices],
        initializer=[max_output, iou_threshold, score_threshold],
    )
    return helper.make_model(graph, opset_imports=[helper.make_operatorsetid("", 13)])


def _make_unary_model(op_type: str, *, name: str) -> onnx.ModelProto:
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 3])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 3])
    node = helper.make_node(op_type, ["x"], ["y"], name=f"{name}Node")
    graph = helper.make_graph([node], f"{name}_graph", [x], [y])
    return helper.make_model(graph, opset_imports=[helper.make_operatorsetid("", 13)])


def _make_unary_rank4_model(op_type: str, *, name: str) -> onnx.ModelProto:
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 3, 8, 8])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 3, 8, 8])
    node = helper.make_node(op_type, ["x"], ["y"], name=f"{name}Node")
    graph = helper.make_graph([node], f"{name}_rank4_graph", [x], [y])
    return helper.make_model(graph, opset_imports=[helper.make_operatorsetid("", 13)])


def _make_unary_rank4_dynamic_hw_model(op_type: str, *, name: str) -> onnx.ModelProto:
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 3, "H", "W"])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 3, "H", "W"])
    node = helper.make_node(op_type, ["x"], ["y"], name=f"{name}Node")
    graph = helper.make_graph([node], f"{name}_rank4_dynamic_hw_graph", [x], [y])
    return helper.make_model(graph, opset_imports=[helper.make_operatorsetid("", 13)])


def _make_clip_relu6_model() -> onnx.ModelProto:
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 3])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 3])
    node = helper.make_node("Clip", ["x"], ["y"], name="ClipNode", min=0.0, max=6.0)
    graph = helper.make_graph([node], "clip_relu6_graph", [x], [y])
    return helper.make_model(graph, opset_imports=[helper.make_operatorsetid("", 13)])


def _make_clip_relu_n1_to_1_model() -> onnx.ModelProto:
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 3])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 3])
    node = helper.make_node("Clip", ["x"], ["y"], name="ClipNode", min=-1.0, max=1.0)
    graph = helper.make_graph([node], "clip_relu_n1_to_1_graph", [x], [y])
    return helper.make_model(graph, opset_imports=[helper.make_operatorsetid("", 13)])


def _make_elu_model() -> onnx.ModelProto:
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 3])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 3])
    node = helper.make_node("Elu", ["x"], ["y"], name="EluNode", alpha=1.0)
    graph = helper.make_graph([node], "elu_graph", [x], [y])
    return helper.make_model(graph, opset_imports=[helper.make_operatorsetid("", 13)])


def _make_hardswish_model() -> onnx.ModelProto:
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 3])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 3])
    node = helper.make_node("HardSwish", ["x"], ["y"], name="HardSwishNode")
    graph = helper.make_graph([node], "hardswish_graph", [x], [y])
    return helper.make_model(graph, opset_imports=[helper.make_operatorsetid("", 14)])


def _make_leakyrelu_model() -> onnx.ModelProto:
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 3])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 3])
    node = helper.make_node("LeakyRelu", ["x"], ["y"], name="LeakyReluNode", alpha=0.2)
    graph = helper.make_graph([node], "leakyrelu_graph", [x], [y])
    return helper.make_model(graph, opset_imports=[helper.make_operatorsetid("", 13)])


def _make_prelu_model() -> onnx.ModelProto:
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 3])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 3])
    slope = numpy_helper.from_array(np.array([0.25], dtype=np.float32), name="slope")
    node = helper.make_node("PRelu", ["x", "slope"], ["y"], name="PReluNode")
    graph = helper.make_graph([node], "prelu_graph", [x], [y], initializer=[slope])
    return helper.make_model(graph, opset_imports=[helper.make_operatorsetid("", 13)])


def _make_gelu_model() -> onnx.ModelProto:
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 3])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 3])
    node = helper.make_node("Gelu", ["x"], ["y"], name="GeluNode")
    graph = helper.make_graph([node], "gelu_graph", [x], [y])
    return helper.make_model(graph, opset_imports=[helper.make_operatorsetid("", 20)])


def _make_pow_square_model() -> onnx.ModelProto:
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 3])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 3])
    exponent = numpy_helper.from_array(np.array([2.0], dtype=np.float32), name="pow_exp")
    node = helper.make_node("Pow", ["x", "pow_exp"], ["y"], name="PowNode")
    graph = helper.make_graph([node], "pow_graph", [x], [y], initializer=[exponent])
    return helper.make_model(graph, opset_imports=[helper.make_operatorsetid("", 13)])


def _make_pow_general_model() -> onnx.ModelProto:
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 3])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 3])
    exponent = numpy_helper.from_array(np.array([1.25], dtype=np.float32), name="pow_general_exp")
    node = helper.make_node("Pow", ["x", "pow_general_exp"], ["y"], name="PowGeneralNode")
    graph = helper.make_graph([node], "pow_general_graph", [x], [y], initializer=[exponent])
    return helper.make_model(graph, opset_imports=[helper.make_operatorsetid("", 13)])


def _make_reciprocal_model() -> onnx.ModelProto:
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 3])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 3])
    node = helper.make_node("Reciprocal", ["x"], ["y"], name="ReciprocalNode")
    graph = helper.make_graph([node], "reciprocal_graph", [x], [y])
    return helper.make_model(graph, opset_imports=[helper.make_operatorsetid("", 13)])


def _make_onehot_model() -> onnx.ModelProto:
    indices = helper.make_tensor_value_info("indices", TensorProto.INT64, [2, 3])
    output = helper.make_tensor_value_info("y", TensorProto.FLOAT, [2, 3, 4])
    depth = numpy_helper.from_array(np.array([4], dtype=np.int64), name="onehot_depth")
    values = numpy_helper.from_array(np.array([0.0, 1.0], dtype=np.float32), name="onehot_values")
    node = helper.make_node(
        "OneHot",
        ["indices", "onehot_depth", "onehot_values"],
        ["y"],
        name="OneHotNode",
        axis=-1,
    )
    graph = helper.make_graph(
        [node],
        "onehot_graph",
        [indices],
        [output],
        initializer=[depth, values],
    )
    return helper.make_model(graph, opset_imports=[helper.make_operatorsetid("", 13)])


def _make_matmul_integer_model() -> onnx.ModelProto:
    x = helper.make_tensor_value_info("x", TensorProto.UINT8, [2, 3])
    a_zero = helper.make_tensor_value_info("a_zero", TensorProto.UINT8, [])
    y = helper.make_tensor_value_info("y", TensorProto.INT32, [2, 4])
    w = numpy_helper.from_array(
        np.array(
            [
                [1, -2, 3, -4],
                [5, -6, 7, -8],
                [9, -10, 11, -12],
            ],
            dtype=np.int8,
        ),
        name="mmi_w",
    )
    b_zero = numpy_helper.from_array(np.zeros((4,), dtype=np.int8), name="mmi_b_zero")
    node = helper.make_node(
        "MatMulInteger",
        ["x", "mmi_w", "a_zero", "mmi_b_zero"],
        ["y"],
        name="MatMulIntegerNode",
    )
    graph = helper.make_graph(
        [node],
        "matmul_integer_graph",
        [x, a_zero],
        [y],
        initializer=[w, b_zero],
    )
    return helper.make_model(graph, opset_imports=[helper.make_operatorsetid("", 13)])


def _make_dynamic_quantize_linear_model() -> onnx.ModelProto:
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [2, 3])
    y = helper.make_tensor_value_info("y", TensorProto.UINT8, [2, 3])
    y_scale = helper.make_tensor_value_info("y_scale", TensorProto.FLOAT, [])
    y_zero = helper.make_tensor_value_info("y_zero", TensorProto.UINT8, [])
    node = helper.make_node(
        "DynamicQuantizeLinear",
        ["x"],
        ["y", "y_scale", "y_zero"],
        name="DynamicQuantizeLinearNode",
    )
    graph = helper.make_graph(
        [node],
        "dynamic_quantize_linear_graph",
        [x],
        [y, y_scale, y_zero],
    )
    return helper.make_model(graph, opset_imports=[helper.make_operatorsetid("", 13)])


def _make_shape_slice_model() -> onnx.ModelProto:
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [2, 3, 4, 5])
    y = helper.make_tensor_value_info("y", TensorProto.INT64, [2])
    node = helper.make_node(
        "Shape",
        ["x"],
        ["y"],
        name="ShapeSliceNode",
        start=1,
        end=3,
    )
    graph = helper.make_graph([node], "shape_slice_graph", [x], [y])
    return helper.make_model(graph, opset_imports=[helper.make_operatorsetid("", 15)])


def _make_pad_dynamic_pads_model() -> onnx.ModelProto:
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 6])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 8])
    width_index = numpy_helper.from_array(np.asarray([1], dtype=np.int64), name="pad_dyn_width_index")
    pad_limit = numpy_helper.from_array(np.asarray([8], dtype=np.int64), name="pad_dyn_limit")
    pad_zeros = numpy_helper.from_array(np.asarray([0, 0, 0], dtype=np.int64), name="pad_dyn_zeros")
    nodes = [
        helper.make_node("Shape", ["x"], ["x_shape"], name="PadDynShape"),
        helper.make_node(
            "Gather",
            ["x_shape", "pad_dyn_width_index"],
            ["x_width"],
            name="PadDynGatherWidth",
            axis=0,
        ),
        helper.make_node("Sub", ["pad_dyn_limit", "x_width"], ["pad_right"], name="PadDynPadRight"),
        helper.make_node(
            "Concat",
            ["pad_dyn_zeros", "pad_right"],
            ["pads"],
            name="PadDynConcatPads",
            axis=0,
        ),
        helper.make_node("Pad", ["x", "pads"], ["y"], name="PadDynNode"),
    ]
    graph = helper.make_graph(
        nodes,
        "pad_dynamic_pads_graph",
        [x],
        [y],
        initializer=[width_index, pad_limit, pad_zeros],
    )
    return helper.make_model(graph, opset_imports=[helper.make_operatorsetid("", 13)])


def _make_pad_reflect_model() -> onnx.ModelProto:
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 4])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 6])
    pads = numpy_helper.from_array(np.asarray([0, 1, 0, 1], dtype=np.int64), name="pad_reflect_pads")
    node = helper.make_node(
        "Pad",
        ["x", "pad_reflect_pads"],
        ["y"],
        name="PadReflectNode",
        mode="reflect",
    )
    graph = helper.make_graph(
        [node],
        "pad_reflect_graph",
        [x],
        [y],
        initializer=[pads],
    )
    return helper.make_model(graph, opset_imports=[helper.make_operatorsetid("", 13)])


def _make_pad_constant_nonzero_model() -> onnx.ModelProto:
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT16, [1, 4])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT16, [1, 6])
    pads = numpy_helper.from_array(
        np.asarray([0, 1, 0, 1], dtype=np.int64),
        name="pad_nonzero_pads",
    )
    constant_value = numpy_helper.from_array(
        np.asarray([-65504.0], dtype=np.float16),
        name="pad_nonzero_value",
    )
    node = helper.make_node(
        "Pad",
        ["x", "pad_nonzero_pads", "pad_nonzero_value"],
        ["y"],
        name="PadConstantNonZeroNode",
        mode="constant",
    )
    graph = helper.make_graph(
        [node],
        "pad_constant_nonzero_graph",
        [x],
        [y],
        initializer=[pads, constant_value],
    )
    return helper.make_model(graph, opset_imports=[helper.make_operatorsetid("", 13)])


def _make_batchnorm_rank3_model() -> onnx.ModelProto:
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 1536, 351])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 1536, 351])
    scale = numpy_helper.from_array(np.ones((1536,), dtype=np.float32), name="bn_rank3_scale")
    bias = numpy_helper.from_array(np.zeros((1536,), dtype=np.float32), name="bn_rank3_bias")
    mean = numpy_helper.from_array(np.zeros((1536,), dtype=np.float32), name="bn_rank3_mean")
    var = numpy_helper.from_array(np.ones((1536,), dtype=np.float32), name="bn_rank3_var")
    bn = helper.make_node(
        "BatchNormalization",
        ["x", "bn_rank3_scale", "bn_rank3_bias", "bn_rank3_mean", "bn_rank3_var"],
        ["y"],
        name="BNRank3",
        epsilon=1e-5,
    )
    graph = helper.make_graph(
        [bn],
        "batchnorm_rank3_graph",
        [x],
        [y],
        initializer=[scale, bias, mean, var],
    )
    return helper.make_model(graph, opset_imports=[helper.make_operatorsetid("", 13)])


def _make_min_topk_dynamic_k_model() -> onnx.ModelProto:
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 10])
    values = helper.make_tensor_value_info("values", TensorProto.FLOAT, [1, "K"])
    indices = helper.make_tensor_value_info("indices", TensorProto.INT64, [1, "K"])
    axis_index = numpy_helper.from_array(np.asarray([1], dtype=np.int64), name="topk_axis_index")
    k_limit = numpy_helper.from_array(np.asarray([5], dtype=np.int64), name="topk_k_limit")
    nodes = [
        helper.make_node("Shape", ["x"], ["x_shape"], name="MinTopKShape"),
        helper.make_node(
            "Gather",
            ["x_shape", "topk_axis_index"],
            ["x_width"],
            name="MinTopKGatherWidth",
            axis=0,
        ),
        helper.make_node("Min", ["topk_k_limit", "x_width"], ["k"], name="MinTopKMin"),
        helper.make_node("TopK", ["x", "k"], ["values", "indices"], name="MinTopKTopK", axis=1),
    ]
    graph = helper.make_graph(
        nodes,
        "min_topk_dynamic_k_graph",
        [x],
        [values, indices],
        initializer=[axis_index, k_limit],
    )
    return helper.make_model(graph, opset_imports=[helper.make_operatorsetid("", 13)])


def _make_legacy_slice_attr_model() -> onnx.ModelProto:
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 3, 4, 4])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 3, 4, 4])
    nodes = [
        helper.make_node(
            "Slice",
            ["x"],
            ["y0"],
            name="LegacySlice0",
            starts=[0],
            ends=[2],
            axes=[2],
        ),
        helper.make_node(
            "Slice",
            ["x"],
            ["y1"],
            name="LegacySlice1",
            starts=[2],
            ends=[9223372036854775807],
            axes=[2],
        ),
        helper.make_node("Concat", ["y0", "y1"], ["y"], name="LegacySliceConcat", axis=2),
    ]
    graph = helper.make_graph(nodes, "legacy_slice_attr_graph", [x], [y])
    return helper.make_model(graph, opset_imports=[helper.make_operatorsetid("", 9)])


def _make_slice_dynamic_end_prefix_rank2_model() -> onnx.ModelProto:
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, "N"])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, "N"])
    starts = numpy_helper.from_array(np.asarray([0, 0], dtype=np.int64), name="slice_dyn_starts")
    nodes = [
        helper.make_node("Shape", ["x"], ["x_shape"], name="SliceDynShape"),
        helper.make_node("Slice", ["x", "slice_dyn_starts", "x_shape"], ["y"], name="SliceDynNode"),
    ]
    graph = helper.make_graph(
        nodes,
        "slice_dynamic_end_prefix_rank2_graph",
        [x],
        [y],
        initializer=[starts],
    )
    return helper.make_model(graph, opset_imports=[helper.make_operatorsetid("", 13)])


def _make_expand_dynamic_shape_model() -> onnx.ModelProto:
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, ["N"])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, ["M"])
    axis0 = numpy_helper.from_array(np.asarray([0], dtype=np.int64), name="expand_dyn_axis0")
    max_count = numpy_helper.from_array(np.asarray(5, dtype=np.int64), name="expand_dyn_max_count")
    minus_one = numpy_helper.from_array(np.asarray([-1], dtype=np.int64), name="expand_dyn_minus_one")
    zero = numpy_helper.from_array(np.asarray(0.0, dtype=np.float32), name="expand_dyn_zero")
    nodes = [
        helper.make_node("Shape", ["x"], ["x_shape"], name="ExpandDynShape"),
        helper.make_node("Gather", ["x_shape", "expand_dyn_axis0"], ["x_count"], name="ExpandDynCount", axis=0),
        helper.make_node("Sub", ["expand_dyn_max_count", "x_count"], ["pad_count"], name="ExpandDynSub"),
        helper.make_node("Reshape", ["pad_count", "expand_dyn_minus_one"], ["expand_shape"], name="ExpandDynReshape"),
        helper.make_node("Expand", ["expand_dyn_zero", "expand_shape"], ["y"], name="ExpandDynNode"),
    ]
    graph = helper.make_graph(
        nodes,
        "expand_dynamic_shape_graph",
        [x],
        [y],
        initializer=[axis0, max_count, minus_one, zero],
    )
    return helper.make_model(graph, opset_imports=[helper.make_operatorsetid("", 13)])


def _make_unsqueeze_scalar_multi_axis_model() -> onnx.ModelProto:
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [2])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 1])
    reduce_axes = numpy_helper.from_array(np.asarray([0], dtype=np.int64), name="unsq_scalar_reduce_axes")
    unsq_axes = numpy_helper.from_array(np.asarray([0, 1], dtype=np.int64), name="unsq_scalar_axes")
    nodes = [
        helper.make_node("ReduceSum", ["x", "unsq_scalar_reduce_axes"], ["s"], name="UnsqScalarReduce", keepdims=0),
        helper.make_node("Unsqueeze", ["s", "unsq_scalar_axes"], ["y"], name="UnsqScalarNode"),
    ]
    graph = helper.make_graph(
        nodes,
        "unsqueeze_scalar_multi_axis_graph",
        [x],
        [y],
        initializer=[reduce_axes, unsq_axes],
    )
    return helper.make_model(graph, opset_imports=[helper.make_operatorsetid("", 13)])


def _make_constant_of_shape_model() -> onnx.ModelProto:
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [2, 3])
    shape = numpy_helper.from_array(np.array([2, 3], dtype=np.int64), name="cos_shape")
    value = numpy_helper.from_array(np.array([0.25], dtype=np.float32), name="cos_value")
    node = helper.make_node(
        "ConstantOfShape",
        ["cos_shape"],
        ["y"],
        name="ConstantOfShapeNode",
        value=value,
    )
    graph = helper.make_graph(
        [node],
        "constant_of_shape_graph",
        [],
        [y],
        initializer=[shape],
    )
    return helper.make_model(graph, opset_imports=[helper.make_operatorsetid("", 13)])


def _make_fused_matmul_model() -> onnx.ModelProto:
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [2, 3])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [2, 4])
    w = numpy_helper.from_array(
        np.array(
            [
                [1.0, 2.0, 3.0],
                [4.0, 5.0, 6.0],
                [7.0, 8.0, 9.0],
                [10.0, 11.0, 12.0],
            ],
            dtype=np.float32,
        ),
        name="fmm_w",
    )
    node = helper.make_node(
        "FusedMatMul",
        ["x", "fmm_w"],
        ["y"],
        name="FusedMatMulNode",
        alpha=0.125,
        transA=0,
        transB=1,
    )
    graph = helper.make_graph([node], "fused_matmul_graph", [x], [y], initializer=[w])
    return helper.make_model(
        graph,
        opset_imports=[
            helper.make_operatorsetid("", 13),
            helper.make_operatorsetid("com.microsoft", 1),
        ],
    )


def _make_matmul_rank4_model() -> onnx.ModelProto:
    x0 = helper.make_tensor_value_info("x0", TensorProto.FLOAT, [1, 3, 8, 8])
    x1 = helper.make_tensor_value_info("x1", TensorProto.FLOAT, [1, 3, 8, 8])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 3, 8, 8])
    node = helper.make_node("MatMul", ["x0", "x1"], ["y"], name="MatMulRank4Node")
    graph = helper.make_graph([node], "matmul_rank4_graph", [x0, x1], [y])
    return helper.make_model(graph, opset_imports=[helper.make_operatorsetid("", 13)])


def _make_space_to_depth_model() -> onnx.ModelProto:
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 2, 4, 4])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 8, 2, 2])
    node = helper.make_node(
        "SpaceToDepth",
        ["x"],
        ["y"],
        name="SpaceToDepthNode",
        blocksize=2,
        mode="DCR",
    )
    graph = helper.make_graph([node], "space_to_depth_graph", [x], [y])
    return helper.make_model(graph, opset_imports=[helper.make_operatorsetid("", 13)])


def _make_depth_to_space_model() -> onnx.ModelProto:
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 8, 2, 2])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 2, 4, 4])
    node = helper.make_node(
        "DepthToSpace",
        ["x"],
        ["y"],
        name="DepthToSpaceNode",
        blocksize=2,
        mode="DCR",
    )
    graph = helper.make_graph([node], "depth_to_space_graph", [x], [y])
    return helper.make_model(graph, opset_imports=[helper.make_operatorsetid("", 13)])


def _make_depth_to_space_crd_model() -> onnx.ModelProto:
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 48, 64, 64])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 3, 256, 256])
    node = helper.make_node(
        "DepthToSpace",
        ["x"],
        ["y"],
        name="DepthToSpaceCRDNode",
        blocksize=4,
        mode="CRD",
    )
    graph = helper.make_graph([node], "depth_to_space_crd_graph", [x], [y])
    return helper.make_model(graph, opset_imports=[helper.make_operatorsetid("", 13)])


def _make_resize_nearest_model() -> onnx.ModelProto:
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 3, 4, 4])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 3, 8, 8])
    roi = numpy_helper.from_array(np.asarray([], dtype=np.float32), name="roi_empty")
    scales = numpy_helper.from_array(np.asarray([1.0, 1.0, 2.0, 2.0], dtype=np.float32), name="resize_scales")
    node = helper.make_node(
        "Resize",
        ["x", "roi_empty", "resize_scales"],
        ["y"],
        name="ResizeNearestNode",
        mode="nearest",
        coordinate_transformation_mode="asymmetric",
        nearest_mode="floor",
    )
    graph = helper.make_graph([node], "resize_nearest_graph", [x], [y], initializer=[roi, scales])
    return helper.make_model(graph, opset_imports=[helper.make_operatorsetid("", 13)])


def _make_resize_cubic_model(cubic_coeff_a: float = -0.75) -> onnx.ModelProto:
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 3, 4, 4])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 3, 8, 8])
    roi = numpy_helper.from_array(np.asarray([], dtype=np.float32), name="resize_cubic_roi_empty")
    scales = numpy_helper.from_array(
        np.asarray([1.0, 1.0, 2.0, 2.0], dtype=np.float32),
        name="resize_cubic_scales",
    )
    node = helper.make_node(
        "Resize",
        ["x", "resize_cubic_roi_empty", "resize_cubic_scales"],
        ["y"],
        name="ResizeCubicNode",
        mode="cubic",
        coordinate_transformation_mode="align_corners",
        cubic_coeff_a=float(cubic_coeff_a),
        nearest_mode="floor",
    )
    graph = helper.make_graph([node], "resize_cubic_graph", [x], [y], initializer=[roi, scales])
    return helper.make_model(graph, opset_imports=[helper.make_operatorsetid("", 13)])


def _make_resize_cubic_batch2_model() -> onnx.ModelProto:
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [2, 3, 4, 4])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [2, 3, 8, 8])
    roi = numpy_helper.from_array(np.asarray([], dtype=np.float32), name="resize_cubic_b2_roi_empty")
    scales = numpy_helper.from_array(
        np.asarray([1.0, 1.0, 2.0, 2.0], dtype=np.float32),
        name="resize_cubic_b2_scales",
    )
    node = helper.make_node(
        "Resize",
        ["x", "resize_cubic_b2_roi_empty", "resize_cubic_b2_scales"],
        ["y"],
        name="ResizeCubicBatch2Node",
        mode="cubic",
        coordinate_transformation_mode="align_corners",
        nearest_mode="floor",
    )
    graph = helper.make_graph([node], "resize_cubic_batch2_graph", [x], [y], initializer=[roi, scales])
    return helper.make_model(graph, opset_imports=[helper.make_operatorsetid("", 13)])


def _make_resize_dynamic_sizes_model() -> onnx.ModelProto:
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, ["N", 3, "H", "W"])
    ref = helper.make_tensor_value_info("ref", TensorProto.FLOAT, ["N", 3, "RH", "RW"])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, ["N", 3, "RH", "RW"])
    roi = numpy_helper.from_array(np.asarray([], dtype=np.float32), name="resize_dyn_roi_empty")
    scales = numpy_helper.from_array(np.asarray([], dtype=np.float32), name="resize_dyn_scales_empty")
    nodes = [
        helper.make_node("Shape", ["ref"], ["resize_dyn_sizes"], name="ResizeDynShapeRef"),
        helper.make_node(
            "Resize",
            ["x", "resize_dyn_roi_empty", "resize_dyn_scales_empty", "resize_dyn_sizes"],
            ["y"],
            name="ResizeDynSizesNode",
            mode="linear",
            coordinate_transformation_mode="pytorch_half_pixel",
            nearest_mode="floor",
        ),
    ]
    graph = helper.make_graph(
        nodes,
        "resize_dynamic_sizes_graph",
        [x, ref],
        [y],
        initializer=[roi, scales],
    )
    return helper.make_model(graph, opset_imports=[helper.make_operatorsetid("", 13)])


def _make_qlinear_sigmoid_model() -> onnx.ModelProto:
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 1, 2, 2])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 1, 2, 2])
    x_scale = numpy_helper.from_array(np.asarray([0.25], dtype=np.float32), name="qsig_x_scale")
    x_zero = numpy_helper.from_array(np.asarray([0], dtype=np.int8), name="qsig_x_zero")
    y_scale = numpy_helper.from_array(np.asarray([1.0 / 256.0], dtype=np.float32), name="qsig_y_scale")
    y_zero = numpy_helper.from_array(np.asarray([-128], dtype=np.int8), name="qsig_y_zero")
    nodes = [
        helper.make_node("QuantizeLinear", ["x", "qsig_x_scale", "qsig_x_zero"], ["x_q"], name="QSigQ0"),
        helper.make_node(
            "QLinearSigmoid",
            ["x_q", "qsig_x_scale", "qsig_x_zero", "qsig_y_scale", "qsig_y_zero"],
            ["y_q"],
            name="QSigNode",
        ),
        helper.make_node("DequantizeLinear", ["y_q", "qsig_y_scale", "qsig_y_zero"], ["y"], name="QSigDQ0"),
    ]
    graph = helper.make_graph(
        nodes,
        "qlinear_sigmoid_graph",
        [x],
        [y],
        initializer=[x_scale, x_zero, y_scale, y_zero],
    )
    return helper.make_model(graph, opset_imports=[helper.make_operatorsetid("", 13)])


def _make_qlinear_global_average_pool_model() -> onnx.ModelProto:
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 2, 4, 4])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 2, 1, 1])
    x_scale = numpy_helper.from_array(np.asarray([0.25], dtype=np.float32), name="qgap_x_scale")
    x_zero = numpy_helper.from_array(np.asarray([0], dtype=np.int8), name="qgap_x_zero")
    y_scale = numpy_helper.from_array(np.asarray([0.125], dtype=np.float32), name="qgap_y_scale")
    y_zero = numpy_helper.from_array(np.asarray([0], dtype=np.int8), name="qgap_y_zero")
    nodes = [
        helper.make_node("QuantizeLinear", ["x", "qgap_x_scale", "qgap_x_zero"], ["x_q"], name="QGapQ0"),
        helper.make_node(
            "QLinearGlobalAveragePool",
            ["x_q", "qgap_x_scale", "qgap_x_zero", "qgap_y_scale", "qgap_y_zero"],
            ["y_q"],
            name="QGapNode",
            channels_last=0,
        ),
        helper.make_node("DequantizeLinear", ["y_q", "qgap_y_scale", "qgap_y_zero"], ["y"], name="QGapDQ0"),
    ]
    graph = helper.make_graph(
        nodes,
        "qlinear_global_average_pool_graph",
        [x],
        [y],
        initializer=[x_scale, x_zero, y_scale, y_zero],
    )
    return helper.make_model(graph, opset_imports=[helper.make_operatorsetid("", 13)])


def _make_qlinear_global_average_pool_dynamic_batch_model() -> onnx.ModelProto:
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, ["N", 2, 4, 4])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, ["N", 2, 1, 1])
    x_scale = numpy_helper.from_array(np.asarray([0.25], dtype=np.float32), name="qgap_dyn_x_scale")
    x_zero = numpy_helper.from_array(np.asarray([0], dtype=np.int8), name="qgap_dyn_x_zero")
    y_scale = numpy_helper.from_array(np.asarray([0.125], dtype=np.float32), name="qgap_dyn_y_scale")
    y_zero = numpy_helper.from_array(np.asarray([0], dtype=np.int8), name="qgap_dyn_y_zero")
    nodes = [
        helper.make_node(
            "QuantizeLinear",
            ["x", "qgap_dyn_x_scale", "qgap_dyn_x_zero"],
            ["x_q"],
            name="QGapDynQ0",
        ),
        helper.make_node(
            "QLinearGlobalAveragePool",
            ["x_q", "qgap_dyn_x_scale", "qgap_dyn_x_zero", "qgap_dyn_y_scale", "qgap_dyn_y_zero"],
            ["y_q"],
            name="QGapDynNode",
            channels_last=0,
        ),
        helper.make_node(
            "DequantizeLinear",
            ["y_q", "qgap_dyn_y_scale", "qgap_dyn_y_zero"],
            ["y"],
            name="QGapDynDQ0",
        ),
    ]
    graph = helper.make_graph(
        nodes,
        "qlinear_global_average_pool_dynamic_batch_graph",
        [x],
        [y],
        initializer=[x_scale, x_zero, y_scale, y_zero],
    )
    return helper.make_model(graph, opset_imports=[helper.make_operatorsetid("", 13)])


def _make_transpose_qlinear_global_average_pool_model() -> onnx.ModelProto:
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 4, 4, 2])
    y = helper.make_tensor_value_info("y_q", TensorProto.INT8, [1, 2, 1, 1])
    x_scale = numpy_helper.from_array(np.asarray([0.25], dtype=np.float32), name="qgap_t_x_scale")
    x_zero = numpy_helper.from_array(np.asarray([0], dtype=np.int8), name="qgap_t_x_zero")
    y_scale = numpy_helper.from_array(np.asarray([0.125], dtype=np.float32), name="qgap_t_y_scale")
    y_zero = numpy_helper.from_array(np.asarray([0], dtype=np.int8), name="qgap_t_y_zero")
    t_perm = numpy_helper.from_array(np.asarray([0, 3, 1, 2], dtype=np.int64), name="qgap_t_perm")
    nodes = [
        helper.make_node(
            "QuantizeLinear",
            ["x", "qgap_t_x_scale", "qgap_t_x_zero"],
            ["x_q"],
            name="QGapTQ0",
        ),
        helper.make_node(
            "Transpose",
            ["x_q", "qgap_t_perm"],
            ["x_q_nchw"],
            name="QGapTTranspose0",
        ),
        helper.make_node(
            "QLinearGlobalAveragePool",
            ["x_q_nchw", "qgap_t_x_scale", "qgap_t_x_zero", "qgap_t_y_scale", "qgap_t_y_zero"],
            ["y_q"],
            name="QGapTNode",
            channels_last=0,
        ),
    ]
    graph = helper.make_graph(
        nodes,
        "transpose_qlinear_global_average_pool_graph",
        [x],
        [y],
        initializer=[x_scale, x_zero, y_scale, y_zero, t_perm],
    )
    return helper.make_model(graph, opset_imports=[helper.make_operatorsetid("", 13)])


def _make_qlinear_concat_model() -> onnx.ModelProto:
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 1, 2, 2])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 2, 2, 2])
    x_scale = numpy_helper.from_array(np.asarray([0.25], dtype=np.float32), name="qcat_x_scale")
    x_zero = numpy_helper.from_array(np.asarray([0], dtype=np.int8), name="qcat_x_zero")
    y_scale = numpy_helper.from_array(np.asarray([0.25], dtype=np.float32), name="qcat_y_scale")
    y_zero = numpy_helper.from_array(np.asarray([0], dtype=np.int8), name="qcat_y_zero")
    nodes = [
        helper.make_node("QuantizeLinear", ["x", "qcat_x_scale", "qcat_x_zero"], ["x_q"], name="QCatQ0"),
        helper.make_node(
            "QLinearConcat",
            [
                "qcat_y_scale",
                "qcat_y_zero",
                "x_q",
                "qcat_x_scale",
                "qcat_x_zero",
                "x_q",
                "qcat_x_scale",
                "qcat_x_zero",
            ],
            ["y_q"],
            name="QCatNode",
            axis=1,
        ),
        helper.make_node("DequantizeLinear", ["y_q", "qcat_y_scale", "qcat_y_zero"], ["y"], name="QCatDQ0"),
    ]
    graph = helper.make_graph(
        nodes,
        "qlinear_concat_graph",
        [x],
        [y],
        initializer=[x_scale, x_zero, y_scale, y_zero],
    )
    return helper.make_model(graph, opset_imports=[helper.make_operatorsetid("", 13)])


def _make_qlinear_concat_conv_layout_chain_model() -> onnx.ModelProto:
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 2, 4, 4])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 3, 1, 1])

    in_scale = numpy_helper.from_array(np.asarray([0.1], dtype=np.float32), name="qlcc_in_scale")
    in_zero = numpy_helper.from_array(np.asarray([0], dtype=np.int8), name="qlcc_in_zero")
    gap_scale = numpy_helper.from_array(np.asarray([0.2], dtype=np.float32), name="qlcc_gap_scale")
    gap_zero = numpy_helper.from_array(np.asarray([0], dtype=np.int8), name="qlcc_gap_zero")
    cat_scale = numpy_helper.from_array(np.asarray([0.15], dtype=np.float32), name="qlcc_cat_scale")
    cat_zero = numpy_helper.from_array(np.asarray([0], dtype=np.int8), name="qlcc_cat_zero")
    conv_w = numpy_helper.from_array(
        np.asarray(
            [
                [[[1]], [[1]], [[1]], [[1]]],
                [[[1]], [[0]], [[-1]], [[1]]],
                [[[0]], [[1]], [[1]], [[0]]],
            ],
            dtype=np.int8,
        ),
        name="qlcc_conv_w",
    )
    conv_w_scale = numpy_helper.from_array(np.asarray([0.25], dtype=np.float32), name="qlcc_conv_w_scale")
    conv_w_zero = numpy_helper.from_array(np.asarray([0], dtype=np.int8), name="qlcc_conv_w_zero")
    conv_y_scale = numpy_helper.from_array(np.asarray([0.12], dtype=np.float32), name="qlcc_conv_y_scale")
    conv_y_zero = numpy_helper.from_array(np.asarray([0], dtype=np.int8), name="qlcc_conv_y_zero")

    nodes = [
        helper.make_node(
            "QuantizeLinear",
            ["x", "qlcc_in_scale", "qlcc_in_zero"],
            ["x_q"],
            name="QLCatConv_Q0",
        ),
        helper.make_node(
            "QLinearGlobalAveragePool",
            ["x_q", "qlcc_in_scale", "qlcc_in_zero", "qlcc_gap_scale", "qlcc_gap_zero"],
            ["gap_q"],
            name="QLCatConv_GAP",
            channels_last=0,
        ),
        helper.make_node(
            "DequantizeLinear",
            ["x_q", "qlcc_in_scale", "qlcc_in_zero"],
            ["x_f"],
            name="QLCatConv_DQ0",
        ),
        helper.make_node(
            "MaxPool",
            ["x_f"],
            ["pool_f"],
            name="QLCatConv_MaxPool",
            kernel_shape=[4, 4],
            strides=[4, 4],
            pads=[0, 0, 0, 0],
        ),
        helper.make_node(
            "QuantizeLinear",
            ["pool_f", "qlcc_in_scale", "qlcc_in_zero"],
            ["pool_q"],
            name="QLCatConv_QPool",
        ),
        helper.make_node(
            "QLinearConcat",
            [
                "qlcc_cat_scale",
                "qlcc_cat_zero",
                "gap_q",
                "qlcc_gap_scale",
                "qlcc_gap_zero",
                "pool_q",
                "qlcc_in_scale",
                "qlcc_in_zero",
            ],
            ["cat_q"],
            name="QLCatConv_Concat",
            axis=1,
        ),
        helper.make_node(
            "QLinearConv",
            [
                "cat_q",
                "qlcc_cat_scale",
                "qlcc_cat_zero",
                "qlcc_conv_w",
                "qlcc_conv_w_scale",
                "qlcc_conv_w_zero",
                "qlcc_conv_y_scale",
                "qlcc_conv_y_zero",
            ],
            ["y_q"],
            name="QLCatConv_Conv",
            kernel_shape=[1, 1],
            pads=[0, 0, 0, 0],
            strides=[1, 1],
            group=1,
        ),
        helper.make_node(
            "DequantizeLinear",
            ["y_q", "qlcc_conv_y_scale", "qlcc_conv_y_zero"],
            ["y"],
            name="QLCatConv_DQ1",
        ),
    ]
    graph = helper.make_graph(
        nodes,
        "qlinear_concat_conv_layout_chain_graph",
        [x],
        [y],
        initializer=[
            in_scale,
            in_zero,
            gap_scale,
            gap_zero,
            cat_scale,
            cat_zero,
            conv_w,
            conv_w_scale,
            conv_w_zero,
            conv_y_scale,
            conv_y_zero,
        ],
    )
    return helper.make_model(graph, opset_imports=[helper.make_operatorsetid("", 13)])


def _make_qlinear_conv_chain_model() -> onnx.ModelProto:
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 1, 2, 2])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 1, 2, 2])

    x_scale = numpy_helper.from_array(np.asarray([0.1], dtype=np.float32), name="x_scale")
    x_zero = numpy_helper.from_array(np.asarray([0], dtype=np.int8), name="x_zero")
    mul_const = numpy_helper.from_array(np.asarray([1], dtype=np.int8), name="mul_const")
    mul_scale = numpy_helper.from_array(np.asarray([0.1], dtype=np.float32), name="mul_scale")
    mul_zero = numpy_helper.from_array(np.asarray([0], dtype=np.int8), name="mul_zero")
    mul_out_scale = numpy_helper.from_array(np.asarray([0.1], dtype=np.float32), name="mul_out_scale")
    conv_w = numpy_helper.from_array(np.asarray([[[[1]]]], dtype=np.int8), name="conv_w")
    conv_w_scale = numpy_helper.from_array(np.asarray([0.2], dtype=np.float32), name="conv_w_scale")
    conv_w_zero = numpy_helper.from_array(np.asarray([0], dtype=np.int8), name="conv_w_zero")
    conv_out_scale = numpy_helper.from_array(np.asarray([0.05], dtype=np.float32), name="conv_out_scale")
    conv_out_zero = numpy_helper.from_array(np.asarray([0], dtype=np.int8), name="conv_out_zero")
    bn_scale = numpy_helper.from_array(np.asarray([1.1], dtype=np.float32), name="bn_scale")
    bn_bias = numpy_helper.from_array(np.asarray([0.0], dtype=np.float32), name="bn_bias")
    bn_mean = numpy_helper.from_array(np.asarray([0.0], dtype=np.float32), name="bn_mean")
    bn_var = numpy_helper.from_array(np.asarray([1.0], dtype=np.float32), name="bn_var")
    prelu_slope = numpy_helper.from_array(np.asarray([0.2], dtype=np.float32), name="prelu_slope")
    q2_scale = numpy_helper.from_array(np.asarray([0.05], dtype=np.float32), name="q2_scale")
    q2_zero = numpy_helper.from_array(np.asarray([0], dtype=np.int8), name="q2_zero")

    nodes = [
        helper.make_node("QuantizeLinear", ["x", "x_scale", "x_zero"], ["x_q"], name="Q0"),
        helper.make_node(
            "QLinearMul",
            ["x_q", "x_scale", "x_zero", "mul_const", "mul_scale", "mul_zero", "mul_out_scale", "mul_zero"],
            ["x_mul_q"],
            name="QMul0",
        ),
        helper.make_node(
            "QLinearConv",
            [
                "x_mul_q",
                "mul_out_scale",
                "mul_zero",
                "conv_w",
                "conv_w_scale",
                "conv_w_zero",
                "conv_out_scale",
                "conv_out_zero",
            ],
            ["conv_q"],
            name="QConv0",
            kernel_shape=[1, 1],
            pads=[0, 0, 0, 0],
            strides=[1, 1],
            group=1,
        ),
        helper.make_node(
            "DequantizeLinear",
            ["conv_q", "conv_out_scale", "conv_out_zero"],
            ["conv_f"],
            name="DQ0",
        ),
        helper.make_node(
            "BatchNormalization",
            ["conv_f", "bn_scale", "bn_bias", "bn_mean", "bn_var"],
            ["bn_out"],
            name="BN0",
            epsilon=1e-5,
        ),
        helper.make_node("PRelu", ["bn_out", "prelu_slope"], ["prelu_out"], name="PRelu0"),
        helper.make_node("QuantizeLinear", ["prelu_out", "q2_scale", "q2_zero"], ["q2"], name="Q1"),
        helper.make_node("DequantizeLinear", ["q2", "q2_scale", "q2_zero"], ["y"], name="DQ1"),
    ]
    graph = helper.make_graph(
        nodes,
        "qlinear_conv_chain_graph",
        [x],
        [y],
        initializer=[
            x_scale,
            x_zero,
            mul_const,
            mul_scale,
            mul_zero,
            mul_out_scale,
            conv_w,
            conv_w_scale,
            conv_w_zero,
            conv_out_scale,
            conv_out_zero,
            bn_scale,
            bn_bias,
            bn_mean,
            bn_var,
            prelu_slope,
            q2_scale,
            q2_zero,
        ],
    )
    return helper.make_model(graph, opset_imports=[helper.make_operatorsetid("", 13)])


def _make_qlinear_conv_pair_model() -> onnx.ModelProto:
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 1, 2, 2])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 1, 2, 2])

    x_scale = numpy_helper.from_array(np.asarray([0.1], dtype=np.float32), name="pair_x_scale")
    x_zero = numpy_helper.from_array(np.asarray([0], dtype=np.int8), name="pair_x_zero")
    w1 = numpy_helper.from_array(np.asarray([[[[1]]]], dtype=np.int8), name="pair_w1")
    w1_scale = numpy_helper.from_array(np.asarray([0.2], dtype=np.float32), name="pair_w1_scale")
    w1_zero = numpy_helper.from_array(np.asarray([0], dtype=np.int8), name="pair_w1_zero")
    y1_scale = numpy_helper.from_array(np.asarray([0.05], dtype=np.float32), name="pair_y1_scale")
    y1_zero = numpy_helper.from_array(np.asarray([0], dtype=np.int8), name="pair_y1_zero")
    w2 = numpy_helper.from_array(np.asarray([[[[1]]]], dtype=np.int8), name="pair_w2")
    w2_scale = numpy_helper.from_array(np.asarray([0.3], dtype=np.float32), name="pair_w2_scale")
    w2_zero = numpy_helper.from_array(np.asarray([0], dtype=np.int8), name="pair_w2_zero")
    y2_scale = numpy_helper.from_array(np.asarray([0.04], dtype=np.float32), name="pair_y2_scale")
    y2_zero = numpy_helper.from_array(np.asarray([0], dtype=np.int8), name="pair_y2_zero")

    nodes = [
        helper.make_node("QuantizeLinear", ["x", "pair_x_scale", "pair_x_zero"], ["x_q"], name="PairQ0"),
        helper.make_node(
            "QLinearConv",
            [
                "x_q",
                "pair_x_scale",
                "pair_x_zero",
                "pair_w1",
                "pair_w1_scale",
                "pair_w1_zero",
                "pair_y1_scale",
                "pair_y1_zero",
            ],
            ["q1"],
            name="PairQConv1",
            kernel_shape=[1, 1],
            pads=[0, 0, 0, 0],
            strides=[1, 1],
            group=1,
        ),
        helper.make_node(
            "QLinearConv",
            [
                "q1",
                "pair_y1_scale",
                "pair_y1_zero",
                "pair_w2",
                "pair_w2_scale",
                "pair_w2_zero",
                "pair_y2_scale",
                "pair_y2_zero",
            ],
            ["q2"],
            name="PairQConv2",
            kernel_shape=[1, 1],
            pads=[0, 0, 0, 0],
            strides=[1, 1],
            group=1,
        ),
        helper.make_node("DequantizeLinear", ["q2", "pair_y2_scale", "pair_y2_zero"], ["y"], name="PairDQ"),
    ]
    graph = helper.make_graph(
        nodes,
        "qlinear_conv_pair_graph",
        [x],
        [y],
        initializer=[
            x_scale,
            x_zero,
            w1,
            w1_scale,
            w1_zero,
            y1_scale,
            y1_zero,
            w2,
            w2_scale,
            w2_zero,
            y2_scale,
            y2_zero,
        ],
    )
    return helper.make_model(graph, opset_imports=[helper.make_operatorsetid("", 13)])


def _make_qlinear_conv_multichannel_model() -> onnx.ModelProto:
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 3, 2, 2])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 4, 2, 2])

    x_scale = numpy_helper.from_array(np.asarray([0.1], dtype=np.float32), name="mc_x_scale")
    x_zero = numpy_helper.from_array(np.asarray([0], dtype=np.int8), name="mc_x_zero")
    # ONNX QLinearConv weights: OIHW
    w = numpy_helper.from_array(
        np.asarray(
            [
                [[[1]], [[2]], [[3]]],
                [[[1]], [[0]], [[-1]]],
                [[[2]], [[1]], [[0]]],
                [[[0]], [[1]], [[2]]],
            ],
            dtype=np.int8,
        ),
        name="mc_w",
    )
    w_scale = numpy_helper.from_array(np.asarray([0.2], dtype=np.float32), name="mc_w_scale")
    w_zero = numpy_helper.from_array(np.asarray([0], dtype=np.int8), name="mc_w_zero")
    y_scale = numpy_helper.from_array(np.asarray([0.05], dtype=np.float32), name="mc_y_scale")
    y_zero = numpy_helper.from_array(np.asarray([0], dtype=np.int8), name="mc_y_zero")

    nodes = [
        helper.make_node("QuantizeLinear", ["x", "mc_x_scale", "mc_x_zero"], ["x_q"], name="QMC0"),
        helper.make_node(
            "QLinearConv",
            [
                "x_q",
                "mc_x_scale",
                "mc_x_zero",
                "mc_w",
                "mc_w_scale",
                "mc_w_zero",
                "mc_y_scale",
                "mc_y_zero",
            ],
            ["y_q"],
            name="QConvMC",
            kernel_shape=[1, 1],
            pads=[0, 0, 0, 0],
            strides=[1, 1],
            group=1,
        ),
        helper.make_node("DequantizeLinear", ["y_q", "mc_y_scale", "mc_y_zero"], ["y"], name="DQMC0"),
    ]
    graph = helper.make_graph(
        nodes,
        "qlinear_conv_multichannel_graph",
        [x],
        [y],
        initializer=[x_scale, x_zero, w, w_scale, w_zero, y_scale, y_zero],
    )
    return helper.make_model(graph, opset_imports=[helper.make_operatorsetid("", 13)])


def _make_qlinear_conv_dynamic_batch_model() -> onnx.ModelProto:
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, ["N", 1, 2, 2])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, ["N", 1, 2, 2])

    x_scale = numpy_helper.from_array(np.asarray([0.1], dtype=np.float32), name="dyn_x_scale")
    x_zero = numpy_helper.from_array(np.asarray([0], dtype=np.int8), name="dyn_x_zero")
    w = numpy_helper.from_array(np.asarray([[[[1]]]], dtype=np.int8), name="dyn_w")
    w_scale = numpy_helper.from_array(np.asarray([0.2], dtype=np.float32), name="dyn_w_scale")
    w_zero = numpy_helper.from_array(np.asarray([0], dtype=np.int8), name="dyn_w_zero")
    y_scale = numpy_helper.from_array(np.asarray([0.05], dtype=np.float32), name="dyn_y_scale")
    y_zero = numpy_helper.from_array(np.asarray([0], dtype=np.int8), name="dyn_y_zero")

    nodes = [
        helper.make_node("QuantizeLinear", ["x", "dyn_x_scale", "dyn_x_zero"], ["x_q"], name="QDyn0"),
        helper.make_node(
            "QLinearConv",
            [
                "x_q",
                "dyn_x_scale",
                "dyn_x_zero",
                "dyn_w",
                "dyn_w_scale",
                "dyn_w_zero",
                "dyn_y_scale",
                "dyn_y_zero",
            ],
            ["y_q"],
            name="QConvDyn",
            kernel_shape=[1, 1],
            pads=[0, 0, 0, 0],
            strides=[1, 1],
            group=1,
        ),
        helper.make_node("DequantizeLinear", ["y_q", "dyn_y_scale", "dyn_y_zero"], ["y"], name="DQDyn0"),
    ]
    graph = helper.make_graph(
        nodes,
        "qlinear_conv_dynamic_batch_graph",
        [x],
        [y],
        initializer=[x_scale, x_zero, w, w_scale, w_zero, y_scale, y_zero],
    )
    return helper.make_model(graph, opset_imports=[helper.make_operatorsetid("", 13)])


def _make_qlinear_conv_unknown_rank_model() -> onnx.ModelProto:
    model = _make_qlinear_conv_multichannel_model()
    for value_info in list(model.graph.input) + list(model.graph.value_info) + list(model.graph.output):
        if (
            value_info.type.HasField("tensor_type")
            and value_info.type.tensor_type.HasField("shape")
        ):
            value_info.type.tensor_type.ClearField("shape")
    return model


def _make_qlinear_conv_valid_explicit_pad_model() -> onnx.ModelProto:
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 1, 4, 4])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 1, 5, 5])

    x_scale = numpy_helper.from_array(np.asarray([0.1], dtype=np.float32), name="qv_x_scale")
    x_zero = numpy_helper.from_array(np.asarray([0], dtype=np.int8), name="qv_x_zero")
    w = numpy_helper.from_array(np.ones((1, 1, 2, 2), dtype=np.int8), name="qv_w")
    w_scale = numpy_helper.from_array(np.asarray([0.2], dtype=np.float32), name="qv_w_scale")
    w_zero = numpy_helper.from_array(np.asarray([0], dtype=np.int8), name="qv_w_zero")
    y_scale = numpy_helper.from_array(np.asarray([0.05], dtype=np.float32), name="qv_y_scale")
    y_zero = numpy_helper.from_array(np.asarray([0], dtype=np.int8), name="qv_y_zero")

    nodes = [
        helper.make_node("QuantizeLinear", ["x", "qv_x_scale", "qv_x_zero"], ["x_q"], name="QV_Q0"),
        helper.make_node(
            "QLinearConv",
            [
                "x_q",
                "qv_x_scale",
                "qv_x_zero",
                "qv_w",
                "qv_w_scale",
                "qv_w_zero",
                "qv_y_scale",
                "qv_y_zero",
            ],
            ["y_q"],
            name="QV_QConv",
            kernel_shape=[2, 2],
            pads=[1, 1, 1, 1],
            strides=[1, 1],
            auto_pad="VALID",
            group=1,
        ),
        helper.make_node("DequantizeLinear", ["y_q", "qv_y_scale", "qv_y_zero"], ["y"], name="QV_DQ0"),
    ]
    graph = helper.make_graph(
        nodes,
        "qlinear_conv_valid_explicit_pad_graph",
        [x],
        [y],
        initializer=[x_scale, x_zero, w, w_scale, w_zero, y_scale, y_zero],
    )
    return helper.make_model(graph, opset_imports=[helper.make_operatorsetid("", 13)])


def _make_terminal_quantize_dequantize_model() -> onnx.ModelProto:
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 1, 2, 2])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 1, 2, 2])
    q_scale = numpy_helper.from_array(np.asarray([0.05], dtype=np.float32), name="tail_q_scale")
    q_zero = numpy_helper.from_array(np.asarray([0], dtype=np.int8), name="tail_q_zero")
    nodes = [
        helper.make_node("Relu", ["x"], ["relu_out"], name="TailRelu"),
        helper.make_node(
            "QuantizeLinear",
            ["relu_out", "tail_q_scale", "tail_q_zero"],
            ["y_q"],
            name="TailQ",
        ),
        helper.make_node(
            "DequantizeLinear",
            ["y_q", "tail_q_scale", "tail_q_zero"],
            ["y"],
            name="TailDQ",
        ),
    ]
    graph = helper.make_graph(
        nodes,
        "terminal_quantize_dequantize_graph",
        [x],
        [y],
        initializer=[q_scale, q_zero],
    )
    return helper.make_model(graph, opset_imports=[helper.make_operatorsetid("", 13)])


def _make_terminal_transpose_dequantize_model() -> onnx.ModelProto:
    x = helper.make_tensor_value_info("x", TensorProto.INT8, [1, 2, 3, 4])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 3, 4, 2])
    q_scale = numpy_helper.from_array(np.asarray([0.1], dtype=np.float32), name="ttd_q_scale")
    q_zero = numpy_helper.from_array(np.asarray([0], dtype=np.int8), name="ttd_q_zero")
    nodes = [
        helper.make_node("Transpose", ["x"], ["x_t"], name="TTD_PreT", perm=[0, 2, 3, 1]),
        helper.make_node("DequantizeLinear", ["x_t", "ttd_q_scale", "ttd_q_zero"], ["y"], name="TTD_DQ"),
    ]
    graph = helper.make_graph(
        nodes,
        "terminal_transpose_dequantize_graph",
        [x],
        [y],
        initializer=[q_scale, q_zero],
    )
    return helper.make_model(graph, opset_imports=[helper.make_operatorsetid("", 13)])


def _make_transpose_quantize_transpose_model() -> onnx.ModelProto:
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 2, 3, 4])
    y = helper.make_tensor_value_info("y", TensorProto.INT8, [1, 2, 3, 4])
    q_scale = numpy_helper.from_array(np.asarray([0.1], dtype=np.float32), name="tqt_q_scale")
    q_zero = numpy_helper.from_array(np.asarray([0], dtype=np.int8), name="tqt_q_zero")
    nodes = [
        helper.make_node("Transpose", ["x"], ["x_t"], name="TQT_PreT", perm=[0, 2, 3, 1]),
        helper.make_node("QuantizeLinear", ["x_t", "tqt_q_scale", "tqt_q_zero"], ["x_q"], name="TQT_Q"),
        helper.make_node("Transpose", ["x_q"], ["y"], name="TQT_PostT", perm=[0, 3, 1, 2]),
    ]
    graph = helper.make_graph(
        nodes,
        "transpose_quantize_transpose_graph",
        [x],
        [y],
        initializer=[q_scale, q_zero],
    )
    return helper.make_model(graph, opset_imports=[helper.make_operatorsetid("", 13)])


def _make_transpose_quantize_transpose_dynamic_batch_model() -> onnx.ModelProto:
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, ["N", 2, 3, 4])
    y = helper.make_tensor_value_info("y", TensorProto.INT8, ["N", 2, 3, 4])
    q_scale = numpy_helper.from_array(np.asarray([0.1], dtype=np.float32), name="tqtd_q_scale")
    q_zero = numpy_helper.from_array(np.asarray([0], dtype=np.int8), name="tqtd_q_zero")
    nodes = [
        helper.make_node("Transpose", ["x"], ["x_t"], name="TQTD_PreT", perm=[0, 2, 3, 1]),
        helper.make_node("QuantizeLinear", ["x_t", "tqtd_q_scale", "tqtd_q_zero"], ["x_q"], name="TQTD_Q"),
        helper.make_node("Transpose", ["x_q"], ["y"], name="TQTD_PostT", perm=[0, 3, 1, 2]),
    ]
    graph = helper.make_graph(
        nodes,
        "transpose_quantize_transpose_dynamic_batch_graph",
        [x],
        [y],
        initializer=[q_scale, q_zero],
    )
    return helper.make_model(graph, opset_imports=[helper.make_operatorsetid("", 13)])


def _make_transpose_quantize_transpose_fanout_model() -> onnx.ModelProto:
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 2, 3, 4])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 2, 3, 4])
    q_scale = numpy_helper.from_array(np.asarray([0.1], dtype=np.float32), name="tqtf_q_scale")
    q_zero = numpy_helper.from_array(np.asarray([0], dtype=np.int8), name="tqtf_q_zero")
    nodes = [
        helper.make_node("Transpose", ["x"], ["x_t"], name="TQTF_PreT", perm=[0, 2, 3, 1]),
        helper.make_node("QuantizeLinear", ["x_t", "tqtf_q_scale", "tqtf_q_zero"], ["x_q"], name="TQTF_Q"),
        helper.make_node("Transpose", ["x_q"], ["x_q0"], name="TQTF_PostT0", perm=[0, 3, 1, 2]),
        helper.make_node("Transpose", ["x_q"], ["x_q1"], name="TQTF_PostT1", perm=[0, 3, 1, 2]),
        helper.make_node("DequantizeLinear", ["x_q0", "tqtf_q_scale", "tqtf_q_zero"], ["x_f0"], name="TQTF_DQ0"),
        helper.make_node("DequantizeLinear", ["x_q1", "tqtf_q_scale", "tqtf_q_zero"], ["x_f1"], name="TQTF_DQ1"),
        helper.make_node("Add", ["x_f0", "x_f1"], ["y"], name="TQTF_Add"),
    ]
    graph = helper.make_graph(
        nodes,
        "transpose_quantize_transpose_fanout_graph",
        [x],
        [y],
        initializer=[q_scale, q_zero],
    )
    return helper.make_model(graph, opset_imports=[helper.make_operatorsetid("", 13)])


def _make_transpose_dequantize_transpose_model() -> onnx.ModelProto:
    x = helper.make_tensor_value_info("x", TensorProto.INT8, [1, 2, 3, 4])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 2, 3, 4])
    q_scale = numpy_helper.from_array(np.asarray([0.1], dtype=np.float32), name="tdt_q_scale")
    q_zero = numpy_helper.from_array(np.asarray([0], dtype=np.int8), name="tdt_q_zero")
    nodes = [
        helper.make_node("Transpose", ["x"], ["x_t"], name="TDT_PreT", perm=[0, 2, 3, 1]),
        helper.make_node("DequantizeLinear", ["x_t", "tdt_q_scale", "tdt_q_zero"], ["x_f"], name="TDT_DQ"),
        helper.make_node("Transpose", ["x_f"], ["y"], name="TDT_PostT", perm=[0, 3, 1, 2]),
    ]
    graph = helper.make_graph(
        nodes,
        "transpose_dequantize_transpose_graph",
        [x],
        [y],
        initializer=[q_scale, q_zero],
    )
    return helper.make_model(graph, opset_imports=[helper.make_operatorsetid("", 13)])


def _make_transpose_cast_sub_mul_transpose_model() -> onnx.ModelProto:
    x = helper.make_tensor_value_info("x", TensorProto.UINT8, [1, 4, 5, 3])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 4, 5, 3])
    sub_bias = numpy_helper.from_array(
        np.asarray([[[[127.5]]]], dtype=np.float32),
        name="tcsm_sub_bias",
    )
    mul_scale = numpy_helper.from_array(np.asarray([0.00784314], dtype=np.float32), name="tcsm_mul_scale")
    nodes = [
        helper.make_node("Transpose", ["x"], ["x_t"], name="TCSM_PreT", perm=[0, 3, 1, 2]),
        helper.make_node("Cast", ["x_t"], ["x_f"], name="TCSM_Cast", to=TensorProto.FLOAT),
        helper.make_node("Sub", ["x_f", "tcsm_sub_bias"], ["x_sub"], name="TCSM_Sub"),
        helper.make_node("Mul", ["x_sub", "tcsm_mul_scale"], ["x_mul"], name="TCSM_Mul"),
        helper.make_node("Transpose", ["x_mul"], ["y"], name="TCSM_PostT", perm=[0, 2, 3, 1]),
    ]
    graph = helper.make_graph(
        nodes,
        "transpose_cast_sub_mul_transpose_graph",
        [x],
        [y],
        initializer=[sub_bias, mul_scale],
    )
    return helper.make_model(graph, opset_imports=[helper.make_operatorsetid("", 13)])


def _make_transpose_binary_fanout_chain_model() -> onnx.ModelProto:
    a = helper.make_tensor_value_info("a", TensorProto.FLOAT, [1, 2, 3, 4])
    b = helper.make_tensor_value_info("b", TensorProto.FLOAT, [1, 2, 3, 4])
    c = helper.make_tensor_value_info("c", TensorProto.FLOAT, [1, 2, 3, 4])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 2, 3, 4])
    z = helper.make_tensor_value_info("z", TensorProto.FLOAT, [1, 2, 3, 4])
    nodes = [
        helper.make_node("Transpose", ["a"], ["a_t"], name="TBFC_PreA", perm=[0, 3, 1, 2]),
        helper.make_node("Transpose", ["b"], ["b_t"], name="TBFC_PreB", perm=[0, 3, 1, 2]),
        helper.make_node("Add", ["a_t", "b_t"], ["ab_t"], name="TBFC_Add0"),
        helper.make_node("Transpose", ["ab_t"], ["ab"], name="TBFC_Post0", perm=[0, 2, 3, 1]),
        helper.make_node("Transpose", ["c"], ["c_t"], name="TBFC_PreC", perm=[0, 3, 1, 2]),
        helper.make_node("Add", ["ab_t", "c_t"], ["abc_t"], name="TBFC_Add1"),
        helper.make_node("Transpose", ["abc_t"], ["abc"], name="TBFC_Post1", perm=[0, 2, 3, 1]),
        helper.make_node("Relu", ["abc"], ["y"], name="TBFC_ReluY"),
        helper.make_node("Relu", ["ab"], ["z"], name="TBFC_ReluZ"),
    ]
    graph = helper.make_graph(nodes, "transpose_binary_fanout_chain_graph", [a, b, c], [y, z])
    return helper.make_model(graph, opset_imports=[helper.make_operatorsetid("", 13)])


def _make_transpose_dequantize_transpose_with_fanout_model() -> onnx.ModelProto:
    x = helper.make_tensor_value_info("x", TensorProto.INT8, [1, 2, 3, 4])
    y1 = helper.make_tensor_value_info("y1", TensorProto.FLOAT, [1, 2, 3, 4])
    y2 = helper.make_tensor_value_info("y2", TensorProto.FLOAT, [1, 3, 4, 2])
    q_scale = numpy_helper.from_array(np.asarray([0.1], dtype=np.float32), name="tdtf_q_scale")
    q_zero = numpy_helper.from_array(np.asarray([0], dtype=np.int8), name="tdtf_q_zero")
    nodes = [
        helper.make_node("Transpose", ["x"], ["x_t"], name="TDTF_PreT", perm=[0, 2, 3, 1]),
        helper.make_node("DequantizeLinear", ["x_t", "tdtf_q_scale", "tdtf_q_zero"], ["x_f1"], name="TDTF_DQ1"),
        helper.make_node("Transpose", ["x_f1"], ["y1"], name="TDTF_PostT", perm=[0, 3, 1, 2]),
        helper.make_node("DequantizeLinear", ["x_t", "tdtf_q_scale", "tdtf_q_zero"], ["y2"], name="TDTF_DQ2"),
    ]
    graph = helper.make_graph(
        nodes,
        "transpose_dequantize_transpose_with_fanout_graph",
        [x],
        [y1, y2],
        initializer=[q_scale, q_zero],
    )
    return helper.make_model(graph, opset_imports=[helper.make_operatorsetid("", 13)])


def _make_transpose_quantize_dequantize_transpose_model() -> onnx.ModelProto:
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 2, 3, 4])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 2, 3, 4])
    q_scale = numpy_helper.from_array(np.asarray([0.1], dtype=np.float32), name="tqdt_q_scale")
    q_zero = numpy_helper.from_array(np.asarray([0], dtype=np.int8), name="tqdt_q_zero")
    nodes = [
        helper.make_node("Transpose", ["x"], ["x_t"], name="TQDT_PreT", perm=[0, 2, 3, 1]),
        helper.make_node("QuantizeLinear", ["x_t", "tqdt_q_scale", "tqdt_q_zero"], ["x_q"], name="TQDT_Q"),
        helper.make_node("DequantizeLinear", ["x_q", "tqdt_q_scale", "tqdt_q_zero"], ["x_f"], name="TQDT_DQ"),
        helper.make_node("Transpose", ["x_f"], ["y"], name="TQDT_PostT", perm=[0, 3, 1, 2]),
    ]
    graph = helper.make_graph(
        nodes,
        "transpose_quantize_dequantize_transpose_graph",
        [x],
        [y],
        initializer=[q_scale, q_zero],
    )
    return helper.make_model(graph, opset_imports=[helper.make_operatorsetid("", 13)])


def _make_transpose_qdq_add_qdq_transpose_residual_model() -> onnx.ModelProto:
    x0 = helper.make_tensor_value_info("x0", TensorProto.FLOAT, [1, 2, 3, 4])
    x1 = helper.make_tensor_value_info("x1", TensorProto.FLOAT, [1, 2, 3, 4])
    y0 = helper.make_tensor_value_info("y0", TensorProto.FLOAT, [1, 2, 3, 4])
    y1 = helper.make_tensor_value_info("y1", TensorProto.FLOAT, [1, 2, 3, 4])
    q_scale = numpy_helper.from_array(np.asarray([0.1], dtype=np.float32), name="tqar_q_scale")
    q_zero = numpy_helper.from_array(np.asarray([0], dtype=np.int8), name="tqar_q_zero")
    nodes = [
        helper.make_node("Transpose", ["x0"], ["x0_t"], name="TQAR_Pre0", perm=[0, 3, 1, 2]),
        helper.make_node("QuantizeLinear", ["x0_t", "tqar_q_scale", "tqar_q_zero"], ["x0_q"], name="TQAR_Q0"),
        helper.make_node("DequantizeLinear", ["x0_q", "tqar_q_scale", "tqar_q_zero"], ["x0_dq"], name="TQAR_DQ0"),
        helper.make_node("Transpose", ["x0_dq"], ["x0_nhwc"], name="TQAR_Post0", perm=[0, 2, 3, 1]),
        helper.make_node("Relu", ["x0_nhwc"], ["y0"], name="TQAR_Relu0"),
        helper.make_node("Transpose", ["x1"], ["x1_t"], name="TQAR_Pre1", perm=[0, 3, 1, 2]),
        helper.make_node("QuantizeLinear", ["x1_t", "tqar_q_scale", "tqar_q_zero"], ["x1_q"], name="TQAR_Q1"),
        helper.make_node("DequantizeLinear", ["x1_q", "tqar_q_scale", "tqar_q_zero"], ["x1_dq"], name="TQAR_DQ1"),
        helper.make_node("Add", ["x0_dq", "x1_dq"], ["sum"], name="TQAR_Add"),
        helper.make_node("QuantizeLinear", ["sum", "tqar_q_scale", "tqar_q_zero"], ["sum_q"], name="TQAR_Q2"),
        helper.make_node("DequantizeLinear", ["sum_q", "tqar_q_scale", "tqar_q_zero"], ["sum_dq"], name="TQAR_DQ2"),
        helper.make_node("Transpose", ["sum_dq"], ["sum_nhwc"], name="TQAR_Post2", perm=[0, 2, 3, 1]),
        helper.make_node("Relu", ["sum_nhwc"], ["y1"], name="TQAR_Relu1"),
    ]
    graph = helper.make_graph(
        nodes,
        "transpose_qdq_add_qdq_transpose_residual_graph",
        [x0, x1],
        [y0, y1],
        initializer=[q_scale, q_zero],
    )
    return helper.make_model(graph, opset_imports=[helper.make_operatorsetid("", 13)])


def _make_transpose_mixed_add_qdq_transpose_residual_model() -> onnx.ModelProto:
    skip = helper.make_tensor_value_info("skip", TensorProto.FLOAT, [1, 2, 3, 4])
    proj = helper.make_tensor_value_info("proj", TensorProto.FLOAT, [1, 2, 3, 4])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 2, 3, 4])
    q_scale = numpy_helper.from_array(np.asarray([0.1], dtype=np.float32), name="tmqar_q_scale")
    q_zero = numpy_helper.from_array(np.asarray([0], dtype=np.int8), name="tmqar_q_zero")
    nodes = [
        helper.make_node("Transpose", ["skip"], ["skip_t"], name="TMQAR_SkipPre", perm=[0, 3, 1, 2]),
        helper.make_node("Transpose", ["proj"], ["proj_t"], name="TMQAR_ProjPre", perm=[0, 3, 1, 2]),
        helper.make_node("QuantizeLinear", ["proj_t", "tmqar_q_scale", "tmqar_q_zero"], ["proj_q"], name="TMQAR_Q0"),
        helper.make_node("DequantizeLinear", ["proj_q", "tmqar_q_scale", "tmqar_q_zero"], ["proj_dq"], name="TMQAR_DQ0"),
        helper.make_node("Add", ["proj_dq", "skip_t"], ["sum"], name="TMQAR_Add"),
        helper.make_node("QuantizeLinear", ["sum", "tmqar_q_scale", "tmqar_q_zero"], ["sum_q"], name="TMQAR_Q1"),
        helper.make_node("DequantizeLinear", ["sum_q", "tmqar_q_scale", "tmqar_q_zero"], ["sum_dq"], name="TMQAR_DQ1"),
        helper.make_node("Transpose", ["sum_dq"], ["y"], name="TMQAR_Post", perm=[0, 2, 3, 1]),
    ]
    graph = helper.make_graph(
        nodes,
        "transpose_mixed_add_qdq_transpose_residual_graph",
        [skip, proj],
        [y],
        initializer=[q_scale, q_zero],
    )
    return helper.make_model(graph, opset_imports=[helper.make_operatorsetid("", 13)])


def _make_transpose_mixed_add_qdq_transpose_residual_with_legacy_fanout_model() -> onnx.ModelProto:
    skip = helper.make_tensor_value_info("skip", TensorProto.FLOAT, [1, 2, 3, 4])
    proj = helper.make_tensor_value_info("proj", TensorProto.FLOAT, [1, 2, 3, 4])
    legacy = helper.make_tensor_value_info("legacy", TensorProto.FLOAT, [1, 2, 3, 4])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 2, 3, 4])
    z = helper.make_tensor_value_info("z", TensorProto.FLOAT, [1, 4, 2, 3])
    q_scale = numpy_helper.from_array(np.asarray([0.1], dtype=np.float32), name="tmlf_q_scale")
    q_zero = numpy_helper.from_array(np.asarray([0], dtype=np.int8), name="tmlf_q_zero")
    nodes = [
        helper.make_node("Transpose", ["skip"], ["skip_t"], name="TMLF_SkipPre", perm=[0, 3, 1, 2]),
        helper.make_node("Transpose", ["proj"], ["proj_t"], name="TMLF_ProjPre", perm=[0, 3, 1, 2]),
        helper.make_node("QuantizeLinear", ["proj_t", "tmlf_q_scale", "tmlf_q_zero"], ["proj_q"], name="TMLF_Q0"),
        helper.make_node("DequantizeLinear", ["proj_q", "tmlf_q_scale", "tmlf_q_zero"], ["proj_dq"], name="TMLF_DQ0"),
        helper.make_node("Add", ["proj_dq", "skip_t"], ["sum"], name="TMLF_Add0"),
        helper.make_node("QuantizeLinear", ["sum", "tmlf_q_scale", "tmlf_q_zero"], ["sum_q"], name="TMLF_Q1"),
        helper.make_node("DequantizeLinear", ["sum_q", "tmlf_q_scale", "tmlf_q_zero"], ["sum_dq"], name="TMLF_DQ1"),
        helper.make_node("Transpose", ["sum_dq"], ["y"], name="TMLF_Post", perm=[0, 2, 3, 1]),
        helper.make_node("Transpose", ["legacy"], ["legacy_t"], name="TMLF_LegacyPre", perm=[0, 3, 1, 2]),
        helper.make_node("Add", ["sum_dq", "legacy_t"], ["z"], name="TMLF_Add1"),
    ]
    graph = helper.make_graph(
        nodes,
        "transpose_mixed_add_qdq_transpose_residual_with_legacy_fanout_graph",
        [skip, proj, legacy],
        [y, z],
        initializer=[q_scale, q_zero],
    )
    return helper.make_model(graph, opset_imports=[helper.make_operatorsetid("", 13)])


def _make_transpose_dequantize_relu6_quantize_transpose_model() -> onnx.ModelProto:
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 2, 3, 4])
    y = helper.make_tensor_value_info("y", TensorProto.INT8, [1, 2, 3, 4])
    q_scale = numpy_helper.from_array(np.asarray([0.1], dtype=np.float32), name="tdrqt_q_scale")
    q_zero = numpy_helper.from_array(np.asarray([0], dtype=np.int8), name="tdrqt_q_zero")
    nodes = [
        helper.make_node("QuantizeLinear", ["x", "tdrqt_q_scale", "tdrqt_q_zero"], ["x_q_src"], name="TDRQT_Q0"),
        helper.make_node("Transpose", ["x_q_src"], ["x_t"], name="TDRQT_PreT", perm=[0, 2, 3, 1]),
        helper.make_node("DequantizeLinear", ["x_t", "tdrqt_q_scale", "tdrqt_q_zero"], ["x_f"], name="TDRQT_DQ"),
        helper.make_node("Clip", ["x_f"], ["x_r6"], name="TDRQT_Relu6", min=0.0, max=6.0),
        helper.make_node("QuantizeLinear", ["x_r6", "tdrqt_q_scale", "tdrqt_q_zero"], ["x_q"], name="TDRQT_Q1"),
        helper.make_node("Transpose", ["x_q"], ["y"], name="TDRQT_PostT", perm=[0, 3, 1, 2]),
    ]
    graph = helper.make_graph(
        nodes,
        "transpose_dequantize_relu6_quantize_transpose_graph",
        [x],
        [y],
        initializer=[q_scale, q_zero],
    )
    return helper.make_model(graph, opset_imports=[helper.make_operatorsetid("", 13)])


def _make_transpose_binary_transpose_model(binary_op: str) -> onnx.ModelProto:
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 2, 3, 4])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 2, 3, 4])
    z = helper.make_tensor_value_info("z", TensorProto.FLOAT, [1, 2, 3, 4])
    nodes = [
        helper.make_node("Transpose", ["x"], ["x_t"], name=f"TBT_PreT0_{binary_op}", perm=[0, 2, 3, 1]),
        helper.make_node("Transpose", ["y"], ["y_t"], name=f"TBT_PreT1_{binary_op}", perm=[0, 2, 3, 1]),
        helper.make_node(binary_op, ["x_t", "y_t"], ["z_t"], name=f"TBT_{binary_op}"),
        helper.make_node("Transpose", ["z_t"], ["z"], name=f"TBT_PostT_{binary_op}", perm=[0, 3, 1, 2]),
    ]
    graph = helper.make_graph(
        nodes,
        f"transpose_{binary_op.lower()}_transpose_graph",
        [x, y],
        [z],
    )
    return helper.make_model(graph, opset_imports=[helper.make_operatorsetid("", 13)])


def _make_transpose_transpose_softmax_model() -> onnx.ModelProto:
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 2, 3, 5])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 3, 2, 5])
    nodes = [
        helper.make_node("Transpose", ["x"], ["x_t0"], name="TTS_Pre0", perm=[0, 3, 1, 2]),
        helper.make_node("Transpose", ["x_t0"], ["x_t1"], name="TTS_Pre1", perm=[0, 3, 2, 1]),
        helper.make_node("Softmax", ["x_t1"], ["y"], name="TTS_Softmax", axis=-1),
    ]
    graph = helper.make_graph(nodes, "transpose_transpose_softmax_graph", [x], [y])
    return helper.make_model(graph, opset_imports=[helper.make_operatorsetid("", 13)])


def _make_softmax_terminal_transpose_model() -> onnx.ModelProto:
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 2, 3, 4])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 4, 3, 2])
    nodes = [
        helper.make_node("Softmax", ["x"], ["s"], name="STT_Softmax", axis=-1),
        helper.make_node("Transpose", ["s"], ["y"], name="STT_OutTranspose", perm=[0, 3, 2, 1]),
    ]
    graph = helper.make_graph(nodes, "softmax_terminal_transpose_graph", [x], [y])
    return helper.make_model(graph, opset_imports=[helper.make_operatorsetid("", 13)])


def _make_softmax_terminal_double_pretranspose_model() -> onnx.ModelProto:
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 2, 3, 4])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 4, 2, 3])
    nodes = [
        helper.make_node("Transpose", ["x"], ["x_t0"], name="STDPT_Pre0", perm=[0, 3, 1, 2]),
        helper.make_node("Transpose", ["x_t0"], ["x_t1"], name="STDPT_Pre1", perm=[0, 3, 2, 1]),
        helper.make_node("Softmax", ["x_t1"], ["s"], name="STDPT_Softmax", axis=-1),
        helper.make_node("Transpose", ["s"], ["y"], name="STDPT_OutTranspose", perm=[0, 3, 2, 1]),
    ]
    graph = helper.make_graph(nodes, "softmax_terminal_double_pretranspose_graph", [x], [y])
    return helper.make_model(graph, opset_imports=[helper.make_operatorsetid("", 13)])


def _make_transpose_swish_transpose_model() -> onnx.ModelProto:
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 2, 3, 4])
    z = helper.make_tensor_value_info("z", TensorProto.FLOAT, [1, 2, 3, 4])
    nodes = [
        helper.make_node("Transpose", ["x"], ["x_t"], name="TSWT_PreT", perm=[0, 3, 1, 2]),
        helper.make_node("Sigmoid", ["x_t"], ["s_t"], name="TSWT_Sigmoid"),
        helper.make_node("Mul", ["x_t", "s_t"], ["y_t"], name="TSWT_Mul"),
        helper.make_node("Transpose", ["y_t"], ["y"], name="TSWT_PostT", perm=[0, 2, 3, 1]),
        helper.make_node("Relu", ["y"], ["z"], name="TSWT_Relu"),
    ]
    graph = helper.make_graph(nodes, "transpose_swish_transpose_graph", [x], [z])
    return helper.make_model(graph, opset_imports=[helper.make_operatorsetid("", 13)])


def _make_transpose_hardsigmoid_transpose_model() -> onnx.ModelProto:
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 2, 3, 4])
    z = helper.make_tensor_value_info("z", TensorProto.FLOAT, [1, 2, 3, 4])
    nodes = [
        helper.make_node("Transpose", ["x"], ["x_t"], name="THST_PreT", perm=[0, 3, 1, 2]),
        helper.make_node("HardSigmoid", ["x_t"], ["hs_t"], name="THST_HardSigmoid", alpha=0.2, beta=0.5),
        helper.make_node("Transpose", ["hs_t"], ["y"], name="THST_PostT", perm=[0, 2, 3, 1]),
        helper.make_node("Relu", ["y"], ["z"], name="THST_Relu"),
    ]
    graph = helper.make_graph(nodes, "transpose_hardsigmoid_transpose_graph", [x], [z])
    return helper.make_model(graph, opset_imports=[helper.make_operatorsetid("", 13)])


def _make_transpose_hardsigmoid_mul_transpose_model() -> onnx.ModelProto:
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 2, 3, 4])
    z = helper.make_tensor_value_info("z", TensorProto.FLOAT, [1, 2, 3, 4])
    nodes = [
        helper.make_node("Transpose", ["x"], ["x_t"], name="THSMT_PreT", perm=[0, 3, 1, 2]),
        helper.make_node("HardSigmoid", ["x_t"], ["hs_t"], name="THSMT_HardSigmoid", alpha=0.2, beta=0.5),
        helper.make_node("Mul", ["x_t", "hs_t"], ["y_t"], name="THSMT_Mul"),
        helper.make_node("Transpose", ["y_t"], ["y"], name="THSMT_PostT", perm=[0, 2, 3, 1]),
        helper.make_node("Relu", ["y"], ["z"], name="THSMT_Relu"),
    ]
    graph = helper.make_graph(nodes, "transpose_hardsigmoid_mul_transpose_graph", [x], [z])
    return helper.make_model(graph, opset_imports=[helper.make_operatorsetid("", 13)])


def _make_transpose_prelu_transpose_model() -> onnx.ModelProto:
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 2, 3, 4])
    z = helper.make_tensor_value_info("z", TensorProto.FLOAT, [1, 2, 3, 4])
    slope = numpy_helper.from_array(
        np.asarray([[[1.0]], [[0.5]], [[0.25]], [[0.125]]], dtype=np.float32),
        name="tpt_alpha",
    )
    nodes = [
        helper.make_node("Transpose", ["x"], ["x_t"], name="TPT_PreT", perm=[0, 3, 1, 2]),
        helper.make_node("PRelu", ["x_t", "tpt_alpha"], ["p_t"], name="TPT_PRelu"),
        helper.make_node("Transpose", ["p_t"], ["y"], name="TPT_PostT", perm=[0, 2, 3, 1]),
        helper.make_node("Relu", ["y"], ["z"], name="TPT_Relu"),
    ]
    graph = helper.make_graph(
        nodes,
        "transpose_prelu_transpose_graph",
        [x],
        [z],
        initializer=[slope],
    )
    return helper.make_model(graph, opset_imports=[helper.make_operatorsetid("", 13)])


def _make_transpose_gelu_transpose_model() -> onnx.ModelProto:
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 2, 3, 4])
    z = helper.make_tensor_value_info("z", TensorProto.FLOAT, [1, 2, 3, 4])
    nodes = [
        helper.make_node("Transpose", ["x"], ["x_t"], name="TGT_PreT", perm=[0, 3, 1, 2]),
        helper.make_node("Gelu", ["x_t"], ["g_t"], name="TGT_Gelu"),
        helper.make_node("Transpose", ["g_t"], ["z"], name="TGT_PostT", perm=[0, 2, 3, 1]),
    ]
    graph = helper.make_graph(nodes, "transpose_gelu_transpose_graph", [x], [z])
    return helper.make_model(graph, opset_imports=[helper.make_operatorsetid("", 20)])


def _make_transpose_gelu_tanh_transpose_model() -> onnx.ModelProto:
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 2, 3, 4])
    z = helper.make_tensor_value_info("z", TensorProto.FLOAT, [1, 2, 3, 4])
    c0 = numpy_helper.from_array(np.asarray(0.044715, dtype=np.float32), name="tgtt_c0")
    c1 = numpy_helper.from_array(np.asarray(0.7978845834732056, dtype=np.float32), name="tgtt_c1")
    c2 = numpy_helper.from_array(np.asarray(1.0, dtype=np.float32), name="tgtt_c2")
    c3 = numpy_helper.from_array(np.asarray(0.5, dtype=np.float32), name="tgtt_c3")
    nodes = [
        helper.make_node("Transpose", ["x"], ["x_t"], name="TGTT_PreT", perm=[0, 3, 1, 2]),
        helper.make_node("Mul", ["x_t", "x_t"], ["x2_t"], name="TGTT_MulSq"),
        helper.make_node("Mul", ["x2_t", "x_t"], ["x3_t"], name="TGTT_MulCube"),
        helper.make_node("Mul", ["x3_t", "tgtt_c0"], ["m0_t"], name="TGTT_MulC0"),
        helper.make_node("Add", ["x_t", "m0_t"], ["a0_t"], name="TGTT_Add0"),
        helper.make_node("Mul", ["a0_t", "tgtt_c1"], ["m1_t"], name="TGTT_MulC1"),
        helper.make_node("Tanh", ["m1_t"], ["t_t"], name="TGTT_Tanh"),
        helper.make_node("Add", ["t_t", "tgtt_c2"], ["a1_t"], name="TGTT_Add1"),
        helper.make_node("Mul", ["x_t", "a1_t"], ["m2_t"], name="TGTT_MulX"),
        helper.make_node("Mul", ["m2_t", "tgtt_c3"], ["y_t"], name="TGTT_MulHalf"),
        helper.make_node("Transpose", ["y_t"], ["y"], name="TGTT_PostT", perm=[0, 2, 3, 1]),
        helper.make_node("Relu", ["y"], ["z"], name="TGTT_Relu"),
    ]
    graph = helper.make_graph(
        nodes,
        "transpose_gelu_tanh_transpose_graph",
        [x],
        [z],
        initializer=[c0, c1, c2, c3],
    )
    return helper.make_model(graph, opset_imports=[helper.make_operatorsetid("", 13)])


def _make_transpose_add_reshape_transpose_with_legacy_model() -> onnx.ModelProto:
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 2, 3, 4])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 2, 3, 4])
    z = helper.make_tensor_value_info("z", TensorProto.FLOAT, [1, 6, 4])
    legacy = helper.make_tensor_value_info("legacy", TensorProto.FLOAT, [1, 4, 2, 3])
    shape = numpy_helper.from_array(np.asarray([1, 4, 6], dtype=np.int64), name="tars_shape")
    nodes = [
        helper.make_node("Transpose", ["x"], ["x_nchw"], name="TARS_PreX", perm=[0, 3, 1, 2]),
        helper.make_node("Transpose", ["y"], ["y_nchw"], name="TARS_PreY", perm=[0, 3, 1, 2]),
        helper.make_node("Add", ["x_nchw", "y_nchw"], ["sum_nchw"], name="TARS_Add"),
        helper.make_node("Reshape", ["sum_nchw", "tars_shape"], ["sum_ncw"], name="TARS_Reshape"),
        helper.make_node("Transpose", ["sum_ncw"], ["z"], name="TARS_Post", perm=[0, 2, 1]),
        helper.make_node("Add", ["sum_nchw", "x_nchw"], ["legacy"], name="TARS_LegacyAdd"),
    ]
    graph = helper.make_graph(
        nodes,
        "transpose_add_reshape_transpose_with_legacy_graph",
        [x, y],
        [z, legacy],
        initializer=[shape],
    )
    return helper.make_model(graph, opset_imports=[helper.make_operatorsetid("", 13)])


def _make_transpose_mulconst_add_reshape_transpose_with_legacy_model() -> onnx.ModelProto:
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 2, 3, 4])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 2, 3, 4])
    z = helper.make_tensor_value_info("z", TensorProto.FLOAT, [1, 6, 4])
    legacy = helper.make_tensor_value_info("legacy", TensorProto.FLOAT, [1, 4, 2, 3])
    shape = numpy_helper.from_array(np.asarray([1, 4, 6], dtype=np.int64), name="tmacrs_shape")
    scale = numpy_helper.from_array(
        np.asarray(np.full((1, 4, 1, 1), 0.5, dtype=np.float32)),
        name="tmacrs_scale",
    )
    nodes = [
        helper.make_node("Transpose", ["x"], ["x_nchw"], name="TMACRS_PreX", perm=[0, 3, 1, 2]),
        helper.make_node("Transpose", ["y"], ["y_nchw"], name="TMACRS_PreY", perm=[0, 3, 1, 2]),
        helper.make_node("Mul", ["y_nchw", "tmacrs_scale"], ["y_scaled_nchw"], name="TMACRS_Mul"),
        helper.make_node("Add", ["x_nchw", "y_scaled_nchw"], ["sum_nchw"], name="TMACRS_Add"),
        helper.make_node("Reshape", ["sum_nchw", "tmacrs_shape"], ["sum_ncw"], name="TMACRS_Reshape"),
        helper.make_node("Transpose", ["sum_ncw"], ["z_mid"], name="TMACRS_Post", perm=[0, 2, 1]),
        helper.make_node("Identity", ["z_mid"], ["z"], name="TMACRS_Out"),
        helper.make_node("Add", ["sum_nchw", "x_nchw"], ["legacy"], name="TMACRS_LegacyAdd"),
    ]
    graph = helper.make_graph(
        nodes,
        "transpose_mulconst_add_reshape_transpose_with_legacy_graph",
        [x, y],
        [z, legacy],
        initializer=[shape, scale],
    )
    return helper.make_model(graph, opset_imports=[helper.make_operatorsetid("", 13)])


def _make_transpose_nested_add_with_shared_skip_model() -> onnx.ModelProto:
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 2, 3, 4])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 2, 3, 4])
    z = helper.make_tensor_value_info("z", TensorProto.FLOAT, [1, 2, 3, 4])
    scale = numpy_helper.from_array(np.asarray(0.5, dtype=np.float32), name="tnas_scale")
    nodes = [
        helper.make_node("Transpose", ["x"], ["x_nchw"], name="TNAS_PreX", perm=[0, 3, 1, 2]),
        helper.make_node("Transpose", ["y"], ["y_nchw"], name="TNAS_PreY", perm=[0, 3, 1, 2]),
        helper.make_node("Mul", ["y_nchw", "tnas_scale"], ["y_scaled_nchw"], name="TNAS_Mul"),
        helper.make_node("Add", ["x_nchw", "y_scaled_nchw"], ["sum0_nchw"], name="TNAS_Add0"),
        helper.make_node("Add", ["sum0_nchw", "x_nchw"], ["sum1_nchw"], name="TNAS_Add1"),
        helper.make_node("Transpose", ["sum1_nchw"], ["sum1_nhwc"], name="TNAS_Post", perm=[0, 2, 3, 1]),
        helper.make_node("Relu", ["sum1_nhwc"], ["z"], name="TNAS_Relu"),
    ]
    graph = helper.make_graph(
        nodes,
        "transpose_nested_add_with_shared_skip_graph",
        [x, y],
        [z],
        initializer=[scale],
    )
    return helper.make_model(graph, opset_imports=[helper.make_operatorsetid("", 13)])


def _make_transpose_swish_add_concat_transpose_model() -> onnx.ModelProto:
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 2, 3, 4])
    z = helper.make_tensor_value_info("z", TensorProto.FLOAT, [1, 2, 3, 8])
    nodes = [
        helper.make_node("Transpose", ["x"], ["a_t"], name="TSACT_PreA", perm=[0, 3, 1, 2]),
        helper.make_node("Sigmoid", ["a_t"], ["a_sig"], name="TSACT_ASig"),
        helper.make_node("Mul", ["a_t", "a_sig"], ["a_swish"], name="TSACT_AMul"),
        helper.make_node("Transpose", ["x"], ["b_t"], name="TSACT_PreB", perm=[0, 3, 1, 2]),
        helper.make_node("Sigmoid", ["b_t"], ["b_sig"], name="TSACT_BSig"),
        helper.make_node("Mul", ["b_t", "b_sig"], ["b_swish"], name="TSACT_BMul"),
        helper.make_node("Transpose", ["x"], ["c_t"], name="TSACT_PreC", perm=[0, 3, 1, 2]),
        helper.make_node("Sigmoid", ["c_t"], ["c_sig"], name="TSACT_CSig"),
        helper.make_node("Mul", ["c_t", "c_sig"], ["c_swish"], name="TSACT_CMul"),
        helper.make_node("Add", ["b_swish", "c_swish"], ["sum_nchw"], name="TSACT_Add"),
        helper.make_node("Concat", ["sum_nchw", "a_swish"], ["cat_nchw"], name="TSACT_Concat", axis=1),
        helper.make_node("Transpose", ["cat_nchw"], ["y"], name="TSACT_Post", perm=[0, 2, 3, 1]),
        helper.make_node("Relu", ["y"], ["z"], name="TSACT_Relu"),
    ]
    graph = helper.make_graph(nodes, "transpose_swish_add_concat_transpose_graph", [x], [z])
    return helper.make_model(graph, opset_imports=[helper.make_operatorsetid("", 13)])


def _make_transpose_swish_add_transpose_cascade_model() -> onnx.ModelProto:
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 2, 3, 4])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 2, 3, 4])
    nodes = [
        helper.make_node("Sigmoid", ["x"], ["a_sig"], name="TSATC_ASig"),
        helper.make_node("Mul", ["x", "a_sig"], ["a_nhwc"], name="TSATC_AMul"),
        helper.make_node("Transpose", ["a_nhwc"], ["b_nchw"], name="TSATC_BT", perm=[0, 3, 1, 2]),
        helper.make_node("Transpose", ["x"], ["c_nchw"], name="TSATC_CT", perm=[0, 3, 1, 2]),
        helper.make_node("Sigmoid", ["c_nchw"], ["c_sig"], name="TSATC_CSig"),
        helper.make_node("Mul", ["c_nchw", "c_sig"], ["c_swish"], name="TSATC_CMul"),
        helper.make_node("Add", ["c_swish", "b_nchw"], ["sum1_nchw"], name="TSATC_Add1"),
        helper.make_node("Transpose", ["sum1_nchw"], ["sum1_nhwc"], name="TSATC_Post1", perm=[0, 2, 3, 1]),
        helper.make_node("Transpose", ["x"], ["d_nchw"], name="TSATC_DT", perm=[0, 3, 1, 2]),
        helper.make_node("Sigmoid", ["d_nchw"], ["d_sig"], name="TSATC_DSig"),
        helper.make_node("Mul", ["d_nchw", "d_sig"], ["d_swish"], name="TSATC_DMul"),
        helper.make_node("Add", ["d_swish", "sum1_nchw"], ["sum2_nchw"], name="TSATC_Add2"),
        helper.make_node("Transpose", ["sum2_nchw"], ["sum2_nhwc"], name="TSATC_Post2", perm=[0, 2, 3, 1]),
        helper.make_node("Relu", ["sum2_nhwc"], ["y"], name="TSATC_Relu"),
    ]
    graph = helper.make_graph(nodes, "transpose_swish_add_transpose_cascade_graph", [x], [y])
    return helper.make_model(graph, opset_imports=[helper.make_operatorsetid("", 13)])


def _make_transpose_slice_concat_transpose_model() -> onnx.ModelProto:
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 8, 8, 3])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 4, 4, 12])
    starts_00 = numpy_helper.from_array(np.asarray([0, 0], dtype=np.int64), name="tsct_starts_00")
    starts_01 = numpy_helper.from_array(np.asarray([0, 1], dtype=np.int64), name="tsct_starts_01")
    starts_10 = numpy_helper.from_array(np.asarray([1, 0], dtype=np.int64), name="tsct_starts_10")
    starts_11 = numpy_helper.from_array(np.asarray([1, 1], dtype=np.int64), name="tsct_starts_11")
    ends = numpy_helper.from_array(np.asarray([8, 8], dtype=np.int64), name="tsct_ends")
    axes = numpy_helper.from_array(np.asarray([2, 3], dtype=np.int64), name="tsct_axes")
    steps = numpy_helper.from_array(np.asarray([2, 2], dtype=np.int64), name="tsct_steps")
    nodes = [
        helper.make_node("Transpose", ["x"], ["x_nchw"], name="TSCT_Pre", perm=[0, 3, 1, 2]),
        helper.make_node("Slice", ["x_nchw", "tsct_starts_00", "tsct_ends", "tsct_axes", "tsct_steps"], ["s00"], name="TSCT_Slice00"),
        helper.make_node("Slice", ["x_nchw", "tsct_starts_01", "tsct_ends", "tsct_axes", "tsct_steps"], ["s01"], name="TSCT_Slice01"),
        helper.make_node("Slice", ["x_nchw", "tsct_starts_10", "tsct_ends", "tsct_axes", "tsct_steps"], ["s10"], name="TSCT_Slice10"),
        helper.make_node("Slice", ["x_nchw", "tsct_starts_11", "tsct_ends", "tsct_axes", "tsct_steps"], ["s11"], name="TSCT_Slice11"),
        helper.make_node("Concat", ["s00", "s10", "s01", "s11"], ["cat_nchw"], name="TSCT_Concat", axis=1),
        helper.make_node("Transpose", ["cat_nchw"], ["cat_nhwc"], name="TSCT_Post", perm=[0, 2, 3, 1]),
        helper.make_node("Relu", ["cat_nhwc"], ["y"], name="TSCT_Relu"),
    ]
    graph = helper.make_graph(
        nodes,
        "transpose_slice_concat_transpose_graph",
        [x],
        [y],
        initializer=[starts_00, starts_01, starts_10, starts_11, ends, axes, steps],
    )
    return helper.make_model(graph, opset_imports=[helper.make_operatorsetid("", 13)])


def _make_transpose_logistic_concat_reshape_model() -> onnx.ModelProto:
    bbox = helper.make_tensor_value_info("bbox", TensorProto.FLOAT, [1, 8, 8, 4])
    obj = helper.make_tensor_value_info("obj", TensorProto.FLOAT, [1, 8, 8, 1])
    cls = helper.make_tensor_value_info("cls", TensorProto.FLOAT, [1, 8, 8, 80])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 85, 64])
    reshape_shape = numpy_helper.from_array(
        np.asarray([1, 85, 64], dtype=np.int64),
        name="tlcr_shape",
    )
    nodes = [
        helper.make_node("Transpose", ["bbox"], ["bbox_nchw"], name="TLCR_PreBbox", perm=[0, 3, 1, 2]),
        helper.make_node("Transpose", ["obj"], ["obj_nchw"], name="TLCR_PreObj", perm=[0, 3, 1, 2]),
        helper.make_node("Transpose", ["cls"], ["cls_nchw"], name="TLCR_PreCls", perm=[0, 3, 1, 2]),
        helper.make_node("Sigmoid", ["obj_nchw"], ["obj_sig"], name="TLCR_ObjSigmoid"),
        helper.make_node("Sigmoid", ["cls_nchw"], ["cls_sig"], name="TLCR_ClsSigmoid"),
        helper.make_node("Concat", ["bbox_nchw", "obj_sig", "cls_sig"], ["cat_nchw"], name="TLCR_Concat", axis=1),
        helper.make_node("Reshape", ["cat_nchw", "tlcr_shape"], ["y"], name="TLCR_Reshape"),
    ]
    graph = helper.make_graph(
        nodes,
        "transpose_logistic_concat_reshape_graph",
        [bbox, obj, cls],
        [y],
        initializer=[reshape_shape],
    )
    return helper.make_model(graph, opset_imports=[helper.make_operatorsetid("", 13)])


def _make_transpose_relu6_transpose_fanout_model() -> onnx.ModelProto:
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 2, 3, 4])
    y0 = helper.make_tensor_value_info("y0", TensorProto.FLOAT, [1, 2, 3, 4])
    y1 = helper.make_tensor_value_info("y1", TensorProto.FLOAT, [1, 2, 3, 4])
    nodes = [
        helper.make_node("Transpose", ["x"], ["x_t"], name="TRTF_PreT", perm=[0, 2, 3, 1]),
        helper.make_node("Clip", ["x_t"], ["r_t"], name="TRTF_Relu6", min=0.0, max=6.0),
        helper.make_node("Transpose", ["r_t"], ["r0"], name="TRTF_PostT0", perm=[0, 3, 1, 2]),
        helper.make_node("Transpose", ["r_t"], ["r1"], name="TRTF_PostT1", perm=[0, 3, 1, 2]),
        helper.make_node("Relu", ["r0"], ["y0"], name="TRTF_Relu0"),
        helper.make_node("Neg", ["r1"], ["y1"], name="TRTF_Neg1"),
    ]
    graph = helper.make_graph(nodes, "transpose_relu6_transpose_fanout_graph", [x], [y0, y1])
    return helper.make_model(graph, opset_imports=[helper.make_operatorsetid("", 13)])


def _make_transpose_relu6_transpose_mixed_fanout_model() -> onnx.ModelProto:
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 2, 3, 4])
    s = helper.make_tensor_value_info("s", TensorProto.FLOAT, [1, 4, 2, 3])
    y0 = helper.make_tensor_value_info("y0", TensorProto.FLOAT, [1, 2, 3, 4])
    y1 = helper.make_tensor_value_info("y1", TensorProto.FLOAT, [1, 4, 2, 3])
    nodes = [
        helper.make_node("Transpose", ["x"], ["x_t"], name="TRTMF_PreT", perm=[0, 3, 1, 2]),
        helper.make_node("Clip", ["x_t"], ["r_t"], name="TRTMF_Relu6", min=0.0, max=6.0),
        helper.make_node("Transpose", ["r_t"], ["r0"], name="TRTMF_PostT", perm=[0, 2, 3, 1]),
        helper.make_node("Relu", ["r0"], ["y0"], name="TRTMF_Relu0"),
        helper.make_node("Add", ["r_t", "s"], ["y1"], name="TRTMF_LegacyAdd"),
    ]
    graph = helper.make_graph(nodes, "transpose_relu6_transpose_mixed_fanout_graph", [x, s], [y0, y1])
    return helper.make_model(graph, opset_imports=[helper.make_operatorsetid("", 13)])


def _make_transpose_relu6_binary_transpose_fanout_model() -> onnx.ModelProto:
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 2, 3, 4])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 2, 3, 4])
    o0 = helper.make_tensor_value_info("o0", TensorProto.FLOAT, [1, 2, 3, 4])
    o1 = helper.make_tensor_value_info("o1", TensorProto.FLOAT, [1, 2, 3, 4])
    nodes = [
        helper.make_node("Transpose", ["x"], ["x_t"], name="TRBTF_PreX", perm=[0, 3, 1, 2]),
        helper.make_node("Clip", ["x_t"], ["x_r6_t"], name="TRBTF_Relu6", min=0.0, max=6.0),
        helper.make_node("Transpose", ["y"], ["y_t"], name="TRBTF_PreY", perm=[0, 3, 1, 2]),
        helper.make_node("Add", ["x_r6_t", "y_t"], ["z_t"], name="TRBTF_Add"),
        helper.make_node("Transpose", ["z_t"], ["z0"], name="TRBTF_Post0", perm=[0, 2, 3, 1]),
        helper.make_node("Transpose", ["z_t"], ["z1"], name="TRBTF_Post1", perm=[0, 2, 3, 1]),
        helper.make_node("Relu", ["z0"], ["o0"], name="TRBTF_Relu0"),
        helper.make_node("Neg", ["z1"], ["o1"], name="TRBTF_Neg1"),
    ]
    graph = helper.make_graph(nodes, "transpose_relu6_binary_transpose_fanout_graph", [x, y], [o0, o1])
    return helper.make_model(graph, opset_imports=[helper.make_operatorsetid("", 13)])


def _make_transpose_relu6_binary_mixed_fanout_model() -> onnx.ModelProto:
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 2, 3, 4])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 2, 3, 4])
    s = helper.make_tensor_value_info("s", TensorProto.FLOAT, [1, 4, 2, 3])
    o0 = helper.make_tensor_value_info("o0", TensorProto.FLOAT, [1, 2, 3, 4])
    o1 = helper.make_tensor_value_info("o1", TensorProto.FLOAT, [1, 4, 2, 3])
    nodes = [
        helper.make_node("Transpose", ["x"], ["x_t"], name="TRBMF_PreX", perm=[0, 3, 1, 2]),
        helper.make_node("Clip", ["x_t"], ["x_r6_t"], name="TRBMF_Relu6", min=0.0, max=6.0),
        helper.make_node("Transpose", ["y"], ["y_t"], name="TRBMF_PreY", perm=[0, 3, 1, 2]),
        helper.make_node("Add", ["x_r6_t", "y_t"], ["z_t"], name="TRBMF_Add"),
        helper.make_node("Transpose", ["z_t"], ["z"], name="TRBMF_Post", perm=[0, 2, 3, 1]),
        helper.make_node("Relu", ["z"], ["o0"], name="TRBMF_Relu0"),
        helper.make_node("Add", ["z_t", "s"], ["o1"], name="TRBMF_LegacyAdd"),
    ]
    graph = helper.make_graph(nodes, "transpose_relu6_binary_mixed_fanout_graph", [x, y, s], [o0, o1])
    return helper.make_model(graph, opset_imports=[helper.make_operatorsetid("", 13)])


def _make_transpose_binary_transpose_fanout_model(binary_op: str) -> onnx.ModelProto:
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 2, 3, 4])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 2, 3, 4])
    s = helper.make_tensor_value_info("s", TensorProto.FLOAT, [1, 4, 2, 3])
    out_nhwc = helper.make_tensor_value_info("out_nhwc", TensorProto.FLOAT, [1, 2, 3, 4])
    out_nchw = helper.make_tensor_value_info("out_nchw", TensorProto.FLOAT, [1, 4, 2, 3])
    nodes = [
        helper.make_node("Transpose", ["x"], ["x_t"], name=f"TBTF_PreT0_{binary_op}", perm=[0, 3, 1, 2]),
        helper.make_node("Transpose", ["y"], ["y_t"], name=f"TBTF_PreT1_{binary_op}", perm=[0, 3, 1, 2]),
        helper.make_node(binary_op, ["x_t", "y_t"], ["z_t"], name=f"TBTF_{binary_op}"),
        helper.make_node("Transpose", ["z_t"], ["z"], name=f"TBTF_PostT_{binary_op}", perm=[0, 2, 3, 1]),
        helper.make_node("Add", ["z_t", "s"], ["out_nchw"], name="TBTF_FanoutAdd"),
        helper.make_node("Relu", ["z"], ["out_nhwc"], name="TBTF_Relu"),
    ]
    graph = helper.make_graph(
        nodes,
        f"transpose_{binary_op.lower()}_transpose_fanout_graph",
        [x, y, s],
        [out_nhwc, out_nchw],
    )
    return helper.make_model(graph, opset_imports=[helper.make_operatorsetid("", 13)])


def _make_transpose_binary_no_post_transpose_model(binary_op: str) -> onnx.ModelProto:
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 2, 3, 4])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 2, 3, 4])
    z = helper.make_tensor_value_info("z", TensorProto.FLOAT, [1, 4, 2, 3])
    nodes = [
        helper.make_node("Transpose", ["x"], ["x_t"], name=f"TBNP_PreT0_{binary_op}", perm=[0, 3, 1, 2]),
        helper.make_node("Transpose", ["y"], ["y_t"], name=f"TBNP_PreT1_{binary_op}", perm=[0, 3, 1, 2]),
        helper.make_node(binary_op, ["x_t", "y_t"], ["z_t"], name=f"TBNP_{binary_op}"),
        helper.make_node("Relu", ["z_t"], ["z"], name=f"TBNP_Relu_{binary_op}"),
    ]
    graph = helper.make_graph(
        nodes,
        f"transpose_{binary_op.lower()}_no_post_transpose_graph",
        [x, y],
        [z],
    )
    return helper.make_model(graph, opset_imports=[helper.make_operatorsetid("", 13)])


def _make_transpose_binary_single_side_transpose_model(
    binary_op: str,
    *,
    transpose_on_lhs: bool,
) -> onnx.ModelProto:
    z = helper.make_tensor_value_info("z", TensorProto.FLOAT, [1, 3, 4, 2])

    if transpose_on_lhs:
        x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 3, 4, 2])
        y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 2, 3, 4])
        nodes = [
            helper.make_node("Transpose", ["x"], ["x_t"], name=f"TSB_PreT0_{binary_op}", perm=[0, 3, 1, 2]),
            helper.make_node(binary_op, ["x_t", "y"], ["z_t"], name=f"TSB_{binary_op}"),
            helper.make_node("Transpose", ["z_t"], ["z"], name=f"TSB_PostT_{binary_op}", perm=[0, 2, 3, 1]),
        ]
        inputs = [x, y]
    else:
        x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 2, 3, 4])
        y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 3, 4, 2])
        nodes = [
            helper.make_node("Transpose", ["y"], ["y_t"], name=f"TSB_PreT1_{binary_op}", perm=[0, 3, 1, 2]),
            helper.make_node(binary_op, ["x", "y_t"], ["z_t"], name=f"TSB_{binary_op}"),
            helper.make_node("Transpose", ["z_t"], ["z"], name=f"TSB_PostT_{binary_op}", perm=[0, 2, 3, 1]),
        ]
        inputs = [x, y]

    graph = helper.make_graph(
        nodes,
        f"transpose_single_side_{binary_op.lower()}_transpose_graph",
        inputs,
        [z],
    )
    return helper.make_model(graph, opset_imports=[helper.make_operatorsetid("", 13)])


def _make_qlinear_fc_chain_model() -> onnx.ModelProto:
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 4])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 2])

    x_scale = numpy_helper.from_array(np.asarray([0.1], dtype=np.float32), name="x_scale_fc")
    x_zero = numpy_helper.from_array(np.asarray([0], dtype=np.int8), name="x_zero_fc")
    mul_const = numpy_helper.from_array(np.asarray([2], dtype=np.int8), name="mul_const_fc")
    mul_scale = numpy_helper.from_array(np.asarray([0.1], dtype=np.float32), name="mul_scale_fc")
    mul_zero = numpy_helper.from_array(np.asarray([0], dtype=np.int8), name="mul_zero_fc")
    mul_out_scale = numpy_helper.from_array(np.asarray([0.2], dtype=np.float32), name="mul_out_scale_fc")
    bn_scale = numpy_helper.from_array(np.asarray([1.0, 0.9, 1.1, 1.2], dtype=np.float32), name="bn_scale_fc")
    bn_bias = numpy_helper.from_array(np.asarray([0.0, 0.0, 0.0, 0.0], dtype=np.float32), name="bn_bias_fc")
    bn_mean = numpy_helper.from_array(np.asarray([0.0, 0.0, 0.0, 0.0], dtype=np.float32), name="bn_mean_fc")
    bn_var = numpy_helper.from_array(np.asarray([1.0, 1.0, 1.0, 1.0], dtype=np.float32), name="bn_var_fc")
    prelu_slope = numpy_helper.from_array(np.asarray([0.1, 0.1, 0.1, 0.1], dtype=np.float32), name="prelu_slope_fc")
    q_mid_scale = numpy_helper.from_array(np.asarray([0.2], dtype=np.float32), name="q_mid_scale_fc")
    q_mid_zero = numpy_helper.from_array(np.asarray([0], dtype=np.int8), name="q_mid_zero_fc")
    mm_w = numpy_helper.from_array(
        np.asarray(
            [
                [1, 0],
                [0, 1],
                [1, 0],
                [0, 1],
            ],
            dtype=np.int8,
        ),
        name="mm_w_fc",
    )
    mm_w_scale = numpy_helper.from_array(np.asarray([0.1], dtype=np.float32), name="mm_w_scale_fc")
    mm_w_zero = numpy_helper.from_array(np.asarray([0], dtype=np.int8), name="mm_w_zero_fc")
    mm_out_scale = numpy_helper.from_array(np.asarray([0.25], dtype=np.float32), name="mm_out_scale_fc")
    mm_out_zero = numpy_helper.from_array(np.asarray([0], dtype=np.int8), name="mm_out_zero_fc")
    add_b = numpy_helper.from_array(np.asarray([[1, -1]], dtype=np.int8), name="add_b_fc")
    add_b_scale = numpy_helper.from_array(np.asarray([0.25], dtype=np.float32), name="add_b_scale_fc")

    nodes = [
        helper.make_node("QuantizeLinear", ["x", "x_scale_fc", "x_zero_fc"], ["x_q"], name="Q0FC"),
        helper.make_node(
            "QLinearMul",
            [
                "x_q",
                "x_scale_fc",
                "x_zero_fc",
                "mul_const_fc",
                "mul_scale_fc",
                "mul_zero_fc",
                "mul_out_scale_fc",
                "mul_zero_fc",
            ],
            ["x_mul_q"],
            name="QMulFC",
        ),
        helper.make_node("DequantizeLinear", ["x_mul_q", "mul_out_scale_fc", "mul_zero_fc"], ["x_dq"], name="DQ0FC"),
        helper.make_node(
            "BatchNormalization",
            ["x_dq", "bn_scale_fc", "bn_bias_fc", "bn_mean_fc", "bn_var_fc"],
            ["x_bn"],
            name="BNFC",
            epsilon=1e-5,
        ),
        helper.make_node("PRelu", ["x_bn", "prelu_slope_fc"], ["x_prelu"], name="PReluFC"),
        helper.make_node("QuantizeLinear", ["x_prelu", "q_mid_scale_fc", "q_mid_zero_fc"], ["x_mid_q"], name="Q1FC"),
        helper.make_node(
            "QLinearMatMul",
            [
                "x_mid_q",
                "q_mid_scale_fc",
                "q_mid_zero_fc",
                "mm_w_fc",
                "mm_w_scale_fc",
                "mm_w_zero_fc",
                "mm_out_scale_fc",
                "mm_out_zero_fc",
            ],
            ["mm_q"],
            name="QMMFC",
        ),
        helper.make_node(
            "QLinearAdd",
            [
                "mm_q",
                "mm_out_scale_fc",
                "mm_out_zero_fc",
                "add_b_fc",
                "add_b_scale_fc",
                "mm_out_zero_fc",
                "mm_out_scale_fc",
                "mm_out_zero_fc",
            ],
            ["y_q"],
            name="QAddFC",
        ),
        helper.make_node("DequantizeLinear", ["y_q", "mm_out_scale_fc", "mm_out_zero_fc"], ["y"], name="DQ1FC"),
    ]
    graph = helper.make_graph(
        nodes,
        "qlinear_fc_chain_graph",
        [x],
        [y],
        initializer=[
            x_scale,
            x_zero,
            mul_const,
            mul_scale,
            mul_zero,
            mul_out_scale,
            bn_scale,
            bn_bias,
            bn_mean,
            bn_var,
            prelu_slope,
            q_mid_scale,
            q_mid_zero,
            mm_w,
            mm_w_scale,
            mm_w_zero,
            mm_out_scale,
            mm_out_zero,
            add_b,
            add_b_scale,
        ],
    )
    return helper.make_model(graph, opset_imports=[helper.make_operatorsetid("", 13)])


def _make_space_to_depth_chain_model() -> onnx.ModelProto:
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 2, 4, 4])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 8, 2, 2])
    shape1 = numpy_helper.from_array(np.array([1, 2, 2, 2, 2, 2], dtype=np.int64), name="shape1")
    shape2 = numpy_helper.from_array(np.array([1, 8, 2, 2], dtype=np.int64), name="shape2")
    n0 = helper.make_node("Reshape", ["x", "shape1"], ["r1"], name="ReshapeNode0")
    n1 = helper.make_node("Transpose", ["r1"], ["t1"], name="TransposeNode", perm=[0, 1, 3, 5, 2, 4])
    n2 = helper.make_node("Reshape", ["t1", "shape2"], ["y"], name="ReshapeNode1")
    graph = helper.make_graph([n0, n1, n2], "space_to_depth_chain_graph", [x], [y], initializer=[shape1, shape2])
    return helper.make_model(graph, opset_imports=[helper.make_operatorsetid("", 13)])


def _make_transpose_attr_only_model() -> onnx.ModelProto:
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 2, 3])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 3, 2])
    node = helper.make_node("Transpose", ["x"], ["y"], name="TransposeAttrOnlyNode", perm=[0, 2, 1])
    graph = helper.make_graph([node], "transpose_attr_only_graph", [x], [y])
    return helper.make_model(graph, opset_imports=[helper.make_operatorsetid("", 13)])


def _make_reshape_minus_one_model() -> onnx.ModelProto:
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 2, 3])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [3, 2])
    shape = numpy_helper.from_array(np.array([-1, 2], dtype=np.int64), name="reshape_shape")
    node = helper.make_node("Reshape", ["x", "reshape_shape"], ["y"], name="ReshapeMinusOneNode")
    graph = helper.make_graph([node], "reshape_minus_one_graph", [x], [y], initializer=[shape])
    return helper.make_model(graph, opset_imports=[helper.make_operatorsetid("", 13)])


def _make_softmax_axis_non_last_model() -> onnx.ModelProto:
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 3, 4])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 3, 4])
    node = helper.make_node("Softmax", ["x"], ["y"], name="SoftmaxAxisNonLastNode", axis=1)
    graph = helper.make_graph([node], "softmax_axis_non_last_graph", [x], [y])
    return helper.make_model(graph, opset_imports=[helper.make_operatorsetid("", 13)])


def _make_logsoftmax_model() -> onnx.ModelProto:
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 3, 4])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 3, 4])
    node = helper.make_node("LogSoftmax", ["x"], ["y"], name="LogSoftmaxNode", axis=2)
    graph = helper.make_graph([node], "logsoftmax_graph", [x], [y])
    return helper.make_model(graph, opset_imports=[helper.make_operatorsetid("", 13)])


def _make_softmax_default_axis_opset13_model() -> onnx.ModelProto:
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 3, 4])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 3, 4])
    node = helper.make_node("Softmax", ["x"], ["y"], name="SoftmaxDefaultAxisOpset13Node")
    graph = helper.make_graph([node], "softmax_default_axis_opset13_graph", [x], [y])
    return helper.make_model(graph, opset_imports=[helper.make_operatorsetid("", 13)])


def _make_softmax_default_axis_opset11_model() -> onnx.ModelProto:
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 3, 4])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 3, 4])
    node = helper.make_node("Softmax", ["x"], ["y"], name="SoftmaxDefaultAxisOpset11Node")
    graph = helper.make_graph([node], "softmax_default_axis_opset11_graph", [x], [y])
    return helper.make_model(graph, opset_imports=[helper.make_operatorsetid("", 11)])


def _make_reduce_axes_concat_const_model() -> onnx.ModelProto:
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 2, 3])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 1, 3])
    c0 = numpy_helper.from_array(np.array([1], dtype=np.int64), name="c0")
    c1 = numpy_helper.from_array(np.array([], dtype=np.int64), name="c1")
    n0 = helper.make_node("Concat", ["c0", "c1"], ["axes"], name="AxesConcatNode", axis=0)
    n1 = helper.make_node("ReduceSum", ["x", "axes"], ["y"], name="ReduceConstFoldNode", keepdims=1)
    graph = helper.make_graph([n0, n1], "reduce_axes_concat_const_graph", [x], [y], initializer=[c0, c1])
    return helper.make_model(graph, opset_imports=[helper.make_operatorsetid("", 13)])


def _make_reduce_mean_model() -> onnx.ModelProto:
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 2, 3])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 2, 1])
    axes = numpy_helper.from_array(np.array([2], dtype=np.int64), name="axes")
    node = helper.make_node(
        "ReduceMean",
        ["x", "axes"],
        ["y"],
        name="ReduceMeanNode",
        keepdims=1,
    )
    graph = helper.make_graph([node], "reduce_mean_graph", [x], [y], initializer=[axes])
    return helper.make_model(graph, opset_imports=[helper.make_operatorsetid("", 13)])


def _make_reduce_sum_model() -> onnx.ModelProto:
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 2, 3])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 1, 3])
    axes = numpy_helper.from_array(np.array([1], dtype=np.int64), name="axes")
    node = helper.make_node(
        "ReduceSum",
        ["x", "axes"],
        ["y"],
        name="ReduceSumNode",
        keepdims=1,
    )
    graph = helper.make_graph([node], "reduce_sum_graph", [x], [y], initializer=[axes])
    return helper.make_model(graph, opset_imports=[helper.make_operatorsetid("", 13)])


def _make_reduce_prod_model() -> onnx.ModelProto:
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 2, 3])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 1, 3])
    axes = numpy_helper.from_array(np.array([1], dtype=np.int64), name="axes")
    node = helper.make_node(
        "ReduceProd",
        ["x", "axes"],
        ["y"],
        name="ReduceProdNode",
        keepdims=1,
    )
    graph = helper.make_graph([node], "reduce_prod_graph", [x], [y], initializer=[axes])
    return helper.make_model(graph, opset_imports=[helper.make_operatorsetid("", 13)])


def _make_squeeze_model() -> onnx.ModelProto:
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 1, 3])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 3])
    axes = numpy_helper.from_array(np.array([1], dtype=np.int64), name="axes")
    node = helper.make_node("Squeeze", ["x", "axes"], ["y"], name="SqueezeNode")
    graph = helper.make_graph([node], "squeeze_graph", [x], [y], initializer=[axes])
    return helper.make_model(graph, opset_imports=[helper.make_operatorsetid("", 13)])


def _make_unsqueeze_model() -> onnx.ModelProto:
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 3])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 1, 3])
    axes = numpy_helper.from_array(np.array([1], dtype=np.int64), name="axes")
    node = helper.make_node("Unsqueeze", ["x", "axes"], ["y"], name="UnsqueezeNode")
    graph = helper.make_graph([node], "unsqueeze_graph", [x], [y], initializer=[axes])
    return helper.make_model(graph, opset_imports=[helper.make_operatorsetid("", 13)])


def _make_unsqueeze_dynamic_axis2_model() -> onnx.ModelProto:
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, "N"])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, "N", 1])
    axes = numpy_helper.from_array(np.array([2], dtype=np.int64), name="unsq_dyn_axes")
    node = helper.make_node("Unsqueeze", ["x", "unsq_dyn_axes"], ["y"], name="UnsqueezeDynamicAxis2Node")
    graph = helper.make_graph(
        [node],
        "unsqueeze_dynamic_axis2_graph",
        [x],
        [y],
        initializer=[axes],
    )
    return helper.make_model(graph, opset_imports=[helper.make_operatorsetid("", 13)])


def _make_unsqueeze_axes_2_3_model() -> onnx.ModelProto:
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 3])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 3, 1, 1])
    axes = numpy_helper.from_array(np.array([2, 3], dtype=np.int64), name="unsq_axes_2_3")
    node = helper.make_node("Unsqueeze", ["x", "unsq_axes_2_3"], ["y"], name="UnsqueezeAxes23Node")
    graph = helper.make_graph(
        [node],
        "unsqueeze_axes_2_3_graph",
        [x],
        [y],
        initializer=[axes],
    )
    return helper.make_model(graph, opset_imports=[helper.make_operatorsetid("", 13)])


def _make_gather_model() -> onnx.ModelProto:
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [2, 4])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [2, 2])
    indices = numpy_helper.from_array(np.array([3, 1], dtype=np.int64), name="indices")
    node = helper.make_node("Gather", ["x", "indices"], ["y"], name="GatherNode", axis=1)
    graph = helper.make_graph([node], "gather_graph", [x], [y], initializer=[indices])
    return helper.make_model(graph, opset_imports=[helper.make_operatorsetid("", 13)])


def _make_gather_singleton_indices_rank_reduced_model() -> onnx.ModelProto:
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 3, 4])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 3])
    indices = numpy_helper.from_array(np.array([0], dtype=np.int64), name="indices")
    node = helper.make_node(
        "Gather",
        ["x", "indices"],
        ["y"],
        name="GatherSingletonIndicesNode",
        axis=2,
    )
    graph = helper.make_graph(
        [node],
        "gather_singleton_indices_rank_reduced_graph",
        [x],
        [y],
        initializer=[indices],
    )
    return helper.make_model(graph, opset_imports=[helper.make_operatorsetid("", 13)])


def _make_gather_singleton_negative_indices_rank_reduced_model() -> onnx.ModelProto:
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 3, 4])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 3])
    indices = numpy_helper.from_array(np.array([-1], dtype=np.int64), name="indices")
    node = helper.make_node(
        "Gather",
        ["x", "indices"],
        ["y"],
        name="GatherSingletonNegativeIndicesNode",
        axis=2,
    )
    graph = helper.make_graph(
        [node],
        "gather_singleton_negative_indices_rank_reduced_graph",
        [x],
        [y],
        initializer=[indices],
    )
    return helper.make_model(graph, opset_imports=[helper.make_operatorsetid("", 13)])


def _make_l2_norm_model() -> onnx.ModelProto:
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 4])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 4])
    node = helper.make_node(
        "LpNormalization",
        ["x"],
        ["y"],
        name="LpNormNode",
        axis=-1,
        p=2,
    )
    graph = helper.make_graph([node], "l2norm_graph", [x], [y])
    return helper.make_model(graph, opset_imports=[helper.make_operatorsetid("", 13)])


def _make_lrn_model() -> onnx.ModelProto:
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 3, 5, 5])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 3, 5, 5])
    node = helper.make_node(
        "LRN",
        ["x"],
        ["y"],
        name="LRNNode",
        size=3,
        alpha=1e-4,
        beta=0.75,
        bias=1.0,
    )
    graph = helper.make_graph([node], "lrn_graph", [x], [y])
    return helper.make_model(graph, opset_imports=[helper.make_operatorsetid("", 13)])


def _make_instance_normalization_model() -> onnx.ModelProto:
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 3, 5, 5])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 3, 5, 5])
    scale = numpy_helper.from_array(
        np.asarray([1.0, 0.5, 2.0], dtype=np.float32),
        name="inst_scale",
    )
    bias = numpy_helper.from_array(
        np.asarray([0.1, -0.2, 0.3], dtype=np.float32),
        name="inst_bias",
    )
    node = helper.make_node(
        "InstanceNormalization",
        ["x", "inst_scale", "inst_bias"],
        ["y"],
        name="InstanceNormNode",
        epsilon=1e-5,
    )
    graph = helper.make_graph(
        [node],
        "instance_norm_graph",
        [x],
        [y],
        initializer=[scale, bias],
    )
    return helper.make_model(graph, opset_imports=[helper.make_operatorsetid("", 13)])


def _make_gather_int32_model() -> onnx.ModelProto:
    x = helper.make_tensor_value_info("x", TensorProto.INT32, [2, 4])
    y = helper.make_tensor_value_info("y", TensorProto.INT32, [2, 2])
    indices = numpy_helper.from_array(np.array([3, 1], dtype=np.int64), name="indices")
    node = helper.make_node("Gather", ["x", "indices"], ["y"], name="GatherIntNode", axis=1)
    graph = helper.make_graph([node], "gather_int_graph", [x], [y], initializer=[indices])
    return helper.make_model(graph, opset_imports=[helper.make_operatorsetid("", 13)])


def _make_gemm_reduce_model() -> onnx.ModelProto:
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 4])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 1])
    w = numpy_helper.from_array(np.ones((3, 4), dtype=np.float32), name="W")
    b = numpy_helper.from_array(np.zeros((3,), dtype=np.float32), name="B")
    axes = numpy_helper.from_array(np.array([1], dtype=np.int64), name="axes")
    gemm = helper.make_node("Gemm", ["x", "W", "B"], ["h"], name="GemmNode", transB=1)
    relu = helper.make_node("Relu", ["h"], ["r"], name="ReluNode")
    reduce = helper.make_node("ReduceMean", ["r", "axes"], ["y"], name="ReduceNode", keepdims=1)
    graph = helper.make_graph([gemm, relu, reduce], "gemm_reduce_graph", [x], [y], initializer=[w, b, axes])
    return helper.make_model(graph, opset_imports=[helper.make_operatorsetid("", 13)])


def _make_shared_input_conv_and_mul_branches_model() -> onnx.ModelProto:
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 3, 8, 8])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 3, 8, 8])
    w0 = numpy_helper.from_array(
        np.ones((3, 3, 1, 1), dtype=np.float32),
        name="sib_w0",
    )
    b0 = numpy_helper.from_array(np.zeros((3,), dtype=np.float32), name="sib_b0")
    scale = numpy_helper.from_array(np.asarray(0.5, dtype=np.float32), name="sib_scale")
    nodes = [
        helper.make_node("Conv", ["x", "sib_w0", "sib_b0"], ["y0"], name="SIB_Conv0"),
        helper.make_node("Mul", ["x", "sib_scale"], ["y1"], name="SIB_Mul1"),
        helper.make_node("Add", ["y0", "y1"], ["y"], name="SIB_Add"),
    ]
    graph = helper.make_graph(
        nodes,
        "shared_input_conv_and_mul_branches_graph",
        [x],
        [y],
        initializer=[w0, b0, scale],
    )
    return helper.make_model(graph, opset_imports=[helper.make_operatorsetid("", 13)])


def _make_einsum_model() -> onnx.ModelProto:
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 3])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [3, 2])
    z = helper.make_tensor_value_info("z", TensorProto.FLOAT, [1, 2])
    node = helper.make_node(
        "Einsum",
        ["x", "y"],
        ["z"],
        name="EinsumNode",
        equation="ij,jk->ik",
    )
    graph = helper.make_graph([node], "einsum_graph", [x, y], [z])
    return helper.make_model(graph, opset_imports=[helper.make_operatorsetid("", 13)])


def _make_einsum_custom_model() -> onnx.ModelProto:
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 3])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [3, 2])
    z = helper.make_tensor_value_info("z", TensorProto.FLOAT, [2, 3])
    node = helper.make_node(
        "Einsum",
        ["x", "y"],
        ["z"],
        name="EinsumCustomNode",
        equation="ij,jk->kj",
    )
    graph = helper.make_graph([node], "einsum_custom_graph", [x, y], [z])
    return helper.make_model(graph, opset_imports=[helper.make_operatorsetid("", 13)])


def _make_einsum_fc_const_model() -> onnx.ModelProto:
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 3])
    z = helper.make_tensor_value_info("z", TensorProto.FLOAT, [1, 2])
    w = numpy_helper.from_array(
        np.array(
            [
                [1.0, 2.0],
                [3.0, 4.0],
                [5.0, 6.0],
            ],
            dtype=np.float32,
        ),
        name="W",
    )
    node = helper.make_node(
        "Einsum",
        ["x", "W"],
        ["z"],
        name="EinsumConstNode",
        equation="ij,jk->ik",
    )
    graph = helper.make_graph([node], "einsum_fc_const_graph", [x], [z], initializer=[w])
    return helper.make_model(graph, opset_imports=[helper.make_operatorsetid("", 13)])


def _requires_flatbuffer_tools() -> bool:
    return shutil.which("flatc") is not None and shutil.which("curl") is not None


@contextmanager
def _temporary_env(updates: dict[str, str]):
    previous = {}
    for key, value in updates.items():
        previous[key] = os.environ.get(key, None)
        os.environ[key] = value
    try:
        yield
    finally:
        for key, old_value in previous.items():
            if old_value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = old_value


def _collect_int8_quant_scale_lengths(tflite_path: str) -> list[int]:
    output_dir = os.path.dirname(tflite_path)
    schema = load_schema_module(output_dir)
    with open(tflite_path, "rb") as f:
        model_bytes = f.read()
    model = schema["ModelT"].InitFromObj(schema["Model"].GetRootAs(model_bytes, 0))
    subgraph = model.subgraphs[0]
    int8_type = getattr(schema["TensorType"], "INT8")
    lengths: list[int] = []
    for tensor in subgraph.tensors:
        if tensor.type != int8_type:
            continue
        if tensor.quantization is None or tensor.quantization.scale is None:
            continue
        lengths.append(len(list(tensor.quantization.scale)))
    return lengths


def _collect_custom_codes(tflite_path: str) -> list[str]:
    output_dir = os.path.dirname(tflite_path)
    schema = load_schema_module(output_dir)
    with open(tflite_path, "rb") as f:
        model_bytes = f.read()
    model = schema["ModelT"].InitFromObj(schema["Model"].GetRootAs(model_bytes, 0))
    builtin_custom = getattr(schema["BuiltinOperator"], "CUSTOM")
    custom_codes: list[str] = []
    for op_code in model.operatorCodes:
        if int(op_code.builtinCode) == int(builtin_custom):
            value = op_code.customCode
            if isinstance(value, bytes):
                custom_codes.append(value.decode("utf-8"))
            else:
                custom_codes.append(str(value))
    return custom_codes


def _collect_builtin_op_names(tflite_path: str) -> list[str]:
    output_dir = os.path.dirname(tflite_path)
    schema = load_schema_module(output_dir)
    with open(tflite_path, "rb") as f:
        model_bytes = f.read()
    model = schema["ModelT"].InitFromObj(schema["Model"].GetRootAs(model_bytes, 0))
    subgraph = model.subgraphs[0]
    op_enum = schema["BuiltinOperator"]
    op_names: list[str] = []
    for op in subgraph.operators:
        op_code = model.operatorCodes[int(op.opcodeIndex)]
        builtin_code = int(op_code.builtinCode)
        resolved = str(builtin_code)
        for attr in dir(op_enum):
            if not attr.isupper():
                continue
            if int(getattr(op_enum, attr)) == builtin_code:
                resolved = attr
                break
        op_names.append(resolved)
    return op_names


def test_tflite_backend_matrix_add() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        model = _make_add_model()
        model_path = _save_model(tmpdir, "add", model)

        tf_out = os.path.join(tmpdir, "tf_converter")
        tf_tflite = _convert(model_path, tf_out, "tf_converter")
        tf_pred = _run_add_inference(tf_tflite)

        if not _requires_flatbuffer_tools():
            pytest.skip("flatbuffer_direct requires flatc and curl")

        fb_out = os.path.join(tmpdir, "flatbuffer_direct")
        fb_tflite = _convert(model_path, fb_out, "flatbuffer_direct")
        fb_pred = _run_add_inference(fb_tflite)

        np.testing.assert_allclose(tf_pred, np.array([[5.0, 7.0, 9.0]], dtype=np.float32), rtol=0.0, atol=1e-6)
        np.testing.assert_allclose(fb_pred, tf_pred, rtol=0.0, atol=1e-6)


@pytest.mark.skipif(not _requires_flatbuffer_tools(), reason="flatbuffer_direct requires flatc and curl")
def test_tflite_backend_matrix_hardswish_rewrite_on_off(monkeypatch: pytest.MonkeyPatch) -> None:
    import onnx2tf.tflite_builder as tflite_builder_backend
    from onnx2tf.tflite_builder.preprocess import clear_preprocess_rules

    with tempfile.TemporaryDirectory() as tmpdir:
        model = _make_hardswish_model()
        model_path = _save_model(tmpdir, "hardswish_rewrite_matrix", model)

        # tf_converter backend should pass without direct preprocess dependency.
        tf_out = os.path.join(tmpdir, "tf_converter")
        tf_tflite = _convert(model_path, tf_out, "tf_converter")
        assert os.path.isfile(tf_tflite)

        # flatbuffer_direct with default preprocess should pass (rewrite enabled).
        fb_out = os.path.join(tmpdir, "flatbuffer_direct")
        fb_tflite = _convert(model_path, fb_out, "flatbuffer_direct")
        assert os.path.isfile(fb_tflite)

        # HardSwish is now lowered as builtin even when rewrite preprocess is disabled.
        clear_preprocess_rules()
        monkeypatch.setattr(
            tflite_builder_backend,
            "register_default_preprocess_rules",
            lambda: None,
        )
        fb_no_rewrite_out = os.path.join(tmpdir, "flatbuffer_direct_no_rewrite")
        _convert(
            model_path,
            fb_no_rewrite_out,
            "flatbuffer_direct",
            report_op_coverage=True,
        )
        fb_no_rewrite_tflite = os.path.join(
            fb_no_rewrite_out,
            "hardswish_rewrite_matrix_float32.tflite",
        )
        assert os.path.isfile(fb_no_rewrite_tflite)
        builtin_names = _collect_builtin_op_names(fb_no_rewrite_tflite)
        assert "HARD_SWISH" in builtin_names
        report_path = os.path.join(
            fb_no_rewrite_out,
            "hardswish_rewrite_matrix_op_coverage_report.json",
        )
        assert os.path.isfile(report_path)
        with open(report_path, "r", encoding="utf-8") as f:
            report = json.load(f)
        assert report["conversion_error"] is None
        assert report["graph_summary"]["unsupported_nodes"] == 0


@pytest.mark.skipif(not _requires_flatbuffer_tools(), reason="flatbuffer_direct requires flatc and curl")
@pytest.mark.parametrize(
    "name, model_factory",
    [
        ("conv", _make_conv_model),
        ("conv1d", _make_conv1d_model),
        ("conv3d", _make_conv3d_model),
        ("convtranspose1d", _make_convtranspose1d_model),
        ("convtranspose2d_output_padding", _make_convtranspose2d_output_padding_model),
        ("convtranspose3d", _make_convtranspose3d_model),
        ("pool", _make_pool_model),
        ("global_max_pool", _make_global_max_pool_model),
        ("gemm", _make_gemm_model),
        ("reduce_mean", _make_reduce_mean_model),
        ("reduce_sum", _make_reduce_sum_model),
        ("reduce_prod", _make_reduce_prod_model),
        ("squeeze", _make_squeeze_model),
        ("unsqueeze", _make_unsqueeze_model),
        ("gather", _make_gather_model),
        ("l2_norm", _make_l2_norm_model),
        ("lrn", _make_lrn_model),
        ("dropout", _make_dropout_model),
        ("instance_norm", _make_instance_normalization_model),
        ("relu", lambda: _make_unary_model("Relu", name="relu")),
        ("tanh", lambda: _make_unary_model("Tanh", name="tanh")),
        ("atan", lambda: _make_unary_model("Atan", name="atan")),
        ("exp", lambda: _make_unary_model("Exp", name="exp")),
        ("sqrt", lambda: _make_unary_model("Sqrt", name="sqrt")),
        ("neg", lambda: _make_unary_model("Neg", name="neg")),
        ("hardswish", _make_hardswish_model),
        ("leakyrelu", _make_leakyrelu_model),
        ("prelu", _make_prelu_model),
        ("gelu", _make_gelu_model),
        ("pow_square", _make_pow_square_model),
        ("dynamic_quantize_linear", _make_dynamic_quantize_linear_model),
        ("shape_slice", _make_shape_slice_model),
        ("fused_matmul", _make_fused_matmul_model),
        ("space_to_depth", _make_space_to_depth_model),
        ("depth_to_space", _make_depth_to_space_model),
        ("depth_to_space_crd", _make_depth_to_space_crd_model),
        ("resize_nearest", _make_resize_nearest_model),
        ("qlinear_sigmoid", _make_qlinear_sigmoid_model),
        ("qlinear_global_average_pool", _make_qlinear_global_average_pool_model),
        ("qlinear_concat", _make_qlinear_concat_model),
        ("space_to_depth_chain", _make_space_to_depth_chain_model),
        ("qlinear_conv_chain", _make_qlinear_conv_chain_model),
        ("qlinear_fc_chain", _make_qlinear_fc_chain_model),
        ("transpose_attr_only", _make_transpose_attr_only_model),
        ("softmax_axis_non_last", _make_softmax_axis_non_last_model),
        ("logsoftmax", _make_logsoftmax_model),
        ("reduce_axes_concat_const", _make_reduce_axes_concat_const_model),
        ("einsum_fc_const", _make_einsum_fc_const_model),
    ],
)
def test_flatbuffer_direct_operator_smoke(name: str, model_factory) -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        model = model_factory()
        model_path = _save_model(tmpdir, name, model)
        out_dir = os.path.join(tmpdir, "out")
        tflite_path = _convert(model_path, out_dir, "flatbuffer_direct")
        if name == "qlinear_fc_chain":
            custom_codes = _collect_custom_codes(tflite_path)
            assert "ONNX_QLINEARMATMUL" in custom_codes
            return
        interpreter = Interpreter(model_path=tflite_path)
        interpreter.allocate_tensors()
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        x = np.ones(input_details[0]["shape"], dtype=np.float32)
        interpreter.set_tensor(input_details[0]["index"], x)
        interpreter.invoke()
        _ = interpreter.get_tensor(output_details[0]["index"])


@pytest.mark.skipif(not _requires_flatbuffer_tools(), reason="flatbuffer_direct requires flatc and curl")
def test_flatbuffer_direct_conv_fp16_generates_op_error_report() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        model = _make_conv_fp16_model()
        model_path = _save_model(tmpdir, "conv_fp16", model)
        out_dir = os.path.join(tmpdir, "out")
        tflite_path = _convert(
            model_path,
            out_dir,
            "flatbuffer_direct",
            report_op_coverage=True,
        )
        assert os.path.isfile(tflite_path)
        report_path = os.path.join(out_dir, "conv_fp16_op_coverage_report.json")
        assert os.path.isfile(report_path)

        interpreter = Interpreter(model_path=tflite_path)
        interpreter.allocate_tensors()
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        x = np.ones(input_details[0]["shape"], dtype=input_details[0]["dtype"])
        interpreter.set_tensor(input_details[0]["index"], x)
        interpreter.invoke()
        _ = interpreter.get_tensor(output_details[0]["index"])


def test_flatbuffer_direct_transpose_chain_optimization() -> None:
    model = _make_qlinear_conv_pair_model()
    register_default_preprocess_rules()
    preprocessed_model, _ = run_preprocess_pipeline(onnx_graph=model)
    model_ir = lower_onnx_to_ir(
        onnx_graph=preprocessed_model,
        output_file_name="transpose_opt_test",
    )
    op_types = [str(op.op_type) for op in model_ir.operators]
    assert op_types.count("TRANSPOSE") == 0
    assert op_types.count("RESHAPE") >= 1


def test_flatbuffer_direct_layout_transpose_chain_removes_inverse_fanout_branch() -> None:
    model_ir = ModelIR(name="layout_transpose_inverse_fanout_branch_test")
    model_ir.inputs = ["x"]
    model_ir.outputs = ["y_nchw", "z"]
    model_ir.tensors["x"] = TensorIR(
        name="x",
        dtype="FLOAT32",
        shape=[1, 4, 4, 3],
        shape_signature=[1, 4, 4, 3],
    )
    model_ir.tensors["pre_perm"] = TensorIR(
        name="pre_perm",
        dtype="INT32",
        shape=[4],
        shape_signature=[4],
        data=np.asarray([0, 3, 1, 2], dtype=np.int32),
        is_variable=False,
    )
    model_ir.tensors["post_perm"] = TensorIR(
        name="post_perm",
        dtype="INT32",
        shape=[4],
        shape_signature=[4],
        data=np.asarray([0, 2, 3, 1], dtype=np.int32),
        is_variable=False,
    )
    model_ir.tensors["x_nchw"] = TensorIR(
        name="x_nchw",
        dtype="FLOAT32",
        shape=[1, 3, 4, 4],
        shape_signature=[1, 3, 4, 4],
    )
    model_ir.tensors["y_nchw"] = TensorIR(
        name="y_nchw",
        dtype="FLOAT32",
        shape=[1, 3, 4, 4],
        shape_signature=[1, 3, 4, 4],
    )
    model_ir.tensors["y_nhwc"] = TensorIR(
        name="y_nhwc",
        dtype="FLOAT32",
        shape=[1, 4, 4, 3],
        shape_signature=[1, 4, 4, 3],
    )
    model_ir.tensors["z"] = TensorIR(
        name="z",
        dtype="FLOAT32",
        shape=[1, 4, 4, 3],
        shape_signature=[1, 4, 4, 3],
    )
    model_ir.operators = [
        OperatorIR(op_type="TRANSPOSE", inputs=["x", "pre_perm"], outputs=["x_nchw"]),
        OperatorIR(op_type="RELU", inputs=["x_nchw"], outputs=["y_nchw"]),
        OperatorIR(op_type="TRANSPOSE", inputs=["x_nchw", "post_perm"], outputs=["y_nhwc"]),
        OperatorIR(op_type="RELU", inputs=["y_nhwc"], outputs=["z"]),
    ]

    stats = _optimize_layout_transpose_chains(model_ir)
    assert stats["removed_inverse_transpose_fanout_branches"] == 1
    op_types = [str(op.op_type) for op in model_ir.operators]
    assert op_types.count("TRANSPOSE") == 1
    z_relu = next(op for op in model_ir.operators if str(op.op_type) == "RELU" and list(op.outputs) == ["z"])
    assert list(z_relu.inputs) == ["x"]


def test_flatbuffer_direct_qlinear_conv_filter_layout_is_ohwi() -> None:
    model = _make_qlinear_conv_multichannel_model()
    register_default_preprocess_rules()
    preprocessed_model, _ = run_preprocess_pipeline(onnx_graph=model)
    model_ir = lower_onnx_to_ir(
        onnx_graph=preprocessed_model,
        output_file_name="qlinear_conv_multichannel_layout_test",
        optimize_layout_transpose_chains=False,
    )

    w_tensor = model_ir.tensors.get("QConvMC_conv_filter_q")
    assert w_tensor is not None
    assert list(w_tensor.shape) == [4, 1, 1, 3]


def test_flatbuffer_direct_input_layout_defaults_to_nhwc_when_enabled() -> None:
    model = _make_qlinear_conv_chain_model()
    register_default_preprocess_rules()
    preprocessed_model, _ = run_preprocess_pipeline(onnx_graph=model)
    model_ir = lower_onnx_to_ir(
        onnx_graph=preprocessed_model,
        output_file_name="input_layout_default_nhwc_test",
        transpose_inputs_to_nhwc=True,
    )
    assert model_ir.inputs == ["x"]
    assert list(model_ir.tensors["x"].shape) == [1, 2, 2, 1]


def test_flatbuffer_direct_keeps_input_layout_contract_for_boundary_transpose() -> None:
    model = _make_input_slice_model()
    register_default_preprocess_rules()
    preprocessed_model, _ = run_preprocess_pipeline(onnx_graph=model)
    model_ir = lower_onnx_to_ir(
        onnx_graph=preprocessed_model,
        output_file_name="input_boundary_transpose_elision_test",
        transpose_inputs_to_nhwc=True,
    )
    assert model_ir.inputs == ["x"]
    assert list(model_ir.tensors["x"].shape) == [1, 4, 4, 3]
    assert "x_onnx_ncx_internal" not in model_ir.tensors
    assert not any(
        str(op.op_type) == "TRANSPOSE"
        and "x_onnx_ncx_internal" in [str(output_name) for output_name in op.outputs]
        for op in model_ir.operators
    )


def test_flatbuffer_direct_boundary_input_transpose_channel_slice_block_elides_op0() -> None:
    model_ir = ModelIR(name="boundary_input_transpose_channel_slice_block_test")
    model_ir.inputs = ["x"]
    model_ir.outputs = ["y", "z"]
    model_ir.tensors["x"] = TensorIR(
        name="x",
        dtype="FLOAT32",
        shape=[1, 2, 2, 3],
        shape_signature=[1, 2, 2, 3],
    )
    model_ir.tensors["x_onnx_ncx_internal_perm"] = TensorIR(
        name="x_onnx_ncx_internal_perm",
        dtype="INT32",
        shape=[4],
        shape_signature=[4],
        data=np.asarray([0, 3, 1, 2], dtype=np.int32),
        is_variable=False,
    )
    model_ir.tensors["x_onnx_ncx_internal"] = TensorIR(
        name="x_onnx_ncx_internal",
        dtype="FLOAT32",
        shape=[1, 3, 2, 2],
        shape_signature=[1, 3, 2, 2],
    )
    for channel_idx in range(3):
        model_ir.tensors[f"b{channel_idx}"] = TensorIR(
            name=f"b{channel_idx}",
            dtype="INT32",
            shape=[4],
            shape_signature=[4],
            data=np.asarray([0, channel_idx, 0, 0], dtype=np.int32),
            is_variable=False,
        )
        model_ir.tensors[f"s{channel_idx}"] = TensorIR(
            name=f"s{channel_idx}",
            dtype="INT32",
            shape=[4],
            shape_signature=[4],
            data=np.asarray([1, 1, 2, 2], dtype=np.int32),
            is_variable=False,
        )
        model_ir.tensors[f"c{channel_idx}"] = TensorIR(
            name=f"c{channel_idx}",
            dtype="FLOAT32",
            shape=[1, 1, 2, 2],
            shape_signature=[1, 1, 2, 2],
        )
    model_ir.tensors["cat"] = TensorIR(
        name="cat",
        dtype="FLOAT32",
        shape=[1, 3, 2, 2],
        shape_signature=[1, 3, 2, 2],
    )
    model_ir.tensors["split_b"] = TensorIR(
        name="split_b",
        dtype="INT32",
        shape=[4],
        shape_signature=[4],
        data=np.asarray([0, 1, 0, 0], dtype=np.int32),
        is_variable=False,
    )
    model_ir.tensors["split_s"] = TensorIR(
        name="split_s",
        dtype="INT32",
        shape=[4],
        shape_signature=[4],
        data=np.asarray([1, 1, 2, 2], dtype=np.int32),
        is_variable=False,
    )
    model_ir.tensors["split1"] = TensorIR(
        name="split1",
        dtype="FLOAT32",
        shape=[1, 1, 2, 2],
        shape_signature=[1, 1, 2, 2],
    )
    model_ir.tensors["y"] = TensorIR(
        name="y",
        dtype="FLOAT32",
        shape=[1, 3, 2, 2],
        shape_signature=[1, 3, 2, 2],
    )
    model_ir.tensors["z"] = TensorIR(
        name="z",
        dtype="FLOAT32",
        shape=[1, 1, 2, 2],
        shape_signature=[1, 1, 2, 2],
    )
    model_ir.operators = [
        OperatorIR(
            op_type="TRANSPOSE",
            inputs=["x", "x_onnx_ncx_internal_perm"],
            outputs=["x_onnx_ncx_internal"],
        ),
        OperatorIR(op_type="SLICE", inputs=["x_onnx_ncx_internal", "b0", "s0"], outputs=["c0"]),
        OperatorIR(op_type="SLICE", inputs=["x_onnx_ncx_internal", "b1", "s1"], outputs=["c1"]),
        OperatorIR(op_type="SLICE", inputs=["x_onnx_ncx_internal", "b2", "s2"], outputs=["c2"]),
        OperatorIR(
            op_type="CONCATENATION",
            inputs=["c0", "c1", "c2"],
            outputs=["cat"],
            options={"axis": 1, "fusedActivationFunction": "NONE"},
        ),
        OperatorIR(op_type="SLICE", inputs=["cat", "split_b", "split_s"], outputs=["split1"]),
        OperatorIR(op_type="RELU", inputs=["x_onnx_ncx_internal"], outputs=["y"]),
        OperatorIR(op_type="RELU", inputs=["split1"], outputs=["z"]),
    ]

    stats = _optimize_boundary_input_transpose_channel_slice_blocks(model_ir)
    assert stats["removed_boundary_input_transpose"] == 1

    assert model_ir.inputs == ["x"]
    assert list(model_ir.tensors["x"].shape) == [1, 2, 2, 3]
    assert str(model_ir.operators[0].op_type) == "SLICE"

    concat_op = next(op for op in model_ir.operators if str(op.op_type) == "CONCATENATION")
    assert int(concat_op.options.get("axis", -1)) == 3

    assert np.array_equal(
        np.asarray(model_ir.tensors["b1"].data),
        np.asarray([0, 0, 0, 1], dtype=np.int32),
    )
    assert np.array_equal(
        np.asarray(model_ir.tensors["s1"].data),
        np.asarray([1, 2, 2, 1], dtype=np.int32),
    )
    assert np.array_equal(
        np.asarray(model_ir.tensors["split_b"].data),
        np.asarray([0, 0, 0, 1], dtype=np.int32),
    )
    assert np.array_equal(
        np.asarray(model_ir.tensors["split_s"].data),
        np.asarray([1, 2, 2, 1], dtype=np.int32),
    )

    assert not any(
        str(op.op_type) == "TRANSPOSE"
        and "x_onnx_ncx_internal" in [str(v) for v in list(op.outputs)]
        for op in model_ir.operators
    )
    assert any(
        str(op.op_type) == "TRANSPOSE"
        and any(str(v).startswith("x_onnx_ncx_internal_local") for v in list(op.outputs))
        for op in model_ir.operators
    )


def test_flatbuffer_direct_boundary_input_slice_depthwise_nhwc_propagation_avoids_bridge_transpose() -> None:
    model_ir = ModelIR(name="boundary_input_slice_depthwise_nhwc_propagation_test")
    model_ir.inputs = ["x"]
    model_ir.outputs = ["y"]
    model_ir.tensors["x"] = TensorIR(
        name="x",
        dtype="FLOAT32",
        shape=[1, 4, 4, 2],
        shape_signature=[1, 4, 4, 2],
    )
    model_ir.tensors["x_onnx_ncx_internal_perm"] = TensorIR(
        name="x_onnx_ncx_internal_perm",
        dtype="INT32",
        shape=[4],
        shape_signature=[4],
        data=np.asarray([0, 3, 1, 2], dtype=np.int32),
        is_variable=False,
    )
    model_ir.tensors["x_onnx_ncx_internal"] = TensorIR(
        name="x_onnx_ncx_internal",
        dtype="FLOAT32",
        shape=[1, 2, 4, 4],
        shape_signature=[1, 2, 4, 4],
    )
    model_ir.tensors["slice_begin"] = TensorIR(
        name="slice_begin",
        dtype="INT32",
        shape=[4],
        shape_signature=[4],
        data=np.asarray([0, 1, 0, 0], dtype=np.int32),
        is_variable=False,
    )
    model_ir.tensors["slice_size"] = TensorIR(
        name="slice_size",
        dtype="INT32",
        shape=[4],
        shape_signature=[4],
        data=np.asarray([1, 1, 4, 4], dtype=np.int32),
        is_variable=False,
    )
    model_ir.tensors["c1"] = TensorIR(
        name="c1",
        dtype="FLOAT32",
        shape=[1, 1, 4, 4],
        shape_signature=[1, 1, 4, 4],
    )
    model_ir.tensors["dw_filter"] = TensorIR(
        name="dw_filter",
        dtype="FLOAT32",
        shape=[1, 3, 3, 1],
        shape_signature=[1, 3, 3, 1],
        data=np.ones((1, 3, 3, 1), dtype=np.float32),
        is_variable=False,
    )
    model_ir.tensors["dw_bias"] = TensorIR(
        name="dw_bias",
        dtype="FLOAT32",
        shape=[1],
        shape_signature=[1],
        data=np.zeros((1,), dtype=np.float32),
        is_variable=False,
    )
    model_ir.tensors["dw_out"] = TensorIR(
        name="dw_out",
        dtype="FLOAT32",
        shape=[1, 1, 4, 4],
        shape_signature=[1, 1, 4, 4],
    )
    model_ir.tensors["dw_out_cast"] = TensorIR(
        name="dw_out_cast",
        dtype="FLOAT16",
        shape=[1, 1, 4, 4],
        shape_signature=[1, 1, 4, 4],
    )
    model_ir.tensors["y"] = TensorIR(
        name="y",
        dtype="FLOAT16",
        shape=[1, 1, 4, 4],
        shape_signature=[1, 1, 4, 4],
    )

    model_ir.operators = [
        OperatorIR(
            op_type="TRANSPOSE",
            inputs=["x", "x_onnx_ncx_internal_perm"],
            outputs=["x_onnx_ncx_internal"],
        ),
        OperatorIR(op_type="SLICE", inputs=["x_onnx_ncx_internal", "slice_begin", "slice_size"], outputs=["c1"]),
        OperatorIR(
            op_type="DEPTHWISE_CONV_2D",
            inputs=["c1", "dw_filter", "dw_bias"],
            outputs=["dw_out"],
            options={
                "padding": "SAME",
                "strideH": 1,
                "strideW": 1,
                "dilationHFactor": 1,
                "dilationWFactor": 1,
                "depthMultiplier": 1,
                "fusedActivationFunction": "NONE",
            },
        ),
        OperatorIR(
            op_type="CAST",
            inputs=["dw_out"],
            outputs=["dw_out_cast"],
            options={"inDataType": "FLOAT32", "outDataType": "FLOAT16"},
        ),
        OperatorIR(op_type="RELU", inputs=["dw_out_cast"], outputs=["y"]),
    ]

    stats = _optimize_boundary_input_transpose_channel_slice_blocks(model_ir)
    assert stats["removed_boundary_input_transpose"] == 1
    assert stats["inserted_local_boundary_transposes"] == 0

    depthwise_op = next(op for op in model_ir.operators if str(op.op_type) == "DEPTHWISE_CONV_2D")
    assert list(depthwise_op.inputs) == ["c1", "dw_filter", "dw_bias"]

    assert list(model_ir.tensors["c1"].shape) == [1, 4, 4, 1]
    assert list(model_ir.tensors["dw_out"].shape) == [1, 4, 4, 1]
    assert list(model_ir.tensors["dw_out_cast"].shape) == [1, 4, 4, 1]
    assert list(model_ir.tensors["y"].shape) == [1, 4, 4, 1]

    assert not any(str(op.op_type) == "TRANSPOSE" for op in model_ir.operators)


def test_flatbuffer_direct_input_layout_respects_keep_shape_absolute() -> None:
    model = _make_qlinear_conv_chain_model()
    register_default_preprocess_rules()
    preprocessed_model, _ = run_preprocess_pipeline(onnx_graph=model)
    model_ir = lower_onnx_to_ir(
        onnx_graph=preprocessed_model,
        output_file_name="input_layout_keep_absolute_test",
        transpose_inputs_to_nhwc=True,
        keep_shape_absolutely_input_names=["x"],
    )
    assert model_ir.inputs == ["x"]
    assert list(model_ir.tensors["x"].shape) == [1, 1, 2, 2]


def test_flatbuffer_direct_depthwise_output_transpose_cast_transpose_chain_elides() -> None:
    model_ir = ModelIR(name="depthwise_output_transpose_cast_transpose_chain_test")
    model_ir.inputs = ["x_nhwc"]
    model_ir.outputs = ["z"]

    model_ir.tensors["x_nhwc"] = TensorIR(
        name="x_nhwc",
        dtype="FLOAT32",
        shape=[1, 4, 4, 1],
        shape_signature=[1, 4, 4, 1],
    )
    model_ir.tensors["dw_filter"] = TensorIR(
        name="dw_filter",
        dtype="FLOAT32",
        shape=[1, 3, 3, 1],
        shape_signature=[1, 3, 3, 1],
        data=np.ones((1, 3, 3, 1), dtype=np.float32),
        is_variable=False,
    )
    model_ir.tensors["dw_bias"] = TensorIR(
        name="dw_bias",
        dtype="FLOAT32",
        shape=[1],
        shape_signature=[1],
        data=np.zeros((1,), dtype=np.float32),
        is_variable=False,
    )
    model_ir.tensors["dw_out_nhwc"] = TensorIR(
        name="dw_out_nhwc",
        dtype="FLOAT32",
        shape=[1, 4, 4, 1],
        shape_signature=[1, 4, 4, 1],
    )
    model_ir.tensors["perm_nhwc_to_nchw"] = TensorIR(
        name="perm_nhwc_to_nchw",
        dtype="INT32",
        shape=[4],
        shape_signature=[4],
        data=np.asarray([0, 3, 1, 2], dtype=np.int32),
        is_variable=False,
    )
    model_ir.tensors["perm_nchw_to_nhwc"] = TensorIR(
        name="perm_nchw_to_nhwc",
        dtype="INT32",
        shape=[4],
        shape_signature=[4],
        data=np.asarray([0, 2, 3, 1], dtype=np.int32),
        is_variable=False,
    )
    model_ir.tensors["dw_out_nchw"] = TensorIR(
        name="dw_out_nchw",
        dtype="FLOAT32",
        shape=[1, 1, 4, 4],
        shape_signature=[1, 1, 4, 4],
    )
    model_ir.tensors["dw_cast_nchw"] = TensorIR(
        name="dw_cast_nchw",
        dtype="FLOAT16",
        shape=[1, 1, 4, 4],
        shape_signature=[1, 1, 4, 4],
    )
    model_ir.tensors["dw_cast_nhwc"] = TensorIR(
        name="dw_cast_nhwc",
        dtype="FLOAT16",
        shape=[1, 4, 4, 1],
        shape_signature=[1, 4, 4, 1],
    )
    model_ir.tensors["z"] = TensorIR(
        name="z",
        dtype="FLOAT16",
        shape=[1, 4, 4, 1],
        shape_signature=[1, 4, 4, 1],
    )

    model_ir.operators = [
        OperatorIR(
            op_type="DEPTHWISE_CONV_2D",
            inputs=["x_nhwc", "dw_filter", "dw_bias"],
            outputs=["dw_out_nhwc"],
            options={
                "padding": "SAME",
                "strideH": 1,
                "strideW": 1,
                "dilationHFactor": 1,
                "dilationWFactor": 1,
                "depthMultiplier": 1,
                "fusedActivationFunction": "NONE",
            },
        ),
        OperatorIR(
            op_type="TRANSPOSE",
            inputs=["dw_out_nhwc", "perm_nhwc_to_nchw"],
            outputs=["dw_out_nchw"],
        ),
        OperatorIR(
            op_type="CAST",
            inputs=["dw_out_nchw"],
            outputs=["dw_cast_nchw"],
            options={"inDataType": "FLOAT32", "outDataType": "FLOAT16"},
        ),
        OperatorIR(
            op_type="TRANSPOSE",
            inputs=["dw_cast_nchw", "perm_nchw_to_nhwc"],
            outputs=["dw_cast_nhwc"],
        ),
        OperatorIR(op_type="RELU", inputs=["dw_cast_nhwc"], outputs=["z"]),
    ]

    stats = _optimize_transposeconv_output_nhwc_passthrough_chains(model_ir)
    assert stats["rewritten_transposeconv_output_nhwc_passthrough_chains"] == 1

    op_types = [str(op.op_type) for op in model_ir.operators]
    assert op_types.count("TRANSPOSE") == 0
    assert op_types.count("DEPTHWISE_CONV_2D") == 1
    assert op_types.count("CAST") == 1

    cast_op = next(op for op in model_ir.operators if str(op.op_type) == "CAST")
    assert list(cast_op.inputs) == ["dw_out_nhwc"]
    assert list(cast_op.outputs) == ["dw_cast_nhwc"]

    relu_op = next(op for op in model_ir.operators if str(op.op_type) == "RELU")
    assert list(relu_op.inputs) == ["dw_cast_nhwc"]


def test_flatbuffer_direct_depthwise_output_transpose_hardswish_transpose_chain_elides() -> None:
    model_ir = ModelIR(name="depthwise_output_transpose_hardswish_transpose_chain_test")
    model_ir.inputs = ["x_nhwc"]
    model_ir.outputs = ["z"]

    model_ir.tensors["x_nhwc"] = TensorIR(
        name="x_nhwc",
        dtype="FLOAT32",
        shape=[1, 4, 4, 1],
        shape_signature=[1, 4, 4, 1],
    )
    model_ir.tensors["dw_filter"] = TensorIR(
        name="dw_filter",
        dtype="FLOAT32",
        shape=[1, 3, 3, 1],
        shape_signature=[1, 3, 3, 1],
        data=np.ones((1, 3, 3, 1), dtype=np.float32),
        is_variable=False,
    )
    model_ir.tensors["dw_bias"] = TensorIR(
        name="dw_bias",
        dtype="FLOAT32",
        shape=[1],
        shape_signature=[1],
        data=np.zeros((1,), dtype=np.float32),
        is_variable=False,
    )
    model_ir.tensors["dw_out_nhwc"] = TensorIR(
        name="dw_out_nhwc",
        dtype="FLOAT32",
        shape=[1, 4, 4, 1],
        shape_signature=[1, 4, 4, 1],
    )
    model_ir.tensors["perm_nhwc_to_nchw"] = TensorIR(
        name="perm_nhwc_to_nchw",
        dtype="INT32",
        shape=[4],
        shape_signature=[4],
        data=np.asarray([0, 3, 1, 2], dtype=np.int32),
        is_variable=False,
    )
    model_ir.tensors["perm_nchw_to_nhwc"] = TensorIR(
        name="perm_nchw_to_nhwc",
        dtype="INT32",
        shape=[4],
        shape_signature=[4],
        data=np.asarray([0, 2, 3, 1], dtype=np.int32),
        is_variable=False,
    )
    model_ir.tensors["dw_out_nchw"] = TensorIR(
        name="dw_out_nchw",
        dtype="FLOAT32",
        shape=[1, 1, 4, 4],
        shape_signature=[1, 1, 4, 4],
    )
    model_ir.tensors["dw_hswish_nchw"] = TensorIR(
        name="dw_hswish_nchw",
        dtype="FLOAT32",
        shape=[1, 1, 4, 4],
        shape_signature=[1, 1, 4, 4],
    )
    model_ir.tensors["dw_hswish_nhwc"] = TensorIR(
        name="dw_hswish_nhwc",
        dtype="FLOAT32",
        shape=[1, 4, 4, 1],
        shape_signature=[1, 4, 4, 1],
    )
    model_ir.tensors["z"] = TensorIR(
        name="z",
        dtype="FLOAT32",
        shape=[1, 4, 4, 1],
        shape_signature=[1, 4, 4, 1],
    )

    model_ir.operators = [
        OperatorIR(
            op_type="DEPTHWISE_CONV_2D",
            inputs=["x_nhwc", "dw_filter", "dw_bias"],
            outputs=["dw_out_nhwc"],
            options={
                "padding": "SAME",
                "strideH": 1,
                "strideW": 1,
                "dilationHFactor": 1,
                "dilationWFactor": 1,
                "depthMultiplier": 1,
                "fusedActivationFunction": "NONE",
            },
        ),
        OperatorIR(
            op_type="TRANSPOSE",
            inputs=["dw_out_nhwc", "perm_nhwc_to_nchw"],
            outputs=["dw_out_nchw"],
        ),
        OperatorIR(
            op_type="HARD_SWISH",
            inputs=["dw_out_nchw"],
            outputs=["dw_hswish_nchw"],
        ),
        OperatorIR(
            op_type="TRANSPOSE",
            inputs=["dw_hswish_nchw", "perm_nchw_to_nhwc"],
            outputs=["dw_hswish_nhwc"],
        ),
        OperatorIR(op_type="RELU", inputs=["dw_hswish_nhwc"], outputs=["z"]),
    ]

    stats = _optimize_transposeconv_output_nhwc_passthrough_chains(model_ir)
    assert stats["rewritten_transposeconv_output_nhwc_passthrough_chains"] == 1

    op_types = [str(op.op_type) for op in model_ir.operators]
    assert op_types.count("TRANSPOSE") == 0
    assert op_types.count("DEPTHWISE_CONV_2D") == 1
    assert op_types.count("HARD_SWISH") == 1

    hard_swish_op = next(op for op in model_ir.operators if str(op.op_type) == "HARD_SWISH")
    assert list(hard_swish_op.inputs) == ["dw_out_nhwc"]
    assert list(hard_swish_op.outputs) == ["dw_hswish_nhwc"]

    relu_op = next(op for op in model_ir.operators if str(op.op_type) == "RELU")
    assert list(relu_op.inputs) == ["dw_hswish_nhwc"]


def test_flatbuffer_direct_transpose_input_remap_preserves_boundary_transpose_for_multibranch_input() -> None:
    model = _make_shared_input_conv_and_mul_branches_model()
    register_default_preprocess_rules()
    preprocessed_model, _ = run_preprocess_pipeline(onnx_graph=model)
    model_ir = lower_onnx_to_ir(
        onnx_graph=preprocessed_model,
        output_file_name="shared_input_transpose_remap_test",
        transpose_inputs_to_nhwc=True,
        optimize_layout_transpose_chains=False,
    )

    internal_input_name = "x_onnx_ncx_internal"
    producer_count = sum(
        1 for op in model_ir.operators if internal_input_name in [str(v) for v in op.outputs]
    )
    consumer_count = sum(
        1 for op in model_ir.operators if internal_input_name in [str(v) for v in op.inputs]
    )
    if consumer_count > 0:
        assert producer_count == 1

    model_inputs = set(str(v) for v in model_ir.inputs)
    produced_tensors = set(
        str(output_name)
        for op in model_ir.operators
        for output_name in list(op.outputs)
    )
    dangling_runtime_inputs = []
    for op in model_ir.operators:
        for input_name in list(op.inputs):
            normalized = str(input_name)
            if normalized in model_inputs or normalized in produced_tensors:
                continue
            tensor = model_ir.tensors.get(normalized, None)
            if tensor is not None and tensor.data is not None:
                continue
            dangling_runtime_inputs.append(normalized)
    assert internal_input_name not in dangling_runtime_inputs


def test_flatbuffer_direct_grouped_conv_default_avoids_split_op() -> None:
    model = _make_grouped_conv_model()
    register_default_preprocess_rules()
    preprocessed_model, _ = run_preprocess_pipeline(onnx_graph=model)
    model_ir = lower_onnx_to_ir(
        onnx_graph=preprocessed_model,
        output_file_name="grouped_conv_default_no_split_test",
        optimize_layout_transpose_chains=False,
    )
    op_types = [str(op.op_type) for op in model_ir.operators]
    assert "SPLIT" not in op_types
    assert "STRIDED_SLICE" not in op_types
    assert op_types.count("CONV_2D") == 1
    assert "CONCATENATION" not in op_types


def test_flatbuffer_direct_grouped_conv_dgc_uses_split_op() -> None:
    model = _make_grouped_conv_model()
    register_default_preprocess_rules()
    preprocessed_model, _ = run_preprocess_pipeline(onnx_graph=model)
    model_ir = lower_onnx_to_ir(
        onnx_graph=preprocessed_model,
        output_file_name="grouped_conv_dgc_split_test",
        disable_group_convolution=True,
        optimize_layout_transpose_chains=False,
    )
    op_types = [str(op.op_type) for op in model_ir.operators]
    assert op_types.count("SPLIT") == 1
    assert op_types.count("CONV_2D") == 2
    assert op_types.count("CONCATENATION") == 1


def test_flatbuffer_direct_reshape_minus_one_resolved_to_static_shape() -> None:
    model = _make_reshape_minus_one_model()
    register_default_preprocess_rules()
    preprocessed_model, _ = run_preprocess_pipeline(onnx_graph=model)
    model_ir = lower_onnx_to_ir(
        onnx_graph=preprocessed_model,
        output_file_name="reshape_minus_one_resolved_test",
    )
    reshape_ops = [op for op in model_ir.operators if str(op.op_type) == "RESHAPE"]
    assert len(reshape_ops) == 1
    reshape_op = reshape_ops[0]
    assert list(reshape_op.options.get("newShape", [])) == [3, 2]
    shape_tensor_name = reshape_op.inputs[1]
    shape_tensor = model_ir.tensors[shape_tensor_name]
    assert shape_tensor is not None
    assert shape_tensor.data is not None
    assert np.asarray(shape_tensor.data).reshape(-1).tolist() == [3, 2]
    output_tensor = model_ir.tensors["y"]
    assert list(output_tensor.shape) == [3, 2]
    assert list(output_tensor.shape_signature) == [3, 2]


def test_flatbuffer_direct_resolve_dynamic_reshape_shapes_pass() -> None:
    model_ir = ModelIR(name="reshape_fixup_test")
    model_ir.inputs = ["x"]
    model_ir.outputs = ["y"]
    model_ir.tensors["x"] = TensorIR(
        name="x",
        dtype="INT8",
        shape=[1, 40, 40, 1],
        shape_signature=[1, 40, 40, 1],
    )
    model_ir.tensors["reshape_shape"] = TensorIR(
        name="reshape_shape",
        dtype="INT32",
        shape=[3],
        shape_signature=[3],
        data=np.asarray([1, -1, 1], dtype=np.int32),
    )
    model_ir.tensors["y"] = TensorIR(
        name="y",
        dtype="INT8",
        shape=[1],
        shape_signature=[-1],
    )
    model_ir.operators.append(
        OperatorIR(
            op_type="RESHAPE",
            inputs=["x", "reshape_shape"],
            outputs=["y"],
            options={"newShape": [1, -1, 1]},
        )
    )

    stats = _resolve_dynamic_reshape_shapes(model_ir)
    assert stats["resolved_dynamic_reshape_shapes"] == 1
    assert list(model_ir.operators[0].options["newShape"]) == [1, 1600, 1]
    assert list(model_ir.tensors["y"].shape) == [1, 1600, 1]
    assert list(model_ir.tensors["y"].shape_signature) == [1, 1600, 1]
    assert np.asarray(model_ir.tensors["reshape_shape"].data).reshape(-1).tolist() == [1, 1600, 1]


def test_flatbuffer_direct_resolve_dynamic_reshape_shapes_zero_copy_dims_pass() -> None:
    model_ir = ModelIR(name="reshape_fixup_zero_copy_test")
    model_ir.inputs = ["x"]
    model_ir.outputs = ["y"]
    model_ir.tensors["x"] = TensorIR(
        name="x",
        dtype="FLOAT32",
        shape=[10, 1, 1, 3],
        shape_signature=[10, 1, 1, 3],
    )
    model_ir.tensors["reshape_shape"] = TensorIR(
        name="reshape_shape",
        dtype="INT32",
        shape=[3],
        shape_signature=[3],
        data=np.asarray([0, 1, 3], dtype=np.int32),
    )
    model_ir.tensors["y"] = TensorIR(
        name="y",
        dtype="FLOAT32",
        shape=[1],
        shape_signature=[-1],
    )
    model_ir.operators.append(
        OperatorIR(
            op_type="RESHAPE",
            inputs=["x", "reshape_shape"],
            outputs=["y"],
            options={
                "newShape": [10, 1, 3],
                "onnxRawNewShape": [0, 1, 3],
                "allowZero": False,
            },
        )
    )

    stats = _resolve_dynamic_reshape_shapes(model_ir)
    assert stats["resolved_dynamic_reshape_shapes"] == 1
    assert list(model_ir.operators[0].options["newShape"]) == [10, 1, 3]
    assert list(model_ir.tensors["y"].shape) == [10, 1, 3]
    assert list(model_ir.tensors["y"].shape_signature) == [10, 1, 3]
    assert np.asarray(model_ir.tensors["reshape_shape"].data).reshape(-1).tolist() == [10, 1, 3]


def test_flatbuffer_direct_logsoftmax_lowering() -> None:
    model = _make_logsoftmax_model()
    register_default_preprocess_rules()
    preprocessed_model, _ = run_preprocess_pipeline(onnx_graph=model)
    model_ir = lower_onnx_to_ir(
        onnx_graph=preprocessed_model,
        output_file_name="logsoftmax_lowering_test",
        allow_custom_ops=False,
    )
    op_types = [str(op.op_type) for op in model_ir.operators]
    assert op_types.count("SOFTMAX") == 1
    assert op_types.count("LOG") == 1
    assert op_types.count("CUSTOM") == 0


def test_flatbuffer_direct_global_max_pool_lowering() -> None:
    model = _make_global_max_pool_model()
    register_default_preprocess_rules()
    preprocessed_model, _ = run_preprocess_pipeline(onnx_graph=model)
    model_ir = lower_onnx_to_ir(
        onnx_graph=preprocessed_model,
        output_file_name="global_max_pool_lowering_test",
        allow_custom_ops=False,
    )
    op_types = [str(op.op_type) for op in model_ir.operators]
    assert op_types.count("REDUCE_MAX") == 1
    assert op_types.count("CUSTOM") == 0


def test_flatbuffer_direct_softmax_default_axis_respects_opset13() -> None:
    model = _make_softmax_default_axis_opset13_model()
    register_default_preprocess_rules()
    preprocessed_model, _ = run_preprocess_pipeline(onnx_graph=model)
    model_ir = lower_onnx_to_ir(
        onnx_graph=preprocessed_model,
        output_file_name="softmax_default_axis_opset13_lowering_test",
        allow_custom_ops=False,
    )
    op_types = [str(op.op_type) for op in model_ir.operators]
    assert op_types.count("SOFTMAX") == 1
    assert op_types.count("TRANSPOSE") == 0
    assert op_types.count("CUSTOM") == 0


def test_flatbuffer_direct_softmax_default_axis_respects_opset11() -> None:
    model = _make_softmax_default_axis_opset11_model()
    model_ir = lower_onnx_to_ir(
        onnx_graph=model,
        output_file_name="softmax_default_axis_opset11_lowering_test",
        allow_custom_ops=False,
    )
    op_types = [str(op.op_type) for op in model_ir.operators]
    assert op_types.count("SOFTMAX") == 1
    assert op_types.count("TRANSPOSE") == 2
    assert op_types.count("CUSTOM") == 0


def test_flatbuffer_direct_lrn_lowering() -> None:
    model = _make_lrn_model()
    register_default_preprocess_rules()
    preprocessed_model, _ = run_preprocess_pipeline(onnx_graph=model)
    model_ir = lower_onnx_to_ir(
        onnx_graph=preprocessed_model,
        output_file_name="lrn_lowering_test",
        allow_custom_ops=False,
    )
    op_types = [str(op.op_type) for op in model_ir.operators]
    assert op_types.count("LOCAL_RESPONSE_NORMALIZATION") == 1
    assert op_types.count("CUSTOM") == 0


def test_flatbuffer_direct_dropout_lowering() -> None:
    model = _make_dropout_model()
    register_default_preprocess_rules()
    preprocessed_model, _ = run_preprocess_pipeline(onnx_graph=model)
    model_ir = lower_onnx_to_ir(
        onnx_graph=preprocessed_model,
        output_file_name="dropout_lowering_test",
        allow_custom_ops=False,
    )
    op_types = [str(op.op_type) for op in model_ir.operators]
    assert op_types.count("RESHAPE") == 1
    assert op_types.count("CUSTOM") == 0


def test_flatbuffer_direct_dropout_with_mask_output_lowering() -> None:
    model = _make_dropout_model(with_mask_output=True)
    register_default_preprocess_rules()
    preprocessed_model, _ = run_preprocess_pipeline(onnx_graph=model)
    model_ir = lower_onnx_to_ir(
        onnx_graph=preprocessed_model,
        output_file_name="dropout_mask_lowering_test",
        allow_custom_ops=False,
    )
    op_types = [str(op.op_type) for op in model_ir.operators]
    assert op_types.count("RESHAPE") == 1
    assert op_types.count("SHAPE") == 1
    assert op_types.count("FILL") == 1
    assert op_types.count("CUSTOM") == 0


def test_flatbuffer_direct_instance_normalization_lowering() -> None:
    model = _make_instance_normalization_model()
    register_default_preprocess_rules()
    preprocessed_model, _ = run_preprocess_pipeline(onnx_graph=model)
    model_ir = lower_onnx_to_ir(
        onnx_graph=preprocessed_model,
        output_file_name="instance_normalization_lowering_test",
        allow_custom_ops=False,
    )
    op_types = [str(op.op_type) for op in model_ir.operators]
    assert op_types.count("MEAN") == 2
    assert op_types.count("CUSTOM") == 0


def test_flatbuffer_direct_pow_lowering() -> None:
    model = _make_pow_general_model()
    register_default_preprocess_rules()
    preprocessed_model, _ = run_preprocess_pipeline(onnx_graph=model)
    model_ir = lower_onnx_to_ir(
        onnx_graph=preprocessed_model,
        output_file_name="pow_lowering_test",
        allow_custom_ops=False,
    )
    op_types = [str(op.op_type) for op in model_ir.operators]
    assert op_types.count("POW") == 1
    assert op_types.count("CUSTOM") == 0


def test_flatbuffer_direct_reciprocal_lowering() -> None:
    model = _make_reciprocal_model()
    register_default_preprocess_rules()
    preprocessed_model, _ = run_preprocess_pipeline(onnx_graph=model)
    model_ir = lower_onnx_to_ir(
        onnx_graph=preprocessed_model,
        output_file_name="reciprocal_lowering_test",
        allow_custom_ops=False,
    )
    op_types = [str(op.op_type) for op in model_ir.operators]
    assert op_types.count("DIV") == 1
    assert op_types.count("CUSTOM") == 0


def test_flatbuffer_direct_onehot_lowering() -> None:
    model = _make_onehot_model()
    register_default_preprocess_rules()
    preprocessed_model, _ = run_preprocess_pipeline(onnx_graph=model)
    model_ir = lower_onnx_to_ir(
        onnx_graph=preprocessed_model,
        output_file_name="onehot_lowering_test",
        allow_custom_ops=False,
    )
    op_types = [str(op.op_type) for op in model_ir.operators]
    assert op_types.count("ONE_HOT") == 1
    assert op_types.count("CUSTOM") == 0


def test_flatbuffer_direct_matmul_integer_lowering() -> None:
    model = _make_matmul_integer_model()
    register_default_preprocess_rules()
    preprocessed_model, _ = run_preprocess_pipeline(onnx_graph=model)
    model_ir = lower_onnx_to_ir(
        onnx_graph=preprocessed_model,
        output_file_name="matmul_integer_lowering_test",
        allow_custom_ops=False,
    )
    op_types = [str(op.op_type) for op in model_ir.operators]
    assert op_types.count("BATCH_MATMUL") == 1
    assert op_types.count("CUSTOM") == 0


def test_flatbuffer_direct_matmul_integer_batch_matmul_runtime_dtypes() -> None:
    model = _make_matmul_integer_model()
    register_default_preprocess_rules()
    preprocessed_model, _ = run_preprocess_pipeline(onnx_graph=model)
    model_ir = lower_onnx_to_ir(
        onnx_graph=preprocessed_model,
        output_file_name="matmul_integer_runtime_dtype_test",
        allow_custom_ops=False,
    )
    allowed = {"FLOAT32", "INT8", "INT16"}
    for op in model_ir.operators:
        if str(op.op_type) != "BATCH_MATMUL":
            continue
        in_dtypes = [str(model_ir.tensors[name].dtype).upper() for name in op.inputs]
        assert all(dtype in allowed for dtype in in_dtypes), in_dtypes


def test_flatbuffer_direct_dynamic_quantize_linear_lowering() -> None:
    model = _make_dynamic_quantize_linear_model()
    register_default_preprocess_rules()
    preprocessed_model, _ = run_preprocess_pipeline(onnx_graph=model)
    model_ir = lower_onnx_to_ir(
        onnx_graph=preprocessed_model,
        output_file_name="dynamic_quantize_linear_lowering_test",
        allow_custom_ops=False,
    )
    op_types = [str(op.op_type) for op in model_ir.operators]
    assert op_types.count("REDUCE_MAX") == 2
    assert op_types.count("CUSTOM") == 0


def test_flatbuffer_direct_shape_lowering() -> None:
    model = _make_shape_slice_model()
    register_default_preprocess_rules()
    preprocessed_model, _ = run_preprocess_pipeline(onnx_graph=model)
    model_ir = lower_onnx_to_ir(
        onnx_graph=preprocessed_model,
        output_file_name="shape_lowering_test",
        allow_custom_ops=False,
    )
    op_types = [str(op.op_type) for op in model_ir.operators]
    assert op_types.count("SHAPE") == 1
    assert op_types.count("SLICE") == 1
    assert op_types.count("CUSTOM") == 0


def test_flatbuffer_direct_unsqueeze_dynamic_axis2_preserves_dynamic_signature() -> None:
    model = _make_unsqueeze_dynamic_axis2_model()
    register_default_preprocess_rules()
    preprocessed_model, _ = run_preprocess_pipeline(onnx_graph=model)
    model_ir = lower_onnx_to_ir(
        onnx_graph=preprocessed_model,
        output_file_name="unsqueeze_dynamic_axis2_signature_test",
        allow_custom_ops=False,
    )

    y_tensor = model_ir.tensors["y"]
    assert list(y_tensor.shape_signature) == [1, -1, 1]

    y_prod = next(op for op in model_ir.operators if str(op.outputs[0]) == "y")
    assert str(y_prod.op_type) == "RESHAPE"
    assert list(y_prod.options.get("newShape", [])) == []


def test_flatbuffer_direct_unsqueeze_axes_2_3_lowering() -> None:
    model = _make_unsqueeze_axes_2_3_model()
    register_default_preprocess_rules()
    preprocessed_model, _ = run_preprocess_pipeline(onnx_graph=model)
    model_ir = lower_onnx_to_ir(
        onnx_graph=preprocessed_model,
        output_file_name="unsqueeze_axes_2_3_lowering_test",
        allow_custom_ops=False,
    )

    op_types = [str(op.op_type) for op in model_ir.operators]
    assert op_types.count("RESHAPE") == 1
    assert op_types.count("CUSTOM") == 0

    y_prod = next(op for op in model_ir.operators if str(op.outputs[0]) == "y")
    assert str(y_prod.op_type) == "RESHAPE"
    assert list(y_prod.options.get("newShape", [])) == [1, 3, 1, 1]


def test_flatbuffer_direct_min_topk_dynamic_k_lowering() -> None:
    model = _make_min_topk_dynamic_k_model()
    register_default_preprocess_rules()
    preprocessed_model, _ = run_preprocess_pipeline(onnx_graph=model)
    model_ir = lower_onnx_to_ir(
        onnx_graph=preprocessed_model,
        output_file_name="min_topk_dynamic_k_lowering_test",
        allow_custom_ops=False,
    )

    op_types = [str(op.op_type) for op in model_ir.operators]
    assert op_types.count("MINIMUM") == 1
    assert op_types.count("TOPK_V2") == 1
    assert op_types.count("CUSTOM") == 0

    topk_op = next(op for op in model_ir.operators if str(op.op_type) == "TOPK_V2")
    k_input_name = str(topk_op.inputs[1])
    k_input_tensor = model_ir.tensors[k_input_name]
    assert str(k_input_tensor.dtype).upper() == "INT32"
    assert list(k_input_tensor.shape) == [1]
    k_input_producers = [
        op
        for op in model_ir.operators
        if str(op.op_type) == "SQUEEZE" and str(op.outputs[0]) == k_input_name
    ]
    assert len(k_input_producers) == 1

    indices_out = model_ir.tensors["indices"]
    assert str(indices_out.dtype).upper() == "INT64"
    cast_to_i64 = [
        op
        for op in model_ir.operators
        if str(op.op_type) == "CAST" and str(op.outputs[0]) == "indices"
    ]
    assert len(cast_to_i64) == 1


def test_flatbuffer_direct_pad_dynamic_pads_lowering() -> None:
    model = _make_pad_dynamic_pads_model()
    register_default_preprocess_rules()
    preprocessed_model, _ = run_preprocess_pipeline(onnx_graph=model)
    model_ir = lower_onnx_to_ir(
        onnx_graph=preprocessed_model,
        output_file_name="pad_dynamic_pads_lowering_test",
        allow_custom_ops=False,
    )

    op_types = [str(op.op_type) for op in model_ir.operators]
    assert op_types.count("PAD") == 1
    assert op_types.count("CUSTOM") == 0

    pad_op = next(op for op in model_ir.operators if str(op.op_type) == "PAD")
    pads_name = str(pad_op.inputs[1])
    pads_tensor = model_ir.tensors[pads_name]
    assert str(pads_tensor.dtype).upper() == "INT32"
    assert list(pads_tensor.shape) == [2, 2]
    assert pads_tensor.data is None

    producers = {}
    for op in model_ir.operators:
        for output_name in op.outputs:
            producers[str(output_name)] = op

    pads_transpose = producers.get(pads_name)
    assert pads_transpose is not None
    assert str(pads_transpose.op_type) == "TRANSPOSE"

    pads_2xrank_name = str(pads_transpose.inputs[0])
    pads_reshape = producers.get(pads_2xrank_name)
    assert pads_reshape is not None
    assert str(pads_reshape.op_type) == "RESHAPE"

    pads_vector_name = str(pads_reshape.inputs[0])
    pads_vector = model_ir.tensors[pads_vector_name]
    assert str(pads_vector.dtype).upper() == "INT32"
    pads_cast = producers.get(pads_vector_name)
    assert pads_cast is not None
    assert str(pads_cast.op_type) == "CAST"


def test_flatbuffer_direct_pad_reflect_lowering() -> None:
    model = _make_pad_reflect_model()
    register_default_preprocess_rules()
    preprocessed_model, _ = run_preprocess_pipeline(onnx_graph=model)
    model_ir = lower_onnx_to_ir(
        onnx_graph=preprocessed_model,
        output_file_name="pad_reflect_lowering_test",
        allow_custom_ops=False,
    )

    op_types = [str(op.op_type) for op in model_ir.operators]
    assert op_types.count("MIRROR_PAD") == 1
    assert op_types.count("CUSTOM") == 0

    mirror_pad_op = next(op for op in model_ir.operators if str(op.op_type) == "MIRROR_PAD")
    assert str(mirror_pad_op.options.get("mode", "")).upper() == "REFLECT"

    pads_name = str(mirror_pad_op.inputs[1])
    pads_tensor = model_ir.tensors[pads_name]
    assert str(pads_tensor.dtype).upper() == "INT32"
    assert list(pads_tensor.shape) == [2, 2]
    assert pads_tensor.data is not None
    np.testing.assert_array_equal(
        pads_tensor.data,
        np.asarray([[0, 0], [1, 1]], dtype=np.int32),
    )


def test_flatbuffer_direct_pad_constant_nonzero_lowering() -> None:
    model = _make_pad_constant_nonzero_model()
    register_default_preprocess_rules()
    preprocessed_model, _ = run_preprocess_pipeline(onnx_graph=model)
    model_ir = lower_onnx_to_ir(
        onnx_graph=preprocessed_model,
        output_file_name="pad_constant_nonzero_lowering_test",
        allow_custom_ops=False,
    )

    op_types = [str(op.op_type) for op in model_ir.operators]
    assert op_types.count("PADV2") == 1
    assert op_types.count("CUSTOM") == 0

    padv2_op = next(op for op in model_ir.operators if str(op.op_type) == "PADV2")
    pad_value_name = str(padv2_op.inputs[2])
    pad_value_tensor = model_ir.tensors[pad_value_name]
    assert str(pad_value_tensor.dtype).upper() == "FLOAT16"
    assert list(pad_value_tensor.shape) == [1]
    assert pad_value_tensor.data is not None
    np.testing.assert_allclose(
        np.asarray(pad_value_tensor.data, dtype=np.float16),
        np.asarray([-65504.0], dtype=np.float16),
        rtol=0.0,
        atol=0.0,
    )


def test_flatbuffer_direct_batch_normalization_rank3_broadcast_coeff_shape() -> None:
    model = _make_batchnorm_rank3_model()
    register_default_preprocess_rules()
    preprocessed_model, _ = run_preprocess_pipeline(onnx_graph=model)
    model_ir = lower_onnx_to_ir(
        onnx_graph=preprocessed_model,
        output_file_name="batchnorm_rank3_broadcast_shape_test",
        allow_custom_ops=False,
    )

    mul_ops = [op for op in model_ir.operators if str(op.op_type) == "MUL"]
    add_ops = [op for op in model_ir.operators if str(op.op_type) == "ADD"]
    assert len(mul_ops) == 1
    assert len(add_ops) == 1

    mul_const = model_ir.tensors[str(mul_ops[0].inputs[1])]
    add_const = model_ir.tensors[str(add_ops[0].inputs[1])]
    assert list(mul_const.shape) == [1, 1536, 1]
    assert list(add_const.shape) == [1, 1536, 1]


def test_flatbuffer_direct_convtranspose2d_output_padding_lowering() -> None:
    model = _make_convtranspose2d_output_padding_model()
    register_default_preprocess_rules()
    preprocessed_model, _ = run_preprocess_pipeline(onnx_graph=model)
    model_ir = lower_onnx_to_ir(
        onnx_graph=preprocessed_model,
        output_file_name="convtranspose2d_output_padding_lowering_test",
        allow_custom_ops=False,
    )

    op_types = [str(op.op_type) for op in model_ir.operators]
    assert op_types.count("TRANSPOSE_CONV") == 1
    assert op_types.count("CUSTOM") == 0

    transpose_conv_op = next(op for op in model_ir.operators if str(op.op_type) == "TRANSPOSE_CONV")
    assert int(transpose_conv_op.options["strideH"]) == 2
    assert int(transpose_conv_op.options["strideW"]) == 2


def test_flatbuffer_direct_conv3d_lowering() -> None:
    model = _make_conv3d_model()
    register_default_preprocess_rules()
    preprocessed_model, _ = run_preprocess_pipeline(onnx_graph=model)
    model_ir = lower_onnx_to_ir(
        onnx_graph=preprocessed_model,
        output_file_name="conv3d_lowering_test",
        allow_custom_ops=False,
    )

    op_types = [str(op.op_type) for op in model_ir.operators]
    assert op_types.count("CONV_3D") == 1
    assert op_types.count("CUSTOM") == 0

    conv3d_op = next(op for op in model_ir.operators if str(op.op_type) == "CONV_3D")
    assert int(conv3d_op.options["strideD"]) == 1
    assert int(conv3d_op.options["strideH"]) == 1
    assert int(conv3d_op.options["strideW"]) == 1


def test_flatbuffer_direct_convtranspose3d_lowering() -> None:
    model = _make_convtranspose3d_model()
    register_default_preprocess_rules()
    preprocessed_model, _ = run_preprocess_pipeline(onnx_graph=model)
    model_ir = lower_onnx_to_ir(
        onnx_graph=preprocessed_model,
        output_file_name="convtranspose3d_lowering_test",
        allow_custom_ops=False,
    )

    op_types = [str(op.op_type) for op in model_ir.operators]
    assert op_types.count("CONV_3D_TRANSPOSE") == 1
    assert op_types.count("CUSTOM") == 0

    conv3dt_op = next(op for op in model_ir.operators if str(op.op_type) == "CONV_3D_TRANSPOSE")
    assert int(conv3dt_op.options["strideD"]) == 2
    assert int(conv3dt_op.options["strideH"]) == 2
    assert int(conv3dt_op.options["strideW"]) == 2


def test_flatbuffer_direct_legacy_slice_attr_lowering() -> None:
    model = _make_legacy_slice_attr_model()
    register_default_preprocess_rules()
    preprocessed_model, _ = run_preprocess_pipeline(onnx_graph=model)
    model_ir = lower_onnx_to_ir(
        onnx_graph=preprocessed_model,
        output_file_name="legacy_slice_attr_lowering_test",
        allow_custom_ops=False,
    )
    op_types = [str(op.op_type) for op in model_ir.operators]
    assert op_types.count("SLICE") == 2
    assert op_types.count("CONCATENATION") == 1
    assert op_types.count("CUSTOM") == 0


def test_flatbuffer_direct_slice_dynamic_end_prefix_rank2_lowering() -> None:
    model = _make_slice_dynamic_end_prefix_rank2_model()
    register_default_preprocess_rules()
    preprocessed_model, _ = run_preprocess_pipeline(onnx_graph=model)
    model_ir = lower_onnx_to_ir(
        onnx_graph=preprocessed_model,
        output_file_name="slice_dynamic_end_prefix_rank2_lowering_test",
        allow_custom_ops=False,
    )
    op_types = [str(op.op_type) for op in model_ir.operators]
    assert op_types.count("STRIDED_SLICE") == 1
    assert op_types.count("CUSTOM") == 0

    ss = next(op for op in model_ir.operators if str(op.op_type) == "STRIDED_SLICE")
    end_name = str(ss.inputs[2])
    end_tensor = model_ir.tensors[end_name]
    assert str(end_tensor.dtype).upper() == "INT32"
    assert end_tensor.data is None


def test_flatbuffer_direct_expand_dynamic_shape_uses_fill() -> None:
    model = _make_expand_dynamic_shape_model()
    register_default_preprocess_rules()
    preprocessed_model, _ = run_preprocess_pipeline(onnx_graph=model)
    model_ir = lower_onnx_to_ir(
        onnx_graph=preprocessed_model,
        output_file_name="expand_dynamic_shape_lowering_test",
        allow_custom_ops=False,
    )

    op_types = [str(op.op_type) for op in model_ir.operators]
    assert op_types.count("FILL") == 1
    assert op_types.count("MUL") >= 1
    assert op_types.count("CUSTOM") == 0

    fill_op = next(op for op in model_ir.operators if str(op.op_type) == "FILL")
    shape_name = str(fill_op.inputs[0])
    shape_tensor = model_ir.tensors[shape_name]
    assert str(shape_tensor.dtype).upper() == "INT32"
    assert shape_tensor.data is None

    output_tensor = model_ir.tensors["y"]
    assert list(output_tensor.shape_signature) == [-1]


def test_flatbuffer_direct_unsqueeze_scalar_multi_axis_shape() -> None:
    model = _make_unsqueeze_scalar_multi_axis_model()
    register_default_preprocess_rules()
    preprocessed_model, _ = run_preprocess_pipeline(onnx_graph=model)
    model_ir = lower_onnx_to_ir(
        onnx_graph=preprocessed_model,
        output_file_name="unsqueeze_scalar_multi_axis_test",
        allow_custom_ops=False,
    )

    y_tensor = model_ir.tensors["y"]
    assert list(y_tensor.shape) == [1, 1]
    assert list(y_tensor.shape_signature) == [1, 1]

    producer = next(op for op in model_ir.operators if "y" in list(op.outputs))
    assert str(producer.op_type) == "RESHAPE"
    assert list(producer.options.get("newShape", [])) == [1, 1]


def test_flatbuffer_direct_resize_dynamic_sizes_lowering() -> None:
    model = _make_resize_dynamic_sizes_model()
    register_default_preprocess_rules()
    preprocessed_model, _ = run_preprocess_pipeline(onnx_graph=model)
    model_ir = lower_onnx_to_ir(
        onnx_graph=preprocessed_model,
        output_file_name="resize_dynamic_sizes_lowering_test",
        allow_custom_ops=False,
    )
    op_types = [str(op.op_type) for op in model_ir.operators]
    assert op_types.count("RESIZE_BILINEAR") == 1
    assert op_types.count("CUSTOM") == 0

    resize_op = next(op for op in model_ir.operators if str(op.op_type) == "RESIZE_BILINEAR")
    size_input_name = str(resize_op.inputs[1])
    size_tensor = model_ir.tensors[size_input_name]
    assert str(size_tensor.dtype).upper() == "INT32"
    assert size_tensor.data is None


def test_flatbuffer_direct_atan_rank4_elides_boundary_input_transpose() -> None:
    model = _make_unary_rank4_model("Atan", name="atan_rank4")
    register_default_preprocess_rules()
    preprocessed_model, _ = run_preprocess_pipeline(onnx_graph=model)
    model_ir = lower_onnx_to_ir(
        onnx_graph=preprocessed_model,
        output_file_name="atan_rank4_layout_opt_test",
        allow_custom_ops=False,
        transpose_inputs_to_nhwc=True,
    )

    op_types = [str(op.op_type) for op in model_ir.operators]
    assert op_types.count("TRANSPOSE") == 0
    assert op_types.count("ATAN2") == 1
    assert list(model_ir.tensors["x"].shape) == [1, 8, 8, 3]
    assert list(model_ir.tensors["y"].shape) == [1, 8, 8, 3]

    atan2_op = next(op for op in model_ir.operators if str(op.op_type) == "ATAN2")
    side_name = str(atan2_op.inputs[1])
    side_tensor = model_ir.tensors.get(side_name)
    assert side_tensor is not None
    assert list(side_tensor.shape) == [1, 8, 8, 3]


def test_flatbuffer_direct_atan_dynamic_hw_uses_rank_matched_singleton_rhs() -> None:
    model = _make_unary_rank4_dynamic_hw_model("Atan", name="atan_dynamic_hw")
    register_default_preprocess_rules()
    preprocessed_model, _ = run_preprocess_pipeline(onnx_graph=model)
    model_ir = lower_onnx_to_ir(
        onnx_graph=preprocessed_model,
        output_file_name="atan_dynamic_hw_layout_opt_test",
        allow_custom_ops=False,
        transpose_inputs_to_nhwc=True,
    )

    op_types = [str(op.op_type) for op in model_ir.operators]
    assert op_types.count("ATAN2") == 1
    assert op_types.count("LESS_EQUAL") == 0
    assert op_types.count("CAST") == 0
    assert op_types.count("TRANSPOSE") == 0

    atan2_op = next(op for op in model_ir.operators if str(op.op_type) == "ATAN2")
    side_name = str(atan2_op.inputs[1])
    side_tensor = model_ir.tensors.get(side_name)
    assert side_tensor is not None
    assert list(side_tensor.shape) == [1, 1, 1, 3]


def test_flatbuffer_direct_resize_cubic_lowering() -> None:
    model = _make_resize_cubic_model()
    register_default_preprocess_rules()
    preprocessed_model, _ = run_preprocess_pipeline(onnx_graph=model)
    model_ir = lower_onnx_to_ir(
        onnx_graph=preprocessed_model,
        output_file_name="resize_cubic_lowering_test",
        allow_custom_ops=False,
    )
    op_types = [str(op.op_type) for op in model_ir.operators]
    assert op_types.count("BATCH_MATMUL") == 2
    assert op_types.count("RESIZE_BILINEAR") == 0
    assert op_types.count("CUSTOM") == 0


def test_flatbuffer_direct_resize_cubic_preserves_batch_dim() -> None:
    model = _make_resize_cubic_batch2_model()
    register_default_preprocess_rules()
    preprocessed_model, _ = run_preprocess_pipeline(onnx_graph=model)
    model_ir = lower_onnx_to_ir(
        onnx_graph=preprocessed_model,
        output_file_name="resize_cubic_preserve_batch_dim_test",
        allow_custom_ops=False,
    )

    y_tensor = model_ir.tensors.get("y")
    assert y_tensor is not None
    assert int(y_tensor.shape[0]) == 2
    assert int(y_tensor.shape_signature[0]) == 2

    cubic_h_in_reshape = next(
        (
            op
            for op in model_ir.operators
            if str(op.op_type) == "RESHAPE"
            and len(op.outputs) == 1
            and "resize_cubic_h_in" in str(op.outputs[0])
        ),
        None,
    )
    assert cubic_h_in_reshape is not None
    reshape_shape = [int(v) for v in list(cubic_h_in_reshape.options.get("newShape", []))]
    assert len(reshape_shape) == 3
    assert reshape_shape[0] == 2

    cubic_h_bmm = next(
        (
            op
            for op in model_ir.operators
            if str(op.op_type) == "BATCH_MATMUL"
            and len(op.outputs) == 1
            and "resize_cubic_h_out" in str(op.outputs[0])
        ),
        None,
    )
    assert cubic_h_bmm is not None
    cubic_h_out = model_ir.tensors.get(str(cubic_h_bmm.outputs[0]))
    assert cubic_h_out is not None
    assert int(cubic_h_out.shape[0]) == 2


@pytest.mark.skipif(not _requires_flatbuffer_tools(), reason="flatbuffer_direct requires flatc and curl")
def test_tf_converter_resize_cubic_avoids_flex_resize_bicubic() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        model = _make_resize_cubic_model()
        model_path = _save_model(tmpdir, "resize_cubic_tf_converter_no_flex", model)
        out_dir = os.path.join(tmpdir, "out")
        tflite_path = _convert(model_path, out_dir, "tf_converter")
        custom_codes = _collect_custom_codes(tflite_path)
        assert "FlexResizeBicubic" not in custom_codes


@pytest.mark.skipif(not _requires_flatbuffer_tools(), reason="flatbuffer_direct requires flatc and curl")
def test_tf_converter_resize_cubic_honors_cubic_coeff_a() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        model = _make_resize_cubic_model(cubic_coeff_a=-0.5)
        model_path = _save_model(tmpdir, "resize_cubic_tf_converter_coeff_a", model)
        out_dir = os.path.join(tmpdir, "out")
        tflite_path = _convert(model_path, out_dir, "tf_converter")

        rng = np.random.default_rng(0)
        x = rng.random((1, 3, 4, 4), dtype=np.float32)

        interpreter = Interpreter(model_path=tflite_path)
        interpreter.allocate_tensors()
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        input_shape = tuple(int(v) for v in list(input_details[0]["shape"]))
        x_feed = x
        if len(input_shape) == 4 and input_shape[1] == x.shape[2] and input_shape[2] == x.shape[3] and input_shape[3] == x.shape[1]:
            x_feed = np.transpose(x, [0, 2, 3, 1])
        interpreter.set_tensor(input_details[0]["index"], x_feed)
        interpreter.invoke()
        tflite_y = interpreter.get_tensor(output_details[0]["index"])
        output_shape = tuple(int(v) for v in list(output_details[0]["shape"]))
        if len(output_shape) == 4 and output_shape[1] == 8 and output_shape[2] == 8 and output_shape[3] == 3:
            tflite_y = np.transpose(tflite_y, [0, 3, 1, 2])

        def _cubic_weight(dist: float, coeff_a: float) -> float:
            t = float(abs(dist))
            if t <= 1.0:
                return ((coeff_a + 2.0) * t * t * t) - ((coeff_a + 3.0) * t * t) + 1.0
            if t < 2.0:
                return (coeff_a * t * t * t) - (5.0 * coeff_a * t * t) + (8.0 * coeff_a * t) - (4.0 * coeff_a)
            return 0.0

        def _align_corners_cubic_matrix(in_size: int, out_size: int, coeff_a: float) -> np.ndarray:
            matrix = np.zeros((out_size, in_size), dtype=np.float32)
            for out_idx in range(out_size):
                src = 0.0 if out_size <= 1 else (float(out_idx) * float(in_size - 1) / float(out_size - 1))
                src_floor = int(np.floor(src))
                for offset in [-1, 0, 1, 2]:
                    src_idx = int(src_floor + offset)
                    weight = _cubic_weight(src - float(src_idx), coeff_a)
                    if src_idx < 0:
                        src_idx = 0
                    elif src_idx >= in_size:
                        src_idx = in_size - 1
                    matrix[out_idx, src_idx] += np.float32(weight)
            return matrix

        x_nhwc = np.transpose(x, [0, 2, 3, 1])
        h_matrix = _align_corners_cubic_matrix(4, 8, -0.5)
        w_matrix = _align_corners_cubic_matrix(4, 8, -0.5)
        tmp = np.zeros((1, 8, 4, 3), dtype=np.float32)
        for out_h_idx in range(8):
            for in_h_idx in range(4):
                tmp[:, out_h_idx, :, :] += h_matrix[out_h_idx, in_h_idx] * x_nhwc[:, in_h_idx, :, :]
        ref_nhwc = np.zeros((1, 8, 8, 3), dtype=np.float32)
        for out_w_idx in range(8):
            for in_w_idx in range(4):
                ref_nhwc[:, :, out_w_idx, :] += w_matrix[out_w_idx, in_w_idx] * tmp[:, :, in_w_idx, :]
        ref_nchw = np.transpose(ref_nhwc, [0, 3, 1, 2])

        max_abs_err = float(np.max(np.abs(ref_nchw - tflite_y)))
        assert max_abs_err <= 1e-4


def test_flatbuffer_direct_constant_of_shape_lowering() -> None:
    model = _make_constant_of_shape_model()
    register_default_preprocess_rules()
    preprocessed_model, _ = run_preprocess_pipeline(onnx_graph=model)
    model_ir = lower_onnx_to_ir(
        onnx_graph=preprocessed_model,
        output_file_name="constant_of_shape_lowering_test",
        allow_custom_ops=False,
    )
    op_types = [str(op.op_type) for op in model_ir.operators]
    assert op_types.count("FILL") == 1
    assert op_types.count("CUSTOM") == 0


def test_flatbuffer_direct_fused_matmul_lowering() -> None:
    model = _make_fused_matmul_model()
    register_default_preprocess_rules()
    preprocessed_model, _ = run_preprocess_pipeline(onnx_graph=model)
    model_ir = lower_onnx_to_ir(
        onnx_graph=preprocessed_model,
        output_file_name="fused_matmul_lowering_test",
        allow_custom_ops=False,
    )
    op_types = [str(op.op_type) for op in model_ir.operators]
    assert op_types.count("BATCH_MATMUL") == 1
    assert op_types.count("MUL") == 1
    assert op_types.count("CUSTOM") == 0


def test_flatbuffer_direct_matmul_rank4_elides_boundary_input_transpose() -> None:
    model = _make_matmul_rank4_model()
    register_default_preprocess_rules()
    preprocessed_model, _ = run_preprocess_pipeline(onnx_graph=model)
    model_ir = lower_onnx_to_ir(
        onnx_graph=preprocessed_model,
        output_file_name="matmul_rank4_layout_opt_test",
        allow_custom_ops=False,
        transpose_inputs_to_nhwc=True,
    )

    op_types = [str(op.op_type) for op in model_ir.operators]
    assert op_types.count("BATCH_MATMUL") == 1
    assert op_types.count("TRANSPOSE") == 0

    bmm_op = next(op for op in model_ir.operators if str(op.op_type) == "BATCH_MATMUL")
    assert list(bmm_op.inputs) == ["x0", "x1"]
    assert list(model_ir.tensors["x0"].shape) == [1, 3, 8, 8]
    assert list(model_ir.tensors["x1"].shape) == [1, 3, 8, 8]


def test_flatbuffer_direct_reduce_l2_avoids_redundant_shape_reshape_identity() -> None:
    model = _make_reduce_l2_model()
    register_default_preprocess_rules()
    preprocessed_model, _ = run_preprocess_pipeline(onnx_graph=model)
    model_ir = lower_onnx_to_ir(
        onnx_graph=preprocessed_model,
        output_file_name="reduce_l2_identity_shape_reshape_elision_test",
        allow_custom_ops=False,
    )

    op_types = [str(op.op_type) for op in model_ir.operators]
    assert op_types.count("SHAPE") == 0
    assert op_types.count("RESHAPE") == 0

    sqrt_ops = [op for op in model_ir.operators if str(op.op_type) == "SQRT"]
    assert len(sqrt_ops) == 1
    sqrt_op = sqrt_ops[0]
    assert len(list(sqrt_op.inputs)) == 1
    assert len(list(sqrt_op.outputs)) == 1
    assert str(sqrt_op.inputs[0]) != str(sqrt_op.outputs[0])
    assert str(sqrt_op.outputs[0]) == "y"


def test_flatbuffer_direct_resolve_dynamic_reshape_zero_copy_dims_pass() -> None:
    model_ir = ModelIR(name="reshape_zero_copy_fixup_test")
    model_ir.inputs = ["x"]
    model_ir.outputs = ["y"]
    model_ir.tensors["x"] = TensorIR(
        name="x",
        dtype="FLOAT32",
        shape=[1, 1, 2, 256],
        shape_signature=[1, 1, 2, 256],
    )
    model_ir.tensors["reshape_shape"] = TensorIR(
        name="reshape_shape",
        dtype="INT32",
        shape=[3],
        shape_signature=[3],
        data=np.asarray([0, 0, -1], dtype=np.int32),
    )
    model_ir.tensors["y"] = TensorIR(
        name="y",
        dtype="FLOAT32",
        shape=[1],
        shape_signature=[-1],
    )
    model_ir.operators.append(
        OperatorIR(
            op_type="RESHAPE",
            inputs=["x", "reshape_shape"],
            outputs=["y"],
            options={
                "newShape": [0, 0, -1],
                "onnxRawNewShape": [0, 0, -1],
                "allowZero": False,
            },
        )
    )

    stats = _resolve_dynamic_reshape_shapes(model_ir)
    assert stats["resolved_dynamic_reshape_shapes"] == 1
    assert list(model_ir.operators[0].options["newShape"]) == [1, 1, 512]
    assert list(model_ir.tensors["y"].shape) == [1, 1, 512]
    assert list(model_ir.tensors["y"].shape_signature) == [1, 1, 512]
    assert np.asarray(model_ir.tensors["reshape_shape"].data).reshape(-1).tolist() == [1, 1, 512]


def test_flatbuffer_direct_reconcile_bilstm_chain_and_resolve_reshape_zero_copy_dims() -> None:
    model_ir = ModelIR(name="reshape_zero_copy_bilstm_chain_fixup_test")
    model_ir.inputs = ["x"]
    model_ir.outputs = ["y"]
    model_ir.tensors["x"] = TensorIR(
        name="x",
        dtype="FLOAT32",
        shape=[25, 1, 512],
        shape_signature=[25, 1, 512],
    )
    model_ir.tensors["merged"] = TensorIR(
        name="merged",
        dtype="FLOAT32",
        shape=[1, 1, 512],
        shape_signature=[1, 1, 512],
    )
    model_ir.tensors["split_axis"] = TensorIR(
        name="split_axis",
        dtype="INT32",
        shape=[1],
        shape_signature=[1],
        data=np.asarray([2], dtype=np.int32),
    )
    model_ir.tensors["fw"] = TensorIR(
        name="fw",
        dtype="FLOAT32",
        shape=[1, 1, 256],
        shape_signature=[1, 1, 256],
    )
    model_ir.tensors["bw"] = TensorIR(
        name="bw",
        dtype="FLOAT32",
        shape=[1, 1, 256],
        shape_signature=[1, 1, 256],
    )
    model_ir.tensors["expand_axis"] = TensorIR(
        name="expand_axis",
        dtype="INT32",
        shape=[1],
        shape_signature=[1],
        data=np.asarray([1], dtype=np.int32),
    )
    model_ir.tensors["fw_expanded"] = TensorIR(
        name="fw_expanded",
        dtype="FLOAT32",
        shape=[1, 1, 1, 256],
        shape_signature=[1, 1, 1, 256],
    )
    model_ir.tensors["bw_expanded"] = TensorIR(
        name="bw_expanded",
        dtype="FLOAT32",
        shape=[1, 1, 1, 256],
        shape_signature=[1, 1, 1, 256],
    )
    model_ir.tensors["lstm_y"] = TensorIR(
        name="lstm_y",
        dtype="FLOAT32",
        shape=[1, 2, 1, 256],
        shape_signature=[1, 2, 1, 256],
    )
    model_ir.tensors["transpose_perm"] = TensorIR(
        name="transpose_perm",
        dtype="INT32",
        shape=[4],
        shape_signature=[4],
        data=np.asarray([0, 2, 1, 3], dtype=np.int32),
    )
    model_ir.tensors["transpose_out"] = TensorIR(
        name="transpose_out",
        dtype="FLOAT32",
        shape=[1, 1, 2, 256],
        shape_signature=[1, 1, 2, 256],
    )
    model_ir.tensors["reshape_shape"] = TensorIR(
        name="reshape_shape",
        dtype="INT32",
        shape=[3],
        shape_signature=[3],
        data=np.asarray([0, 0, -1], dtype=np.int32),
    )
    model_ir.tensors["y"] = TensorIR(
        name="y",
        dtype="FLOAT32",
        shape=[1, 1, 512],
        shape_signature=[1, 1, 512],
    )
    model_ir.operators.extend(
        [
            OperatorIR(
                op_type="BIDIRECTIONAL_SEQUENCE_LSTM",
                inputs=["x"],
                outputs=["merged"],
                options={"timeMajor": True},
            ),
            OperatorIR(
                op_type="SPLIT",
                inputs=["split_axis", "merged"],
                outputs=["fw", "bw"],
                options={"numSplits": 2},
            ),
            OperatorIR(
                op_type="EXPAND_DIMS",
                inputs=["fw", "expand_axis"],
                outputs=["fw_expanded"],
            ),
            OperatorIR(
                op_type="EXPAND_DIMS",
                inputs=["bw", "expand_axis"],
                outputs=["bw_expanded"],
            ),
            OperatorIR(
                op_type="CONCATENATION",
                inputs=["fw_expanded", "bw_expanded"],
                outputs=["lstm_y"],
                options={"axis": 1},
            ),
            OperatorIR(
                op_type="TRANSPOSE",
                inputs=["lstm_y", "transpose_perm"],
                outputs=["transpose_out"],
            ),
            OperatorIR(
                op_type="RESHAPE",
                inputs=["transpose_out", "reshape_shape"],
                outputs=["y"],
                options={
                    "newShape": [0, 0, -1],
                    "onnxRawNewShape": [0, 0, -1],
                    "allowZero": False,
                },
            ),
        ]
    )

    stats_reconcile = _reconcile_static_tensor_shapes(model_ir)
    stats_resolve = _resolve_dynamic_reshape_shapes(model_ir)
    assert stats_reconcile["reconciled_static_tensor_shapes"] > 0
    assert stats_resolve["resolved_dynamic_reshape_shapes"] == 1
    assert list(model_ir.tensors["transpose_out"].shape) == [25, 1, 2, 256]
    assert list(model_ir.tensors["y"].shape) == [25, 1, 512]
    assert np.asarray(model_ir.tensors["reshape_shape"].data).reshape(-1).tolist() == [25, 1, 512]


def test_flatbuffer_direct_reconcile_cast_propagates_output_shape_signature() -> None:
    model_ir = ModelIR(name="reconcile_cast_shape_signature_test")
    model_ir.inputs = ["x"]
    model_ir.outputs = ["y"]
    model_ir.tensors["x"] = TensorIR(
        name="x",
        dtype="INT32",
        shape=[1, 2],
        shape_signature=[-1, 2],
    )
    model_ir.tensors["y"] = TensorIR(
        name="y",
        dtype="FLOAT32",
        shape=[1, 2],
        shape_signature=[1, 2],
    )
    model_ir.operators = [
        OperatorIR(
            op_type="CAST",
            inputs=["x"],
            outputs=["y"],
            options={"inDataType": "INT32", "outDataType": "FLOAT32"},
        )
    ]

    stats = _reconcile_static_tensor_shapes(model_ir)
    assert stats["reconciled_static_tensor_shapes"] >= 1
    assert list(model_ir.tensors["y"].shape) == [1, 2]
    assert list(model_ir.tensors["y"].shape_signature) == [-1, 2]


def test_flatbuffer_direct_reconcile_neg_propagates_output_shape_signature() -> None:
    model_ir = ModelIR(name="reconcile_neg_shape_signature_test")
    model_ir.inputs = ["x"]
    model_ir.outputs = ["y"]
    model_ir.tensors["x"] = TensorIR(
        name="x",
        dtype="FLOAT32",
        shape=[1, 2],
        shape_signature=[-1, 2],
    )
    model_ir.tensors["y"] = TensorIR(
        name="y",
        dtype="FLOAT32",
        shape=[1, 2],
        shape_signature=[1, 2],
    )
    model_ir.operators = [
        OperatorIR(
            op_type="NEG",
            inputs=["x"],
            outputs=["y"],
        )
    ]

    stats = _reconcile_static_tensor_shapes(model_ir)
    assert stats["reconciled_static_tensor_shapes"] >= 1
    assert list(model_ir.tensors["y"].shape) == [1, 2]
    assert list(model_ir.tensors["y"].shape_signature) == [-1, 2]


def test_flatbuffer_direct_reconcile_floor_mod_propagates_output_shape_signature() -> None:
    model_ir = ModelIR(name="reconcile_floor_mod_shape_signature_test")
    model_ir.inputs = ["x"]
    model_ir.outputs = ["y"]
    model_ir.tensors["x"] = TensorIR(
        name="x",
        dtype="INT32",
        shape=[1],
        shape_signature=[-1],
    )
    model_ir.tensors["c"] = TensorIR(
        name="c",
        dtype="INT32",
        shape=[1],
        shape_signature=[1],
        data=np.asarray([2], dtype=np.int32),
    )
    model_ir.tensors["y"] = TensorIR(
        name="y",
        dtype="INT32",
        shape=[1],
        shape_signature=[1],
    )
    model_ir.operators = [
        OperatorIR(
            op_type="FLOOR_MOD",
            inputs=["x", "c"],
            outputs=["y"],
        )
    ]

    stats = _reconcile_static_tensor_shapes(model_ir)
    assert stats["reconciled_static_tensor_shapes"] >= 1
    assert list(model_ir.tensors["y"].shape) == [1]
    assert list(model_ir.tensors["y"].shape_signature) == [-1]


def test_flatbuffer_direct_reconcile_maximum_propagates_output_shape_signature() -> None:
    model_ir = ModelIR(name="reconcile_maximum_shape_signature_test")
    model_ir.inputs = ["x"]
    model_ir.outputs = ["y"]
    model_ir.tensors["x"] = TensorIR(
        name="x",
        dtype="FLOAT32",
        shape=[1, 3, 1],
        shape_signature=[-1, 3, 1],
    )
    model_ir.tensors["c"] = TensorIR(
        name="c",
        dtype="FLOAT32",
        shape=[1, 1, 4],
        shape_signature=[1, 1, 4],
        data=np.asarray([[[0.0, 0.0, 0.0, 0.0]]], dtype=np.float32),
    )
    model_ir.tensors["y"] = TensorIR(
        name="y",
        dtype="FLOAT32",
        shape=[1, 3, 4],
        shape_signature=[1, 3, 4],
    )
    model_ir.operators = [
        OperatorIR(
            op_type="MAXIMUM",
            inputs=["x", "c"],
            outputs=["y"],
        )
    ]

    stats = _reconcile_static_tensor_shapes(model_ir)
    assert stats["reconciled_static_tensor_shapes"] >= 1
    assert list(model_ir.tensors["y"].shape) == [1, 3, 4]
    assert list(model_ir.tensors["y"].shape_signature) == [-1, 3, 4]


def test_flatbuffer_direct_reconcile_minimum_propagates_output_shape_signature() -> None:
    model_ir = ModelIR(name="reconcile_minimum_shape_signature_test")
    model_ir.inputs = ["x"]
    model_ir.outputs = ["y"]
    model_ir.tensors["x"] = TensorIR(
        name="x",
        dtype="FLOAT32",
        shape=[1, 3, 1],
        shape_signature=[-1, 3, 1],
    )
    model_ir.tensors["c"] = TensorIR(
        name="c",
        dtype="FLOAT32",
        shape=[1, 1, 4],
        shape_signature=[1, 1, 4],
        data=np.asarray([[[1.0, 1.0, 1.0, 1.0]]], dtype=np.float32),
    )
    model_ir.tensors["y"] = TensorIR(
        name="y",
        dtype="FLOAT32",
        shape=[1, 3, 4],
        shape_signature=[1, 3, 4],
    )
    model_ir.operators = [
        OperatorIR(
            op_type="MINIMUM",
            inputs=["x", "c"],
            outputs=["y"],
        )
    ]

    stats = _reconcile_static_tensor_shapes(model_ir)
    assert stats["reconciled_static_tensor_shapes"] >= 1
    assert list(model_ir.tensors["y"].shape) == [1, 3, 4]
    assert list(model_ir.tensors["y"].shape_signature) == [-1, 3, 4]


def test_flatbuffer_direct_reconcile_batch_matmul_propagates_output_shape_signature() -> None:
    model_ir = ModelIR(name="reconcile_batch_matmul_shape_signature_test")
    model_ir.inputs = ["a", "b"]
    model_ir.outputs = ["y"]
    model_ir.tensors["a"] = TensorIR(
        name="a",
        dtype="FLOAT32",
        shape=[1, 2, 4],
        shape_signature=[-1, 2, 4],
    )
    model_ir.tensors["b"] = TensorIR(
        name="b",
        dtype="FLOAT32",
        shape=[1, 4, 3],
        shape_signature=[-1, 4, 3],
    )
    model_ir.tensors["y"] = TensorIR(
        name="y",
        dtype="FLOAT32",
        shape=[1, 2, 3],
        shape_signature=[1, 2, 3],
    )
    model_ir.operators = [
        OperatorIR(
            op_type="BATCH_MATMUL",
            inputs=["a", "b"],
            outputs=["y"],
            options={"adjX": False, "adjY": False},
        )
    ]

    stats = _reconcile_static_tensor_shapes(model_ir)
    assert stats["reconciled_static_tensor_shapes"] >= 1
    assert list(model_ir.tensors["y"].shape) == [1, 2, 3]
    assert list(model_ir.tensors["y"].shape_signature) == [-1, 2, 3]


def test_flatbuffer_direct_reconcile_slice_preserves_dynamic_full_extent_size() -> None:
    model_ir = ModelIR(name="reconcile_slice_dynamic_full_extent_test")
    model_ir.inputs = ["x"]
    model_ir.outputs = ["y"]
    model_ir.tensors["x"] = TensorIR(
        name="x",
        dtype="FLOAT32",
        shape=[1, 1, 1, 2],
        shape_signature=[1, 1, -1, 2],
    )
    model_ir.tensors["slice_begin"] = TensorIR(
        name="slice_begin",
        dtype="INT32",
        shape=[4],
        shape_signature=[4],
        data=np.asarray([0, 0, 0, 0], dtype=np.int32),
        is_variable=False,
    )
    model_ir.tensors["slice_size"] = TensorIR(
        name="slice_size",
        dtype="INT32",
        shape=[4],
        shape_signature=[4],
        data=np.asarray([-1, -1, -1, 1], dtype=np.int32),
        is_variable=False,
    )
    model_ir.tensors["y"] = TensorIR(
        name="y",
        dtype="FLOAT32",
        shape=[1, 1, 1, 1],
        shape_signature=[1, 1, -1, 1],
    )
    model_ir.operators = [
        OperatorIR(
            op_type="SLICE",
            inputs=["x", "slice_begin", "slice_size"],
            outputs=["y"],
        )
    ]

    _ = _reconcile_static_tensor_shapes(model_ir)
    slice_size_values = np.asarray(model_ir.tensors["slice_size"].data, dtype=np.int32).reshape(-1).tolist()
    assert slice_size_values == [-1, -1, -1, 1]
    assert list(model_ir.tensors["y"].shape_signature) == [1, 1, -1, 1]


def test_flatbuffer_direct_transpose_mul_add_const_rewrite_keeps_nhwc_consts_stable() -> None:
    model_ir = ModelIR(name="transpose_mul_add_const_rewrite_idempotent_const_shape_test")
    model_ir.inputs = ["x_nhwc"]
    model_ir.outputs = ["z"]
    model_ir.tensors["x_nhwc"] = TensorIR(
        name="x_nhwc",
        dtype="FLOAT32",
        shape=[1, 56, 56, 40],
        shape_signature=[1, 56, 56, 40],
    )
    model_ir.tensors["pre_perm"] = TensorIR(
        name="pre_perm",
        dtype="INT32",
        shape=[4],
        shape_signature=[4],
        data=np.asarray([0, 3, 1, 2], dtype=np.int32),
        is_variable=False,
    )
    model_ir.tensors["x_nchw"] = TensorIR(
        name="x_nchw",
        dtype="FLOAT32",
        shape=[1, 40, 56, 56],
        shape_signature=[1, 40, 56, 56],
    )
    model_ir.tensors["mul_const"] = TensorIR(
        name="mul_const",
        dtype="FLOAT32",
        shape=[1, 1, 1, 40],
        shape_signature=[1, 1, 1, 40],
        data=np.ones((1, 1, 1, 40), dtype=np.float32),
        is_variable=False,
    )
    model_ir.tensors["mul_out"] = TensorIR(
        name="mul_out",
        dtype="FLOAT32",
        shape=[1, 40, 56, 56],
        shape_signature=[1, 40, 56, 56],
    )
    model_ir.tensors["add_const"] = TensorIR(
        name="add_const",
        dtype="FLOAT32",
        shape=[1, 1, 1, 40],
        shape_signature=[1, 1, 1, 40],
        data=np.ones((1, 1, 1, 40), dtype=np.float32),
        is_variable=False,
    )
    model_ir.tensors["add_out"] = TensorIR(
        name="add_out",
        dtype="FLOAT32",
        shape=[1, 40, 56, 56],
        shape_signature=[1, 40, 56, 56],
    )
    model_ir.tensors["post_perm"] = TensorIR(
        name="post_perm",
        dtype="INT32",
        shape=[4],
        shape_signature=[4],
        data=np.asarray([0, 2, 3, 1], dtype=np.int32),
        is_variable=False,
    )
    model_ir.tensors["y_nhwc"] = TensorIR(
        name="y_nhwc",
        dtype="FLOAT32",
        shape=[1, 56, 56, 40],
        shape_signature=[1, 56, 56, 40],
    )
    model_ir.tensors["z"] = TensorIR(
        name="z",
        dtype="FLOAT32",
        shape=[1, 56, 56, 40],
        shape_signature=[1, 56, 56, 40],
    )
    model_ir.operators = [
        OperatorIR(op_type="TRANSPOSE", inputs=["x_nhwc", "pre_perm"], outputs=["x_nchw"]),
        OperatorIR(
            op_type="MUL",
            inputs=["x_nchw", "mul_const"],
            outputs=["mul_out"],
            options={"fusedActivationFunction": "NONE"},
        ),
        OperatorIR(
            op_type="ADD",
            inputs=["mul_out", "add_const"],
            outputs=["add_out"],
            options={"fusedActivationFunction": "NONE"},
        ),
        OperatorIR(op_type="TRANSPOSE", inputs=["add_out", "post_perm"], outputs=["y_nhwc"]),
        OperatorIR(op_type="RELU", inputs=["y_nhwc"], outputs=["z"]),
    ]

    stats = _optimize_transpose_mul_add_const_prepost_nhwc_chains(model_ir)
    assert stats["optimized_transpose_mul_add_const_prepost_nhwc_chains"] == 1
    assert list(model_ir.tensors["mul_const"].shape) == [1, 1, 1, 40]
    assert list(model_ir.tensors["add_const"].shape) == [1, 1, 1, 40]

    stats_second = _optimize_transpose_mul_add_const_prepost_nhwc_chains(model_ir)
    assert stats_second["optimized_transpose_mul_add_const_prepost_nhwc_chains"] == 0
    assert list(model_ir.tensors["mul_const"].shape) == [1, 1, 1, 40]
    assert list(model_ir.tensors["add_const"].shape) == [1, 1, 1, 40]


def test_flatbuffer_direct_transpose_mul_add_const_rewrite_does_not_mutate_on_partial_match() -> None:
    model_ir = ModelIR(name="transpose_mul_add_const_partial_match_no_mutation_test")
    model_ir.inputs = ["x_nhwc", "residual_nchw"]
    model_ir.outputs = ["z"]
    model_ir.tensors["x_nhwc"] = TensorIR(
        name="x_nhwc",
        dtype="FLOAT32",
        shape=[1, 56, 56, 40],
        shape_signature=[1, 56, 56, 40],
    )
    model_ir.tensors["residual_nchw"] = TensorIR(
        name="residual_nchw",
        dtype="FLOAT32",
        shape=[1, 40, 56, 56],
        shape_signature=[1, 40, 56, 56],
    )
    model_ir.tensors["pre_perm"] = TensorIR(
        name="pre_perm",
        dtype="INT32",
        shape=[4],
        shape_signature=[4],
        data=np.asarray([0, 3, 1, 2], dtype=np.int32),
        is_variable=False,
    )
    model_ir.tensors["x_nchw"] = TensorIR(
        name="x_nchw",
        dtype="FLOAT32",
        shape=[1, 40, 56, 56],
        shape_signature=[1, 40, 56, 56],
    )
    model_ir.tensors["mul_const"] = TensorIR(
        name="mul_const",
        dtype="FLOAT32",
        shape=[1, 40, 1, 1],
        shape_signature=[1, 40, 1, 1],
        data=np.ones((1, 40, 1, 1), dtype=np.float32),
        is_variable=False,
    )
    model_ir.tensors["mul_out"] = TensorIR(
        name="mul_out",
        dtype="FLOAT32",
        shape=[1, 40, 56, 56],
        shape_signature=[1, 40, 56, 56],
    )
    model_ir.tensors["add_out"] = TensorIR(
        name="add_out",
        dtype="FLOAT32",
        shape=[1, 40, 56, 56],
        shape_signature=[1, 40, 56, 56],
    )
    model_ir.tensors["post_perm"] = TensorIR(
        name="post_perm",
        dtype="INT32",
        shape=[4],
        shape_signature=[4],
        data=np.asarray([0, 2, 3, 1], dtype=np.int32),
        is_variable=False,
    )
    model_ir.tensors["y_nhwc"] = TensorIR(
        name="y_nhwc",
        dtype="FLOAT32",
        shape=[1, 56, 56, 40],
        shape_signature=[1, 56, 56, 40],
    )
    model_ir.tensors["z"] = TensorIR(
        name="z",
        dtype="FLOAT32",
        shape=[1, 56, 56, 40],
        shape_signature=[1, 56, 56, 40],
    )
    model_ir.operators = [
        OperatorIR(op_type="TRANSPOSE", inputs=["x_nhwc", "pre_perm"], outputs=["x_nchw"]),
        OperatorIR(
            op_type="MUL",
            inputs=["x_nchw", "mul_const"],
            outputs=["mul_out"],
            options={"fusedActivationFunction": "NONE"},
        ),
        OperatorIR(
            op_type="ADD",
            inputs=["mul_out", "residual_nchw"],
            outputs=["add_out"],
            options={"fusedActivationFunction": "NONE"},
        ),
        OperatorIR(op_type="TRANSPOSE", inputs=["add_out", "post_perm"], outputs=["y_nhwc"]),
        OperatorIR(op_type="RELU", inputs=["y_nhwc"], outputs=["z"]),
    ]

    stats = _optimize_transpose_mul_add_const_prepost_nhwc_chains(model_ir)
    assert stats["optimized_transpose_mul_add_const_prepost_nhwc_chains"] == 0
    assert list(model_ir.tensors["mul_const"].shape) == [1, 40, 1, 1]


def test_flatbuffer_direct_transpose_elementwise_roundtrip_nhwc_nchw_fanout_chain_optimized() -> None:
    model_ir = ModelIR(name="transpose_elementwise_roundtrip_nhwc_nchw_fanout_chain_opt_test")
    model_ir.inputs = ["x_nhwc"]
    model_ir.outputs = ["z0", "z1"]

    model_ir.tensors["x_nhwc"] = TensorIR(
        name="x_nhwc",
        dtype="FLOAT32",
        shape=[1, 6, 6, 8],
        shape_signature=[1, 6, 6, 8],
    )
    model_ir.tensors["pre_perm"] = TensorIR(
        name="pre_perm",
        dtype="INT32",
        shape=[4],
        shape_signature=[4],
        data=np.asarray([0, 3, 1, 2], dtype=np.int32),
        is_variable=False,
    )
    model_ir.tensors["post_perm"] = TensorIR(
        name="post_perm",
        dtype="INT32",
        shape=[4],
        shape_signature=[4],
        data=np.asarray([0, 2, 3, 1], dtype=np.int32),
        is_variable=False,
    )
    model_ir.tensors["x_nchw"] = TensorIR(
        name="x_nchw",
        dtype="FLOAT32",
        shape=[1, 8, 6, 6],
        shape_signature=[1, 8, 6, 6],
    )
    model_ir.tensors["mul_scale"] = TensorIR(
        name="mul_scale",
        dtype="FLOAT32",
        shape=[1, 8, 1, 1],
        shape_signature=[1, 8, 1, 1],
        data=np.ones((1, 8, 1, 1), dtype=np.float32),
        is_variable=False,
    )
    model_ir.tensors["branch0_mul"] = TensorIR(
        name="branch0_mul",
        dtype="FLOAT32",
        shape=[1, 8, 6, 6],
        shape_signature=[1, 8, 6, 6],
    )
    model_ir.tensors["branch0_erf"] = TensorIR(
        name="branch0_erf",
        dtype="FLOAT32",
        shape=[1, 8, 6, 6],
        shape_signature=[1, 8, 6, 6],
    )
    model_ir.tensors["add_bias"] = TensorIR(
        name="add_bias",
        dtype="FLOAT32",
        shape=[1, 8, 1, 1],
        shape_signature=[1, 8, 1, 1],
        data=np.zeros((1, 8, 1, 1), dtype=np.float32),
        is_variable=False,
    )
    model_ir.tensors["branch0_out_nchw"] = TensorIR(
        name="branch0_out_nchw",
        dtype="FLOAT32",
        shape=[1, 8, 6, 6],
        shape_signature=[1, 8, 6, 6],
    )
    model_ir.tensors["branch1_sign"] = TensorIR(
        name="branch1_sign",
        dtype="FLOAT32",
        shape=[1, 8, 6, 6],
        shape_signature=[1, 8, 6, 6],
    )
    model_ir.tensors["branch1_scale"] = TensorIR(
        name="branch1_scale",
        dtype="FLOAT32",
        shape=[1, 8, 1, 1],
        shape_signature=[1, 8, 1, 1],
        data=np.full((1, 8, 1, 1), 0.5, dtype=np.float32),
        is_variable=False,
    )
    model_ir.tensors["branch1_out_nchw"] = TensorIR(
        name="branch1_out_nchw",
        dtype="FLOAT32",
        shape=[1, 8, 6, 6],
        shape_signature=[1, 8, 6, 6],
    )
    model_ir.tensors["y0"] = TensorIR(
        name="y0",
        dtype="FLOAT32",
        shape=[1, 6, 6, 8],
        shape_signature=[1, 6, 6, 8],
    )
    model_ir.tensors["y1"] = TensorIR(
        name="y1",
        dtype="FLOAT32",
        shape=[1, 6, 6, 8],
        shape_signature=[1, 6, 6, 8],
    )
    model_ir.tensors["z0"] = TensorIR(
        name="z0",
        dtype="FLOAT32",
        shape=[1, 6, 6, 8],
        shape_signature=[1, 6, 6, 8],
    )
    model_ir.tensors["z1"] = TensorIR(
        name="z1",
        dtype="FLOAT32",
        shape=[1, 6, 6, 8],
        shape_signature=[1, 6, 6, 8],
    )
    model_ir.operators = [
        OperatorIR(op_type="TRANSPOSE", inputs=["x_nhwc", "pre_perm"], outputs=["x_nchw"]),
        OperatorIR(op_type="MUL", inputs=["x_nchw", "mul_scale"], outputs=["branch0_mul"]),
        OperatorIR(op_type="ERF", inputs=["branch0_mul"], outputs=["branch0_erf"]),
        OperatorIR(op_type="ADD", inputs=["branch0_erf", "add_bias"], outputs=["branch0_out_nchw"]),
        OperatorIR(op_type="SIGN", inputs=["x_nchw"], outputs=["branch1_sign"]),
        OperatorIR(op_type="MUL", inputs=["branch1_sign", "branch1_scale"], outputs=["branch1_out_nchw"]),
        OperatorIR(op_type="TRANSPOSE", inputs=["branch0_out_nchw", "post_perm"], outputs=["y0"]),
        OperatorIR(op_type="TRANSPOSE", inputs=["branch1_out_nchw", "post_perm"], outputs=["y1"]),
        OperatorIR(op_type="RELU", inputs=["y0"], outputs=["z0"]),
        OperatorIR(op_type="RELU", inputs=["y1"], outputs=["z1"]),
    ]

    stats = _optimize_transpose_elementwise_roundtrip_nhwc_nchw_fanout_chains(model_ir)
    assert stats["optimized_transpose_elementwise_roundtrip_nhwc_nchw_fanout_chains"] == 1

    assert not any(str(op.op_type) == "TRANSPOSE" for op in model_ir.operators)
    assert list(model_ir.tensors["mul_scale"].shape) == [1, 1, 1, 8]
    assert list(model_ir.tensors["add_bias"].shape) == [1, 1, 1, 8]
    assert list(model_ir.tensors["branch1_scale"].shape) == [1, 1, 1, 8]

    mul0_op = next(op for op in model_ir.operators if list(op.outputs) == ["branch0_mul"])
    sign_op = next(op for op in model_ir.operators if list(op.outputs) == ["branch1_sign"])
    assert [str(v) for v in list(mul0_op.inputs)][0] == "x_nhwc"
    assert [str(v) for v in list(sign_op.inputs)] == ["x_nhwc"]

    add_op = next(op for op in model_ir.operators if str(op.op_type) == "ADD")
    mul1_op = next(op for op in model_ir.operators if str(op.op_type) == "MUL" and list(op.outputs) == ["y1"])
    assert list(add_op.outputs) == ["y0"]
    assert list(mul1_op.outputs) == ["y1"]


def test_flatbuffer_direct_conv_output_transpose_nhwc_passthrough_chain_optimized() -> None:
    model_ir = ModelIR(name="conv_output_transpose_nhwc_passthrough_opt_test")
    model_ir.inputs = ["x_nhwc"]
    model_ir.outputs = ["z"]

    model_ir.tensors["x_nhwc"] = TensorIR(
        name="x_nhwc",
        dtype="FLOAT32",
        shape=[1, 6, 6, 8],
        shape_signature=[1, 6, 6, 8],
    )
    model_ir.tensors["conv_filter"] = TensorIR(
        name="conv_filter",
        dtype="FLOAT32",
        shape=[1, 1, 8, 8],
        shape_signature=[1, 1, 8, 8],
        data=np.ones((1, 1, 8, 8), dtype=np.float32),
        is_variable=False,
    )
    model_ir.tensors["conv_bias"] = TensorIR(
        name="conv_bias",
        dtype="FLOAT32",
        shape=[8],
        shape_signature=[8],
        data=np.zeros((8,), dtype=np.float32),
        is_variable=False,
    )
    model_ir.tensors["conv_out_nhwc"] = TensorIR(
        name="conv_out_nhwc",
        dtype="FLOAT32",
        shape=[1, 6, 6, 8],
        shape_signature=[1, 6, 6, 8],
    )
    model_ir.tensors["pre_perm"] = TensorIR(
        name="pre_perm",
        dtype="INT32",
        shape=[4],
        shape_signature=[4],
        data=np.asarray([0, 3, 1, 2], dtype=np.int32),
        is_variable=False,
    )
    model_ir.tensors["conv_out_nchw"] = TensorIR(
        name="conv_out_nchw",
        dtype="FLOAT32",
        shape=[1, 8, 6, 6],
        shape_signature=[1, 8, 6, 6],
    )
    model_ir.tensors["mul_scale"] = TensorIR(
        name="mul_scale",
        dtype="FLOAT32",
        shape=[1, 8, 1, 1],
        shape_signature=[1, 8, 1, 1],
        data=np.ones((1, 8, 1, 1), dtype=np.float32),
        is_variable=False,
    )
    model_ir.tensors["mul_out_nchw"] = TensorIR(
        name="mul_out_nchw",
        dtype="FLOAT32",
        shape=[1, 8, 6, 6],
        shape_signature=[1, 8, 6, 6],
    )
    model_ir.tensors["add_bias"] = TensorIR(
        name="add_bias",
        dtype="FLOAT32",
        shape=[1, 8, 1, 1],
        shape_signature=[1, 8, 1, 1],
        data=np.zeros((1, 8, 1, 1), dtype=np.float32),
        is_variable=False,
    )
    model_ir.tensors["add_out_nchw"] = TensorIR(
        name="add_out_nchw",
        dtype="FLOAT32",
        shape=[1, 8, 6, 6],
        shape_signature=[1, 8, 6, 6],
    )
    model_ir.tensors["z"] = TensorIR(
        name="z",
        dtype="FLOAT32",
        shape=[1, 8, 6, 6],
        shape_signature=[1, 8, 6, 6],
    )
    model_ir.operators = [
        OperatorIR(op_type="CONV_2D", inputs=["x_nhwc", "conv_filter", "conv_bias"], outputs=["conv_out_nhwc"]),
        OperatorIR(op_type="TRANSPOSE", inputs=["conv_out_nhwc", "pre_perm"], outputs=["conv_out_nchw"]),
        OperatorIR(op_type="MUL", inputs=["conv_out_nchw", "mul_scale"], outputs=["mul_out_nchw"]),
        OperatorIR(op_type="ADD", inputs=["mul_out_nchw", "add_bias"], outputs=["add_out_nchw"]),
        OperatorIR(op_type="RELU", inputs=["add_out_nchw"], outputs=["z"]),
    ]

    stats = _optimize_convpool_output_transpose_nhwc_passthrough_chains(model_ir)
    assert stats["optimized_convpool_output_transpose_nhwc_passthrough_chains"] == 1

    # Leading post-conv transpose is removed.
    assert not any(str(op.op_type) == "TRANSPOSE" and list(op.outputs) == ["conv_out_nchw"] for op in model_ir.operators)

    mul_op = next(op for op in model_ir.operators if str(op.op_type) == "MUL")
    assert [str(v) for v in list(mul_op.inputs)][0] == "conv_out_nhwc"


def test_flatbuffer_direct_transpose_pre_add_mul_add_prelu_nhwc_chain_optimized() -> None:
    model_ir = ModelIR(name="transpose_pre_add_mul_add_prelu_nhwc_chain_opt_test")
    model_ir.inputs = ["a_nhwc", "b_nhwc", "legacy_ref_nchw"]
    model_ir.outputs = ["z", "legacy_cat"]
    model_ir.tensors["a_nhwc"] = TensorIR(
        name="a_nhwc",
        dtype="FLOAT32",
        shape=[1, 6, 6, 32],
        shape_signature=[1, 6, 6, 32],
    )
    model_ir.tensors["b_nhwc"] = TensorIR(
        name="b_nhwc",
        dtype="FLOAT32",
        shape=[1, 6, 6, 32],
        shape_signature=[1, 6, 6, 32],
    )
    model_ir.tensors["legacy_ref_nchw"] = TensorIR(
        name="legacy_ref_nchw",
        dtype="FLOAT32",
        shape=[1, 32, 6, 6],
        shape_signature=[1, 32, 6, 6],
    )
    model_ir.tensors["pre_perm"] = TensorIR(
        name="pre_perm",
        dtype="INT32",
        shape=[4],
        shape_signature=[4],
        data=np.asarray([0, 3, 1, 2], dtype=np.int32),
        is_variable=False,
    )
    model_ir.tensors["post_perm"] = TensorIR(
        name="post_perm",
        dtype="INT32",
        shape=[4],
        shape_signature=[4],
        data=np.asarray([0, 2, 3, 1], dtype=np.int32),
        is_variable=False,
    )
    model_ir.tensors["a_nchw"] = TensorIR(
        name="a_nchw",
        dtype="FLOAT32",
        shape=[1, 32, 6, 6],
        shape_signature=[1, 32, 6, 6],
    )
    model_ir.tensors["b_nchw"] = TensorIR(
        name="b_nchw",
        dtype="FLOAT32",
        shape=[1, 32, 6, 6],
        shape_signature=[1, 32, 6, 6],
    )
    model_ir.tensors["add_out"] = TensorIR(
        name="add_out",
        dtype="FLOAT32",
        shape=[1, 32, 6, 6],
        shape_signature=[1, 32, 6, 6],
    )
    model_ir.tensors["mul_const"] = TensorIR(
        name="mul_const",
        dtype="FLOAT32",
        shape=[1, 32, 1, 1],
        shape_signature=[1, 32, 1, 1],
        data=np.ones((1, 32, 1, 1), dtype=np.float32),
        is_variable=False,
    )
    model_ir.tensors["mul_out"] = TensorIR(
        name="mul_out",
        dtype="FLOAT32",
        shape=[1, 32, 6, 6],
        shape_signature=[1, 32, 6, 6],
    )
    model_ir.tensors["add2_const"] = TensorIR(
        name="add2_const",
        dtype="FLOAT32",
        shape=[1, 32, 1, 1],
        shape_signature=[1, 32, 1, 1],
        data=np.zeros((1, 32, 1, 1), dtype=np.float32),
        is_variable=False,
    )
    model_ir.tensors["add2_out"] = TensorIR(
        name="add2_out",
        dtype="FLOAT32",
        shape=[1, 32, 6, 6],
        shape_signature=[1, 32, 6, 6],
    )
    model_ir.tensors["prelu_alpha"] = TensorIR(
        name="prelu_alpha",
        dtype="FLOAT32",
        shape=[1, 32, 1, 1],
        shape_signature=[1, 32, 1, 1],
        data=np.full((1, 32, 1, 1), 0.125, dtype=np.float32),
        is_variable=False,
    )
    model_ir.tensors["prelu_out"] = TensorIR(
        name="prelu_out",
        dtype="FLOAT32",
        shape=[1, 32, 6, 6],
        shape_signature=[1, 32, 6, 6],
    )
    model_ir.tensors["y_nhwc"] = TensorIR(
        name="y_nhwc",
        dtype="FLOAT32",
        shape=[1, 6, 6, 32],
        shape_signature=[1, 6, 6, 32],
    )
    model_ir.tensors["z"] = TensorIR(
        name="z",
        dtype="FLOAT32",
        shape=[1, 6, 6, 32],
        shape_signature=[1, 6, 6, 32],
    )
    model_ir.tensors["legacy_cat"] = TensorIR(
        name="legacy_cat",
        dtype="FLOAT32",
        shape=[1, 64, 6, 6],
        shape_signature=[1, 64, 6, 6],
    )
    model_ir.operators = [
        OperatorIR(op_type="TRANSPOSE", inputs=["a_nhwc", "pre_perm"], outputs=["a_nchw"]),
        OperatorIR(op_type="TRANSPOSE", inputs=["b_nhwc", "pre_perm"], outputs=["b_nchw"]),
        OperatorIR(
            op_type="ADD",
            inputs=["a_nchw", "b_nchw"],
            outputs=["add_out"],
            options={"fusedActivationFunction": "NONE"},
        ),
        OperatorIR(
            op_type="MUL",
            inputs=["add_out", "mul_const"],
            outputs=["mul_out"],
            options={"fusedActivationFunction": "NONE"},
        ),
        OperatorIR(
            op_type="ADD",
            inputs=["mul_out", "add2_const"],
            outputs=["add2_out"],
            options={"fusedActivationFunction": "NONE"},
        ),
        OperatorIR(op_type="PRELU", inputs=["add2_out", "prelu_alpha"], outputs=["prelu_out"]),
        OperatorIR(op_type="TRANSPOSE", inputs=["prelu_out", "post_perm"], outputs=["y_nhwc"]),
        OperatorIR(op_type="RELU", inputs=["y_nhwc"], outputs=["z"]),
        OperatorIR(
            op_type="CONCATENATION",
            inputs=["legacy_ref_nchw", "prelu_out"],
            outputs=["legacy_cat"],
            options={"axis": 1, "fusedActivationFunction": "NONE"},
        ),
    ]

    stats = _optimize_transpose_pre_add_mul_add_prelu_nhwc_chains(model_ir)
    assert stats["optimized_transpose_pre_add_mul_add_prelu_nhwc_chains"] == 1

    add_op = next(op for op in model_ir.operators if str(op.op_type) == "ADD" and list(op.outputs) == ["add_out"])
    assert list(add_op.inputs) == ["a_nhwc", "b_nhwc"]
    assert list(model_ir.tensors["prelu_alpha"].shape) == [1, 1, 1, 32]
    assert not any(
        str(op.op_type) == "TRANSPOSE" and list(op.outputs) in (["a_nchw"], ["b_nchw"])
        for op in model_ir.operators
    )
    assert any(
        str(op.op_type) == "TRANSPOSE" and list(op.outputs) == ["prelu_out"]
        for op in model_ir.operators
    )


def test_flatbuffer_direct_transpose_logistic_muladd_prepost_nhwc_chain_optimized() -> None:
    model_ir = ModelIR(name="transpose_logistic_muladd_prepost_nhwc_chain_opt_test")
    model_ir.inputs = ["skip_nhwc", "data_nhwc", "gate_nhwc"]
    model_ir.outputs = ["z"]
    model_ir.tensors["skip_nhwc"] = TensorIR(
        name="skip_nhwc",
        dtype="FLOAT32",
        shape=[1, 6, 6, 8],
        shape_signature=[1, 6, 6, 8],
    )
    model_ir.tensors["data_nhwc"] = TensorIR(
        name="data_nhwc",
        dtype="FLOAT32",
        shape=[1, 6, 6, 8],
        shape_signature=[1, 6, 6, 8],
    )
    model_ir.tensors["gate_nhwc"] = TensorIR(
        name="gate_nhwc",
        dtype="FLOAT32",
        shape=[1, 1, 1, 8],
        shape_signature=[1, 1, 1, 8],
    )
    model_ir.tensors["pre_perm"] = TensorIR(
        name="pre_perm",
        dtype="INT32",
        shape=[4],
        shape_signature=[4],
        data=np.asarray([0, 3, 1, 2], dtype=np.int32),
        is_variable=False,
    )
    model_ir.tensors["post_perm"] = TensorIR(
        name="post_perm",
        dtype="INT32",
        shape=[4],
        shape_signature=[4],
        data=np.asarray([0, 2, 3, 1], dtype=np.int32),
        is_variable=False,
    )
    model_ir.tensors["skip_nchw"] = TensorIR(
        name="skip_nchw",
        dtype="FLOAT32",
        shape=[1, 8, 6, 6],
        shape_signature=[1, 8, 6, 6],
    )
    model_ir.tensors["data_nchw"] = TensorIR(
        name="data_nchw",
        dtype="FLOAT32",
        shape=[1, 8, 6, 6],
        shape_signature=[1, 8, 6, 6],
    )
    model_ir.tensors["gate_nchw"] = TensorIR(
        name="gate_nchw",
        dtype="FLOAT32",
        shape=[1, 8, 1, 1],
        shape_signature=[1, 8, 1, 1],
    )
    model_ir.tensors["sig_nchw"] = TensorIR(
        name="sig_nchw",
        dtype="FLOAT32",
        shape=[1, 8, 1, 1],
        shape_signature=[1, 8, 1, 1],
    )
    model_ir.tensors["mul_nchw"] = TensorIR(
        name="mul_nchw",
        dtype="FLOAT32",
        shape=[1, 8, 6, 6],
        shape_signature=[1, 8, 6, 6],
    )
    model_ir.tensors["add_nchw"] = TensorIR(
        name="add_nchw",
        dtype="FLOAT32",
        shape=[1, 8, 6, 6],
        shape_signature=[1, 8, 6, 6],
    )
    model_ir.tensors["y_nhwc"] = TensorIR(
        name="y_nhwc",
        dtype="FLOAT32",
        shape=[1, 6, 6, 8],
        shape_signature=[1, 6, 6, 8],
    )
    model_ir.tensors["reshape_shape"] = TensorIR(
        name="reshape_shape",
        dtype="INT64",
        shape=[3],
        shape_signature=[3],
        data=np.asarray([1, 36, 8], dtype=np.int64),
        is_variable=False,
    )
    model_ir.tensors["z"] = TensorIR(
        name="z",
        dtype="FLOAT32",
        shape=[1, 36, 8],
        shape_signature=[1, 36, 8],
    )
    model_ir.operators = [
        OperatorIR(op_type="TRANSPOSE", inputs=["skip_nhwc", "pre_perm"], outputs=["skip_nchw"]),
        OperatorIR(op_type="TRANSPOSE", inputs=["data_nhwc", "pre_perm"], outputs=["data_nchw"]),
        OperatorIR(op_type="TRANSPOSE", inputs=["gate_nhwc", "pre_perm"], outputs=["gate_nchw"]),
        OperatorIR(op_type="LOGISTIC", inputs=["gate_nchw"], outputs=["sig_nchw"]),
        OperatorIR(op_type="MUL", inputs=["data_nchw", "sig_nchw"], outputs=["mul_nchw"]),
        OperatorIR(op_type="ADD", inputs=["mul_nchw", "skip_nchw"], outputs=["add_nchw"]),
        OperatorIR(op_type="TRANSPOSE", inputs=["add_nchw", "post_perm"], outputs=["y_nhwc"]),
        OperatorIR(op_type="RESHAPE", inputs=["y_nhwc", "reshape_shape"], outputs=["z"]),
    ]

    stats = _optimize_transpose_logistic_muladd_prepost_nhwc_chains(model_ir)
    assert stats["optimized_transpose_logistic_muladd_prepost_nhwc_chains"] == 1

    op_types = [str(op.op_type) for op in model_ir.operators]
    assert op_types.count("TRANSPOSE") == 0

    logistic_op = next(op for op in model_ir.operators if str(op.op_type) == "LOGISTIC")
    assert [str(v) for v in list(logistic_op.inputs)] == ["gate_nhwc"]

    mul_op = next(op for op in model_ir.operators if str(op.op_type) == "MUL")
    assert set(str(v) for v in list(mul_op.inputs)) == {"data_nhwc", "sig_nchw"}

    add_op = next(op for op in model_ir.operators if str(op.op_type) == "ADD")
    assert list(add_op.outputs) == ["y_nhwc"]
    assert set(str(v) for v in list(add_op.inputs)) == {"mul_nchw", "skip_nhwc"}

    reshape_op = next(op for op in model_ir.operators if str(op.op_type) == "RESHAPE")
    assert [str(v) for v in list(reshape_op.inputs)] == ["y_nhwc", "reshape_shape"]

    assert list(model_ir.tensors["sig_nchw"].shape) == [1, 1, 1, 8]
    assert list(model_ir.tensors["mul_nchw"].shape) == [1, 6, 6, 8]


def test_flatbuffer_direct_transpose_logistic_muladd_prepost_nhwc_chain_optimized_with_legacy_users() -> None:
    model_ir = ModelIR(name="transpose_logistic_muladd_prepost_nhwc_chain_opt_legacy_test")
    model_ir.inputs = ["skip_nhwc", "data_nhwc", "gate_nhwc"]
    model_ir.outputs = ["z", "legacy_z"]
    model_ir.tensors["skip_nhwc"] = TensorIR(
        name="skip_nhwc",
        dtype="FLOAT32",
        shape=[1, 6, 6, 8],
        shape_signature=[1, 6, 6, 8],
    )
    model_ir.tensors["data_nhwc"] = TensorIR(
        name="data_nhwc",
        dtype="FLOAT32",
        shape=[1, 6, 6, 8],
        shape_signature=[1, 6, 6, 8],
    )
    model_ir.tensors["gate_nhwc"] = TensorIR(
        name="gate_nhwc",
        dtype="FLOAT32",
        shape=[1, 1, 1, 8],
        shape_signature=[1, 1, 1, 8],
    )
    model_ir.tensors["pre_perm"] = TensorIR(
        name="pre_perm",
        dtype="INT32",
        shape=[4],
        shape_signature=[4],
        data=np.asarray([0, 3, 1, 2], dtype=np.int32),
        is_variable=False,
    )
    model_ir.tensors["post_perm"] = TensorIR(
        name="post_perm",
        dtype="INT32",
        shape=[4],
        shape_signature=[4],
        data=np.asarray([0, 2, 3, 1], dtype=np.int32),
        is_variable=False,
    )
    model_ir.tensors["mean_axes"] = TensorIR(
        name="mean_axes",
        dtype="INT32",
        shape=[2],
        shape_signature=[2],
        data=np.asarray([2, 3], dtype=np.int32),
        is_variable=False,
    )
    model_ir.tensors["reshape_shape"] = TensorIR(
        name="reshape_shape",
        dtype="INT64",
        shape=[3],
        shape_signature=[3],
        data=np.asarray([1, 36, 8], dtype=np.int64),
        is_variable=False,
    )
    model_ir.tensors["legacy_bias"] = TensorIR(
        name="legacy_bias",
        dtype="FLOAT32",
        shape=[1, 8, 6, 6],
        shape_signature=[1, 8, 6, 6],
        data=np.ones((1, 8, 6, 6), dtype=np.float32),
        is_variable=False,
    )
    model_ir.tensors["skip_nchw"] = TensorIR(
        name="skip_nchw",
        dtype="FLOAT32",
        shape=[1, 8, 6, 6],
        shape_signature=[1, 8, 6, 6],
    )
    model_ir.tensors["data_nchw"] = TensorIR(
        name="data_nchw",
        dtype="FLOAT32",
        shape=[1, 8, 6, 6],
        shape_signature=[1, 8, 6, 6],
    )
    model_ir.tensors["gate_nchw"] = TensorIR(
        name="gate_nchw",
        dtype="FLOAT32",
        shape=[1, 8, 1, 1],
        shape_signature=[1, 8, 1, 1],
    )
    model_ir.tensors["sig_nchw"] = TensorIR(
        name="sig_nchw",
        dtype="FLOAT32",
        shape=[1, 8, 1, 1],
        shape_signature=[1, 8, 1, 1],
    )
    model_ir.tensors["mul_nchw"] = TensorIR(
        name="mul_nchw",
        dtype="FLOAT32",
        shape=[1, 8, 6, 6],
        shape_signature=[1, 8, 6, 6],
    )
    model_ir.tensors["add_nchw"] = TensorIR(
        name="add_nchw",
        dtype="FLOAT32",
        shape=[1, 8, 6, 6],
        shape_signature=[1, 8, 6, 6],
    )
    model_ir.tensors["y_nhwc"] = TensorIR(
        name="y_nhwc",
        dtype="FLOAT32",
        shape=[1, 6, 6, 8],
        shape_signature=[1, 6, 6, 8],
    )
    model_ir.tensors["z"] = TensorIR(
        name="z",
        dtype="FLOAT32",
        shape=[1, 36, 8],
        shape_signature=[1, 36, 8],
    )
    model_ir.tensors["data_stats_nchw"] = TensorIR(
        name="data_stats_nchw",
        dtype="FLOAT32",
        shape=[1, 8, 1, 1],
        shape_signature=[1, 8, 1, 1],
    )
    model_ir.tensors["legacy_sum"] = TensorIR(
        name="legacy_sum",
        dtype="FLOAT32",
        shape=[1, 8, 6, 6],
        shape_signature=[1, 8, 6, 6],
    )
    model_ir.tensors["legacy_z"] = TensorIR(
        name="legacy_z",
        dtype="FLOAT32",
        shape=[1, 8, 6, 6],
        shape_signature=[1, 8, 6, 6],
    )
    model_ir.operators = [
        OperatorIR(op_type="TRANSPOSE", inputs=["skip_nhwc", "pre_perm"], outputs=["skip_nchw"]),
        OperatorIR(op_type="TRANSPOSE", inputs=["data_nhwc", "pre_perm"], outputs=["data_nchw"]),
        OperatorIR(op_type="TRANSPOSE", inputs=["gate_nhwc", "pre_perm"], outputs=["gate_nchw"]),
        OperatorIR(op_type="LOGISTIC", inputs=["gate_nchw"], outputs=["sig_nchw"]),
        OperatorIR(op_type="MUL", inputs=["data_nchw", "sig_nchw"], outputs=["mul_nchw"]),
        OperatorIR(
            op_type="MEAN",
            inputs=["data_nchw", "mean_axes"],
            outputs=["data_stats_nchw"],
            options={"keepDims": True},
        ),
        OperatorIR(op_type="ADD", inputs=["mul_nchw", "skip_nchw"], outputs=["add_nchw"]),
        OperatorIR(op_type="TRANSPOSE", inputs=["add_nchw", "post_perm"], outputs=["y_nhwc"]),
        OperatorIR(op_type="ADD", inputs=["add_nchw", "legacy_bias"], outputs=["legacy_sum"]),
        OperatorIR(op_type="RELU", inputs=["legacy_sum"], outputs=["legacy_z"]),
        OperatorIR(op_type="RESHAPE", inputs=["y_nhwc", "reshape_shape"], outputs=["z"]),
    ]

    stats = _optimize_transpose_logistic_muladd_prepost_nhwc_chains(model_ir)
    assert stats["optimized_transpose_logistic_muladd_prepost_nhwc_chains"] == 1

    logistic_op = next(op for op in model_ir.operators if str(op.op_type) == "LOGISTIC")
    assert [str(v) for v in list(logistic_op.inputs)] == ["gate_nhwc"]

    mul_op = next(op for op in model_ir.operators if str(op.op_type) == "MUL")
    assert set(str(v) for v in list(mul_op.inputs)) == {"data_nhwc", "sig_nchw"}

    add_op = next(op for op in model_ir.operators if str(op.op_type) == "ADD" and list(op.outputs) == ["y_nhwc"])
    assert set(str(v) for v in list(add_op.inputs)) == {"mul_nchw", "skip_nhwc"}

    # Legacy NCHW path is preserved via one retained adapter transpose.
    adapter_transpose = next(
        op
        for op in model_ir.operators
        if str(op.op_type) == "TRANSPOSE" and list(op.outputs) == ["add_nchw"]
    )
    assert [str(v) for v in list(adapter_transpose.inputs)] == ["y_nhwc", "post_perm"]
    np.testing.assert_array_equal(
        np.asarray(model_ir.tensors["post_perm"].data, dtype=np.int32),
        np.asarray([0, 3, 1, 2], dtype=np.int32),
    )

    # data_nchw transpose remains because MEAN still consumes NCHW.
    assert any(str(op.op_type) == "TRANSPOSE" and list(op.outputs) == ["data_nchw"] for op in model_ir.operators)
    assert not any(str(op.op_type) == "TRANSPOSE" and list(op.outputs) == ["skip_nchw"] for op in model_ir.operators)
    assert not any(str(op.op_type) == "TRANSPOSE" and list(op.outputs) == ["gate_nchw"] for op in model_ir.operators)

    reshape_op = next(op for op in model_ir.operators if str(op.op_type) == "RESHAPE")
    assert [str(v) for v in list(reshape_op.inputs)] == ["y_nhwc", "reshape_shape"]


def test_flatbuffer_direct_transpose_weighted_add_swish_prepost_nhwc_chain_optimized() -> None:
    model_ir = ModelIR(name="transpose_weighted_add_swish_prepost_nhwc_chain_opt_test")
    model_ir.inputs = ["a_nhwc", "b_nhwc"]
    model_ir.outputs = ["z"]

    model_ir.tensors["a_nhwc"] = TensorIR(
        name="a_nhwc",
        dtype="FLOAT32",
        shape=[1, 6, 6, 8],
        shape_signature=[1, 6, 6, 8],
    )
    model_ir.tensors["b_nhwc"] = TensorIR(
        name="b_nhwc",
        dtype="FLOAT32",
        shape=[1, 6, 6, 8],
        shape_signature=[1, 6, 6, 8],
    )
    model_ir.tensors["pre_perm"] = TensorIR(
        name="pre_perm",
        dtype="INT32",
        shape=[4],
        shape_signature=[4],
        data=np.asarray([0, 3, 1, 2], dtype=np.int32),
        is_variable=False,
    )
    model_ir.tensors["post_perm"] = TensorIR(
        name="post_perm",
        dtype="INT32",
        shape=[4],
        shape_signature=[4],
        data=np.asarray([0, 2, 3, 1], dtype=np.int32),
        is_variable=False,
    )
    # Scalar-like runtime weights (e.g. Gather outputs in BiFPN).
    model_ir.tensors["w0"] = TensorIR(
        name="w0",
        dtype="FLOAT32",
        shape=[1],
        shape_signature=[1],
    )
    model_ir.tensors["w1"] = TensorIR(
        name="w1",
        dtype="FLOAT32",
        shape=[1],
        shape_signature=[1],
    )
    model_ir.tensors["a_nchw"] = TensorIR(
        name="a_nchw",
        dtype="FLOAT32",
        shape=[1, 8, 6, 6],
        shape_signature=[1, 8, 6, 6],
    )
    model_ir.tensors["b_nchw"] = TensorIR(
        name="b_nchw",
        dtype="FLOAT32",
        shape=[1, 8, 6, 6],
        shape_signature=[1, 8, 6, 6],
    )
    model_ir.tensors["wa_nchw"] = TensorIR(
        name="wa_nchw",
        dtype="FLOAT32",
        shape=[1, 8, 6, 6],
        shape_signature=[1, 8, 6, 6],
    )
    model_ir.tensors["wb_nchw"] = TensorIR(
        name="wb_nchw",
        dtype="FLOAT32",
        shape=[1, 8, 6, 6],
        shape_signature=[1, 8, 6, 6],
    )
    model_ir.tensors["sum_nchw"] = TensorIR(
        name="sum_nchw",
        dtype="FLOAT32",
        shape=[1, 8, 6, 6],
        shape_signature=[1, 8, 6, 6],
    )
    model_ir.tensors["sig_nchw"] = TensorIR(
        name="sig_nchw",
        dtype="FLOAT32",
        shape=[1, 8, 6, 6],
        shape_signature=[1, 8, 6, 6],
    )
    model_ir.tensors["sw_nchw"] = TensorIR(
        name="sw_nchw",
        dtype="FLOAT32",
        shape=[1, 8, 6, 6],
        shape_signature=[1, 8, 6, 6],
    )
    model_ir.tensors["y_nhwc"] = TensorIR(
        name="y_nhwc",
        dtype="FLOAT32",
        shape=[1, 6, 6, 8],
        shape_signature=[1, 6, 6, 8],
    )
    model_ir.tensors["reshape_shape"] = TensorIR(
        name="reshape_shape",
        dtype="INT64",
        shape=[3],
        shape_signature=[3],
        data=np.asarray([1, 36, 8], dtype=np.int64),
        is_variable=False,
    )
    model_ir.tensors["z"] = TensorIR(
        name="z",
        dtype="FLOAT32",
        shape=[1, 36, 8],
        shape_signature=[1, 36, 8],
    )

    model_ir.operators = [
        OperatorIR(op_type="TRANSPOSE", inputs=["a_nhwc", "pre_perm"], outputs=["a_nchw"]),
        OperatorIR(op_type="TRANSPOSE", inputs=["b_nhwc", "pre_perm"], outputs=["b_nchw"]),
        OperatorIR(op_type="MUL", inputs=["a_nchw", "w0"], outputs=["wa_nchw"]),
        OperatorIR(op_type="MUL", inputs=["b_nchw", "w1"], outputs=["wb_nchw"]),
        OperatorIR(op_type="ADD", inputs=["wa_nchw", "wb_nchw"], outputs=["sum_nchw"]),
        OperatorIR(op_type="LOGISTIC", inputs=["sum_nchw"], outputs=["sig_nchw"]),
        OperatorIR(op_type="MUL", inputs=["sum_nchw", "sig_nchw"], outputs=["sw_nchw"]),
        OperatorIR(op_type="TRANSPOSE", inputs=["sw_nchw", "post_perm"], outputs=["y_nhwc"]),
        OperatorIR(op_type="RESHAPE", inputs=["y_nhwc", "reshape_shape"], outputs=["z"]),
    ]

    stats = _optimize_transpose_weighted_add_swish_prepost_nhwc_chains(model_ir)
    assert stats["optimized_transpose_weighted_add_swish_prepost_nhwc_chains"] == 1

    op_types = [str(op.op_type) for op in model_ir.operators]
    assert op_types.count("TRANSPOSE") == 0

    mul_ops = [op for op in model_ir.operators if str(op.op_type) == "MUL" and list(op.outputs) in (["wa_nchw"], ["wb_nchw"])]
    assert len(mul_ops) == 2
    assert set(str(v) for v in list(mul_ops[0].inputs)) & {"a_nhwc", "b_nhwc"}
    assert set(str(v) for v in list(mul_ops[1].inputs)) & {"a_nhwc", "b_nhwc"}

    swish_mul_op = next(op for op in model_ir.operators if str(op.op_type) == "MUL" and list(op.outputs) == ["y_nhwc"])
    assert set(str(v) for v in list(swish_mul_op.inputs)) == {"sum_nchw", "sig_nchw"}

    reshape_op = next(op for op in model_ir.operators if str(op.op_type) == "RESHAPE")
    assert [str(v) for v in list(reshape_op.inputs)] == ["y_nhwc", "reshape_shape"]


def test_flatbuffer_direct_transpose_nested_weighted_add_swish_prepost_nhwc_chain_optimized() -> None:
    model_ir = ModelIR(name="transpose_nested_weighted_add_swish_prepost_nhwc_chain_opt_test")
    model_ir.inputs = ["a_nhwc", "b_nhwc", "c_nhwc"]
    model_ir.outputs = ["z"]

    for input_name in ("a_nhwc", "b_nhwc", "c_nhwc"):
        model_ir.tensors[input_name] = TensorIR(
            name=input_name,
            dtype="FLOAT32",
            shape=[1, 6, 6, 8],
            shape_signature=[1, 6, 6, 8],
        )
    model_ir.tensors["pre_perm"] = TensorIR(
        name="pre_perm",
        dtype="INT32",
        shape=[4],
        shape_signature=[4],
        data=np.asarray([0, 3, 1, 2], dtype=np.int32),
        is_variable=False,
    )
    model_ir.tensors["post_perm"] = TensorIR(
        name="post_perm",
        dtype="INT32",
        shape=[4],
        shape_signature=[4],
        data=np.asarray([0, 2, 3, 1], dtype=np.int32),
        is_variable=False,
    )
    for weight_name in ("w0", "w1", "w2"):
        model_ir.tensors[weight_name] = TensorIR(
            name=weight_name,
            dtype="FLOAT32",
            shape=[1],
            shape_signature=[1],
        )
    for tensor_name in (
        "a_nchw",
        "b_nchw",
        "c_nchw",
        "wa_nchw",
        "wb_nchw",
        "wc_nchw",
        "sum1_nchw",
        "sum2_nchw",
        "sig_nchw",
        "sw_nchw",
    ):
        model_ir.tensors[tensor_name] = TensorIR(
            name=tensor_name,
            dtype="FLOAT32",
            shape=[1, 8, 6, 6],
            shape_signature=[1, 8, 6, 6],
        )
    model_ir.tensors["y_nhwc"] = TensorIR(
        name="y_nhwc",
        dtype="FLOAT32",
        shape=[1, 6, 6, 8],
        shape_signature=[1, 6, 6, 8],
    )
    model_ir.tensors["reshape_shape"] = TensorIR(
        name="reshape_shape",
        dtype="INT64",
        shape=[3],
        shape_signature=[3],
        data=np.asarray([1, 36, 8], dtype=np.int64),
        is_variable=False,
    )
    model_ir.tensors["z"] = TensorIR(
        name="z",
        dtype="FLOAT32",
        shape=[1, 36, 8],
        shape_signature=[1, 36, 8],
    )

    model_ir.operators = [
        OperatorIR(op_type="TRANSPOSE", inputs=["a_nhwc", "pre_perm"], outputs=["a_nchw"]),
        OperatorIR(op_type="TRANSPOSE", inputs=["b_nhwc", "pre_perm"], outputs=["b_nchw"]),
        OperatorIR(op_type="TRANSPOSE", inputs=["c_nhwc", "pre_perm"], outputs=["c_nchw"]),
        OperatorIR(op_type="MUL", inputs=["a_nchw", "w0"], outputs=["wa_nchw"]),
        OperatorIR(op_type="MUL", inputs=["b_nchw", "w1"], outputs=["wb_nchw"]),
        OperatorIR(op_type="MUL", inputs=["c_nchw", "w2"], outputs=["wc_nchw"]),
        OperatorIR(op_type="ADD", inputs=["wa_nchw", "wb_nchw"], outputs=["sum1_nchw"]),
        OperatorIR(op_type="ADD", inputs=["sum1_nchw", "wc_nchw"], outputs=["sum2_nchw"]),
        OperatorIR(op_type="LOGISTIC", inputs=["sum2_nchw"], outputs=["sig_nchw"]),
        OperatorIR(op_type="MUL", inputs=["sum2_nchw", "sig_nchw"], outputs=["sw_nchw"]),
        OperatorIR(op_type="TRANSPOSE", inputs=["sw_nchw", "post_perm"], outputs=["y_nhwc"]),
        OperatorIR(op_type="RESHAPE", inputs=["y_nhwc", "reshape_shape"], outputs=["z"]),
    ]

    stats = _optimize_transpose_nested_weighted_add_swish_prepost_nhwc_chains(model_ir)
    assert stats["optimized_transpose_nested_weighted_add_swish_prepost_nhwc_chains"] == 1
    assert [str(op.op_type) for op in model_ir.operators].count("TRANSPOSE") == 0

    for output_name, expected_input in (
        ("wa_nchw", "a_nhwc"),
        ("wb_nchw", "b_nhwc"),
        ("wc_nchw", "c_nhwc"),
    ):
        mul_op = next(op for op in model_ir.operators if str(op.op_type) == "MUL" and list(op.outputs) == [output_name])
        assert expected_input in [str(v) for v in list(mul_op.inputs)]

    swish_mul_op = next(op for op in model_ir.operators if str(op.op_type) == "MUL" and list(op.outputs) == ["y_nhwc"])
    assert set(str(v) for v in list(swish_mul_op.inputs)) == {"sum2_nchw", "sig_nchw"}

    reshape_op = next(op for op in model_ir.operators if str(op.op_type) == "RESHAPE")
    assert [str(v) for v in list(reshape_op.inputs)] == ["y_nhwc", "reshape_shape"]


def test_flatbuffer_direct_transpose_pad_prepost_nhwc_chain_optimized() -> None:
    model_ir = ModelIR(name="transpose_pad_prepost_nhwc_chain_opt_test")
    model_ir.inputs = ["x_nhwc"]
    model_ir.outputs = ["z"]

    model_ir.tensors["x_nhwc"] = TensorIR(
        name="x_nhwc",
        dtype="FLOAT32",
        shape=[1, 6, 8, 4],
        shape_signature=[1, 6, 8, 4],
    )
    model_ir.tensors["pre_perm"] = TensorIR(
        name="pre_perm",
        dtype="INT32",
        shape=[4],
        shape_signature=[4],
        data=np.asarray([0, 3, 1, 2], dtype=np.int32),
        is_variable=False,
    )
    model_ir.tensors["post_perm"] = TensorIR(
        name="post_perm",
        dtype="INT32",
        shape=[4],
        shape_signature=[4],
        data=np.asarray([0, 2, 3, 1], dtype=np.int32),
        is_variable=False,
    )
    model_ir.tensors["pads"] = TensorIR(
        name="pads",
        dtype="INT32",
        shape=[4, 2],
        shape_signature=[4, 2],
        data=np.asarray([[0, 0], [1, 2], [3, 4], [5, 6]], dtype=np.int32),
        is_variable=False,
    )
    model_ir.tensors["x_nchw"] = TensorIR(
        name="x_nchw",
        dtype="FLOAT32",
        shape=[1, 4, 6, 8],
        shape_signature=[1, 4, 6, 8],
    )
    model_ir.tensors["pad_nchw"] = TensorIR(
        name="pad_nchw",
        dtype="FLOAT32",
        shape=[1, 7, 13, 19],
        shape_signature=[1, 7, 13, 19],
    )
    model_ir.tensors["y_nhwc"] = TensorIR(
        name="y_nhwc",
        dtype="FLOAT32",
        shape=[1, 13, 19, 7],
        shape_signature=[1, 13, 19, 7],
    )
    model_ir.tensors["z"] = TensorIR(
        name="z",
        dtype="FLOAT32",
        shape=[1, 13, 19, 7],
        shape_signature=[1, 13, 19, 7],
    )

    model_ir.operators = [
        OperatorIR(op_type="TRANSPOSE", inputs=["x_nhwc", "pre_perm"], outputs=["x_nchw"]),
        OperatorIR(op_type="PAD", inputs=["x_nchw", "pads"], outputs=["pad_nchw"]),
        OperatorIR(op_type="TRANSPOSE", inputs=["pad_nchw", "post_perm"], outputs=["y_nhwc"]),
        OperatorIR(op_type="RELU", inputs=["y_nhwc"], outputs=["z"]),
    ]

    stats = _optimize_transpose_pad_prepost_nhwc_chains(model_ir)
    assert stats["optimized_transpose_pad_prepost_nhwc_chains"] == 1

    op_types = [str(op.op_type) for op in model_ir.operators]
    assert op_types.count("TRANSPOSE") == 0

    pad_op = next(op for op in model_ir.operators if str(op.op_type) == "PAD")
    assert [str(v) for v in list(pad_op.inputs)] == ["x_nhwc", "pads"]
    assert [str(v) for v in list(pad_op.outputs)] == ["y_nhwc"]

    np.testing.assert_array_equal(
        np.asarray(model_ir.tensors["pads"].data, dtype=np.int32),
        np.asarray([[0, 0], [3, 4], [5, 6], [1, 2]], dtype=np.int32),
    )

    relu_op = next(op for op in model_ir.operators if str(op.op_type) == "RELU")
    assert [str(v) for v in list(relu_op.inputs)] == ["y_nhwc"]


def test_flatbuffer_direct_transpose_3d_leaky_logistic_muladd_ndhwc_chain_optimized() -> None:
    model_ir = ModelIR(name="transpose_3d_leaky_logistic_muladd_ndhwc_chain_opt_test")
    model_ir.inputs = ["base_nhwc", "skip_ndhwc", "gate_ndhwc"]
    model_ir.outputs = ["y0", "y1"]

    model_ir.tensors["base_nhwc"] = TensorIR(
        name="base_nhwc",
        dtype="FLOAT32",
        shape=[1, 6, 10, 8],
        shape_signature=[1, 6, 10, 8],
    )
    model_ir.tensors["skip_ndhwc"] = TensorIR(
        name="skip_ndhwc",
        dtype="FLOAT32",
        shape=[1, 6, 6, 10, 8],
        shape_signature=[1, 6, 6, 10, 8],
    )
    model_ir.tensors["gate_ndhwc"] = TensorIR(
        name="gate_ndhwc",
        dtype="FLOAT32",
        shape=[1, 6, 6, 10, 8],
        shape_signature=[1, 6, 6, 10, 8],
    )
    model_ir.tensors["perm4_pre"] = TensorIR(
        name="perm4_pre",
        dtype="INT32",
        shape=[4],
        shape_signature=[4],
        data=np.asarray([0, 3, 1, 2], dtype=np.int32),
        is_variable=False,
    )
    model_ir.tensors["perm5_pre"] = TensorIR(
        name="perm5_pre",
        dtype="INT32",
        shape=[5],
        shape_signature=[5],
        data=np.asarray([0, 4, 1, 2, 3], dtype=np.int32),
        is_variable=False,
    )
    model_ir.tensors["perm5_post"] = TensorIR(
        name="perm5_post",
        dtype="INT32",
        shape=[5],
        shape_signature=[5],
        data=np.asarray([0, 2, 3, 4, 1], dtype=np.int32),
        is_variable=False,
    )
    model_ir.tensors["base_reshape_shape"] = TensorIR(
        name="base_reshape_shape",
        dtype="INT64",
        shape=[5],
        shape_signature=[5],
        data=np.asarray([1, 8, 1, 6, 10], dtype=np.int64),
        is_variable=False,
    )
    model_ir.tensors["base_nchw"] = TensorIR(
        name="base_nchw",
        dtype="FLOAT32",
        shape=[1, 8, 6, 10],
        shape_signature=[1, 8, 6, 10],
    )
    model_ir.tensors["base_ncdhw"] = TensorIR(
        name="base_ncdhw",
        dtype="FLOAT32",
        shape=[1, 8, 1, 6, 10],
        shape_signature=[1, 8, 1, 6, 10],
    )
    model_ir.tensors["skip_ncdhw"] = TensorIR(
        name="skip_ncdhw",
        dtype="FLOAT32",
        shape=[1, 8, 6, 6, 10],
        shape_signature=[1, 8, 6, 6, 10],
    )
    model_ir.tensors["skip_leaky_ncdhw"] = TensorIR(
        name="skip_leaky_ncdhw",
        dtype="FLOAT32",
        shape=[1, 8, 6, 6, 10],
        shape_signature=[1, 8, 6, 6, 10],
    )
    model_ir.tensors["add0_ncdhw"] = TensorIR(
        name="add0_ncdhw",
        dtype="FLOAT32",
        shape=[1, 8, 6, 6, 10],
        shape_signature=[1, 8, 6, 6, 10],
    )
    model_ir.tensors["add0_ndhwc"] = TensorIR(
        name="add0_ndhwc",
        dtype="FLOAT32",
        shape=[1, 6, 6, 10, 8],
        shape_signature=[1, 6, 6, 10, 8],
    )
    model_ir.tensors["gate_ncdhw"] = TensorIR(
        name="gate_ncdhw",
        dtype="FLOAT32",
        shape=[1, 8, 6, 6, 10],
        shape_signature=[1, 8, 6, 6, 10],
    )
    model_ir.tensors["gate_sig_ncdhw"] = TensorIR(
        name="gate_sig_ncdhw",
        dtype="FLOAT32",
        shape=[1, 8, 6, 6, 10],
        shape_signature=[1, 8, 6, 6, 10],
    )
    model_ir.tensors["mul1_ncdhw"] = TensorIR(
        name="mul1_ncdhw",
        dtype="FLOAT32",
        shape=[1, 8, 6, 6, 10],
        shape_signature=[1, 8, 6, 6, 10],
    )
    model_ir.tensors["add1_ncdhw"] = TensorIR(
        name="add1_ncdhw",
        dtype="FLOAT32",
        shape=[1, 8, 6, 6, 10],
        shape_signature=[1, 8, 6, 6, 10],
    )
    model_ir.tensors["add1_ndhwc"] = TensorIR(
        name="add1_ndhwc",
        dtype="FLOAT32",
        shape=[1, 6, 6, 10, 8],
        shape_signature=[1, 6, 6, 10, 8],
    )
    model_ir.tensors["y0"] = TensorIR(
        name="y0",
        dtype="FLOAT32",
        shape=[1, 6, 6, 10, 8],
        shape_signature=[1, 6, 6, 10, 8],
    )
    model_ir.tensors["y1"] = TensorIR(
        name="y1",
        dtype="FLOAT32",
        shape=[1, 6, 6, 10, 8],
        shape_signature=[1, 6, 6, 10, 8],
    )

    model_ir.operators = [
        OperatorIR(op_type="TRANSPOSE", inputs=["base_nhwc", "perm4_pre"], outputs=["base_nchw"]),
        OperatorIR(op_type="RESHAPE", inputs=["base_nchw", "base_reshape_shape"], outputs=["base_ncdhw"]),
        OperatorIR(op_type="TRANSPOSE", inputs=["skip_ndhwc", "perm5_pre"], outputs=["skip_ncdhw"]),
        OperatorIR(op_type="LEAKY_RELU", inputs=["skip_ncdhw"], outputs=["skip_leaky_ncdhw"], options={"alpha": 0.1}),
        OperatorIR(op_type="ADD", inputs=["skip_leaky_ncdhw", "base_ncdhw"], outputs=["add0_ncdhw"]),
        OperatorIR(op_type="TRANSPOSE", inputs=["add0_ncdhw", "perm5_post"], outputs=["add0_ndhwc"]),
        OperatorIR(op_type="TRANSPOSE", inputs=["gate_ndhwc", "perm5_pre"], outputs=["gate_ncdhw"]),
        OperatorIR(op_type="LOGISTIC", inputs=["gate_ncdhw"], outputs=["gate_sig_ncdhw"]),
        OperatorIR(op_type="MUL", inputs=["gate_sig_ncdhw", "base_ncdhw"], outputs=["mul1_ncdhw"]),
        OperatorIR(op_type="ADD", inputs=["mul1_ncdhw", "skip_leaky_ncdhw"], outputs=["add1_ncdhw"]),
        OperatorIR(op_type="TRANSPOSE", inputs=["add1_ncdhw", "perm5_post"], outputs=["add1_ndhwc"]),
        OperatorIR(op_type="RELU", inputs=["add0_ndhwc"], outputs=["y0"]),
        OperatorIR(op_type="RELU", inputs=["add1_ndhwc"], outputs=["y1"]),
    ]

    stats = _optimize_transpose_3d_leaky_logistic_muladd_ndhwc_chains(model_ir)
    assert stats["optimized_transpose_3d_leaky_logistic_muladd_ndhwc_chains"] == 1

    assert not any(str(op.op_type) == "TRANSPOSE" for op in model_ir.operators)

    reshape_op = next(op for op in model_ir.operators if str(op.op_type) == "RESHAPE")
    assert [str(v) for v in list(reshape_op.inputs)] == ["base_nhwc", "base_reshape_shape"]
    np.testing.assert_array_equal(
        np.asarray(model_ir.tensors["base_reshape_shape"].data, dtype=np.int64),
        np.asarray([1, 1, 6, 10, 8], dtype=np.int64),
    )

    skip_leaky_op = next(op for op in model_ir.operators if str(op.op_type) == "LEAKY_RELU")
    assert [str(v) for v in list(skip_leaky_op.inputs)] == ["skip_ndhwc"]

    logistic_op = next(op for op in model_ir.operators if str(op.op_type) == "LOGISTIC")
    assert [str(v) for v in list(logistic_op.inputs)] == ["gate_ndhwc"]

    add_ops = [op for op in model_ir.operators if str(op.op_type) == "ADD"]
    assert [str(v) for v in list(add_ops[0].outputs)] == ["add0_ndhwc"]
    assert [str(v) for v in list(add_ops[1].outputs)] == ["add1_ndhwc"]

    assert list(model_ir.tensors["base_ncdhw"].shape) == [1, 1, 6, 10, 8]
    assert list(model_ir.tensors["skip_leaky_ncdhw"].shape) == [1, 6, 6, 10, 8]


def test_flatbuffer_direct_transpose_pre_unary_reshape_transpose_suffix_nhwc_chain_optimized() -> None:
    model_ir = ModelIR(name="transpose_pre_unary_reshape_transpose_suffix_nhwc_chain_opt_test")
    model_ir.inputs = ["x_nhwc"]
    model_ir.outputs = ["z"]
    model_ir.tensors["x_nhwc"] = TensorIR(
        name="x_nhwc",
        dtype="FLOAT32",
        shape=[1, 5, 7, 4],
        shape_signature=[1, 5, 7, 4],
    )
    model_ir.tensors["pre_perm"] = TensorIR(
        name="pre_perm",
        dtype="INT32",
        shape=[4],
        shape_signature=[4],
        data=np.asarray([0, 3, 1, 2], dtype=np.int32),
        is_variable=False,
    )
    model_ir.tensors["post_perm"] = TensorIR(
        name="post_perm",
        dtype="INT32",
        shape=[3],
        shape_signature=[3],
        data=np.asarray([0, 2, 1], dtype=np.int32),
        is_variable=False,
    )
    model_ir.tensors["reshape_shape"] = TensorIR(
        name="reshape_shape",
        dtype="INT64",
        shape=[3],
        shape_signature=[3],
        data=np.asarray([1, 4, 35], dtype=np.int64),
        is_variable=False,
    )
    model_ir.tensors["x_nchw"] = TensorIR(
        name="x_nchw",
        dtype="FLOAT32",
        shape=[1, 4, 5, 7],
        shape_signature=[1, 4, 5, 7],
    )
    model_ir.tensors["u_nchw"] = TensorIR(
        name="u_nchw",
        dtype="FLOAT32",
        shape=[1, 4, 5, 7],
        shape_signature=[1, 4, 5, 7],
    )
    model_ir.tensors["r_nchw"] = TensorIR(
        name="r_nchw",
        dtype="FLOAT32",
        shape=[1, 4, 35],
        shape_signature=[1, 4, 35],
    )
    model_ir.tensors["z"] = TensorIR(
        name="z",
        dtype="FLOAT32",
        shape=[1, 35, 4],
        shape_signature=[1, 35, 4],
    )
    model_ir.operators = [
        OperatorIR(op_type="TRANSPOSE", inputs=["x_nhwc", "pre_perm"], outputs=["x_nchw"]),
        OperatorIR(op_type="LEAKY_RELU", inputs=["x_nchw"], outputs=["u_nchw"], options={"alpha": 0.1}),
        OperatorIR(
            op_type="RESHAPE",
            inputs=["u_nchw", "reshape_shape"],
            outputs=["r_nchw"],
            options={"newShape": [1, 4, 35], "onnxRawNewShape": [1, 4, -1]},
        ),
        OperatorIR(op_type="TRANSPOSE", inputs=["r_nchw", "post_perm"], outputs=["z"]),
    ]

    stats = _optimize_transpose_pre_unary_reshape_transpose_suffix_nhwc_chains(model_ir)
    assert stats["optimized_transpose_pre_unary_reshape_transpose_suffix_nhwc_chains"] == 1

    op_types = [str(op.op_type) for op in model_ir.operators]
    assert op_types.count("TRANSPOSE") == 0

    unary_op = next(op for op in model_ir.operators if str(op.op_type) == "LEAKY_RELU")
    assert [str(v) for v in list(unary_op.inputs)] == ["x_nhwc"]

    reshape_op = next(op for op in model_ir.operators if str(op.op_type) == "RESHAPE")
    assert [str(v) for v in list(reshape_op.inputs)] == ["u_nchw", "reshape_shape"]
    assert [str(v) for v in list(reshape_op.outputs)] == ["z"]
    assert list(np.asarray(model_ir.tensors["reshape_shape"].data).reshape(-1)) == [1, 35, 4]
    assert reshape_op.options.get("newShape") == [1, 35, 4]
    assert reshape_op.options.get("onnxRawNewShape") == [1, -1, 4]

    assert list(model_ir.tensors["u_nchw"].shape) == [1, 5, 7, 4]
    assert list(model_ir.tensors["z"].shape) == [1, 35, 4]


def test_flatbuffer_direct_transpose_reshape_transpose_to_expanddims_nhwc_chain_optimized() -> None:
    model_ir = ModelIR(name="transpose_reshape_transpose_to_expanddims_nhwc_chain_opt_test")
    model_ir.inputs = ["x_nhwc"]
    model_ir.outputs = ["z"]
    model_ir.tensors["x_nhwc"] = TensorIR(
        name="x_nhwc",
        dtype="FLOAT32",
        shape=[1, 80, 80, 85],
        shape_signature=[1, 80, 80, 85],
    )
    model_ir.tensors["pre_perm"] = TensorIR(
        name="pre_perm",
        dtype="INT32",
        shape=[4],
        shape_signature=[4],
        data=np.asarray([0, 3, 1, 2], dtype=np.int32),
        is_variable=False,
    )
    model_ir.tensors["reshape_shape"] = TensorIR(
        name="reshape_shape",
        dtype="INT64",
        shape=[5],
        shape_signature=[5],
        data=np.asarray([1, 1, 85, 80, 80], dtype=np.int64),
        is_variable=False,
    )
    model_ir.tensors["post_perm"] = TensorIR(
        name="post_perm",
        dtype="INT32",
        shape=[5],
        shape_signature=[5],
        data=np.asarray([0, 1, 3, 4, 2], dtype=np.int32),
        is_variable=False,
    )
    model_ir.tensors["x_nchw"] = TensorIR(
        name="x_nchw",
        dtype="FLOAT32",
        shape=[1, 85, 80, 80],
        shape_signature=[1, 85, 80, 80],
    )
    model_ir.tensors["r_n1chw"] = TensorIR(
        name="r_n1chw",
        dtype="FLOAT32",
        shape=[1, 1, 85, 80, 80],
        shape_signature=[1, 1, 85, 80, 80],
    )
    model_ir.tensors["z"] = TensorIR(
        name="z",
        dtype="FLOAT32",
        shape=[1, 1, 80, 80, 85],
        shape_signature=[1, 1, 80, 80, 85],
    )
    model_ir.operators = [
        OperatorIR(op_type="TRANSPOSE", inputs=["x_nhwc", "pre_perm"], outputs=["x_nchw"]),
        OperatorIR(
            op_type="RESHAPE",
            inputs=["x_nchw", "reshape_shape"],
            outputs=["r_n1chw"],
            options={"newShape": [1, 1, 85, 80, 80], "onnxRawNewShape": [1, 1, 85, 80, 80]},
        ),
        OperatorIR(op_type="TRANSPOSE", inputs=["r_n1chw", "post_perm"], outputs=["z"]),
    ]

    stats = _optimize_transpose_reshape_transpose_to_expanddims_nhwc_chains(model_ir)
    assert stats["optimized_transpose_reshape_transpose_to_expanddims_nhwc_chains"] == 1

    op_types = [str(op.op_type) for op in model_ir.operators]
    assert op_types == ["RESHAPE"]
    reshape_op = model_ir.operators[0]
    assert [str(v) for v in list(reshape_op.inputs)] == ["x_nhwc", "reshape_shape"]
    assert [str(v) for v in list(reshape_op.outputs)] == ["z"]
    assert list(np.asarray(model_ir.tensors["reshape_shape"].data).reshape(-1)) == [1, 1, 80, 80, 85]
    assert reshape_op.options.get("newShape") == [1, 1, 80, 80, 85]
    assert reshape_op.options.get("onnxRawNewShape") == [1, 1, 80, 80, 85]


def test_flatbuffer_direct_transpose_reshape_transpose_to_expanddims_nhwc_channel_tail_chain_optimized() -> None:
    model_ir = ModelIR(name="transpose_reshape_transpose_to_expanddims_nhwc_channel_tail_chain_opt_test")
    model_ir.inputs = ["x_nhwc"]
    model_ir.outputs = ["z"]
    model_ir.tensors["x_nhwc"] = TensorIR(
        name="x_nhwc",
        dtype="FLOAT32",
        shape=[1, 20, 20, 88],
        shape_signature=[1, 20, 20, 88],
    )
    model_ir.tensors["pre_perm"] = TensorIR(
        name="pre_perm",
        dtype="INT32",
        shape=[4],
        shape_signature=[4],
        data=np.asarray([0, 3, 1, 2], dtype=np.int32),
        is_variable=False,
    )
    model_ir.tensors["reshape_shape"] = TensorIR(
        name="reshape_shape",
        dtype="INT32",
        shape=[5],
        shape_signature=[5],
        data=np.asarray([1, 88, 20, 20, 1], dtype=np.int32),
        is_variable=False,
    )
    model_ir.tensors["post_perm"] = TensorIR(
        name="post_perm",
        dtype="INT32",
        shape=[5],
        shape_signature=[5],
        data=np.asarray([0, 2, 3, 1, 4], dtype=np.int32),
        is_variable=False,
    )
    model_ir.tensors["x_nchw"] = TensorIR(
        name="x_nchw",
        dtype="FLOAT32",
        shape=[1, 88, 20, 20],
        shape_signature=[1, 88, 20, 20],
    )
    model_ir.tensors["r_nchw1"] = TensorIR(
        name="r_nchw1",
        dtype="FLOAT32",
        shape=[1, 88, 20, 20, 1],
        shape_signature=[1, 88, 20, 20, 1],
    )
    model_ir.tensors["z"] = TensorIR(
        name="z",
        dtype="FLOAT32",
        shape=[1, 20, 20, 88, 1],
        shape_signature=[1, 20, 20, 88, 1],
    )
    model_ir.operators = [
        OperatorIR(op_type="TRANSPOSE", inputs=["x_nhwc", "pre_perm"], outputs=["x_nchw"]),
        OperatorIR(
            op_type="RESHAPE",
            inputs=["x_nchw", "reshape_shape"],
            outputs=["r_nchw1"],
            options={"newShape": [1, 88, 20, 20, 1], "onnxRawNewShape": [1, 88, 20, 20, 1]},
        ),
        OperatorIR(op_type="TRANSPOSE", inputs=["r_nchw1", "post_perm"], outputs=["z"]),
    ]

    stats = _optimize_transpose_reshape_transpose_to_expanddims_nhwc_chains(model_ir)
    assert stats["optimized_transpose_reshape_transpose_to_expanddims_nhwc_chains"] == 1

    op_types = [str(op.op_type) for op in model_ir.operators]
    assert op_types == ["RESHAPE"]
    reshape_op = model_ir.operators[0]
    assert [str(v) for v in list(reshape_op.inputs)] == ["x_nhwc", "reshape_shape"]
    assert [str(v) for v in list(reshape_op.outputs)] == ["z"]
    assert list(np.asarray(model_ir.tensors["reshape_shape"].data).reshape(-1)) == [1, 20, 20, 88, 1]
    assert reshape_op.options.get("newShape") == [1, 20, 20, 88, 1]
    assert reshape_op.options.get("onnxRawNewShape") == [1, 20, 20, 88, 1]


def test_flatbuffer_direct_transpose_mul_add_const_prelu_prepost_terminal_optimized() -> None:
    model_ir = ModelIR(name="transpose_mul_add_const_prelu_prepost_terminal_opt_test")
    model_ir.inputs = ["x_nhwc", "legacy_nchw"]
    model_ir.outputs = ["z", "legacy_cat"]
    model_ir.tensors["x_nhwc"] = TensorIR(
        name="x_nhwc",
        dtype="FLOAT32",
        shape=[1, 8, 8, 16],
        shape_signature=[1, 8, 8, 16],
    )
    model_ir.tensors["legacy_nchw"] = TensorIR(
        name="legacy_nchw",
        dtype="FLOAT32",
        shape=[1, 16, 8, 8],
        shape_signature=[1, 16, 8, 8],
    )
    model_ir.tensors["pre_perm"] = TensorIR(
        name="pre_perm",
        dtype="INT32",
        shape=[4],
        shape_signature=[4],
        data=np.asarray([0, 3, 1, 2], dtype=np.int32),
        is_variable=False,
    )
    model_ir.tensors["post_perm"] = TensorIR(
        name="post_perm",
        dtype="INT32",
        shape=[4],
        shape_signature=[4],
        data=np.asarray([0, 2, 3, 1], dtype=np.int32),
        is_variable=False,
    )
    model_ir.tensors["x_nchw"] = TensorIR(
        name="x_nchw",
        dtype="FLOAT32",
        shape=[1, 16, 8, 8],
        shape_signature=[1, 16, 8, 8],
    )
    model_ir.tensors["mul_const"] = TensorIR(
        name="mul_const",
        dtype="FLOAT32",
        shape=[1, 16, 1, 1],
        shape_signature=[1, 16, 1, 1],
        data=np.ones((1, 16, 1, 1), dtype=np.float32),
        is_variable=False,
    )
    model_ir.tensors["mul_out"] = TensorIR(
        name="mul_out",
        dtype="FLOAT32",
        shape=[1, 16, 8, 8],
        shape_signature=[1, 16, 8, 8],
    )
    model_ir.tensors["add_const"] = TensorIR(
        name="add_const",
        dtype="FLOAT32",
        shape=[1, 16, 1, 1],
        shape_signature=[1, 16, 1, 1],
        data=np.zeros((1, 16, 1, 1), dtype=np.float32),
        is_variable=False,
    )
    model_ir.tensors["add_out"] = TensorIR(
        name="add_out",
        dtype="FLOAT32",
        shape=[1, 16, 8, 8],
        shape_signature=[1, 16, 8, 8],
    )
    model_ir.tensors["prelu_alpha"] = TensorIR(
        name="prelu_alpha",
        dtype="FLOAT32",
        shape=[1, 16, 1, 1],
        shape_signature=[1, 16, 1, 1],
        data=np.full((1, 16, 1, 1), 0.25, dtype=np.float32),
        is_variable=False,
    )
    model_ir.tensors["prelu_out"] = TensorIR(
        name="prelu_out",
        dtype="FLOAT32",
        shape=[1, 16, 8, 8],
        shape_signature=[1, 16, 8, 8],
    )
    model_ir.tensors["y_nhwc"] = TensorIR(
        name="y_nhwc",
        dtype="FLOAT32",
        shape=[1, 8, 8, 16],
        shape_signature=[1, 8, 8, 16],
    )
    model_ir.tensors["z"] = TensorIR(
        name="z",
        dtype="FLOAT32",
        shape=[1, 8, 8, 16],
        shape_signature=[1, 8, 8, 16],
    )
    model_ir.tensors["legacy_cat"] = TensorIR(
        name="legacy_cat",
        dtype="FLOAT32",
        shape=[1, 32, 8, 8],
        shape_signature=[1, 32, 8, 8],
    )
    model_ir.operators = [
        OperatorIR(op_type="TRANSPOSE", inputs=["x_nhwc", "pre_perm"], outputs=["x_nchw"]),
        OperatorIR(
            op_type="MUL",
            inputs=["x_nchw", "mul_const"],
            outputs=["mul_out"],
            options={"fusedActivationFunction": "NONE"},
        ),
        OperatorIR(
            op_type="ADD",
            inputs=["mul_out", "add_const"],
            outputs=["add_out"],
            options={"fusedActivationFunction": "NONE"},
        ),
        OperatorIR(op_type="PRELU", inputs=["add_out", "prelu_alpha"], outputs=["prelu_out"]),
        OperatorIR(op_type="TRANSPOSE", inputs=["prelu_out", "post_perm"], outputs=["y_nhwc"]),
        OperatorIR(op_type="RELU", inputs=["y_nhwc"], outputs=["z"]),
        OperatorIR(
            op_type="CONCATENATION",
            inputs=["legacy_nchw", "prelu_out"],
            outputs=["legacy_cat"],
            options={"axis": 1, "fusedActivationFunction": "NONE"},
        ),
    ]

    stats = _optimize_transpose_mul_add_const_prelu_prepost_nhwc_terminal_chains(model_ir)
    assert stats["optimized_transpose_mul_add_const_prelu_prepost_nhwc_terminal_chains"] == 1
    assert list(model_ir.tensors["mul_const"].shape) == [1, 1, 1, 16]
    assert list(model_ir.tensors["add_const"].shape) == [1, 1, 1, 16]
    assert list(model_ir.tensors["prelu_alpha"].shape) == [1, 1, 1, 16]
    assert any(
        str(op.op_type) == "TRANSPOSE" and list(op.outputs) == ["prelu_out"]
        for op in model_ir.operators
    )


def test_flatbuffer_direct_singleton_spatial_transpose_before_flatten_reshape_removed() -> None:
    model_ir = ModelIR(name="singleton_spatial_transpose_flatten_test")
    model_ir.inputs = ["x_nhwc"]
    model_ir.outputs = ["z"]
    model_ir.tensors["x_nhwc"] = TensorIR(
        name="x_nhwc",
        dtype="FLOAT32",
        shape=[1, 1, 1, 48],
        shape_signature=[1, 1, 1, 48],
    )
    model_ir.tensors["pre_perm"] = TensorIR(
        name="pre_perm",
        dtype="INT32",
        shape=[4],
        shape_signature=[4],
        data=np.asarray([0, 3, 1, 2], dtype=np.int32),
        is_variable=False,
    )
    model_ir.tensors["x_nchw"] = TensorIR(
        name="x_nchw",
        dtype="FLOAT32",
        shape=[1, 48, 1, 1],
        shape_signature=[1, 48, 1, 1],
    )
    model_ir.tensors["flat_shape"] = TensorIR(
        name="flat_shape",
        dtype="INT32",
        shape=[2],
        shape_signature=[2],
        data=np.asarray([1, 48], dtype=np.int32),
        is_variable=False,
    )
    model_ir.tensors["y"] = TensorIR(
        name="y",
        dtype="FLOAT32",
        shape=[1, 48],
        shape_signature=[1, 48],
    )
    model_ir.tensors["z"] = TensorIR(
        name="z",
        dtype="FLOAT32",
        shape=[1, 48],
        shape_signature=[1, 48],
    )
    model_ir.operators = [
        OperatorIR(op_type="TRANSPOSE", inputs=["x_nhwc", "pre_perm"], outputs=["x_nchw"]),
        OperatorIR(
            op_type="RESHAPE",
            inputs=["x_nchw", "flat_shape"],
            outputs=["y"],
            options={"newShape": [1, 48]},
        ),
        OperatorIR(op_type="RELU", inputs=["y"], outputs=["z"]),
    ]

    stats = _optimize_singleton_spatial_nhwc_transpose_reshape_flatten(model_ir)
    assert stats["optimized_singleton_spatial_nhwc_transpose_reshape_flatten"] == 1
    assert not any(str(op.op_type) == "TRANSPOSE" for op in model_ir.operators)
    reshape_op = next(op for op in model_ir.operators if str(op.op_type) == "RESHAPE")
    assert list(reshape_op.inputs) == ["x_nhwc", "flat_shape"]


def test_flatbuffer_direct_singleton_spatial_transpose_identity_reshape_flatten_removed() -> None:
    model_ir = ModelIR(name="singleton_spatial_transpose_identity_reshape_flatten_test")
    model_ir.inputs = ["x_nhwc"]
    model_ir.outputs = ["z"]
    model_ir.tensors["x_nhwc"] = TensorIR(
        name="x_nhwc",
        dtype="FLOAT32",
        shape=[1, 1, 1, 48],
        shape_signature=[1, 1, 1, 48],
    )
    model_ir.tensors["pre_perm"] = TensorIR(
        name="pre_perm",
        dtype="INT32",
        shape=[4],
        shape_signature=[4],
        data=np.asarray([0, 3, 1, 2], dtype=np.int32),
        is_variable=False,
    )
    model_ir.tensors["x_nchw"] = TensorIR(
        name="x_nchw",
        dtype="FLOAT32",
        shape=[1, 48, 1, 1],
        shape_signature=[1, 48, 1, 1],
    )
    model_ir.tensors["keep_shape"] = TensorIR(
        name="keep_shape",
        dtype="INT32",
        shape=[4],
        shape_signature=[4],
        data=np.asarray([1, 48, 1, 1], dtype=np.int32),
        is_variable=False,
    )
    model_ir.tensors["x_keep"] = TensorIR(
        name="x_keep",
        dtype="FLOAT32",
        shape=[1, 48, 1, 1],
        shape_signature=[1, 48, 1, 1],
    )
    model_ir.tensors["flat_shape"] = TensorIR(
        name="flat_shape",
        dtype="INT32",
        shape=[2],
        shape_signature=[2],
        data=np.asarray([1, 48], dtype=np.int32),
        is_variable=False,
    )
    model_ir.tensors["y"] = TensorIR(
        name="y",
        dtype="FLOAT32",
        shape=[1, 48],
        shape_signature=[1, 48],
    )
    model_ir.tensors["z"] = TensorIR(
        name="z",
        dtype="FLOAT32",
        shape=[1, 48],
        shape_signature=[1, 48],
    )
    model_ir.operators = [
        OperatorIR(op_type="TRANSPOSE", inputs=["x_nhwc", "pre_perm"], outputs=["x_nchw"]),
        OperatorIR(
            op_type="RESHAPE",
            inputs=["x_nchw", "keep_shape"],
            outputs=["x_keep"],
            options={"newShape": [1, 48, 1, 1]},
        ),
        OperatorIR(
            op_type="RESHAPE",
            inputs=["x_keep", "flat_shape"],
            outputs=["y"],
            options={"newShape": [1, 48]},
        ),
        OperatorIR(op_type="RELU", inputs=["y"], outputs=["z"]),
    ]

    stats = _optimize_singleton_spatial_nhwc_transpose_reshape_flatten(model_ir)
    assert stats["optimized_singleton_spatial_nhwc_transpose_reshape_flatten"] == 1
    assert not any(str(op.op_type) == "TRANSPOSE" for op in model_ir.operators)
    reshape_ops = [op for op in model_ir.operators if str(op.op_type) == "RESHAPE"]
    assert len(reshape_ops) == 1
    assert list(reshape_ops[0].inputs) == ["x_nhwc", "flat_shape"]


def test_flatbuffer_direct_sinet_concat_resize_affine_transpose_chain_optimized() -> None:
    model_ir = ModelIR(name="sinet_concat_resize_affine_transpose_opt_test")
    model_ir.inputs = ["pre_nhwc", "b0_nhwc", "rz_nhwc"]
    model_ir.outputs = ["z"]
    model_ir.tensors["pre_nhwc"] = TensorIR(
        name="pre_nhwc",
        dtype="FLOAT32",
        shape=[1, 8, 8, 32],
        shape_signature=[1, 8, 8, 32],
    )
    model_ir.tensors["b0_nhwc"] = TensorIR(
        name="b0_nhwc",
        dtype="FLOAT32",
        shape=[1, 8, 8, 16],
        shape_signature=[1, 8, 8, 16],
    )
    model_ir.tensors["rz_nhwc"] = TensorIR(
        name="rz_nhwc",
        dtype="FLOAT32",
        shape=[1, 8, 8, 16],
        shape_signature=[1, 8, 8, 16],
    )
    model_ir.tensors["perm_nhwc_to_nchw"] = TensorIR(
        name="perm_nhwc_to_nchw",
        dtype="INT32",
        shape=[4],
        shape_signature=[4],
        data=np.asarray([0, 3, 1, 2], dtype=np.int32),
        is_variable=False,
    )
    model_ir.tensors["perm_nchw_to_nhwc"] = TensorIR(
        name="perm_nchw_to_nhwc",
        dtype="INT32",
        shape=[4],
        shape_signature=[4],
        data=np.asarray([0, 2, 3, 1], dtype=np.int32),
        is_variable=False,
    )
    for name, shape in [
        ("pre_nchw", [1, 32, 8, 8]),
        ("b0_nchw", [1, 16, 8, 8]),
        ("rz_nchw", [1, 16, 8, 8]),
        ("b1_mul", [1, 16, 8, 8]),
        ("b1_add", [1, 16, 8, 8]),
        ("concat_nchw", [1, 32, 8, 8]),
        ("add0_out", [1, 32, 8, 8]),
        ("mul2_out", [1, 32, 8, 8]),
        ("add2_out", [1, 32, 8, 8]),
        ("prelu_out", [1, 32, 8, 8]),
        ("y_nhwc", [1, 8, 8, 32]),
        ("z", [1, 8, 8, 32]),
    ]:
        model_ir.tensors[name] = TensorIR(
            name=name,
            dtype="FLOAT32",
            shape=shape,
            shape_signature=shape,
        )
    model_ir.tensors["mul_b_const"] = TensorIR(
        name="mul_b_const",
        dtype="FLOAT32",
        shape=[1, 16, 1, 1],
        shape_signature=[1, 16, 1, 1],
        data=np.ones((1, 16, 1, 1), dtype=np.float32),
        is_variable=False,
    )
    model_ir.tensors["add_b_const"] = TensorIR(
        name="add_b_const",
        dtype="FLOAT32",
        shape=[1, 16, 1, 1],
        shape_signature=[1, 16, 1, 1],
        data=np.zeros((1, 16, 1, 1), dtype=np.float32),
        is_variable=False,
    )
    model_ir.tensors["mul2_const"] = TensorIR(
        name="mul2_const",
        dtype="FLOAT32",
        shape=[1, 32, 1, 1],
        shape_signature=[1, 32, 1, 1],
        data=np.ones((1, 32, 1, 1), dtype=np.float32),
        is_variable=False,
    )
    model_ir.tensors["add2_const"] = TensorIR(
        name="add2_const",
        dtype="FLOAT32",
        shape=[1, 32, 1, 1],
        shape_signature=[1, 32, 1, 1],
        data=np.zeros((1, 32, 1, 1), dtype=np.float32),
        is_variable=False,
    )
    model_ir.tensors["prelu_alpha"] = TensorIR(
        name="prelu_alpha",
        dtype="FLOAT32",
        shape=[1, 32, 1, 1],
        shape_signature=[1, 32, 1, 1],
        data=np.full((1, 32, 1, 1), 0.25, dtype=np.float32),
        is_variable=False,
    )
    model_ir.operators = [
        OperatorIR(op_type="TRANSPOSE", inputs=["pre_nhwc", "perm_nhwc_to_nchw"], outputs=["pre_nchw"]),
        OperatorIR(op_type="TRANSPOSE", inputs=["b0_nhwc", "perm_nhwc_to_nchw"], outputs=["b0_nchw"]),
        OperatorIR(op_type="TRANSPOSE", inputs=["rz_nhwc", "perm_nhwc_to_nchw"], outputs=["rz_nchw"]),
        OperatorIR(
            op_type="MUL",
            inputs=["rz_nchw", "mul_b_const"],
            outputs=["b1_mul"],
            options={"fusedActivationFunction": "NONE"},
        ),
        OperatorIR(
            op_type="ADD",
            inputs=["b1_mul", "add_b_const"],
            outputs=["b1_add"],
            options={"fusedActivationFunction": "NONE"},
        ),
        OperatorIR(
            op_type="CONCATENATION",
            inputs=["b0_nchw", "b1_add"],
            outputs=["concat_nchw"],
            options={"axis": 1, "fusedActivationFunction": "NONE"},
        ),
        OperatorIR(
            op_type="ADD",
            inputs=["pre_nchw", "concat_nchw"],
            outputs=["add0_out"],
            options={"fusedActivationFunction": "NONE"},
        ),
        OperatorIR(
            op_type="MUL",
            inputs=["add0_out", "mul2_const"],
            outputs=["mul2_out"],
            options={"fusedActivationFunction": "NONE"},
        ),
        OperatorIR(
            op_type="ADD",
            inputs=["mul2_out", "add2_const"],
            outputs=["add2_out"],
            options={"fusedActivationFunction": "NONE"},
        ),
        OperatorIR(op_type="PRELU", inputs=["add2_out", "prelu_alpha"], outputs=["prelu_out"]),
        OperatorIR(op_type="TRANSPOSE", inputs=["prelu_out", "perm_nchw_to_nhwc"], outputs=["y_nhwc"]),
        OperatorIR(op_type="RELU", inputs=["y_nhwc"], outputs=["z"]),
    ]

    stats = _optimize_sinet_concat_resize_affine_transpose_chains(model_ir)
    assert stats["optimized_sinet_concat_resize_affine_transpose_chains"] == 1
    assert not any(str(op.op_type) == "TRANSPOSE" for op in model_ir.operators)
    concat_op = next(op for op in model_ir.operators if str(op.op_type) == "CONCATENATION")
    assert int(concat_op.options.get("axis", -1)) == 3
    add0_op = next(op for op in model_ir.operators if str(op.op_type) == "ADD" and list(op.outputs) == ["add0_out"])
    assert "pre_nhwc" in list(add0_op.inputs)
    mul_b_op = next(op for op in model_ir.operators if str(op.op_type) == "MUL" and list(op.outputs) == ["b1_mul"])
    assert "rz_nhwc" in list(mul_b_op.inputs)
    assert list(model_ir.tensors["mul_b_const"].shape) == [1, 1, 1, 16]
    assert list(model_ir.tensors["add_b_const"].shape) == [1, 1, 1, 16]
    assert list(model_ir.tensors["mul2_const"].shape) == [1, 1, 1, 32]
    assert list(model_ir.tensors["add2_const"].shape) == [1, 1, 1, 32]
    assert list(model_ir.tensors["prelu_alpha"].shape) == [1, 1, 1, 32]


def test_flatbuffer_direct_sinet_dual_resize_affine_transpose_chain_optimized() -> None:
    model_ir = ModelIR(name="sinet_dual_resize_affine_transpose_opt_test")
    model_ir.inputs = ["pre_nhwc", "r0_nhwc", "r1_nhwc"]
    model_ir.outputs = ["z"]

    for name, shape in [
        ("pre_nhwc", [1, 8, 8, 32]),
        ("r0_nhwc", [1, 8, 8, 16]),
        ("r1_nhwc", [1, 8, 8, 16]),
        ("pre_nchw", [1, 32, 8, 8]),
        ("rz0_nhwc", [1, 8, 8, 16]),
        ("rz1_nhwc", [1, 8, 8, 16]),
        ("b0_nchw", [1, 16, 8, 8]),
        ("b1_nchw", [1, 16, 8, 8]),
        ("b0_mul", [1, 16, 8, 8]),
        ("b0_add", [1, 16, 8, 8]),
        ("b1_mul", [1, 16, 8, 8]),
        ("b1_add", [1, 16, 8, 8]),
        ("concat_nchw", [1, 32, 8, 8]),
        ("add0_out", [1, 32, 8, 8]),
        ("mul2_out", [1, 32, 8, 8]),
        ("add2_out", [1, 32, 8, 8]),
        ("prelu_out", [1, 32, 8, 8]),
        ("y_nhwc", [1, 8, 8, 32]),
        ("z", [1, 8, 8, 32]),
    ]:
        model_ir.tensors[name] = TensorIR(
            name=name,
            dtype="FLOAT32",
            shape=shape,
            shape_signature=shape,
        )

    model_ir.tensors["perm_nhwc_to_nchw"] = TensorIR(
        name="perm_nhwc_to_nchw",
        dtype="INT32",
        shape=[4],
        shape_signature=[4],
        data=np.asarray([0, 3, 1, 2], dtype=np.int32),
        is_variable=False,
    )
    model_ir.tensors["perm_nchw_to_nhwc"] = TensorIR(
        name="perm_nchw_to_nhwc",
        dtype="INT32",
        shape=[4],
        shape_signature=[4],
        data=np.asarray([0, 2, 3, 1], dtype=np.int32),
        is_variable=False,
    )
    model_ir.tensors["resize0_size"] = TensorIR(
        name="resize0_size",
        dtype="INT32",
        shape=[2],
        shape_signature=[2],
        data=np.asarray([8, 8], dtype=np.int32),
        is_variable=False,
    )
    model_ir.tensors["resize1_size"] = TensorIR(
        name="resize1_size",
        dtype="INT32",
        shape=[2],
        shape_signature=[2],
        data=np.asarray([8, 8], dtype=np.int32),
        is_variable=False,
    )

    for const_name, channels, value in [
        ("b0_mul_const", 16, 1.0),
        ("b0_add_const", 16, 0.0),
        ("b1_mul_const", 16, 1.0),
        ("b1_add_const", 16, 0.0),
        ("mul2_const", 32, 1.0),
        ("add2_const", 32, 0.0),
        ("prelu_alpha", 32, 0.25),
    ]:
        arr = np.full((1, channels, 1, 1), value, dtype=np.float32)
        model_ir.tensors[const_name] = TensorIR(
            name=const_name,
            dtype="FLOAT32",
            shape=list(arr.shape),
            shape_signature=list(arr.shape),
            data=np.asarray(arr),
            is_variable=False,
        )

    model_ir.operators = [
        OperatorIR(op_type="TRANSPOSE", inputs=["pre_nhwc", "perm_nhwc_to_nchw"], outputs=["pre_nchw"]),
        OperatorIR(
            op_type="RESIZE_BILINEAR",
            inputs=["r0_nhwc", "resize0_size"],
            outputs=["rz0_nhwc"],
            options={"alignCorners": True, "halfPixelCenters": False},
        ),
        OperatorIR(op_type="TRANSPOSE", inputs=["rz0_nhwc", "perm_nhwc_to_nchw"], outputs=["b0_nchw"]),
        OperatorIR(
            op_type="MUL",
            inputs=["b0_nchw", "b0_mul_const"],
            outputs=["b0_mul"],
            options={"fusedActivationFunction": "NONE"},
        ),
        OperatorIR(
            op_type="ADD",
            inputs=["b0_mul", "b0_add_const"],
            outputs=["b0_add"],
            options={"fusedActivationFunction": "NONE"},
        ),
        OperatorIR(
            op_type="RESIZE_BILINEAR",
            inputs=["r1_nhwc", "resize1_size"],
            outputs=["rz1_nhwc"],
            options={"alignCorners": True, "halfPixelCenters": False},
        ),
        OperatorIR(op_type="TRANSPOSE", inputs=["rz1_nhwc", "perm_nhwc_to_nchw"], outputs=["b1_nchw"]),
        OperatorIR(
            op_type="MUL",
            inputs=["b1_nchw", "b1_mul_const"],
            outputs=["b1_mul"],
            options={"fusedActivationFunction": "NONE"},
        ),
        OperatorIR(
            op_type="ADD",
            inputs=["b1_mul", "b1_add_const"],
            outputs=["b1_add"],
            options={"fusedActivationFunction": "NONE"},
        ),
        OperatorIR(
            op_type="CONCATENATION",
            inputs=["b0_add", "b1_add"],
            outputs=["concat_nchw"],
            options={"axis": 1, "fusedActivationFunction": "NONE"},
        ),
        OperatorIR(
            op_type="ADD",
            inputs=["pre_nchw", "concat_nchw"],
            outputs=["add0_out"],
            options={"fusedActivationFunction": "NONE"},
        ),
        OperatorIR(
            op_type="MUL",
            inputs=["add0_out", "mul2_const"],
            outputs=["mul2_out"],
            options={"fusedActivationFunction": "NONE"},
        ),
        OperatorIR(
            op_type="ADD",
            inputs=["mul2_out", "add2_const"],
            outputs=["add2_out"],
            options={"fusedActivationFunction": "NONE"},
        ),
        OperatorIR(op_type="PRELU", inputs=["add2_out", "prelu_alpha"], outputs=["prelu_out"]),
        OperatorIR(op_type="TRANSPOSE", inputs=["prelu_out", "perm_nchw_to_nhwc"], outputs=["y_nhwc"]),
        OperatorIR(op_type="RELU", inputs=["y_nhwc"], outputs=["z"]),
    ]

    stats = _optimize_sinet_dual_resize_affine_transpose_chains(model_ir)
    assert stats["optimized_sinet_dual_resize_affine_transpose_chains"] == 1
    assert not any(str(op.op_type) == "TRANSPOSE" for op in model_ir.operators)
    concat_op = next(op for op in model_ir.operators if str(op.op_type) == "CONCATENATION")
    assert int(concat_op.options.get("axis", -1)) == 3
    add0_op = next(op for op in model_ir.operators if str(op.op_type) == "ADD" and list(op.outputs) == ["add0_out"])
    assert "pre_nhwc" in list(add0_op.inputs)
    assert list(model_ir.tensors["b0_mul_const"].shape) == [1, 1, 1, 16]
    assert list(model_ir.tensors["b0_add_const"].shape) == [1, 1, 1, 16]
    assert list(model_ir.tensors["b1_mul_const"].shape) == [1, 1, 1, 16]
    assert list(model_ir.tensors["b1_add_const"].shape) == [1, 1, 1, 16]
    assert list(model_ir.tensors["mul2_const"].shape) == [1, 1, 1, 32]
    assert list(model_ir.tensors["add2_const"].shape) == [1, 1, 1, 32]
    assert list(model_ir.tensors["prelu_alpha"].shape) == [1, 1, 1, 32]


def test_flatbuffer_direct_transpose_pre_concat_nhwc_partial_match_does_not_mutate_swish_input() -> None:
    model_ir = ModelIR(name="transpose_pre_concat_nhwc_partial_match_no_mutation_test")
    model_ir.inputs = ["x_nhwc", "residual_nchw"]
    model_ir.outputs = ["z"]
    model_ir.tensors["x_nhwc"] = TensorIR(
        name="x_nhwc",
        dtype="FLOAT32",
        shape=[1, 6, 6, 384],
        shape_signature=[1, 6, 6, 384],
    )
    model_ir.tensors["residual_nchw"] = TensorIR(
        name="residual_nchw",
        dtype="FLOAT32",
        shape=[1, 192, 6, 6],
        shape_signature=[1, 192, 6, 6],
    )
    model_ir.tensors["pre_perm"] = TensorIR(
        name="pre_perm",
        dtype="INT32",
        shape=[4],
        shape_signature=[4],
        data=np.asarray([0, 3, 1, 2], dtype=np.int32),
        is_variable=False,
    )
    model_ir.tensors["post_perm"] = TensorIR(
        name="post_perm",
        dtype="INT32",
        shape=[4],
        shape_signature=[4],
        data=np.asarray([0, 2, 3, 1], dtype=np.int32),
        is_variable=False,
    )
    model_ir.tensors["x_nchw"] = TensorIR(
        name="x_nchw",
        dtype="FLOAT32",
        shape=[1, 384, 6, 6],
        shape_signature=[1, 384, 6, 6],
    )
    model_ir.tensors["sigmoid_out"] = TensorIR(
        name="sigmoid_out",
        dtype="FLOAT32",
        shape=[1, 384, 6, 6],
        shape_signature=[1, 384, 6, 6],
    )
    model_ir.tensors["swish_out"] = TensorIR(
        name="swish_out",
        dtype="FLOAT32",
        shape=[1, 384, 6, 6],
        shape_signature=[1, 384, 6, 6],
    )
    model_ir.tensors["concat_nchw"] = TensorIR(
        name="concat_nchw",
        dtype="FLOAT32",
        shape=[1, 576, 6, 6],
        shape_signature=[1, 576, 6, 6],
    )
    model_ir.tensors["concat_nhwc"] = TensorIR(
        name="concat_nhwc",
        dtype="FLOAT32",
        shape=[1, 6, 6, 576],
        shape_signature=[1, 6, 6, 576],
    )
    model_ir.tensors["z"] = TensorIR(
        name="z",
        dtype="FLOAT32",
        shape=[1, 6, 6, 576],
        shape_signature=[1, 6, 6, 576],
    )
    model_ir.operators = [
        OperatorIR(op_type="TRANSPOSE", inputs=["x_nhwc", "pre_perm"], outputs=["x_nchw"]),
        OperatorIR(op_type="LOGISTIC", inputs=["x_nchw"], outputs=["sigmoid_out"]),
        OperatorIR(
            op_type="MUL",
            inputs=["x_nchw", "sigmoid_out"],
            outputs=["swish_out"],
            options={"fusedActivationFunction": "NONE"},
        ),
        OperatorIR(
            op_type="CONCATENATION",
            inputs=["swish_out", "residual_nchw"],
            outputs=["concat_nchw"],
            options={"axis": 1, "fusedActivationFunction": "NONE"},
        ),
        OperatorIR(op_type="TRANSPOSE", inputs=["concat_nchw", "post_perm"], outputs=["concat_nhwc"]),
        OperatorIR(op_type="RELU", inputs=["concat_nhwc"], outputs=["z"]),
    ]

    stats = _optimize_transpose_pre_concat_nhwc_chains(model_ir)
    assert stats["optimized_transpose_pre_concat_nhwc_chains"] == 0
    assert list(model_ir.tensors["swish_out"].shape) == [1, 384, 6, 6]
    assert list(model_ir.tensors["swish_out"].shape_signature) == [1, 384, 6, 6]
    assert [str(op.op_type) for op in model_ir.operators] == [
        "TRANSPOSE",
        "LOGISTIC",
        "MUL",
        "CONCATENATION",
        "TRANSPOSE",
        "RELU",
    ]


def test_flatbuffer_direct_transpose_pre_concat_nhwc_relu_and_swish_inputs_optimized() -> None:
    model_ir = ModelIR(name="transpose_pre_concat_nhwc_relu_swish_opt_test")
    model_ir.inputs = ["x0_nhwc", "x1_nhwc"]
    model_ir.outputs = ["z"]
    model_ir.tensors["x0_nhwc"] = TensorIR(
        name="x0_nhwc",
        dtype="FLOAT32",
        shape=[1, 6, 6, 192],
        shape_signature=[1, 6, 6, 192],
    )
    model_ir.tensors["x1_nhwc"] = TensorIR(
        name="x1_nhwc",
        dtype="FLOAT32",
        shape=[1, 6, 6, 384],
        shape_signature=[1, 6, 6, 384],
    )
    model_ir.tensors["pre_perm"] = TensorIR(
        name="pre_perm",
        dtype="INT32",
        shape=[4],
        shape_signature=[4],
        data=np.asarray([0, 3, 1, 2], dtype=np.int32),
        is_variable=False,
    )
    model_ir.tensors["post_perm"] = TensorIR(
        name="post_perm",
        dtype="INT32",
        shape=[4],
        shape_signature=[4],
        data=np.asarray([0, 2, 3, 1], dtype=np.int32),
        is_variable=False,
    )
    model_ir.tensors["x0_nchw"] = TensorIR(
        name="x0_nchw",
        dtype="FLOAT32",
        shape=[1, 192, 6, 6],
        shape_signature=[1, 192, 6, 6],
    )
    model_ir.tensors["relu_out"] = TensorIR(
        name="relu_out",
        dtype="FLOAT32",
        shape=[1, 192, 6, 6],
        shape_signature=[1, 192, 6, 6],
    )
    model_ir.tensors["x1_nchw"] = TensorIR(
        name="x1_nchw",
        dtype="FLOAT32",
        shape=[1, 384, 6, 6],
        shape_signature=[1, 384, 6, 6],
    )
    model_ir.tensors["sigmoid_out"] = TensorIR(
        name="sigmoid_out",
        dtype="FLOAT32",
        shape=[1, 384, 6, 6],
        shape_signature=[1, 384, 6, 6],
    )
    model_ir.tensors["swish_out"] = TensorIR(
        name="swish_out",
        dtype="FLOAT32",
        shape=[1, 384, 6, 6],
        shape_signature=[1, 384, 6, 6],
    )
    model_ir.tensors["concat_nchw"] = TensorIR(
        name="concat_nchw",
        dtype="FLOAT32",
        shape=[1, 576, 6, 6],
        shape_signature=[1, 576, 6, 6],
    )
    model_ir.tensors["concat_nhwc"] = TensorIR(
        name="concat_nhwc",
        dtype="FLOAT32",
        shape=[1, 6, 6, 576],
        shape_signature=[1, 6, 6, 576],
    )
    model_ir.tensors["z"] = TensorIR(
        name="z",
        dtype="FLOAT32",
        shape=[1, 6, 6, 576],
        shape_signature=[1, 6, 6, 576],
    )
    model_ir.operators = [
        OperatorIR(op_type="TRANSPOSE", inputs=["x0_nhwc", "pre_perm"], outputs=["x0_nchw"]),
        OperatorIR(op_type="RELU", inputs=["x0_nchw"], outputs=["relu_out"]),
        OperatorIR(op_type="TRANSPOSE", inputs=["x1_nhwc", "pre_perm"], outputs=["x1_nchw"]),
        OperatorIR(op_type="LOGISTIC", inputs=["x1_nchw"], outputs=["sigmoid_out"]),
        OperatorIR(
            op_type="MUL",
            inputs=["x1_nchw", "sigmoid_out"],
            outputs=["swish_out"],
            options={"fusedActivationFunction": "NONE"},
        ),
        OperatorIR(
            op_type="CONCATENATION",
            inputs=["swish_out", "relu_out"],
            outputs=["concat_nchw"],
            options={"axis": 1, "fusedActivationFunction": "NONE"},
        ),
        OperatorIR(op_type="TRANSPOSE", inputs=["concat_nchw", "post_perm"], outputs=["concat_nhwc"]),
        OperatorIR(op_type="RELU", inputs=["concat_nhwc"], outputs=["z"]),
    ]

    stats = _optimize_transpose_pre_concat_nhwc_chains(model_ir)
    assert stats["optimized_transpose_pre_concat_nhwc_chains"] == 1
    op_types = [str(op.op_type) for op in model_ir.operators]
    assert op_types.count("TRANSPOSE") == 0
    concat_op = next(op for op in model_ir.operators if str(op.op_type) == "CONCATENATION")
    assert int(concat_op.options.get("axis", -1)) == 3
    assert list(model_ir.tensors["relu_out"].shape) == [1, 6, 6, 192]
    assert list(model_ir.tensors["swish_out"].shape) == [1, 6, 6, 384]


def test_flatbuffer_direct_transpose_pre_concat_ndhwc_unary_inputs_optimized() -> None:
    model_ir = ModelIR(name="transpose_pre_concat_ndhwc_unary_opt_test")
    model_ir.inputs = ["x0_ndhwc", "x1_ndhwc"]
    model_ir.outputs = ["y"]
    model_ir.tensors["x0_ndhwc"] = TensorIR(
        name="x0_ndhwc",
        dtype="FLOAT32",
        shape=[1, 2, 3, 4, 5],
        shape_signature=[1, 2, 3, 4, 5],
    )
    model_ir.tensors["x1_ndhwc"] = TensorIR(
        name="x1_ndhwc",
        dtype="FLOAT32",
        shape=[1, 2, 3, 4, 6],
        shape_signature=[1, 2, 3, 4, 6],
    )
    model_ir.tensors["pre_perm"] = TensorIR(
        name="pre_perm",
        dtype="INT32",
        shape=[5],
        shape_signature=[5],
        data=np.asarray([0, 4, 1, 2, 3], dtype=np.int32),
        is_variable=False,
    )
    model_ir.tensors["post_perm"] = TensorIR(
        name="post_perm",
        dtype="INT32",
        shape=[5],
        shape_signature=[5],
        data=np.asarray([0, 2, 3, 4, 1], dtype=np.int32),
        is_variable=False,
    )
    model_ir.tensors["x0_ncdhw"] = TensorIR(
        name="x0_ncdhw",
        dtype="FLOAT32",
        shape=[1, 5, 2, 3, 4],
        shape_signature=[1, 5, 2, 3, 4],
    )
    model_ir.tensors["x1_ncdhw"] = TensorIR(
        name="x1_ncdhw",
        dtype="FLOAT32",
        shape=[1, 6, 2, 3, 4],
        shape_signature=[1, 6, 2, 3, 4],
    )
    model_ir.tensors["x1_leaky_ncdhw"] = TensorIR(
        name="x1_leaky_ncdhw",
        dtype="FLOAT32",
        shape=[1, 6, 2, 3, 4],
        shape_signature=[1, 6, 2, 3, 4],
    )
    model_ir.tensors["concat_ncdhw"] = TensorIR(
        name="concat_ncdhw",
        dtype="FLOAT32",
        shape=[1, 11, 2, 3, 4],
        shape_signature=[1, 11, 2, 3, 4],
    )
    model_ir.tensors["concat_ndhwc"] = TensorIR(
        name="concat_ndhwc",
        dtype="FLOAT32",
        shape=[1, 2, 3, 4, 11],
        shape_signature=[1, 2, 3, 4, 11],
    )
    model_ir.tensors["y"] = TensorIR(
        name="y",
        dtype="FLOAT32",
        shape=[1, 2, 3, 4, 11],
        shape_signature=[1, 2, 3, 4, 11],
    )
    model_ir.operators = [
        OperatorIR(op_type="TRANSPOSE", inputs=["x0_ndhwc", "pre_perm"], outputs=["x0_ncdhw"]),
        OperatorIR(op_type="TRANSPOSE", inputs=["x1_ndhwc", "pre_perm"], outputs=["x1_ncdhw"]),
        OperatorIR(op_type="LEAKY_RELU", inputs=["x1_ncdhw"], outputs=["x1_leaky_ncdhw"]),
        OperatorIR(
            op_type="CONCATENATION",
            inputs=["x0_ncdhw", "x1_leaky_ncdhw"],
            outputs=["concat_ncdhw"],
            options={"axis": 1, "fusedActivationFunction": "NONE"},
        ),
        OperatorIR(op_type="TRANSPOSE", inputs=["concat_ncdhw", "post_perm"], outputs=["concat_ndhwc"]),
        OperatorIR(op_type="RELU", inputs=["concat_ndhwc"], outputs=["y"]),
    ]

    stats = _optimize_transpose_pre_concat_ndhwc_chains(model_ir)
    assert stats["optimized_transpose_pre_concat_ndhwc_chains"] == 1

    op_types = [str(op.op_type) for op in model_ir.operators]
    assert op_types.count("TRANSPOSE") == 0
    concat_op = next(op for op in model_ir.operators if str(op.op_type) == "CONCATENATION")
    assert list(concat_op.inputs) == ["x0_ndhwc", "x1_leaky_ncdhw"]
    assert list(concat_op.outputs) == ["concat_ndhwc"]
    assert int(concat_op.options.get("axis", -1)) == 4
    assert list(model_ir.tensors["x1_leaky_ncdhw"].shape) == [1, 2, 3, 4, 6]
    assert list(model_ir.tensors["concat_ndhwc"].shape) == [1, 2, 3, 4, 11]
    relu_op = next(op for op in model_ir.operators if str(op.op_type) == "RELU")
    assert list(relu_op.inputs) == ["concat_ndhwc"]


def test_flatbuffer_direct_transpose_conv3d_leaky_mul_unsqueeze_ndhwc_chain_optimized() -> None:
    model_ir = ModelIR(name="transpose_conv3d_leaky_mul_unsqueeze_ndhwc_opt_test")
    model_ir.inputs = ["sem_ndhwc", "conv_ndhwc"]
    model_ir.outputs = ["y"]

    model_ir.tensors["sem_ndhwc"] = TensorIR(
        name="sem_ndhwc",
        dtype="FLOAT32",
        shape=[1, 48, 80, 8],
        shape_signature=[1, 48, 80, 8],
    )
    model_ir.tensors["conv_ndhwc"] = TensorIR(
        name="conv_ndhwc",
        dtype="FLOAT32",
        shape=[1, 48, 48, 80, 8],
        shape_signature=[1, 48, 48, 80, 8],
    )
    model_ir.tensors["perm_ndhwc_to_ncdhw"] = TensorIR(
        name="perm_ndhwc_to_ncdhw",
        dtype="INT32",
        shape=[5],
        shape_signature=[5],
        data=np.asarray([0, 4, 1, 2, 3], dtype=np.int32),
        is_variable=False,
    )
    model_ir.tensors["perm_nhwc_to_nchw"] = TensorIR(
        name="perm_nhwc_to_nchw",
        dtype="INT32",
        shape=[4],
        shape_signature=[4],
        data=np.asarray([0, 3, 1, 2], dtype=np.int32),
        is_variable=False,
    )
    model_ir.tensors["perm_ncdhw_to_ndhwc"] = TensorIR(
        name="perm_ncdhw_to_ndhwc",
        dtype="INT32",
        shape=[5],
        shape_signature=[5],
        data=np.asarray([0, 2, 3, 4, 1], dtype=np.int32),
        is_variable=False,
    )
    model_ir.tensors["sem_ncdhw"] = TensorIR(
        name="sem_ncdhw",
        dtype="FLOAT32",
        shape=[1, 8, 48, 80],
        shape_signature=[1, 8, 48, 80],
    )
    model_ir.tensors["unsqueeze_shape"] = TensorIR(
        name="unsqueeze_shape",
        dtype="INT32",
        shape=[5],
        shape_signature=[5],
        data=np.asarray([1, 8, 1, 48, 80], dtype=np.int32),
        is_variable=False,
    )
    model_ir.tensors["gate_ncdhw"] = TensorIR(
        name="gate_ncdhw",
        dtype="FLOAT32",
        shape=[1, 8, 1, 48, 80],
        shape_signature=[1, 8, 1, 48, 80],
    )
    model_ir.tensors["conv_ncdhw"] = TensorIR(
        name="conv_ncdhw",
        dtype="FLOAT32",
        shape=[1, 8, 48, 48, 80],
        shape_signature=[1, 8, 48, 48, 80],
    )
    model_ir.tensors["act_ncdhw"] = TensorIR(
        name="act_ncdhw",
        dtype="FLOAT32",
        shape=[1, 8, 48, 48, 80],
        shape_signature=[1, 8, 48, 48, 80],
    )
    model_ir.tensors["mul_ncdhw"] = TensorIR(
        name="mul_ncdhw",
        dtype="FLOAT32",
        shape=[1, 8, 48, 48, 80],
        shape_signature=[1, 8, 48, 48, 80],
    )
    model_ir.tensors["mul_ndhwc"] = TensorIR(
        name="mul_ndhwc",
        dtype="FLOAT32",
        shape=[1, 48, 48, 80, 8],
        shape_signature=[1, 48, 48, 80, 8],
    )
    model_ir.tensors["y"] = TensorIR(
        name="y",
        dtype="FLOAT32",
        shape=[1, 48, 48, 80, 4],
        shape_signature=[1, 48, 48, 80, 4],
    )

    model_ir.operators = [
        OperatorIR(
            op_type="TRANSPOSE",
            inputs=["sem_ndhwc", "perm_nhwc_to_nchw"],
            outputs=["sem_ncdhw"],
        ),
        OperatorIR(
            op_type="RESHAPE",
            inputs=["sem_ncdhw", "unsqueeze_shape"],
            outputs=["gate_ncdhw"],
            options={"newShape": [1, 8, 1, 48, 80]},
        ),
        OperatorIR(
            op_type="TRANSPOSE",
            inputs=["conv_ndhwc", "perm_ndhwc_to_ncdhw"],
            outputs=["conv_ncdhw"],
        ),
        OperatorIR(
            op_type="LEAKY_RELU",
            inputs=["conv_ncdhw"],
            outputs=["act_ncdhw"],
            options={"alpha": 0.1},
        ),
        OperatorIR(
            op_type="MUL",
            inputs=["act_ncdhw", "gate_ncdhw"],
            outputs=["mul_ncdhw"],
        ),
        OperatorIR(
            op_type="TRANSPOSE",
            inputs=["mul_ncdhw", "perm_ncdhw_to_ndhwc"],
            outputs=["mul_ndhwc"],
        ),
        OperatorIR(
            op_type="CONV_3D",
            inputs=["mul_ndhwc", "dummy_w", "dummy_b"],
            outputs=["y"],
            options={},
        ),
    ]

    model_ir.tensors["dummy_w"] = TensorIR(
        name="dummy_w",
        dtype="FLOAT32",
        shape=[4, 1, 1, 1, 8],
        shape_signature=[4, 1, 1, 1, 8],
        data=np.ones((4, 1, 1, 1, 8), dtype=np.float32),
        is_variable=False,
    )
    model_ir.tensors["dummy_b"] = TensorIR(
        name="dummy_b",
        dtype="FLOAT32",
        shape=[4],
        shape_signature=[4],
        data=np.zeros((4,), dtype=np.float32),
        is_variable=False,
    )

    stats = _optimize_transpose_conv3d_leaky_mul_unsqueeze_ndhwc_chains(model_ir)
    assert stats["optimized_transpose_conv3d_leaky_mul_unsqueeze_ndhwc_chains"] == 1

    op_types = [str(op.op_type) for op in model_ir.operators]
    assert op_types.count("TRANSPOSE") == 0

    reshape_op = next(op for op in model_ir.operators if str(op.op_type) == "RESHAPE")
    assert list(reshape_op.inputs) == ["sem_ndhwc", "unsqueeze_shape"]
    assert model_ir.tensors["unsqueeze_shape"].data is not None
    np.testing.assert_array_equal(
        np.asarray(model_ir.tensors["unsqueeze_shape"].data, dtype=np.int32),
        np.asarray([1, 1, 48, 80, 8], dtype=np.int32),
    )
    assert list(model_ir.tensors["gate_ncdhw"].shape) == [1, 1, 48, 80, 8]

    leaky_op = next(op for op in model_ir.operators if str(op.op_type) == "LEAKY_RELU")
    assert list(leaky_op.inputs) == ["conv_ndhwc"]
    assert list(model_ir.tensors["act_ncdhw"].shape) == [1, 48, 48, 80, 8]

    mul_op = next(op for op in model_ir.operators if str(op.op_type) == "MUL")
    assert list(mul_op.outputs) == ["mul_ndhwc"]
    conv3d_op = next(op for op in model_ir.operators if str(op.op_type) == "CONV_3D")
    assert list(conv3d_op.inputs)[0] == "mul_ndhwc"


def test_flatbuffer_direct_transpose_pre_concat_single_post_adapter_supports_relu_inputs() -> None:
    model_ir = ModelIR(name="transpose_pre_concat_single_post_adapter_relu_test")
    model_ir.inputs = ["x0_nhwc", "x1_nhwc"]
    model_ir.outputs = ["z"]
    model_ir.tensors["x0_nhwc"] = TensorIR(
        name="x0_nhwc",
        dtype="FLOAT32",
        shape=[1, 4, 4, 8],
        shape_signature=[1, 4, 4, 8],
    )
    model_ir.tensors["x1_nhwc"] = TensorIR(
        name="x1_nhwc",
        dtype="FLOAT32",
        shape=[1, 4, 4, 8],
        shape_signature=[1, 4, 4, 8],
    )
    model_ir.tensors["pre_perm"] = TensorIR(
        name="pre_perm",
        dtype="INT32",
        shape=[4],
        shape_signature=[4],
        data=np.asarray([0, 3, 1, 2], dtype=np.int32),
        is_variable=False,
    )
    model_ir.tensors["x0_nchw"] = TensorIR(
        name="x0_nchw",
        dtype="FLOAT32",
        shape=[1, 8, 4, 4],
        shape_signature=[1, 8, 4, 4],
    )
    model_ir.tensors["x1_nchw"] = TensorIR(
        name="x1_nchw",
        dtype="FLOAT32",
        shape=[1, 8, 4, 4],
        shape_signature=[1, 8, 4, 4],
    )
    model_ir.tensors["r0"] = TensorIR(
        name="r0",
        dtype="FLOAT32",
        shape=[1, 8, 4, 4],
        shape_signature=[1, 8, 4, 4],
    )
    model_ir.tensors["r1"] = TensorIR(
        name="r1",
        dtype="FLOAT32",
        shape=[1, 8, 4, 4],
        shape_signature=[1, 8, 4, 4],
    )
    model_ir.tensors["cat_nchw"] = TensorIR(
        name="cat_nchw",
        dtype="FLOAT32",
        shape=[1, 16, 4, 4],
        shape_signature=[1, 16, 4, 4],
    )
    model_ir.tensors["reshape_shape"] = TensorIR(
        name="reshape_shape",
        dtype="INT32",
        shape=[3],
        shape_signature=[3],
        data=np.asarray([1, 16, 16], dtype=np.int32),
        is_variable=False,
    )
    model_ir.tensors["z"] = TensorIR(
        name="z",
        dtype="FLOAT32",
        shape=[1, 16, 16],
        shape_signature=[1, 16, 16],
    )
    model_ir.operators = [
        OperatorIR(op_type="TRANSPOSE", inputs=["x0_nhwc", "pre_perm"], outputs=["x0_nchw"]),
        OperatorIR(op_type="RELU", inputs=["x0_nchw"], outputs=["r0"]),
        OperatorIR(op_type="TRANSPOSE", inputs=["x1_nhwc", "pre_perm"], outputs=["x1_nchw"]),
        OperatorIR(op_type="RELU", inputs=["x1_nchw"], outputs=["r1"]),
        OperatorIR(
            op_type="CONCATENATION",
            inputs=["r0", "r1"],
            outputs=["cat_nchw"],
            options={"axis": 1, "fusedActivationFunction": "NONE"},
        ),
        OperatorIR(op_type="RESHAPE", inputs=["cat_nchw", "reshape_shape"], outputs=["z"]),
    ]

    stats = _optimize_transpose_input_chains_pre_concat_to_single_post_adapter(model_ir)
    assert stats["optimized_transpose_input_chains_pre_concat_to_single_post_adapter"] == 1

    op_types = [str(op.op_type) for op in model_ir.operators]
    assert op_types.count("TRANSPOSE") == 1
    concat_idx = op_types.index("CONCATENATION")
    concat_op = model_ir.operators[concat_idx]
    assert int(concat_op.options.get("axis", -1)) == 3
    assert str(model_ir.operators[concat_idx + 1].op_type) == "TRANSPOSE"
    assert model_ir.operators[concat_idx + 1].outputs == ["cat_nchw"]
    assert model_ir.operators[concat_idx + 2].inputs[0] == "cat_nchw"

    relu_ops = [op for op in model_ir.operators if str(op.op_type) == "RELU"]
    assert relu_ops[0].inputs == ["x0_nhwc"]
    assert relu_ops[1].inputs == ["x1_nhwc"]
    assert list(model_ir.tensors["r0"].shape) == [1, 4, 4, 8]
    assert list(model_ir.tensors["r1"].shape) == [1, 4, 4, 8]


def test_flatbuffer_direct_transpose_pre_concat_single_post_adapter_supports_unary_after_singleton_reshape() -> None:
    model_ir = ModelIR(name="transpose_pre_concat_single_post_adapter_unary_singleton_reshape_test")
    model_ir.inputs = ["b0_nhwc", "b1_nhwc", "b2_nhwc"]
    model_ir.outputs = ["out_nchw"]
    model_ir.tensors["b0_nhwc"] = TensorIR(
        name="b0_nhwc",
        dtype="FLOAT32",
        shape=[1, 22, 22, 1],
        shape_signature=[1, 22, 22, 1],
    )
    model_ir.tensors["b1_nhwc"] = TensorIR(
        name="b1_nhwc",
        dtype="FLOAT32",
        shape=[1, 22, 22, 4],
        shape_signature=[1, 22, 22, 4],
    )
    model_ir.tensors["b2_nhwc"] = TensorIR(
        name="b2_nhwc",
        dtype="FLOAT32",
        shape=[1, 22, 22, 80],
        shape_signature=[1, 22, 22, 80],
    )
    model_ir.tensors["to_nchw_perm"] = TensorIR(
        name="to_nchw_perm",
        dtype="INT32",
        shape=[4],
        shape_signature=[4],
        data=np.asarray([0, 3, 1, 2], dtype=np.int32),
        is_variable=False,
    )
    model_ir.tensors["shape_nchw"] = TensorIR(
        name="shape_nchw",
        dtype="INT32",
        shape=[4],
        shape_signature=[4],
        data=np.asarray([1, 1, 22, 22], dtype=np.int32),
        is_variable=False,
    )
    model_ir.tensors["b0_nchw"] = TensorIR(
        name="b0_nchw",
        dtype="FLOAT32",
        shape=[1, 1, 22, 22],
        shape_signature=[1, 1, 22, 22],
    )
    model_ir.tensors["s0_nchw"] = TensorIR(
        name="s0_nchw",
        dtype="FLOAT32",
        shape=[1, 1, 22, 22],
        shape_signature=[1, 1, 22, 22],
    )
    model_ir.tensors["b1_nchw"] = TensorIR(
        name="b1_nchw",
        dtype="FLOAT32",
        shape=[1, 4, 22, 22],
        shape_signature=[1, 4, 22, 22],
    )
    model_ir.tensors["b2_nchw"] = TensorIR(
        name="b2_nchw",
        dtype="FLOAT32",
        shape=[1, 80, 22, 22],
        shape_signature=[1, 80, 22, 22],
    )
    model_ir.tensors["out_nchw"] = TensorIR(
        name="out_nchw",
        dtype="FLOAT32",
        shape=[1, 85, 22, 22],
        shape_signature=[1, 85, 22, 22],
    )
    model_ir.operators = [
        OperatorIR(op_type="RESHAPE", inputs=["b0_nhwc", "shape_nchw"], outputs=["b0_nchw"]),
        OperatorIR(op_type="LOGISTIC", inputs=["b0_nchw"], outputs=["s0_nchw"]),
        OperatorIR(op_type="TRANSPOSE", inputs=["b1_nhwc", "to_nchw_perm"], outputs=["b1_nchw"]),
        OperatorIR(op_type="TRANSPOSE", inputs=["b2_nhwc", "to_nchw_perm"], outputs=["b2_nchw"]),
        OperatorIR(
            op_type="CONCATENATION",
            inputs=["s0_nchw", "b1_nchw", "b2_nchw"],
            outputs=["out_nchw"],
            options={"axis": 1, "fusedActivationFunction": "NONE"},
        ),
    ]

    stats = _optimize_transpose_input_chains_pre_concat_to_single_post_adapter(model_ir)
    assert stats["optimized_transpose_input_chains_pre_concat_to_single_post_adapter"] == 1

    op_types = [str(op.op_type) for op in model_ir.operators]
    assert op_types.count("RESHAPE") == 0
    assert op_types.count("TRANSPOSE") == 1
    concat_idx = op_types.index("CONCATENATION")
    concat_op = model_ir.operators[concat_idx]
    assert int(concat_op.options.get("axis", -1)) == 3
    assert str(model_ir.operators[concat_idx + 1].op_type) == "TRANSPOSE"
    assert model_ir.operators[concat_idx + 1].outputs == ["out_nchw"]

    logistic_op = next(op for op in model_ir.operators if str(op.op_type) == "LOGISTIC")
    assert logistic_op.inputs == ["b0_nhwc"]
    assert list(model_ir.tensors["s0_nchw"].shape) == [1, 22, 22, 1]


def test_flatbuffer_direct_shufflenet_transpose_shuffle_chain_optimized() -> None:
    model_ir = ModelIR(name="shufflenet_transpose_shuffle_chain_opt_test")
    model_ir.inputs = ["x0_nhwc", "x1_nhwc"]
    model_ir.outputs = ["out_nchw"]

    model_ir.tensors["x0_nhwc"] = TensorIR(
        name="x0_nhwc",
        dtype="FLOAT32",
        shape=[1, 2, 2, 4],
        shape_signature=[1, 2, 2, 4],
    )
    model_ir.tensors["x1_nhwc"] = TensorIR(
        name="x1_nhwc",
        dtype="FLOAT32",
        shape=[1, 2, 2, 4],
        shape_signature=[1, 2, 2, 4],
    )
    model_ir.tensors["to_nchw_perm"] = TensorIR(
        name="to_nchw_perm",
        dtype="INT32",
        shape=[4],
        shape_signature=[4],
        data=np.asarray([0, 3, 1, 2], dtype=np.int32),
        is_variable=False,
    )
    model_ir.tensors["to_nhwc_perm"] = TensorIR(
        name="to_nhwc_perm",
        dtype="INT32",
        shape=[4],
        shape_signature=[4],
        data=np.asarray([0, 2, 3, 1], dtype=np.int32),
        is_variable=False,
    )
    model_ir.tensors["perm_102"] = TensorIR(
        name="perm_102",
        dtype="INT32",
        shape=[3],
        shape_signature=[3],
        data=np.asarray([1, 0, 2], dtype=np.int32),
        is_variable=False,
    )
    model_ir.tensors["shape_r1"] = TensorIR(
        name="shape_r1",
        dtype="INT32",
        shape=[3],
        shape_signature=[3],
        data=np.asarray([4, 2, 4], dtype=np.int32),
        is_variable=False,
    )
    model_ir.tensors["shape_r2"] = TensorIR(
        name="shape_r2",
        dtype="INT32",
        shape=[5],
        shape_signature=[5],
        data=np.asarray([2, 1, 4, 2, 2], dtype=np.int32),
        is_variable=False,
    )
    model_ir.tensors["idx0"] = TensorIR(
        name="idx0",
        dtype="INT32",
        shape=[],
        shape_signature=[],
        data=np.asarray(0, dtype=np.int32),
        is_variable=False,
    )
    model_ir.tensors["idx1"] = TensorIR(
        name="idx1",
        dtype="INT32",
        shape=[],
        shape_signature=[],
        data=np.asarray(1, dtype=np.int32),
        is_variable=False,
    )

    model_ir.tensors["x_nhwc"] = TensorIR(
        name="x_nhwc",
        dtype="FLOAT32",
        shape=[1, 2, 2, 8],
        shape_signature=[1, 2, 2, 8],
    )
    model_ir.tensors["x_nchw"] = TensorIR(
        name="x_nchw",
        dtype="FLOAT32",
        shape=[1, 8, 2, 2],
        shape_signature=[1, 8, 2, 2],
    )
    model_ir.tensors["r1"] = TensorIR(
        name="r1",
        dtype="FLOAT32",
        shape=[4, 2, 4],
        shape_signature=[4, 2, 4],
    )
    model_ir.tensors["t1"] = TensorIR(
        name="t1",
        dtype="FLOAT32",
        shape=[2, 4, 4],
        shape_signature=[2, 4, 4],
    )
    model_ir.tensors["r2"] = TensorIR(
        name="r2",
        dtype="FLOAT32",
        shape=[2, 1, 4, 2, 2],
        shape_signature=[2, 1, 4, 2, 2],
    )
    model_ir.tensors["g0_nchw"] = TensorIR(
        name="g0_nchw",
        dtype="FLOAT32",
        shape=[1, 4, 2, 2],
        shape_signature=[1, 4, 2, 2],
    )
    model_ir.tensors["g1_nchw"] = TensorIR(
        name="g1_nchw",
        dtype="FLOAT32",
        shape=[1, 4, 2, 2],
        shape_signature=[1, 4, 2, 2],
    )
    model_ir.tensors["g1_nhwc"] = TensorIR(
        name="g1_nhwc",
        dtype="FLOAT32",
        shape=[1, 2, 2, 4],
        shape_signature=[1, 2, 2, 4],
    )
    model_ir.tensors["b1"] = TensorIR(
        name="b1",
        dtype="FLOAT32",
        shape=[1, 2, 2, 4],
        shape_signature=[1, 2, 2, 4],
    )
    model_ir.tensors["b2"] = TensorIR(
        name="b2",
        dtype="FLOAT32",
        shape=[1, 2, 2, 4],
        shape_signature=[1, 2, 2, 4],
    )
    model_ir.tensors["b3"] = TensorIR(
        name="b3",
        dtype="FLOAT32",
        shape=[1, 2, 2, 4],
        shape_signature=[1, 2, 2, 4],
    )
    model_ir.tensors["b3_nchw"] = TensorIR(
        name="b3_nchw",
        dtype="FLOAT32",
        shape=[1, 4, 2, 2],
        shape_signature=[1, 4, 2, 2],
    )
    model_ir.tensors["b3_relu_nchw"] = TensorIR(
        name="b3_relu_nchw",
        dtype="FLOAT32",
        shape=[1, 4, 2, 2],
        shape_signature=[1, 4, 2, 2],
    )
    model_ir.tensors["out_nchw"] = TensorIR(
        name="out_nchw",
        dtype="FLOAT32",
        shape=[1, 8, 2, 2],
        shape_signature=[1, 8, 2, 2],
    )

    model_ir.operators = [
        OperatorIR(
            op_type="CONCATENATION",
            inputs=["x0_nhwc", "x1_nhwc"],
            outputs=["x_nhwc"],
            options={"axis": 3, "fusedActivationFunction": "NONE"},
        ),
        OperatorIR(op_type="TRANSPOSE", inputs=["x_nhwc", "to_nchw_perm"], outputs=["x_nchw"]),
        OperatorIR(op_type="RESHAPE", inputs=["x_nchw", "shape_r1"], outputs=["r1"]),
        OperatorIR(op_type="TRANSPOSE", inputs=["r1", "perm_102"], outputs=["t1"]),
        OperatorIR(op_type="RESHAPE", inputs=["t1", "shape_r2"], outputs=["r2"]),
        OperatorIR(op_type="GATHER", inputs=["r2", "idx0"], outputs=["g0_nchw"], options={"axis": 0, "batchDims": 0}),
        OperatorIR(op_type="GATHER", inputs=["r2", "idx1"], outputs=["g1_nchw"], options={"axis": 0, "batchDims": 0}),
        OperatorIR(op_type="TRANSPOSE", inputs=["g1_nchw", "to_nhwc_perm"], outputs=["g1_nhwc"]),
        OperatorIR(op_type="CONV_2D", inputs=["g1_nhwc"], outputs=["b1"], options={"padding": "SAME", "strideH": 1, "strideW": 1, "filterHeight": 1, "filterWidth": 1, "fusedActivationFunction": "NONE"}),
        OperatorIR(op_type="DEPTHWISE_CONV_2D", inputs=["b1"], outputs=["b2"], options={"padding": "SAME", "strideH": 1, "strideW": 1, "depthMultiplier": 1, "dilationHFactor": 1, "dilationWFactor": 1, "fusedActivationFunction": "NONE"}),
        OperatorIR(op_type="CONV_2D", inputs=["b2"], outputs=["b3"], options={"padding": "SAME", "strideH": 1, "strideW": 1, "filterHeight": 1, "filterWidth": 1, "fusedActivationFunction": "NONE"}),
        OperatorIR(op_type="TRANSPOSE", inputs=["b3", "to_nchw_perm"], outputs=["b3_nchw"]),
        OperatorIR(op_type="RELU", inputs=["b3_nchw"], outputs=["b3_relu_nchw"]),
        OperatorIR(
            op_type="CONCATENATION",
            inputs=["g0_nchw", "b3_relu_nchw"],
            outputs=["out_nchw"],
            options={"axis": 1, "fusedActivationFunction": "NONE"},
        ),
    ]

    stats = _optimize_shufflenet_transpose_shuffle_chains(model_ir)
    assert stats["optimized_shufflenet_transpose_shuffle_chains"] == 1

    op_types = [str(op.op_type) for op in model_ir.operators]
    assert op_types.count("TRANSPOSE") <= 2

    concat_ops = [op for op in model_ir.operators if str(op.op_type) == "CONCATENATION"]
    assert len(concat_ops) == 2
    # Tail concat is rewritten to NHWC + post transpose adapter.
    assert int(concat_ops[1].options.get("axis", -1)) == 3
    tail_concat_idx = op_types.index("CONCATENATION", op_types.index("CONCATENATION") + 1)
    assert str(model_ir.operators[tail_concat_idx + 1].op_type) == "TRANSPOSE"
    assert model_ir.operators[tail_concat_idx + 1].outputs == ["out_nchw"]

    gather_ops = [op for op in model_ir.operators if str(op.op_type) == "GATHER"]
    assert len(gather_ops) >= 2
    assert all(int(op.options.get("axis", -1)) == 3 for op in gather_ops[:2])


def test_flatbuffer_direct_shufflenet_transpose_slice_shuffle_chain_optimized() -> None:
    model_ir = ModelIR(name="shufflenet_transpose_slice_shuffle_chain_opt_test")
    model_ir.inputs = ["x0_nhwc", "x1_nhwc"]
    model_ir.outputs = ["out_nchw"]

    model_ir.tensors["x0_nhwc"] = TensorIR(
        name="x0_nhwc",
        dtype="FLOAT32",
        shape=[1, 2, 2, 4],
        shape_signature=[1, 2, 2, 4],
    )
    model_ir.tensors["x1_nhwc"] = TensorIR(
        name="x1_nhwc",
        dtype="FLOAT32",
        shape=[1, 2, 2, 4],
        shape_signature=[1, 2, 2, 4],
    )
    model_ir.tensors["to_nchw_perm"] = TensorIR(
        name="to_nchw_perm",
        dtype="INT32",
        shape=[4],
        shape_signature=[4],
        data=np.asarray([0, 3, 1, 2], dtype=np.int32),
        is_variable=False,
    )
    model_ir.tensors["to_nhwc_perm"] = TensorIR(
        name="to_nhwc_perm",
        dtype="INT32",
        shape=[4],
        shape_signature=[4],
        data=np.asarray([0, 2, 3, 1], dtype=np.int32),
        is_variable=False,
    )
    model_ir.tensors["perm_swap"] = TensorIR(
        name="perm_swap",
        dtype="INT32",
        shape=[5],
        shape_signature=[5],
        data=np.asarray([0, 2, 1, 3, 4], dtype=np.int32),
        is_variable=False,
    )
    model_ir.tensors["shape_r1"] = TensorIR(
        name="shape_r1",
        dtype="INT32",
        shape=[5],
        shape_signature=[5],
        data=np.asarray([1, 2, 4, 2, 2], dtype=np.int32),
        is_variable=False,
    )
    model_ir.tensors["shape_r2"] = TensorIR(
        name="shape_r2",
        dtype="INT32",
        shape=[4],
        shape_signature=[4],
        data=np.asarray([1, 8, 2, 2], dtype=np.int32),
        is_variable=False,
    )
    model_ir.tensors["slice0_begin"] = TensorIR(
        name="slice0_begin",
        dtype="INT32",
        shape=[4],
        shape_signature=[4],
        data=np.asarray([0, 0, 0, 0], dtype=np.int32),
        is_variable=False,
    )
    model_ir.tensors["slice0_size"] = TensorIR(
        name="slice0_size",
        dtype="INT32",
        shape=[4],
        shape_signature=[4],
        data=np.asarray([-1, 4, -1, -1], dtype=np.int32),
        is_variable=False,
    )
    model_ir.tensors["slice1_begin"] = TensorIR(
        name="slice1_begin",
        dtype="INT32",
        shape=[4],
        shape_signature=[4],
        data=np.asarray([0, 4, 0, 0], dtype=np.int32),
        is_variable=False,
    )
    model_ir.tensors["slice1_size"] = TensorIR(
        name="slice1_size",
        dtype="INT32",
        shape=[4],
        shape_signature=[4],
        data=np.asarray([-1, 4, -1, -1], dtype=np.int32),
        is_variable=False,
    )

    model_ir.tensors["x_nhwc"] = TensorIR(
        name="x_nhwc",
        dtype="FLOAT32",
        shape=[1, 2, 2, 8],
        shape_signature=[1, 2, 2, 8],
    )
    model_ir.tensors["x_nchw"] = TensorIR(
        name="x_nchw",
        dtype="FLOAT32",
        shape=[1, 8, 2, 2],
        shape_signature=[1, 8, 2, 2],
    )
    model_ir.tensors["r1"] = TensorIR(
        name="r1",
        dtype="FLOAT32",
        shape=[1, 2, 4, 2, 2],
        shape_signature=[1, 2, 4, 2, 2],
    )
    model_ir.tensors["t1"] = TensorIR(
        name="t1",
        dtype="FLOAT32",
        shape=[1, 4, 2, 2, 2],
        shape_signature=[1, 4, 2, 2, 2],
    )
    model_ir.tensors["r2"] = TensorIR(
        name="r2",
        dtype="FLOAT32",
        shape=[1, 8, 2, 2],
        shape_signature=[1, 8, 2, 2],
    )
    model_ir.tensors["s0_nchw"] = TensorIR(
        name="s0_nchw",
        dtype="FLOAT32",
        shape=[1, 4, 2, 2],
        shape_signature=[1, 4, 2, 2],
    )
    model_ir.tensors["s1_nchw"] = TensorIR(
        name="s1_nchw",
        dtype="FLOAT32",
        shape=[1, 4, 2, 2],
        shape_signature=[1, 4, 2, 2],
    )
    model_ir.tensors["s1_nhwc"] = TensorIR(
        name="s1_nhwc",
        dtype="FLOAT32",
        shape=[1, 2, 2, 4],
        shape_signature=[1, 2, 2, 4],
    )
    model_ir.tensors["b1"] = TensorIR(
        name="b1",
        dtype="FLOAT32",
        shape=[1, 2, 2, 4],
        shape_signature=[1, 2, 2, 4],
    )
    model_ir.tensors["b2"] = TensorIR(
        name="b2",
        dtype="FLOAT32",
        shape=[1, 2, 2, 4],
        shape_signature=[1, 2, 2, 4],
    )
    model_ir.tensors["b3"] = TensorIR(
        name="b3",
        dtype="FLOAT32",
        shape=[1, 2, 2, 4],
        shape_signature=[1, 2, 2, 4],
    )
    model_ir.tensors["b3_nchw"] = TensorIR(
        name="b3_nchw",
        dtype="FLOAT32",
        shape=[1, 4, 2, 2],
        shape_signature=[1, 4, 2, 2],
    )
    model_ir.tensors["b3_relu_nchw"] = TensorIR(
        name="b3_relu_nchw",
        dtype="FLOAT32",
        shape=[1, 4, 2, 2],
        shape_signature=[1, 4, 2, 2],
    )
    model_ir.tensors["out_nchw"] = TensorIR(
        name="out_nchw",
        dtype="FLOAT32",
        shape=[1, 8, 2, 2],
        shape_signature=[1, 8, 2, 2],
    )

    model_ir.operators = [
        OperatorIR(
            op_type="CONCATENATION",
            inputs=["x0_nhwc", "x1_nhwc"],
            outputs=["x_nhwc"],
            options={"axis": 3, "fusedActivationFunction": "NONE"},
        ),
        OperatorIR(op_type="TRANSPOSE", inputs=["x_nhwc", "to_nchw_perm"], outputs=["x_nchw"]),
        OperatorIR(op_type="RESHAPE", inputs=["x_nchw", "shape_r1"], outputs=["r1"]),
        OperatorIR(op_type="TRANSPOSE", inputs=["r1", "perm_swap"], outputs=["t1"]),
        OperatorIR(op_type="RESHAPE", inputs=["t1", "shape_r2"], outputs=["r2"]),
        OperatorIR(op_type="SLICE", inputs=["r2", "slice0_begin", "slice0_size"], outputs=["s0_nchw"]),
        OperatorIR(op_type="SLICE", inputs=["r2", "slice1_begin", "slice1_size"], outputs=["s1_nchw"]),
        OperatorIR(op_type="TRANSPOSE", inputs=["s1_nchw", "to_nhwc_perm"], outputs=["s1_nhwc"]),
        OperatorIR(op_type="CONV_2D", inputs=["s1_nhwc"], outputs=["b1"], options={"padding": "SAME", "strideH": 1, "strideW": 1, "filterHeight": 1, "filterWidth": 1, "fusedActivationFunction": "NONE"}),
        OperatorIR(op_type="DEPTHWISE_CONV_2D", inputs=["b1"], outputs=["b2"], options={"padding": "SAME", "strideH": 1, "strideW": 1, "depthMultiplier": 1, "dilationHFactor": 1, "dilationWFactor": 1, "fusedActivationFunction": "NONE"}),
        OperatorIR(op_type="CONV_2D", inputs=["b2"], outputs=["b3"], options={"padding": "SAME", "strideH": 1, "strideW": 1, "filterHeight": 1, "filterWidth": 1, "fusedActivationFunction": "NONE"}),
        OperatorIR(op_type="TRANSPOSE", inputs=["b3", "to_nchw_perm"], outputs=["b3_nchw"]),
        OperatorIR(op_type="RELU", inputs=["b3_nchw"], outputs=["b3_relu_nchw"]),
        OperatorIR(
            op_type="CONCATENATION",
            inputs=["s0_nchw", "b3_relu_nchw"],
            outputs=["out_nchw"],
            options={"axis": 1, "fusedActivationFunction": "NONE"},
        ),
    ]

    stats = _optimize_shufflenet_transpose_shuffle_chains(model_ir)
    assert stats["optimized_shufflenet_transpose_shuffle_chains"] == 1

    op_types = [str(op.op_type) for op in model_ir.operators]
    assert op_types.count("SLICE") == 0

    concat_ops = [op for op in model_ir.operators if str(op.op_type) == "CONCATENATION"]
    assert len(concat_ops) == 2
    assert int(concat_ops[1].options.get("axis", -1)) == 3
    tail_concat_idx = op_types.index("CONCATENATION", op_types.index("CONCATENATION") + 1)
    assert str(model_ir.operators[tail_concat_idx + 1].op_type) == "TRANSPOSE"
    assert model_ir.operators[tail_concat_idx + 1].outputs == ["out_nchw"]

    gather_ops = [op for op in model_ir.operators if str(op.op_type) == "GATHER"]
    assert len(gather_ops) >= 2
    assert all(int(op.options.get("axis", -1)) == 3 for op in gather_ops[:2])
    gather0_indices = np.asarray(model_ir.tensors[str(gather_ops[0].inputs[1])].data, dtype=np.int32).reshape(-1)
    gather1_indices = np.asarray(model_ir.tensors[str(gather_ops[1].inputs[1])].data, dtype=np.int32).reshape(-1)
    assert np.array_equal(gather0_indices, np.asarray([0, 4, 1, 5], dtype=np.int32))
    assert np.array_equal(gather1_indices, np.asarray([2, 6, 3, 7], dtype=np.int32))


def test_flatbuffer_direct_transpose_slice_logistic_concat_reshape_tail_nhwc_chain_optimized() -> None:
    model_ir = ModelIR(name="transpose_slice_logistic_concat_reshape_tail_nhwc_opt_test")
    model_ir.inputs = ["x0_nhwc", "x1_nhwc", "x2_nhwc", "x3_nhwc"]
    model_ir.outputs = ["out_nchw"]

    def _add_tensor(name: str, dtype: str, shape: list[int], data: np.ndarray | None = None) -> None:
        model_ir.tensors[name] = TensorIR(
            name=name,
            dtype=dtype,
            shape=[int(v) for v in list(shape)],
            shape_signature=[int(v) for v in list(shape)],
            data=(None if data is None else np.asarray(data)),
            is_variable=False,
        )

    _add_tensor(
        "to_nchw_perm",
        "INT32",
        [4],
        np.asarray([0, 3, 1, 2], dtype=np.int32),
    )

    branch_specs = [
        ("x0", 52, 52, 2704),
        ("x1", 26, 26, 676),
        ("x2", 13, 13, 169),
        ("x3", 7, 7, 49),
    ]

    operators: list[OperatorIR] = []
    reshape_outputs: list[str] = []
    for branch_name, height, width, spatial in branch_specs:
        x_nhwc = f"{branch_name}_nhwc"
        x_nchw = f"{branch_name}_nchw"
        slice0_out = f"{branch_name}_slice0"
        slice1_out = f"{branch_name}_slice1"
        sig_out = f"{branch_name}_sig"
        cat_out = f"{branch_name}_cat"
        reshape_out = f"{branch_name}_reshape"
        reshape_shape = f"{branch_name}_reshape_shape"
        split0_begin = f"{branch_name}_split0_begin"
        split0_size = f"{branch_name}_split0_size"
        split1_begin = f"{branch_name}_split1_begin"
        split1_size = f"{branch_name}_split1_size"

        _add_tensor(x_nhwc, "FLOAT32", [1, int(height), int(width), 37])
        _add_tensor(x_nchw, "FLOAT32", [1, 37, int(height), int(width)])
        _add_tensor(slice0_out, "FLOAT32", [1, 5, int(height), int(width)])
        _add_tensor(slice1_out, "FLOAT32", [1, 32, int(height), int(width)])
        _add_tensor(sig_out, "FLOAT32", [1, 5, int(height), int(width)])
        _add_tensor(cat_out, "FLOAT32", [1, 37, int(height), int(width)])
        _add_tensor(reshape_out, "FLOAT32", [1, 37, int(spatial)])
        _add_tensor(
            reshape_shape,
            "INT32",
            [3],
            np.asarray([1, 37, int(spatial)], dtype=np.int32),
        )
        _add_tensor(
            split0_begin,
            "INT32",
            [4],
            np.asarray([0, 0, 0, 0], dtype=np.int32),
        )
        _add_tensor(
            split0_size,
            "INT32",
            [4],
            np.asarray([1, 5, int(height), int(width)], dtype=np.int32),
        )
        _add_tensor(
            split1_begin,
            "INT32",
            [4],
            np.asarray([0, 5, 0, 0], dtype=np.int32),
        )
        _add_tensor(
            split1_size,
            "INT32",
            [4],
            np.asarray([1, 32, int(height), int(width)], dtype=np.int32),
        )

        operators.extend(
            [
                OperatorIR(
                    op_type="TRANSPOSE",
                    inputs=[x_nhwc, "to_nchw_perm"],
                    outputs=[x_nchw],
                ),
                OperatorIR(
                    op_type="SLICE",
                    inputs=[x_nchw, split0_begin, split0_size],
                    outputs=[slice0_out],
                ),
                OperatorIR(
                    op_type="SLICE",
                    inputs=[x_nchw, split1_begin, split1_size],
                    outputs=[slice1_out],
                ),
                OperatorIR(
                    op_type="LOGISTIC",
                    inputs=[slice0_out],
                    outputs=[sig_out],
                ),
                OperatorIR(
                    op_type="CONCATENATION",
                    inputs=[sig_out, slice1_out],
                    outputs=[cat_out],
                    options={"axis": 1, "fusedActivationFunction": "NONE"},
                ),
                OperatorIR(
                    op_type="RESHAPE",
                    inputs=[cat_out, reshape_shape],
                    outputs=[reshape_out],
                    options={
                        "newShape": [1, 37, int(spatial)],
                        "onnxRawNewShape": [1, 37, -1],
                        "allowZero": False,
                    },
                ),
            ]
        )
        reshape_outputs.append(reshape_out)

    _add_tensor("out_nchw", "FLOAT32", [1, 37, 3598])
    operators.append(
        OperatorIR(
            op_type="CONCATENATION",
            inputs=[str(v) for v in reshape_outputs],
            outputs=["out_nchw"],
            options={"axis": 2, "fusedActivationFunction": "NONE"},
        )
    )
    model_ir.operators = list(operators)

    stats = _optimize_transpose_slice_logistic_concat_reshape_tail_nhwc_chains(model_ir)
    assert stats["optimized_transpose_slice_logistic_concat_reshape_tail_nhwc_chains"] == 1

    op_types = [str(op.op_type) for op in model_ir.operators]
    assert op_types.count("TRANSPOSE") == 1

    transpose_op = next(op for op in model_ir.operators if str(op.op_type) == "TRANSPOSE")
    assert list(transpose_op.outputs) == ["out_nchw"]
    perm_vals = np.asarray(model_ir.tensors[str(transpose_op.inputs[1])].data, dtype=np.int32).reshape(-1).tolist()
    assert perm_vals == [0, 2, 1]

    tail_concat = next(
        op
        for op in model_ir.operators
        if str(op.op_type) == "CONCATENATION"
        and len(list(op.inputs)) == len(reshape_outputs)
        and set(str(v) for v in list(op.inputs)) == set(str(v) for v in reshape_outputs)
    )
    assert int(tail_concat.options.get("axis", -1)) == 1
    assert list(tail_concat.outputs) == [str(transpose_op.inputs[0])]

    for branch_name, height, width, spatial in branch_specs:
        cat_op = next(op for op in model_ir.operators if list(op.outputs) == [f"{branch_name}_cat"])
        assert int(cat_op.options.get("axis", -1)) == 3
        reshape_op = next(op for op in model_ir.operators if list(op.outputs) == [f"{branch_name}_reshape"])
        assert list(reshape_op.options.get("newShape", [])) == [1, int(spatial), 37]
        assert list(reshape_op.options.get("onnxRawNewShape", [])) == [1, -1, 37]
        shape_vals = np.asarray(model_ir.tensors[f"{branch_name}_reshape_shape"].data, dtype=np.int32).reshape(-1).tolist()
        assert shape_vals == [1, int(spatial), 37]

        split1_begin_vals = np.asarray(model_ir.tensors[f"{branch_name}_split1_begin"].data, dtype=np.int32).reshape(-1).tolist()
        split1_size_vals = np.asarray(model_ir.tensors[f"{branch_name}_split1_size"].data, dtype=np.int32).reshape(-1).tolist()
        assert split1_begin_vals == [0, 0, 0, 5]
        assert split1_size_vals == [1, int(height), int(width), 32]


def test_flatbuffer_direct_transpose_cost_volume_scatter_ndhwc_chain_optimized() -> None:
    model_ir = ModelIR(name="transpose_cost_volume_scatter_ndhwc_opt_test")
    model_ir.inputs = ["desc0_nhwc", "desc1_nhwc"]
    model_ir.outputs = ["conv_out"]

    def _add_tensor(name: str, dtype: str, shape: list[int], data: np.ndarray | None = None) -> None:
        model_ir.tensors[name] = TensorIR(
            name=name,
            dtype=dtype,
            shape=[int(v) for v in list(shape)],
            shape_signature=[int(v) for v in list(shape)],
            data=(None if data is None else np.asarray(data)),
            is_variable=False,
        )

    _add_tensor("desc0_nhwc", "FLOAT32", [1, 3, 6, 4])
    _add_tensor("desc1_nhwc", "FLOAT32", [1, 3, 6, 4])
    _add_tensor("desc0_nchw", "FLOAT32", [1, 4, 3, 6])
    _add_tensor("desc1_nchw", "FLOAT32", [1, 4, 3, 6])
    _add_tensor("slice_begin", "INT32", [4], np.asarray([0, 0, 0, 1], dtype=np.int32))
    _add_tensor("slice_size", "INT32", [4], np.asarray([1, 4, 3, 5], dtype=np.int32))
    _add_tensor("slice0", "FLOAT32", [1, 4, 3, 5])
    _add_tensor("slice1", "FLOAT32", [1, 4, 3, 5])
    _add_tensor("mul_out", "FLOAT32", [1, 4, 3, 5])
    _add_tensor("mean_axes", "INT32", [1], np.asarray([1], dtype=np.int32))
    _add_tensor("mean_out", "FLOAT32", [1, 1, 3, 5])
    _add_tensor("reshape_shape", "INT32", [5], np.asarray([1, 1, 1, 3, 5], dtype=np.int32))
    _add_tensor("reshape_out", "FLOAT32", [1, 1, 1, 3, 5])
    _add_tensor(
        "scatter_shape",
        "INT32",
        [5],
        np.asarray([1, 1, 3, 4, 6], dtype=np.int32),
    )
    index_data = np.zeros((1, 1, 1, 3, 5, 5), dtype=np.float32)
    for h in range(3):
        for w in range(5):
            index_data[0, 0, 0, h, w] = np.asarray([0, 0, 0, h, w + 1], dtype=np.float32)
    _add_tensor("indices_f32", "FLOAT32", [1, 1, 1, 3, 5, 5], index_data)
    _add_tensor("indices_i32", "INT32", [1, 1, 1, 3, 5, 5])
    _add_tensor("scatter_ncdhw", "FLOAT32", [1, 1, 3, 4, 6])
    _add_tensor("scatter_ndhwc", "FLOAT32", [1, 3, 4, 6, 1])
    _add_tensor("conv_w", "FLOAT32", [1, 1, 1, 1, 2])
    _add_tensor("conv_b", "FLOAT32", [2])
    _add_tensor("conv_out", "FLOAT32", [1, 3, 4, 6, 2])
    _add_tensor(
        "perm_nhwc_to_nchw",
        "INT32",
        [4],
        np.asarray([0, 3, 1, 2], dtype=np.int32),
    )
    _add_tensor(
        "perm_ncdhw_to_ndhwc",
        "INT32",
        [5],
        np.asarray([0, 2, 3, 4, 1], dtype=np.int32),
    )

    model_ir.operators = [
        OperatorIR(
            op_type="TRANSPOSE",
            inputs=["desc0_nhwc", "perm_nhwc_to_nchw"],
            outputs=["desc0_nchw"],
        ),
        OperatorIR(
            op_type="TRANSPOSE",
            inputs=["desc1_nhwc", "perm_nhwc_to_nchw"],
            outputs=["desc1_nchw"],
        ),
        OperatorIR(
            op_type="SLICE",
            inputs=["desc0_nchw", "slice_begin", "slice_size"],
            outputs=["slice0"],
        ),
        OperatorIR(
            op_type="SLICE",
            inputs=["desc1_nchw", "slice_begin", "slice_size"],
            outputs=["slice1"],
        ),
        OperatorIR(
            op_type="MUL",
            inputs=["slice0", "slice1"],
            outputs=["mul_out"],
        ),
        OperatorIR(
            op_type="MEAN",
            inputs=["mul_out", "mean_axes"],
            outputs=["mean_out"],
            options={"keepDims": True},
        ),
        OperatorIR(
            op_type="RESHAPE",
            inputs=["mean_out", "reshape_shape"],
            outputs=["reshape_out"],
            options={"newShape": [1, 1, 1, 3, 5], "onnxRawNewShape": [1, 1, 1, 3, 5]},
        ),
        OperatorIR(
            op_type="CAST",
            inputs=["indices_f32"],
            outputs=["indices_i32"],
            options={"inDataType": "FLOAT32", "outDataType": "INT32"},
        ),
        OperatorIR(
            op_type="SCATTER_ND",
            inputs=["indices_i32", "reshape_out", "scatter_shape"],
            outputs=["scatter_ncdhw"],
        ),
        OperatorIR(
            op_type="TRANSPOSE",
            inputs=["scatter_ncdhw", "perm_ncdhw_to_ndhwc"],
            outputs=["scatter_ndhwc"],
        ),
        OperatorIR(
            op_type="CONV_3D",
            inputs=["scatter_ndhwc", "conv_w", "conv_b"],
            outputs=["conv_out"],
            options={
                "padding": "SAME",
                "strideD": 1,
                "strideH": 1,
                "strideW": 1,
                "dilationDFactor": 1,
                "dilationHFactor": 1,
                "dilationWFactor": 1,
                "fusedActivationFunction": "NONE",
            },
        ),
    ]

    stats = _optimize_transpose_cost_volume_scatter_ndhwc_chains(model_ir)
    assert stats["optimized_transpose_cost_volume_scatter_ndhwc_chains"] == 1

    op_types = [str(op.op_type) for op in model_ir.operators]
    assert op_types.count("TRANSPOSE") == 0

    slice_ops = [op for op in model_ir.operators if str(op.op_type) == "SLICE"]
    assert len(slice_ops) == 2
    for slice_op in slice_ops:
        assert str(slice_op.inputs[0]) in {"desc0_nhwc", "desc1_nhwc"}
        slice_begin_vals = np.asarray(
            model_ir.tensors[str(slice_op.inputs[1])].data,
            dtype=np.int32,
        ).reshape(-1).tolist()
        slice_size_vals = np.asarray(
            model_ir.tensors[str(slice_op.inputs[2])].data,
            dtype=np.int32,
        ).reshape(-1).tolist()
        assert slice_begin_vals == [0, 0, 1, 0]
        assert slice_size_vals == [1, 3, 5, 4]

    mean_op = next(op for op in model_ir.operators if str(op.op_type) == "MEAN")
    mean_axes_vals = np.asarray(
        model_ir.tensors[str(mean_op.inputs[1])].data,
        dtype=np.int32,
    ).reshape(-1).tolist()
    assert mean_axes_vals == [3]

    scatter_op = next(op for op in model_ir.operators if str(op.op_type) == "SCATTER_ND")
    scatter_shape_vals = np.asarray(
        model_ir.tensors[str(scatter_op.inputs[2])].data,
        dtype=np.int32,
    ).reshape(-1).tolist()
    assert scatter_shape_vals == [1, 3, 4, 6, 1]

    cast_op = next(op for op in model_ir.operators if str(op.op_type) == "CAST")
    remapped_indices = np.asarray(
        model_ir.tensors[str(cast_op.inputs[0])].data,
        dtype=np.float32,
    )
    assert remapped_indices.shape == (1, 1, 1, 3, 5, 5)
    assert remapped_indices[0, 0, 0, 0, 0].tolist() == [0.0, 0.0, 0.0, 1.0, 0.0]

    conv_op = next(op for op in model_ir.operators if str(op.op_type) == "CONV_3D")
    assert str(conv_op.inputs[0]) == "scatter_ncdhw"
    assert model_ir.tensors["scatter_ncdhw"].shape == [1, 3, 4, 6, 1]
    assert model_ir.tensors["slice0"].shape == [1, 3, 5, 4]


def test_flatbuffer_direct_shufflenet_reshape_transpose_shuffle_nhwc_chain_optimized() -> None:
    model_ir = ModelIR(name="shufflenet_reshape_transpose_shuffle_nhwc_opt_test")
    model_ir.inputs = ["x_nhwc"]
    model_ir.outputs = ["y0", "y1"]

    model_ir.tensors["x_nhwc"] = TensorIR(
        name="x_nhwc",
        dtype="FLOAT32",
        shape=[1, 80, 80, 24],
        shape_signature=[1, 80, 80, 24],
    )
    model_ir.tensors["perm_nhwc_to_nchw"] = TensorIR(
        name="perm_nhwc_to_nchw",
        dtype="INT32",
        shape=[4],
        shape_signature=[4],
        data=np.asarray([0, 3, 1, 2], dtype=np.int32),
        is_variable=False,
    )
    model_ir.tensors["perm_swap"] = TensorIR(
        name="perm_swap",
        dtype="INT32",
        shape=[5],
        shape_signature=[5],
        data=np.asarray([0, 2, 1, 3, 4], dtype=np.int32),
        is_variable=False,
    )
    model_ir.tensors["perm_nchw_to_nhwc"] = TensorIR(
        name="perm_nchw_to_nhwc",
        dtype="INT32",
        shape=[4],
        shape_signature=[4],
        data=np.asarray([0, 2, 3, 1], dtype=np.int32),
        is_variable=False,
    )
    model_ir.tensors["shape_r1"] = TensorIR(
        name="shape_r1",
        dtype="INT32",
        shape=[5],
        shape_signature=[5],
        data=np.asarray([1, 2, 12, 80, 80], dtype=np.int32),
        is_variable=False,
    )
    model_ir.tensors["shape_r2"] = TensorIR(
        name="shape_r2",
        dtype="INT32",
        shape=[4],
        shape_signature=[4],
        data=np.asarray([1, 24, 80, 80], dtype=np.int32),
        is_variable=False,
    )
    model_ir.tensors["x_nchw"] = TensorIR(
        name="x_nchw",
        dtype="FLOAT32",
        shape=[1, 24, 80, 80],
        shape_signature=[1, 24, 80, 80],
    )
    model_ir.tensors["r1"] = TensorIR(
        name="r1",
        dtype="FLOAT32",
        shape=[1, 2, 12, 80, 80],
        shape_signature=[1, 2, 12, 80, 80],
    )
    model_ir.tensors["t1"] = TensorIR(
        name="t1",
        dtype="FLOAT32",
        shape=[1, 12, 2, 80, 80],
        shape_signature=[1, 12, 2, 80, 80],
    )
    model_ir.tensors["r2"] = TensorIR(
        name="r2",
        dtype="FLOAT32",
        shape=[1, 24, 80, 80],
        shape_signature=[1, 24, 80, 80],
    )
    model_ir.tensors["y_nhwc"] = TensorIR(
        name="y_nhwc",
        dtype="FLOAT32",
        shape=[1, 80, 80, 24],
        shape_signature=[1, 80, 80, 24],
    )
    model_ir.tensors["y0"] = TensorIR(
        name="y0",
        dtype="FLOAT32",
        shape=[1, 80, 80, 24],
        shape_signature=[1, 80, 80, 24],
    )
    model_ir.tensors["y1"] = TensorIR(
        name="y1",
        dtype="FLOAT32",
        shape=[1, 80, 80, 24],
        shape_signature=[1, 80, 80, 24],
    )

    model_ir.operators = [
        OperatorIR(op_type="TRANSPOSE", inputs=["x_nhwc", "perm_nhwc_to_nchw"], outputs=["x_nchw"]),
        OperatorIR(
            op_type="RESHAPE",
            inputs=["x_nchw", "shape_r1"],
            outputs=["r1"],
            options={"newShape": [1, 2, 12, 80, 80]},
        ),
        OperatorIR(op_type="TRANSPOSE", inputs=["r1", "perm_swap"], outputs=["t1"]),
        OperatorIR(
            op_type="RESHAPE",
            inputs=["t1", "shape_r2"],
            outputs=["r2"],
            options={"newShape": [1, 24, 80, 80]},
        ),
        OperatorIR(op_type="TRANSPOSE", inputs=["r2", "perm_nchw_to_nhwc"], outputs=["y_nhwc"]),
        OperatorIR(op_type="RELU", inputs=["y_nhwc"], outputs=["y0"]),
        OperatorIR(op_type="RELU", inputs=["y_nhwc"], outputs=["y1"]),
    ]

    stats = _optimize_shufflenet_reshape_transpose_shuffle_nhwc_chains(model_ir)
    assert stats["optimized_shufflenet_reshape_transpose_shuffle_nhwc_chains"] == 1

    op_types = [str(op.op_type) for op in model_ir.operators]
    assert op_types == ["GATHER", "RELU", "RELU"]
    gather_op = model_ir.operators[0]
    assert int(gather_op.options.get("axis", -1)) == 3
    gather_indices_name = str(gather_op.inputs[1])
    gather_indices = np.asarray(model_ir.tensors[gather_indices_name].data, dtype=np.int32).reshape(-1)
    expected = np.asarray([int((k % 2) * 12 + (k // 2)) for k in range(24)], dtype=np.int32)
    assert np.array_equal(gather_indices, expected)

def test_flatbuffer_direct_transpose_pre_add_nhwc_shared_concat_and_unary_input() -> None:
    model_ir = ModelIR(name="transpose_pre_add_shared_concat_and_unary_test")
    model_ir.inputs = ["x0_nhwc", "x1_nhwc"]
    model_ir.outputs = ["legacy_cat"]
    model_ir.tensors["x0_nhwc"] = TensorIR(
        name="x0_nhwc",
        dtype="FLOAT32",
        shape=[1, 6, 6, 64],
        shape_signature=[1, 6, 6, 64],
    )
    model_ir.tensors["x1_nhwc"] = TensorIR(
        name="x1_nhwc",
        dtype="FLOAT32",
        shape=[1, 6, 6, 64],
        shape_signature=[1, 6, 6, 64],
    )
    model_ir.tensors["pre_perm"] = TensorIR(
        name="pre_perm",
        dtype="INT32",
        shape=[4],
        shape_signature=[4],
        data=np.asarray([0, 3, 1, 2], dtype=np.int32),
        is_variable=False,
    )
    model_ir.tensors["post_perm"] = TensorIR(
        name="post_perm",
        dtype="INT32",
        shape=[4],
        shape_signature=[4],
        data=np.asarray([0, 2, 3, 1], dtype=np.int32),
        is_variable=False,
    )
    model_ir.tensors["x0_nchw"] = TensorIR(
        name="x0_nchw",
        dtype="FLOAT32",
        shape=[1, 64, 6, 6],
        shape_signature=[1, 64, 6, 6],
    )
    model_ir.tensors["x1_nchw"] = TensorIR(
        name="x1_nchw",
        dtype="FLOAT32",
        shape=[1, 64, 6, 6],
        shape_signature=[1, 64, 6, 6],
    )
    model_ir.tensors["x1_relu"] = TensorIR(
        name="x1_relu",
        dtype="FLOAT32",
        shape=[1, 64, 6, 6],
        shape_signature=[1, 64, 6, 6],
    )
    model_ir.tensors["sum_nchw"] = TensorIR(
        name="sum_nchw",
        dtype="FLOAT32",
        shape=[1, 64, 6, 6],
        shape_signature=[1, 64, 6, 6],
    )
    model_ir.tensors["y_nhwc"] = TensorIR(
        name="y_nhwc",
        dtype="FLOAT32",
        shape=[1, 6, 6, 64],
        shape_signature=[1, 6, 6, 64],
    )
    model_ir.tensors["legacy_cat"] = TensorIR(
        name="legacy_cat",
        dtype="FLOAT32",
        shape=[1, 128, 6, 6],
        shape_signature=[1, 128, 6, 6],
    )
    model_ir.operators = [
        OperatorIR(op_type="TRANSPOSE", inputs=["x0_nhwc", "pre_perm"], outputs=["x0_nchw"]),
        OperatorIR(op_type="TRANSPOSE", inputs=["x1_nhwc", "pre_perm"], outputs=["x1_nchw"]),
        OperatorIR(op_type="RELU", inputs=["x1_nchw"], outputs=["x1_relu"]),
        OperatorIR(
            op_type="ADD",
            inputs=["x0_nchw", "x1_relu"],
            outputs=["sum_nchw"],
            options={"fusedActivationFunction": "NONE"},
        ),
        OperatorIR(op_type="TRANSPOSE", inputs=["sum_nchw", "post_perm"], outputs=["y_nhwc"]),
        OperatorIR(
            op_type="CONCATENATION",
            inputs=["x0_nchw", "sum_nchw"],
            outputs=["legacy_cat"],
            options={"axis": 1, "fusedActivationFunction": "NONE"},
        ),
    ]

    stats = _optimize_transpose_pre_add_nhwc_chains(model_ir)
    assert stats["optimized_transpose_pre_add_nhwc_chains"] == 1

    add_op = next(op for op in model_ir.operators if str(op.op_type) == "ADD")
    assert list(add_op.inputs) == ["x0_nhwc", "x1_relu"]
    assert list(model_ir.tensors["x1_relu"].shape) == [1, 6, 6, 64]
    assert any(
        str(op.op_type) == "TRANSPOSE" and list(op.outputs) == ["sum_nchw"]
        for op in model_ir.operators
    )


def test_flatbuffer_direct_transpose_pre_add_nhwc_with_leakyrelu_inputs() -> None:
    model_ir = ModelIR(name="transpose_pre_add_nhwc_with_leakyrelu_inputs_test")
    model_ir.inputs = ["a_nhwc", "b_nhwc"]
    model_ir.outputs = ["tail"]
    model_ir.tensors["a_nhwc"] = TensorIR(
        name="a_nhwc",
        dtype="FLOAT32",
        shape=[1, 8, 8, 16],
        shape_signature=[1, 8, 8, 16],
    )
    model_ir.tensors["b_nhwc"] = TensorIR(
        name="b_nhwc",
        dtype="FLOAT32",
        shape=[1, 8, 8, 16],
        shape_signature=[1, 8, 8, 16],
    )
    model_ir.tensors["pre_perm"] = TensorIR(
        name="pre_perm",
        dtype="INT32",
        shape=[4],
        shape_signature=[4],
        data=np.asarray([0, 3, 1, 2], dtype=np.int32),
        is_variable=False,
    )
    model_ir.tensors["post_perm"] = TensorIR(
        name="post_perm",
        dtype="INT32",
        shape=[4],
        shape_signature=[4],
        data=np.asarray([0, 2, 3, 1], dtype=np.int32),
        is_variable=False,
    )
    model_ir.tensors["a_nchw"] = TensorIR(
        name="a_nchw",
        dtype="FLOAT32",
        shape=[1, 16, 8, 8],
        shape_signature=[1, 16, 8, 8],
    )
    model_ir.tensors["b_nchw"] = TensorIR(
        name="b_nchw",
        dtype="FLOAT32",
        shape=[1, 16, 8, 8],
        shape_signature=[1, 16, 8, 8],
    )
    model_ir.tensors["a_lrelu"] = TensorIR(
        name="a_lrelu",
        dtype="FLOAT32",
        shape=[1, 16, 8, 8],
        shape_signature=[1, 16, 8, 8],
    )
    model_ir.tensors["b_lrelu"] = TensorIR(
        name="b_lrelu",
        dtype="FLOAT32",
        shape=[1, 16, 8, 8],
        shape_signature=[1, 16, 8, 8],
    )
    model_ir.tensors["sum_nchw"] = TensorIR(
        name="sum_nchw",
        dtype="FLOAT32",
        shape=[1, 16, 8, 8],
        shape_signature=[1, 16, 8, 8],
    )
    model_ir.tensors["z_nhwc"] = TensorIR(
        name="z_nhwc",
        dtype="FLOAT32",
        shape=[1, 8, 8, 16],
        shape_signature=[1, 8, 8, 16],
    )
    model_ir.tensors["tail"] = TensorIR(
        name="tail",
        dtype="FLOAT32",
        shape=[1, 8, 8, 16],
        shape_signature=[1, 8, 8, 16],
    )

    model_ir.operators = [
        OperatorIR(op_type="TRANSPOSE", inputs=["a_nhwc", "pre_perm"], outputs=["a_nchw"]),
        OperatorIR(op_type="TRANSPOSE", inputs=["b_nhwc", "pre_perm"], outputs=["b_nchw"]),
        OperatorIR(op_type="LEAKY_RELU", inputs=["a_nchw"], outputs=["a_lrelu"], options={"alpha": 0.1}),
        OperatorIR(op_type="LEAKY_RELU", inputs=["b_nchw"], outputs=["b_lrelu"], options={"alpha": 0.1}),
        OperatorIR(
            op_type="ADD",
            inputs=["a_lrelu", "b_lrelu"],
            outputs=["sum_nchw"],
            options={"fusedActivationFunction": "NONE"},
        ),
        OperatorIR(op_type="TRANSPOSE", inputs=["sum_nchw", "post_perm"], outputs=["z_nhwc"]),
        OperatorIR(op_type="RELU", inputs=["z_nhwc"], outputs=["tail"]),
    ]

    stats = _optimize_transpose_pre_add_nhwc_chains(model_ir)
    assert stats["optimized_transpose_pre_add_nhwc_chains"] == 1

    # Input transposes are removed and LEAKY_RELU runs in NHWC.
    assert not any(
        str(op.op_type) == "TRANSPOSE" and list(op.outputs) in (["a_nchw"], ["b_nchw"])
        for op in model_ir.operators
    )
    assert list(model_ir.tensors["a_lrelu"].shape) == [1, 8, 8, 16]
    assert list(model_ir.tensors["b_lrelu"].shape) == [1, 8, 8, 16]
    add_op = next(op for op in model_ir.operators if str(op.op_type) == "ADD")
    assert list(add_op.inputs) == ["a_lrelu", "b_lrelu"]
    # Post transpose is removed and downstream consumer uses NHWC tensor directly.
    assert not any(str(op.op_type) == "TRANSPOSE" and list(op.outputs) == ["z_nhwc"] for op in model_ir.operators)
    relu_op = next(op for op in model_ir.operators if str(op.op_type) == "RELU")
    relu_input_name = str(relu_op.inputs[0])
    assert relu_input_name in {"sum_nchw", "z_nhwc"}
    assert list(model_ir.tensors[relu_input_name].shape) == [1, 8, 8, 16]


def test_flatbuffer_direct_transpose_pre_add_nhwc_with_nested_add_const_affine() -> None:
    model_ir = ModelIR(name="transpose_pre_add_nhwc_with_nested_add_const_affine_test")
    model_ir.inputs = ["main_nhwc", "skip_nhwc"]
    model_ir.outputs = ["z"]
    model_ir.tensors["main_nhwc"] = TensorIR(
        name="main_nhwc",
        dtype="FLOAT32",
        shape=[1, 8, 8, 4],
        shape_signature=[1, 8, 8, 4],
    )
    model_ir.tensors["skip_nhwc"] = TensorIR(
        name="skip_nhwc",
        dtype="FLOAT32",
        shape=[1, 8, 8, 4],
        shape_signature=[1, 8, 8, 4],
    )
    model_ir.tensors["pre_perm"] = TensorIR(
        name="pre_perm",
        dtype="INT32",
        shape=[4],
        shape_signature=[4],
        data=np.asarray([0, 3, 1, 2], dtype=np.int32),
        is_variable=False,
    )
    model_ir.tensors["post_perm"] = TensorIR(
        name="post_perm",
        dtype="INT32",
        shape=[4],
        shape_signature=[4],
        data=np.asarray([0, 2, 3, 1], dtype=np.int32),
        is_variable=False,
    )
    model_ir.tensors["main_nchw"] = TensorIR(
        name="main_nchw",
        dtype="FLOAT32",
        shape=[1, 4, 8, 8],
        shape_signature=[1, 4, 8, 8],
    )
    model_ir.tensors["skip_nchw"] = TensorIR(
        name="skip_nchw",
        dtype="FLOAT32",
        shape=[1, 4, 8, 8],
        shape_signature=[1, 4, 8, 8],
    )
    model_ir.tensors["gamma_const"] = TensorIR(
        name="gamma_const",
        dtype="FLOAT32",
        shape=[1, 4, 1, 1],
        shape_signature=[1, 4, 1, 1],
        data=np.ones((1, 4, 1, 1), dtype=np.float32),
        is_variable=False,
    )
    model_ir.tensors["beta_const"] = TensorIR(
        name="beta_const",
        dtype="FLOAT32",
        shape=[1, 4, 1, 1],
        shape_signature=[1, 4, 1, 1],
        data=np.zeros((1, 4, 1, 1), dtype=np.float32),
        is_variable=False,
    )
    model_ir.tensors["skip_mul"] = TensorIR(
        name="skip_mul",
        dtype="FLOAT32",
        shape=[1, 4, 8, 8],
        shape_signature=[1, 4, 8, 8],
    )
    model_ir.tensors["skip_affine_nchw"] = TensorIR(
        name="skip_affine_nchw",
        dtype="FLOAT32",
        shape=[1, 4, 8, 8],
        shape_signature=[1, 4, 8, 8],
    )
    model_ir.tensors["sum_nchw"] = TensorIR(
        name="sum_nchw",
        dtype="FLOAT32",
        shape=[1, 4, 8, 8],
        shape_signature=[1, 4, 8, 8],
    )
    model_ir.tensors["y_nhwc"] = TensorIR(
        name="y_nhwc",
        dtype="FLOAT32",
        shape=[1, 8, 8, 4],
        shape_signature=[1, 8, 8, 4],
    )
    model_ir.tensors["z"] = TensorIR(
        name="z",
        dtype="FLOAT32",
        shape=[1, 8, 8, 4],
        shape_signature=[1, 8, 8, 4],
    )

    model_ir.operators = [
        OperatorIR(op_type="TRANSPOSE", inputs=["skip_nhwc", "pre_perm"], outputs=["skip_nchw"]),
        OperatorIR(
            op_type="MUL",
            inputs=["skip_nchw", "gamma_const"],
            outputs=["skip_mul"],
            options={"fusedActivationFunction": "NONE"},
        ),
        OperatorIR(
            op_type="ADD",
            inputs=["skip_mul", "beta_const"],
            outputs=["skip_affine_nchw"],
            options={"fusedActivationFunction": "NONE"},
        ),
        OperatorIR(op_type="TRANSPOSE", inputs=["main_nhwc", "pre_perm"], outputs=["main_nchw"]),
        OperatorIR(
            op_type="ADD",
            inputs=["main_nchw", "skip_affine_nchw"],
            outputs=["sum_nchw"],
            options={"fusedActivationFunction": "NONE"},
        ),
        OperatorIR(op_type="TRANSPOSE", inputs=["sum_nchw", "post_perm"], outputs=["y_nhwc"]),
        OperatorIR(op_type="RELU", inputs=["y_nhwc"], outputs=["z"]),
    ]

    stats = _optimize_transpose_pre_add_nhwc_chains(model_ir)
    assert stats["optimized_transpose_pre_add_nhwc_chains"] == 1

    assert not any(str(op.op_type) == "TRANSPOSE" for op in model_ir.operators)
    top_add_op = next(
        op
        for op in model_ir.operators
        if str(op.op_type) == "ADD" and "main_nhwc" in list(op.inputs)
    )
    assert list(top_add_op.inputs) == ["main_nhwc", "skip_affine_nchw"]
    nested_add_op = next(
        op
        for op in model_ir.operators
        if str(op.op_type) == "ADD" and list(op.outputs) == ["skip_affine_nchw"]
    )
    assert list(nested_add_op.inputs) == ["skip_mul", "beta_const"]
    assert list(model_ir.tensors["gamma_const"].shape) == [1, 1, 1, 4]
    assert list(model_ir.tensors["beta_const"].shape) == [1, 1, 1, 4]
    assert list(model_ir.tensors["skip_mul"].shape) == [1, 8, 8, 4]
    assert list(model_ir.tensors["skip_affine_nchw"].shape) == [1, 8, 8, 4]


def test_flatbuffer_direct_transpose_pre_concat_nhwc_shared_direct_and_nested_add_optimized() -> None:
    model_ir = ModelIR(name="transpose_pre_concat_nhwc_shared_direct_nested_add_test")
    model_ir.inputs = ["a_nhwc", "b_nhwc", "c_nhwc", "d_nhwc"]
    model_ir.outputs = ["z"]
    model_ir.tensors["a_nhwc"] = TensorIR(
        name="a_nhwc",
        dtype="FLOAT32",
        shape=[1, 6, 6, 32],
        shape_signature=[1, 6, 6, 32],
    )
    model_ir.tensors["b_nhwc"] = TensorIR(
        name="b_nhwc",
        dtype="FLOAT32",
        shape=[1, 6, 6, 32],
        shape_signature=[1, 6, 6, 32],
    )
    model_ir.tensors["c_nhwc"] = TensorIR(
        name="c_nhwc",
        dtype="FLOAT32",
        shape=[1, 6, 6, 32],
        shape_signature=[1, 6, 6, 32],
    )
    model_ir.tensors["d_nhwc"] = TensorIR(
        name="d_nhwc",
        dtype="FLOAT32",
        shape=[1, 6, 6, 32],
        shape_signature=[1, 6, 6, 32],
    )
    model_ir.tensors["pre_perm"] = TensorIR(
        name="pre_perm",
        dtype="INT32",
        shape=[4],
        shape_signature=[4],
        data=np.asarray([0, 3, 1, 2], dtype=np.int32),
        is_variable=False,
    )
    model_ir.tensors["post_perm"] = TensorIR(
        name="post_perm",
        dtype="INT32",
        shape=[4],
        shape_signature=[4],
        data=np.asarray([0, 2, 3, 1], dtype=np.int32),
        is_variable=False,
    )
    model_ir.tensors["a_nchw"] = TensorIR(
        name="a_nchw",
        dtype="FLOAT32",
        shape=[1, 32, 6, 6],
        shape_signature=[1, 32, 6, 6],
    )
    model_ir.tensors["a_relu"] = TensorIR(
        name="a_relu",
        dtype="FLOAT32",
        shape=[1, 32, 6, 6],
        shape_signature=[1, 32, 6, 6],
    )
    model_ir.tensors["b_nchw"] = TensorIR(
        name="b_nchw",
        dtype="FLOAT32",
        shape=[1, 32, 6, 6],
        shape_signature=[1, 32, 6, 6],
    )
    model_ir.tensors["c_nchw"] = TensorIR(
        name="c_nchw",
        dtype="FLOAT32",
        shape=[1, 32, 6, 6],
        shape_signature=[1, 32, 6, 6],
    )
    model_ir.tensors["d_nchw"] = TensorIR(
        name="d_nchw",
        dtype="FLOAT32",
        shape=[1, 32, 6, 6],
        shape_signature=[1, 32, 6, 6],
    )
    model_ir.tensors["d_relu"] = TensorIR(
        name="d_relu",
        dtype="FLOAT32",
        shape=[1, 32, 6, 6],
        shape_signature=[1, 32, 6, 6],
    )
    model_ir.tensors["sum_nchw"] = TensorIR(
        name="sum_nchw",
        dtype="FLOAT32",
        shape=[1, 32, 6, 6],
        shape_signature=[1, 32, 6, 6],
    )
    model_ir.tensors["cat_nchw"] = TensorIR(
        name="cat_nchw",
        dtype="FLOAT32",
        shape=[1, 128, 6, 6],
        shape_signature=[1, 128, 6, 6],
    )
    model_ir.tensors["cat_nhwc"] = TensorIR(
        name="cat_nhwc",
        dtype="FLOAT32",
        shape=[1, 6, 6, 128],
        shape_signature=[1, 6, 6, 128],
    )
    model_ir.tensors["z"] = TensorIR(
        name="z",
        dtype="FLOAT32",
        shape=[1, 6, 6, 128],
        shape_signature=[1, 6, 6, 128],
    )
    model_ir.operators = [
        OperatorIR(op_type="TRANSPOSE", inputs=["a_nhwc", "pre_perm"], outputs=["a_nchw"]),
        OperatorIR(op_type="RELU", inputs=["a_nchw"], outputs=["a_relu"]),
        OperatorIR(op_type="TRANSPOSE", inputs=["b_nhwc", "pre_perm"], outputs=["b_nchw"]),
        OperatorIR(op_type="TRANSPOSE", inputs=["c_nhwc", "pre_perm"], outputs=["c_nchw"]),
        OperatorIR(op_type="TRANSPOSE", inputs=["d_nhwc", "pre_perm"], outputs=["d_nchw"]),
        OperatorIR(op_type="RELU", inputs=["d_nchw"], outputs=["d_relu"]),
        OperatorIR(
            op_type="ADD",
            inputs=["c_nchw", "d_relu"],
            outputs=["sum_nchw"],
            options={"fusedActivationFunction": "NONE"},
        ),
        OperatorIR(
            op_type="CONCATENATION",
            inputs=["a_relu", "b_nchw", "c_nchw", "sum_nchw"],
            outputs=["cat_nchw"],
            options={"axis": 1, "fusedActivationFunction": "NONE"},
        ),
        OperatorIR(op_type="TRANSPOSE", inputs=["cat_nchw", "post_perm"], outputs=["cat_nhwc"]),
        OperatorIR(op_type="RELU", inputs=["cat_nhwc"], outputs=["z"]),
    ]

    stats = _optimize_transpose_pre_concat_nhwc_chains(model_ir)
    assert stats["optimized_transpose_pre_concat_nhwc_chains"] == 1

    concat_op = next(op for op in model_ir.operators if str(op.op_type) == "CONCATENATION")
    assert int(concat_op.options.get("axis", -1)) == 3
    assert list(concat_op.inputs) == ["a_relu", "b_nhwc", "c_nhwc", "sum_nchw"]
    assert list(model_ir.tensors["sum_nchw"].shape) == [1, 6, 6, 32]

    transpose_ops = [op for op in model_ir.operators if str(op.op_type) == "TRANSPOSE"]
    assert len(transpose_ops) == 1
    remaining_transpose = transpose_ops[0]
    assert list(remaining_transpose.outputs) == ["c_nchw"]
    all_inputs = [str(v) for op in model_ir.operators for v in list(op.inputs)]
    assert "c_nchw" not in all_inputs


def test_flatbuffer_direct_transpose_pre_concat_nhwc_pad_input_optimized() -> None:
    model_ir = ModelIR(name="transpose_pre_concat_nhwc_pad_input_opt_test")
    model_ir.inputs = ["a_nhwc", "b_nhwc"]
    model_ir.outputs = ["y"]
    model_ir.tensors["a_nhwc"] = TensorIR(
        name="a_nhwc",
        dtype="FLOAT32",
        shape=[1, 4, 5, 8],
        shape_signature=[1, 4, 5, 8],
    )
    model_ir.tensors["b_nhwc"] = TensorIR(
        name="b_nhwc",
        dtype="FLOAT32",
        shape=[1, 4, 4, 8],
        shape_signature=[1, 4, 4, 8],
    )
    model_ir.tensors["pre_perm"] = TensorIR(
        name="pre_perm",
        dtype="INT32",
        shape=[4],
        shape_signature=[4],
        data=np.asarray([0, 3, 1, 2], dtype=np.int32),
        is_variable=False,
    )
    model_ir.tensors["post_perm"] = TensorIR(
        name="post_perm",
        dtype="INT32",
        shape=[4],
        shape_signature=[4],
        data=np.asarray([0, 2, 3, 1], dtype=np.int32),
        is_variable=False,
    )
    model_ir.tensors["pad_nchw_pads"] = TensorIR(
        name="pad_nchw_pads",
        dtype="INT32",
        shape=[4, 2],
        shape_signature=[4, 2],
        data=np.asarray(
            [
                [0, 0],  # N
                [0, 0],  # C
                [0, 0],  # H
                [0, 1],  # W
            ],
            dtype=np.int32,
        ),
        is_variable=False,
    )
    model_ir.tensors["a_nchw"] = TensorIR(
        name="a_nchw",
        dtype="FLOAT32",
        shape=[1, 8, 4, 5],
        shape_signature=[1, 8, 4, 5],
    )
    model_ir.tensors["b_nchw"] = TensorIR(
        name="b_nchw",
        dtype="FLOAT32",
        shape=[1, 8, 4, 4],
        shape_signature=[1, 8, 4, 4],
    )
    model_ir.tensors["b_pad_nchw"] = TensorIR(
        name="b_pad_nchw",
        dtype="FLOAT32",
        shape=[1, 8, 4, 5],
        shape_signature=[1, 8, 4, 5],
    )
    model_ir.tensors["cat_nchw"] = TensorIR(
        name="cat_nchw",
        dtype="FLOAT32",
        shape=[1, 16, 4, 5],
        shape_signature=[1, 16, 4, 5],
    )
    model_ir.tensors["cat_nhwc"] = TensorIR(
        name="cat_nhwc",
        dtype="FLOAT32",
        shape=[1, 4, 5, 16],
        shape_signature=[1, 4, 5, 16],
    )
    model_ir.tensors["y"] = TensorIR(
        name="y",
        dtype="FLOAT32",
        shape=[1, 4, 5, 16],
        shape_signature=[1, 4, 5, 16],
    )
    model_ir.operators = [
        OperatorIR(op_type="TRANSPOSE", inputs=["a_nhwc", "pre_perm"], outputs=["a_nchw"]),
        OperatorIR(op_type="TRANSPOSE", inputs=["b_nhwc", "pre_perm"], outputs=["b_nchw"]),
        OperatorIR(op_type="PAD", inputs=["b_nchw", "pad_nchw_pads"], outputs=["b_pad_nchw"]),
        OperatorIR(
            op_type="CONCATENATION",
            inputs=["a_nchw", "b_pad_nchw"],
            outputs=["cat_nchw"],
            options={"axis": 1, "fusedActivationFunction": "NONE"},
        ),
        OperatorIR(op_type="TRANSPOSE", inputs=["cat_nchw", "post_perm"], outputs=["cat_nhwc"]),
        OperatorIR(op_type="RELU", inputs=["cat_nhwc"], outputs=["y"]),
    ]

    stats = _optimize_transpose_pre_concat_nhwc_chains(model_ir)
    assert stats["optimized_transpose_pre_concat_nhwc_chains"] == 1

    op_types = [str(op.op_type) for op in model_ir.operators]
    assert op_types.count("TRANSPOSE") == 0

    pad_op = next(op for op in model_ir.operators if str(op.op_type) == "PAD")
    assert list(pad_op.inputs) == ["b_nhwc", "pad_nchw_pads"]
    assert np.array_equal(
        np.asarray(model_ir.tensors["pad_nchw_pads"].data),
        np.asarray(
            [
                [0, 0],  # N
                [0, 0],  # H
                [0, 1],  # W
                [0, 0],  # C
            ],
            dtype=np.int32,
        ),
    )
    assert list(model_ir.tensors["b_pad_nchw"].shape) == [1, 4, 5, 8]

    concat_op = next(op for op in model_ir.operators if str(op.op_type) == "CONCATENATION")
    assert int(concat_op.options.get("axis", -1)) == 3
    assert list(concat_op.inputs) == ["a_nhwc", "b_pad_nchw"]
    assert list(concat_op.outputs) == ["cat_nhwc"]


def test_flatbuffer_direct_transpose_axis3_const_concat_bridge_nhwc_chain_optimized() -> None:
    model_ir = ModelIR(name="transpose_axis3_const_concat_bridge_nhwc_opt_test")
    model_ir.inputs = ["stem_nhwc"]
    model_ir.outputs = ["y_nhwc", "mean_nchw"]

    model_ir.tensors["stem_nhwc"] = TensorIR(
        name="stem_nhwc",
        dtype="FLOAT32",
        shape=[1, 1, 96, 64],
        shape_signature=[1, 1, 96, 64],
    )
    model_ir.tensors["perm_nhwc_to_nchw"] = TensorIR(
        name="perm_nhwc_to_nchw",
        dtype="INT32",
        shape=[4],
        shape_signature=[4],
        data=np.asarray([0, 3, 1, 2], dtype=np.int32),
        is_variable=False,
    )
    model_ir.tensors["perm_nchw_to_nhwc"] = TensorIR(
        name="perm_nchw_to_nhwc",
        dtype="INT32",
        shape=[4],
        shape_signature=[4],
        data=np.asarray([0, 2, 3, 1], dtype=np.int32),
        is_variable=False,
    )
    model_ir.tensors["mean_axis_nchw"] = TensorIR(
        name="mean_axis_nchw",
        dtype="INT32",
        shape=[1],
        shape_signature=[1],
        data=np.asarray([1], dtype=np.int32),
        is_variable=False,
    )
    const_nchw = np.arange(64, dtype=np.float32).reshape(1, 64, 1, 1)
    model_ir.tensors["const_token_nchw"] = TensorIR(
        name="const_token_nchw",
        dtype="FLOAT32",
        shape=[1, 64, 1, 1],
        shape_signature=[1, 64, 1, 1],
        data=np.asarray(const_nchw),
        is_variable=False,
    )
    model_ir.tensors["stem_nchw"] = TensorIR(
        name="stem_nchw",
        dtype="FLOAT32",
        shape=[1, 64, 1, 96],
        shape_signature=[1, 64, 1, 96],
    )
    model_ir.tensors["concat_nchw"] = TensorIR(
        name="concat_nchw",
        dtype="FLOAT32",
        shape=[1, 64, 1, 97],
        shape_signature=[1, 64, 1, 97],
    )
    model_ir.tensors["concat_nhwc"] = TensorIR(
        name="concat_nhwc",
        dtype="FLOAT32",
        shape=[1, 1, 97, 64],
        shape_signature=[1, 1, 97, 64],
    )
    model_ir.tensors["mean_nchw"] = TensorIR(
        name="mean_nchw",
        dtype="FLOAT32",
        shape=[1, 1, 1, 97],
        shape_signature=[1, 1, 1, 97],
    )
    model_ir.tensors["y_nhwc"] = TensorIR(
        name="y_nhwc",
        dtype="FLOAT32",
        shape=[1, 1, 97, 64],
        shape_signature=[1, 1, 97, 64],
    )
    model_ir.operators = [
        OperatorIR(
            op_type="TRANSPOSE",
            inputs=["stem_nhwc", "perm_nhwc_to_nchw"],
            outputs=["stem_nchw"],
        ),
        OperatorIR(
            op_type="CONCATENATION",
            inputs=["const_token_nchw", "stem_nchw"],
            outputs=["concat_nchw"],
            options={"axis": 3, "fusedActivationFunction": "NONE"},
        ),
        OperatorIR(
            op_type="TRANSPOSE",
            inputs=["concat_nchw", "perm_nchw_to_nhwc"],
            outputs=["concat_nhwc"],
        ),
        OperatorIR(
            op_type="MEAN",
            inputs=["concat_nchw", "mean_axis_nchw"],
            outputs=["mean_nchw"],
            options={"keepDims": True},
        ),
        OperatorIR(
            op_type="RELU",
            inputs=["concat_nhwc"],
            outputs=["y_nhwc"],
        ),
    ]

    stats = _optimize_transpose_axis3_const_concat_bridge_nhwc_chains(model_ir)
    assert stats["optimized_transpose_axis3_const_concat_bridge_nhwc_chains"] == 1

    op_types = [str(op.op_type) for op in model_ir.operators]
    assert op_types.count("TRANSPOSE") == 1
    assert op_types.count("CONCATENATION") == 1

    concat_op = next(op for op in model_ir.operators if str(op.op_type) == "CONCATENATION")
    assert int(concat_op.options.get("axis", -1)) == 2
    assert list(concat_op.inputs) == ["const_token_nchw", "stem_nhwc"]
    assert list(model_ir.tensors["concat_nchw"].shape) == [1, 1, 97, 64]

    const_converted = np.asarray(model_ir.tensors["const_token_nchw"].data)
    assert list(const_converted.shape) == [1, 1, 1, 64]
    assert np.array_equal(const_converted, np.transpose(const_nchw, (0, 2, 3, 1)))

    relu_op = next(op for op in model_ir.operators if str(op.op_type) == "RELU")
    assert list(relu_op.inputs) == ["concat_nchw"]

    bridge_transpose = next(op for op in model_ir.operators if str(op.op_type) == "TRANSPOSE")
    bridge_name = str(list(bridge_transpose.outputs)[0])
    assert str(list(bridge_transpose.inputs)[0]) == "concat_nchw"
    assert str(list(bridge_transpose.inputs)[1]) == "perm_nhwc_to_nchw"
    mean_op = next(op for op in model_ir.operators if str(op.op_type) == "MEAN")
    assert str(list(mean_op.inputs)[0]) == bridge_name


def test_flatbuffer_direct_transpose_elementwise_concat_conv_nhwc_groups_optimized() -> None:
    model_ir = ModelIR(name="transpose_elementwise_concat_conv_nhwc_groups_opt_test")
    model_ir.inputs = ["a_nhwc", "b_nhwc", "c_nhwc"]
    model_ir.outputs = ["y0", "y1"]
    model_ir.tensors["a_nhwc"] = TensorIR(
        name="a_nhwc",
        dtype="FLOAT32",
        shape=[1, 6, 6, 16],
        shape_signature=[1, 6, 6, 16],
    )
    model_ir.tensors["b_nhwc"] = TensorIR(
        name="b_nhwc",
        dtype="FLOAT32",
        shape=[1, 6, 6, 16],
        shape_signature=[1, 6, 6, 16],
    )
    model_ir.tensors["c_nhwc"] = TensorIR(
        name="c_nhwc",
        dtype="FLOAT32",
        shape=[1, 6, 6, 16],
        shape_signature=[1, 6, 6, 16],
    )
    model_ir.tensors["pre_perm"] = TensorIR(
        name="pre_perm",
        dtype="INT32",
        shape=[4],
        shape_signature=[4],
        data=np.asarray([0, 3, 1, 2], dtype=np.int32),
        is_variable=False,
    )
    model_ir.tensors["post_perm"] = TensorIR(
        name="post_perm",
        dtype="INT32",
        shape=[4],
        shape_signature=[4],
        data=np.asarray([0, 2, 3, 1], dtype=np.int32),
        is_variable=False,
    )
    model_ir.tensors["scale_nchw"] = TensorIR(
        name="scale_nchw",
        dtype="FLOAT32",
        shape=[1, 16, 1, 1],
        shape_signature=[1, 16, 1, 1],
        data=np.ones((1, 16, 1, 1), dtype=np.float32),
        is_variable=False,
    )
    model_ir.tensors["a_nchw"] = TensorIR(
        name="a_nchw",
        dtype="FLOAT32",
        shape=[1, 16, 6, 6],
        shape_signature=[1, 16, 6, 6],
    )
    model_ir.tensors["b_nchw"] = TensorIR(
        name="b_nchw",
        dtype="FLOAT32",
        shape=[1, 16, 6, 6],
        shape_signature=[1, 16, 6, 6],
    )
    model_ir.tensors["c_nchw"] = TensorIR(
        name="c_nchw",
        dtype="FLOAT32",
        shape=[1, 16, 6, 6],
        shape_signature=[1, 16, 6, 6],
    )
    model_ir.tensors["mul_out"] = TensorIR(
        name="mul_out",
        dtype="FLOAT32",
        shape=[1, 16, 6, 6],
        shape_signature=[1, 16, 6, 6],
    )
    model_ir.tensors["sum_out"] = TensorIR(
        name="sum_out",
        dtype="FLOAT32",
        shape=[1, 16, 6, 6],
        shape_signature=[1, 16, 6, 6],
    )
    model_ir.tensors["cat0_nchw"] = TensorIR(
        name="cat0_nchw",
        dtype="FLOAT32",
        shape=[1, 32, 6, 6],
        shape_signature=[1, 32, 6, 6],
    )
    model_ir.tensors["cat1_nchw"] = TensorIR(
        name="cat1_nchw",
        dtype="FLOAT32",
        shape=[1, 32, 6, 6],
        shape_signature=[1, 32, 6, 6],
    )
    model_ir.tensors["cat0_nhwc"] = TensorIR(
        name="cat0_nhwc",
        dtype="FLOAT32",
        shape=[1, 6, 6, 32],
        shape_signature=[1, 6, 6, 32],
    )
    model_ir.tensors["cat1_nhwc"] = TensorIR(
        name="cat1_nhwc",
        dtype="FLOAT32",
        shape=[1, 6, 6, 32],
        shape_signature=[1, 6, 6, 32],
    )
    model_ir.tensors["conv0_w"] = TensorIR(
        name="conv0_w",
        dtype="FLOAT32",
        shape=[8, 1, 1, 32],
        shape_signature=[8, 1, 1, 32],
        data=np.ones((8, 1, 1, 32), dtype=np.float32),
        is_variable=False,
    )
    model_ir.tensors["conv0_b"] = TensorIR(
        name="conv0_b",
        dtype="FLOAT32",
        shape=[8],
        shape_signature=[8],
        data=np.zeros((8,), dtype=np.float32),
        is_variable=False,
    )
    model_ir.tensors["conv1_w"] = TensorIR(
        name="conv1_w",
        dtype="FLOAT32",
        shape=[8, 1, 1, 32],
        shape_signature=[8, 1, 1, 32],
        data=np.ones((8, 1, 1, 32), dtype=np.float32),
        is_variable=False,
    )
    model_ir.tensors["conv1_b"] = TensorIR(
        name="conv1_b",
        dtype="FLOAT32",
        shape=[8],
        shape_signature=[8],
        data=np.zeros((8,), dtype=np.float32),
        is_variable=False,
    )
    model_ir.tensors["y0"] = TensorIR(
        name="y0",
        dtype="FLOAT32",
        shape=[1, 6, 6, 8],
        shape_signature=[1, 6, 6, 8],
    )
    model_ir.tensors["y1"] = TensorIR(
        name="y1",
        dtype="FLOAT32",
        shape=[1, 6, 6, 8],
        shape_signature=[1, 6, 6, 8],
    )
    conv_options = {
        "padding": "SAME",
        "strideH": 1,
        "strideW": 1,
        "dilationHFactor": 1,
        "dilationWFactor": 1,
        "fusedActivationFunction": "NONE",
    }
    model_ir.operators = [
        OperatorIR(op_type="TRANSPOSE", inputs=["a_nhwc", "pre_perm"], outputs=["a_nchw"]),
        OperatorIR(op_type="TRANSPOSE", inputs=["b_nhwc", "pre_perm"], outputs=["b_nchw"]),
        OperatorIR(op_type="TRANSPOSE", inputs=["c_nhwc", "pre_perm"], outputs=["c_nchw"]),
        OperatorIR(
            op_type="MUL",
            inputs=["a_nchw", "scale_nchw"],
            outputs=["mul_out"],
            options={"fusedActivationFunction": "NONE"},
        ),
        OperatorIR(
            op_type="ADD",
            inputs=["mul_out", "b_nchw"],
            outputs=["sum_out"],
            options={"fusedActivationFunction": "NONE"},
        ),
        OperatorIR(
            op_type="CONCATENATION",
            inputs=["sum_out", "c_nchw"],
            outputs=["cat0_nchw"],
            options={"axis": 1, "fusedActivationFunction": "NONE"},
        ),
        OperatorIR(op_type="TRANSPOSE", inputs=["cat0_nchw", "post_perm"], outputs=["cat0_nhwc"]),
        OperatorIR(op_type="CONV_2D", inputs=["cat0_nhwc", "conv0_w", "conv0_b"], outputs=["y0"], options=conv_options),
        OperatorIR(
            op_type="CONCATENATION",
            inputs=["sum_out", "b_nchw"],
            outputs=["cat1_nchw"],
            options={"axis": 1, "fusedActivationFunction": "NONE"},
        ),
        OperatorIR(op_type="TRANSPOSE", inputs=["cat1_nchw", "post_perm"], outputs=["cat1_nhwc"]),
        OperatorIR(op_type="CONV_2D", inputs=["cat1_nhwc", "conv1_w", "conv1_b"], outputs=["y1"], options=conv_options),
    ]

    stats = _optimize_transpose_elementwise_concat_conv_nhwc_groups(model_ir)
    assert stats["optimized_transpose_elementwise_concat_conv_nhwc_groups"] == 1

    op_types = [str(op.op_type) for op in model_ir.operators]
    assert op_types.count("TRANSPOSE") == 0
    concat_ops = [op for op in model_ir.operators if str(op.op_type) == "CONCATENATION"]
    assert len(concat_ops) == 2
    assert all(int(op.options.get("axis", -1)) == 3 for op in concat_ops)
    assert list(model_ir.tensors["scale_nchw"].shape) == [1, 1, 1, 16]
    conv_ops = [op for op in model_ir.operators if str(op.op_type) == "CONV_2D"]
    assert list(conv_ops[0].inputs)[0] == "cat0_nhwc"
    assert list(conv_ops[1].inputs)[0] == "cat1_nhwc"


def test_flatbuffer_direct_transpose_leakyrelu_concat_conv_nhwc_groups_optimized() -> None:
    model_ir = ModelIR(name="transpose_leakyrelu_concat_conv_nhwc_groups_opt_test")
    model_ir.inputs = ["a_nhwc", "b_nhwc"]
    model_ir.outputs = ["y"]
    model_ir.tensors["a_nhwc"] = TensorIR(
        name="a_nhwc",
        dtype="FLOAT32",
        shape=[1, 6, 6, 16],
        shape_signature=[1, 6, 6, 16],
    )
    model_ir.tensors["b_nhwc"] = TensorIR(
        name="b_nhwc",
        dtype="FLOAT32",
        shape=[1, 6, 6, 16],
        shape_signature=[1, 6, 6, 16],
    )
    model_ir.tensors["pre_perm"] = TensorIR(
        name="pre_perm",
        dtype="INT32",
        shape=[4],
        shape_signature=[4],
        data=np.asarray([0, 3, 1, 2], dtype=np.int32),
        is_variable=False,
    )
    model_ir.tensors["post_perm"] = TensorIR(
        name="post_perm",
        dtype="INT32",
        shape=[4],
        shape_signature=[4],
        data=np.asarray([0, 2, 3, 1], dtype=np.int32),
        is_variable=False,
    )
    model_ir.tensors["a_nchw"] = TensorIR(
        name="a_nchw",
        dtype="FLOAT32",
        shape=[1, 16, 6, 6],
        shape_signature=[1, 16, 6, 6],
    )
    model_ir.tensors["b_nchw"] = TensorIR(
        name="b_nchw",
        dtype="FLOAT32",
        shape=[1, 16, 6, 6],
        shape_signature=[1, 16, 6, 6],
    )
    model_ir.tensors["a_act_nchw"] = TensorIR(
        name="a_act_nchw",
        dtype="FLOAT32",
        shape=[1, 16, 6, 6],
        shape_signature=[1, 16, 6, 6],
    )
    model_ir.tensors["b_act_nchw"] = TensorIR(
        name="b_act_nchw",
        dtype="FLOAT32",
        shape=[1, 16, 6, 6],
        shape_signature=[1, 16, 6, 6],
    )
    model_ir.tensors["cat_nchw"] = TensorIR(
        name="cat_nchw",
        dtype="FLOAT32",
        shape=[1, 32, 6, 6],
        shape_signature=[1, 32, 6, 6],
    )
    model_ir.tensors["cat_nhwc"] = TensorIR(
        name="cat_nhwc",
        dtype="FLOAT32",
        shape=[1, 6, 6, 32],
        shape_signature=[1, 6, 6, 32],
    )
    model_ir.tensors["conv_w"] = TensorIR(
        name="conv_w",
        dtype="FLOAT32",
        shape=[8, 1, 1, 32],
        shape_signature=[8, 1, 1, 32],
        data=np.ones((8, 1, 1, 32), dtype=np.float32),
        is_variable=False,
    )
    model_ir.tensors["conv_b"] = TensorIR(
        name="conv_b",
        dtype="FLOAT32",
        shape=[8],
        shape_signature=[8],
        data=np.zeros((8,), dtype=np.float32),
        is_variable=False,
    )
    model_ir.tensors["y"] = TensorIR(
        name="y",
        dtype="FLOAT32",
        shape=[1, 6, 6, 8],
        shape_signature=[1, 6, 6, 8],
    )
    conv_options = {
        "padding": "SAME",
        "strideH": 1,
        "strideW": 1,
        "dilationHFactor": 1,
        "dilationWFactor": 1,
        "fusedActivationFunction": "NONE",
    }
    model_ir.operators = [
        OperatorIR(op_type="TRANSPOSE", inputs=["a_nhwc", "pre_perm"], outputs=["a_nchw"]),
        OperatorIR(op_type="TRANSPOSE", inputs=["b_nhwc", "pre_perm"], outputs=["b_nchw"]),
        OperatorIR(op_type="LEAKY_RELU", inputs=["a_nchw"], outputs=["a_act_nchw"], options={"alpha": 0.125}),
        OperatorIR(op_type="LEAKY_RELU", inputs=["b_nchw"], outputs=["b_act_nchw"], options={"alpha": 0.125}),
        OperatorIR(
            op_type="CONCATENATION",
            inputs=["a_act_nchw", "b_act_nchw"],
            outputs=["cat_nchw"],
            options={"axis": 1, "fusedActivationFunction": "NONE"},
        ),
        OperatorIR(op_type="TRANSPOSE", inputs=["cat_nchw", "post_perm"], outputs=["cat_nhwc"]),
        OperatorIR(op_type="CONV_2D", inputs=["cat_nhwc", "conv_w", "conv_b"], outputs=["y"], options=conv_options),
    ]

    stats = _optimize_transpose_elementwise_concat_conv_nhwc_groups(model_ir)
    assert stats["optimized_transpose_elementwise_concat_conv_nhwc_groups"] == 1
    assert [str(op.op_type) for op in model_ir.operators].count("TRANSPOSE") == 0
    concat_op = next(op for op in model_ir.operators if str(op.op_type) == "CONCATENATION")
    assert int(concat_op.options.get("axis", -1)) == 3
    leaky_ops = [op for op in model_ir.operators if str(op.op_type) == "LEAKY_RELU"]
    assert [str(v) for v in list(leaky_ops[0].inputs)] == ["a_nhwc"]
    assert [str(v) for v in list(leaky_ops[1].inputs)] == ["b_nhwc"]


def test_flatbuffer_direct_transpose_conv_attention_nhwc_propagation_optimized() -> None:
    model_ir = ModelIR(name="transpose_conv_attention_nhwc_propagation_opt_test")
    model_ir.inputs = ["x_nhwc"]
    model_ir.outputs = ["z"]
    model_ir.tensors["x_nhwc"] = TensorIR(
        name="x_nhwc",
        dtype="FLOAT32",
        shape=[1, 8, 8, 16],
        shape_signature=[1, 8, 8, 16],
    )
    model_ir.tensors["pre_perm"] = TensorIR(
        name="pre_perm",
        dtype="INT32",
        shape=[4],
        shape_signature=[4],
        data=np.asarray([0, 3, 1, 2], dtype=np.int32),
        is_variable=False,
    )
    model_ir.tensors["post_perm"] = TensorIR(
        name="post_perm",
        dtype="INT32",
        shape=[4],
        shape_signature=[4],
        data=np.asarray([0, 2, 3, 1], dtype=np.int32),
        is_variable=False,
    )
    model_ir.tensors["mean_axes"] = TensorIR(
        name="mean_axes",
        dtype="INT32",
        shape=[2],
        shape_signature=[2],
        data=np.asarray([2, 3], dtype=np.int32),
        is_variable=False,
    )
    model_ir.tensors["x_nchw"] = TensorIR(
        name="x_nchw",
        dtype="FLOAT32",
        shape=[1, 16, 8, 8],
        shape_signature=[1, 16, 8, 8],
    )
    model_ir.tensors["a_nchw"] = TensorIR(
        name="a_nchw",
        dtype="FLOAT32",
        shape=[1, 16, 8, 8],
        shape_signature=[1, 16, 8, 8],
    )
    model_ir.tensors["mean_nchw"] = TensorIR(
        name="mean_nchw",
        dtype="FLOAT32",
        shape=[1, 16, 1, 1],
        shape_signature=[1, 16, 1, 1],
    )
    model_ir.tensors["mean_nhwc"] = TensorIR(
        name="mean_nhwc",
        dtype="FLOAT32",
        shape=[1, 1, 1, 16],
        shape_signature=[1, 1, 1, 16],
    )
    model_ir.tensors["conv1_w"] = TensorIR(
        name="conv1_w",
        dtype="FLOAT32",
        shape=[8, 1, 1, 16],
        shape_signature=[8, 1, 1, 16],
        data=np.ones((8, 1, 1, 16), dtype=np.float32),
        is_variable=False,
    )
    model_ir.tensors["conv1_b"] = TensorIR(
        name="conv1_b",
        dtype="FLOAT32",
        shape=[8],
        shape_signature=[8],
        data=np.zeros((8,), dtype=np.float32),
        is_variable=False,
    )
    model_ir.tensors["conv1_out"] = TensorIR(
        name="conv1_out",
        dtype="FLOAT32",
        shape=[1, 1, 1, 8],
        shape_signature=[1, 1, 1, 8],
    )
    model_ir.tensors["conv2_w"] = TensorIR(
        name="conv2_w",
        dtype="FLOAT32",
        shape=[16, 1, 1, 8],
        shape_signature=[16, 1, 1, 8],
        data=np.ones((16, 1, 1, 8), dtype=np.float32),
        is_variable=False,
    )
    model_ir.tensors["conv2_b"] = TensorIR(
        name="conv2_b",
        dtype="FLOAT32",
        shape=[16],
        shape_signature=[16],
        data=np.zeros((16,), dtype=np.float32),
        is_variable=False,
    )
    model_ir.tensors["conv2_out_nhwc"] = TensorIR(
        name="conv2_out_nhwc",
        dtype="FLOAT32",
        shape=[1, 1, 1, 16],
        shape_signature=[1, 1, 1, 16],
    )
    model_ir.tensors["sig_in_nchw"] = TensorIR(
        name="sig_in_nchw",
        dtype="FLOAT32",
        shape=[1, 16, 1, 1],
        shape_signature=[1, 16, 1, 1],
    )
    model_ir.tensors["sig_nchw"] = TensorIR(
        name="sig_nchw",
        dtype="FLOAT32",
        shape=[1, 16, 1, 1],
        shape_signature=[1, 16, 1, 1],
    )
    model_ir.tensors["mul_nchw"] = TensorIR(
        name="mul_nchw",
        dtype="FLOAT32",
        shape=[1, 16, 8, 8],
        shape_signature=[1, 16, 8, 8],
    )
    model_ir.tensors["y_nhwc"] = TensorIR(
        name="y_nhwc",
        dtype="FLOAT32",
        shape=[1, 8, 8, 16],
        shape_signature=[1, 8, 8, 16],
    )
    model_ir.tensors["z"] = TensorIR(
        name="z",
        dtype="FLOAT32",
        shape=[1, 8, 8, 16],
        shape_signature=[1, 8, 8, 16],
    )
    conv_options = {
        "padding": "SAME",
        "strideH": 1,
        "strideW": 1,
        "dilationHFactor": 1,
        "dilationWFactor": 1,
        "fusedActivationFunction": "NONE",
    }
    model_ir.operators = [
        OperatorIR(op_type="TRANSPOSE", inputs=["x_nhwc", "pre_perm"], outputs=["x_nchw"]),
        OperatorIR(op_type="RELU", inputs=["x_nchw"], outputs=["a_nchw"]),
        OperatorIR(op_type="MEAN", inputs=["a_nchw", "mean_axes"], outputs=["mean_nchw"], options={"keepDims": True}),
        OperatorIR(op_type="TRANSPOSE", inputs=["mean_nchw", "post_perm"], outputs=["mean_nhwc"]),
        OperatorIR(op_type="CONV_2D", inputs=["mean_nhwc", "conv1_w", "conv1_b"], outputs=["conv1_out"], options=conv_options),
        OperatorIR(op_type="CONV_2D", inputs=["conv1_out", "conv2_w", "conv2_b"], outputs=["conv2_out_nhwc"], options=conv_options),
        OperatorIR(op_type="TRANSPOSE", inputs=["conv2_out_nhwc", "pre_perm"], outputs=["sig_in_nchw"]),
        OperatorIR(op_type="LOGISTIC", inputs=["sig_in_nchw"], outputs=["sig_nchw"]),
        OperatorIR(
            op_type="MUL",
            inputs=["a_nchw", "sig_nchw"],
            outputs=["mul_nchw"],
            options={"fusedActivationFunction": "NONE"},
        ),
        OperatorIR(op_type="TRANSPOSE", inputs=["mul_nchw", "post_perm"], outputs=["y_nhwc"]),
        OperatorIR(op_type="RELU", inputs=["y_nhwc"], outputs=["z"]),
    ]

    stats = _optimize_transpose_conv_attention_nhwc_propagation_chains(model_ir)
    assert stats["optimized_transpose_conv_attention_nhwc_propagation_chains"] == 1

    op_types = [str(op.op_type) for op in model_ir.operators]
    assert op_types.count("TRANSPOSE") == 0
    assert np.array_equal(np.asarray(model_ir.tensors["mean_axes"].data), np.asarray([1, 2], dtype=np.int32))

    conv_ops = [op for op in model_ir.operators if str(op.op_type) == "CONV_2D"]
    assert list(conv_ops[0].inputs)[0] == "mean_nchw"
    logistic_op = next(op for op in model_ir.operators if str(op.op_type) == "LOGISTIC")
    assert list(logistic_op.inputs) == ["conv2_out_nhwc"]
    mul_op = next(op for op in model_ir.operators if str(op.op_type) == "MUL")
    assert list(mul_op.outputs) == ["y_nhwc"]
    assert list(model_ir.tensors["a_nchw"].shape) == [1, 8, 8, 16]


def test_flatbuffer_direct_transpose_conv_attention_hardsigmoid_nhwc_propagation_optimized() -> None:
    model_ir = ModelIR(name="transpose_conv_attention_hardsigmoid_nhwc_propagation_opt_test")
    model_ir.inputs = ["x_nhwc"]
    model_ir.outputs = ["z"]
    model_ir.tensors["x_nhwc"] = TensorIR(
        name="x_nhwc",
        dtype="FLOAT32",
        shape=[1, 8, 8, 16],
        shape_signature=[1, 8, 8, 16],
    )
    model_ir.tensors["pre_perm"] = TensorIR(
        name="pre_perm",
        dtype="INT32",
        shape=[4],
        shape_signature=[4],
        data=np.asarray([0, 3, 1, 2], dtype=np.int32),
        is_variable=False,
    )
    model_ir.tensors["post_perm"] = TensorIR(
        name="post_perm",
        dtype="INT32",
        shape=[4],
        shape_signature=[4],
        data=np.asarray([0, 2, 3, 1], dtype=np.int32),
        is_variable=False,
    )
    model_ir.tensors["mean_axes"] = TensorIR(
        name="mean_axes",
        dtype="INT32",
        shape=[2],
        shape_signature=[2],
        data=np.asarray([2, 3], dtype=np.int32),
        is_variable=False,
    )
    model_ir.tensors["x_nchw"] = TensorIR(
        name="x_nchw",
        dtype="FLOAT32",
        shape=[1, 16, 8, 8],
        shape_signature=[1, 16, 8, 8],
    )
    model_ir.tensors["a_nchw"] = TensorIR(
        name="a_nchw",
        dtype="FLOAT32",
        shape=[1, 16, 8, 8],
        shape_signature=[1, 16, 8, 8],
    )
    model_ir.tensors["mean_nchw"] = TensorIR(
        name="mean_nchw",
        dtype="FLOAT32",
        shape=[1, 16, 1, 1],
        shape_signature=[1, 16, 1, 1],
    )
    model_ir.tensors["mean_nhwc"] = TensorIR(
        name="mean_nhwc",
        dtype="FLOAT32",
        shape=[1, 1, 1, 16],
        shape_signature=[1, 1, 1, 16],
    )
    model_ir.tensors["conv1_w"] = TensorIR(
        name="conv1_w",
        dtype="FLOAT32",
        shape=[8, 1, 1, 16],
        shape_signature=[8, 1, 1, 16],
        data=np.ones((8, 1, 1, 16), dtype=np.float32),
        is_variable=False,
    )
    model_ir.tensors["conv1_b"] = TensorIR(
        name="conv1_b",
        dtype="FLOAT32",
        shape=[8],
        shape_signature=[8],
        data=np.zeros((8,), dtype=np.float32),
        is_variable=False,
    )
    model_ir.tensors["conv1_out"] = TensorIR(
        name="conv1_out",
        dtype="FLOAT32",
        shape=[1, 1, 1, 8],
        shape_signature=[1, 1, 1, 8],
    )
    model_ir.tensors["conv2_w"] = TensorIR(
        name="conv2_w",
        dtype="FLOAT32",
        shape=[16, 1, 1, 8],
        shape_signature=[16, 1, 1, 8],
        data=np.ones((16, 1, 1, 8), dtype=np.float32),
        is_variable=False,
    )
    model_ir.tensors["conv2_b"] = TensorIR(
        name="conv2_b",
        dtype="FLOAT32",
        shape=[16],
        shape_signature=[16],
        data=np.zeros((16,), dtype=np.float32),
        is_variable=False,
    )
    model_ir.tensors["conv2_out_nhwc"] = TensorIR(
        name="conv2_out_nhwc",
        dtype="FLOAT32",
        shape=[1, 1, 1, 16],
        shape_signature=[1, 1, 1, 16],
    )
    model_ir.tensors["sig_in_nchw"] = TensorIR(
        name="sig_in_nchw",
        dtype="FLOAT32",
        shape=[1, 16, 1, 1],
        shape_signature=[1, 16, 1, 1],
    )
    model_ir.tensors["hs_alpha"] = TensorIR(
        name="hs_alpha",
        dtype="FLOAT32",
        shape=[1],
        shape_signature=[1],
        data=np.asarray([1.0 / 6.0], dtype=np.float32),
        is_variable=False,
    )
    model_ir.tensors["hs_beta"] = TensorIR(
        name="hs_beta",
        dtype="FLOAT32",
        shape=[1],
        shape_signature=[1],
        data=np.asarray([0.5], dtype=np.float32),
        is_variable=False,
    )
    model_ir.tensors["sig_mul_nchw"] = TensorIR(
        name="sig_mul_nchw",
        dtype="FLOAT32",
        shape=[1, 16, 1, 1],
        shape_signature=[1, 16, 1, 1],
    )
    model_ir.tensors["sig_add_nchw"] = TensorIR(
        name="sig_add_nchw",
        dtype="FLOAT32",
        shape=[1, 16, 1, 1],
        shape_signature=[1, 16, 1, 1],
    )
    model_ir.tensors["sig_nchw"] = TensorIR(
        name="sig_nchw",
        dtype="FLOAT32",
        shape=[1, 16, 1, 1],
        shape_signature=[1, 16, 1, 1],
    )
    model_ir.tensors["mul_nchw"] = TensorIR(
        name="mul_nchw",
        dtype="FLOAT32",
        shape=[1, 16, 8, 8],
        shape_signature=[1, 16, 8, 8],
    )
    model_ir.tensors["y_nhwc"] = TensorIR(
        name="y_nhwc",
        dtype="FLOAT32",
        shape=[1, 8, 8, 16],
        shape_signature=[1, 8, 8, 16],
    )
    model_ir.tensors["z"] = TensorIR(
        name="z",
        dtype="FLOAT32",
        shape=[1, 8, 8, 16],
        shape_signature=[1, 8, 8, 16],
    )
    conv_options = {
        "padding": "SAME",
        "strideH": 1,
        "strideW": 1,
        "dilationHFactor": 1,
        "dilationWFactor": 1,
        "fusedActivationFunction": "NONE",
    }
    model_ir.operators = [
        OperatorIR(op_type="TRANSPOSE", inputs=["x_nhwc", "pre_perm"], outputs=["x_nchw"]),
        OperatorIR(op_type="RELU", inputs=["x_nchw"], outputs=["a_nchw"]),
        OperatorIR(op_type="MEAN", inputs=["a_nchw", "mean_axes"], outputs=["mean_nchw"], options={"keepDims": True}),
        OperatorIR(op_type="TRANSPOSE", inputs=["mean_nchw", "post_perm"], outputs=["mean_nhwc"]),
        OperatorIR(op_type="CONV_2D", inputs=["mean_nhwc", "conv1_w", "conv1_b"], outputs=["conv1_out"], options=conv_options),
        OperatorIR(op_type="CONV_2D", inputs=["conv1_out", "conv2_w", "conv2_b"], outputs=["conv2_out_nhwc"], options=conv_options),
        OperatorIR(op_type="TRANSPOSE", inputs=["conv2_out_nhwc", "pre_perm"], outputs=["sig_in_nchw"]),
        OperatorIR(
            op_type="MUL",
            inputs=["sig_in_nchw", "hs_alpha"],
            outputs=["sig_mul_nchw"],
            options={"fusedActivationFunction": "NONE"},
        ),
        OperatorIR(
            op_type="ADD",
            inputs=["sig_mul_nchw", "hs_beta"],
            outputs=["sig_add_nchw"],
            options={"fusedActivationFunction": "NONE"},
        ),
        OperatorIR(op_type="RELU_0_TO_1", inputs=["sig_add_nchw"], outputs=["sig_nchw"]),
        OperatorIR(
            op_type="MUL",
            inputs=["a_nchw", "sig_nchw"],
            outputs=["mul_nchw"],
            options={"fusedActivationFunction": "NONE"},
        ),
        OperatorIR(op_type="TRANSPOSE", inputs=["mul_nchw", "post_perm"], outputs=["y_nhwc"]),
        OperatorIR(op_type="RELU", inputs=["y_nhwc"], outputs=["z"]),
    ]

    stats = _optimize_transpose_conv_attention_nhwc_propagation_chains(model_ir)
    assert stats["optimized_transpose_conv_attention_nhwc_propagation_chains"] == 1

    op_types = [str(op.op_type) for op in model_ir.operators]
    assert op_types.count("TRANSPOSE") == 0
    assert np.array_equal(np.asarray(model_ir.tensors["mean_axes"].data), np.asarray([1, 2], dtype=np.int32))

    gate_mul_op = next(op for op in model_ir.operators if list(op.outputs) == ["sig_mul_nchw"])
    assert list(gate_mul_op.inputs) == ["conv2_out_nhwc", "hs_alpha"]
    final_mul_op = next(op for op in model_ir.operators if list(op.outputs) == ["y_nhwc"])
    assert list(final_mul_op.inputs)[0] == "a_nchw"
    assert list(model_ir.tensors["a_nchw"].shape) == [1, 8, 8, 16]


def test_flatbuffer_direct_transpose_conv_attention_hardswish_activation_nhwc_propagation_optimized() -> None:
    model_ir = ModelIR(name="transpose_conv_attention_hardswish_activation_nhwc_propagation_opt_test")
    model_ir.inputs = ["x_nhwc"]
    model_ir.outputs = ["z"]
    model_ir.tensors["x_nhwc"] = TensorIR(
        name="x_nhwc",
        dtype="FLOAT32",
        shape=[1, 8, 8, 16],
        shape_signature=[1, 8, 8, 16],
    )
    model_ir.tensors["pre_perm"] = TensorIR(
        name="pre_perm",
        dtype="INT32",
        shape=[4],
        shape_signature=[4],
        data=np.asarray([0, 3, 1, 2], dtype=np.int32),
        is_variable=False,
    )
    model_ir.tensors["post_perm"] = TensorIR(
        name="post_perm",
        dtype="INT32",
        shape=[4],
        shape_signature=[4],
        data=np.asarray([0, 2, 3, 1], dtype=np.int32),
        is_variable=False,
    )
    model_ir.tensors["mean_axes"] = TensorIR(
        name="mean_axes",
        dtype="INT32",
        shape=[2],
        shape_signature=[2],
        data=np.asarray([2, 3], dtype=np.int32),
        is_variable=False,
    )
    model_ir.tensors["x_nchw"] = TensorIR(
        name="x_nchw",
        dtype="FLOAT32",
        shape=[1, 16, 8, 8],
        shape_signature=[1, 16, 8, 8],
    )
    model_ir.tensors["act_sig_mul_nchw"] = TensorIR(
        name="act_sig_mul_nchw",
        dtype="FLOAT32",
        shape=[1, 16, 8, 8],
        shape_signature=[1, 16, 8, 8],
    )
    model_ir.tensors["act_sig_add_nchw"] = TensorIR(
        name="act_sig_add_nchw",
        dtype="FLOAT32",
        shape=[1, 16, 8, 8],
        shape_signature=[1, 16, 8, 8],
    )
    model_ir.tensors["act_sig_nchw"] = TensorIR(
        name="act_sig_nchw",
        dtype="FLOAT32",
        shape=[1, 16, 8, 8],
        shape_signature=[1, 16, 8, 8],
    )
    model_ir.tensors["a_nchw"] = TensorIR(
        name="a_nchw",
        dtype="FLOAT32",
        shape=[1, 16, 8, 8],
        shape_signature=[1, 16, 8, 8],
    )
    model_ir.tensors["mean_nchw"] = TensorIR(
        name="mean_nchw",
        dtype="FLOAT32",
        shape=[1, 16, 1, 1],
        shape_signature=[1, 16, 1, 1],
    )
    model_ir.tensors["mean_nhwc"] = TensorIR(
        name="mean_nhwc",
        dtype="FLOAT32",
        shape=[1, 1, 1, 16],
        shape_signature=[1, 1, 1, 16],
    )
    model_ir.tensors["conv1_w"] = TensorIR(
        name="conv1_w",
        dtype="FLOAT32",
        shape=[8, 1, 1, 16],
        shape_signature=[8, 1, 1, 16],
        data=np.ones((8, 1, 1, 16), dtype=np.float32),
        is_variable=False,
    )
    model_ir.tensors["conv1_b"] = TensorIR(
        name="conv1_b",
        dtype="FLOAT32",
        shape=[8],
        shape_signature=[8],
        data=np.zeros((8,), dtype=np.float32),
        is_variable=False,
    )
    model_ir.tensors["conv1_out"] = TensorIR(
        name="conv1_out",
        dtype="FLOAT32",
        shape=[1, 1, 1, 8],
        shape_signature=[1, 1, 1, 8],
    )
    model_ir.tensors["conv2_w"] = TensorIR(
        name="conv2_w",
        dtype="FLOAT32",
        shape=[16, 1, 1, 8],
        shape_signature=[16, 1, 1, 8],
        data=np.ones((16, 1, 1, 8), dtype=np.float32),
        is_variable=False,
    )
    model_ir.tensors["conv2_b"] = TensorIR(
        name="conv2_b",
        dtype="FLOAT32",
        shape=[16],
        shape_signature=[16],
        data=np.zeros((16,), dtype=np.float32),
        is_variable=False,
    )
    model_ir.tensors["conv2_out_nhwc"] = TensorIR(
        name="conv2_out_nhwc",
        dtype="FLOAT32",
        shape=[1, 1, 1, 16],
        shape_signature=[1, 1, 1, 16],
    )
    model_ir.tensors["gate_in_nchw"] = TensorIR(
        name="gate_in_nchw",
        dtype="FLOAT32",
        shape=[1, 16, 1, 1],
        shape_signature=[1, 16, 1, 1],
    )
    model_ir.tensors["hs_alpha"] = TensorIR(
        name="hs_alpha",
        dtype="FLOAT32",
        shape=[1],
        shape_signature=[1],
        data=np.asarray([1.0 / 6.0], dtype=np.float32),
        is_variable=False,
    )
    model_ir.tensors["hs_beta"] = TensorIR(
        name="hs_beta",
        dtype="FLOAT32",
        shape=[1],
        shape_signature=[1],
        data=np.asarray([0.5], dtype=np.float32),
        is_variable=False,
    )
    model_ir.tensors["gate_mul_nchw"] = TensorIR(
        name="gate_mul_nchw",
        dtype="FLOAT32",
        shape=[1, 16, 1, 1],
        shape_signature=[1, 16, 1, 1],
    )
    model_ir.tensors["gate_add_nchw"] = TensorIR(
        name="gate_add_nchw",
        dtype="FLOAT32",
        shape=[1, 16, 1, 1],
        shape_signature=[1, 16, 1, 1],
    )
    model_ir.tensors["sig_nchw"] = TensorIR(
        name="sig_nchw",
        dtype="FLOAT32",
        shape=[1, 16, 1, 1],
        shape_signature=[1, 16, 1, 1],
    )
    model_ir.tensors["mul_nchw"] = TensorIR(
        name="mul_nchw",
        dtype="FLOAT32",
        shape=[1, 16, 8, 8],
        shape_signature=[1, 16, 8, 8],
    )
    model_ir.tensors["y_nhwc"] = TensorIR(
        name="y_nhwc",
        dtype="FLOAT32",
        shape=[1, 8, 8, 16],
        shape_signature=[1, 8, 8, 16],
    )
    model_ir.tensors["z"] = TensorIR(
        name="z",
        dtype="FLOAT32",
        shape=[1, 8, 8, 16],
        shape_signature=[1, 8, 8, 16],
    )
    conv_options = {
        "padding": "SAME",
        "strideH": 1,
        "strideW": 1,
        "dilationHFactor": 1,
        "dilationWFactor": 1,
        "fusedActivationFunction": "NONE",
    }
    model_ir.operators = [
        OperatorIR(op_type="TRANSPOSE", inputs=["x_nhwc", "pre_perm"], outputs=["x_nchw"]),
        OperatorIR(
            op_type="MUL",
            inputs=["x_nchw", "hs_alpha"],
            outputs=["act_sig_mul_nchw"],
            options={"fusedActivationFunction": "NONE"},
        ),
        OperatorIR(
            op_type="ADD",
            inputs=["act_sig_mul_nchw", "hs_beta"],
            outputs=["act_sig_add_nchw"],
            options={"fusedActivationFunction": "NONE"},
        ),
        OperatorIR(op_type="RELU_0_TO_1", inputs=["act_sig_add_nchw"], outputs=["act_sig_nchw"]),
        OperatorIR(
            op_type="MUL",
            inputs=["x_nchw", "act_sig_nchw"],
            outputs=["a_nchw"],
            options={"fusedActivationFunction": "NONE"},
        ),
        OperatorIR(op_type="MEAN", inputs=["a_nchw", "mean_axes"], outputs=["mean_nchw"], options={"keepDims": True}),
        OperatorIR(op_type="TRANSPOSE", inputs=["mean_nchw", "post_perm"], outputs=["mean_nhwc"]),
        OperatorIR(op_type="CONV_2D", inputs=["mean_nhwc", "conv1_w", "conv1_b"], outputs=["conv1_out"], options=conv_options),
        OperatorIR(op_type="CONV_2D", inputs=["conv1_out", "conv2_w", "conv2_b"], outputs=["conv2_out_nhwc"], options=conv_options),
        OperatorIR(op_type="TRANSPOSE", inputs=["conv2_out_nhwc", "pre_perm"], outputs=["gate_in_nchw"]),
        OperatorIR(
            op_type="MUL",
            inputs=["gate_in_nchw", "hs_alpha"],
            outputs=["gate_mul_nchw"],
            options={"fusedActivationFunction": "NONE"},
        ),
        OperatorIR(
            op_type="ADD",
            inputs=["gate_mul_nchw", "hs_beta"],
            outputs=["gate_add_nchw"],
            options={"fusedActivationFunction": "NONE"},
        ),
        OperatorIR(op_type="RELU_0_TO_1", inputs=["gate_add_nchw"], outputs=["sig_nchw"]),
        OperatorIR(
            op_type="MUL",
            inputs=["a_nchw", "sig_nchw"],
            outputs=["mul_nchw"],
            options={"fusedActivationFunction": "NONE"},
        ),
        OperatorIR(op_type="TRANSPOSE", inputs=["mul_nchw", "post_perm"], outputs=["y_nhwc"]),
        OperatorIR(op_type="RELU", inputs=["y_nhwc"], outputs=["z"]),
    ]

    stats = _optimize_transpose_conv_attention_nhwc_propagation_chains(model_ir)
    assert stats["optimized_transpose_conv_attention_nhwc_propagation_chains"] == 1

    op_types = [str(op.op_type) for op in model_ir.operators]
    assert op_types.count("TRANSPOSE") == 0
    assert np.array_equal(np.asarray(model_ir.tensors["mean_axes"].data), np.asarray([1, 2], dtype=np.int32))

    act_mul_op = next(op for op in model_ir.operators if list(op.outputs) == ["a_nchw"])
    assert list(act_mul_op.inputs)[0] == "x_nhwc"
    act_gate_mul_op = next(op for op in model_ir.operators if list(op.outputs) == ["act_sig_mul_nchw"])
    assert list(act_gate_mul_op.inputs)[0] == "x_nhwc"
    gate_mul_op = next(op for op in model_ir.operators if list(op.outputs) == ["gate_mul_nchw"])
    assert list(gate_mul_op.inputs) == ["conv2_out_nhwc", "hs_alpha"]
    final_mul_op = next(op for op in model_ir.operators if list(op.outputs) == ["y_nhwc"])
    assert list(final_mul_op.inputs)[0] == "a_nchw"
    assert list(model_ir.tensors["a_nchw"].shape) == [1, 8, 8, 16]


def test_flatbuffer_direct_transpose_conv_hardswish_mean_conv_hardswish_mean_chain_optimized() -> None:
    model_ir = ModelIR(name="transpose_conv_hardswish_mean_conv_hardswish_mean_chain_opt_test")
    model_ir.inputs = ["x_nhwc"]
    model_ir.outputs = ["z"]
    model_ir.tensors["x_nhwc"] = TensorIR(
        name="x_nhwc",
        dtype="FLOAT32",
        shape=[1, 7, 7, 8],
        shape_signature=[1, 7, 7, 8],
    )
    model_ir.tensors["pre_perm"] = TensorIR(
        name="pre_perm",
        dtype="INT32",
        shape=[4],
        shape_signature=[4],
        data=np.asarray([0, 3, 1, 2], dtype=np.int32),
        is_variable=False,
    )
    model_ir.tensors["post_perm"] = TensorIR(
        name="post_perm",
        dtype="INT32",
        shape=[4],
        shape_signature=[4],
        data=np.asarray([0, 2, 3, 1], dtype=np.int32),
        is_variable=False,
    )
    model_ir.tensors["x_nchw"] = TensorIR(
        name="x_nchw",
        dtype="FLOAT32",
        shape=[1, 8, 7, 7],
        shape_signature=[1, 8, 7, 7],
    )
    model_ir.tensors["add0_bias"] = TensorIR(
        name="add0_bias",
        dtype="FLOAT32",
        shape=[1],
        shape_signature=[1],
        data=np.asarray([3.0], dtype=np.float32),
        is_variable=False,
    )
    model_ir.tensors["mul0_scale"] = TensorIR(
        name="mul0_scale",
        dtype="FLOAT32",
        shape=[1],
        shape_signature=[1],
        data=np.asarray([1.0 / 6.0], dtype=np.float32),
        is_variable=False,
    )
    model_ir.tensors["add0_out"] = TensorIR(
        name="add0_out",
        dtype="FLOAT32",
        shape=[1, 8, 7, 7],
        shape_signature=[1, 8, 7, 7],
    )
    model_ir.tensors["scale0_out"] = TensorIR(
        name="scale0_out",
        dtype="FLOAT32",
        shape=[1, 8, 7, 7],
        shape_signature=[1, 8, 7, 7],
    )
    model_ir.tensors["hsw0_out"] = TensorIR(
        name="hsw0_out",
        dtype="FLOAT32",
        shape=[1, 8, 7, 7],
        shape_signature=[1, 8, 7, 7],
    )
    model_ir.tensors["mean0_axes"] = TensorIR(
        name="mean0_axes",
        dtype="INT32",
        shape=[2],
        shape_signature=[2],
        data=np.asarray([2, 3], dtype=np.int32),
        is_variable=False,
    )
    model_ir.tensors["mean0_out"] = TensorIR(
        name="mean0_out",
        dtype="FLOAT32",
        shape=[1, 8, 1, 1],
        shape_signature=[1, 8, 1, 1],
    )
    model_ir.tensors["mean0_nhwc"] = TensorIR(
        name="mean0_nhwc",
        dtype="FLOAT32",
        shape=[1, 1, 1, 8],
        shape_signature=[1, 1, 1, 8],
    )
    model_ir.tensors["conv_w"] = TensorIR(
        name="conv_w",
        dtype="FLOAT32",
        shape=[16, 1, 1, 8],
        shape_signature=[16, 1, 1, 8],
        data=np.ones((16, 1, 1, 8), dtype=np.float32),
        is_variable=False,
    )
    model_ir.tensors["conv_b"] = TensorIR(
        name="conv_b",
        dtype="FLOAT32",
        shape=[16],
        shape_signature=[16],
        data=np.zeros((16,), dtype=np.float32),
        is_variable=False,
    )
    model_ir.tensors["conv_out_nhwc"] = TensorIR(
        name="conv_out_nhwc",
        dtype="FLOAT32",
        shape=[1, 1, 1, 16],
        shape_signature=[1, 1, 1, 16],
    )
    model_ir.tensors["conv_out_nchw"] = TensorIR(
        name="conv_out_nchw",
        dtype="FLOAT32",
        shape=[1, 16, 1, 1],
        shape_signature=[1, 16, 1, 1],
    )
    model_ir.tensors["add1_bias"] = TensorIR(
        name="add1_bias",
        dtype="FLOAT32",
        shape=[1],
        shape_signature=[1],
        data=np.asarray([3.0], dtype=np.float32),
        is_variable=False,
    )
    model_ir.tensors["mul1_scale"] = TensorIR(
        name="mul1_scale",
        dtype="FLOAT32",
        shape=[1],
        shape_signature=[1],
        data=np.asarray([1.0 / 6.0], dtype=np.float32),
        is_variable=False,
    )
    model_ir.tensors["add1_out"] = TensorIR(
        name="add1_out",
        dtype="FLOAT32",
        shape=[1, 16, 1, 1],
        shape_signature=[1, 16, 1, 1],
    )
    model_ir.tensors["scale1_out"] = TensorIR(
        name="scale1_out",
        dtype="FLOAT32",
        shape=[1, 16, 1, 1],
        shape_signature=[1, 16, 1, 1],
    )
    model_ir.tensors["hsw1_out"] = TensorIR(
        name="hsw1_out",
        dtype="FLOAT32",
        shape=[1, 16, 1, 1],
        shape_signature=[1, 16, 1, 1],
    )
    model_ir.tensors["mean1_axes"] = TensorIR(
        name="mean1_axes",
        dtype="INT32",
        shape=[2],
        shape_signature=[2],
        data=np.asarray([2, 3], dtype=np.int32),
        is_variable=False,
    )
    model_ir.tensors["mean1_out"] = TensorIR(
        name="mean1_out",
        dtype="FLOAT32",
        shape=[1, 16, 1, 1],
        shape_signature=[1, 16, 1, 1],
    )
    model_ir.tensors["reshape_shape"] = TensorIR(
        name="reshape_shape",
        dtype="INT32",
        shape=[2],
        shape_signature=[2],
        data=np.asarray([1, 16], dtype=np.int32),
        is_variable=False,
    )
    model_ir.tensors["z"] = TensorIR(
        name="z",
        dtype="FLOAT32",
        shape=[1, 16],
        shape_signature=[1, 16],
    )
    conv_options = {
        "padding": "SAME",
        "strideH": 1,
        "strideW": 1,
        "dilationHFactor": 1,
        "dilationWFactor": 1,
        "fusedActivationFunction": "NONE",
    }
    model_ir.operators = [
        OperatorIR(op_type="TRANSPOSE", inputs=["x_nhwc", "pre_perm"], outputs=["x_nchw"]),
        OperatorIR(op_type="ADD", inputs=["x_nchw", "add0_bias"], outputs=["add0_out"]),
        OperatorIR(op_type="MUL", inputs=["add0_out", "mul0_scale"], outputs=["scale0_out"]),
        OperatorIR(op_type="MUL", inputs=["x_nchw", "scale0_out"], outputs=["hsw0_out"]),
        OperatorIR(op_type="MEAN", inputs=["hsw0_out", "mean0_axes"], outputs=["mean0_out"], options={"keepDims": True}),
        OperatorIR(op_type="TRANSPOSE", inputs=["mean0_out", "post_perm"], outputs=["mean0_nhwc"]),
        OperatorIR(op_type="CONV_2D", inputs=["mean0_nhwc", "conv_w", "conv_b"], outputs=["conv_out_nhwc"], options=conv_options),
        OperatorIR(op_type="TRANSPOSE", inputs=["conv_out_nhwc", "pre_perm"], outputs=["conv_out_nchw"]),
        OperatorIR(op_type="ADD", inputs=["conv_out_nchw", "add1_bias"], outputs=["add1_out"]),
        OperatorIR(op_type="MUL", inputs=["add1_out", "mul1_scale"], outputs=["scale1_out"]),
        OperatorIR(op_type="MUL", inputs=["conv_out_nchw", "scale1_out"], outputs=["hsw1_out"]),
        OperatorIR(op_type="MEAN", inputs=["hsw1_out", "mean1_axes"], outputs=["mean1_out"], options={"keepDims": True}),
        OperatorIR(op_type="RESHAPE", inputs=["mean1_out", "reshape_shape"], outputs=["z"], options={"newShape": [1, 16]}),
    ]

    stats = _optimize_transpose_conv_attention_nhwc_propagation_chains(model_ir)
    assert stats["optimized_transpose_conv_attention_nhwc_propagation_chains"] == 1

    op_types = [str(op.op_type) for op in model_ir.operators]
    assert op_types.count("TRANSPOSE") == 0
    assert np.array_equal(np.asarray(model_ir.tensors["mean0_axes"].data), np.asarray([1, 2], dtype=np.int32))
    assert np.array_equal(np.asarray(model_ir.tensors["mean1_axes"].data), np.asarray([1, 2], dtype=np.int32))

    conv_op = next(op for op in model_ir.operators if str(op.op_type) == "CONV_2D")
    assert list(conv_op.inputs)[0] == "mean0_out"

    add0_op = next(op for op in model_ir.operators if list(op.outputs) == ["add0_out"])
    mul0_op = next(op for op in model_ir.operators if list(op.outputs) == ["hsw0_out"])
    assert "x_nhwc" in list(add0_op.inputs)
    assert "x_nhwc" in list(mul0_op.inputs)

    add1_op = next(op for op in model_ir.operators if list(op.outputs) == ["add1_out"])
    mul1_op = next(op for op in model_ir.operators if list(op.outputs) == ["hsw1_out"])
    assert "conv_out_nhwc" in list(add1_op.inputs)
    assert "conv_out_nhwc" in list(mul1_op.inputs)

    assert list(model_ir.tensors["mean0_out"].shape) == [1, 1, 1, 8]
    assert list(model_ir.tensors["mean1_out"].shape) == [1, 1, 1, 16]


def test_flatbuffer_direct_transpose_csp_attention_nhwc_chain_optimized() -> None:
    model_ir = ModelIR(name="transpose_csp_attention_nhwc_chain_opt_test")
    model_ir.inputs = ["short_nhwc", "main_nhwc", "point_nhwc"]
    model_ir.outputs = ["z"]
    model_ir.tensors["short_nhwc"] = TensorIR(
        name="short_nhwc",
        dtype="FLOAT32",
        shape=[1, 16, 16, 24],
        shape_signature=[1, 16, 16, 24],
    )
    model_ir.tensors["main_nhwc"] = TensorIR(
        name="main_nhwc",
        dtype="FLOAT32",
        shape=[1, 16, 16, 24],
        shape_signature=[1, 16, 16, 24],
    )
    model_ir.tensors["point_nhwc"] = TensorIR(
        name="point_nhwc",
        dtype="FLOAT32",
        shape=[1, 16, 16, 24],
        shape_signature=[1, 16, 16, 24],
    )
    model_ir.tensors["pre_perm"] = TensorIR(
        name="pre_perm",
        dtype="INT32",
        shape=[4],
        shape_signature=[4],
        data=np.asarray([0, 3, 1, 2], dtype=np.int32),
        is_variable=False,
    )
    model_ir.tensors["post_perm"] = TensorIR(
        name="post_perm",
        dtype="INT32",
        shape=[4],
        shape_signature=[4],
        data=np.asarray([0, 2, 3, 1], dtype=np.int32),
        is_variable=False,
    )
    model_ir.tensors["mean_axes"] = TensorIR(
        name="mean_axes",
        dtype="INT32",
        shape=[2],
        shape_signature=[2],
        data=np.asarray([2, 3], dtype=np.int32),
        is_variable=False,
    )
    model_ir.tensors["hs_alpha"] = TensorIR(
        name="hs_alpha",
        dtype="FLOAT32",
        shape=[1],
        shape_signature=[1],
        data=np.asarray([1.0 / 6.0], dtype=np.float32),
        is_variable=False,
    )
    model_ir.tensors["hs_beta"] = TensorIR(
        name="hs_beta",
        dtype="FLOAT32",
        shape=[1],
        shape_signature=[1],
        data=np.asarray([0.5], dtype=np.float32),
        is_variable=False,
    )
    model_ir.tensors["gate_w"] = TensorIR(
        name="gate_w",
        dtype="FLOAT32",
        shape=[48, 1, 1, 48],
        shape_signature=[48, 1, 1, 48],
        data=np.ones((48, 1, 1, 48), dtype=np.float32),
        is_variable=False,
    )
    model_ir.tensors["gate_b"] = TensorIR(
        name="gate_b",
        dtype="FLOAT32",
        shape=[48],
        shape_signature=[48],
        data=np.zeros((48,), dtype=np.float32),
        is_variable=False,
    )
    model_ir.tensors["short_nchw"] = TensorIR(
        name="short_nchw",
        dtype="FLOAT32",
        shape=[1, 24, 16, 16],
        shape_signature=[1, 24, 16, 16],
    )
    model_ir.tensors["short_sig"] = TensorIR(
        name="short_sig",
        dtype="FLOAT32",
        shape=[1, 24, 16, 16],
        shape_signature=[1, 24, 16, 16],
    )
    model_ir.tensors["short_branch"] = TensorIR(
        name="short_branch",
        dtype="FLOAT32",
        shape=[1, 24, 16, 16],
        shape_signature=[1, 24, 16, 16],
    )
    model_ir.tensors["main_nchw"] = TensorIR(
        name="main_nchw",
        dtype="FLOAT32",
        shape=[1, 24, 16, 16],
        shape_signature=[1, 24, 16, 16],
    )
    model_ir.tensors["point_nchw"] = TensorIR(
        name="point_nchw",
        dtype="FLOAT32",
        shape=[1, 24, 16, 16],
        shape_signature=[1, 24, 16, 16],
    )
    model_ir.tensors["point_sig"] = TensorIR(
        name="point_sig",
        dtype="FLOAT32",
        shape=[1, 24, 16, 16],
        shape_signature=[1, 24, 16, 16],
    )
    model_ir.tensors["point_branch"] = TensorIR(
        name="point_branch",
        dtype="FLOAT32",
        shape=[1, 24, 16, 16],
        shape_signature=[1, 24, 16, 16],
    )
    model_ir.tensors["add_nchw"] = TensorIR(
        name="add_nchw",
        dtype="FLOAT32",
        shape=[1, 24, 16, 16],
        shape_signature=[1, 24, 16, 16],
    )
    model_ir.tensors["concat_nchw"] = TensorIR(
        name="concat_nchw",
        dtype="FLOAT32",
        shape=[1, 48, 16, 16],
        shape_signature=[1, 48, 16, 16],
    )
    model_ir.tensors["mean_nchw"] = TensorIR(
        name="mean_nchw",
        dtype="FLOAT32",
        shape=[1, 48, 1, 1],
        shape_signature=[1, 48, 1, 1],
    )
    model_ir.tensors["mean_nhwc"] = TensorIR(
        name="mean_nhwc",
        dtype="FLOAT32",
        shape=[1, 1, 1, 48],
        shape_signature=[1, 1, 1, 48],
    )
    model_ir.tensors["gate_conv_out_nhwc"] = TensorIR(
        name="gate_conv_out_nhwc",
        dtype="FLOAT32",
        shape=[1, 1, 1, 48],
        shape_signature=[1, 1, 1, 48],
    )
    model_ir.tensors["gate_in_nchw"] = TensorIR(
        name="gate_in_nchw",
        dtype="FLOAT32",
        shape=[1, 48, 1, 1],
        shape_signature=[1, 48, 1, 1],
    )
    model_ir.tensors["gate_mul"] = TensorIR(
        name="gate_mul",
        dtype="FLOAT32",
        shape=[1, 48, 1, 1],
        shape_signature=[1, 48, 1, 1],
    )
    model_ir.tensors["gate_add"] = TensorIR(
        name="gate_add",
        dtype="FLOAT32",
        shape=[1, 48, 1, 1],
        shape_signature=[1, 48, 1, 1],
    )
    model_ir.tensors["gate_nchw"] = TensorIR(
        name="gate_nchw",
        dtype="FLOAT32",
        shape=[1, 48, 1, 1],
        shape_signature=[1, 48, 1, 1],
    )
    model_ir.tensors["attn_mul_nchw"] = TensorIR(
        name="attn_mul_nchw",
        dtype="FLOAT32",
        shape=[1, 48, 16, 16],
        shape_signature=[1, 48, 16, 16],
    )
    model_ir.tensors["y_nhwc"] = TensorIR(
        name="y_nhwc",
        dtype="FLOAT32",
        shape=[1, 16, 16, 48],
        shape_signature=[1, 16, 16, 48],
    )
    model_ir.tensors["z"] = TensorIR(
        name="z",
        dtype="FLOAT32",
        shape=[1, 16, 16, 48],
        shape_signature=[1, 16, 16, 48],
    )
    conv_options = {
        "padding": "SAME",
        "strideH": 1,
        "strideW": 1,
        "dilationHFactor": 1,
        "dilationWFactor": 1,
        "fusedActivationFunction": "NONE",
    }
    model_ir.operators = [
        OperatorIR(op_type="TRANSPOSE", inputs=["short_nhwc", "pre_perm"], outputs=["short_nchw"]),
        OperatorIR(op_type="LOGISTIC", inputs=["short_nchw"], outputs=["short_sig"]),
        OperatorIR(op_type="MUL", inputs=["short_nchw", "short_sig"], outputs=["short_branch"]),
        OperatorIR(op_type="TRANSPOSE", inputs=["main_nhwc", "pre_perm"], outputs=["main_nchw"]),
        OperatorIR(op_type="TRANSPOSE", inputs=["point_nhwc", "pre_perm"], outputs=["point_nchw"]),
        OperatorIR(op_type="LOGISTIC", inputs=["point_nchw"], outputs=["point_sig"]),
        OperatorIR(op_type="MUL", inputs=["point_nchw", "point_sig"], outputs=["point_branch"]),
        OperatorIR(op_type="ADD", inputs=["main_nchw", "point_branch"], outputs=["add_nchw"]),
        OperatorIR(
            op_type="CONCATENATION",
            inputs=["add_nchw", "short_branch"],
            outputs=["concat_nchw"],
            options={"axis": 1, "fusedActivationFunction": "NONE"},
        ),
        OperatorIR(
            op_type="MEAN",
            inputs=["concat_nchw", "mean_axes"],
            outputs=["mean_nchw"],
            options={"keepDims": True},
        ),
        OperatorIR(op_type="TRANSPOSE", inputs=["mean_nchw", "post_perm"], outputs=["mean_nhwc"]),
        OperatorIR(
            op_type="CONV_2D",
            inputs=["mean_nhwc", "gate_w", "gate_b"],
            outputs=["gate_conv_out_nhwc"],
            options=conv_options,
        ),
        OperatorIR(op_type="TRANSPOSE", inputs=["gate_conv_out_nhwc", "pre_perm"], outputs=["gate_in_nchw"]),
        OperatorIR(op_type="MUL", inputs=["gate_in_nchw", "hs_alpha"], outputs=["gate_mul"]),
        OperatorIR(op_type="ADD", inputs=["gate_mul", "hs_beta"], outputs=["gate_add"]),
        OperatorIR(op_type="RELU_0_TO_1", inputs=["gate_add"], outputs=["gate_nchw"]),
        OperatorIR(op_type="MUL", inputs=["concat_nchw", "gate_nchw"], outputs=["attn_mul_nchw"]),
        OperatorIR(op_type="TRANSPOSE", inputs=["attn_mul_nchw", "post_perm"], outputs=["y_nhwc"]),
        OperatorIR(op_type="RELU", inputs=["y_nhwc"], outputs=["z"]),
    ]

    stats = _optimize_transpose_csp_attention_nhwc_chains(model_ir)
    assert stats["optimized_transpose_csp_attention_nhwc_chains"] == 1

    op_types = [str(op.op_type) for op in model_ir.operators]
    assert op_types.count("TRANSPOSE") == 0
    concat_op = next(op for op in model_ir.operators if str(op.op_type) == "CONCATENATION")
    assert int(concat_op.options.get("axis", -1)) == 3
    np.testing.assert_array_equal(
        np.asarray(model_ir.tensors["mean_axes"].data, dtype=np.int32),
        np.asarray([1, 2], dtype=np.int32),
    )
    conv_op = next(op for op in model_ir.operators if str(op.op_type) == "CONV_2D")
    assert list(conv_op.inputs)[0] == "mean_nchw"
    gate_mul_op = next(op for op in model_ir.operators if list(op.outputs) == ["gate_mul"])
    assert list(gate_mul_op.inputs)[0] == "gate_conv_out_nhwc"
    final_mul_op = next(op for op in model_ir.operators if str(op.op_type) == "MUL" and list(op.outputs) == ["y_nhwc"])
    assert set(list(final_mul_op.inputs)) == {"concat_nchw", "gate_nchw"}
    assert list(model_ir.tensors["concat_nchw"].shape) == [1, 16, 16, 48]


def test_flatbuffer_direct_transpose_csp_attention_nhwc_chain_without_main_add_optimized() -> None:
    model_ir = ModelIR(name="transpose_csp_attention_nhwc_chain_no_add_opt_test")
    model_ir.inputs = ["short_nhwc", "point_nhwc"]
    model_ir.outputs = ["z"]
    model_ir.tensors["short_nhwc"] = TensorIR(
        name="short_nhwc",
        dtype="FLOAT32",
        shape=[1, 20, 20, 192],
        shape_signature=[1, 20, 20, 192],
    )
    model_ir.tensors["point_nhwc"] = TensorIR(
        name="point_nhwc",
        dtype="FLOAT32",
        shape=[1, 20, 20, 192],
        shape_signature=[1, 20, 20, 192],
    )
    model_ir.tensors["pre_perm"] = TensorIR(
        name="pre_perm",
        dtype="INT32",
        shape=[4],
        shape_signature=[4],
        data=np.asarray([0, 3, 1, 2], dtype=np.int32),
        is_variable=False,
    )
    model_ir.tensors["post_perm"] = TensorIR(
        name="post_perm",
        dtype="INT32",
        shape=[4],
        shape_signature=[4],
        data=np.asarray([0, 2, 3, 1], dtype=np.int32),
        is_variable=False,
    )
    model_ir.tensors["mean_axes"] = TensorIR(
        name="mean_axes",
        dtype="INT32",
        shape=[2],
        shape_signature=[2],
        data=np.asarray([2, 3], dtype=np.int32),
        is_variable=False,
    )
    model_ir.tensors["hs_alpha"] = TensorIR(
        name="hs_alpha",
        dtype="FLOAT32",
        shape=[1],
        shape_signature=[1],
        data=np.asarray([1.0 / 6.0], dtype=np.float32),
        is_variable=False,
    )
    model_ir.tensors["hs_beta"] = TensorIR(
        name="hs_beta",
        dtype="FLOAT32",
        shape=[1],
        shape_signature=[1],
        data=np.asarray([0.5], dtype=np.float32),
        is_variable=False,
    )
    model_ir.tensors["gate_w"] = TensorIR(
        name="gate_w",
        dtype="FLOAT32",
        shape=[384, 1, 1, 384],
        shape_signature=[384, 1, 1, 384],
        data=np.ones((384, 1, 1, 384), dtype=np.float32),
        is_variable=False,
    )
    model_ir.tensors["gate_b"] = TensorIR(
        name="gate_b",
        dtype="FLOAT32",
        shape=[384],
        shape_signature=[384],
        data=np.zeros((384,), dtype=np.float32),
        is_variable=False,
    )
    model_ir.tensors["short_nchw"] = TensorIR(
        name="short_nchw",
        dtype="FLOAT32",
        shape=[1, 192, 20, 20],
        shape_signature=[1, 192, 20, 20],
    )
    model_ir.tensors["short_sig"] = TensorIR(
        name="short_sig",
        dtype="FLOAT32",
        shape=[1, 192, 20, 20],
        shape_signature=[1, 192, 20, 20],
    )
    model_ir.tensors["short_branch"] = TensorIR(
        name="short_branch",
        dtype="FLOAT32",
        shape=[1, 192, 20, 20],
        shape_signature=[1, 192, 20, 20],
    )
    model_ir.tensors["point_nchw"] = TensorIR(
        name="point_nchw",
        dtype="FLOAT32",
        shape=[1, 192, 20, 20],
        shape_signature=[1, 192, 20, 20],
    )
    model_ir.tensors["point_sig"] = TensorIR(
        name="point_sig",
        dtype="FLOAT32",
        shape=[1, 192, 20, 20],
        shape_signature=[1, 192, 20, 20],
    )
    model_ir.tensors["point_branch"] = TensorIR(
        name="point_branch",
        dtype="FLOAT32",
        shape=[1, 192, 20, 20],
        shape_signature=[1, 192, 20, 20],
    )
    model_ir.tensors["concat_nchw"] = TensorIR(
        name="concat_nchw",
        dtype="FLOAT32",
        shape=[1, 384, 20, 20],
        shape_signature=[1, 384, 20, 20],
    )
    model_ir.tensors["mean_nchw"] = TensorIR(
        name="mean_nchw",
        dtype="FLOAT32",
        shape=[1, 384, 1, 1],
        shape_signature=[1, 384, 1, 1],
    )
    model_ir.tensors["mean_nhwc"] = TensorIR(
        name="mean_nhwc",
        dtype="FLOAT32",
        shape=[1, 1, 1, 384],
        shape_signature=[1, 1, 1, 384],
    )
    model_ir.tensors["gate_conv_out_nhwc"] = TensorIR(
        name="gate_conv_out_nhwc",
        dtype="FLOAT32",
        shape=[1, 1, 1, 384],
        shape_signature=[1, 1, 1, 384],
    )
    model_ir.tensors["gate_in_nchw"] = TensorIR(
        name="gate_in_nchw",
        dtype="FLOAT32",
        shape=[1, 384, 1, 1],
        shape_signature=[1, 384, 1, 1],
    )
    model_ir.tensors["gate_mul"] = TensorIR(
        name="gate_mul",
        dtype="FLOAT32",
        shape=[1, 384, 1, 1],
        shape_signature=[1, 384, 1, 1],
    )
    model_ir.tensors["gate_add"] = TensorIR(
        name="gate_add",
        dtype="FLOAT32",
        shape=[1, 384, 1, 1],
        shape_signature=[1, 384, 1, 1],
    )
    model_ir.tensors["gate_nchw"] = TensorIR(
        name="gate_nchw",
        dtype="FLOAT32",
        shape=[1, 384, 1, 1],
        shape_signature=[1, 384, 1, 1],
    )
    model_ir.tensors["attn_mul_nchw"] = TensorIR(
        name="attn_mul_nchw",
        dtype="FLOAT32",
        shape=[1, 384, 20, 20],
        shape_signature=[1, 384, 20, 20],
    )
    model_ir.tensors["y_nhwc"] = TensorIR(
        name="y_nhwc",
        dtype="FLOAT32",
        shape=[1, 20, 20, 384],
        shape_signature=[1, 20, 20, 384],
    )
    model_ir.tensors["z"] = TensorIR(
        name="z",
        dtype="FLOAT32",
        shape=[1, 20, 20, 384],
        shape_signature=[1, 20, 20, 384],
    )
    conv_options = {
        "padding": "SAME",
        "strideH": 1,
        "strideW": 1,
        "dilationHFactor": 1,
        "dilationWFactor": 1,
        "fusedActivationFunction": "NONE",
    }
    model_ir.operators = [
        OperatorIR(op_type="TRANSPOSE", inputs=["short_nhwc", "pre_perm"], outputs=["short_nchw"]),
        OperatorIR(op_type="LOGISTIC", inputs=["short_nchw"], outputs=["short_sig"]),
        OperatorIR(op_type="MUL", inputs=["short_nchw", "short_sig"], outputs=["short_branch"]),
        OperatorIR(op_type="TRANSPOSE", inputs=["point_nhwc", "pre_perm"], outputs=["point_nchw"]),
        OperatorIR(op_type="LOGISTIC", inputs=["point_nchw"], outputs=["point_sig"]),
        OperatorIR(op_type="MUL", inputs=["point_nchw", "point_sig"], outputs=["point_branch"]),
        OperatorIR(
            op_type="CONCATENATION",
            inputs=["point_branch", "short_branch"],
            outputs=["concat_nchw"],
            options={"axis": 1, "fusedActivationFunction": "NONE"},
        ),
        OperatorIR(
            op_type="MEAN",
            inputs=["concat_nchw", "mean_axes"],
            outputs=["mean_nchw"],
            options={"keepDims": True},
        ),
        OperatorIR(op_type="TRANSPOSE", inputs=["mean_nchw", "post_perm"], outputs=["mean_nhwc"]),
        OperatorIR(
            op_type="CONV_2D",
            inputs=["mean_nhwc", "gate_w", "gate_b"],
            outputs=["gate_conv_out_nhwc"],
            options=conv_options,
        ),
        OperatorIR(op_type="TRANSPOSE", inputs=["gate_conv_out_nhwc", "pre_perm"], outputs=["gate_in_nchw"]),
        OperatorIR(op_type="MUL", inputs=["gate_in_nchw", "hs_alpha"], outputs=["gate_mul"]),
        OperatorIR(op_type="ADD", inputs=["gate_mul", "hs_beta"], outputs=["gate_add"]),
        OperatorIR(op_type="RELU_0_TO_1", inputs=["gate_add"], outputs=["gate_nchw"]),
        OperatorIR(op_type="MUL", inputs=["concat_nchw", "gate_nchw"], outputs=["attn_mul_nchw"]),
        OperatorIR(op_type="TRANSPOSE", inputs=["attn_mul_nchw", "post_perm"], outputs=["y_nhwc"]),
        OperatorIR(op_type="RELU", inputs=["y_nhwc"], outputs=["z"]),
    ]

    stats = _optimize_transpose_csp_attention_nhwc_chains(model_ir)
    assert stats["optimized_transpose_csp_attention_nhwc_chains"] == 1

    op_types = [str(op.op_type) for op in model_ir.operators]
    assert op_types.count("TRANSPOSE") == 0
    concat_op = next(op for op in model_ir.operators if str(op.op_type) == "CONCATENATION")
    assert int(concat_op.options.get("axis", -1)) == 3
    np.testing.assert_array_equal(
        np.asarray(model_ir.tensors["mean_axes"].data, dtype=np.int32),
        np.asarray([1, 2], dtype=np.int32),
    )
    gate_mul_op = next(op for op in model_ir.operators if list(op.outputs) == ["gate_mul"])
    assert list(gate_mul_op.inputs)[0] == "gate_conv_out_nhwc"
    final_mul_op = next(op for op in model_ir.operators if str(op.op_type) == "MUL" and list(op.outputs) == ["y_nhwc"])
    assert set(list(final_mul_op.inputs)) == {"concat_nchw", "gate_nchw"}
    assert list(model_ir.tensors["concat_nchw"].shape) == [1, 20, 20, 384]


def test_flatbuffer_direct_hardsigmoid_mul_transpose_passthrough_with_legacy_fanout_keeps_adapter() -> None:
    model_ir = ModelIR(name="hardsigmoid_mul_transpose_passthrough_legacy_fanout_test")
    model_ir.inputs = ["x_nhwc"]
    model_ir.outputs = ["y_nhwc", "m_nhwc"]
    model_ir.tensors["x_nhwc"] = TensorIR(
        name="x_nhwc",
        dtype="FLOAT32",
        shape=[1, 4, 4, 8],
        shape_signature=[1, 4, 4, 8],
    )
    model_ir.tensors["pre_perm"] = TensorIR(
        name="pre_perm",
        dtype="INT32",
        shape=[4],
        shape_signature=[4],
        data=np.asarray([0, 3, 1, 2], dtype=np.int32),
        is_variable=False,
    )
    model_ir.tensors["post_perm"] = TensorIR(
        name="post_perm",
        dtype="INT32",
        shape=[4],
        shape_signature=[4],
        data=np.asarray([0, 2, 3, 1], dtype=np.int32),
        is_variable=False,
    )
    model_ir.tensors["x_nchw"] = TensorIR(
        name="x_nchw",
        dtype="FLOAT32",
        shape=[1, 8, 4, 4],
        shape_signature=[1, 8, 4, 4],
    )
    model_ir.tensors["hs_alpha"] = TensorIR(
        name="hs_alpha",
        dtype="FLOAT32",
        shape=[1],
        shape_signature=[1],
        data=np.asarray([1.0 / 6.0], dtype=np.float32),
        is_variable=False,
    )
    model_ir.tensors["hs_beta"] = TensorIR(
        name="hs_beta",
        dtype="FLOAT32",
        shape=[1],
        shape_signature=[1],
        data=np.asarray([0.5], dtype=np.float32),
        is_variable=False,
    )
    model_ir.tensors["hs_mul_out"] = TensorIR(
        name="hs_mul_out",
        dtype="FLOAT32",
        shape=[1, 8, 4, 4],
        shape_signature=[1, 8, 4, 4],
    )
    model_ir.tensors["hs_add_out"] = TensorIR(
        name="hs_add_out",
        dtype="FLOAT32",
        shape=[1, 8, 4, 4],
        shape_signature=[1, 8, 4, 4],
    )
    model_ir.tensors["hs_out"] = TensorIR(
        name="hs_out",
        dtype="FLOAT32",
        shape=[1, 8, 4, 4],
        shape_signature=[1, 8, 4, 4],
    )
    model_ir.tensors["mul_out_nchw"] = TensorIR(
        name="mul_out_nchw",
        dtype="FLOAT32",
        shape=[1, 8, 4, 4],
        shape_signature=[1, 8, 4, 4],
    )
    model_ir.tensors["y_nhwc"] = TensorIR(
        name="y_nhwc",
        dtype="FLOAT32",
        shape=[1, 4, 4, 8],
        shape_signature=[1, 4, 4, 8],
    )
    model_ir.tensors["mean_axes_nchw"] = TensorIR(
        name="mean_axes_nchw",
        dtype="INT32",
        shape=[2],
        shape_signature=[2],
        data=np.asarray([2, 3], dtype=np.int32),
        is_variable=False,
    )
    model_ir.tensors["m_nchw"] = TensorIR(
        name="m_nchw",
        dtype="FLOAT32",
        shape=[1, 8, 1, 1],
        shape_signature=[1, 8, 1, 1],
    )
    model_ir.tensors["m_nhwc"] = TensorIR(
        name="m_nhwc",
        dtype="FLOAT32",
        shape=[1, 1, 1, 8],
        shape_signature=[1, 1, 1, 8],
    )
    model_ir.operators = [
        OperatorIR(op_type="TRANSPOSE", inputs=["x_nhwc", "pre_perm"], outputs=["x_nchw"]),
        OperatorIR(
            op_type="MUL",
            inputs=["x_nchw", "hs_alpha"],
            outputs=["hs_mul_out"],
            options={"fusedActivationFunction": "NONE"},
        ),
        OperatorIR(
            op_type="ADD",
            inputs=["hs_mul_out", "hs_beta"],
            outputs=["hs_add_out"],
            options={"fusedActivationFunction": "NONE"},
        ),
        OperatorIR(op_type="RELU_0_TO_1", inputs=["hs_add_out"], outputs=["hs_out"]),
        OperatorIR(
            op_type="MUL",
            inputs=["x_nchw", "hs_out"],
            outputs=["mul_out_nchw"],
            options={"fusedActivationFunction": "NONE"},
        ),
        OperatorIR(op_type="TRANSPOSE", inputs=["mul_out_nchw", "post_perm"], outputs=["y_nhwc"]),
        OperatorIR(op_type="MEAN", inputs=["mul_out_nchw", "mean_axes_nchw"], outputs=["m_nchw"], options={"keepDims": True}),
        OperatorIR(op_type="TRANSPOSE", inputs=["m_nchw", "post_perm"], outputs=["m_nhwc"]),
    ]

    stats = _optimize_hardsigmoid_mul_transpose_passthrough_chains(model_ir)
    assert stats["rewritten_hardsigmoid_mul_transpose_passthrough_chains"] == 1

    op_types = [str(op.op_type) for op in model_ir.operators]
    # Keep one adapter for the legacy MEAN branch, but drop the original pre/post pair.
    assert op_types.count("TRANSPOSE") == 2
    assert op_types[0] == "MUL"
    hs_mul_op = model_ir.operators[0]
    assert list(hs_mul_op.inputs)[0] == "x_nhwc"
    residual_mul_op = next(op for op in model_ir.operators if str(op.op_type) == "MUL" and list(op.outputs) == ["y_nhwc"])
    assert list(residual_mul_op.inputs)[0] == "x_nhwc"
    mean_op = next(op for op in model_ir.operators if str(op.op_type) == "MEAN")
    assert list(mean_op.inputs)[0] == "mul_out_nchw"
    adapter_op = next(
        op
        for op in model_ir.operators
        if str(op.op_type) == "TRANSPOSE"
        and list(op.outputs) == ["mul_out_nchw"]
    )
    assert list(adapter_op.inputs)[0] == "y_nhwc"


def test_flatbuffer_direct_fold_conv_mul_add_affine_chain_single_path() -> None:
    model_ir = ModelIR(name="fold_conv_mul_add_affine_single_path_test")
    model_ir.inputs = ["x"]
    model_ir.outputs = ["y"]
    model_ir.tensors["x"] = TensorIR(
        name="x",
        dtype="FLOAT32",
        shape=[1, 4, 4, 2],
        shape_signature=[1, 4, 4, 2],
    )
    base_filter = np.asarray(
        [
            [[[1.0, 2.0]]],
            [[[3.0, 4.0]]],
            [[[5.0, 6.0]]],
        ],
        dtype=np.float32,
    )  # [O,H,W,I] = [3,1,1,2]
    base_bias = np.asarray([0.1, 0.2, 0.3], dtype=np.float32)
    mul_coeff = np.asarray([1.0, 2.0, 3.0], dtype=np.float32)
    add_coeff = np.asarray([0.01, 0.02, 0.03], dtype=np.float32)
    model_ir.tensors["w"] = TensorIR(
        name="w",
        dtype="FLOAT32",
        shape=[3, 1, 1, 2],
        shape_signature=[3, 1, 1, 2],
        data=np.asarray(base_filter),
        is_variable=False,
    )
    model_ir.tensors["b"] = TensorIR(
        name="b",
        dtype="FLOAT32",
        shape=[3],
        shape_signature=[3],
        data=np.asarray(base_bias),
        is_variable=False,
    )
    model_ir.tensors["mul_c"] = TensorIR(
        name="mul_c",
        dtype="FLOAT32",
        shape=[3],
        shape_signature=[3],
        data=np.asarray(mul_coeff),
        is_variable=False,
    )
    model_ir.tensors["add_c"] = TensorIR(
        name="add_c",
        dtype="FLOAT32",
        shape=[3],
        shape_signature=[3],
        data=np.asarray(add_coeff),
        is_variable=False,
    )
    model_ir.tensors["conv_out"] = TensorIR(
        name="conv_out",
        dtype="FLOAT32",
        shape=[1, 4, 4, 3],
        shape_signature=[1, 4, 4, 3],
    )
    model_ir.tensors["mul_out"] = TensorIR(
        name="mul_out",
        dtype="FLOAT32",
        shape=[1, 4, 4, 3],
        shape_signature=[1, 4, 4, 3],
    )
    model_ir.tensors["add_out"] = TensorIR(
        name="add_out",
        dtype="FLOAT32",
        shape=[1, 4, 4, 3],
        shape_signature=[1, 4, 4, 3],
    )
    model_ir.tensors["y"] = TensorIR(
        name="y",
        dtype="FLOAT32",
        shape=[1, 4, 4, 3],
        shape_signature=[1, 4, 4, 3],
    )
    conv_options = {
        "padding": "SAME",
        "strideH": 1,
        "strideW": 1,
        "dilationHFactor": 1,
        "dilationWFactor": 1,
        "fusedActivationFunction": "NONE",
    }
    model_ir.operators = [
        OperatorIR(op_type="CONV_2D", inputs=["x", "w", "b"], outputs=["conv_out"], options=conv_options),
        OperatorIR(
            op_type="MUL",
            inputs=["conv_out", "mul_c"],
            outputs=["mul_out"],
            options={"fusedActivationFunction": "NONE"},
        ),
        OperatorIR(
            op_type="ADD",
            inputs=["mul_out", "add_c"],
            outputs=["add_out"],
            options={"fusedActivationFunction": "NONE"},
        ),
        OperatorIR(op_type="RELU", inputs=["add_out"], outputs=["y"]),
    ]

    stats = _optimize_fold_conv_mul_add_affine_chains(model_ir)
    assert stats["folded_conv_mul_add_affine_chains"] == 1

    op_types = [str(op.op_type) for op in model_ir.operators]
    assert op_types == ["CONV_2D", "RELU"]
    conv_op = model_ir.operators[0]
    assert list(conv_op.outputs) == ["add_out"]
    relu_op = model_ir.operators[1]
    assert list(relu_op.inputs) == ["add_out"]

    expected_w = base_filter * mul_coeff.reshape(3, 1, 1, 1)
    expected_b = base_bias * mul_coeff + add_coeff
    assert np.allclose(np.asarray(model_ir.tensors["w"].data), expected_w, rtol=0.0, atol=1e-7)
    assert np.allclose(np.asarray(model_ir.tensors["b"].data), expected_b, rtol=0.0, atol=1e-7)


def test_flatbuffer_direct_fold_conv_mul_add_affine_chain_skips_conv_fanout() -> None:
    model_ir = ModelIR(name="fold_conv_mul_add_affine_conv_fanout_skip_test")
    model_ir.inputs = ["x"]
    model_ir.outputs = ["y", "branch"]
    model_ir.tensors["x"] = TensorIR(
        name="x",
        dtype="FLOAT32",
        shape=[1, 4, 4, 2],
        shape_signature=[1, 4, 4, 2],
    )
    model_ir.tensors["w"] = TensorIR(
        name="w",
        dtype="FLOAT32",
        shape=[3, 1, 1, 2],
        shape_signature=[3, 1, 1, 2],
        data=np.ones((3, 1, 1, 2), dtype=np.float32),
        is_variable=False,
    )
    model_ir.tensors["b"] = TensorIR(
        name="b",
        dtype="FLOAT32",
        shape=[3],
        shape_signature=[3],
        data=np.zeros((3,), dtype=np.float32),
        is_variable=False,
    )
    model_ir.tensors["mul_c"] = TensorIR(
        name="mul_c",
        dtype="FLOAT32",
        shape=[1],
        shape_signature=[1],
        data=np.asarray([2.0], dtype=np.float32),
        is_variable=False,
    )
    model_ir.tensors["add_c"] = TensorIR(
        name="add_c",
        dtype="FLOAT32",
        shape=[1],
        shape_signature=[1],
        data=np.asarray([0.5], dtype=np.float32),
        is_variable=False,
    )
    for name in ["conv_out", "mul_out", "add_out", "y", "branch"]:
        model_ir.tensors[name] = TensorIR(
            name=name,
            dtype="FLOAT32",
            shape=[1, 4, 4, 3],
            shape_signature=[1, 4, 4, 3],
        )
    conv_options = {
        "padding": "SAME",
        "strideH": 1,
        "strideW": 1,
        "dilationHFactor": 1,
        "dilationWFactor": 1,
        "fusedActivationFunction": "NONE",
    }
    model_ir.operators = [
        OperatorIR(op_type="CONV_2D", inputs=["x", "w", "b"], outputs=["conv_out"], options=conv_options),
        OperatorIR(
            op_type="MUL",
            inputs=["conv_out", "mul_c"],
            outputs=["mul_out"],
            options={"fusedActivationFunction": "NONE"},
        ),
        OperatorIR(
            op_type="ADD",
            inputs=["mul_out", "add_c"],
            outputs=["add_out"],
            options={"fusedActivationFunction": "NONE"},
        ),
        OperatorIR(op_type="RELU", inputs=["add_out"], outputs=["y"]),
        OperatorIR(op_type="RELU", inputs=["conv_out"], outputs=["branch"]),
    ]

    stats = _optimize_fold_conv_mul_add_affine_chains(model_ir)
    assert stats["folded_conv_mul_add_affine_chains"] == 0
    op_types = [str(op.op_type) for op in model_ir.operators]
    assert op_types == ["CONV_2D", "MUL", "ADD", "RELU", "RELU"]


def test_flatbuffer_direct_fold_conv_mul_affine_chain_single_path() -> None:
    model_ir = ModelIR(name="fold_conv_mul_affine_single_path_test")
    model_ir.inputs = ["x"]
    model_ir.outputs = ["y"]
    model_ir.tensors["x"] = TensorIR(
        name="x",
        dtype="FLOAT32",
        shape=[1, 4, 4, 2],
        shape_signature=[1, 4, 4, 2],
    )
    base_filter = np.asarray(
        [
            [[[1.0, 2.0]]],
            [[[3.0, 4.0]]],
            [[[5.0, 6.0]]],
        ],
        dtype=np.float32,
    )  # [O,H,W,I] = [3,1,1,2]
    base_bias = np.asarray([0.1, 0.2, 0.3], dtype=np.float32)
    mul_coeff = np.asarray([2.0, 3.0, 4.0], dtype=np.float32)
    model_ir.tensors["w"] = TensorIR(
        name="w",
        dtype="FLOAT32",
        shape=[3, 1, 1, 2],
        shape_signature=[3, 1, 1, 2],
        data=np.asarray(base_filter),
        is_variable=False,
    )
    model_ir.tensors["b"] = TensorIR(
        name="b",
        dtype="FLOAT32",
        shape=[3],
        shape_signature=[3],
        data=np.asarray(base_bias),
        is_variable=False,
    )
    model_ir.tensors["mul_c"] = TensorIR(
        name="mul_c",
        dtype="FLOAT32",
        shape=[1, 1, 3],
        shape_signature=[1, 1, 3],
        data=np.asarray(mul_coeff).reshape(1, 1, 3),
        is_variable=False,
    )
    model_ir.tensors["conv_out"] = TensorIR(
        name="conv_out",
        dtype="FLOAT32",
        shape=[1, 4, 4, 3],
        shape_signature=[1, 4, 4, 3],
    )
    model_ir.tensors["mul_out"] = TensorIR(
        name="mul_out",
        dtype="FLOAT32",
        shape=[1, 4, 4, 3],
        shape_signature=[1, 4, 4, 3],
    )
    model_ir.tensors["y"] = TensorIR(
        name="y",
        dtype="FLOAT32",
        shape=[1, 4, 4, 3],
        shape_signature=[1, 4, 4, 3],
    )
    conv_options = {
        "padding": "SAME",
        "strideH": 1,
        "strideW": 1,
        "dilationHFactor": 1,
        "dilationWFactor": 1,
        "fusedActivationFunction": "NONE",
    }
    model_ir.operators = [
        OperatorIR(op_type="CONV_2D", inputs=["x", "w", "b"], outputs=["conv_out"], options=conv_options),
        OperatorIR(
            op_type="MUL",
            inputs=["conv_out", "mul_c"],
            outputs=["mul_out"],
            options={"fusedActivationFunction": "NONE"},
        ),
        OperatorIR(op_type="RELU", inputs=["mul_out"], outputs=["y"]),
    ]

    stats = _optimize_fold_conv_mul_add_affine_chains(model_ir)
    assert stats["folded_conv_mul_add_affine_chains"] == 1
    assert stats["folded_conv_mul_only_affine_chains"] == 1
    assert stats["folded_conv_mul_add_only_affine_chains"] == 0

    op_types = [str(op.op_type) for op in model_ir.operators]
    assert op_types == ["CONV_2D", "RELU"]
    conv_op = model_ir.operators[0]
    assert list(conv_op.outputs) == ["mul_out"]
    relu_op = model_ir.operators[1]
    assert list(relu_op.inputs) == ["mul_out"]

    expected_w = base_filter * mul_coeff.reshape(3, 1, 1, 1)
    expected_b = base_bias * mul_coeff
    assert np.allclose(np.asarray(model_ir.tensors["w"].data), expected_w, rtol=0.0, atol=1e-7)
    assert np.allclose(np.asarray(model_ir.tensors["b"].data), expected_b, rtol=0.0, atol=1e-7)


def test_flatbuffer_direct_fold_conv_relu_mul_add_affine_chain_folds_mul_only() -> None:
    model_ir = ModelIR(name="fold_conv_relu_mul_add_affine_mul_only_test")
    model_ir.inputs = ["x"]
    model_ir.outputs = ["y"]
    model_ir.tensors["x"] = TensorIR(
        name="x",
        dtype="FLOAT32",
        shape=[1, 4, 4, 2],
        shape_signature=[1, 4, 4, 2],
    )
    base_filter = np.asarray(
        [
            [[[1.0, 2.0]]],
            [[[3.0, 4.0]]],
            [[[5.0, 6.0]]],
        ],
        dtype=np.float32,
    )
    base_bias = np.asarray([0.1, 0.2, 0.3], dtype=np.float32)
    mul_coeff = np.asarray([2.0, 3.0, 4.0], dtype=np.float32)
    add_coeff = np.asarray([0.5, 0.25, 0.125], dtype=np.float32)
    model_ir.tensors["w"] = TensorIR(
        name="w",
        dtype="FLOAT32",
        shape=[3, 1, 1, 2],
        shape_signature=[3, 1, 1, 2],
        data=np.asarray(base_filter),
        is_variable=False,
    )
    model_ir.tensors["b"] = TensorIR(
        name="b",
        dtype="FLOAT32",
        shape=[3],
        shape_signature=[3],
        data=np.asarray(base_bias),
        is_variable=False,
    )
    model_ir.tensors["mul_c"] = TensorIR(
        name="mul_c",
        dtype="FLOAT32",
        shape=[1, 1, 1, 3],
        shape_signature=[1, 1, 1, 3],
        data=np.asarray(mul_coeff).reshape(1, 1, 1, 3),
        is_variable=False,
    )
    model_ir.tensors["add_c"] = TensorIR(
        name="add_c",
        dtype="FLOAT32",
        shape=[3],
        shape_signature=[3],
        data=np.asarray(add_coeff),
        is_variable=False,
    )
    for name in ["conv_out", "mul_out", "add_out", "y"]:
        model_ir.tensors[name] = TensorIR(
            name=name,
            dtype="FLOAT32",
            shape=[1, 4, 4, 3],
            shape_signature=[1, 4, 4, 3],
        )
    conv_options = {
        "padding": "SAME",
        "strideH": 1,
        "strideW": 1,
        "dilationHFactor": 1,
        "dilationWFactor": 1,
        "fusedActivationFunction": "RELU",
    }
    model_ir.operators = [
        OperatorIR(op_type="CONV_2D", inputs=["x", "w", "b"], outputs=["conv_out"], options=conv_options),
        OperatorIR(
            op_type="MUL",
            inputs=["conv_out", "mul_c"],
            outputs=["mul_out"],
            options={"fusedActivationFunction": "NONE"},
        ),
        OperatorIR(
            op_type="ADD",
            inputs=["mul_out", "add_c"],
            outputs=["add_out"],
            options={"fusedActivationFunction": "NONE"},
        ),
        OperatorIR(op_type="RELU", inputs=["add_out"], outputs=["y"]),
    ]

    stats = _optimize_fold_conv_mul_add_affine_chains(model_ir)
    assert stats["folded_conv_mul_add_affine_chains"] == 1
    assert stats["folded_conv_mul_only_affine_chains"] == 1
    assert stats["folded_conv_mul_add_only_affine_chains"] == 0

    op_types = [str(op.op_type) for op in model_ir.operators]
    assert op_types == ["CONV_2D", "ADD", "RELU"]
    conv_op = model_ir.operators[0]
    assert list(conv_op.outputs) == ["mul_out"]
    add_op = model_ir.operators[1]
    assert list(add_op.inputs) == ["mul_out", "add_c"]

    expected_w = base_filter * mul_coeff.reshape(3, 1, 1, 1)
    expected_b = base_bias * mul_coeff
    assert np.allclose(np.asarray(model_ir.tensors["w"].data), expected_w, rtol=0.0, atol=1e-7)
    assert np.allclose(np.asarray(model_ir.tensors["b"].data), expected_b, rtol=0.0, atol=1e-7)


def test_flatbuffer_direct_fold_conv_relu_mul_affine_chain_skips_negative_scale() -> None:
    model_ir = ModelIR(name="fold_conv_relu_mul_affine_negative_scale_skip_test")
    model_ir.inputs = ["x"]
    model_ir.outputs = ["y"]
    model_ir.tensors["x"] = TensorIR(
        name="x",
        dtype="FLOAT32",
        shape=[1, 4, 4, 2],
        shape_signature=[1, 4, 4, 2],
    )
    model_ir.tensors["w"] = TensorIR(
        name="w",
        dtype="FLOAT32",
        shape=[3, 1, 1, 2],
        shape_signature=[3, 1, 1, 2],
        data=np.ones((3, 1, 1, 2), dtype=np.float32),
        is_variable=False,
    )
    model_ir.tensors["b"] = TensorIR(
        name="b",
        dtype="FLOAT32",
        shape=[3],
        shape_signature=[3],
        data=np.zeros((3,), dtype=np.float32),
        is_variable=False,
    )
    model_ir.tensors["mul_c"] = TensorIR(
        name="mul_c",
        dtype="FLOAT32",
        shape=[3],
        shape_signature=[3],
        data=np.asarray([1.0, -1.0, 2.0], dtype=np.float32),
        is_variable=False,
    )
    for name in ["conv_out", "mul_out", "y"]:
        model_ir.tensors[name] = TensorIR(
            name=name,
            dtype="FLOAT32",
            shape=[1, 4, 4, 3],
            shape_signature=[1, 4, 4, 3],
        )
    conv_options = {
        "padding": "SAME",
        "strideH": 1,
        "strideW": 1,
        "dilationHFactor": 1,
        "dilationWFactor": 1,
        "fusedActivationFunction": "RELU",
    }
    model_ir.operators = [
        OperatorIR(op_type="CONV_2D", inputs=["x", "w", "b"], outputs=["conv_out"], options=conv_options),
        OperatorIR(
            op_type="MUL",
            inputs=["conv_out", "mul_c"],
            outputs=["mul_out"],
            options={"fusedActivationFunction": "NONE"},
        ),
        OperatorIR(op_type="RELU", inputs=["mul_out"], outputs=["y"]),
    ]

    stats = _optimize_fold_conv_mul_add_affine_chains(model_ir)
    assert stats["folded_conv_mul_add_affine_chains"] == 0
    op_types = [str(op.op_type) for op in model_ir.operators]
    assert op_types == ["CONV_2D", "MUL", "RELU"]


def test_flatbuffer_direct_fold_conv_add_affine_chain_single_path() -> None:
    model_ir = ModelIR(name="fold_conv_add_affine_single_path_test")
    model_ir.inputs = ["x"]
    model_ir.outputs = ["y"]
    model_ir.tensors["x"] = TensorIR(
        name="x",
        dtype="FLOAT32",
        shape=[1, 4, 4, 2],
        shape_signature=[1, 4, 4, 2],
    )
    base_filter = np.asarray(
        [
            [[[1.0, 2.0]]],
            [[[3.0, 4.0]]],
            [[[5.0, 6.0]]],
        ],
        dtype=np.float32,
    )  # [O,H,W,I] = [3,1,1,2]
    base_bias = np.asarray([0.1, 0.2, 0.3], dtype=np.float32)
    add_coeff = np.asarray([0.5, 0.25, 0.125], dtype=np.float32)
    model_ir.tensors["w"] = TensorIR(
        name="w",
        dtype="FLOAT32",
        shape=[3, 1, 1, 2],
        shape_signature=[3, 1, 1, 2],
        data=np.asarray(base_filter),
        is_variable=False,
    )
    model_ir.tensors["b"] = TensorIR(
        name="b",
        dtype="FLOAT32",
        shape=[3],
        shape_signature=[3],
        data=np.asarray(base_bias),
        is_variable=False,
    )
    model_ir.tensors["add_c"] = TensorIR(
        name="add_c",
        dtype="FLOAT32",
        shape=[1, 1, 3],
        shape_signature=[1, 1, 3],
        data=np.asarray(add_coeff).reshape(1, 1, 3),
        is_variable=False,
    )
    model_ir.tensors["conv_out"] = TensorIR(
        name="conv_out",
        dtype="FLOAT32",
        shape=[1, 4, 4, 3],
        shape_signature=[1, 4, 4, 3],
    )
    model_ir.tensors["add_out"] = TensorIR(
        name="add_out",
        dtype="FLOAT32",
        shape=[1, 4, 4, 3],
        shape_signature=[1, 4, 4, 3],
    )
    model_ir.tensors["y"] = TensorIR(
        name="y",
        dtype="FLOAT32",
        shape=[1, 4, 4, 3],
        shape_signature=[1, 4, 4, 3],
    )
    conv_options = {
        "padding": "SAME",
        "strideH": 1,
        "strideW": 1,
        "dilationHFactor": 1,
        "dilationWFactor": 1,
        "fusedActivationFunction": "NONE",
    }
    model_ir.operators = [
        OperatorIR(op_type="CONV_2D", inputs=["x", "w", "b"], outputs=["conv_out"], options=conv_options),
        OperatorIR(
            op_type="ADD",
            inputs=["conv_out", "add_c"],
            outputs=["add_out"],
            options={"fusedActivationFunction": "NONE"},
        ),
        OperatorIR(op_type="RELU", inputs=["add_out"], outputs=["y"]),
    ]

    stats = _optimize_fold_conv_mul_add_affine_chains(model_ir)
    assert stats["folded_conv_mul_add_affine_chains"] == 1
    assert stats["folded_conv_add_only_affine_chains"] == 1
    assert stats["folded_conv_mul_only_affine_chains"] == 0
    assert stats["folded_conv_mul_add_only_affine_chains"] == 0

    op_types = [str(op.op_type) for op in model_ir.operators]
    assert op_types == ["CONV_2D", "RELU"]
    conv_op = model_ir.operators[0]
    assert list(conv_op.outputs) == ["add_out"]
    relu_op = model_ir.operators[1]
    assert list(relu_op.inputs) == ["add_out"]

    expected_w = base_filter
    expected_b = base_bias + add_coeff
    assert np.allclose(np.asarray(model_ir.tensors["w"].data), expected_w, rtol=0.0, atol=1e-7)
    assert np.allclose(np.asarray(model_ir.tensors["b"].data), expected_b, rtol=0.0, atol=1e-7)


def test_flatbuffer_direct_fold_conv_add_affine_chain_skips_add_fanout() -> None:
    model_ir = ModelIR(name="fold_conv_add_affine_add_fanout_skip_test")
    model_ir.inputs = ["x"]
    model_ir.outputs = ["y0", "y1"]
    model_ir.tensors["x"] = TensorIR(
        name="x",
        dtype="FLOAT32",
        shape=[1, 4, 4, 2],
        shape_signature=[1, 4, 4, 2],
    )
    model_ir.tensors["w"] = TensorIR(
        name="w",
        dtype="FLOAT32",
        shape=[3, 1, 1, 2],
        shape_signature=[3, 1, 1, 2],
        data=np.ones((3, 1, 1, 2), dtype=np.float32),
        is_variable=False,
    )
    model_ir.tensors["b"] = TensorIR(
        name="b",
        dtype="FLOAT32",
        shape=[3],
        shape_signature=[3],
        data=np.zeros((3,), dtype=np.float32),
        is_variable=False,
    )
    model_ir.tensors["add_c"] = TensorIR(
        name="add_c",
        dtype="FLOAT32",
        shape=[1],
        shape_signature=[1],
        data=np.asarray([0.5], dtype=np.float32),
        is_variable=False,
    )
    for name in ["conv_out", "add_out", "y0", "y1"]:
        model_ir.tensors[name] = TensorIR(
            name=name,
            dtype="FLOAT32",
            shape=[1, 4, 4, 3],
            shape_signature=[1, 4, 4, 3],
        )
    conv_options = {
        "padding": "SAME",
        "strideH": 1,
        "strideW": 1,
        "dilationHFactor": 1,
        "dilationWFactor": 1,
        "fusedActivationFunction": "NONE",
    }
    model_ir.operators = [
        OperatorIR(op_type="CONV_2D", inputs=["x", "w", "b"], outputs=["conv_out"], options=conv_options),
        OperatorIR(
            op_type="ADD",
            inputs=["conv_out", "add_c"],
            outputs=["add_out"],
            options={"fusedActivationFunction": "NONE"},
        ),
        OperatorIR(op_type="RELU", inputs=["add_out"], outputs=["y0"]),
        OperatorIR(op_type="RELU6", inputs=["add_out"], outputs=["y1"]),
    ]

    stats = _optimize_fold_conv_mul_add_affine_chains(model_ir)
    assert stats["folded_conv_mul_add_affine_chains"] == 0
    assert stats["folded_conv_add_only_affine_chains"] == 0
    op_types = [str(op.op_type) for op in model_ir.operators]
    assert op_types == ["CONV_2D", "ADD", "RELU", "RELU6"]


def test_flatbuffer_direct_fold_conv_add_affine_chain_skips_leading_channel_coeff_shape() -> None:
    model_ir = ModelIR(name="fold_conv_add_affine_leading_channel_coeff_skip_test")
    model_ir.inputs = ["x"]
    model_ir.outputs = ["y"]
    model_ir.tensors["x"] = TensorIR(
        name="x",
        dtype="FLOAT32",
        shape=[1, 4, 4, 2],
        shape_signature=[1, 4, 4, 2],
    )
    model_ir.tensors["w"] = TensorIR(
        name="w",
        dtype="FLOAT32",
        shape=[3, 1, 1, 2],
        shape_signature=[3, 1, 1, 2],
        data=np.ones((3, 1, 1, 2), dtype=np.float32),
        is_variable=False,
    )
    model_ir.tensors["b"] = TensorIR(
        name="b",
        dtype="FLOAT32",
        shape=[3],
        shape_signature=[3],
        data=np.zeros((3,), dtype=np.float32),
        is_variable=False,
    )
    # [C,1,1] is not a valid NHWC trailing-channel broadcast form.
    # Treating this as channelwise can misfold NCHW-intent constants.
    model_ir.tensors["add_c"] = TensorIR(
        name="add_c",
        dtype="FLOAT32",
        shape=[3, 1, 1],
        shape_signature=[3, 1, 1],
        data=np.asarray([0.5, 0.25, 0.125], dtype=np.float32).reshape(3, 1, 1),
        is_variable=False,
    )
    for name in ["conv_out", "add_out", "y"]:
        model_ir.tensors[name] = TensorIR(
            name=name,
            dtype="FLOAT32",
            shape=[1, 4, 4, 3],
            shape_signature=[1, 4, 4, 3],
        )
    conv_options = {
        "padding": "SAME",
        "strideH": 1,
        "strideW": 1,
        "dilationHFactor": 1,
        "dilationWFactor": 1,
        "fusedActivationFunction": "NONE",
    }
    model_ir.operators = [
        OperatorIR(op_type="CONV_2D", inputs=["x", "w", "b"], outputs=["conv_out"], options=conv_options),
        OperatorIR(
            op_type="ADD",
            inputs=["conv_out", "add_c"],
            outputs=["add_out"],
            options={"fusedActivationFunction": "NONE"},
        ),
        OperatorIR(op_type="RELU", inputs=["add_out"], outputs=["y"]),
    ]

    stats = _optimize_fold_conv_mul_add_affine_chains(model_ir)
    assert stats["folded_conv_mul_add_affine_chains"] == 0
    assert stats["folded_conv_add_only_affine_chains"] == 0
    op_types = [str(op.op_type) for op in model_ir.operators]
    assert op_types == ["CONV_2D", "ADD", "RELU"]


def test_flatbuffer_direct_fold_conv_add_affine_chain_folds_relu6_muldiv_suffix() -> None:
    model_ir = ModelIR(name="fold_conv_add_affine_relu6_muldiv_suffix_fold_test")
    model_ir.inputs = ["x"]
    model_ir.outputs = ["y"]
    model_ir.tensors["x"] = TensorIR(
        name="x",
        dtype="FLOAT32",
        shape=[1, 4, 4, 2],
        shape_signature=[1, 4, 4, 2],
    )
    model_ir.tensors["w"] = TensorIR(
        name="w",
        dtype="FLOAT32",
        shape=[3, 1, 1, 2],
        shape_signature=[3, 1, 1, 2],
        data=np.ones((3, 1, 1, 2), dtype=np.float32),
        is_variable=False,
    )
    model_ir.tensors["b"] = TensorIR(
        name="b",
        dtype="FLOAT32",
        shape=[3],
        shape_signature=[3],
        data=np.zeros((3,), dtype=np.float32),
        is_variable=False,
    )
    model_ir.tensors["add_c"] = TensorIR(
        name="add_c",
        dtype="FLOAT32",
        shape=[1],
        shape_signature=[1],
        data=np.asarray([0.5], dtype=np.float32),
        is_variable=False,
    )
    model_ir.tensors["div_c"] = TensorIR(
        name="div_c",
        dtype="FLOAT32",
        shape=[1],
        shape_signature=[1],
        data=np.asarray([6.0], dtype=np.float32),
        is_variable=False,
    )
    model_ir.tensors["mul_c"] = TensorIR(
        name="mul_c",
        dtype="FLOAT32",
        shape=[1],
        shape_signature=[1],
        data=np.asarray([0.25], dtype=np.float32),
        is_variable=False,
    )
    for name in ["conv_out", "add_out", "relu6_out", "div_out", "y"]:
        model_ir.tensors[name] = TensorIR(
            name=name,
            dtype="FLOAT32",
            shape=[1, 4, 4, 3],
            shape_signature=[1, 4, 4, 3],
        )
    conv_options = {
        "padding": "SAME",
        "strideH": 1,
        "strideW": 1,
        "dilationHFactor": 1,
        "dilationWFactor": 1,
        "fusedActivationFunction": "NONE",
    }
    model_ir.operators = [
        OperatorIR(op_type="CONV_2D", inputs=["x", "w", "b"], outputs=["conv_out"], options=conv_options),
        OperatorIR(
            op_type="ADD",
            inputs=["conv_out", "add_c"],
            outputs=["add_out"],
            options={"fusedActivationFunction": "NONE"},
        ),
        OperatorIR(op_type="RELU6", inputs=["add_out"], outputs=["relu6_out"]),
        OperatorIR(
            op_type="DIV",
            inputs=["relu6_out", "div_c"],
            outputs=["div_out"],
            options={"fusedActivationFunction": "NONE"},
        ),
        OperatorIR(
            op_type="MUL",
            inputs=["div_out", "mul_c"],
            outputs=["y"],
            options={"fusedActivationFunction": "NONE"},
        ),
    ]

    stats = _optimize_fold_conv_mul_add_affine_chains(model_ir)
    assert stats["folded_conv_mul_add_affine_chains"] == 1
    assert stats["folded_conv_add_only_affine_chains"] == 1
    op_types = [str(op.op_type) for op in model_ir.operators]
    assert op_types == ["CONV_2D", "RELU6", "DIV", "MUL"]
    conv_op = model_ir.operators[0]
    assert list(conv_op.outputs) == ["add_out"]


def test_flatbuffer_direct_fold_conv_add_affine_chain_folds_fused_add_activation() -> None:
    model_ir = ModelIR(name="fold_conv_add_affine_fused_add_activation_fold_test")
    model_ir.inputs = ["x"]
    model_ir.outputs = ["y"]
    model_ir.tensors["x"] = TensorIR(
        name="x",
        dtype="FLOAT32",
        shape=[1, 4, 4, 2],
        shape_signature=[1, 4, 4, 2],
    )
    model_ir.tensors["w"] = TensorIR(
        name="w",
        dtype="FLOAT32",
        shape=[3, 1, 1, 2],
        shape_signature=[3, 1, 1, 2],
        data=np.ones((3, 1, 1, 2), dtype=np.float32),
        is_variable=False,
    )
    model_ir.tensors["b"] = TensorIR(
        name="b",
        dtype="FLOAT32",
        shape=[3],
        shape_signature=[3],
        data=np.zeros((3,), dtype=np.float32),
        is_variable=False,
    )
    model_ir.tensors["add_c"] = TensorIR(
        name="add_c",
        dtype="FLOAT32",
        shape=[1],
        shape_signature=[1],
        data=np.asarray([0.5], dtype=np.float32),
        is_variable=False,
    )
    model_ir.tensors["mul_c"] = TensorIR(
        name="mul_c",
        dtype="FLOAT32",
        shape=[1],
        shape_signature=[1],
        data=np.asarray([0.25], dtype=np.float32),
        is_variable=False,
    )
    for name in ["conv_out", "add_out", "y"]:
        model_ir.tensors[name] = TensorIR(
            name=name,
            dtype="FLOAT32",
            shape=[1, 4, 4, 3],
            shape_signature=[1, 4, 4, 3],
        )
    conv_options = {
        "padding": "SAME",
        "strideH": 1,
        "strideW": 1,
        "dilationHFactor": 1,
        "dilationWFactor": 1,
        "fusedActivationFunction": "NONE",
    }
    model_ir.operators = [
        OperatorIR(op_type="CONV_2D", inputs=["x", "w", "b"], outputs=["conv_out"], options=conv_options),
        OperatorIR(
            op_type="ADD",
            inputs=["conv_out", "add_c"],
            outputs=["add_out"],
            options={"fusedActivationFunction": "RELU6"},
        ),
        OperatorIR(
            op_type="MUL",
            inputs=["add_out", "mul_c"],
            outputs=["y"],
            options={"fusedActivationFunction": "NONE"},
        ),
    ]

    stats = _optimize_fold_conv_mul_add_affine_chains(model_ir)
    assert stats["folded_conv_mul_add_affine_chains"] == 1
    assert stats["folded_conv_add_only_affine_chains"] == 1
    op_types = [str(op.op_type) for op in model_ir.operators]
    assert op_types == ["CONV_2D", "MUL"]
    conv_op = model_ir.operators[0]
    conv_opts = dict(conv_op.options) if isinstance(conv_op.options, dict) else {}
    assert str(conv_opts.get("fusedActivationFunction", "NONE")) == "RELU6"
    assert list(conv_op.outputs) == ["add_out"]


def test_flatbuffer_direct_asin_transpose_passthrough_chain() -> None:
    model_ir = ModelIR(name="asin_transpose_passthrough_test")
    model_ir.inputs = ["x_nhwc"]
    model_ir.outputs = ["y"]
    model_ir.tensors["x_nhwc"] = TensorIR(
        name="x_nhwc",
        dtype="FLOAT32",
        shape=[1, 4, 4, 3],
        shape_signature=[1, 4, 4, 3],
    )
    model_ir.tensors["perm"] = TensorIR(
        name="perm",
        dtype="INT32",
        shape=[4],
        shape_signature=[4],
        data=np.asarray([0, 3, 1, 2], dtype=np.int32),
        is_variable=False,
    )
    model_ir.tensors["x_nchw"] = TensorIR(
        name="x_nchw",
        dtype="FLOAT32",
        shape=[1, 3, 4, 4],
        shape_signature=[1, 3, 4, 4],
    )
    model_ir.tensors["one"] = TensorIR(
        name="one",
        dtype="FLOAT32",
        shape=[],
        shape_signature=[],
        data=np.asarray(1.0, dtype=np.float32),
        is_variable=False,
    )
    for name in ["x_sq", "one_minus_x_sq", "denom", "y"]:
        model_ir.tensors[name] = TensorIR(
            name=name,
            dtype="FLOAT32",
            shape=[1, 3, 4, 4],
            shape_signature=[1, 3, 4, 4],
        )

    model_ir.operators = [
        OperatorIR(op_type="TRANSPOSE", inputs=["x_nhwc", "perm"], outputs=["x_nchw"]),
        OperatorIR(
            op_type="MUL",
            inputs=["x_nchw", "x_nchw"],
            outputs=["x_sq"],
            options={"fusedActivationFunction": "NONE"},
        ),
        OperatorIR(
            op_type="SUB",
            inputs=["one", "x_sq"],
            outputs=["one_minus_x_sq"],
            options={"fusedActivationFunction": "NONE"},
        ),
        OperatorIR(op_type="SQRT", inputs=["one_minus_x_sq"], outputs=["denom"]),
        OperatorIR(op_type="ATAN2", inputs=["x_nchw", "denom"], outputs=["y"]),
    ]

    stats = _optimize_asin_transpose_passthrough_chains(model_ir)
    assert stats["rewritten_asin_transpose_passthrough_chains"] == 1
    op_types = [str(op.op_type) for op in model_ir.operators]
    assert op_types == ["MUL", "SUB", "SQRT", "ATAN2"]
    mul_op = model_ir.operators[0]
    assert list(mul_op.inputs) == ["x_nhwc", "x_nhwc"]
    atan2_op = model_ir.operators[-1]
    assert "x_nhwc" in list(atan2_op.inputs)
    assert list(atan2_op.outputs) == ["y"]
    assert list(model_ir.tensors["y"].shape) == [1, 4, 4, 3]


def test_flatbuffer_direct_erf_transpose_passthrough_chain() -> None:
    model_ir = ModelIR(name="erf_transpose_passthrough_test")
    model_ir.inputs = ["x_nhwc"]
    model_ir.outputs = ["y"]
    model_ir.tensors["x_nhwc"] = TensorIR(
        name="x_nhwc",
        dtype="FLOAT32",
        shape=[1, 4, 4, 3],
        shape_signature=[1, 4, 4, 3],
    )
    model_ir.tensors["perm"] = TensorIR(
        name="perm",
        dtype="INT32",
        shape=[4],
        shape_signature=[4],
        data=np.asarray([0, 3, 1, 2], dtype=np.int32),
        is_variable=False,
    )
    for name in [
        "x_nchw",
        "abs_out",
        "sign_out",
        "px",
        "one_plus_px",
        "t_out",
        "abs_sq",
        "neg_abs_sq",
        "exp_out",
        "s1_mul",
        "s1_add",
        "s2_mul",
        "s2_add",
        "s3_mul",
        "s3_add",
        "s4_mul",
        "s4_add",
        "poly_out",
        "poly_exp",
        "one_minus",
        "y",
    ]:
        model_ir.tensors[name] = TensorIR(
            name=name,
            dtype="FLOAT32",
            shape=[1, 3, 4, 4],
            shape_signature=[1, 3, 4, 4],
        )
    for c_name, c_val in [
        ("one", 1.0),
        ("p", 0.3275911),
        ("minus_one", -1.0),
        ("a5", 1.061405429),
        ("a4", -1.453152027),
        ("a3", 1.421413741),
        ("a2", -0.284496736),
        ("a1", 0.254829592),
    ]:
        model_ir.tensors[c_name] = TensorIR(
            name=c_name,
            dtype="FLOAT32",
            shape=[],
            shape_signature=[],
            data=np.asarray(c_val, dtype=np.float32),
            is_variable=False,
        )

    model_ir.operators = [
        OperatorIR(op_type="TRANSPOSE", inputs=["x_nhwc", "perm"], outputs=["x_nchw"]),
        OperatorIR(op_type="ABS", inputs=["x_nchw"], outputs=["abs_out"]),
        OperatorIR(op_type="SIGN", inputs=["x_nchw"], outputs=["sign_out"]),
        OperatorIR(op_type="MUL", inputs=["abs_out", "p"], outputs=["px"], options={"fusedActivationFunction": "NONE"}),
        OperatorIR(op_type="ADD", inputs=["one", "px"], outputs=["one_plus_px"], options={"fusedActivationFunction": "NONE"}),
        OperatorIR(op_type="DIV", inputs=["one", "one_plus_px"], outputs=["t_out"], options={"fusedActivationFunction": "NONE"}),
        OperatorIR(op_type="MUL", inputs=["abs_out", "abs_out"], outputs=["abs_sq"], options={"fusedActivationFunction": "NONE"}),
        OperatorIR(op_type="MUL", inputs=["abs_sq", "minus_one"], outputs=["neg_abs_sq"], options={"fusedActivationFunction": "NONE"}),
        OperatorIR(op_type="EXP", inputs=["neg_abs_sq"], outputs=["exp_out"]),
        OperatorIR(op_type="MUL", inputs=["a5", "t_out"], outputs=["s1_mul"], options={"fusedActivationFunction": "NONE"}),
        OperatorIR(op_type="ADD", inputs=["s1_mul", "a4"], outputs=["s1_add"], options={"fusedActivationFunction": "NONE"}),
        OperatorIR(op_type="MUL", inputs=["s1_add", "t_out"], outputs=["s2_mul"], options={"fusedActivationFunction": "NONE"}),
        OperatorIR(op_type="ADD", inputs=["s2_mul", "a3"], outputs=["s2_add"], options={"fusedActivationFunction": "NONE"}),
        OperatorIR(op_type="MUL", inputs=["s2_add", "t_out"], outputs=["s3_mul"], options={"fusedActivationFunction": "NONE"}),
        OperatorIR(op_type="ADD", inputs=["s3_mul", "a2"], outputs=["s3_add"], options={"fusedActivationFunction": "NONE"}),
        OperatorIR(op_type="MUL", inputs=["s3_add", "t_out"], outputs=["s4_mul"], options={"fusedActivationFunction": "NONE"}),
        OperatorIR(op_type="ADD", inputs=["s4_mul", "a1"], outputs=["s4_add"], options={"fusedActivationFunction": "NONE"}),
        OperatorIR(op_type="MUL", inputs=["s4_add", "t_out"], outputs=["poly_out"], options={"fusedActivationFunction": "NONE"}),
        OperatorIR(op_type="MUL", inputs=["poly_out", "exp_out"], outputs=["poly_exp"], options={"fusedActivationFunction": "NONE"}),
        OperatorIR(op_type="SUB", inputs=["one", "poly_exp"], outputs=["one_minus"], options={"fusedActivationFunction": "NONE"}),
        OperatorIR(op_type="MUL", inputs=["sign_out", "one_minus"], outputs=["y"], options={"fusedActivationFunction": "NONE"}),
    ]

    stats = _optimize_erf_transpose_passthrough_chains(model_ir)
    assert stats["rewritten_erf_transpose_passthrough_chains"] == 1
    op_types = [str(op.op_type) for op in model_ir.operators]
    assert op_types.count("TRANSPOSE") == 0
    abs_op = next(op for op in model_ir.operators if str(op.op_type) == "ABS")
    sign_op = next(op for op in model_ir.operators if str(op.op_type) == "SIGN")
    assert list(abs_op.inputs) == ["x_nhwc"]
    assert list(sign_op.inputs) == ["x_nhwc"]
    assert list(model_ir.tensors["y"].shape) == [1, 4, 4, 3]


def test_flatbuffer_direct_transpose_mean_prepost_nhwc_passthrough_chain() -> None:
    model_ir = ModelIR(name="transpose_mean_prepost_nhwc_passthrough_test")
    model_ir.inputs = ["x_nhwc"]
    model_ir.outputs = ["y_nhwc"]
    model_ir.tensors["x_nhwc"] = TensorIR(
        name="x_nhwc",
        dtype="FLOAT32",
        shape=[1, 4, 4, 8],
        shape_signature=[1, 4, 4, 8],
    )
    model_ir.tensors["pre_perm"] = TensorIR(
        name="pre_perm",
        dtype="INT32",
        shape=[4],
        shape_signature=[4],
        data=np.asarray([0, 3, 1, 2], dtype=np.int32),
        is_variable=False,
    )
    model_ir.tensors["post_perm"] = TensorIR(
        name="post_perm",
        dtype="INT32",
        shape=[4],
        shape_signature=[4],
        data=np.asarray([0, 2, 3, 1], dtype=np.int32),
        is_variable=False,
    )
    model_ir.tensors["mean_axes_nchw"] = TensorIR(
        name="mean_axes_nchw",
        dtype="INT32",
        shape=[2],
        shape_signature=[2],
        data=np.asarray([2, 3], dtype=np.int32),
        is_variable=False,
    )
    model_ir.tensors["x_nchw"] = TensorIR(
        name="x_nchw",
        dtype="FLOAT32",
        shape=[1, 8, 4, 4],
        shape_signature=[1, 8, 4, 4],
    )
    model_ir.tensors["m_nchw"] = TensorIR(
        name="m_nchw",
        dtype="FLOAT32",
        shape=[1, 8, 1, 1],
        shape_signature=[1, 8, 1, 1],
    )
    model_ir.tensors["y_nhwc"] = TensorIR(
        name="y_nhwc",
        dtype="FLOAT32",
        shape=[1, 1, 1, 8],
        shape_signature=[1, 1, 1, 8],
    )
    model_ir.operators = [
        OperatorIR(op_type="TRANSPOSE", inputs=["x_nhwc", "pre_perm"], outputs=["x_nchw"]),
        OperatorIR(op_type="MEAN", inputs=["x_nchw", "mean_axes_nchw"], outputs=["m_nchw"], options={"keepDims": True}),
        OperatorIR(op_type="TRANSPOSE", inputs=["m_nchw", "post_perm"], outputs=["y_nhwc"]),
    ]

    stats = _optimize_transpose_mean_prepost_nhwc_passthrough_chains(model_ir)
    assert stats["optimized_transpose_mean_prepost_nhwc_passthrough_chains"] == 1
    assert [str(op.op_type) for op in model_ir.operators] == ["MEAN"]
    mean_op = model_ir.operators[0]
    assert list(mean_op.inputs) == ["x_nhwc", "mean_axes_nchw"]
    assert list(mean_op.outputs) == ["y_nhwc"]
    assert np.array_equal(np.asarray(model_ir.tensors["mean_axes_nchw"].data, dtype=np.int32), np.asarray([1, 2], dtype=np.int32))


def test_flatbuffer_direct_transpose_mean_prepost_nhwc_passthrough_keeps_pre_on_fanout() -> None:
    model_ir = ModelIR(name="transpose_mean_prepost_nhwc_pre_fanout_keep_pre_test")
    model_ir.inputs = ["x_nhwc"]
    model_ir.outputs = ["y_nhwc", "side_nchw"]
    model_ir.tensors["x_nhwc"] = TensorIR(
        name="x_nhwc",
        dtype="FLOAT32",
        shape=[1, 4, 4, 8],
        shape_signature=[1, 4, 4, 8],
    )
    model_ir.tensors["pre_perm"] = TensorIR(
        name="pre_perm",
        dtype="INT32",
        shape=[4],
        shape_signature=[4],
        data=np.asarray([0, 3, 1, 2], dtype=np.int32),
        is_variable=False,
    )
    model_ir.tensors["post_perm"] = TensorIR(
        name="post_perm",
        dtype="INT32",
        shape=[4],
        shape_signature=[4],
        data=np.asarray([0, 2, 3, 1], dtype=np.int32),
        is_variable=False,
    )
    model_ir.tensors["mean_axes_nchw"] = TensorIR(
        name="mean_axes_nchw",
        dtype="INT32",
        shape=[2],
        shape_signature=[2],
        data=np.asarray([2, 3], dtype=np.int32),
        is_variable=False,
    )
    for name, shape in [
        ("x_nchw", [1, 8, 4, 4]),
        ("m_nchw", [1, 8, 1, 1]),
        ("y_nhwc", [1, 1, 1, 8]),
        ("side_nchw", [1, 8, 4, 4]),
    ]:
        model_ir.tensors[name] = TensorIR(
            name=name,
            dtype="FLOAT32",
            shape=shape,
            shape_signature=shape,
        )
    model_ir.operators = [
        OperatorIR(op_type="TRANSPOSE", inputs=["x_nhwc", "pre_perm"], outputs=["x_nchw"]),
        OperatorIR(op_type="MEAN", inputs=["x_nchw", "mean_axes_nchw"], outputs=["m_nchw"], options={"keepDims": True}),
        OperatorIR(op_type="TRANSPOSE", inputs=["m_nchw", "post_perm"], outputs=["y_nhwc"]),
        OperatorIR(op_type="RELU", inputs=["x_nchw"], outputs=["side_nchw"]),
    ]

    stats = _optimize_transpose_mean_prepost_nhwc_passthrough_chains(model_ir)
    assert stats["optimized_transpose_mean_prepost_nhwc_passthrough_chains"] == 1
    assert [str(op.op_type) for op in model_ir.operators] == ["TRANSPOSE", "MEAN", "RELU"]
    mean_op = model_ir.operators[1]
    assert list(mean_op.inputs) == ["x_nhwc", "mean_axes_nchw"]
    assert list(mean_op.outputs) == ["y_nhwc"]
    assert np.array_equal(
        np.asarray(model_ir.tensors["mean_axes_nchw"].data, dtype=np.int32),
        np.asarray([1, 2], dtype=np.int32),
    )
    relu_op = model_ir.operators[2]
    assert list(relu_op.inputs) == ["x_nchw"]


def test_flatbuffer_direct_transpose_pre_unary_mean_terminal_nhwc_chain() -> None:
    model_ir = ModelIR(name="transpose_pre_unary_mean_terminal_nhwc_test")
    model_ir.inputs = ["x_nhwc"]
    model_ir.outputs = ["y"]
    model_ir.tensors["x_nhwc"] = TensorIR(
        name="x_nhwc",
        dtype="FLOAT32",
        shape=[1, 4, 4, 8],
        shape_signature=[1, 4, 4, 8],
    )
    model_ir.tensors["pre_perm"] = TensorIR(
        name="pre_perm",
        dtype="INT32",
        shape=[4],
        shape_signature=[4],
        data=np.asarray([0, 3, 1, 2], dtype=np.int32),
        is_variable=False,
    )
    model_ir.tensors["mean_axes_nchw"] = TensorIR(
        name="mean_axes_nchw",
        dtype="INT32",
        shape=[2],
        shape_signature=[2],
        data=np.asarray([2, 3], dtype=np.int32),
        is_variable=False,
    )
    model_ir.tensors["reshape_shape"] = TensorIR(
        name="reshape_shape",
        dtype="INT32",
        shape=[2],
        shape_signature=[2],
        data=np.asarray([1, 8], dtype=np.int32),
        is_variable=False,
    )
    model_ir.tensors["x_nchw"] = TensorIR(
        name="x_nchw",
        dtype="FLOAT32",
        shape=[1, 8, 4, 4],
        shape_signature=[1, 8, 4, 4],
    )
    model_ir.tensors["u_nchw"] = TensorIR(
        name="u_nchw",
        dtype="FLOAT32",
        shape=[1, 8, 4, 4],
        shape_signature=[1, 8, 4, 4],
    )
    model_ir.tensors["m_nchw"] = TensorIR(
        name="m_nchw",
        dtype="FLOAT32",
        shape=[1, 8, 1, 1],
        shape_signature=[1, 8, 1, 1],
    )
    model_ir.tensors["y"] = TensorIR(
        name="y",
        dtype="FLOAT32",
        shape=[1, 8],
        shape_signature=[1, 8],
    )
    model_ir.operators = [
        OperatorIR(op_type="TRANSPOSE", inputs=["x_nhwc", "pre_perm"], outputs=["x_nchw"]),
        OperatorIR(op_type="RELU6", inputs=["x_nchw"], outputs=["u_nchw"]),
        OperatorIR(op_type="MEAN", inputs=["u_nchw", "mean_axes_nchw"], outputs=["m_nchw"], options={"keepDims": True}),
        OperatorIR(op_type="RESHAPE", inputs=["m_nchw", "reshape_shape"], outputs=["y"], options={"newShape": [1, 8]}),
    ]

    stats = _optimize_transpose_pre_unary_mean_terminal_nhwc_chains(model_ir)
    assert stats["optimized_transpose_pre_unary_mean_terminal_nhwc_chains"] == 1
    assert [str(op.op_type) for op in model_ir.operators] == ["RELU6", "MEAN", "RESHAPE"]
    relu_op = model_ir.operators[0]
    assert list(relu_op.inputs) == ["x_nhwc"]
    assert list(model_ir.tensors["u_nchw"].shape) == [1, 4, 4, 8]
    mean_op = model_ir.operators[1]
    assert list(mean_op.inputs) == ["u_nchw", "mean_axes_nchw"]
    assert np.array_equal(
        np.asarray(model_ir.tensors["mean_axes_nchw"].data, dtype=np.int32),
        np.asarray([1, 2], dtype=np.int32),
    )
    assert list(model_ir.tensors["m_nchw"].shape) == [1, 1, 1, 8]


def test_flatbuffer_direct_transpose_se_conv_mul_prepost_nhwc_chain() -> None:
    model_ir = ModelIR(name="transpose_se_conv_mul_prepost_nhwc_test")
    model_ir.inputs = ["x_nhwc"]
    model_ir.outputs = ["z"]

    def _add_tensor(name: str, shape: list[int], dtype: str = "FLOAT32", data: np.ndarray | None = None) -> None:
        model_ir.tensors[name] = TensorIR(
            name=name,
            dtype=dtype,
            shape=[int(v) for v in shape],
            shape_signature=[int(v) for v in shape],
            data=data,
            is_variable=False if data is not None else True,
        )

    _add_tensor("x_nhwc", [1, 4, 4, 8])
    _add_tensor("x_nchw", [1, 8, 4, 4])
    _add_tensor("sig1_nchw", [1, 8, 4, 4])
    _add_tensor("sw_nchw", [1, 8, 4, 4])
    _add_tensor("mean_axes_nchw", [2], "INT32", np.asarray([2, 3], dtype=np.int32))
    _add_tensor("mean_nchw", [1, 8, 1, 1])
    _add_tensor("mean_nhwc", [1, 1, 1, 8])
    _add_tensor("gate_pre_in_nhwc", [1, 1, 1, 8])
    _add_tensor("gate_nchw", [1, 8, 1, 1])
    _add_tensor("sig2_nchw", [1, 8, 1, 1])
    _add_tensor("y_nchw", [1, 8, 4, 4])
    _add_tensor("y_nhwc", [1, 4, 4, 8])
    _add_tensor("z", [1, 4, 4, 8])
    _add_tensor("perm_nhwc_to_nchw", [4], "INT32", np.asarray([0, 3, 1, 2], dtype=np.int32))
    _add_tensor("perm_nchw_to_nhwc", [4], "INT32", np.asarray([0, 2, 3, 1], dtype=np.int32))
    _add_tensor("conv_w", [1, 1, 8, 8], data=np.ones((1, 1, 8, 8), dtype=np.float32))
    _add_tensor("conv_b", [8], data=np.zeros((8,), dtype=np.float32))

    model_ir.operators = [
        OperatorIR(op_type="TRANSPOSE", inputs=["x_nhwc", "perm_nhwc_to_nchw"], outputs=["x_nchw"]),
        OperatorIR(op_type="LOGISTIC", inputs=["x_nchw"], outputs=["sig1_nchw"]),
        OperatorIR(op_type="MUL", inputs=["x_nchw", "sig1_nchw"], outputs=["sw_nchw"]),
        OperatorIR(
            op_type="MEAN",
            inputs=["sw_nchw", "mean_axes_nchw"],
            outputs=["mean_nchw"],
            options={"keepDims": True, "axes": [2, 3]},
        ),
        OperatorIR(op_type="TRANSPOSE", inputs=["mean_nchw", "perm_nchw_to_nhwc"], outputs=["mean_nhwc"]),
        OperatorIR(
            op_type="CONV_2D",
            inputs=["mean_nhwc", "conv_w", "conv_b"],
            outputs=["gate_pre_in_nhwc"],
            options={
                "padding": "SAME",
                "strideH": 1,
                "strideW": 1,
                "dilationHFactor": 1,
                "dilationWFactor": 1,
                "fusedActivationFunction": "NONE",
            },
        ),
        OperatorIR(op_type="TRANSPOSE", inputs=["gate_pre_in_nhwc", "perm_nhwc_to_nchw"], outputs=["gate_nchw"]),
        OperatorIR(op_type="LOGISTIC", inputs=["gate_nchw"], outputs=["sig2_nchw"]),
        OperatorIR(op_type="MUL", inputs=["sw_nchw", "sig2_nchw"], outputs=["y_nchw"]),
        OperatorIR(op_type="TRANSPOSE", inputs=["y_nchw", "perm_nchw_to_nhwc"], outputs=["y_nhwc"]),
        OperatorIR(op_type="RELU", inputs=["y_nhwc"], outputs=["z"]),
    ]

    stats = _optimize_transpose_se_conv_mul_prepost_nhwc_chains(model_ir)
    assert stats["optimized_transpose_se_conv_mul_prepost_nhwc_chains"] == 1
    assert all(str(op.op_type) != "TRANSPOSE" for op in model_ir.operators)

    mean_axes_vals = np.asarray(model_ir.tensors["mean_axes_nchw"].data, dtype=np.int32).reshape(-1).tolist()
    assert mean_axes_vals == [1, 2]

    log1_op = next(op for op in model_ir.operators if str(op.op_type) == "LOGISTIC" and list(op.outputs) == ["sig1_nchw"])
    assert list(log1_op.inputs) == ["x_nhwc"]
    log2_op = next(op for op in model_ir.operators if str(op.op_type) == "LOGISTIC" and list(op.outputs) == ["sig2_nchw"])
    assert list(log2_op.inputs) == ["gate_pre_in_nhwc"]

    mul2_op = next(op for op in model_ir.operators if str(op.op_type) == "MUL" and "sig2_nchw" in list(op.inputs))
    assert list(mul2_op.outputs) == ["y_nhwc"]
    relu_op = next(op for op in model_ir.operators if str(op.op_type) == "RELU")
    assert list(relu_op.inputs) == ["y_nhwc"]


def test_flatbuffer_direct_transpose_se_conv_mul_prepost_nhwc_chain_with_squeeze_reshape_gate() -> None:
    model_ir = ModelIR(name="transpose_se_conv_mul_prepost_nhwc_squeeze_reshape_gate_test")
    model_ir.inputs = ["x_nhwc"]
    model_ir.outputs = ["z"]

    def _add_tensor(name: str, shape: list[int], dtype: str = "FLOAT32", data: np.ndarray | None = None) -> None:
        model_ir.tensors[name] = TensorIR(
            name=name,
            dtype=dtype,
            shape=[int(v) for v in shape],
            shape_signature=[int(v) for v in shape],
            data=data,
            is_variable=False if data is not None else True,
        )

    _add_tensor("x_nhwc", [1, 4, 4, 8])
    _add_tensor("x_nchw", [1, 8, 4, 4])
    _add_tensor("sig1_nchw", [1, 8, 4, 4])
    _add_tensor("sw_nchw", [1, 8, 4, 4])
    _add_tensor("mean_axes_nchw", [2], "INT32", np.asarray([2, 3], dtype=np.int32))
    _add_tensor("mean_nchw", [1, 8, 1, 1])
    _add_tensor("mean_sq", [1, 8])
    _add_tensor("mean_nhwc", [1, 1, 1, 8])
    _add_tensor("gate_pre_in_nhwc", [1, 1, 1, 8])
    _add_tensor("gate_nchw", [1, 8, 1, 1])
    _add_tensor("sig2_nchw", [1, 8, 1, 1])
    _add_tensor("y_nchw", [1, 8, 4, 4])
    _add_tensor("y_nhwc", [1, 4, 4, 8])
    _add_tensor("z", [1, 4, 4, 8])
    _add_tensor("perm_nhwc_to_nchw", [4], "INT32", np.asarray([0, 3, 1, 2], dtype=np.int32))
    _add_tensor("perm_nchw_to_nhwc", [4], "INT32", np.asarray([0, 2, 3, 1], dtype=np.int32))
    _add_tensor("conv_w", [1, 1, 8, 8], data=np.ones((1, 1, 8, 8), dtype=np.float32))
    _add_tensor("conv_b", [8], data=np.zeros((8,), dtype=np.float32))
    _add_tensor("reshape_mean_to_nhwc", [4], "INT32", np.asarray([1, 1, 1, 8], dtype=np.int32))
    _add_tensor("reshape_gate_to_nchw", [4], "INT32", np.asarray([1, 8, 1, 1], dtype=np.int32))

    model_ir.operators = [
        OperatorIR(op_type="TRANSPOSE", inputs=["x_nhwc", "perm_nhwc_to_nchw"], outputs=["x_nchw"]),
        OperatorIR(op_type="LOGISTIC", inputs=["x_nchw"], outputs=["sig1_nchw"]),
        OperatorIR(op_type="MUL", inputs=["x_nchw", "sig1_nchw"], outputs=["sw_nchw"]),
        OperatorIR(
            op_type="MEAN",
            inputs=["sw_nchw", "mean_axes_nchw"],
            outputs=["mean_nchw"],
            options={"keepDims": True, "axes": [2, 3]},
        ),
        OperatorIR(op_type="SQUEEZE", inputs=["mean_nchw"], outputs=["mean_sq"], options={"squeezeDims": [2, 3]}),
        OperatorIR(
            op_type="RESHAPE",
            inputs=["mean_sq", "reshape_mean_to_nhwc"],
            outputs=["mean_nhwc"],
            options={"newShape": [1, 1, 1, 8]},
        ),
        OperatorIR(
            op_type="CONV_2D",
            inputs=["mean_nhwc", "conv_w", "conv_b"],
            outputs=["gate_pre_in_nhwc"],
            options={
                "padding": "SAME",
                "strideH": 1,
                "strideW": 1,
                "dilationHFactor": 1,
                "dilationWFactor": 1,
                "fusedActivationFunction": "NONE",
            },
        ),
        OperatorIR(
            op_type="RESHAPE",
            inputs=["gate_pre_in_nhwc", "reshape_gate_to_nchw"],
            outputs=["gate_nchw"],
            options={"newShape": [1, 8, 1, 1]},
        ),
        OperatorIR(op_type="LOGISTIC", inputs=["gate_nchw"], outputs=["sig2_nchw"]),
        OperatorIR(op_type="MUL", inputs=["sw_nchw", "sig2_nchw"], outputs=["y_nchw"]),
        OperatorIR(op_type="TRANSPOSE", inputs=["y_nchw", "perm_nchw_to_nhwc"], outputs=["y_nhwc"]),
        OperatorIR(op_type="RELU", inputs=["y_nhwc"], outputs=["z"]),
    ]

    stats = _optimize_transpose_se_conv_mul_prepost_nhwc_chains(model_ir)
    assert stats["optimized_transpose_se_conv_mul_prepost_nhwc_chains"] == 1
    assert all(str(op.op_type) != "TRANSPOSE" for op in model_ir.operators)

    mean_axes_vals = np.asarray(model_ir.tensors["mean_axes_nchw"].data, dtype=np.int32).reshape(-1).tolist()
    assert mean_axes_vals == [1, 2]

    squeeze_op = next(op for op in model_ir.operators if str(op.op_type) == "SQUEEZE")
    assert sorted(int(v) for v in list(squeeze_op.options.get("squeezeDims", []))) == [1, 2]

    gate_shape_vals = np.asarray(model_ir.tensors["reshape_gate_to_nchw"].data, dtype=np.int32).reshape(-1).tolist()
    assert gate_shape_vals == [1, 1, 1, 8]

    log1_op = next(op for op in model_ir.operators if str(op.op_type) == "LOGISTIC" and list(op.outputs) == ["sig1_nchw"])
    assert list(log1_op.inputs) == ["x_nhwc"]

    mul2_op = next(op for op in model_ir.operators if str(op.op_type) == "MUL" and "sig2_nchw" in list(op.inputs))
    assert list(mul2_op.outputs) == ["y_nhwc"]
    assert list(model_ir.tensors["gate_nchw"].shape) == [1, 1, 1, 8]
    assert list(model_ir.tensors["sig2_nchw"].shape) == [1, 1, 1, 8]


def test_flatbuffer_direct_transpose_se_conv_mul_prepost_nhwc_chain_with_affine_gate() -> None:
    model_ir = ModelIR(name="transpose_se_conv_mul_prepost_nhwc_affine_gate_test")
    model_ir.inputs = ["x_nhwc"]
    model_ir.outputs = ["z"]

    def _add_tensor(name: str, shape: list[int], dtype: str = "FLOAT32", data: np.ndarray | None = None) -> None:
        model_ir.tensors[name] = TensorIR(
            name=name,
            dtype=dtype,
            shape=[int(v) for v in shape],
            shape_signature=[int(v) for v in shape],
            data=data,
            is_variable=False if data is not None else True,
        )

    _add_tensor("x_nhwc", [1, 4, 4, 8])
    _add_tensor("x_nchw", [1, 8, 4, 4])
    _add_tensor("sig1_nchw", [1, 8, 4, 4])
    _add_tensor("sw_nchw", [1, 8, 4, 4])
    _add_tensor("mean_axes_nchw", [2], "INT32", np.asarray([2, 3], dtype=np.int32))
    _add_tensor("mean_nchw", [1, 8, 1, 1])
    _add_tensor("mean_nhwc", [1, 1, 1, 8])
    _add_tensor("gate_pre_in_nhwc", [1, 1, 1, 8])
    _add_tensor("gate_nchw", [1, 8, 1, 1])
    _add_tensor("gate_add_nchw", [1, 8, 1, 1])
    _add_tensor("gate_affine_nchw", [1, 8, 1, 1])
    _add_tensor("y_nchw", [1, 8, 4, 4])
    _add_tensor("y_nhwc", [1, 4, 4, 8])
    _add_tensor("z", [1, 4, 4, 8])
    _add_tensor("perm_nhwc_to_nchw", [4], "INT32", np.asarray([0, 3, 1, 2], dtype=np.int32))
    _add_tensor("perm_nchw_to_nhwc", [4], "INT32", np.asarray([0, 2, 3, 1], dtype=np.int32))
    _add_tensor("conv_w", [8, 1, 1, 8], data=np.ones((8, 1, 1, 8), dtype=np.float32))
    _add_tensor("conv_b", [8], data=np.zeros((8,), dtype=np.float32))
    _add_tensor("gate_add_const", [1], data=np.asarray([3.0], dtype=np.float32))
    _add_tensor("gate_mul_const", [1], data=np.asarray([1.0 / 6.0], dtype=np.float32))

    model_ir.operators = [
        OperatorIR(op_type="TRANSPOSE", inputs=["x_nhwc", "perm_nhwc_to_nchw"], outputs=["x_nchw"]),
        OperatorIR(op_type="LOGISTIC", inputs=["x_nchw"], outputs=["sig1_nchw"]),
        OperatorIR(op_type="MUL", inputs=["x_nchw", "sig1_nchw"], outputs=["sw_nchw"]),
        OperatorIR(
            op_type="MEAN",
            inputs=["sw_nchw", "mean_axes_nchw"],
            outputs=["mean_nchw"],
            options={"keepDims": True, "axes": [2, 3]},
        ),
        OperatorIR(op_type="TRANSPOSE", inputs=["mean_nchw", "perm_nchw_to_nhwc"], outputs=["mean_nhwc"]),
        OperatorIR(
            op_type="CONV_2D",
            inputs=["mean_nhwc", "conv_w", "conv_b"],
            outputs=["gate_pre_in_nhwc"],
            options={
                "padding": "SAME",
                "strideH": 1,
                "strideW": 1,
                "dilationHFactor": 1,
                "dilationWFactor": 1,
                "fusedActivationFunction": "NONE",
            },
        ),
        OperatorIR(op_type="TRANSPOSE", inputs=["gate_pre_in_nhwc", "perm_nhwc_to_nchw"], outputs=["gate_nchw"]),
        OperatorIR(op_type="ADD", inputs=["gate_nchw", "gate_add_const"], outputs=["gate_add_nchw"]),
        OperatorIR(op_type="MUL", inputs=["gate_add_nchw", "gate_mul_const"], outputs=["gate_affine_nchw"]),
        OperatorIR(op_type="MUL", inputs=["sw_nchw", "gate_affine_nchw"], outputs=["y_nchw"]),
        OperatorIR(op_type="TRANSPOSE", inputs=["y_nchw", "perm_nchw_to_nhwc"], outputs=["y_nhwc"]),
        OperatorIR(op_type="RELU", inputs=["y_nhwc"], outputs=["z"]),
    ]

    stats = _optimize_transpose_se_conv_mul_prepost_nhwc_chains(model_ir)
    assert stats["optimized_transpose_se_conv_mul_prepost_nhwc_chains"] == 1
    assert all(str(op.op_type) != "TRANSPOSE" for op in model_ir.operators)

    mean_axes_vals = np.asarray(model_ir.tensors["mean_axes_nchw"].data, dtype=np.int32).reshape(-1).tolist()
    assert mean_axes_vals == [1, 2]

    log1_op = next(op for op in model_ir.operators if str(op.op_type) == "LOGISTIC")
    assert list(log1_op.inputs) == ["x_nhwc"]

    gate_add_op = next(op for op in model_ir.operators if str(op.op_type) == "ADD")
    assert "gate_pre_in_nhwc" in list(gate_add_op.inputs)

    mul2_op = next(
        op
        for op in model_ir.operators
        if str(op.op_type) == "MUL" and "gate_affine_nchw" in list(op.inputs)
    )
    assert list(mul2_op.outputs) == ["y_nhwc"]
    relu_op = next(op for op in model_ir.operators if str(op.op_type) == "RELU")
    assert list(relu_op.inputs) == ["y_nhwc"]

    assert list(model_ir.tensors["gate_add_nchw"].shape) == [1, 1, 1, 8]
    assert list(model_ir.tensors["gate_affine_nchw"].shape) == [1, 1, 1, 8]


def test_flatbuffer_direct_batchmatmul_affine_transpose_input_chains() -> None:
    model_ir = ModelIR(name="batchmatmul_affine_transpose_inputs_test")
    model_ir.inputs = ["lhs_nhwc", "rhs_nhwc"]
    model_ir.outputs = ["bmm_out"]

    def _add_tensor(name: str, shape: list[int], dtype: str = "FLOAT32", data: np.ndarray | None = None) -> None:
        model_ir.tensors[name] = TensorIR(
            name=name,
            dtype=dtype,
            shape=[int(v) for v in shape],
            shape_signature=[int(v) for v in shape],
            data=data,
            is_variable=False if data is not None else True,
        )

    _add_tensor("lhs_nhwc", [1, 8, 8, 96])
    _add_tensor("lhs_nchw", [1, 96, 8, 8])
    _add_tensor("lhs_mul_out", [1, 96, 8, 8])
    _add_tensor("lhs_add_out", [1, 96, 8, 8])
    _add_tensor("lhs_reshape", [1, 96, 64])
    _add_tensor("lhs_mat", [1, 64, 96])

    _add_tensor("rhs_nhwc", [1, 16, 16, 96])
    _add_tensor("rhs_nchw", [1, 96, 16, 16])
    _add_tensor("rhs_mul_out", [1, 96, 16, 16])
    _add_tensor("rhs_add_out", [1, 96, 16, 16])
    _add_tensor("rhs_reshape", [1, 96, 256])

    _add_tensor("bmm_out", [1, 64, 256])

    _add_tensor("perm_nhwc_to_nchw", [4], "INT32", np.asarray([0, 3, 1, 2], dtype=np.int32))
    _add_tensor("perm_swap_last2_rank3", [3], "INT32", np.asarray([0, 2, 1], dtype=np.int32))
    _add_tensor("lhs_mul_const", [1, 96, 1, 1], data=np.ones((1, 96, 1, 1), dtype=np.float32))
    _add_tensor("lhs_add_const", [1, 96, 1, 1], data=np.zeros((1, 96, 1, 1), dtype=np.float32))
    _add_tensor("rhs_mul_const", [1, 96, 1, 1], data=np.ones((1, 96, 1, 1), dtype=np.float32))
    _add_tensor("rhs_add_const", [1, 96, 1, 1], data=np.zeros((1, 96, 1, 1), dtype=np.float32))
    _add_tensor("lhs_shape", [3], "INT32", np.asarray([1, 96, 64], dtype=np.int32))
    _add_tensor("rhs_shape", [3], "INT32", np.asarray([1, 96, 256], dtype=np.int32))

    model_ir.operators = [
        OperatorIR(op_type="TRANSPOSE", inputs=["lhs_nhwc", "perm_nhwc_to_nchw"], outputs=["lhs_nchw"]),
        OperatorIR(op_type="MUL", inputs=["lhs_nchw", "lhs_mul_const"], outputs=["lhs_mul_out"]),
        OperatorIR(op_type="ADD", inputs=["lhs_mul_out", "lhs_add_const"], outputs=["lhs_add_out"]),
        OperatorIR(op_type="RESHAPE", inputs=["lhs_add_out", "lhs_shape"], outputs=["lhs_reshape"], options={"newShape": [1, 96, 64]}),
        OperatorIR(op_type="TRANSPOSE", inputs=["lhs_reshape", "perm_swap_last2_rank3"], outputs=["lhs_mat"]),
        OperatorIR(op_type="TRANSPOSE", inputs=["rhs_nhwc", "perm_nhwc_to_nchw"], outputs=["rhs_nchw"]),
        OperatorIR(op_type="MUL", inputs=["rhs_nchw", "rhs_mul_const"], outputs=["rhs_mul_out"]),
        OperatorIR(op_type="ADD", inputs=["rhs_mul_out", "rhs_add_const"], outputs=["rhs_add_out"]),
        OperatorIR(op_type="RESHAPE", inputs=["rhs_add_out", "rhs_shape"], outputs=["rhs_reshape"], options={"newShape": [1, 96, 256]}),
        OperatorIR(
            op_type="BATCH_MATMUL",
            inputs=["lhs_mat", "rhs_reshape"],
            outputs=["bmm_out"],
            options={"adjX": False, "adjY": False},
        ),
    ]

    stats = _optimize_batchmatmul_affine_transpose_input_chains(model_ir)
    assert stats["optimized_batchmatmul_affine_transpose_input_chains"] == 1
    assert all(str(op.op_type) != "TRANSPOSE" for op in model_ir.operators)

    bmm_op = next(op for op in model_ir.operators if str(op.op_type) == "BATCH_MATMUL")
    assert list(bmm_op.inputs) == ["lhs_reshape", "rhs_reshape"]
    assert bool(dict(bmm_op.options).get("adjY", False))

    lhs_mul_op = next(op for op in model_ir.operators if str(op.op_type) == "MUL" and list(op.outputs) == ["lhs_mul_out"])
    rhs_mul_op = next(op for op in model_ir.operators if str(op.op_type) == "MUL" and list(op.outputs) == ["rhs_mul_out"])
    assert list(lhs_mul_op.inputs)[0] == "lhs_nhwc"
    assert list(rhs_mul_op.inputs)[0] == "rhs_nhwc"

    lhs_shape_vals = np.asarray(model_ir.tensors["lhs_shape"].data, dtype=np.int32).reshape(-1).tolist()
    rhs_shape_vals = np.asarray(model_ir.tensors["rhs_shape"].data, dtype=np.int32).reshape(-1).tolist()
    assert lhs_shape_vals == [1, 64, 96]
    assert rhs_shape_vals == [1, 256, 96]


def test_flatbuffer_direct_batchmatmul_reshape_se_nhwc_chains() -> None:
    model_ir = ModelIR(name="batchmatmul_reshape_se_nhwc_chain_test")
    model_ir.inputs = ["lhs_mat", "rhs_mat"]
    model_ir.outputs = ["z"]

    def _add_tensor(name: str, shape: list[int], dtype: str = "FLOAT32", data: np.ndarray | None = None) -> None:
        model_ir.tensors[name] = TensorIR(
            name=name,
            dtype=dtype,
            shape=[int(v) for v in shape],
            shape_signature=[int(v) for v in shape],
            data=data,
            is_variable=False if data is not None else True,
        )

    _add_tensor("lhs_mat", [1, 64, 96])
    _add_tensor("rhs_mat", [1, 256, 96])
    _add_tensor("bmm_out", [1, 64, 256])
    _add_tensor("x_nchw", [1, 64, 16, 16])
    _add_tensor("mean_nchw", [1, 64, 1, 1])
    _add_tensor("mean_nhwc", [1, 1, 1, 64])
    _add_tensor("conv1_out", [1, 1, 1, 64])
    _add_tensor("conv2_out_nhwc", [1, 1, 1, 64])
    _add_tensor("gate_nchw", [1, 64, 1, 1])
    _add_tensor("gate_sig", [1, 64, 1, 1])
    _add_tensor("y_nchw", [1, 64, 16, 16])
    _add_tensor("y_nhwc", [1, 16, 16, 64])
    _add_tensor("z", [1, 16, 16, 64])

    _add_tensor("shape_x", [4], "INT32", np.asarray([1, 64, 16, 16], dtype=np.int32))
    _add_tensor("mean_axes", [2], "INT32", np.asarray([2, 3], dtype=np.int32))
    _add_tensor("perm_nhwc_to_nchw", [4], "INT32", np.asarray([0, 3, 1, 2], dtype=np.int32))
    _add_tensor("perm_nchw_to_nhwc", [4], "INT32", np.asarray([0, 2, 3, 1], dtype=np.int32))
    _add_tensor("conv_w", [64, 1, 1, 64], data=np.ones((64, 1, 1, 64), dtype=np.float32))
    _add_tensor("conv_b", [64], data=np.zeros((64,), dtype=np.float32))

    model_ir.operators = [
        OperatorIR(
            op_type="BATCH_MATMUL",
            inputs=["lhs_mat", "rhs_mat"],
            outputs=["bmm_out"],
            options={"adjX": False, "adjY": True},
        ),
        OperatorIR(op_type="RESHAPE", inputs=["bmm_out", "shape_x"], outputs=["x_nchw"], options={"newShape": [1, 64, 16, 16]}),
        OperatorIR(op_type="MEAN", inputs=["x_nchw", "mean_axes"], outputs=["mean_nchw"], options={"keepDims": True, "axes": [2, 3]}),
        OperatorIR(op_type="TRANSPOSE", inputs=["mean_nchw", "perm_nchw_to_nhwc"], outputs=["mean_nhwc"]),
        OperatorIR(
            op_type="CONV_2D",
            inputs=["mean_nhwc", "conv_w", "conv_b"],
            outputs=["conv1_out"],
            options={"padding": "SAME", "strideH": 1, "strideW": 1, "dilationHFactor": 1, "dilationWFactor": 1},
        ),
        OperatorIR(
            op_type="CONV_2D",
            inputs=["conv1_out", "conv_w", "conv_b"],
            outputs=["conv2_out_nhwc"],
            options={"padding": "SAME", "strideH": 1, "strideW": 1, "dilationHFactor": 1, "dilationWFactor": 1},
        ),
        OperatorIR(op_type="TRANSPOSE", inputs=["conv2_out_nhwc", "perm_nhwc_to_nchw"], outputs=["gate_nchw"]),
        OperatorIR(op_type="LOGISTIC", inputs=["gate_nchw"], outputs=["gate_sig"]),
        OperatorIR(op_type="MUL", inputs=["x_nchw", "gate_sig"], outputs=["y_nchw"]),
        OperatorIR(op_type="TRANSPOSE", inputs=["y_nchw", "perm_nchw_to_nhwc"], outputs=["y_nhwc"]),
        OperatorIR(op_type="RELU", inputs=["y_nhwc"], outputs=["z"]),
    ]

    stats = _optimize_batchmatmul_reshape_se_nhwc_chains(model_ir)
    assert stats["optimized_batchmatmul_reshape_se_nhwc_chains"] == 1
    assert all(str(op.op_type) != "TRANSPOSE" for op in model_ir.operators)

    bmm_op = next(op for op in model_ir.operators if str(op.op_type) == "BATCH_MATMUL")
    assert list(bmm_op.inputs) == ["rhs_mat", "lhs_mat"]
    assert bool(dict(bmm_op.options).get("adjY", False))

    shape_vals = np.asarray(model_ir.tensors["shape_x"].data, dtype=np.int32).reshape(-1).tolist()
    assert shape_vals == [1, 16, 16, 64]
    mean_axes_vals = np.asarray(model_ir.tensors["mean_axes"].data, dtype=np.int32).reshape(-1).tolist()
    assert mean_axes_vals == [1, 2]

    gate_op = next(op for op in model_ir.operators if str(op.op_type) == "LOGISTIC")
    assert list(gate_op.inputs) == ["conv2_out_nhwc"]

    mul_op = next(op for op in model_ir.operators if str(op.op_type) == "MUL")
    assert list(mul_op.outputs) == ["y_nhwc"]
    relu_op = next(op for op in model_ir.operators if str(op.op_type) == "RELU")
    assert list(relu_op.inputs) == ["y_nhwc"]


def test_flatbuffer_direct_transpose_se_fc_mul_prepost_nhwc_chain() -> None:
    model_ir = ModelIR(name="transpose_se_fc_mul_prepost_nhwc_test")
    model_ir.inputs = ["x_nhwc"]
    model_ir.outputs = ["y"]

    def _add_tensor(name: str, shape: list[int], dtype: str = "FLOAT32", data: np.ndarray | None = None) -> None:
        model_ir.tensors[name] = TensorIR(
            name=name,
            dtype=dtype,
            shape=[int(v) for v in shape],
            shape_signature=[int(v) for v in shape],
            data=data,
            is_variable=False if data is not None else True,
        )

    _add_tensor("x_nhwc", [1, 80, 80, 12])
    _add_tensor("x_nchw", [1, 12, 80, 80])
    _add_tensor("pool_nhwc", [1, 1, 1, 12])
    _add_tensor("pool_nchw", [1, 12, 1, 1])
    _add_tensor("fc_in", [1, 12])
    _add_tensor("fc0_out", [1, 12])
    _add_tensor("fc0_act", [1, 12])
    _add_tensor("fc1_out", [1, 12])
    _add_tensor("gate_fc", [1, 12])
    _add_tensor("gate_nchw", [1, 12, 1, 1])
    _add_tensor("mul_out_nchw", [1, 12, 80, 80])
    _add_tensor("mul_out_nhwc", [1, 80, 80, 12])
    _add_tensor("y", [1, 80, 80, 16])
    _add_tensor("perm_nhwc_to_nchw", [4], "INT32", np.asarray([0, 3, 1, 2], dtype=np.int32))
    _add_tensor("perm_nchw_to_nhwc", [4], "INT32", np.asarray([0, 2, 3, 1], dtype=np.int32))
    _add_tensor("shape_fc_in", [2], "INT32", np.asarray([1, 12], dtype=np.int32))
    _add_tensor("shape_gate", [4], "INT32", np.asarray([1, 12, 1, 1], dtype=np.int32))
    _add_tensor("fc0_w", [12, 12], data=np.ones((12, 12), dtype=np.float32))
    _add_tensor("fc0_b", [12], data=np.zeros((12,), dtype=np.float32))
    _add_tensor("prelu0_slope", [12], data=np.ones((12,), dtype=np.float32))
    _add_tensor("fc1_w", [12, 12], data=np.ones((12, 12), dtype=np.float32))
    _add_tensor("fc1_b", [12], data=np.zeros((12,), dtype=np.float32))
    _add_tensor("prelu1_slope", [12], data=np.ones((12,), dtype=np.float32))
    _add_tensor("conv_w", [1, 1, 12, 16], data=np.ones((1, 1, 12, 16), dtype=np.float32))
    _add_tensor("conv_b", [16], data=np.zeros((16,), dtype=np.float32))

    model_ir.operators = [
        OperatorIR(op_type="TRANSPOSE", inputs=["x_nhwc", "perm_nhwc_to_nchw"], outputs=["x_nchw"]),
        OperatorIR(
            op_type="AVERAGE_POOL_2D",
            inputs=["x_nhwc"],
            outputs=["pool_nhwc"],
            options={
                "padding": "VALID",
                "strideH": 80,
                "strideW": 80,
                "filterHeight": 80,
                "filterWidth": 80,
                "fusedActivationFunction": "NONE",
            },
        ),
        OperatorIR(op_type="TRANSPOSE", inputs=["pool_nhwc", "perm_nhwc_to_nchw"], outputs=["pool_nchw"]),
        OperatorIR(
            op_type="RESHAPE",
            inputs=["pool_nchw", "shape_fc_in"],
            outputs=["fc_in"],
            options={"newShape": [1, 12], "onnxRawNewShape": [1, 12]},
        ),
        OperatorIR(op_type="FULLY_CONNECTED", inputs=["fc_in", "fc0_w", "fc0_b"], outputs=["fc0_out"], options={}),
        OperatorIR(op_type="PRELU", inputs=["fc0_out", "prelu0_slope"], outputs=["fc0_act"], options={}),
        OperatorIR(op_type="FULLY_CONNECTED", inputs=["fc0_act", "fc1_w", "fc1_b"], outputs=["fc1_out"], options={}),
        OperatorIR(op_type="PRELU", inputs=["fc1_out", "prelu1_slope"], outputs=["gate_fc"], options={}),
        OperatorIR(
            op_type="RESHAPE",
            inputs=["gate_fc", "shape_gate"],
            outputs=["gate_nchw"],
            options={"newShape": [1, 12, 1, 1], "onnxRawNewShape": [1, 12, 1, 1]},
        ),
        OperatorIR(op_type="MUL", inputs=["gate_nchw", "x_nchw"], outputs=["mul_out_nchw"], options={}),
        OperatorIR(op_type="TRANSPOSE", inputs=["mul_out_nchw", "perm_nchw_to_nhwc"], outputs=["mul_out_nhwc"]),
        OperatorIR(
            op_type="CONV_2D",
            inputs=["mul_out_nhwc", "conv_w", "conv_b"],
            outputs=["y"],
            options={
                "padding": "SAME",
                "strideH": 1,
                "strideW": 1,
                "dilationHFactor": 1,
                "dilationWFactor": 1,
                "fusedActivationFunction": "NONE",
            },
        ),
    ]

    stats = _optimize_transpose_se_fc_mul_prepost_nhwc_chains(model_ir)
    assert stats["optimized_transpose_se_fc_mul_prepost_nhwc_chains"] == 1

    op_types = [str(op.op_type) for op in model_ir.operators]
    assert op_types == [
        "AVERAGE_POOL_2D",
        "RESHAPE",
        "FULLY_CONNECTED",
        "PRELU",
        "FULLY_CONNECTED",
        "PRELU",
        "RESHAPE",
        "MUL",
        "CONV_2D",
    ]
    assert op_types.count("TRANSPOSE") == 0

    gate_shape_vals = np.asarray(model_ir.tensors["shape_gate"].data, dtype=np.int32).reshape(-1).tolist()
    assert gate_shape_vals == [1, 1, 1, 12]

    gate_reshape_op = model_ir.operators[6]
    assert list(gate_reshape_op.options.get("newShape", [])) == [1, 1, 1, 12]
    assert list(gate_reshape_op.options.get("onnxRawNewShape", [])) == [1, 1, 1, 12]

    mul_op = model_ir.operators[7]
    assert "x_nhwc" in list(mul_op.inputs)
    assert "gate_nchw" in list(mul_op.inputs)

    conv_op = model_ir.operators[8]
    assert list(conv_op.inputs)[0] == "mul_out_nhwc"


def test_flatbuffer_direct_transpose_se_fc_mul_prepost_nhwc_chain_without_pool_pretranspose() -> None:
    model_ir = ModelIR(name="transpose_se_fc_mul_prepost_nhwc_no_pool_pretranspose_test")
    model_ir.inputs = ["x_nhwc"]
    model_ir.outputs = ["y"]

    def _add_tensor(name: str, shape: list[int], dtype: str = "FLOAT32", data: np.ndarray | None = None) -> None:
        model_ir.tensors[name] = TensorIR(
            name=name,
            dtype=dtype,
            shape=[int(v) for v in shape],
            shape_signature=[int(v) for v in shape],
            data=data,
            is_variable=False if data is not None else True,
        )

    _add_tensor("x_nhwc", [1, 80, 80, 12])
    _add_tensor("x_nchw", [1, 12, 80, 80])
    _add_tensor("pool_nhwc", [1, 1, 1, 12])
    _add_tensor("fc_in", [1, 12])
    _add_tensor("fc_out", [1, 12])
    _add_tensor("gate_fc", [1, 12])
    _add_tensor("gate_nchw", [1, 12, 1, 1])
    _add_tensor("mul_out_nchw", [1, 12, 80, 80])
    _add_tensor("mul_out_nhwc", [1, 80, 80, 12])
    _add_tensor("y", [1, 80, 80, 16])
    _add_tensor("perm_nhwc_to_nchw", [4], "INT32", np.asarray([0, 3, 1, 2], dtype=np.int32))
    _add_tensor("perm_nchw_to_nhwc", [4], "INT32", np.asarray([0, 2, 3, 1], dtype=np.int32))
    _add_tensor("shape_fc_in", [2], "INT32", np.asarray([1, 12], dtype=np.int32))
    _add_tensor("shape_gate", [4], "INT32", np.asarray([1, 12, 1, 1], dtype=np.int32))
    _add_tensor("fc_w", [12, 12], data=np.ones((12, 12), dtype=np.float32))
    _add_tensor("fc_b", [12], data=np.zeros((12,), dtype=np.float32))
    _add_tensor("prelu_slope", [12], data=np.ones((12,), dtype=np.float32))
    _add_tensor("conv_w", [1, 1, 12, 16], data=np.ones((1, 1, 12, 16), dtype=np.float32))
    _add_tensor("conv_b", [16], data=np.zeros((16,), dtype=np.float32))

    model_ir.operators = [
        OperatorIR(op_type="TRANSPOSE", inputs=["x_nhwc", "perm_nhwc_to_nchw"], outputs=["x_nchw"]),
        OperatorIR(
            op_type="AVERAGE_POOL_2D",
            inputs=["x_nhwc"],
            outputs=["pool_nhwc"],
            options={
                "padding": "VALID",
                "strideH": 80,
                "strideW": 80,
                "filterHeight": 80,
                "filterWidth": 80,
                "fusedActivationFunction": "NONE",
            },
        ),
        OperatorIR(
            op_type="RESHAPE",
            inputs=["pool_nhwc", "shape_fc_in"],
            outputs=["fc_in"],
            options={"newShape": [1, 12], "onnxRawNewShape": [1, 12]},
        ),
        OperatorIR(op_type="FULLY_CONNECTED", inputs=["fc_in", "fc_w", "fc_b"], outputs=["fc_out"], options={}),
        OperatorIR(op_type="PRELU", inputs=["fc_out", "prelu_slope"], outputs=["gate_fc"], options={}),
        OperatorIR(
            op_type="RESHAPE",
            inputs=["gate_fc", "shape_gate"],
            outputs=["gate_nchw"],
            options={"newShape": [1, 12, 1, 1], "onnxRawNewShape": [1, 12, 1, 1]},
        ),
        OperatorIR(op_type="MUL", inputs=["gate_nchw", "x_nchw"], outputs=["mul_out_nchw"], options={}),
        OperatorIR(op_type="TRANSPOSE", inputs=["mul_out_nchw", "perm_nchw_to_nhwc"], outputs=["mul_out_nhwc"]),
        OperatorIR(
            op_type="CONV_2D",
            inputs=["mul_out_nhwc", "conv_w", "conv_b"],
            outputs=["y"],
            options={
                "padding": "SAME",
                "strideH": 1,
                "strideW": 1,
                "dilationHFactor": 1,
                "dilationWFactor": 1,
                "fusedActivationFunction": "NONE",
            },
        ),
    ]

    stats = _optimize_transpose_se_fc_mul_prepost_nhwc_chains(model_ir)
    assert stats["optimized_transpose_se_fc_mul_prepost_nhwc_chains"] == 1

    op_types = [str(op.op_type) for op in model_ir.operators]
    assert op_types == ["AVERAGE_POOL_2D", "RESHAPE", "FULLY_CONNECTED", "PRELU", "RESHAPE", "MUL", "CONV_2D"]
    assert op_types.count("TRANSPOSE") == 0
    gate_shape_vals = np.asarray(model_ir.tensors["shape_gate"].data, dtype=np.int32).reshape(-1).tolist()
    assert gate_shape_vals == [1, 1, 1, 12]
    mul_op = model_ir.operators[5]
    assert "x_nhwc" in list(mul_op.inputs)
    assert "gate_nchw" in list(mul_op.inputs)


def test_flatbuffer_direct_singleton_channel_layout_transpose_rewritten_to_reshape() -> None:
    model_ir = ModelIR(name="singleton_channel_transpose_to_reshape_test")
    model_ir.inputs = ["x_nhwc"]
    model_ir.outputs = ["y_nchw"]
    model_ir.tensors["x_nhwc"] = TensorIR(
        name="x_nhwc",
        dtype="FLOAT32",
        shape=[1, 8, 8, 1],
        shape_signature=[1, 8, 8, 1],
    )
    model_ir.tensors["perm_nhwc_to_nchw"] = TensorIR(
        name="perm_nhwc_to_nchw",
        dtype="INT32",
        shape=[4],
        shape_signature=[4],
        data=np.asarray([0, 3, 1, 2], dtype=np.int32),
        is_variable=False,
    )
    model_ir.tensors["x_nchw"] = TensorIR(
        name="x_nchw",
        dtype="FLOAT32",
        shape=[1, 1, 8, 8],
        shape_signature=[1, 1, 8, 8],
    )
    model_ir.tensors["y_nchw"] = TensorIR(
        name="y_nchw",
        dtype="FLOAT32",
        shape=[1, 1, 8, 8],
        shape_signature=[1, 1, 8, 8],
    )
    model_ir.operators = [
        OperatorIR(op_type="TRANSPOSE", inputs=["x_nhwc", "perm_nhwc_to_nchw"], outputs=["x_nchw"]),
        OperatorIR(op_type="RELU", inputs=["x_nchw"], outputs=["y_nchw"]),
    ]

    stats = _optimize_singleton_channel_layout_transpose_to_reshape(model_ir)
    assert stats["rewritten_singleton_channel_layout_transpose_to_reshape"] == 1
    assert [str(op.op_type) for op in model_ir.operators] == ["RESHAPE", "RELU"]
    reshape_op = model_ir.operators[0]
    assert reshape_op.options.get("newShape") == [1, 1, 8, 8]
    assert len(list(reshape_op.inputs)) == 2


def test_flatbuffer_direct_singleton_spatial_layout_transpose_rewritten_to_reshape() -> None:
    model_ir = ModelIR(name="singleton_spatial_transpose_to_reshape_test")
    model_ir.inputs = ["x_nhwc", "y_nchw"]
    model_ir.outputs = ["x_nchw", "y_nhwc"]
    model_ir.tensors["x_nhwc"] = TensorIR(
        name="x_nhwc",
        dtype="FLOAT32",
        shape=[1, 1, 1, 5],
        shape_signature=[1, 1, 1, 5],
    )
    model_ir.tensors["y_nchw"] = TensorIR(
        name="y_nchw",
        dtype="FLOAT32",
        shape=[1, 7, 1, 1],
        shape_signature=[1, 7, 1, 1],
    )
    model_ir.tensors["perm_nhwc_to_nchw"] = TensorIR(
        name="perm_nhwc_to_nchw",
        dtype="INT32",
        shape=[4],
        shape_signature=[4],
        data=np.asarray([0, 3, 1, 2], dtype=np.int32),
        is_variable=False,
    )
    model_ir.tensors["perm_nchw_to_nhwc"] = TensorIR(
        name="perm_nchw_to_nhwc",
        dtype="INT32",
        shape=[4],
        shape_signature=[4],
        data=np.asarray([0, 2, 3, 1], dtype=np.int32),
        is_variable=False,
    )
    model_ir.tensors["x_nchw"] = TensorIR(
        name="x_nchw",
        dtype="FLOAT32",
        shape=[1, 5, 1, 1],
        shape_signature=[1, 5, 1, 1],
    )
    model_ir.tensors["y_nhwc"] = TensorIR(
        name="y_nhwc",
        dtype="FLOAT32",
        shape=[1, 1, 1, 7],
        shape_signature=[1, 1, 1, 7],
    )
    model_ir.operators = [
        OperatorIR(op_type="TRANSPOSE", inputs=["x_nhwc", "perm_nhwc_to_nchw"], outputs=["x_nchw"]),
        OperatorIR(op_type="TRANSPOSE", inputs=["y_nchw", "perm_nchw_to_nhwc"], outputs=["y_nhwc"]),
    ]

    stats = _optimize_singleton_channel_layout_transpose_to_reshape(model_ir)
    assert stats["rewritten_singleton_channel_layout_transpose_to_reshape"] == 2
    assert [str(op.op_type) for op in model_ir.operators] == ["RESHAPE", "RESHAPE"]
    assert model_ir.operators[0].options.get("newShape") == [1, 5, 1, 1]
    assert model_ir.operators[1].options.get("newShape") == [1, 1, 1, 7]


def test_flatbuffer_direct_singleton_moved_axis_transpose_rewritten_to_reshape() -> None:
    model_ir = ModelIR(name="singleton_moved_axis_transpose_to_reshape_test")
    model_ir.inputs = ["x"]
    model_ir.outputs = ["y"]
    model_ir.tensors["x"] = TensorIR(
        name="x",
        dtype="FLOAT32",
        shape=[1, 1, 64, 97],
        shape_signature=[1, 1, 64, 97],
    )
    model_ir.tensors["perm_swap_h_w"] = TensorIR(
        name="perm_swap_h_w",
        dtype="INT32",
        shape=[4],
        shape_signature=[4],
        data=np.asarray([0, 2, 1, 3], dtype=np.int32),
        is_variable=False,
    )
    model_ir.tensors["y"] = TensorIR(
        name="y",
        dtype="FLOAT32",
        shape=[1, 64, 1, 97],
        shape_signature=[1, 64, 1, 97],
    )
    model_ir.operators = [
        OperatorIR(op_type="TRANSPOSE", inputs=["x", "perm_swap_h_w"], outputs=["y"]),
    ]

    stats = _optimize_singleton_channel_layout_transpose_to_reshape(model_ir)
    assert stats["rewritten_singleton_channel_layout_transpose_to_reshape"] == 1
    assert [str(op.op_type) for op in model_ir.operators] == ["RESHAPE"]
    assert model_ir.operators[0].options.get("newShape") == [1, 64, 1, 97]


def test_flatbuffer_direct_non_singleton_order_change_transpose_not_rewritten_to_reshape() -> None:
    model_ir = ModelIR(name="non_singleton_order_change_transpose_test")
    model_ir.inputs = ["x"]
    model_ir.outputs = ["y"]
    model_ir.tensors["x"] = TensorIR(
        name="x",
        dtype="FLOAT32",
        shape=[1, 2, 64, 97],
        shape_signature=[1, 2, 64, 97],
    )
    model_ir.tensors["perm_rotate"] = TensorIR(
        name="perm_rotate",
        dtype="INT32",
        shape=[4],
        shape_signature=[4],
        data=np.asarray([0, 2, 3, 1], dtype=np.int32),
        is_variable=False,
    )
    model_ir.tensors["y"] = TensorIR(
        name="y",
        dtype="FLOAT32",
        shape=[1, 64, 97, 2],
        shape_signature=[1, 64, 97, 2],
    )
    model_ir.operators = [
        OperatorIR(op_type="TRANSPOSE", inputs=["x", "perm_rotate"], outputs=["y"]),
    ]

    stats = _optimize_singleton_channel_layout_transpose_to_reshape(model_ir)
    assert stats["rewritten_singleton_channel_layout_transpose_to_reshape"] == 0
    assert [str(op.op_type) for op in model_ir.operators] == ["TRANSPOSE"]


def test_flatbuffer_direct_singleton_layout_reshape_unary_passthrough_rewritten() -> None:
    model_ir = ModelIR(name="singleton_layout_reshape_unary_passthrough_test")
    model_ir.inputs = ["x_nhwc"]
    model_ir.outputs = ["z"]
    model_ir.tensors["x_nhwc"] = TensorIR(
        name="x_nhwc",
        dtype="FLOAT32",
        shape=[1, 1, 1, 80],
        shape_signature=[1, 1, 1, 80],
    )
    model_ir.tensors["x_nchw"] = TensorIR(
        name="x_nchw",
        dtype="FLOAT32",
        shape=[1, 80, 1, 1],
        shape_signature=[1, 80, 1, 1],
    )
    model_ir.tensors["u_nchw"] = TensorIR(
        name="u_nchw",
        dtype="FLOAT32",
        shape=[1, 80, 1, 1],
        shape_signature=[1, 80, 1, 1],
    )
    model_ir.tensors["u_nhwc"] = TensorIR(
        name="u_nhwc",
        dtype="FLOAT32",
        shape=[1, 1, 1, 80],
        shape_signature=[1, 1, 1, 80],
    )
    model_ir.tensors["bias"] = TensorIR(
        name="bias",
        dtype="FLOAT32",
        shape=[1, 1, 1, 80],
        shape_signature=[1, 1, 1, 80],
        data=np.ones((1, 1, 1, 80), dtype=np.float32),
        is_variable=False,
    )
    model_ir.tensors["z"] = TensorIR(
        name="z",
        dtype="FLOAT32",
        shape=[1, 1, 1, 80],
        shape_signature=[1, 1, 1, 80],
    )
    model_ir.tensors["shape_nchw"] = TensorIR(
        name="shape_nchw",
        dtype="INT32",
        shape=[4],
        shape_signature=[4],
        data=np.asarray([1, 80, 1, 1], dtype=np.int32),
        is_variable=False,
    )
    model_ir.tensors["shape_nhwc"] = TensorIR(
        name="shape_nhwc",
        dtype="INT32",
        shape=[4],
        shape_signature=[4],
        data=np.asarray([1, 1, 1, 80], dtype=np.int32),
        is_variable=False,
    )
    model_ir.operators = [
        OperatorIR(op_type="RESHAPE", inputs=["x_nhwc", "shape_nchw"], outputs=["x_nchw"], options={"newShape": [1, 80, 1, 1]}),
        OperatorIR(op_type="RELU", inputs=["x_nchw"], outputs=["u_nchw"]),
        OperatorIR(op_type="RESHAPE", inputs=["u_nchw", "shape_nhwc"], outputs=["u_nhwc"], options={"newShape": [1, 1, 1, 80]}),
        OperatorIR(op_type="ADD", inputs=["u_nhwc", "bias"], outputs=["z"]),
    ]

    stats = _optimize_singleton_layout_reshape_unary_passthrough_chains(model_ir)
    assert stats["rewritten_singleton_layout_reshape_unary_passthrough_chains"] == 1
    assert [str(op.op_type) for op in model_ir.operators] == ["RELU", "ADD"]
    assert list(model_ir.operators[0].inputs) == ["x_nhwc"]
    assert list(model_ir.operators[0].outputs) == ["u_nhwc"]
    assert list(model_ir.operators[1].inputs) == ["u_nhwc", "bias"]


def test_flatbuffer_direct_squeeze_reshape_identity_chains_removed() -> None:
    model_ir = ModelIR(name="squeeze_reshape_identity_chains_test")
    model_ir.inputs = ["mean_out"]
    model_ir.outputs = ["conv_out"]
    model_ir.tensors["mean_out"] = TensorIR(
        name="mean_out",
        dtype="FLOAT32",
        shape=[1, 1, 1, 32],
        shape_signature=[1, 1, 1, 32],
    )
    model_ir.tensors["mean_sq"] = TensorIR(
        name="mean_sq",
        dtype="FLOAT32",
        shape=[1, 32],
        shape_signature=[1, 32],
    )
    model_ir.tensors["mean_nhwc"] = TensorIR(
        name="mean_nhwc",
        dtype="FLOAT32",
        shape=[1, 1, 1, 32],
        shape_signature=[1, 1, 1, 32],
    )
    model_ir.tensors["reshape_shape"] = TensorIR(
        name="reshape_shape",
        dtype="INT32",
        shape=[4],
        shape_signature=[4],
        data=np.asarray([1, 1, 1, 32], dtype=np.int32),
        is_variable=False,
    )
    model_ir.tensors["conv_w"] = TensorIR(
        name="conv_w",
        dtype="FLOAT32",
        shape=[8, 1, 1, 32],
        shape_signature=[8, 1, 1, 32],
        data=np.ones((8, 1, 1, 32), dtype=np.float32),
        is_variable=False,
    )
    model_ir.tensors["conv_b"] = TensorIR(
        name="conv_b",
        dtype="FLOAT32",
        shape=[8],
        shape_signature=[8],
        data=np.zeros((8,), dtype=np.float32),
        is_variable=False,
    )
    model_ir.tensors["conv_out"] = TensorIR(
        name="conv_out",
        dtype="FLOAT32",
        shape=[1, 1, 1, 8],
        shape_signature=[1, 1, 1, 8],
    )
    model_ir.operators = [
        OperatorIR(op_type="SQUEEZE", inputs=["mean_out"], outputs=["mean_sq"], options={"squeezeDims": [1, 2]}),
        OperatorIR(
            op_type="RESHAPE",
            inputs=["mean_sq", "reshape_shape"],
            outputs=["mean_nhwc"],
            options={"newShape": [1, 1, 1, 32]},
        ),
        OperatorIR(
            op_type="CONV_2D",
            inputs=["mean_nhwc", "conv_w", "conv_b"],
            outputs=["conv_out"],
            options={
                "padding": "SAME",
                "strideH": 1,
                "strideW": 1,
                "dilationHFactor": 1,
                "dilationWFactor": 1,
                "fusedActivationFunction": "NONE",
            },
        ),
    ]

    stats = _optimize_squeeze_reshape_identity_chains(model_ir)
    assert stats["optimized_squeeze_reshape_identity_chains"] == 1
    assert [str(op.op_type) for op in model_ir.operators] == ["CONV_2D"]
    assert list(model_ir.operators[0].inputs) == ["mean_out", "conv_w", "conv_b"]


def test_flatbuffer_direct_consecutive_inverse_singleton_layout_reshapes_removed() -> None:
    model_ir = ModelIR(name="consecutive_inverse_singleton_layout_reshapes_test")
    model_ir.inputs = ["x_nhwc"]
    model_ir.outputs = ["y_nhwc"]
    model_ir.tensors["x_nhwc"] = TensorIR(
        name="x_nhwc",
        dtype="FLOAT32",
        shape=[1, 1, 1, 80],
        shape_signature=[1, 1, 1, 80],
    )
    model_ir.tensors["src_nhwc"] = TensorIR(
        name="src_nhwc",
        dtype="FLOAT32",
        shape=[1, 1, 1, 80],
        shape_signature=[1, 1, 1, 80],
    )
    model_ir.tensors["x_nchw"] = TensorIR(
        name="x_nchw",
        dtype="FLOAT32",
        shape=[1, 80, 1, 1],
        shape_signature=[1, 80, 1, 1],
    )
    model_ir.tensors["y_nhwc"] = TensorIR(
        name="y_nhwc",
        dtype="FLOAT32",
        shape=[1, 1, 1, 80],
        shape_signature=[1, 1, 1, 80],
    )
    model_ir.tensors["shape_nchw"] = TensorIR(
        name="shape_nchw",
        dtype="INT32",
        shape=[4],
        shape_signature=[4],
        data=np.asarray([1, 80, 1, 1], dtype=np.int32),
        is_variable=False,
    )
    model_ir.tensors["shape_nhwc"] = TensorIR(
        name="shape_nhwc",
        dtype="INT32",
        shape=[4],
        shape_signature=[4],
        data=np.asarray([1, 1, 1, 80], dtype=np.int32),
        is_variable=False,
    )
    model_ir.operators = [
        OperatorIR(op_type="RELU", inputs=["x_nhwc"], outputs=["src_nhwc"]),
        OperatorIR(op_type="RESHAPE", inputs=["src_nhwc", "shape_nchw"], outputs=["x_nchw"], options={"newShape": [1, 80, 1, 1]}),
        OperatorIR(op_type="RESHAPE", inputs=["x_nchw", "shape_nhwc"], outputs=["y_nhwc"], options={"newShape": [1, 1, 1, 80]}),
    ]

    stats = _optimize_consecutive_inverse_singleton_layout_reshapes(model_ir)
    assert stats["rewritten_consecutive_inverse_singleton_layout_reshapes"] == 1
    assert [str(op.op_type) for op in model_ir.operators] == ["RELU"]
    assert model_ir.outputs == ["y_nhwc"]
    assert "y_nhwc" in model_ir.tensors


def test_flatbuffer_direct_consecutive_reshape_passthrough_chains_removed() -> None:
    model_ir = ModelIR(name="consecutive_reshape_passthrough_test")
    model_ir.inputs = ["x0", "x1"]
    model_ir.outputs = ["y0", "y1"]

    model_ir.tensors["x0"] = TensorIR(name="x0", dtype="FLOAT32", shape=[1, 1, 1, 512], shape_signature=[1, 1, 1, 512])
    model_ir.tensors["m0"] = TensorIR(name="m0", dtype="FLOAT32", shape=[1, 512, 1, 1], shape_signature=[1, 512, 1, 1])
    model_ir.tensors["y0"] = TensorIR(name="y0", dtype="FLOAT32", shape=[1, 512], shape_signature=[1, 512])
    model_ir.tensors["x1"] = TensorIR(name="x1", dtype="FLOAT32", shape=[1, 515], shape_signature=[1, 515])
    model_ir.tensors["m1"] = TensorIR(name="m1", dtype="FLOAT32", shape=[1, 515, 1, 1], shape_signature=[1, 515, 1, 1])
    model_ir.tensors["y1"] = TensorIR(name="y1", dtype="FLOAT32", shape=[1, 1, 1, 515], shape_signature=[1, 1, 1, 515])
    model_ir.tensors["shape_m0"] = TensorIR(
        name="shape_m0",
        dtype="INT32",
        shape=[4],
        shape_signature=[4],
        data=np.asarray([1, 512, 1, 1], dtype=np.int32),
        is_variable=False,
    )
    model_ir.tensors["shape_y0"] = TensorIR(
        name="shape_y0",
        dtype="INT32",
        shape=[2],
        shape_signature=[2],
        data=np.asarray([1, 512], dtype=np.int32),
        is_variable=False,
    )
    model_ir.tensors["shape_m1"] = TensorIR(
        name="shape_m1",
        dtype="INT32",
        shape=[4],
        shape_signature=[4],
        data=np.asarray([1, 515, 1, 1], dtype=np.int32),
        is_variable=False,
    )
    model_ir.tensors["shape_y1"] = TensorIR(
        name="shape_y1",
        dtype="INT32",
        shape=[4],
        shape_signature=[4],
        data=np.asarray([1, 1, 1, 515], dtype=np.int32),
        is_variable=False,
    )
    model_ir.operators = [
        OperatorIR(op_type="RESHAPE", inputs=["x0", "shape_m0"], outputs=["m0"], options={"newShape": [1, 512, 1, 1]}),
        OperatorIR(op_type="RESHAPE", inputs=["m0", "shape_y0"], outputs=["y0"], options={"newShape": [1, 512]}),
        OperatorIR(op_type="RESHAPE", inputs=["x1", "shape_m1"], outputs=["m1"], options={"newShape": [1, 515, 1, 1]}),
        OperatorIR(op_type="RESHAPE", inputs=["m1", "shape_y1"], outputs=["y1"], options={"newShape": [1, 1, 1, 515]}),
    ]

    stats = _optimize_consecutive_reshape_passthrough_chains(model_ir)
    assert stats["rewritten_consecutive_reshape_passthrough_chains"] == 2
    assert [str(op.op_type) for op in model_ir.operators] == ["RESHAPE", "RESHAPE"]
    assert list(model_ir.operators[0].inputs) == ["x0", "shape_y0"]
    assert list(model_ir.operators[1].inputs) == ["x1", "shape_y1"]


def test_flatbuffer_direct_flatten_concat_expanddims_to_nhwc_concat_rewritten() -> None:
    model_ir = ModelIR(name="flatten_concat_expanddims_to_nhwc_concat_test")
    model_ir.inputs = ["a4d", "b2d"]
    model_ir.outputs = ["z4d"]
    model_ir.tensors["a4d"] = TensorIR(
        name="a4d",
        dtype="FLOAT32",
        shape=[1, 1, 1, 4],
        shape_signature=[1, 1, 1, 4],
    )
    model_ir.tensors["a2d"] = TensorIR(
        name="a2d",
        dtype="FLOAT32",
        shape=[1, 4],
        shape_signature=[1, 4],
    )
    model_ir.tensors["b2d"] = TensorIR(
        name="b2d",
        dtype="FLOAT32",
        shape=[1, 3],
        shape_signature=[1, 3],
    )
    model_ir.tensors["c2d"] = TensorIR(
        name="c2d",
        dtype="FLOAT32",
        shape=[1, 7],
        shape_signature=[1, 7],
    )
    model_ir.tensors["z4d"] = TensorIR(
        name="z4d",
        dtype="FLOAT32",
        shape=[1, 1, 1, 7],
        shape_signature=[1, 1, 1, 7],
    )
    model_ir.tensors["shape_a2d"] = TensorIR(
        name="shape_a2d",
        dtype="INT32",
        shape=[2],
        shape_signature=[2],
        data=np.asarray([1, 4], dtype=np.int32),
        is_variable=False,
    )
    model_ir.tensors["shape_z4d"] = TensorIR(
        name="shape_z4d",
        dtype="INT32",
        shape=[4],
        shape_signature=[4],
        data=np.asarray([1, 1, 1, 7], dtype=np.int32),
        is_variable=False,
    )
    model_ir.operators = [
        OperatorIR(op_type="RESHAPE", inputs=["a4d", "shape_a2d"], outputs=["a2d"], options={"newShape": [1, 4]}),
        OperatorIR(op_type="CONCATENATION", inputs=["a2d", "b2d"], outputs=["c2d"], options={"axis": 1}),
        OperatorIR(op_type="RESHAPE", inputs=["c2d", "shape_z4d"], outputs=["z4d"], options={"newShape": [1, 1, 1, 7]}),
    ]

    stats = _optimize_flatten_concat_expanddims_to_nhwc_concat(model_ir)
    assert stats["rewritten_flatten_concat_expanddims_to_nhwc_concat"] == 1
    op_types = [str(op.op_type) for op in model_ir.operators]
    assert op_types.count("RESHAPE") == 1
    assert op_types.count("CONCATENATION") == 1
    concat_op = next(op for op in model_ir.operators if str(op.op_type) == "CONCATENATION")
    assert int(concat_op.options.get("axis", -1)) == 3
    assert list(concat_op.outputs) == ["z4d"]


def test_flatbuffer_direct_non_singleton_channel_transpose_kept() -> None:
    model_ir = ModelIR(name="non_singleton_channel_transpose_kept_test")
    model_ir.inputs = ["x_nhwc"]
    model_ir.outputs = ["x_nchw"]
    model_ir.tensors["x_nhwc"] = TensorIR(
        name="x_nhwc",
        dtype="FLOAT32",
        shape=[1, 8, 8, 2],
        shape_signature=[1, 8, 8, 2],
    )
    model_ir.tensors["perm_nhwc_to_nchw"] = TensorIR(
        name="perm_nhwc_to_nchw",
        dtype="INT32",
        shape=[4],
        shape_signature=[4],
        data=np.asarray([0, 3, 1, 2], dtype=np.int32),
        is_variable=False,
    )
    model_ir.tensors["x_nchw"] = TensorIR(
        name="x_nchw",
        dtype="FLOAT32",
        shape=[1, 2, 8, 8],
        shape_signature=[1, 2, 8, 8],
    )
    model_ir.operators = [
        OperatorIR(op_type="TRANSPOSE", inputs=["x_nhwc", "perm_nhwc_to_nchw"], outputs=["x_nchw"]),
    ]

    stats = _optimize_singleton_channel_layout_transpose_to_reshape(model_ir)
    assert stats["rewritten_singleton_channel_layout_transpose_to_reshape"] == 0
    assert [str(op.op_type) for op in model_ir.operators] == ["TRANSPOSE"]


def test_flatbuffer_direct_singleton_channel_dynamic_signature_transpose_kept() -> None:
    model_ir = ModelIR(name="singleton_channel_dynamic_signature_transpose_kept_test")
    model_ir.inputs = ["x_nhwc"]
    model_ir.outputs = ["x_nchw"]
    model_ir.tensors["x_nhwc"] = TensorIR(
        name="x_nhwc",
        dtype="FLOAT32",
        shape=[1, 1, 1, 1],
        shape_signature=[1, 1, -1, 1],
    )
    model_ir.tensors["perm_nhwc_to_nchw"] = TensorIR(
        name="perm_nhwc_to_nchw",
        dtype="INT32",
        shape=[4],
        shape_signature=[4],
        data=np.asarray([0, 3, 1, 2], dtype=np.int32),
        is_variable=False,
    )
    model_ir.tensors["x_nchw"] = TensorIR(
        name="x_nchw",
        dtype="FLOAT32",
        shape=[1, 1, 1, 1],
        shape_signature=[1, 1, 1, -1],
    )
    model_ir.operators = [
        OperatorIR(op_type="TRANSPOSE", inputs=["x_nhwc", "perm_nhwc_to_nchw"], outputs=["x_nchw"]),
    ]

    stats = _optimize_singleton_channel_layout_transpose_to_reshape(model_ir)
    assert stats["rewritten_singleton_channel_layout_transpose_to_reshape"] == 0
    assert [str(op.op_type) for op in model_ir.operators] == ["TRANSPOSE"]


def test_flatbuffer_direct_singleton_reshape_concat_post_transpose_rewritten_to_nhwc() -> None:
    model_ir = ModelIR(name="singleton_reshape_concat_post_transpose_to_nhwc_test")
    model_ir.inputs = ["split0_nhwc", "split1_nhwc", "split2_nhwc", "clip_nhwc"]
    model_ir.outputs = ["z"]

    def _add_tensor(name: str, shape: list[int]) -> None:
        model_ir.tensors[name] = TensorIR(
            name=name,
            dtype="FLOAT32",
            shape=[int(v) for v in shape],
            shape_signature=[int(v) for v in shape],
        )

    _add_tensor("split0_nhwc", [1, 8, 8, 1])
    _add_tensor("split1_nhwc", [1, 8, 8, 1])
    _add_tensor("split2_nhwc", [1, 8, 8, 1])
    _add_tensor("clip_nhwc", [1, 8, 8, 1])
    _add_tensor("split0_nchw", [1, 1, 8, 8])
    _add_tensor("split1_nchw", [1, 1, 8, 8])
    _add_tensor("split2_nchw", [1, 1, 8, 8])
    _add_tensor("clip_nchw", [1, 1, 8, 8])
    _add_tensor("mul_out", [1, 1, 8, 8])
    _add_tensor("concat_nchw", [1, 4, 8, 8])
    _add_tensor("concat_nhwc", [1, 8, 8, 4])
    _add_tensor("conv_w", [4, 1, 1, 4])
    _add_tensor("conv_b", [4])
    _add_tensor("z", [1, 8, 8, 4])
    model_ir.tensors["conv_w"].data = np.ones((4, 1, 1, 4), dtype=np.float32)
    model_ir.tensors["conv_w"].is_variable = False
    model_ir.tensors["conv_b"].data = np.zeros((4,), dtype=np.float32)
    model_ir.tensors["conv_b"].is_variable = False

    def _shape_const(name: str, shape: list[int]) -> None:
        model_ir.tensors[name] = TensorIR(
            name=name,
            dtype="INT32",
            shape=[len(shape)],
            shape_signature=[len(shape)],
            data=np.asarray(shape, dtype=np.int32),
            is_variable=False,
        )

    _shape_const("to_nchw_shape", [1, 1, 8, 8])
    _shape_const("to_nhwc_shape", [1, 8, 8, 1])

    model_ir.tensors["perm_nchw_to_nhwc"] = TensorIR(
        name="perm_nchw_to_nhwc",
        dtype="INT32",
        shape=[4],
        shape_signature=[4],
        data=np.asarray([0, 2, 3, 1], dtype=np.int32),
        is_variable=False,
    )

    conv_options = {
        "padding": "SAME",
        "strideH": 1,
        "strideW": 1,
        "dilationHFactor": 1,
        "dilationWFactor": 1,
        "fusedActivationFunction": "NONE",
    }
    model_ir.operators = [
        OperatorIR(op_type="RESHAPE", inputs=["split0_nhwc", "to_nchw_shape"], outputs=["split0_nchw"], options={"newShape": [1, 1, 8, 8]}),
        OperatorIR(op_type="RESHAPE", inputs=["split1_nhwc", "to_nchw_shape"], outputs=["split1_nchw"], options={"newShape": [1, 1, 8, 8]}),
        OperatorIR(op_type="RESHAPE", inputs=["split2_nhwc", "to_nchw_shape"], outputs=["split2_nchw"], options={"newShape": [1, 1, 8, 8]}),
        OperatorIR(op_type="RESHAPE", inputs=["clip_nhwc", "to_nchw_shape"], outputs=["clip_nchw"], options={"newShape": [1, 1, 8, 8]}),
        # keep split0_nchw live outside concat so its reshape must remain
        OperatorIR(op_type="MUL", inputs=["split0_nchw", "clip_nchw"], outputs=["mul_out"], options={"fusedActivationFunction": "NONE"}),
        OperatorIR(
            op_type="CONCATENATION",
            inputs=["split0_nchw", "split1_nchw", "split2_nchw", "clip_nchw"],
            outputs=["concat_nchw"],
            options={"axis": 1, "fusedActivationFunction": "NONE"},
        ),
        OperatorIR(op_type="TRANSPOSE", inputs=["concat_nchw", "perm_nchw_to_nhwc"], outputs=["concat_nhwc"]),
        OperatorIR(op_type="CONV_2D", inputs=["concat_nhwc", "conv_w", "conv_b"], outputs=["z"], options=conv_options),
    ]

    stats = _optimize_singleton_reshape_concat_post_transpose_nhwc_chains(model_ir)
    assert stats["rewritten_singleton_reshape_concat_post_transpose_nhwc_chains"] == 1

    op_types = [str(op.op_type) for op in model_ir.operators]
    assert op_types.count("TRANSPOSE") == 0
    concat_op = next(op for op in model_ir.operators if str(op.op_type) == "CONCATENATION")
    assert int(concat_op.options.get("axis", -1)) == 3
    assert list(concat_op.outputs) == ["concat_nhwc"]
    assert "split1_nchw" not in list(concat_op.inputs)
    assert "split2_nchw" not in list(concat_op.inputs)
    assert "split1_nhwc" in list(concat_op.inputs)
    assert "split2_nhwc" in list(concat_op.inputs)

    # split0 reshape is still required by MUL branch
    mul_op = next(op for op in model_ir.operators if str(op.op_type) == "MUL")
    assert "split0_nchw" in list(mul_op.inputs)


def test_flatbuffer_direct_duplicate_transpose_fanout_deduplicates_shared_layout_adapter() -> None:
    model_ir = ModelIR(name="duplicate_transpose_fanout_dedup_test")
    model_ir.inputs = ["x"]
    model_ir.outputs = ["y0", "y1"]
    model_ir.tensors["x"] = TensorIR(
        name="x",
        dtype="FLOAT32",
        shape=[1, 16, 8, 8],
        shape_signature=[1, 16, 8, 8],
    )
    model_ir.tensors["perm"] = TensorIR(
        name="perm",
        dtype="INT32",
        shape=[4],
        shape_signature=[4],
        data=np.asarray([0, 2, 3, 1], dtype=np.int32),
        is_variable=False,
    )
    model_ir.tensors["x_t0"] = TensorIR(
        name="x_t0",
        dtype="FLOAT32",
        shape=[1, 8, 8, 16],
        shape_signature=[1, 8, 8, 16],
    )
    model_ir.tensors["x_t1"] = TensorIR(
        name="x_t1",
        dtype="FLOAT32",
        shape=[1, 8, 8, 16],
        shape_signature=[1, 8, 8, 16],
    )
    model_ir.tensors["y0"] = TensorIR(
        name="y0",
        dtype="FLOAT32",
        shape=[1, 8, 8, 16],
        shape_signature=[1, 8, 8, 16],
    )
    model_ir.tensors["y1"] = TensorIR(
        name="y1",
        dtype="FLOAT32",
        shape=[1, 8, 8, 16],
        shape_signature=[1, 8, 8, 16],
    )
    model_ir.operators = [
        OperatorIR(op_type="TRANSPOSE", inputs=["x", "perm"], outputs=["x_t0"]),
        OperatorIR(op_type="TRANSPOSE", inputs=["x", "perm"], outputs=["x_t1"]),
        OperatorIR(op_type="RELU", inputs=["x_t0"], outputs=["y0"]),
        OperatorIR(op_type="RELU", inputs=["x_t1"], outputs=["y1"]),
    ]

    stats = _optimize_duplicate_transpose_fanout(model_ir)
    assert stats["removed_duplicate_transpose_fanout"] == 1
    assert [str(op.op_type) for op in model_ir.operators].count("TRANSPOSE") == 1
    relu_ops = [op for op in model_ir.operators if str(op.op_type) == "RELU"]
    assert len(relu_ops) == 2
    assert list(relu_ops[0].inputs) == ["x_t0"]
    assert list(relu_ops[1].inputs) == ["x_t0"]


def test_flatbuffer_direct_center_size_offset_terminal_transpose_optimization() -> None:
    model_ir = ModelIR(name="center_size_offset_terminal_transpose_opt_test")
    model_ir.inputs = ["center_nhwc", "size_nhwc", "offset_nhwc"]
    model_ir.outputs = ["center_flat", "gather_size", "gather_offset"]
    model_ir.tensors["center_nhwc"] = TensorIR(
        name="center_nhwc",
        dtype="FLOAT32",
        shape=[1, 7, 7, 1],
        shape_signature=[1, 7, 7, 1],
    )
    model_ir.tensors["size_nhwc"] = TensorIR(
        name="size_nhwc",
        dtype="FLOAT32",
        shape=[1, 7, 7, 2],
        shape_signature=[1, 7, 7, 2],
    )
    model_ir.tensors["offset_nhwc"] = TensorIR(
        name="offset_nhwc",
        dtype="FLOAT32",
        shape=[1, 7, 7, 2],
        shape_signature=[1, 7, 7, 2],
    )
    model_ir.tensors["perm_nhwc_to_nchw"] = TensorIR(
        name="perm_nhwc_to_nchw",
        dtype="INT32",
        shape=[4],
        shape_signature=[4],
        data=np.asarray([0, 3, 1, 2], dtype=np.int32),
        is_variable=False,
    )
    model_ir.tensors["center_nchw"] = TensorIR(
        name="center_nchw",
        dtype="FLOAT32",
        shape=[1, 1, 7, 7],
        shape_signature=[1, 1, 7, 7],
    )
    model_ir.tensors["size_nchw"] = TensorIR(
        name="size_nchw",
        dtype="FLOAT32",
        shape=[1, 2, 7, 7],
        shape_signature=[1, 2, 7, 7],
    )
    model_ir.tensors["offset_nchw"] = TensorIR(
        name="offset_nchw",
        dtype="FLOAT32",
        shape=[1, 2, 7, 7],
        shape_signature=[1, 2, 7, 7],
    )
    model_ir.tensors["clip_min"] = TensorIR(
        name="clip_min",
        dtype="FLOAT32",
        shape=[1],
        shape_signature=[1],
        data=np.asarray([0.0], dtype=np.float32),
        is_variable=False,
    )
    model_ir.tensors["clip_max"] = TensorIR(
        name="clip_max",
        dtype="FLOAT32",
        shape=[1],
        shape_signature=[1],
        data=np.asarray([1.0], dtype=np.float32),
        is_variable=False,
    )
    model_ir.tensors["center_sig"] = TensorIR(
        name="center_sig",
        dtype="FLOAT32",
        shape=[1, 1, 7, 7],
        shape_signature=[1, 1, 7, 7],
    )
    model_ir.tensors["center_clip_min_out"] = TensorIR(
        name="center_clip_min_out",
        dtype="FLOAT32",
        shape=[1, 1, 7, 7],
        shape_signature=[1, 1, 7, 7],
    )
    model_ir.tensors["center_clip_out"] = TensorIR(
        name="center_clip_out",
        dtype="FLOAT32",
        shape=[1, 1, 7, 7],
        shape_signature=[1, 1, 7, 7],
    )
    model_ir.tensors["size_sig"] = TensorIR(
        name="size_sig",
        dtype="FLOAT32",
        shape=[1, 2, 7, 7],
        shape_signature=[1, 2, 7, 7],
    )
    model_ir.tensors["size_clip_min_out"] = TensorIR(
        name="size_clip_min_out",
        dtype="FLOAT32",
        shape=[1, 2, 7, 7],
        shape_signature=[1, 2, 7, 7],
    )
    model_ir.tensors["size_clip_out"] = TensorIR(
        name="size_clip_out",
        dtype="FLOAT32",
        shape=[1, 2, 7, 7],
        shape_signature=[1, 2, 7, 7],
    )
    model_ir.tensors["center_flat_shape"] = TensorIR(
        name="center_flat_shape",
        dtype="INT64",
        shape=[2],
        shape_signature=[2],
        data=np.asarray([1, 49], dtype=np.int64),
        is_variable=False,
    )
    model_ir.tensors["size_reshape_shape"] = TensorIR(
        name="size_reshape_shape",
        dtype="INT64",
        shape=[3],
        shape_signature=[3],
        data=np.asarray([1, 2, 49], dtype=np.int64),
        is_variable=False,
    )
    model_ir.tensors["offset_reshape_shape"] = TensorIR(
        name="offset_reshape_shape",
        dtype="INT64",
        shape=[3],
        shape_signature=[3],
        data=np.asarray([1, 2, 49], dtype=np.int64),
        is_variable=False,
    )
    model_ir.tensors["center_flat"] = TensorIR(
        name="center_flat",
        dtype="FLOAT32",
        shape=[1, 49],
        shape_signature=[1, 49],
    )
    model_ir.tensors["size_reshaped"] = TensorIR(
        name="size_reshaped",
        dtype="FLOAT32",
        shape=[1, 2, 49],
        shape_signature=[1, 2, 49],
    )
    model_ir.tensors["offset_reshaped"] = TensorIR(
        name="offset_reshaped",
        dtype="FLOAT32",
        shape=[1, 2, 49],
        shape_signature=[1, 2, 49],
    )
    model_ir.tensors["indices_i32"] = TensorIR(
        name="indices_i32",
        dtype="INT32",
        shape=[1, 2, 1],
        shape_signature=[1, 2, 1],
        data=np.asarray([[[5], [5]]], dtype=np.int32),
        is_variable=False,
    )
    model_ir.tensors["axis_coord_shape"] = TensorIR(
        name="axis_coord_shape",
        dtype="INT64",
        shape=[4],
        shape_signature=[4],
        data=np.asarray([1, 2, 1, 1], dtype=np.int64),
        is_variable=False,
    )
    model_ir.tensors["axis_coord"] = TensorIR(
        name="axis_coord",
        dtype="INT32",
        shape=[1, 2, 1, 1],
        shape_signature=[1, 2, 1, 1],
    )
    model_ir.tensors["coord0"] = TensorIR(
        name="coord0",
        dtype="INT32",
        shape=[1, 2, 1, 1],
        shape_signature=[1, 2, 1, 1],
        data=np.asarray([[[[0]], [[0]]]], dtype=np.int32),
        is_variable=False,
    )
    model_ir.tensors["coord1"] = TensorIR(
        name="coord1",
        dtype="INT32",
        shape=[1, 2, 1, 1],
        shape_signature=[1, 2, 1, 1],
        data=np.asarray([[[[0]], [[1]]]], dtype=np.int32),
        is_variable=False,
    )
    model_ir.tensors["coords_size"] = TensorIR(
        name="coords_size",
        dtype="INT32",
        shape=[1, 2, 1, 3],
        shape_signature=[1, 2, 1, 3],
    )
    model_ir.tensors["coords_offset"] = TensorIR(
        name="coords_offset",
        dtype="INT32",
        shape=[1, 2, 1, 3],
        shape_signature=[1, 2, 1, 3],
    )
    model_ir.tensors["gather_size"] = TensorIR(
        name="gather_size",
        dtype="FLOAT32",
        shape=[1, 2, 1],
        shape_signature=[1, 2, 1],
    )
    model_ir.tensors["gather_offset"] = TensorIR(
        name="gather_offset",
        dtype="FLOAT32",
        shape=[1, 2, 1],
        shape_signature=[1, 2, 1],
    )

    model_ir.operators = [
        OperatorIR(op_type="TRANSPOSE", inputs=["center_nhwc", "perm_nhwc_to_nchw"], outputs=["center_nchw"]),
        OperatorIR(op_type="TRANSPOSE", inputs=["size_nhwc", "perm_nhwc_to_nchw"], outputs=["size_nchw"]),
        OperatorIR(op_type="TRANSPOSE", inputs=["offset_nhwc", "perm_nhwc_to_nchw"], outputs=["offset_nchw"]),
        OperatorIR(op_type="LOGISTIC", inputs=["center_nchw"], outputs=["center_sig"]),
        OperatorIR(op_type="LOGISTIC", inputs=["size_nchw"], outputs=["size_sig"]),
        OperatorIR(op_type="MAXIMUM", inputs=["center_sig", "clip_min"], outputs=["center_clip_min_out"]),
        OperatorIR(op_type="MINIMUM", inputs=["center_clip_min_out", "clip_max"], outputs=["center_clip_out"]),
        OperatorIR(op_type="MAXIMUM", inputs=["size_sig", "clip_min"], outputs=["size_clip_min_out"]),
        OperatorIR(op_type="MINIMUM", inputs=["size_clip_min_out", "clip_max"], outputs=["size_clip_out"]),
        OperatorIR(
            op_type="RESHAPE",
            inputs=["center_clip_out", "center_flat_shape"],
            outputs=["center_flat"],
            options={"newShape": [1, 49]},
        ),
        OperatorIR(
            op_type="RESHAPE",
            inputs=["size_clip_out", "size_reshape_shape"],
            outputs=["size_reshaped"],
            options={"newShape": [1, 2, 49], "onnxRawNewShape": [1, 2, 49]},
        ),
        OperatorIR(
            op_type="RESHAPE",
            inputs=["offset_nchw", "offset_reshape_shape"],
            outputs=["offset_reshaped"],
            options={"newShape": [1, 2, 49], "onnxRawNewShape": [1, 2, 49]},
        ),
        OperatorIR(
            op_type="RESHAPE",
            inputs=["indices_i32", "axis_coord_shape"],
            outputs=["axis_coord"],
            options={"newShape": [1, 2, 1, 1]},
        ),
        OperatorIR(
            op_type="CONCATENATION",
            inputs=["coord0", "coord1", "axis_coord"],
            outputs=["coords_size"],
            options={"axis": 3, "fusedActivationFunction": "NONE"},
        ),
        OperatorIR(op_type="GATHER_ND", inputs=["size_reshaped", "coords_size"], outputs=["gather_size"]),
        OperatorIR(
            op_type="CONCATENATION",
            inputs=["coord0", "coord1", "axis_coord"],
            outputs=["coords_offset"],
            options={"axis": 3, "fusedActivationFunction": "NONE"},
        ),
        OperatorIR(op_type="GATHER_ND", inputs=["offset_reshaped", "coords_offset"], outputs=["gather_offset"]),
    ]

    stats = _optimize_center_size_offset_terminal_transpose_chains(model_ir)
    assert stats["optimized_center_size_offset_terminal_transpose_chains"] == 1
    assert [str(op.op_type) for op in model_ir.operators].count("TRANSPOSE") == 0

    size_shape = model_ir.tensors["size_reshape_shape"]
    offset_shape = model_ir.tensors["offset_reshape_shape"]
    assert np.asarray(size_shape.data).tolist() == [1, 49, 2]
    assert np.asarray(offset_shape.data).tolist() == [1, 49, 2]

    concat_size = next(op for op in model_ir.operators if list(op.outputs) == ["coords_size"])
    concat_offset = next(op for op in model_ir.operators if list(op.outputs) == ["coords_offset"])
    assert list(concat_size.inputs) == ["coord0", "axis_coord", "coord1"]
    assert list(concat_offset.inputs) == ["coord0", "axis_coord", "coord1"]

    center_log = next(op for op in model_ir.operators if list(op.outputs) == ["center_sig"])
    size_log = next(op for op in model_ir.operators if list(op.outputs) == ["size_sig"])
    offset_reshape = next(op for op in model_ir.operators if list(op.outputs) == ["offset_reshaped"])
    assert list(center_log.inputs) == ["center_nhwc"]
    assert list(size_log.inputs) == ["size_nhwc"]
    assert list(offset_reshape.inputs)[0] == "offset_nhwc"


def test_flatbuffer_direct_qlinear_global_average_pool_lowering() -> None:
    model = _make_qlinear_global_average_pool_model()
    register_default_preprocess_rules()
    preprocessed_model, _ = run_preprocess_pipeline(onnx_graph=model)
    model_ir = lower_onnx_to_ir(
        onnx_graph=preprocessed_model,
        output_file_name="qlinear_global_average_pool_lowering_test",
    )
    op_types = [str(op.op_type) for op in model_ir.operators]
    assert op_types.count("AVERAGE_POOL_2D") == 1
    assert op_types.count("MEAN") == 0
    assert op_types.count("DEQUANTIZE") >= 1

    y = model_ir.tensors.get("y")
    assert y is not None
    assert list(y.shape) == [1, 1, 1, 2]
    assert list(y.shape_signature) == [1, 1, 1, 2]


def test_flatbuffer_direct_qlinear_concat_lowering() -> None:
    model = _make_qlinear_concat_model()
    register_default_preprocess_rules()
    preprocessed_model, _ = run_preprocess_pipeline(onnx_graph=model)
    model_ir = lower_onnx_to_ir(
        onnx_graph=preprocessed_model,
        output_file_name="qlinear_concat_lowering_test",
    )
    op_types = [str(op.op_type) for op in model_ir.operators]
    assert op_types.count("CONCATENATION") == 1
    assert op_types.count("DEQUANTIZE") >= 1
    assert op_types.count("QUANTIZE") >= 1

    y = model_ir.tensors.get("y")
    assert y is not None
    assert list(y.shape) == [1, 2, 2, 2]
    assert list(y.shape_signature) == [1, 2, 2, 2]


def test_flatbuffer_direct_elides_inverse_transpose_chain_at_generation() -> None:
    model = _make_qlinear_conv_pair_model()
    register_default_preprocess_rules()
    preprocessed_model, _ = run_preprocess_pipeline(onnx_graph=model)
    model_ir = lower_onnx_to_ir(
        onnx_graph=preprocessed_model,
        output_file_name="transpose_generation_elide_test",
        optimize_layout_transpose_chains=False,
    )
    transpose_ops = [op for op in model_ir.operators if str(op.op_type) == "TRANSPOSE"]
    transpose_outputs = {str(op.outputs[0]) for op in transpose_ops if len(op.outputs) == 1}

    # PairQConv1 output(NCHW) -> PairQConv2 input(NHWC) bridge is suppressed at generation time.
    assert "PairQConv2_input_nhwc" not in transpose_outputs
    assert len(transpose_ops) == 1


def test_flatbuffer_direct_no_dead_operator_outputs_after_prune() -> None:
    model = _make_qlinear_conv_pair_model()
    register_default_preprocess_rules()
    preprocessed_model, _ = run_preprocess_pipeline(onnx_graph=model)
    model_ir = lower_onnx_to_ir(
        onnx_graph=preprocessed_model,
        output_file_name="dead_operator_prune_test",
        optimize_layout_transpose_chains=False,
    )

    consumed_tensors = set()
    produced_tensors = []
    for op in model_ir.operators:
        for input_name in op.inputs:
            consumed_tensors.add(str(input_name))
        for output_name in op.outputs:
            produced_tensors.append(str(output_name))

    graph_outputs = set(str(v) for v in model_ir.outputs)
    dead_outputs = [
        name for name in produced_tensors
        if name not in consumed_tensors and name not in graph_outputs
    ]
    assert dead_outputs == []


def test_flatbuffer_direct_leakyrelu_lowering_uses_builtin_without_layout_optimizations() -> None:
    model = _make_leakyrelu_model()
    register_default_preprocess_rules()
    preprocessed_model, _ = run_preprocess_pipeline(onnx_graph=model)
    model_ir = lower_onnx_to_ir(
        onnx_graph=preprocessed_model,
        output_file_name="leakyrelu_builtin_no_layout_opt_test",
        optimize_layout_transpose_chains=False,
    )

    op_types = [str(op.op_type) for op in model_ir.operators]
    assert op_types == ["LEAKY_RELU"]
    leaky_op = model_ir.operators[0]
    assert math.isclose(float(leaky_op.options.get("alpha", -1.0)), 0.2, rel_tol=0.0, abs_tol=1e-7)


def test_flatbuffer_direct_hardswish_lowering_uses_builtin_without_layout_optimizations() -> None:
    model = _make_hardswish_model()
    register_default_preprocess_rules()
    preprocessed_model, _ = run_preprocess_pipeline(onnx_graph=model)
    model_ir = lower_onnx_to_ir(
        onnx_graph=preprocessed_model,
        output_file_name="hardswish_builtin_no_layout_opt_test",
        optimize_layout_transpose_chains=False,
    )

    op_types = [str(op.op_type) for op in model_ir.operators]
    assert op_types == ["HARD_SWISH"]
    hardswish_op = model_ir.operators[0]
    assert list(hardswish_op.inputs) == ["x"]
    assert list(hardswish_op.outputs) == ["y"]


def test_flatbuffer_direct_clip_relu_n1_to_1_lowering_uses_builtin_without_layout_optimizations() -> None:
    model = _make_clip_relu_n1_to_1_model()
    register_default_preprocess_rules()
    preprocessed_model, _ = run_preprocess_pipeline(onnx_graph=model)
    model_ir = lower_onnx_to_ir(
        onnx_graph=preprocessed_model,
        output_file_name="clip_relu_n1_to_1_builtin_no_layout_opt_test",
        optimize_layout_transpose_chains=False,
    )

    op_types = [str(op.op_type) for op in model_ir.operators]
    assert op_types == ["RELU_N1_TO_1"]
    clip_op = model_ir.operators[0]
    assert list(clip_op.inputs) == ["x"]
    assert list(clip_op.outputs) == ["y"]


@pytest.mark.parametrize(
    ("activation", "activation_params", "expected_terminal_op", "expected_conv_fused_activation"),
    [
        ("Relu", None, "CONV_2D", "RELU"),
        ("Tanh", None, "TANH", None),
        ("Sigmoid", None, "LOGISTIC", None),
        ("LeakyRelu", [0.125], "LEAKY_RELU", None),
        ("Clip", [0.0, 6.0], "CONV_2D", "RELU6"),
        ("Clip", [-1.0, 1.0], "RELU_N1_TO_1", None),
        ("HardSigmoid", [0.2, 0.5], "RELU_0_TO_1", None),
    ],
)
def test_flatbuffer_direct_fused_conv_builtin_lowering(
    activation: str,
    activation_params: list[float] | None,
    expected_terminal_op: str,
    expected_conv_fused_activation: str | None,
) -> None:
    model = _make_fused_conv_model(
        activation=activation,
        activation_params=activation_params,
    )
    register_default_preprocess_rules()
    preprocessed_model, _ = run_preprocess_pipeline(onnx_graph=model)
    model_ir = lower_onnx_to_ir(
        onnx_graph=preprocessed_model,
        output_file_name=f"fused_conv_{activation.lower()}_builtin_lowering_test",
        allow_custom_ops=False,
    )

    op_types = [str(op.op_type) for op in model_ir.operators]
    assert op_types.count("CONV_2D") == 1
    assert op_types.count("CUSTOM") == 0

    output_producer = next((op for op in model_ir.operators if "y" in set(op.outputs)), None)
    assert output_producer is not None
    assert str(output_producer.op_type) == expected_terminal_op
    if expected_conv_fused_activation is not None:
        assert str(output_producer.options.get("fusedActivationFunction", "NONE")).upper() == str(
            expected_conv_fused_activation
        )

    if str(activation).lower() == "hardsigmoid":
        # MAXIMUM(0.0)->MINIMUM(1.0) folding may collapse HardSigmoid affine
        # scaffolding into RELU_0_TO_1 without explicit MUL/ADD.
        assert op_types.count("RELU_0_TO_1") >= 1

    if expected_terminal_op == "LEAKY_RELU":
        assert math.isclose(
            float(output_producer.options.get("alpha", -1.0)),
            0.125,
            rel_tol=0.0,
            abs_tol=1e-7,
        )


def test_flatbuffer_direct_fused_conv_clip_unknown_intermediate_shape_builtin_lowering() -> None:
    model = _make_fused_conv_clip_chain_unknown_intermediate_model()
    register_default_preprocess_rules()
    preprocessed_model, _ = run_preprocess_pipeline(onnx_graph=model)
    model_ir = lower_onnx_to_ir(
        onnx_graph=preprocessed_model,
        output_file_name="fused_conv_clip_unknown_intermediate_shape_builtin_lowering_test",
        allow_custom_ops=False,
    )

    op_types = [str(op.op_type) for op in model_ir.operators]
    assert op_types.count("CUSTOM") == 0
    assert op_types.count("CONV_2D") == 2

    conv_ops = [op for op in model_ir.operators if str(op.op_type) == "CONV_2D"]
    assert len(conv_ops) == 2
    first_conv_out = str(conv_ops[0].outputs[0])
    first_conv_tensor = model_ir.tensors.get(first_conv_out)
    assert first_conv_tensor is not None
    assert len(list(first_conv_tensor.shape)) == 4

    output_producer = next((op for op in model_ir.operators if "output" in set(op.outputs)), None)
    assert output_producer is not None
    assert str(output_producer.op_type) == "CONV_2D"
    assert str(output_producer.options.get("fusedActivationFunction", "NONE")).upper() == "RELU6"


def test_flatbuffer_direct_fused_conv_leakyrelu_unknown_intermediate_shape_reduces_redundant_transposes() -> None:
    model = _make_fused_conv_leakyrelu_chain_unknown_intermediate_model()
    register_default_preprocess_rules()
    preprocessed_model, _ = run_preprocess_pipeline(onnx_graph=model)
    model_ir = lower_onnx_to_ir(
        onnx_graph=preprocessed_model,
        output_file_name="fused_conv_leakyrelu_unknown_intermediate_shape_layout_test",
        allow_custom_ops=False,
    )

    op_types = [str(op.op_type) for op in model_ir.operators]
    assert op_types.count("CUSTOM") == 0
    assert op_types.count("CONV_2D") == 2
    assert op_types.count("LEAKY_RELU") == 2
    # Middle NCHW<->NHWC bridge after first LEAKY_RELU should be elided.
    # Only boundary adapter can remain in lowered IR.
    assert op_types.count("TRANSPOSE") <= 1
    first_leaky_idx = next(i for i, op_type in enumerate(op_types) if op_type == "LEAKY_RELU")
    assert op_types[first_leaky_idx + 1] == "CONV_2D"


def test_flatbuffer_direct_fused_conv_tanh_keeps_explicit_tanh_ops() -> None:
    model = _make_fused_conv_tanh_chain_unknown_intermediate_model()
    register_default_preprocess_rules()
    preprocessed_model, _ = run_preprocess_pipeline(onnx_graph=model)
    model_ir = lower_onnx_to_ir(
        onnx_graph=preprocessed_model,
        output_file_name="fused_conv_tanh_unknown_intermediate_shape_layout_test",
        allow_custom_ops=False,
    )

    op_types = [str(op.op_type) for op in model_ir.operators]
    assert op_types.count("CUSTOM") == 0
    assert op_types.count("CONV_2D") == 2
    assert op_types.count("TANH") == 2
    conv_ops = [op for op in model_ir.operators if str(op.op_type) == "CONV_2D"]
    assert all(str(op.options.get("fusedActivationFunction", "NONE")).upper() == "NONE" for op in conv_ops)


def test_flatbuffer_direct_serialize_model_prunes_dead_ops() -> None:
    model_ir = ModelIR(name="serialize_prune_test")
    model_ir.inputs = ["x"]
    model_ir.outputs = ["y"]
    model_ir.tensors["x"] = TensorIR(name="x", dtype="FLOAT32", shape=[1, 3])
    model_ir.tensors["y"] = TensorIR(name="y", dtype="FLOAT32", shape=[1, 3])
    model_ir.tensors["tmp"] = TensorIR(name="tmp", dtype="FLOAT32", shape=[1, 3])
    model_ir.tensors["dead_out"] = TensorIR(name="dead_out", dtype="FLOAT32", shape=[1, 3])
    model_ir.tensors["c"] = TensorIR(
        name="c",
        dtype="FLOAT32",
        shape=[1, 3],
        data=np.array([[1.0, 2.0, 3.0]], dtype=np.float32),
    )
    model_ir.tensors["perm"] = TensorIR(
        name="perm",
        dtype="INT32",
        shape=[2],
        data=np.array([0, 1], dtype=np.int32),
    )
    model_ir.operators = [
        OperatorIR(op_type="ADD", inputs=["x", "c"], outputs=["tmp"]),
        OperatorIR(op_type="RELU", inputs=["tmp"], outputs=["y"]),
        OperatorIR(op_type="TRANSPOSE", inputs=["tmp", "perm"], outputs=["dead_out"]),
    ]

    with tempfile.TemporaryDirectory() as tmpdir:
        schema_tflite = load_schema_module(tmpdir)
        model_bytes = serialize_model(schema_tflite=schema_tflite, model_ir=model_ir)
        model_obj = schema_tflite["Model"].GetRootAsModel(model_bytes, 0)
        subgraph = model_obj.Subgraphs(0)
        assert subgraph.OperatorsLength() == 2
        tensor_names = {
            subgraph.Tensors(i).Name().decode()
            for i in range(subgraph.TensorsLength())
        }
        assert "dead_out" not in tensor_names
        assert "perm" not in tensor_names


def test_flatbuffer_direct_serialize_model_supports_one_hot() -> None:
    model_ir = ModelIR(name="serialize_one_hot_test")
    model_ir.inputs = ["indices"]
    model_ir.outputs = ["y"]
    model_ir.tensors["indices"] = TensorIR(name="indices", dtype="INT32", shape=[2])
    model_ir.tensors["depth"] = TensorIR(
        name="depth",
        dtype="INT32",
        shape=[1],
        data=np.asarray(3, dtype=np.int32),
    )
    model_ir.tensors["on"] = TensorIR(
        name="on",
        dtype="FLOAT32",
        shape=[1],
        data=np.asarray(1.0, dtype=np.float32),
    )
    model_ir.tensors["off"] = TensorIR(
        name="off",
        dtype="FLOAT32",
        shape=[1],
        data=np.asarray(0.0, dtype=np.float32),
    )
    model_ir.tensors["y"] = TensorIR(name="y", dtype="FLOAT32", shape=[2, 3])
    model_ir.operators = [
        OperatorIR(
            op_type="ONE_HOT",
            inputs=["indices", "depth", "on", "off"],
            outputs=["y"],
            options={"axis": -1},
        ),
    ]

    with tempfile.TemporaryDirectory() as tmpdir:
        schema_tflite = load_schema_module(tmpdir)
        model_bytes = serialize_model(schema_tflite=schema_tflite, model_ir=model_ir)
        model_obj = schema_tflite["Model"].GetRootAsModel(model_bytes, 0)
        subgraph = model_obj.Subgraphs(0)
        assert subgraph.OperatorsLength() == 1


def test_flatbuffer_direct_serialize_model_supports_pow() -> None:
    model_ir = ModelIR(name="serialize_pow_test")
    model_ir.inputs = ["x"]
    model_ir.outputs = ["y"]
    model_ir.tensors["x"] = TensorIR(name="x", dtype="FLOAT32", shape=[1, 3])
    model_ir.tensors["exp"] = TensorIR(
        name="exp",
        dtype="FLOAT32",
        shape=[1],
        data=np.asarray(1.25, dtype=np.float32),
    )
    model_ir.tensors["y"] = TensorIR(name="y", dtype="FLOAT32", shape=[1, 3])
    model_ir.operators = [
        OperatorIR(
            op_type="POW",
            inputs=["x", "exp"],
            outputs=["y"],
        ),
    ]

    with tempfile.TemporaryDirectory() as tmpdir:
        schema_tflite = load_schema_module(tmpdir)
        model_bytes = serialize_model(schema_tflite=schema_tflite, model_ir=model_ir)
        model_obj = schema_tflite["Model"].GetRootAsModel(model_bytes, 0)
        subgraph = model_obj.Subgraphs(0)
        assert subgraph.OperatorsLength() == 1


def test_flatbuffer_direct_serialize_model_supports_leaky_relu() -> None:
    model_ir = ModelIR(name="serialize_leaky_relu_test")
    model_ir.inputs = ["x"]
    model_ir.outputs = ["y"]
    model_ir.tensors["x"] = TensorIR(name="x", dtype="FLOAT32", shape=[1, 3])
    model_ir.tensors["y"] = TensorIR(name="y", dtype="FLOAT32", shape=[1, 3])
    model_ir.operators = [
        OperatorIR(
            op_type="LEAKY_RELU",
            inputs=["x"],
            outputs=["y"],
            options={"alpha": 0.2},
        ),
    ]

    with tempfile.TemporaryDirectory() as tmpdir:
        schema_tflite = load_schema_module(tmpdir)
        model_bytes = serialize_model(schema_tflite=schema_tflite, model_ir=model_ir)
        model_obj = schema_tflite["Model"].GetRootAsModel(model_bytes, 0)
        subgraph = model_obj.Subgraphs(0)
        assert subgraph.OperatorsLength() == 1
        op_code = model_obj.OperatorCodes(subgraph.Operators(0).OpcodeIndex())
        assert int(op_code.BuiltinCode()) == int(schema_tflite["BuiltinOperator"].LEAKY_RELU)


def test_flatbuffer_direct_serialize_model_supports_mirror_pad() -> None:
    model_ir = ModelIR(name="serialize_mirror_pad_test")
    model_ir.inputs = ["x"]
    model_ir.outputs = ["y"]
    model_ir.tensors["x"] = TensorIR(name="x", dtype="FLOAT32", shape=[1, 4])
    model_ir.tensors["pads"] = TensorIR(
        name="pads",
        dtype="INT32",
        shape=[2, 2],
        data=np.asarray([[0, 0], [1, 1]], dtype=np.int32),
    )
    model_ir.tensors["y"] = TensorIR(name="y", dtype="FLOAT32", shape=[1, 6])
    model_ir.operators = [
        OperatorIR(
            op_type="MIRROR_PAD",
            inputs=["x", "pads"],
            outputs=["y"],
            options={"mode": "REFLECT"},
        ),
    ]

    with tempfile.TemporaryDirectory() as tmpdir:
        schema_tflite = load_schema_module(tmpdir)
        model_bytes = serialize_model(schema_tflite=schema_tflite, model_ir=model_ir)
        model_obj = schema_tflite["ModelT"].InitFromObj(schema_tflite["Model"].GetRootAs(model_bytes, 0))
        subgraph = model_obj.subgraphs[0]
        assert len(subgraph.operators) == 1
        op = subgraph.operators[0]
        op_code = model_obj.operatorCodes[int(op.opcodeIndex)]
        assert int(op_code.builtinCode) == int(schema_tflite["BuiltinOperator"].MIRROR_PAD)
        assert int(op.builtinOptionsType) == int(schema_tflite["BuiltinOptions"].MirrorPadOptions)
        assert int(op.builtinOptions.mode) == int(schema_tflite["MirrorPadMode"].REFLECT)


def test_flatbuffer_direct_serialize_model_supports_fill() -> None:
    model_ir = ModelIR(name="serialize_fill_test")
    model_ir.outputs = ["y"]
    model_ir.tensors["shape"] = TensorIR(
        name="shape",
        dtype="INT32",
        shape=[2],
        data=np.asarray([2, 3], dtype=np.int32),
    )
    model_ir.tensors["value"] = TensorIR(
        name="value",
        dtype="FLOAT32",
        shape=[],
        data=np.asarray(0.25, dtype=np.float32),
    )
    model_ir.tensors["y"] = TensorIR(name="y", dtype="FLOAT32", shape=[2, 3])
    model_ir.operators = [
        OperatorIR(
            op_type="FILL",
            inputs=["shape", "value"],
            outputs=["y"],
        ),
    ]

    with tempfile.TemporaryDirectory() as tmpdir:
        schema_tflite = load_schema_module(tmpdir)
        model_bytes = serialize_model(schema_tflite=schema_tflite, model_ir=model_ir)
        model_obj = schema_tflite["Model"].GetRootAsModel(model_bytes, 0)
        subgraph = model_obj.Subgraphs(0)
        assert subgraph.OperatorsLength() == 1


def test_flatbuffer_direct_serialize_model_sanitizes_negative_tensor_shape() -> None:
    model_ir = ModelIR(name="serialize_negative_shape_sanitize_test")
    model_ir.inputs = ["x"]
    model_ir.outputs = ["y"]
    model_ir.tensors["x"] = TensorIR(name="x", dtype="FLOAT32", shape=[1], shape_signature=[-1])
    model_ir.tensors["y"] = TensorIR(name="y", dtype="FLOAT32", shape=[-1], shape_signature=[-1])
    model_ir.operators = [
        OperatorIR(
            op_type="RESHAPE",
            inputs=["x"],
            outputs=["y"],
            options={"newShape": [-1]},
        ),
    ]

    with tempfile.TemporaryDirectory() as tmpdir:
        schema_tflite = load_schema_module(tmpdir)
        model_bytes = serialize_model(schema_tflite=schema_tflite, model_ir=model_ir)
        model_obj = schema_tflite["Model"].GetRootAsModel(model_bytes, 0)
        subgraph = model_obj.Subgraphs(0)
        y_idx = subgraph.Outputs(0)
        y_tensor = subgraph.Tensors(y_idx)
        assert [int(y_tensor.Shape(i)) for i in range(y_tensor.ShapeLength())] == [1]
        assert [int(y_tensor.ShapeSignature(i)) for i in range(y_tensor.ShapeSignatureLength())] == [-1]


def test_flatbuffer_direct_terminal_quantize_dequantize_optimization() -> None:
    model = _make_terminal_quantize_dequantize_model()
    register_default_preprocess_rules()
    preprocessed_model, _ = run_preprocess_pipeline(onnx_graph=model)
    model_ir = lower_onnx_to_ir(
        onnx_graph=preprocessed_model,
        output_file_name="terminal_qdq_opt_test",
    )
    op_types = [str(op.op_type) for op in model_ir.operators]
    assert "QUANTIZE" not in op_types
    assert "DEQUANTIZE" not in op_types
    assert op_types.count("RELU") == 1


def test_flatbuffer_direct_terminal_transpose_before_dequantize_sanitization() -> None:
    model = _make_terminal_transpose_dequantize_model()
    register_default_preprocess_rules()
    preprocessed_model, _ = run_preprocess_pipeline(onnx_graph=model)
    model_ir = lower_onnx_to_ir(
        onnx_graph=preprocessed_model,
        output_file_name="terminal_transpose_before_dequantize_sanitize_test",
    )

    dequantize_ops = [op for op in model_ir.operators if str(op.op_type) == "DEQUANTIZE"]
    transpose_ops = [op for op in model_ir.operators if str(op.op_type) == "TRANSPOSE"]
    assert len(dequantize_ops) == 1
    assert len(transpose_ops) == 0

    dq_op = dequantize_ops[0]
    assert dq_op.outputs[0] == "y"


def test_flatbuffer_direct_transpose_quantize_transpose_optimization() -> None:
    model = _make_transpose_quantize_transpose_model()
    register_default_preprocess_rules()
    preprocessed_model, _ = run_preprocess_pipeline(onnx_graph=model)
    model_ir = lower_onnx_to_ir(
        onnx_graph=preprocessed_model,
        output_file_name="transpose_quantize_transpose_opt_test",
    )
    op_types = [str(op.op_type) for op in model_ir.operators]
    assert op_types.count("TRANSPOSE") == 0
    assert op_types.count("QUANTIZE") == 1


def test_flatbuffer_direct_transpose_quantize_transpose_fanout_optimization() -> None:
    model = _make_transpose_quantize_transpose_fanout_model()
    register_default_preprocess_rules()
    preprocessed_model, _ = run_preprocess_pipeline(onnx_graph=model)
    model_ir = lower_onnx_to_ir(
        onnx_graph=preprocessed_model,
        output_file_name="transpose_quantize_transpose_fanout_opt_test",
    )
    op_types = [str(op.op_type) for op in model_ir.operators]
    assert op_types.count("TRANSPOSE") == 0
    assert op_types.count("QUANTIZE") == 1
    assert op_types.count("DEQUANTIZE") == 2
    assert op_types.count("ADD") == 1


def test_flatbuffer_direct_transpose_quantize_transpose_preserves_dynamic_batch_signature() -> None:
    model = _make_transpose_quantize_transpose_dynamic_batch_model()
    register_default_preprocess_rules()
    preprocessed_model, _ = run_preprocess_pipeline(onnx_graph=model)
    model_ir = lower_onnx_to_ir(
        onnx_graph=preprocessed_model,
        output_file_name="transpose_quantize_transpose_dynamic_batch_opt_test",
    )
    quant_ops = [op for op in model_ir.operators if str(op.op_type) == "QUANTIZE"]
    assert len(quant_ops) == 1
    out_name = quant_ops[0].outputs[0]
    out_tensor = model_ir.tensors[out_name]
    assert out_tensor is not None
    assert out_tensor.shape_signature is not None
    assert int(out_tensor.shape_signature[0]) == -1


def test_flatbuffer_direct_transpose_binary_fanout_chain_optimization() -> None:
    model = _make_transpose_binary_fanout_chain_model()
    register_default_preprocess_rules()
    preprocessed_model, _ = run_preprocess_pipeline(onnx_graph=model)
    model_ir = lower_onnx_to_ir(
        onnx_graph=preprocessed_model,
        output_file_name="transpose_binary_fanout_chain_opt_test",
    )
    op_types = [str(op.op_type) for op in model_ir.operators]
    assert op_types.count("TRANSPOSE") == 0
    assert op_types.count("ADD") == 2
    assert op_types.count("RELU") == 1
    add_ops = [op for op in model_ir.operators if str(op.op_type) == "ADD"]
    assert any(str(op.options.get("fusedActivationFunction", "NONE")).upper() == "RELU" for op in add_ops)


def test_flatbuffer_direct_gather_singleton_const_indices_scalarized_for_rank_reduced_output() -> None:
    model = _make_gather_singleton_indices_rank_reduced_model()
    register_default_preprocess_rules()
    preprocessed_model, _ = run_preprocess_pipeline(onnx_graph=model)
    model_ir = lower_onnx_to_ir(
        onnx_graph=preprocessed_model,
        output_file_name="gather_singleton_indices_scalarized_test",
    )
    gather_ops = [op for op in model_ir.operators if str(op.op_type) == "GATHER"]
    assert len(gather_ops) == 1
    gather_op = gather_ops[0]
    gather_indices_name = str(gather_op.inputs[1])
    gather_indices_tensor = model_ir.tensors[gather_indices_name]
    assert gather_indices_name != "indices"
    assert list(gather_indices_tensor.shape) == []
    assert list(gather_indices_tensor.shape_signature) == []

    gather_output_tensor = model_ir.tensors[str(gather_op.outputs[0])]
    assert list(gather_output_tensor.shape_signature) == [1, 3]


def test_flatbuffer_direct_gather_singleton_negative_const_indices_wrapped_before_scalarization() -> None:
    model = _make_gather_singleton_negative_indices_rank_reduced_model()
    register_default_preprocess_rules()
    preprocessed_model, _ = run_preprocess_pipeline(onnx_graph=model)
    model_ir = lower_onnx_to_ir(
        onnx_graph=preprocessed_model,
        output_file_name="gather_singleton_negative_indices_scalarized_test",
    )
    gather_ops = [op for op in model_ir.operators if str(op.op_type) == "GATHER"]
    assert len(gather_ops) == 1
    gather_op = gather_ops[0]
    gather_indices_name = str(gather_op.inputs[1])
    gather_indices_tensor = model_ir.tensors[gather_indices_name]
    assert gather_indices_name != "indices"
    assert list(gather_indices_tensor.shape) == []
    assert list(gather_indices_tensor.shape_signature) == []
    gather_indices = np.asarray(gather_indices_tensor.data, dtype=np.int64).reshape(-1)
    assert gather_indices.tolist() == [3]


def test_flatbuffer_direct_nms_scalar_const_inputs_do_not_emit_squeeze() -> None:
    model = _make_non_max_suppression_model()
    register_default_preprocess_rules()
    preprocessed_model, _ = run_preprocess_pipeline(onnx_graph=model)
    model_ir = lower_onnx_to_ir(
        onnx_graph=preprocessed_model,
        output_file_name="nms_scalar_const_inputs_test",
    )
    op_types = [str(op.op_type) for op in model_ir.operators]
    assert op_types.count("NON_MAX_SUPPRESSION_V4") + op_types.count("NON_MAX_SUPPRESSION_V5") == 1
    # boxes/scores squeeze remain, but scalar const inputs are now fed directly.
    assert op_types.count("SQUEEZE") == 2

    nms_op = next(
        op
        for op in model_ir.operators
        if str(op.op_type) in {"NON_MAX_SUPPRESSION_V4", "NON_MAX_SUPPRESSION_V5"}
    )
    for scalar_input_name in list(nms_op.inputs)[2:5]:
        scalar_tensor = model_ir.tensors.get(str(scalar_input_name), None)
        assert scalar_tensor is not None
        assert scalar_tensor.data is not None
        assert list(scalar_tensor.shape) == []


def test_flatbuffer_direct_nms_switch_version_selects_builtin_variant() -> None:
    model = _make_non_max_suppression_model()
    register_default_preprocess_rules()
    preprocessed_model, _ = run_preprocess_pipeline(onnx_graph=model)

    model_ir_v4 = lower_onnx_to_ir(
        onnx_graph=preprocessed_model,
        output_file_name="nms_switch_v4_test",
        switch_nms_version="v4",
    )
    nms_ops_v4 = [op for op in model_ir_v4.operators if str(op.op_type) == "NON_MAX_SUPPRESSION_V4"]
    assert len(nms_ops_v4) == 1
    assert len(list(nms_ops_v4[0].inputs)) == 5
    assert len(list(nms_ops_v4[0].outputs)) == 2
    assert all(str(op.op_type) != "NON_MAX_SUPPRESSION_V5" for op in model_ir_v4.operators)

    model_ir_v5 = lower_onnx_to_ir(
        onnx_graph=preprocessed_model,
        output_file_name="nms_switch_v5_test",
        switch_nms_version="v5",
    )
    nms_ops_v5 = [op for op in model_ir_v5.operators if str(op.op_type) == "NON_MAX_SUPPRESSION_V5"]
    assert len(nms_ops_v5) == 1
    assert len(list(nms_ops_v5[0].inputs)) == 6
    assert len(list(nms_ops_v5[0].outputs)) == 3
    assert all(str(op.op_type) != "NON_MAX_SUPPRESSION_V4" for op in model_ir_v5.operators)


def test_flatbuffer_direct_multiclass_nms_without_onwa_matches_default_behavior() -> None:
    model = _make_non_max_suppression_multiclass_model()
    register_default_preprocess_rules()
    preprocessed_model, _ = run_preprocess_pipeline(onnx_graph=model)
    model_ir = lower_onnx_to_ir(
        onnx_graph=preprocessed_model,
        output_file_name="nms_multiclass_without_onwa_test",
        output_nms_with_argmax=False,
    )
    op_types = [str(op.op_type) for op in model_ir.operators]
    assert op_types.count("NON_MAX_SUPPRESSION_V4") + op_types.count("NON_MAX_SUPPRESSION_V5") == 3
    assert op_types.count("ARG_MAX") == 0
    assert op_types.count("REDUCE_MAX") == 0
    assert op_types.count("CUSTOM") == 0

    concat_axis0_ops = [
        op
        for op in model_ir.operators
        if str(op.op_type) == "CONCATENATION" and int(op.options.get("axis", -1)) == 0
    ]
    assert len(concat_axis0_ops) >= 1
    assert any(len(list(op.inputs)) == 3 for op in concat_axis0_ops)

    output_tensor = model_ir.tensors.get("selected_indices")
    assert output_tensor is not None
    assert list(output_tensor.shape_signature) == [-1, 3]


@pytest.mark.skipif(not _requires_flatbuffer_tools(), reason="flatbuffer_direct requires flatc and curl")
def test_flatbuffer_direct_multiclass_nms_auto_argmax_retry_avoids_custom_lowering() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        model = _make_non_max_suppression_multiclass_model()
        model_path = _save_model(tmpdir, "nms_multiclass_auto_argmax", model)
        out_dir = os.path.join(tmpdir, "flatbuffer_direct")
        tflite_path = _convert(
            model_path=model_path,
            output_dir=out_dir,
            backend="flatbuffer_direct",
            report_op_coverage=True,
        )
        assert os.path.isfile(tflite_path)

        report_path = os.path.join(out_dir, "nms_multiclass_auto_argmax_op_coverage_report.json")
        assert os.path.isfile(report_path)
        with open(report_path, "r", encoding="utf-8") as f:
            report = json.load(f)

        assert report["graph_custom_ops"] == []
        nms_reports = [
            node
            for node in report["graph_node_reports"]
            if str(node.get("onnx_op", "")) == "NonMaxSuppression"
        ]
        assert len(nms_reports) == 1
        assert nms_reports[0]["dispatch_mode"] == "builtin"


def test_flatbuffer_direct_conv2d_intermediate_preserves_dynamic_batch_signature() -> None:
    model = _make_conv_dynamic_batch_model()
    register_default_preprocess_rules()
    preprocessed_model, _ = run_preprocess_pipeline(onnx_graph=model)
    model_ir = lower_onnx_to_ir(
        onnx_graph=preprocessed_model,
        output_file_name="conv_dynamic_batch_signature_test",
    )
    conv_ops = [op for op in model_ir.operators if str(op.op_type) == "CONV_2D"]
    assert len(conv_ops) == 1
    out_name = conv_ops[0].outputs[0]
    out_tensor = model_ir.tensors[out_name]
    assert out_tensor.shape_signature is not None
    assert int(out_tensor.shape_signature[0]) == -1


def test_flatbuffer_direct_conv_stride2_notset_symmetric_pad_uses_explicit_pad() -> None:
    model = _make_conv_stride2_symmetric_pad_model()
    register_default_preprocess_rules()
    preprocessed_model, _ = run_preprocess_pipeline(onnx_graph=model)
    model_ir = lower_onnx_to_ir(
        onnx_graph=preprocessed_model,
        output_file_name="conv_stride2_notset_symmetric_pad_test",
    )
    op_types = [str(op.op_type) for op in model_ir.operators]
    assert op_types.count("PAD") >= 1
    assert op_types.count("CONV_2D") == 1
    conv_op = next(op for op in model_ir.operators if str(op.op_type) == "CONV_2D")
    assert str(conv_op.options.get("padding", "")) == "VALID"


def test_flatbuffer_direct_qlinear_conv_valid_nonzero_pad_uses_explicit_pad() -> None:
    model = _make_qlinear_conv_valid_explicit_pad_model()
    register_default_preprocess_rules()
    preprocessed_model, _ = run_preprocess_pipeline(onnx_graph=model)
    model_ir = lower_onnx_to_ir(
        onnx_graph=preprocessed_model,
        output_file_name="qlinear_conv_valid_nonzero_pad_explicit_pad_test",
    )
    op_types = [str(op.op_type) for op in model_ir.operators]
    assert op_types.count("PAD") >= 1
    conv_ops = [op for op in model_ir.operators if str(op.op_type) == "CONV_2D"]
    assert len(conv_ops) == 1
    assert str(conv_ops[0].options.get("padding", "")) == "VALID"


def test_flatbuffer_direct_maxpool2d_intermediate_preserves_dynamic_batch_signature() -> None:
    model = _make_maxpool_dynamic_batch_model()
    register_default_preprocess_rules()
    preprocessed_model, _ = run_preprocess_pipeline(onnx_graph=model)
    model_ir = lower_onnx_to_ir(
        onnx_graph=preprocessed_model,
        output_file_name="maxpool_dynamic_batch_signature_test",
    )
    maxpool_ops = [op for op in model_ir.operators if str(op.op_type) == "MAX_POOL_2D"]
    assert len(maxpool_ops) == 1
    out_name = maxpool_ops[0].outputs[0]
    out_tensor = model_ir.tensors[out_name]
    assert out_tensor.shape_signature is not None
    assert int(out_tensor.shape_signature[0]) == -1


def test_flatbuffer_direct_average_pool_exclude_pad_uses_divisor_correction() -> None:
    model = _make_average_pool_exclude_pad_model()
    register_default_preprocess_rules()
    preprocessed_model, _ = run_preprocess_pipeline(onnx_graph=model)
    model_ir = lower_onnx_to_ir(
        onnx_graph=preprocessed_model,
        output_file_name="average_pool_exclude_pad_correction_test",
    )
    op_types = [str(op.op_type) for op in model_ir.operators]
    assert op_types.count("AVERAGE_POOL_2D") == 2
    assert op_types.count("DIV") == 1


@pytest.mark.skipif(not _requires_flatbuffer_tools(), reason="flatbuffer_direct requires flatc and curl")
def test_flatbuffer_direct_average_pool_exclude_pad_runtime_match() -> None:
    def _reference_avgpool_exclude_pad(x_nchw: np.ndarray) -> np.ndarray:
        out = np.zeros((1, 1, 2, 2), dtype=np.float32)
        for oh in range(2):
            for ow in range(2):
                h_start = int(oh * 2 - 1)
                w_start = int(ow * 2 - 1)
                vals: list[float] = []
                for ih in range(h_start, h_start + 3):
                    for iw in range(w_start, w_start + 3):
                        if 0 <= ih < 4 and 0 <= iw < 4:
                            vals.append(float(x_nchw[0, 0, ih, iw]))
                out[0, 0, oh, ow] = float(sum(vals) / max(len(vals), 1))
        return out

    with tempfile.TemporaryDirectory() as tmpdir:
        model = _make_average_pool_exclude_pad_model()
        model_path = _save_model(tmpdir, "average_pool_exclude_pad", model)
        out_dir = os.path.join(tmpdir, "out")
        tflite_path = _convert(model_path, out_dir, "flatbuffer_direct")
        interpreter = Interpreter(model_path=tflite_path)
        interpreter.allocate_tensors()
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        x = np.asarray(
            [[[[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0], [9.0, 10.0, 11.0, 12.0], [13.0, 14.0, 15.0, 16.0]]]],
            dtype=np.float32,
        )
        input_shape = tuple(int(v) for v in input_details[0]["shape"])
        if input_shape == tuple(x.shape):
            x_feed = x
        elif input_shape == (1, 4, 4, 1):
            x_feed = np.transpose(x, (0, 2, 3, 1))
        else:
            raise AssertionError(f"Unexpected input shape: {input_shape}")
        interpreter.set_tensor(input_details[0]["index"], x_feed)
        interpreter.invoke()
        y = interpreter.get_tensor(output_details[0]["index"])
        if tuple(y.shape) == (1, 2, 2, 1):
            y = np.transpose(y, (0, 3, 1, 2))
        ref = _reference_avgpool_exclude_pad(x)
        np.testing.assert_allclose(y, ref, rtol=0.0, atol=1e-6)


def test_flatbuffer_direct_qlinear_conv2d_intermediate_preserves_dynamic_batch_signature() -> None:
    model = _make_qlinear_conv_dynamic_batch_model()
    register_default_preprocess_rules()
    preprocessed_model, _ = run_preprocess_pipeline(onnx_graph=model)
    model_ir = lower_onnx_to_ir(
        onnx_graph=preprocessed_model,
        output_file_name="qlinear_conv_dynamic_batch_signature_test",
    )
    conv_ops = [op for op in model_ir.operators if str(op.op_type) == "CONV_2D"]
    assert len(conv_ops) == 1
    out_name = conv_ops[0].outputs[0]
    out_tensor = model_ir.tensors[out_name]
    assert out_tensor.quantization is not None
    assert out_tensor.shape_signature is not None
    assert int(out_tensor.shape_signature[0]) == -1


def test_flatbuffer_direct_qlinear_conv_unknown_rank_placeholder_is_promoted_to_rank4() -> None:
    model = _make_qlinear_conv_unknown_rank_model()
    register_default_preprocess_rules()
    preprocessed_model, _ = run_preprocess_pipeline(onnx_graph=model)
    model_ir = lower_onnx_to_ir(
        onnx_graph=preprocessed_model,
        output_file_name="qlinear_conv_unknown_rank_signature_test",
    )

    conv_ops = [op for op in model_ir.operators if str(op.op_type) == "CONV_2D"]
    assert len(conv_ops) == 1

    conv_out = model_ir.tensors[conv_ops[0].outputs[0]]
    assert len(list(conv_out.shape)) == 4
    assert int(conv_out.shape[3]) == 4
    assert conv_out.shape_signature is not None
    assert len(list(conv_out.shape_signature)) == 4


def test_flatbuffer_direct_qlinear_global_average_pool_intermediate_preserves_dynamic_batch_signature() -> None:
    model = _make_qlinear_global_average_pool_dynamic_batch_model()
    register_default_preprocess_rules()
    preprocessed_model, _ = run_preprocess_pipeline(onnx_graph=model)
    model_ir = lower_onnx_to_ir(
        onnx_graph=preprocessed_model,
        output_file_name="qlinear_global_average_pool_dynamic_batch_signature_test",
    )

    avg_pool_ops = [op for op in model_ir.operators if str(op.op_type) == "AVERAGE_POOL_2D"]
    assert len(avg_pool_ops) == 1
    avg_tensor = model_ir.tensors[avg_pool_ops[0].outputs[0]]
    assert avg_tensor.shape_signature is not None
    assert int(avg_tensor.shape_signature[0]) == -1

    y_tensor = model_ir.tensors["y"]
    assert y_tensor.shape_signature is not None
    assert int(y_tensor.shape_signature[0]) == -1


def test_flatbuffer_direct_transpose_qlinear_global_average_pool_bridge_optimization() -> None:
    model = _make_transpose_qlinear_global_average_pool_model()
    register_default_preprocess_rules()
    preprocessed_model, _ = run_preprocess_pipeline(onnx_graph=model)
    model_ir = lower_onnx_to_ir(
        onnx_graph=preprocessed_model,
        output_file_name="transpose_qlinear_global_average_pool_bridge_opt_test",
    )

    op_types = [str(op.op_type) for op in model_ir.operators]
    assert op_types.count("AVERAGE_POOL_2D") == 1
    assert op_types.count("MEAN") == 0
    assert op_types.count("DEQUANTIZE") == 0
    assert op_types.count("QUANTIZE") == 1

    assert all(
        not (str(op.op_type) == "TRANSPOSE" and len(op.outputs) == 1 and op.outputs[0] == "x_q_nchw")
        for op in model_ir.operators
    )

    avg_pool_ops = [op for op in model_ir.operators if str(op.op_type) == "AVERAGE_POOL_2D"]
    assert len(avg_pool_ops) == 1
    assert avg_pool_ops[0].inputs == ["x_q"]

    producer = None
    for op in model_ir.operators:
        if "y_q" in set(op.outputs):
            producer = op
            break
    assert producer is not None
    assert str(producer.op_type) == "TRANSPOSE"


def test_flatbuffer_direct_qlinear_concat_conv_layout_propagation() -> None:
    model = _make_qlinear_concat_conv_layout_chain_model()
    register_default_preprocess_rules()
    preprocessed_model, _ = run_preprocess_pipeline(onnx_graph=model)
    model_ir = lower_onnx_to_ir(
        onnx_graph=preprocessed_model,
        output_file_name="qlinear_concat_conv_layout_propagation_test",
    )

    concat_ops = [op for op in model_ir.operators if str(op.op_type) == "CONCATENATION"]
    assert len(concat_ops) == 1
    concat_axis = int(concat_ops[0].options.get("axis", -1))
    assert concat_axis == 3

    cat_q = model_ir.tensors.get("cat_q")
    assert cat_q is not None
    assert list(cat_q.shape) == [1, 1, 1, 4]

    assert all(
        not (
            str(op.op_type) == "TRANSPOSE"
            and len(op.outputs) == 1
            and op.outputs[0] == "QLCatConv_Conv_input_nhwc"
        )
        for op in model_ir.operators
    )

    # If a QUANTIZE input is already in dequantized flow, avoid redundant Q->DQ before CONCAT.
    assert all(
        not (
            str(op.op_type) == "DEQUANTIZE"
            and len(op.inputs) == 1
            and op.inputs[0] == "pool_q"
        )
        for op in model_ir.operators
    )
    concat_inputs = list(concat_ops[0].inputs)
    assert "QLCatConv_MaxPool_output_nhwc" in concat_inputs


def test_flatbuffer_direct_fuse_add_relu_activation_chain() -> None:
    model = _make_add_relu_model()
    register_default_preprocess_rules()
    preprocessed_model, _ = run_preprocess_pipeline(onnx_graph=model)
    model_ir = lower_onnx_to_ir(
        onnx_graph=preprocessed_model,
        output_file_name="fuse_add_relu_activation_chain_test",
    )
    add_ops = [op for op in model_ir.operators if str(op.op_type) == "ADD"]
    relu_ops = [op for op in model_ir.operators if str(op.op_type) == "RELU"]

    assert len(add_ops) == 1
    assert len(relu_ops) == 0
    assert str(add_ops[0].options.get("fusedActivationFunction", "NONE")).upper() == "RELU"
    assert list(add_ops[0].outputs) == ["z"]


def test_flatbuffer_direct_fuse_add_relu_after_transpose_pre_add_rewrite() -> None:
    model = _make_transpose_wrapped_add_relu_model()
    register_default_preprocess_rules()
    preprocessed_model, _ = run_preprocess_pipeline(onnx_graph=model)
    model_ir = lower_onnx_to_ir(
        onnx_graph=preprocessed_model,
        output_file_name="fuse_add_relu_after_transpose_pre_add_rewrite_test",
    )

    add_ops = [op for op in model_ir.operators if str(op.op_type) == "ADD"]
    relu_ops = [op for op in model_ir.operators if str(op.op_type) == "RELU"]
    transpose_ops = [op for op in model_ir.operators if str(op.op_type) == "TRANSPOSE"]

    assert len(add_ops) == 1
    assert len(relu_ops) == 1
    assert len(transpose_ops) == 0
    assert str(add_ops[0].options.get("fusedActivationFunction", "NONE")).upper() == "RELU"
    assert "__skip_add_activation_fuse__" not in add_ops[0].options
    assert list(add_ops[0].outputs) == ["sum_nhwc"]


def test_flatbuffer_direct_fuse_conv_relu_n1_to_1_activation_chain() -> None:
    model_ir = ModelIR(name="fuse_conv_relu_n1_to_1_activation_chain_test")
    model_ir.inputs = ["x"]
    model_ir.outputs = ["z"]
    model_ir.tensors["x"] = TensorIR(name="x", dtype="FLOAT32", shape=[1, 8, 8, 3], shape_signature=[1, 8, 8, 3])
    model_ir.tensors["w"] = TensorIR(
        name="w",
        dtype="FLOAT32",
        shape=[8, 3, 3, 3],
        shape_signature=[8, 3, 3, 3],
        data=np.ones((8, 3, 3, 3), dtype=np.float32),
        is_variable=False,
    )
    model_ir.tensors["b"] = TensorIR(
        name="b",
        dtype="FLOAT32",
        shape=[8],
        shape_signature=[8],
        data=np.zeros((8,), dtype=np.float32),
        is_variable=False,
    )
    model_ir.tensors["y"] = TensorIR(name="y", dtype="FLOAT32", shape=[1, 8, 8, 8], shape_signature=[1, 8, 8, 8])
    model_ir.tensors["z"] = TensorIR(name="z", dtype="FLOAT32", shape=[1, 8, 8, 8], shape_signature=[1, 8, 8, 8])
    model_ir.operators = [
        OperatorIR(
            op_type="CONV_2D",
            inputs=["x", "w", "b"],
            outputs=["y"],
            options={
                "padding": "SAME",
                "strideH": 1,
                "strideW": 1,
                "dilationHFactor": 1,
                "dilationWFactor": 1,
                "fusedActivationFunction": "NONE",
            },
        ),
        OperatorIR(op_type="RELU_N1_TO_1", inputs=["y"], outputs=["z"]),
    ]

    stats = _optimize_fuse_conv_activation_chains(model_ir)
    assert stats["fused_conv_activation_chains"] == 1
    assert stats["fused_activation_chains_total"] == 1
    assert [str(op.op_type) for op in model_ir.operators] == ["CONV_2D"]
    conv_op = model_ir.operators[0]
    assert str(conv_op.options.get("fusedActivationFunction", "NONE")).upper() == "RELU_N1_TO_1"
    assert list(conv_op.outputs) == ["z"]


def test_flatbuffer_direct_fuse_depthwise_conv_relu_n1_to_1_activation_chain() -> None:
    model_ir = ModelIR(name="fuse_depthwise_conv_relu_n1_to_1_activation_chain_test")
    model_ir.inputs = ["x"]
    model_ir.outputs = ["z"]
    model_ir.tensors["x"] = TensorIR(name="x", dtype="FLOAT32", shape=[1, 8, 8, 4], shape_signature=[1, 8, 8, 4])
    model_ir.tensors["w"] = TensorIR(
        name="w",
        dtype="FLOAT32",
        shape=[1, 3, 3, 4],
        shape_signature=[1, 3, 3, 4],
        data=np.ones((1, 3, 3, 4), dtype=np.float32),
        is_variable=False,
    )
    model_ir.tensors["b"] = TensorIR(
        name="b",
        dtype="FLOAT32",
        shape=[4],
        shape_signature=[4],
        data=np.zeros((4,), dtype=np.float32),
        is_variable=False,
    )
    model_ir.tensors["y"] = TensorIR(name="y", dtype="FLOAT32", shape=[1, 8, 8, 4], shape_signature=[1, 8, 8, 4])
    model_ir.tensors["z"] = TensorIR(name="z", dtype="FLOAT32", shape=[1, 8, 8, 4], shape_signature=[1, 8, 8, 4])
    model_ir.operators = [
        OperatorIR(
            op_type="DEPTHWISE_CONV_2D",
            inputs=["x", "w", "b"],
            outputs=["y"],
            options={
                "padding": "SAME",
                "strideH": 1,
                "strideW": 1,
                "dilationHFactor": 1,
                "dilationWFactor": 1,
                "depthMultiplier": 1,
                "fusedActivationFunction": "NONE",
            },
        ),
        OperatorIR(op_type="RELU_N1_TO_1", inputs=["y"], outputs=["z"]),
    ]

    stats = _optimize_fuse_conv_activation_chains(model_ir)
    assert stats["fused_conv_activation_chains"] == 1
    assert stats["fused_activation_chains_total"] == 1
    assert [str(op.op_type) for op in model_ir.operators] == ["DEPTHWISE_CONV_2D"]
    depthwise_conv_op = model_ir.operators[0]
    assert str(depthwise_conv_op.options.get("fusedActivationFunction", "NONE")).upper() == "RELU_N1_TO_1"
    assert list(depthwise_conv_op.outputs) == ["z"]


@pytest.mark.parametrize(
    ("binary_onnx_op", "binary_tflite_op"),
    [
        ("Add", "ADD"),
        ("Sub", "SUB"),
        ("Mul", "MUL"),
        ("Div", "DIV"),
    ],
)
@pytest.mark.parametrize(
    ("activation_onnx_op", "activation_tflite_op", "fused_activation"),
    [
        ("Relu", "RELU", "RELU"),
        ("Relu6", "RELU6", "RELU6"),
    ],
)
def test_flatbuffer_direct_fuse_binary_activation_chain(
    binary_onnx_op: str,
    binary_tflite_op: str,
    activation_onnx_op: str,
    activation_tflite_op: str,
    fused_activation: str,
) -> None:
    model = _make_binary_activation_model(binary_onnx_op, activation_onnx_op)
    register_default_preprocess_rules()
    preprocessed_model, _ = run_preprocess_pipeline(onnx_graph=model)
    model_ir = lower_onnx_to_ir(
        onnx_graph=preprocessed_model,
        output_file_name=f"fuse_{binary_onnx_op.lower()}_{activation_onnx_op.lower()}_chain_test",
    )

    binary_ops = [op for op in model_ir.operators if str(op.op_type) == binary_tflite_op]
    activation_ops = [op for op in model_ir.operators if str(op.op_type) == activation_tflite_op]

    assert len(binary_ops) == 1
    assert len(activation_ops) == 0
    assert str(binary_ops[0].options.get("fusedActivationFunction", "NONE")).upper() == fused_activation
    assert list(binary_ops[0].outputs) == ["z"]


@pytest.mark.parametrize(
    ("binary_onnx_op", "binary_tflite_op", "activation_onnx_op", "activation_tflite_op"),
    [
        ("Add", "ADD", "Sigmoid", "LOGISTIC"),
        ("Sub", "SUB", "Sigmoid", "LOGISTIC"),
        ("Mul", "MUL", "Sigmoid", "LOGISTIC"),
        ("Div", "DIV", "Sigmoid", "LOGISTIC"),
        ("Add", "ADD", "Tanh", "TANH"),
        ("Sub", "SUB", "Tanh", "TANH"),
        ("Mul", "MUL", "Tanh", "TANH"),
        ("Div", "DIV", "Tanh", "TANH"),
    ],
)
def test_flatbuffer_direct_no_fuse_binary_activation_chain(
    binary_onnx_op: str,
    binary_tflite_op: str,
    activation_onnx_op: str,
    activation_tflite_op: str,
) -> None:
    model = _make_binary_activation_model(binary_onnx_op, activation_onnx_op)
    register_default_preprocess_rules()
    preprocessed_model, _ = run_preprocess_pipeline(onnx_graph=model)
    model_ir = lower_onnx_to_ir(
        onnx_graph=preprocessed_model,
        output_file_name=f"no_fuse_{binary_onnx_op.lower()}_{activation_onnx_op.lower()}_chain_test",
    )

    binary_ops = [op for op in model_ir.operators if str(op.op_type) == binary_tflite_op]
    activation_ops = [op for op in model_ir.operators if str(op.op_type) == activation_tflite_op]

    assert len(binary_ops) == 1
    assert len(activation_ops) == 1
    assert str(binary_ops[0].options.get("fusedActivationFunction", "NONE")).upper() == "NONE"
    assert list(binary_ops[0].outputs) == ["a0"]
    assert list(activation_ops[0].outputs) == ["z"]


def test_flatbuffer_direct_lstm_forward_builtin_lowering() -> None:
    model = _make_forward_lstm_model()
    register_default_preprocess_rules()
    preprocessed_model, _ = run_preprocess_pipeline(onnx_graph=model)
    model_ir = lower_onnx_to_ir(
        onnx_graph=preprocessed_model,
        output_file_name="lstm_forward_builtin_lowering_test",
        allow_custom_ops=False,
    )

    op_types = [str(op.op_type) for op in model_ir.operators]
    assert op_types.count("UNIDIRECTIONAL_SEQUENCE_LSTM") == 1
    assert op_types.count("BIDIRECTIONAL_SEQUENCE_LSTM") == 0
    assert "SPLIT" not in op_types
    assert op_types.count("CUSTOM") == 0
    assert "CONCATENATION" not in op_types

    y_tensor = model_ir.tensors.get("y")
    assert y_tensor is not None
    assert list(y_tensor.shape) == [3, 1, 1, 2]
    assert list(y_tensor.shape_signature) == [3, 1, 1, 2]


def test_flatbuffer_direct_lstm_reverse_builtin_lowering() -> None:
    model = _make_reverse_lstm_model()
    register_default_preprocess_rules()
    preprocessed_model, _ = run_preprocess_pipeline(onnx_graph=model)
    model_ir = lower_onnx_to_ir(
        onnx_graph=preprocessed_model,
        output_file_name="lstm_reverse_builtin_lowering_test",
        allow_custom_ops=False,
    )

    op_types = [str(op.op_type) for op in model_ir.operators]
    assert op_types.count("UNIDIRECTIONAL_SEQUENCE_LSTM") == 1
    assert op_types.count("BIDIRECTIONAL_SEQUENCE_LSTM") == 0
    assert op_types.count("REVERSE_V2") == 2
    assert op_types.count("CUSTOM") == 0

    y_tensor = model_ir.tensors.get("y")
    assert y_tensor is not None
    assert list(y_tensor.shape) == [3, 1, 1, 2]
    assert list(y_tensor.shape_signature) == [3, 1, 1, 2]


def test_flatbuffer_direct_lstm_forward_state_io_builtin_lowering() -> None:
    model = _make_forward_lstm_with_state_io_model()
    register_default_preprocess_rules()
    preprocessed_model, _ = run_preprocess_pipeline(onnx_graph=model)
    model_ir = lower_onnx_to_ir(
        onnx_graph=preprocessed_model,
        output_file_name="lstm_forward_state_io_builtin_lowering_test",
        allow_custom_ops=False,
    )

    op_types = [str(op.op_type) for op in model_ir.operators]
    assert op_types.count("UNIDIRECTIONAL_SEQUENCE_LSTM") == 1
    assert op_types.count("CUSTOM") == 0
    assert op_types.count("RESHAPE") >= 3

    y_tensor = model_ir.tensors.get("y")
    h_out_tensor = model_ir.tensors.get("h_out")
    c_out_tensor = model_ir.tensors.get("c_out")
    assert y_tensor is not None
    assert h_out_tensor is not None
    assert c_out_tensor is not None
    assert list(y_tensor.shape) == [4, 1, 1, 2]
    assert list(h_out_tensor.shape) == [1, 1, 2]
    assert list(c_out_tensor.shape) == [1, 1, 2]


def test_flatbuffer_direct_rnn_reverse_builtin_lowering() -> None:
    model = _make_reverse_rnn_model()
    register_default_preprocess_rules()
    preprocessed_model, _ = run_preprocess_pipeline(onnx_graph=model)
    model_ir = lower_onnx_to_ir(
        onnx_graph=preprocessed_model,
        output_file_name="rnn_reverse_builtin_lowering_test",
        allow_custom_ops=False,
    )

    op_types = [str(op.op_type) for op in model_ir.operators]
    assert op_types.count("UNIDIRECTIONAL_SEQUENCE_RNN") == 1
    assert op_types.count("REVERSE_V2") == 2
    assert op_types.count("CUSTOM") == 0
    rnn_ops = [op for op in model_ir.operators if str(op.op_type) == "UNIDIRECTIONAL_SEQUENCE_RNN"]
    assert str(rnn_ops[0].options.get("fusedActivationFunction", "NONE")).upper() == "RELU"

    y_tensor = model_ir.tensors.get("y")
    y_h_tensor = model_ir.tensors.get("y_h")
    assert y_tensor is not None
    assert y_h_tensor is not None
    assert list(y_tensor.shape) == [4, 1, 1, 3]
    assert list(y_h_tensor.shape) == [1, 1, 3]


def test_flatbuffer_direct_rnn_bidirectional_builtin_lowering() -> None:
    model = _make_bidirectional_rnn_model()
    register_default_preprocess_rules()
    preprocessed_model, _ = run_preprocess_pipeline(onnx_graph=model)
    model_ir = lower_onnx_to_ir(
        onnx_graph=preprocessed_model,
        output_file_name="rnn_bidirectional_builtin_lowering_test",
        allow_custom_ops=False,
    )

    op_types = [str(op.op_type) for op in model_ir.operators]
    assert op_types.count("UNIDIRECTIONAL_SEQUENCE_RNN") == 2
    assert op_types.count("REVERSE_V2") == 2
    assert op_types.count("CONCATENATION") >= 2
    assert op_types.count("CUSTOM") == 0
    rnn_ops = [op for op in model_ir.operators if str(op.op_type) == "UNIDIRECTIONAL_SEQUENCE_RNN"]
    fused = [str(op.options.get("fusedActivationFunction", "NONE")).upper() for op in rnn_ops]
    assert fused.count("TANH") == 1
    assert fused.count("RELU") == 1

    y_tensor = model_ir.tensors.get("y")
    y_h_tensor = model_ir.tensors.get("y_h")
    assert y_tensor is not None
    assert y_h_tensor is not None
    assert list(y_tensor.shape) == [4, 2, 1, 3]
    assert list(y_h_tensor.shape) == [2, 1, 3]


def test_flatbuffer_direct_transpose_mul_const_add_transpose_optimization() -> None:
    model = _make_transpose_mul_const_add_transpose_model()
    register_default_preprocess_rules()
    preprocessed_model, _ = run_preprocess_pipeline(onnx_graph=model)
    model_ir = lower_onnx_to_ir(
        onnx_graph=preprocessed_model,
        output_file_name="transpose_mul_const_add_transpose_opt_test",
    )
    op_types = [str(op.op_type) for op in model_ir.operators]
    assert op_types.count("TRANSPOSE") == 0
    assert op_types.count("MUL") == 1
    assert op_types.count("ADD") == 1
    assert op_types.count("RELU") == 0
    add_ops = [op for op in model_ir.operators if str(op.op_type) == "ADD"]
    assert len(add_ops) == 1
    assert str(add_ops[0].options.get("fusedActivationFunction", "NONE")).upper() == "RELU"
    mul_ops = [op for op in model_ir.operators if str(op.op_type) == "MUL"]
    assert len(mul_ops) == 1
    scale_tensor = model_ir.tensors.get("tmcat_scale_nchw")
    assert scale_tensor is not None
    assert list(scale_tensor.shape) == [1, 1, 1, 4]


def test_flatbuffer_direct_transpose_dequantize_transpose_optimization() -> None:
    model = _make_transpose_dequantize_transpose_model()
    register_default_preprocess_rules()
    preprocessed_model, _ = run_preprocess_pipeline(onnx_graph=model)
    model_ir = lower_onnx_to_ir(
        onnx_graph=preprocessed_model,
        output_file_name="transpose_dequantize_transpose_opt_test",
    )
    op_types = [str(op.op_type) for op in model_ir.operators]
    assert op_types.count("TRANSPOSE") == 0
    assert op_types.count("DEQUANTIZE") == 1


def test_flatbuffer_direct_transpose_cast_sub_mul_transpose_optimization() -> None:
    model = _make_transpose_cast_sub_mul_transpose_model()
    register_default_preprocess_rules()
    preprocessed_model, _ = run_preprocess_pipeline(onnx_graph=model)
    model_ir = lower_onnx_to_ir(
        onnx_graph=preprocessed_model,
        output_file_name="transpose_cast_sub_mul_transpose_opt_test",
    )
    op_types = [str(op.op_type) for op in model_ir.operators]
    assert op_types.count("TRANSPOSE") == 0
    assert op_types.count("CAST") == 1
    assert op_types.count("SUB") == 1
    assert op_types.count("MUL") == 1

    cast_op = next(op for op in model_ir.operators if str(op.op_type) == "CAST")
    assert [str(v) for v in list(cast_op.inputs)] == ["x"]
    mul_op = next(op for op in model_ir.operators if str(op.op_type) == "MUL")
    assert [str(v) for v in list(mul_op.outputs)] == ["y"]

    y_tensor = model_ir.tensors.get("y")
    assert y_tensor is not None
    assert list(y_tensor.shape) == [1, 4, 5, 3]


def test_flatbuffer_direct_transpose_dequantize_transpose_optimization_fanout_safe() -> None:
    model = _make_transpose_dequantize_transpose_with_fanout_model()
    register_default_preprocess_rules()
    preprocessed_model, _ = run_preprocess_pipeline(onnx_graph=model)
    model_ir = lower_onnx_to_ir(
        onnx_graph=preprocessed_model,
        output_file_name="transpose_dequantize_transpose_fanout_opt_test",
    )
    op_types = [str(op.op_type) for op in model_ir.operators]
    assert op_types.count("DEQUANTIZE") == 2
    assert op_types.count("TRANSPOSE") == 0


def test_flatbuffer_direct_transpose_quantize_dequantize_transpose_optimization() -> None:
    model = _make_transpose_quantize_dequantize_transpose_model()
    register_default_preprocess_rules()
    preprocessed_model, _ = run_preprocess_pipeline(onnx_graph=model)
    model_ir = lower_onnx_to_ir(
        onnx_graph=preprocessed_model,
        output_file_name="transpose_quantize_dequantize_transpose_opt_test",
    )
    op_types = [str(op.op_type) for op in model_ir.operators]
    assert op_types.count("TRANSPOSE") == 0
    assert op_types.count("QUANTIZE") == 1
    assert op_types.count("DEQUANTIZE") == 1


def test_flatbuffer_direct_transpose_qdq_add_qdq_transpose_residual_optimization() -> None:
    model = _make_transpose_qdq_add_qdq_transpose_residual_model()
    register_default_preprocess_rules()
    preprocessed_model, _ = run_preprocess_pipeline(onnx_graph=model)
    model_ir = lower_onnx_to_ir(
        onnx_graph=preprocessed_model,
        output_file_name="transpose_qdq_add_qdq_transpose_residual_opt_test",
    )
    op_types = [str(op.op_type) for op in model_ir.operators]
    assert op_types.count("TRANSPOSE") == 0
    assert op_types.count("QUANTIZE") == 3
    assert op_types.count("DEQUANTIZE") == 3
    assert op_types.count("ADD") == 1


def test_flatbuffer_direct_transpose_mixed_add_qdq_transpose_residual_optimization() -> None:
    model = _make_transpose_mixed_add_qdq_transpose_residual_model()
    register_default_preprocess_rules()
    preprocessed_model, _ = run_preprocess_pipeline(onnx_graph=model)
    model_ir = lower_onnx_to_ir(
        onnx_graph=preprocessed_model,
        output_file_name="transpose_mixed_add_qdq_transpose_residual_opt_test",
    )
    op_types = [str(op.op_type) for op in model_ir.operators]
    assert op_types.count("TRANSPOSE") == 0
    assert op_types.count("QUANTIZE") == 1
    assert op_types.count("DEQUANTIZE") == 1
    assert op_types.count("ADD") == 1


def test_flatbuffer_direct_transpose_mixed_add_qdq_transpose_residual_with_legacy_fanout_optimization() -> None:
    model = _make_transpose_mixed_add_qdq_transpose_residual_with_legacy_fanout_model()
    register_default_preprocess_rules()
    preprocessed_model, _ = run_preprocess_pipeline(onnx_graph=model)
    model_ir = lower_onnx_to_ir(
        onnx_graph=preprocessed_model,
        output_file_name="transpose_mixed_add_qdq_transpose_residual_legacy_fanout_opt_test",
    )
    op_types = [str(op.op_type) for op in model_ir.operators]
    assert op_types.count("TRANSPOSE") == 2
    assert op_types.count("QUANTIZE") == 2
    assert op_types.count("DEQUANTIZE") == 2
    assert op_types.count("ADD") == 2
    producer_of_y = next(
        (op for op in model_ir.operators if "y" in [str(v) for v in op.outputs]),
        None,
    )
    assert producer_of_y is not None
    assert str(producer_of_y.op_type) != "TRANSPOSE"


def test_flatbuffer_direct_transpose_dequantize_relu6_quantize_transpose_optimization() -> None:
    model = _make_transpose_dequantize_relu6_quantize_transpose_model()
    register_default_preprocess_rules()
    preprocessed_model, _ = run_preprocess_pipeline(onnx_graph=model)
    model_ir = lower_onnx_to_ir(
        onnx_graph=preprocessed_model,
        output_file_name="transpose_dequantize_relu6_quantize_transpose_opt_test",
    )
    op_types = [str(op.op_type) for op in model_ir.operators]
    assert op_types.count("TRANSPOSE") == 0
    assert op_types.count("DEQUANTIZE") == 1
    assert op_types.count("RELU6") == 1
    assert op_types.count("QUANTIZE") == 2


def test_flatbuffer_direct_transpose_relu6_transpose_fanout_optimization() -> None:
    model = _make_transpose_relu6_transpose_fanout_model()
    register_default_preprocess_rules()
    preprocessed_model, _ = run_preprocess_pipeline(onnx_graph=model)
    model_ir = lower_onnx_to_ir(
        onnx_graph=preprocessed_model,
        output_file_name="transpose_relu6_transpose_fanout_opt_test",
    )
    op_types = [str(op.op_type) for op in model_ir.operators]
    assert op_types.count("TRANSPOSE") == 0
    assert op_types.count("RELU6") == 1
    assert op_types.count("RELU") == 1
    assert op_types.count("NEG") == 1


def test_flatbuffer_direct_transpose_relu6_transpose_mixed_fanout_optimization() -> None:
    model = _make_transpose_relu6_transpose_mixed_fanout_model()
    register_default_preprocess_rules()
    preprocessed_model, _ = run_preprocess_pipeline(onnx_graph=model)
    model_ir = lower_onnx_to_ir(
        onnx_graph=preprocessed_model,
        output_file_name="transpose_relu6_transpose_mixed_fanout_opt_test",
    )
    op_types = [str(op.op_type) for op in model_ir.operators]
    assert op_types.count("TRANSPOSE") == 1
    assert op_types.count("RELU6") == 1
    assert op_types.count("RELU") == 1
    assert op_types.count("ADD") == 1


def test_flatbuffer_direct_transpose_relu6_binary_transpose_fanout_optimization() -> None:
    model = _make_transpose_relu6_binary_transpose_fanout_model()
    register_default_preprocess_rules()
    preprocessed_model, _ = run_preprocess_pipeline(onnx_graph=model)
    model_ir = lower_onnx_to_ir(
        onnx_graph=preprocessed_model,
        output_file_name="transpose_relu6_binary_transpose_fanout_opt_test",
    )
    op_types = [str(op.op_type) for op in model_ir.operators]
    assert op_types.count("TRANSPOSE") == 0
    assert op_types.count("RELU6") == 1
    assert op_types.count("ADD") == 1
    assert op_types.count("RELU") == 1
    assert op_types.count("NEG") == 1


def test_flatbuffer_direct_transpose_relu6_binary_mixed_fanout_optimization() -> None:
    model = _make_transpose_relu6_binary_mixed_fanout_model()
    register_default_preprocess_rules()
    preprocessed_model, _ = run_preprocess_pipeline(onnx_graph=model)
    model_ir = lower_onnx_to_ir(
        onnx_graph=preprocessed_model,
        output_file_name="transpose_relu6_binary_mixed_fanout_opt_test",
    )
    op_types = [str(op.op_type) for op in model_ir.operators]
    assert op_types.count("TRANSPOSE") == 1
    assert op_types.count("RELU6") == 1
    assert op_types.count("ADD") == 2
    assert op_types.count("RELU") == 1


@pytest.mark.parametrize("binary_op", ["Add", "Sub", "Mul", "Div"])
def test_flatbuffer_direct_transpose_binary_transpose_optimization(binary_op: str) -> None:
    model = _make_transpose_binary_transpose_model(binary_op)
    register_default_preprocess_rules()
    preprocessed_model, _ = run_preprocess_pipeline(onnx_graph=model)
    model_ir = lower_onnx_to_ir(
        onnx_graph=preprocessed_model,
        output_file_name=f"transpose_{binary_op.lower()}_transpose_opt_test",
    )
    op_types = [str(op.op_type) for op in model_ir.operators]
    expected_binary = binary_op.upper()
    assert op_types.count("TRANSPOSE") == 0
    assert op_types.count(expected_binary) == 1


def test_flatbuffer_direct_transpose_transpose_softmax_compose_optimization() -> None:
    model = _make_transpose_transpose_softmax_model()
    register_default_preprocess_rules()
    preprocessed_model, _ = run_preprocess_pipeline(onnx_graph=model)
    model_ir = lower_onnx_to_ir(
        onnx_graph=preprocessed_model,
        output_file_name="transpose_transpose_softmax_compose_opt_test",
    )
    op_types = [str(op.op_type) for op in model_ir.operators]
    assert op_types.count("SOFTMAX") == 1
    assert op_types.count("TRANSPOSE") == 1


def test_flatbuffer_direct_softmax_terminal_transpose_preserved() -> None:
    model = _make_softmax_terminal_transpose_model()
    register_default_preprocess_rules()
    preprocessed_model, _ = run_preprocess_pipeline(onnx_graph=model)
    model_ir = lower_onnx_to_ir(
        onnx_graph=preprocessed_model,
        output_file_name="softmax_terminal_transpose_preserve_test",
    )
    op_types = [str(op.op_type) for op in model_ir.operators]
    assert op_types.count("SOFTMAX") == 1
    assert op_types.count("TRANSPOSE") == 1
    y_tensor = model_ir.tensors.get("y")
    assert y_tensor is not None
    assert list(y_tensor.shape) == [1, 4, 3, 2]


def test_flatbuffer_direct_softmax_terminal_double_pretranspose_canonicalization() -> None:
    model = _make_softmax_terminal_double_pretranspose_model()
    register_default_preprocess_rules()
    preprocessed_model, _ = run_preprocess_pipeline(onnx_graph=model)
    model_ir = lower_onnx_to_ir(
        onnx_graph=preprocessed_model,
        output_file_name="softmax_terminal_double_pretranspose_canon_test",
    )
    op_types = [str(op.op_type) for op in model_ir.operators]
    assert op_types.count("SOFTMAX") == 1
    assert op_types.count("TRANSPOSE") == 0

    softmax_op = next(op for op in model_ir.operators if str(op.op_type) == "SOFTMAX")
    assert list(softmax_op.inputs) == ["x"]
    assert list(softmax_op.outputs) == ["y"]

    y_tensor = model_ir.tensors.get("y")
    assert y_tensor is not None
    assert list(y_tensor.shape) == [1, 2, 3, 4]


def test_flatbuffer_direct_transpose_swish_transpose_optimization() -> None:
    model = _make_transpose_swish_transpose_model()
    register_default_preprocess_rules()
    preprocessed_model, _ = run_preprocess_pipeline(onnx_graph=model)
    model_ir = lower_onnx_to_ir(
        onnx_graph=preprocessed_model,
        output_file_name="transpose_swish_transpose_opt_test",
    )
    op_types = [str(op.op_type) for op in model_ir.operators]
    assert op_types.count("TRANSPOSE") == 0
    assert op_types.count("LOGISTIC") == 1
    assert op_types.count("MUL") == 1
    assert op_types.count("RELU") == 0


def test_flatbuffer_direct_maximum_minimum_chain_rewrites_to_relu_0_to_1() -> None:
    model_ir = ModelIR(name="maximum_minimum_relu0to1_opt_test")
    model_ir.inputs = ["x"]
    model_ir.outputs = ["y"]
    model_ir.tensors["x"] = TensorIR(
        name="x",
        dtype="FLOAT32",
        shape=[1, 2, 3, 4],
        shape_signature=[1, 2, 3, 4],
    )
    model_ir.tensors["zero"] = TensorIR(
        name="zero",
        dtype="FLOAT32",
        shape=[],
        shape_signature=[],
        data=np.asarray(0.0, dtype=np.float32),
        is_variable=False,
    )
    model_ir.tensors["one"] = TensorIR(
        name="one",
        dtype="FLOAT32",
        shape=[],
        shape_signature=[],
        data=np.asarray(1.0, dtype=np.float32),
        is_variable=False,
    )
    model_ir.tensors["m"] = TensorIR(
        name="m",
        dtype="FLOAT32",
        shape=[1, 2, 3, 4],
        shape_signature=[1, 2, 3, 4],
    )
    model_ir.tensors["y"] = TensorIR(
        name="y",
        dtype="FLOAT32",
        shape=[1, 2, 3, 4],
        shape_signature=[1, 2, 3, 4],
    )
    model_ir.operators = [
        OperatorIR(op_type="MAXIMUM", inputs=["x", "zero"], outputs=["m"]),
        OperatorIR(op_type="MINIMUM", inputs=["m", "one"], outputs=["y"]),
    ]

    stats = _optimize_maximum_minimum_relu0to1_chains(model_ir)
    assert stats["rewritten_maximum_minimum_relu0to1_chains"] == 1
    op_types = [str(op.op_type) for op in model_ir.operators]
    assert op_types == ["RELU_0_TO_1"]
    assert [str(v) for v in list(model_ir.operators[0].inputs)] == ["x"]
    assert [str(v) for v in list(model_ir.operators[0].outputs)] == ["y"]


def test_flatbuffer_direct_maximum_with_zero_input2_rewrites_to_relu() -> None:
    model_ir = ModelIR(name="maximum_input2_zero_relu_opt_test")
    model_ir.inputs = ["x"]
    model_ir.outputs = ["y"]
    model_ir.tensors["x"] = TensorIR(
        name="x",
        dtype="FLOAT32",
        shape=[1, 2, 3, 4],
        shape_signature=[1, 2, 3, 4],
    )
    model_ir.tensors["zero"] = TensorIR(
        name="zero",
        dtype="FLOAT32",
        shape=[],
        shape_signature=[],
        data=np.asarray(0.0, dtype=np.float32),
        is_variable=False,
    )
    model_ir.tensors["y"] = TensorIR(
        name="y",
        dtype="FLOAT32",
        shape=[1, 2, 3, 4],
        shape_signature=[1, 2, 3, 4],
    )
    model_ir.operators = [
        OperatorIR(op_type="MAXIMUM", inputs=["x", "zero"], outputs=["y"]),
    ]

    stats = _optimize_maximum_with_zero_input2_to_relu(model_ir)
    assert stats["rewritten_maximum_with_zero_input2_to_relu"] == 1
    op_types = [str(op.op_type) for op in model_ir.operators]
    assert op_types == ["RELU"]
    assert [str(v) for v in list(model_ir.operators[0].inputs)] == ["x"]
    assert [str(v) for v in list(model_ir.operators[0].outputs)] == ["y"]


def test_flatbuffer_direct_maximum_with_zero_input1_is_not_rewritten() -> None:
    model_ir = ModelIR(name="maximum_input1_zero_no_relu_opt_test")
    model_ir.inputs = ["x"]
    model_ir.outputs = ["y"]
    model_ir.tensors["x"] = TensorIR(
        name="x",
        dtype="FLOAT32",
        shape=[1, 2, 3, 4],
        shape_signature=[1, 2, 3, 4],
    )
    model_ir.tensors["zero"] = TensorIR(
        name="zero",
        dtype="FLOAT32",
        shape=[],
        shape_signature=[],
        data=np.asarray(0.0, dtype=np.float32),
        is_variable=False,
    )
    model_ir.tensors["y"] = TensorIR(
        name="y",
        dtype="FLOAT32",
        shape=[1, 2, 3, 4],
        shape_signature=[1, 2, 3, 4],
    )
    model_ir.operators = [
        OperatorIR(op_type="MAXIMUM", inputs=["zero", "x"], outputs=["y"]),
    ]

    stats = _optimize_maximum_with_zero_input2_to_relu(model_ir)
    assert stats["rewritten_maximum_with_zero_input2_to_relu"] == 0
    op_types = [str(op.op_type) for op in model_ir.operators]
    assert op_types == ["MAXIMUM"]


def test_flatbuffer_direct_transpose_hardsigmoid_transpose_optimization() -> None:
    model = _make_transpose_hardsigmoid_transpose_model()
    register_default_preprocess_rules()
    preprocessed_model, _ = run_preprocess_pipeline(onnx_graph=model)
    model_ir = lower_onnx_to_ir(
        onnx_graph=preprocessed_model,
        output_file_name="transpose_hardsigmoid_transpose_opt_test",
    )
    op_types = [str(op.op_type) for op in model_ir.operators]
    assert op_types.count("TRANSPOSE") == 0
    assert op_types.count("MUL") == 1
    assert op_types.count("ADD") == 1
    assert op_types.count("MAXIMUM") == 0
    assert op_types.count("MINIMUM") == 0
    assert op_types.count("RELU_0_TO_1") == 1
    assert op_types.count("RELU") == 1


def test_flatbuffer_direct_transpose_hardsigmoid_mul_transpose_optimization() -> None:
    model = _make_transpose_hardsigmoid_mul_transpose_model()
    register_default_preprocess_rules()
    preprocessed_model, _ = run_preprocess_pipeline(onnx_graph=model)
    model_ir = lower_onnx_to_ir(
        onnx_graph=preprocessed_model,
        output_file_name="transpose_hardsigmoid_mul_transpose_opt_test",
    )
    op_types = [str(op.op_type) for op in model_ir.operators]
    assert op_types.count("TRANSPOSE") == 0
    assert op_types.count("MUL") == 2
    assert op_types.count("ADD") == 1
    assert op_types.count("MAXIMUM") == 0
    assert op_types.count("MINIMUM") == 0
    assert op_types.count("RELU_0_TO_1") == 1


def test_flatbuffer_direct_transpose_prelu_transpose_optimization() -> None:
    model = _make_transpose_prelu_transpose_model()
    register_default_preprocess_rules()
    preprocessed_model, _ = run_preprocess_pipeline(onnx_graph=model)
    model_ir = lower_onnx_to_ir(
        onnx_graph=preprocessed_model,
        output_file_name="transpose_prelu_transpose_opt_test",
    )
    op_types = [str(op.op_type) for op in model_ir.operators]
    assert op_types.count("TRANSPOSE") == 0
    assert op_types.count("PRELU") == 1
    prelu_op = next(op for op in model_ir.operators if str(op.op_type) == "PRELU")
    assert [str(v) for v in list(prelu_op.inputs)][0] == "x"
    assert [str(v) for v in list(prelu_op.outputs)][0] == "y"
    alpha_name = str(prelu_op.inputs[1])
    alpha_tensor = model_ir.tensors.get(alpha_name, None)
    assert alpha_tensor is not None
    assert isinstance(alpha_tensor.data, np.ndarray)
    assert list(np.asarray(alpha_tensor.data).shape) == [1, 1, 1, 4]


def test_flatbuffer_direct_transpose_gelu_transpose_optimization() -> None:
    model = _make_transpose_gelu_transpose_model()
    register_default_preprocess_rules()
    preprocessed_model, _ = run_preprocess_pipeline(onnx_graph=model)
    model_ir = lower_onnx_to_ir(
        onnx_graph=preprocessed_model,
        output_file_name="transpose_gelu_transpose_opt_test",
    )
    op_types = [str(op.op_type) for op in model_ir.operators]
    assert op_types.count("TRANSPOSE") == 0
    assert op_types.count("GELU") == 1
    gelu_op = next(op for op in model_ir.operators if str(op.op_type) == "GELU")
    assert [str(v) for v in list(gelu_op.inputs)] == ["x"]
    assert [str(v) for v in list(gelu_op.outputs)] == ["z"]


def test_flatbuffer_direct_transpose_gelu_tanh_transpose_optimization() -> None:
    model = _make_transpose_gelu_tanh_transpose_model()
    register_default_preprocess_rules()
    preprocessed_model, _ = run_preprocess_pipeline(onnx_graph=model)
    model_ir = lower_onnx_to_ir(
        onnx_graph=preprocessed_model,
        output_file_name="transpose_gelu_tanh_transpose_opt_test",
    )
    op_types = [str(op.op_type) for op in model_ir.operators]
    assert op_types.count("TRANSPOSE") == 0
    assert op_types.count("TANH") == 1
    assert op_types.count("MUL") == 6
    assert op_types.count("ADD") == 2
    assert op_types.count("RELU") == 0


def test_flatbuffer_direct_transpose_add_reshape_transpose_suffix_optimization() -> None:
    model = _make_transpose_add_reshape_transpose_with_legacy_model()
    register_default_preprocess_rules()
    preprocessed_model, _ = run_preprocess_pipeline(onnx_graph=model)
    model_ir_raw = lower_onnx_to_ir(
        onnx_graph=preprocessed_model,
        output_file_name="transpose_add_reshape_suffix_raw_test",
        optimize_layout_transpose_chains=False,
    )
    model_ir_opt = lower_onnx_to_ir(
        onnx_graph=preprocessed_model,
        output_file_name="transpose_add_reshape_suffix_opt_test",
    )

    raw_transpose_count = sum(1 for op in model_ir_raw.operators if str(op.op_type) == "TRANSPOSE")
    opt_transpose_count = sum(1 for op in model_ir_opt.operators if str(op.op_type) == "TRANSPOSE")
    assert opt_transpose_count < raw_transpose_count

    for op in model_ir_opt.operators:
        if str(op.op_type) != "TRANSPOSE" or len(op.inputs) < 2:
            continue
        perm_tensor = model_ir_opt.tensors.get(str(op.inputs[1]))
        if perm_tensor is None or perm_tensor.data is None:
            continue
        perm = [int(v) for v in np.asarray(perm_tensor.data).reshape(-1).tolist()]
        assert perm != [0, 2, 1]

    producer_of_z = None
    for op in model_ir_opt.operators:
        if "z" in [str(v) for v in op.outputs]:
            producer_of_z = op
            break
    assert producer_of_z is not None
    assert str(producer_of_z.op_type) == "RESHAPE"


def test_flatbuffer_direct_transpose_mulconst_add_reshape_transpose_suffix_optimization() -> None:
    model = _make_transpose_mulconst_add_reshape_transpose_with_legacy_model()
    register_default_preprocess_rules()
    preprocessed_model, _ = run_preprocess_pipeline(onnx_graph=model)
    model_ir_raw = lower_onnx_to_ir(
        onnx_graph=preprocessed_model,
        output_file_name="transpose_mulconst_add_reshape_suffix_raw_test",
        optimize_layout_transpose_chains=False,
    )
    model_ir_opt = lower_onnx_to_ir(
        onnx_graph=preprocessed_model,
        output_file_name="transpose_mulconst_add_reshape_suffix_opt_test",
    )

    raw_transpose_count = sum(1 for op in model_ir_raw.operators if str(op.op_type) == "TRANSPOSE")
    opt_transpose_count = sum(1 for op in model_ir_opt.operators if str(op.op_type) == "TRANSPOSE")
    assert opt_transpose_count < raw_transpose_count

    output_names_by_op = {
        str(output_name)
        for op in model_ir_opt.operators
        if str(op.op_type) == "TRANSPOSE"
        for output_name in list(op.outputs)
    }
    assert "y_nchw" not in output_names_by_op

    producer_of_z_mid = next(
        (op for op in model_ir_opt.operators if "z_mid" in [str(v) for v in op.outputs]),
        None,
    )
    if producer_of_z_mid is not None:
        assert str(producer_of_z_mid.op_type) == "RESHAPE"
    else:
        producer_of_z = next(
            (op for op in model_ir_opt.operators if "z" in [str(v) for v in op.outputs]),
            None,
        )
        assert producer_of_z is not None
        assert str(producer_of_z.op_type) == "RESHAPE"


def test_flatbuffer_direct_transpose_nested_add_with_shared_skip_optimization() -> None:
    model = _make_transpose_nested_add_with_shared_skip_model()
    register_default_preprocess_rules()
    preprocessed_model, _ = run_preprocess_pipeline(onnx_graph=model)
    model_ir_opt = lower_onnx_to_ir(
        onnx_graph=preprocessed_model,
        output_file_name="transpose_nested_add_with_shared_skip_opt_test",
    )
    op_types = [str(op.op_type) for op in model_ir_opt.operators]
    assert op_types.count("TRANSPOSE") == 0
    assert op_types.count("MUL") == 1
    add_ops = [op for op in model_ir_opt.operators if str(op.op_type) == "ADD"]
    assert len(add_ops) == 2
    producer_of_z = next(
        (op for op in model_ir_opt.operators if "z" in [str(v) for v in op.outputs]),
        None,
    )
    assert producer_of_z is not None
    assert str(producer_of_z.op_type) == "ADD"


def test_flatbuffer_direct_transpose_swish_add_concat_transpose_optimization() -> None:
    model = _make_transpose_swish_add_concat_transpose_model()
    register_default_preprocess_rules()
    preprocessed_model, _ = run_preprocess_pipeline(onnx_graph=model)
    model_ir = lower_onnx_to_ir(
        onnx_graph=preprocessed_model,
        output_file_name="transpose_swish_add_concat_transpose_opt_test",
    )
    op_types = [str(op.op_type) for op in model_ir.operators]
    assert op_types.count("TRANSPOSE") == 0
    assert op_types.count("LOGISTIC") == 3
    assert op_types.count("MUL") == 3
    assert op_types.count("ADD") == 1
    assert op_types.count("CONCATENATION") == 1
    concat_ops = [op for op in model_ir.operators if str(op.op_type) == "CONCATENATION"]
    assert len(concat_ops) == 1
    assert int(concat_ops[0].options.get("axis", -1)) == 3


def test_flatbuffer_direct_transpose_swish_add_transpose_cascade_optimization() -> None:
    model = _make_transpose_swish_add_transpose_cascade_model()
    register_default_preprocess_rules()
    preprocessed_model, _ = run_preprocess_pipeline(onnx_graph=model)
    model_ir = lower_onnx_to_ir(
        onnx_graph=preprocessed_model,
        output_file_name="transpose_swish_add_transpose_cascade_opt_test",
    )
    op_types = [str(op.op_type) for op in model_ir.operators]
    assert op_types.count("TRANSPOSE") == 0
    assert op_types.count("ADD") == 2
    assert op_types.count("LOGISTIC") == 3
    assert op_types.count("MUL") == 3
    assert op_types.count("RELU") == 0
    add_ops = [op for op in model_ir.operators if str(op.op_type) == "ADD"]
    assert any(str(op.options.get("fusedActivationFunction", "NONE")).upper() == "RELU" for op in add_ops)


def test_flatbuffer_direct_transpose_slice_concat_transpose_optimization() -> None:
    model = _make_transpose_slice_concat_transpose_model()
    register_default_preprocess_rules()
    preprocessed_model, _ = run_preprocess_pipeline(onnx_graph=model)
    model_ir = lower_onnx_to_ir(
        onnx_graph=preprocessed_model,
        output_file_name="transpose_slice_concat_transpose_opt_test",
    )
    op_types = [str(op.op_type) for op in model_ir.operators]
    assert op_types.count("TRANSPOSE") == 0
    assert op_types.count("STRIDED_SLICE") == 4
    assert op_types.count("CONCATENATION") == 1
    assert op_types.count("RELU") == 1
    concat_ops = [op for op in model_ir.operators if str(op.op_type) == "CONCATENATION"]
    assert len(concat_ops) == 1
    assert int(concat_ops[0].options.get("axis", -1)) == 3


def test_flatbuffer_direct_transpose_logistic_concat_reshape_optimization() -> None:
    model = _make_transpose_logistic_concat_reshape_model()
    register_default_preprocess_rules()
    preprocessed_model, _ = run_preprocess_pipeline(onnx_graph=model)
    model_ir = lower_onnx_to_ir(
        onnx_graph=preprocessed_model,
        output_file_name="transpose_logistic_concat_reshape_opt_test",
    )
    op_types = [str(op.op_type) for op in model_ir.operators]
    assert op_types.count("TRANSPOSE") == 1
    assert op_types.count("LOGISTIC") == 2
    assert op_types.count("CONCATENATION") == 1
    assert op_types.count("RESHAPE") == 1
    concat_ops = [op for op in model_ir.operators if str(op.op_type) == "CONCATENATION"]
    assert len(concat_ops) == 1
    assert int(concat_ops[0].options.get("axis", -1)) == 3


@pytest.mark.parametrize("binary_op", ["Add", "Sub", "Mul", "Div"])
def test_flatbuffer_direct_transpose_binary_transpose_fanout_optimization(binary_op: str) -> None:
    model = _make_transpose_binary_transpose_fanout_model(binary_op)
    register_default_preprocess_rules()
    preprocessed_model, _ = run_preprocess_pipeline(onnx_graph=model)
    model_ir = lower_onnx_to_ir(
        onnx_graph=preprocessed_model,
        output_file_name=f"transpose_{binary_op.lower()}_transpose_fanout_opt_test",
    )
    op_types = [str(op.op_type) for op in model_ir.operators]
    expected_binary = binary_op.upper()
    assert op_types.count("TRANSPOSE") == 1
    expected_binary_count = 2 if expected_binary == "ADD" else 1
    assert op_types.count(expected_binary) == expected_binary_count
    expected_add_count = 2 if expected_binary == "ADD" else 1
    assert op_types.count("ADD") == expected_add_count
    assert op_types.count("RELU") == 1


@pytest.mark.parametrize("binary_op", ["Add", "Sub", "Mul", "Div"])
def test_flatbuffer_direct_transpose_binary_no_post_transpose_optimization(binary_op: str) -> None:
    model = _make_transpose_binary_no_post_transpose_model(binary_op)
    register_default_preprocess_rules()
    preprocessed_model, _ = run_preprocess_pipeline(onnx_graph=model)
    model_ir = lower_onnx_to_ir(
        onnx_graph=preprocessed_model,
        output_file_name=f"transpose_{binary_op.lower()}_no_post_transpose_opt_test",
    )
    op_types = [str(op.op_type) for op in model_ir.operators]
    expected_binary = binary_op.upper()
    assert op_types.count("TRANSPOSE") == 1
    assert op_types.count(expected_binary) == 1
    assert op_types.count("RELU") == 1


@pytest.mark.parametrize("binary_op", ["Add", "Sub", "Mul", "Div"])
@pytest.mark.parametrize("transpose_on_lhs", [True, False])
def test_flatbuffer_direct_transpose_binary_single_side_transpose_optimization(
    binary_op: str,
    transpose_on_lhs: bool,
) -> None:
    model = _make_transpose_binary_single_side_transpose_model(
        binary_op,
        transpose_on_lhs=transpose_on_lhs,
    )
    register_default_preprocess_rules()
    preprocessed_model, _ = run_preprocess_pipeline(onnx_graph=model)
    model_ir = lower_onnx_to_ir(
        onnx_graph=preprocessed_model,
        output_file_name=f"transpose_single_side_{binary_op.lower()}_transpose_opt_test",
    )

    expected_binary = binary_op.upper()
    op_types = [str(op.op_type) for op in model_ir.operators]
    assert op_types.count(expected_binary) == 1
    assert op_types.count("TRANSPOSE") == 1

    binary_ops = [op for op in model_ir.operators if str(op.op_type) == expected_binary]
    assert len(binary_ops) == 1
    binary_ir = binary_ops[0]
    assert binary_ir.outputs == ["z"]

    transpose_ops = [op for op in model_ir.operators if str(op.op_type) == "TRANSPOSE"]
    assert len(transpose_ops) == 1
    assert transpose_ops[0].outputs[0] in set(binary_ir.inputs)


@pytest.mark.skipif(not _requires_flatbuffer_tools(), reason="flatbuffer_direct requires flatc and curl")
def test_flatbuffer_direct_clip_relu6_smoke() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        model = _make_clip_relu6_model()
        model_path = _save_model(tmpdir, "clip_relu6", model)
        out_dir = os.path.join(tmpdir, "out")
        tflite_path = _convert(model_path, out_dir, "flatbuffer_direct")

        interpreter = Interpreter(model_path=tflite_path)
        interpreter.allocate_tensors()
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        x = np.array([[-1.0, 3.0, 10.0]], dtype=np.float32)
        interpreter.set_tensor(input_details[0]["index"], x)
        interpreter.invoke()
        y = interpreter.get_tensor(output_details[0]["index"])
        np.testing.assert_allclose(
            y,
            np.array([[0.0, 3.0, 6.0]], dtype=np.float32),
            rtol=0.0,
            atol=1e-6,
        )


@pytest.mark.skipif(not _requires_flatbuffer_tools(), reason="flatbuffer_direct requires flatc and curl")
def test_flatbuffer_direct_clip_relu_n1_to_1_smoke() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        model = _make_clip_relu_n1_to_1_model()
        model_path = _save_model(tmpdir, "clip_relu_n1_to_1", model)
        out_dir = os.path.join(tmpdir, "out")
        tflite_path = _convert(model_path, out_dir, "flatbuffer_direct")

        op_names = _collect_builtin_op_names(tflite_path)
        assert "RELU_N1_TO_1" in op_names

        interpreter = Interpreter(model_path=tflite_path)
        interpreter.allocate_tensors()
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        x = np.array([[-3.0, -0.25, 2.0]], dtype=np.float32)
        interpreter.set_tensor(input_details[0]["index"], x)
        interpreter.invoke()
        y = interpreter.get_tensor(output_details[0]["index"])
        np.testing.assert_allclose(
            y,
            np.array([[-1.0, -0.25, 1.0]], dtype=np.float32),
            rtol=0.0,
            atol=1e-6,
        )


@pytest.mark.skipif(not _requires_flatbuffer_tools(), reason="flatbuffer_direct requires flatc and curl")
def test_flatbuffer_direct_prelu_emits_builtin_prelu() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        model = _make_prelu_model()
        model_path = _save_model(tmpdir, "prelu_builtin", model)
        out_dir = os.path.join(tmpdir, "out")
        tflite_path = _convert(model_path, out_dir, "flatbuffer_direct")
        op_names = _collect_builtin_op_names(tflite_path)
        assert "PRELU" in op_names


@pytest.mark.skipif(not _requires_flatbuffer_tools(), reason="flatbuffer_direct requires flatc and curl")
def test_flatbuffer_direct_leakyrelu_emits_builtin_leaky_relu() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        model = _make_leakyrelu_model()
        model_path = _save_model(tmpdir, "leakyrelu_builtin", model)
        out_dir = os.path.join(tmpdir, "out")
        tflite_path = _convert(model_path, out_dir, "flatbuffer_direct")
        op_names = _collect_builtin_op_names(tflite_path)
        assert "LEAKY_RELU" in op_names


@pytest.mark.skipif(not _requires_flatbuffer_tools(), reason="flatbuffer_direct requires flatc and curl")
def test_flatbuffer_direct_instance_normalization_emits_builtin_chain() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        model = _make_instance_normalization_model()
        model_path = _save_model(tmpdir, "instance_norm_builtin", model)
        out_dir = os.path.join(tmpdir, "out")
        tflite_path = _convert(
            model_path,
            out_dir,
            "flatbuffer_direct",
            report_op_coverage=True,
        )
        op_names = _collect_builtin_op_names(tflite_path)
        custom_codes = _collect_custom_codes(tflite_path)
        assert "MEAN" in op_names
        assert "SUB" in op_names
        assert "SQRT" in op_names
        assert "ONNX_INSTANCENORMALIZATION" not in custom_codes
        report_path = os.path.join(out_dir, "instance_norm_builtin_op_coverage_report.json")
        assert os.path.isfile(report_path)
        with open(report_path, "r", encoding="utf-8") as f:
            report = json.load(f)
        assert report["conversion_error"] is None
        assert report["graph_summary"]["custom_lowered_nodes"] == 0
        assert any(
            r["onnx_op"] == "InstanceNormalization" and r["dispatch_mode"] == "builtin"
            for r in report["graph_node_reports"]
        )


@pytest.mark.skipif(not _requires_flatbuffer_tools(), reason="flatbuffer_direct requires flatc and curl")
def test_flatbuffer_direct_dropout_emits_builtin_chain() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        model = _make_dropout_model()
        model_path = _save_model(tmpdir, "dropout_builtin", model)
        out_dir = os.path.join(tmpdir, "out")
        tflite_path = _convert(
            model_path,
            out_dir,
            "flatbuffer_direct",
            report_op_coverage=True,
        )
        custom_codes = _collect_custom_codes(tflite_path)
        assert "ONNX_DROPOUT" not in custom_codes
        report_path = os.path.join(out_dir, "dropout_builtin_op_coverage_report.json")
        assert os.path.isfile(report_path)
        with open(report_path, "r", encoding="utf-8") as f:
            report = json.load(f)
        assert report["conversion_error"] is None
        assert report["graph_summary"]["custom_lowered_nodes"] == 0
        assert any(
            r["onnx_op"] == "Dropout" and r["dispatch_mode"] == "builtin"
            for r in report["graph_node_reports"]
        )


@pytest.mark.skipif(not _requires_flatbuffer_tools(), reason="flatbuffer_direct requires flatc and curl")
def test_flatbuffer_direct_custom_op_candidate_disabled_fails() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        model = _make_einsum_custom_model()
        model_path = _save_model(tmpdir, "einsum_custom_disabled", model)
        out_dir = os.path.join(tmpdir, "out")
        _convert(
            model_path,
            out_dir,
            "flatbuffer_direct",
            report_op_coverage=True,
        )
        report_path = os.path.join(out_dir, "einsum_custom_disabled_op_coverage_report.json")
        assert os.path.isfile(report_path)
        with open(report_path, "r", encoding="utf-8") as f:
            report = json.load(f)
        assert "Einsum" in report["graph_custom_ops"]
        assert report["graph_summary"]["custom_lowered_nodes"] == 1
        assert any(
            node["onnx_op"] == "Einsum" and node["dispatch_mode"] == "custom"
            for node in report["graph_node_reports"]
        )


@pytest.mark.skipif(not _requires_flatbuffer_tools(), reason="flatbuffer_direct requires flatc and curl")
def test_flatbuffer_direct_custom_op_enabled_generates_custom_code() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        model = _make_einsum_custom_model()
        model_path = _save_model(tmpdir, "einsum_custom_enabled", model)
        out_dir = os.path.join(tmpdir, "out")
        tflite_path = _convert(
            model_path,
            out_dir,
            "flatbuffer_direct",
            flatbuffer_direct_allow_custom_ops=True,
            flatbuffer_direct_custom_op_allowlist=["Einsum"],
        )
        assert os.path.isfile(tflite_path)
        custom_codes = _collect_custom_codes(tflite_path)
        assert "ONNX_EINSUM" in custom_codes


@pytest.mark.skipif(not _requires_flatbuffer_tools(), reason="flatbuffer_direct requires flatc and curl")
def test_flatbuffer_direct_custom_op_not_in_allowlist_fails() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        model = _make_einsum_custom_model()
        model_path = _save_model(tmpdir, "einsum_custom_allowlist_fail", model)
        out_dir = os.path.join(tmpdir, "out")
        with pytest.raises(NotImplementedError, match="custom_op_not_in_allowlist"):
            _convert(
                model_path,
                out_dir,
                "flatbuffer_direct",
                flatbuffer_direct_allow_custom_ops=True,
                flatbuffer_direct_custom_op_allowlist=["TopK"],
            )


@pytest.mark.skipif(not _requires_flatbuffer_tools(), reason="flatbuffer_direct requires flatc and curl")
def test_flatbuffer_direct_custom_op_coverage_report() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        model = _make_einsum_custom_model()
        model_path = _save_model(tmpdir, "einsum_custom_report", model)
        out_dir = os.path.join(tmpdir, "out")
        _convert(
            model_path,
            out_dir,
            "flatbuffer_direct",
            report_op_coverage=True,
            flatbuffer_direct_allow_custom_ops=True,
            flatbuffer_direct_custom_op_allowlist=["Einsum"],
        )
        report_path = os.path.join(out_dir, "einsum_custom_report_op_coverage_report.json")
        assert os.path.isfile(report_path)
        with open(report_path, "r", encoding="utf-8") as f:
            report = json.load(f)
        assert report["custom_op_policy"]["allow_custom_ops"] is True
        assert "Einsum" in report["graph_custom_ops"]
        assert report["graph_summary"]["custom_lowered_nodes"] == 1
        assert (
            report["custom_op_policy"]["candidate_count_excluding_builtin_supported"]
            == report["custom_op_policy"]["candidate_count"]
            - len(report["custom_op_policy"]["candidate_ops_now_builtin_supported"])
        )
        assert "Einsum" in report["custom_op_policy"]["candidate_ops_now_builtin_supported"]
        assert "Einsum" in report["custom_op_policy"]["allowlist_builtin_supported_ops"]
        assert "Einsum" in report["custom_op_policy"]["allowlist_custom_candidate_ops"]
        assert report["custom_op_policy"]["allowlist_unknown_ops"] == []
        node_reports = report["graph_node_reports"]
        assert any(
            r["onnx_op"] == "Einsum" and r["dispatch_mode"] == "custom"
            for r in node_reports
        )


@pytest.mark.skipif(not _requires_flatbuffer_tools(), reason="flatbuffer_direct requires flatc and curl")
def test_flatbuffer_direct_einsum_builtin_preferred_over_custom() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        model = _make_einsum_fc_const_model()
        model_path = _save_model(tmpdir, "einsum_builtin_preferred", model)
        out_dir = os.path.join(tmpdir, "out")
        tflite_path = _convert(
            model_path,
            out_dir,
            "flatbuffer_direct",
            flatbuffer_direct_allow_custom_ops=True,
            flatbuffer_direct_custom_op_allowlist=["Einsum"],
            report_op_coverage=True,
        )
        assert os.path.isfile(tflite_path)
        custom_codes = _collect_custom_codes(tflite_path)
        assert "ONNX_EINSUM" not in custom_codes

        report_path = os.path.join(out_dir, "einsum_builtin_preferred_op_coverage_report.json")
        assert os.path.isfile(report_path)
        with open(report_path, "r", encoding="utf-8") as f:
            report = json.load(f)
        assert report["graph_summary"]["custom_lowered_nodes"] == 0
        assert "Einsum" not in report["graph_custom_ops"]
        assert any(
            r["onnx_op"] == "Einsum" and r["dispatch_mode"] == "builtin"
            for r in report["graph_node_reports"]
        )


@pytest.mark.skipif(not _requires_flatbuffer_tools(), reason="flatbuffer_direct requires flatc and curl")
def test_flatbuffer_direct_einsum_nonconst_rhs_builtin() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        model = _make_einsum_model()
        model_path = _save_model(tmpdir, "einsum_nonconst_builtin", model)
        out_dir = os.path.join(tmpdir, "out")
        tflite_path = _convert(
            model_path,
            out_dir,
            "flatbuffer_direct",
            flatbuffer_direct_allow_custom_ops=True,
            flatbuffer_direct_custom_op_allowlist=["Einsum"],
            report_op_coverage=True,
        )
        assert os.path.isfile(tflite_path)
        custom_codes = _collect_custom_codes(tflite_path)
        assert "ONNX_EINSUM" not in custom_codes

        report_path = os.path.join(out_dir, "einsum_nonconst_builtin_op_coverage_report.json")
        assert os.path.isfile(report_path)
        with open(report_path, "r", encoding="utf-8") as f:
            report = json.load(f)
        assert report["graph_summary"]["custom_lowered_nodes"] == 0
        assert "Einsum" not in report["graph_custom_ops"]
        assert any(
            r["onnx_op"] == "Einsum" and r["dispatch_mode"] == "builtin"
            for r in report["graph_node_reports"]
        )


@pytest.mark.skipif(not _requires_flatbuffer_tools(), reason="flatbuffer_direct requires flatc and curl")
def test_flatbuffer_direct_fallback_to_tf_converter_smoke() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        model = _make_elu_model()
        model_path = _save_model(tmpdir, "elu_fallback", model)
        out_dir = os.path.join(tmpdir, "out")
        _convert(
            model_path,
            out_dir,
            "flatbuffer_direct",
            flatbuffer_direct_fallback_to_tf_converter=True,
            report_op_coverage=True,
        )

        assert os.path.isfile(os.path.join(out_dir, "elu_fallback_float32.tflite"))
        assert os.path.isfile(os.path.join(out_dir, "elu_fallback_float16.tflite"))
        report_path = os.path.join(out_dir, "elu_fallback_op_coverage_report.json")
        assert os.path.isfile(report_path)
        with open(report_path, "r", encoding="utf-8") as f:
            report = json.load(f)
        assert report["conversion_error"] is None
        assert report["graph_summary"]["unsupported_nodes"] == 0


@pytest.mark.skipif(not _requires_flatbuffer_tools(), reason="flatbuffer_direct requires flatc and curl")
def test_flatbuffer_direct_integration_quant_eval_coverage_smoke() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        model = _make_gemm_model()
        model_path = _save_model(tmpdir, "gemm_integ", model)
        out_dir = os.path.join(tmpdir, "out")
        _convert(
            model_path,
            out_dir,
            "flatbuffer_direct",
            output_dynamic_range_quantized_tflite=True,
            output_integer_quantized_tflite=True,
            eval_with_onnx=True,
            eval_num_samples=2,
            eval_target_tflite="full_integer_quant",
            eval_compare_mode="dequant",
            report_op_coverage=True,
        )

        expected_files = [
            "gemm_integ_float32.tflite",
            "gemm_integ_float16.tflite",
            "gemm_integ_dynamic_range_quant.tflite",
            "gemm_integ_integer_quant.tflite",
            "gemm_integ_full_integer_quant.tflite",
            "gemm_integ_accuracy_report.json",
            "gemm_integ_op_coverage_report.json",
        ]
        for name in expected_files:
            assert os.path.isfile(os.path.join(out_dir, name))

        with open(os.path.join(out_dir, "gemm_integ_accuracy_report.json"), "r", encoding="utf-8") as f:
            acc = json.load(f)
        assert acc["num_samples"] == 2

        with open(os.path.join(out_dir, "gemm_integ_op_coverage_report.json"), "r", encoding="utf-8") as f:
            cov = json.load(f)
        assert "schema_policy_counts" in cov
        assert cov["schema_unresolved_ops"] == []


@pytest.mark.skipif(not _requires_flatbuffer_tools(), reason="flatbuffer_direct requires flatc and curl")
def test_flatbuffer_direct_integration_split_eval_coverage_smoke() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        model = _make_add_chain_model()
        model_path = _save_model(tmpdir, "add_chain_integ", model)
        out_dir = os.path.join(tmpdir, "out")
        _convert(
            model_path,
            out_dir,
            "flatbuffer_direct",
            auto_split_tflite_by_size=True,
            tflite_split_max_bytes=10_000_000,
            tflite_split_target_bytes=1,
            eval_split_models=True,
            eval_num_samples=2,
            report_op_coverage=True,
        )

        expected_files = [
            "add_chain_integ_split_plan.json",
            "add_chain_integ_split_manifest.json",
            "add_chain_integ_split_accuracy_report.json",
            "add_chain_integ_op_coverage_report.json",
        ]
        for name in expected_files:
            assert os.path.isfile(os.path.join(out_dir, name))

        with open(os.path.join(out_dir, "add_chain_integ_split_accuracy_report.json"), "r", encoding="utf-8") as f:
            split_report = json.load(f)
        assert split_report["reference_mode"] == "unsplit_tflite"


@pytest.mark.skipif(not _requires_flatbuffer_tools(), reason="flatbuffer_direct requires flatc and curl")
def test_flatbuffer_direct_gather_int32_dtype_boundary() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        model = _make_gather_int32_model()
        model_path = _save_model(tmpdir, "gather_int32", model)
        out_dir = os.path.join(tmpdir, "out")
        tflite_path = _convert(model_path, out_dir, "flatbuffer_direct")

        interpreter = Interpreter(model_path=tflite_path)
        interpreter.allocate_tensors()
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        x = np.array(
            [
                [10, 20, 30, 40],
                [1, 2, 3, 4],
            ],
            dtype=np.int32,
        )
        interpreter.set_tensor(input_details[0]["index"], x)
        interpreter.invoke()
        y = interpreter.get_tensor(output_details[0]["index"])
        np.testing.assert_array_equal(
            y,
            np.array(
                [
                    [40, 20],
                    [4, 2],
                ],
                dtype=np.int32,
            ),
        )


@pytest.mark.skipif(not _requires_flatbuffer_tools(), reason="flatbuffer_direct requires flatc and curl")
def test_flatbuffer_direct_dynamic_range_quantized_smoke() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        model = _make_gemm_model()
        model_path = _save_model(tmpdir, "gemm_dq", model)
        out_dir = os.path.join(tmpdir, "out")
        _convert(
            model_path,
            out_dir,
            "flatbuffer_direct",
            output_dynamic_range_quantized_tflite=True,
        )
        tflite_path = os.path.join(out_dir, "gemm_dq_dynamic_range_quant.tflite")
        assert os.path.isfile(tflite_path)

        interpreter = Interpreter(model_path=tflite_path)
        interpreter.allocate_tensors()
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        x = np.array([[1.0, 2.0, 3.0, 4.0]], dtype=np.float32)
        interpreter.set_tensor(input_details[0]["index"], x)
        interpreter.invoke()
        y = interpreter.get_tensor(output_details[0]["index"])
        assert y.shape == (1, 3)


@pytest.mark.skipif(not _requires_flatbuffer_tools(), reason="flatbuffer_direct requires flatc and curl")
def test_flatbuffer_direct_dynamic_range_quantized_add_const_smoke() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        model = _make_add_const_model()
        model_path = _save_model(tmpdir, "add_const_dq", model)
        out_dir = os.path.join(tmpdir, "out")
        _convert(
            model_path,
            out_dir,
            "flatbuffer_direct",
            output_dynamic_range_quantized_tflite=True,
        )
        tflite_path = os.path.join(out_dir, "add_const_dq_dynamic_range_quant.tflite")
        assert os.path.isfile(tflite_path)

        interpreter = Interpreter(model_path=tflite_path)
        interpreter.allocate_tensors()
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        x = np.array([[1.0, 1.0, 1.0]], dtype=np.float32)
        interpreter.set_tensor(input_details[0]["index"], x)
        interpreter.invoke()
        y = interpreter.get_tensor(output_details[0]["index"])
        np.testing.assert_allclose(
            y,
            np.array([[2.0, 3.0, 4.0]], dtype=np.float32),
            rtol=0.0,
            atol=5e-2,
        )


@pytest.mark.skipif(not _requires_flatbuffer_tools(), reason="flatbuffer_direct requires flatc and curl")
@pytest.mark.parametrize(
    "quant_type, expected_multi_scale",
    [
        ("per-channel", True),
        ("per-tensor", False),
    ],
)
def test_flatbuffer_direct_dynamic_range_quantized_fc_quant_type(
    quant_type: str,
    expected_multi_scale: bool,
) -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        model = _make_gemm_model()
        model_path = _save_model(tmpdir, f"gemm_{quant_type}", model)
        out_dir = os.path.join(tmpdir, "out")
        _convert(
            model_path,
            out_dir,
            "flatbuffer_direct",
            output_dynamic_range_quantized_tflite=True,
            quant_type=quant_type,
        )
        tflite_path = os.path.join(
            out_dir,
            f"gemm_{quant_type}_dynamic_range_quant.tflite",
        )
        assert os.path.isfile(tflite_path)

        scale_lengths = _collect_int8_quant_scale_lengths(tflite_path)
        assert len(scale_lengths) > 0
        if expected_multi_scale:
            assert any(length > 1 for length in scale_lengths)
        else:
            assert all(length == 1 for length in scale_lengths)

        interpreter = Interpreter(model_path=tflite_path)
        interpreter.allocate_tensors()


@pytest.mark.skipif(not _requires_flatbuffer_tools(), reason="flatbuffer_direct requires flatc and curl")
def test_flatbuffer_direct_integer_quantized_smoke() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        model = _make_gemm_model()
        model_path = _save_model(tmpdir, "gemm_iq", model)
        out_dir = os.path.join(tmpdir, "out")
        _convert(
            model_path,
            out_dir,
            "flatbuffer_direct",
            output_integer_quantized_tflite=True,
            quant_type="per-channel",
            input_quant_dtype="int8",
            output_quant_dtype="int8",
        )

        integer_tflite = os.path.join(out_dir, "gemm_iq_integer_quant.tflite")
        full_integer_tflite = os.path.join(out_dir, "gemm_iq_full_integer_quant.tflite")
        integer_i16_tflite = os.path.join(out_dir, "gemm_iq_integer_quant_with_int16_act.tflite")
        full_integer_i16_tflite = os.path.join(out_dir, "gemm_iq_full_integer_quant_with_int16_act.tflite")
        assert os.path.isfile(integer_tflite)
        assert os.path.isfile(full_integer_tflite)
        assert os.path.isfile(integer_i16_tflite)
        assert os.path.isfile(full_integer_i16_tflite)

        # integer_quant: float input/output path
        interpreter = Interpreter(model_path=integer_tflite)
        interpreter.allocate_tensors()
        in_details = interpreter.get_input_details()
        out_details = interpreter.get_output_details()
        x = np.array([[1.0, 2.0, 3.0, 4.0]], dtype=np.float32)
        interpreter.set_tensor(in_details[0]["index"], x)
        interpreter.invoke()
        y = interpreter.get_tensor(out_details[0]["index"])
        assert y.shape == (1, 3)

        # full_integer_quant: quantized io path
        interpreter2 = Interpreter(model_path=full_integer_tflite)
        interpreter2.allocate_tensors()
        in2 = interpreter2.get_input_details()
        out2 = interpreter2.get_output_details()
        assert in2[0]["dtype"] == np.int8
        assert out2[0]["dtype"] == np.int8
        xq = np.zeros(in2[0]["shape"], dtype=np.int8)
        interpreter2.set_tensor(in2[0]["index"], xq)
        interpreter2.invoke()
        yq = interpreter2.get_tensor(out2[0]["index"])
        assert yq.dtype == np.int8

        # integer_quant_with_int16_act: float input/output path
        interpreter3 = Interpreter(model_path=integer_i16_tflite)
        interpreter3.allocate_tensors()
        in3 = interpreter3.get_input_details()
        out3 = interpreter3.get_output_details()
        x3 = np.array([[1.0, 2.0, 3.0, 4.0]], dtype=np.float32)
        interpreter3.set_tensor(in3[0]["index"], x3)
        interpreter3.invoke()
        y3 = interpreter3.get_tensor(out3[0]["index"])
        assert y3.shape == (1, 3)

        # full_integer_quant_with_int16_act: int16 input/output path
        interpreter4 = Interpreter(model_path=full_integer_i16_tflite)
        interpreter4.allocate_tensors()
        in4 = interpreter4.get_input_details()
        out4 = interpreter4.get_output_details()
        assert in4[0]["dtype"] == np.int16
        assert out4[0]["dtype"] == np.int16
        x4 = np.zeros(in4[0]["shape"], dtype=np.int16)
        interpreter4.set_tensor(in4[0]["index"], x4)
        interpreter4.invoke()
        y4 = interpreter4.get_tensor(out4[0]["index"])
        assert y4.dtype == np.int16


@pytest.mark.skipif(not _requires_flatbuffer_tools(), reason="flatbuffer_direct requires flatc and curl")
def test_flatbuffer_direct_integer_quantized_reduce_compatibility() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        model = _make_gemm_reduce_model()
        model_path = _save_model(tmpdir, "gemm_reduce_iq", model)
        out_dir = os.path.join(tmpdir, "out")
        _convert(
            model_path,
            out_dir,
            "flatbuffer_direct",
            output_integer_quantized_tflite=True,
            quant_type="per-channel",
        )
        tflite_path = os.path.join(out_dir, "gemm_reduce_iq_integer_quant.tflite")
        assert os.path.isfile(tflite_path)

        interpreter = Interpreter(model_path=tflite_path)
        interpreter.allocate_tensors()
        in_details = interpreter.get_input_details()
        out_details = interpreter.get_output_details()
        x = np.array([[1.0, 2.0, 3.0, 4.0]], dtype=np.float32)
        interpreter.set_tensor(in_details[0]["index"], x)
        interpreter.invoke()
        y = interpreter.get_tensor(out_details[0]["index"])
        assert y.shape == (1, 1)


@pytest.mark.skipif(not _requires_flatbuffer_tools(), reason="flatbuffer_direct requires flatc and curl")
def test_flatbuffer_direct_dynamic_range_percentile_calibration_smoke() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        model = _make_gemm_model()
        model_path = _save_model(tmpdir, "gemm_pct", model)
        out_dir = os.path.join(tmpdir, "out")
        with _temporary_env(
            {
                "ONNX2TF_FLATBUFFER_DIRECT_CALIBRATION_METHOD": "percentile",
                "ONNX2TF_FLATBUFFER_DIRECT_CALIBRATION_PERCENTILE": "99.0",
            }
        ):
            _convert(
                model_path,
                out_dir,
                "flatbuffer_direct",
                output_dynamic_range_quantized_tflite=True,
            )
        tflite_path = os.path.join(out_dir, "gemm_pct_dynamic_range_quant.tflite")
        assert os.path.isfile(tflite_path)


@pytest.mark.skipif(not _requires_flatbuffer_tools(), reason="flatbuffer_direct requires flatc and curl")
def test_flatbuffer_direct_dynamic_range_threshold_control() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        model = _make_gemm_model()
        model_path = _save_model(tmpdir, "gemm_th", model)
        out_dir = os.path.join(tmpdir, "out")
        with _temporary_env(
            {
                "ONNX2TF_FLATBUFFER_DIRECT_QUANT_MIN_ABS_MAX": "1000.0",
            }
        ):
            with pytest.raises(NotImplementedError):
                _convert(
                    model_path,
                    out_dir,
                    "flatbuffer_direct",
                    output_dynamic_range_quantized_tflite=True,
                )


@pytest.mark.skipif(not _requires_flatbuffer_tools(), reason="flatbuffer_direct requires flatc and curl")
def test_flatbuffer_direct_accuracy_report_generation() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        model = _make_add_model()
        model_path = _save_model(tmpdir, "add_eval", model)
        out_dir = os.path.join(tmpdir, "out")
        _convert(
            model_path,
            out_dir,
            "flatbuffer_direct",
            eval_with_onnx=True,
            eval_num_samples=3,
        )

        report_path = os.path.join(out_dir, "add_eval_accuracy_report.json")
        assert os.path.isfile(report_path)

        with open(report_path, "r", encoding="utf-8") as f:
            report = json.load(f)

        assert report["schema_version"] == 1
        assert report["num_samples"] == 3
        assert report["seed"] == 0
        assert report["inputs_source"] == "seeded_random"
        assert report["compare_mode"] == "raw"
        assert report["evaluation_pass"] is True
        assert report["allclose_summary"]["pass"] is True
        assert "overall_metrics" in report
        assert "per_output_metrics" in report
        assert "z" in report["per_output_metrics"]
        assert report["overall_metrics"]["max_abs"] <= 1e-6


@pytest.mark.skipif(not _requires_flatbuffer_tools(), reason="flatbuffer_direct requires flatc and curl")
def test_flatbuffer_direct_accuracy_report_quant_dequant_mode() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        model = _make_gemm_model()
        model_path = _save_model(tmpdir, "gemm_eval_q", model)
        out_dir = os.path.join(tmpdir, "out")
        _convert(
            model_path,
            out_dir,
            "flatbuffer_direct",
            output_integer_quantized_tflite=True,
            eval_with_onnx=True,
            eval_num_samples=3,
            eval_target_tflite="full_integer_quant",
            eval_compare_mode="dequant",
        )

        report_path = os.path.join(out_dir, "gemm_eval_q_accuracy_report.json")
        assert os.path.isfile(report_path)
        with open(report_path, "r", encoding="utf-8") as f:
            report = json.load(f)
        assert report["compare_mode"] == "dequant"
        assert report["has_quantized_outputs"] is True
        assert "metric_threshold_judgement" in report


@pytest.mark.skipif(not _requires_flatbuffer_tools(), reason="flatbuffer_direct requires flatc and curl")
def test_flatbuffer_direct_accuracy_report_fail_on_threshold() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        model = _make_gemm_model()
        model_path = _save_model(tmpdir, "gemm_eval_fail", model)
        out_dir = os.path.join(tmpdir, "out")
        with pytest.raises(RuntimeError):
            _convert(
                model_path,
                out_dir,
                "flatbuffer_direct",
                output_integer_quantized_tflite=True,
                eval_with_onnx=True,
                eval_num_samples=2,
                eval_target_tflite="full_integer_quant",
                eval_compare_mode="raw",
                eval_fail_on_threshold=True,
            )
        report_path = os.path.join(out_dir, "gemm_eval_fail_accuracy_report.json")
        assert os.path.isfile(report_path)
        with open(report_path, "r", encoding="utf-8") as f:
            report = json.load(f)
        assert report["evaluation_pass"] is False


@pytest.mark.skipif(not _requires_flatbuffer_tools(), reason="flatbuffer_direct requires flatc and curl")
def test_flatbuffer_direct_split_plan_report_smoke() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        model = _make_gemm_model()
        model_path = _save_model(tmpdir, "gemm_split_plan", model)
        out_dir = os.path.join(tmpdir, "out")
        _convert(
            model_path,
            out_dir,
            "flatbuffer_direct",
            auto_split_tflite_by_size=True,
            tflite_split_max_bytes=10_000_000,
            tflite_split_target_bytes=9_000_000,
        )
        report_path = os.path.join(out_dir, "gemm_split_plan_split_plan.json")
        assert os.path.isfile(report_path)
        with open(report_path, "r", encoding="utf-8") as f:
            report = json.load(f)
        assert report["schema_version"] == 1
        assert report["plan_valid"] is True
        assert report["total_estimated_bytes"] > 0
        assert len(report["partitions"]) >= 1


def test_parse_auto_split_size_to_bytes_units() -> None:
    from onnx2tf.onnx2tf import _parse_size_to_bytes

    assert _parse_size_to_bytes("1KB", param_name="x") == 1_024
    assert _parse_size_to_bytes("2mb", param_name="x") == 2 * 1_048_576
    assert _parse_size_to_bytes("1.5GB", param_name="x") == int(1.5 * 1_073_741_824)
    assert _parse_size_to_bytes("256", default_unit="MB", param_name="x") == 256 * 1_048_576


@pytest.mark.skipif(not _requires_flatbuffer_tools(), reason="flatbuffer_direct requires flatc and curl")
def test_flatbuffer_direct_split_plan_uses_auto_split_max_size_when_specified() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        model = _make_gemm_model()
        model_path = _save_model(tmpdir, "gemm_split_size_opt", model)
        out_dir = os.path.join(tmpdir, "out")
        _convert(
            model_path,
            out_dir,
            "flatbuffer_direct",
            auto_split_tflite_by_size=True,
            auto_split_max_size="256KB",
            tflite_split_max_bytes=10_000_000,
            tflite_split_target_bytes=9_000_000,
        )
        report_path = os.path.join(out_dir, "gemm_split_size_opt_split_plan.json")
        assert os.path.isfile(report_path)
        with open(report_path, "r", encoding="utf-8") as f:
            report = json.load(f)
        assert report["target_max_bytes"] == 256 * 1024
        assert report["hard_max_bytes"] >= report["target_max_bytes"]


@pytest.mark.skipif(not _requires_flatbuffer_tools(), reason="flatbuffer_direct requires flatc and curl")
def test_flatbuffer_direct_split_manifest_and_partition_outputs() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        model = _make_add_chain_model()
        model_path = _save_model(tmpdir, "add_chain_split", model)
        out_dir = os.path.join(tmpdir, "out")
        _convert(
            model_path,
            out_dir,
            "flatbuffer_direct",
            auto_split_tflite_by_size=True,
            tflite_split_max_bytes=10_000_000,
            tflite_split_target_bytes=1,
        )

        manifest_path = os.path.join(out_dir, "add_chain_split_split_manifest.json")
        assert os.path.isfile(manifest_path)
        with open(manifest_path, "r", encoding="utf-8") as f:
            manifest = json.load(f)
        assert manifest["schema_version"] == 1
        assert manifest["base_model"] == "add_chain_split.tflite"
        assert manifest["target_max_bytes"] == 1
        assert len(manifest["partitions"]) >= 1
        assert "edges" in manifest

        part_files = sorted(
            glob.glob(os.path.join(out_dir, "add_chain_split_[0-9][0-9][0-9][0-9].tflite"))
        )
        assert len(part_files) >= 1
        for part_file in part_files:
            interpreter = Interpreter(model_path=part_file)
            interpreter.allocate_tensors()


@pytest.mark.skipif(not _requires_flatbuffer_tools(), reason="flatbuffer_direct requires flatc and curl")
def test_flatbuffer_direct_split_accuracy_report_with_unsplit_reference() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        model = _make_add_chain_model()
        model_path = _save_model(tmpdir, "add_chain_eval_split", model)
        out_dir = os.path.join(tmpdir, "out")
        _convert(
            model_path,
            out_dir,
            "flatbuffer_direct",
            auto_split_tflite_by_size=True,
            tflite_split_max_bytes=10_000_000,
            tflite_split_target_bytes=1,
            eval_split_models=True,
            eval_split_reference="unsplit_tflite",
            eval_num_samples=3,
        )
        report_path = os.path.join(out_dir, "add_chain_eval_split_split_accuracy_report.json")
        assert os.path.isfile(report_path)
        with open(report_path, "r", encoding="utf-8") as f:
            report = json.load(f)
        assert report["reference_mode"] == "unsplit_tflite"
        assert report["evaluation_pass"] is True
        assert report["allclose_summary"]["pass"] is True


@pytest.mark.skipif(not _requires_flatbuffer_tools(), reason="flatbuffer_direct requires flatc and curl")
def test_split_accuracy_report_fail_on_threshold() -> None:
    from onnx2tf.tflite_builder.split_accuracy_evaluator import evaluate_split_manifest_outputs

    with tempfile.TemporaryDirectory() as tmpdir:
        model = _make_add_chain_model()
        model_path = _save_model(tmpdir, "add_chain_eval_split_fail", model)
        out_dir = os.path.join(tmpdir, "out")
        _convert(
            model_path,
            out_dir,
            "flatbuffer_direct",
            auto_split_tflite_by_size=True,
            tflite_split_max_bytes=10_000_000,
            tflite_split_target_bytes=1,
        )
        split_manifest_path = os.path.join(out_dir, "add_chain_eval_split_fail_split_manifest.json")
        reference_tflite_path = os.path.join(out_dir, "add_chain_eval_split_fail_float32.tflite")
        onnx_graph = onnx.load(model_path)
        report_path = os.path.join(out_dir, "add_chain_eval_split_fail_split_accuracy_report.json")
        with pytest.raises(RuntimeError):
            evaluate_split_manifest_outputs(
                onnx_graph=onnx_graph,
                split_manifest_path=split_manifest_path,
                reference_mode="unsplit_tflite",
                reference_tflite_path=reference_tflite_path,
                output_report_path=report_path,
                num_samples=2,
                fail_on_threshold=True,
                metric_thresholds={
                    "max_abs": 0.0,
                    "mean_abs": 0.0,
                    "rmse": 0.0,
                    "cosine_similarity": 1.1,
                },
            )
        assert os.path.isfile(report_path)
        with open(report_path, "r", encoding="utf-8") as f:
            report = json.load(f)
        assert report["evaluation_pass"] is False


@pytest.mark.skipif(not _requires_flatbuffer_tools(), reason="flatbuffer_direct requires flatc and curl")
def test_flatbuffer_direct_op_coverage_report_generation() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        model = _make_add_model()
        model_path = _save_model(tmpdir, "add_cov", model)
        out_dir = os.path.join(tmpdir, "out")
        _convert(
            model_path,
            out_dir,
            "flatbuffer_direct",
            report_op_coverage=True,
        )
        report_path = os.path.join(out_dir, "add_cov_op_coverage_report.json")
        assert os.path.isfile(report_path)
        with open(report_path, "r", encoding="utf-8") as f:
            report = json.load(f)
        assert report["schema_version"] == 1
        assert report["conversion_error"] is None
        assert "Add" in report["graph_ops"]
        assert report["graph_summary"]["unsupported_nodes"] == 0
        assert report["graph_summary"]["coverage_ratio"] == 1.0
        assert "preprocess_report" in report
        assert report["preprocess_report"]["schema_version"] == 1
        assert report["preprocess_report"]["summary"]["executed_rule_count"] >= 0


@pytest.mark.skipif(not _requires_flatbuffer_tools(), reason="flatbuffer_direct requires flatc and curl")
def test_flatbuffer_direct_where_neg_inf_broadcast_no_nan() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        model = _make_where_neg_inf_broadcast_model()
        model_path = _save_model(tmpdir, "where_neg_inf_broadcast", model)
        out_dir = os.path.join(tmpdir, "out")
        tflite_path = _convert(
            model_path,
            out_dir,
            "flatbuffer_direct",
        )
        interpreter = Interpreter(model_path=tflite_path)
        interpreter.allocate_tensors()
        in_details = interpreter.get_input_details()
        out_details = interpreter.get_output_details()
        by_name = {str(v["name"]): v for v in in_details}

        cond_shape = tuple(int(v) for v in by_name["cond"]["shape"])
        y_shape = tuple(int(v) for v in by_name["y"]["shape"])
        cond = np.zeros(cond_shape, dtype=np.bool_)
        y = np.linspace(
            -1.0,
            1.0,
            num=int(np.prod(y_shape)),
            dtype=np.float32,
        ).reshape(y_shape)

        interpreter.set_tensor(by_name["cond"]["index"], cond)
        interpreter.set_tensor(by_name["y"]["index"], y)
        interpreter.invoke()
        out = interpreter.get_tensor(out_details[0]["index"])

        assert bool(np.isfinite(out).all())
        expected = y
        if tuple(expected.shape) != tuple(out.shape):
            if (
                expected.ndim == 4
                and out.ndim == 4
                and tuple(out.shape)
                == (
                    int(expected.shape[0]),
                    int(expected.shape[1]),
                    int(expected.shape[3]),
                    int(expected.shape[2]),
                )
            ):
                expected = np.transpose(expected, (0, 1, 3, 2))
            else:
                raise AssertionError(
                    f"Unexpected output shape. out={tuple(out.shape)} expected={tuple(expected.shape)}"
                )
        np.testing.assert_allclose(out, expected, rtol=0.0, atol=0.0)


@pytest.mark.skipif(not _requires_flatbuffer_tools(), reason="flatbuffer_direct requires flatc and curl")
def test_flatbuffer_direct_op_coverage_report_on_unsupported_op() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        model = _make_elu_model()
        model_path = _save_model(tmpdir, "elu_cov", model)
        out_dir = os.path.join(tmpdir, "out")
        _convert(
            model_path,
            out_dir,
            "flatbuffer_direct",
            report_op_coverage=True,
        )
        report_path = os.path.join(out_dir, "elu_cov_op_coverage_report.json")
        assert os.path.isfile(report_path)
        with open(report_path, "r", encoding="utf-8") as f:
            report = json.load(f)
        assert report["conversion_error"] is None
        assert report["graph_summary"]["unsupported_nodes"] == 0
        assert any(
            node["onnx_op"] == "Elu" and node["dispatch_mode"] == "builtin"
            for node in report["graph_node_reports"]
        )


@pytest.mark.skipif(not _requires_flatbuffer_tools(), reason="flatbuffer_direct requires flatc and curl")
def test_flatbuffer_direct_erf_tile_scatternd_builtin_smoke() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        model = _make_erf_tile_scatternd_model()
        model_path = _save_model(tmpdir, "erf_tile_scatternd", model)
        out_dir = os.path.join(tmpdir, "out")
        tflite_path = _convert(
            model_path,
            out_dir,
            "flatbuffer_direct",
            report_op_coverage=True,
        )
        assert os.path.isfile(tflite_path)

        report_path = os.path.join(out_dir, "erf_tile_scatternd_op_coverage_report.json")
        with open(report_path, "r", encoding="utf-8") as f:
            report = json.load(f)
        assert report["graph_custom_ops"] == []
        assert report["graph_summary"]["custom_lowered_nodes"] == 0
        assert report["graph_summary"]["unsupported_nodes"] == 0

        interpreter = Interpreter(model_path=tflite_path)
        interpreter.allocate_tensors()
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        by_name = {d["name"]: d for d in input_details}

        x = np.asarray(
            [
                [-0.5, 0.0, 0.5],
                [1.0, -1.0, 0.25],
            ],
            dtype=np.float32,
        )
        updates = np.asarray([10.0, -20.0], dtype=np.float32)
        interpreter.set_tensor(by_name["x"]["index"], x)
        interpreter.set_tensor(by_name["updates"]["index"], updates)
        interpreter.invoke()
        y = interpreter.get_tensor(output_details[0]["index"])

        expected = np.tile(np.vectorize(math.erf)(x).astype(np.float32), (1, 2))
        expected[0, 1] = updates[0]
        expected[1, 4] = updates[1]
        np.testing.assert_allclose(y, expected, rtol=1e-4, atol=1e-4)


def _make_flatten_dynamic_first_dim_model() -> onnx.ModelProto:
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, ["N", 1, 4])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, ["N", 4])
    node = helper.make_node("Flatten", ["x"], ["y"], name="FlattenDynamicNode", axis=1)
    graph = helper.make_graph([node], "flatten_dynamic_graph", [x], [y])
    return helper.make_model(graph, opset_imports=[helper.make_operatorsetid("", 13)])


def test_flatbuffer_direct_flatten_preserves_dynamic_dim_in_reshape_shape() -> None:
    model_ir = lower_onnx_to_ir(
        _make_flatten_dynamic_first_dim_model(),
        output_file_name="flatten_dynamic_dim",
    )
    reshape_ops = [op for op in model_ir.operators if str(op.op_type) == "RESHAPE"]
    assert len(reshape_ops) == 1
    reshape_op = reshape_ops[0]
    assert list(reshape_op.options.get("newShape", [])) == [-1, 4]

    shape_tensor_name = str(reshape_op.inputs[1])
    shape_tensor = model_ir.tensors[shape_tensor_name]
    np.testing.assert_array_equal(
        np.asarray(shape_tensor.data, dtype=np.int32).reshape(-1),
        np.asarray([-1, 4], dtype=np.int32),
    )


def _make_loop_static_unroll_model() -> onnx.ModelProto:
    x = helper.make_tensor_value_info("loop_x", TensorProto.FLOAT, [3])
    y = helper.make_tensor_value_info("loop_y", TensorProto.FLOAT, [3])

    trip_count = numpy_helper.from_array(np.asarray(8, dtype=np.int64), name="loop_trip_count")
    cond_init = numpy_helper.from_array(np.asarray(True, dtype=np.bool_), name="loop_cond_init")

    body_iter = helper.make_tensor_value_info("i", TensorProto.INT64, [])
    body_cond = helper.make_tensor_value_info("cond", TensorProto.BOOL, [])
    body_state = helper.make_tensor_value_info("state_in", TensorProto.FLOAT, [3])
    body_cond_out = helper.make_tensor_value_info("cond_out", TensorProto.BOOL, [])
    body_state_out = helper.make_tensor_value_info("state_out", TensorProto.FLOAT, [3])
    body_nodes = [
        helper.make_node("Identity", ["cond"], ["cond_out"], name="LoopBodyCondIdentity"),
        helper.make_node("Sigmoid", ["state_in"], ["state_out"], name="LoopBodySigmoid"),
    ]
    body_graph = helper.make_graph(
        body_nodes,
        "loop_body",
        [body_iter, body_cond, body_state],
        [body_cond_out, body_state_out],
    )

    loop_node = helper.make_node(
        "Loop",
        ["loop_trip_count", "loop_cond_init", "loop_x"],
        ["loop_y"],
        name="LoopStaticUnroll",
        body=body_graph,
    )
    graph = helper.make_graph(
        [loop_node],
        "loop_static_unroll_graph",
        [x],
        [y],
        initializer=[trip_count, cond_init],
    )
    return helper.make_model(graph, opset_imports=[helper.make_operatorsetid("", 16)])


def _make_loop_while_model() -> onnx.ModelProto:
    x = helper.make_tensor_value_info("loop_x", TensorProto.FLOAT, [3])
    counter = helper.make_tensor_value_info("loop_counter", TensorProto.INT64, [])
    y = helper.make_tensor_value_info("loop_y", TensorProto.FLOAT, [3])

    max_trip = numpy_helper.from_array(np.asarray(9223372036854775807, dtype=np.int64), name="loop_max_trip")
    cond_limit = numpy_helper.from_array(np.asarray(64, dtype=np.int64), name="loop_cond_limit")

    cond_node = helper.make_node("Less", ["loop_counter", "loop_cond_limit"], ["loop_cond"], name="LoopTopCond")

    body_iter = helper.make_tensor_value_info("iter", TensorProto.INT64, [])
    body_cond = helper.make_tensor_value_info("cond", TensorProto.BOOL, [])
    body_state = helper.make_tensor_value_info("state_in", TensorProto.FLOAT, [3])
    body_counter = helper.make_tensor_value_info("counter_in", TensorProto.INT64, [])
    body_cond_out = helper.make_tensor_value_info("cond_out", TensorProto.BOOL, [])
    body_state_out = helper.make_tensor_value_info("state_out", TensorProto.FLOAT, [3])
    body_counter_out = helper.make_tensor_value_info("counter_out", TensorProto.INT64, [])
    body_counter_one = numpy_helper.from_array(np.asarray(1, dtype=np.int64), name="body_counter_one")
    body_counter_limit = numpy_helper.from_array(np.asarray(64, dtype=np.int64), name="body_counter_limit")
    body_nodes = [
        helper.make_node("Sigmoid", ["state_in"], ["state_out"], name="LoopBodySigmoid"),
        helper.make_node("Add", ["counter_in", "body_counter_one"], ["counter_out"], name="LoopBodyCounterAdd"),
        helper.make_node("Less", ["counter_out", "body_counter_limit"], ["cond_out"], name="LoopBodyLess"),
    ]
    body_graph = helper.make_graph(
        body_nodes,
        "loop_while_body",
        [body_iter, body_cond, body_state, body_counter],
        [body_cond_out, body_state_out, body_counter_out],
        initializer=[body_counter_one, body_counter_limit],
    )

    loop_node = helper.make_node(
        "Loop",
        ["loop_max_trip", "loop_cond", "loop_x", "loop_counter"],
        ["loop_y", "loop_counter_out"],
        name="LoopWhileNode",
        body=body_graph,
    )
    graph = helper.make_graph(
        [cond_node, loop_node],
        "loop_while_graph",
        [x, counter],
        [y],
        initializer=[max_trip, cond_limit],
    )
    return helper.make_model(graph, opset_imports=[helper.make_operatorsetid("", 16)])


def test_flatbuffer_direct_loop_static_unroll_lowers_to_builtin_ops_without_custom() -> None:
    model_ir = lower_onnx_to_ir(
        _make_loop_static_unroll_model(),
        output_file_name="loop_static_unroll_builtin",
    )
    op_types = [str(op.op_type) for op in model_ir.operators]
    assert "CUSTOM" not in op_types
    assert "WHILE" in op_types
    assert len(model_ir.subgraphs) == 2
    out_tensor = model_ir.tensors["loop_y"]
    assert list(out_tensor.shape) == [3]


def test_flatbuffer_direct_loop_while_lowers_to_builtin_ops_without_custom() -> None:
    model_ir = lower_onnx_to_ir(
        _make_loop_while_model(),
        output_file_name="loop_while_builtin",
    )
    op_types = [str(op.op_type) for op in model_ir.operators]
    assert "CUSTOM" not in op_types
    assert "WHILE" in op_types
    assert len(model_ir.subgraphs) == 2
    assert len(model_ir.subgraphs[0].operators) > 0
    assert len(model_ir.subgraphs[1].operators) > 0


@pytest.mark.skipif(not _requires_flatbuffer_tools(), reason="flatbuffer_direct requires flatc and curl")
def test_flatbuffer_direct_loop_static_unroll_op_coverage_reports_builtin_loop() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        model = _make_loop_static_unroll_model()
        model_path = _save_model(tmpdir, "loop_static_unroll", model)
        out_dir = os.path.join(tmpdir, "out")
        tflite_path = _convert(
            model_path=model_path,
            output_dir=out_dir,
            backend="flatbuffer_direct",
            report_op_coverage=True,
        )
        assert os.path.isfile(tflite_path)

        report_path = os.path.join(out_dir, "loop_static_unroll_op_coverage_report.json")
        assert os.path.isfile(report_path)
        with open(report_path, "r", encoding="utf-8") as f:
            report = json.load(f)
        assert report["graph_custom_ops"] == []
        loop_reports = [
            node
            for node in report["graph_node_reports"]
            if str(node.get("onnx_op", "")) == "Loop"
        ]
        assert len(loop_reports) == 1
        assert loop_reports[0]["dispatch_mode"] == "builtin"


@pytest.mark.skipif(not _requires_flatbuffer_tools(), reason="flatbuffer_direct requires flatc and curl")
def test_flatbuffer_direct_loop_while_op_coverage_reports_builtin_loop() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        model = _make_loop_while_model()
        model_path = _save_model(tmpdir, "loop_while", model)
        out_dir = os.path.join(tmpdir, "out")
        tflite_path = _convert(
            model_path=model_path,
            output_dir=out_dir,
            backend="flatbuffer_direct",
            report_op_coverage=True,
        )
        assert os.path.isfile(tflite_path)

        report_path = os.path.join(out_dir, "loop_while_op_coverage_report.json")
        assert os.path.isfile(report_path)
        with open(report_path, "r", encoding="utf-8") as f:
            report = json.load(f)
        assert report["graph_custom_ops"] == []
        loop_reports = [
            node
            for node in report["graph_node_reports"]
            if str(node.get("onnx_op", "")) == "Loop"
        ]
        assert len(loop_reports) == 1
        assert loop_reports[0]["dispatch_mode"] == "builtin"


def _make_if_nms_guard_direct_model() -> onnx.ModelProto:
    boxes = helper.make_tensor_value_info("boxes", TensorProto.FLOAT, ["N", 4])
    scores = helper.make_tensor_value_info("scores", TensorProto.FLOAT, ["N"])
    idxs = helper.make_tensor_value_info("idxs", TensorProto.INT64, ["N"])
    cond = helper.make_tensor_value_info("cond", TensorProto.BOOL, [])
    keep = helper.make_tensor_value_info("keep", TensorProto.INT64, ["K"])

    top_then_empty = numpy_helper.from_array(np.asarray([], dtype=np.int64), name="top_then_empty")
    top_then_out = helper.make_tensor_value_info("top_then_empty", TensorProto.INT64, [0])
    top_then_graph = helper.make_graph([], "if_top_then", [], [top_then_out], initializer=[top_then_empty])

    nested_then_axis = numpy_helper.from_array(np.asarray([1], dtype=np.int64), name="nested_then_axis")
    nested_then_node = helper.make_node(
        "Squeeze",
        ["nms_gathered", "nested_then_axis"],
        ["nested_then_out"],
        name="NestedThenSqueeze",
    )
    nested_then_out = helper.make_tensor_value_info("nested_then_out", TensorProto.INT64, ["K"])
    nested_then_graph = helper.make_graph(
        [nested_then_node],
        "if_nested_then",
        [],
        [nested_then_out],
        initializer=[nested_then_axis],
    )
    nested_else_node = helper.make_node(
        "Identity",
        ["nms_gathered"],
        ["nested_else_out"],
        name="NestedElseIdentity",
    )
    nested_else_out = helper.make_tensor_value_info("nested_else_out", TensorProto.INT64, ["K", 1])
    nested_else_graph = helper.make_graph([nested_else_node], "if_nested_else", [], [nested_else_out])

    eq_lhs = numpy_helper.from_array(np.asarray([1], dtype=np.int64), name="eq_lhs")
    eq_rhs = numpy_helper.from_array(np.asarray([1], dtype=np.int64), name="eq_rhs")
    unsq_scores_axes = numpy_helper.from_array(np.asarray([0, 1], dtype=np.int64), name="unsq_scores_axes")
    add_one = numpy_helper.from_array(np.asarray(1.0, dtype=np.float32), name="add_one")
    unsq_offsets_axes = numpy_helper.from_array(np.asarray([1], dtype=np.int64), name="unsq_offsets_axes")
    unsq_boxes_axes = numpy_helper.from_array(np.asarray([0], dtype=np.int64), name="unsq_boxes_axes")
    nms_max_output = numpy_helper.from_array(np.asarray([9223372036854775807], dtype=np.int64), name="nms_max_output")
    nms_iou = numpy_helper.from_array(np.asarray([0.5], dtype=np.float32), name="nms_iou")
    nms_gather_index = numpy_helper.from_array(np.asarray([2], dtype=np.int64), name="nms_gather_index")

    else_nodes = [
        helper.make_node("ReduceMax", ["boxes"], ["max_coordinate"], name="IfElseReduceMax", keepdims=0),
        helper.make_node("Cast", ["idxs"], ["idxs_f"], name="IfElseCast", to=TensorProto.FLOAT),
        helper.make_node("Equal", ["eq_lhs", "eq_rhs"], ["nested_cond"], name="IfElseEqual"),
        helper.make_node("Unsqueeze", ["scores", "unsq_scores_axes"], ["scores_nms"], name="IfElseUnsqueezeScores"),
        helper.make_node("Add", ["max_coordinate", "add_one"], ["max_plus_one"], name="IfElseAdd"),
        helper.make_node("Mul", ["idxs_f", "max_plus_one"], ["offsets"], name="IfElseMul"),
        helper.make_node("Unsqueeze", ["offsets", "unsq_offsets_axes"], ["offsets_2d"], name="IfElseUnsqueezeOffsets"),
        helper.make_node("Add", ["boxes", "offsets_2d"], ["boxes_for_nms"], name="IfElseAddBoxes"),
        helper.make_node("Unsqueeze", ["boxes_for_nms", "unsq_boxes_axes"], ["boxes_nms"], name="IfElseUnsqueezeBoxes"),
        helper.make_node(
            "NonMaxSuppression",
            ["boxes_nms", "scores_nms", "nms_max_output", "nms_iou"],
            ["nms_selected"],
            name="IfElseNMS",
        ),
        helper.make_node("Gather", ["nms_selected", "nms_gather_index"], ["nms_gathered"], name="IfElseGather", axis=1),
        helper.make_node(
            "If",
            ["nested_cond"],
            ["keep"],
            name="IfElseNestedIf",
            then_branch=nested_then_graph,
            else_branch=nested_else_graph,
        ),
    ]
    else_out = helper.make_tensor_value_info("keep", TensorProto.INT64, ["K"])
    top_else_graph = helper.make_graph(
        else_nodes,
        "if_top_else",
        [],
        [else_out],
        initializer=[
            eq_lhs,
            eq_rhs,
            unsq_scores_axes,
            add_one,
            unsq_offsets_axes,
            unsq_boxes_axes,
            nms_max_output,
            nms_iou,
            nms_gather_index,
        ],
    )

    top_if = helper.make_node(
        "If",
        ["cond"],
        ["keep"],
        name="IfNmsGuardNode",
        then_branch=top_then_graph,
        else_branch=top_else_graph,
    )
    graph = helper.make_graph([top_if], "if_nms_guard_direct_graph", [boxes, scores, idxs, cond], [keep])
    return helper.make_model(graph, opset_imports=[helper.make_operatorsetid("", 13)])


def _make_if_p1_model() -> onnx.ModelProto:
    x1 = helper.make_tensor_value_info("If_p1_input1", TensorProto.FLOAT, [1, 100])
    x2 = helper.make_tensor_value_info("If_p1_input2", TensorProto.FLOAT, [2, 100])
    y = helper.make_tensor_value_info("If_p1_output", TensorProto.FLOAT, ["N", 100])

    then_const = numpy_helper.from_array(np.asarray(1.0, dtype=np.float32), name="then_const")
    else_const = numpy_helper.from_array(np.asarray(2.0, dtype=np.float32), name="else_const")

    then_node = helper.make_node(
        "Add",
        ["If_p1_input1", "then_const"],
        ["then_out"],
        name="ThenAdd",
    )
    then_out = helper.make_tensor_value_info("then_out", TensorProto.FLOAT, [1, 100])
    then_graph = helper.make_graph(
        [then_node],
        "if_p1_then",
        [],
        [then_out],
        initializer=[then_const],
    )

    else_node = helper.make_node(
        "Add",
        ["If_p1_input2", "else_const"],
        ["else_out"],
        name="ElseAdd",
    )
    else_out = helper.make_tensor_value_info("else_out", TensorProto.FLOAT, [2, 100])
    else_graph = helper.make_graph(
        [else_node],
        "if_p1_else",
        [],
        [else_out],
        initializer=[else_const],
    )

    nodes = [
        helper.make_node("ReduceSum", ["If_p1_input1"], ["sum1"], name="Sum1", keepdims=0),
        helper.make_node("ReduceSum", ["If_p1_input2"], ["sum2"], name="Sum2", keepdims=0),
        helper.make_node("Greater", ["sum1", "sum2"], ["cond"], name="Cond"),
        helper.make_node(
            "If",
            ["cond"],
            ["If_p1_output"],
            name="If_p1",
            then_branch=then_graph,
            else_branch=else_graph,
        ),
    ]
    graph = helper.make_graph(nodes, "if_p1_graph", [x1, x2], [y])
    return helper.make_model(graph, opset_imports=[helper.make_operatorsetid("", 11)])


def _make_if_p2_model() -> onnx.ModelProto:
    x1 = helper.make_tensor_value_info("If_p2_input1", TensorProto.FLOAT, [1, 100])
    x2 = helper.make_tensor_value_info("If_p2_input2", TensorProto.FLOAT, [2, 100])
    y = helper.make_tensor_value_info("If_p2_output", TensorProto.FLOAT, None)

    then_const = numpy_helper.from_array(np.asarray(1.0, dtype=np.float32))
    then_const_node = helper.make_node(
        "Constant",
        [],
        ["then_const"],
        name="ThenConst",
        value=then_const,
    )
    then_add = helper.make_node(
        "Add",
        ["If_p2_input1", "then_const"],
        ["then_add_out"],
        name="ThenAdd",
    )
    then_seq = helper.make_node(
        "SequenceConstruct",
        ["then_add_out"],
        ["then_seq_out"],
        name="ThenSequenceConstruct",
    )
    then_out = helper.make_tensor_value_info("then_seq_out", TensorProto.FLOAT, None)
    then_graph = helper.make_graph(
        [then_const_node, then_add, then_seq],
        "if_p2_then",
        [],
        [then_out],
    )

    else_const1 = numpy_helper.from_array(np.asarray(3.0, dtype=np.float32))
    else_const2 = numpy_helper.from_array(np.asarray(4.0, dtype=np.float32))
    else_const1_node = helper.make_node(
        "Constant",
        [],
        ["else_const1"],
        name="ElseConst1",
        value=else_const1,
    )
    else_const2_node = helper.make_node(
        "Constant",
        [],
        ["else_const2"],
        name="ElseConst2",
        value=else_const2,
    )
    else_add1 = helper.make_node(
        "Add",
        ["If_p2_input1", "else_const1"],
        ["else_add1_out"],
        name="ElseAdd1",
    )
    else_add2 = helper.make_node(
        "Add",
        ["If_p2_input2", "else_const2"],
        ["else_add2_out"],
        name="ElseAdd2",
    )
    else_seq = helper.make_node(
        "SequenceConstruct",
        ["else_add1_out", "else_add2_out"],
        ["else_seq_out"],
        name="ElseSequenceConstruct",
    )
    else_out = helper.make_tensor_value_info("else_seq_out", TensorProto.FLOAT, None)
    else_graph = helper.make_graph(
        [else_const1_node, else_add1, else_const2_node, else_add2, else_seq],
        "if_p2_else",
        [],
        [else_out],
    )

    nodes = [
        helper.make_node("ReduceSum", ["If_p2_input1"], ["sum1"], name="Sum1", keepdims=0),
        helper.make_node("ReduceSum", ["If_p2_input2"], ["sum2"], name="Sum2", keepdims=0),
        helper.make_node("Greater", ["sum1", "sum2"], ["cond_bool"], name="Cond"),
        helper.make_node("Cast", ["cond_bool"], ["cond"], name="CondCast", to=TensorProto.BOOL),
        helper.make_node(
            "If",
            ["cond"],
            ["If_p2_output"],
            name="If_p2",
            then_branch=then_graph,
            else_branch=else_graph,
        ),
    ]
    graph = helper.make_graph(nodes, "if_p2_graph", [x1, x2], [y])
    return helper.make_model(graph, opset_imports=[helper.make_operatorsetid("", 11)])


def _make_if_p3_model() -> onnx.ModelProto:
    x1 = helper.make_tensor_value_info("If_p3_input1", TensorProto.FLOAT, [100])
    x2 = helper.make_tensor_value_info("If_p3_input2", TensorProto.FLOAT, [100])
    y = helper.make_tensor_value_info("If_p3_output", TensorProto.FLOAT, [100])

    then_const = numpy_helper.from_array(np.asarray(1.0, dtype=np.float32), name="then_const")
    then_add = helper.make_node(
        "Add",
        ["If_p3_input1", "then_const"],
        ["then_out"],
        name="ThenAdd",
    )
    then_out = helper.make_tensor_value_info("then_out", TensorProto.FLOAT, [100])
    then_graph = helper.make_graph(
        [then_add],
        "if_p3_then",
        [],
        [then_out],
        initializer=[then_const],
    )

    nested_then_const = numpy_helper.from_array(np.asarray(2.0, dtype=np.float32), name="nested_then_const")
    nested_then_add = helper.make_node(
        "Add",
        ["If_p3_input1", "nested_then_const"],
        ["nested_then_out"],
        name="NestedThenAdd",
    )
    nested_then_out = helper.make_tensor_value_info("nested_then_out", TensorProto.FLOAT, [100])
    nested_then_graph = helper.make_graph(
        [nested_then_add],
        "if_p3_nested_then",
        [],
        [nested_then_out],
        initializer=[nested_then_const],
    )

    nested_else_const = numpy_helper.from_array(np.asarray(3.0, dtype=np.float32), name="nested_else_const")
    nested_else_add = helper.make_node(
        "Add",
        ["If_p3_input2", "nested_else_const"],
        ["nested_else_out"],
        name="NestedElseAdd",
    )
    nested_else_out = helper.make_tensor_value_info("nested_else_out", TensorProto.FLOAT, [100])
    nested_else_graph = helper.make_graph(
        [nested_else_add],
        "if_p3_nested_else",
        [],
        [nested_else_out],
        initializer=[nested_else_const],
    )

    else_nodes = [
        helper.make_node("ReduceMin", ["If_p3_input1"], ["min1"], name="ElseReduceMin1", keepdims=0),
        helper.make_node("ReduceMin", ["If_p3_input2"], ["min2"], name="ElseReduceMin2", keepdims=0),
        helper.make_node("Greater", ["min1", "min2"], ["nested_cond"], name="ElseNestedCond"),
        helper.make_node(
            "If",
            ["nested_cond"],
            ["else_out"],
            name="ElseNestedIf",
            then_branch=nested_then_graph,
            else_branch=nested_else_graph,
        ),
    ]
    else_out = helper.make_tensor_value_info("else_out", TensorProto.FLOAT, [100])
    else_graph = helper.make_graph(
        else_nodes,
        "if_p3_else",
        [],
        [else_out],
    )

    nodes = [
        helper.make_node("ReduceSum", ["If_p3_input1"], ["sum1"], name="TopSum1", keepdims=0),
        helper.make_node("ReduceSum", ["If_p3_input2"], ["sum2"], name="TopSum2", keepdims=0),
        helper.make_node("Greater", ["sum1", "sum2"], ["cond"], name="TopCond"),
        helper.make_node(
            "If",
            ["cond"],
            ["If_p3_output"],
            name="If_p3",
            then_branch=then_graph,
            else_branch=else_graph,
        ),
    ]
    graph = helper.make_graph(nodes, "if_p3_graph", [x1, x2], [y])
    return helper.make_model(graph, opset_imports=[helper.make_operatorsetid("", 11)])


def test_flatbuffer_direct_if_p1_lowers_to_builtin_ops_without_custom() -> None:
    model_ir = lower_onnx_to_ir(
        _make_if_p1_model(),
        output_file_name="if_p1_builtin",
    )
    op_types = [str(op.op_type) for op in model_ir.operators]
    assert "CUSTOM" not in op_types
    assert "CONCATENATION" in op_types
    assert "SLICE" in op_types
    out_tensor = model_ir.tensors["If_p1_output"]
    assert out_tensor.shape_signature is not None
    assert int(out_tensor.shape_signature[0]) == -1
    assert int(out_tensor.shape_signature[1]) == 100


def test_flatbuffer_direct_if_p2_lowers_to_builtin_ops_without_custom() -> None:
    model_ir = lower_onnx_to_ir(
        _make_if_p2_model(),
        output_file_name="if_p2_builtin",
    )
    op_types = [str(op.op_type) for op in model_ir.operators]
    assert "CUSTOM" not in op_types
    assert "CONCATENATION" in op_types
    assert "SLICE" in op_types
    out_tensor = model_ir.tensors["If_p2_output"]
    assert out_tensor.shape_signature is not None
    assert int(out_tensor.shape_signature[0]) == -1
    assert int(out_tensor.shape_signature[1]) == 100


def test_flatbuffer_direct_if_p3_lowers_to_builtin_ops_without_custom() -> None:
    model_ir = lower_onnx_to_ir(
        _make_if_p3_model(),
        output_file_name="if_p3_builtin",
    )
    op_types = [str(op.op_type) for op in model_ir.operators]
    assert "CUSTOM" not in op_types
    assert "REDUCE_MIN" in op_types
    assert "GREATER" in op_types
    assert "ADD" in op_types
    assert any(op in op_types for op in {"SELECT", "MUL"})
    out_tensor = model_ir.tensors["If_p3_output"]
    assert out_tensor.shape_signature is not None
    assert int(out_tensor.shape_signature[0]) == 100


@pytest.mark.skipif(not _requires_flatbuffer_tools(), reason="flatbuffer_direct requires flatc and curl")
def test_flatbuffer_direct_if_p1_op_coverage_reports_builtin_if() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        model = _make_if_p1_model()
        model_path = _save_model(tmpdir, "if_p1_11", model)
        out_dir = os.path.join(tmpdir, "out")
        tflite_path = _convert(
            model_path=model_path,
            output_dir=out_dir,
            backend="flatbuffer_direct",
            report_op_coverage=True,
        )
        assert os.path.isfile(tflite_path)

        report_path = os.path.join(out_dir, "if_p1_11_op_coverage_report.json")
        assert os.path.isfile(report_path)
        with open(report_path, "r", encoding="utf-8") as f:
            report = json.load(f)
        assert report["graph_custom_ops"] == []
        if_reports = [
            node
            for node in report["graph_node_reports"]
            if str(node.get("onnx_op", "")) == "If"
        ]
        assert len(if_reports) == 1
        assert if_reports[0]["dispatch_mode"] == "builtin"


@pytest.mark.skipif(not _requires_flatbuffer_tools(), reason="flatbuffer_direct requires flatc and curl")
def test_flatbuffer_direct_if_p2_op_coverage_reports_builtin_if() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        model = _make_if_p2_model()
        model_path = _save_model(tmpdir, "if_p2_11", model)
        out_dir = os.path.join(tmpdir, "out")
        tflite_path = _convert(
            model_path=model_path,
            output_dir=out_dir,
            backend="flatbuffer_direct",
            report_op_coverage=True,
        )
        assert os.path.isfile(tflite_path)

        report_path = os.path.join(out_dir, "if_p2_11_op_coverage_report.json")
        assert os.path.isfile(report_path)
        with open(report_path, "r", encoding="utf-8") as f:
            report = json.load(f)
        assert report["graph_custom_ops"] == []
        if_reports = [
            node
            for node in report["graph_node_reports"]
            if str(node.get("onnx_op", "")) == "If"
        ]
        assert len(if_reports) == 1
        assert if_reports[0]["dispatch_mode"] == "builtin"


@pytest.mark.skipif(not _requires_flatbuffer_tools(), reason="flatbuffer_direct requires flatc and curl")
def test_flatbuffer_direct_if_p3_op_coverage_reports_builtin_if() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        model = _make_if_p3_model()
        model_path = _save_model(tmpdir, "if_p3_11", model)
        out_dir = os.path.join(tmpdir, "out")
        tflite_path = _convert(
            model_path=model_path,
            output_dir=out_dir,
            backend="flatbuffer_direct",
            report_op_coverage=True,
        )
        assert os.path.isfile(tflite_path)

        report_path = os.path.join(out_dir, "if_p3_11_op_coverage_report.json")
        assert os.path.isfile(report_path)
        with open(report_path, "r", encoding="utf-8") as f:
            report = json.load(f)
        assert report["graph_custom_ops"] == []
        if_reports = [
            node
            for node in report["graph_node_reports"]
            if str(node.get("onnx_op", "")) == "If"
        ]
        assert len(if_reports) >= 1
        assert all(str(node.get("dispatch_mode", "")) == "builtin" for node in if_reports)


def test_flatbuffer_direct_if_nms_guard_slice_dtype_consistent() -> None:
    model_ir = lower_onnx_to_ir(
        _make_if_nms_guard_direct_model(),
        output_file_name="if_nms_guard_direct",
        allow_custom_ops=True,
        custom_op_allowlist=["If"],
    )
    assert all(str(op.op_type) != "CUSTOM" for op in model_ir.operators)
    mismatches = []
    for op in model_ir.operators:
        if str(op.op_type) != "SLICE" or len(op.inputs) < 1 or len(op.outputs) != 1:
            continue
        in_name = str(op.inputs[0])
        out_name = str(op.outputs[0])
        in_tensor = model_ir.tensors.get(in_name)
        out_tensor = model_ir.tensors.get(out_name)
        if in_tensor is None or out_tensor is None:
            continue
        if str(in_tensor.dtype) != str(out_tensor.dtype):
            mismatches.append((in_name, in_tensor.dtype, out_name, out_tensor.dtype))
    assert mismatches == []


def test_flatbuffer_direct_if_nms_guard_nms_max_output_not_pinned_to_one() -> None:
    model_ir = lower_onnx_to_ir(
        _make_if_nms_guard_direct_model(),
        output_file_name="if_nms_guard_nms_dynamic_max_output",
        allow_custom_ops=True,
        custom_op_allowlist=["If"],
    )
    producers = {}
    for idx, op in enumerate(model_ir.operators):
        for out_name in op.outputs:
            producers[str(out_name)] = (idx, op)

    nms_ops = [
        op
        for op in model_ir.operators
        if str(op.op_type) in {"NON_MAX_SUPPRESSION_V4", "NON_MAX_SUPPRESSION_V5"}
    ]
    assert len(nms_ops) >= 1
    max_output_input = str(nms_ops[0].inputs[2])
    producer = producers.get(max_output_input, None)
    assert producer is not None
    assert str(producer[1].op_type) == "MINIMUM"


def test_flatbuffer_direct_gaze_keep_shape_inputs_fuses_late_conv_relu_chain() -> None:
    model_path = os.path.join(os.getcwd(), "gaze_estimation_adas_0002.onnx")
    if not os.path.isfile(model_path):
        pytest.skip("gaze_estimation_adas_0002.onnx not found")

    model = onnx.load(model_path)
    model_ir = lower_onnx_to_ir(
        onnx_graph=model,
        output_file_name="gaze_keep_shape_inputs_final_conv_relu_fuse_test",
        keep_shape_absolutely_input_names=[
            "right_eye_image__0",
            "left_eye_image__0",
            "head_pose_angles__0",
        ],
    )

    # Regression target:
    # Conv__58_output_nhwc -> RELU used to survive until serialization.
    relu_output_name = "StatefulPartitionedCall/model/conv2d_15/Conv2D__38:0"
    producer = next(
        (
            op
            for op in model_ir.operators
            if relu_output_name in [str(v) for v in list(op.outputs)]
        ),
        None,
    )
    assert producer is not None
    assert str(producer.op_type) == "CONV_2D"
    producer_opts = dict(producer.options) if isinstance(producer.options, dict) else {}
    assert str(producer_opts.get("fusedActivationFunction", "NONE")).upper() == "RELU"

    assert all(str(op.op_type) != "RELU" for op in model_ir.operators)
    assert "Conv__58_output_nhwc" not in model_ir.tensors
