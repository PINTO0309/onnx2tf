from __future__ import annotations

import copy
from dataclasses import fields, is_dataclass
from typing import Any

import numpy as np
import pytest

from onnx2tf.tflite_builder.ir import (
    ModelIR,
    OperatorIR,
    QuantParamIR,
    TensorIR,
)
from onnx2tf.tflite_builder.lower_from_onnx2tf import (
    _optimize_transpose_mean_maxpool_concat_conv_chains,
)


def _normalize(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return {
            "dtype": str(value.dtype),
            "shape": list(value.shape),
            "values": value.tolist(),
        }
    if is_dataclass(value):
        return {
            field.name: _normalize(getattr(value, field.name))
            for field in fields(value)
        }
    if isinstance(value, dict):
        return {str(key): _normalize(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_normalize(item) for item in value]
    return value


def _tensor(
    model_ir: ModelIR,
    name: str,
    shape: list[int],
    *,
    dtype: str = "FLOAT32",
    data: np.ndarray | None = None,
    signature: list[int] | None = None,
    quantization: QuantParamIR | None = None,
) -> None:
    model_ir.tensors[name] = TensorIR(
        name=name,
        dtype=dtype,
        shape=list(shape),
        shape_signature=(
            list(signature) if signature is not None else list(shape)
        ),
        data=data,
        is_variable=data is None,
        quantization=quantization,
    )


def _build_mean_maxpool_concat_chain(
    *,
    post_count: int = 1,
    dynamic_batch: bool = False,
) -> ModelIR:
    model_ir = ModelIR("mean_maxpool_concat_characterization")
    model_ir.inputs = ["q_raw_nhwc"]
    model_ir.outputs = [f"conv_y_{index}" for index in range(post_count)]
    nhwc_signature = [-1, 4, 6, 3] if dynamic_batch else [1, 4, 6, 3]
    nchw_signature = [-1, 3, 4, 6] if dynamic_batch else [1, 3, 4, 6]
    pooled_nhwc_signature = (
        [-1, 1, 1, 3] if dynamic_batch else [1, 1, 1, 3]
    )
    pooled_nchw_signature = (
        [-1, 3, 1, 1] if dynamic_batch else [1, 3, 1, 1]
    )
    concat_nchw_signature = (
        [-1, 6, 1, 1] if dynamic_batch else [1, 6, 1, 1]
    )
    concat_nhwc_signature = (
        [-1, 1, 1, 6] if dynamic_batch else [1, 1, 1, 6]
    )

    _tensor(
        model_ir,
        "q_raw_nhwc",
        [1, 4, 6, 3],
        dtype="INT8",
        signature=nhwc_signature,
        quantization=QuantParamIR(scale=[0.125], zero_point=[0]),
    )
    _tensor(
        model_ir,
        "q_nchw",
        [1, 3, 4, 6],
        dtype="INT8",
        signature=nchw_signature,
        quantization=QuantParamIR(scale=[0.125], zero_point=[0]),
    )
    _tensor(
        model_ir,
        "dq_mean_nchw",
        [1, 3, 4, 6],
        signature=nchw_signature,
    )
    _tensor(
        model_ir,
        "mean_nchw",
        [1, 3, 1, 1],
        signature=pooled_nchw_signature,
    )
    _tensor(
        model_ir,
        "dq_pool_nhwc",
        [1, 4, 6, 3],
        signature=nhwc_signature,
    )
    _tensor(
        model_ir,
        "pool_nhwc",
        [1, 1, 1, 3],
        signature=pooled_nhwc_signature,
    )
    _tensor(
        model_ir,
        "pool_nchw",
        [1, 3, 1, 1],
        signature=pooled_nchw_signature,
    )
    _tensor(
        model_ir,
        "concat_nchw",
        [1, 6, 1, 1],
        signature=concat_nchw_signature,
    )
    _tensor(
        model_ir,
        "q_cat_nchw",
        [1, 6, 1, 1],
        dtype="INT8",
        signature=concat_nchw_signature,
        quantization=QuantParamIR(
            scale=[0.1] * 6,
            zero_point=[0] * 6,
            quantized_dimension=1,
        ),
    )
    _tensor(
        model_ir,
        "mean_axes",
        [2],
        dtype="INT32",
        data=np.asarray([2, 3], dtype=np.int32),
    )
    _tensor(
        model_ir,
        "perm_nhwc_to_nchw",
        [4],
        dtype="INT32",
        data=np.asarray([0, 3, 1, 2], dtype=np.int32),
    )
    _tensor(
        model_ir,
        "perm_nchw_to_nhwc",
        [4],
        dtype="INT32",
        data=np.asarray([0, 2, 3, 1], dtype=np.int32),
    )
    _tensor(
        model_ir,
        "conv_filter",
        [2, 1, 1, 6],
        data=np.ones((2, 1, 1, 6), dtype=np.float32),
    )
    _tensor(
        model_ir,
        "conv_bias",
        [2],
        data=np.zeros((2,), dtype=np.float32),
    )

    operators = [
        OperatorIR(
            "TRANSPOSE",
            ["q_raw_nhwc", "perm_nhwc_to_nchw"],
            ["q_nchw"],
        ),
        OperatorIR("DEQUANTIZE", ["q_nchw"], ["dq_mean_nchw"]),
        OperatorIR(
            "MEAN",
            ["dq_mean_nchw", "mean_axes"],
            ["mean_nchw"],
            {"keepDims": True},
        ),
        OperatorIR("DEQUANTIZE", ["q_raw_nhwc"], ["dq_pool_nhwc"]),
        OperatorIR("MAX_POOL_2D", ["dq_pool_nhwc"], ["pool_nhwc"]),
        OperatorIR(
            "TRANSPOSE",
            ["pool_nhwc", "perm_nhwc_to_nchw"],
            ["pool_nchw"],
        ),
        OperatorIR(
            "CONCATENATION",
            ["mean_nchw", "pool_nchw"],
            ["concat_nchw"],
            {"axis": 1, "fusedActivationFunction": "NONE"},
        ),
        OperatorIR("QUANTIZE", ["concat_nchw"], ["q_cat_nchw"]),
    ]
    for index in range(post_count):
        post_name = f"post_nhwc_{index}"
        output_name = f"conv_y_{index}"
        _tensor(
            model_ir,
            post_name,
            [1, 1, 1, 6],
            dtype="INT8",
            signature=concat_nhwc_signature,
            quantization=QuantParamIR(scale=[0.1], zero_point=[0]),
        )
        _tensor(
            model_ir,
            output_name,
            [1, 1, 1, 2],
            dtype="INT8",
            signature=(
                [-1, 1, 1, 2]
                if dynamic_batch
                else [1, 1, 1, 2]
            ),
            quantization=QuantParamIR(scale=[0.1], zero_point=[0]),
        )
        operators.extend(
            [
                OperatorIR(
                    "TRANSPOSE",
                    ["q_cat_nchw", "perm_nchw_to_nhwc"],
                    [post_name],
                ),
                OperatorIR(
                    "CONV_2D",
                    [post_name, "conv_filter", "conv_bias"],
                    [output_name],
                    {"padding": "SAME", "strideH": 1, "strideW": 1},
                ),
            ]
        )
    model_ir.operators = operators
    return model_ir


def _prefix_model_ir(model_ir: ModelIR, prefix: str) -> ModelIR:
    prefixed = copy.deepcopy(model_ir)
    tensor_names = {name: f"{prefix}{name}" for name in prefixed.tensors}
    prefixed.tensors = {
        tensor_names[name]: tensor
        for name, tensor in prefixed.tensors.items()
    }
    for name, tensor in prefixed.tensors.items():
        tensor.name = name
    prefixed.inputs = [tensor_names[name] for name in prefixed.inputs]
    prefixed.outputs = [tensor_names[name] for name in prefixed.outputs]
    for op in prefixed.operators:
        op.inputs = [tensor_names[name] for name in op.inputs]
        op.outputs = [tensor_names[name] for name in op.outputs]
    return prefixed


def test_mean_maxpool_concat_chain_rewrites_to_nhwc() -> None:
    model_ir = _build_mean_maxpool_concat_chain()

    stats = _optimize_transpose_mean_maxpool_concat_conv_chains(model_ir)

    assert stats == {
        "optimized_transpose_mean_maxpool_concat_conv_chains": 1,
    }
    assert not any(op.op_type == "TRANSPOSE" for op in model_ir.operators)
    dq_mean = next(op for op in model_ir.operators if op.outputs == ["dq_mean_nchw"])
    concat = next(op for op in model_ir.operators if op.op_type == "CONCATENATION")
    conv = next(op for op in model_ir.operators if op.op_type == "CONV_2D")
    assert dq_mean.inputs == ["q_raw_nhwc"]
    assert np.array_equal(
        np.asarray(model_ir.tensors["mean_axes"].data),
        np.asarray([1, 2], dtype=np.int32),
    )
    assert model_ir.tensors["dq_mean_nchw"].shape == [1, 4, 6, 3]
    assert model_ir.tensors["mean_nchw"].shape == [1, 1, 1, 3]
    assert concat.inputs == ["mean_nchw", "pool_nhwc"]
    assert concat.options["axis"] == 3
    assert model_ir.tensors["concat_nchw"].shape == [1, 1, 1, 6]
    assert model_ir.tensors["q_cat_nchw"].shape == [1, 1, 1, 6]
    assert model_ir.tensors[
        "q_cat_nchw"
    ].quantization.quantized_dimension == 3
    assert conv.inputs[0] == "q_cat_nchw"
    assert "q_nchw" not in model_ir.tensors
    assert "pool_nchw" not in model_ir.tensors
    assert "post_nhwc_0" not in model_ir.tensors


def test_mean_maxpool_concat_preserves_dynamic_batch_signature() -> None:
    model_ir = _build_mean_maxpool_concat_chain(dynamic_batch=True)

    stats = _optimize_transpose_mean_maxpool_concat_conv_chains(model_ir)

    assert stats == {
        "optimized_transpose_mean_maxpool_concat_conv_chains": 1,
    }
    assert model_ir.tensors["dq_mean_nchw"].shape_signature == [-1, 4, 6, 3]
    assert model_ir.tensors["mean_nchw"].shape_signature == [-1, 1, 1, 3]
    assert model_ir.tensors["concat_nchw"].shape_signature == [-1, 1, 1, 6]
    assert model_ir.tensors["q_cat_nchw"].shape_signature == [-1, 1, 1, 6]


def test_mean_maxpool_concat_removes_multiple_post_transposes() -> None:
    model_ir = _build_mean_maxpool_concat_chain(post_count=2)

    stats = _optimize_transpose_mean_maxpool_concat_conv_chains(model_ir)

    assert stats == {
        "optimized_transpose_mean_maxpool_concat_conv_chains": 1,
    }
    assert not any(op.op_type == "TRANSPOSE" for op in model_ir.operators)
    assert [
        op.inputs[0]
        for op in model_ir.operators
        if op.op_type == "CONV_2D"
    ] == ["q_cat_nchw", "q_cat_nchw"]


def test_mean_maxpool_concat_fixed_point_rewrites_multiple_chains() -> None:
    model_ir = _build_mean_maxpool_concat_chain()
    second = _prefix_model_ir(
        _build_mean_maxpool_concat_chain(),
        "second_",
    )
    model_ir.inputs.extend(second.inputs)
    model_ir.outputs.extend(second.outputs)
    model_ir.tensors.update(second.tensors)
    model_ir.operators.extend(second.operators)

    stats = _optimize_transpose_mean_maxpool_concat_conv_chains(model_ir)

    assert stats == {
        "optimized_transpose_mean_maxpool_concat_conv_chains": 2,
    }
    assert not any(op.op_type == "TRANSPOSE" for op in model_ir.operators)


def test_mean_maxpool_concat_is_idempotent_after_rewrite() -> None:
    model_ir = _build_mean_maxpool_concat_chain()
    assert _optimize_transpose_mean_maxpool_concat_conv_chains(model_ir) == {
        "optimized_transpose_mean_maxpool_concat_conv_chains": 1,
    }
    after_first = _normalize(copy.deepcopy(model_ir))

    assert _optimize_transpose_mean_maxpool_concat_conv_chains(model_ir) == {
        "optimized_transpose_mean_maxpool_concat_conv_chains": 0,
    }
    assert _normalize(model_ir) == after_first


@pytest.mark.parametrize(
    "case",
    [
        "wrong_pre_perm",
        "public_pre_output",
        "pre_output_fanout",
        "mean_without_keepdims",
        "wrong_mean_axes",
        "wrong_axes_tensor_dtype",
        "wrong_axes_data_dtype",
        "quantized_axes",
        "wrong_pool_post_perm",
        "wrong_concat_axis",
        "wrong_qcat_quantized_dimension",
        "public_post_output",
        "nontranspose_post_user",
    ],
)
def test_mean_maxpool_concat_rejects_unsafe_boundaries(case: str) -> None:
    model_ir = _build_mean_maxpool_concat_chain()
    if case == "wrong_pre_perm":
        model_ir.tensors["perm_nhwc_to_nchw"].data = np.asarray(
            [0, 1, 2, 3],
            dtype=np.int32,
        )
    elif case == "public_pre_output":
        model_ir.outputs.append("q_nchw")
    elif case == "pre_output_fanout":
        _tensor(model_ir, "side_out", [1, 3, 4, 6])
        model_ir.operators.append(OperatorIR("RELU", ["q_nchw"], ["side_out"]))
    elif case == "mean_without_keepdims":
        mean = next(op for op in model_ir.operators if op.op_type == "MEAN")
        mean.options["keepDims"] = False
    elif case == "wrong_mean_axes":
        model_ir.tensors["mean_axes"].data = np.asarray(
            [1, 2],
            dtype=np.int32,
        )
    elif case == "wrong_axes_tensor_dtype":
        model_ir.tensors["mean_axes"].dtype = "INT64"
        model_ir.tensors["mean_axes"].data = np.asarray(
            [2, 3],
            dtype=np.int64,
        )
    elif case == "wrong_axes_data_dtype":
        model_ir.tensors["mean_axes"].data = np.asarray(
            [2, 3],
            dtype=np.int64,
        )
    elif case == "quantized_axes":
        model_ir.tensors["mean_axes"].quantization = QuantParamIR(
            scale=[1.0],
            zero_point=[0],
        )
    elif case == "wrong_pool_post_perm":
        pool_post = next(op for op in model_ir.operators if op.outputs == ["pool_nchw"])
        pool_post.inputs[1] = "perm_nchw_to_nhwc"
    elif case == "wrong_concat_axis":
        concat = next(op for op in model_ir.operators if op.op_type == "CONCATENATION")
        concat.options["axis"] = 2
    elif case == "wrong_qcat_quantized_dimension":
        model_ir.tensors["q_cat_nchw"].quantization.quantized_dimension = 0
    elif case == "public_post_output":
        model_ir.outputs.append("post_nhwc_0")
    elif case == "nontranspose_post_user":
        post = next(op for op in model_ir.operators if op.outputs == ["post_nhwc_0"])
        post.op_type = "RELU"
    else:
        raise AssertionError(case)
    before = _normalize(copy.deepcopy(model_ir))

    stats = _optimize_transpose_mean_maxpool_concat_conv_chains(model_ir)

    assert stats == {
        "optimized_transpose_mean_maxpool_concat_conv_chains": 0,
    }
    assert _normalize(model_ir) == before


@pytest.mark.parametrize("tensor_name", ["q_raw_nhwc", "pool_nhwc"])
def test_mean_maxpool_concat_rejects_short_signatures_atomically(
    tensor_name: str,
) -> None:
    model_ir = _build_mean_maxpool_concat_chain()
    model_ir.tensors[tensor_name].shape_signature = [1, 3]
    before = _normalize(copy.deepcopy(model_ir))

    stats = _optimize_transpose_mean_maxpool_concat_conv_chains(model_ir)

    assert stats == {
        "optimized_transpose_mean_maxpool_concat_conv_chains": 0,
    }
    assert _normalize(model_ir) == before


@pytest.mark.parametrize("invalid_extra", ["missing", "rank3", "short_signature"])
def test_mean_maxpool_concat_prevalidates_every_concat_input_atomically(
    invalid_extra: str,
) -> None:
    model_ir = _build_mean_maxpool_concat_chain()
    concat = next(
        op for op in model_ir.operators if op.op_type == "CONCATENATION"
    )
    concat.inputs.append("extra_input")
    if invalid_extra != "missing":
        shape = [1, 1, 3] if invalid_extra == "rank3" else [1, 1, 1, 2]
        signature = [1, 1] if invalid_extra == "short_signature" else shape
        _tensor(
            model_ir,
            "extra_input",
            shape,
            signature=signature,
        )
        model_ir.inputs.append("extra_input")
    before = _normalize(copy.deepcopy(model_ir))

    stats = _optimize_transpose_mean_maxpool_concat_conv_chains(model_ir)

    assert stats == {
        "optimized_transpose_mean_maxpool_concat_conv_chains": 0,
    }
    assert _normalize(model_ir) == before


@pytest.mark.parametrize("ownership", ["shared", "public", "input", "variable"])
def test_mean_maxpool_concat_preserves_nonlocal_axes_ownership(
    ownership: str,
) -> None:
    model_ir = _build_mean_maxpool_concat_chain()
    if ownership == "shared":
        _tensor(model_ir, "side_mean_input", [1, 3, 4, 6])
        _tensor(model_ir, "side_mean_output", [1, 3, 1, 1])
        model_ir.inputs.append("side_mean_input")
        model_ir.outputs.append("side_mean_output")
        model_ir.operators.append(
            OperatorIR(
                "MEAN",
                ["side_mean_input", "mean_axes"],
                ["side_mean_output"],
                {"keepDims": True},
            )
        )
    elif ownership == "public":
        model_ir.outputs.append("mean_axes")
    elif ownership == "input":
        model_ir.inputs.append("mean_axes")
    elif ownership == "variable":
        model_ir.tensors["mean_axes"].is_variable = True
    else:
        raise AssertionError(ownership)
    before = _normalize(copy.deepcopy(model_ir))

    stats = _optimize_transpose_mean_maxpool_concat_conv_chains(model_ir)

    assert stats == {
        "optimized_transpose_mean_maxpool_concat_conv_chains": 0,
    }
    assert _normalize(model_ir) == before
