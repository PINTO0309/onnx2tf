from __future__ import annotations

from copy import deepcopy

import numpy as np

from onnx2tf.tflite_builder.ir import ModelIR, OperatorIR, TensorIR
from onnx2tf.tflite_builder.lower_from_onnx2tf import (
    _sanitize_probable_nhwc_axis_sensitive_ops,
)
from onnx2tf.tflite_builder.passes.probable_nhwc_axis_sanitizer import (
    sanitize_probable_nhwc_axis_sensitive_ops,
)


def _fingerprint(model_ir: ModelIR) -> tuple[object, ...]:
    return (
        tuple(model_ir.inputs),
        tuple(model_ir.outputs),
        tuple(
            (
                operator.op_type,
                tuple(operator.inputs),
                tuple(operator.outputs),
                repr(operator.options),
            )
            for operator in model_ir.operators
        ),
        tuple(
            (
                name,
                tensor.dtype,
                tuple(tensor.shape),
                (
                    None
                    if tensor.shape_signature is None
                    else tuple(tensor.shape_signature)
                ),
                tensor.logical_layout,
                tensor.physical_layout,
                repr(tensor.quantization),
                (
                    None
                    if tensor.data is None
                    else (
                        str(np.asarray(tensor.data).dtype),
                        tuple(np.asarray(tensor.data).shape),
                        tuple(np.asarray(tensor.data).reshape(-1).tolist()),
                    )
                ),
            )
            for name, tensor in sorted(model_ir.tensors.items())
        ),
        repr(model_ir.metadata),
    )


def _run_owner_and_wrapper(
    model_ir: ModelIR,
) -> tuple[ModelIR, dict[str, int]]:
    owner_model_ir = deepcopy(model_ir)
    wrapper_model_ir = deepcopy(model_ir)
    owner_stats = sanitize_probable_nhwc_axis_sensitive_ops(owner_model_ir)
    wrapper_stats = _sanitize_probable_nhwc_axis_sensitive_ops(wrapper_model_ir)
    assert owner_stats == wrapper_stats
    assert _fingerprint(owner_model_ir) == _fingerprint(wrapper_model_ir)
    return owner_model_ir, owner_stats


def _add_tensor(
    model_ir: ModelIR,
    name: str,
    shape: list[int],
    *,
    dtype: str = "FLOAT32",
    data: np.ndarray | None = None,
) -> None:
    model_ir.tensors[name] = TensorIR(
        name=name,
        dtype=dtype,
        shape=[int(value) for value in shape],
        shape_signature=[int(value) for value in shape],
        data=data,
        is_variable=data is None,
    )


def test_probable_nhwc_sanitizer_clones_split_axis_and_restores_output() -> None:
    model_ir = ModelIR("probable_nhwc_split_axis_test")
    model_ir.inputs = ["x", "other"]
    model_ir.outputs = ["output"]
    _add_tensor(model_ir, "x", [1, 8, 8, 4])
    _add_tensor(model_ir, "split_0", [1, 2, 8, 4])
    _add_tensor(model_ir, "split_1", [1, 2, 8, 4])
    _add_tensor(model_ir, "output", [1, 2, 8, 4])
    _add_tensor(model_ir, "other", [1, 4])
    _add_tensor(model_ir, "other_0", [1, 2])
    _add_tensor(model_ir, "other_1", [1, 2])
    _add_tensor(
        model_ir,
        "axis",
        [1],
        dtype="INT32",
        data=np.asarray([1], dtype=np.int32),
    )
    model_ir.operators = [
        OperatorIR(
            op_type="SPLIT",
            inputs=["axis", "x"],
            outputs=["split_0", "split_1"],
            options={"numSplits": 2},
        ),
        OperatorIR(
            op_type="RELU",
            inputs=["split_0"],
            outputs=["output"],
            options={},
        ),
        OperatorIR(
            op_type="SPLIT",
            inputs=["axis", "other"],
            outputs=["other_0", "other_1"],
            options={"numSplits": 2},
        ),
    ]

    model_ir, stats = _run_owner_and_wrapper(model_ir)

    assert stats == {
        "sanitized_probable_nhwc_axis_sensitive_ops": 1,
        "inserted_probable_nhwc_terminal_transposes": 1,
    }
    first_split = model_ir.operators[0]
    assert first_split.inputs == ["axis_nhwc", "x"]
    assert np.array_equal(
        model_ir.tensors["axis_nhwc"].data,
        np.asarray([3], dtype=np.int32),
    )
    assert np.array_equal(
        model_ir.tensors["axis"].data,
        np.asarray([1], dtype=np.int32),
    )
    assert model_ir.tensors["split_0"].shape == [1, 8, 8, 2]
    assert model_ir.tensors["split_1"].shape == [1, 8, 8, 2]

    relu = model_ir.operators[1]
    terminal = model_ir.operators[2]
    assert relu.outputs == ["output_nhwc"]
    assert terminal.op_type == "TRANSPOSE"
    assert terminal.inputs == ["output_nhwc", "output_perm"]
    assert terminal.outputs == ["output"]
    assert model_ir.tensors["output_nhwc"].shape == [1, 8, 8, 2]
    assert model_ir.tensors["output"].shape == [1, 2, 8, 8]
    assert np.array_equal(
        model_ir.tensors["output_perm"].data,
        np.asarray([0, 3, 1, 2], dtype=np.int32),
    )


def test_probable_nhwc_sanitizer_repairs_concat_and_slice_metadata() -> None:
    model_ir = ModelIR("probable_nhwc_concat_slice_test")
    _add_tensor(model_ir, "concat_a", [1, 8, 8, 2])
    _add_tensor(model_ir, "concat_b", [1, 8, 8, 3])
    _add_tensor(model_ir, "concat_out", [1, 16, 8, 3])
    _add_tensor(model_ir, "slice_x", [1, 8, 8, 4])
    _add_tensor(model_ir, "slice_out", [1, 1, 8, 8])
    _add_tensor(
        model_ir,
        "begin",
        [4],
        dtype="INT32",
        data=np.asarray([0, 1, 0, 0], dtype=np.int32),
    )
    _add_tensor(
        model_ir,
        "size",
        [4],
        dtype="INT32",
        data=np.asarray([1, 1, 8, 8], dtype=np.int32),
    )
    model_ir.operators = [
        OperatorIR(
            op_type="CONCATENATION",
            inputs=["concat_a", "concat_b"],
            outputs=["concat_out"],
            options={"axis": 1},
        ),
        OperatorIR(
            op_type="SLICE",
            inputs=["slice_x", "begin", "size"],
            outputs=["slice_out"],
            options={},
        ),
    ]

    model_ir, stats = _run_owner_and_wrapper(model_ir)

    assert stats == {
        "sanitized_probable_nhwc_axis_sensitive_ops": 2,
        "inserted_probable_nhwc_terminal_transposes": 0,
    }
    assert model_ir.operators[0].options["axis"] == 3
    assert model_ir.tensors["concat_out"].shape == [1, 8, 8, 5]
    assert model_ir.tensors["concat_out"].shape_signature == [1, 8, 8, 5]
    assert np.array_equal(
        model_ir.tensors["begin"].data,
        np.asarray([0, 0, 0, 1], dtype=np.int32),
    )
    assert np.array_equal(
        model_ir.tensors["size"].data,
        np.asarray([1, 8, 8, 1], dtype=np.int32),
    )
    assert model_ir.tensors["slice_out"].shape == [1, 8, 8, 1]
    assert model_ir.tensors["slice_out"].shape_signature == [1, 8, 8, 1]


def test_probable_nhwc_sanitizer_propagates_unary_and_binary_metadata() -> None:
    model_ir = ModelIR("probable_nhwc_metadata_propagation_test")
    _add_tensor(model_ir, "x", [1, 8, 8, 4])
    _add_tensor(model_ir, "relu_out", [1, 4, 8, 8])
    _add_tensor(model_ir, "residual", [1, 8, 8, 1])
    _add_tensor(model_ir, "output", [1, 4, 8, 8])
    model_ir.operators = [
        OperatorIR(
            op_type="RELU",
            inputs=["x"],
            outputs=["relu_out"],
            options={},
        ),
        OperatorIR(
            op_type="ADD",
            inputs=["relu_out", "residual"],
            outputs=["output"],
            options={},
        ),
    ]

    model_ir, stats = _run_owner_and_wrapper(model_ir)

    assert stats == {
        "sanitized_probable_nhwc_axis_sensitive_ops": 0,
        "inserted_probable_nhwc_terminal_transposes": 0,
    }
    assert model_ir.tensors["relu_out"].shape == [1, 8, 8, 4]
    assert model_ir.tensors["relu_out"].shape_signature == [1, 8, 8, 4]
    assert model_ir.tensors["output"].shape == [1, 8, 8, 4]
    assert model_ir.tensors["output"].shape_signature == [1, 8, 8, 4]


def test_probable_nhwc_sanitizer_preserves_explicit_nchw_concat_axis() -> None:
    model_ir = ModelIR("explicit_nchw_concat_axis_test")
    model_ir.outputs = ["output"]
    model_ir.metadata["onnx_public_layout_map"] = {"output": "NCHW"}
    concat_inputs: list[str] = []
    for idx in range(4):
        source_name = f"source_{idx}_nhwc"
        nchw_name = f"branch_{idx}_nchw"
        perm_name = f"perm_{idx}"
        _add_tensor(model_ir, source_name, [1, 7, 7, 64])
        _add_tensor(
            model_ir,
            perm_name,
            [4],
            dtype="INT32",
            data=np.asarray([0, 3, 1, 2], dtype=np.int32),
        )
        _add_tensor(model_ir, nchw_name, [1, 64, 7, 7])
        model_ir.operators.append(
            OperatorIR(
                op_type="TRANSPOSE",
                inputs=[source_name, perm_name],
                outputs=[nchw_name],
            )
        )
        concat_inputs.append(nchw_name)
    _add_tensor(model_ir, "output", [1, 256, 7, 7])
    model_ir.operators.append(
        OperatorIR(
            op_type="CONCATENATION",
            inputs=concat_inputs,
            outputs=["output"],
            options={"axis": 1, "fusedActivationFunction": "NONE"},
        )
    )

    model_ir, stats = _run_owner_and_wrapper(model_ir)

    concat = model_ir.operators[-1]
    assert stats["sanitized_probable_nhwc_axis_sensitive_ops"] == 0
    assert int(concat.options["axis"]) == 1
    assert list(model_ir.tensors["output"].shape) == [1, 256, 7, 7]
