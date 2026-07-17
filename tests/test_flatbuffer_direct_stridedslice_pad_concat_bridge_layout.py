from __future__ import annotations

import ast
import copy
from dataclasses import fields, is_dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pytest

from onnx2tf.tflite_builder.core.graph import ModelIRGraphIndex
from onnx2tf.tflite_builder.core.validation import (
    validate_model_ir_invariants,
)
from onnx2tf.tflite_builder.ir import (
    ModelIR,
    OperatorIR,
    QuantParamIR,
    TensorIR,
)
from onnx2tf.tflite_builder.lower_from_onnx2tf import (
    _optimize_transpose_stridedslice_pad_concat_mul_add_posttranspose_nhwc_chains,
)
from onnx2tf.tflite_builder.passes.stridedslice_pad_concat_bridge_layout import (
    _optimize_transpose_stridedslice_pad_concat_mul_add_posttranspose_nhwc_chains as _optimize_transpose_stridedslice_pad_concat_mul_add_posttranspose_nhwc_chains_owner,
)
from onnx2tf.tflite_builder.passes.terminal_slice_concat_recovery_orchestration import (
    TERMINAL_SLICE_CONCAT_RECOVERY_PASS_IDS,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
_STATS = {
    "optimized_transpose_stridedslice_pad_concat_mul_add_posttranspose_nhwc_chains": 1,
}
_ZERO_STATS = {
    "optimized_transpose_stridedslice_pad_concat_mul_add_posttranspose_nhwc_chains": 0,
}


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
    signature: list[int] | None = None,
    data: np.ndarray | None = None,
    is_variable: bool = False,
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
        is_variable=bool(is_variable),
        quantization=quantization,
    )


def _add_chain(
    model_ir: ModelIR,
    *,
    prefix: str,
    dynamic_batch: bool = False,
    mirror_pad: bool = False,
    extra_add: bool = False,
) -> None:
    batch = -1 if dynamic_batch else 1
    int_max = np.iinfo(np.int32).max
    names = {
        "x0": f"{prefix}x0_nhwc",
        "x1": f"{prefix}x1_nhwc",
        "p0": f"{prefix}x0_nchw",
        "p1": f"{prefix}x1_nchw",
        "s0": f"{prefix}x0_slice",
        "s1": f"{prefix}x1_slice",
        "pad0": f"{prefix}x0_pad",
        "pad1": f"{prefix}x1_pad",
        "concat": f"{prefix}cat_nchw",
        "mul_const": f"{prefix}mul_const",
        "mul_out": f"{prefix}mul_out",
        "post": f"{prefix}mul_out_nhwc",
        "bias": f"{prefix}add_bias",
        "output": f"{prefix}y",
        "output2": f"{prefix}y2",
        "begin": f"{prefix}slice_begin",
        "end": f"{prefix}slice_end",
        "stride": f"{prefix}slice_stride",
        "pads": f"{prefix}pads",
        "to_nchw": f"{prefix}to_nchw_perm",
        "to_nhwc": f"{prefix}to_nhwc_perm",
    }
    model_ir.inputs.extend([names["x0"], names["x1"]])
    model_ir.outputs.append(names["output"])
    if extra_add:
        model_ir.outputs.append(names["output2"])

    for name, shape, signature in (
        (names["x0"], [1, 3, 2, 2], [batch, 3, 2, 2]),
        (names["x1"], [1, 3, 2, 2], [batch, 3, 2, 2]),
        (names["p0"], [1, 2, 3, 2], [batch, 2, 3, 2]),
        (names["p1"], [1, 2, 3, 2], [batch, 2, 3, 2]),
        (names["s0"], [1, 2, 3, 1], [batch, 2, 3, 1]),
        (names["s1"], [1, 2, 3, 1], [batch, 2, 3, 1]),
        (names["pad0"], [1, 2, 3, 1], [batch, 2, 3, 1]),
        (names["pad1"], [1, 2, 3, 1], [batch, 2, 3, 1]),
        (names["concat"], [1, 4, 3, 1], [batch, 4, 3, 1]),
        (names["mul_out"], [1, 4, 3, 1], [batch, 4, 3, 1]),
        (names["post"], [1, 3, 1, 4], [batch, 3, 1, 4]),
        (names["output"], [1, 3, 1, 4], [batch, 3, 1, 4]),
    ):
        _tensor(model_ir, name, shape, signature=signature)
    if extra_add:
        _tensor(
            model_ir,
            names["output2"],
            [1, 3, 1, 4],
            signature=[batch, 3, 1, 4],
        )

    for name, values in (
        (names["to_nchw"], [0, 3, 1, 2]),
        (names["to_nhwc"], [0, 2, 3, 1]),
        (names["begin"], [0, 0, 0, 0]),
        (names["end"], [int_max, int_max, int_max, -1]),
        (names["stride"], [1, 1, 1, 1]),
    ):
        _tensor(
            model_ir,
            name,
            [4],
            dtype="INT32",
            data=np.asarray(values, dtype=np.int32),
        )
    _tensor(
        model_ir,
        names["pads"],
        [4, 2],
        dtype="INT32",
        data=np.asarray(
            [[0, 0], [1, 2], [3, 4], [5, 6]],
            dtype=np.int32,
        ),
    )
    _tensor(
        model_ir,
        names["mul_const"],
        [1, 4, 1, 1],
        data=np.arange(1, 5, dtype=np.float32).reshape(1, 4, 1, 1),
    )
    _tensor(
        model_ir,
        names["bias"],
        [1, 1, 1, 4],
        data=np.arange(4, dtype=np.float32).reshape(1, 1, 1, 4),
    )
    if extra_add:
        _tensor(
            model_ir,
            f"{prefix}add_bias2",
            [1],
            data=np.asarray([1.0], dtype=np.float32),
        )

    slice_options = {
        "beginMask": 0,
        "endMask": 7,
        "ellipsisMask": 0,
        "newAxisMask": 0,
        "shrinkAxisMask": 0,
    }
    pad_type = "MIRROR_PAD" if mirror_pad else "PAD"
    model_ir.operators.extend(
        [
            OperatorIR(
                "TRANSPOSE",
                [names["x0"], names["to_nchw"]],
                [names["p0"]],
            ),
            OperatorIR(
                "STRIDED_SLICE",
                [names["p0"], names["begin"], names["end"], names["stride"]],
                [names["s0"]],
                options=dict(slice_options),
                version=2,
                onnx_node_name=f"{prefix}slice0",
                onnx_op_type="Slice",
            ),
            OperatorIR(
                pad_type,
                [names["s0"], names["pads"]],
                [names["pad0"]],
                options={"mode": "REFLECT"} if mirror_pad else {},
                version=2,
                onnx_node_name=f"{prefix}pad0",
                onnx_op_type="Pad",
            ),
            OperatorIR(
                "TRANSPOSE",
                [names["x1"], names["to_nchw"]],
                [names["p1"]],
            ),
            OperatorIR(
                "STRIDED_SLICE",
                [names["p1"], names["begin"], names["end"], names["stride"]],
                [names["s1"]],
                options=dict(slice_options),
                version=2,
                onnx_node_name=f"{prefix}slice1",
                onnx_op_type="Slice",
            ),
            OperatorIR(
                pad_type,
                [names["s1"], names["pads"]],
                [names["pad1"]],
                options={"mode": "REFLECT"} if mirror_pad else {},
                version=2,
                onnx_node_name=f"{prefix}pad1",
                onnx_op_type="Pad",
            ),
            OperatorIR(
                "CONCATENATION",
                [names["pad0"], names["pad1"]],
                [names["concat"]],
                options={"axis": 1, "fusedActivationFunction": "NONE"},
                axis_semantics={"axis": "physical"},
                version=2,
                onnx_node_name=f"{prefix}concat",
                onnx_op_type="Concat",
            ),
            OperatorIR(
                "MUL",
                [names["concat"], names["mul_const"]],
                [names["mul_out"]],
            ),
            OperatorIR(
                "TRANSPOSE",
                [names["mul_out"], names["to_nhwc"]],
                [names["post"]],
            ),
            OperatorIR(
                "ADD",
                [names["post"], names["bias"]],
                [names["output"]],
            ),
        ]
    )
    if extra_add:
        model_ir.operators.append(
            OperatorIR(
                "ADD",
                [f"{prefix}add_bias2", names["post"]],
                [names["output2"]],
            )
        )


def _model(
    *,
    branches: int = 1,
    dynamic_batch: bool = False,
    mirror_pad: bool = False,
    extra_add: bool = False,
) -> ModelIR:
    model_ir = ModelIR("stridedslice_pad_concat_bridge_characterization")
    for branch_index in range(int(branches)):
        _add_chain(
            model_ir,
            prefix=f"branch{branch_index}_",
            dynamic_batch=dynamic_batch,
            mirror_pad=mirror_pad,
            extra_add=extra_add,
        )
    return model_ir


def _owner_wrapper_case(case: str) -> ModelIR:
    if case == "dynamic":
        return _model(dynamic_batch=True)
    if case == "multiple":
        return _model(branches=2)
    if case == "multi-add":
        return _model(extra_add=True)
    if case == "mirror-pad":
        return _model(mirror_pad=True)
    model_ir = _model()

    if case == "scalar":
        tensor = model_ir.tensors["branch0_mul_const"]
        tensor.shape = [1]
        tensor.shape_signature = [1]
        tensor.data = np.asarray([2.0], dtype=np.float32)
    elif case == "shared-constants":
        for name in (
            "branch0_slice_end",
            "branch0_pads",
            "branch0_mul_const",
        ):
            output = f"{name}_copy"
            _tensor(
                model_ir,
                output,
                list(np.asarray(model_ir.tensors[name].data).shape),
            )
            model_ir.outputs.append(output)
            model_ir.operators.append(
                OperatorIR("IDENTITY", [name], [output])
            )
    elif case == "public-index-output":
        model_ir.outputs.append("branch0_slice_end")
    elif case == "public-index-input":
        model_ir.inputs.append("branch0_slice_begin")
    elif case == "wrong-index-dtype":
        tensor = model_ir.tensors["branch0_pads"]
        tensor.dtype = "FLOAT32"
        tensor.data = np.asarray(tensor.data, dtype=np.float32)
    elif case == "public-mul":
        model_ir.outputs.append("branch0_mul_const")
    elif case == "variable-mul":
        model_ir.tensors["branch0_mul_const"].is_variable = True
    elif case == "per-axis":
        for name in (
            "branch0_x0_slice",
            "branch0_x1_slice",
            "branch0_x0_pad",
            "branch0_x1_pad",
            "branch0_cat_nchw",
            "branch0_mul_const",
            "branch0_mul_out",
        ):
            model_ir.tensors[name].quantization = QuantParamIR(
                scale=[0.1, 0.2, 0.3, 0.4],
                zero_point=[0, 0, 0, 0],
                quantized_dimension=1,
            )
    elif case == "unmatched":
        _tensor(model_ir, "unused", [1])
        model_ir.tensors["branch0_to_nhwc_perm"].data = np.asarray(
            [0, 3, 1, 2],
            dtype=np.int32,
        )
    elif case == "missing-metadata":
        del model_ir.tensors["branch0_x0_slice"]
    elif case == "missing-post":
        del model_ir.tensors["branch0_mul_out_nhwc"]
    elif case == "malformed-options":
        model_ir.operators[1].options["beginMask"] = None
    elif case == "reverse-topology":
        model_ir.operators[1], model_ir.operators[2] = (
            model_ir.operators[2],
            model_ir.operators[1],
        )
    elif case == "public-intermediate":
        model_ir.outputs.append("branch0_x0_slice")
    elif case == "duplicate-source":
        model_ir.operators.extend(
            [
                OperatorIR(
                    "IDENTITY",
                    ["branch0_x1_nhwc"],
                    ["branch0_x0_nhwc"],
                ),
                OperatorIR(
                    "IDENTITY",
                    ["branch0_x0_nhwc"],
                    ["branch0_x0_nhwc"],
                ),
            ]
        )
    return model_ir


@pytest.mark.parametrize(
    "case",
    [
        "ordinary",
        "dynamic",
        "multiple",
        "multi-add",
        "mirror-pad",
        "scalar",
        "shared-constants",
        "public-index-output",
        "public-index-input",
        "wrong-index-dtype",
        "public-mul",
        "variable-mul",
        "per-axis",
        "unmatched",
        "missing-metadata",
        "missing-post",
        "malformed-options",
        "reverse-topology",
        "public-intermediate",
        "duplicate-source",
    ],
)
def test_slice_pad_concat_owner_and_wrapper_are_identical(
    case: str,
) -> None:
    owner_model = _owner_wrapper_case(case)
    wrapper_model = copy.deepcopy(owner_model)

    owner_stats = (
        _optimize_transpose_stridedslice_pad_concat_mul_add_posttranspose_nhwc_chains_owner(
            owner_model
        )
    )
    wrapper_stats = (
        _optimize_transpose_stridedslice_pad_concat_mul_add_posttranspose_nhwc_chains(
            wrapper_model
        )
    )

    assert wrapper_stats == owner_stats
    assert _normalize(wrapper_model) == _normalize(owner_model)


def _assert_transactional_rejection(model_ir: ModelIR) -> None:
    before = _normalize(copy.deepcopy(model_ir))

    stats = (
        _optimize_transpose_stridedslice_pad_concat_mul_add_posttranspose_nhwc_chains(
            model_ir
        )
    )

    assert stats == _ZERO_STATS
    assert _normalize(model_ir) == before


@pytest.mark.parametrize("dynamic_batch", [False, True])
def test_slice_pad_concat_rewrites_ordinary_chain(
    dynamic_batch: bool,
) -> None:
    model_ir = _model(dynamic_batch=dynamic_batch)

    stats = (
        _optimize_transpose_stridedslice_pad_concat_mul_add_posttranspose_nhwc_chains(
            model_ir
        )
    )

    assert stats == _STATS
    assert [operator.op_type for operator in model_ir.operators] == [
        "STRIDED_SLICE",
        "PAD",
        "STRIDED_SLICE",
        "PAD",
        "CONCATENATION",
        "MUL",
        "ADD",
    ]
    slice0, pad0, slice1, pad1, concat, mul, add = model_ir.operators
    assert slice0.inputs[0] == "branch0_x0_nhwc"
    assert slice1.inputs[0] == "branch0_x1_nhwc"
    for slice_op, node_name in (
        (slice0, "branch0_slice0"),
        (slice1, "branch0_slice1"),
    ):
        assert slice_op.options == {
            "beginMask": 0,
            "endMask": 11,
            "ellipsisMask": 0,
            "newAxisMask": 0,
            "shrinkAxisMask": 0,
        }
        assert slice_op.version == 2
        assert slice_op.onnx_node_name == node_name
        assert slice_op.onnx_op_type == "Slice"
    for pad_op, node_name in (
        (pad0, "branch0_pad0"),
        (pad1, "branch0_pad1"),
    ):
        assert pad_op.options == {}
        assert pad_op.version == 2
        assert pad_op.onnx_node_name == node_name
        assert pad_op.onnx_op_type == "Pad"
    assert concat.options == {
        "axis": 3,
        "fusedActivationFunction": "NONE",
    }
    assert concat.axis_semantics == {"axis": "physical"}
    assert concat.version == 2
    assert concat.onnx_node_name == "branch0_concat"
    assert concat.onnx_op_type == "Concat"
    assert mul.outputs == ["branch0_mul_out_nhwc"]
    assert add.inputs[0] == "branch0_mul_out_nhwc"

    int_max = np.iinfo(np.int32).max
    assert np.asarray(
        model_ir.tensors["branch0_slice_end"].data
    ).tolist() == [int_max, int_max, -1, int_max]
    assert np.asarray(model_ir.tensors["branch0_pads"].data).tolist() == [
        [0, 0],
        [3, 4],
        [5, 6],
        [1, 2],
    ]
    assert model_ir.tensors["branch0_x0_slice"].shape == [1, 3, 1, 2]
    assert model_ir.tensors["branch0_x0_pad"].shape == [1, 3, 1, 2]
    assert model_ir.tensors["branch0_cat_nchw"].shape == [1, 3, 1, 4]
    assert model_ir.tensors["branch0_mul_const"].shape == [1, 1, 1, 4]
    assert model_ir.tensors["branch0_mul_out_nhwc"].shape == [1, 3, 1, 4]
    expected_signature = (
        [-1, 3, 1, 4] if dynamic_batch else [1, 3, 1, 4]
    )
    assert model_ir.tensors[
        "branch0_mul_out_nhwc"
    ].shape_signature == expected_signature
    assert validate_model_ir_invariants(model_ir) == []


def test_slice_pad_concat_rewrites_multiple_and_fixed_point() -> None:
    model_ir = _model(branches=2)

    first = (
        _optimize_transpose_stridedslice_pad_concat_mul_add_posttranspose_nhwc_chains(
            model_ir
        )
    )
    after_first = _normalize(copy.deepcopy(model_ir))
    second = (
        _optimize_transpose_stridedslice_pad_concat_mul_add_posttranspose_nhwc_chains(
            model_ir
        )
    )

    assert first == {
        "optimized_transpose_stridedslice_pad_concat_mul_add_posttranspose_nhwc_chains": 2,
    }
    assert second == _ZERO_STATS
    assert _normalize(model_ir) == after_first


def test_slice_pad_concat_reuses_one_graph_index(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model_ir = _model(branches=2)
    refresh_count = 0
    original_refresh = ModelIRGraphIndex.refresh

    def counted_refresh(graph_index: ModelIRGraphIndex) -> None:
        nonlocal refresh_count
        refresh_count += 1
        original_refresh(graph_index)

    monkeypatch.setattr(ModelIRGraphIndex, "refresh", counted_refresh)

    stats = (
        _optimize_transpose_stridedslice_pad_concat_mul_add_posttranspose_nhwc_chains(
            model_ir
        )
    )

    assert stats == {
        "optimized_transpose_stridedslice_pad_concat_mul_add_posttranspose_nhwc_chains": 2,
    }
    assert refresh_count == 1


def test_slice_pad_concat_preserves_multiple_add_users() -> None:
    model_ir = _model(extra_add=True)

    stats = (
        _optimize_transpose_stridedslice_pad_concat_mul_add_posttranspose_nhwc_chains(
            model_ir
        )
    )

    assert stats == _STATS
    add_ops = [op for op in model_ir.operators if op.op_type == "ADD"]
    assert len(add_ops) == 2
    assert all(
        "branch0_mul_out_nhwc" in [str(value) for value in add.inputs]
        for add in add_ops
    )


def test_slice_pad_concat_preserves_mirror_pad() -> None:
    model_ir = _model(mirror_pad=True)

    stats = (
        _optimize_transpose_stridedslice_pad_concat_mul_add_posttranspose_nhwc_chains(
            model_ir
        )
    )

    assert stats == _STATS
    mirror_pads = [
        op for op in model_ir.operators if op.op_type == "MIRROR_PAD"
    ]
    assert len(mirror_pads) == 2
    assert all(op.options == {"mode": "REFLECT"} for op in mirror_pads)


def test_slice_pad_concat_accepts_scalar_mul_constant() -> None:
    model_ir = _model()
    tensor = model_ir.tensors["branch0_mul_const"]
    tensor.shape = [1]
    tensor.shape_signature = [1]
    tensor.data = np.asarray([2.0], dtype=np.float32)

    stats = (
        _optimize_transpose_stridedslice_pad_concat_mul_add_posttranspose_nhwc_chains(
            model_ir
        )
    )

    assert stats == _STATS
    assert tensor.shape == [1]
    assert np.asarray(tensor.data).tolist() == [2.0]


def test_slice_pad_concat_clones_shared_constants_collision_safely() -> None:
    model_ir = _model()
    shared_names = (
        "branch0_slice_end",
        "branch0_pads",
        "branch0_mul_const",
    )
    originals = {
        name: np.asarray(model_ir.tensors[name].data).copy()
        for name in shared_names
    }
    for name in shared_names:
        output = f"{name}_copy"
        _tensor(model_ir, output, list(np.asarray(originals[name]).shape))
        model_ir.outputs.append(output)
        model_ir.operators.append(
            OperatorIR("IDENTITY", [name], [output])
        )
    for name in (
        "branch0_slice_end_nhwc_end",
        "branch0_pads_nhwc_pads",
        "branch0_mul_const_nhwc",
    ):
        _tensor(
            model_ir,
            name,
            [1],
            data=np.asarray([99], dtype=np.int32),
        )
        model_ir.outputs.append(name)

    stats = (
        _optimize_transpose_stridedslice_pad_concat_mul_add_posttranspose_nhwc_chains(
            model_ir
        )
    )

    assert stats == _STATS
    for name in shared_names:
        assert np.array_equal(model_ir.tensors[name].data, originals[name])
    slice_ops = [
        op for op in model_ir.operators if op.op_type == "STRIDED_SLICE"
    ]
    assert all(op.inputs[2] != "branch0_slice_end" for op in slice_ops)
    assert len({op.inputs[2] for op in slice_ops}) == 1
    pad_ops = [
        op
        for op in model_ir.operators
        if op.op_type in {"PAD", "MIRROR_PAD"}
    ]
    assert all(op.inputs[1] != "branch0_pads" for op in pad_ops)
    assert len({op.inputs[1] for op in pad_ops}) == 1
    mul = next(op for op in model_ir.operators if op.op_type == "MUL")
    assert "branch0_mul_const" not in mul.inputs
    for name in (
        "branch0_slice_end_nhwc_end",
        "branch0_pads_nhwc_pads",
        "branch0_mul_const_nhwc",
    ):
        assert np.asarray(model_ir.tensors[name].data).tolist() == [99]


def test_slice_pad_concat_does_not_prune_unmatched_graph() -> None:
    model_ir = _model()
    _tensor(model_ir, "unused", [1])
    model_ir.tensors["branch0_to_nhwc_perm"].data = np.asarray(
        [0, 3, 1, 2],
        dtype=np.int32,
    )

    stats = (
        _optimize_transpose_stridedslice_pad_concat_mul_add_posttranspose_nhwc_chains(
            model_ir
        )
    )

    assert stats == _ZERO_STATS
    assert "unused" in model_ir.tensors
    assert model_ir.metadata == {}


@pytest.mark.parametrize(
    "mutate",
    [
        pytest.param(
            lambda model_ir: model_ir.tensors[
                "branch0_to_nhwc_perm"
            ].__setattr__(
                "data",
                np.asarray([0, 3, 1, 2], dtype=np.int32),
            ),
            id="wrong-post-permutation",
        ),
        pytest.param(
            lambda model_ir: model_ir.outputs.append("branch0_mul_out"),
            id="public-mul-output",
        ),
        pytest.param(
            lambda model_ir: model_ir.outputs.append(
                "branch0_mul_out_nhwc"
            ),
            id="public-post-output",
        ),
        pytest.param(
            lambda model_ir: model_ir.operators[6].options.__setitem__(
                "axis",
                3,
            ),
            id="wrong-concat-axis",
        ),
        pytest.param(
            lambda model_ir: model_ir.outputs.append("branch0_cat_nchw"),
            id="public-concat-output",
        ),
        pytest.param(
            lambda model_ir: model_ir.operators[6].inputs.pop(),
            id="single-concat-input",
        ),
        pytest.param(
            lambda model_ir: model_ir.tensors[
                "branch0_mul_const"
            ].__setattr__("data", None),
            id="dynamic-mul-constant",
        ),
        pytest.param(
            lambda model_ir: model_ir.tensors[
                "branch0_add_bias"
            ].__setattr__("data", None),
            id="dynamic-add-bias",
        ),
        pytest.param(
            lambda model_ir: model_ir.tensors[
                "branch0_add_bias"
            ].__setattr__(
                "data",
                np.zeros((1, 4, 1, 1), dtype=np.float32),
            ),
            id="nchw-add-bias",
        ),
        pytest.param(
            lambda model_ir: _append_identity_consumer(
                model_ir,
                "branch0_cat_nchw",
                "concat_copy",
            ),
            id="concat-fanout",
        ),
        pytest.param(
            lambda model_ir: _append_identity_consumer(
                model_ir,
                "branch0_x0_pad",
                "pad_copy",
            ),
            id="pad-fanout",
        ),
        pytest.param(
            lambda model_ir: _append_identity_consumer(
                model_ir,
                "branch0_x0_slice",
                "slice_copy",
            ),
            id="slice-fanout",
        ),
        pytest.param(
            lambda model_ir: _append_identity_consumer(
                model_ir,
                "branch0_x0_nchw",
                "pre_copy",
            ),
            id="pre-fanout",
        ),
        pytest.param(
            lambda model_ir: model_ir.operators[1].options.__setitem__(
                "newAxisMask",
                1,
            ),
            id="unsupported-slice-mask",
        ),
        pytest.param(
            lambda model_ir: model_ir.tensors[
                "branch0_slice_end"
            ].__setattr__(
                "data",
                np.asarray([1, 2, 3], dtype=np.int32),
            ),
            id="short-slice-end",
        ),
        pytest.param(
            lambda model_ir: model_ir.tensors[
                "branch0_pads"
            ].__setattr__(
                "data",
                np.zeros((3, 2), dtype=np.int32),
            ),
            id="invalid-pads",
        ),
        pytest.param(
            lambda model_ir: model_ir.outputs.append("branch0_x0_nchw"),
            id="public-pre-output",
        ),
    ],
)
def test_slice_pad_concat_preserves_existing_rejections(
    mutate: Callable[[ModelIR], object],
) -> None:
    model_ir = _model()
    mutate(model_ir)

    _assert_transactional_rejection(model_ir)


def _append_identity_consumer(
    model_ir: ModelIR,
    input_name: str,
    output_name: str,
) -> None:
    source = model_ir.tensors[input_name]
    _tensor(model_ir, output_name, list(source.shape))
    model_ir.outputs.append(output_name)
    model_ir.operators.append(
        OperatorIR("IDENTITY", [input_name], [output_name])
    )


@pytest.mark.parametrize(
    "case",
    [
        "missing-source",
        "missing-slice",
        "missing-pad",
        "missing-concat",
        "missing-mul-output",
        "missing-post-output",
        "rank-three-source",
        "short-slice-signature",
        "short-pad-signature",
        "short-concat-signature",
        "short-mul-signature",
    ],
)
def test_slice_pad_concat_rejects_incomplete_metadata(case: str) -> None:
    model_ir = _model()
    missing_names = {
        "missing-source": "branch0_x0_nhwc",
        "missing-slice": "branch0_x0_slice",
        "missing-pad": "branch0_x0_pad",
        "missing-concat": "branch0_cat_nchw",
        "missing-mul-output": "branch0_mul_out",
        "missing-post-output": "branch0_mul_out_nhwc",
    }
    if case in missing_names:
        del model_ir.tensors[missing_names[case]]
    elif case == "rank-three-source":
        tensor = model_ir.tensors["branch0_x0_nhwc"]
        tensor.shape = [1, 3, 2]
        tensor.shape_signature = [1, 3, 2]
    else:
        signature_names = {
            "short-slice-signature": "branch0_x0_slice",
            "short-pad-signature": "branch0_x0_pad",
            "short-concat-signature": "branch0_cat_nchw",
            "short-mul-signature": "branch0_mul_out",
        }
        model_ir.tensors[signature_names[case]].shape_signature = [1, 2, 3]

    _assert_transactional_rejection(model_ir)


@pytest.mark.parametrize(
    "role",
    ["slice_begin", "slice_end", "slice_stride", "pads"],
)
@pytest.mark.parametrize(
    "ownership",
    ["public-input", "variable", "wrong-dtype", "quantized"],
)
def test_slice_pad_concat_rejects_unsafe_index_constant(
    role: str,
    ownership: str,
) -> None:
    model_ir = _model()
    tensor = model_ir.tensors[f"branch0_{role}"]
    if ownership == "public-input":
        model_ir.inputs.append(tensor.name)
    elif ownership == "variable":
        tensor.is_variable = True
    elif ownership == "wrong-dtype":
        tensor.dtype = "FLOAT32"
        tensor.data = np.asarray(tensor.data, dtype=np.float32)
    elif ownership == "quantized":
        tensor.quantization = QuantParamIR(
            scale=[0.1],
            zero_point=[0],
            quantized_dimension=0,
        )

    _assert_transactional_rejection(model_ir)


@pytest.mark.parametrize("role", ["slice_end", "pads"])
def test_slice_pad_concat_clones_public_index_constant_output(
    role: str,
) -> None:
    model_ir = _model()
    tensor_name = f"branch0_{role}"
    original = np.asarray(model_ir.tensors[tensor_name].data).copy()
    model_ir.outputs.append(tensor_name)

    stats = (
        _optimize_transpose_stridedslice_pad_concat_mul_add_posttranspose_nhwc_chains(
            model_ir
        )
    )

    assert stats == _STATS
    if role == "slice_end":
        users = [
            op
            for op in model_ir.operators
            if op.op_type == "STRIDED_SLICE"
        ]
        assert all(op.inputs[2] != tensor_name for op in users)
        assert len({op.inputs[2] for op in users}) == 1
    else:
        users = [
            op
            for op in model_ir.operators
            if op.op_type in {"PAD", "MIRROR_PAD"}
        ]
        assert all(op.inputs[1] != tensor_name for op in users)
        assert len({op.inputs[1] for op in users}) == 1
    assert np.array_equal(model_ir.tensors[tensor_name].data, original)


@pytest.mark.parametrize("ownership", ["public-input", "variable"])
def test_slice_pad_concat_preserves_mul_constant_ownership(
    ownership: str,
) -> None:
    model_ir = _model()
    tensor = model_ir.tensors["branch0_mul_const"]
    if ownership == "public-input":
        model_ir.inputs.append(tensor.name)
    else:
        tensor.is_variable = True

    _assert_transactional_rejection(model_ir)


def test_slice_pad_concat_clones_public_mul_constant_output() -> None:
    model_ir = _model()
    tensor_name = "branch0_mul_const"
    original = np.asarray(model_ir.tensors[tensor_name].data).copy()
    model_ir.outputs.append(tensor_name)

    stats = (
        _optimize_transpose_stridedslice_pad_concat_mul_add_posttranspose_nhwc_chains(
            model_ir
        )
    )

    assert stats == _STATS
    mul = next(op for op in model_ir.operators if op.op_type == "MUL")
    assert tensor_name not in mul.inputs
    assert np.array_equal(model_ir.tensors[tensor_name].data, original)


def test_slice_pad_concat_remaps_per_axis_quantization() -> None:
    model_ir = _model()
    for name in (
        "branch0_x0_slice",
        "branch0_x1_slice",
        "branch0_x0_pad",
        "branch0_x1_pad",
        "branch0_cat_nchw",
        "branch0_mul_const",
        "branch0_mul_out",
    ):
        model_ir.tensors[name].quantization = QuantParamIR(
            scale=[0.1, 0.2, 0.3, 0.4],
            zero_point=[0, 0, 0, 0],
            quantized_dimension=1,
        )

    stats = (
        _optimize_transpose_stridedslice_pad_concat_mul_add_posttranspose_nhwc_chains(
            model_ir
        )
    )

    assert stats == _STATS
    for name in (
        "branch0_x0_slice",
        "branch0_x1_slice",
        "branch0_x0_pad",
        "branch0_x1_pad",
        "branch0_cat_nchw",
        "branch0_mul_const",
        "branch0_mul_out_nhwc",
    ):
        quantization = model_ir.tensors[name].quantization
        assert isinstance(quantization, QuantParamIR)
        assert quantization.quantized_dimension == 3


@pytest.mark.parametrize("role", ["slice", "pad"])
def test_slice_pad_concat_rejects_public_branch_intermediate(
    role: str,
) -> None:
    model_ir = _model()
    name = (
        "branch0_x0_slice" if role == "slice" else "branch0_x0_pad"
    )
    model_ir.outputs.append(name)

    _assert_transactional_rejection(model_ir)


@pytest.mark.parametrize(
    "case",
    [
        "duplicate-post-output",
        "reverse-post-add",
        "reverse-slice-pad",
        "public-pre-input",
        "reverse-source-pre",
        "duplicate-source-producer",
    ],
)
def test_slice_pad_concat_rejects_invalid_topology(case: str) -> None:
    model_ir = _model()
    if case == "duplicate-post-output":
        model_ir.operators.insert(
            8,
            OperatorIR(
                "IDENTITY",
                ["branch0_x0_nhwc"],
                ["branch0_mul_out_nhwc"],
            ),
        )
    elif case == "reverse-post-add":
        model_ir.operators[8], model_ir.operators[9] = (
            model_ir.operators[9],
            model_ir.operators[8],
        )
    elif case == "reverse-slice-pad":
        model_ir.operators[1], model_ir.operators[2] = (
            model_ir.operators[2],
            model_ir.operators[1],
        )
    elif case == "public-pre-input":
        model_ir.inputs.append("branch0_x0_nchw")
    elif case == "reverse-source-pre":
        model_ir.operators.append(
            OperatorIR(
                "IDENTITY",
                ["branch0_x1_nhwc"],
                ["branch0_x0_nhwc"],
            )
        )
    elif case == "duplicate-source-producer":
        model_ir.operators.extend(
            [
                OperatorIR(
                    "IDENTITY",
                    ["branch0_x1_nhwc"],
                    ["branch0_x0_nhwc"],
                ),
                OperatorIR(
                    "IDENTITY",
                    ["branch0_x0_nhwc"],
                    ["branch0_x0_nhwc"],
                ),
            ]
        )

    _assert_transactional_rejection(model_ir)


@pytest.mark.parametrize("case", ["concat-axis", "slice-mask"])
def test_slice_pad_concat_rejects_malformed_options(case: str) -> None:
    model_ir = _model()
    if case == "concat-axis":
        model_ir.operators[6].options["axis"] = None
    else:
        model_ir.operators[1].options["beginMask"] = None

    _assert_transactional_rejection(model_ir)


def test_slice_pad_concat_keeps_owner_wrapper_and_ordered_calls() -> None:
    pass_path = (
        REPO_ROOT
        / "onnx2tf"
        / "tflite_builder"
        / "passes"
        / "stridedslice_pad_concat_bridge_layout.py"
    )
    pass_tree = ast.parse(pass_path.read_text(encoding="utf-8"))
    owner = next(
        node
        for node in pass_tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name
        == "_optimize_transpose_stridedslice_pad_concat_mul_add_posttranspose_nhwc_chains"
    )
    assert owner.end_lineno - owner.lineno + 1 == 1100
    assert sum(isinstance(node, ast.While) for node in ast.walk(owner)) == 2
    assert not any(
        isinstance(node, (ast.Import, ast.ImportFrom))
        and (
            any(
                alias.name == "onnx2tf.tflite_builder.lower_from_onnx2tf"
                for alias in node.names
            )
            if isinstance(node, ast.Import)
            else node.module == "onnx2tf.tflite_builder.lower_from_onnx2tf"
        )
        for node in pass_tree.body
    )

    lowering_path = (
        REPO_ROOT / "onnx2tf" / "tflite_builder" / "lower_from_onnx2tf.py"
    )
    lowering_tree = ast.parse(lowering_path.read_text(encoding="utf-8"))
    wrapper = next(
        node
        for node in lowering_tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name
        == "_optimize_transpose_stridedslice_pad_concat_mul_add_posttranspose_nhwc_chains"
    )
    assert len(wrapper.body) == 1
    assert isinstance(wrapper.body[0], ast.Return)
    wrapper_call = wrapper.body[0].value
    assert isinstance(wrapper_call, ast.Call)
    assert isinstance(wrapper_call.func, ast.Name)
    assert (
        wrapper_call.func.id
        == "_optimize_transpose_stridedslice_pad_concat_mul_add_posttranspose_nhwc_chains_pass"
    )
    assert len(wrapper_call.args) == 1
    assert isinstance(wrapper_call.args[0], ast.Name)
    assert wrapper_call.args[0].id == "model_ir"
    assert wrapper_call.keywords == []

    lowerer = next(
        node
        for node in lowering_tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "lower_onnx_to_ir"
    )
    calls = [
        node
        for node in ast.walk(lowerer)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id
        == "_optimize_transpose_stridedslice_pad_concat_mul_add_posttranspose_nhwc_chains"
    ]
    assert (
        len(calls)
        + TERMINAL_SLICE_CONCAT_RECOVERY_PASS_IDS.count(
            "_optimize_transpose_stridedslice_pad_concat_mul_add_posttranspose_nhwc_chains"
        )
        == 3
    )
    for call in calls:
        assert len(call.args) == 1
        assert isinstance(call.args[0], ast.Name)
        assert call.args[0].id == "model_ir"
        assert call.keywords == []


@pytest.mark.xfail(
    strict=True,
    reason="the second terminal slice/pad/concat result is still discarded",
)
def test_second_terminal_slice_pad_concat_captures_complete_mutation_evidence() -> None:
    lowering_tree = ast.parse(
        (
            REPO_ROOT
            / "onnx2tf"
            / "tflite_builder"
            / "lower_from_onnx2tf.py"
        ).read_text(encoding="utf-8")
    )
    lowerer = next(
        node
        for node in lowering_tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "lower_onnx_to_ir"
    )
    invocation_index = next(
        index
        for index, statement in enumerate(lowerer.body)
        if isinstance(statement, ast.Assign)
        and len(statement.targets) == 1
        and isinstance(statement.targets[0], ast.Name)
        and statement.targets[0].id == "_terminal_slice_pad_concat_stats"
    )
    invocation = lowerer.body[invocation_index]
    assert isinstance(invocation, ast.Assign)
    assert isinstance(invocation.value, ast.Call)
    assert isinstance(invocation.value.func, ast.Name)
    assert invocation.value.func.id == (
        "_optimize_transpose_stridedslice_pad_concat_mul_add_posttranspose_nhwc_chains"
    )
    assert len(invocation.value.args) == 1
    assert isinstance(invocation.value.args[0], ast.Name)
    assert invocation.value.args[0].id == "model_ir"
    assert invocation.value.keywords == []

    previous = lowerer.body[invocation_index - 1]
    assert isinstance(previous, ast.Expr)
    assert isinstance(previous.value, ast.Call)
    assert isinstance(previous.value.func, ast.Name)
    assert previous.value.func.id == (
        "_run_terminal_affine_concat_split_recovery_sequence"
    )
    following = lowerer.body[invocation_index + 1]
    assert isinstance(following, ast.Assign)
    assert len(following.targets) == 1
    assert isinstance(following.targets[0], ast.Name)
    assert following.targets[0].id == "late_spp_results"

    direct_statements = [
        statement
        for statement in lowerer.body
        if any(
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id
            == "_optimize_transpose_stridedslice_pad_concat_mul_add_posttranspose_nhwc_chains"
            for node in ast.walk(statement)
        )
    ]
    assert len(direct_statements) == 2
    assert isinstance(direct_statements[0], ast.Expr)
    assert direct_statements[1] is invocation
