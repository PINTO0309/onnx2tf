from __future__ import annotations

import ast
import copy
from dataclasses import fields, is_dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pytest

from onnx2tf.tflite_builder.core.validation import validate_model_ir_invariants
from onnx2tf.tflite_builder.ir import (
    ModelIR,
    OperatorIR,
    QuantParamIR,
    TensorIR,
)
from onnx2tf.tflite_builder.lower_from_onnx2tf import (
    _canonicalize_softmax_transpose_chains,
)
from onnx2tf.tflite_builder.passes.softmax_transpose_canonicalization import (
    _canonicalize_softmax_transpose_chains as _canonicalize_softmax_transpose_chains_owner,
)
from onnx2tf.tflite_builder.passes.quantized_recovery_orchestration import (
    QUANTIZED_ACTIVATION_BINARY_PASS_IDS,
)
from onnx2tf.tflite_builder.passes.layout_attention_quantized_suffix_orchestration import (
    LAYOUT_ATTENTION_QUANTIZED_SUFFIX_PASS_IDS,
)
from onnx2tf.tflite_builder.passes.terminal_softmax_layout import (
    _SOFTMAX_NHWC_PROPAGATED_MARKER,
)


REPO_ROOT = Path(__file__).resolve().parents[1]


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
    is_variable: bool | None = None,
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
        is_variable=data is None if is_variable is None else bool(is_variable),
        quantization=quantization,
    )


def _add_branch(
    model_ir: ModelIR,
    *,
    prefix: str,
    terminal: bool,
    shared_inner_permutation: bool = False,
) -> None:
    source = f"{prefix}source"
    pre_previous_output = f"{prefix}pre_previous_output"
    softmax_input = f"{prefix}softmax_input"
    softmax_output = f"{prefix}softmax_output"
    post_output = f"{prefix}post_output"
    final_output = post_output if terminal else f"{prefix}final_output"
    pre_previous_permutation = f"{prefix}perm_nhwc_to_nchw"
    pre_permutation = f"{prefix}perm_nchw_to_nwhc"
    post_permutation = (
        pre_permutation
        if shared_inner_permutation
        else f"{prefix}post_perm_nchw_to_nwhc"
    )

    model_ir.inputs.append(source)
    model_ir.outputs.append(final_output)
    _tensor(
        model_ir,
        source,
        [1, 2, 3, 4],
        signature=[-1, 2, 3, 4],
    )
    _tensor(
        model_ir,
        pre_previous_output,
        [1, 4, 2, 3],
        signature=[-1, 4, 2, 3],
    )
    _tensor(
        model_ir,
        softmax_input,
        [1, 3, 2, 4],
        signature=[-1, 3, 2, 4],
    )
    _tensor(
        model_ir,
        softmax_output,
        [1, 3, 2, 4],
        signature=[-1, 3, 2, 4],
    )
    _tensor(
        model_ir,
        post_output,
        [1, 4, 2, 3],
        signature=[-1, 4, 2, 3],
    )
    if not terminal:
        _tensor(
            model_ir,
            final_output,
            [1, 4, 2, 3],
            signature=[-1, 4, 2, 3],
        )

    _tensor(
        model_ir,
        pre_previous_permutation,
        [4],
        dtype="INT32",
        data=np.asarray([0, 3, 1, 2], dtype=np.int32),
    )
    _tensor(
        model_ir,
        pre_permutation,
        [4],
        dtype="INT32",
        data=np.asarray([0, 3, 2, 1], dtype=np.int32),
    )
    if not shared_inner_permutation:
        _tensor(
            model_ir,
            post_permutation,
            [4],
            dtype="INT32",
            data=np.asarray([0, 3, 2, 1], dtype=np.int32),
        )

    model_ir.operators.extend(
        [
            OperatorIR(
                "TRANSPOSE",
                [source, pre_previous_permutation],
                [pre_previous_output],
            ),
            OperatorIR(
                "TRANSPOSE",
                [pre_previous_output, pre_permutation],
                [softmax_input],
            ),
            OperatorIR(
                "SOFTMAX",
                [softmax_input],
                [softmax_output],
                options={"axis": 3, "beta": 0.75},
                axis_semantics={"axis": "physical"},
                version=2,
                onnx_node_name=f"{prefix}softmax",
                onnx_op_type="Softmax",
            ),
            OperatorIR(
                "TRANSPOSE",
                [softmax_output, post_permutation],
                [post_output],
            ),
        ]
    )
    if not terminal:
        model_ir.operators.append(
            OperatorIR("IDENTITY", [post_output], [final_output])
        )


def _model(
    *,
    branches: int = 1,
    terminal: bool = False,
    shared_inner_permutation: bool = False,
) -> ModelIR:
    model_ir = ModelIR("softmax_transpose_canonicalization")
    for branch_index in range(int(branches)):
        _add_branch(
            model_ir,
            prefix=f"branch{branch_index}_",
            terminal=terminal,
            shared_inner_permutation=shared_inner_permutation,
        )
    return model_ir


def _assert_transactional_rejection(model_ir: ModelIR) -> None:
    before = _normalize(copy.deepcopy(model_ir))

    stats = _canonicalize_softmax_transpose_chains(model_ir)

    assert stats == {"canonicalized_softmax_transpose_chains": 0}
    assert _normalize(model_ir) == before


def _owner_wrapper_case(case: str) -> ModelIR:
    if case == "multiple":
        return _model(branches=2)
    if case == "shared-permutation":
        return _model(shared_inner_permutation=True)
    if case == "terminal":
        return _model(terminal=True)

    model_ir = _model()
    if case == "public-permutation-output":
        model_ir.outputs.append("branch0_perm_nchw_to_nwhc")
    elif case == "negative-last-axis":
        model_ir.operators[2].options["axis"] = -1
    elif case == "pruning":
        _tensor(model_ir, "unused", [1])
        model_ir.tensors["branch0_perm_nchw_to_nwhc"].data = np.asarray(
            [0, 2, 3, 1],
            dtype=np.int32,
        )
    elif case == "unsafe-axis":
        model_ir.operators[2].options["axis"] = 1
    elif case == "incomplete-metadata":
        model_ir.tensors["branch0_softmax_output"].shape_signature = [1, 3, 2]
    elif case == "post-plan-rejection":
        model_ir.tensors[
            "branch0_post_perm_nchw_to_nwhc"
        ].is_variable = True
    elif case != "static-dynamic-signature":
        raise ValueError(f"unsupported owner/wrapper case: {case}")
    return model_ir


@pytest.mark.parametrize(
    "case",
    [
        "static-dynamic-signature",
        "multiple",
        "shared-permutation",
        "public-permutation-output",
        "negative-last-axis",
        "terminal",
        "pruning",
        "unsafe-axis",
        "incomplete-metadata",
        "post-plan-rejection",
    ],
)
def test_softmax_transpose_owner_matches_lowerer_wrapper(case: str) -> None:
    owner_model_ir = _owner_wrapper_case(case)
    wrapper_model_ir = copy.deepcopy(owner_model_ir)

    owner_stats = _canonicalize_softmax_transpose_chains_owner(owner_model_ir)
    wrapper_stats = _canonicalize_softmax_transpose_chains(wrapper_model_ir)

    assert owner_stats == wrapper_stats
    assert _normalize(owner_model_ir) == _normalize(wrapper_model_ir)


def test_softmax_transpose_canonicalization_preserves_operator_contract() -> None:
    model_ir = _model()
    softmax = model_ir.operators[2]

    stats = _canonicalize_softmax_transpose_chains(model_ir)

    assert stats == {"canonicalized_softmax_transpose_chains": 1}
    assert np.asarray(
        model_ir.tensors[model_ir.operators[0].inputs[1]].data
    ).tolist() == [0, 3, 1, 2]
    assert np.asarray(
        model_ir.tensors[model_ir.operators[1].inputs[1]].data
    ).tolist() == [0, 2, 3, 1]
    assert np.asarray(
        model_ir.tensors[model_ir.operators[3].inputs[1]].data
    ).tolist() == [0, 3, 1, 2]
    assert softmax.options == {
        "axis": 3,
        "beta": 0.75,
        _SOFTMAX_NHWC_PROPAGATED_MARKER: True,
    }
    assert softmax.axis_semantics == {"axis": "physical"}
    assert softmax.version == 2
    assert softmax.onnx_node_name == "branch0_softmax"
    assert softmax.onnx_op_type == "Softmax"
    assert model_ir.tensors["branch0_softmax_input"].shape == [1, 2, 3, 4]
    assert model_ir.tensors["branch0_softmax_input"].shape_signature == [
        -1,
        2,
        3,
        4,
    ]
    assert validate_model_ir_invariants(model_ir) == []


def test_softmax_transpose_canonicalization_accepts_negative_last_axis() -> None:
    model_ir = _model()
    model_ir.operators[2].options["axis"] = -1

    stats = _canonicalize_softmax_transpose_chains(model_ir)

    assert stats == {"canonicalized_softmax_transpose_chains": 1}
    assert model_ir.operators[2].options["axis"] == -1


def test_softmax_transpose_canonicalization_rewrites_terminal_output() -> None:
    model_ir = _model(terminal=True)

    stats = _canonicalize_softmax_transpose_chains(model_ir)

    assert stats == {"canonicalized_softmax_transpose_chains": 1}
    assert model_ir.outputs == ["branch0_post_output"]
    assert model_ir.operators[-1].outputs == ["branch0_post_output"]


def test_softmax_transpose_canonicalization_reaches_fixed_point_in_graph_order() -> None:
    model_ir = _model(branches=2)

    first_stats = _canonicalize_softmax_transpose_chains(model_ir)
    after_first = _normalize(copy.deepcopy(model_ir))
    second_stats = _canonicalize_softmax_transpose_chains(model_ir)

    assert first_stats == {"canonicalized_softmax_transpose_chains": 2}
    assert second_stats == {"canonicalized_softmax_transpose_chains": 0}
    assert _normalize(model_ir) == after_first
    marked_softmaxes = [
        operator
        for operator in model_ir.operators
        if operator.op_type == "SOFTMAX"
        and operator.options.get(_SOFTMAX_NHWC_PROPAGATED_MARKER) is True
    ]
    assert len(marked_softmaxes) == 2


def test_softmax_transpose_canonicalization_clones_shared_permutation() -> None:
    model_ir = _model(shared_inner_permutation=True)

    stats = _canonicalize_softmax_transpose_chains(model_ir)

    assert stats == {"canonicalized_softmax_transpose_chains": 1}
    pre_permutation = model_ir.operators[1].inputs[1]
    post_permutation = model_ir.operators[3].inputs[1]
    assert pre_permutation == "branch0_perm_nchw_to_nwhc_canon"
    assert post_permutation == "branch0_perm_nchw_to_nwhc_canon_1"
    assert pre_permutation != post_permutation
    assert np.asarray(model_ir.tensors[pre_permutation].data).tolist() == [
        0,
        2,
        3,
        1,
    ]
    assert np.asarray(model_ir.tensors[post_permutation].data).tolist() == [
        0,
        3,
        1,
        2,
    ]
    assert "branch0_perm_nchw_to_nwhc" not in model_ir.tensors
    lineage = model_ir.metadata["tensor_lineage_events"]
    assert [event["kind"] for event in lineage] == [
        "replace_input",
        "replace_input",
        "prune_unused_tensors",
    ]


def test_softmax_transpose_canonicalization_prunes_on_unmatched_graph() -> None:
    model_ir = _model()
    model_ir.tensors["unused"] = TensorIR(
        name="unused",
        dtype="FLOAT32",
        shape=[1],
        shape_signature=[1],
    )
    model_ir.tensors["branch0_perm_nchw_to_nwhc"].data = np.asarray(
        [0, 2, 3, 1],
        dtype=np.int32,
    )

    stats = _canonicalize_softmax_transpose_chains(model_ir)

    assert stats == {"canonicalized_softmax_transpose_chains": 0}
    assert "unused" not in model_ir.tensors
    assert model_ir.metadata["tensor_lineage_events"] == [
        {
            "kind": "prune_unused_tensors",
            "removed_names": ["unused"],
            "event_index": 0,
        }
    ]


@pytest.mark.parametrize(
    "mutate",
    [
        pytest.param(
            lambda model_ir: model_ir.tensors[
                "branch0_perm_nchw_to_nwhc"
            ].__setattr__(
                "data",
                np.asarray([0, 2, 3, 1], dtype=np.int32),
            ),
            id="wrong-pre-permutation",
        ),
        pytest.param(
            lambda model_ir: model_ir.tensors[
                "branch0_post_perm_nchw_to_nwhc"
            ].__setattr__(
                "data",
                np.asarray([0, 2, 3, 1], dtype=np.int32),
            ),
            id="wrong-post-permutation",
        ),
        pytest.param(
            lambda model_ir: model_ir.operators[2].inputs.append(
                "branch0_softmax_input"
            ),
            id="softmax-input-arity",
        ),
        pytest.param(
            lambda model_ir: model_ir.operators[3].outputs.append(
                "branch0_final_output"
            ),
            id="post-output-arity",
        ),
        pytest.param(
            lambda model_ir: model_ir.outputs.append(
                "branch0_softmax_input"
            ),
            id="public-softmax-input",
        ),
        pytest.param(
            lambda model_ir: model_ir.outputs.append(
                "branch0_softmax_output"
            ),
            id="public-softmax-output",
        ),
        pytest.param(
            lambda model_ir: (
                _tensor(model_ir, "side_output", [1, 3, 2, 4]),
                model_ir.outputs.append("side_output"),
                model_ir.operators.append(
                    OperatorIR(
                        "IDENTITY",
                        ["branch0_softmax_input"],
                        ["side_output"],
                    )
                ),
            ),
            id="pre-fanout",
        ),
        pytest.param(
            lambda model_ir: (
                _tensor(model_ir, "side_output", [1, 3, 2, 4]),
                model_ir.outputs.append("side_output"),
                model_ir.operators.append(
                    OperatorIR(
                        "IDENTITY",
                        ["branch0_softmax_output"],
                        ["side_output"],
                    )
                ),
            ),
            id="softmax-fanout",
        ),
        pytest.param(
            lambda model_ir: (
                model_ir.outputs.append("branch0_post_output"),
            ),
            id="public-consumed-post-output",
        ),
        pytest.param(
            lambda model_ir: model_ir.tensors[
                "branch0_pre_previous_output"
            ].__setattr__(
                "quantization",
                QuantParamIR(
                    scale=[0.1, 0.2, 0.3, 0.4],
                    zero_point=[0, 0, 0, 0],
                    quantized_dimension=1,
                ),
            ),
            id="per-axis-quantization",
        ),
    ],
)
def test_softmax_transpose_canonicalization_rejects_existing_unsafe_contracts(
    mutate: Callable[[ModelIR], object],
) -> None:
    model_ir = _model()
    mutate(model_ir)

    _assert_transactional_rejection(model_ir)


def test_softmax_transpose_canonicalization_updates_complete_metadata() -> None:
    model_ir = _model()

    stats = _canonicalize_softmax_transpose_chains(model_ir)

    assert stats == {"canonicalized_softmax_transpose_chains": 1}
    for name in ("branch0_softmax_input", "branch0_softmax_output"):
        assert model_ir.tensors[name].shape == [1, 2, 3, 4]
        assert model_ir.tensors[name].shape_signature == [-1, 2, 3, 4]
    assert model_ir.tensors["branch0_post_output"].shape == [1, 4, 2, 3]
    assert model_ir.tensors["branch0_post_output"].shape_signature == [
        -1,
        4,
        2,
        3,
    ]


@pytest.mark.parametrize("axis", [0, 1, 2, -2, 4, "invalid"])
def test_softmax_transpose_canonicalization_rejects_unsafe_axis(
    axis: object,
) -> None:
    model_ir = _model()
    model_ir.operators[2].options["axis"] = axis

    _assert_transactional_rejection(model_ir)


@pytest.mark.parametrize(
    "case",
    [
        "missing-pre-input",
        "missing-softmax-input",
        "missing-softmax-output",
        "missing-post-output",
        "rank-three-pre-input",
        "short-softmax-input-signature",
        "short-softmax-output-signature",
    ],
)
def test_softmax_transpose_canonicalization_rejects_incomplete_metadata(
    case: str,
) -> None:
    model_ir = _model()
    if case == "missing-pre-input":
        del model_ir.tensors["branch0_pre_previous_output"]
    elif case == "missing-softmax-input":
        del model_ir.tensors["branch0_softmax_input"]
    elif case == "missing-softmax-output":
        del model_ir.tensors["branch0_softmax_output"]
    elif case == "missing-post-output":
        del model_ir.tensors["branch0_post_output"]
    elif case == "rank-three-pre-input":
        model_ir.tensors["branch0_pre_previous_output"].shape = [1, 4, 6]
    elif case == "short-softmax-input-signature":
        model_ir.tensors["branch0_softmax_input"].shape_signature = [1, 3, 2]
    elif case == "short-softmax-output-signature":
        model_ir.tensors["branch0_softmax_output"].shape_signature = [1, 3, 2]

    _assert_transactional_rejection(model_ir)


@pytest.mark.parametrize(
    "case",
    [
        "public-input",
        "variable",
        "tensor-dtype",
        "buffer-dtype",
        "quantized",
        "post-variable",
    ],
)
def test_softmax_transpose_canonicalization_rejects_unsafe_permutation_owner(
    case: str,
) -> None:
    model_ir = _model()
    permutation = model_ir.tensors["branch0_perm_nchw_to_nwhc"]
    if case == "public-input":
        model_ir.inputs.append(permutation.name)
    elif case == "variable":
        permutation.is_variable = True
    elif case == "tensor-dtype":
        permutation.dtype = "FLOAT32"
    elif case == "buffer-dtype":
        permutation.data = np.asarray([0, 3, 2, 1], dtype=np.int64)
    elif case == "quantized":
        permutation.quantization = QuantParamIR(scale=[1.0], zero_point=[0])
    elif case == "post-variable":
        model_ir.tensors[
            "branch0_post_perm_nchw_to_nwhc"
        ].is_variable = True

    _assert_transactional_rejection(model_ir)


def test_softmax_transpose_canonicalization_clones_public_permutation_output() -> None:
    model_ir = _model()
    permutation_name = "branch0_perm_nchw_to_nwhc"
    original_data = np.asarray(
        model_ir.tensors[permutation_name].data
    ).copy()
    model_ir.outputs.append(permutation_name)

    stats = _canonicalize_softmax_transpose_chains(model_ir)

    assert stats == {"canonicalized_softmax_transpose_chains": 1}
    assert model_ir.operators[1].inputs[1] != permutation_name
    assert np.array_equal(model_ir.tensors[permutation_name].data, original_data)


@pytest.mark.parametrize(
    "case",
    [
        "duplicate-softmax-output",
        "duplicate-post-output",
        "reverse-softmax-post",
        "public-internal-input",
    ],
)
def test_softmax_transpose_canonicalization_rejects_invalid_topology(
    case: str,
) -> None:
    model_ir = _model()
    if case == "duplicate-softmax-output":
        model_ir.operators.insert(
            3,
            OperatorIR(
                "IDENTITY",
                ["branch0_source"],
                ["branch0_softmax_output"],
            ),
        )
    elif case == "duplicate-post-output":
        model_ir.operators.insert(
            4,
            OperatorIR(
                "IDENTITY",
                ["branch0_source"],
                ["branch0_post_output"],
            ),
        )
    elif case == "reverse-softmax-post":
        model_ir.operators[2], model_ir.operators[3] = (
            model_ir.operators[3],
            model_ir.operators[2],
        )
    elif case == "public-internal-input":
        model_ir.inputs.append("branch0_softmax_input")

    _assert_transactional_rejection(model_ir)


def test_softmax_transpose_canonicalization_keeps_ordered_boundaries() -> None:
    lowering_path = (
        REPO_ROOT / "onnx2tf" / "tflite_builder" / "lower_from_onnx2tf.py"
    )
    owner_path = (
        REPO_ROOT
        / "onnx2tf"
        / "tflite_builder"
        / "passes"
        / "softmax_transpose_canonicalization.py"
    )
    owner_source = owner_path.read_text(encoding="utf-8")
    assert "lower_from_onnx2tf" not in owner_source
    owner_tree = ast.parse(owner_source)
    owner = next(
        node
        for node in owner_tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "_canonicalize_softmax_transpose_chains"
    )
    assert any(isinstance(node, ast.While) for node in ast.walk(owner))

    lowering_tree = ast.parse(lowering_path.read_text(encoding="utf-8"))
    wrapper = next(
        node
        for node in lowering_tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "_canonicalize_softmax_transpose_chains"
    )
    assert len(wrapper.body) == 1
    assert isinstance(wrapper.body[0], ast.Return)
    assert isinstance(wrapper.body[0].value, ast.Call)
    assert isinstance(wrapper.body[0].value.func, ast.Name)
    assert (
        wrapper.body[0].value.func.id
        == "_canonicalize_softmax_transpose_chains_pass"
    )
    assert len(wrapper.body[0].value.args) == 1
    assert isinstance(wrapper.body[0].value.args[0], ast.Name)
    assert wrapper.body[0].value.args[0].id == "model_ir"
    assert wrapper.body[0].value.keywords == []

    canonicalization_index = QUANTIZED_ACTIVATION_BINARY_PASS_IDS.index(
        "_canonicalize_softmax_transpose_chains"
    )
    assert (
        QUANTIZED_ACTIVATION_BINARY_PASS_IDS[canonicalization_index - 1],
        QUANTIZED_ACTIVATION_BINARY_PASS_IDS[canonicalization_index + 1],
    ) == (
        "_optimize_dequant_logistic_quantize_chains",
        "_run_safe_binary_bridge_recovery_sequence",
    )

    suffix_index = LAYOUT_ATTENTION_QUANTIZED_SUFFIX_PASS_IDS.index(
        "_canonicalize_softmax_transpose_chains"
    )
    assert (
        LAYOUT_ATTENTION_QUANTIZED_SUFFIX_PASS_IDS[suffix_index - 1],
        (
            LAYOUT_ATTENTION_QUANTIZED_SUFFIX_PASS_IDS[suffix_index + 1]
            if suffix_index + 1
            < len(LAYOUT_ATTENTION_QUANTIZED_SUFFIX_PASS_IDS)
            else None
        ),
    ) == ("_optimize_dequant_logistic_quantize_chains", None)
