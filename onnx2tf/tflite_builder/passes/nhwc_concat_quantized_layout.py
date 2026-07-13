from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from onnx2tf.tflite_builder.core.graph import ModelIRGraphIndex
from onnx2tf.tflite_builder.core.layout import LayoutState
from onnx2tf.tflite_builder.core.model_ir_pass_state import (
    ModelIRPassState,
    preflight_required_op_types,
    run_model_ir_pass_group,
)
from onnx2tf.tflite_builder.core.model_ir_utils import (
    _clone_quantization,
    _permute_tensor_metadata_if_rank_matches,
    _prune_unused_tensors,
    _read_transpose_perm,
    _replace_tensor_inputs,
    _set_operator_inputs,
    _set_operator_outputs,
)
from onnx2tf.tflite_builder.core.passes import PassPhase, PassSpec
from onnx2tf.tflite_builder.ir import ModelIR, OperatorIR, QuantParamIR
from onnx2tf.tflite_builder.passes.nhwc_concat_layout import (
    _NhwcConcatInputPlan,
    _apply_add_input_plan,
    _apply_dequantize_input_plan,
    _apply_leaky_input_plan,
    _apply_prelu_input_plan,
    _apply_softmax_input_plan,
    _apply_slice_input_plan,
    _apply_split_input_plan,
    _apply_swish_input_plan,
    _resolve_dequantize_input_plan,
    _resolve_add_input_plan,
    _resolve_leaky_input_plan,
    _resolve_prelu_input_plan,
    _resolve_softmax_input_plan,
    _resolve_slice_input_plan,
    _resolve_split_input_plan,
    _resolve_swish_input_plan,
)
from onnx2tf.tflite_builder.passes.nhwc_concat_pad import (
    NhwcConcatPadPlan,
    apply_nhwc_concat_pad_plan,
    resolve_nhwc_concat_pad_plan,
)


_PERM_NHWC_TO_NCHW = [0, 3, 1, 2]
_PERM_NCHW_TO_NHWC = [0, 2, 3, 1]
_DIRECT_STATS_KEY = (
    "optimized_transpose_pre_concat_nhwc_quantized_direct_chains"
)
_UNARY_STATS_KEY = (
    "optimized_transpose_pre_concat_nhwc_quantized_unary_chains"
)
_PAD_STATS_KEY = "optimized_transpose_pre_concat_nhwc_quantized_pad_chains"
_UNARY_PAD_STATS_KEY = (
    "optimized_transpose_pre_concat_nhwc_quantized_unary_pad_chains"
)
_ALL_PAD_STATS_KEY = (
    "optimized_transpose_pre_concat_nhwc_quantized_all_pad_chains"
)
_SWISH_STATS_KEY = (
    "optimized_transpose_pre_concat_nhwc_quantized_swish_chains"
)
_DEQUANTIZE_STATS_KEY = (
    "optimized_transpose_pre_concat_nhwc_quantized_dequantize_chains"
)
_PRELU_STATS_KEY = (
    "optimized_transpose_pre_concat_nhwc_quantized_prelu_chains"
)
_SOFTMAX_STATS_KEY = (
    "optimized_transpose_pre_concat_nhwc_quantized_softmax_chains"
)
_LEAKY_STATS_KEY = (
    "optimized_transpose_pre_concat_nhwc_quantized_leaky_chains"
)
_SLICE_STATS_KEY = (
    "optimized_transpose_pre_concat_nhwc_quantized_slice_chains"
)
_SPLIT_STATS_KEY = (
    "optimized_transpose_pre_concat_nhwc_quantized_split_chains"
)
_ADD_STATS_KEY = (
    "optimized_transpose_pre_concat_nhwc_quantized_add_chains"
)
_QUANTIZED_FAMILY_SPECS: Tuple[Tuple[str, str, int], ...] = (
    ("direct", _DIRECT_STATS_KEY, 10),
    ("unary", _UNARY_STATS_KEY, 20),
    ("pad", _PAD_STATS_KEY, 30),
    ("unary_pad", _UNARY_PAD_STATS_KEY, 40),
    ("all_pad", _ALL_PAD_STATS_KEY, 50),
    ("swish", _SWISH_STATS_KEY, 60),
    ("dequantize", _DEQUANTIZE_STATS_KEY, 70),
    ("prelu", _PRELU_STATS_KEY, 80),
    ("softmax", _SOFTMAX_STATS_KEY, 90),
    ("leaky", _LEAKY_STATS_KEY, 100),
    ("slice", _SLICE_STATS_KEY, 110),
    ("split", _SPLIT_STATS_KEY, 120),
    ("add", _ADD_STATS_KEY, 130),
)
_UNARY_OPS = {"RELU", "RELU6", "LOGISTIC", "TANH", "GELU"}


@dataclass(frozen=True)
class _QuantizedInputPlan:
    kind: str
    adapter_op: OperatorIR
    concat_input: str
    source_name: str
    remove_adapter: bool
    unary_op: Optional[OperatorIR] = None
    pad_plan: Optional[NhwcConcatPadPlan] = None
    nhwc_plan: Optional[_NhwcConcatInputPlan] = None


@dataclass(frozen=True)
class _QuantizedConcatCandidate:
    input_plans: Tuple[_QuantizedInputPlan, ...]
    concat_op: OperatorIR
    concat_output: str
    quantize_op: OperatorIR
    quantized_output: str
    post_ops: Tuple[OperatorIR, ...]
    post_outputs: Tuple[str, ...]


@dataclass(frozen=True)
class _QuantizedInputContract:
    allowed_kinds: frozenset[str]
    required_kinds: frozenset[str]
    minimum_inputs: int = 1
    exact_kind_counts: Tuple[Tuple[str, int], ...] = ()
    validate_shapes: bool = True


def _input_contract(
    *allowed_kinds: str,
    required_kinds: Tuple[str, ...],
    minimum_inputs: int = 1,
    exact_kind_counts: Tuple[Tuple[str, int], ...] = (),
    validate_shapes: bool = True,
) -> _QuantizedInputContract:
    return _QuantizedInputContract(
        allowed_kinds=frozenset(allowed_kinds),
        required_kinds=frozenset(required_kinds),
        minimum_inputs=minimum_inputs,
        exact_kind_counts=exact_kind_counts,
        validate_shapes=validate_shapes,
    )


_QUANTIZED_INPUT_CONTRACTS = {
    "direct": _input_contract("direct", required_kinds=("direct",), validate_shapes=False),
    "unary": _input_contract("direct", "unary", required_kinds=("unary",)),
    "pad": _input_contract("direct", "pad", required_kinds=("direct", "pad")),
    "unary_pad": _input_contract("direct", "unary", "pad", required_kinds=("unary", "pad")),
    "all_pad": _input_contract("pad", required_kinds=("pad",), minimum_inputs=2),
    "swish": _input_contract("direct", "unary", "swish", required_kinds=("swish",)),
    "dequantize": _input_contract("direct", "dequantize", required_kinds=("dequantize",)),
    "prelu": _input_contract("direct", "prelu", required_kinds=("prelu",)),
    "softmax": _input_contract(
        "direct",
        "softmax",
        required_kinds=("direct", "softmax"),
        exact_kind_counts=(("softmax", 1),),
    ),
    "leaky": _input_contract("direct", "unary", "leaky", required_kinds=("leaky",)),
    "slice": _input_contract("direct", "slice", required_kinds=("slice",)),
    "split": _input_contract("direct", "split", required_kinds=("split",)),
    "add": _input_contract("direct", "unary", "add", required_kinds=("add",)),
}
if {
    family for family, _, _ in _QUANTIZED_FAMILY_SPECS
} != set(_QUANTIZED_INPUT_CONTRACTS):
    raise RuntimeError("quantized Concat pass and input contracts diverged")


def _input_plans_satisfy_contract(
    input_plans: List[_QuantizedInputPlan],
    contract: _QuantizedInputContract,
) -> bool:
    if len(input_plans) < contract.minimum_inputs:
        return False
    kind_counts: Dict[str, int] = {}
    for input_plan in input_plans:
        kind_counts[input_plan.kind] = kind_counts.get(input_plan.kind, 0) + 1
    kinds = frozenset(kind_counts)
    return (
        kinds.issubset(contract.allowed_kinds)
        and contract.required_kinds.issubset(kinds)
        and all(
            kind_counts.get(kind, 0) == expected_count
            for kind, expected_count in contract.exact_kind_counts
        )
    )


def _clone_permuted_quantization(
    quantization: Any,
    permutation: List[int],
) -> Any:
    cloned = _clone_quantization(quantization)
    if isinstance(cloned, QuantParamIR):
        old_dimension = int(cloned.quantized_dimension)
        if 0 <= old_dimension < len(permutation):
            cloned.quantized_dimension = int(permutation.index(old_dimension))
    elif isinstance(cloned, dict) and "quantized_dimension" in cloned:
        old_dimension = int(cloned["quantized_dimension"])
        if 0 <= old_dimension < len(permutation):
            cloned["quantized_dimension"] = int(
                permutation.index(old_dimension)
            )
    return cloned


def _resolve_direct_input_plan(
    model_ir: ModelIR,
    graph_index: ModelIRGraphIndex,
    *,
    concat_input: str,
    concat_index: int,
    model_outputs: set[str],
) -> Optional[_QuantizedInputPlan]:
    adapter_op = graph_index.producer(concat_input)
    if (
        adapter_op is None
        or str(adapter_op.op_type) != "TRANSPOSE"
        or len(adapter_op.inputs) < 2
        or len(adapter_op.outputs) != 1
        or str(adapter_op.outputs[0]) != concat_input
        or _read_transpose_perm(model_ir, adapter_op)
        != _PERM_NHWC_TO_NCHW
    ):
        return None
    source_name = str(adapter_op.inputs[0])
    source_tensor = model_ir.tensors.get(source_name)
    input_consumers = set(graph_index.consumer_indices(concat_input))
    if (
        source_tensor is None
        or len(list(source_tensor.shape)) != 4
        or int(concat_index) not in input_consumers
    ):
        return None
    return _QuantizedInputPlan(
        kind="direct",
        adapter_op=adapter_op,
        concat_input=concat_input,
        source_name=source_name,
        remove_adapter=(
            input_consumers == {int(concat_index)}
            and concat_input not in model_outputs
        ),
    )


def _resolve_unary_input_plan(
    model_ir: ModelIR,
    graph_index: ModelIRGraphIndex,
    *,
    concat_input: str,
    concat_index: int,
    model_outputs: set[str],
) -> Optional[_QuantizedInputPlan]:
    unary_op = graph_index.producer(concat_input)
    unary_index = (
        None if unary_op is None else graph_index.operator_index(unary_op)
    )
    if (
        unary_op is None
        or unary_index is None
        or str(unary_op.op_type) not in _UNARY_OPS
        or len(unary_op.inputs) != 1
        or len(unary_op.outputs) != 1
        or str(unary_op.outputs[0]) != concat_input
        or concat_input in model_outputs
        or set(graph_index.consumer_indices(concat_input))
        != {int(concat_index)}
    ):
        return None
    adapter_output = str(unary_op.inputs[0])
    adapter_op = graph_index.producer(adapter_output)
    adapter_index = (
        None if adapter_op is None else graph_index.operator_index(adapter_op)
    )
    if (
        adapter_op is None
        or adapter_index is None
        or str(adapter_op.op_type) != "TRANSPOSE"
        or len(adapter_op.inputs) < 2
        or len(adapter_op.outputs) != 1
        or str(adapter_op.outputs[0]) != adapter_output
        or _read_transpose_perm(model_ir, adapter_op)
        != _PERM_NHWC_TO_NCHW
        or adapter_output in model_outputs
        or set(graph_index.consumer_indices(adapter_output))
        != {int(unary_index)}
    ):
        return None
    source_name = str(adapter_op.inputs[0])
    source_tensor = model_ir.tensors.get(source_name)
    output_tensor = model_ir.tensors.get(concat_input)
    if (
        source_tensor is None
        or len(list(source_tensor.shape)) != 4
        or output_tensor is None
        or len(list(output_tensor.shape)) != 4
    ):
        return None
    return _QuantizedInputPlan(
        kind="unary",
        adapter_op=adapter_op,
        unary_op=unary_op,
        concat_input=concat_input,
        source_name=source_name,
        remove_adapter=True,
    )


def _resolve_pad_input_plan(
    model_ir: ModelIR,
    graph_index: ModelIRGraphIndex,
    *,
    concat_input: str,
    concat_index: int,
    model_outputs: set[str],
    public_names: set[str],
) -> Optional[_QuantizedInputPlan]:
    pad_plan = resolve_nhwc_concat_pad_plan(
        model_ir,
        graph_index,
        output_name=concat_input,
        concat_index=concat_index,
        model_outputs=model_outputs,
        public_names=public_names,
    )
    if pad_plan is None:
        return None
    return _QuantizedInputPlan(
        kind="pad",
        adapter_op=pad_plan.adapter_op,
        pad_plan=pad_plan,
        concat_input=concat_input,
        source_name=pad_plan.source_name,
        remove_adapter=pad_plan.remove_adapter,
    )


def _wrap_shared_input_plan(
    *,
    concat_input: str,
    input_plan: Optional[_NhwcConcatInputPlan],
) -> Optional[_QuantizedInputPlan]:
    if input_plan is None:
        return None
    return _QuantizedInputPlan(
        kind=input_plan.kind,
        adapter_op=input_plan.adapter_op,
        nhwc_plan=input_plan,
        concat_input=concat_input,
        source_name=input_plan.source_name,
        remove_adapter=input_plan.remove_adapter,
    )


def _resolve_swish_quantized_input_plan(
    model_ir: ModelIR,
    graph_index: ModelIRGraphIndex,
    *,
    concat_input: str,
    concat_index: int,
    model_outputs: set[str],
) -> Optional[_QuantizedInputPlan]:
    return _wrap_shared_input_plan(
        concat_input=concat_input,
        input_plan=_resolve_swish_input_plan(
            model_ir,
            graph_index,
            input_name=concat_input,
            concat_index=concat_index,
            model_outputs=model_outputs,
        ),
    )


def _resolve_dequantize_quantized_input_plan(
    model_ir: ModelIR,
    graph_index: ModelIRGraphIndex,
    *,
    concat_input: str,
    concat_index: int,
    model_outputs: set[str],
) -> Optional[_QuantizedInputPlan]:
    dequantize_plan = _resolve_dequantize_input_plan(
        model_ir,
        graph_index,
        input_name=concat_input,
        concat_index=concat_index,
        model_outputs=model_outputs,
    )
    if dequantize_plan is None:
        return None
    source_tensor = model_ir.tensors.get(dequantize_plan.source_name)
    if source_tensor is None or len(list(source_tensor.shape)) != 4:
        return None
    return _wrap_shared_input_plan(
        concat_input=concat_input,
        input_plan=dequantize_plan,
    )


def _resolve_prelu_quantized_input_plan(
    model_ir: ModelIR,
    graph_index: ModelIRGraphIndex,
    *,
    concat_input: str,
    concat_index: int,
    model_outputs: set[str],
    public_names: set[str],
) -> Optional[_QuantizedInputPlan]:
    return _wrap_shared_input_plan(
        concat_input=concat_input,
        input_plan=_resolve_prelu_input_plan(
            model_ir,
            graph_index,
            input_name=concat_input,
            concat_index=concat_index,
            model_outputs=model_outputs,
            public_names=public_names,
        ),
    )


def _resolve_softmax_quantized_input_plan(
    model_ir: ModelIR,
    graph_index: ModelIRGraphIndex,
    *,
    concat_input: str,
    concat_index: int,
    model_outputs: set[str],
) -> Optional[_QuantizedInputPlan]:
    return _wrap_shared_input_plan(
        concat_input=concat_input,
        input_plan=_resolve_softmax_input_plan(
            model_ir,
            graph_index,
            input_name=concat_input,
            concat_index=concat_index,
            model_outputs=model_outputs,
        ),
    )


def _resolve_leaky_quantized_input_plan(
    model_ir: ModelIR,
    graph_index: ModelIRGraphIndex,
    *,
    concat_input: str,
    concat_index: int,
    model_outputs: set[str],
) -> Optional[_QuantizedInputPlan]:
    return _wrap_shared_input_plan(
        concat_input=concat_input,
        input_plan=_resolve_leaky_input_plan(
            model_ir,
            graph_index,
            input_name=concat_input,
            concat_index=concat_index,
            model_outputs=model_outputs,
        ),
    )


def _resolve_slice_quantized_input_plan(
    model_ir: ModelIR,
    graph_index: ModelIRGraphIndex,
    *,
    concat_input: str,
    concat_index: int,
    model_outputs: set[str],
    public_names: set[str],
) -> Optional[_QuantizedInputPlan]:
    slice_plan = _resolve_slice_input_plan(
        model_ir,
        graph_index,
        input_name=concat_input,
        concat_index=concat_index,
        model_outputs=model_outputs,
        public_names=public_names,
    )
    if slice_plan is None or slice_plan.output_post_adapter_ops:
        return None
    return _wrap_shared_input_plan(
        concat_input=concat_input,
        input_plan=slice_plan,
    )


def _resolve_split_quantized_input_plan(
    model_ir: ModelIR,
    graph_index: ModelIRGraphIndex,
    *,
    concat_input: str,
    concat_index: int,
    model_outputs: set[str],
    public_names: set[str],
) -> Optional[_QuantizedInputPlan]:
    split_plan = _resolve_split_input_plan(
        model_ir,
        graph_index,
        input_name=concat_input,
        concat_index=concat_index,
        model_outputs=model_outputs,
        public_names=public_names,
    )
    if split_plan is None or split_plan.output_post_adapter_ops:
        return None
    return _wrap_shared_input_plan(
        concat_input=concat_input,
        input_plan=split_plan,
    )


def _resolve_add_quantized_input_plan(
    model_ir: ModelIR,
    graph_index: ModelIRGraphIndex,
    *,
    concat_input: str,
    concat_index: int,
    model_outputs: set[str],
) -> Optional[_QuantizedInputPlan]:
    def _has_only_supported_add_leaves(
        input_plan: _NhwcConcatInputPlan,
    ) -> bool:
        if input_plan.kind in {
            "direct",
            "unary",
            "swish",
            "dequantize",
            "prelu",
            "softmax",
            "leaky",
            "pad",
            "slice",
            "split",
        }:
            return not input_plan.output_post_adapter_ops
        return (
            input_plan.kind == "add"
            and not input_plan.output_post_adapter_ops
            and len(input_plan.add_operand_plans) == 2
            and all(
                _has_only_supported_add_leaves(operand_plan)
                for operand_plan in input_plan.add_operand_plans
            )
        )

    add_plan = _resolve_add_input_plan(
        model_ir,
        graph_index,
        input_name=concat_input,
        concat_index=concat_index,
        model_outputs=model_outputs,
    )
    if (
        add_plan is None
        or not _has_only_supported_add_leaves(add_plan)
    ):
        return None
    return _wrap_shared_input_plan(
        concat_input=concat_input,
        input_plan=add_plan,
    )


def _resolve_quantized_concat_candidate(
    model_ir: ModelIR,
    graph_index: ModelIRGraphIndex,
    *,
    family: str,
) -> Optional[_QuantizedConcatCandidate]:
    contract = _QUANTIZED_INPUT_CONTRACTS.get(family)
    if contract is None:
        return None
    model_outputs = {str(name) for name in model_ir.outputs}
    public_names = {
        *[str(name) for name in model_ir.inputs],
        *model_outputs,
    }
    for concat_op in model_ir.operators:
        concat_index = graph_index.operator_index(concat_op)
        if (
            concat_index is None
            or str(concat_op.op_type) != "CONCATENATION"
            or len(concat_op.inputs) == 0
            or len(concat_op.outputs) != 1
        ):
            continue
        concat_output = str(concat_op.outputs[0])
        concat_tensor = model_ir.tensors.get(concat_output)
        concat_axis = int(concat_op.options.get("axis", 1))
        if concat_axis < 0:
            concat_axis += 4
        if (
            concat_axis != 1
            or concat_output in model_outputs
            or concat_tensor is None
            or len(list(concat_tensor.shape)) != 4
        ):
            continue

        concat_users = graph_index.consumer_indices(concat_output)
        if len(concat_users) != 1:
            continue
        quantize_op = model_ir.operators[int(concat_users[0])]
        quantize_index = graph_index.operator_index(quantize_op)
        if (
            quantize_index is None
            or str(quantize_op.op_type) != "QUANTIZE"
            or len(quantize_op.inputs) != 1
            or len(quantize_op.outputs) != 1
            or str(quantize_op.inputs[0]) != concat_output
        ):
            continue
        quantized_output = str(quantize_op.outputs[0])
        quantized_tensor = model_ir.tensors.get(quantized_output)
        if (
            quantized_output in model_outputs
            or quantized_tensor is None
            or len(list(quantized_tensor.shape)) != 4
        ):
            continue

        post_ops: List[OperatorIR] = []
        post_outputs: List[str] = []
        quantized_users = graph_index.consumer_indices(quantized_output)
        if not quantized_users:
            continue
        posts_valid = True
        for post_index in quantized_users:
            post_op = model_ir.operators[int(post_index)]
            if (
                str(post_op.op_type) != "TRANSPOSE"
                or len(post_op.inputs) < 2
                or len(post_op.outputs) != 1
                or str(post_op.inputs[0]) != quantized_output
                or _read_transpose_perm(model_ir, post_op)
                != _PERM_NCHW_TO_NHWC
                or str(post_op.outputs[0]) in model_outputs
            ):
                posts_valid = False
                break
            post_output = str(post_op.outputs[0])
            post_tensor = model_ir.tensors.get(post_output)
            if post_tensor is None or len(list(post_tensor.shape)) != 4:
                posts_valid = False
                break
            post_ops.append(post_op)
            post_outputs.append(post_output)
        if not posts_valid or not post_ops:
            continue

        input_plans: List[_QuantizedInputPlan] = []
        inputs_valid = True
        for concat_input in [str(name) for name in concat_op.inputs]:
            input_plan = _resolve_direct_input_plan(
                model_ir,
                graph_index,
                concat_input=concat_input,
                concat_index=int(concat_index),
                model_outputs=model_outputs,
            )
            if input_plan is None and family in {
                "unary",
                "unary_pad",
                "swish",
                "leaky",
                "add",
            }:
                input_plan = _resolve_unary_input_plan(
                    model_ir,
                    graph_index,
                    concat_input=concat_input,
                    concat_index=int(concat_index),
                    model_outputs=model_outputs,
                )
            if input_plan is None and family == "swish":
                input_plan = _resolve_swish_quantized_input_plan(
                    model_ir,
                    graph_index,
                    concat_input=concat_input,
                    concat_index=int(concat_index),
                    model_outputs=model_outputs,
                )
            if input_plan is None and family == "dequantize":
                input_plan = _resolve_dequantize_quantized_input_plan(
                    model_ir,
                    graph_index,
                    concat_input=concat_input,
                    concat_index=int(concat_index),
                    model_outputs=model_outputs,
                )
            if input_plan is None and family == "prelu":
                input_plan = _resolve_prelu_quantized_input_plan(
                    model_ir,
                    graph_index,
                    concat_input=concat_input,
                    concat_index=int(concat_index),
                    model_outputs=model_outputs,
                    public_names=public_names,
                )
            if input_plan is None and family == "softmax":
                input_plan = _resolve_softmax_quantized_input_plan(
                    model_ir,
                    graph_index,
                    concat_input=concat_input,
                    concat_index=int(concat_index),
                    model_outputs=model_outputs,
                )
            if input_plan is None and family == "leaky":
                input_plan = _resolve_leaky_quantized_input_plan(
                    model_ir,
                    graph_index,
                    concat_input=concat_input,
                    concat_index=int(concat_index),
                    model_outputs=model_outputs,
                )
            if input_plan is None and family == "slice":
                input_plan = _resolve_slice_quantized_input_plan(
                    model_ir,
                    graph_index,
                    concat_input=concat_input,
                    concat_index=int(concat_index),
                    model_outputs=model_outputs,
                    public_names=public_names,
                )
            if input_plan is None and family == "split":
                input_plan = _resolve_split_quantized_input_plan(
                    model_ir,
                    graph_index,
                    concat_input=concat_input,
                    concat_index=int(concat_index),
                    model_outputs=model_outputs,
                    public_names=public_names,
                )
            if input_plan is None and family == "add":
                input_plan = _resolve_add_quantized_input_plan(
                    model_ir,
                    graph_index,
                    concat_input=concat_input,
                    concat_index=int(concat_index),
                    model_outputs=model_outputs,
                )
            if input_plan is None and family in {
                "pad",
                "unary_pad",
                "all_pad",
            }:
                input_plan = _resolve_pad_input_plan(
                    model_ir,
                    graph_index,
                    concat_input=concat_input,
                    concat_index=int(concat_index),
                    model_outputs=model_outputs,
                    public_names=public_names,
                )
            if input_plan is None:
                inputs_valid = False
                break
            input_plans.append(input_plan)
        if (
            not inputs_valid
            or not _input_plans_satisfy_contract(input_plans, contract)
        ):
            continue
        if contract.validate_shapes:
            reference_shape: Optional[List[int]] = None
            shapes_compatible = True
            for input_plan in input_plans:
                tensor = model_ir.tensors.get(
                    input_plan.source_name
                    if input_plan.kind == "direct"
                    else input_plan.concat_input
                )
                if tensor is None or len(list(tensor.shape)) != 4:
                    shapes_compatible = False
                    break
                shape = [int(value) for value in tensor.shape]
                if input_plan.kind != "direct":
                    shape = [shape[index] for index in _PERM_NCHW_TO_NHWC]
                if reference_shape is None:
                    reference_shape = shape
                elif any(
                    int(shape[index]) != int(reference_shape[index])
                    for index in (0, 1, 2)
                ):
                    shapes_compatible = False
                    break
            if not shapes_compatible:
                continue
        return _QuantizedConcatCandidate(
            input_plans=tuple(input_plans),
            concat_op=concat_op,
            concat_output=concat_output,
            quantize_op=quantize_op,
            quantized_output=quantized_output,
            post_ops=tuple(post_ops),
            post_outputs=tuple(post_outputs),
        )
    return None


def _has_quantized_concat_candidate(
    pass_state: ModelIRPassState,
    *,
    family: str,
) -> bool:
    return (
        _resolve_quantized_concat_candidate(
            pass_state.model_ir,
            pass_state.graph_index,
            family=family,
        )
        is not None
    )
def _optimize_quantized_concat_chains(
    model_ir: ModelIR,
    *,
    family: str,
    stats_key: str,
    graph_index: ModelIRGraphIndex | None = None,
    layout_state: LayoutState | None = None,
) -> Dict[str, int]:
    graph_index = graph_index or ModelIRGraphIndex(model_ir)
    optimized = 0
    while True:
        candidate = _resolve_quantized_concat_candidate(
            model_ir,
            graph_index,
            family=family,
        )
        if candidate is None:
            break

        new_concat_inputs: List[str] = []
        materialized_pads: Dict[str, str] = {}
        materialized_alphas: Dict[
            Tuple[str, Optional[Tuple[int, ...]], Tuple[int, ...]],
            str,
        ] = {}
        materialized_int_parameters: Dict[
            Tuple[str, Tuple[int, ...]],
            str,
        ] = {}
        applied_split_operators: set[int] = set()
        applied_add_operators: set[int] = set()
        for input_plan in candidate.input_plans:
            if input_plan.nhwc_plan is not None:
                if input_plan.kind == "dequantize":
                    _apply_dequantize_input_plan(
                        model_ir,
                        graph_index,
                        input_plan.nhwc_plan,
                    )
                elif input_plan.kind == "swish":
                    _apply_swish_input_plan(
                        model_ir,
                        graph_index,
                        input_plan.nhwc_plan,
                    )
                elif input_plan.kind == "prelu":
                    _apply_prelu_input_plan(
                        model_ir,
                        graph_index,
                        input_plan.nhwc_plan,
                        materialized_alphas=materialized_alphas,
                    )
                elif input_plan.kind == "softmax":
                    _apply_softmax_input_plan(
                        model_ir,
                        graph_index,
                        input_plan.nhwc_plan,
                    )
                elif input_plan.kind == "leaky":
                    _apply_leaky_input_plan(
                        model_ir,
                        graph_index,
                        input_plan.nhwc_plan,
                    )
                elif input_plan.kind == "slice":
                    _apply_slice_input_plan(
                        model_ir,
                        graph_index,
                        input_plan.nhwc_plan,
                        materialized=materialized_int_parameters,
                    )
                elif input_plan.kind == "split":
                    assert input_plan.nhwc_plan.split_op is not None
                    split_operator_id = id(input_plan.nhwc_plan.split_op)
                    if split_operator_id not in applied_split_operators:
                        _apply_split_input_plan(
                            model_ir,
                            graph_index,
                            input_plan.nhwc_plan,
                            materialized=materialized_int_parameters,
                        )
                        applied_split_operators.add(split_operator_id)
                elif input_plan.kind == "add":
                    _apply_add_input_plan(
                        model_ir,
                        graph_index,
                        input_plan.nhwc_plan,
                        materialized_int_parameters=(
                            materialized_int_parameters
                        ),
                        materialized_pads=materialized_pads,
                        materialized_prelu_alphas=materialized_alphas,
                        applied_split_operators=applied_split_operators,
                        applied_add_operators=applied_add_operators,
                    )
                else:
                    raise RuntimeError(
                        "unsupported shared quantized Concat input plan: "
                        f"{input_plan.kind}"
                    )
                new_concat_inputs.append(input_plan.concat_input)
            elif input_plan.unary_op is not None:
                _set_operator_inputs(
                    model_ir=model_ir,
                    op=input_plan.unary_op,
                    new_inputs=[input_plan.source_name],
                    graph_index=graph_index,
                )
                unary_tensor = model_ir.tensors.get(input_plan.concat_input)
                _permute_tensor_metadata_if_rank_matches(
                    unary_tensor,
                    _PERM_NCHW_TO_NHWC,
                )
                if unary_tensor is not None:
                    unary_tensor.quantization = _clone_permuted_quantization(
                        unary_tensor.quantization,
                        _PERM_NCHW_TO_NHWC,
                    )
                new_concat_inputs.append(input_plan.concat_input)
            elif input_plan.pad_plan is not None:
                apply_nhwc_concat_pad_plan(
                    model_ir,
                    graph_index,
                    input_plan.pad_plan,
                    materialized_pads=materialized_pads,
                )
                new_concat_inputs.append(input_plan.concat_input)
            else:
                new_concat_inputs.append(input_plan.source_name)
        _set_operator_inputs(
            model_ir=model_ir,
            op=candidate.concat_op,
            new_inputs=new_concat_inputs,
            graph_index=graph_index,
        )
        candidate.concat_op.options["axis"] = 3
        concat_tensor = model_ir.tensors.get(candidate.concat_output)
        _permute_tensor_metadata_if_rank_matches(
            concat_tensor,
            _PERM_NCHW_TO_NHWC,
        )
        if concat_tensor is not None:
            concat_tensor.quantization = _clone_permuted_quantization(
                concat_tensor.quantization,
                _PERM_NCHW_TO_NHWC,
            )

        canonical_output = candidate.post_outputs[0]
        _set_operator_outputs(
            model_ir=model_ir,
            op=candidate.quantize_op,
            new_outputs=[canonical_output],
            graph_index=graph_index,
        )
        for alias_output in candidate.post_outputs[1:]:
            _replace_tensor_inputs(
                model_ir=model_ir,
                src_name=alias_output,
                dst_name=canonical_output,
                graph_index=graph_index,
            )

        quantized_tensor = model_ir.tensors.get(candidate.quantized_output)
        canonical_tensor = model_ir.tensors.get(canonical_output)
        if canonical_tensor is not None and quantized_tensor is not None:
            canonical_tensor.dtype = str(quantized_tensor.dtype)
            canonical_tensor.quantization = _clone_permuted_quantization(
                quantized_tensor.quantization,
                _PERM_NCHW_TO_NHWC,
            )
        if canonical_tensor is not None and concat_tensor is not None:
            canonical_tensor.shape = [int(value) for value in concat_tensor.shape]
            canonical_tensor.shape_signature = (
                [int(value) for value in concat_tensor.shape_signature]
                if concat_tensor.shape_signature is not None
                else [int(value) for value in concat_tensor.shape]
            )

        public_names = {
            *[str(name) for name in model_ir.inputs],
            *[str(name) for name in model_ir.outputs],
        }

        def _adapter_is_now_removable(adapter_op: OperatorIR) -> bool:
            output_names = [str(name) for name in adapter_op.outputs]
            return bool(output_names) and all(
                output_name not in public_names
                and not graph_index.consumer_indices(output_name)
                for output_name in output_names
            )

        def _walk_shared_plans(
            input_plan: _NhwcConcatInputPlan,
        ) -> List[_NhwcConcatInputPlan]:
            plans = [input_plan]
            for operand_plan in input_plan.add_operand_plans:
                plans.extend(_walk_shared_plans(operand_plan))
            return plans

        shared_plans = [
            shared_plan
            for input_plan in candidate.input_plans
            if input_plan.nhwc_plan is not None
            for shared_plan in _walk_shared_plans(input_plan.nhwc_plan)
        ]
        remove_ops = [*candidate.post_ops]
        remove_ops.extend(
            input_plan.adapter_op
            for input_plan in candidate.input_plans
            if input_plan.remove_adapter
            or _adapter_is_now_removable(input_plan.adapter_op)
        )
        remove_ops.extend(
            shared_plan.adapter_op
            for shared_plan in shared_plans
            if shared_plan.remove_adapter
            or _adapter_is_now_removable(shared_plan.adapter_op)
        )
        remove_ops.extend(
            adapter_op
            for shared_plan in shared_plans
            for adapter_op in shared_plan.extra_source_adapter_ops
            if _adapter_is_now_removable(adapter_op)
        )
        remove_ops.extend(
            post_op
            for shared_plan in shared_plans
            for post_op in shared_plan.output_post_adapter_ops
        )
        remove_indices = sorted(
            {
                int(index)
                for op in remove_ops
                if (index := graph_index.operator_index(op)) is not None
            },
            reverse=True,
        )
        for remove_index in remove_indices:
            graph_index.remove_operator(remove_index)
        optimized += 1

    if optimized > 0:
        _prune_unused_tensors(model_ir, layout_state=layout_state)
        if layout_state is not None:
            layout_state.sync_from_model_ir(model_ir)
    return {stats_key: int(optimized)}


def run_nhwc_concat_quantized_layout_cleanup(
    model_ir: ModelIR,
    *,
    layout_state: LayoutState | None = None,
    diagnostics: List[Dict[str, Any]] | None = None,
) -> Dict[str, int]:
    """Lift bounded Concat→Quantize post-adapter families into NHWC."""

    def _make_callback(family: str, stats_key: str):
        def _run(
            pass_state: ModelIRPassState,
        ) -> Dict[str, int | bool]:
            stats = _optimize_quantized_concat_chains(
                pass_state.model_ir,
                family=family,
                stats_key=stats_key,
                graph_index=pass_state.graph_index,
                layout_state=pass_state.layout_state,
            )
            return {
                **stats,
                "changed": bool(stats.get(stats_key, 0)),
            }

        return _run

    specs = [
        PassSpec(
            pass_id=f"layout.nhwc_pre_concat_quantized_{family}",
            phase=PassPhase.LAYOUT_PLAN,
            callback=_make_callback(family, stats_key),
            precondition=(
                lambda pass_state, family=family: (
                    _has_quantized_concat_candidate(
                        pass_state,
                        family=family,
                    )
                )
            ),
            priority=priority,
            transactional=True,
        )
        for family, stats_key, priority in _QUANTIZED_FAMILY_SPECS
    ]
    details, _ = run_model_ir_pass_group(
        model_ir,
        specs=specs,
        layout_state=layout_state,
        default_details={
            stats_key: 0
            for _, stats_key, _ in _QUANTIZED_FAMILY_SPECS
        },
        diagnostics=diagnostics,
        preflight=lambda candidate_model: preflight_required_op_types(
            candidate_model,
            {"TRANSPOSE", "CONCATENATION", "QUANTIZE"},
        ),
    )
    return {str(key): int(value) for key, value in details.items()}
