from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np


LOGICAL_LAYOUT_UNKNOWN = "UNKNOWN"
LOGICAL_LAYOUT_NCW = "NCW"
LOGICAL_LAYOUT_NWC = "NWC"
LOGICAL_LAYOUT_NCHW = "NCHW"
LOGICAL_LAYOUT_NHWC = "NHWC"
LOGICAL_LAYOUT_NCDHW = "NCDHW"
LOGICAL_LAYOUT_NDHWC = "NDHWC"

_VALID_LOGICAL_LAYOUTS = {
    LOGICAL_LAYOUT_UNKNOWN,
    LOGICAL_LAYOUT_NCW,
    LOGICAL_LAYOUT_NWC,
    LOGICAL_LAYOUT_NCHW,
    LOGICAL_LAYOUT_NHWC,
    LOGICAL_LAYOUT_NCDHW,
    LOGICAL_LAYOUT_NDHWC,
}


@dataclass
class QuantParamIR:
    scale: List[float]
    zero_point: List[int]
    quantized_dimension: int = 0
    min: Optional[List[float]] = None
    max: Optional[List[float]] = None


@dataclass
class TensorIR:
    name: str
    dtype: str
    shape: List[int]
    shape_signature: Optional[List[int]] = None
    data: Optional[np.ndarray] = None
    is_variable: bool = False
    quantization: Optional[Union[Dict[str, Any], QuantParamIR]] = None
    logical_layout: str = LOGICAL_LAYOUT_UNKNOWN


@dataclass
class OperatorIR:
    op_type: str
    inputs: List[str]
    outputs: List[str]
    options: Dict[str, Any] = field(default_factory=dict)
    axis_semantics: Dict[str, str] = field(default_factory=dict)
    version: int = 1


@dataclass
class ModelIR:
    name: str
    description: str = "onnx2tf flatbuffer_direct"
    tensors: Dict[str, TensorIR] = field(default_factory=dict)
    operators: List[OperatorIR] = field(default_factory=list)
    inputs: List[str] = field(default_factory=list)
    outputs: List[str] = field(default_factory=list)
    subgraphs: List["ModelIR"] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


def normalize_dim_to_shape_and_signature(dim: Any) -> Tuple[int, int]:
    if isinstance(dim, (int, np.integer)):
        if int(dim) >= 0:
            return int(dim), int(dim)
    return 1, -1


def normalize_onnx_shape(shape: Optional[List[Any]]) -> Tuple[List[int], List[int]]:
    if shape is None:
        return [1], [-1]
    norm_shape: List[int] = []
    signature: List[int] = []
    for dim in shape:
        s, sig = normalize_dim_to_shape_and_signature(dim)
        norm_shape.append(s)
        signature.append(sig)
    if len(norm_shape) == 0:
        # Scalar tensors are represented as rank-1 in many tflite paths.
        return [1], [1]
    return norm_shape, signature


def normalize_logical_layout(layout: Optional[str]) -> str:
    normalized = str(layout or LOGICAL_LAYOUT_UNKNOWN).upper()
    if normalized not in _VALID_LOGICAL_LAYOUTS:
        return LOGICAL_LAYOUT_UNKNOWN
    return normalized


def channel_first_logical_layout(rank: int) -> str:
    if int(rank) == 3:
        return LOGICAL_LAYOUT_NCW
    if int(rank) == 4:
        return LOGICAL_LAYOUT_NCHW
    if int(rank) == 5:
        return LOGICAL_LAYOUT_NCDHW
    return LOGICAL_LAYOUT_UNKNOWN


def channel_last_logical_layout(rank: int) -> str:
    if int(rank) == 3:
        return LOGICAL_LAYOUT_NWC
    if int(rank) == 4:
        return LOGICAL_LAYOUT_NHWC
    if int(rank) == 5:
        return LOGICAL_LAYOUT_NDHWC
    return LOGICAL_LAYOUT_UNKNOWN


def is_channel_first_logical_layout(layout: Optional[str]) -> bool:
    return normalize_logical_layout(layout) in {
        LOGICAL_LAYOUT_NCW,
        LOGICAL_LAYOUT_NCHW,
        LOGICAL_LAYOUT_NCDHW,
    }


def is_channel_last_logical_layout(layout: Optional[str]) -> bool:
    return normalize_logical_layout(layout) in {
        LOGICAL_LAYOUT_NWC,
        LOGICAL_LAYOUT_NHWC,
        LOGICAL_LAYOUT_NDHWC,
    }


def logical_layout_rank(layout: Optional[str]) -> int:
    normalized = normalize_logical_layout(layout)
    if normalized in {LOGICAL_LAYOUT_NCW, LOGICAL_LAYOUT_NWC}:
        return 3
    if normalized in {LOGICAL_LAYOUT_NCHW, LOGICAL_LAYOUT_NHWC}:
        return 4
    if normalized in {LOGICAL_LAYOUT_NCDHW, LOGICAL_LAYOUT_NDHWC}:
        return 5
    return -1


def logical_layout_permutation(
    *,
    source_layout: Optional[str],
    target_layout: Optional[str],
) -> Optional[List[int]]:
    source = normalize_logical_layout(source_layout)
    target = normalize_logical_layout(target_layout)
    source_rank = logical_layout_rank(source)
    target_rank = logical_layout_rank(target)
    if source_rank <= 0 or int(source_rank) != int(target_rank):
        return None
    if source == target:
        return list(range(source_rank))
    if is_channel_last_logical_layout(source) and is_channel_first_logical_layout(target):
        if int(source_rank) == 3:
            return [0, 2, 1]
        if int(source_rank) == 4:
            return [0, 3, 1, 2]
        if int(source_rank) == 5:
            return [0, 4, 1, 2, 3]
    if is_channel_first_logical_layout(source) and is_channel_last_logical_layout(target):
        if int(source_rank) == 3:
            return [0, 2, 1]
        if int(source_rank) == 4:
            return [0, 2, 3, 1]
        if int(source_rank) == 5:
            return [0, 2, 3, 4, 1]
    return None


def remap_layout_through_permute(
    *,
    layout: Optional[str],
    perm: List[int],
) -> str:
    normalized = normalize_logical_layout(layout)
    rank = logical_layout_rank(normalized)
    if rank <= 0 or len(list(perm)) != int(rank):
        return LOGICAL_LAYOUT_UNKNOWN
    if list(perm) == logical_layout_permutation(
        source_layout=normalized,
        target_layout=channel_first_logical_layout(rank),
    ):
        return channel_first_logical_layout(rank)
    if list(perm) == logical_layout_permutation(
        source_layout=normalized,
        target_layout=channel_last_logical_layout(rank),
    ):
        return channel_last_logical_layout(rank)
    if list(perm) == list(range(rank)):
        return normalized
    return LOGICAL_LAYOUT_UNKNOWN


def rewrite_axis_for_layout(
    *,
    axis: int,
    source_layout: Optional[str],
    target_layout: Optional[str],
    rank: Optional[int] = None,
) -> int:
    effective_rank = int(rank) if rank is not None else logical_layout_rank(source_layout)
    perm = logical_layout_permutation(
        source_layout=source_layout,
        target_layout=target_layout,
    )
    if perm is None or int(effective_rank) <= 0:
        return int(axis)
    resolved_axis = int(axis)
    if resolved_axis < 0:
        resolved_axis += int(effective_rank)
    if resolved_axis < 0 or resolved_axis >= int(effective_rank):
        return int(axis)
    return int(perm.index(resolved_axis))


def clone_model_ir_with_float16(model_ir: ModelIR) -> ModelIR:
    clone = ModelIR(
        name=model_ir.name,
        description=model_ir.description,
        metadata=dict(model_ir.metadata),
    )
    clone.inputs = list(model_ir.inputs)
    clone.outputs = list(model_ir.outputs)
    clone.subgraphs = [clone_model_ir_with_float16(subgraph) for subgraph in model_ir.subgraphs]
    clone.operators = [
        OperatorIR(
            op_type=op.op_type,
            inputs=list(op.inputs),
            outputs=list(op.outputs),
            options=dict(op.options),
            axis_semantics=dict(op.axis_semantics),
            version=op.version,
        ) for op in model_ir.operators
    ]
    for name, tensor in model_ir.tensors.items():
        new_data = tensor.data
        new_dtype = tensor.dtype
        if tensor.dtype == "FLOAT32":
            new_dtype = "FLOAT16"
            if tensor.data is not None:
                new_data = tensor.data.astype(np.float16)
        clone.tensors[name] = TensorIR(
            name=tensor.name,
            dtype=new_dtype,
            shape=list(tensor.shape),
            shape_signature=list(tensor.shape_signature) if tensor.shape_signature is not None else None,
            data=new_data.copy() if isinstance(new_data, np.ndarray) else new_data,
            is_variable=tensor.is_variable,
            quantization=(
                dict(tensor.quantization)
                if isinstance(tensor.quantization, dict)
                else QuantParamIR(
                    scale=list(tensor.quantization.scale),
                    zero_point=list(tensor.quantization.zero_point),
                    quantized_dimension=int(tensor.quantization.quantized_dimension),
                    min=list(tensor.quantization.min)
                    if tensor.quantization.min is not None
                    else None,
                    max=list(tensor.quantization.max)
                    if tensor.quantization.max is not None
                    else None,
                )
                if isinstance(tensor.quantization, QuantParamIR)
                else tensor.quantization
            ),
            logical_layout=normalize_logical_layout(tensor.logical_layout),
        )
    return clone


def _rewrite_float16_token_to_float32(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            key: _rewrite_float16_token_to_float32(item)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [_rewrite_float16_token_to_float32(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_rewrite_float16_token_to_float32(item) for item in value)
    if isinstance(value, str) and str(value).upper() == "FLOAT16":
        return "FLOAT32"
    return value


def clone_model_ir_with_float32(model_ir: ModelIR) -> ModelIR:
    clone = ModelIR(
        name=model_ir.name,
        description=model_ir.description,
        metadata=dict(model_ir.metadata),
    )
    clone.inputs = list(model_ir.inputs)
    clone.outputs = list(model_ir.outputs)
    clone.subgraphs = [clone_model_ir_with_float32(subgraph) for subgraph in model_ir.subgraphs]
    clone.operators = [
        OperatorIR(
            op_type=op.op_type,
            inputs=list(op.inputs),
            outputs=list(op.outputs),
            options=_rewrite_float16_token_to_float32(dict(op.options)),
            axis_semantics=dict(op.axis_semantics),
            version=op.version,
        ) for op in model_ir.operators
    ]
    for name, tensor in model_ir.tensors.items():
        new_data = tensor.data
        new_dtype = str(tensor.dtype).upper()
        if new_dtype == "FLOAT16":
            new_dtype = "FLOAT32"
            if tensor.data is not None:
                new_data = tensor.data.astype(np.float32)
        clone.tensors[name] = TensorIR(
            name=tensor.name,
            dtype=new_dtype,
            shape=list(tensor.shape),
            shape_signature=list(tensor.shape_signature) if tensor.shape_signature is not None else None,
            data=new_data.copy() if isinstance(new_data, np.ndarray) else new_data,
            is_variable=tensor.is_variable,
            quantization=(
                dict(tensor.quantization)
                if isinstance(tensor.quantization, dict)
                else QuantParamIR(
                    scale=list(tensor.quantization.scale),
                    zero_point=list(tensor.quantization.zero_point),
                    quantized_dimension=int(tensor.quantization.quantized_dimension),
                    min=list(tensor.quantization.min)
                    if tensor.quantization.min is not None
                    else None,
                    max=list(tensor.quantization.max)
                    if tensor.quantization.max is not None
                    else None,
                )
                if isinstance(tensor.quantization, QuantParamIR)
                else tensor.quantization
            ),
            logical_layout=normalize_logical_layout(tensor.logical_layout),
        )
    return clone


def prune_identity_cast_operators(
    model_ir: ModelIR,
    *,
    preserve_model_outputs: bool = True,
) -> int:
    """
    Remove redundant CAST operators where in/out dtypes are identical.

    Example:
      CAST(inDataType=FLOAT32, outDataType=FLOAT32): a -> b
    can be removed by rewiring consumers of `b` to `a`.

    By default, casts whose output tensor is a model output are preserved to
    avoid changing boundary tensor names/contracts.
    """
    rewritten = 0

    while True:
        changed = False
        for cast_idx, cast_op in enumerate(model_ir.operators):
            if str(cast_op.op_type) != "CAST" or len(cast_op.inputs) != 1 or len(cast_op.outputs) != 1:
                continue

            in_name = str(cast_op.inputs[0])
            out_name = str(cast_op.outputs[0])
            if in_name == "" or out_name == "" or in_name == out_name:
                continue
            if bool(preserve_model_outputs) and out_name in set(str(v) for v in model_ir.outputs):
                continue

            in_tensor = model_ir.tensors.get(in_name, None)
            out_tensor = model_ir.tensors.get(out_name, None)
            in_dtype = str(cast_op.options.get("inDataType", "")).upper()
            out_dtype = str(cast_op.options.get("outDataType", "")).upper()
            if in_dtype == "" and in_tensor is not None:
                in_dtype = str(in_tensor.dtype).upper()
            if out_dtype == "" and out_tensor is not None:
                out_dtype = str(out_tensor.dtype).upper()
            if in_dtype == "" or out_dtype == "" or in_dtype != out_dtype:
                continue

            for op in model_ir.operators:
                if len(op.inputs) > 0:
                    op.inputs = [
                        in_name if str(input_name) == out_name else input_name
                        for input_name in op.inputs
                    ]
            if not bool(preserve_model_outputs):
                model_ir.outputs = [
                    in_name if str(output_name) == out_name else output_name
                    for output_name in model_ir.outputs
                ]

            del model_ir.operators[int(cast_idx)]
            rewritten += 1
            changed = True
            break

        if not changed:
            break

    if rewritten > 0:
        _prune_unused_tensors(model_ir)

    return int(rewritten)


def _prune_unused_tensors(model_ir: ModelIR) -> None:
    used_tensor_names = set(str(v) for v in list(model_ir.inputs) + list(model_ir.outputs))
    for op in model_ir.operators:
        used_tensor_names.update(str(v) for v in op.inputs)
        used_tensor_names.update(str(v) for v in op.outputs)
    for tensor_name in [name for name in list(model_ir.tensors.keys()) if str(name) not in used_tensor_names]:
        del model_ir.tensors[tensor_name]


def _build_tensor_consumer_map(model_ir: ModelIR) -> Dict[str, List[int]]:
    consumers: Dict[str, List[int]] = {}
    for op_idx, op in enumerate(model_ir.operators):
        for input_name in op.inputs:
            key = str(input_name)
            if key == "":
                continue
            consumers.setdefault(key, []).append(int(op_idx))
    return consumers


def _is_identity_perm(perm: List[int]) -> bool:
    return list(int(v) for v in perm) == list(range(len(perm)))


def _is_inverse_perm(perm_a: List[int], perm_b: List[int]) -> bool:
    if len(perm_a) != len(perm_b):
        return False
    rank = len(perm_a)
    if sorted(int(v) for v in perm_a) != list(range(rank)):
        return False
    if sorted(int(v) for v in perm_b) != list(range(rank)):
        return False
    return all(int(perm_b[int(perm_a[i])]) == int(i) for i in range(rank))


def _compose_permutations(perm_pre: List[int], perm_post: List[int]) -> Optional[List[int]]:
    if len(perm_pre) != len(perm_post) or len(perm_pre) == 0:
        return None
    rank = len(perm_pre)
    if sorted(int(v) for v in perm_pre) != list(range(rank)):
        return None
    if sorted(int(v) for v in perm_post) != list(range(rank)):
        return None
    try:
        return [int(perm_pre[int(v)]) for v in perm_post]
    except Exception:
        return None


def _replace_tensor_inputs(model_ir: ModelIR, old_name: str, new_name: str) -> None:
    for op in model_ir.operators:
        if len(op.inputs) == 0:
            continue
        op.inputs = [
            new_name if str(input_name) == str(old_name) else input_name
            for input_name in op.inputs
        ]


def _read_transpose_perm(model_ir: ModelIR, op: OperatorIR) -> Optional[List[int]]:
    if str(op.op_type) != "TRANSPOSE":
        return None
    if len(op.inputs) >= 2:
        perm_tensor = model_ir.tensors.get(str(op.inputs[1]), None)
        if perm_tensor is not None and perm_tensor.data is not None:
            try:
                perm = [int(v) for v in np.asarray(perm_tensor.data).reshape(-1).tolist()]
                if len(perm) > 0 and sorted(perm) == list(range(len(perm))):
                    return perm
            except Exception:
                pass
    opt_perm = op.options.get("perm", None)
    if isinstance(opt_perm, (list, tuple)) and len(opt_perm) > 0:
        try:
            perm = [int(v) for v in list(opt_perm)]
            if sorted(perm) == list(range(len(perm))):
                return perm
        except Exception:
            return None
    return None


def _write_transpose_perm(
    model_ir: ModelIR,
    op: OperatorIR,
    perm: List[int],
) -> None:
    perm_values = [int(v) for v in list(perm)]
    op.options["perm"] = list(perm_values)
    if len(op.inputs) >= 2:
        perm_name = str(op.inputs[1])
        perm_tensor = model_ir.tensors.get(perm_name, None)
        if perm_tensor is not None:
            perm_arr = np.asarray(perm_values, dtype=np.int32)
            perm_tensor.data = perm_arr
            perm_tensor.dtype = "INT32"
            perm_tensor.shape = [int(perm_arr.size)]
            perm_tensor.shape_signature = [int(perm_arr.size)]
            return
    base_name = str(op.outputs[0]) if len(op.outputs) == 1 else f"transpose_perm_{len(model_ir.tensors)}"
    new_perm_name = f"{base_name}_perm"
    suffix = 1
    while new_perm_name in model_ir.tensors:
        new_perm_name = f"{base_name}_perm_{suffix}"
        suffix += 1
    perm_arr = np.asarray(perm_values, dtype=np.int32)
    model_ir.tensors[new_perm_name] = TensorIR(
        name=new_perm_name,
        dtype="INT32",
        shape=[int(perm_arr.size)],
        shape_signature=[int(perm_arr.size)],
        data=perm_arr,
    )
    if len(op.inputs) >= 2:
        op.inputs[1] = new_perm_name
    else:
        op.inputs.append(new_perm_name)


def optimize_redundant_transpose_operators(
    model_ir: ModelIR,
    *,
    preserve_model_outputs: bool = True,
) -> Dict[str, int]:
    """
    Reduce redundant TRANSPOSE chains:
    - remove identity TRANSPOSE
    - remove inverse consecutive TRANSPOSE pairs
    - compose consecutive TRANSPOSE pairs into one TRANSPOSE
    """
    removed_identity = 0
    removed_inverse_pairs = 0
    composed_pairs = 0

    while True:
        changed = False
        consumers = _build_tensor_consumer_map(model_ir)

        # 1) Remove identity transpose.
        for op_idx, op in enumerate(model_ir.operators):
            if str(op.op_type) != "TRANSPOSE" or len(op.outputs) != 1:
                continue
            perm = _read_transpose_perm(model_ir, op)
            if perm is None or not _is_identity_perm(perm):
                continue
            in_name = str(op.inputs[0]) if len(op.inputs) >= 1 else ""
            out_name = str(op.outputs[0])
            if in_name == "" or out_name == "":
                continue
            if bool(preserve_model_outputs) and out_name in set(str(v) for v in model_ir.outputs):
                continue
            _replace_tensor_inputs(model_ir, out_name, in_name)
            del model_ir.operators[int(op_idx)]
            removed_identity += 1
            changed = True
            break
        if changed:
            continue

        # 2) Simplify consecutive transpose pair.
        for op_idx, pre_op in enumerate(model_ir.operators):
            if str(pre_op.op_type) != "TRANSPOSE" or len(pre_op.outputs) != 1 or len(pre_op.inputs) < 1:
                continue
            bridge_name = str(pre_op.outputs[0])
            bridge_users = [int(v) for v in consumers.get(bridge_name, [])]
            if len(bridge_users) != 1:
                continue
            post_idx = int(bridge_users[0])
            if post_idx == int(op_idx):
                continue
            post_op = model_ir.operators[int(post_idx)]
            if str(post_op.op_type) != "TRANSPOSE" or len(post_op.outputs) != 1 or len(post_op.inputs) < 1:
                continue
            if str(post_op.inputs[0]) != bridge_name:
                continue
            pre_perm = _read_transpose_perm(model_ir, pre_op)
            post_perm = _read_transpose_perm(model_ir, post_op)
            if pre_perm is None or post_perm is None:
                continue

            composed_perm = _compose_permutations(pre_perm, post_perm)
            if composed_perm is None:
                continue

            pre_input = str(pre_op.inputs[0])
            post_output = str(post_op.outputs[0])
            post_op.inputs[0] = pre_input

            if _is_identity_perm(composed_perm):
                if bool(preserve_model_outputs) and post_output in set(str(v) for v in model_ir.outputs):
                    _write_transpose_perm(
                        model_ir=model_ir,
                        op=post_op,
                        perm=[int(v) for v in range(len(composed_perm))],
                    )
                    del model_ir.operators[int(op_idx)]
                    changed = True
                    break
                _replace_tensor_inputs(model_ir, post_output, pre_input)
                for remove_idx in sorted([int(op_idx), int(post_idx)], reverse=True):
                    del model_ir.operators[remove_idx]
                removed_inverse_pairs += 1
                changed = True
                break

            _write_transpose_perm(
                model_ir=model_ir,
                op=post_op,
                perm=composed_perm,
            )
            del model_ir.operators[int(op_idx)]
            composed_pairs += 1
            changed = True
            break

        if not changed:
            break

    if removed_identity > 0 or removed_inverse_pairs > 0 or composed_pairs > 0:
        _prune_unused_tensors(model_ir)

    return {
        "removed_identity_transpose": int(removed_identity),
        "removed_inverse_transpose_pairs": int(removed_inverse_pairs),
        "composed_consecutive_transpose_pairs": int(composed_pairs),
    }


def _permute_shape(values: List[int], perm: List[int]) -> Optional[List[int]]:
    if len(list(values)) != len(list(perm)):
        return None
    return [int(values[int(idx)]) for idx in perm]


def _boundary_current_layout(
    *,
    tensor: TensorIR,
    boundary_signature: Optional[List[int]],
    assume_channel_last_names: set[str],
) -> str:
    rank = len(list(tensor.shape))
    if rank not in {3, 4, 5}:
        return LOGICAL_LAYOUT_UNKNOWN
    if str(tensor.name) in assume_channel_last_names:
        return channel_last_logical_layout(rank)
    if isinstance(boundary_signature, list) and len(boundary_signature) == rank:
        current_shape = [int(v) for v in list(tensor.shape)]
        cf_shape = [int(v) if int(v) > 0 else 1 for v in list(boundary_signature)]
        if current_shape == cf_shape:
            return channel_first_logical_layout(rank)
        cl_shape = _permute_shape(
            cf_shape,
            logical_layout_permutation(
                source_layout=channel_first_logical_layout(rank),
                target_layout=channel_last_logical_layout(rank),
            ) or [],
        )
        if cl_shape is not None and current_shape == cl_shape:
            return channel_last_logical_layout(rank)
    lowered_name = str(tensor.name).lower()
    if any(token in lowered_name for token in ["_nwc", "_nhwc", "_ndhwc"]):
        return channel_last_logical_layout(rank)
    if any(token in lowered_name for token in ["_ncw", "_nchw", "_ncdhw", "_onnx_ncx_internal"]):
        return channel_first_logical_layout(rank)
    return normalize_logical_layout(tensor.logical_layout)


def infer_model_ir_logical_layouts(model_ir: ModelIR) -> Dict[str, str]:
    boundary_map = model_ir.metadata.get("onnx_boundary_shape_signature_map", {})
    if not isinstance(boundary_map, dict):
        boundary_map = {}
    assume_channel_last_names = {
        str(v)
        for v in model_ir.metadata.get("assume_channel_last_layout_tensor_names", [])
        if str(v) != ""
    }
    recurrent_public_boundary_context = any(
        str(op.op_type) in {
            "GRU",
            "LSTM",
            "RNN",
            "UNIDIRECTIONAL_SEQUENCE_RNN",
            "UNIDIRECTIONAL_SEQUENCE_LSTM",
            "BIDIRECTIONAL_SEQUENCE_LSTM",
        }
        for op in model_ir.operators
    )
    public_layout_map: Dict[str, str] = {}
    for tensor_name in list(model_ir.inputs) + list(model_ir.outputs):
        tensor = model_ir.tensors.get(str(tensor_name), None)
        if tensor is None:
            continue
        rank = len(list(tensor.shape))
        if rank in {3, 4, 5}:
            if recurrent_public_boundary_context and rank == 3:
                public_layout_map[str(tensor_name)] = channel_last_logical_layout(rank)
            else:
                public_layout_map[str(tensor_name)] = channel_first_logical_layout(rank)
    model_ir.metadata["onnx_public_layout_map"] = dict(public_layout_map)

    for tensor_name, tensor in model_ir.tensors.items():
        tensor.logical_layout = normalize_logical_layout(tensor.logical_layout)
        if str(tensor_name) in public_layout_map:
            tensor.logical_layout = _boundary_current_layout(
                tensor=tensor,
                boundary_signature=boundary_map.get(str(tensor_name), None),
                assume_channel_last_names=assume_channel_last_names,
            )
        elif tensor.logical_layout == LOGICAL_LAYOUT_UNKNOWN:
            rank = len(list(tensor.shape))
            lowered_name = str(tensor_name).lower()
            if rank in {3, 4, 5} and any(token in lowered_name for token in ["_nwc", "_nhwc", "_ndhwc"]):
                tensor.logical_layout = channel_last_logical_layout(rank)
            elif rank in {3, 4, 5} and any(token in lowered_name for token in ["_ncw", "_nchw", "_ncdhw", "_onnx_ncx_internal"]):
                tensor.logical_layout = channel_first_logical_layout(rank)

    layout_passthrough_ops = {
        "ABS",
        "ADD",
        "AVERAGE_POOL_2D",
        "BROADCAST_TO",
        "CAST",
        "CEIL",
        "CONV_2D",
        "CONV_3D",
        "CONV_3D_TRANSPOSE",
        "COS",
        "DEPTHWISE_CONV_2D",
        "DIV",
        "ELU",
        "EXP",
        "EXPAND_DIMS",
        "FLOOR",
        "GATHER",
        "GATHER_ND",
        "IDENTITY",
        "LEAKY_RELU",
        "LOG",
        "LOGICAL_AND",
        "LOGICAL_NOT",
        "LOGICAL_OR",
        "LOGISTIC",
        "L2_NORMALIZATION",
        "MAXIMUM",
        "MAX_POOL_2D",
        "MEAN",
        "MINIMUM",
        "MIRROR_PAD",
        "MUL",
        "NEG",
        "PAD",
        "PADV2",
        "PACK",
        "POW",
        "PRELU",
        "RELU",
        "RELU6",
        "RELU_N1_TO_1",
        "RELU_0_TO_1",
        "RESHAPE",
        "RESIZE_BILINEAR",
        "RESIZE_NEAREST_NEIGHBOR",
        "REVERSE_V2",
        "SELECT",
        "SELECT_V2",
        "SHAPE",
        "SIGN",
        "SIN",
        "SLICE",
        "SOFTMAX",
        "SPACE_TO_DEPTH",
        "SQRT",
        "SQUEEZE",
        "STRIDED_SLICE",
        "SUB",
        "SUM",
        "TANH",
        "TILE",
        "TRANSPOSE_CONV",
        "UNPACK",
        "WHERE",
    }

    max_iter = max(1, int(len(model_ir.operators)) * 4)
    for _ in range(max_iter):
        changed = False
        for op in model_ir.operators:
            op_type = str(op.op_type)
            output_names = [str(v) for v in list(op.outputs)]
            output_tensors = [model_ir.tensors.get(name, None) for name in output_names]
            input_names = [str(v) for v in list(op.inputs)]
            input_tensors = [model_ir.tensors.get(name, None) for name in input_names]

            if op_type == "TRANSPOSE" and len(input_tensors) >= 1 and len(output_tensors) == 1:
                source_tensor = input_tensors[0]
                target_tensor = output_tensors[0]
                if source_tensor is None or target_tensor is None:
                    continue
                perm = _read_transpose_perm(model_ir, op)
                if perm is None:
                    continue
                source_layout = normalize_logical_layout(source_tensor.logical_layout)
                target_layout = normalize_logical_layout(target_tensor.logical_layout)
                if source_layout != LOGICAL_LAYOUT_UNKNOWN:
                    remapped = remap_layout_through_permute(layout=source_layout, perm=perm)
                    if remapped != LOGICAL_LAYOUT_UNKNOWN and remapped != target_layout:
                        target_tensor.logical_layout = remapped
                        changed = True
                elif target_layout != LOGICAL_LAYOUT_UNKNOWN:
                    for candidate_rank in [len(perm)]:
                        cf_layout = channel_first_logical_layout(candidate_rank)
                        cl_layout = channel_last_logical_layout(candidate_rank)
                        for candidate in [cf_layout, cl_layout]:
                            if remap_layout_through_permute(layout=candidate, perm=perm) == target_layout:
                                source_tensor.logical_layout = candidate
                                changed = True
                                break
                continue

            if op_type == "SPLIT":
                data_input = input_tensors[1] if len(input_tensors) >= 2 else input_tensors[0] if len(input_tensors) >= 1 else None
                if data_input is None:
                    continue
                data_layout = normalize_logical_layout(data_input.logical_layout)
                if data_layout != LOGICAL_LAYOUT_UNKNOWN:
                    for output_tensor in output_tensors:
                        if output_tensor is not None and normalize_logical_layout(output_tensor.logical_layout) != data_layout:
                            output_tensor.logical_layout = data_layout
                            changed = True
                else:
                    known_output_layouts = {
                        normalize_logical_layout(t.logical_layout)
                        for t in output_tensors
                        if t is not None and normalize_logical_layout(t.logical_layout) != LOGICAL_LAYOUT_UNKNOWN
                    }
                    if len(known_output_layouts) == 1:
                        resolved = next(iter(known_output_layouts))
                        data_input.logical_layout = resolved
                        changed = True
                continue

            if op_type == "CONCATENATION":
                known_input_layouts = {
                    normalize_logical_layout(t.logical_layout)
                    for t in input_tensors
                    if t is not None and normalize_logical_layout(t.logical_layout) != LOGICAL_LAYOUT_UNKNOWN
                }
                if len(known_input_layouts) == 1:
                    resolved = next(iter(known_input_layouts))
                    for output_tensor in output_tensors:
                        if output_tensor is not None and normalize_logical_layout(output_tensor.logical_layout) != resolved:
                            output_tensor.logical_layout = resolved
                            changed = True
                else:
                    known_output_layouts = {
                        normalize_logical_layout(t.logical_layout)
                        for t in output_tensors
                        if t is not None and normalize_logical_layout(t.logical_layout) != LOGICAL_LAYOUT_UNKNOWN
                    }
                    if len(known_output_layouts) == 1:
                        resolved = next(iter(known_output_layouts))
                        for input_tensor in input_tensors:
                            if input_tensor is None:
                                continue
                            input_rank = len(list(input_tensor.shape))
                            if input_rank not in {3, 4, 5}:
                                continue
                            target_layout = resolved
                            if is_channel_first_logical_layout(resolved):
                                target_layout = channel_first_logical_layout(input_rank)
                            elif is_channel_last_logical_layout(resolved):
                                target_layout = channel_last_logical_layout(input_rank)
                            if normalize_logical_layout(input_tensor.logical_layout) != target_layout:
                                input_tensor.logical_layout = target_layout
                                changed = True
                continue

            if op_type in layout_passthrough_ops:
                resolved_layout = LOGICAL_LAYOUT_UNKNOWN
                for input_tensor in input_tensors:
                    if input_tensor is None:
                        continue
                    candidate = normalize_logical_layout(input_tensor.logical_layout)
                    if candidate == LOGICAL_LAYOUT_UNKNOWN:
                        continue
                    resolved_layout = candidate
                    break
                if resolved_layout == LOGICAL_LAYOUT_UNKNOWN:
                    known_outputs = {
                        normalize_logical_layout(t.logical_layout)
                        for t in output_tensors
                        if t is not None and normalize_logical_layout(t.logical_layout) != LOGICAL_LAYOUT_UNKNOWN
                    }
                    if len(known_outputs) == 1:
                        resolved_layout = next(iter(known_outputs))
                if resolved_layout == LOGICAL_LAYOUT_UNKNOWN:
                    continue
                for input_tensor in input_tensors:
                    if input_tensor is None:
                        continue
                    input_rank = len(list(input_tensor.shape))
                    if input_rank not in {3, 4, 5}:
                        continue
                    if is_channel_first_logical_layout(resolved_layout):
                        input_target_layout = channel_first_logical_layout(input_rank)
                    elif is_channel_last_logical_layout(resolved_layout):
                        input_target_layout = channel_last_logical_layout(input_rank)
                    else:
                        input_target_layout = LOGICAL_LAYOUT_UNKNOWN
                    if input_target_layout != LOGICAL_LAYOUT_UNKNOWN and normalize_logical_layout(input_tensor.logical_layout) != input_target_layout:
                        input_tensor.logical_layout = input_target_layout
                        changed = True
                for output_tensor in output_tensors:
                    if output_tensor is None:
                        continue
                    output_rank = len(list(output_tensor.shape))
                    if output_rank in {3, 4, 5}:
                        if is_channel_first_logical_layout(resolved_layout):
                            target_layout = channel_first_logical_layout(output_rank)
                        elif is_channel_last_logical_layout(resolved_layout):
                            target_layout = channel_last_logical_layout(output_rank)
                        else:
                            target_layout = LOGICAL_LAYOUT_UNKNOWN
                        if target_layout != LOGICAL_LAYOUT_UNKNOWN and normalize_logical_layout(output_tensor.logical_layout) != target_layout:
                            output_tensor.logical_layout = target_layout
                            changed = True
        if not changed:
            break

    return {
        str(name): normalize_logical_layout(tensor.logical_layout)
        for name, tensor in model_ir.tensors.items()
    }


def validate_model_ir_layout_annotations(model_ir: ModelIR) -> List[str]:
    problems: List[str] = []
    for tensor_name, tensor in sorted(model_ir.tensors.items()):
        layout = normalize_logical_layout(tensor.logical_layout)
        rank = len(list(tensor.shape))
        if rank not in {3, 4, 5}:
            continue
        if layout == LOGICAL_LAYOUT_UNKNOWN:
            continue
        if logical_layout_rank(layout) != int(rank):
            problems.append(
                f"tensor={tensor_name} shape={list(tensor.shape)} logical_layout={layout}"
            )
    return problems
