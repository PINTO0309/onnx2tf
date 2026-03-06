from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np


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


@dataclass
class OperatorIR:
    op_type: str
    inputs: List[str]
    outputs: List[str]
    options: Dict[str, Any] = field(default_factory=dict)
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
