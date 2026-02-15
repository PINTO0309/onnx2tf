from __future__ import annotations

import copy
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import onnx
from onnx import numpy_helper

from onnx2tf.tflite_builder.dispatcher import dispatch_node
from onnx2tf.tflite_builder.op_registry import (
    NodeValidationError,
    get_custom_op_candidate_ops,
    get_supported_onnx_ops,
    resolve_node_dispatch,
)
from onnx2tf.tflite_builder.ir import (
    ModelIR,
    OperatorIR,
    QuantParamIR,
    TensorIR,
    normalize_onnx_shape,
)
from onnx2tf.tflite_builder.tensor_buffer_builder import tflite_dtype_from_numpy


_ONNX_TYPE_TO_TFLITE_DTYPE = {
    onnx.TensorProto.FLOAT: "FLOAT32",
    onnx.TensorProto.FLOAT16: "FLOAT16",
    onnx.TensorProto.DOUBLE: "FLOAT64",
    onnx.TensorProto.INT8: "INT8",
    onnx.TensorProto.INT16: "INT16",
    onnx.TensorProto.INT32: "INT32",
    onnx.TensorProto.INT64: "INT64",
    onnx.TensorProto.UINT8: "UINT8",
    onnx.TensorProto.UINT16: "UINT16",
    onnx.TensorProto.UINT32: "UINT32",
    onnx.TensorProto.UINT64: "UINT64",
    onnx.TensorProto.BOOL: "BOOL",
}


def _dtype_from_onnx_elem_type(elem_type: Optional[int]) -> str:
    if elem_type is None:
        return "FLOAT32"
    if elem_type not in _ONNX_TYPE_TO_TFLITE_DTYPE:
        raise NotImplementedError(f"Unsupported ONNX dtype in flatbuffer_direct: elem_type={elem_type}")
    return _ONNX_TYPE_TO_TFLITE_DTYPE[elem_type]


def _extract_tensor_info(
    onnx_graph: onnx.ModelProto,
) -> Tuple[Dict[str, List[Any]], Dict[str, str]]:
    shape_map: Dict[str, List[Any]] = {}
    dtype_map: Dict[str, str] = {}

    def _fill_value_info(value_info):
        if not value_info.type.HasField("tensor_type"):
            return
        name = value_info.name
        tensor_type = value_info.type.tensor_type
        dims: List[Any] = []
        if tensor_type.HasField("shape"):
            for d in tensor_type.shape.dim:
                if d.HasField("dim_value") and d.dim_value >= 0:
                    dims.append(int(d.dim_value))
                else:
                    dims.append(-1)
        shape_map[name] = dims
        dtype_map[name] = _dtype_from_onnx_elem_type(tensor_type.elem_type)

    for vi in onnx_graph.graph.input:
        _fill_value_info(vi)
    for vi in onnx_graph.graph.value_info:
        _fill_value_info(vi)
    for vi in onnx_graph.graph.output:
        _fill_value_info(vi)

    for ini in onnx_graph.graph.initializer:
        arr = numpy_helper.to_array(ini)
        shape_map[ini.name] = list(arr.shape)
        dtype_map[ini.name] = tflite_dtype_from_numpy(arr.dtype)

    return shape_map, dtype_map


def _graph_has_missing_rank_info(onnx_graph: onnx.ModelProto) -> bool:
    value_infos = (
        list(onnx_graph.graph.input)
        + list(onnx_graph.graph.value_info)
        + list(onnx_graph.graph.output)
    )
    for vi in value_infos:
        if not vi.type.HasField("tensor_type"):
            continue
        if not vi.type.tensor_type.HasField("shape"):
            return True
    return False


def _infer_shapes_with_fallback(onnx_graph: onnx.ModelProto) -> onnx.ModelProto:
    inferred_graph = onnx_graph
    try:
        inferred_graph = onnx.shape_inference.infer_shapes(inferred_graph)
    except Exception:
        pass

    if not _graph_has_missing_rank_info(inferred_graph):
        return inferred_graph

    try:
        from onnxruntime.tools.symbolic_shape_infer import SymbolicShapeInference
    except Exception:
        return inferred_graph

    try:
        return SymbolicShapeInference.infer_shapes(
            inferred_graph,
            auto_merge=True,
            guess_output_rank=True,
        )
    except TypeError:
        try:
            return SymbolicShapeInference.infer_shapes(inferred_graph)
        except Exception:
            return inferred_graph
    except Exception:
        return inferred_graph


class LoweringContext:
    def __init__(
        self,
        model_ir: ModelIR,
        shape_map: Dict[str, List[Any]],
        dtype_map: Dict[str, str],
        constants: Dict[str, np.ndarray],
        allow_custom_ops: bool = False,
        custom_op_allowlist: Optional[List[str]] = None,
        tensor_consumer_count: Optional[Dict[str, int]] = None,
        graph_output_names: Optional[List[str]] = None,
    ):
        self.model_ir = model_ir
        self.shape_map = shape_map
        self.dtype_map = dtype_map
        self.constants = constants
        self.allow_custom_ops = bool(allow_custom_ops)
        self.custom_op_allowlist = (
            list(custom_op_allowlist) if custom_op_allowlist is not None else None
        )
        self.tensor_consumer_count = (
            dict(tensor_consumer_count)
            if isinstance(tensor_consumer_count, dict)
            else {}
        )
        self.graph_output_names = set(graph_output_names) if graph_output_names is not None else set()
        self._serial = 0

    def _next_name(self, base: str) -> str:
        self._serial += 1
        return f"{base}_{self._serial}"

    def get_tensor_shape(self, name: str) -> List[int]:
        if name in self.model_ir.tensors:
            return list(self.model_ir.tensors[name].shape)
        shape = self.shape_map.get(name, None)
        norm_shape, _ = normalize_onnx_shape(shape)
        return norm_shape

    def get_tensor_dtype(self, name: str) -> str:
        if name in self.model_ir.tensors:
            return self.model_ir.tensors[name].dtype
        return self.dtype_map.get(name, "FLOAT32")

    def get_constant_array(self, name: str) -> Optional[np.ndarray]:
        if name in self.constants:
            return self.constants[name]
        t = self.model_ir.tensors.get(name, None)
        if t is not None and isinstance(t.data, np.ndarray):
            return t.data
        return None

    def ensure_tensor(self, name: str, dtype: Optional[str] = None, shape: Optional[List[int]] = None) -> str:
        if name == "":
            raise ValueError("Tensor name must not be empty in flatbuffer_direct lowering.")
        if name in self.model_ir.tensors:
            return name
        if dtype is None:
            dtype = self.dtype_map.get(name, "FLOAT32")
        if shape is None:
            shape = self.shape_map.get(name, None)
        shape, signature = normalize_onnx_shape(shape)
        self.model_ir.tensors[name] = TensorIR(
            name=name,
            dtype=dtype,
            shape=list(shape),
            shape_signature=list(signature),
            data=self.constants.get(name, None),
        )
        return name

    def add_const_tensor(self, base_name: str, data: np.ndarray) -> str:
        name = base_name
        if name in self.model_ir.tensors:
            name = self._next_name(base_name)
        data = np.asarray(data)
        dtype = tflite_dtype_from_numpy(data.dtype)
        shape, signature = normalize_onnx_shape(list(data.shape))
        self.model_ir.tensors[name] = TensorIR(
            name=name,
            dtype=dtype,
            shape=shape,
            shape_signature=signature,
            data=data,
        )
        self.constants[name] = data
        return name

    def add_intermediate_tensor(self, base_name: str, dtype: str, shape: List[int]) -> str:
        if base_name == "":
            raise ValueError("Tensor name must not be empty in flatbuffer_direct lowering.")
        name = base_name
        if name in self.model_ir.tensors:
            name = self._next_name(base_name)
        norm_shape, signature = normalize_onnx_shape(shape)
        self.model_ir.tensors[name] = TensorIR(
            name=name,
            dtype=dtype,
            shape=norm_shape,
            shape_signature=signature,
            data=None,
        )
        return name

    def add_operator(self, op: OperatorIR) -> None:
        self.model_ir.operators.append(op)


class _NodeWrap:
    def __init__(
        self,
        n: onnx.NodeProto,
        input_name_remap: Optional[Dict[str, str]] = None,
    ):
        self.name = n.name if n.name else n.op_type
        self.op = n.op_type
        self.attrs = {}
        for a in n.attribute:
            if a.type == onnx.AttributeProto.INT:
                self.attrs[a.name] = int(a.i)
            elif a.type == onnx.AttributeProto.FLOAT:
                self.attrs[a.name] = float(a.f)
            elif a.type == onnx.AttributeProto.INTS:
                self.attrs[a.name] = [int(v) for v in a.ints]
            elif a.type == onnx.AttributeProto.FLOATS:
                self.attrs[a.name] = [float(v) for v in a.floats]
            elif a.type == onnx.AttributeProto.STRING:
                self.attrs[a.name] = a.s.decode("utf-8")
            else:
                pass
        remap = input_name_remap if isinstance(input_name_remap, dict) else {}
        self.inputs = [
            type("In", (), {"name": remap.get(i, i)})
            for i in n.input
            if i != ""
        ]
        self.outputs = [type("Out", (), {"name": o}) for o in n.output if o != ""]


def _build_tensor_consumer_map(model_ir: ModelIR) -> Dict[str, List[int]]:
    consumers: Dict[str, List[int]] = {}
    for op_idx, op in enumerate(model_ir.operators):
        for input_name in op.inputs:
            if input_name not in consumers:
                consumers[input_name] = []
            consumers[input_name].append(op_idx)
    return consumers


def _read_transpose_perm(model_ir: ModelIR, op: OperatorIR) -> Optional[List[int]]:
    if str(op.op_type) != "TRANSPOSE":
        return None
    if len(op.inputs) < 2:
        return None
    perm_tensor_name = op.inputs[1]
    perm_tensor = model_ir.tensors.get(perm_tensor_name, None)
    if perm_tensor is None or perm_tensor.data is None:
        return None
    perm = np.asarray(perm_tensor.data).reshape(-1)
    if perm.size == 0:
        return None
    return [int(v) for v in perm.tolist()]


def _is_identity_perm(perm: List[int]) -> bool:
    return perm == [int(i) for i in range(len(perm))]


def _is_inverse_perm(perm_a: List[int], perm_b: List[int]) -> bool:
    if len(perm_a) != len(perm_b):
        return False
    rank = len(perm_a)
    if sorted(perm_a) != [int(i) for i in range(rank)]:
        return False
    if sorted(perm_b) != [int(i) for i in range(rank)]:
        return False
    for idx, value in enumerate(perm_a):
        if perm_b[value] != idx:
            return False
    return True


def _invert_perm(perm: List[int]) -> Optional[List[int]]:
    rank = len(perm)
    if sorted(perm) != [int(i) for i in range(rank)]:
        return None
    inv = [0 for _ in range(rank)]
    for idx, value in enumerate(perm):
        inv[int(value)] = int(idx)
    return inv


def _replace_tensor_inputs(model_ir: ModelIR, src_name: str, dst_name: str) -> None:
    for op in model_ir.operators:
        if len(op.inputs) > 0:
            op.inputs = [dst_name if input_name == src_name else input_name for input_name in op.inputs]


def _prune_unused_tensors(model_ir: ModelIR) -> None:
    used_tensor_names = set(model_ir.inputs + model_ir.outputs)
    for op in model_ir.operators:
        used_tensor_names.update(op.inputs)
        used_tensor_names.update(op.outputs)
    unused_tensor_names = [name for name in model_ir.tensors.keys() if name not in used_tensor_names]
    for name in unused_tensor_names:
        del model_ir.tensors[name]


def _prune_dead_operators(model_ir: ModelIR) -> Dict[str, int]:
    """
    Remove operators that do not contribute to graph outputs.

    This pass performs a reverse liveness walk from model outputs and keeps only
    operators whose outputs are required by downstream live operators or graph outputs.
    """
    if len(model_ir.operators) == 0:
        return {"removed_dead_operators": 0}

    live_tensors = set(model_ir.outputs)
    keep_flags = [False for _ in model_ir.operators]

    for op_idx in range(len(model_ir.operators) - 1, -1, -1):
        op = model_ir.operators[op_idx]
        outputs = list(op.outputs)
        if len(outputs) == 0:
            continue
        if any(out_name in live_tensors for out_name in outputs):
            keep_flags[op_idx] = True
            for input_name in op.inputs:
                live_tensors.add(input_name)

    removed = int(sum(1 for keep in keep_flags if not keep))
    if removed == 0:
        return {"removed_dead_operators": 0}

    model_ir.operators = [
        op for idx, op in enumerate(model_ir.operators) if keep_flags[idx]
    ]
    _prune_unused_tensors(model_ir)
    return {"removed_dead_operators": removed}


def _build_tensor_producer_map(model_ir: ModelIR) -> Dict[str, int]:
    producers: Dict[str, int] = {}
    for op_idx, op in enumerate(model_ir.operators):
        for output_name in op.outputs:
            producers[output_name] = op_idx
    return producers


def _rename_tensor_globally(
    model_ir: ModelIR,
    old_name: str,
    new_name: str,
) -> None:
    if old_name == new_name:
        return

    for op in model_ir.operators:
        if len(op.inputs) > 0:
            op.inputs = [new_name if input_name == old_name else input_name for input_name in op.inputs]
        if len(op.outputs) > 0:
            op.outputs = [new_name if output_name == old_name else output_name for output_name in op.outputs]

    if len(model_ir.inputs) > 0:
        model_ir.inputs = [new_name if input_name == old_name else input_name for input_name in model_ir.inputs]
    if len(model_ir.outputs) > 0:
        model_ir.outputs = [new_name if output_name == old_name else output_name for output_name in model_ir.outputs]

    old_tensor = model_ir.tensors.get(old_name, None)
    if old_tensor is None:
        if new_name in model_ir.tensors:
            del model_ir.tensors[new_name]
        return
    old_tensor.name = new_name
    if new_name in model_ir.tensors and new_name != old_name:
        del model_ir.tensors[new_name]
    del model_ir.tensors[old_name]
    model_ir.tensors[new_name] = old_tensor


def _clone_quantization(quantization: Any) -> Any:
    if quantization is None:
        return None
    if isinstance(quantization, QuantParamIR):
        return QuantParamIR(
            scale=list(quantization.scale),
            zero_point=list(quantization.zero_point),
            quantized_dimension=int(quantization.quantized_dimension),
            min=list(quantization.min) if quantization.min is not None else None,
            max=list(quantization.max) if quantization.max is not None else None,
        )
    return copy.deepcopy(quantization)


def _quantize_prelu_alpha(
    alpha: np.ndarray,
    target_dtype: str,
) -> Tuple[np.ndarray, QuantParamIR]:
    alpha = np.asarray(alpha, dtype=np.float32)
    if str(target_dtype) == "INT8":
        max_abs = float(np.max(np.abs(alpha))) if alpha.size > 0 else 0.0
        scale = max(max_abs / 127.0, 1e-8)
        quantized = np.clip(np.round(alpha / scale), -128, 127).astype(np.int8)
        return quantized, QuantParamIR(
            scale=[float(scale)],
            zero_point=[0],
            quantized_dimension=0,
        )
    if str(target_dtype) == "UINT8":
        mn = float(np.min(alpha)) if alpha.size > 0 else 0.0
        mx = float(np.max(alpha)) if alpha.size > 0 else 0.0
        scale = max((mx - mn) / 255.0, 1e-8)
        zp = int(np.round(-mn / scale))
        zp = int(np.clip(zp, 0, 255))
        quantized = np.clip(np.round(alpha / scale) + zp, 0, 255).astype(np.uint8)
        return quantized, QuantParamIR(
            scale=[float(scale)],
            zero_point=[int(zp)],
            quantized_dimension=0,
        )
    raise NotImplementedError(f"Quantized PRELU alpha requires INT8/UINT8. got={target_dtype}")


def _quantize_tensor_per_tensor(
    values: np.ndarray,
    target_dtype: str,
) -> Tuple[np.ndarray, QuantParamIR]:
    values = np.asarray(values, dtype=np.float32)
    if str(target_dtype) == "INT8":
        max_abs = float(np.max(np.abs(values))) if values.size > 0 else 0.0
        scale = max(max_abs / 127.0, 1e-8)
        quantized = np.clip(np.round(values / scale), -128, 127).astype(np.int8)
        return quantized, QuantParamIR(
            scale=[float(scale)],
            zero_point=[0],
            quantized_dimension=0,
        )
    if str(target_dtype) == "UINT8":
        mn = float(np.min(values)) if values.size > 0 else 0.0
        mx = float(np.max(values)) if values.size > 0 else 0.0
        scale = max((mx - mn) / 255.0, 1e-8)
        zp = int(np.round(-mn / scale))
        zp = int(np.clip(zp, 0, 255))
        quantized = np.clip(np.round(values / scale) + zp, 0, 255).astype(np.uint8)
        return quantized, QuantParamIR(
            scale=[float(scale)],
            zero_point=[int(zp)],
            quantized_dimension=0,
        )
    raise NotImplementedError(f"Per-tensor quantization supports INT8/UINT8 only. got={target_dtype}")


def _get_per_tensor_scale_zero_point(quantization: Any) -> Optional[Tuple[float, int]]:
    if quantization is None:
        return None
    if not _is_per_tensor_quantization(quantization):
        return None
    if isinstance(quantization, QuantParamIR):
        if len(list(quantization.scale)) == 0 or len(list(quantization.zero_point)) == 0:
            return None
        return float(quantization.scale[0]), int(quantization.zero_point[0])
    if isinstance(quantization, dict):
        try:
            scale = np.asarray(quantization.get("scale", []), dtype=np.float32).reshape(-1)
            zero_point = np.asarray(quantization.get("zero_point", []), dtype=np.int64).reshape(-1)
            if scale.size == 0 or zero_point.size == 0:
                return None
            return float(scale[0]), int(zero_point[0])
        except Exception:
            return None
    return None


def _is_same_per_tensor_quantization(
    quantization_a: Any,
    quantization_b: Any,
    *,
    rtol: float = 1e-6,
    atol: float = 1e-8,
) -> bool:
    qparams_a = _get_per_tensor_scale_zero_point(quantization_a)
    qparams_b = _get_per_tensor_scale_zero_point(quantization_b)
    if qparams_a is None or qparams_b is None:
        return False
    scale_a, zp_a = qparams_a
    scale_b, zp_b = qparams_b
    if int(zp_a) != int(zp_b):
        return False
    return bool(np.isclose(float(scale_a), float(scale_b), rtol=rtol, atol=atol))


def _quant_scale_count(quantization: Any) -> int:
    if quantization is None:
        return 0
    if isinstance(quantization, QuantParamIR):
        return len(list(quantization.scale))
    if isinstance(quantization, dict):
        scale = quantization.get("scale", None)
        if scale is None:
            return 0
        if isinstance(scale, (list, tuple)):
            return len(list(scale))
        try:
            return int(np.asarray(scale).size)
        except Exception:
            return 0
    return 0


def _is_per_tensor_quantization(quantization: Any) -> bool:
    count = _quant_scale_count(quantization)
    return count <= 1


def _permute_shape(shape: Optional[List[int]], perm: List[int]) -> Optional[List[int]]:
    if shape is None:
        return None
    rank = len(shape)
    if len(perm) != rank:
        return None
    if sorted(perm) != [int(i) for i in range(rank)]:
        return None
    return [int(shape[int(axis)]) for axis in perm]


def _shapes_equal(shape_a: Optional[List[int]], shape_b: Optional[List[int]]) -> bool:
    if shape_a is None or shape_b is None:
        return False
    if len(shape_a) != len(shape_b):
        return False
    return all(int(a) == int(b) for a, b in zip(shape_a, shape_b))


def _is_unknown_shape(shape: Optional[List[int]]) -> bool:
    if shape is None:
        return True
    if len(shape) == 0:
        return True
    if any(int(dim) < 0 for dim in shape):
        return True
    # Internal placeholder shape used before static shape propagation.
    if len(shape) == 1 and int(shape[0]) == 1:
        return True
    return False


def _shapes_match_if_known(shape_a: Optional[List[int]], shape_b: Optional[List[int]]) -> bool:
    if _is_unknown_shape(shape_a) or _is_unknown_shape(shape_b):
        return True
    return _shapes_equal(shape_a, shape_b)


def _all_per_tensor_quantized(tensors: List[Optional[TensorIR]]) -> bool:
    for tensor in tensors:
        if tensor is None:
            continue
        if tensor.quantization is None:
            continue
        if not _is_per_tensor_quantization(tensor.quantization):
            return False
    return True


def _resolve_reshape_new_shape_from_static_input(
    new_shape: List[int],
    input_signature: Optional[List[int]],
) -> Optional[List[int]]:
    if len(new_shape) == 0:
        return None
    candidate = [int(v) for v in new_shape]
    if all(int(dim) >= 0 for dim in candidate):
        return list(candidate)

    # Without allowzero metadata we avoid rewriting zero-dim requests.
    if any(int(dim) == 0 for dim in candidate):
        return None

    minus_one_indices = [idx for idx, dim in enumerate(candidate) if int(dim) == -1]
    if len(minus_one_indices) != 1:
        return None
    if input_signature is None or len(input_signature) == 0:
        return None
    if any(int(dim) <= 0 for dim in input_signature):
        return None

    known_product = 1
    for dim in candidate:
        if int(dim) == -1:
            continue
        if int(dim) <= 0:
            return None
        known_product *= int(dim)
    if known_product <= 0:
        return None

    input_product = int(np.prod(np.asarray(input_signature, dtype=np.int64)))
    if input_product <= 0 or input_product % known_product != 0:
        return None
    inferred = int(input_product // known_product)
    if inferred <= 0:
        return None
    candidate[minus_one_indices[0]] = inferred
    return candidate


def _resolve_dynamic_reshape_shapes(model_ir: ModelIR) -> Dict[str, int]:
    resolved_count = 0
    for op in model_ir.operators:
        if str(op.op_type) != "RESHAPE":
            continue
        if len(op.inputs) < 1 or len(op.outputs) != 1:
            continue

        raw_new_shape = op.options.get("newShape", [])
        try:
            new_shape = [int(v) for v in np.asarray(raw_new_shape).reshape(-1).tolist()]
        except Exception:
            continue

        input_tensor = model_ir.tensors.get(op.inputs[0], None)
        if input_tensor is None:
            continue
        input_signature = (
            list(input_tensor.shape_signature)
            if input_tensor.shape_signature is not None
            else list(input_tensor.shape)
        )
        resolved_shape = _resolve_reshape_new_shape_from_static_input(
            new_shape=new_shape,
            input_signature=input_signature,
        )
        if resolved_shape is None:
            continue

        changed = False
        if [int(v) for v in new_shape] != [int(v) for v in resolved_shape]:
            op.options["newShape"] = [int(v) for v in resolved_shape]
            changed = True

        if len(op.inputs) >= 2:
            shape_tensor = model_ir.tensors.get(op.inputs[1], None)
            if shape_tensor is not None:
                existing_shape_data = None
                if shape_tensor.data is not None:
                    existing_shape_data = [
                        int(v) for v in np.asarray(shape_tensor.data).reshape(-1).tolist()
                    ]
                if existing_shape_data != [int(v) for v in resolved_shape]:
                    shape_tensor.data = np.asarray(resolved_shape, dtype=np.int32)
                    shape_tensor.dtype = "INT32"
                    shape_tensor.shape = [int(len(resolved_shape))]
                    shape_tensor.shape_signature = [int(len(resolved_shape))]
                    changed = True

        output_tensor = model_ir.tensors.get(op.outputs[0], None)
        if output_tensor is not None:
            output_shape = list(output_tensor.shape)
            output_signature = (
                list(output_tensor.shape_signature)
                if output_tensor.shape_signature is not None
                else None
            )
            if output_shape != [int(v) for v in resolved_shape] or output_signature != [int(v) for v in resolved_shape]:
                output_tensor.shape = [int(v) for v in resolved_shape]
                output_tensor.shape_signature = [int(v) for v in resolved_shape]
                changed = True

        if changed:
            resolved_count += 1

    return {"resolved_dynamic_reshape_shapes": int(resolved_count)}


def _is_fully_known_positive_shape(shape: Optional[List[int]]) -> bool:
    if shape is None or len(shape) == 0:
        return False
    return all(int(dim) > 0 for dim in shape)


def _read_const_ints_from_tensor(tensor: Optional[TensorIR]) -> Optional[List[int]]:
    if tensor is None or tensor.data is None:
        return None
    try:
        return [int(v) for v in np.asarray(tensor.data).reshape(-1).tolist()]
    except Exception:
        return None


def _broadcast_static_shapes(
    shape_a: Optional[List[int]],
    shape_b: Optional[List[int]],
) -> Optional[List[int]]:
    if not _is_fully_known_positive_shape(shape_a) or not _is_fully_known_positive_shape(shape_b):
        return None
    a = [int(v) for v in list(shape_a)]
    b = [int(v) for v in list(shape_b)]
    rank = max(len(a), len(b))
    a = [1] * (rank - len(a)) + a
    b = [1] * (rank - len(b)) + b
    out: List[int] = []
    for dim_a, dim_b in zip(a, b):
        if int(dim_a) == int(dim_b):
            out.append(int(dim_a))
            continue
        if int(dim_a) == 1:
            out.append(int(dim_b))
            continue
        if int(dim_b) == 1:
            out.append(int(dim_a))
            continue
        return None
    return out


def _infer_conv_out_dim(
    in_size: int,
    kernel_size: int,
    stride: int,
    dilation: int,
    padding: str,
) -> Optional[int]:
    if any(int(v) <= 0 for v in [in_size, kernel_size, stride, dilation]):
        return None
    effective_kernel = int((int(kernel_size) - 1) * int(dilation) + 1)
    mode = str(padding).upper()
    if mode == "SAME":
        return int((int(in_size) + int(stride) - 1) // int(stride))
    if mode == "VALID":
        return int((int(in_size) - int(effective_kernel)) // int(stride) + 1)
    return None


def _reconcile_static_tensor_shapes(model_ir: ModelIR) -> Dict[str, int]:
    """
    Recompute static tensor shapes after aggressive graph rewrites.

    Some transpose-bridge optimizations intentionally relax local shape guards and can
    leave stale static metadata. This pass performs a conservative forward fixed-point
    shape propagation and syncs `shape` / `shape_signature` for common TFLite ops.
    """
    updated_tensors = 0

    def _update_tensor_shape(
        tensor_name: str,
        new_shape: Optional[List[int]],
    ) -> bool:
        nonlocal updated_tensors
        if new_shape is None:
            return False
        if not _is_fully_known_positive_shape(new_shape):
            return False
        tensor = model_ir.tensors.get(tensor_name, None)
        if tensor is None:
            return False
        normalized = [int(v) for v in list(new_shape)]
        signature = [int(v) for v in list(normalized)]
        if tensor.shape == normalized and tensor.shape_signature == signature:
            return False
        tensor.shape = normalized
        tensor.shape_signature = signature
        updated_tensors += 1
        return True

    max_passes = 32
    for _ in range(max_passes):
        changed = False
        for op in model_ir.operators:
            op_type = str(op.op_type)
            inputs = [str(v) for v in list(op.inputs)]
            outputs = [str(v) for v in list(op.outputs)]
            if len(outputs) == 0:
                continue

            # Pass-through ops: output shape == first input shape.
            if op_type in {"QUANTIZE", "DEQUANTIZE", "SOFTMAX", "LOGISTIC", "TANH", "RELU", "PRELU"}:
                if len(inputs) >= 1:
                    in_tensor = model_ir.tensors.get(inputs[0], None)
                    if in_tensor is not None and _is_fully_known_positive_shape(in_tensor.shape):
                        changed |= _update_tensor_shape(outputs[0], list(in_tensor.shape))
                continue

            if op_type == "TRANSPOSE" and len(inputs) >= 2 and len(outputs) == 1:
                in_tensor = model_ir.tensors.get(inputs[0], None)
                out_tensor = model_ir.tensors.get(outputs[0], None)
                perm_tensor = model_ir.tensors.get(inputs[1], None)
                perm = _read_const_ints_from_tensor(perm_tensor)
                if perm is None:
                    continue
                perm = [int(v) for v in perm]
                if in_tensor is not None and _is_fully_known_positive_shape(in_tensor.shape):
                    out_shape = _permute_shape(list(in_tensor.shape), perm)
                    changed |= _update_tensor_shape(outputs[0], out_shape)
                if out_tensor is not None and _is_fully_known_positive_shape(out_tensor.shape):
                    inv_perm = _invert_perm(perm)
                    if inv_perm is not None:
                        in_shape = _permute_shape(list(out_tensor.shape), inv_perm)
                        changed |= _update_tensor_shape(inputs[0], in_shape)
                continue

            if op_type in {"ADD", "SUB", "MUL", "DIV"} and len(inputs) >= 2 and len(outputs) == 1:
                in0 = model_ir.tensors.get(inputs[0], None)
                in1 = model_ir.tensors.get(inputs[1], None)
                shape0 = list(in0.shape) if in0 is not None else None
                shape1 = list(in1.shape) if in1 is not None else None
                out_shape = _broadcast_static_shapes(shape0, shape1)
                if out_shape is not None:
                    changed |= _update_tensor_shape(outputs[0], out_shape)
                continue

            if op_type == "CONCATENATION" and len(inputs) >= 1 and len(outputs) == 1:
                axis = op.options.get("axis", None)
                if axis is None:
                    continue
                in_shapes: List[List[int]] = []
                ranks: List[int] = []
                valid = True
                for input_name in inputs:
                    t = model_ir.tensors.get(input_name, None)
                    if t is None or not _is_fully_known_positive_shape(t.shape):
                        valid = False
                        break
                    shape = [int(v) for v in list(t.shape)]
                    in_shapes.append(shape)
                    ranks.append(len(shape))
                if not valid or len(in_shapes) == 0 or len(set(ranks)) != 1:
                    continue
                rank = int(ranks[0])
                axis_new = int(axis)
                if axis_new < 0:
                    axis_new += rank
                if axis_new < 0 or axis_new >= rank:
                    continue
                out_shape = list(in_shapes[0])
                compatible = True
                for shape in in_shapes[1:]:
                    for dim_idx in range(rank):
                        if dim_idx == axis_new:
                            continue
                        if int(shape[dim_idx]) != int(out_shape[dim_idx]):
                            compatible = False
                            break
                    if not compatible:
                        break
                    out_shape[axis_new] += int(shape[axis_new])
                if compatible:
                    changed |= _update_tensor_shape(outputs[0], out_shape)
                continue

            if op_type == "RESHAPE" and len(inputs) >= 1 and len(outputs) == 1:
                input_tensor = model_ir.tensors.get(inputs[0], None)
                if input_tensor is None:
                    continue
                input_signature = (
                    list(input_tensor.shape_signature)
                    if input_tensor.shape_signature is not None
                    else list(input_tensor.shape)
                )
                raw_new_shape = op.options.get("newShape", [])
                try:
                    new_shape = [int(v) for v in np.asarray(raw_new_shape).reshape(-1).tolist()]
                except Exception:
                    new_shape = []
                resolved = _resolve_reshape_new_shape_from_static_input(
                    new_shape=new_shape,
                    input_signature=input_signature,
                )
                if resolved is not None and _is_fully_known_positive_shape(resolved):
                    changed |= _update_tensor_shape(outputs[0], resolved)
                continue

            if op_type in {"CONV_2D", "DEPTHWISE_CONV_2D"} and len(inputs) >= 2 and len(outputs) == 1:
                in_tensor = model_ir.tensors.get(inputs[0], None)
                filter_tensor = model_ir.tensors.get(inputs[1], None)
                if (
                    in_tensor is None
                    or filter_tensor is None
                    or not _is_fully_known_positive_shape(in_tensor.shape)
                    or not _is_fully_known_positive_shape(filter_tensor.shape)
                ):
                    continue
                in_shape = [int(v) for v in list(in_tensor.shape)]
                filter_shape = [int(v) for v in list(filter_tensor.shape)]
                if len(in_shape) != 4 or len(filter_shape) != 4:
                    continue
                padding = str(op.options.get("padding", "SAME"))
                stride_h = int(op.options.get("strideH", 1))
                stride_w = int(op.options.get("strideW", 1))
                dilation_h = int(op.options.get("dilationHFactor", 1))
                dilation_w = int(op.options.get("dilationWFactor", 1))
                kernel_h = int(filter_shape[1])
                kernel_w = int(filter_shape[2])
                out_h = _infer_conv_out_dim(in_shape[1], kernel_h, stride_h, dilation_h, padding)
                out_w = _infer_conv_out_dim(in_shape[2], kernel_w, stride_w, dilation_w, padding)
                if out_h is None or out_w is None or int(out_h) <= 0 or int(out_w) <= 0:
                    continue
                if op_type == "CONV_2D":
                    out_c = int(filter_shape[0])
                else:
                    # TFLite depthwise filter layout: [1, KH, KW, OC]
                    out_c = int(filter_shape[3])
                out_shape = [int(in_shape[0]), int(out_h), int(out_w), int(out_c)]
                changed |= _update_tensor_shape(outputs[0], out_shape)
                continue

            if op_type in {"AVERAGE_POOL_2D", "MAX_POOL_2D"} and len(inputs) >= 1 and len(outputs) == 1:
                in_tensor = model_ir.tensors.get(inputs[0], None)
                if in_tensor is None or not _is_fully_known_positive_shape(in_tensor.shape):
                    continue
                in_shape = [int(v) for v in list(in_tensor.shape)]
                if len(in_shape) != 4:
                    continue

                padding = str(op.options.get("padding", "SAME")).upper()
                stride_h = int(op.options.get("strideH", 1))
                stride_w = int(op.options.get("strideW", 1))
                filter_h = int(op.options.get("filterHeight", 1))
                filter_w = int(op.options.get("filterWidth", 1))

                out_h = _infer_conv_out_dim(in_shape[1], filter_h, stride_h, 1, padding)
                out_w = _infer_conv_out_dim(in_shape[2], filter_w, stride_w, 1, padding)

                # GlobalAveragePool lowering can become stale after aggressive transpose/layout
                # rewrites. Recover invalid VALID+stride=1 average-pool metadata by snapping
                # pool kernel to the current input spatial size.
                if (
                    op_type == "AVERAGE_POOL_2D"
                    and (out_h is None or out_w is None or int(out_h) <= 0 or int(out_w) <= 0)
                    and padding == "VALID"
                    and int(stride_h) == 1
                    and int(stride_w) == 1
                ):
                    if int(filter_h) > int(in_shape[1]) or int(filter_w) > int(in_shape[2]):
                        filter_h = int(in_shape[1])
                        filter_w = int(in_shape[2])
                        op.options["filterHeight"] = int(filter_h)
                        op.options["filterWidth"] = int(filter_w)
                        out_h = _infer_conv_out_dim(in_shape[1], filter_h, stride_h, 1, padding)
                        out_w = _infer_conv_out_dim(in_shape[2], filter_w, stride_w, 1, padding)

                if out_h is None or out_w is None or int(out_h) <= 0 or int(out_w) <= 0:
                    continue
                out_shape = [int(in_shape[0]), int(out_h), int(out_w), int(in_shape[3])]
                changed |= _update_tensor_shape(outputs[0], out_shape)
                continue

            if op_type == "RESIZE_BILINEAR" and len(inputs) >= 2 and len(outputs) == 1:
                in_tensor = model_ir.tensors.get(inputs[0], None)
                size_tensor = model_ir.tensors.get(inputs[1], None)
                if in_tensor is None or not _is_fully_known_positive_shape(in_tensor.shape):
                    continue
                in_shape = [int(v) for v in list(in_tensor.shape)]
                if len(in_shape) != 4:
                    continue
                size_vals = _read_const_ints_from_tensor(size_tensor)
                if size_vals is None or len(size_vals) < 2:
                    continue
                out_h = int(size_vals[0])
                out_w = int(size_vals[1])
                if out_h <= 0 or out_w <= 0:
                    continue
                out_shape = [int(in_shape[0]), int(out_h), int(out_w), int(in_shape[3])]
                changed |= _update_tensor_shape(outputs[0], out_shape)
                continue

        if not changed:
            break

    return {"reconciled_static_tensor_shapes": int(updated_tensors)}


def _optimize_transpose_quant_dequant_bridges(model_ir: ModelIR) -> Dict[str, int]:
    """
    Fold transpose-quantize/dequantize-transpose bridges when both transposes cancel.

    Target patterns:
      X --Transpose(P)--> A --QUANTIZE--> B --Transpose(inv(P))--> Y
      X --Transpose(P)--> A --DEQUANTIZE--> B --Transpose(inv(P))--> Y

    This is safe for per-tensor quantization only.
    """
    removed_bridge_pairs = 0

    while True:
        changed = False
        consumers = _build_tensor_consumer_map(model_ir)
        producers = _build_tensor_producer_map(model_ir)

        # Pattern A:
        #   X -T(P)-> A -Q-> B -DQ-> C -T(invP)-> Y
        # -> X -Q-> B -DQ-> Y
        for pre_idx, pre_op in enumerate(model_ir.operators):
            if str(pre_op.op_type) != "TRANSPOSE" or len(pre_op.inputs) < 2 or len(pre_op.outputs) != 1:
                continue
            bridge_a = pre_op.outputs[0]
            q_users = consumers.get(bridge_a, [])
            if len(q_users) != 1:
                continue
            q_idx = int(q_users[0])
            q_op = model_ir.operators[q_idx]
            if str(q_op.op_type) != "QUANTIZE" or len(q_op.inputs) != 1 or len(q_op.outputs) != 1:
                continue
            if q_op.inputs[0] != bridge_a:
                continue

            bridge_b = q_op.outputs[0]
            dq_users = consumers.get(bridge_b, [])
            if len(dq_users) != 1:
                continue
            dq_idx = int(dq_users[0])
            dq_op = model_ir.operators[dq_idx]
            if str(dq_op.op_type) != "DEQUANTIZE" or len(dq_op.inputs) != 1 or len(dq_op.outputs) != 1:
                continue
            if dq_op.inputs[0] != bridge_b:
                continue

            bridge_c = dq_op.outputs[0]
            post_users = consumers.get(bridge_c, [])
            if len(post_users) != 1:
                continue
            post_idx = int(post_users[0])
            post_op = model_ir.operators[post_idx]
            if str(post_op.op_type) != "TRANSPOSE" or len(post_op.inputs) < 2 or len(post_op.outputs) != 1:
                continue
            if post_op.inputs[0] != bridge_c:
                continue

            perm_pre = _read_transpose_perm(model_ir, pre_op)
            perm_post = _read_transpose_perm(model_ir, post_op)
            if perm_pre is None or perm_post is None:
                continue
            if not _is_inverse_perm(perm_pre, perm_post):
                continue

            q_out_tensor = model_ir.tensors.get(bridge_b, None)
            if q_out_tensor is None or not _is_per_tensor_quantization(q_out_tensor.quantization):
                continue

            if bridge_a in model_ir.outputs or bridge_b in model_ir.outputs or bridge_c in model_ir.outputs:
                continue

            q_op.inputs = [pre_op.inputs[0]]
            dq_op.outputs = [post_op.outputs[0]]

            new_dq_out = model_ir.tensors.get(post_op.outputs[0], None)
            old_dq_out = model_ir.tensors.get(bridge_c, None)
            if new_dq_out is not None and old_dq_out is not None:
                new_dq_out.dtype = str(old_dq_out.dtype)

            for remove_idx in sorted([pre_idx, post_idx], reverse=True):
                del model_ir.operators[remove_idx]
            removed_bridge_pairs += 1
            changed = True
            break

        if changed:
            continue

        # Pattern B:
        #   X -T(P)-> A -(Q|DQ)-> B -T(invP)-> Y
        # -> X -(Q|DQ)-> Y
        for mid_idx, mid_op in enumerate(model_ir.operators):
            if str(mid_op.op_type) not in {"QUANTIZE", "DEQUANTIZE"}:
                continue
            if len(mid_op.inputs) != 1 or len(mid_op.outputs) != 1:
                continue

            bridge_in = mid_op.inputs[0]
            bridge_out = mid_op.outputs[0]

            pre_idx = producers.get(bridge_in, None)
            if pre_idx is None:
                continue
            pre_op = model_ir.operators[pre_idx]
            if str(pre_op.op_type) != "TRANSPOSE" or len(pre_op.outputs) != 1 or len(pre_op.inputs) < 2:
                continue
            if pre_op.outputs[0] != bridge_in:
                continue
            in_users = set(consumers.get(bridge_in, []))
            if int(mid_idx) not in in_users:
                continue
            can_remove_pre = in_users == {int(mid_idx)}

            post_users = [int(v) for v in consumers.get(bridge_out, []) if int(v) != int(mid_idx)]
            if len(post_users) == 0:
                continue

            perm_pre = _read_transpose_perm(model_ir, pre_op)
            if perm_pre is None:
                continue

            post_indices: List[int] = []
            post_output_names: List[str] = []
            valid_posts = True
            for post_idx in post_users:
                post_op = model_ir.operators[int(post_idx)]
                if (
                    str(post_op.op_type) != "TRANSPOSE"
                    or len(post_op.inputs) < 2
                    or len(post_op.outputs) != 1
                    or post_op.inputs[0] != bridge_out
                ):
                    valid_posts = False
                    break
                perm_post = _read_transpose_perm(model_ir, post_op)
                if perm_post is None or not _is_inverse_perm(perm_pre, perm_post):
                    valid_posts = False
                    break
                post_indices.append(int(post_idx))
                post_output_names.append(str(post_op.outputs[0]))
            if not valid_posts or len(post_indices) == 0:
                continue

            # Keep visible output names stable.
            if bridge_out in model_ir.outputs:
                continue
            if can_remove_pre and bridge_in in model_ir.outputs:
                continue
            single_post = len(post_indices) == 1
            if (not single_post) and any(
                post_output_name in model_ir.outputs for post_output_name in post_output_names
            ):
                continue

            pre_input_name = pre_op.inputs[0]
            representative_post_output_name = post_output_names[0]

            if str(mid_op.op_type) == "QUANTIZE":
                bridge_out_tensor = model_ir.tensors.get(bridge_out, None)
                if bridge_out_tensor is None:
                    continue
                if not _is_per_tensor_quantization(bridge_out_tensor.quantization):
                    continue
                mid_op.inputs = [pre_input_name]
                if single_post:
                    post_output_name = representative_post_output_name
                    mid_op.outputs = [post_output_name]
                    post_output_tensor = model_ir.tensors.get(post_output_name, None)
                    if post_output_tensor is not None:
                        post_output_tensor.dtype = str(bridge_out_tensor.dtype)
                        post_output_tensor.quantization = _clone_quantization(
                            bridge_out_tensor.quantization
                        )
                else:
                    rep_post_tensor = model_ir.tensors.get(representative_post_output_name, None)
                    if rep_post_tensor is not None:
                        bridge_out_tensor.shape = list(rep_post_tensor.shape)
                        bridge_out_tensor.shape_signature = (
                            list(rep_post_tensor.shape_signature)
                            if rep_post_tensor.shape_signature is not None
                            else list(rep_post_tensor.shape)
                        )
                        bridge_out_tensor.dtype = str(rep_post_tensor.dtype)
                    bridge_out_tensor.quantization = _clone_quantization(
                        bridge_out_tensor.quantization
                    )
            else:
                bridge_in_tensor = model_ir.tensors.get(bridge_in, None)
                if bridge_in_tensor is None:
                    continue
                if not _is_per_tensor_quantization(bridge_in_tensor.quantization):
                    continue
                mid_op.inputs = [pre_input_name]
                if single_post:
                    post_output_name = representative_post_output_name
                    mid_op.outputs = [post_output_name]
                    pre_input_tensor = model_ir.tensors.get(pre_input_name, None)
                    if pre_input_tensor is not None and pre_input_tensor.quantization is None:
                        pre_input_tensor.quantization = _clone_quantization(
                            bridge_in_tensor.quantization
                        )
                    post_output_tensor = model_ir.tensors.get(post_output_name, None)
                    bridge_out_tensor = model_ir.tensors.get(bridge_out, None)
                    if post_output_tensor is not None and bridge_out_tensor is not None:
                        post_output_tensor.dtype = str(bridge_out_tensor.dtype)
                else:
                    pre_input_tensor = model_ir.tensors.get(pre_input_name, None)
                    if pre_input_tensor is not None and pre_input_tensor.quantization is None:
                        pre_input_tensor.quantization = _clone_quantization(
                            bridge_in_tensor.quantization
                        )
                    bridge_out_tensor = model_ir.tensors.get(bridge_out, None)
                    rep_post_tensor = model_ir.tensors.get(representative_post_output_name, None)
                    if bridge_out_tensor is not None and rep_post_tensor is not None:
                        bridge_out_tensor.shape = list(rep_post_tensor.shape)
                        bridge_out_tensor.shape_signature = (
                            list(rep_post_tensor.shape_signature)
                            if rep_post_tensor.shape_signature is not None
                            else list(rep_post_tensor.shape)
                        )
                        bridge_out_tensor.dtype = str(rep_post_tensor.dtype)

            if not single_post:
                for post_output_name in post_output_names:
                    _replace_tensor_inputs(model_ir, post_output_name, bridge_out)

            remove_indices = list(post_indices)
            if can_remove_pre:
                remove_indices.append(pre_idx)
            for remove_idx in sorted(remove_indices, reverse=True):
                del model_ir.operators[remove_idx]
            removed_bridge_pairs += int(1 if single_post else max(1, len(post_indices)))
            changed = True
            break

        if not changed:
            break

    _prune_unused_tensors(model_ir)
    return {
        "removed_transpose_quantize_dequantize_bridges": int(removed_bridge_pairs),
    }


def _optimize_duplicate_transpose_fanout(model_ir: ModelIR) -> Dict[str, int]:
    """
    Deduplicate fan-out TRANSPOSE nodes with identical input and permutation.

    Target pattern:
      X --TRANSPOSE(P)--> Y0
      X --TRANSPOSE(P)--> Y1
      ...

    Rewritten:
      X --TRANSPOSE(P)--> Y0
      (all uses of Y1, ... are rewired to Y0; duplicate TRANSPOSE nodes removed)
    """
    removed_duplicates = 0

    while True:
        changed = False
        canonical_by_key: Dict[Tuple[str, Tuple[int, ...]], int] = {}

        for op_idx, op in enumerate(model_ir.operators):
            if str(op.op_type) != "TRANSPOSE":
                continue
            if len(op.inputs) < 2 or len(op.outputs) != 1:
                continue

            input_name = str(op.inputs[0])
            output_name = str(op.outputs[0])
            perm = _read_transpose_perm(model_ir, op)
            if perm is None:
                continue

            key = (input_name, tuple(int(v) for v in perm))
            canonical_idx = canonical_by_key.get(key, None)
            if canonical_idx is None:
                canonical_by_key[key] = int(op_idx)
                continue

            if output_name in model_ir.outputs:
                # Preserve user-visible graph output names.
                continue

            canonical_op = model_ir.operators[int(canonical_idx)]
            if len(canonical_op.outputs) != 1:
                continue
            canonical_output = str(canonical_op.outputs[0])
            if canonical_output == output_name:
                continue

            canonical_tensor = model_ir.tensors.get(canonical_output, None)
            duplicate_tensor = model_ir.tensors.get(output_name, None)
            if canonical_tensor is not None and duplicate_tensor is not None:
                if canonical_tensor.shape == [1] and duplicate_tensor.shape != [1]:
                    canonical_tensor.shape = [int(v) for v in list(duplicate_tensor.shape)]
                    canonical_tensor.shape_signature = (
                        [int(v) for v in list(duplicate_tensor.shape_signature)]
                        if duplicate_tensor.shape_signature is not None
                        else [int(v) for v in list(duplicate_tensor.shape)]
                    )
                if canonical_tensor.quantization is None and duplicate_tensor.quantization is not None:
                    canonical_tensor.quantization = _clone_quantization(duplicate_tensor.quantization)
                if str(canonical_tensor.dtype) == "FLOAT32" and str(duplicate_tensor.dtype) != "FLOAT32":
                    canonical_tensor.dtype = str(duplicate_tensor.dtype)

            _replace_tensor_inputs(model_ir, output_name, canonical_output)
            del model_ir.operators[int(op_idx)]
            removed_duplicates += 1
            changed = True
            break

        if not changed:
            break

    _prune_unused_tensors(model_ir)
    return {
        "removed_duplicate_transpose_fanout": int(removed_duplicates),
    }


def _optimize_transpose_dequant_prelu_quantize_bridges(model_ir: ModelIR) -> Dict[str, int]:
    """
    Fold transpose wrappers around DEQUANTIZE->PRELU->QUANTIZE chains.

    Target pattern:
      Xq --Transpose(P)--> Aq --DEQUANTIZE--> A --PRELU(alpha)--> B --QUANTIZE--> Bq --Transpose(inv(P))--> Yq

    Rewritten:
      Xq --DEQUANTIZE--> A --PRELU(alpha')--> B --QUANTIZE--> Yq

    Safety conditions:
    - Chain is linear (single consumer at each bridge tensor)
    - Pre/Post transpose permutations are exact inverses
    - Quantized tensors use per-tensor quantization only
    - PRELU alpha tensor is constant if rank remap is required
    """
    removed_prelu_bridges = 0

    while True:
        changed = False
        consumers = _build_tensor_consumer_map(model_ir)

        for pre_idx, pre_op in enumerate(model_ir.operators):
            if str(pre_op.op_type) != "TRANSPOSE" or len(pre_op.inputs) < 2 or len(pre_op.outputs) != 1:
                continue

            perm_pre = _read_transpose_perm(model_ir, pre_op)
            if perm_pre is None:
                continue

            bridge_q_in = str(pre_op.outputs[0])
            dq_users = consumers.get(bridge_q_in, [])
            if len(dq_users) != 1:
                continue
            dq_idx = int(dq_users[0])
            dq_op = model_ir.operators[dq_idx]
            if str(dq_op.op_type) != "DEQUANTIZE" or len(dq_op.inputs) != 1 or len(dq_op.outputs) != 1:
                continue
            if str(dq_op.inputs[0]) != bridge_q_in:
                continue

            bridge_f_in = str(dq_op.outputs[0])
            prelu_users = consumers.get(bridge_f_in, [])
            if len(prelu_users) != 1:
                continue
            prelu_idx = int(prelu_users[0])
            prelu_op = model_ir.operators[prelu_idx]
            if str(prelu_op.op_type) != "PRELU" or len(prelu_op.inputs) != 2 or len(prelu_op.outputs) != 1:
                continue
            if str(prelu_op.inputs[0]) != bridge_f_in:
                continue

            bridge_f_out = str(prelu_op.outputs[0])
            q_users = consumers.get(bridge_f_out, [])
            if len(q_users) != 1:
                continue
            q_idx = int(q_users[0])
            q_op = model_ir.operators[q_idx]
            if str(q_op.op_type) != "QUANTIZE" or len(q_op.inputs) != 1 or len(q_op.outputs) != 1:
                continue
            if str(q_op.inputs[0]) != bridge_f_out:
                continue

            bridge_q_out = str(q_op.outputs[0])
            post_users = consumers.get(bridge_q_out, [])
            if len(post_users) != 1:
                continue
            post_idx = int(post_users[0])
            post_op = model_ir.operators[post_idx]
            if str(post_op.op_type) != "TRANSPOSE" or len(post_op.inputs) < 2 or len(post_op.outputs) != 1:
                continue
            if str(post_op.inputs[0]) != bridge_q_out:
                continue

            perm_post = _read_transpose_perm(model_ir, post_op)
            if perm_post is None or not _is_inverse_perm(perm_pre, perm_post):
                continue

            # Keep user-visible output names stable and avoid breaking observable intermediates.
            if (
                bridge_q_in in model_ir.outputs
                or bridge_f_in in model_ir.outputs
                or bridge_f_out in model_ir.outputs
                or bridge_q_out in model_ir.outputs
            ):
                continue

            q_src_name = str(pre_op.inputs[0])
            q_dst_name = str(post_op.outputs[0])
            if q_src_name in model_ir.outputs:
                continue

            q_src_tensor = model_ir.tensors.get(q_src_name, None)
            q_mid_in_tensor = model_ir.tensors.get(bridge_q_in, None)
            q_mid_out_tensor = model_ir.tensors.get(bridge_q_out, None)
            q_dst_tensor = model_ir.tensors.get(q_dst_name, None)
            if not _all_per_tensor_quantized([q_src_tensor, q_mid_in_tensor, q_mid_out_tensor, q_dst_tensor]):
                continue

            # PRELU alpha layout follows data layout. When alpha rank matches the transposed rank,
            # remap alpha using post permutation so PRELU can run directly on non-transposed data.
            alpha_name = str(prelu_op.inputs[1])
            alpha_tensor = model_ir.tensors.get(alpha_name, None)
            if alpha_tensor is not None and isinstance(alpha_tensor.data, np.ndarray):
                alpha_data = np.asarray(alpha_tensor.data)
                if alpha_data.ndim == len(perm_post):
                    alpha_data = np.transpose(alpha_data, axes=perm_post)
                    alpha_tensor.data = alpha_data
                    alpha_tensor.shape = [int(v) for v in alpha_data.shape]
                    alpha_tensor.shape_signature = [int(v) for v in alpha_data.shape]

            dq_op.inputs = [q_src_name]
            q_op.outputs = [q_dst_name]

            # Update bridge tensor metadata to the non-transposed layout.
            q_src_shape = list(q_src_tensor.shape) if q_src_tensor is not None else None
            q_src_signature = (
                list(q_src_tensor.shape_signature)
                if q_src_tensor is not None and q_src_tensor.shape_signature is not None
                else q_src_shape
            )
            if q_src_shape is not None:
                dq_out_tensor = model_ir.tensors.get(bridge_f_in, None)
                prelu_out_tensor = model_ir.tensors.get(bridge_f_out, None)
                if dq_out_tensor is not None:
                    dq_out_tensor.shape = [int(v) for v in q_src_shape]
                    dq_out_tensor.shape_signature = (
                        [int(v) for v in q_src_signature]
                        if q_src_signature is not None
                        else [int(v) for v in q_src_shape]
                    )
                if prelu_out_tensor is not None:
                    prelu_out_tensor.shape = [int(v) for v in q_src_shape]
                    prelu_out_tensor.shape_signature = (
                        [int(v) for v in q_src_signature]
                        if q_src_signature is not None
                        else [int(v) for v in q_src_shape]
                    )

            if q_dst_tensor is not None and q_mid_out_tensor is not None:
                q_dst_tensor.dtype = str(q_mid_out_tensor.dtype)
                q_dst_tensor.quantization = _clone_quantization(q_mid_out_tensor.quantization)

            for remove_idx in sorted([pre_idx, post_idx], reverse=True):
                del model_ir.operators[remove_idx]
            removed_prelu_bridges += 1
            changed = True
            break

        if not changed:
            break

    _prune_unused_tensors(model_ir)
    return {
        "removed_transpose_dequant_prelu_quantize_bridges": int(removed_prelu_bridges),
    }


def _optimize_transpose_dequant_mul_add_prelu_quantize_bridges(model_ir: ModelIR) -> Dict[str, int]:
    """
    Fold transpose wrappers around DEQUANTIZE->MUL->ADD->PRELU->QUANTIZE chains.

    Target pattern:
      Xq --Transpose(P)--> Aq --DEQUANTIZE--> A --MUL(c1)--> B --ADD(c2)--> C
         --PRELU(alpha)--> D --QUANTIZE--> Dq --Transpose(inv(P))--> Yq

    Rewritten:
      Xq --DEQUANTIZE--> A --MUL(c1')--> B --ADD(c2')--> C --PRELU(alpha')--> D --QUANTIZE--> Yq

    Safety conditions:
    - Chain is linear (single consumer at each bridge tensor)
    - Pre/Post transpose permutations are exact inverses
    - Quantized tensors use per-tensor quantization only
    - Side inputs c1/c2/alpha are constants; rank-matched constants are layout-remapped
    """
    removed_mul_add_prelu_bridges = 0

    def _unique_tensor_name(base: str) -> str:
        name = str(base)
        suffix = 1
        while name in model_ir.tensors:
            name = f"{base}_{suffix}"
            suffix += 1
        return name

    while True:
        changed = False
        consumers = _build_tensor_consumer_map(model_ir)

        for pre_idx, pre_op in enumerate(model_ir.operators):
            if str(pre_op.op_type) != "TRANSPOSE" or len(pre_op.inputs) < 2 or len(pre_op.outputs) != 1:
                continue

            perm_pre = _read_transpose_perm(model_ir, pre_op)
            if perm_pre is None:
                continue

            bridge_q_in = str(pre_op.outputs[0])
            dq_users = consumers.get(bridge_q_in, [])
            if len(dq_users) != 1:
                continue
            dq_idx = int(dq_users[0])
            dq_op = model_ir.operators[dq_idx]
            if str(dq_op.op_type) != "DEQUANTIZE" or len(dq_op.inputs) != 1 or len(dq_op.outputs) != 1:
                continue
            if str(dq_op.inputs[0]) != bridge_q_in:
                continue

            bridge_f_in = str(dq_op.outputs[0])
            mul_users = consumers.get(bridge_f_in, [])
            if len(mul_users) != 1:
                continue
            mul_idx = int(mul_users[0])
            mul_op = model_ir.operators[mul_idx]
            if str(mul_op.op_type) != "MUL" or len(mul_op.inputs) != 2 or len(mul_op.outputs) != 1:
                continue
            mul_in0 = str(mul_op.inputs[0])
            mul_in1 = str(mul_op.inputs[1])
            if mul_in0 == bridge_f_in:
                mul_const_input_index = 1
            elif mul_in1 == bridge_f_in:
                mul_const_input_index = 0
            else:
                continue

            bridge_mul_out = str(mul_op.outputs[0])
            add_users = consumers.get(bridge_mul_out, [])
            if len(add_users) != 1:
                continue
            add_idx = int(add_users[0])
            add_op = model_ir.operators[add_idx]
            if str(add_op.op_type) != "ADD" or len(add_op.inputs) != 2 or len(add_op.outputs) != 1:
                continue
            add_in0 = str(add_op.inputs[0])
            add_in1 = str(add_op.inputs[1])
            if add_in0 == bridge_mul_out:
                add_const_input_index = 1
            elif add_in1 == bridge_mul_out:
                add_const_input_index = 0
            else:
                continue

            bridge_add_out = str(add_op.outputs[0])
            prelu_users = consumers.get(bridge_add_out, [])
            if len(prelu_users) != 1:
                continue
            prelu_idx = int(prelu_users[0])
            prelu_op = model_ir.operators[prelu_idx]
            if str(prelu_op.op_type) != "PRELU" or len(prelu_op.inputs) != 2 or len(prelu_op.outputs) != 1:
                continue
            if str(prelu_op.inputs[0]) != bridge_add_out:
                continue

            bridge_prelu_out = str(prelu_op.outputs[0])
            q_users = consumers.get(bridge_prelu_out, [])
            if len(q_users) != 1:
                continue
            q_idx = int(q_users[0])
            q_op = model_ir.operators[q_idx]
            if str(q_op.op_type) != "QUANTIZE" or len(q_op.inputs) != 1 or len(q_op.outputs) != 1:
                continue
            if str(q_op.inputs[0]) != bridge_prelu_out:
                continue

            bridge_q_out = str(q_op.outputs[0])
            post_users = consumers.get(bridge_q_out, [])
            if len(post_users) != 1:
                continue
            post_idx = int(post_users[0])
            post_op = model_ir.operators[post_idx]
            if str(post_op.op_type) != "TRANSPOSE" or len(post_op.inputs) < 2 or len(post_op.outputs) != 1:
                continue
            if str(post_op.inputs[0]) != bridge_q_out:
                continue

            perm_post = _read_transpose_perm(model_ir, post_op)
            if perm_post is None or not _is_inverse_perm(perm_pre, perm_post):
                continue

            # Keep user-visible output names stable and avoid breaking observable intermediates.
            if (
                bridge_q_in in model_ir.outputs
                or bridge_f_in in model_ir.outputs
                or bridge_mul_out in model_ir.outputs
                or bridge_add_out in model_ir.outputs
                or bridge_prelu_out in model_ir.outputs
                or bridge_q_out in model_ir.outputs
            ):
                continue

            q_src_name = str(pre_op.inputs[0])
            q_dst_name = str(post_op.outputs[0])
            if q_src_name in model_ir.outputs:
                continue

            q_src_tensor = model_ir.tensors.get(q_src_name, None)
            q_mid_in_tensor = model_ir.tensors.get(bridge_q_in, None)
            q_mid_out_tensor = model_ir.tensors.get(bridge_q_out, None)
            q_dst_tensor = model_ir.tensors.get(q_dst_name, None)
            if not _all_per_tensor_quantized([q_src_tensor, q_mid_in_tensor, q_mid_out_tensor, q_dst_tensor]):
                continue

            mul_const_name = str(mul_op.inputs[mul_const_input_index])
            add_const_name = str(add_op.inputs[add_const_input_index])
            alpha_name = str(prelu_op.inputs[1])
            mul_const_tensor = model_ir.tensors.get(mul_const_name, None)
            add_const_tensor = model_ir.tensors.get(add_const_name, None)
            alpha_tensor = model_ir.tensors.get(alpha_name, None)
            if (
                mul_const_tensor is None
                or add_const_tensor is None
                or alpha_tensor is None
                or not isinstance(mul_const_tensor.data, np.ndarray)
                or not isinstance(add_const_tensor.data, np.ndarray)
                or not isinstance(alpha_tensor.data, np.ndarray)
            ):
                continue

            def _remap_constant_input_for_op(
                *,
                op: OperatorIR,
                op_idx: int,
                input_index: int,
            ) -> bool:
                const_name = str(op.inputs[input_index])
                const_tensor = model_ir.tensors.get(const_name, None)
                if const_tensor is None or not isinstance(const_tensor.data, np.ndarray):
                    return False
                const_data = np.asarray(const_tensor.data)
                if const_data.ndim != len(perm_post):
                    return True
                transposed_data = np.transpose(const_data, axes=perm_post)
                const_users = consumers.get(const_name, [])
                if len(const_users) == 1 and int(const_users[0]) == int(op_idx):
                    const_tensor.data = np.asarray(transposed_data)
                    const_tensor.shape = [int(v) for v in transposed_data.shape]
                    const_tensor.shape_signature = [int(v) for v in transposed_data.shape]
                    return True
                new_name = _unique_tensor_name(f"{const_name}_nhwc")
                model_ir.tensors[new_name] = TensorIR(
                    name=new_name,
                    dtype=str(const_tensor.dtype),
                    shape=[int(v) for v in transposed_data.shape],
                    shape_signature=[int(v) for v in transposed_data.shape],
                    data=np.asarray(transposed_data),
                    is_variable=False,
                    quantization=_clone_quantization(const_tensor.quantization),
                )
                op.inputs[input_index] = new_name
                return True

            if not _remap_constant_input_for_op(
                op=mul_op,
                op_idx=mul_idx,
                input_index=mul_const_input_index,
            ):
                continue
            if not _remap_constant_input_for_op(
                op=add_op,
                op_idx=add_idx,
                input_index=add_const_input_index,
            ):
                continue
            if not _remap_constant_input_for_op(
                op=prelu_op,
                op_idx=prelu_idx,
                input_index=1,
            ):
                continue

            dq_op.inputs = [q_src_name]
            q_op.outputs = [q_dst_name]

            # Update bridge tensor metadata to the non-transposed layout.
            q_src_shape = list(q_src_tensor.shape) if q_src_tensor is not None else None
            q_src_signature = (
                list(q_src_tensor.shape_signature)
                if q_src_tensor is not None and q_src_tensor.shape_signature is not None
                else q_src_shape
            )
            if q_src_shape is not None:
                dq_out_tensor = model_ir.tensors.get(bridge_f_in, None)
                mul_out_tensor = model_ir.tensors.get(bridge_mul_out, None)
                add_out_tensor = model_ir.tensors.get(bridge_add_out, None)
                prelu_out_tensor = model_ir.tensors.get(bridge_prelu_out, None)
                if dq_out_tensor is not None:
                    dq_out_tensor.shape = [int(v) for v in q_src_shape]
                    dq_out_tensor.shape_signature = (
                        [int(v) for v in q_src_signature]
                        if q_src_signature is not None
                        else [int(v) for v in q_src_shape]
                    )
                if mul_out_tensor is not None:
                    mul_out_tensor.shape = [int(v) for v in q_src_shape]
                    mul_out_tensor.shape_signature = (
                        [int(v) for v in q_src_signature]
                        if q_src_signature is not None
                        else [int(v) for v in q_src_shape]
                    )
                if add_out_tensor is not None:
                    add_out_tensor.shape = [int(v) for v in q_src_shape]
                    add_out_tensor.shape_signature = (
                        [int(v) for v in q_src_signature]
                        if q_src_signature is not None
                        else [int(v) for v in q_src_shape]
                    )
                if prelu_out_tensor is not None:
                    prelu_out_tensor.shape = [int(v) for v in q_src_shape]
                    prelu_out_tensor.shape_signature = (
                        [int(v) for v in q_src_signature]
                        if q_src_signature is not None
                        else [int(v) for v in q_src_shape]
                    )

            if q_dst_tensor is not None and q_mid_out_tensor is not None:
                q_dst_tensor.dtype = str(q_mid_out_tensor.dtype)
                q_dst_tensor.quantization = _clone_quantization(q_mid_out_tensor.quantization)

            for remove_idx in sorted([pre_idx, post_idx], reverse=True):
                del model_ir.operators[remove_idx]
            removed_mul_add_prelu_bridges += 1
            changed = True
            break

        if not changed:
            break

    _prune_unused_tensors(model_ir)
    return {
        "removed_transpose_dequant_mul_add_prelu_quantize_bridges": int(removed_mul_add_prelu_bridges),
    }


def _optimize_transpose_dequant_prelu_transpose_bridges(model_ir: ModelIR) -> Dict[str, int]:
    """
    Fold transpose wrappers around DEQUANTIZE->PRELU chains.

    Target pattern:
      Xq --Transpose(P)--> Aq --DEQUANTIZE--> A --PRELU(alpha)--> B --Transpose(inv(P))--> Y

    Rewritten:
      Xq --DEQUANTIZE--> A' --PRELU(alpha')--> Y

    Safety conditions:
    - Chain is linear (single consumer at each bridge tensor)
    - Pre/Post transpose permutations are exact inverses
    - Quantized tensors on DEQUANTIZE input path are per-tensor
    """
    removed_prelu_transpose_bridges = 0

    def _unique_tensor_name(base: str) -> str:
        name = str(base)
        suffix = 1
        while name in model_ir.tensors:
            name = f"{base}_{suffix}"
            suffix += 1
        return name

    while True:
        changed = False
        consumers = _build_tensor_consumer_map(model_ir)

        for pre_idx, pre_op in enumerate(model_ir.operators):
            if str(pre_op.op_type) != "TRANSPOSE" or len(pre_op.inputs) < 2 or len(pre_op.outputs) != 1:
                continue

            perm_pre = _read_transpose_perm(model_ir, pre_op)
            if perm_pre is None:
                continue

            bridge_q_in = str(pre_op.outputs[0])
            dq_users = consumers.get(bridge_q_in, [])
            if len(dq_users) != 1:
                continue
            dq_idx = int(dq_users[0])
            dq_op = model_ir.operators[dq_idx]
            if str(dq_op.op_type) != "DEQUANTIZE" or len(dq_op.inputs) != 1 or len(dq_op.outputs) != 1:
                continue
            if str(dq_op.inputs[0]) != bridge_q_in:
                continue

            bridge_f_in = str(dq_op.outputs[0])
            prelu_users = consumers.get(bridge_f_in, [])
            if len(prelu_users) != 1:
                continue
            prelu_idx = int(prelu_users[0])
            prelu_op = model_ir.operators[prelu_idx]
            if str(prelu_op.op_type) != "PRELU" or len(prelu_op.inputs) != 2 or len(prelu_op.outputs) != 1:
                continue
            if str(prelu_op.inputs[0]) != bridge_f_in:
                continue

            bridge_f_out = str(prelu_op.outputs[0])
            post_users = consumers.get(bridge_f_out, [])
            if len(post_users) != 1:
                continue
            post_idx = int(post_users[0])
            post_op = model_ir.operators[post_idx]
            if str(post_op.op_type) != "TRANSPOSE" or len(post_op.inputs) < 2 or len(post_op.outputs) != 1:
                continue
            if str(post_op.inputs[0]) != bridge_f_out:
                continue

            perm_post = _read_transpose_perm(model_ir, post_op)
            if perm_post is None or not _is_inverse_perm(perm_pre, perm_post):
                continue

            if (
                bridge_q_in in model_ir.outputs
                or bridge_f_in in model_ir.outputs
                or bridge_f_out in model_ir.outputs
                or post_op.outputs[0] in model_ir.outputs
            ):
                continue

            q_src_name = str(pre_op.inputs[0])
            y_name = str(post_op.outputs[0])
            q_src_tensor = model_ir.tensors.get(q_src_name, None)
            q_mid_tensor = model_ir.tensors.get(bridge_q_in, None)
            if not _all_per_tensor_quantized([q_src_tensor, q_mid_tensor]):
                continue

            alpha_name = str(prelu_op.inputs[1])
            alpha_tensor = model_ir.tensors.get(alpha_name, None)
            if alpha_tensor is not None:
                alpha_data = alpha_tensor.data
                if isinstance(alpha_data, np.ndarray) and alpha_data.ndim == len(perm_post):
                    transposed_alpha = np.transpose(alpha_data, axes=perm_post)
                    alpha_users = consumers.get(alpha_name, [])
                    if len(alpha_users) == 1 and int(alpha_users[0]) == int(prelu_idx):
                        alpha_tensor.data = transposed_alpha
                        alpha_tensor.shape = [int(v) for v in transposed_alpha.shape]
                        alpha_tensor.shape_signature = [int(v) for v in transposed_alpha.shape]
                    else:
                        new_alpha_name = _unique_tensor_name(f"{alpha_name}_nhwc")
                        model_ir.tensors[new_alpha_name] = TensorIR(
                            name=new_alpha_name,
                            dtype=str(alpha_tensor.dtype),
                            shape=[int(v) for v in transposed_alpha.shape],
                            shape_signature=[int(v) for v in transposed_alpha.shape],
                            data=np.asarray(transposed_alpha),
                            is_variable=False,
                            quantization=_clone_quantization(alpha_tensor.quantization),
                        )
                        prelu_op.inputs[1] = new_alpha_name

            dq_op.inputs = [q_src_name]
            prelu_op.outputs = [y_name]

            q_src_shape = list(q_src_tensor.shape) if q_src_tensor is not None else None
            q_src_signature = (
                list(q_src_tensor.shape_signature)
                if q_src_tensor is not None and q_src_tensor.shape_signature is not None
                else q_src_shape
            )
            if q_src_shape is not None:
                dq_out_tensor = model_ir.tensors.get(bridge_f_in, None)
                y_tensor = model_ir.tensors.get(y_name, None)
                if dq_out_tensor is not None:
                    dq_out_tensor.shape = [int(v) for v in q_src_shape]
                    dq_out_tensor.shape_signature = (
                        [int(v) for v in q_src_signature]
                        if q_src_signature is not None
                        else [int(v) for v in q_src_shape]
                    )
                if y_tensor is not None:
                    y_tensor.shape = [int(v) for v in q_src_shape]
                    y_tensor.shape_signature = (
                        [int(v) for v in q_src_signature]
                        if q_src_signature is not None
                        else [int(v) for v in q_src_shape]
                    )

            for remove_idx in sorted([pre_idx, post_idx], reverse=True):
                del model_ir.operators[remove_idx]
            removed_prelu_transpose_bridges += 1
            changed = True
            break

        if not changed:
            break

    _prune_unused_tensors(model_ir)
    return {
        "removed_transpose_dequant_prelu_transpose_bridges": int(removed_prelu_transpose_bridges),
    }


def _optimize_transpose_binary_bridges(model_ir: ModelIR) -> Dict[str, int]:
    """
    Fold symmetric transpose wrappers around elementwise binary ops.

    Target pattern:
      A --Transpose(P)--> A'
      B --Transpose(P)--> B'
      A',B' --(ADD|SUB|MUL|DIV)--> C'
      C' --Transpose(inv(P))--> C

    Constraints:
    - Both pre-transposed tensors are consumed only by the binary op.
    - Binary output is consumed only by the post transpose.
    - pre/post permutations are exact inverses.
    """
    removed_binary_bridges = 0
    removed_binary_asymmetric_bridges = 0
    binary_ops = {"ADD", "SUB", "MUL", "DIV"}
    enable_fanout_pattern_c = False

    while True:
        changed = False
        consumers = _build_tensor_consumer_map(model_ir)
        producers = _build_tensor_producer_map(model_ir)

        for mid_idx, mid_op in enumerate(model_ir.operators):
            if str(mid_op.op_type) not in binary_ops:
                continue
            if len(mid_op.inputs) != 2 or len(mid_op.outputs) != 1:
                continue

            in0_name = mid_op.inputs[0]
            in1_name = mid_op.inputs[1]
            out_name = mid_op.outputs[0]

            pre0_idx = producers.get(in0_name, None)
            pre1_idx = producers.get(in1_name, None)
            if pre0_idx is None or pre1_idx is None:
                continue
            pre0_op = model_ir.operators[pre0_idx]
            pre1_op = model_ir.operators[pre1_idx]
            if str(pre0_op.op_type) != "TRANSPOSE" or str(pre1_op.op_type) != "TRANSPOSE":
                continue
            if len(pre0_op.inputs) < 2 or len(pre1_op.inputs) < 2:
                continue
            if len(pre0_op.outputs) != 1 or len(pre1_op.outputs) != 1:
                continue
            if pre0_op.outputs[0] != in0_name or pre1_op.outputs[0] != in1_name:
                continue

            in0_users = set(consumers.get(in0_name, []))
            in1_users = set(consumers.get(in1_name, []))
            if in0_users != {int(mid_idx)} or in1_users != {int(mid_idx)}:
                continue

            perm0 = _read_transpose_perm(model_ir, pre0_op)
            perm1 = _read_transpose_perm(model_ir, pre1_op)
            if perm0 is None or perm1 is None:
                continue
            if perm0 != perm1:
                continue

            out_users = {int(v) for v in consumers.get(out_name, []) if int(v) != int(mid_idx)}
            if len(out_users) == 0:
                continue

            post_candidates: List[int] = []
            for user_idx in sorted(list(out_users)):
                post_candidate = model_ir.operators[int(user_idx)]
                if (
                    str(post_candidate.op_type) != "TRANSPOSE"
                    or len(post_candidate.inputs) < 2
                    or len(post_candidate.outputs) != 1
                    or post_candidate.inputs[0] != out_name
                ):
                    continue
                perm_post = _read_transpose_perm(model_ir, post_candidate)
                if perm_post is None or not _is_inverse_perm(perm0, perm_post):
                    continue
                post_candidates.append(int(user_idx))

            synthesize_post = False
            if len(post_candidates) == 0:
                if any(str(model_ir.operators[int(user_idx)].op_type) == "TRANSPOSE" for user_idx in out_users):
                    continue
                synthesize_post = True
            if len(post_candidates) > 1:
                continue
            post_idx = None
            post_op = None
            post_perm_name = None
            keep_post_for_fanout = False
            if not synthesize_post:
                post_idx = int(post_candidates[0])
                post_op = model_ir.operators[post_idx]
                post_perm_name = post_op.inputs[1]
                extra_users = set(out_users) - {int(post_idx)}
                # Extra users may remain on out_name, but must not be transpose producers.
                if any(str(model_ir.operators[int(user_idx)].op_type) == "TRANSPOSE" for user_idx in extra_users):
                    continue
                keep_post_for_fanout = len(extra_users) > 0

            # Keep user-visible output names stable.
            if in0_name in model_ir.outputs or in1_name in model_ir.outputs or out_name in model_ir.outputs:
                continue

            original_in0 = pre0_op.inputs[0]
            original_in1 = pre1_op.inputs[0]
            final_out = (
                post_op.outputs[0]
                if post_op is not None
                else out_name
            )

            mid_op.inputs = [original_in0, original_in1]
            if synthesize_post:
                out_tensor = model_ir.tensors.get(out_name, None)
                if out_tensor is None:
                    continue
                if not _is_per_tensor_quantization(out_tensor.quantization):
                    continue
                inv_perm = _invert_perm(perm0)
                if inv_perm is None:
                    continue
                raw_shape = _permute_shape(list(out_tensor.shape), inv_perm)
                raw_signature = _permute_shape(
                    list(out_tensor.shape_signature)
                    if out_tensor.shape_signature is not None
                    else list(out_tensor.shape),
                    inv_perm,
                )
                if raw_shape is None:
                    continue
                if raw_signature is None:
                    raw_signature = list(raw_shape)
                raw_out_name = f"{out_name}__raw"
                serial = 0
                while raw_out_name in model_ir.tensors:
                    serial += 1
                    raw_out_name = f"{out_name}__raw_{serial}"
                model_ir.tensors[raw_out_name] = TensorIR(
                    name=raw_out_name,
                    dtype=str(out_tensor.dtype),
                    shape=[int(v) for v in raw_shape],
                    shape_signature=[int(v) for v in raw_signature],
                    data=None,
                    is_variable=False,
                    quantization=_clone_quantization(out_tensor.quantization),
                )
                mid_op.outputs = [raw_out_name]
            else:
                mid_op.outputs = [final_out]

            final_tensor = model_ir.tensors.get(final_out, None)
            old_out_tensor = model_ir.tensors.get(out_name, None)
            if final_tensor is not None and old_out_tensor is not None:
                final_tensor.dtype = str(old_out_tensor.dtype)
                if final_tensor.quantization is None and old_out_tensor.quantization is not None:
                    final_tensor.quantization = _clone_quantization(old_out_tensor.quantization)

            if keep_post_for_fanout and post_op is not None and post_perm_name is not None:
                perm_tensor = model_ir.tensors.get(post_perm_name, None)
                if perm_tensor is not None:
                    perm_tensor.data = np.asarray(perm0, dtype=np.int32)
                post_op.inputs = [final_out, post_perm_name]
                post_op.outputs = [out_name]

            if synthesize_post:
                reuse_perm_name = pre0_op.inputs[1]
                insert_at = int(min(out_users))
                remove_indices = sorted(list({int(pre0_idx), int(pre1_idx)}), reverse=True)
                shift = sum(1 for idx in remove_indices if int(idx) < insert_at)
                for remove_idx in remove_indices:
                    del model_ir.operators[remove_idx]
                adjusted_insert_at = int(insert_at - shift)
                model_ir.operators.insert(
                    adjusted_insert_at,
                    OperatorIR(
                        op_type="TRANSPOSE",
                        inputs=[raw_out_name, reuse_perm_name],
                        outputs=[out_name],
                    ),
                )
            else:
                remove_indices = (
                    sorted(list({int(pre0_idx), int(pre1_idx)}), reverse=True)
                    if keep_post_for_fanout
                    else sorted(list({int(pre0_idx), int(pre1_idx), int(post_idx)}), reverse=True)
                )
                for remove_idx in remove_indices:
                    del model_ir.operators[remove_idx]
            removed_binary_bridges += 1
            changed = True
            break

        if changed:
            continue

        # Pattern B:
        #   A,B' --(ADD|SUB|MUL|DIV)--> C' --T(invP)--> C
        # where B' is produced by T(P) from B.
        # Rewrite to:
        #   A' --T(invP)--> A'
        #   A',B --(ADD|SUB|MUL|DIV)--> C
        # This removes one transpose around the binary op.
        for mid_idx, mid_op in enumerate(model_ir.operators):
            if str(mid_op.op_type) not in binary_ops:
                continue
            if len(mid_op.inputs) != 2 or len(mid_op.outputs) != 1:
                continue

            in0_name = mid_op.inputs[0]
            in1_name = mid_op.inputs[1]
            out_name = mid_op.outputs[0]

            pre0_idx = producers.get(in0_name, None)
            pre1_idx = producers.get(in1_name, None)
            pre0_is_transpose = False
            pre1_is_transpose = False
            pre0_op = None
            pre1_op = None

            if pre0_idx is not None:
                pre0_op = model_ir.operators[int(pre0_idx)]
                pre0_is_transpose = (
                    str(pre0_op.op_type) == "TRANSPOSE"
                    and len(pre0_op.inputs) >= 2
                    and len(pre0_op.outputs) == 1
                    and pre0_op.outputs[0] == in0_name
                )
            if pre1_idx is not None:
                pre1_op = model_ir.operators[int(pre1_idx)]
                pre1_is_transpose = (
                    str(pre1_op.op_type) == "TRANSPOSE"
                    and len(pre1_op.inputs) >= 2
                    and len(pre1_op.outputs) == 1
                    and pre1_op.outputs[0] == in1_name
                )

            # Asymmetric-only pass: exactly one input is wrapped by pre-transpose.
            if pre0_is_transpose == pre1_is_transpose:
                continue

            if pre0_is_transpose:
                pre_idx = int(pre0_idx)
                pre_op = pre0_op
                transposed_input_name = in0_name
                plain_input_name = in1_name
                transpose_on_lhs = True
            else:
                pre_idx = int(pre1_idx)
                pre_op = pre1_op
                transposed_input_name = in1_name
                plain_input_name = in0_name
                transpose_on_lhs = False

            if pre_op is None:
                continue
            if len(pre_op.inputs) < 2 or len(pre_op.outputs) != 1:
                continue

            transposed_users = set(consumers.get(transposed_input_name, []))
            if transposed_users != {int(mid_idx)}:
                continue

            out_users = set(consumers.get(out_name, []))
            if len(out_users) != 1:
                continue
            post_idx = int(list(out_users)[0])
            if post_idx == mid_idx:
                continue
            post_op = model_ir.operators[post_idx]
            if str(post_op.op_type) != "TRANSPOSE" or len(post_op.inputs) < 2 or len(post_op.outputs) != 1:
                continue
            if post_op.inputs[0] != out_name:
                continue

            perm_pre = _read_transpose_perm(model_ir, pre_op)
            perm_post = _read_transpose_perm(model_ir, post_op)
            if perm_pre is None or perm_post is None:
                continue
            if not _is_inverse_perm(perm_pre, perm_post):
                continue

            # Keep user-visible output names stable.
            if transposed_input_name in model_ir.outputs or out_name in model_ir.outputs:
                continue

            raw_from_pre_name = pre_op.inputs[0]
            final_out_name = post_op.outputs[0]

            transposed_tensor = model_ir.tensors.get(transposed_input_name, None)
            plain_tensor = model_ir.tensors.get(plain_input_name, None)
            raw_tensor = model_ir.tensors.get(raw_from_pre_name, None)
            out_tensor = model_ir.tensors.get(out_name, None)
            final_out_tensor = model_ir.tensors.get(final_out_name, None)

            if not _all_per_tensor_quantized(
                [transposed_tensor, plain_tensor, raw_tensor, out_tensor, final_out_tensor]
            ):
                continue

            transposed_input_shape = (
                list(transposed_tensor.shape) if transposed_tensor is not None else None
            )
            plain_input_shape = list(plain_tensor.shape) if plain_tensor is not None else None
            plain_input_signature = (
                list(plain_tensor.shape_signature)
                if plain_tensor is not None and plain_tensor.shape_signature is not None
                else plain_input_shape
            )
            raw_pre_shape = list(raw_tensor.shape) if raw_tensor is not None else None
            final_out_shape = list(final_out_tensor.shape) if final_out_tensor is not None else None

            # Require no broadcasting ambiguity.
            if not _shapes_equal(transposed_input_shape, plain_input_shape):
                continue
            expected_raw_shape = _permute_shape(plain_input_shape, perm_post)
            expected_raw_signature = _permute_shape(plain_input_signature, perm_post)
            if expected_raw_shape is None:
                continue
            if expected_raw_signature is None:
                expected_raw_signature = list(expected_raw_shape)
            if raw_pre_shape is not None and not _shapes_equal(expected_raw_shape, raw_pre_shape):
                continue
            if final_out_shape is not None and not _shapes_equal(expected_raw_shape, final_out_shape):
                continue

            # Reuse pre-transpose node by moving it to the other input with inverse perm.
            pre_op.inputs = [plain_input_name, post_op.inputs[1]]
            pre_op.outputs = [transposed_input_name]

            if transpose_on_lhs:
                # op(T(A), B) then T(invP) -> op(A, T(invP)(B))
                mid_op.inputs = [raw_from_pre_name, transposed_input_name]
            else:
                # op(A, T(B)) then T(invP) -> op(T(invP)(A), B)
                mid_op.inputs = [transposed_input_name, raw_from_pre_name]
            mid_op.outputs = [final_out_name]

            if transposed_tensor is not None:
                transposed_tensor.shape = list(expected_raw_shape)
                transposed_tensor.shape_signature = list(expected_raw_signature)
                if plain_tensor is not None:
                    transposed_tensor.dtype = str(plain_tensor.dtype)
                    if transposed_tensor.quantization is None and plain_tensor.quantization is not None:
                        transposed_tensor.quantization = _clone_quantization(plain_tensor.quantization)

            if final_out_tensor is not None and out_tensor is not None:
                final_out_tensor.dtype = str(out_tensor.dtype)
                if final_out_tensor.quantization is None and out_tensor.quantization is not None:
                    final_out_tensor.quantization = _clone_quantization(out_tensor.quantization)

            del model_ir.operators[post_idx]
            removed_binary_asymmetric_bridges += 1
            changed = True
            break

        if changed:
            continue

        # Pattern C:
        # Fanout-aware binary bridge optimization.
        # Allows output consumers to include:
        # - inverse TRANSPOSE nodes (removed)
        # - downstream binary ops (kept, enabling chained rewrites)
        if not enable_fanout_pattern_c:
            break
        binary_chain_cache: Dict[int, bool] = {}

        def _is_rewritable_binary_chain(
            start_idx: int,
            base_perm_local: List[int],
            visiting: Optional[set[int]] = None,
        ) -> bool:
            if start_idx in binary_chain_cache:
                return bool(binary_chain_cache[start_idx])
            if visiting is None:
                visiting = set()
            if int(start_idx) in visiting:
                binary_chain_cache[int(start_idx)] = False
                return False
            if start_idx < 0 or start_idx >= len(model_ir.operators):
                binary_chain_cache[int(start_idx)] = False
                return False
            op_local = model_ir.operators[int(start_idx)]
            if str(op_local.op_type) not in binary_ops or len(op_local.outputs) != 1:
                binary_chain_cache[int(start_idx)] = False
                return False
            out_local = str(op_local.outputs[0])
            if out_local in model_ir.outputs:
                binary_chain_cache[int(start_idx)] = False
                return False

            downstream = [int(v) for v in consumers.get(out_local, []) if int(v) != int(start_idx)]
            if len(downstream) == 0:
                binary_chain_cache[int(start_idx)] = False
                return False

            visiting.add(int(start_idx))
            for down_idx in downstream:
                down_op = model_ir.operators[int(down_idx)]
                if (
                    str(down_op.op_type) == "TRANSPOSE"
                    and len(down_op.inputs) >= 2
                    and len(down_op.outputs) == 1
                    and down_op.inputs[0] == out_local
                ):
                    down_perm = _read_transpose_perm(model_ir, down_op)
                    if down_perm is None or not _is_inverse_perm(base_perm_local, down_perm):
                        visiting.remove(int(start_idx))
                        binary_chain_cache[int(start_idx)] = False
                        return False
                    if str(down_op.outputs[0]) in model_ir.outputs:
                        visiting.remove(int(start_idx))
                        binary_chain_cache[int(start_idx)] = False
                        return False
                    continue
                if str(down_op.op_type) in binary_ops and out_local in set(down_op.inputs):
                    if not _is_rewritable_binary_chain(
                        int(down_idx),
                        base_perm_local,
                        visiting,
                    ):
                        visiting.remove(int(start_idx))
                        binary_chain_cache[int(start_idx)] = False
                        return False
                    continue
                visiting.remove(int(start_idx))
                binary_chain_cache[int(start_idx)] = False
                return False

            visiting.remove(int(start_idx))
            binary_chain_cache[int(start_idx)] = True
            return True

        for mid_idx, mid_op in enumerate(model_ir.operators):
            if str(mid_op.op_type) not in binary_ops:
                continue
            if len(mid_op.inputs) != 2 or len(mid_op.outputs) != 1:
                continue

            in0_name = mid_op.inputs[0]
            in1_name = mid_op.inputs[1]
            out_name = mid_op.outputs[0]

            if out_name in model_ir.outputs:
                continue

            def _input_info(input_name: str) -> Tuple[str, Optional[int], Optional[List[int]]]:
                pre_idx_local = producers.get(input_name, None)
                if pre_idx_local is None:
                    return input_name, None, None
                pre_op_local = model_ir.operators[int(pre_idx_local)]
                if (
                    str(pre_op_local.op_type) != "TRANSPOSE"
                    or len(pre_op_local.inputs) < 2
                    or len(pre_op_local.outputs) != 1
                    or pre_op_local.outputs[0] != input_name
                ):
                    return input_name, None, None
                if set(consumers.get(input_name, [])) != {int(mid_idx)}:
                    return input_name, None, None
                perm_local = _read_transpose_perm(model_ir, pre_op_local)
                if perm_local is None:
                    return input_name, None, None
                return pre_op_local.inputs[0], int(pre_idx_local), perm_local

            raw_in0_name, pre0_idx, perm0 = _input_info(in0_name)
            raw_in1_name, pre1_idx, perm1 = _input_info(in1_name)

            if perm0 is None and perm1 is None:
                continue
            if in0_name in model_ir.outputs or in1_name in model_ir.outputs:
                continue

            perm_candidates = []
            if perm0 is not None:
                perm_candidates.append(list(perm0))
            if perm1 is not None:
                perm_candidates.append(list(perm1))
            if len(perm_candidates) == 0:
                continue
            base_perm = perm_candidates[0]
            if any(perm != base_perm for perm in perm_candidates[1:]):
                continue

            raw_in0_tensor = model_ir.tensors.get(raw_in0_name, None)
            raw_in1_tensor = model_ir.tensors.get(raw_in1_name, None)
            if raw_in0_tensor is None or raw_in1_tensor is None:
                continue
            raw_shape0 = list(raw_in0_tensor.shape)
            raw_shape1 = list(raw_in1_tensor.shape)
            if not _shapes_equal(raw_shape0, raw_shape1):
                continue
            raw_signature0 = (
                list(raw_in0_tensor.shape_signature)
                if raw_in0_tensor.shape_signature is not None
                else list(raw_shape0)
            )

            out_users = [int(v) for v in consumers.get(out_name, []) if int(v) != int(mid_idx)]
            if len(out_users) == 0:
                continue

            post_indices: List[int] = []
            post_output_names: List[str] = []
            users_valid = True

            for user_idx in out_users:
                user_op = model_ir.operators[int(user_idx)]
                if (
                    str(user_op.op_type) == "TRANSPOSE"
                    and len(user_op.inputs) >= 2
                    and len(user_op.outputs) == 1
                    and user_op.inputs[0] == out_name
                ):
                    perm_post = _read_transpose_perm(model_ir, user_op)
                    if perm_post is None or not _is_inverse_perm(base_perm, perm_post):
                        users_valid = False
                        break
                    post_output_name = str(user_op.outputs[0])
                    if post_output_name in model_ir.outputs:
                        users_valid = False
                        break
                    post_indices.append(int(user_idx))
                    post_output_names.append(post_output_name)
                    continue

                if str(user_op.op_type) in binary_ops and out_name in set(user_op.inputs):
                    if len(user_op.inputs) != 2:
                        users_valid = False
                        break
                    if user_op.inputs[0] == out_name:
                        other_input_name = user_op.inputs[1]
                    elif user_op.inputs[1] == out_name:
                        other_input_name = user_op.inputs[0]
                    else:
                        users_valid = False
                        break

                    other_ok = False
                    other_prod_idx = producers.get(other_input_name, None)
                    if other_prod_idx is not None:
                        other_prod_op = model_ir.operators[int(other_prod_idx)]
                        if (
                            str(other_prod_op.op_type) == "TRANSPOSE"
                            and len(other_prod_op.inputs) >= 2
                            and len(other_prod_op.outputs) == 1
                            and other_prod_op.outputs[0] == other_input_name
                            and set(consumers.get(other_input_name, [])) == {int(user_idx)}
                        ):
                            other_perm = _read_transpose_perm(model_ir, other_prod_op)
                            if other_perm is not None and other_perm == base_perm:
                                other_ok = True
                    if not other_ok:
                        other_tensor = model_ir.tensors.get(other_input_name, None)
                        if other_tensor is not None and _shapes_equal(list(other_tensor.shape), raw_shape0):
                            other_ok = True
                    if not other_ok:
                        users_valid = False
                        break

                    if not _is_rewritable_binary_chain(int(user_idx), base_perm):
                        users_valid = False
                        break

                    continue

                users_valid = False
                break

            if not users_valid:
                continue

            removable_pre_indices = [idx for idx in [pre0_idx, pre1_idx] if idx is not None]
            if len(removable_pre_indices) == 0 and len(post_indices) == 0:
                continue

            # Rewire binary inputs to raw layout.
            mid_op.inputs = [raw_in0_name, raw_in1_name]

            out_tensor = model_ir.tensors.get(out_name, None)
            if out_tensor is not None:
                out_tensor.shape = list(raw_shape0)
                out_tensor.shape_signature = list(raw_signature0)

            for post_output_name in post_output_names:
                _replace_tensor_inputs(model_ir, post_output_name, out_name)

            remove_indices = sorted(list(set(removable_pre_indices + post_indices)), reverse=True)
            for remove_idx in remove_indices:
                del model_ir.operators[remove_idx]

            removed_binary_bridges += int(max(1, len(post_indices) + len(removable_pre_indices)))
            changed = True
            break

        if not changed:
            break

    _prune_unused_tensors(model_ir)
    return {
        "removed_transpose_binary_bridges": int(removed_binary_bridges),
        "removed_transpose_binary_asymmetric_bridges": int(removed_binary_asymmetric_bridges),
    }


def _optimize_transpose_binary_asymmetric_fanout_bridges(model_ir: ModelIR) -> Dict[str, int]:
    """
    Fold asymmetric transpose wrappers around binary ops when output has fanout.

    Target pattern:
      T(P)(X) + Y -> Z
      T(inv(P))(Z) -> Z_nhwc
      (Z may have extra non-transpose consumers)

    Rewritten:
      X + T(inv(P))(Y) -> Z_nhwc
      (optional adapter) T(P)(Z_nhwc) -> Z

    This reduces one transpose in fanout cases where the simple asymmetric bridge
    optimization cannot apply.
    """
    rewritten = 0
    binary_ops = {"ADD", "SUB", "MUL", "DIV"}

    while True:
        changed = False
        consumers = _build_tensor_consumer_map(model_ir)
        producers = _build_tensor_producer_map(model_ir)

        for mid_idx, mid_op in enumerate(model_ir.operators):
            if str(mid_op.op_type) not in binary_ops:
                continue
            if len(mid_op.inputs) != 2 or len(mid_op.outputs) != 1:
                continue

            in0_name = str(mid_op.inputs[0])
            in1_name = str(mid_op.inputs[1])
            out_name = str(mid_op.outputs[0])

            pre0_idx = producers.get(in0_name, None)
            pre1_idx = producers.get(in1_name, None)
            pre0_is_transpose = False
            pre1_is_transpose = False
            pre0_op = None
            pre1_op = None

            if pre0_idx is not None:
                pre0_op = model_ir.operators[int(pre0_idx)]
                pre0_is_transpose = (
                    str(pre0_op.op_type) == "TRANSPOSE"
                    and len(pre0_op.inputs) >= 2
                    and len(pre0_op.outputs) == 1
                    and str(pre0_op.outputs[0]) == in0_name
                )
            if pre1_idx is not None:
                pre1_op = model_ir.operators[int(pre1_idx)]
                pre1_is_transpose = (
                    str(pre1_op.op_type) == "TRANSPOSE"
                    and len(pre1_op.inputs) >= 2
                    and len(pre1_op.outputs) == 1
                    and str(pre1_op.outputs[0]) == in1_name
                )

            # Exactly one pre-transposed input.
            if pre0_is_transpose == pre1_is_transpose:
                continue

            if pre0_is_transpose:
                pre_idx = int(pre0_idx)
                pre_op = pre0_op
                transposed_input_name = in0_name
                plain_input_name = in1_name
                transpose_on_lhs = True
            else:
                pre_idx = int(pre1_idx)
                pre_op = pre1_op
                transposed_input_name = in1_name
                plain_input_name = in0_name
                transpose_on_lhs = False

            if pre_op is None:
                continue

            transposed_users = [int(v) for v in consumers.get(transposed_input_name, [])]
            if transposed_users != [int(mid_idx)] and set(transposed_users) != {int(mid_idx)}:
                continue

            out_users = [int(v) for v in consumers.get(out_name, []) if int(v) != int(mid_idx)]
            if len(out_users) == 0:
                continue

            perm_pre = _read_transpose_perm(model_ir, pre_op)
            if perm_pre is None:
                continue
            inv_perm = _invert_perm(perm_pre)
            if inv_perm is None:
                continue

            post_idx = None
            post_op = None
            for user_idx in out_users:
                user_op = model_ir.operators[int(user_idx)]
                if (
                    str(user_op.op_type) == "TRANSPOSE"
                    and len(user_op.inputs) >= 2
                    and len(user_op.outputs) == 1
                    and str(user_op.inputs[0]) == out_name
                ):
                    perm_post = _read_transpose_perm(model_ir, user_op)
                    if perm_post is not None and perm_post == inv_perm:
                        post_idx = int(user_idx)
                        post_op = user_op
                        break
            if post_idx is None or post_op is None:
                continue

            post_output_name = str(post_op.outputs[0])
            if post_output_name in model_ir.outputs and out_name in model_ir.outputs:
                # Avoid ambiguous observable output rewrites.
                continue

            # Find an existing transpose(plain_input, inv_perm) to reuse.
            plain_users = [int(v) for v in consumers.get(plain_input_name, [])]
            plain_transpose_idx = None
            plain_transpose_out = None
            for plain_user_idx in plain_users:
                plain_user_op = model_ir.operators[int(plain_user_idx)]
                if (
                    str(plain_user_op.op_type) == "TRANSPOSE"
                    and len(plain_user_op.inputs) >= 2
                    and len(plain_user_op.outputs) == 1
                    and str(plain_user_op.inputs[0]) == plain_input_name
                ):
                    perm_plain = _read_transpose_perm(model_ir, plain_user_op)
                    if perm_plain is not None and perm_plain == inv_perm:
                        plain_transpose_idx = int(plain_user_idx)
                        plain_transpose_out = str(plain_user_op.outputs[0])
                        break
            if plain_transpose_idx is None or plain_transpose_out is None:
                continue

            raw_from_pre_name = str(pre_op.inputs[0])
            if transpose_on_lhs:
                mid_op.inputs = [raw_from_pre_name, plain_transpose_out]
            else:
                mid_op.inputs = [plain_transpose_out, raw_from_pre_name]
            mid_op.outputs = [post_output_name]

            out_tensor = model_ir.tensors.get(out_name, None)
            post_out_tensor = model_ir.tensors.get(post_output_name, None)
            if out_tensor is not None and post_out_tensor is not None:
                post_out_tensor.dtype = str(out_tensor.dtype)
                if post_out_tensor.quantization is None and out_tensor.quantization is not None:
                    post_out_tensor.quantization = _clone_quantization(out_tensor.quantization)

            extra_users = [int(v) for v in out_users if int(v) != int(post_idx)]
            adapter_needed = len(extra_users) > 0 or out_name in model_ir.outputs
            if adapter_needed:
                adapter = OperatorIR(
                    op_type="TRANSPOSE",
                    inputs=[post_output_name, str(pre_op.inputs[1])],
                    outputs=[out_name],
                )

            remove_indices = sorted(list({int(pre_idx), int(post_idx)}), reverse=True)
            shift = sum(1 for ridx in remove_indices if int(ridx) < int(mid_idx))
            for ridx in remove_indices:
                del model_ir.operators[int(ridx)]
            mid_new_idx = int(mid_idx - shift)
            if adapter_needed:
                model_ir.operators.insert(int(mid_new_idx) + 1, adapter)

            rewritten += 1
            changed = True
            break

        if not changed:
            break

    _prune_unused_tensors(model_ir)
    return {
        "rewritten_transpose_binary_asymmetric_fanout_bridges": int(rewritten),
    }


def _optimize_transpose_binary_full_post_fanout_bridges(model_ir: ModelIR) -> Dict[str, int]:
    """
    Fold a strict transpose-binary-transpose fanout pattern.

    Target:
      T(P)(A) + T(P)(B) -> C
      C -> T(inv(P)) -> C0
      C -> T(inv(P)) -> C1
      ...

    Rewrite:
      A + B -> C0
      (all uses of C1... are rewired to C0)

    Safety:
    - Binary op has exactly 2 inputs / 1 output (ADD|SUB|MUL|DIV)
    - Both inputs are produced by TRANSPOSE with identical permutation
    - Both pre-transpose outputs are consumed only by the binary op
    - All consumers of binary output are inverse TRANSPOSE ops
    - No broadcasting ambiguity: all participating tensor shapes match exactly
    - Intermediate tensors are not graph outputs
    """
    rewritten = 0
    binary_ops = {"ADD", "SUB", "MUL", "DIV"}

    while True:
        changed = False
        consumers = _build_tensor_consumer_map(model_ir)
        producers = _build_tensor_producer_map(model_ir)

        for mid_idx, mid_op in enumerate(model_ir.operators):
            if str(mid_op.op_type) not in binary_ops:
                continue
            if len(mid_op.inputs) != 2 or len(mid_op.outputs) != 1:
                continue

            in0_name = str(mid_op.inputs[0])
            in1_name = str(mid_op.inputs[1])
            out_name = str(mid_op.outputs[0])

            pre0_idx = producers.get(in0_name, None)
            pre1_idx = producers.get(in1_name, None)
            if pre0_idx is None or pre1_idx is None:
                continue
            pre0_op = model_ir.operators[int(pre0_idx)]
            pre1_op = model_ir.operators[int(pre1_idx)]
            if str(pre0_op.op_type) != "TRANSPOSE" or str(pre1_op.op_type) != "TRANSPOSE":
                continue
            if len(pre0_op.inputs) < 2 or len(pre1_op.inputs) < 2:
                continue
            if len(pre0_op.outputs) != 1 or len(pre1_op.outputs) != 1:
                continue
            if str(pre0_op.outputs[0]) != in0_name or str(pre1_op.outputs[0]) != in1_name:
                continue

            # Pre-transposed tensors must be exclusive to the binary op.
            if set(consumers.get(in0_name, [])) != {int(mid_idx)}:
                continue
            if set(consumers.get(in1_name, [])) != {int(mid_idx)}:
                continue

            perm_pre0 = _read_transpose_perm(model_ir, pre0_op)
            perm_pre1 = _read_transpose_perm(model_ir, pre1_op)
            if perm_pre0 is None or perm_pre1 is None:
                continue
            if perm_pre0 != perm_pre1:
                continue

            out_users = [int(v) for v in consumers.get(out_name, []) if int(v) != int(mid_idx)]
            if len(out_users) < 2:
                continue

            post_indices: List[int] = []
            post_output_names: List[str] = []
            for user_idx in out_users:
                post_op = model_ir.operators[int(user_idx)]
                if (
                    str(post_op.op_type) != "TRANSPOSE"
                    or len(post_op.inputs) < 2
                    or len(post_op.outputs) != 1
                    or str(post_op.inputs[0]) != out_name
                ):
                    post_indices = []
                    break
                perm_post = _read_transpose_perm(model_ir, post_op)
                if perm_post is None or not _is_inverse_perm(perm_pre0, perm_post):
                    post_indices = []
                    break
                post_indices.append(int(user_idx))
                post_output_names.append(str(post_op.outputs[0]))
            if len(post_indices) != len(out_users):
                continue

            if (
                in0_name in model_ir.outputs
                or in1_name in model_ir.outputs
                or out_name in model_ir.outputs
                or any(name in model_ir.outputs for name in post_output_names)
            ):
                continue

            raw_in0_name = str(pre0_op.inputs[0])
            raw_in1_name = str(pre1_op.inputs[0])
            raw_in0_tensor = model_ir.tensors.get(raw_in0_name, None)
            raw_in1_tensor = model_ir.tensors.get(raw_in1_name, None)
            in0_tensor = model_ir.tensors.get(in0_name, None)
            in1_tensor = model_ir.tensors.get(in1_name, None)
            out_tensor = model_ir.tensors.get(out_name, None)
            if not _all_per_tensor_quantized(
                [raw_in0_tensor, raw_in1_tensor, in0_tensor, in1_tensor, out_tensor]
            ):
                continue

            # Conservative shape guards (no broadcasting / no ambiguous layout rewrite).
            raw_shape0 = list(raw_in0_tensor.shape) if raw_in0_tensor is not None else None
            raw_shape1 = list(raw_in1_tensor.shape) if raw_in1_tensor is not None else None
            if not _shapes_match_if_known(raw_shape0, raw_shape1):
                continue
            in_shape0 = list(in0_tensor.shape) if in0_tensor is not None else None
            in_shape1 = list(in1_tensor.shape) if in1_tensor is not None else None
            if not _shapes_match_if_known(in_shape0, in_shape1):
                continue

            valid_post_shapes = True
            for post_output_name in post_output_names:
                post_tensor = model_ir.tensors.get(post_output_name, None)
                post_shape = list(post_tensor.shape) if post_tensor is not None else None
                if not _shapes_match_if_known(post_shape, raw_shape0):
                    valid_post_shapes = False
                    break
            if not valid_post_shapes:
                continue

            canonical_post_output = str(post_output_names[0])
            canonical_tensor = model_ir.tensors.get(canonical_post_output, None)

            mid_op.inputs = [raw_in0_name, raw_in1_name]
            mid_op.outputs = [canonical_post_output]

            if canonical_tensor is not None:
                if not _is_unknown_shape(raw_shape0):
                    canonical_tensor.shape = [int(v) for v in raw_shape0]
                    raw_sig0 = (
                        list(raw_in0_tensor.shape_signature)
                        if raw_in0_tensor is not None and raw_in0_tensor.shape_signature is not None
                        else raw_shape0
                    )
                    canonical_tensor.shape_signature = [int(v) for v in raw_sig0]
                if out_tensor is not None:
                    canonical_tensor.dtype = str(out_tensor.dtype)
                    canonical_tensor.quantization = _clone_quantization(out_tensor.quantization)

            for post_output_name in post_output_names[1:]:
                _replace_tensor_inputs(model_ir, post_output_name, canonical_post_output)

            remove_indices = sorted(
                list(set([int(pre0_idx), int(pre1_idx)] + [int(idx) for idx in post_indices])),
                reverse=True,
            )
            for remove_idx in remove_indices:
                del model_ir.operators[remove_idx]

            rewritten += 1
            changed = True
            break

        if not changed:
            break

    _prune_unused_tensors(model_ir)
    return {"rewritten_transpose_binary_full_post_fanout_bridges": int(rewritten)}


def _optimize_transpose_binary_single_post_bridges_safe(model_ir: ModelIR) -> Dict[str, int]:
    """
    Conservative transpose-binary bridge folding with exactly one inverse post-transpose.

    Target:
      T(P)(A), T(P)(B) -> C
      C -> T(inv(P)) -> C_raw
      (C may also have extra non-transpose consumers)

    Rewrite:
      A, B -> C_raw
      If C has extra consumers:
        keep adapter T(P): C_raw -> C
      else:
        remove post-transpose.

    This pass intentionally uses strict shape guards to avoid layout corruption.
    """
    rewritten = 0
    binary_ops = {"ADD", "SUB", "MUL", "DIV"}

    while True:
        changed = False
        consumers = _build_tensor_consumer_map(model_ir)
        producers = _build_tensor_producer_map(model_ir)

        for mid_idx, mid_op in enumerate(model_ir.operators):
            if str(mid_op.op_type) not in binary_ops:
                continue
            if len(mid_op.inputs) != 2 or len(mid_op.outputs) != 1:
                continue

            in0_name = str(mid_op.inputs[0])
            in1_name = str(mid_op.inputs[1])
            out_name = str(mid_op.outputs[0])

            pre0_idx = producers.get(in0_name, None)
            pre1_idx = producers.get(in1_name, None)
            if pre0_idx is None or pre1_idx is None:
                continue
            pre0_op = model_ir.operators[int(pre0_idx)]
            pre1_op = model_ir.operators[int(pre1_idx)]
            if str(pre0_op.op_type) != "TRANSPOSE" or str(pre1_op.op_type) != "TRANSPOSE":
                continue
            if len(pre0_op.inputs) < 2 or len(pre1_op.inputs) < 2:
                continue
            if len(pre0_op.outputs) != 1 or len(pre1_op.outputs) != 1:
                continue
            if str(pre0_op.outputs[0]) != in0_name or str(pre1_op.outputs[0]) != in1_name:
                continue

            if set(consumers.get(in0_name, [])) != {int(mid_idx)}:
                continue
            if set(consumers.get(in1_name, [])) != {int(mid_idx)}:
                continue

            perm_pre0 = _read_transpose_perm(model_ir, pre0_op)
            perm_pre1 = _read_transpose_perm(model_ir, pre1_op)
            if perm_pre0 is None or perm_pre1 is None:
                continue
            if perm_pre0 != perm_pre1:
                continue

            out_users = {int(v) for v in consumers.get(out_name, []) if int(v) != int(mid_idx)}
            if len(out_users) == 0:
                continue

            post_candidates: List[int] = []
            for user_idx in sorted(list(out_users)):
                post_op = model_ir.operators[int(user_idx)]
                if (
                    str(post_op.op_type) != "TRANSPOSE"
                    or len(post_op.inputs) < 2
                    or len(post_op.outputs) != 1
                    or str(post_op.inputs[0]) != out_name
                ):
                    continue
                perm_post = _read_transpose_perm(model_ir, post_op)
                if perm_post is None or not _is_inverse_perm(perm_pre0, perm_post):
                    continue
                post_candidates.append(int(user_idx))

            if len(post_candidates) != 1:
                continue

            post_idx = int(post_candidates[0])
            post_op = model_ir.operators[post_idx]
            post_out_name = str(post_op.outputs[0])
            post_perm_name = str(post_op.inputs[1])
            extra_users = set(out_users) - {int(post_idx)}

            # Avoid ambiguous fanout rewrites if any extra fanout is another transpose.
            if any(str(model_ir.operators[int(v)].op_type) == "TRANSPOSE" for v in extra_users):
                continue

            if in0_name in model_ir.outputs or in1_name in model_ir.outputs:
                continue

            raw_in0_name = str(pre0_op.inputs[0])
            raw_in1_name = str(pre1_op.inputs[0])

            raw_in0_tensor = model_ir.tensors.get(raw_in0_name, None)
            raw_in1_tensor = model_ir.tensors.get(raw_in1_name, None)
            in0_tensor = model_ir.tensors.get(in0_name, None)
            in1_tensor = model_ir.tensors.get(in1_name, None)
            out_tensor = model_ir.tensors.get(out_name, None)
            post_out_tensor = model_ir.tensors.get(post_out_name, None)

            if not _all_per_tensor_quantized(
                [raw_in0_tensor, raw_in1_tensor, in0_tensor, in1_tensor, out_tensor, post_out_tensor]
            ):
                continue

            raw_shape0 = list(raw_in0_tensor.shape) if raw_in0_tensor is not None else None
            raw_shape1 = list(raw_in1_tensor.shape) if raw_in1_tensor is not None else None
            # NOTE:
            # Some optimized branches can retain stale static shapes in IR while runtime
            # shapes are valid. For strict single-post bridge chains, rely on topology
            # constraints and quantization compatibility instead of hard shape equality.

            mid_op.inputs = [raw_in0_name, raw_in1_name]
            mid_op.outputs = [post_out_name]

            if post_out_tensor is not None and out_tensor is not None:
                post_out_tensor.dtype = str(out_tensor.dtype)
                post_out_tensor.quantization = _clone_quantization(out_tensor.quantization)
                if not _is_unknown_shape(raw_shape0):
                    post_out_tensor.shape = [int(v) for v in list(raw_shape0)]
                    raw_sig0 = (
                        list(raw_in0_tensor.shape_signature)
                        if raw_in0_tensor is not None and raw_in0_tensor.shape_signature is not None
                        else list(raw_shape0)
                    )
                    post_out_tensor.shape_signature = [int(v) for v in list(raw_sig0)]

            if len(extra_users) > 0:
                perm_tensor = model_ir.tensors.get(post_perm_name, None)
                if perm_tensor is not None:
                    perm_tensor.data = np.asarray(perm_pre0, dtype=np.int32)
                post_op.inputs = [post_out_name, post_perm_name]
                post_op.outputs = [out_name]
                remove_indices = sorted(list({int(pre0_idx), int(pre1_idx)}), reverse=True)
            else:
                remove_indices = sorted(list({int(pre0_idx), int(pre1_idx), int(post_idx)}), reverse=True)

            for remove_idx in remove_indices:
                del model_ir.operators[remove_idx]

            rewritten += 1
            changed = True
            break

        if not changed:
            break

    _prune_unused_tensors(model_ir)
    return {"rewritten_transpose_binary_single_post_bridges_safe": int(rewritten)}


def _optimize_transpose_binary_mixed_fanout_bridges_safe(model_ir: ModelIR) -> Dict[str, int]:
    """
    Fold transpose-binary pattern when output fanout mixes:
    - inverse TRANSPOSE consumers (layout adapters)
    - non-TRANSPOSE consumers (legacy layout path)

    Rewrite keeps one transpose adapter for legacy consumers and removes the
    inverse transpose fanout consumers.
    """
    rewritten = 0
    binary_ops = {"ADD", "SUB", "MUL", "DIV"}

    while True:
        changed = False
        consumers = _build_tensor_consumer_map(model_ir)
        producers = _build_tensor_producer_map(model_ir)

        for mid_idx, mid_op in enumerate(model_ir.operators):
            if str(mid_op.op_type) not in binary_ops:
                continue
            if len(mid_op.inputs) != 2 or len(mid_op.outputs) != 1:
                continue

            in0_name = str(mid_op.inputs[0])
            in1_name = str(mid_op.inputs[1])
            out_name = str(mid_op.outputs[0])

            pre0_idx = producers.get(in0_name, None)
            pre1_idx = producers.get(in1_name, None)
            if pre0_idx is None or pre1_idx is None:
                continue
            pre0_op = model_ir.operators[int(pre0_idx)]
            pre1_op = model_ir.operators[int(pre1_idx)]
            if str(pre0_op.op_type) != "TRANSPOSE" or str(pre1_op.op_type) != "TRANSPOSE":
                continue
            if len(pre0_op.inputs) < 2 or len(pre1_op.inputs) < 2:
                continue
            if len(pre0_op.outputs) != 1 or len(pre1_op.outputs) != 1:
                continue
            if str(pre0_op.outputs[0]) != in0_name or str(pre1_op.outputs[0]) != in1_name:
                continue

            if set(consumers.get(in0_name, [])) != {int(mid_idx)}:
                continue
            if set(consumers.get(in1_name, [])) != {int(mid_idx)}:
                continue

            perm_pre0 = _read_transpose_perm(model_ir, pre0_op)
            perm_pre1 = _read_transpose_perm(model_ir, pre1_op)
            if perm_pre0 is None or perm_pre1 is None or perm_pre0 != perm_pre1:
                continue

            out_users = [int(v) for v in consumers.get(out_name, []) if int(v) != int(mid_idx)]
            if len(out_users) < 2:
                continue

            post_indices: List[int] = []
            post_output_names: List[str] = []
            legacy_users: List[int] = []
            valid = True
            for user_idx in out_users:
                user_op = model_ir.operators[int(user_idx)]
                if (
                    str(user_op.op_type) == "TRANSPOSE"
                    and len(user_op.inputs) >= 2
                    and len(user_op.outputs) == 1
                    and str(user_op.inputs[0]) == out_name
                ):
                    perm_post = _read_transpose_perm(model_ir, user_op)
                    if perm_post is None or not _is_inverse_perm(perm_pre0, perm_post):
                        valid = False
                        break
                    post_indices.append(int(user_idx))
                    post_output_names.append(str(user_op.outputs[0]))
                else:
                    legacy_users.append(int(user_idx))

            if not valid:
                continue
            if len(post_indices) == 0 or len(legacy_users) == 0:
                continue

            if (
                in0_name in model_ir.outputs
                or in1_name in model_ir.outputs
                or out_name in model_ir.outputs
                or any(name in model_ir.outputs for name in post_output_names)
            ):
                continue

            raw_in0_name = str(pre0_op.inputs[0])
            raw_in1_name = str(pre1_op.inputs[0])
            raw_in0_tensor = model_ir.tensors.get(raw_in0_name, None)
            raw_in1_tensor = model_ir.tensors.get(raw_in1_name, None)
            in0_tensor = model_ir.tensors.get(in0_name, None)
            in1_tensor = model_ir.tensors.get(in1_name, None)
            out_tensor = model_ir.tensors.get(out_name, None)
            if not _all_per_tensor_quantized(
                [raw_in0_tensor, raw_in1_tensor, in0_tensor, in1_tensor, out_tensor]
            ):
                continue

            raw_shape0 = list(raw_in0_tensor.shape) if raw_in0_tensor is not None else None
            raw_shape1 = list(raw_in1_tensor.shape) if raw_in1_tensor is not None else None
            if not _shapes_match_if_known(raw_shape0, raw_shape1):
                continue
            in_shape0 = list(in0_tensor.shape) if in0_tensor is not None else None
            in_shape1 = list(in1_tensor.shape) if in1_tensor is not None else None
            if not _shapes_match_if_known(in_shape0, in_shape1):
                continue
            if not _is_unknown_shape(raw_shape0) and not _is_unknown_shape(in_shape0):
                expected_trans_shape = _permute_shape(raw_shape0, perm_pre0)
                if expected_trans_shape is None or not _shapes_equal(expected_trans_shape, in_shape0):
                    continue

            valid_post_shapes = True
            for post_output_name in post_output_names:
                post_tensor = model_ir.tensors.get(post_output_name, None)
                post_shape = list(post_tensor.shape) if post_tensor is not None else None
                if not _shapes_match_if_known(post_shape, raw_shape0):
                    valid_post_shapes = False
                    break
            if not valid_post_shapes:
                continue

            keep_post_idx = int(post_indices[0])
            keep_post_op = model_ir.operators[int(keep_post_idx)]
            canonical_raw_name = str(keep_post_op.outputs[0])
            keep_perm_name = str(keep_post_op.inputs[1])
            canonical_raw_tensor = model_ir.tensors.get(canonical_raw_name, None)

            mid_op.inputs = [raw_in0_name, raw_in1_name]
            mid_op.outputs = [canonical_raw_name]

            if canonical_raw_tensor is not None and out_tensor is not None:
                canonical_raw_tensor.dtype = str(out_tensor.dtype)
                canonical_raw_tensor.quantization = _clone_quantization(out_tensor.quantization)
                if not _is_unknown_shape(raw_shape0):
                    canonical_raw_tensor.shape = [int(v) for v in list(raw_shape0)]
                    raw_sig0 = (
                        list(raw_in0_tensor.shape_signature)
                        if raw_in0_tensor is not None and raw_in0_tensor.shape_signature is not None
                        else list(raw_shape0)
                    )
                    canonical_raw_tensor.shape_signature = [int(v) for v in list(raw_sig0)]

            for post_output_name in post_output_names[1:]:
                _replace_tensor_inputs(model_ir, post_output_name, canonical_raw_name)

            # Keep one adapter transpose for legacy consumers: raw -> original out layout.
            keep_perm_tensor = model_ir.tensors.get(keep_perm_name, None)
            if keep_perm_tensor is not None:
                keep_perm_tensor.data = np.asarray(perm_pre0, dtype=np.int32)
            keep_post_op.inputs = [canonical_raw_name, keep_perm_name]
            keep_post_op.outputs = [out_name]

            remove_indices = sorted(
                list(set([int(pre0_idx), int(pre1_idx)] + [int(idx) for idx in post_indices[1:]])),
                reverse=True,
            )
            for remove_idx in remove_indices:
                del model_ir.operators[remove_idx]

            rewritten += 1
            changed = True
            break

        if not changed:
            break

    _prune_unused_tensors(model_ir)
    return {"rewritten_transpose_binary_mixed_fanout_bridges_safe": int(rewritten)}


def _optimize_transpose_binary_symmetric_legacy_only_bridges_safe(model_ir: ModelIR) -> Dict[str, int]:
    """
    Fold symmetric pre-transpose binary pattern when output has only legacy consumers.

    Pattern:
      T(P)(A), T(P)(B) -> C
      C -> legacy consumers only (no inverse-post transpose)

    Rewrite:
      A, B -> C_raw
      T(P)(C_raw) -> C

    Net effect: remove 2 pre-transposes and insert 1 adapter transpose.
    """
    rewritten = 0
    binary_ops = {"ADD", "SUB", "MUL", "DIV"}

    while True:
        changed = False
        consumers = _build_tensor_consumer_map(model_ir)
        producers = _build_tensor_producer_map(model_ir)

        for mid_idx, mid_op in enumerate(model_ir.operators):
            if str(mid_op.op_type) not in binary_ops:
                continue
            if len(mid_op.inputs) != 2 or len(mid_op.outputs) != 1:
                continue

            in0_name = str(mid_op.inputs[0])
            in1_name = str(mid_op.inputs[1])
            out_name = str(mid_op.outputs[0])

            pre0_idx = producers.get(in0_name, None)
            pre1_idx = producers.get(in1_name, None)
            if pre0_idx is None or pre1_idx is None:
                continue
            pre0_op = model_ir.operators[int(pre0_idx)]
            pre1_op = model_ir.operators[int(pre1_idx)]
            if str(pre0_op.op_type) != "TRANSPOSE" or str(pre1_op.op_type) != "TRANSPOSE":
                continue
            if len(pre0_op.inputs) < 2 or len(pre1_op.inputs) < 2:
                continue
            if len(pre0_op.outputs) != 1 or len(pre1_op.outputs) != 1:
                continue
            if str(pre0_op.outputs[0]) != in0_name or str(pre1_op.outputs[0]) != in1_name:
                continue

            if set(consumers.get(in0_name, [])) != {int(mid_idx)}:
                continue
            if set(consumers.get(in1_name, [])) != {int(mid_idx)}:
                continue

            perm_pre0 = _read_transpose_perm(model_ir, pre0_op)
            perm_pre1 = _read_transpose_perm(model_ir, pre1_op)
            if perm_pre0 is None or perm_pre1 is None or perm_pre0 != perm_pre1:
                continue

            out_users = [int(v) for v in consumers.get(out_name, []) if int(v) != int(mid_idx)]
            if len(out_users) == 0:
                continue
            # This pass targets legacy-only fanout; inverse post-transpose is handled elsewhere.
            has_inverse_post = False
            for user_idx in out_users:
                user_op = model_ir.operators[int(user_idx)]
                if (
                    str(user_op.op_type) == "TRANSPOSE"
                    and len(user_op.inputs) >= 2
                    and len(user_op.outputs) == 1
                    and str(user_op.inputs[0]) == out_name
                ):
                    perm_post = _read_transpose_perm(model_ir, user_op)
                    if perm_post is not None and _is_inverse_perm(perm_pre0, perm_post):
                        has_inverse_post = True
                        break
            if has_inverse_post:
                continue

            if in0_name in model_ir.outputs or in1_name in model_ir.outputs:
                continue

            raw_in0_name = str(pre0_op.inputs[0])
            raw_in1_name = str(pre1_op.inputs[0])

            raw_in0_tensor = model_ir.tensors.get(raw_in0_name, None)
            raw_in1_tensor = model_ir.tensors.get(raw_in1_name, None)
            in0_tensor = model_ir.tensors.get(in0_name, None)
            in1_tensor = model_ir.tensors.get(in1_name, None)
            out_tensor = model_ir.tensors.get(out_name, None)
            if not _all_per_tensor_quantized(
                [raw_in0_tensor, raw_in1_tensor, in0_tensor, in1_tensor, out_tensor]
            ):
                continue

            raw_shape0 = list(raw_in0_tensor.shape) if raw_in0_tensor is not None else None
            raw_shape1 = list(raw_in1_tensor.shape) if raw_in1_tensor is not None else None
            if not _shapes_match_if_known(raw_shape0, raw_shape1):
                continue

            # Prepare raw output tensor.
            raw_out_name = f"{out_name}__raw"
            suffix = 1
            while raw_out_name in model_ir.tensors:
                raw_out_name = f"{out_name}__raw_{suffix}"
                suffix += 1

            raw_sig0 = (
                list(raw_in0_tensor.shape_signature)
                if raw_in0_tensor is not None and raw_in0_tensor.shape_signature is not None
                else raw_shape0
            )
            model_ir.tensors[raw_out_name] = TensorIR(
                name=raw_out_name,
                dtype=str(out_tensor.dtype) if out_tensor is not None else "INT8",
                shape=[int(v) for v in list(raw_shape0)] if not _is_unknown_shape(raw_shape0) else [1],
                shape_signature=(
                    [int(v) for v in list(raw_sig0)]
                    if raw_sig0 is not None and not _is_unknown_shape(raw_sig0)
                    else [1]
                ),
                data=None,
                is_variable=False,
                quantization=_clone_quantization(out_tensor.quantization if out_tensor is not None else None),
            )

            mid_op.inputs = [raw_in0_name, raw_in1_name]
            mid_op.outputs = [raw_out_name]

            insert_at = min(out_users)
            remove_indices = sorted([int(pre0_idx), int(pre1_idx)], reverse=True)
            shift = sum(1 for idx in remove_indices if idx < insert_at)
            for idx in remove_indices:
                del model_ir.operators[idx]
            adjusted_insert_at = int(insert_at - shift)
            model_ir.operators.insert(
                adjusted_insert_at,
                OperatorIR(
                    op_type="TRANSPOSE",
                    inputs=[raw_out_name, str(pre0_op.inputs[1])],
                    outputs=[out_name],
                ),
            )

            rewritten += 1
            changed = True
            break

        if not changed:
            break

    _prune_unused_tensors(model_ir)
    return {"rewritten_transpose_binary_symmetric_legacy_only_bridges_safe": int(rewritten)}


def _is_singleton_constant_tensor(
    model_ir: ModelIR,
    tensor_name: str,
) -> bool:
    tensor = model_ir.tensors.get(str(tensor_name), None)
    if tensor is None or tensor.data is None:
        return False
    try:
        array = np.asarray(tensor.data)
    except Exception:
        return False
    return int(array.size) == 1


def _permute_tensor_metadata_if_rank_matches(
    tensor: Optional[TensorIR],
    perm: List[int],
) -> None:
    if tensor is None:
        return
    shape_src = list(tensor.shape) if tensor.shape is not None else None
    if shape_src is not None and len(shape_src) == len(perm):
        new_shape = _permute_shape(shape_src, perm)
        if new_shape is not None:
            tensor.shape = [int(v) for v in list(new_shape)]
    signature_src = (
        list(tensor.shape_signature)
        if tensor.shape_signature is not None
        else (list(tensor.shape) if tensor.shape is not None else None)
    )
    if signature_src is not None and len(signature_src) == len(perm):
        new_signature = _permute_shape(signature_src, perm)
        if new_signature is not None:
            tensor.shape_signature = [int(v) for v in list(new_signature)]


def _optimize_leading_input_transpose_passthrough_chains(model_ir: ModelIR) -> Dict[str, int]:
    """
    Fold leading input-boundary transpose chains through layout-agnostic ops.

    Target:
      X_in(NHWC) --T(P)--> X_ncx --(SUB|ADD|MUL|DIV|QUANTIZE|DEQUANTIZE)*--> Y_ncx --T(inv(P))--> Y_nhwc

    Rewrite:
      X_in(NHWC) --(same chain)*--> Y_nhwc

    Safety:
    - Leading transpose input must be a model input tensor.
    - Chain must be strictly linear (single-consumer on the main path).
    - Binary ops in the chain must use singleton constants on the side input.
    - Quantization in the chain must remain per-tensor only.
    """
    rewritten = 0
    unary_passthrough_ops = {"QUANTIZE", "DEQUANTIZE"}
    binary_passthrough_ops = {"ADD", "SUB", "MUL", "DIV"}

    while True:
        changed = False
        consumers = _build_tensor_consumer_map(model_ir)
        model_inputs = set(model_ir.inputs)
        model_outputs = set(model_ir.outputs)

        for pre_idx, pre_op in enumerate(model_ir.operators):
            if str(pre_op.op_type) != "TRANSPOSE":
                continue
            if len(pre_op.inputs) < 2 or len(pre_op.outputs) != 1:
                continue

            pre_input_name = str(pre_op.inputs[0])
            if pre_input_name not in model_inputs:
                continue

            perm_pre = _read_transpose_perm(model_ir, pre_op)
            if perm_pre is None:
                continue
            perm_post_expected = _invert_perm(perm_pre)
            if perm_post_expected is None:
                continue

            pre_output_name = str(pre_op.outputs[0])
            if pre_output_name in model_outputs:
                continue

            # Build a strict linear passthrough chain.
            chain_indices: List[int] = []
            chain_ops: List[OperatorIR] = []
            chain_output_names: List[str] = []
            current_tensor = pre_output_name
            while True:
                current_users = [int(v) for v in consumers.get(current_tensor, [])]
                if len(current_users) != 1:
                    break
                op_idx = int(current_users[0])
                op = model_ir.operators[op_idx]
                op_type = str(op.op_type)

                if op_type in unary_passthrough_ops:
                    if len(op.inputs) != 1 or len(op.outputs) != 1:
                        break
                    if str(op.inputs[0]) != current_tensor:
                        break
                    out_name = str(op.outputs[0])
                    if out_name in model_outputs:
                        break
                    out_tensor = model_ir.tensors.get(out_name, None)
                    if out_tensor is not None and not _is_per_tensor_quantization(out_tensor.quantization):
                        break
                    chain_indices.append(int(op_idx))
                    chain_ops.append(op)
                    chain_output_names.append(out_name)
                    current_tensor = out_name
                    continue

                if op_type in binary_passthrough_ops:
                    if len(op.inputs) != 2 or len(op.outputs) != 1:
                        break
                    input_0 = str(op.inputs[0])
                    input_1 = str(op.inputs[1])
                    if input_0 == current_tensor:
                        side_input_name = input_1
                    elif input_1 == current_tensor:
                        side_input_name = input_0
                    else:
                        break
                    if not _is_singleton_constant_tensor(model_ir, side_input_name):
                        break
                    side_tensor = model_ir.tensors.get(side_input_name, None)
                    if side_tensor is not None and not _is_per_tensor_quantization(side_tensor.quantization):
                        break
                    out_name = str(op.outputs[0])
                    if out_name in model_outputs:
                        break
                    out_tensor = model_ir.tensors.get(out_name, None)
                    if out_tensor is not None and not _is_per_tensor_quantization(out_tensor.quantization):
                        break
                    chain_indices.append(int(op_idx))
                    chain_ops.append(op)
                    chain_output_names.append(out_name)
                    current_tensor = out_name
                    continue

                break

            if len(chain_ops) == 0:
                continue

            # Chain must end with exactly one inverse transpose.
            tail_users = [int(v) for v in consumers.get(current_tensor, [])]
            if len(tail_users) != 1:
                continue
            post_idx = int(tail_users[0])
            if post_idx in set(chain_indices):
                continue
            post_op = model_ir.operators[post_idx]
            if str(post_op.op_type) != "TRANSPOSE":
                continue
            if len(post_op.inputs) < 2 or len(post_op.outputs) != 1:
                continue
            if str(post_op.inputs[0]) != current_tensor:
                continue
            perm_post = _read_transpose_perm(model_ir, post_op)
            if perm_post is None or perm_post != perm_post_expected:
                continue

            # Ensure chain topology remains linear on the main path.
            linear_ok = True
            previous_tensor_name = pre_output_name
            chain_index_to_pos = {int(idx): pos for pos, idx in enumerate(chain_indices)}
            for op_idx, op in zip(chain_indices, chain_ops):
                if previous_tensor_name not in set(str(v) for v in op.inputs):
                    linear_ok = False
                    break
                out_name = str(op.outputs[0]) if len(op.outputs) > 0 else ""
                if out_name == "":
                    linear_ok = False
                    break
                expected_users = [int(v) for v in consumers.get(out_name, [])]
                pos = int(chain_index_to_pos[int(op_idx)])
                if pos < len(chain_indices) - 1:
                    if expected_users != [int(chain_indices[pos + 1])]:
                        linear_ok = False
                        break
                else:
                    if expected_users != [int(post_idx)]:
                        linear_ok = False
                        break
                previous_tensor_name = out_name
            if not linear_ok:
                continue

            old_last_name = str(chain_ops[-1].outputs[0])
            post_output_name = str(post_op.outputs[0])
            if old_last_name in model_outputs:
                continue

            # Rewire chain head: consume model input directly.
            first_op = chain_ops[0]
            first_input_names = [str(v) for v in first_op.inputs]
            if first_input_names[0] == pre_output_name:
                first_op.inputs = [pre_input_name] + first_input_names[1:]
            elif len(first_input_names) > 1 and first_input_names[1] == pre_output_name:
                first_op.inputs = [first_input_names[0], pre_input_name]
            else:
                continue

            # Chain tail now produces post-transpose output name directly.
            chain_ops[-1].outputs = [post_output_name]

            # Update metadata to NHWC-side layout.
            for out_name in chain_output_names[:-1]:
                _permute_tensor_metadata_if_rank_matches(
                    model_ir.tensors.get(out_name, None),
                    perm_post_expected,
                )
            old_last_tensor = model_ir.tensors.get(old_last_name, None)
            post_output_tensor = model_ir.tensors.get(post_output_name, None)
            if post_output_tensor is not None and old_last_tensor is not None:
                post_output_tensor.dtype = str(old_last_tensor.dtype)
                post_output_tensor.quantization = _clone_quantization(old_last_tensor.quantization)
                _permute_tensor_metadata_if_rank_matches(
                    post_output_tensor,
                    perm_post_expected,
                )

            # Remove boundary transposes.
            remove_indices = sorted(list({int(pre_idx), int(post_idx)}), reverse=True)
            for remove_idx in remove_indices:
                del model_ir.operators[int(remove_idx)]

            rewritten += 1
            changed = True
            break

        if not changed:
            break

    _prune_unused_tensors(model_ir)
    return {"rewritten_leading_input_transpose_passthrough_chains": int(rewritten)}


def _optimize_dequant_prelu_quantize_chains(model_ir: ModelIR) -> Dict[str, int]:
    """
    Fold DEQUANTIZE->PRELU->QUANTIZE into quantized PRELU.

    Target pattern:
      Xq --DEQUANTIZE--> Xf --PRELU(alpha_f)--> Yf --QUANTIZE--> Yq

    Rewritten:
      Xq --PRELU(alpha_q)--> Yq
    """
    folded = 0

    def _unique_tensor_name(base: str) -> str:
        name = str(base)
        suffix = 1
        while name in model_ir.tensors:
            name = f"{base}_{suffix}"
            suffix += 1
        return name

    while True:
        changed = False
        consumers = _build_tensor_consumer_map(model_ir)

        for dq_idx, dq_op in enumerate(model_ir.operators):
            if str(dq_op.op_type) != "DEQUANTIZE" or len(dq_op.inputs) != 1 or len(dq_op.outputs) != 1:
                continue
            q_in_name = str(dq_op.inputs[0])
            f_in_name = str(dq_op.outputs[0])

            prelu_users = consumers.get(f_in_name, [])
            if len(prelu_users) != 1:
                continue
            prelu_idx = int(prelu_users[0])
            prelu_op = model_ir.operators[prelu_idx]
            if str(prelu_op.op_type) != "PRELU" or len(prelu_op.inputs) != 2 or len(prelu_op.outputs) != 1:
                continue
            if str(prelu_op.inputs[0]) != f_in_name:
                continue

            f_out_name = str(prelu_op.outputs[0])
            q_users = consumers.get(f_out_name, [])
            if len(q_users) != 1:
                continue
            q_idx = int(q_users[0])
            q_op = model_ir.operators[q_idx]
            if str(q_op.op_type) != "QUANTIZE" or len(q_op.inputs) != 1 or len(q_op.outputs) != 1:
                continue
            if str(q_op.inputs[0]) != f_out_name:
                continue
            q_out_name = str(q_op.outputs[0])

            if f_in_name in model_ir.outputs or f_out_name in model_ir.outputs:
                continue

            q_in_tensor = model_ir.tensors.get(q_in_name, None)
            q_out_tensor = model_ir.tensors.get(q_out_name, None)
            alpha_name = str(prelu_op.inputs[1])
            alpha_tensor = model_ir.tensors.get(alpha_name, None)
            if q_in_tensor is None or q_out_tensor is None or alpha_tensor is None:
                continue
            if not _all_per_tensor_quantized([q_in_tensor, q_out_tensor]):
                continue

            target_dtype = str(q_in_tensor.dtype)
            if target_dtype not in {"INT8", "UINT8"}:
                continue
            if str(q_out_tensor.dtype) != target_dtype:
                continue
            if not isinstance(alpha_tensor.data, np.ndarray):
                continue

            try:
                alpha_q, alpha_qparams = _quantize_prelu_alpha(alpha_tensor.data, target_dtype)
            except Exception:
                continue

            alpha_users = consumers.get(alpha_name, [])
            if len(alpha_users) == 1 and int(alpha_users[0]) == int(prelu_idx):
                alpha_q_name = alpha_name
                alpha_tensor.data = np.asarray(alpha_q)
                alpha_tensor.dtype = target_dtype
                alpha_tensor.shape = [int(v) for v in alpha_q.shape]
                alpha_tensor.shape_signature = [int(v) for v in alpha_q.shape]
                alpha_tensor.quantization = alpha_qparams
            else:
                alpha_q_name = _unique_tensor_name(f"{alpha_name}_q")
                model_ir.tensors[alpha_q_name] = TensorIR(
                    name=alpha_q_name,
                    dtype=target_dtype,
                    shape=[int(v) for v in alpha_q.shape],
                    shape_signature=[int(v) for v in alpha_q.shape],
                    data=np.asarray(alpha_q),
                    is_variable=False,
                    quantization=alpha_qparams,
                )

            prelu_op.inputs = [q_in_name, alpha_q_name]
            prelu_op.outputs = [q_out_name]

            for remove_idx in sorted([dq_idx, q_idx], reverse=True):
                del model_ir.operators[remove_idx]
            folded += 1
            changed = True
            break

        if not changed:
            break

    _prune_unused_tensors(model_ir)
    return {"folded_dequant_prelu_quantize_chains": int(folded)}


def _optimize_dequant_prelu_depthwise_quantize_chains(model_ir: ModelIR) -> Dict[str, int]:
    """
    Fold DEQUANTIZE->PRELU->DEPTHWISE_CONV_2D->QUANTIZE into quantized PRELU+DEPTHWISE_CONV_2D.

    Target pattern:
      Xq --DEQUANTIZE--> Xf --PRELU(alpha_f)--> Pf --DEPTHWISE_CONV_2D(w_f,b_f)--> Yf --QUANTIZE--> Yq

    Rewritten:
      Xq --PRELU(alpha_q)--> Pq --DEPTHWISE_CONV_2D(w_q,b_q)--> Yq
    """
    folded = 0

    def _unique_tensor_name(base: str) -> str:
        name = str(base)
        suffix = 1
        while name in model_ir.tensors:
            name = f"{base}_{suffix}"
            suffix += 1
        return name

    while True:
        changed = False
        consumers = _build_tensor_consumer_map(model_ir)

        for dq_idx, dq_op in enumerate(model_ir.operators):
            if str(dq_op.op_type) != "DEQUANTIZE" or len(dq_op.inputs) != 1 or len(dq_op.outputs) != 1:
                continue
            q_in_name = str(dq_op.inputs[0])
            f_in_name = str(dq_op.outputs[0])

            prelu_users = consumers.get(f_in_name, [])
            if len(prelu_users) != 1:
                continue
            prelu_idx = int(prelu_users[0])
            prelu_op = model_ir.operators[prelu_idx]
            if str(prelu_op.op_type) != "PRELU" or len(prelu_op.inputs) != 2 or len(prelu_op.outputs) != 1:
                continue
            if str(prelu_op.inputs[0]) != f_in_name:
                continue
            p_f_name = str(prelu_op.outputs[0])

            dw_users = consumers.get(p_f_name, [])
            if len(dw_users) != 1:
                continue
            dw_idx = int(dw_users[0])
            dw_op = model_ir.operators[dw_idx]
            if str(dw_op.op_type) != "DEPTHWISE_CONV_2D" or len(dw_op.inputs) != 3 or len(dw_op.outputs) != 1:
                continue
            if str(dw_op.inputs[0]) != p_f_name:
                continue
            y_f_name = str(dw_op.outputs[0])

            q_users = consumers.get(y_f_name, [])
            if len(q_users) != 1:
                continue
            q_idx = int(q_users[0])
            q_op = model_ir.operators[q_idx]
            if str(q_op.op_type) != "QUANTIZE" or len(q_op.inputs) != 1 or len(q_op.outputs) != 1:
                continue
            if str(q_op.inputs[0]) != y_f_name:
                continue
            y_q_name = str(q_op.outputs[0])

            if f_in_name in model_ir.outputs or p_f_name in model_ir.outputs or y_f_name in model_ir.outputs:
                continue

            q_in_tensor = model_ir.tensors.get(q_in_name, None)
            y_q_tensor = model_ir.tensors.get(y_q_name, None)
            alpha_tensor = model_ir.tensors.get(str(prelu_op.inputs[1]), None)
            w_f_tensor = model_ir.tensors.get(str(dw_op.inputs[1]), None)
            b_f_tensor = model_ir.tensors.get(str(dw_op.inputs[2]), None)
            if (
                q_in_tensor is None
                or y_q_tensor is None
                or alpha_tensor is None
                or w_f_tensor is None
                or b_f_tensor is None
            ):
                continue
            if not _all_per_tensor_quantized([q_in_tensor, y_q_tensor]):
                continue

            target_dtype = str(q_in_tensor.dtype)
            if target_dtype not in {"INT8", "UINT8"}:
                continue
            if str(y_q_tensor.dtype) != target_dtype:
                continue
            if not isinstance(alpha_tensor.data, np.ndarray):
                continue
            if not isinstance(w_f_tensor.data, np.ndarray):
                continue
            if not isinstance(b_f_tensor.data, np.ndarray):
                continue

            x_qparams = _get_per_tensor_scale_zero_point(q_in_tensor.quantization)
            if x_qparams is None:
                continue
            x_scale, _x_zero_point = x_qparams

            weights_f = np.asarray(w_f_tensor.data, dtype=np.float32)
            if weights_f.ndim != 4:
                continue
            bias_f = np.asarray(b_f_tensor.data, dtype=np.float32).reshape(-1)
            if bias_f.size != int(weights_f.shape[-1]):
                continue

            try:
                alpha_q, alpha_qparams = _quantize_prelu_alpha(alpha_tensor.data, target_dtype)
                w_q, w_qparams = _quantize_tensor_per_tensor(weights_f, target_dtype)
            except Exception:
                continue

            w_scale = float(w_qparams.scale[0])
            bias_scale = max(float(x_scale * w_scale), 1e-12)
            bias_q = np.clip(
                np.round(bias_f / bias_scale),
                np.iinfo(np.int32).min,
                np.iinfo(np.int32).max,
            ).astype(np.int32)
            b_qparams = QuantParamIR(
                scale=[float(bias_scale)],
                zero_point=[0],
                quantized_dimension=0,
            )

            alpha_q_name = _unique_tensor_name(f"{prelu_op.inputs[1]}_q")
            model_ir.tensors[alpha_q_name] = TensorIR(
                name=alpha_q_name,
                dtype=target_dtype,
                shape=[int(v) for v in alpha_q.shape],
                shape_signature=[int(v) for v in alpha_q.shape],
                data=np.asarray(alpha_q),
                is_variable=False,
                quantization=alpha_qparams,
            )

            w_q_name = _unique_tensor_name(f"{dw_op.inputs[1]}_q")
            model_ir.tensors[w_q_name] = TensorIR(
                name=w_q_name,
                dtype=target_dtype,
                shape=[int(v) for v in w_q.shape],
                shape_signature=[int(v) for v in w_q.shape],
                data=np.asarray(w_q),
                is_variable=False,
                quantization=w_qparams,
            )

            b_q_name = _unique_tensor_name(f"{dw_op.inputs[2]}_q")
            model_ir.tensors[b_q_name] = TensorIR(
                name=b_q_name,
                dtype="INT32",
                shape=[int(v) for v in bias_q.shape],
                shape_signature=[int(v) for v in bias_q.shape],
                data=np.asarray(bias_q),
                is_variable=False,
                quantization=b_qparams,
            )

            p_q_tensor = model_ir.tensors.get(p_f_name, None)
            if p_q_tensor is not None:
                p_q_tensor.dtype = target_dtype
                p_q_tensor.quantization = _clone_quantization(q_in_tensor.quantization)

            prelu_op.inputs = [q_in_name, alpha_q_name]
            dw_op.inputs = [p_f_name, w_q_name, b_q_name]
            dw_op.outputs = [y_q_name]

            for remove_idx in sorted([dq_idx, q_idx], reverse=True):
                del model_ir.operators[remove_idx]
            folded += 1
            changed = True
            break

        if not changed:
            break

    _prune_unused_tensors(model_ir)
    return {"folded_dequant_prelu_depthwise_quantize_chains": int(folded)}


def _optimize_dequant_reshape_quantize_chains(model_ir: ModelIR) -> Dict[str, int]:
    """
    Fold DEQUANTIZE->RESHAPE->QUANTIZE into quantized RESHAPE.

    Target pattern:
      Xq --DEQUANTIZE--> Xf --RESHAPE(shape)--> Yf --QUANTIZE--> Yq

    Rewritten:
      Xq --RESHAPE(shape)--> Yq

    Safety conditions:
    - Chain is linear (single consumer at each bridge tensor)
    - input/output quantized tensors use equivalent per-tensor quantization
    - input/output quantized dtypes are identical
    """
    folded = 0

    while True:
        changed = False
        consumers = _build_tensor_consumer_map(model_ir)

        for dq_idx, dq_op in enumerate(model_ir.operators):
            if str(dq_op.op_type) != "DEQUANTIZE" or len(dq_op.inputs) != 1 or len(dq_op.outputs) != 1:
                continue
            q_in_name = str(dq_op.inputs[0])
            f_in_name = str(dq_op.outputs[0])

            reshape_users = consumers.get(f_in_name, [])
            if len(reshape_users) != 1:
                continue
            reshape_idx = int(reshape_users[0])
            reshape_op = model_ir.operators[reshape_idx]
            if str(reshape_op.op_type) != "RESHAPE" or len(reshape_op.inputs) < 1 or len(reshape_op.outputs) != 1:
                continue
            if str(reshape_op.inputs[0]) != f_in_name:
                continue
            f_out_name = str(reshape_op.outputs[0])

            q_users = consumers.get(f_out_name, [])
            if len(q_users) != 1:
                continue
            q_idx = int(q_users[0])
            q_op = model_ir.operators[q_idx]
            if str(q_op.op_type) != "QUANTIZE" or len(q_op.inputs) != 1 or len(q_op.outputs) != 1:
                continue
            if str(q_op.inputs[0]) != f_out_name:
                continue
            q_out_name = str(q_op.outputs[0])

            if f_in_name in model_ir.outputs or f_out_name in model_ir.outputs:
                continue

            q_in_tensor = model_ir.tensors.get(q_in_name, None)
            q_out_tensor = model_ir.tensors.get(q_out_name, None)
            f_out_tensor = model_ir.tensors.get(f_out_name, None)
            if q_in_tensor is None or q_out_tensor is None:
                continue
            if not _all_per_tensor_quantized([q_in_tensor, q_out_tensor]):
                continue
            if not _is_same_per_tensor_quantization(
                q_in_tensor.quantization,
                q_out_tensor.quantization,
            ):
                continue

            q_in_dtype = str(q_in_tensor.dtype)
            q_out_dtype = str(q_out_tensor.dtype)
            if q_in_dtype != q_out_dtype:
                continue
            if q_in_dtype in {"FLOAT16", "FLOAT32", "FLOAT64", "BOOL", "STRING"}:
                continue

            reshape_op.inputs[0] = q_in_name
            reshape_op.outputs = [q_out_name]

            q_out_tensor.dtype = q_in_dtype
            q_out_tensor.quantization = _clone_quantization(q_in_tensor.quantization)
            if f_out_tensor is not None:
                q_out_tensor.shape = [int(v) for v in list(f_out_tensor.shape)]
                if f_out_tensor.shape_signature is not None:
                    q_out_tensor.shape_signature = [
                        int(v) for v in list(f_out_tensor.shape_signature)
                    ]
                else:
                    q_out_tensor.shape_signature = [int(v) for v in list(f_out_tensor.shape)]

            for remove_idx in sorted([dq_idx, q_idx], reverse=True):
                del model_ir.operators[remove_idx]
            folded += 1
            changed = True
            break

        if not changed:
            break

    _prune_unused_tensors(model_ir)
    return {"folded_dequant_reshape_quantize_chains": int(folded)}


def _optimize_dequant_softmax_quantize_chains(model_ir: ModelIR) -> Dict[str, int]:
    """
    Fold DEQUANTIZE->SOFTMAX->QUANTIZE into quantized SOFTMAX when TFLite-compatible.

    Target pattern:
      Xq --DEQUANTIZE--> Xf --SOFTMAX(beta=1.0)--> Yf --QUANTIZE--> Yq

    Rewritten:
      Xq --SOFTMAX(int8/uint8)--> Yq

    Safety conditions:
    - Chain is linear (single consumer at each bridge tensor)
    - input/output quantized tensors are per-tensor INT8/UINT8
    - output quantization matches TFLite quantized softmax canonical params
      (INT8: scale=1/256, zp=-128; UINT8: scale=1/256, zp=0)
    - Softmax beta is approximately 1.0
    """
    folded = 0

    def _is_supported_softmax_qparams(dtype: str, quantization: Any) -> bool:
        qparams = _get_per_tensor_scale_zero_point(quantization)
        if qparams is None:
            return False
        scale, zero_point = qparams
        if not np.isclose(float(scale), 1.0 / 256.0, rtol=0.0, atol=1e-7):
            return False
        dtype_u = str(dtype).upper()
        if dtype_u == "INT8":
            return int(zero_point) == -128
        if dtype_u == "UINT8":
            return int(zero_point) == 0
        return False

    while True:
        changed = False
        consumers = _build_tensor_consumer_map(model_ir)

        for dq_idx, dq_op in enumerate(model_ir.operators):
            if str(dq_op.op_type) != "DEQUANTIZE" or len(dq_op.inputs) != 1 or len(dq_op.outputs) != 1:
                continue
            q_in_name = str(dq_op.inputs[0])
            f_in_name = str(dq_op.outputs[0])

            softmax_users = consumers.get(f_in_name, [])
            if len(softmax_users) != 1:
                continue
            softmax_idx = int(softmax_users[0])
            softmax_op = model_ir.operators[softmax_idx]
            if str(softmax_op.op_type) != "SOFTMAX" or len(softmax_op.inputs) != 1 or len(softmax_op.outputs) != 1:
                continue
            if str(softmax_op.inputs[0]) != f_in_name:
                continue
            f_out_name = str(softmax_op.outputs[0])

            q_users = consumers.get(f_out_name, [])
            if len(q_users) != 1:
                continue
            q_idx = int(q_users[0])
            q_op = model_ir.operators[q_idx]
            if str(q_op.op_type) != "QUANTIZE" or len(q_op.inputs) != 1 or len(q_op.outputs) != 1:
                continue
            if str(q_op.inputs[0]) != f_out_name:
                continue
            q_out_name = str(q_op.outputs[0])

            if f_in_name in model_ir.outputs or f_out_name in model_ir.outputs:
                continue

            q_in_tensor = model_ir.tensors.get(q_in_name, None)
            q_out_tensor = model_ir.tensors.get(q_out_name, None)
            f_in_tensor = model_ir.tensors.get(f_in_name, None)
            f_out_tensor = model_ir.tensors.get(f_out_name, None)
            if q_in_tensor is None or q_out_tensor is None:
                continue
            if not _all_per_tensor_quantized([q_in_tensor, q_out_tensor]):
                continue

            q_in_dtype = str(q_in_tensor.dtype).upper()
            q_out_dtype = str(q_out_tensor.dtype).upper()
            if q_in_dtype not in {"INT8", "UINT8"} or q_out_dtype != q_in_dtype:
                continue
            if not _is_supported_softmax_qparams(q_out_dtype, q_out_tensor.quantization):
                continue

            beta = float(softmax_op.options.get("beta", 1.0))
            if not np.isclose(beta, 1.0, rtol=0.0, atol=1e-6):
                continue

            if f_in_tensor is not None and not _shapes_match_if_known(q_in_tensor.shape, f_in_tensor.shape):
                continue
            if f_out_tensor is not None and not _shapes_match_if_known(q_out_tensor.shape, f_out_tensor.shape):
                continue

            softmax_op.inputs = [q_in_name]
            softmax_op.outputs = [q_out_name]
            # TFLite quantized SOFTMAX opcode version differs by dtype.
            softmax_op.version = 2 if q_out_dtype == "INT8" else 1

            # Keep output tensor metadata aligned to the pre-quantized softmax output shape.
            if f_out_tensor is not None:
                q_out_tensor.shape = [int(v) for v in list(f_out_tensor.shape)]
                if f_out_tensor.shape_signature is not None:
                    q_out_tensor.shape_signature = [int(v) for v in list(f_out_tensor.shape_signature)]
                else:
                    q_out_tensor.shape_signature = [int(v) for v in list(f_out_tensor.shape)]

            for remove_idx in sorted([dq_idx, q_idx], reverse=True):
                del model_ir.operators[remove_idx]
            folded += 1
            changed = True
            break

        if not changed:
            break

    _prune_unused_tensors(model_ir)
    return {"folded_dequant_softmax_quantize_chains": int(folded)}


def _optimize_terminal_quantize_dequantize(model_ir: ModelIR) -> Dict[str, int]:
    """
    Remove terminal QUANTIZE->DEQUANTIZE pairs for float outputs when safe.

    Target pattern:
      x_float -> QUANTIZE -> q -> DEQUANTIZE -> y_float (graph output)

    Safety conditions:
    - q is consumed only by that DEQUANTIZE
    - y_float is a graph output and has no internal consumers
    - x_float is produced by another op and consumed only by QUANTIZE
      (allows preserving output name by renaming x_float -> y_float)
    """
    removed_pairs = 0

    while True:
        changed = False
        consumers = _build_tensor_consumer_map(model_ir)
        producers = _build_tensor_producer_map(model_ir)

        for q_idx, q_op in enumerate(model_ir.operators):
            if str(q_op.op_type) != "QUANTIZE":
                continue
            if len(q_op.inputs) != 1 or len(q_op.outputs) != 1:
                continue

            float_input_name = q_op.inputs[0]
            quantized_name = q_op.outputs[0]
            if float_input_name in model_ir.inputs:
                continue

            quantized_users = consumers.get(quantized_name, [])
            if len(quantized_users) != 1:
                continue

            dq_idx = int(quantized_users[0])
            if dq_idx == q_idx:
                continue
            dq_op = model_ir.operators[dq_idx]
            if str(dq_op.op_type) != "DEQUANTIZE":
                continue
            if len(dq_op.inputs) != 1 or len(dq_op.outputs) != 1:
                continue
            if dq_op.inputs[0] != quantized_name:
                continue

            float_output_name = dq_op.outputs[0]
            if float_output_name not in model_ir.outputs:
                continue
            if len(consumers.get(float_output_name, [])) > 0:
                continue

            float_input_users = consumers.get(float_input_name, [])
            if len(float_input_users) != 1 or int(float_input_users[0]) != q_idx:
                continue
            if float_input_name not in producers:
                continue
            if float_input_name in model_ir.outputs:
                continue

            _rename_tensor_globally(
                model_ir=model_ir,
                old_name=float_input_name,
                new_name=float_output_name,
            )

            for remove_idx in sorted([q_idx, dq_idx], reverse=True):
                del model_ir.operators[remove_idx]
            removed_pairs += 1
            changed = True
            break

        if not changed:
            break

    _prune_unused_tensors(model_ir)
    return {
        "removed_terminal_quantize_dequantize_pairs": int(removed_pairs),
    }


def _sanitize_terminal_transpose_before_dequantize(model_ir: ModelIR) -> Dict[str, int]:
    """
    Sanitize terminal TRANSPOSE->DEQUANTIZE by swapping to DEQUANTIZE->TRANSPOSE.

    Target pattern:
      q --TRANSPOSE(P)--> q_t --DEQUANTIZE--> y(graph output)

    Rewritten:
      q --DEQUANTIZE--> y_pre --TRANSPOSE(P)--> y(graph output)

    Safety conditions:
    - y is a graph output without internal consumers
    - q_t is consumed only by that DEQUANTIZE
    - q/q_t quantization is per-tensor (to avoid axis remap ambiguity)
    """
    sanitized = 0
    removed_terminal_dequantize_transpose = 0

    def _unique_tensor_name(base: str) -> str:
        name = str(base)
        suffix = 1
        while name in model_ir.tensors:
            name = f"{base}_{suffix}"
            suffix += 1
        return name

    while True:
        changed = False
        consumers = _build_tensor_consumer_map(model_ir)
        producers = _build_tensor_producer_map(model_ir)

        for dq_idx, dq_op in enumerate(model_ir.operators):
            if str(dq_op.op_type) != "DEQUANTIZE":
                continue
            if len(dq_op.inputs) != 1 or len(dq_op.outputs) != 1:
                continue

            dq_input_name = str(dq_op.inputs[0])
            dq_output_name = str(dq_op.outputs[0])
            if dq_output_name not in model_ir.outputs:
                continue
            if len(consumers.get(dq_output_name, [])) > 0:
                continue
            if dq_input_name in model_ir.outputs:
                continue

            transpose_idx = producers.get(dq_input_name, None)
            if transpose_idx is None:
                continue
            transpose_op = model_ir.operators[int(transpose_idx)]
            if str(transpose_op.op_type) != "TRANSPOSE":
                continue
            if len(transpose_op.inputs) < 2 or len(transpose_op.outputs) != 1:
                continue
            if str(transpose_op.outputs[0]) != dq_input_name:
                continue

            q_input_name = str(transpose_op.inputs[0])
            if q_input_name in model_ir.outputs:
                continue

            dq_input_users = [int(v) for v in consumers.get(dq_input_name, [])]
            if len(dq_input_users) != 1 or int(dq_input_users[0]) != int(dq_idx):
                continue

            q_tensor = model_ir.tensors.get(q_input_name, None)
            qt_tensor = model_ir.tensors.get(dq_input_name, None)
            y_tensor = model_ir.tensors.get(dq_output_name, None)
            if q_tensor is None or qt_tensor is None or y_tensor is None:
                continue
            if not _all_per_tensor_quantized([q_tensor, qt_tensor]):
                continue

            q_shape = list(q_tensor.shape)
            q_signature = (
                list(q_tensor.shape_signature)
                if q_tensor.shape_signature is not None
                else list(q_shape)
            )
            if len(q_shape) == 0:
                continue
            perm = _read_transpose_perm(model_ir, transpose_op)
            if perm is None:
                continue
            expected_out_shape = _permute_shape(q_shape, perm)
            expected_out_signature = _permute_shape(q_signature, perm)
            if expected_out_shape is None or expected_out_signature is None:
                continue

            pre_transpose_name = _unique_tensor_name(f"{dq_output_name}_before_transpose")
            model_ir.tensors[pre_transpose_name] = TensorIR(
                name=pre_transpose_name,
                dtype=str(y_tensor.dtype),
                shape=[int(v) for v in q_shape],
                shape_signature=[int(v) for v in q_signature],
                data=None,
            )

            dq_op.inputs = [q_input_name]
            dq_op.outputs = [pre_transpose_name]
            transpose_op.inputs = [pre_transpose_name, str(transpose_op.inputs[1])]
            transpose_op.outputs = [dq_output_name]

            y_tensor.shape = [int(v) for v in expected_out_shape]
            y_tensor.shape_signature = [int(v) for v in expected_out_signature]

            transpose_idx_int = int(transpose_idx)
            if transpose_idx_int < int(dq_idx):
                moved = model_ir.operators.pop(transpose_idx_int)
                insert_idx = int(dq_idx) - 1
                model_ir.operators.insert(insert_idx + 1, moved)

            sanitized += 1
            changed = True
            break

        if changed:
            continue

        # Pattern B:
        #   q --DEQUANTIZE--> y_pre --TRANSPOSE(P)--> y(graph output)
        #   => q --DEQUANTIZE--> y(graph output)
        # Keep user-visible output tensor name stable by renaming y_pre -> y.
        for transpose_idx, transpose_op in enumerate(model_ir.operators):
            if str(transpose_op.op_type) != "TRANSPOSE":
                continue
            if len(transpose_op.inputs) < 2 or len(transpose_op.outputs) != 1:
                continue

            pre_output_name = str(transpose_op.inputs[0])
            final_output_name = str(transpose_op.outputs[0])
            if final_output_name not in model_ir.outputs:
                continue
            if len(consumers.get(final_output_name, [])) > 0:
                continue
            if pre_output_name in model_ir.outputs:
                continue

            pre_users = [int(v) for v in consumers.get(pre_output_name, [])]
            if len(pre_users) != 1 or int(pre_users[0]) != int(transpose_idx):
                continue

            dq_idx = producers.get(pre_output_name, None)
            if dq_idx is None:
                continue
            dq_op = model_ir.operators[int(dq_idx)]
            if str(dq_op.op_type) != "DEQUANTIZE":
                continue
            if len(dq_op.inputs) != 1 or len(dq_op.outputs) != 1:
                continue
            if str(dq_op.outputs[0]) != pre_output_name:
                continue

            _rename_tensor_globally(
                model_ir=model_ir,
                old_name=pre_output_name,
                new_name=final_output_name,
            )
            del model_ir.operators[int(transpose_idx)]
            removed_terminal_dequantize_transpose += 1
            changed = True
            break

        if not changed:
            break

    _prune_unused_tensors(model_ir)
    return {
        "sanitized_terminal_transpose_before_dequantize": int(sanitized),
        "removed_terminal_dequantize_transpose": int(removed_terminal_dequantize_transpose),
    }


def _optimize_concat_pre_quantize_dequantize(model_ir: ModelIR) -> Dict[str, int]:
    """
    Bypass redundant QUANTIZE->DEQUANTIZE immediately before CONCATENATION inputs.

    Target pattern:
      ... -> x_float --QUANTIZE--> q --DEQUANTIZE--> x_dq --CONCATENATION--> ...

    Rewritten:
      ... -> x_float ---------------------------------> --CONCATENATION--> ...

    Safety conditions:
    - q is consumed only by that DEQUANTIZE
    - neither q nor x_dq is a graph output
    - x_float is already in a dequantized flow:
      - producer(x_float) is DEQUANTIZE, or
      - producer(x_float) has at least one DEQUANTIZE input
    """
    bypassed = 0

    def _has_dequantized_origin(
        tensor_name: str,
        producers: Dict[str, int],
        max_depth: int = 6,
    ) -> bool:
        traceable_ops = {
            "TRANSPOSE",
            "RESHAPE",
            "MAX_POOL_2D",
            "AVERAGE_POOL_2D",
            "PAD",
            "ADD",
            "SUB",
            "MUL",
            "DIV",
            "CONCATENATION",
            "MEAN",
        }
        visited = set()
        stack: List[Tuple[str, int]] = [(str(tensor_name), 0)]
        while len(stack) > 0:
            current_name, depth = stack.pop()
            key = (str(current_name), int(depth))
            if key in visited:
                continue
            visited.add(key)
            if int(depth) > int(max_depth):
                continue

            producer_idx = producers.get(str(current_name), None)
            if producer_idx is None:
                continue
            producer = model_ir.operators[int(producer_idx)]
            producer_type = str(producer.op_type)
            if producer_type == "DEQUANTIZE":
                return True
            if producer_type not in traceable_ops:
                continue
            for input_name in list(producer.inputs):
                stack.append((str(input_name), int(depth) + 1))
        return False

    while True:
        changed = False
        consumers = _build_tensor_consumer_map(model_ir)
        producers = _build_tensor_producer_map(model_ir)

        for concat_idx, concat_op in enumerate(model_ir.operators):
            if str(concat_op.op_type) != "CONCATENATION":
                continue
            if len(concat_op.inputs) == 0:
                continue

            concat_inputs = list(concat_op.inputs)
            for input_pos, concat_input_name in enumerate(concat_inputs):
                dq_idx = producers.get(str(concat_input_name), None)
                if dq_idx is None:
                    continue
                dq_op = model_ir.operators[int(dq_idx)]
                if str(dq_op.op_type) != "DEQUANTIZE":
                    continue
                if len(dq_op.inputs) != 1 or len(dq_op.outputs) != 1:
                    continue
                if str(dq_op.outputs[0]) != str(concat_input_name):
                    continue

                quantized_name = str(dq_op.inputs[0])
                if quantized_name in model_ir.outputs or str(concat_input_name) in model_ir.outputs:
                    continue

                q_idx = producers.get(quantized_name, None)
                if q_idx is None:
                    continue
                q_op = model_ir.operators[int(q_idx)]
                if str(q_op.op_type) != "QUANTIZE":
                    continue
                if len(q_op.inputs) != 1 or len(q_op.outputs) != 1:
                    continue
                if str(q_op.outputs[0]) != quantized_name:
                    continue

                q_users = [int(v) for v in consumers.get(quantized_name, [])]
                if len(q_users) != 1 or int(q_users[0]) != int(dq_idx):
                    continue

                float_name = str(q_op.inputs[0])
                float_tensor = model_ir.tensors.get(float_name, None)
                dq_tensor = model_ir.tensors.get(str(concat_input_name), None)
                if float_tensor is None or dq_tensor is None:
                    continue
                if str(float_tensor.dtype).upper().startswith("INT"):
                    continue
                if not _shapes_equal(list(float_tensor.shape), list(dq_tensor.shape)):
                    continue
                if not _has_dequantized_origin(float_name, producers):
                    continue

                concat_op.inputs[input_pos] = float_name
                bypassed += 1
                changed = True
                break

            if changed:
                break

        if not changed:
            break

    _prune_unused_tensors(model_ir)
    return {
        "bypassed_concat_pre_quantize_dequantize": int(bypassed),
    }


def _optimize_transpose_dequantize_mean_quantize_bridges(model_ir: ModelIR) -> Dict[str, int]:
    """
    Rewrite TRANSPOSE->DEQUANTIZE->MEAN->QUANTIZE bridges by moving TRANSPOSE after MEAN.

    Target pattern:
      X --TRANSPOSE(P)--> A --DEQUANTIZE--> B --MEAN(axes=K, keepDims=True)--> C --QUANTIZE--> Y

    Rewritten:
      X --DEQUANTIZE--> B --MEAN(axes=map(P,K), keepDims=True)--> C'
        --TRANSPOSE(P)--> C --QUANTIZE--> Y

    This preserves Y layout while avoiding transposing a large activation tensor.
    """
    moved_bridges = 0

    def _unique_tensor_name(base: str) -> str:
        name = str(base)
        suffix = 1
        while name in model_ir.tensors:
            name = f"{base}_{suffix}"
            suffix += 1
        return name

    while True:
        changed = False
        consumers = _build_tensor_consumer_map(model_ir)

        for pre_idx, pre_op in enumerate(model_ir.operators):
            if str(pre_op.op_type) != "TRANSPOSE" or len(pre_op.inputs) < 2 or len(pre_op.outputs) != 1:
                continue
            perm_pre = _read_transpose_perm(model_ir, pre_op)
            if perm_pre is None:
                continue

            pre_input = str(pre_op.inputs[0])
            pre_output = str(pre_op.outputs[0])
            if pre_output in model_ir.outputs:
                continue

            dq_users = consumers.get(pre_output, [])
            if len(dq_users) != 1:
                continue
            dq_idx = int(dq_users[0])
            dq_op = model_ir.operators[dq_idx]
            if str(dq_op.op_type) != "DEQUANTIZE" or len(dq_op.inputs) != 1 or len(dq_op.outputs) != 1:
                continue
            if str(dq_op.inputs[0]) != pre_output:
                continue
            dq_output = str(dq_op.outputs[0])
            if dq_output in model_ir.outputs:
                continue

            mean_users = consumers.get(dq_output, [])
            if len(mean_users) != 1:
                continue
            mean_idx = int(mean_users[0])
            mean_op = model_ir.operators[mean_idx]
            if str(mean_op.op_type) != "MEAN" or len(mean_op.inputs) < 2 or len(mean_op.outputs) != 1:
                continue
            if str(mean_op.inputs[0]) != dq_output:
                continue
            if not bool(mean_op.options.get("keepDims", False)):
                continue

            axes_name = str(mean_op.inputs[1])
            axes_users = consumers.get(axes_name, [])
            if len(axes_users) != 1:
                continue
            axes_tensor = model_ir.tensors.get(axes_name, None)
            if axes_tensor is None or axes_tensor.data is None:
                continue
            old_axes_raw = [int(v) for v in np.asarray(axes_tensor.data).reshape(-1).tolist()]
            if len(old_axes_raw) == 0:
                continue

            mean_output = str(mean_op.outputs[0])
            if mean_output in model_ir.outputs:
                continue

            q_users = consumers.get(mean_output, [])
            if len(q_users) != 1:
                continue
            q_idx = int(q_users[0])
            q_op = model_ir.operators[q_idx]
            if str(q_op.op_type) != "QUANTIZE" or len(q_op.inputs) != 1 or len(q_op.outputs) != 1:
                continue
            if str(q_op.inputs[0]) != mean_output:
                continue

            input_tensor = model_ir.tensors.get(pre_input, None)
            dq_tensor = model_ir.tensors.get(dq_output, None)
            mean_tensor = model_ir.tensors.get(mean_output, None)
            if input_tensor is None or dq_tensor is None or mean_tensor is None:
                continue
            rank = len(list(input_tensor.shape))
            if rank <= 0 or len(perm_pre) != rank:
                continue

            old_axes: List[int] = []
            valid_axes = True
            for axis in old_axes_raw:
                norm_axis = int(axis)
                if norm_axis < 0:
                    norm_axis += rank
                if norm_axis < 0 or norm_axis >= rank:
                    valid_axes = False
                    break
                old_axes.append(norm_axis)
            if not valid_axes:
                continue

            new_axes = [int(perm_pre[axis]) for axis in old_axes]

            # 1) Bypass pre-transpose for DEQUANTIZE.
            dq_op.inputs = [pre_input]
            dq_tensor.shape = list(input_tensor.shape)
            dq_tensor.shape_signature = (
                list(input_tensor.shape_signature)
                if input_tensor.shape_signature is not None
                else list(input_tensor.shape)
            )

            # 2) Update MEAN axes to match input layout.
            axes_arr = np.asarray(new_axes, dtype=np.int32)
            axes_tensor.data = axes_arr
            axes_tensor.dtype = "INT32"
            axes_tensor.shape = [int(len(new_axes))]
            axes_tensor.shape_signature = [int(len(new_axes))]

            # 3) Recompute MEAN output tensor metadata in new layout.
            reduced_shape = list(dq_tensor.shape)
            reduced_signature = (
                list(dq_tensor.shape_signature)
                if dq_tensor.shape_signature is not None
                else list(dq_tensor.shape)
            )
            for axis in new_axes:
                if 0 <= int(axis) < len(reduced_shape):
                    reduced_shape[int(axis)] = 1
                if 0 <= int(axis) < len(reduced_signature):
                    reduced_signature[int(axis)] = 1
            mean_tensor.shape = list(reduced_shape)
            mean_tensor.shape_signature = list(reduced_signature)

            # 4) Insert a transpose before QUANTIZE to preserve QUANTIZE input layout.
            q_bridge_name = _unique_tensor_name(f"{mean_output}_for_quant")
            q_bridge_shape = _permute_shape(list(mean_tensor.shape), perm_pre)
            q_bridge_signature = _permute_shape(list(mean_tensor.shape_signature), perm_pre)
            if q_bridge_shape is None or q_bridge_signature is None:
                continue
            model_ir.tensors[q_bridge_name] = TensorIR(
                name=q_bridge_name,
                dtype=str(mean_tensor.dtype),
                shape=[int(v) for v in q_bridge_shape],
                shape_signature=[int(v) for v in q_bridge_signature],
                data=None,
            )

            perm_name = _unique_tensor_name(f"{q_bridge_name}_perm")
            perm_data = np.asarray(perm_pre, dtype=np.int32)
            model_ir.tensors[perm_name] = TensorIR(
                name=perm_name,
                dtype="INT32",
                shape=[int(perm_data.size)],
                shape_signature=[int(perm_data.size)],
                data=perm_data,
            )

            q_op.inputs = [q_bridge_name]
            model_ir.operators.insert(
                q_idx,
                OperatorIR(
                    op_type="TRANSPOSE",
                    inputs=[mean_output, perm_name],
                    outputs=[q_bridge_name],
                ),
            )

            # 5) Remove the original pre-transpose.
            remove_idx = int(pre_idx if pre_idx < q_idx else pre_idx + 1)
            del model_ir.operators[remove_idx]

            moved_bridges += 1
            changed = True
            break

        if not changed:
            break

    _prune_unused_tensors(model_ir)
    return {
        "moved_transpose_dequantize_mean_quantize_bridges": int(moved_bridges),
    }


def _optimize_nhwc_propagation_qlinear_concat_conv(model_ir: ModelIR) -> Dict[str, int]:
    """
    Propagate NHWC layout through a QLinearConcat -> QLinearConv bridge.

    Target IR shape:
      (NCHW quantized inputs)
        -> DEQUANTIZE[*]
        -> CONCATENATION(axis=1)
        -> QUANTIZE(q_out_nchw)
        -> TRANSPOSE([0,2,3,1])  # q_out_nhwc
        -> (CONV_2D|DEPTHWISE_CONV_2D)

    Rewrites:
    - Fold per-input NCHW adapters (TRANSPOSE [0,3,1,2]) into DQ inputs when safe.
    - Convert CONCAT axis from NCHW to NHWC.
    - Remove post-QUANTIZE TRANSPOSE adapters to CONV inputs.
    """
    propagated = 0
    rank4_perm_nchw_to_nhwc = [0, 2, 3, 1]
    rank4_perm_nhwc_to_nchw = [0, 3, 1, 2]
    inv_nchw_to_nhwc = _invert_perm(rank4_perm_nchw_to_nhwc)
    if inv_nchw_to_nhwc is None:
        return {"propagated_qlinear_concat_conv_nhwc_chains": 0}

    def _remap_qdim_for_permute(tensor: Optional[TensorIR], perm: List[int]) -> None:
        if tensor is None or tensor.quantization is None:
            return
        if _quant_scale_count(tensor.quantization) <= 1:
            return
        inv = _invert_perm(perm)
        if inv is None:
            return
        old_qdim = int(tensor.quantization.quantized_dimension)
        if 0 <= old_qdim < len(inv):
            tensor.quantization.quantized_dimension = int(inv[old_qdim])

    def _is_nchw_nhwc_reinterpret_safe(tensor: Optional[TensorIR]) -> bool:
        if tensor is None:
            return False
        shape = list(tensor.shape)
        if len(shape) != 4:
            return False
        signature = (
            list(tensor.shape_signature)
            if tensor.shape_signature is not None
            else list(shape)
        )
        if len(signature) != 4:
            return False
        # Reinterpretation is safe when spatial dimensions are statically singleton.
        return int(shape[2]) == 1 and int(shape[3]) == 1 and int(signature[2]) == 1 and int(signature[3]) == 1

    def _permute_tensor_shape_signature(tensor: Optional[TensorIR], perm: List[int]) -> bool:
        if tensor is None:
            return False
        new_shape = _permute_shape(list(tensor.shape), perm)
        signature_source = (
            list(tensor.shape_signature)
            if tensor.shape_signature is not None
            else list(tensor.shape)
        )
        new_signature = _permute_shape(signature_source, perm)
        if new_shape is None or new_signature is None:
            return False
        tensor.shape = [int(v) for v in new_shape]
        tensor.shape_signature = [int(v) for v in new_signature]
        _remap_qdim_for_permute(tensor, perm)
        return True

    while True:
        changed = False
        consumers = _build_tensor_consumer_map(model_ir)
        producers = _build_tensor_producer_map(model_ir)

        for concat_idx, concat_op in enumerate(model_ir.operators):
            if str(concat_op.op_type) != "CONCATENATION" or len(concat_op.inputs) == 0 or len(concat_op.outputs) != 1:
                continue
            concat_out_name = str(concat_op.outputs[0])
            if concat_out_name in model_ir.outputs:
                continue

            q_users = [int(v) for v in consumers.get(concat_out_name, [])]
            if len(q_users) != 1:
                continue
            q_idx = int(q_users[0])
            q_op = model_ir.operators[q_idx]
            if str(q_op.op_type) != "QUANTIZE" or len(q_op.inputs) != 1 or len(q_op.outputs) != 1:
                continue
            if str(q_op.inputs[0]) != concat_out_name:
                continue
            q_out_name = str(q_op.outputs[0])
            if q_out_name in model_ir.outputs:
                continue

            post_users = [int(v) for v in consumers.get(q_out_name, [])]
            if len(post_users) == 0:
                continue

            removable_post_indices: List[int] = []
            post_output_names: List[str] = []
            valid_posts = True
            for post_idx in post_users:
                post_op = model_ir.operators[post_idx]
                if str(post_op.op_type) != "TRANSPOSE" or len(post_op.inputs) < 2 or len(post_op.outputs) != 1:
                    valid_posts = False
                    break
                if str(post_op.inputs[0]) != q_out_name:
                    valid_posts = False
                    break
                perm_post = _read_transpose_perm(model_ir, post_op)
                if perm_post != rank4_perm_nchw_to_nhwc:
                    valid_posts = False
                    break
                post_output_name = str(post_op.outputs[0])
                if post_output_name in model_ir.outputs:
                    valid_posts = False
                    break
                removable_post_indices.append(int(post_idx))
                post_output_names.append(post_output_name)
            if not valid_posts:
                continue

            # Collect and rewrite concat input DQ adapters.
            removable_pre_indices: List[int] = []
            convertible = True
            for dq_input_name in list(concat_op.inputs):
                dq_idx = producers.get(str(dq_input_name), None)
                if dq_idx is None:
                    convertible = False
                    break
                dq_op = model_ir.operators[int(dq_idx)]
                if str(dq_op.op_type) != "DEQUANTIZE" or len(dq_op.inputs) != 1 or len(dq_op.outputs) != 1:
                    convertible = False
                    break
                if str(dq_op.outputs[0]) != str(dq_input_name):
                    convertible = False
                    break

                q_in_name = str(dq_op.inputs[0])
                if q_in_name in model_ir.outputs:
                    convertible = False
                    break

                q_in_tensor = model_ir.tensors.get(q_in_name, None)
                if q_in_tensor is None or len(list(q_in_tensor.shape)) != 4:
                    convertible = False
                    break

                q_in_producer_idx = producers.get(q_in_name, None)
                if q_in_producer_idx is None:
                    convertible = False
                    break
                q_in_producer = model_ir.operators[int(q_in_producer_idx)]

                # Pattern 1:
                #   q_raw --TRANSPOSE(0,3,1,2)--> q_nchw --DEQUANTIZE--> dq
                if (
                    str(q_in_producer.op_type) == "TRANSPOSE"
                    and len(q_in_producer.inputs) >= 2
                    and len(q_in_producer.outputs) == 1
                    and _read_transpose_perm(model_ir, q_in_producer) == rank4_perm_nhwc_to_nchw
                ):
                    q_raw_name = str(q_in_producer.inputs[0])
                    q_in_users = [int(v) for v in consumers.get(q_in_name, [])]
                    if len(q_in_users) != 1 or int(q_in_users[0]) != int(dq_idx):
                        convertible = False
                        break
                    if q_in_name in model_ir.outputs:
                        convertible = False
                        break

                    dq_op.inputs = [q_raw_name]
                    q_raw_tensor = model_ir.tensors.get(q_raw_name, None)
                    dq_out_tensor = model_ir.tensors.get(str(dq_op.outputs[0]), None)
                    if q_raw_tensor is None or dq_out_tensor is None:
                        convertible = False
                        break
                    dq_out_tensor.shape = [int(v) for v in list(q_raw_tensor.shape)]
                    dq_out_tensor.shape_signature = (
                        [int(v) for v in list(q_raw_tensor.shape_signature)]
                        if q_raw_tensor.shape_signature is not None
                        else [int(v) for v in list(q_raw_tensor.shape)]
                    )
                    removable_pre_indices.append(int(q_in_producer_idx))
                    continue

                # Pattern 2:
                #   x_nhwc --TRANSPOSE(0,3,1,2)--> x_nchw --QUANTIZE--> q_nchw --DEQUANTIZE--> dq
                if (
                    str(q_in_producer.op_type) == "QUANTIZE"
                    and len(q_in_producer.inputs) == 1
                    and len(q_in_producer.outputs) == 1
                    and str(q_in_producer.outputs[0]) == q_in_name
                ):
                    q_float_name = str(q_in_producer.inputs[0])
                    q_float_producer_idx = producers.get(q_float_name, None)
                    if q_float_producer_idx is not None:
                        q_float_producer = model_ir.operators[int(q_float_producer_idx)]
                        if (
                            str(q_float_producer.op_type) == "TRANSPOSE"
                            and len(q_float_producer.inputs) >= 2
                            and len(q_float_producer.outputs) == 1
                            and _read_transpose_perm(model_ir, q_float_producer) == rank4_perm_nhwc_to_nchw
                        ):
                            q_float_users = [int(v) for v in consumers.get(q_float_name, [])]
                            if len(q_float_users) != 1 or int(q_float_users[0]) != int(q_in_producer_idx):
                                convertible = False
                                break
                            if q_float_name in model_ir.outputs:
                                convertible = False
                                break

                            q_in_producer.inputs = [str(q_float_producer.inputs[0])]
                            if not _permute_tensor_shape_signature(q_in_tensor, rank4_perm_nchw_to_nhwc):
                                convertible = False
                                break

                            dq_out_tensor = model_ir.tensors.get(str(dq_op.outputs[0]), None)
                            if dq_out_tensor is None:
                                convertible = False
                                break
                            dq_out_tensor.shape = [int(v) for v in list(q_in_tensor.shape)]
                            dq_out_tensor.shape_signature = (
                                [int(v) for v in list(q_in_tensor.shape_signature)]
                                if q_in_tensor.shape_signature is not None
                                else [int(v) for v in list(q_in_tensor.shape)]
                            )
                            removable_pre_indices.append(int(q_float_producer_idx))
                            continue

                # Pattern 3:
                #   q_nchw --DEQUANTIZE--> dq, where q_nchw is effectively layout-invariant
                #   (e.g., NCHW N,C,1,1), so we can reinterpret metadata without data movement.
                q_in_users = [int(v) for v in consumers.get(q_in_name, [])]
                if len(q_in_users) == 1 and int(q_in_users[0]) == int(dq_idx):
                    if _is_nchw_nhwc_reinterpret_safe(q_in_tensor):
                        if not _permute_tensor_shape_signature(q_in_tensor, rank4_perm_nchw_to_nhwc):
                            convertible = False
                            break
                        dq_out_tensor = model_ir.tensors.get(str(dq_op.outputs[0]), None)
                        if dq_out_tensor is None:
                            convertible = False
                            break
                        dq_out_tensor.shape = [int(v) for v in list(q_in_tensor.shape)]
                        dq_out_tensor.shape_signature = (
                            [int(v) for v in list(q_in_tensor.shape_signature)]
                            if q_in_tensor.shape_signature is not None
                            else [int(v) for v in list(q_in_tensor.shape)]
                        )
                        continue

                convertible = False
                break

            if not convertible:
                continue

            concat_axis_old = int(concat_op.options.get("axis", 1))
            if concat_axis_old < 0 or concat_axis_old >= len(inv_nchw_to_nhwc):
                continue
            concat_axis_new = int(inv_nchw_to_nhwc[concat_axis_old])
            concat_op.options["axis"] = int(concat_axis_new)

            concat_input_tensors: List[Optional[TensorIR]] = [
                model_ir.tensors.get(str(input_name), None)
                for input_name in concat_op.inputs
            ]
            if any(t is None for t in concat_input_tensors):
                continue
            first_tensor = concat_input_tensors[0]
            if first_tensor is None:
                continue
            rank = len(list(first_tensor.shape))
            if rank != 4:
                continue

            concat_shape = [int(v) for v in list(first_tensor.shape)]
            concat_signature = (
                [int(v) for v in list(first_tensor.shape_signature)]
                if first_tensor.shape_signature is not None
                else [int(v) for v in list(first_tensor.shape)]
            )
            dynamic_concat_axis = False
            for tensor in concat_input_tensors[1:]:
                if tensor is None:
                    continue
                if len(list(tensor.shape)) != rank:
                    convertible = False
                    break
                concat_shape[concat_axis_new] += int(tensor.shape[concat_axis_new])
            if not convertible:
                continue
            for tensor in concat_input_tensors:
                if tensor is None:
                    continue
                sig = (
                    [int(v) for v in list(tensor.shape_signature)]
                    if tensor.shape_signature is not None
                    else [int(v) for v in list(tensor.shape)]
                )
                if int(sig[concat_axis_new]) < 0:
                    dynamic_concat_axis = True
                    break
            if dynamic_concat_axis:
                concat_signature[concat_axis_new] = -1
            else:
                concat_signature[concat_axis_new] = int(
                    sum(
                        int(
                            (
                                t.shape_signature[concat_axis_new]
                                if t is not None and t.shape_signature is not None
                                else t.shape[concat_axis_new]
                            )
                        )
                        for t in concat_input_tensors
                        if t is not None
                    )
                )

            concat_out_tensor = model_ir.tensors.get(concat_out_name, None)
            if concat_out_tensor is None:
                continue
            concat_out_tensor.shape = [int(v) for v in concat_shape]
            concat_out_tensor.shape_signature = [int(v) for v in concat_signature]

            q_out_tensor = model_ir.tensors.get(q_out_name, None)
            if q_out_tensor is None:
                continue
            q_out_tensor.shape = [int(v) for v in concat_shape]
            q_out_tensor.shape_signature = [int(v) for v in concat_signature]
            _remap_qdim_for_permute(q_out_tensor, rank4_perm_nchw_to_nhwc)

            # Remove output transpose adapters and reconnect their consumers to q_out directly.
            for post_idx in removable_post_indices:
                post_op = model_ir.operators[int(post_idx)]
                post_out_name = str(post_op.outputs[0])
                _replace_tensor_inputs(model_ir, post_out_name, q_out_name)

            remove_indices = sorted(
                set(int(v) for v in (removable_pre_indices + removable_post_indices)),
                reverse=True,
            )
            for remove_idx in remove_indices:
                del model_ir.operators[int(remove_idx)]

            propagated += 1
            changed = True
            break

        if not changed:
            break

    _prune_unused_tensors(model_ir)
    return {
        "propagated_qlinear_concat_conv_nhwc_chains": int(propagated),
    }


def _canonicalize_softmax_transpose_chains(model_ir: ModelIR) -> Dict[str, int]:
    """
    Canonicalize ONNX-driven transpose chains around SOFTMAX to expose removable pairs.

    Target template:
      T0(0,3,1,2) -> T1(0,3,2,1) -> SOFTMAX -> T2(0,3,2,1)

    Rewritten to:
      T0(0,3,1,2) -> T1(0,2,3,1) -> SOFTMAX -> T2(0,3,1,2)

    Then T0/T1 become inverse pairs and are removed by `_optimize_layout_transpose_chains`.
    """
    rewritten = 0
    rank4_perm_nhwc_to_nchw = [0, 3, 1, 2]
    rank4_perm_nchw_to_nhwc = [0, 2, 3, 1]
    rank4_perm_nchw_to_nwhc = [0, 3, 2, 1]

    def _unique_tensor_name(base: str) -> str:
        name = str(base)
        suffix = 1
        while name in model_ir.tensors:
            name = f"{base}_{suffix}"
            suffix += 1
        return name

    def _write_transpose_perm_for_op(
        op_idx: int,
        new_perm: List[int],
        consumers: Dict[str, List[int]],
    ) -> bool:
        op = model_ir.operators[int(op_idx)]
        if str(op.op_type) != "TRANSPOSE" or len(op.inputs) < 2:
            return False
        perm_name = str(op.inputs[1])
        perm_users = [int(v) for v in consumers.get(perm_name, [])]
        perm_data = np.asarray(new_perm, dtype=np.int32)

        # If perm tensor is shared, clone it to avoid side effects on other TRANSPOSE ops.
        if len(perm_users) != 1 or int(perm_users[0]) != int(op_idx):
            new_perm_name = _unique_tensor_name(f"{perm_name}_canon")
            model_ir.tensors[new_perm_name] = TensorIR(
                name=new_perm_name,
                dtype="INT32",
                shape=[int(len(new_perm))],
                shape_signature=[int(len(new_perm))],
                data=perm_data,
                is_variable=False,
                quantization=None,
            )
            op.inputs[1] = new_perm_name
            return True

        perm_tensor = model_ir.tensors.get(perm_name, None)
        if perm_tensor is None:
            return False
        perm_tensor.data = perm_data
        perm_tensor.dtype = "INT32"
        perm_tensor.shape = [int(len(new_perm))]
        perm_tensor.shape_signature = [int(len(new_perm))]
        return True

    while True:
        changed = False
        consumers = _build_tensor_consumer_map(model_ir)
        producers = _build_tensor_producer_map(model_ir)

        for softmax_idx, softmax_op in enumerate(model_ir.operators):
            if str(softmax_op.op_type) != "SOFTMAX" or len(softmax_op.inputs) != 1 or len(softmax_op.outputs) != 1:
                continue
            softmax_in = str(softmax_op.inputs[0])
            softmax_out = str(softmax_op.outputs[0])
            if softmax_in in model_ir.outputs or softmax_out in model_ir.outputs:
                continue

            pre_idx = producers.get(softmax_in, None)
            if pre_idx is None:
                continue
            pre_op = model_ir.operators[int(pre_idx)]
            if str(pre_op.op_type) != "TRANSPOSE" or len(pre_op.inputs) < 2 or len(pre_op.outputs) != 1:
                continue
            if str(pre_op.outputs[0]) != softmax_in:
                continue
            if _read_transpose_perm(model_ir, pre_op) != rank4_perm_nchw_to_nwhc:
                continue
            pre_users = [int(v) for v in consumers.get(softmax_in, [])]
            if pre_users != [int(softmax_idx)]:
                continue

            pre_input = str(pre_op.inputs[0])
            pre_prev_idx = producers.get(pre_input, None)
            if pre_prev_idx is None:
                continue
            pre_prev_op = model_ir.operators[int(pre_prev_idx)]
            if (
                str(pre_prev_op.op_type) != "TRANSPOSE"
                or len(pre_prev_op.inputs) < 2
                or len(pre_prev_op.outputs) != 1
                or str(pre_prev_op.outputs[0]) != pre_input
            ):
                continue
            if _read_transpose_perm(model_ir, pre_prev_op) != rank4_perm_nhwc_to_nchw:
                continue
            pre_prev_users = [int(v) for v in consumers.get(pre_input, [])]
            if pre_prev_users != [int(pre_idx)]:
                continue

            post_users = [int(v) for v in consumers.get(softmax_out, [])]
            if len(post_users) != 1:
                continue
            post_idx = int(post_users[0])
            post_op = model_ir.operators[int(post_idx)]
            if str(post_op.op_type) != "TRANSPOSE" or len(post_op.inputs) < 2 or len(post_op.outputs) != 1:
                continue
            if str(post_op.inputs[0]) != softmax_out:
                continue
            if _read_transpose_perm(model_ir, post_op) != rank4_perm_nchw_to_nwhc:
                continue
            if str(post_op.outputs[0]) in model_ir.outputs:
                continue

            # Avoid qdim remapping complexity for per-axis activation quantization.
            pre_in_tensor = model_ir.tensors.get(pre_input, None)
            softmax_out_tensor = model_ir.tensors.get(softmax_out, None)
            post_out_tensor = model_ir.tensors.get(str(post_op.outputs[0]), None)
            if not _all_per_tensor_quantized([pre_in_tensor, softmax_out_tensor, post_out_tensor]):
                continue

            if not _write_transpose_perm_for_op(int(pre_idx), rank4_perm_nchw_to_nhwc, consumers):
                continue
            if not _write_transpose_perm_for_op(int(post_idx), rank4_perm_nhwc_to_nchw, consumers):
                continue

            # Refresh static metadata conservatively; final reconcile pass will refine.
            if pre_in_tensor is not None:
                pre_out_tensor = model_ir.tensors.get(softmax_in, None)
                if pre_out_tensor is not None:
                    new_shape = _permute_shape(list(pre_in_tensor.shape), rank4_perm_nchw_to_nhwc)
                    sig_src = (
                        list(pre_in_tensor.shape_signature)
                        if pre_in_tensor.shape_signature is not None
                        else list(pre_in_tensor.shape)
                    )
                    new_sig = _permute_shape(sig_src, rank4_perm_nchw_to_nhwc)
                    if new_shape is not None:
                        pre_out_tensor.shape = [int(v) for v in new_shape]
                    if new_sig is not None:
                        pre_out_tensor.shape_signature = [int(v) for v in new_sig]

            if softmax_out_tensor is not None:
                post_out_tensor = model_ir.tensors.get(str(post_op.outputs[0]), None)
                if post_out_tensor is not None:
                    new_shape = _permute_shape(list(softmax_out_tensor.shape), rank4_perm_nhwc_to_nchw)
                    sig_src = (
                        list(softmax_out_tensor.shape_signature)
                        if softmax_out_tensor.shape_signature is not None
                        else list(softmax_out_tensor.shape)
                    )
                    new_sig = _permute_shape(sig_src, rank4_perm_nhwc_to_nchw)
                    if new_shape is not None:
                        post_out_tensor.shape = [int(v) for v in new_shape]
                    if new_sig is not None:
                        post_out_tensor.shape_signature = [int(v) for v in new_sig]

            rewritten += 1
            changed = True
            break

        if not changed:
            break

    _prune_unused_tensors(model_ir)
    return {"canonicalized_softmax_transpose_chains": int(rewritten)}


def _optimize_layout_transpose_chains(model_ir: ModelIR) -> Dict[str, int]:
    """
    Eliminate redundant TRANSPOSE chains introduced by channel-first/channel-last bridging.

    This pass removes:
    - identity transpose: X --Transpose(identity)--> Y
    - inverse transpose pairs with single-edge bridge:
        A --Transpose(P)--> B --Transpose(inv(P))--> C
      by directly reconnecting C consumers to A.

    The optimization is output-safe: graph outputs are never renamed.
    """
    removed_identity = 0
    removed_inverse_pairs = 0
    iterations = 0

    while True:
        iterations += 1
        changed = False
        consumers = _build_tensor_consumer_map(model_ir)

        # 1) Remove identity transpose.
        for op_idx, op in enumerate(model_ir.operators):
            if str(op.op_type) != "TRANSPOSE":
                continue
            if len(op.inputs) < 2 or len(op.outputs) != 1:
                continue
            perm = _read_transpose_perm(model_ir, op)
            if perm is None or not _is_identity_perm(perm):
                continue
            transposed_output = op.outputs[0]
            transposed_input = op.inputs[0]
            if transposed_output in model_ir.outputs:
                continue
            _replace_tensor_inputs(model_ir, transposed_output, transposed_input)
            del model_ir.operators[op_idx]
            removed_identity += 1
            changed = True
            break
        if changed:
            continue

        # 2) Remove inverse transpose pair.
        for op_idx, op in enumerate(model_ir.operators):
            if str(op.op_type) != "TRANSPOSE":
                continue
            if len(op.inputs) < 2 or len(op.outputs) != 1:
                continue
            bridge_tensor = op.outputs[0]
            bridge_users = consumers.get(bridge_tensor, [])
            if len(bridge_users) != 1:
                continue
            next_op_idx = int(bridge_users[0])
            if next_op_idx == op_idx:
                continue
            next_op = model_ir.operators[next_op_idx]
            if str(next_op.op_type) != "TRANSPOSE":
                continue
            if len(next_op.inputs) < 2 or len(next_op.outputs) != 1:
                continue
            if next_op.inputs[0] != bridge_tensor:
                continue

            perm_1 = _read_transpose_perm(model_ir, op)
            perm_2 = _read_transpose_perm(model_ir, next_op)
            if perm_1 is None or perm_2 is None:
                continue
            if not _is_inverse_perm(perm_1, perm_2):
                continue

            transpose_1_input = op.inputs[0]
            transpose_1_output = op.outputs[0]
            transpose_2_output = next_op.outputs[0]

            # Keep user-visible output names stable.
            if transpose_1_output in model_ir.outputs or transpose_2_output in model_ir.outputs:
                continue

            _replace_tensor_inputs(model_ir, transpose_2_output, transpose_1_input)
            for remove_idx in sorted([op_idx, next_op_idx], reverse=True):
                del model_ir.operators[remove_idx]
            removed_inverse_pairs += 1
            changed = True
            break

        if not changed:
            break

    _prune_unused_tensors(model_ir)
    return {
        "iterations": int(iterations),
        "removed_identity_transpose": int(removed_identity),
        "removed_inverse_transpose_pairs": int(removed_inverse_pairs),
    }


def _collect_schema_ops_for_range(
    *,
    opset_min: int = 13,
    opset_max: int = 18,
) -> List[str]:
    schema_ops = set()
    for schema in onnx.defs.get_all_schemas_with_history():
        if schema.domain != "":
            continue
        if int(schema.since_version) < int(opset_min) or int(schema.since_version) > int(opset_max):
            continue
        schema_ops.add(str(schema.name))
    return sorted(schema_ops)


def _build_schema_policy_matrix(
    *,
    schema_ops: set,
    supported_registry_ops: set,
    custom_candidate_ops: set,
) -> List[Dict[str, Any]]:
    matrix: List[Dict[str, Any]] = []
    for op in sorted(list(schema_ops)):
        if op in supported_registry_ops:
            policy = "builtin_supported"
        elif op in custom_candidate_ops:
            policy = "custom_candidate"
        else:
            policy = "explicit_error"
        matrix.append(
            {
                "onnx_op": str(op),
                "policy": str(policy),
            }
        )
    return matrix


def build_op_coverage_report(
    *,
    onnx_graph: onnx.ModelProto,
    output_file_name: str,
    opset_min: int = 13,
    opset_max: int = 18,
    conversion_error: Optional[str] = None,
    allow_custom_ops: bool = False,
    custom_op_allowlist: Optional[List[str]] = None,
    preprocess_report: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    onnx_graph = _infer_shapes_with_fallback(onnx_graph)

    shape_map, dtype_map = _extract_tensor_info(onnx_graph)
    constants: Dict[str, np.ndarray] = {}
    for ini in onnx_graph.graph.initializer:
        constants[ini.name] = np.asarray(numpy_helper.to_array(ini))
    tensor_consumer_count: Dict[str, int] = {}
    for node in onnx_graph.graph.node:
        for input_name in node.input:
            if str(input_name) == "":
                continue
            tensor_consumer_count[str(input_name)] = int(tensor_consumer_count.get(str(input_name), 0) + 1)
    graph_output_names = [str(o.name) for o in onnx_graph.graph.output]

    model_ir = ModelIR(name=output_file_name)
    ctx = LoweringContext(
        model_ir=model_ir,
        shape_map=shape_map,
        dtype_map=dtype_map,
        constants=constants,
        allow_custom_ops=allow_custom_ops,
        custom_op_allowlist=custom_op_allowlist,
        tensor_consumer_count=tensor_consumer_count,
        graph_output_names=graph_output_names,
    )
    initializer_names = {ini.name for ini in onnx_graph.graph.initializer}
    for graph_input in onnx_graph.graph.input:
        if graph_input.name in initializer_names:
            continue
        ctx.ensure_tensor(graph_input.name)
        model_ir.inputs.append(graph_input.name)
    for name, value in constants.items():
        if name not in model_ir.tensors:
            ctx.add_const_tensor(name, value)

    node_reports: List[Dict[str, Any]] = []
    unsupported_nodes: List[Dict[str, Any]] = []
    custom_lowered_nodes: List[Dict[str, Any]] = []
    graph_unique_ops: set = set()
    for node in onnx_graph.graph.node:
        node_name = node.name if node.name else node.op_type
        graph_unique_ops.add(node.op_type)
        if node.op_type == "Constant":
            value_attr = None
            for attr in node.attribute:
                if attr.name == "value":
                    value_attr = attr
                    break
            if value_attr is not None and len(node.output) > 0:
                const_array = np.asarray(numpy_helper.to_array(value_attr.t))
                out_name = node.output[0]
                if out_name in model_ir.tensors:
                    t = model_ir.tensors[out_name]
                    t.data = const_array
                    t.dtype = tflite_dtype_from_numpy(const_array.dtype)
                    t.shape, t.shape_signature = normalize_onnx_shape(list(const_array.shape))
                    constants[out_name] = const_array
                else:
                    _added = ctx.add_const_tensor(out_name, const_array)
                    if _added != out_name:
                        model_ir.tensors[out_name] = model_ir.tensors.pop(_added)
                        model_ir.tensors[out_name].name = out_name
                        constants[out_name] = constants.pop(_added)
            node_reports.append(
                {
                    "node_name": node_name,
                    "onnx_op": "Constant",
                    "supported": True,
                    "reason_code": "handled_inline",
                    "message": "Constant node is handled inline in lowering pass.",
                }
            )
            continue

        wrapped = _NodeWrap(node)
        try:
            resolution = resolve_node_dispatch(wrapped, ctx)
            dispatch_mode = str(resolution.dispatch_mode)
            reason_code = resolution.reason_code
            message = resolution.message
            node_reports.append(
                {
                    "node_name": node_name,
                    "onnx_op": node.op_type,
                    "supported": True,
                    "dispatch_mode": dispatch_mode,
                    "reason_code": reason_code,
                    "message": message,
                }
            )
            if dispatch_mode == "custom":
                custom_lowered_nodes.append(
                    {
                        "node_name": node_name,
                        "onnx_op": node.op_type,
                        "reason_code": reason_code,
                        "message": message,
                    }
                )
        except NodeValidationError as ve:
            issue = ve.to_dict()
            issue["supported"] = False
            issue["dispatch_mode"] = "unsupported"
            node_reports.append(issue)
            unsupported_nodes.append(issue)
        except Exception as ex:
            issue = {
                "node_name": node_name,
                "onnx_op": node.op_type,
                "supported": False,
                "dispatch_mode": "unsupported",
                "reason_code": "validation_exception",
                "message": str(ex),
            }
            node_reports.append(issue)
            unsupported_nodes.append(issue)

    supported_registry_ops = set(get_supported_onnx_ops())
    custom_candidate_ops = set(get_custom_op_candidate_ops())
    schema_ops = set(_collect_schema_ops_for_range(opset_min=opset_min, opset_max=opset_max))
    schema_policy_matrix = _build_schema_policy_matrix(
        schema_ops=schema_ops,
        supported_registry_ops=supported_registry_ops,
        custom_candidate_ops=custom_candidate_ops,
    )
    schema_policy_counts: Dict[str, int] = {
        "builtin_supported": 0,
        "custom_candidate": 0,
        "explicit_error": 0,
    }
    for item in schema_policy_matrix:
        p = str(item["policy"])
        schema_policy_counts[p] = int(schema_policy_counts.get(p, 0) + 1)
    graph_ops = sorted(graph_unique_ops)
    supported_graph_ops = sorted(
        list({r["onnx_op"] for r in node_reports if r["supported"] is True})
    )
    unsupported_graph_ops = sorted(
        list({r["onnx_op"] for r in node_reports if r["supported"] is False})
    )
    reason_counts: Dict[str, int] = {}
    for issue in unsupported_nodes:
        reason = str(issue.get("reason_code", "unknown"))
        reason_counts[reason] = int(reason_counts.get(reason, 0) + 1)

    normalized_allowlist = (
        [str(v).strip() for v in custom_op_allowlist if str(v).strip() != ""]
        if custom_op_allowlist is not None
        else None
    )
    allowlist_set = (
        {str(v) for v in normalized_allowlist}
        if normalized_allowlist is not None
        else set()
    )
    candidate_ops_now_builtin_supported = sorted(
        list(custom_candidate_ops & supported_registry_ops)
    )
    allowlist_builtin_supported_ops = sorted(
        list(allowlist_set & supported_registry_ops)
    )
    allowlist_custom_candidate_ops = sorted(
        list(allowlist_set & custom_candidate_ops)
    )
    allowlist_unknown_ops = sorted(
        list(
            allowlist_set
            - supported_registry_ops
            - custom_candidate_ops
            - schema_ops
        )
    )

    total_nodes = len(node_reports)
    supported_nodes = len([r for r in node_reports if r["supported"] is True])
    coverage = float(supported_nodes / total_nodes) if total_nodes > 0 else 1.0
    report: Dict[str, Any] = {
        "schema_version": 1,
        "target_opset_min": int(opset_min),
        "target_opset_max": int(opset_max),
        "supported_onnx_ops_registry": sorted(list(supported_registry_ops)),
        "custom_op_candidate_ops": sorted(list(custom_candidate_ops)),
        "schema_onnx_ops_target_range": sorted(list(schema_ops)),
        "schema_policy_matrix": schema_policy_matrix,
        "schema_policy_counts": schema_policy_counts,
        "schema_unresolved_ops": [],
        "registry_missing_from_schema_range": sorted(list(schema_ops - supported_registry_ops)),
        "registry_extra_outside_schema_range": sorted(list(supported_registry_ops - schema_ops)),
        "graph_ops": graph_ops,
        "graph_supported_ops": supported_graph_ops,
        "graph_unsupported_ops": unsupported_graph_ops,
        "graph_custom_ops": sorted(list({r["onnx_op"] for r in custom_lowered_nodes})),
        "graph_node_reports": node_reports,
        "custom_lowered_nodes": custom_lowered_nodes,
        "unsupported_nodes": unsupported_nodes,
        "unsupported_reason_counts": reason_counts,
        "graph_summary": {
            "total_nodes": int(total_nodes),
            "supported_nodes": int(supported_nodes),
            "custom_lowered_nodes": int(len(custom_lowered_nodes)),
            "unsupported_nodes": int(total_nodes - supported_nodes),
            "coverage_ratio": float(coverage),
        },
        "custom_op_policy": {
            "allow_custom_ops": bool(allow_custom_ops),
            "custom_op_allowlist": (
                [str(v) for v in normalized_allowlist]
                if normalized_allowlist is not None
                else None
            ),
            "candidate_count": int(len(custom_candidate_ops)),
            "candidate_count_excluding_builtin_supported": int(
                len(custom_candidate_ops - supported_registry_ops)
            ),
            "candidate_ops_now_builtin_supported": candidate_ops_now_builtin_supported,
            "allowlist_builtin_supported_ops": allowlist_builtin_supported_ops,
            "allowlist_custom_candidate_ops": allowlist_custom_candidate_ops,
            "allowlist_unknown_ops": allowlist_unknown_ops,
        },
        "preprocess_report": (
            dict(preprocess_report)
            if isinstance(preprocess_report, dict)
            else {
                "schema_version": 1,
                "pipeline_version": 1,
                "registered_rule_ids": [],
                "enabled_rule_ids": [],
                "applied_rules": [],
                "summary": {
                    "registered_rule_count": 0,
                    "enabled_rule_count": 0,
                    "executed_rule_count": 0,
                    "changed_rule_count": 0,
                    "total_matched_nodes": 0,
                    "total_rewritten_nodes": 0,
                },
            }
        ),
        "conversion_error": conversion_error,
    }
    return report


def write_op_coverage_report(
    *,
    report: Dict[str, Any],
    output_report_path: str,
) -> str:
    import json
    import os

    os.makedirs(os.path.dirname(output_report_path) or ".", exist_ok=True)
    with open(output_report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    return output_report_path


def lower_onnx_to_ir(
    onnx_graph: onnx.ModelProto,
    output_file_name: str,
    allow_custom_ops: bool = False,
    custom_op_allowlist: Optional[List[str]] = None,
    optimize_layout_transpose_chains: bool = True,
    transpose_inputs_to_nhwc: bool = False,
    keep_ncw_or_nchw_or_ncdhw_input_names: Optional[List[str]] = None,
    keep_nwc_or_nhwc_or_ndhwc_input_names: Optional[List[str]] = None,
    keep_shape_absolutely_input_names: Optional[List[str]] = None,
) -> ModelIR:
    onnx_graph = _infer_shapes_with_fallback(onnx_graph)

    shape_map, dtype_map = _extract_tensor_info(onnx_graph)
    constants: Dict[str, np.ndarray] = {}
    for ini in onnx_graph.graph.initializer:
        constants[ini.name] = np.asarray(numpy_helper.to_array(ini))
    tensor_consumer_count: Dict[str, int] = {}
    for node in onnx_graph.graph.node:
        for input_name in node.input:
            if str(input_name) == "":
                continue
            tensor_consumer_count[str(input_name)] = int(
                tensor_consumer_count.get(str(input_name), 0) + 1
            )
    graph_output_names = [str(o.name) for o in onnx_graph.graph.output]

    model_ir = ModelIR(name=output_file_name)
    ctx = LoweringContext(
        model_ir=model_ir,
        shape_map=shape_map,
        dtype_map=dtype_map,
        constants=constants,
        allow_custom_ops=allow_custom_ops,
        custom_op_allowlist=custom_op_allowlist,
        tensor_consumer_count=tensor_consumer_count,
        graph_output_names=graph_output_names,
    )

    keep_ncw_input_names = {
        str(v) for v in (keep_ncw_or_nchw_or_ncdhw_input_names or [])
    }
    keep_nwc_input_names = {
        str(v) for v in (keep_nwc_or_nhwc_or_ndhwc_input_names or [])
    }
    keep_shape_abs_input_names = {
        str(v) for v in (keep_shape_absolutely_input_names or [])
    }
    input_name_remap: Dict[str, str] = {}

    # Inputs
    initializer_names = {ini.name for ini in onnx_graph.graph.initializer}
    for graph_input in onnx_graph.graph.input:
        if graph_input.name in initializer_names:
            continue
        input_name = str(graph_input.name)
        ctx.ensure_tensor(input_name)
        input_tensor = model_ir.tensors[input_name]
        model_ir.inputs.append(input_name)

        if not transpose_inputs_to_nhwc:
            continue

        input_rank = len(list(input_tensor.shape))
        if input_rank not in [3, 4, 5]:
            continue

        keep_shape_abs = input_name in keep_shape_abs_input_names
        keep_ncw = input_name in keep_ncw_input_names
        keep_nwc = (
            input_name in keep_nwc_input_names
            and input_name not in keep_ncw_input_names
        )
        if keep_shape_abs or keep_ncw or keep_nwc:
            continue

        original_shape = list(input_tensor.shape)
        original_signature = (
            list(input_tensor.shape_signature)
            if input_tensor.shape_signature is not None
            else list(input_tensor.shape)
        )
        if input_rank == 3:
            perm_internal_to_external = [0, 2, 1]
            perm_external_to_internal = [0, 2, 1]
        elif input_rank == 4:
            perm_internal_to_external = [0, 2, 3, 1]
            perm_external_to_internal = [0, 3, 1, 2]
        else:
            perm_internal_to_external = [0, 2, 3, 4, 1]
            perm_external_to_internal = [0, 4, 1, 2, 3]

        external_shape = _permute_shape(original_shape, perm_internal_to_external)
        external_signature = _permute_shape(
            original_signature, perm_internal_to_external
        )
        if external_shape is None or external_signature is None:
            continue

        input_tensor.shape = list(external_shape)
        input_tensor.shape_signature = list(external_signature)

        internal_input_name = ctx.add_intermediate_tensor(
            f"{input_name}_onnx_ncx_internal",
            dtype=str(input_tensor.dtype),
            shape=original_shape,
        )
        internal_tensor = model_ir.tensors[internal_input_name]
        internal_tensor.shape_signature = list(original_signature)
        internal_tensor.quantization = _clone_quantization(input_tensor.quantization)

        perm_name = ctx.add_const_tensor(
            f"{internal_input_name}_perm",
            np.asarray(perm_external_to_internal, dtype=np.int32),
        )
        ctx.add_operator(
            OperatorIR(
                op_type="TRANSPOSE",
                inputs=[input_name, perm_name],
                outputs=[internal_input_name],
            )
        )
        input_name_remap[input_name] = internal_input_name

    # Initializers as tensors
    for name, value in constants.items():
        if name not in model_ir.tensors:
            ctx.add_const_tensor(name, value)

    # Nodes
    for node in onnx_graph.graph.node:
        if node.op_type == "Constant":
            output_name = node.output[0]
            value_attr = None
            for attr in node.attribute:
                if attr.name == "value":
                    value_attr = attr
                    break
            if value_attr is None:
                raise NotImplementedError(f"Constant node without value is not supported. op={node.name}")
            const_array = np.asarray(numpy_helper.to_array(value_attr.t))
            if output_name in model_ir.tensors:
                # Replace existing placeholder tensor data.
                t = model_ir.tensors[output_name]
                t.data = const_array
                t.dtype = tflite_dtype_from_numpy(const_array.dtype)
                t.shape, t.shape_signature = normalize_onnx_shape(list(const_array.shape))
                constants[output_name] = const_array
            else:
                name = ctx.add_const_tensor(output_name, const_array)
                if name != output_name:
                    # keep graph output name stable if collision happened.
                    model_ir.tensors[output_name] = model_ir.tensors.pop(name)
                    model_ir.tensors[output_name].name = output_name
                    constants[output_name] = constants.pop(name)
            continue

        wrapped = _NodeWrap(
            node,
            input_name_remap=input_name_remap,
        )
        try:
            dispatch_node(wrapped, ctx)
        except NodeValidationError as ve:
            raise NotImplementedError(
                f"flatbuffer_direct validation failed: "
                f"op={ve.node_op} node={ve.node_name} "
                f"reason_code={ve.reason_code} message={ve.message}"
            ) from ve

    # Outputs
    for graph_output in onnx_graph.graph.output:
        ctx.ensure_tensor(graph_output.name)
        model_ir.outputs.append(graph_output.name)

    if optimize_layout_transpose_chains:
        # NOTE:
        # Binary/fanout transpose rewrites are currently disabled because they can
        # invalidate tensor shape/layout assumptions in multi-branch int8 graphs.
        enable_transpose_binary_bridge_optimizations = False
        enable_duplicate_transpose_fanout_optimizations = False

        _optimize_layout_transpose_chains(model_ir)
        _optimize_transpose_quant_dequant_bridges(model_ir)
        _optimize_leading_input_transpose_passthrough_chains(model_ir)
        _optimize_transpose_dequant_mul_add_prelu_quantize_bridges(model_ir)
        _optimize_transpose_dequant_prelu_quantize_bridges(model_ir)
        _optimize_transpose_dequant_prelu_transpose_bridges(model_ir)
        _optimize_dequant_prelu_quantize_chains(model_ir)
        _optimize_dequant_prelu_depthwise_quantize_chains(model_ir)
        _optimize_dequant_reshape_quantize_chains(model_ir)
        _optimize_dequant_softmax_quantize_chains(model_ir)
        _canonicalize_softmax_transpose_chains(model_ir)
        _optimize_transpose_binary_symmetric_legacy_only_bridges_safe(model_ir)
        _optimize_transpose_binary_single_post_bridges_safe(model_ir)
        _optimize_transpose_binary_mixed_fanout_bridges_safe(model_ir)
        _optimize_transpose_binary_asymmetric_fanout_bridges(model_ir)
        _optimize_transpose_binary_full_post_fanout_bridges(model_ir)
        if enable_transpose_binary_bridge_optimizations:
            _optimize_transpose_binary_bridges(model_ir)
        if enable_duplicate_transpose_fanout_optimizations:
            _optimize_duplicate_transpose_fanout(model_ir)
        # Binary bridge rewrites can introduce new transpose-(q|dq)-transpose patterns.
        _optimize_transpose_quant_dequant_bridges(model_ir)
        _optimize_leading_input_transpose_passthrough_chains(model_ir)
        _optimize_transpose_dequant_mul_add_prelu_quantize_bridges(model_ir)
        if enable_duplicate_transpose_fanout_optimizations:
            _optimize_duplicate_transpose_fanout(model_ir)
        _optimize_transpose_dequant_prelu_quantize_bridges(model_ir)
        _optimize_transpose_dequant_prelu_transpose_bridges(model_ir)
        _optimize_dequant_prelu_quantize_chains(model_ir)
        _optimize_dequant_prelu_depthwise_quantize_chains(model_ir)
        _optimize_dequant_reshape_quantize_chains(model_ir)
        _optimize_dequant_softmax_quantize_chains(model_ir)
        _canonicalize_softmax_transpose_chains(model_ir)
        _optimize_transpose_binary_symmetric_legacy_only_bridges_safe(model_ir)
        _optimize_transpose_binary_single_post_bridges_safe(model_ir)
        _optimize_transpose_binary_mixed_fanout_bridges_safe(model_ir)
        _optimize_transpose_binary_asymmetric_fanout_bridges(model_ir)
        _optimize_transpose_binary_full_post_fanout_bridges(model_ir)
        _optimize_transpose_dequantize_mean_quantize_bridges(model_ir)
        _optimize_nhwc_propagation_qlinear_concat_conv(model_ir)
        _optimize_concat_pre_quantize_dequantize(model_ir)
        _optimize_transpose_quant_dequant_bridges(model_ir)
        _optimize_leading_input_transpose_passthrough_chains(model_ir)
        _optimize_transpose_dequant_mul_add_prelu_quantize_bridges(model_ir)
        if enable_duplicate_transpose_fanout_optimizations:
            _optimize_duplicate_transpose_fanout(model_ir)
        _optimize_transpose_dequant_prelu_quantize_bridges(model_ir)
        _optimize_transpose_dequant_prelu_transpose_bridges(model_ir)
        _optimize_dequant_prelu_quantize_chains(model_ir)
        _optimize_dequant_prelu_depthwise_quantize_chains(model_ir)
        _optimize_dequant_reshape_quantize_chains(model_ir)
        _optimize_dequant_softmax_quantize_chains(model_ir)
        _canonicalize_softmax_transpose_chains(model_ir)
        _optimize_layout_transpose_chains(model_ir)
        _optimize_transpose_binary_symmetric_legacy_only_bridges_safe(model_ir)
        _optimize_transpose_binary_single_post_bridges_safe(model_ir)
        _optimize_transpose_binary_mixed_fanout_bridges_safe(model_ir)
        _optimize_transpose_binary_asymmetric_fanout_bridges(model_ir)
        _optimize_transpose_binary_full_post_fanout_bridges(model_ir)
    _sanitize_terminal_transpose_before_dequantize(model_ir)
    _optimize_terminal_quantize_dequantize(model_ir)
    _resolve_dynamic_reshape_shapes(model_ir)
    _prune_dead_operators(model_ir)
    _reconcile_static_tensor_shapes(model_ir)
    if optimize_layout_transpose_chains:
        # Final recovery sweep:
        # some transpose-binary patterns become shape-safe only after static
        # metadata reconciliation, so run bridge passes once more.
        _optimize_nhwc_propagation_qlinear_concat_conv(model_ir)
        _optimize_concat_pre_quantize_dequantize(model_ir)
        _optimize_transpose_quant_dequant_bridges(model_ir)
        _optimize_leading_input_transpose_passthrough_chains(model_ir)
        _optimize_transpose_dequant_mul_add_prelu_quantize_bridges(model_ir)
        _optimize_dequant_softmax_quantize_chains(model_ir)
        _canonicalize_softmax_transpose_chains(model_ir)
        _optimize_transpose_binary_symmetric_legacy_only_bridges_safe(model_ir)
        _optimize_transpose_binary_single_post_bridges_safe(model_ir)
        _optimize_transpose_binary_mixed_fanout_bridges_safe(model_ir)
        _optimize_transpose_binary_asymmetric_fanout_bridges(model_ir)
        _optimize_transpose_binary_full_post_fanout_bridges(model_ir)
        _optimize_layout_transpose_chains(model_ir)
        _prune_dead_operators(model_ir)
        _reconcile_static_tensor_shapes(model_ir)
    # Recovery sweeps above can re-introduce terminal TRANSPOSE->DEQUANTIZE.
    # Run terminal sanitizers once more at the very end.
    _sanitize_terminal_transpose_before_dequantize(model_ir)
    _optimize_terminal_quantize_dequantize(model_ir)
    _prune_dead_operators(model_ir)
    _reconcile_static_tensor_shapes(model_ir)

    return model_ir
