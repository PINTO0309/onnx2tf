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
    try:
        onnx_graph = onnx.shape_inference.infer_shapes(onnx_graph)
    except Exception:
        pass

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
    try:
        onnx_graph = onnx.shape_inference.infer_shapes(onnx_graph)
    except Exception:
        pass

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
        _optimize_layout_transpose_chains(model_ir)
        _optimize_transpose_quant_dequant_bridges(model_ir)
        _optimize_transpose_binary_bridges(model_ir)
        # Binary bridge rewrites can introduce new transpose-(q|dq)-transpose patterns.
        _optimize_transpose_quant_dequant_bridges(model_ir)
        _optimize_transpose_dequantize_mean_quantize_bridges(model_ir)
        _optimize_nhwc_propagation_qlinear_concat_conv(model_ir)
        _optimize_concat_pre_quantize_dequantize(model_ir)
        _optimize_transpose_quant_dequant_bridges(model_ir)
        _optimize_layout_transpose_chains(model_ir)
    _sanitize_terminal_transpose_before_dequantize(model_ir)
    _optimize_terminal_quantize_dequantize(model_ir)
    _resolve_dynamic_reshape_shapes(model_ir)
    _prune_dead_operators(model_ir)

    return model_ir
