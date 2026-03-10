from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from typing import Any, Callable, Dict, List, Optional, Sequence, Set

import numpy as np
import tensorflow as tf

from onnx2tf.tflite_builder.ir import ModelIR, OperatorIR, TensorIR


_TF_DTYPE_BY_TFLITE_DTYPE: Dict[str, tf.dtypes.DType] = {
    "BOOL": tf.bool,
    "INT8": tf.int8,
    "INT16": tf.int16,
    "INT32": tf.int32,
    "INT64": tf.int64,
    "UINT8": tf.uint8,
    "UINT16": tf.uint16,
    "UINT32": tf.uint32,
    "UINT64": tf.uint64,
    "FLOAT16": tf.float16,
    "FLOAT32": tf.float32,
    "FLOAT64": tf.float64,
    "STRING": tf.string,
}


_UNARY_KERNELS: Dict[str, Callable[[tf.Tensor], tf.Tensor]] = {
    "ABS": tf.math.abs,
    "CEIL": tf.math.ceil,
    "COS": tf.math.cos,
    "ELU": tf.nn.elu,
    "EXP": tf.math.exp,
    "FLOOR": tf.math.floor,
    "LOG": tf.math.log,
    "NEG": tf.math.negative,
    "RELU": tf.nn.relu,
    "RELU6": tf.nn.relu6,
    "LOGISTIC": tf.math.sigmoid,
    "SIGN": tf.math.sign,
    "SIN": tf.math.sin,
    "SQRT": tf.math.sqrt,
    "TANH": tf.math.tanh,
}


_BINARY_KERNELS: Dict[str, Callable[[tf.Tensor, tf.Tensor], tf.Tensor]] = {
    "ADD": tf.math.add,
    "ATAN2": tf.math.atan2,
    "SUB": tf.math.subtract,
    "MUL": tf.math.multiply,
    "DIV": tf.math.divide,
    "FLOOR_MOD": tf.math.floormod,
    "MAXIMUM": tf.math.maximum,
    "MINIMUM": tf.math.minimum,
    "POW": tf.math.pow,
    "RIGHT_SHIFT": tf.bitwise.right_shift,
    "EQUAL": tf.math.equal,
    "NOT_EQUAL": tf.math.not_equal,
    "GREATER": tf.math.greater,
    "GREATER_EQUAL": tf.math.greater_equal,
    "LESS": tf.math.less,
    "LESS_EQUAL": tf.math.less_equal,
    "LOGICAL_AND": tf.math.logical_and,
    "LOGICAL_OR": tf.math.logical_or,
}


class ModelIRSavedModelExportError(RuntimeError):
    pass


@dataclass(frozen=True)
class _CustomOpSummary:
    custom_code: str
    onnx_op: str
    onnx_node_name: str


def _normalized_shape_signature(tensor: TensorIR) -> List[Optional[int]]:
    signature = (
        list(tensor.shape_signature)
        if tensor.shape_signature is not None
        else list(tensor.shape)
    )
    normalized: List[Optional[int]] = []
    for dim in signature:
        d = int(dim)
        normalized.append(None if d < 0 else d)
    return normalized


def _tf_dtype_from_tensor_ir(tensor: TensorIR) -> tf.dtypes.DType:
    key = str(tensor.dtype).upper()
    if key not in _TF_DTYPE_BY_TFLITE_DTYPE:
        raise ModelIRSavedModelExportError(
            f"Unsupported TensorIR dtype for SavedModel export: dtype={tensor.dtype} tensor={tensor.name}"
        )
    return _TF_DTYPE_BY_TFLITE_DTYPE[key]


def _numpy_dtype_from_tensor_ir(tensor: TensorIR) -> np.dtype:
    return np.dtype(_tf_dtype_from_tensor_ir(tensor).as_numpy_dtype)


def _discover_builder_op_types() -> Set[str]:
    op_types: Set[str] = set()
    op_builders_dir = Path(__file__).resolve().parent / "op_builders"
    if not op_builders_dir.exists():
        return op_types
    single_pattern = re.compile(r'OperatorIR\(\s*op_type="([A-Z0-9_]+)"')
    conditional_pattern = re.compile(
        r'OperatorIR\(\s*op_type="([A-Z0-9_]+)"\s*if\s+.+?\s+else\s+"([A-Z0-9_]+)"',
        re.DOTALL,
    )
    for py_file in sorted(op_builders_dir.glob("*.py")):
        try:
            text = py_file.read_text(encoding="utf-8")
        except Exception:
            continue
        for match in single_pattern.finditer(text):
            op_types.add(str(match.group(1)))
        for match in conditional_pattern.finditer(text):
            op_types.add(str(match.group(1)))
            op_types.add(str(match.group(2)))
    return op_types


KNOWN_MODEL_IR_OP_TYPES: Set[str] = _discover_builder_op_types()


def get_known_model_ir_op_types() -> Set[str]:
    return set(KNOWN_MODEL_IR_OP_TYPES)


def _collect_all_ops(model_ir: ModelIR) -> List[OperatorIR]:
    all_ops: List[OperatorIR] = []
    all_ops.extend(list(model_ir.operators))
    for subgraph in model_ir.subgraphs:
        all_ops.extend(_collect_all_ops(subgraph))
    return all_ops


def _collect_custom_ops(model_ir: ModelIR) -> List[_CustomOpSummary]:
    summaries: List[_CustomOpSummary] = []
    seen = set()
    for op in _collect_all_ops(model_ir):
        if str(op.op_type) != "CUSTOM":
            continue
        options = op.options if isinstance(op.options, dict) else {}
        summary = _CustomOpSummary(
            custom_code=str(options.get("customCode", "CUSTOM")).strip() or "CUSTOM",
            onnx_op=str(options.get("onnxOp", "")).strip(),
            onnx_node_name=str(options.get("onnxNodeName", "")).strip(),
        )
        key = (summary.custom_code, summary.onnx_op, summary.onnx_node_name)
        if key in seen:
            continue
        seen.add(key)
        summaries.append(summary)
    summaries.sort(key=lambda v: (v.custom_code, v.onnx_op, v.onnx_node_name))
    return summaries


def _collect_model_op_types(model_ir: ModelIR) -> Set[str]:
    ops: Set[str] = set()
    for op in _collect_all_ops(model_ir):
        ops.add(str(op.op_type))
    return ops


def _ensure_no_custom_ops(model_ir: ModelIR) -> None:
    custom_ops = _collect_custom_ops(model_ir)
    if len(custom_ops) == 0:
        return
    details = ", ".join(
        [
            (
                f"customCode={v.custom_code} "
                f"onnxOp={v.onnx_op or '-'} "
                f"onnxNodeName={v.onnx_node_name or '-'}"
            )
            for v in custom_ops
        ]
    )
    raise ModelIRSavedModelExportError(
        "flatbuffer_direct_output_saved_model does not support CUSTOM ops. "
        f"Found {len(custom_ops)} CUSTOM node(s): {details}"
    )


def _apply_fused_activation(x: tf.Tensor, fused: str) -> tf.Tensor:
    key = str(fused).upper()
    if key in {"", "NONE"}:
        return x
    if key == "RELU":
        return tf.nn.relu(x)
    if key == "RELU6":
        return tf.nn.relu6(x)
    if key == "RELU_N1_TO_1":
        return tf.clip_by_value(x, -1.0, 1.0)
    if key == "RELU_0_TO_1":
        return tf.clip_by_value(x, 0.0, 1.0)
    if key == "TANH":
        return tf.math.tanh(x)
    return x


def _apply_activation_by_name(x: tf.Tensor, activation: str) -> tf.Tensor:
    key = str(activation).upper()
    if key in {"", "NONE", "TANH"}:
        return tf.math.tanh(x) if key == "TANH" else x
    if key == "RELU":
        return tf.nn.relu(x)
    if key == "RELU6":
        return tf.nn.relu6(x)
    if key == "SIGMOID":
        return tf.math.sigmoid(x)
    return x


def _parse_bool_option(value: Any, default: bool = False) -> bool:
    if value is None:
        return bool(default)
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, np.integer)):
        return bool(int(value))
    if isinstance(value, str):
        key = value.strip().lower()
        if key in {"1", "true", "yes", "on"}:
            return True
        if key in {"0", "false", "no", "off", ""}:
            return False
    return bool(value)


class _GraphExecutor:
    def __init__(
        self,
        *,
        model_ir: ModelIR,
        root_constants: Dict[str, tf.Tensor],
        kernels: Dict[str, Callable[["_GraphExecutor", OperatorIR, Dict[str, tf.Tensor]], None]],
    ) -> None:
        self.model_ir = model_ir
        self._root_constants = root_constants
        self._kernels = kernels
        self._subgraph_executors: Dict[int, "_GraphExecutor"] = {}
        self._local_constants: Dict[str, tf.Tensor] = {}
        for tensor_name, tensor in self.model_ir.tensors.items():
            if tensor.data is None:
                continue
            if tensor_name in self._root_constants:
                self._local_constants[tensor_name] = self._root_constants[tensor_name]
                continue
            np_dtype = _numpy_dtype_from_tensor_ir(tensor)
            self._local_constants[tensor_name] = tf.constant(
                np.asarray(tensor.data).astype(np_dtype, copy=False),
                dtype=_tf_dtype_from_tensor_ir(tensor),
                name=str(tensor_name),
            )

    def _resolve_tensor(self, tensor_name: str, env: Dict[str, tf.Tensor]) -> tf.Tensor:
        key = str(tensor_name)
        if key == "":
            raise ModelIRSavedModelExportError("Empty tensor name cannot be resolved.")
        if key in env:
            return env[key]
        if key in self._local_constants:
            tensor = self._local_constants[key]
            env[key] = tensor
            return tensor
        if key in self._root_constants:
            tensor = self._root_constants[key]
            env[key] = tensor
            return tensor
        tensor_meta = self.model_ir.tensors.get(key, None)
        if tensor_meta is not None and tensor_meta.data is not None:
            np_dtype = _numpy_dtype_from_tensor_ir(tensor_meta)
            tensor = tf.constant(
                np.asarray(tensor_meta.data).astype(np_dtype, copy=False),
                dtype=_tf_dtype_from_tensor_ir(tensor_meta),
                name=key,
            )
            env[key] = tensor
            return tensor
        raise ModelIRSavedModelExportError(
            f"Tensor is unbound during SavedModel export: tensor={key}"
        )

    def _resolve_optional_tensor(
        self,
        tensor_name: str,
        env: Dict[str, tf.Tensor],
    ) -> Optional[tf.Tensor]:
        key = str(tensor_name)
        if key == "":
            return None
        return self._resolve_tensor(key, env)

    def _assign_outputs(
        self,
        op: OperatorIR,
        values: Sequence[Optional[tf.Tensor]],
        env: Dict[str, tf.Tensor],
    ) -> None:
        if len(op.outputs) != len(values):
            raise ModelIRSavedModelExportError(
                f"Output arity mismatch: op={op.op_type} expected={len(op.outputs)} got={len(values)}"
            )
        for out_name, value in zip(op.outputs, values):
            key = str(out_name)
            if key == "" or value is None:
                continue
            env[key] = value

    def run(
        self,
        *,
        inputs: Dict[str, tf.Tensor],
        requested_outputs: Optional[List[str]] = None,
    ) -> Dict[str, tf.Tensor]:
        env: Dict[str, tf.Tensor] = dict(inputs)
        for op in self.model_ir.operators:
            op_type = str(op.op_type)
            if op_type not in self._kernels:
                raise ModelIRSavedModelExportError(
                    f"SavedModel kernel is not registered for op_type={op_type}"
                )
            self._kernels[op_type](self, op, env)
        output_names = (
            list(requested_outputs)
            if requested_outputs is not None
            else list(self.model_ir.outputs)
        )
        outputs: Dict[str, tf.Tensor] = {}
        for output_name in output_names:
            key = str(output_name)
            outputs[key] = self._resolve_tensor(key, env)
        return outputs

    def _get_subgraph_executor(self, subgraph_index_1based: int) -> "_GraphExecutor":
        index = int(subgraph_index_1based) - 1
        if index < 0 or index >= len(self.model_ir.subgraphs):
            raise ModelIRSavedModelExportError(
                f"Subgraph index is out of range: index={subgraph_index_1based}"
            )
        if index not in self._subgraph_executors:
            self._subgraph_executors[index] = _GraphExecutor(
                model_ir=self.model_ir.subgraphs[index],
                root_constants=self._root_constants,
                kernels=self._kernels,
            )
        return self._subgraph_executors[index]


def _kernel_unary(
    executor: _GraphExecutor,
    op: OperatorIR,
    env: Dict[str, tf.Tensor],
) -> None:
    fn = _UNARY_KERNELS[str(op.op_type)]
    x = executor._resolve_tensor(op.inputs[0], env)
    y = fn(x)
    executor._assign_outputs(op, [y], env)


def _kernel_binary(
    executor: _GraphExecutor,
    op: OperatorIR,
    env: Dict[str, tf.Tensor],
) -> None:
    fn = _BINARY_KERNELS[str(op.op_type)]
    x = executor._resolve_tensor(op.inputs[0], env)
    y = executor._resolve_tensor(op.inputs[1], env)
    z = fn(x, y)
    z = _apply_fused_activation(
        z,
        str(op.options.get("fusedActivationFunction", "NONE")),
    )
    executor._assign_outputs(op, [z], env)


def _kernel_logical_not(
    executor: _GraphExecutor,
    op: OperatorIR,
    env: Dict[str, tf.Tensor],
) -> None:
    x = executor._resolve_tensor(op.inputs[0], env)
    executor._assign_outputs(op, [tf.math.logical_not(x)], env)


def _kernel_cast(
    executor: _GraphExecutor,
    op: OperatorIR,
    env: Dict[str, tf.Tensor],
) -> None:
    x = executor._resolve_tensor(op.inputs[0], env)
    out_dtype_name = str(op.options.get("outDataType", "")).upper()
    if out_dtype_name == "":
        out_tensor = executor.model_ir.tensors.get(str(op.outputs[0]), None)
        if out_tensor is None:
            raise ModelIRSavedModelExportError(
                f"Cannot infer CAST output dtype. op={op.op_type} output={op.outputs[0]}"
            )
        out_dtype = _tf_dtype_from_tensor_ir(out_tensor)
    else:
        if out_dtype_name not in _TF_DTYPE_BY_TFLITE_DTYPE:
            raise ModelIRSavedModelExportError(
                f"Unsupported CAST output dtype: dtype={out_dtype_name}"
            )
        out_dtype = _TF_DTYPE_BY_TFLITE_DTYPE[out_dtype_name]
    executor._assign_outputs(op, [tf.cast(x, out_dtype)], env)


def _kernel_reshape(
    executor: _GraphExecutor,
    op: OperatorIR,
    env: Dict[str, tf.Tensor],
) -> None:
    x = executor._resolve_tensor(op.inputs[0], env)
    if "newShape" in op.options:
        shape = tf.constant(
            [int(v) for v in list(op.options.get("newShape", []))], dtype=tf.int32
        )
    elif len(op.inputs) >= 2:
        shape = tf.cast(executor._resolve_tensor(op.inputs[1], env), tf.int32)
    else:
        raise ModelIRSavedModelExportError(
            f"RESHAPE requires newShape option or shape input. op={op.op_type}"
        )
    executor._assign_outputs(op, [tf.reshape(x, shape)], env)


def _kernel_transpose(
    executor: _GraphExecutor,
    op: OperatorIR,
    env: Dict[str, tf.Tensor],
) -> None:
    x = executor._resolve_tensor(op.inputs[0], env)
    if len(op.inputs) >= 2:
        perm = tf.cast(executor._resolve_tensor(op.inputs[1], env), tf.int32)
    else:
        perm = tf.constant(
            [int(v) for v in list(op.options.get("perm", []))], dtype=tf.int32
        )
    executor._assign_outputs(op, [tf.transpose(x, perm=perm)], env)


def _kernel_concat(
    executor: _GraphExecutor,
    op: OperatorIR,
    env: Dict[str, tf.Tensor],
) -> None:
    axis = int(op.options.get("axis", 0))
    values = [executor._resolve_tensor(name, env) for name in op.inputs]
    y = tf.concat(values, axis=axis)
    y = _apply_fused_activation(y, str(op.options.get("fusedActivationFunction", "NONE")))
    executor._assign_outputs(op, [y], env)


def _kernel_squeeze(
    executor: _GraphExecutor,
    op: OperatorIR,
    env: Dict[str, tf.Tensor],
) -> None:
    x = executor._resolve_tensor(op.inputs[0], env)
    axes = [int(v) for v in list(op.options.get("squeezeDims", []))]
    executor._assign_outputs(op, [tf.squeeze(x, axis=axes if len(axes) > 0 else None)], env)


def _kernel_expand_dims(
    executor: _GraphExecutor,
    op: OperatorIR,
    env: Dict[str, tf.Tensor],
) -> None:
    x = executor._resolve_tensor(op.inputs[0], env)
    if len(op.inputs) >= 2:
        axis = tf.cast(executor._resolve_tensor(op.inputs[1], env), tf.int32)
    else:
        axis = tf.constant(int(op.options.get("axis", 0)), dtype=tf.int32)
    executor._assign_outputs(op, [tf.expand_dims(x, axis)], env)


def _kernel_split(
    executor: _GraphExecutor,
    op: OperatorIR,
    env: Dict[str, tf.Tensor],
) -> None:
    num_splits = int(op.options.get("numSplits", len(op.outputs)))
    axis: Any = 0
    if len(op.inputs) >= 2:
        first = executor._resolve_tensor(op.inputs[0], env)
        second = executor._resolve_tensor(op.inputs[1], env)

        def _is_axis_candidate(t: tf.Tensor) -> bool:
            if t.dtype not in {tf.int8, tf.int16, tf.int32, tf.int64}:
                return False
            if t.shape.rank == 0:
                return True
            if t.shape.rank == 1 and t.shape[0] == 1:
                return True
            static_value = tf.get_static_value(t)
            if static_value is None:
                return False
            return np.asarray(static_value).size == 1

        if _is_axis_candidate(first):
            axis_tensor = tf.cast(first, tf.int32)
            x = second
        elif _is_axis_candidate(second):
            axis_tensor = tf.cast(second, tf.int32)
            x = first
        else:
            # Fallback to TFLite Split input order: [axis, input]
            axis_tensor = tf.cast(first, tf.int32)
            x = second

        axis_value = tf.get_static_value(axis_tensor)
        axis = int(axis_value) if axis_value is not None else tf.reshape(axis_tensor, [])
    elif "axis" in op.options:
        x = executor._resolve_tensor(op.inputs[0], env)
        axis = int(op.options["axis"])
    else:
        x = executor._resolve_tensor(op.inputs[0], env)
    values = tf.split(x, num_or_size_splits=num_splits, axis=axis)
    executor._assign_outputs(op, list(values), env)


def _kernel_pack(
    executor: _GraphExecutor,
    op: OperatorIR,
    env: Dict[str, tf.Tensor],
) -> None:
    axis = int(op.options.get("axis", 0))
    values = [executor._resolve_tensor(name, env) for name in op.inputs]
    executor._assign_outputs(op, [tf.stack(values, axis=axis)], env)


def _kernel_unpack(
    executor: _GraphExecutor,
    op: OperatorIR,
    env: Dict[str, tf.Tensor],
) -> None:
    x = executor._resolve_tensor(op.inputs[0], env)
    axis = int(op.options.get("axis", 0))
    values = tf.unstack(x, axis=axis, num=len(op.outputs))
    executor._assign_outputs(op, list(values), env)


def _kernel_slice(
    executor: _GraphExecutor,
    op: OperatorIR,
    env: Dict[str, tf.Tensor],
) -> None:
    x = executor._resolve_tensor(op.inputs[0], env)
    begin = tf.cast(executor._resolve_tensor(op.inputs[1], env), tf.int32)
    size = tf.cast(executor._resolve_tensor(op.inputs[2], env), tf.int32)
    executor._assign_outputs(op, [tf.slice(x, begin, size)], env)


def _kernel_strided_slice(
    executor: _GraphExecutor,
    op: OperatorIR,
    env: Dict[str, tf.Tensor],
) -> None:
    x = executor._resolve_tensor(op.inputs[0], env)
    begin = tf.cast(executor._resolve_tensor(op.inputs[1], env), tf.int32)
    end = tf.cast(executor._resolve_tensor(op.inputs[2], env), tf.int32)
    strides = tf.cast(executor._resolve_tensor(op.inputs[3], env), tf.int32)
    y = tf.strided_slice(
        x,
        begin=begin,
        end=end,
        strides=strides,
        begin_mask=int(op.options.get("beginMask", 0)),
        end_mask=int(op.options.get("endMask", 0)),
        ellipsis_mask=int(op.options.get("ellipsisMask", 0)),
        new_axis_mask=int(op.options.get("newAxisMask", 0)),
        shrink_axis_mask=int(op.options.get("shrinkAxisMask", 0)),
    )
    executor._assign_outputs(op, [y], env)


def _kernel_shape(
    executor: _GraphExecutor,
    op: OperatorIR,
    env: Dict[str, tf.Tensor],
) -> None:
    x = executor._resolve_tensor(op.inputs[0], env)
    out_dtype_name = str(op.options.get("outType", "INT32")).upper()
    out_dtype = _TF_DTYPE_BY_TFLITE_DTYPE.get(out_dtype_name, tf.int32)
    executor._assign_outputs(op, [tf.shape(x, out_type=out_dtype)], env)


def _kernel_fill(
    executor: _GraphExecutor,
    op: OperatorIR,
    env: Dict[str, tf.Tensor],
) -> None:
    dims = tf.cast(executor._resolve_tensor(op.inputs[0], env), tf.int32)
    value = executor._resolve_tensor(op.inputs[1], env)
    executor._assign_outputs(op, [tf.fill(dims, tf.reshape(value, []))], env)


def _kernel_range(
    executor: _GraphExecutor,
    op: OperatorIR,
    env: Dict[str, tf.Tensor],
) -> None:
    start = executor._resolve_tensor(op.inputs[0], env)
    limit = executor._resolve_tensor(op.inputs[1], env)
    delta = executor._resolve_tensor(op.inputs[2], env)
    executor._assign_outputs(op, [tf.range(start, limit, delta)], env)


def _kernel_softmax(
    executor: _GraphExecutor,
    op: OperatorIR,
    env: Dict[str, tf.Tensor],
) -> None:
    x = executor._resolve_tensor(op.inputs[0], env)
    beta = float(op.options.get("beta", 1.0))
    if beta != 1.0:
        x = tf.cast(beta, x.dtype) * x
    executor._assign_outputs(op, [tf.nn.softmax(x)], env)


def _kernel_reduce(
    tf_reduce_fn: Callable[..., tf.Tensor],
) -> Callable[[_GraphExecutor, OperatorIR, Dict[str, tf.Tensor]], None]:
    def _impl(executor: _GraphExecutor, op: OperatorIR, env: Dict[str, tf.Tensor]) -> None:
        x = executor._resolve_tensor(op.inputs[0], env)
        if len(op.inputs) >= 2:
            axis = tf.cast(executor._resolve_tensor(op.inputs[1], env), tf.int32)
        else:
            axis = None
        keepdims = _parse_bool_option(op.options.get("keepDims", True), default=True)
        y = tf_reduce_fn(x, axis=axis, keepdims=keepdims)
        executor._assign_outputs(op, [y], env)
    return _impl


def _kernel_pad(
    executor: _GraphExecutor,
    op: OperatorIR,
    env: Dict[str, tf.Tensor],
) -> None:
    x = executor._resolve_tensor(op.inputs[0], env)
    paddings = tf.cast(executor._resolve_tensor(op.inputs[1], env), tf.int32)
    if str(op.op_type) == "PADV2":
        constant_values = executor._resolve_tensor(op.inputs[2], env)
        y = tf.pad(x, paddings, mode="CONSTANT", constant_values=tf.reshape(constant_values, []))
    else:
        y = tf.pad(x, paddings)
    executor._assign_outputs(op, [y], env)


def _kernel_where(
    executor: _GraphExecutor,
    op: OperatorIR,
    env: Dict[str, tf.Tensor],
) -> None:
    if len(op.inputs) == 1:
        cond = executor._resolve_tensor(op.inputs[0], env)
        executor._assign_outputs(op, [tf.where(cond)], env)
        return
    cond = executor._resolve_tensor(op.inputs[0], env)
    x = executor._resolve_tensor(op.inputs[1], env)
    y = executor._resolve_tensor(op.inputs[2], env)
    executor._assign_outputs(op, [tf.where(cond, x, y)], env)


def _kernel_gather(
    executor: _GraphExecutor,
    op: OperatorIR,
    env: Dict[str, tf.Tensor],
) -> None:
    params = executor._resolve_tensor(op.inputs[0], env)
    indices = executor._resolve_tensor(op.inputs[1], env)
    axis = int(op.options.get("axis", 0))
    batch_dims = int(op.options.get("batchDims", 0))
    params_rank = params.shape.rank
    axis_resolved = int(axis)
    if params_rank is not None:
        if axis_resolved < 0:
            axis_resolved += int(params_rank)
        axis_resolved = max(0, min(axis_resolved, int(params_rank) - 1))
    indices_i64 = tf.cast(indices, tf.int64)
    axis_dim = tf.cast(tf.shape(params)[axis_resolved], tf.int64)
    safe_axis_dim = tf.maximum(axis_dim, tf.constant(1, dtype=tf.int64))
    indices_non_negative = tf.where(
        indices_i64 < 0,
        indices_i64 + safe_axis_dim,
        indices_i64,
    )
    safe_indices = tf.cast(indices_non_negative, indices.dtype)
    y = tf.gather(params, safe_indices, axis=axis, batch_dims=batch_dims)
    executor._assign_outputs(op, [y], env)


def _kernel_gather_nd(
    executor: _GraphExecutor,
    op: OperatorIR,
    env: Dict[str, tf.Tensor],
) -> None:
    params = executor._resolve_tensor(op.inputs[0], env)
    indices = executor._resolve_tensor(op.inputs[1], env)
    y = tf.gather_nd(params, indices)
    executor._assign_outputs(op, [y], env)


def _kernel_scatter_nd(
    executor: _GraphExecutor,
    op: OperatorIR,
    env: Dict[str, tf.Tensor],
) -> None:
    indices = executor._resolve_tensor(op.inputs[0], env)
    updates = executor._resolve_tensor(op.inputs[1], env)
    shape = tf.cast(executor._resolve_tensor(op.inputs[2], env), tf.int32)
    y = tf.scatter_nd(indices, updates, shape=shape)
    executor._assign_outputs(op, [y], env)


def _kernel_tile(
    executor: _GraphExecutor,
    op: OperatorIR,
    env: Dict[str, tf.Tensor],
) -> None:
    x = executor._resolve_tensor(op.inputs[0], env)
    multiples = tf.cast(executor._resolve_tensor(op.inputs[1], env), tf.int32)
    executor._assign_outputs(op, [tf.tile(x, multiples)], env)


def _kernel_broadcast_to(
    executor: _GraphExecutor,
    op: OperatorIR,
    env: Dict[str, tf.Tensor],
) -> None:
    x = executor._resolve_tensor(op.inputs[0], env)
    shape = tf.cast(executor._resolve_tensor(op.inputs[1], env), tf.int32)
    executor._assign_outputs(op, [tf.broadcast_to(x, shape)], env)


def _kernel_arg(
    is_max: bool,
) -> Callable[[_GraphExecutor, OperatorIR, Dict[str, tf.Tensor]], None]:
    def _impl(executor: _GraphExecutor, op: OperatorIR, env: Dict[str, tf.Tensor]) -> None:
        x = executor._resolve_tensor(op.inputs[0], env)
        axis = tf.cast(executor._resolve_tensor(op.inputs[1], env), tf.int32)
        output_type_name = str(op.options.get("outputType", "INT64")).upper()
        out_type = _TF_DTYPE_BY_TFLITE_DTYPE.get(output_type_name, tf.int64)
        if is_max:
            y = tf.argmax(x, axis=axis, output_type=out_type)
        else:
            y = tf.argmin(x, axis=axis, output_type=out_type)
        executor._assign_outputs(op, [y], env)
    return _impl


def _kernel_topk(
    executor: _GraphExecutor,
    op: OperatorIR,
    env: Dict[str, tf.Tensor],
) -> None:
    x = executor._resolve_tensor(op.inputs[0], env)
    k = tf.cast(executor._resolve_tensor(op.inputs[1], env), tf.int32)
    values, indices = tf.math.top_k(x, k=tf.reshape(k, []))
    executor._assign_outputs(op, [values, indices], env)


def _kernel_relu_clipped(
    min_value: float,
    max_value: float,
) -> Callable[[_GraphExecutor, OperatorIR, Dict[str, tf.Tensor]], None]:
    def _impl(executor: _GraphExecutor, op: OperatorIR, env: Dict[str, tf.Tensor]) -> None:
        x = executor._resolve_tensor(op.inputs[0], env)
        executor._assign_outputs(op, [tf.clip_by_value(x, min_value, max_value)], env)
    return _impl


def _kernel_fully_connected(
    executor: _GraphExecutor,
    op: OperatorIR,
    env: Dict[str, tf.Tensor],
) -> None:
    x = executor._resolve_tensor(op.inputs[0], env)
    w = executor._resolve_tensor(op.inputs[1], env)
    b = executor._resolve_optional_tensor(op.inputs[2], env) if len(op.inputs) >= 3 else None

    keep_num_dims = _parse_bool_option(op.options.get("keepNumDims", False), default=False)
    if not keep_num_dims and x.shape.rank is not None and x.shape.rank > 2:
        x = tf.reshape(x, [tf.shape(x)[0], -1])
    elif x.shape.rank is not None and x.shape.rank == 1:
        x = tf.expand_dims(x, axis=0)

    y = tf.linalg.matmul(x, w, transpose_b=True)
    if b is not None:
        y = tf.nn.bias_add(y, b)
    y = _apply_fused_activation(y, str(op.options.get("fusedActivationFunction", "NONE")))
    executor._assign_outputs(op, [y], env)


def _kernel_conv2d(
    executor: _GraphExecutor,
    op: OperatorIR,
    env: Dict[str, tf.Tensor],
) -> None:
    def _positive_dim(value: Any) -> Optional[int]:
        try:
            dim = int(value)
        except Exception:
            return None
        if dim <= 0:
            return None
        return dim

    def _tensor_ir_dim(tensor_name: str, axis: int) -> Optional[int]:
        tensor_ir = executor.model_ir.tensors.get(str(tensor_name), None)
        if tensor_ir is None:
            return None
        shape = list(tensor_ir.shape) if tensor_ir.shape is not None else []
        if len(shape) == 0:
            return None
        idx = int(axis)
        if idx < 0:
            idx += len(shape)
        if idx < 0 or idx >= len(shape):
            return None
        return _positive_dim(shape[idx])

    x = executor._resolve_tensor(op.inputs[0], env)
    w = executor._resolve_tensor(op.inputs[1], env)
    b = executor._resolve_optional_tensor(op.inputs[2], env) if len(op.inputs) >= 3 else None
    # TFLite conv filter: [out, h, w, in] -> TF conv filter: [h, w, in, out]
    w_tf = tf.transpose(w, perm=[1, 2, 3, 0])
    strides = [1, int(op.options.get("strideH", 1)), int(op.options.get("strideW", 1)), 1]
    dilations = [1, int(op.options.get("dilationHFactor", 1)), int(op.options.get("dilationWFactor", 1)), 1]
    padding = str(op.options.get("padding", "SAME")).upper()
    input_channels = _positive_dim(x.shape[-1])
    filter_in_channels = _positive_dim(w.shape[3])
    output_channels = _positive_dim(w.shape[0])
    if input_channels is None and len(op.inputs) >= 1:
        input_channels = _tensor_ir_dim(op.inputs[0], -1)
    if filter_in_channels is None and len(op.inputs) >= 2:
        filter_in_channels = _tensor_ir_dim(op.inputs[1], 3)
    if output_channels is None and len(op.inputs) >= 2:
        output_channels = _tensor_ir_dim(op.inputs[1], 0)
    groups = 1
    if (
        input_channels is not None
        and filter_in_channels is not None
        and int(input_channels) % int(filter_in_channels) == 0
    ):
        groups = int(input_channels) // int(filter_in_channels)

    if groups > 1:
        if output_channels is None or int(output_channels) % int(groups) != 0:
            raise ModelIRSavedModelExportError(
                "Grouped CONV_2D has unsupported channel partition. "
                f"op={op.op_type} input_channels={input_channels} "
                f"filter_in_channels={filter_in_channels} output_channels={output_channels} groups={groups}"
            )
        out_per_group = int(output_channels) // int(groups)
        x_split = tf.split(x, num_or_size_splits=int(groups), axis=-1)
        y_parts: List[tf.Tensor] = []
        for group_idx, x_part in enumerate(x_split):
            start = int(group_idx * out_per_group)
            end = int(start + out_per_group)
            w_part = w_tf[:, :, :, start:end]
            y_part = tf.nn.conv2d(
                x_part,
                w_part,
                strides=strides,
                dilations=dilations,
                padding=padding,
            )
            y_parts.append(y_part)
        y = tf.concat(y_parts, axis=-1)
    else:
        y = tf.nn.conv2d(
            x,
            w_tf,
            strides=strides,
            dilations=dilations,
            padding=padding,
        )
    if b is not None:
        y = tf.nn.bias_add(y, b)
    y = _apply_fused_activation(y, str(op.options.get("fusedActivationFunction", "NONE")))
    executor._assign_outputs(op, [y], env)


def _kernel_depthwise_conv2d(
    executor: _GraphExecutor,
    op: OperatorIR,
    env: Dict[str, tf.Tensor],
) -> None:
    x = executor._resolve_tensor(op.inputs[0], env)
    w = executor._resolve_tensor(op.inputs[1], env)
    b = executor._resolve_optional_tensor(op.inputs[2], env) if len(op.inputs) >= 3 else None
    w_shape = tf.shape(w)
    w_tf = tf.reshape(
        tf.transpose(w, perm=[1, 2, 3, 0]),
        [w_shape[1], w_shape[2], w_shape[3], w_shape[0]],
    )
    strides = [1, int(op.options.get("strideH", 1)), int(op.options.get("strideW", 1)), 1]
    dilations = [1, int(op.options.get("dilationHFactor", 1)), int(op.options.get("dilationWFactor", 1)), 1]
    y = tf.nn.depthwise_conv2d(
        x,
        w_tf,
        strides=strides,
        dilations=dilations,
        padding=str(op.options.get("padding", "SAME")).upper(),
    )
    if b is not None:
        y = tf.nn.bias_add(y, b)
    y = _apply_fused_activation(y, str(op.options.get("fusedActivationFunction", "NONE")))
    executor._assign_outputs(op, [y], env)


def _kernel_pool2d(
    is_max_pool: bool,
) -> Callable[[_GraphExecutor, OperatorIR, Dict[str, tf.Tensor]], None]:
    def _impl(executor: _GraphExecutor, op: OperatorIR, env: Dict[str, tf.Tensor]) -> None:
        x = executor._resolve_tensor(op.inputs[0], env)
        ksize = [1, int(op.options.get("filterHeight", 1)), int(op.options.get("filterWidth", 1)), 1]
        strides = [1, int(op.options.get("strideH", 1)), int(op.options.get("strideW", 1)), 1]
        padding = str(op.options.get("padding", "SAME")).upper()
        if is_max_pool:
            y = tf.nn.max_pool2d(x, ksize=ksize, strides=strides, padding=padding)
        else:
            y = tf.nn.avg_pool2d(x, ksize=ksize, strides=strides, padding=padding)
        y = _apply_fused_activation(y, str(op.options.get("fusedActivationFunction", "NONE")))
        executor._assign_outputs(op, [y], env)
    return _impl


def _kernel_resize(
    method: str,
) -> Callable[[_GraphExecutor, OperatorIR, Dict[str, tf.Tensor]], None]:
    def _impl(executor: _GraphExecutor, op: OperatorIR, env: Dict[str, tf.Tensor]) -> None:
        x = executor._resolve_tensor(op.inputs[0], env)
        if len(op.inputs) >= 2:
            size = tf.cast(executor._resolve_tensor(op.inputs[1], env), tf.int32)
        else:
            new_h = int(op.options.get("newHeight", 0))
            new_w = int(op.options.get("newWidth", 0))
            size = tf.constant([new_h, new_w], dtype=tf.int32)
        align_corners = _parse_bool_option(op.options.get("alignCorners", False), default=False)
        half_pixel_centers = _parse_bool_option(
            op.options.get("halfPixelCenters", False),
            default=False,
        )
        if method == "nearest":
            y = tf.raw_ops.ResizeNearestNeighbor(
                images=x,
                size=size,
                align_corners=align_corners,
                half_pixel_centers=half_pixel_centers,
            )
        else:
            y = tf.raw_ops.ResizeBilinear(
                images=x,
                size=size,
                align_corners=align_corners,
                half_pixel_centers=half_pixel_centers,
            )
        executor._assign_outputs(op, [y], env)
    return _impl


def _kernel_while(
    executor: _GraphExecutor,
    op: OperatorIR,
    env: Dict[str, tf.Tensor],
) -> None:
    cond_index = int(op.options.get("condSubgraphIndex", 0))
    body_index = int(op.options.get("bodySubgraphIndex", 0))
    cond_executor = executor._get_subgraph_executor(cond_index)
    body_executor = executor._get_subgraph_executor(body_index)
    loop_vars = tuple(executor._resolve_tensor(name, env) for name in op.inputs)

    def _cond(*args: tf.Tensor) -> tf.Tensor:
        cond_inputs = {
            name: value for name, value in zip(cond_executor.model_ir.inputs, args)
        }
        cond_outputs = cond_executor.run(inputs=cond_inputs)
        first_output_name = str(cond_executor.model_ir.outputs[0])
        return tf.cast(cond_outputs[first_output_name], tf.bool)

    def _body(*args: tf.Tensor) -> tuple[tf.Tensor, ...]:
        body_inputs = {
            name: value for name, value in zip(body_executor.model_ir.inputs, args)
        }
        body_outputs = body_executor.run(inputs=body_inputs)
        return tuple(body_outputs[name] for name in body_executor.model_ir.outputs)

    result = tf.while_loop(cond=_cond, body=_body, loop_vars=loop_vars)
    if not isinstance(result, (tuple, list)):
        result_values: List[tf.Tensor] = [result]
    else:
        result_values = list(result)
    executor._assign_outputs(op, result_values, env)


def _kernel_batch_matmul(
    executor: _GraphExecutor,
    op: OperatorIR,
    env: Dict[str, tf.Tensor],
) -> None:
    x = executor._resolve_tensor(op.inputs[0], env)
    y = executor._resolve_tensor(op.inputs[1], env)
    adj_x = _parse_bool_option(op.options.get("adjX", False), default=False)
    adj_y = _parse_bool_option(op.options.get("adjY", False), default=False)
    z = tf.linalg.matmul(x, y, transpose_a=adj_x, transpose_b=adj_y)
    z = _apply_fused_activation(z, str(op.options.get("fusedActivationFunction", "NONE")))
    executor._assign_outputs(op, [z], env)


def _kernel_depth_to_space(
    executor: _GraphExecutor,
    op: OperatorIR,
    env: Dict[str, tf.Tensor],
) -> None:
    x = executor._resolve_tensor(op.inputs[0], env)
    block_size = int(op.options.get("blockSize", 1))
    executor._assign_outputs(op, [tf.nn.depth_to_space(x, block_size=block_size)], env)


def _kernel_space_to_depth(
    executor: _GraphExecutor,
    op: OperatorIR,
    env: Dict[str, tf.Tensor],
) -> None:
    x = executor._resolve_tensor(op.inputs[0], env)
    block_size = int(op.options.get("blockSize", 1))
    executor._assign_outputs(op, [tf.nn.space_to_depth(x, block_size=block_size)], env)


def _kernel_reverse_v2(
    executor: _GraphExecutor,
    op: OperatorIR,
    env: Dict[str, tf.Tensor],
) -> None:
    x = executor._resolve_tensor(op.inputs[0], env)
    axis = tf.cast(executor._resolve_tensor(op.inputs[1], env), tf.int32)
    executor._assign_outputs(op, [tf.reverse(x, axis=axis)], env)


def _kernel_reverse_sequence(
    executor: _GraphExecutor,
    op: OperatorIR,
    env: Dict[str, tf.Tensor],
) -> None:
    x = executor._resolve_tensor(op.inputs[0], env)
    seq_lengths = tf.cast(executor._resolve_tensor(op.inputs[1], env), tf.int32)
    seq_axis = int(op.options.get("seqDim", 0))
    batch_axis = int(op.options.get("batchDim", 1))
    y = tf.reverse_sequence(
        x,
        seq_lengths=seq_lengths,
        seq_axis=seq_axis,
        batch_axis=batch_axis,
    )
    executor._assign_outputs(op, [y], env)


def _kernel_one_hot(
    executor: _GraphExecutor,
    op: OperatorIR,
    env: Dict[str, tf.Tensor],
) -> None:
    indices = executor._resolve_tensor(op.inputs[0], env)
    depth = tf.cast(executor._resolve_tensor(op.inputs[1], env), tf.int32)
    on_value = executor._resolve_tensor(op.inputs[2], env)
    off_value = executor._resolve_tensor(op.inputs[3], env)
    axis = int(op.options.get("axis", -1))
    y = tf.one_hot(
        indices=indices,
        depth=tf.reshape(depth, []),
        on_value=tf.reshape(on_value, []),
        off_value=tf.reshape(off_value, []),
        axis=axis,
    )
    executor._assign_outputs(op, [y], env)


def _kernel_cumsum(
    executor: _GraphExecutor,
    op: OperatorIR,
    env: Dict[str, tf.Tensor],
) -> None:
    x = executor._resolve_tensor(op.inputs[0], env)
    axis = tf.cast(executor._resolve_tensor(op.inputs[1], env), tf.int32)
    exclusive = _parse_bool_option(op.options.get("exclusive", False), default=False)
    reverse = _parse_bool_option(op.options.get("reverse", False), default=False)
    y = tf.math.cumsum(
        x,
        axis=tf.reshape(axis, []),
        exclusive=exclusive,
        reverse=reverse,
    )
    executor._assign_outputs(op, [y], env)


def _kernel_l2_norm(
    executor: _GraphExecutor,
    op: OperatorIR,
    env: Dict[str, tf.Tensor],
) -> None:
    x = executor._resolve_tensor(op.inputs[0], env)
    y = tf.math.l2_normalize(x, axis=-1)
    y = _apply_fused_activation(y, str(op.options.get("fusedActivationFunction", "NONE")))
    executor._assign_outputs(op, [y], env)


def _kernel_local_response_norm(
    executor: _GraphExecutor,
    op: OperatorIR,
    env: Dict[str, tf.Tensor],
) -> None:
    x = executor._resolve_tensor(op.inputs[0], env)
    radius = int(op.options.get("radius", 0))
    bias = float(op.options.get("bias", 1.0))
    alpha = float(op.options.get("alpha", 0.0))
    beta = float(op.options.get("beta", 0.0))
    y = tf.nn.local_response_normalization(
        x,
        depth_radius=radius,
        bias=bias,
        alpha=alpha,
        beta=beta,
    )
    executor._assign_outputs(op, [y], env)


def _kernel_leaky_relu(
    executor: _GraphExecutor,
    op: OperatorIR,
    env: Dict[str, tf.Tensor],
) -> None:
    x = executor._resolve_tensor(op.inputs[0], env)
    alpha = float(op.options.get("alpha", 0.2))
    executor._assign_outputs(op, [tf.nn.leaky_relu(x, alpha=alpha)], env)


def _kernel_prelu(
    executor: _GraphExecutor,
    op: OperatorIR,
    env: Dict[str, tf.Tensor],
) -> None:
    x = executor._resolve_tensor(op.inputs[0], env)
    alpha = executor._resolve_tensor(op.inputs[1], env)
    y = tf.maximum(x, tf.zeros_like(x)) + alpha * tf.minimum(x, tf.zeros_like(x))
    executor._assign_outputs(op, [y], env)


def _kernel_mirror_pad(
    executor: _GraphExecutor,
    op: OperatorIR,
    env: Dict[str, tf.Tensor],
) -> None:
    x = executor._resolve_tensor(op.inputs[0], env)
    paddings = tf.cast(executor._resolve_tensor(op.inputs[1], env), tf.int32)
    mode = str(op.options.get("mode", "REFLECT")).upper()
    if mode not in {"REFLECT", "SYMMETRIC"}:
        mode = "REFLECT"
    y = tf.pad(x, paddings=paddings, mode=mode)
    executor._assign_outputs(op, [y], env)


def _kernel_random_standard_normal(
    executor: _GraphExecutor,
    op: OperatorIR,
    env: Dict[str, tf.Tensor],
) -> None:
    shape = tf.cast(executor._resolve_tensor(op.inputs[0], env), tf.int32)
    out_tensor = executor.model_ir.tensors.get(str(op.outputs[0]), None)
    out_dtype = _tf_dtype_from_tensor_ir(out_tensor) if out_tensor is not None else tf.float32
    seed = int(op.options.get("seed", 0))
    seed2 = int(op.options.get("seed2", 0))
    if seed != 0:
        tf.random.set_seed(seed ^ seed2)
    y = tf.random.normal(shape=shape, dtype=out_dtype)
    executor._assign_outputs(op, [y], env)


def _kernel_random_uniform(
    executor: _GraphExecutor,
    op: OperatorIR,
    env: Dict[str, tf.Tensor],
) -> None:
    shape = tf.cast(executor._resolve_tensor(op.inputs[0], env), tf.int32)
    out_tensor = executor.model_ir.tensors.get(str(op.outputs[0]), None)
    out_dtype = _tf_dtype_from_tensor_ir(out_tensor) if out_tensor is not None else tf.float32
    seed = int(op.options.get("seed", 0))
    seed2 = int(op.options.get("seed2", 0))
    if seed != 0:
        tf.random.set_seed(seed ^ seed2)
    sample_dtype = out_dtype if out_dtype.is_floating else tf.float32
    y = tf.random.uniform(shape=shape, dtype=sample_dtype)
    if sample_dtype != out_dtype:
        y = tf.cast(y, out_dtype)
    executor._assign_outputs(op, [y], env)


def _kernel_transpose_conv(
    executor: _GraphExecutor,
    op: OperatorIR,
    env: Dict[str, tf.Tensor],
) -> None:
    output_shape = tf.cast(executor._resolve_tensor(op.inputs[0], env), tf.int32)
    w = executor._resolve_tensor(op.inputs[1], env)
    x = executor._resolve_tensor(op.inputs[2], env)
    b = executor._resolve_optional_tensor(op.inputs[3], env) if len(op.inputs) >= 4 else None
    # TFLite transpose-conv filter: [out, h, w, in] -> TF: [h, w, out, in]
    w_tf = tf.transpose(w, perm=[1, 2, 0, 3])
    strides = [1, int(op.options.get("strideH", 1)), int(op.options.get("strideW", 1)), 1]
    y = tf.nn.conv2d_transpose(
        x,
        filters=w_tf,
        output_shape=output_shape,
        strides=strides,
        padding=str(op.options.get("padding", "SAME")).upper(),
    )
    if b is not None:
        y = tf.nn.bias_add(y, b)
    y = _apply_fused_activation(y, str(op.options.get("fusedActivationFunction", "NONE")))
    executor._assign_outputs(op, [y], env)


def _kernel_conv3d(
    executor: _GraphExecutor,
    op: OperatorIR,
    env: Dict[str, tf.Tensor],
) -> None:
    x = executor._resolve_tensor(op.inputs[0], env)
    w = executor._resolve_tensor(op.inputs[1], env)
    b = executor._resolve_optional_tensor(op.inputs[2], env) if len(op.inputs) >= 3 else None
    # TFLite conv3d filter: [out, d, h, w, in] -> TF: [d, h, w, in, out]
    w_tf = tf.transpose(w, perm=[1, 2, 3, 4, 0])
    strides = [
        1,
        int(op.options.get("strideD", 1)),
        int(op.options.get("strideH", 1)),
        int(op.options.get("strideW", 1)),
        1,
    ]
    dilations = [
        1,
        int(op.options.get("dilationDFactor", 1)),
        int(op.options.get("dilationHFactor", 1)),
        int(op.options.get("dilationWFactor", 1)),
        1,
    ]
    y = tf.nn.conv3d(
        x,
        filters=w_tf,
        strides=strides,
        dilations=dilations,
        padding=str(op.options.get("padding", "SAME")).upper(),
    )
    if b is not None:
        y = y + tf.reshape(b, [1, 1, 1, 1, -1])
    y = _apply_fused_activation(y, str(op.options.get("fusedActivationFunction", "NONE")))
    executor._assign_outputs(op, [y], env)


def _kernel_conv3d_transpose(
    executor: _GraphExecutor,
    op: OperatorIR,
    env: Dict[str, tf.Tensor],
) -> None:
    output_shape = tf.cast(executor._resolve_tensor(op.inputs[0], env), tf.int32)
    w = executor._resolve_tensor(op.inputs[1], env)
    x = executor._resolve_tensor(op.inputs[2], env)
    b = executor._resolve_optional_tensor(op.inputs[3], env) if len(op.inputs) >= 4 else None
    # TFLite conv3d-transpose filter: [out, d, h, w, in] -> TF: [d, h, w, out, in]
    w_tf = tf.transpose(w, perm=[1, 2, 3, 0, 4])
    strides = [
        1,
        int(op.options.get("strideD", 1)),
        int(op.options.get("strideH", 1)),
        int(op.options.get("strideW", 1)),
        1,
    ]
    y = tf.nn.conv3d_transpose(
        x,
        filters=w_tf,
        output_shape=output_shape,
        strides=strides,
        padding=str(op.options.get("padding", "SAME")).upper(),
    )
    if b is not None:
        y = y + tf.reshape(b, [1, 1, 1, 1, -1])
    y = _apply_fused_activation(y, str(op.options.get("fusedActivationFunction", "NONE")))
    executor._assign_outputs(op, [y], env)


def _extract_quant_scale_zero_point(tensor: Optional[TensorIR]) -> tuple[Optional[float], Optional[float]]:
    if tensor is None:
        return None, None
    quant = tensor.quantization
    if quant is None:
        return None, None
    scale_list: List[float]
    zero_list: List[float]
    if isinstance(quant, dict):
        scale_list = [float(v) for v in list(quant.get("scale", []))]
        zero_list = [float(v) for v in list(quant.get("zero_point", []))]
    else:
        scale_list = [float(v) for v in list(getattr(quant, "scale", []))]
        zero_list = [float(v) for v in list(getattr(quant, "zero_point", []))]
    if len(scale_list) == 0 or len(zero_list) == 0:
        return None, None
    return float(scale_list[0]), float(zero_list[0])


def _kernel_dequantize(
    executor: _GraphExecutor,
    op: OperatorIR,
    env: Dict[str, tf.Tensor],
) -> None:
    x = executor._resolve_tensor(op.inputs[0], env)
    input_tensor = executor.model_ir.tensors.get(str(op.inputs[0]), None)
    output_tensor = executor.model_ir.tensors.get(str(op.outputs[0]), None)
    scale, zero_point = _extract_quant_scale_zero_point(input_tensor)
    if scale is None or zero_point is None:
        y = tf.cast(x, _tf_dtype_from_tensor_ir(output_tensor) if output_tensor is not None else tf.float32)
        executor._assign_outputs(op, [y], env)
        return
    y = (tf.cast(x, tf.float32) - tf.cast(zero_point, tf.float32)) * tf.cast(scale, tf.float32)
    if output_tensor is not None:
        y = tf.cast(y, _tf_dtype_from_tensor_ir(output_tensor))
    executor._assign_outputs(op, [y], env)


def _kernel_quantize(
    executor: _GraphExecutor,
    op: OperatorIR,
    env: Dict[str, tf.Tensor],
) -> None:
    x = executor._resolve_tensor(op.inputs[0], env)
    output_tensor = executor.model_ir.tensors.get(str(op.outputs[0]), None)
    if output_tensor is None:
        executor._assign_outputs(op, [x], env)
        return
    output_dtype = _tf_dtype_from_tensor_ir(output_tensor)
    scale, zero_point = _extract_quant_scale_zero_point(output_tensor)
    if scale is None or zero_point is None:
        executor._assign_outputs(op, [tf.cast(x, output_dtype)], env)
        return
    q = tf.math.round(tf.cast(x, tf.float32) / tf.cast(scale, tf.float32) + tf.cast(zero_point, tf.float32))
    np_dtype = np.dtype(output_dtype.as_numpy_dtype)
    if np.issubdtype(np_dtype, np.integer):
        info = np.iinfo(np_dtype)
        q = tf.clip_by_value(q, float(info.min), float(info.max))
    y = tf.cast(q, output_dtype)
    executor._assign_outputs(op, [y], env)


def _kernel_unique(
    executor: _GraphExecutor,
    op: OperatorIR,
    env: Dict[str, tf.Tensor],
) -> None:
    x = executor._resolve_tensor(op.inputs[0], env)
    x_flat = tf.reshape(x, [-1])
    values, indices = tf.unique(x_flat)
    idx_out_dtype = str(op.options.get("idxOutType", "INT32")).upper()
    idx_dtype = _TF_DTYPE_BY_TFLITE_DTYPE.get(idx_out_dtype, tf.int32)
    outputs: List[tf.Tensor] = [values]
    if len(op.outputs) >= 2:
        outputs.append(tf.cast(indices, idx_dtype))
    executor._assign_outputs(op, outputs, env)


def _kernel_non_max_suppression_v5(
    executor: _GraphExecutor,
    op: OperatorIR,
    env: Dict[str, tf.Tensor],
) -> None:
    boxes = executor._resolve_tensor(op.inputs[0], env)
    scores = executor._resolve_tensor(op.inputs[1], env)
    max_output_size = tf.cast(executor._resolve_tensor(op.inputs[2], env), tf.int32)
    iou_threshold = tf.cast(executor._resolve_tensor(op.inputs[3], env), tf.float32)
    score_threshold = tf.cast(executor._resolve_tensor(op.inputs[4], env), tf.float32)
    soft_nms_sigma = (
        tf.cast(executor._resolve_tensor(op.inputs[5], env), tf.float32)
        if len(op.inputs) >= 6
        else tf.constant(0.0, dtype=tf.float32)
    )

    boxes_rank = boxes.shape.rank
    scores_rank = scores.shape.rank
    if boxes_rank == 3:
        boxes = boxes[0]
    if scores_rank == 3:
        scores = scores[0]
    if scores.shape.rank == 2:
        scores = tf.reduce_max(scores, axis=-1)

    selected_indices, selected_scores = tf.image.non_max_suppression_with_scores(
        boxes=boxes,
        scores=scores,
        max_output_size=tf.reshape(max_output_size, []),
        iou_threshold=tf.reshape(iou_threshold, []),
        score_threshold=tf.reshape(score_threshold, []),
        soft_nms_sigma=tf.reshape(soft_nms_sigma, []),
    )
    valid_outputs = tf.shape(selected_indices, out_type=tf.int32)[0]
    executor._assign_outputs(
        op,
        [selected_indices, selected_scores, valid_outputs],
        env,
    )


def _kernel_non_max_suppression_v4(
    executor: _GraphExecutor,
    op: OperatorIR,
    env: Dict[str, tf.Tensor],
) -> None:
    boxes = executor._resolve_tensor(op.inputs[0], env)
    scores = executor._resolve_tensor(op.inputs[1], env)
    max_output_size = tf.cast(executor._resolve_tensor(op.inputs[2], env), tf.int32)
    iou_threshold = tf.cast(executor._resolve_tensor(op.inputs[3], env), tf.float32)
    score_threshold = tf.cast(executor._resolve_tensor(op.inputs[4], env), tf.float32)

    boxes_rank = boxes.shape.rank
    scores_rank = scores.shape.rank
    if boxes_rank == 3:
        boxes = boxes[0]
    if scores_rank == 3:
        scores = scores[0]
    if scores.shape.rank == 2:
        scores = tf.reduce_max(scores, axis=-1)

    selected_indices = tf.image.non_max_suppression(
        boxes=boxes,
        scores=scores,
        max_output_size=tf.reshape(max_output_size, []),
        iou_threshold=tf.reshape(iou_threshold, []),
        score_threshold=tf.reshape(score_threshold, []),
    )
    valid_outputs = tf.shape(selected_indices, out_type=tf.int32)[0]
    executor._assign_outputs(
        op,
        [selected_indices, valid_outputs],
        env,
    )


def _resolve_optional_by_index(
    executor: _GraphExecutor,
    op: OperatorIR,
    env: Dict[str, tf.Tensor],
    index: int,
) -> Optional[tf.Tensor]:
    if index >= len(op.inputs):
        return None
    return executor._resolve_optional_tensor(op.inputs[index], env)


def _run_unidirectional_sequence_lstm(
    *,
    x: tf.Tensor,
    wi: tf.Tensor,
    wf: tf.Tensor,
    wc: tf.Tensor,
    wo: tf.Tensor,
    ri: tf.Tensor,
    rf: tf.Tensor,
    rc: tf.Tensor,
    ro: tf.Tensor,
    bi: tf.Tensor,
    bf: tf.Tensor,
    bc: tf.Tensor,
    bo: tf.Tensor,
    h0: tf.Tensor,
    c0: tf.Tensor,
    projection_weights: Optional[tf.Tensor],
    projection_bias: Optional[tf.Tensor],
    cell_clip: float,
    proj_clip: float,
    time_major: bool,
    fused_activation: str,
    reverse_time: bool = False,
) -> tf.Tensor:
    x_time = x if bool(time_major) else tf.transpose(x, perm=[1, 0, 2])
    if reverse_time:
        x_time = tf.reverse(x_time, axis=[0])

    def _step(
        state: tuple[tf.Tensor, tf.Tensor],
        x_t: tf.Tensor,
    ) -> tuple[tf.Tensor, tf.Tensor]:
        h_prev, c_prev = state
        i = tf.math.sigmoid(
            tf.linalg.matmul(x_t, wi, transpose_b=True) +
            tf.linalg.matmul(h_prev, ri, transpose_b=True) +
            bi
        )
        f = tf.math.sigmoid(
            tf.linalg.matmul(x_t, wf, transpose_b=True) +
            tf.linalg.matmul(h_prev, rf, transpose_b=True) +
            bf
        )
        c_bar = tf.math.tanh(
            tf.linalg.matmul(x_t, wc, transpose_b=True) +
            tf.linalg.matmul(h_prev, rc, transpose_b=True) +
            bc
        )
        c = f * c_prev + i * c_bar
        if cell_clip > 0.0:
            c = tf.clip_by_value(c, -cell_clip, cell_clip)
        o = tf.math.sigmoid(
            tf.linalg.matmul(x_t, wo, transpose_b=True) +
            tf.linalg.matmul(h_prev, ro, transpose_b=True) +
            bo
        )
        h = o * _apply_activation_by_name(c, fused_activation)
        if projection_weights is not None:
            h = tf.linalg.matmul(h, projection_weights, transpose_b=True)
            if projection_bias is not None:
                h = h + projection_bias
            if proj_clip > 0.0:
                h = tf.clip_by_value(h, -proj_clip, proj_clip)
        return h, c

    h_seq, _ = tf.scan(
        fn=_step,
        elems=x_time,
        initializer=(h0, c0),
    )
    if reverse_time:
        h_seq = tf.reverse(h_seq, axis=[0])
    if not bool(time_major):
        h_seq = tf.transpose(h_seq, perm=[1, 0, 2])
    return h_seq


def _kernel_unidirectional_sequence_lstm(
    executor: _GraphExecutor,
    op: OperatorIR,
    env: Dict[str, tf.Tensor],
) -> None:
    if len(op.inputs) < 24:
        raise ModelIRSavedModelExportError(
            "UNIDIRECTIONAL_SEQUENCE_LSTM expects 24 inputs."
        )
    unsupported_optional_indices = [9, 10, 11, 20, 21, 22, 23]
    for idx in unsupported_optional_indices:
        if str(op.inputs[idx]).strip() != "":
            raise ModelIRSavedModelExportError(
                "UNIDIRECTIONAL_SEQUENCE_LSTM with peephole/auxiliary inputs is not supported."
            )

    x = executor._resolve_tensor(op.inputs[0], env)
    wi = executor._resolve_tensor(op.inputs[1], env)
    wf = executor._resolve_tensor(op.inputs[2], env)
    wc = executor._resolve_tensor(op.inputs[3], env)
    wo = executor._resolve_tensor(op.inputs[4], env)
    ri = executor._resolve_tensor(op.inputs[5], env)
    rf = executor._resolve_tensor(op.inputs[6], env)
    rc = executor._resolve_tensor(op.inputs[7], env)
    ro = executor._resolve_tensor(op.inputs[8], env)
    bi = executor._resolve_tensor(op.inputs[12], env)
    bf = executor._resolve_tensor(op.inputs[13], env)
    bc = executor._resolve_tensor(op.inputs[14], env)
    bo = executor._resolve_tensor(op.inputs[15], env)

    projection_weights = _resolve_optional_by_index(executor, op, env, 16)
    projection_bias = _resolve_optional_by_index(executor, op, env, 17)
    h0 = _resolve_optional_by_index(executor, op, env, 18)
    c0 = _resolve_optional_by_index(executor, op, env, 19)

    batch_size = tf.shape(x)[1] if _parse_bool_option(op.options.get("timeMajor", True), default=True) else tf.shape(x)[0]
    hidden_size = tf.shape(wi)[0]
    if h0 is None:
        h0 = tf.zeros([batch_size, hidden_size], dtype=x.dtype)
    if c0 is None:
        c0 = tf.zeros([batch_size, hidden_size], dtype=x.dtype)

    y = _run_unidirectional_sequence_lstm(
        x=x,
        wi=wi,
        wf=wf,
        wc=wc,
        wo=wo,
        ri=ri,
        rf=rf,
        rc=rc,
        ro=ro,
        bi=bi,
        bf=bf,
        bc=bc,
        bo=bo,
        h0=h0,
        c0=c0,
        projection_weights=projection_weights,
        projection_bias=projection_bias,
        cell_clip=float(op.options.get("cellClip", 0.0)),
        proj_clip=float(op.options.get("projClip", 0.0)),
        time_major=_parse_bool_option(op.options.get("timeMajor", True), default=True),
        fused_activation=str(op.options.get("fusedActivationFunction", "TANH")),
        reverse_time=False,
    )
    executor._assign_outputs(op, [y], env)


def _kernel_bidirectional_sequence_lstm(
    executor: _GraphExecutor,
    op: OperatorIR,
    env: Dict[str, tf.Tensor],
) -> None:
    if len(op.inputs) < 48:
        raise ModelIRSavedModelExportError(
            "BIDIRECTIONAL_SEQUENCE_LSTM expects 48 inputs."
        )
    unsupported_optional_indices = [
        9, 10, 11, 26, 27, 28, 39, 40, 41, 42, 43, 44, 45, 46, 47,
    ]
    for idx in unsupported_optional_indices:
        if str(op.inputs[idx]).strip() != "":
            raise ModelIRSavedModelExportError(
                "BIDIRECTIONAL_SEQUENCE_LSTM with peephole/auxiliary inputs is not supported."
            )

    x = executor._resolve_tensor(op.inputs[0], env)
    time_major = _parse_bool_option(op.options.get("timeMajor", True), default=True)
    fused_activation = str(op.options.get("fusedActivationFunction", "TANH"))
    cell_clip = float(op.options.get("cellClip", 0.0))
    proj_clip = float(op.options.get("projClip", 0.0))

    fw_h0 = _resolve_optional_by_index(executor, op, env, 35)
    fw_c0 = _resolve_optional_by_index(executor, op, env, 36)
    bw_h0 = _resolve_optional_by_index(executor, op, env, 37)
    bw_c0 = _resolve_optional_by_index(executor, op, env, 38)

    batch_size = tf.shape(x)[1] if time_major else tf.shape(x)[0]
    fw_hidden_size = tf.shape(executor._resolve_tensor(op.inputs[1], env))[0]
    bw_hidden_size = tf.shape(executor._resolve_tensor(op.inputs[18], env))[0]
    if fw_h0 is None:
        fw_h0 = tf.zeros([batch_size, fw_hidden_size], dtype=x.dtype)
    if fw_c0 is None:
        fw_c0 = tf.zeros([batch_size, fw_hidden_size], dtype=x.dtype)
    if bw_h0 is None:
        bw_h0 = tf.zeros([batch_size, bw_hidden_size], dtype=x.dtype)
    if bw_c0 is None:
        bw_c0 = tf.zeros([batch_size, bw_hidden_size], dtype=x.dtype)

    fw_y = _run_unidirectional_sequence_lstm(
        x=x,
        wi=executor._resolve_tensor(op.inputs[1], env),
        wf=executor._resolve_tensor(op.inputs[2], env),
        wc=executor._resolve_tensor(op.inputs[3], env),
        wo=executor._resolve_tensor(op.inputs[4], env),
        ri=executor._resolve_tensor(op.inputs[5], env),
        rf=executor._resolve_tensor(op.inputs[6], env),
        rc=executor._resolve_tensor(op.inputs[7], env),
        ro=executor._resolve_tensor(op.inputs[8], env),
        bi=executor._resolve_tensor(op.inputs[12], env),
        bf=executor._resolve_tensor(op.inputs[13], env),
        bc=executor._resolve_tensor(op.inputs[14], env),
        bo=executor._resolve_tensor(op.inputs[15], env),
        h0=fw_h0,
        c0=fw_c0,
        projection_weights=_resolve_optional_by_index(executor, op, env, 16),
        projection_bias=_resolve_optional_by_index(executor, op, env, 17),
        cell_clip=cell_clip,
        proj_clip=proj_clip,
        time_major=time_major,
        fused_activation=fused_activation,
        reverse_time=False,
    )
    bw_y = _run_unidirectional_sequence_lstm(
        x=x,
        wi=executor._resolve_tensor(op.inputs[18], env),
        wf=executor._resolve_tensor(op.inputs[19], env),
        wc=executor._resolve_tensor(op.inputs[20], env),
        wo=executor._resolve_tensor(op.inputs[21], env),
        ri=executor._resolve_tensor(op.inputs[22], env),
        rf=executor._resolve_tensor(op.inputs[23], env),
        rc=executor._resolve_tensor(op.inputs[24], env),
        ro=executor._resolve_tensor(op.inputs[25], env),
        bi=executor._resolve_tensor(op.inputs[29], env),
        bf=executor._resolve_tensor(op.inputs[30], env),
        bc=executor._resolve_tensor(op.inputs[31], env),
        bo=executor._resolve_tensor(op.inputs[32], env),
        h0=bw_h0,
        c0=bw_c0,
        projection_weights=_resolve_optional_by_index(executor, op, env, 33),
        projection_bias=_resolve_optional_by_index(executor, op, env, 34),
        cell_clip=cell_clip,
        proj_clip=proj_clip,
        time_major=time_major,
        fused_activation=fused_activation,
        reverse_time=True,
    )
    if _parse_bool_option(op.options.get("mergeOutputs", True), default=True):
        y = tf.concat([fw_y, bw_y], axis=-1)
        executor._assign_outputs(op, [y], env)
    else:
        stacked = tf.stack([fw_y, bw_y], axis=2 if time_major else 1)
        executor._assign_outputs(op, [stacked], env)


def _kernel_unidirectional_sequence_rnn(
    executor: _GraphExecutor,
    op: OperatorIR,
    env: Dict[str, tf.Tensor],
) -> None:
    if len(op.inputs) < 5:
        raise ModelIRSavedModelExportError(
            "UNIDIRECTIONAL_SEQUENCE_RNN expects 5 inputs."
        )
    x = executor._resolve_tensor(op.inputs[0], env)
    w = executor._resolve_tensor(op.inputs[1], env)
    r = executor._resolve_tensor(op.inputs[2], env)
    b = executor._resolve_tensor(op.inputs[3], env)
    h0 = _resolve_optional_by_index(executor, op, env, 4)
    time_major = _parse_bool_option(op.options.get("timeMajor", True), default=True)
    activation_name = str(op.options.get("fusedActivationFunction", "TANH"))

    x_time = x if time_major else tf.transpose(x, perm=[1, 0, 2])
    batch_size = tf.shape(x_time)[1]
    hidden_size = tf.shape(w)[0]
    if h0 is None:
        h0 = tf.zeros([batch_size, hidden_size], dtype=x.dtype)

    def _step(h_prev: tf.Tensor, x_t: tf.Tensor) -> tf.Tensor:
        pre = tf.linalg.matmul(x_t, w, transpose_b=True) + tf.linalg.matmul(h_prev, r, transpose_b=True) + b
        return _apply_activation_by_name(pre, activation_name)

    h_seq = tf.scan(fn=_step, elems=x_time, initializer=h0)
    if not time_major:
        h_seq = tf.transpose(h_seq, perm=[1, 0, 2])
    executor._assign_outputs(op, [h_seq], env)


def _register_supported_kernels() -> Dict[str, Callable[[_GraphExecutor, OperatorIR, Dict[str, tf.Tensor]], None]]:
    kernels: Dict[str, Callable[[_GraphExecutor, OperatorIR, Dict[str, tf.Tensor]], None]] = {}
    for op_type in _UNARY_KERNELS.keys():
        kernels[op_type] = _kernel_unary
    for op_type in _BINARY_KERNELS.keys():
        kernels[op_type] = _kernel_binary
    kernels["LOGICAL_NOT"] = _kernel_logical_not
    kernels["CAST"] = _kernel_cast
    kernels["RESHAPE"] = _kernel_reshape
    kernels["TRANSPOSE"] = _kernel_transpose
    kernels["CONCATENATION"] = _kernel_concat
    kernels["SQUEEZE"] = _kernel_squeeze
    kernels["EXPAND_DIMS"] = _kernel_expand_dims
    kernels["SPLIT"] = _kernel_split
    kernels["PACK"] = _kernel_pack
    kernels["UNPACK"] = _kernel_unpack
    kernels["SLICE"] = _kernel_slice
    kernels["STRIDED_SLICE"] = _kernel_strided_slice
    kernels["SHAPE"] = _kernel_shape
    kernels["FILL"] = _kernel_fill
    kernels["RANGE"] = _kernel_range
    kernels["SOFTMAX"] = _kernel_softmax
    kernels["SUM"] = _kernel_reduce(tf.reduce_sum)
    kernels["MEAN"] = _kernel_reduce(tf.reduce_mean)
    kernels["REDUCE_MAX"] = _kernel_reduce(tf.reduce_max)
    kernels["REDUCE_MIN"] = _kernel_reduce(tf.reduce_min)
    kernels["REDUCE_PROD"] = _kernel_reduce(tf.reduce_prod)
    kernels["REDUCE_ANY"] = _kernel_reduce(tf.reduce_any)
    kernels["PAD"] = _kernel_pad
    kernels["PADV2"] = _kernel_pad
    kernels["WHERE"] = _kernel_where
    kernels["SELECT"] = _kernel_where
    kernels["SELECT_V2"] = _kernel_where
    kernels["GATHER"] = _kernel_gather
    kernels["GATHER_ND"] = _kernel_gather_nd
    kernels["SCATTER_ND"] = _kernel_scatter_nd
    kernels["TILE"] = _kernel_tile
    kernels["BROADCAST_TO"] = _kernel_broadcast_to
    kernels["ARG_MAX"] = _kernel_arg(is_max=True)
    kernels["ARG_MIN"] = _kernel_arg(is_max=False)
    kernels["TOPK_V2"] = _kernel_topk
    kernels["RELU_N1_TO_1"] = _kernel_relu_clipped(-1.0, 1.0)
    kernels["RELU_0_TO_1"] = _kernel_relu_clipped(0.0, 1.0)
    kernels["LEAKY_RELU"] = _kernel_leaky_relu
    kernels["PRELU"] = _kernel_prelu
    kernels["L2_NORMALIZATION"] = _kernel_l2_norm
    kernels["LOCAL_RESPONSE_NORMALIZATION"] = _kernel_local_response_norm
    kernels["CUMSUM"] = _kernel_cumsum
    kernels["ONE_HOT"] = _kernel_one_hot
    kernels["REVERSE_V2"] = _kernel_reverse_v2
    kernels["DEPTH_TO_SPACE"] = _kernel_depth_to_space
    kernels["SPACE_TO_DEPTH"] = _kernel_space_to_depth
    kernels["MIRROR_PAD"] = _kernel_mirror_pad
    kernels["RANDOM_STANDARD_NORMAL"] = _kernel_random_standard_normal
    kernels["RANDOM_UNIFORM"] = _kernel_random_uniform
    kernels["REVERSE_SEQUENCE"] = _kernel_reverse_sequence
    kernels["FULLY_CONNECTED"] = _kernel_fully_connected
    kernels["BATCH_MATMUL"] = _kernel_batch_matmul
    kernels["CONV_2D"] = _kernel_conv2d
    kernels["DEPTHWISE_CONV_2D"] = _kernel_depthwise_conv2d
    kernels["TRANSPOSE_CONV"] = _kernel_transpose_conv
    kernels["CONV_3D"] = _kernel_conv3d
    kernels["CONV_3D_TRANSPOSE"] = _kernel_conv3d_transpose
    kernels["AVERAGE_POOL_2D"] = _kernel_pool2d(is_max_pool=False)
    kernels["MAX_POOL_2D"] = _kernel_pool2d(is_max_pool=True)
    kernels["RESIZE_BILINEAR"] = _kernel_resize(method="bilinear")
    kernels["RESIZE_NEAREST_NEIGHBOR"] = _kernel_resize(method="nearest")
    kernels["DEQUANTIZE"] = _kernel_dequantize
    kernels["QUANTIZE"] = _kernel_quantize
    kernels["UNIQUE"] = _kernel_unique
    kernels["NON_MAX_SUPPRESSION_V4"] = _kernel_non_max_suppression_v4
    kernels["NON_MAX_SUPPRESSION_V5"] = _kernel_non_max_suppression_v5
    kernels["UNIDIRECTIONAL_SEQUENCE_LSTM"] = _kernel_unidirectional_sequence_lstm
    kernels["BIDIRECTIONAL_SEQUENCE_LSTM"] = _kernel_bidirectional_sequence_lstm
    kernels["UNIDIRECTIONAL_SEQUENCE_RNN"] = _kernel_unidirectional_sequence_rnn
    kernels["WHILE"] = _kernel_while
    return kernels


_REGISTERED_KERNELS = _register_supported_kernels()
SUPPORTED_KERNEL_OP_TYPES: Set[str] = set(_REGISTERED_KERNELS.keys())


def get_supported_kernel_op_types() -> Set[str]:
    return set(SUPPORTED_KERNEL_OP_TYPES)


def _collect_unsupported_ops_for_model(model_ir: ModelIR) -> List[str]:
    model_ops = _collect_model_op_types(model_ir)
    unsupported = sorted(
        list(
            {
                op_type
                for op_type in model_ops
                if op_type not in SUPPORTED_KERNEL_OP_TYPES and op_type not in {"CUSTOM", "MODEL"}
            }
        )
    )
    return unsupported


class _SavedModelModule(tf.Module):
    def __init__(self, *, model_ir: ModelIR) -> None:
        module_name = re.sub(r"[^0-9A-Za-z_]", "_", str(model_ir.name))
        if module_name == "":
            module_name = "onnx2tf_model"
        if module_name[0].isdigit():
            module_name = f"m_{module_name}"
        super().__init__(name=module_name)
        self._model_ir = model_ir
        self._input_names = list(model_ir.inputs)
        self._output_names = list(model_ir.outputs)
        self._input_signature = self._build_input_signature()
        self._root_constants = self._build_root_constants()
        self._executor = _GraphExecutor(
            model_ir=self._model_ir,
            root_constants=self._root_constants,
            kernels=_REGISTERED_KERNELS,
        )

    def _build_root_constants(self) -> Dict[str, tf.Tensor]:
        constants: Dict[str, tf.Tensor] = {}
        for tensor_name, tensor in self._model_ir.tensors.items():
            if tensor.data is None:
                continue
            np_dtype = _numpy_dtype_from_tensor_ir(tensor)
            constants[str(tensor_name)] = tf.constant(
                np.asarray(tensor.data).astype(np_dtype, copy=False),
                dtype=_tf_dtype_from_tensor_ir(tensor),
                name=str(tensor_name),
            )
        return constants

    def _build_input_signature(self) -> List[tf.TensorSpec]:
        signature: List[tf.TensorSpec] = []
        for tensor_name in self._input_names:
            tensor = self._model_ir.tensors.get(str(tensor_name), None)
            if tensor is None:
                raise ModelIRSavedModelExportError(
                    f"ModelIR input tensor metadata is missing: input={tensor_name}"
                )
            spec = tf.TensorSpec(
                shape=_normalized_shape_signature(tensor),
                dtype=_tf_dtype_from_tensor_ir(tensor),
                name=str(tensor_name),
            )
            signature.append(spec)
        return signature

    def _serve_impl(self, *args: tf.Tensor) -> Dict[str, tf.Tensor]:
        inputs = {name: value for name, value in zip(self._input_names, args)}
        outputs = self._executor.run(inputs=inputs)
        return {name: outputs[name] for name in self._output_names}

    def build_signature_function(self) -> Callable[..., Dict[str, tf.Tensor]]:
        return tf.function(self._serve_impl, input_signature=self._input_signature)


def export_saved_model_from_model_ir(
    *,
    model_ir: ModelIR,
    output_folder_path: str,
) -> str:
    _ensure_no_custom_ops(model_ir)
    unsupported_ops = _collect_unsupported_ops_for_model(model_ir)
    if len(unsupported_ops) > 0:
        raise ModelIRSavedModelExportError(
            "ModelIR->SavedModel exporter does not support some op types in this model. "
            f"unsupported_op_types={unsupported_ops}"
        )

    module = _SavedModelModule(model_ir=model_ir)
    signature_fn = module.build_signature_function()
    tf.saved_model.save(
        module,
        output_folder_path,
        signatures={"serving_default": signature_fn},
    )
    return str(output_folder_path)
