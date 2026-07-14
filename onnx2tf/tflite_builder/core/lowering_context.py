from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np

from onnx2tf.tflite_builder.core.session import ConversionSession
from onnx2tf.tflite_builder.core.shape_resolution import (
    shape_hint_only_adds_singleton_or_dynamic_axes,
)
from onnx2tf.tflite_builder.ir import (
    ModelIR,
    OperatorIR,
    TensorIR,
    normalize_logical_layout,
    normalize_onnx_shape,
)
from onnx2tf.tflite_builder.tensor_buffer_builder import tflite_dtype_from_numpy


class LoweringContext:
    """Per-conversion state exposed to ONNX op-family lowerers."""

    def __init__(
        self,
        model_ir: ModelIR,
        shape_map: Dict[str, List[Any]],
        dtype_map: Dict[str, str],
        constants: Dict[str, np.ndarray],
        onnx_model: Optional[Any] = None,
        allow_custom_ops: bool = False,
        custom_op_allowlist: Optional[List[str]] = None,
        disable_group_convolution: bool = False,
        tensor_consumer_count: Optional[Dict[str, int]] = None,
        graph_output_names: Optional[List[str]] = None,
        output_nms_with_argmax: bool = False,
        switch_nms_version: str = "v4",
        mvn_epsilon: float = 1e-10,
        disable_suppression_flextranspose: bool = False,
        number_of_dimensions_after_flextranspose_compression: int = 6,
        disable_suppression_flexstridedslice: bool = False,
        number_of_dimensions_after_flexstridedslice_compression: int = 5,
        optimization_for_gpu_delegate: bool = False,
        replace_argmax_to_reducemax_and_indices_is_int64: bool = False,
        replace_argmax_to_reducemax_and_indices_is_float32: bool = False,
        replace_argmax_to_fused_argmax_and_indices_is_int64: bool = False,
        replace_argmax_to_fused_argmax_and_indices_is_float32: bool = False,
        fused_argmax_scale_ratio: float = 0.5,
        replace_to_pseudo_operators: Optional[List[str]] = None,
        session: Optional[ConversionSession] = None,
    ) -> None:
        self.session = session
        self.model_ir = model_ir
        self.shape_map = shape_map
        self.dtype_map = dtype_map
        self.constants = constants
        self.onnx_model = onnx_model
        self.allow_custom_ops = bool(allow_custom_ops)
        self.custom_op_allowlist = (
            list(custom_op_allowlist) if custom_op_allowlist is not None else None
        )
        self.disable_group_convolution = bool(disable_group_convolution)
        self.tensor_consumer_count = (
            session.tensor_consumer_count
            if session is not None
            else dict(tensor_consumer_count or {})
        )
        self.graph_output_names = set(graph_output_names or [])
        self.output_nms_with_argmax = bool(output_nms_with_argmax)
        self.mvn_epsilon = float(mvn_epsilon)
        self.disable_suppression_flextranspose = bool(disable_suppression_flextranspose)
        self.number_of_dimensions_after_flextranspose_compression = int(
            max(2, min(6, int(number_of_dimensions_after_flextranspose_compression)))
        )
        self.disable_suppression_flexstridedslice = bool(
            disable_suppression_flexstridedslice
        )
        self.number_of_dimensions_after_flexstridedslice_compression = int(
            max(1, min(5, int(number_of_dimensions_after_flexstridedslice_compression)))
        )
        self.optimization_for_gpu_delegate = bool(optimization_for_gpu_delegate)
        self.replace_argmax_to_reducemax_and_indices_is_int64 = bool(
            replace_argmax_to_reducemax_and_indices_is_int64
        )
        self.replace_argmax_to_reducemax_and_indices_is_float32 = bool(
            replace_argmax_to_reducemax_and_indices_is_float32
        )
        self.replace_argmax_to_fused_argmax_and_indices_is_int64 = bool(
            replace_argmax_to_fused_argmax_and_indices_is_int64
        )
        self.replace_argmax_to_fused_argmax_and_indices_is_float32 = bool(
            replace_argmax_to_fused_argmax_and_indices_is_float32
        )
        self.fused_argmax_scale_ratio = float(fused_argmax_scale_ratio)
        self.argmax_mode = "none"
        if self.replace_argmax_to_reducemax_and_indices_is_int64:
            self.argmax_mode = "reducemax_int64"
        elif self.replace_argmax_to_reducemax_and_indices_is_float32:
            self.argmax_mode = "reducemax_float32"
        elif self.replace_argmax_to_fused_argmax_and_indices_is_int64:
            self.argmax_mode = "fused_int64"
        elif self.replace_argmax_to_fused_argmax_and_indices_is_float32:
            self.argmax_mode = "fused_float32"
        self.fused_argmax_restore_shapes: Dict[str, Dict[str, Any]] = {}
        nms_version = str(switch_nms_version).strip().lower()
        if nms_version not in {"v4", "v5"}:
            raise ValueError(
                "switch_nms_version must be 'v4' or 'v5'. "
                f"got: {switch_nms_version}"
            )
        self.switch_nms_version = nms_version
        self.replace_to_pseudo_operators = {
            str(value).strip().lower()
            for value in (replace_to_pseudo_operators or [])
            if str(value).strip()
        }
        self._serial = 0
        self.ir_tensor_producers: Dict[str, OperatorIR] = {}
        self.onnx_tensor_consumers: Dict[str, List[Any]] = (
            session.graph_index.consumers if session is not None else {}
        )
        self.onnx_tensor_producers: Dict[str, Any] = (
            session.graph_index.producers if session is not None else {}
        )
        if session is None and onnx_model is not None and hasattr(onnx_model, "graph"):
            for graph_node in onnx_model.graph.node:
                for output_name in graph_node.output:
                    key = str(output_name).strip()
                    if key:
                        self.onnx_tensor_producers[key] = graph_node
                for input_name in graph_node.input:
                    key = str(input_name).strip()
                    if key:
                        self.onnx_tensor_consumers.setdefault(key, []).append(graph_node)

    def _next_name(self, base: str) -> str:
        self._serial += 1
        return f"{base}_{self._serial}"

    def get_tensor_shape(self, name: str) -> List[int]:
        if name in self.model_ir.tensors:
            return list(self.model_ir.tensors[name].shape)
        shape, _ = normalize_onnx_shape(self.shape_map.get(name))
        return shape

    def get_tensor_dtype(self, name: str) -> str:
        if name in self.model_ir.tensors:
            return self.model_ir.tensors[name].dtype
        return self.dtype_map.get(name, "FLOAT32")

    def get_constant_array(self, name: str) -> Optional[np.ndarray]:
        if name in self.constants:
            return self.constants[name]
        tensor = self.model_ir.tensors.get(name)
        return tensor.data if tensor is not None and isinstance(tensor.data, np.ndarray) else None

    def _record_layout(self, name: str) -> None:
        if self.session is None:
            return
        tensor = self.model_ir.tensors[name]
        self.session.layout_state.set(
            name,
            logical=tensor.logical_layout,
            physical=tensor.physical_layout,
        )

    def set_tensor_layout(
        self,
        name: str,
        *,
        logical: Optional[str] = None,
        physical: Optional[str] = None,
    ) -> None:
        """Update tensor layout metadata and the Session-owned layout state."""

        tensor_name = str(name)
        if tensor_name not in self.model_ir.tensors:
            raise KeyError(
                f"Tensor layout cannot be updated before creation: {tensor_name}"
            )
        tensor = self.model_ir.tensors[tensor_name]
        if logical is not None:
            tensor.logical_layout = normalize_logical_layout(logical)
        if physical is not None:
            tensor.physical_layout = normalize_logical_layout(physical)
        self._record_layout(tensor_name)

    def ensure_tensor(
        self,
        name: str,
        dtype: Optional[str] = None,
        shape: Optional[List[int]] = None,
    ) -> str:
        if not name:
            raise ValueError("Tensor name must not be empty in flatbuffer_direct lowering.")
        if name in self.model_ir.tensors:
            tensor = self.model_ir.tensors[name]
            rank_hint = self.shape_map.get(name, None)
            hinted_rank = len(rank_hint) if isinstance(rank_hint, list) else 0
            current_shape = [int(v) for v in list(tensor.shape)]
            if int(hinted_rank) > len(current_shape):
                producer = self.ir_tensor_producers.get(str(name), None)
                current_signature = (
                    [int(v) for v in list(tensor.shape_signature)]
                    if tensor.shape_signature is not None
                    else list(current_shape)
                )
                preserve_producer_rank = bool(
                    producer is not None
                    and len(current_signature) == len(current_shape)
                    and all(int(value) > 0 for value in current_signature)
                    and shape_hint_only_adds_singleton_or_dynamic_axes(
                        resolved_shape=current_shape,
                        shape_hint=rank_hint,
                    )
                )
                if not preserve_producer_rank:
                    missing_rank = int(hinted_rank) - len(current_shape)
                    tensor.shape = [1 for _ in range(missing_rank)] + current_shape
                    tensor.shape_signature = [
                        -1 for _ in range(missing_rank)
                    ] + current_signature
            self._record_layout(name)
            return name
        resolved_shape, signature = normalize_onnx_shape(
            self.shape_map.get(name) if shape is None else shape
        )
        self.model_ir.tensors[name] = TensorIR(
            name=name,
            dtype=dtype or self.dtype_map.get(name, "FLOAT32"),
            shape=list(resolved_shape),
            shape_signature=list(signature),
            data=self.constants.get(name),
            onnx_tensor_name=name,
        )
        self._record_layout(name)
        return name

    def add_const_tensor(self, base_name: str, data: np.ndarray) -> str:
        name = base_name if base_name not in self.model_ir.tensors else self._next_name(base_name)
        value = np.asarray(data)
        shape, signature = normalize_onnx_shape(list(value.shape))
        self.model_ir.tensors[name] = TensorIR(
            name=name,
            dtype=tflite_dtype_from_numpy(value.dtype),
            shape=shape,
            shape_signature=signature,
            data=value,
        )
        self._record_layout(name)
        self.constants[name] = value
        return name

    def add_intermediate_tensor(self, base_name: str, dtype: str, shape: List[int]) -> str:
        if not base_name:
            raise ValueError("Tensor name must not be empty in flatbuffer_direct lowering.")
        name = base_name if base_name not in self.model_ir.tensors else self._next_name(base_name)
        resolved_shape, signature = normalize_onnx_shape(shape)
        self.model_ir.tensors[name] = TensorIR(
            name=name,
            dtype=dtype,
            shape=resolved_shape,
            shape_signature=signature,
        )
        self._record_layout(name)
        return name

    def add_operator(self, op: OperatorIR) -> None:
        self.model_ir.operators.append(op)
        for output_name in op.outputs:
            if str(output_name):
                self.ir_tensor_producers[str(output_name)] = op
