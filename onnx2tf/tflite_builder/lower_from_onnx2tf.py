from __future__ import annotations

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
    ):
        self.model_ir = model_ir
        self.shape_map = shape_map
        self.dtype_map = dtype_map
        self.constants = constants
        self.allow_custom_ops = bool(allow_custom_ops)
        self.custom_op_allowlist = (
            list(custom_op_allowlist) if custom_op_allowlist is not None else None
        )
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
    def __init__(self, n: onnx.NodeProto):
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
        self.inputs = [type("In", (), {"name": i}) for i in n.input if i != ""]
        self.outputs = [type("Out", (), {"name": o}) for o in n.output if o != ""]


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


def build_op_coverage_report(
    *,
    onnx_graph: onnx.ModelProto,
    output_file_name: str,
    opset_min: int = 13,
    opset_max: int = 18,
    conversion_error: Optional[str] = None,
    allow_custom_ops: bool = False,
    custom_op_allowlist: Optional[List[str]] = None,
) -> Dict[str, Any]:
    try:
        onnx_graph = onnx.shape_inference.infer_shapes(onnx_graph)
    except Exception:
        pass

    shape_map, dtype_map = _extract_tensor_info(onnx_graph)
    constants: Dict[str, np.ndarray] = {}
    for ini in onnx_graph.graph.initializer:
        constants[ini.name] = np.asarray(numpy_helper.to_array(ini))

    model_ir = ModelIR(name=output_file_name)
    ctx = LoweringContext(
        model_ir=model_ir,
        shape_map=shape_map,
        dtype_map=dtype_map,
        constants=constants,
        allow_custom_ops=allow_custom_ops,
        custom_op_allowlist=custom_op_allowlist,
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
                [str(v) for v in custom_op_allowlist]
                if custom_op_allowlist is not None
                else None
            ),
            "candidate_count": int(len(custom_candidate_ops)),
        },
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
) -> ModelIR:
    try:
        onnx_graph = onnx.shape_inference.infer_shapes(onnx_graph)
    except Exception:
        pass

    shape_map, dtype_map = _extract_tensor_info(onnx_graph)
    constants: Dict[str, np.ndarray] = {}
    for ini in onnx_graph.graph.initializer:
        constants[ini.name] = np.asarray(numpy_helper.to_array(ini))

    model_ir = ModelIR(name=output_file_name)
    ctx = LoweringContext(
        model_ir=model_ir,
        shape_map=shape_map,
        dtype_map=dtype_map,
        constants=constants,
        allow_custom_ops=allow_custom_ops,
        custom_op_allowlist=custom_op_allowlist,
    )

    # Inputs
    initializer_names = {ini.name for ini in onnx_graph.graph.initializer}
    for graph_input in onnx_graph.graph.input:
        if graph_input.name in initializer_names:
            continue
        ctx.ensure_tensor(graph_input.name)
        model_ir.inputs.append(graph_input.name)

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

        wrapped = _NodeWrap(node)
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

    return model_ir
