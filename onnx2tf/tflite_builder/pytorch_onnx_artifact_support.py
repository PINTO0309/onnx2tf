from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

import onnx

from onnx2tf.tflite_builder.ir import (
    channel_first_logical_layout,
    channel_last_logical_layout,
)
from onnx2tf.tflite_builder.pytorch_layout_utils import (
    _perm_cf_to_cl,
    _perm_cl_to_cf,
    _read_onnx_squeeze_axes,
    _read_onnx_unsqueeze_axes,
)

from onnx2tf.tflite_builder.pytorch_onnx_layout_passes import (
    _onnx_fold_pad_layout_bridges_in_place,
    _onnx_fold_relu_layout_bridges_in_place,
    _onnx_fold_residual_add_layout_bridges_in_place,
)
from onnx2tf.tflite_builder.pytorch_onnx_optimizer import (
    _optimize_dynamo_exported_onnx_in_place,
)
from onnx2tf.tflite_builder.pytorch_onnx_utils import (
    _clear_onnx_graph_and_node_metadata_in_place,
    _onnx_repair_inferred_shapes_in_place,
)


def _read_onnx_transpose_perm(node: Any) -> Optional[List[int]]:
    if str(getattr(node, "op_type", "")) != "Transpose":
        return None
    for attr in list(getattr(node, "attribute", [])):
        if str(getattr(attr, "name", "")) != "perm":
            continue
        try:
            values = onnx.helper.get_attribute_value(attr)
        except Exception:
            return None
        try:
            return [int(v) for v in list(values)]
        except Exception:
            return None
        return None


def _is_onnx_boundary_layout_passthrough_node(
    *,
    node: Any,
    source_tensor_name: str,
) -> bool:
    passthrough_op_types = {
        "Abs",
        "Add",
        "Cast",
        "Clip",
        "Div",
        "Identity",
        "LeakyRelu",
        "Mul",
        "Relu",
        "Sigmoid",
        "Softmax",
        "Sub",
        "Tanh",
    }
    if str(getattr(node, "op_type", "")) not in passthrough_op_types:
        return False
    inputs = [str(v) for v in list(getattr(node, "input", []))]
    outputs = [str(v) for v in list(getattr(node, "output", []))]
    return len(outputs) == 1 and str(source_tensor_name) in set(inputs)


def _infer_public_layouts_from_onnx_graph(reference_onnx_graph: Any) -> Dict[str, str]:
    graph = getattr(reference_onnx_graph, "graph", None)
    if graph is None:
        return {}
    consumers: Dict[str, List[Any]] = {}
    producer_by_output: Dict[str, Any] = {}
    for node in list(graph.node):
        for output_name in list(getattr(node, "output", [])):
            producer_by_output[str(output_name)] = node
        for input_name in list(getattr(node, "input", [])):
            consumers.setdefault(str(input_name), []).append(node)

    def _walk_input_boundary(tensor_name: str, rank: int) -> Optional[str]:
        current_tensor_name = str(tensor_name)
        for _ in range(4):
            user_nodes = consumers.get(current_tensor_name, [])
            if len(user_nodes) != 1:
                return None
            node = user_nodes[0]
            perm = _read_onnx_transpose_perm(node)
            if perm == _perm_cl_to_cf(rank):
                return channel_last_logical_layout(rank)
            if perm == _perm_cf_to_cl(rank):
                return channel_first_logical_layout(rank)
            if not _is_onnx_boundary_layout_passthrough_node(
                node=node,
                source_tensor_name=current_tensor_name,
            ):
                return None
            current_tensor_name = str(list(getattr(node, "output", []))[0])
        return None

    def _walk_output_boundary(tensor_name: str, rank: int) -> Optional[str]:
        current_tensor_name = str(tensor_name)
        for _ in range(4):
            node = producer_by_output.get(current_tensor_name, None)
            if node is None:
                return None
            perm = _read_onnx_transpose_perm(node)
            if perm == _perm_cf_to_cl(rank):
                return channel_last_logical_layout(rank)
            if perm == _perm_cl_to_cf(rank):
                return channel_first_logical_layout(rank)
            inputs = [str(v) for v in list(getattr(node, "input", []))]
            if len(inputs) != 1:
                return None
            previous_tensor_name = inputs[0]
            if not _is_onnx_boundary_layout_passthrough_node(
                node=node,
                source_tensor_name=previous_tensor_name,
            ):
                return None
            current_tensor_name = previous_tensor_name
        return None

    public_layout_map: Dict[str, str] = {}
    for value_info in list(graph.input):
        tensor_name = str(getattr(value_info, "name", ""))
        if tensor_name == "":
            continue
        tensor_type = getattr(value_info, "type", None)
        tensor_shape = getattr(getattr(tensor_type, "tensor_type", None), "shape", None)
        dims = (
            list(getattr(tensor_shape, "dim", [])) if tensor_shape is not None else []
        )
        rank = len(dims)
        if rank not in {3, 4, 5}:
            continue
        inferred_layout = _walk_input_boundary(tensor_name, rank)
        if inferred_layout is not None:
            public_layout_map[tensor_name] = inferred_layout
    for value_info in list(graph.output):
        tensor_name = str(getattr(value_info, "name", ""))
        if tensor_name == "":
            continue
        tensor_type = getattr(value_info, "type", None)
        tensor_shape = getattr(getattr(tensor_type, "tensor_type", None), "shape", None)
        dims = (
            list(getattr(tensor_shape, "dim", [])) if tensor_shape is not None else []
        )
        rank = len(dims)
        if rank not in {3, 4, 5}:
            continue
        inferred_layout = _walk_output_boundary(tensor_name, rank)
        if inferred_layout is not None:
            public_layout_map[tensor_name] = inferred_layout
    return public_layout_map


def _infer_batchless_rank3_image_boundaries_from_onnx_graph(
    reference_onnx_graph: Any,
) -> Set[str]:
    graph = getattr(reference_onnx_graph, "graph", None)
    if graph is None:
        return set()
    consumers: Dict[str, List[Any]] = {}
    producer_by_output: Dict[str, Any] = {}
    for node in list(graph.node):
        for output_name in list(getattr(node, "output", [])):
            producer_by_output[str(output_name)] = node
        for input_name in list(getattr(node, "input", [])):
            consumers.setdefault(str(input_name), []).append(node)

    def _input_is_batchless_channel_first_image(tensor_name: str, rank: int) -> bool:
        if rank != 3:
            return False
        current_tensor_name = str(tensor_name)
        for _ in range(4):
            user_nodes = consumers.get(current_tensor_name, [])
            if len(user_nodes) != 1:
                return False
            node = user_nodes[0]
            axes = _read_onnx_unsqueeze_axes(node)
            if axes == [0]:
                return True
            if not _is_onnx_boundary_layout_passthrough_node(
                node=node,
                source_tensor_name=current_tensor_name,
            ):
                return False
            current_tensor_name = str(list(getattr(node, "output", []))[0])
        return False

    def _output_is_batchless_channel_first_image(tensor_name: str, rank: int) -> bool:
        if rank != 3:
            return False
        current_tensor_name = str(tensor_name)
        for _ in range(4):
            node = producer_by_output.get(current_tensor_name, None)
            if node is None:
                return False
            axes = _read_onnx_squeeze_axes(node)
            if axes == [0]:
                return True
            inputs = [str(v) for v in list(getattr(node, "input", []))]
            if len(inputs) != 1:
                return False
            previous_tensor_name = inputs[0]
            if not _is_onnx_boundary_layout_passthrough_node(
                node=node,
                source_tensor_name=previous_tensor_name,
            ):
                return False
            current_tensor_name = previous_tensor_name
        return False

    boundary_names: Set[str] = set()
    for value_info in list(graph.input):
        tensor_name = str(getattr(value_info, "name", ""))
        if tensor_name == "":
            continue
        tensor_type = getattr(value_info, "type", None)
        tensor_shape = getattr(getattr(tensor_type, "tensor_type", None), "shape", None)
        dims = (
            list(getattr(tensor_shape, "dim", [])) if tensor_shape is not None else []
        )
        if _input_is_batchless_channel_first_image(tensor_name, len(dims)):
            boundary_names.add(tensor_name)
    for value_info in list(graph.output):
        tensor_name = str(getattr(value_info, "name", ""))
        if tensor_name == "":
            continue
        tensor_type = getattr(value_info, "type", None)
        tensor_shape = getattr(getattr(tensor_type, "tensor_type", None), "shape", None)
        dims = (
            list(getattr(tensor_shape, "dim", [])) if tensor_shape is not None else []
        )
        if _output_is_batchless_channel_first_image(tensor_name, len(dims)):
            boundary_names.add(tensor_name)
    return boundary_names


def _onnx_model_uses_external_data(model: onnx.ModelProto) -> bool:
    return any(
        initializer.data_location == onnx.TensorProto.EXTERNAL
        for initializer in model.graph.initializer
    )


def _inspect_onnx_uses_external_data(onnx_path: Path) -> bool:
    model = onnx.load(str(onnx_path), load_external_data=False)
    return _onnx_model_uses_external_data(model)


def _restore_missing_onnx_output_shapes_from_package_metadata(
    model: onnx.ModelProto,
    *,
    package_dir: Path,
) -> None:
    metadata_path = package_dir / "metadata.json"
    if not metadata_path.exists():
        return
    try:
        with open(metadata_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)
    except Exception:
        return
    tensor_meta_map = metadata.get("tensors", {})
    if not isinstance(tensor_meta_map, dict):
        return
    for output in model.graph.output:
        tensor_type = getattr(output.type, "tensor_type", None)
        if tensor_type is None:
            continue
        shape = getattr(tensor_type, "shape", None)
        if shape is None or len(list(shape.dim)) != 0:
            continue
        tensor_meta = tensor_meta_map.get(str(output.name), {})
        if not isinstance(tensor_meta, dict):
            continue
        shape_values = tensor_meta.get("shape_signature", tensor_meta.get("shape", []))
        if not isinstance(shape_values, list):
            shape_values = tensor_meta.get("shape", [])
        if not isinstance(shape_values, list) or len(shape_values) == 0:
            continue
        for dim_index, raw_dim_value in enumerate(shape_values):
            output_dim = shape.dim.add()
            try:
                dim_value = int(raw_dim_value)
            except Exception:
                output_dim.dim_param = str(raw_dim_value)
                continue
            if dim_value > 0:
                output_dim.dim_value = dim_value
            else:
                sanitized_output_name = re.sub(
                    r"[^0-9A-Za-z_]", "_", str(output.name)
                ).strip("_")
                if sanitized_output_name == "":
                    sanitized_output_name = "output"
                output_dim.dim_param = f"{sanitized_output_name}_dim_{dim_index}"


def _sanitize_dynamo_exported_onnx_metadata(onnx_path: Path) -> None:
    external_data_sidecar_path = onnx_path.with_name(f"{onnx_path.name}.data")
    original_uses_external_data = _inspect_onnx_uses_external_data(onnx_path)
    model = onnx.load(str(onnx_path))
    _onnx_fold_relu_layout_bridges_in_place(model.graph)
    _onnx_fold_pad_layout_bridges_in_place(model.graph)
    _onnx_fold_residual_add_layout_bridges_in_place(model.graph)
    _onnx_repair_inferred_shapes_in_place(model)
    _onnx_fold_residual_add_layout_bridges_in_place(model.graph)
    _onnx_repair_inferred_shapes_in_place(model)
    _optimize_dynamo_exported_onnx_in_place(model)
    import onnx2tf.gs as gs

    model = gs.export_onnx(gs.import_onnx(model).cleanup().toposort())
    _onnx_repair_inferred_shapes_in_place(model)
    _restore_missing_onnx_output_shapes_from_package_metadata(
        model, package_dir=onnx_path.parent
    )
    del model.metadata_props[:]
    _clear_onnx_graph_and_node_metadata_in_place(model.graph)
    onnx.checker.check_model(model)
    if original_uses_external_data:
        onnx.save_model(
            model,
            str(onnx_path),
            save_as_external_data=True,
            all_tensors_to_one_file=True,
            location=external_data_sidecar_path.name,
            size_threshold=0,
        )
    else:
        onnx.save_model(
            model,
            str(onnx_path),
            save_as_external_data=False,
        )
    if (
        not _inspect_onnx_uses_external_data(onnx_path)
        and external_data_sidecar_path.exists()
    ):
        external_data_sidecar_path.unlink()
