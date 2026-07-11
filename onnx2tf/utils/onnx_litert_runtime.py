from __future__ import annotations

import ast
import os
import re
import subprocess
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import onnx
import psutil
from ai_edge_litert.interpreter import Interpreter
from onnx.external_data_helper import uses_external_data
from onnx.serialization import ProtoSerializer

import onnx2tf.gs as gs
from onnx2tf.utils.logging import info
from onnx2tf.utils.onnxruntime_compat import prepare_onnx_graph_for_onnxruntime
from onnx2tf.utils.tempdir_cleanup import make_managed_tempdir

try:
    import cv2
except Exception:
    cv2 = None

try:
    import onnxruntime as ort
except Exception:
    ort = None


_DEFAULT_DUMMY_SHAPE_HINTS: Optional[List[str]] = None
_DEFAULT_DUMMY_VALUE_HINTS: Optional[List[str]] = None
_DEFAULT_TFLITE_SCHEMA_REPOSITORY: str = "google-ai-edge/LiteRT"
_DEFAULT_TFLITE_SCHEMA_TAG: str = "v2.1.2"
_DEFAULT_TFLITE_SCHEMA_RELATIVE_PATH: str = "tflite/converter/schema/schema.fbs"
_BUNDLED_SCHEMA_DIR: str = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "tflite_builder",
    "schema",
)
_BUNDLED_SCHEMA_FBS_PATH: str = os.path.join(_BUNDLED_SCHEMA_DIR, "schema.fbs")
_BUNDLED_SCHEMA_PY_PATH: str = os.path.join(_BUNDLED_SCHEMA_DIR, "schema_generated.py")


def set_dummy_shape_hints(shape_hints: Optional[List[str]]) -> None:
    global _DEFAULT_DUMMY_SHAPE_HINTS
    _DEFAULT_DUMMY_SHAPE_HINTS = shape_hints


def set_dummy_value_hints(value_hints: Optional[List[str]]) -> None:
    global _DEFAULT_DUMMY_VALUE_HINTS
    _DEFAULT_DUMMY_VALUE_HINTS = value_hints


def get_tflite_schema_fbs_url() -> str:
    repository = os.environ.get(
        "ONNX2TF_TFLITE_SCHEMA_REPOSITORY",
        _DEFAULT_TFLITE_SCHEMA_REPOSITORY,
    ).strip().strip("/")
    schema_tag = os.environ.get(
        "ONNX2TF_TFLITE_SCHEMA_TAG",
        _DEFAULT_TFLITE_SCHEMA_TAG,
    ).strip()
    schema_relative_path = os.environ.get(
        "ONNX2TF_TFLITE_SCHEMA_RELATIVE_PATH",
        _DEFAULT_TFLITE_SCHEMA_RELATIVE_PATH,
    ).strip().lstrip("/")
    if repository == "":
        repository = _DEFAULT_TFLITE_SCHEMA_REPOSITORY
    if schema_tag == "":
        schema_tag = _DEFAULT_TFLITE_SCHEMA_TAG
    if schema_relative_path == "":
        schema_relative_path = _DEFAULT_TFLITE_SCHEMA_RELATIVE_PATH
    return f"https://raw.githubusercontent.com/{repository}/{schema_tag}/{schema_relative_path}"


def _get_default_tflite_schema_fbs_url() -> str:
    return (
        f"https://raw.githubusercontent.com/"
        f"{_DEFAULT_TFLITE_SCHEMA_REPOSITORY}/"
        f"{_DEFAULT_TFLITE_SCHEMA_TAG}/"
        f"{_DEFAULT_TFLITE_SCHEMA_RELATIVE_PATH}"
    )


def _is_default_tflite_schema_source() -> bool:
    return get_tflite_schema_fbs_url() == _get_default_tflite_schema_fbs_url()


def _copy_schema_artifact_if_needed(
    *,
    src_path: str,
    dst_path: str,
    force_copy: bool = False,
) -> bool:
    if not os.path.isfile(src_path) or os.path.getsize(src_path) <= 0:
        return False
    if force_copy or not os.path.isfile(dst_path) or os.path.getsize(dst_path) <= 0:
        import shutil

        shutil.copyfile(src_path, dst_path)
        return True
    return True


def ensure_tflite_schema_artifacts(
    *,
    output_folder_path: str,
    force_regenerate_schema_py: bool = False,
) -> Tuple[str, str]:
    os.makedirs(output_folder_path, exist_ok=True)

    schema_fbs_path = os.path.join(output_folder_path, "schema.fbs")
    schema_py_path = os.path.join(output_folder_path, "schema_generated.py")
    default_schema_source = _is_default_tflite_schema_source()
    bundled_schema_available = (
        os.path.isfile(_BUNDLED_SCHEMA_PY_PATH)
        and os.path.getsize(_BUNDLED_SCHEMA_PY_PATH) > 0
        and os.path.isfile(_BUNDLED_SCHEMA_FBS_PATH)
        and os.path.getsize(_BUNDLED_SCHEMA_FBS_PATH) > 0
    )
    if not bundled_schema_available:
        raise RuntimeError(
            "Bundled TFLite schema artifacts are missing. "
            f"expected={_BUNDLED_SCHEMA_DIR}"
        )

    if not default_schema_source:
        print(
            "WARNING: ONNX2TF_TFLITE_SCHEMA_TAG/REPOSITORY/RELATIVE_PATH are set, "
            "but schema override is no longer used. Falling back to bundled schema artifacts.",
            flush=True,
        )

    _copy_schema_artifact_if_needed(
        src_path=_BUNDLED_SCHEMA_PY_PATH,
        dst_path=schema_py_path,
        force_copy=force_regenerate_schema_py,
    )
    _copy_schema_artifact_if_needed(
        src_path=_BUNDLED_SCHEMA_FBS_PATH,
        dst_path=schema_fbs_path,
        force_copy=False,
    )
    return schema_fbs_path, schema_py_path


def _parse_value_hint_scalar(value: str) -> Optional[Any]:
    try:
        parsed = ast.literal_eval(value)
    except Exception:
        try:
            parsed = float(value)
        except Exception:
            return None
    if isinstance(parsed, (list, tuple, dict, set, np.ndarray)):
        return None
    if isinstance(parsed, (int, float, bool, np.number)):
        return parsed
    return None


def _parse_value_hints(
    value_hints: Optional[List[str]]
) -> Tuple[Dict[str, Any], Optional[Any], bool]:
    if not value_hints:
        return {}, None, False
    hints: Dict[str, Any] = {}
    default_value: Optional[Any] = None
    for hint in value_hints:
        if not isinstance(hint, str):
            continue
        parts = hint.rsplit(":", 1)
        if len(parts) != 2:
            continue
        key = parts[0].strip()
        parsed = _parse_value_hint_scalar(parts[1].strip())
        if parsed is None:
            continue
        if key == "*":
            default_value = parsed
        elif key != "":
            hints[key] = parsed
    return hints, default_value, default_value is not None


def _make_safe_ort_memmap_filename(
    *,
    output_index: int,
    output_name: str,
) -> str:
    safe_name = re.sub(r"[^0-9A-Za-z._-]+", "_", str(output_name))
    safe_name = re.sub(r"_+", "_", safe_name).strip("._")
    if safe_name == "":
        safe_name = "output"
    max_stem_len = 220
    stem = f"ort_output_{int(output_index)}_{safe_name}"
    if len(stem) > max_stem_len:
        stem = stem[:max_stem_len].rstrip("._")
    return f"{stem}.npy"


def check_cuda_enabled() -> bool:
    try:
        output = subprocess.check_output("nvidia-smi", shell=True)
        return "nvidia-smi" in output.decode().lower()
    except Exception:
        return False


def check_model_has_external_data(model: onnx.ModelProto) -> bool:
    def iter_tensors_in_graph(g):
        for t in g.initializer:
            yield t
        for t in g.sparse_initializer:
            yield t
        for n in g.node:
            for a in n.attribute:
                if a.type == onnx.AttributeProto.TENSOR:
                    yield a.t
                elif a.type == onnx.AttributeProto.TENSORS:
                    for t in a.tensors:
                        yield t
                elif a.type == onnx.AttributeProto.GRAPH:
                    yield from iter_tensors_in_graph(a.g)
                elif a.type == onnx.AttributeProto.GRAPHS:
                    for sg in a.graphs:
                        yield from iter_tensors_in_graph(sg)

    return any(
        isinstance(t, onnx.TensorProto) and uses_external_data(t)
        for t in iter_tensors_in_graph(model.graph)
    )


def check_has_external_data(input_onnx_file_path: str) -> bool:
    model = onnx.load(input_onnx_file_path, load_external_data=False)
    return check_model_has_external_data(model)


def _resize_test_data_to_nchw(
    *,
    test_data_nhwc: np.ndarray,
    input_shape: tuple[int, ...],
    np_input_dtype: np.dtype,
) -> np.ndarray:
    if len(input_shape) != 4:
        raise ValueError(
            "test_data_nhwc-backed dummy inference currently supports rank-4 inputs only. "
            f"shape={input_shape}"
        )
    if cv2 is None:
        raise ImportError(
            "opencv-python is required for test_data_nhwc-backed dummy inference without TensorFlow."
        )

    resized = [
        cv2.resize(
            np.asarray(image),
            dsize=(int(input_shape[3]), int(input_shape[2])),
            interpolation=cv2.INTER_LINEAR,
        )
        for image in np.asarray(test_data_nhwc)
    ]
    resized_nhwc = np.asarray(resized, dtype=np.float32)
    return np.transpose(resized_nhwc, (0, 3, 1, 2)).astype(np_input_dtype)


def dummy_onnx_inference(
    *,
    onnx_graph: onnx.ModelProto,
    output_names: List[str],
    test_data_nhwc: Optional[np.ndarray] = None,
    custom_input_op_name_np_data_path: Optional[Any] = None,
    tf_layers_dict: Optional[Dict] = None,
    use_cuda: bool = False,
    disable_strict_mode: bool = False,
    enable_ort_output_memmap: bool = False,
    ort_output_memmap_dir: Optional[str] = None,
    ort_output_memmap_paths_for_cleanup: Optional[List[str]] = None,
    ort_disable_graph_optimization: bool = False,
    shape_hints: Optional[List[str]] = None,
    value_hints: Optional[List[str]] = None,
    input_datas_for_validation: Optional[Dict[str, np.ndarray]] = None,
) -> List[np.ndarray]:
    if shape_hints is None:
        shape_hints = _DEFAULT_DUMMY_SHAPE_HINTS
    if value_hints is None:
        value_hints = _DEFAULT_DUMMY_VALUE_HINTS
    if ort is None:
        raise ImportError("onnxruntime is required for dummy_onnx_inference.")

    onnx_graph, _ = prepare_onnx_graph_for_onnxruntime(onnx_graph)

    domain: str = onnx_graph.domain
    ir_version: int = onnx_graph.ir_version
    meta_data = {"domain": domain, "ir_version": ir_version}
    metadata_props = None
    if hasattr(onnx_graph, "metadata_props"):
        metadata_props = onnx_graph.metadata_props
    gs_graph = gs.import_onnx(onnx_graph)

    for node in gs_graph.nodes:
        input_shape = node.inputs[0].shape if len(node.inputs) > 0 else None
        input_rank = len(input_shape) if input_shape is not None else 0
        default_axes = [axis for axis in range(1, input_rank)] if input_rank > 1 else [0]
        if (
            gs_graph.opset <= 17
            and node.op in ["ReduceMax", "ReduceMean", "ReduceMin", "ReduceProd"]
            and "axes" not in node.attrs
        ):
            node.attrs["axes"] = default_axes
        elif (
            gs_graph.opset > 17
            and node.op in ["ReduceMax", "ReduceMean", "ReduceMin", "ReduceProd"]
            and len(node.inputs) == 1
        ):
            node.inputs.append(
                gs.Constant(
                    f"{node.name}_axes",
                    values=np.asarray(default_axes, dtype=np.int64),
                )
            )
        elif gs_graph.opset <= 12 and node.op in ["ReduceSum"] and "axes" not in node.attrs:
            node.attrs["axes"] = default_axes
        elif gs_graph.opset > 12 and node.op in ["ReduceSum"] and len(node.inputs) == 1:
            node.inputs.append(
                gs.Constant(
                    f"{node.name}_axes",
                    values=np.asarray(default_axes, dtype=np.int64),
                )
            )

    inferred_output_dtype_by_name: Dict[str, Any] = {}
    producer_op_type_by_output_name: Dict[str, str] = {}
    initializer_dtype_by_name: Dict[str, Any] = {}
    value_info_dtype_by_name: Dict[str, Any] = {}
    for initializer in onnx_graph.graph.initializer:
        try:
            initializer_dtype_by_name[initializer.name] = np.asarray(
                onnx.numpy_helper.to_array(initializer)
            ).dtype
        except Exception:
            continue
    for value_info in (
        list(onnx_graph.graph.value_info)
        + list(onnx_graph.graph.output)
        + list(onnx_graph.graph.input)
    ):
        try:
            tensor_type = value_info.type.tensor_type
            if int(tensor_type.elem_type) != 0:
                value_info_dtype_by_name[value_info.name] = onnx.helper.tensor_dtype_to_np_dtype(
                    int(tensor_type.elem_type)
                )
        except Exception:
            continue

    def _resolve_dtype_from_tensor_name(tensor_name: str) -> Optional[Any]:
        if tensor_name in initializer_dtype_by_name:
            return initializer_dtype_by_name[tensor_name]
        if tensor_name in value_info_dtype_by_name:
            return value_info_dtype_by_name[tensor_name]
        return None

    def _resolve_qlinear_output_zero_point_input_index(
        node_op_type: str,
        input_count: int,
    ) -> Optional[int]:
        if node_op_type == "QuantizeLinear":
            return 2 if input_count >= 3 else None
        if node_op_type == "QLinearConcat":
            return 1 if input_count >= 2 else None
        if node_op_type in {"QLinearConv", "QLinearMatMul", "QLinearAdd", "QLinearMul"}:
            return 7 if input_count >= 8 else None
        if node_op_type == "QGemm":
            return 8 if input_count >= 9 else None
        if node_op_type in {
            "QLinearLeakyRelu",
            "QLinearSigmoid",
            "QLinearSoftmax",
            "QLinearAveragePool",
            "QLinearGlobalAveragePool",
        }:
            return 4 if input_count >= 5 else None
        if node_op_type.startswith("QLinear") and input_count >= 1:
            return input_count - 1
        return None

    for node in onnx_graph.graph.node:
        node_op_type = str(node.op_type)
        zp_index = _resolve_qlinear_output_zero_point_input_index(
            node_op_type=node_op_type,
            input_count=len(node.input),
        )
        zero_point_dtype = None
        if zp_index is not None:
            zero_point_dtype = _resolve_dtype_from_tensor_name(str(node.input[zp_index]))
        for node_output_name in node.output:
            if not node_output_name:
                continue
            producer_op_type_by_output_name[node_output_name] = node_op_type
            if zero_point_dtype is not None:
                inferred_output_dtype_by_name[node_output_name] = zero_point_dtype
            elif node_output_name in value_info_dtype_by_name:
                inferred_output_dtype_by_name[node_output_name] = value_info_dtype_by_name[
                    node_output_name
                ]

    passthrough_ops = {
        "Transpose",
        "Reshape",
        "Split",
        "Resize",
        "Identity",
        "Squeeze",
        "Unsqueeze",
        "Flatten",
        "Slice",
        "Pad",
    }
    known_dtype_by_tensor: Dict[str, Any] = {}
    known_dtype_by_tensor.update(initializer_dtype_by_name)
    known_dtype_by_tensor.update(value_info_dtype_by_name)
    known_dtype_by_tensor.update(inferred_output_dtype_by_name)
    changed = True
    while changed:
        changed = False
        for node in onnx_graph.graph.node:
            node_op_type = str(node.op_type)
            if node_op_type not in passthrough_ops or len(node.input) == 0:
                continue
            src_dtype = known_dtype_by_tensor.get(str(node.input[0]), None)
            if src_dtype is None:
                continue
            for node_output_name in node.output:
                if not node_output_name:
                    continue
                if known_dtype_by_tensor.get(node_output_name, None) is None:
                    known_dtype_by_tensor[node_output_name] = src_dtype
                    changed = True
    for tensor_name, tensor_dtype in known_dtype_by_tensor.items():
        if tensor_name not in inferred_output_dtype_by_name:
            inferred_output_dtype_by_name[tensor_name] = tensor_dtype

    gs_graph.outputs = []
    for graph_node in gs_graph.nodes:
        for node_output in graph_node.outputs:
            if node_output.name in output_names:
                if node_output.dtype is None:
                    inferred_dtype = inferred_output_dtype_by_name.get(node_output.name, None)
                    if inferred_dtype is None:
                        producer_op_type = producer_op_type_by_output_name.get(node_output.name, "")
                        if producer_op_type == "DequantizeLinear":
                            inferred_dtype = np.float32
                        elif (
                            producer_op_type == "QuantizeLinear"
                            or producer_op_type.startswith("QLinear")
                            or producer_op_type == "QGemm"
                        ):
                            inferred_dtype = np.int8
                        else:
                            inferred_dtype = np.float32
                    node_output.dtype = inferred_dtype
                gs_graph.outputs.append(node_output)

    new_onnx_graph = gs.export_onnx(graph=gs_graph, do_type_check=False, **meta_data)
    if metadata_props is not None:
        new_onnx_graph.metadata_props.extend(metadata_props)
    # ONNX GraphSurgeon does not preserve node domains on export. Reapply the
    # shared ORT compatibility mapping to the graph that is actually executed.
    new_onnx_graph, _ = prepare_onnx_graph_for_onnxruntime(new_onnx_graph)

    tmp_onnx_path = ""
    tmp_onnx_external_weights_path = ""
    try:
        serializer: ProtoSerializer = onnx._get_serializer(fmt="protobuf")
        serialized_graph = serializer.serialize_proto(proto=new_onnx_graph)
    except ValueError:
        tmp_onnx_path = "tmp.onnx"
        tmp_onnx_external_weights_path = "tmp_external.weights"
        onnx.save(
            proto=new_onnx_graph,
            f=tmp_onnx_path,
            save_as_external_data=True,
            location=tmp_onnx_external_weights_path,
        )
        serialized_graph = tmp_onnx_path

    sess_options = ort.SessionOptions()
    logical_cpu_count = psutil.cpu_count(logical=True) or 1
    sess_options.intra_op_num_threads = max(1, logical_cpu_count - 1)
    if bool(ort_disable_graph_optimization):
        try:
            sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
        except Exception:
            pass

    if use_cuda and check_cuda_enabled():
        try:
            onnx_session = ort.InferenceSession(
                path_or_bytes=serialized_graph,
                sess_options=sess_options,
                providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
            )
        except Exception:
            onnx_session = ort.InferenceSession(
                path_or_bytes=serialized_graph,
                sess_options=sess_options,
                providers=["CPUExecutionProvider"],
            )
    else:
        onnx_session = ort.InferenceSession(
            path_or_bytes=serialized_graph,
            sess_options=sess_options,
            providers=["CPUExecutionProvider"],
        )

    onnx_inputs = gs_graph.inputs
    input_names: List[str] = [inp.name for inp in onnx_inputs]
    input_sizes: List[List[Optional[Union[int, str]]]] = [
        list(inp.shape) if inp.shape is not None else []
        for inp in onnx_inputs
    ]

    if shape_hints is None:
        first_input_batch_dim: Optional[int] = None
        if len(input_sizes) > 0 and len(input_sizes[0]) > 0:
            first_dim = input_sizes[0][0]
            if isinstance(first_dim, int):
                first_input_batch_dim = first_dim
        new_input_sizes = []
        for input_size in input_sizes:
            new_input_size = []
            for idx, dim in enumerate(input_size):
                if (
                    idx == 0
                    and first_input_batch_dim is not None
                    and len(input_sizes[0]) == len(input_size)
                    and (dim is None or isinstance(dim, str))
                ):
                    new_input_size.append(first_input_batch_dim)
                elif dim is None or isinstance(dim, str):
                    new_input_size.append(1)
                else:
                    new_input_size.append(dim)
            new_input_sizes.append(new_input_size)
        input_sizes = new_input_sizes
    else:
        shape_hints_dict = {}
        for hint in shape_hints:
            parts = hint.rsplit(":", 1)
            if len(parts) == 2:
                shape_hints_dict[parts[0]] = [int(val) for val in parts[1].split(",")]
        for i, (input_name, original_shape) in enumerate(zip(input_names, input_sizes)):
            if input_name in shape_hints_dict:
                updated_shape = shape_hints_dict[input_name]
                for j, (orig_dim, hint_dim) in enumerate(zip(original_shape, updated_shape)):
                    if orig_dim is not None and not isinstance(orig_dim, str):
                        updated_shape[j] = orig_dim
                    else:
                        updated_shape[j] = hint_dim
                input_sizes[i] = updated_shape

    input_sizes = [
        [int(dim) if not isinstance(dim, str) and dim is not None else 1 for dim in input_size]
        for input_size in input_sizes
    ]

    input_dtypes: List[Any] = [inp.dtype for inp in onnx_inputs]
    input_size_map = {name: tuple(size) for name, size in zip(input_names, input_sizes)}
    input_datas: Dict[str, np.ndarray] = {}
    value_hints_dict, default_value, has_default = _parse_value_hints(value_hints)

    if custom_input_op_name_np_data_path:
        for param in custom_input_op_name_np_data_path:
            input_op_name = str(param[0])
            numpy_file_path = str(param[1])
            custom_input_data: np.ndarray = np.load(numpy_file_path)
            input_op_info: Optional[Dict[str, Any]] = (
                tf_layers_dict.get(input_op_name, None)
                if isinstance(tf_layers_dict, dict)
                else None
            )
            if input_op_info is not None:
                ncw_nchw_ncdhw_perm: Optional[List[int]] = input_op_info.get(
                    "ncw_nchw_ncdhw_perm",
                    None,
                )
                if ncw_nchw_ncdhw_perm is not None:
                    expected_shape = input_size_map.get(input_op_name, tuple(custom_input_data.shape))
                    if tuple(custom_input_data.shape) != expected_shape:
                        permuted_shape = tuple(
                            custom_input_data.shape[i] for i in ncw_nchw_ncdhw_perm
                        )
                        if permuted_shape == expected_shape:
                            custom_input_data = custom_input_data.transpose(ncw_nchw_ncdhw_perm)
                onnx_batch_size = input_op_info["shape"][0]
                cdata_batch_size = custom_input_data.shape[0]
                if (
                    isinstance(onnx_batch_size, int)
                    and onnx_batch_size != cdata_batch_size
                    and cdata_batch_size > 1
                ):
                    custom_input_data = custom_input_data[0:onnx_batch_size, ...]
                elif isinstance(onnx_batch_size, str) and cdata_batch_size > 1:
                    custom_input_data = custom_input_data[0:1, ...]

            input_datas[input_op_name] = custom_input_data
    else:
        for input_name, input_size, input_dtype in zip(input_names, input_sizes, input_dtypes):
            input_shape = tuple(dim if isinstance(dim, int) else 1 for dim in input_size)
            np_input_dtype = np.dtype(input_dtype)
            hint_value = value_hints_dict.get(
                input_name,
                default_value if has_default else None,
            )
            if hint_value is not None:
                input_datas[input_name] = np.full(input_shape, hint_value, dtype=np_input_dtype)
            elif test_data_nhwc is None:
                if np.issubdtype(np_input_dtype, np.integer) or np.issubdtype(
                    np_input_dtype,
                    np.bool_,
                ):
                    input_datas[input_name] = np.zeros(input_shape, dtype=np_input_dtype)
                else:
                    input_datas[input_name] = np.ones(input_shape, dtype=np_input_dtype)
            else:
                input_datas[input_name] = _resize_test_data_to_nchw(
                    test_data_nhwc=np.asarray(test_data_nhwc),
                    input_shape=input_shape,
                    np_input_dtype=np_input_dtype,
                )

    if input_datas_for_validation is not None:
        input_datas_for_validation.update(input_datas)

    dtype_sizes = {
        np.dtype("float16"): 2,
        np.dtype("float32"): 4,
        np.dtype("float64"): 8,
        np.dtype("uint8"): 1,
        np.dtype("uint16"): 2,
        np.dtype("uint32"): 4,
        np.dtype("uint64"): 8,
        np.dtype("int8"): 1,
        np.dtype("int16"): 2,
        np.dtype("int32"): 4,
        np.dtype("int64"): 8,
        np.dtype("bool_"): 1,
    }
    total_output_size = 0
    for gs_graph_output in gs_graph.outputs:
        op_output_size = 1
        if gs_graph_output.shape is not None:
            for s in gs_graph_output.shape:
                if isinstance(s, (int, np.integer)):
                    op_output_size *= s
            total_output_size += op_output_size * dtype_sizes.get(gs_graph_output.dtype, 4)

    mem_available = psutil.virtual_memory().available * 0.80 // 1024 // 1024 // 1024
    total_output_size_gb = total_output_size // 1024 // 1024 // 1024
    use_memmap_outputs = bool(enable_ort_output_memmap)
    if (
        not disable_strict_mode
        and total_output_size_gb > mem_available
        and not use_memmap_outputs
    ):
        if tmp_onnx_path:
            os.remove(tmp_onnx_path)
            os.remove(tmp_onnx_external_weights_path)
        raise Exception(
            "The tool skipped dummy inference to avoid SWAP processing because the total size "
            f"of the tensor of inference results exceeded about {mem_available} GB. "
            f"(results: {total_output_size_gb} GB)"
        )

    output_names_order = [out.name for out in gs_graph.outputs]
    if use_memmap_outputs:
        output_shapes: List[List[int]] = []
        for out in gs_graph.outputs:
            shape = out.shape
            if shape is None:
                if tmp_onnx_path:
                    os.remove(tmp_onnx_path)
                    os.remove(tmp_onnx_external_weights_path)
                raise Exception(
                    "onnxruntime output memmap requires static output shapes. "
                    "Provide --shape_hints or reduce validation outputs."
                )
            normalized_shape: List[int] = []
            for dim in shape:
                if not isinstance(dim, (int, np.integer)):
                    if tmp_onnx_path:
                        os.remove(tmp_onnx_path)
                        os.remove(tmp_onnx_external_weights_path)
                    raise Exception(
                        "onnxruntime output memmap requires static output shapes. "
                        "Provide --shape_hints or reduce validation outputs."
                    )
                normalized_shape.append(int(dim))
            output_shapes.append(normalized_shape)

        memmap_dir = ort_output_memmap_dir
        if memmap_dir is None:
            memmap_dir = make_managed_tempdir(
                prefix="onnx2tf_ort_mm_",
                stale_prefixes=["onnx2tf_ort_mm_"],
            )
        os.makedirs(memmap_dir, exist_ok=True)

        disk_free = psutil.disk_usage(memmap_dir).free
        if total_output_size > disk_free:
            if tmp_onnx_path:
                os.remove(tmp_onnx_path)
                os.remove(tmp_onnx_external_weights_path)
            raise Exception(
                "Not enough disk space for memmap outputs. "
                f"Required: {total_output_size} bytes, Free: {disk_free} bytes."
            )

        info(
            f"onnxruntime output memmap enabled. Outputs: {len(output_names_order)}, Path: {memmap_dir}"
        )

        io_binding = onnx_session.io_binding()
        for input_name, input_data in input_datas.items():
            if not input_data.flags["C_CONTIGUOUS"]:
                input_data = np.ascontiguousarray(input_data)
                input_datas[input_name] = input_data
            io_binding.bind_input(
                input_name,
                "cpu",
                0,
                input_data.dtype,
                input_data.shape,
                input_data.__array_interface__["data"][0],
            )

        memmap_outputs = {}
        for idx, (output_name, output_shape) in enumerate(zip(output_names_order, output_shapes)):
            memmap_path = os.path.join(
                memmap_dir,
                _make_safe_ort_memmap_filename(output_index=idx, output_name=output_name),
            )
            if ort_output_memmap_paths_for_cleanup is not None:
                ort_output_memmap_paths_for_cleanup.append(memmap_path)
            output_dtype = gs_graph.outputs[idx].dtype
            if output_dtype is None:
                output_dtype = inferred_output_dtype_by_name.get(output_name, np.float32)
            output_dtype = np.dtype(output_dtype)
            memmap_array = np.memmap(
                memmap_path,
                dtype=output_dtype,
                mode="w+",
                shape=tuple(output_shape),
            )
            memmap_outputs[output_name] = memmap_array
            io_binding.bind_output(
                output_name,
                "cpu",
                0,
                output_dtype,
                output_shape,
                memmap_array.__array_interface__["data"][0],
            )

        onnx_session.run_with_iobinding(io_binding)
        io_binding.synchronize_outputs()
        for memmap_array in memmap_outputs.values():
            memmap_array.flush()
        outputs = [memmap_outputs[name] for name in output_names_order]
    else:
        outputs = onnx_session.run(output_names_order, input_datas)

    if tmp_onnx_path:
        os.remove(tmp_onnx_path)
        os.remove(tmp_onnx_external_weights_path)
    return [np.asarray(output) for output in outputs]


def weights_export(
    *,
    extract_target_tflite_file_path: str,
    output_weights_file_path: str,
) -> None:
    import h5py

    interpreter = Interpreter(model_path=extract_target_tflite_file_path)
    interpreter.allocate_tensors()
    input_indexes = [input_detail["index"] for input_detail in interpreter.get_input_details()]
    output_indexes = [output_detail["index"] for output_detail in interpreter.get_output_details()]
    with h5py.File(output_weights_file_path, "w") as f:
        for tensor_detail in interpreter.get_tensor_details():
            tensor_index = tensor_detail["index"]
            if tensor_index in input_indexes or tensor_index in output_indexes:
                continue
            try:
                dataset = f.create_dataset(
                    name=tensor_detail["name"],
                    data=interpreter.get_tensor(tensor_index),
                )
                del dataset
            except Exception:
                pass
