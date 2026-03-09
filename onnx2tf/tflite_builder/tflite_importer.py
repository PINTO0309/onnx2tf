from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from onnx2tf.tflite_builder.ir import ModelIR, OperatorIR, TensorIR
from onnx2tf.tflite_builder.schema_loader import load_schema_module


class TFLiteModelIRImportError(RuntimeError):
    pass


_SUPPORTED_TENSOR_DTYPES: Dict[str, np.dtype] = {
    "BOOL": np.dtype(np.bool_),
    "INT8": np.dtype(np.int8),
    "INT16": np.dtype(np.int16),
    "INT32": np.dtype(np.int32),
    "INT64": np.dtype(np.int64),
    "UINT8": np.dtype(np.uint8),
    "UINT16": np.dtype(np.uint16),
    "UINT32": np.dtype(np.uint32),
    "UINT64": np.dtype(np.uint64),
    "FLOAT16": np.dtype(np.float16),
    "FLOAT32": np.dtype(np.float32),
    "FLOAT64": np.dtype(np.float64),
}

_ENUM_FIELD_NAMES: Dict[str, str] = {
    "padding": "Padding",
    "fusedActivationFunction": "ActivationFunctionType",
    "inDataType": "TensorType",
    "outDataType": "TensorType",
    "outputType": "TensorType",
    "idxOutType": "TensorType",
    "outType": "TensorType",
    "quantizedBiasType": "TensorType",
    "mode": "MirrorPadMode",
}


def _as_list(value: Any) -> List[Any]:
    if value is None:
        return []
    if isinstance(value, np.ndarray):
        return np.asarray(value).reshape(-1).tolist()
    if isinstance(value, (list, tuple)):
        return list(value)
    try:
        return list(value)
    except Exception:
        return []


def _as_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, np.ndarray):
        if value.dtype == np.uint8:
            try:
                return bytes(value.astype(np.uint8, copy=False).tolist()).decode("utf-8")
            except Exception:
                return str(value)
        return str(value)
    if isinstance(value, (bytes, bytearray)):
        try:
            return bytes(value).decode("utf-8")
        except Exception:
            return str(bytes(value))
    return str(value)


def _enum_value_to_name(schema_tflite: Dict[str, Any], enum_name: str, value: int) -> str:
    enum_cls = schema_tflite.get(enum_name, None)
    if enum_cls is None:
        return str(int(value))
    for name, raw_value in enum_cls.__dict__.items():
        if name.startswith("_"):
            continue
        if not isinstance(raw_value, int):
            continue
        if int(raw_value) == int(value):
            return str(name)
    return str(int(value))


def _builtin_operator_name(schema_tflite: Dict[str, Any], builtin_code: int) -> str:
    builtin = schema_tflite.get("BuiltinOperator", None)
    if builtin is None:
        raise TFLiteModelIRImportError("BuiltinOperator enum is unavailable in schema module.")
    for name, raw_value in builtin.__dict__.items():
        if name.startswith("_"):
            continue
        if not isinstance(raw_value, int):
            continue
        if int(raw_value) == int(builtin_code):
            return str(name)
    raise TFLiteModelIRImportError(
        f"Failed to resolve BuiltinOperator name from code={builtin_code}."
    )


def _to_serializable_value(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (bytes, bytearray)):
        return bytes(value)
    if isinstance(value, list):
        return [_to_serializable_value(v) for v in value]
    if isinstance(value, tuple):
        return [_to_serializable_value(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _to_serializable_value(v) for k, v in value.items()}
    return value


def _extract_options_dict(
    *,
    schema_tflite: Dict[str, Any],
    builtin_options: Any,
) -> Dict[str, Any]:
    if builtin_options is None:
        return {}
    attrs = getattr(builtin_options, "__dict__", None)
    if not isinstance(attrs, dict):
        return {}

    options: Dict[str, Any] = {}
    for key, raw_value in attrs.items():
        if str(key).startswith("_"):
            continue
        value = _to_serializable_value(raw_value)
        if value is None:
            continue
        enum_name = _ENUM_FIELD_NAMES.get(str(key), None)
        if enum_name is not None:
            if isinstance(value, (int, np.integer)):
                value = _enum_value_to_name(schema_tflite, enum_name, int(value))
        options[str(key)] = value
    return options


def _extract_quantization_dict(quantization: Any) -> Optional[Dict[str, Any]]:
    if quantization is None:
        return None
    attrs = getattr(quantization, "__dict__", None)
    if not isinstance(attrs, dict):
        return None
    result: Dict[str, Any] = {}
    for key, raw_value in attrs.items():
        if str(key).startswith("_"):
            continue
        value = _to_serializable_value(raw_value)
        if value is None:
            continue
        result[str(key)] = value
    return result if len(result) > 0 else None


def _normalize_shape(values: Any) -> List[int]:
    if values is None:
        return []
    return [int(v) for v in np.asarray(values).reshape(-1).tolist()]


def _tensor_name_candidates(
    *,
    raw_name: str,
    fallback_name: str,
    preferred_boundary_name: Optional[str],
) -> List[str]:
    candidates: List[str] = []
    if preferred_boundary_name is not None and str(preferred_boundary_name).strip() != "":
        candidates.append(str(preferred_boundary_name).strip())
    if str(raw_name).strip() != "":
        candidates.append(str(raw_name).strip())
    candidates.append(str(fallback_name))
    return candidates


def _build_tensor_name_map(
    *,
    subgraph_index: int,
    tensors: List[Any],
    preferred_boundary_names: Dict[int, str],
) -> Dict[int, str]:
    name_map: Dict[int, str] = {}
    used: set[str] = set()

    for tensor_index, tensor in enumerate(tensors):
        fallback = f"sg{subgraph_index}_tensor{tensor_index}"
        raw_name = _as_text(getattr(tensor, "name", ""))
        preferred = preferred_boundary_names.get(int(tensor_index), None)
        candidates = _tensor_name_candidates(
            raw_name=raw_name,
            fallback_name=fallback,
            preferred_boundary_name=preferred,
        )
        selected = None
        for candidate in candidates:
            if candidate not in used:
                selected = candidate
                break
        if selected is None:
            suffix = 1
            while True:
                candidate = f"{fallback}_{suffix}"
                if candidate not in used:
                    selected = candidate
                    break
                suffix += 1
        used.add(str(selected))
        name_map[int(tensor_index)] = str(selected)

    return name_map


def _buffer_to_numpy(
    *,
    tensor_name: str,
    tensor_dtype: str,
    tensor_shape: List[int],
    buffer_obj: Any,
) -> Optional[np.ndarray]:
    if tensor_dtype == "STRING":
        raise TFLiteModelIRImportError(
            "STRING tensor is not supported in tflite direct import. "
            f"tensor={tensor_name}"
        )
    if tensor_dtype not in _SUPPORTED_TENSOR_DTYPES:
        raise TFLiteModelIRImportError(
            "Tensor dtype is not supported in tflite direct import. "
            f"tensor={tensor_name} dtype={tensor_dtype}"
        )

    raw_data = getattr(buffer_obj, "data", None)
    if raw_data is None:
        return None

    if isinstance(raw_data, np.ndarray):
        data_bytes = bytes(raw_data.astype(np.uint8, copy=False).tolist())
    elif isinstance(raw_data, (bytes, bytearray)):
        data_bytes = bytes(raw_data)
    else:
        try:
            data_bytes = bytes(raw_data)
        except Exception:
            data_bytes = b""

    if len(data_bytes) == 0:
        return None

    np_dtype = _SUPPORTED_TENSOR_DTYPES[tensor_dtype]
    if np_dtype == np.dtype(np.bool_):
        raw_u8 = np.frombuffer(data_bytes, dtype=np.uint8)
        array = (raw_u8 != 0).astype(np.bool_)
    else:
        if len(data_bytes) % int(np_dtype.itemsize) != 0:
            raise TFLiteModelIRImportError(
                "Tensor buffer size is not aligned with dtype itemsize. "
                f"tensor={tensor_name} dtype={tensor_dtype} "
                f"buffer_size={len(data_bytes)} itemsize={np_dtype.itemsize}"
            )
        array = np.frombuffer(data_bytes, dtype=np_dtype)

    if len(tensor_shape) == 0:
        if array.size == 1:
            return np.asarray(array.reshape(()))
        return np.asarray(array)

    expected_numel = int(np.prod(np.asarray(tensor_shape, dtype=np.int64)))
    if expected_numel == int(array.size):
        return np.asarray(array.reshape(tuple(int(v) for v in tensor_shape)))
    if expected_numel == 0 and int(array.size) == 0:
        return np.asarray(array.reshape(tuple(int(v) for v in tensor_shape)))

    raise TFLiteModelIRImportError(
        "Tensor buffer size does not match tensor shape. "
        f"tensor={tensor_name} shape={tensor_shape} expected_numel={expected_numel} actual_numel={array.size}"
    )


def _decode_operator(
    *,
    schema_tflite: Dict[str, Any],
    op_obj: Any,
    opcodes: List[Any],
    tensor_name_map: Dict[int, str],
) -> OperatorIR:
    opcode_index = int(getattr(op_obj, "opcodeIndex", 0))
    if opcode_index < 0 or opcode_index >= len(opcodes):
        raise TFLiteModelIRImportError(f"Operator opcodeIndex is out of range: {opcode_index}")
    opcode = opcodes[opcode_index]

    builtin_code = int(getattr(opcode, "builtinCode", 0))
    op_type = _builtin_operator_name(schema_tflite, builtin_code)
    custom_code = _as_text(getattr(opcode, "customCode", "")).strip()
    if op_type == "CUSTOM" and custom_code == "":
        custom_code = "CUSTOM"

    version = int(getattr(opcode, "version", 1))

    inputs: List[str] = []
    for input_index in _as_list(getattr(op_obj, "inputs", None)):
        input_i = int(input_index)
        if input_i < 0:
            continue
        if input_i not in tensor_name_map:
            raise TFLiteModelIRImportError(
                f"Operator input tensor index is invalid: {input_i}"
            )
        inputs.append(str(tensor_name_map[input_i]))

    outputs: List[str] = []
    for output_index in _as_list(getattr(op_obj, "outputs", None)):
        output_i = int(output_index)
        if output_i < 0:
            continue
        if output_i not in tensor_name_map:
            raise TFLiteModelIRImportError(
                f"Operator output tensor index is invalid: {output_i}"
            )
        outputs.append(str(tensor_name_map[output_i]))

    options = _extract_options_dict(
        schema_tflite=schema_tflite,
        builtin_options=getattr(op_obj, "builtinOptions", None),
    )

    if op_type == "CUSTOM":
        options["customCode"] = str(custom_code)
        custom_options = getattr(op_obj, "customOptions", None)
        if custom_options is not None:
            if isinstance(custom_options, np.ndarray):
                options["customOptions"] = bytes(
                    custom_options.astype(np.uint8, copy=False).tolist()
                )
            elif isinstance(custom_options, (bytes, bytearray)):
                options["customOptions"] = bytes(custom_options)
            else:
                try:
                    options["customOptions"] = bytes(custom_options)
                except Exception:
                    pass

    return OperatorIR(
        op_type=str(op_type),
        inputs=inputs,
        outputs=outputs,
        options=options,
        version=int(version),
    )


def _find_serving_default_signature(model_obj: Any) -> Optional[Any]:
    signature_defs = _as_list(getattr(model_obj, "signatureDefs", None))
    for signature in signature_defs:
        if int(getattr(signature, "subgraphIndex", -1)) != 0:
            continue
        signature_key = _as_text(getattr(signature, "signatureKey", ""))
        if signature_key == "serving_default":
            return signature
    for signature in signature_defs:
        if int(getattr(signature, "subgraphIndex", -1)) == 0:
            return signature
    return None


def _build_boundary_preferred_name_maps(model_obj: Any) -> Tuple[Dict[int, str], Dict[int, str]]:
    signature = _find_serving_default_signature(model_obj)
    input_names: Dict[int, str] = {}
    output_names: Dict[int, str] = {}
    if signature is None:
        return input_names, output_names

    for tensor_map in _as_list(getattr(signature, "inputs", None)):
        tensor_index = int(getattr(tensor_map, "tensorIndex", -1))
        if tensor_index < 0:
            continue
        tensor_name = _as_text(getattr(tensor_map, "name", "")).strip()
        if tensor_name == "":
            continue
        input_names[tensor_index] = tensor_name

    for tensor_map in _as_list(getattr(signature, "outputs", None)):
        tensor_index = int(getattr(tensor_map, "tensorIndex", -1))
        if tensor_index < 0:
            continue
        tensor_name = _as_text(getattr(tensor_map, "name", "")).strip()
        if tensor_name == "":
            continue
        output_names[tensor_index] = tensor_name

    return input_names, output_names


def _import_subgraph(
    *,
    schema_tflite: Dict[str, Any],
    model_obj: Any,
    subgraph_index: int,
    preferred_input_names: Optional[Dict[int, str]] = None,
    preferred_output_names: Optional[Dict[int, str]] = None,
) -> ModelIR:
    subgraphs = _as_list(getattr(model_obj, "subgraphs", None))
    if subgraph_index < 0 or subgraph_index >= len(subgraphs):
        raise TFLiteModelIRImportError(f"Subgraph index is out of range: {subgraph_index}")
    subgraph = subgraphs[subgraph_index]

    tensors = _as_list(getattr(subgraph, "tensors", None))
    operators = _as_list(getattr(subgraph, "operators", None))
    inputs = [int(v) for v in _as_list(getattr(subgraph, "inputs", None))]
    outputs = [int(v) for v in _as_list(getattr(subgraph, "outputs", None))]

    preferred: Dict[int, str] = {}
    if preferred_input_names is not None:
        preferred.update({int(k): str(v) for k, v in preferred_input_names.items()})
    if preferred_output_names is not None:
        preferred.update({int(k): str(v) for k, v in preferred_output_names.items()})

    tensor_name_map = _build_tensor_name_map(
        subgraph_index=subgraph_index,
        tensors=tensors,
        preferred_boundary_names=preferred,
    )

    model_ir = ModelIR(
        name=_as_text(getattr(subgraph, "name", "")) or f"subgraph_{subgraph_index}",
        description=_as_text(getattr(model_obj, "description", "")) or "onnx2tf tflite import",
    )

    buffers = _as_list(getattr(model_obj, "buffers", None))
    for tensor_index, tensor_obj in enumerate(tensors):
        tensor_name = str(tensor_name_map[int(tensor_index)])
        tensor_dtype = _enum_value_to_name(
            schema_tflite,
            "TensorType",
            int(getattr(tensor_obj, "type", 0)),
        )
        tensor_shape = _normalize_shape(getattr(tensor_obj, "shape", None))
        tensor_shape_signature_raw = getattr(tensor_obj, "shapeSignature", None)
        tensor_shape_signature = (
            _normalize_shape(tensor_shape_signature_raw)
            if tensor_shape_signature_raw is not None
            else None
        )
        tensor_is_variable = bool(getattr(tensor_obj, "isVariable", False))

        buffer_index = int(getattr(tensor_obj, "buffer", 0))
        if buffer_index < 0 or buffer_index >= len(buffers):
            raise TFLiteModelIRImportError(
                "Tensor buffer index is out of range. "
                f"tensor={tensor_name} buffer_index={buffer_index}"
            )
        tensor_data = _buffer_to_numpy(
            tensor_name=tensor_name,
            tensor_dtype=tensor_dtype,
            tensor_shape=tensor_shape,
            buffer_obj=buffers[buffer_index],
        )

        external_buffer = getattr(tensor_obj, "externalBuffer", None)
        if external_buffer is not None:
            attrs = getattr(external_buffer, "__dict__", {})
            if isinstance(attrs, dict) and len(attrs) > 0:
                raise TFLiteModelIRImportError(
                    "Tensor externalBuffer is not supported in tflite direct import. "
                    f"tensor={tensor_name}"
                )

        quantization = _extract_quantization_dict(getattr(tensor_obj, "quantization", None))

        model_ir.tensors[tensor_name] = TensorIR(
            name=tensor_name,
            dtype=tensor_dtype,
            shape=tensor_shape,
            shape_signature=tensor_shape_signature,
            data=tensor_data,
            is_variable=tensor_is_variable,
            quantization=quantization,
        )

    operator_codes = _as_list(getattr(model_obj, "operatorCodes", None))
    for op_obj in operators:
        model_ir.operators.append(
            _decode_operator(
                schema_tflite=schema_tflite,
                op_obj=op_obj,
                opcodes=operator_codes,
                tensor_name_map=tensor_name_map,
            )
        )

    for input_index in inputs:
        if input_index < 0:
            continue
        if input_index in tensor_name_map:
            model_ir.inputs.append(str(tensor_name_map[input_index]))
    for output_index in outputs:
        if output_index < 0:
            continue
        if output_index in tensor_name_map:
            model_ir.outputs.append(str(tensor_name_map[output_index]))

    return model_ir


def import_model_ir_from_tflite(
    *,
    tflite_file_path: str,
    output_folder_path: Optional[str] = None,
) -> ModelIR:
    tflite_path = Path(str(tflite_file_path))
    if not tflite_path.exists():
        raise FileNotFoundError(f"Input tflite file does not exist: {tflite_file_path}")

    schema_output_dir = (
        str(output_folder_path)
        if output_folder_path is not None and str(output_folder_path).strip() != ""
        else str(tflite_path.parent)
    )
    schema_tflite = load_schema_module(schema_output_dir)

    with open(tflite_path, "rb") as f:
        model_bytes = f.read()

    model_obj = schema_tflite["ModelT"].InitFromObj(
        schema_tflite["Model"].GetRootAs(model_bytes, 0)
    )
    subgraphs = _as_list(getattr(model_obj, "subgraphs", None))
    if len(subgraphs) == 0:
        raise TFLiteModelIRImportError(
            "Input tflite does not contain any subgraphs."
        )

    preferred_inputs, preferred_outputs = _build_boundary_preferred_name_maps(model_obj)

    root_model_ir = _import_subgraph(
        schema_tflite=schema_tflite,
        model_obj=model_obj,
        subgraph_index=0,
        preferred_input_names=preferred_inputs,
        preferred_output_names=preferred_outputs,
    )
    root_model_ir.name = str(tflite_path.stem)
    root_model_ir.description = _as_text(
        getattr(model_obj, "description", "")
    ) or "onnx2tf tflite direct import"
    root_model_ir.metadata["onnx_public_layout_map"] = {}

    for subgraph_index in range(1, len(subgraphs)):
        child_ir = _import_subgraph(
            schema_tflite=schema_tflite,
            model_obj=model_obj,
            subgraph_index=subgraph_index,
            preferred_input_names=None,
            preferred_output_names=None,
        )
        root_model_ir.subgraphs.append(child_ir)

    return root_model_ir
