from __future__ import annotations

import hashlib
import keyword
import re
from typing import Dict, List, Sequence, Set, Tuple

import numpy as np

from onnx2tf.tflite_builder.ir import (
    ModelIR,
    is_channel_last_logical_layout,
    normalize_logical_layout,
)


_PYTORCH_LOCAL_NAME_MAX_LENGTH = 40


_GENERATED_NAME_DROP_TOKENS = {
    "model",
    "readvariableop",
    "read",
    "variable",
    "resource",
}


_GENERATED_NAME_TOKEN_ALIASES = {
    "block": "b",
    "pool": "p",
    "conv": "cv",
    "conv2d": "cv",
    "depthwise": "dw",
    "depthwise1": "dw",
    "fusedbatchnorm": "bn",
    "fusedbatchnormv3": "bn",
    "batchnorm": "bn",
    "bn": "bn",
    "relu": "relu",
    "sigmoid": "sig",
    "add": "add",
    "mul": "mul",
    "mean": "mean",
    "transpose": "tr",
    "reshape": "rs",
    "input": "in",
    "output": "out",
    "channel": "ch",
    "first": "first",
    "nhwc": "nhwc",
}


_GENERATED_NAME_SUFFIX_PATTERNS: Sequence[Tuple[str, List[str]]] = (
    ("_input_nhwc__channel_first", ["in", "cf"]),
    ("_output_nhwc__channel_first", ["out", "cf"]),
    ("_input_nwc__channel_first", ["in", "cf"]),
    ("_output_nwc__channel_first", ["out", "cf"]),
    ("_input_ndhwc__channel_first", ["in", "cf"]),
    ("_output_ndhwc__channel_first", ["out", "cf"]),
    ("__channel_first", ["cf"]),
    ("_input_nhwc", ["in", "nhwc"]),
    ("_output_nhwc", ["out", "nhwc"]),
    ("_input_nwc", ["in", "nhwc"]),
    ("_output_nwc", ["out", "nhwc"]),
    ("_input_ndhwc", ["in", "nhwc"]),
    ("_output_ndhwc", ["out", "nhwc"]),
    ("_input", ["in"]),
    ("_output", ["out"]),
)


def _make_tensor_storage_name_map(model_ir: ModelIR) -> Dict[str, str]:
    used_names: Set[str] = set()
    storage_name_map: Dict[str, str] = {}
    for tensor_name, tensor in sorted(model_ir.tensors.items()):
        if not isinstance(tensor.data, np.ndarray):
            continue
        base_name = re.sub(r"[^0-9A-Za-z_]", "_", str(tensor_name)).strip("_")
        if base_name == "":
            base_name = "tensor"
        if base_name[0].isdigit():
            base_name = f"tensor_{base_name}"
        candidate = base_name
        suffix = 1
        while candidate in used_names:
            candidate = f"{base_name}_{suffix}"
            suffix += 1
        used_names.add(candidate)
        storage_name_map[str(tensor_name)] = candidate
    return storage_name_map


def _sanitize_python_identifier(name: str, *, prefix: str) -> str:
    identifier = re.sub(r"[^0-9A-Za-z_]", "_", str(name)).strip("_")
    if identifier == "":
        identifier = str(prefix)
    if identifier[0].isdigit():
        identifier = f"{prefix}_{identifier}"
    if keyword.iskeyword(identifier):
        identifier = f"{identifier}_{prefix}"
    return identifier


def _extract_generated_name_suffix_tokens(name: str) -> Tuple[str, List[str]]:
    lowered = str(name).lower()
    for raw_suffix, suffix_tokens in _GENERATED_NAME_SUFFIX_PATTERNS:
        if lowered.endswith(raw_suffix):
            return str(name)[: len(str(name)) - len(raw_suffix)], list(suffix_tokens)
    return str(name), []


def _split_generated_name_piece(piece: str) -> List[str]:
    compact_piece = re.sub(r"[^0-9A-Za-z]+", "", str(piece))
    if compact_piece == "":
        return []
    lowered_piece = compact_piece.lower()
    if lowered_piece in _GENERATED_NAME_DROP_TOKENS:
        return []
    for pattern, token_template in (
        (r"conv2d(\d+)", "cv{}"),
        (r"conv(\d+)", "cv{}"),
        (r"depthwise(\d+)", "dw{}"),
        (r"relu(\d+)", "relu{}"),
        (r"sigmoid(\d+)", "sig{}"),
        (r"add(\d+)", "add{}"),
        (r"mul(\d+)", "mul{}"),
        (r"mean(\d+)", "mean{}"),
        (r"reshape(\d+)", "rs{}"),
        (r"transpose(\d+)", "tr{}"),
        (r"pool(\d+)", "p{}"),
        (r"block(\d+)", "b{}"),
        (r"bn(\d+)", "bn{}"),
    ):
        matched = re.fullmatch(pattern, lowered_piece)
        if matched is not None:
            return [str(token_template).format(matched.group(1))]
    whole_alias = _GENERATED_NAME_TOKEN_ALIASES.get(lowered_piece, None)
    if whole_alias is not None:
        return [str(whole_alias)]
    split_tokens: List[str] = []
    for fragment in re.findall(
        r"[A-Z]+(?=[A-Z][a-z]|[0-9]|$)|[A-Z]?[a-z]+|[0-9]+", compact_piece
    ):
        lowered_fragment = str(fragment).lower()
        if lowered_fragment in _GENERATED_NAME_DROP_TOKENS or lowered_fragment == "":
            continue
        split_tokens.append(
            str(_GENERATED_NAME_TOKEN_ALIASES.get(lowered_fragment, lowered_fragment))
        )
    return split_tokens if len(split_tokens) > 0 else [lowered_piece]


def _collapse_generated_name_tokens(tokens: Sequence[str]) -> List[str]:
    collapsed: List[str] = []
    for token in tokens:
        token_str = str(token)
        if token_str == "":
            continue
        if (
            token_str.isdigit()
            and len(collapsed) > 0
            and not collapsed[-1].isdigit()
            and not bool(re.search(r"\d$", collapsed[-1]))
        ):
            collapsed[-1] = f"{collapsed[-1]}{token_str}"
            continue
        if len(collapsed) > 0 and collapsed[-1] == token_str:
            continue
        collapsed.append(token_str)
    return collapsed


def _shorten_generated_python_identifier(
    name: str,
    *,
    prefix: str,
    max_length: int = _PYTORCH_LOCAL_NAME_MAX_LENGTH,
) -> str:
    stem_name, suffix_tokens = _extract_generated_name_suffix_tokens(str(name))
    base_tokens: List[str] = []
    for piece in re.split(r"_+", str(stem_name)):
        base_tokens.extend(_split_generated_name_piece(piece))
    base_tokens = _collapse_generated_name_tokens(base_tokens)
    candidate_tokens = base_tokens + list(suffix_tokens)
    candidate = _sanitize_python_identifier("_".join(candidate_tokens), prefix=prefix)
    if len(candidate) <= int(max_length):
        return candidate
    digest = hashlib.sha1(str(name).encode("utf-8")).hexdigest()[:4]
    core_tokens = list(base_tokens)
    leading_tokens = core_tokens[:3]
    trailing_tokens: List[str] = []
    for token in reversed(core_tokens):
        if token in leading_tokens or token in trailing_tokens:
            continue
        trailing_tokens.insert(0, token)
        if len(trailing_tokens) >= 2:
            break
    compressed_tokens = _collapse_generated_name_tokens(
        [*leading_tokens, *trailing_tokens, *suffix_tokens, digest]
    )
    candidate = _sanitize_python_identifier("_".join(compressed_tokens), prefix=prefix)
    trim_tokens = list(compressed_tokens[:-1])
    while len(candidate) > int(max_length) and len(trim_tokens) > 0:
        trim_index = max(
            range(len(trim_tokens)),
            key=lambda idx: (
                len(trim_tokens[idx])
                if trim_tokens[idx] not in {"in", "out", "cf", "nhwc"}
                else -1
            ),
        )
        if len(trim_tokens[trim_index]) > 3:
            trim_tokens[trim_index] = trim_tokens[trim_index][:-1]
        elif len(trim_tokens) > 1:
            del trim_tokens[trim_index]
        else:
            break
        candidate = _sanitize_python_identifier(
            "_".join([*trim_tokens, digest]), prefix=prefix
        )
    if len(candidate) <= int(max_length):
        return candidate
    fallback = _sanitize_python_identifier(f"{prefix}_{digest}", prefix=prefix)
    return fallback[: int(max_length)]


def _make_unique_identifier(base_name: str, used_names: Set[str]) -> str:
    candidate = str(base_name)
    suffix = 1
    while candidate in used_names:
        candidate = f"{base_name}_{suffix}"
        suffix += 1
    used_names.add(candidate)
    return candidate


def _direct_codegen_module_attr_base(op_type: str) -> str:
    names = {
        "CONV_2D": "conv2d",
        "DEPTHWISE_CONV_2D": "depthwise_conv2d",
        "TRANSPOSE_CONV": "conv_transpose2d",
        "CONV_3D": "conv3d",
        "CONV_3D_TRANSPOSE": "conv_transpose3d",
        "FULLY_CONNECTED": "linear",
        "PRELU": "prelu",
        "UNIDIRECTIONAL_SEQUENCE_RNN": "sequence_rnn",
        "UNIDIRECTIONAL_SEQUENCE_LSTM": "sequence_lstm",
        "BIDIRECTIONAL_SEQUENCE_LSTM": "bidirectional_sequence_lstm",
    }
    return str(names.get(str(op_type), str(op_type).lower()))


def _build_tensor_var_name_map(model_ir: ModelIR) -> Dict[str, str]:
    used_names: Set[str] = set()
    mapping: Dict[str, str] = {}

    def _canonical_tensor_var_source_name(tensor_name: str) -> str:
        tensor = model_ir.tensors.get(str(tensor_name), None)
        base_name = str(tensor_name)
        if tensor is not None:
            layout = normalize_logical_layout(tensor.logical_layout)
            if not is_channel_last_logical_layout(layout):
                base_name = re.sub(
                    r"_(?:nhwc|nwc|ndhwc)$",
                    "",
                    base_name,
                    flags=re.IGNORECASE,
                )
        return base_name

    for tensor_name in (
        list(model_ir.inputs)
        + [str(out) for op in model_ir.operators for out in op.outputs]
        + list(model_ir.outputs)
    ):
        if str(tensor_name) in mapping:
            continue
        base = _shorten_generated_python_identifier(
            _canonical_tensor_var_source_name(str(tensor_name)),
            prefix="t",
        )
        mapping[str(tensor_name)] = _make_unique_identifier(base, used_names)
    return mapping


def _build_buffer_attr_name_map(
    *,
    model_ir: ModelIR,
    tensor_storage_name_map: Dict[str, str],
    excluded_tensor_names: Set[str],
) -> Dict[str, str]:
    used_names: Set[str] = set()
    mapping: Dict[str, str] = {}
    for tensor_name, tensor in sorted(model_ir.tensors.items()):
        if str(tensor_name) in excluded_tensor_names or not isinstance(
            tensor.data, np.ndarray
        ):
            continue
        storage_name = tensor_storage_name_map.get(str(tensor_name), str(tensor_name))
        base = _sanitize_python_identifier(f"const_{storage_name}", prefix="const")
        mapping[str(tensor_name)] = _make_unique_identifier(base, used_names)
    return mapping
