from __future__ import annotations

from typing import Dict, Iterable, Sequence, Tuple

import numpy as np
import onnx
from onnx import numpy_helper


def _constant_arrays(model: onnx.ModelProto) -> Dict[str, np.ndarray]:
    constants = {
        str(initializer.name): np.asarray(numpy_helper.to_array(initializer))
        for initializer in model.graph.initializer
    }
    for node in model.graph.node:
        if str(node.op_type) != "Constant" or len(node.output) != 1:
            continue
        for attribute in node.attribute:
            if str(attribute.name) == "value" and attribute.HasField("t"):
                constants[str(node.output[0])] = np.asarray(
                    numpy_helper.to_array(attribute.t)
                )
                break
    return constants


def _canonical_name(name: str) -> str:
    return "".join(character for character in str(name).lower() if character.isalnum())


def _pyramid_spatial_shapes(split_sizes: Sequence[int]) -> np.ndarray | None:
    sizes = [int(value) for value in split_sizes]
    if len(sizes) < 2 or any(value <= 0 for value in sizes):
        return None
    candidates: list[list[tuple[int, int]]] = []
    first_size = int(sizes[0])
    for height in range(1, int(np.sqrt(first_size)) + 1):
        if first_size % height != 0:
            continue
        width = int(first_size // height)
        current_height = int(height)
        current_width = int(width)
        levels: list[tuple[int, int]] = []
        valid = True
        for size in sizes:
            if int(current_height * current_width) != int(size):
                valid = False
                break
            levels.append((int(current_height), int(current_width)))
            current_height = int((current_height + 1) // 2)
            current_width = int((current_width + 1) // 2)
        if valid:
            candidates.append(levels)
    if len(candidates) != 1:
        return None
    return np.asarray(candidates[0], dtype=np.int64)


def _pyramid_spatial_shapes_from_total(
    *,
    sequence_length: int,
    level_count: int,
) -> np.ndarray | None:
    total = int(sequence_length)
    levels = int(level_count)
    if total <= 0 or levels < 2:
        return None

    def _levels(height: int, width: int) -> list[tuple[int, int]]:
        shapes: list[tuple[int, int]] = []
        for _ in range(levels):
            shapes.append((int(height), int(width)))
            height = int((height + 1) // 2)
            width = int((width + 1) // 2)
        return shapes

    # Require both final pyramid dimensions to remain greater than one. This
    # rejects degenerate 1xN factorizations that are mathematically possible
    # but do not describe a multi-scale image feature pyramid.
    minimum_base_dimension = int(2 ** (levels - 1) + 1)
    candidates: list[list[tuple[int, int]]] = []
    for height in range(
        minimum_base_dimension,
        int(np.sqrt(total)) + 1,
    ):
        low = max(int(height), minimum_base_dimension)
        high = int(total // height)
        while low <= high:
            width = int((low + high) // 2)
            shapes = _levels(height, width)
            candidate_total = int(sum(h * w for h, w in shapes))
            if candidate_total < total:
                low = int(width + 1)
            elif candidate_total > total:
                high = int(width - 1)
            else:
                candidates.append(shapes)
                break
    if len(candidates) != 1:
        return None
    return np.asarray(candidates[0], dtype=np.int64)


def _spatial_shape_candidate(
    *,
    model: onnx.ModelProto,
    constants: Dict[str, np.ndarray],
    level_count: int,
    known_sequence_lengths: Iterable[int],
) -> np.ndarray | None:
    known_lengths = {int(value) for value in known_sequence_lengths if int(value) > 0}
    candidates: list[np.ndarray] = []
    for node in model.graph.node:
        if str(node.op_type) != "Split" or len(node.input) < 2:
            continue
        split_values = constants.get(str(node.input[1]))
        if split_values is None:
            continue
        sizes = [int(value) for value in np.asarray(split_values).reshape(-1).tolist()]
        if (
            len(sizes) != int(level_count)
            or max(sizes, default=0) <= 1
            or (known_lengths and int(sum(sizes)) not in known_lengths)
        ):
            continue
        spatial_shapes = _pyramid_spatial_shapes(sizes)
        if spatial_shapes is not None:
            candidates.append(spatial_shapes)
    if not candidates:
        for sequence_length in sorted(known_lengths):
            spatial_shapes = _pyramid_spatial_shapes_from_total(
                sequence_length=int(sequence_length),
                level_count=int(level_count),
            )
            if spatial_shapes is not None:
                candidates.append(spatial_shapes)
    unique_candidates = {
        tuple(tuple(int(value) for value in row) for row in candidate.tolist()): candidate
        for candidate in candidates
    }
    if len(unique_candidates) != 1:
        return None
    return next(iter(unique_candidates.values()))


def build_attention_control_input_overrides(
    *,
    onnx_graph: onnx.ModelProto,
    input_specs: Sequence[Tuple[str, np.dtype, Tuple[int, ...]]],
) -> Dict[str, np.ndarray]:
    """Build deterministic valid controls for static multi-scale attention graphs."""
    constants = _constant_arrays(onnx_graph)
    input_spec_map = {
        str(name): (np.dtype(dtype), tuple(int(value) for value in shape))
        for name, dtype, shape in input_specs
    }
    sequence_lengths: set[int] = set()
    for input_name, (_, shape) in input_spec_map.items():
        canonical_name = _canonical_name(input_name)
        if "spatialshape" in canonical_name or "validratio" in canonical_name:
            continue
        if len(shape) >= 3 and int(shape[-2]) > 1:
            sequence_lengths.add(int(shape[-2]))
        elif len(shape) == 2 and int(shape[-1]) > 1:
            sequence_lengths.add(int(shape[-1]))
    consumers: Dict[str, list[onnx.NodeProto]] = {}
    for node in onnx_graph.graph.node:
        for input_name in node.input:
            consumers.setdefault(str(input_name), []).append(node)

    overrides: Dict[str, np.ndarray] = {}
    for input_name, (dtype, shape) in input_spec_map.items():
        canonical_name = _canonical_name(input_name)
        if (
            "spatialshape" in canonical_name
            and len(shape) == 2
            and int(shape[1]) == 2
            and int(shape[0]) > 1
        ):
            candidate = _spatial_shape_candidate(
                model=onnx_graph,
                constants=constants,
                level_count=int(shape[0]),
                known_sequence_lengths=sequence_lengths,
            )
            if candidate is not None:
                overrides[input_name] = candidate.astype(dtype).reshape(shape)
            continue

        if (
            "validratio" in canonical_name
            and len(shape) == 3
            and int(shape[-1]) == 2
        ):
            overrides[input_name] = np.ones(shape, dtype=dtype)
            continue

        if not np.issubdtype(dtype, np.floating) or len(shape) < 2:
            continue
        has_direct_bool_cast = False
        for consumer in consumers.get(input_name, []):
            if str(consumer.op_type) != "Cast":
                continue
            for attribute in consumer.attribute:
                if (
                    str(attribute.name) == "to"
                    and int(attribute.i) == int(onnx.TensorProto.BOOL)
                ):
                    has_direct_bool_cast = True
                    break
            if has_direct_bool_cast:
                break
        if has_direct_bool_cast:
            overrides[input_name] = np.zeros(shape, dtype=dtype)

    return overrides
