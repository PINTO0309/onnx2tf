from __future__ import annotations

import re
from typing import List, Sequence, Set


def _convert_nhwc_pad_to_nchw_pad_values(pad_values: Sequence[int]) -> List[int] | None:
    values = [int(value) for value in list(pad_values)]
    if len(values) % 2 != 0 or len(values) > 8:
        return None
    nhwc_inner_to_outer = ["C", "W", "H", "N"]
    nchw_inner_to_outer = ["W", "H", "C", "N"]
    semantic_pairs = {name: [0, 0] for name in nhwc_inner_to_outer}
    pair_count = len(values) // 2
    for idx in range(pair_count):
        semantic_pairs[nhwc_inner_to_outer[idx]] = [
            int(values[idx * 2]),
            int(values[idx * 2 + 1]),
        ]
    output_pairs = [semantic_pairs[name] for name in nchw_inner_to_outer]
    last_nonzero = -1
    for idx, pair in enumerate(output_pairs):
        if pair != [0, 0]:
            last_nonzero = idx
    if last_nonzero < 0:
        return []
    flattened: List[int] = []
    for pair in output_pairs[: last_nonzero + 1]:
        flattened.extend([int(pair[0]), int(pair[1])])
    return flattened

def _convert_nchw_pad_to_nhwc_pad_values(pad_values: Sequence[int]) -> List[int] | None:
    values = [int(value) for value in list(pad_values)]
    if len(values) % 2 != 0 or len(values) > 8:
        return None
    nchw_inner_to_outer = ["W", "H", "C", "N"]
    nhwc_inner_to_outer = ["C", "W", "H", "N"]
    semantic_pairs = {name: [0, 0] for name in nchw_inner_to_outer}
    pair_count = len(values) // 2
    for idx in range(pair_count):
        semantic_pairs[nchw_inner_to_outer[idx]] = [
            int(values[idx * 2]),
            int(values[idx * 2 + 1]),
        ]
    output_pairs = [semantic_pairs[name] for name in nhwc_inner_to_outer]
    last_nonzero = -1
    for idx, pair in enumerate(output_pairs):
        if pair != [0, 0]:
            last_nonzero = idx
    if last_nonzero < 0:
        return []
    flattened: List[int] = []
    for pair in output_pairs[: last_nonzero + 1]:
        flattened.extend([int(pair[0]), int(pair[1])])
    return flattened

def _infer_unique_channel_count_from_rank4_shape(shape: Sequence[int]) -> int | None:
    shape_values = [int(value) for value in list(shape)]
    if len(shape_values) != 4:
        return None
    dims = [int(value) for value in shape_values[1:]]
    candidates = [
        int(value)
        for value in dims
        if int(value) > 1 and dims.count(int(value)) == 1
    ]
    if len(candidates) == 1:
        return int(candidates[0])
    if len(candidates) > 1:
        return int(max(candidates))
    return None

def _fast_precanonicalize_expr_identifiers(expr: str) -> Set[str]:
    return {
        token
        for token in re.findall(r"[A-Za-z_][A-Za-z0-9_]*", str(expr))
        if token not in {"torch", "self", "True", "False"}
    }
