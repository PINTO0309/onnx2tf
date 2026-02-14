from __future__ import annotations

from typing import Any, List

import numpy as np

from onnx2tf.tflite_builder.ir import OperatorIR


def resolve_padding(node: Any) -> str:
    auto_pad = str(node.attrs.get("auto_pad", "NOTSET"))
    if auto_pad in ["SAME_UPPER", "SAME_LOWER"]:
        return "SAME"
    pads = list(node.attrs.get("pads", [0, 0, 0, 0]))
    if len(pads) == 4 and sum([abs(int(v)) for v in pads]) == 0:
        return "VALID"
    if len(pads) == 4:
        top, left, bottom, right = [int(v) for v in pads]
        if top == bottom and left == right and top >= 0 and left >= 0:
            return "SAME"
    raise NotImplementedError(
        "Only zero pads, symmetric pads, or SAME auto_pad are supported in flatbuffer_direct. "
        f"op={node.name} pads={pads} auto_pad={auto_pad}"
    )


def make_transpose(
    ctx: Any,
    input_name: str,
    output_name: str,
    perm_values: List[int],
) -> None:
    perm_name = ctx.add_const_tensor(
        f"{output_name}_perm",
        np.asarray(perm_values, dtype=np.int32),
    )
    ctx.add_operator(
        OperatorIR(
            op_type="TRANSPOSE",
            inputs=[input_name, perm_name],
            outputs=[output_name],
        )
    )
