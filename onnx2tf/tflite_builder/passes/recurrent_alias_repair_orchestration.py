from __future__ import annotations

from typing import Dict, Optional

from onnx2tf.tflite_builder.core.graph import ModelIRGraphIndex
from onnx2tf.tflite_builder.ir import ModelIR
from onnx2tf.tflite_builder.passes.recurrent_alias import (
    repair_orphan_recurrent_step_tensors,
)


def repair_orphan_recurrent_step_tensors_summary(
    model_ir: ModelIR,
    *,
    graph_index: Optional[ModelIRGraphIndex] = None,
) -> Dict[str, int]:
    """Repair recurrent aliases and normalize the mutation count."""

    repaired = repair_orphan_recurrent_step_tensors(
        model_ir,
        graph_index=graph_index,
    )
    return {"repaired_orphan_recurrent_step_tensors": int(repaired)}
