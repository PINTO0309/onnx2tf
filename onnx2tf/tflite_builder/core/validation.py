from __future__ import annotations

from typing import List

from onnx2tf.tflite_builder.core.graph import ModelIRGraphIndex
from onnx2tf.tflite_builder.core.passes import OrderedPassManager, PassPhase, PassSpec
from onnx2tf.tflite_builder.ir import ModelIR, normalize_logical_layout


class ModelIRInvariantError(RuntimeError):
    pass


def validate_model_ir_invariants(
    model_ir: ModelIR,
    graph_index: ModelIRGraphIndex | None = None,
) -> List[str]:
    """Return structural errors that make exporter behaviour ambiguous."""

    problems: List[str] = []
    index = (
        graph_index
        if graph_index is not None and graph_index.model_ir is model_ir
        else ModelIRGraphIndex(model_ir)
    )
    for name, producers in sorted(index.duplicate_producers.items()):
        problems.append(f"duplicate_producer:{name}:{producers}")

    known = set(str(name) for name in model_ir.tensors)
    known.update(str(name) for name in model_ir.inputs)
    for op_index, op in enumerate(model_ir.operators):
        for name in op.inputs:
            if str(name).strip() == "":
                # TFLite uses an empty tensor name to preserve omitted optional
                # input slots (for example the 24/48-input LSTM builtins).
                continue
            if str(name) not in known:
                problems.append(f"missing_input_tensor:{op_index}:{name}")
        for name in op.outputs:
            if str(name).strip() == "":
                continue
            if str(name) not in known:
                problems.append(f"missing_output_tensor:{op_index}:{name}")

    for name in model_ir.inputs:
        if str(name) not in model_ir.tensors:
            problems.append(f"missing_graph_input_tensor:{name}")
    for name in model_ir.outputs:
        if str(name) not in model_ir.tensors:
            problems.append(f"missing_graph_output_tensor:{name}")

    for name, tensor in model_ir.tensors.items():
        if not str(name).strip():
            problems.append("empty_tensor_name")
        if len(tensor.shape) != len(tensor.shape_signature or tensor.shape):
            problems.append(f"shape_signature_rank_mismatch:{name}")
        if normalize_logical_layout(tensor.logical_layout) != tensor.logical_layout:
            problems.append(f"invalid_logical_layout:{name}:{tensor.logical_layout}")
        if normalize_logical_layout(tensor.physical_layout) != tensor.physical_layout:
            problems.append(f"invalid_physical_layout:{name}:{tensor.physical_layout}")
    return problems


def assert_model_ir_invariants(model_ir: ModelIR) -> None:
    problems = validate_model_ir_invariants(model_ir)
    if problems:
        raise ModelIRInvariantError(
            "ModelIR invariant validation failed: " + "; ".join(problems[:16])
        )


def run_model_ir_validation_pipeline(model_ir: ModelIR) -> None:
    """Run the final invariant gate through the shared ordered pass engine."""

    manager = OrderedPassManager[ModelIR](validator=validate_model_ir_invariants)
    manager.register(
        PassSpec(
            pass_id="model_ir.structural_invariants",
            phase=PassPhase.VALIDATE,
            callback=lambda _: {"changed": False},
        )
    )
    try:
        manager.run(model_ir)
    except RuntimeError as error:
        raise ModelIRInvariantError(str(error)) from error
