"""Stable internal contracts for the flatbuffer-direct pipeline.

The public API intentionally remains in :mod:`onnx2tf`.  Objects exported by
this package are internal seams which let the legacy lowering implementation
move to ordered, independently testable stages without changing user-facing
behaviour.
"""

from onnx2tf.tflite_builder.core.contracts import (
    ArtifactPlan,
    ConversionRequest,
    ConversionResult,
)
from onnx2tf.tflite_builder.core.graph import GraphIndex, ModelIRGraphIndex
from onnx2tf.tflite_builder.core.layout import LayoutState
from onnx2tf.tflite_builder.core.lowering_registry import (
    LoweringRegistry,
    LoweringResolution,
)
from onnx2tf.tflite_builder.core.lowering_context import LoweringContext
from onnx2tf.tflite_builder.core.node import NodeView
from onnx2tf.tflite_builder.core.op_contracts import (
    DispatchEntry,
    DispatchResolution,
    NodeValidationError,
    ValidationSpec,
)
from onnx2tf.tflite_builder.core.passes import (
    OrderedPassManager,
    PassPhase,
    PassResult,
    PassSpec,
)
from onnx2tf.tflite_builder.core.session import ConversionSession
from onnx2tf.tflite_builder.core.validation import (
    ModelIRInvariantError,
    run_model_ir_validation_pipeline,
    validate_model_ir_invariants,
)

__all__ = [
    "ArtifactPlan",
    "ConversionRequest",
    "ConversionResult",
    "ConversionSession",
    "GraphIndex",
    "LayoutState",
    "ModelIRGraphIndex",
    "ModelIRInvariantError",
    "LoweringRegistry",
    "LoweringResolution",
    "LoweringContext",
    "NodeView",
    "DispatchEntry",
    "DispatchResolution",
    "NodeValidationError",
    "ValidationSpec",
    "OrderedPassManager",
    "PassPhase",
    "PassResult",
    "PassSpec",
    "run_model_ir_validation_pipeline",
    "validate_model_ir_invariants",
]
