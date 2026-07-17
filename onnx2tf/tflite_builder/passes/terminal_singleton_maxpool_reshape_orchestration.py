from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

from onnx2tf.tflite_builder.core.layout import LayoutState
from onnx2tf.tflite_builder.core.model_ir_pass_state import ModelIRPassStateScope
from onnx2tf.tflite_builder.ir import ModelIR
from onnx2tf.tflite_builder.passes.graph_cleanup import (
    run_consecutive_reshape_cleanup,
)
from onnx2tf.tflite_builder.passes.recovery_orchestration import (
    RecoveryInvocation,
    run_recovery_invocations,
)
from onnx2tf.tflite_builder.passes.singleton_maxpool_layout import (
    run_singleton_maxpool_layout_cleanup,
)


TERMINAL_SINGLETON_MAXPOOL_RESHAPE_PASS_IDS = (
    "run_singleton_maxpool_layout_cleanup",
    "run_consecutive_reshape_cleanup",
)


@dataclass(frozen=True)
class TerminalSingletonMaxPoolReshapeContext:
    model_ir: ModelIR
    layout_state: LayoutState | None
    diagnostics: List[Dict[str, Any]]


def build_terminal_singleton_maxpool_reshape_invocations(
    context: TerminalSingletonMaxPoolReshapeContext,
) -> Tuple[RecoveryInvocation, ...]:
    state_scope = ModelIRPassStateScope(
        context.model_ir,
        layout_state=context.layout_state,
    )
    keyword_args = (
        ("layout_state", context.layout_state),
        ("diagnostics", context.diagnostics),
        ("state_scope", state_scope),
    )
    return (
        RecoveryInvocation(
            pass_id=TERMINAL_SINGLETON_MAXPOOL_RESHAPE_PASS_IDS[0],
            callback=run_singleton_maxpool_layout_cleanup,
            args=(context.model_ir,),
            keyword_args=keyword_args,
        ),
        # Boundary cleanup can recreate a terminal no-op RESHAPE.
        # Run one last pass to remove shape-preserving single reshapes.
        RecoveryInvocation(
            pass_id=TERMINAL_SINGLETON_MAXPOOL_RESHAPE_PASS_IDS[1],
            callback=run_consecutive_reshape_cleanup,
            args=(context.model_ir,),
            keyword_args=keyword_args,
        ),
    )


def run_terminal_singleton_maxpool_reshape(
    context: TerminalSingletonMaxPoolReshapeContext,
) -> None:
    run_recovery_invocations(
        build_terminal_singleton_maxpool_reshape_invocations(context),
        expected_pass_ids=TERMINAL_SINGLETON_MAXPOOL_RESHAPE_PASS_IDS,
        phase_name="terminal singleton-maxpool/reshape",
    )
