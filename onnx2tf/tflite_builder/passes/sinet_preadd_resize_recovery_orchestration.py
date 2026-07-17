from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Tuple

from onnx2tf.tflite_builder.core.layout import LayoutState
from onnx2tf.tflite_builder.ir import ModelIR
from onnx2tf.tflite_builder.passes.recovery_orchestration import (
    RecoveryInvocation,
    run_recovery_invocations,
)
from onnx2tf.tflite_builder.passes.residual_affine_fanout_layout import (
    optimize_transpose_pre_add_mul_add_transpose_fanout_nhwc_chains,
)
from onnx2tf.tflite_builder.passes.residual_affine_prelu_layout import (
    optimize_transpose_pre_add_mul_add_prelu_nhwc_chains,
)
from onnx2tf.tflite_builder.passes.sinet_concat_resize_layout import (
    optimize_sinet_concat_resize_affine_transpose_chains,
)
from onnx2tf.tflite_builder.passes.sinet_dual_resize_layout import (
    optimize_sinet_dual_resize_affine_transpose_chains,
)
from onnx2tf.tflite_builder.passes.sinet_softmax_mask_layout import (
    optimize_sinet_softmax_mask_residual_nhwc_tail_chains,
)
from onnx2tf.tflite_builder.passes.sinet_tail_concat_layout import (
    optimize_sinet_concat_resize_affine_tail_concat_transpose_chains,
)


SINET_PREADD_RESIZE_RECOVERY_PASS_IDS = (
    "_optimize_transpose_pre_add_mul_add_prelu_nhwc_chains",
    "_optimize_transpose_pre_add_mul_add_transpose_fanout_nhwc_chains",
    "_optimize_sinet_concat_resize_affine_transpose_chains",
    "_optimize_sinet_dual_resize_affine_transpose_chains",
    "_optimize_sinet_concat_resize_affine_tail_concat_transpose_chains",
    "_optimize_sinet_softmax_mask_residual_nhwc_tail_chains",
)


@dataclass(frozen=True)
class SINetPreaddResizeRecoveryContext:
    model_ir: ModelIR
    layout_state: LayoutState | None


def _model_invocation(
    pass_id: str,
    callback: Callable[..., Any],
    context: SINetPreaddResizeRecoveryContext,
    *,
    include_layout: bool = False,
) -> RecoveryInvocation:
    keyword_args = ()
    if include_layout:
        keyword_args = (("layout_state", context.layout_state),)
    return RecoveryInvocation(
        pass_id=pass_id,
        callback=callback,
        args=(context.model_ir,),
        keyword_args=keyword_args,
    )


def build_sinet_preadd_resize_recovery_invocations(
    context: SINetPreaddResizeRecoveryContext,
) -> Tuple[RecoveryInvocation, ...]:
    return (
        _model_invocation(
            SINET_PREADD_RESIZE_RECOVERY_PASS_IDS[0],
            optimize_transpose_pre_add_mul_add_prelu_nhwc_chains,
            context,
        ),
        _model_invocation(
            SINET_PREADD_RESIZE_RECOVERY_PASS_IDS[1],
            optimize_transpose_pre_add_mul_add_transpose_fanout_nhwc_chains,
            context,
        ),
        # Generic rewrites can recreate SiNet concat+resize transpose adapters.
        _model_invocation(
            SINET_PREADD_RESIZE_RECOVERY_PASS_IDS[2],
            optimize_sinet_concat_resize_affine_transpose_chains,
            context,
            include_layout=True,
        ),
        _model_invocation(
            SINET_PREADD_RESIZE_RECOVERY_PASS_IDS[3],
            optimize_sinet_dual_resize_affine_transpose_chains,
            context,
            include_layout=True,
        ),
        _model_invocation(
            SINET_PREADD_RESIZE_RECOVERY_PASS_IDS[4],
            optimize_sinet_concat_resize_affine_tail_concat_transpose_chains,
            context,
            include_layout=True,
        ),
        _model_invocation(
            SINET_PREADD_RESIZE_RECOVERY_PASS_IDS[5],
            optimize_sinet_softmax_mask_residual_nhwc_tail_chains,
            context,
            include_layout=True,
        ),
    )


def run_sinet_preadd_resize_recovery(
    context: SINetPreaddResizeRecoveryContext,
) -> None:
    run_recovery_invocations(
        build_sinet_preadd_resize_recovery_invocations(context),
        expected_pass_ids=SINET_PREADD_RESIZE_RECOVERY_PASS_IDS,
        phase_name="SINet pre-add/resize recovery",
    )
