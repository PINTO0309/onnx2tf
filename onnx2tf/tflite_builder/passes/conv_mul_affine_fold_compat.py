from __future__ import annotations

from typing import Dict, Optional

import numpy as np

from onnx2tf.tflite_builder.core.graph import ModelIRGraphIndex
from onnx2tf.tflite_builder.core.layout import LayoutState
from onnx2tf.tflite_builder.core.model_ir_utils import (
    _build_tensor_consumer_map,
    _prune_unused_tensors,
    _set_operator_inputs,
    _set_operator_outputs,
)
from onnx2tf.tflite_builder.ir import ModelIR, OperatorIR, TensorIR
from onnx2tf.tflite_builder.passes.conv_mul_affine_fold import (
    optimize_conv_mul_affine_mul_only_chains as _optimize_conv_mul_affine_mul_only_chains_pass,
)


def optimize_fold_conv_mul_add_affine_chains(
    model_ir: ModelIR,
    *,
    enable_conv_add_only_fold: bool = True,
    layout_state: Optional[LayoutState] = None,
) -> Dict[str, int]:
    """
    Fold single-path affine chains after CONV_2D into CONV_2D.

    Targets:
      x --CONV_2D(w,b)--> y
      y --ADD(a)--> o
        or
      y --MUL(m)--> z
      (optional) z --ADD(a)--> o

    Rewrite 1:
      x --CONV_2D(w,b')--> o
      where:
        b'[oc] = b[oc] + a[oc]

    Rewrite 2:
      x --CONV_2D(w',b')--> z

    Rewrite 3:
      z --ADD(a)--> o
      x --CONV_2D(w',b')--> o
      where:
        w'[oc,kh,kw,ic] = w[oc,kh,kw,ic] * m[oc]
        b'[oc]          = b[oc] * m[oc] + a[oc]

    Rewrite 4 (RELU-preserving partial fold):
      x --CONV_2D(w,b, fused=RELU)--> y
      y --MUL(m)--> z
      (optional) z --ADD(a)--> o
      x --CONV_2D(w',b', fused=RELU)--> z
      (optional) z --ADD(a)--> o
      where m[oc] >= 0 and:
        w'[oc,kh,kw,ic] = w[oc,kh,kw,ic] * m[oc]
        b'[oc]          = b[oc] * m[oc]

      NOTE: ADD is intentionally kept because a post-RELU affine offset
      cannot, in general, be represented by CONV_2D fused activations.

    Safety:
    - Conv output must have exactly one consumer (single path).
    - ADD/MUL side input must be constant representable as scalar or NHWC-channelwise.
    - When MUL is followed by ADD, ADD side input must also be constant representable as scalar or NHWC-channelwise.
    - Conv filter/bias must be constant and not shared by other operators.
    - Chains with Conv output fan-out (>=2) are excluded.
    """

    indexed_stats = _optimize_conv_mul_affine_mul_only_chains_pass(
        model_ir,
        graph_index=ModelIRGraphIndex(model_ir),
        layout_state=layout_state,
    )
    tensors_before_fallback_prune = (
        set(str(name) for name in model_ir.tensors)
        if layout_state is not None
        else set()
    )

    def _extract_nhwc_channelwise_coeff(
        *,
        tensor: Optional[TensorIR],
        out_channels: int,
        allow_leading_channel_axis: bool = True,
    ) -> Optional[np.ndarray]:
        if tensor is None or tensor.data is None:
            return None
        if int(out_channels) <= 0:
            return None
        raw = np.asarray(tensor.data)
        if raw.size == 0:
            return None
        coeff = np.asarray(raw, dtype=np.float32)
        if coeff.size == 1:
            return np.full((int(out_channels),), float(coeff.reshape(-1)[0]), dtype=np.float32)
        shape = [int(v) for v in list(coeff.shape)]
        if len(shape) == 1 and int(shape[0]) == int(out_channels):
            return coeff.reshape(int(out_channels)).astype(np.float32, copy=False)
        if bool(allow_leading_channel_axis) and len(shape) >= 2 and int(shape[0]) == int(out_channels):
            if int(np.prod(shape[1:])) == 1:
                return coeff.reshape(int(out_channels)).astype(np.float32, copy=False)
        if len(shape) >= 2 and int(shape[-1]) == int(out_channels):
            if int(np.prod(shape[:-1])) == 1:
                return coeff.reshape(int(out_channels)).astype(np.float32, copy=False)
        return None

    def _unique_tensor_name(base: str) -> str:
        name = str(base)
        suffix = 1
        while name in model_ir.tensors:
            name = f"{base}_{suffix}"
            suffix += 1
        return name

    folded = int(indexed_stats.get("folded_conv_mul_add_affine_chains", 0))
    folded_conv_add_only = int(
        indexed_stats.get("folded_conv_add_only_affine_chains", 0)
    )
    folded_conv_mul_only = int(
        indexed_stats.get("folded_conv_mul_only_affine_chains", 0)
    )
    folded_conv_mul_add = int(
        indexed_stats.get("folded_conv_mul_add_only_affine_chains", 0)
    )

    while True:
        changed = False
        consumers = _build_tensor_consumer_map(model_ir)
        model_outputs = set(  # noqa: F841 - preserve legacy local state
            str(v) for v in model_ir.outputs
        )

        for conv_idx, conv_op in enumerate(model_ir.operators):
            if str(conv_op.op_type) != "CONV_2D" or len(conv_op.inputs) < 2 or len(conv_op.outputs) != 1:
                continue
            conv_opts = dict(conv_op.options) if isinstance(conv_op.options, dict) else {}
            conv_fused_activation = str(
                conv_opts.get("fusedActivationFunction", "NONE")
            ).upper()
            if conv_fused_activation not in {"NONE", "RELU"}:
                continue
            conv_out_name = str(conv_op.outputs[0])
            conv_users = [int(v) for v in consumers.get(conv_out_name, [])]
            if len(conv_users) != 1:
                continue

            first_idx = int(conv_users[0])
            first_op = model_ir.operators[int(first_idx)]
            first_type = str(first_op.op_type)

            add_only_mode = False
            mul_side_tensor: Optional[TensorIR] = None
            mul_out_name: Optional[str] = None
            mul_idx: Optional[int] = None

            add_only_idx: Optional[int] = None
            add_only_op: Optional[OperatorIR] = None
            add_only_side_tensor: Optional[TensorIR] = None
            add_only_out_name: Optional[str] = None
            add_only_fused_activation: str = "NONE"

            if first_type == "MUL":
                if len(first_op.inputs) != 2 or len(first_op.outputs) != 1:
                    continue
                mul_idx = int(first_idx)
                mul_op = first_op
                mul_inputs = [str(v) for v in list(mul_op.inputs)]
                if mul_inputs[0] == conv_out_name:
                    mul_side_name = str(mul_inputs[1])
                elif mul_inputs[1] == conv_out_name:
                    mul_side_name = str(mul_inputs[0])
                else:
                    continue
                mul_side_tensor = model_ir.tensors.get(mul_side_name, None)
                if mul_side_tensor is None or mul_side_tensor.data is None:
                    continue
                mul_out_name = str(mul_op.outputs[0])
                mul_users = [int(v) for v in consumers.get(mul_out_name, [])]
            elif first_type == "ADD":
                if conv_fused_activation != "NONE":
                    continue
                if not bool(enable_conv_add_only_fold):
                    continue
                if len(first_op.inputs) != 2 or len(first_op.outputs) != 1:
                    continue
                add_only_opts = dict(first_op.options) if isinstance(first_op.options, dict) else {}
                add_only_fused_activation = str(
                    add_only_opts.get("fusedActivationFunction", "NONE")
                ).upper()
                if add_only_fused_activation not in {"NONE", "RELU", "RELU6"}:
                    continue
                add_only_idx = int(first_idx)
                add_only_op = first_op
                add_only_inputs = [str(v) for v in list(add_only_op.inputs)]
                if add_only_inputs[0] == conv_out_name:
                    add_only_side_name = str(add_only_inputs[1])
                elif add_only_inputs[1] == conv_out_name:
                    add_only_side_name = str(add_only_inputs[0])
                else:
                    continue
                add_only_side_tensor = model_ir.tensors.get(add_only_side_name, None)
                if add_only_side_tensor is None or add_only_side_tensor.data is None:
                    continue
                add_only_out_name = str(add_only_op.outputs[0])
                add_only_users = [int(v) for v in consumers.get(add_only_out_name, [])]
                # Keep this rewrite strict to single-path chains.
                if len(add_only_users) != 1:
                    continue
                add_only_mode = True
                mul_users = []
            else:
                continue

            has_add = False
            add_idx: Optional[int] = None
            add_op: Optional[OperatorIR] = None
            add_side_name: Optional[str] = None
            add_side_tensor: Optional[TensorIR] = None
            keep_add_after_mul_fold = False
            if not add_only_mode and len(mul_users) == 1:
                candidate_add_idx = int(mul_users[0])
                candidate_add_op = model_ir.operators[int(candidate_add_idx)]
                if str(candidate_add_op.op_type) == "ADD" and len(candidate_add_op.inputs) == 2 and len(candidate_add_op.outputs) == 1:
                    candidate_add_opts = dict(candidate_add_op.options) if isinstance(candidate_add_op.options, dict) else {}
                    if str(candidate_add_opts.get("fusedActivationFunction", "NONE")).upper() != "NONE":
                        continue
                    add_inputs = [str(v) for v in list(candidate_add_op.inputs)]
                    candidate_add_side_name: Optional[str] = None
                    if add_inputs[0] == str(mul_out_name):
                        candidate_add_side_name = str(add_inputs[1])
                    elif add_inputs[1] == str(mul_out_name):
                        candidate_add_side_name = str(add_inputs[0])
                    if candidate_add_side_name is not None:
                        candidate_add_side_tensor = model_ir.tensors.get(candidate_add_side_name, None)
                        if candidate_add_side_tensor is not None and candidate_add_side_tensor.data is not None:
                            has_add = True
                            add_idx = int(candidate_add_idx)
                            add_op = candidate_add_op
                            add_side_name = str(  # noqa: F841 - legacy state
                                candidate_add_side_name
                            )
                            add_side_tensor = candidate_add_side_tensor

            filter_name = str(conv_op.inputs[1])
            filter_tensor = model_ir.tensors.get(filter_name, None)
            if (
                filter_tensor is None
                or filter_tensor.data is None
                or bool(filter_tensor.is_variable)
                or filter_tensor.quantization is not None
            ):
                continue
            filter_users = [int(v) for v in consumers.get(filter_name, [])]
            if set(int(v) for v in filter_users) != {int(conv_idx)}:
                continue
            filter_data = np.asarray(filter_tensor.data)
            if filter_data.ndim != 4:
                continue
            out_channels = int(filter_data.shape[0])
            if out_channels <= 0:
                continue

            bias_name: Optional[str] = None
            bias_tensor: Optional[TensorIR] = None
            if len(conv_op.inputs) >= 3:
                bias_name = str(conv_op.inputs[2])
                bias_tensor = model_ir.tensors.get(bias_name, None)
                if (
                    bias_tensor is None
                    or bias_tensor.data is None
                    or bool(bias_tensor.is_variable)
                    or bias_tensor.quantization is not None
                ):
                    continue
                bias_users = [int(v) for v in consumers.get(str(bias_name), [])]
                if set(int(v) for v in bias_users) != {int(conv_idx)}:
                    continue
                bias_values = np.asarray(bias_tensor.data).reshape(-1)
                if int(bias_values.size) != int(out_channels):
                    continue
            else:
                bias_name = _unique_tensor_name(f"{conv_out_name}_affine_fold_bias")
                bias_tensor = TensorIR(
                    name=str(bias_name),
                    dtype=str(filter_tensor.dtype),
                    shape=[int(out_channels)],
                    shape_signature=[int(out_channels)],
                    data=np.zeros((int(out_channels),), dtype=np.float32),
                    is_variable=False,
                    quantization=None,
                )
                model_ir.tensors[str(bias_name)] = bias_tensor
                _set_operator_inputs(
                    model_ir=model_ir,
                    op=conv_op,
                    new_inputs=[str(conv_op.inputs[0]), str(filter_name), str(bias_name)],
                )

            if add_only_mode:
                mul_coeff = np.ones((int(out_channels),), dtype=np.float32)
                add_coeff = _extract_nhwc_channelwise_coeff(
                    tensor=add_only_side_tensor,
                    out_channels=int(out_channels),
                    # Keep add-only folding strict: accept only scalar,
                    # [C], or trailing-channel broadcast forms (..., C)
                    # with all non-channel dims == 1.
                    allow_leading_channel_axis=False,
                )
                if add_coeff is None:
                    continue
                folded_output_name = str(add_only_out_name)
                remove_indices = [int(add_only_idx)]
            else:
                mul_coeff = _extract_nhwc_channelwise_coeff(
                    tensor=mul_side_tensor,
                    out_channels=int(out_channels),
                )
                if mul_coeff is None:
                    continue
                if conv_fused_activation == "RELU" and bool(np.any(mul_coeff < 0.0)):
                    # For RELU-preserving rewrite, scale must be non-negative:
                    # relu(z) * m == relu(z * m) only when m >= 0.
                    continue
                add_coeff = np.zeros((int(out_channels),), dtype=np.float32)
                if has_add:
                    if conv_fused_activation == "RELU":
                        # Keep ADD in place; fold only MUL into CONV_2D.
                        keep_add_after_mul_fold = True
                        folded_output_name = str(mul_out_name)
                        remove_indices = [int(mul_idx)]
                    else:
                        add_coeff = _extract_nhwc_channelwise_coeff(
                            tensor=add_side_tensor,
                            out_channels=int(out_channels),
                        )
                        if add_coeff is None:
                            continue
                        assert add_op is not None
                        assert add_idx is not None
                        folded_output_name = str(add_op.outputs[0])
                        remove_indices = [int(mul_idx), int(add_idx)]
                else:
                    folded_output_name = str(mul_out_name)
                    remove_indices = [int(mul_idx)]
            if not np.isfinite(mul_coeff).all() or not np.isfinite(add_coeff).all():
                continue

            filter_dtype = np.asarray(filter_tensor.data).dtype
            folded_filter = (
                np.asarray(filter_tensor.data, dtype=np.float32)
                * np.asarray(mul_coeff, dtype=np.float32).reshape(int(out_channels), 1, 1, 1)
            )
            filter_tensor.data = np.asarray(folded_filter, dtype=filter_dtype)

            assert bias_tensor is not None
            bias_dtype = np.asarray(bias_tensor.data).dtype
            bias_values = np.asarray(bias_tensor.data, dtype=np.float32).reshape(int(out_channels))
            folded_bias = bias_values * np.asarray(mul_coeff, dtype=np.float32) + np.asarray(add_coeff, dtype=np.float32)
            bias_tensor.data = np.asarray(folded_bias, dtype=bias_dtype)
            bias_tensor.shape = [int(out_channels)]
            bias_tensor.shape_signature = [int(out_channels)]

            _set_operator_outputs(
                model_ir=model_ir,
                op=conv_op,
                new_outputs=[folded_output_name],
            )
            if add_only_mode and str(add_only_fused_activation) != "NONE":
                if not isinstance(conv_op.options, dict):
                    conv_op.options = {}
                conv_op.options["fusedActivationFunction"] = str(add_only_fused_activation)

            for remove_idx in sorted(remove_indices, reverse=True):
                del model_ir.operators[int(remove_idx)]

            folded += 1
            if add_only_mode:
                folded_conv_add_only += 1
            else:
                if has_add and not bool(keep_add_after_mul_fold):
                    folded_conv_mul_add += 1
                else:
                    folded_conv_mul_only += 1
            changed = True
            break

        if not changed:
            break

    _prune_unused_tensors(model_ir)
    if layout_state is not None:
        layout_state.remove(
            tensors_before_fallback_prune
            - set(str(name) for name in model_ir.tensors)
        )
    return {
        "folded_conv_mul_add_affine_chains": int(folded),
        "folded_conv_add_only_affine_chains": int(folded_conv_add_only),
        "folded_conv_mul_only_affine_chains": int(folded_conv_mul_only),
        "folded_conv_mul_add_only_affine_chains": int(folded_conv_mul_add),
    }
