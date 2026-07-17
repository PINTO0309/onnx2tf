from __future__ import annotations

from typing import Dict, Optional

from onnx2tf.tflite_builder.core.graph import ModelIRGraphIndex
from onnx2tf.tflite_builder.core.layout import LayoutState
from onnx2tf.tflite_builder.core.model_ir_utils import (
    _build_tensor_consumer_map,
    _prune_unused_tensors,
    _read_const_ints_from_tensor,
    _read_transpose_perm,
    _set_operator_inputs,
    _set_operator_outputs,
    _shapes_match_if_known,
    _write_const_ints_to_tensor,
)
from onnx2tf.tflite_builder.ir import ModelIR, normalize_onnx_shape
from onnx2tf.tflite_builder.passes.expanddims_reshape_layout import (
    optimize_transpose_factorized_expanddims_nhwc_chains as _optimize_transpose_factorized_expanddims_nhwc_chains_pass,
)


def optimize_transpose_reshape_transpose_to_expanddims_nhwc_chains_compat(
    model_ir: ModelIR,
    *,
    layout_state: Optional[LayoutState] = None,
) -> Dict[str, int]:
    """
    Collapse NHWC->NCHW->RESHAPE->N?HWC wrappers into a single RESHAPE.

    Target:
      x_nhwc --TRANSPOSE(0,3,1,2)--> x_nchw
      x_nchw --RESHAPE([N,1,C,H,W])--> r
      r      --TRANSPOSE(0,1,3,4,2)--> y

    Rewrite:
      x_nhwc --RESHAPE([N,1,H,W,C])--> y

    This pattern appears in some YOLO heads where layout wrappers are only used
    to insert a singleton anchor axis before 5D decode operations.
    """
    indexed_stats = _optimize_transpose_factorized_expanddims_nhwc_chains_pass(
        model_ir,
        graph_index=ModelIRGraphIndex(model_ir),
        layout_state=layout_state,
    )
    rewritten = int(
        indexed_stats.get(
            "optimized_transpose_reshape_transpose_to_expanddims_nhwc_chains",
            0,
        )
    )
    tensors_before_fallback_prune = (
        set(str(name) for name in model_ir.tensors)
        if layout_state is not None
        else set()
    )
    perm_nhwc_to_nchw = [0, 3, 1, 2]
    perm_n1chw_to_n1hwc = [0, 1, 3, 4, 2]
    perm_nchw1_to_nhwc1 = [0, 2, 3, 1, 4]
    perm_nhwab_to_nahwb = [0, 3, 1, 2, 4]

    def _dims_compatible(a: int, b: int) -> bool:
        if int(a) < 0 or int(b) < 0:
            return True
        return int(a) == int(b)

    while True:
        changed = False
        consumers = _build_tensor_consumer_map(model_ir)
        model_outputs = set(str(v) for v in model_ir.outputs)

        for pre_idx, pre_op in enumerate(model_ir.operators):
            if str(pre_op.op_type) != "TRANSPOSE" or len(pre_op.inputs) < 2 or len(pre_op.outputs) != 1:
                continue
            if _read_transpose_perm(model_ir, pre_op) != perm_nhwc_to_nchw:
                continue

            pre_in_name = str(pre_op.inputs[0])
            pre_out_name = str(pre_op.outputs[0])
            pre_users = [int(v) for v in consumers.get(pre_out_name, [])]
            if len(pre_users) != 1:
                continue
            reshape_idx = int(pre_users[0])
            reshape_op = model_ir.operators[int(reshape_idx)]
            if (
                str(reshape_op.op_type) != "RESHAPE"
                or len(reshape_op.inputs) < 1
                or len(reshape_op.outputs) != 1
                or str(reshape_op.inputs[0]) != pre_out_name
            ):
                continue

            reshape_out_name = str(reshape_op.outputs[0])
            reshape_users = [int(v) for v in consumers.get(reshape_out_name, [])]
            if len(reshape_users) != 1:
                continue
            post_idx = int(reshape_users[0])
            post_op = model_ir.operators[int(post_idx)]
            if (
                str(post_op.op_type) != "TRANSPOSE"
                or len(post_op.inputs) < 2
                or len(post_op.outputs) != 1
                or str(post_op.inputs[0]) != reshape_out_name
            ):
                continue
            post_perm = _read_transpose_perm(model_ir, post_op)
            if post_perm not in [perm_n1chw_to_n1hwc, perm_nchw1_to_nhwc1]:
                continue

            post_out_name = str(post_op.outputs[0])
            if pre_out_name in model_outputs or reshape_out_name in model_outputs:
                continue
            input_tensor = model_ir.tensors.get(pre_in_name, None)
            pre_out_tensor = model_ir.tensors.get(pre_out_name, None)
            reshape_out_tensor = model_ir.tensors.get(reshape_out_name, None)
            post_out_tensor = model_ir.tensors.get(post_out_name, None)
            if (
                input_tensor is None
                or pre_out_tensor is None
                or reshape_out_tensor is None
                or post_out_tensor is None
            ):
                continue

            in_shape = [int(v) for v in list(input_tensor.shape)]
            pre_out_shape = [int(v) for v in list(pre_out_tensor.shape)]
            reshape_out_shape = [int(v) for v in list(reshape_out_tensor.shape)]
            post_out_shape = [int(v) for v in list(post_out_tensor.shape)]
            if len(in_shape) != 4 or len(pre_out_shape) != 4 or len(reshape_out_shape) != 5 or len(post_out_shape) != 5:
                continue

            n, h, w, c = in_shape
            expected_pre_out = [n, c, h, w]
            if not _shapes_match_if_known(pre_out_shape, expected_pre_out):
                continue
            if post_perm == perm_n1chw_to_n1hwc:
                expected_reshape_out = [n, 1, c, h, w]
                expected_post_out = [n, 1, h, w, c]
                reshape_singleton_axis = 1
                post_singleton_axis = 1
            elif post_perm == perm_nchw1_to_nhwc1:
                expected_reshape_out = [n, c, h, w, 1]
                expected_post_out = [n, h, w, c, 1]
                reshape_singleton_axis = 4
                post_singleton_axis = 4
            else:
                continue

            # Case A:
            #   NHWC -> NCHW -> RESHAPE(insert singleton) -> transpose back
            #   => NHWC -> RESHAPE(insert singleton in NHWC order)
            if (
                _shapes_match_if_known(reshape_out_shape, expected_reshape_out)
                and _shapes_match_if_known(post_out_shape, expected_post_out)
                and _dims_compatible(int(reshape_out_shape[int(reshape_singleton_axis)]), 1)
                and _dims_compatible(int(post_out_shape[int(post_singleton_axis)]), 1)
            ):
                target_shape = [int(v) for v in list(post_out_shape)]
                if len(reshape_op.inputs) >= 2:
                    shape_tensor = model_ir.tensors.get(str(reshape_op.inputs[1]), None)
                    if _read_const_ints_from_tensor(shape_tensor) is None:
                        continue
                    if not _write_const_ints_to_tensor(shape_tensor, target_shape):
                        continue
                if isinstance(reshape_op.options, dict):
                    reshape_opts = dict(reshape_op.options)
                    for key in ["newShape", "onnxRawNewShape"]:
                        if isinstance(reshape_opts.get(key, None), list):
                            reshape_opts[key] = [int(v) for v in target_shape]
                    reshape_op.options = reshape_opts

                _set_operator_inputs(
                    model_ir=model_ir,
                    op=reshape_op,
                    new_inputs=[pre_in_name] + [str(v) for v in list(reshape_op.inputs[1:])],
                )
                _set_operator_outputs(
                    model_ir=model_ir,
                    op=reshape_op,
                    new_outputs=[post_out_name],
                )

                for remove_idx in sorted([int(post_idx), int(pre_idx)], reverse=True):
                    del model_ir.operators[int(remove_idx)]

                rewritten += 1
                changed = True
                break

            # Case B (YOLO decode style):
            #   NHWC -> NCHW -> RESHAPE([N,A,B,H,W]) -> TRANSPOSE([0,1,3,4,2]) -> [N,A,H,W,B]
            #   => NHWC -> RESHAPE([N,H,W,A,B]) -> TRANSPOSE([0,3,1,2,4]) -> [N,A,H,W,B]
            if post_perm == perm_n1chw_to_n1hwc:
                a = int(reshape_out_shape[1])
                b = int(reshape_out_shape[2])
                rh = int(reshape_out_shape[3])
                rw = int(reshape_out_shape[4])
                expected_post_out_factorized = [n, a, h, w, b]
                if not _shapes_match_if_known(post_out_shape, expected_post_out_factorized):
                    continue
                if not _dims_compatible(rh, h) or not _dims_compatible(rw, w):
                    continue
                if int(c) > 0 and int(a) > 0 and int(b) > 0 and int(c) != int(a) * int(b):
                    continue

                target_reshape_shape = [int(n), int(h), int(w), int(a), int(b)]
                if len(reshape_op.inputs) < 2:
                    continue
                reshape_shape_name = str(reshape_op.inputs[1])
                if (
                    reshape_shape_name in model_outputs
                    or set(
                        int(value)
                        for value in consumers.get(reshape_shape_name, [])
                    )
                    != {int(reshape_idx)}
                ):
                    continue
                reshape_shape_tensor = model_ir.tensors.get(
                    reshape_shape_name,
                    None,
                )
                reshape_shape_values = _read_const_ints_from_tensor(
                    reshape_shape_tensor
                )
                if (
                    reshape_shape_values is None
                    or reshape_shape_values == target_reshape_shape
                ):
                    continue
                if len(post_op.inputs) < 2:
                    continue
                post_perm_name = str(post_op.inputs[1])
                if (
                    post_perm_name in model_outputs
                    or set(
                        int(value)
                        for value in consumers.get(post_perm_name, [])
                    )
                    != {int(post_idx)}
                ):
                    continue
                post_perm_tensor = model_ir.tensors.get(post_perm_name, None)
                post_perm_values = _read_const_ints_from_tensor(post_perm_tensor)
                if (
                    post_perm_values is None
                    or post_perm_values == perm_nhwab_to_nahwb
                ):
                    continue

                if not _write_const_ints_to_tensor(
                    reshape_shape_tensor,
                    target_reshape_shape,
                ):
                    continue

                if isinstance(reshape_op.options, dict):
                    reshape_opts = dict(reshape_op.options)
                    for key in ["newShape", "onnxRawNewShape"]:
                        if isinstance(reshape_opts.get(key, None), list):
                            reshape_opts[key] = [int(v) for v in target_reshape_shape]
                    reshape_op.options = reshape_opts

                if not _write_const_ints_to_tensor(post_perm_tensor, perm_nhwab_to_nahwb):
                    continue

                reshape_shape_norm, reshape_signature_norm = normalize_onnx_shape(target_reshape_shape)
                reshape_out_tensor.shape = [int(v) for v in reshape_shape_norm]
                reshape_out_tensor.shape_signature = [int(v) for v in reshape_signature_norm]

                _set_operator_inputs(
                    model_ir=model_ir,
                    op=reshape_op,
                    new_inputs=[pre_in_name] + [str(v) for v in list(reshape_op.inputs[1:])],
                )
                del model_ir.operators[int(pre_idx)]

                rewritten += 1
                changed = True
                break

        if not changed:
            break

    _prune_unused_tensors(model_ir)
    if layout_state is not None:
        layout_state.remove(
            tensors_before_fallback_prune - set(str(name) for name in model_ir.tensors)
        )
    return {
        "optimized_transpose_reshape_transpose_to_expanddims_nhwc_chains": int(rewritten)
    }
