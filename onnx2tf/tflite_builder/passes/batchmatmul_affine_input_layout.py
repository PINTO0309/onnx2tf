from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np

from onnx2tf.tflite_builder.core.model_ir_utils import (
    _build_tensor_consumer_map,
    _build_tensor_producer_map,
    _permute_tensor_metadata_if_rank_matches,
    _prune_unused_tensors,
    _read_const_ints_from_tensor,
    _read_transpose_perm,
    _replace_operator_input_at,
    _write_const_ints_to_tensor,
)
from onnx2tf.tflite_builder.ir import ModelIR, OperatorIR


def optimize_batchmatmul_affine_transpose_input_chains(model_ir: ModelIR) -> Dict[str, int]:
    """
    Fold NCHW bridge transposes around affine branches feeding BATCH_MATMUL.

    Target:
      lhs_nhwc --T(0,3,1,2)--> lhs_nchw --MUL(const)--ADD(const)--RESHAPE([N,C,S])--T(0,2,1)--> lhs
      rhs_nhwc --T(0,3,1,2)--> rhs_nchw --MUL(const)--ADD(const)--RESHAPE([N,C,M])--> rhs
      BATCH_MATMUL(lhs, rhs, adjY=False)

    Rewrite:
      lhs_nhwc --MUL(const_nhwc)--ADD(const_nhwc)--RESHAPE([N,S,C])--> lhs
      rhs_nhwc --MUL(const_nhwc)--ADD(const_nhwc)--RESHAPE([N,M,C])--> rhs
      BATCH_MATMUL(lhs, rhs, adjY=True)
    """
    rewritten = 0
    perm_nhwc_to_nchw = [0, 3, 1, 2]
    perm_nchw_to_nhwc = [0, 2, 3, 1]
    perm_swap_last2_rank3 = [0, 2, 1]

    def _is_nchw_channelwise_or_singleton_const(tensor_name: str) -> bool:
        tensor = model_ir.tensors.get(str(tensor_name), None)
        if tensor is None or tensor.data is None:
            return False
        try:
            array = np.asarray(tensor.data)
        except Exception:
            return False
        if int(array.size) == 1:
            return True
        if array.ndim != 4:
            return False
        return (
            int(array.shape[0]) == 1
            and int(array.shape[2]) == 1
            and int(array.shape[3]) == 1
            and int(array.shape[1]) > 0
        )

    def _permute_const_nchw_to_nhwc(tensor_name: str) -> bool:
        tensor = model_ir.tensors.get(str(tensor_name), None)
        if tensor is None or tensor.data is None:
            return False
        try:
            array = np.asarray(tensor.data)
        except Exception:
            return False
        if array.ndim != 4:
            return False
        try:
            transposed = np.transpose(array, axes=perm_nchw_to_nhwc)
        except Exception:
            return False
        tensor.data = np.asarray(transposed)
        tensor.shape = [int(v) for v in list(transposed.shape)]
        tensor.shape_signature = [int(v) for v in list(transposed.shape)]
        return True

    def _match_affine_branch_from_reshape_output(
        *,
        reshape_output_name: str,
        terminal_consumer_idx: int,
        consumers: Dict[str, List[int]],
        producers: Dict[str, int],
        model_outputs: set[str],
    ) -> Optional[Dict[str, Any]]:
        reshape_idx = producers.get(str(reshape_output_name), None)
        if reshape_idx is None:
            return None
        reshape_op = model_ir.operators[int(reshape_idx)]
        if (
            str(reshape_op.op_type) != "RESHAPE"
            or len(reshape_op.inputs) < 2
            or len(reshape_op.outputs) != 1
            or str(reshape_op.outputs[0]) != str(reshape_output_name)
        ):
            return None
        if set(int(v) for v in consumers.get(str(reshape_output_name), [])) != {int(terminal_consumer_idx)}:
            return None

        add_out_name = str(reshape_op.inputs[0])
        add_idx = producers.get(add_out_name, None)
        if add_idx is None:
            return None
        add_op = model_ir.operators[int(add_idx)]
        if (
            str(add_op.op_type) != "ADD"
            or len(add_op.inputs) != 2
            or len(add_op.outputs) != 1
            or str(add_op.outputs[0]) != str(add_out_name)
        ):
            return None
        if set(int(v) for v in consumers.get(add_out_name, [])) != {int(reshape_idx)}:
            return None

        add_inputs = [str(v) for v in list(add_op.inputs)]
        mul_out_name: Optional[str] = None
        add_side_name: Optional[str] = None
        for candidate_name, side_name in [(add_inputs[0], add_inputs[1]), (add_inputs[1], add_inputs[0])]:
            candidate_idx = producers.get(str(candidate_name), None)
            if candidate_idx is None:
                continue
            candidate_op = model_ir.operators[int(candidate_idx)]
            if str(candidate_op.op_type) != "MUL":
                continue
            if not _is_nchw_channelwise_or_singleton_const(str(side_name)):
                return None
            mul_out_name = str(candidate_name)
            add_side_name = str(side_name)
            break
        if mul_out_name is None or add_side_name is None:
            return None

        mul_idx = producers.get(mul_out_name, None)
        if mul_idx is None:
            return None
        mul_op = model_ir.operators[int(mul_idx)]
        if (
            str(mul_op.op_type) != "MUL"
            or len(mul_op.inputs) != 2
            or len(mul_op.outputs) != 1
            or str(mul_op.outputs[0]) != str(mul_out_name)
        ):
            return None
        if set(int(v) for v in consumers.get(mul_out_name, [])) != {int(add_idx)}:
            return None

        mul_inputs = [str(v) for v in list(mul_op.inputs)]
        pre_output_name: Optional[str] = None
        mul_const_name: Optional[str] = None
        mul_data_input_index: Optional[int] = None
        pre_idx: Optional[int] = None
        pre_op: Optional[OperatorIR] = None
        for input_index, input_name in enumerate(mul_inputs):
            side_name = str(mul_inputs[1 - int(input_index)])
            candidate_idx = producers.get(str(input_name), None)
            if candidate_idx is None:
                continue
            candidate_op = model_ir.operators[int(candidate_idx)]
            if (
                str(candidate_op.op_type) != "TRANSPOSE"
                or len(candidate_op.inputs) < 2
                or len(candidate_op.outputs) != 1
                or str(candidate_op.outputs[0]) != str(input_name)
                or _read_transpose_perm(model_ir, candidate_op) != perm_nhwc_to_nchw
            ):
                continue
            if not _is_nchw_channelwise_or_singleton_const(side_name):
                return None
            pre_output_name = str(input_name)
            mul_const_name = str(side_name)
            mul_data_input_index = int(input_index)
            pre_idx = int(candidate_idx)
            pre_op = candidate_op
            break
        if (
            pre_idx is None
            or pre_op is None
            or pre_output_name is None
            or mul_const_name is None
            or mul_data_input_index is None
        ):
            return None
        if pre_output_name in model_outputs:
            return None
        if set(int(v) for v in consumers.get(pre_output_name, [])) != {int(mul_idx)}:
            return None

        source_nhwc_name = str(pre_op.inputs[0])
        if source_nhwc_name in model_outputs:
            return None

        return {
            "pre_idx": int(pre_idx),
            "mul_idx": int(mul_idx),
            "add_idx": int(add_idx),
            "reshape_idx": int(reshape_idx),
            "source_nhwc_name": str(source_nhwc_name),
            "mul_out_name": str(mul_out_name),
            "add_out_name": str(add_out_name),
            "reshape_out_name": str(reshape_output_name),
            "mul_const_name": str(mul_const_name),
            "add_const_name": str(add_side_name),
            "mul_data_input_index": int(mul_data_input_index),
            "reshape_shape_name": str(reshape_op.inputs[1]),
        }

    while True:
        changed = False
        consumers = _build_tensor_consumer_map(model_ir)
        producers = _build_tensor_producer_map(model_ir)
        model_outputs = set(str(v) for v in model_ir.outputs)

        for bmm_idx, bmm_op in enumerate(model_ir.operators):
            if str(bmm_op.op_type) != "BATCH_MATMUL" or len(bmm_op.inputs) != 2:
                continue
            bmm_options = dict(bmm_op.options) if isinstance(bmm_op.options, dict) else {}
            if bool(bmm_options.get("adjY", False)):
                continue

            lhs_input_name = str(bmm_op.inputs[0])
            rhs_input_name = str(bmm_op.inputs[1])

            lhs_post_idx = producers.get(lhs_input_name, None)
            if lhs_post_idx is None:
                continue
            lhs_post_op = model_ir.operators[int(lhs_post_idx)]
            if (
                str(lhs_post_op.op_type) != "TRANSPOSE"
                or len(lhs_post_op.inputs) < 2
                or len(lhs_post_op.outputs) != 1
                or str(lhs_post_op.outputs[0]) != lhs_input_name
                or _read_transpose_perm(model_ir, lhs_post_op) != perm_swap_last2_rank3
            ):
                continue
            if set(int(v) for v in consumers.get(lhs_input_name, [])) != {int(bmm_idx)}:
                continue
            lhs_reshape_out_name = str(lhs_post_op.inputs[0])

            lhs_branch = _match_affine_branch_from_reshape_output(
                reshape_output_name=lhs_reshape_out_name,
                terminal_consumer_idx=int(lhs_post_idx),
                consumers=consumers,
                producers=producers,
                model_outputs=model_outputs,
            )
            if lhs_branch is None:
                continue

            rhs_branch = _match_affine_branch_from_reshape_output(
                reshape_output_name=rhs_input_name,
                terminal_consumer_idx=int(bmm_idx),
                consumers=consumers,
                producers=producers,
                model_outputs=model_outputs,
            )
            if rhs_branch is None:
                continue

            shape_rewrite_ok = True
            for branch in [lhs_branch, rhs_branch]:
                mul_op = model_ir.operators[int(branch["mul_idx"])]
                _replace_operator_input_at(
                    model_ir=model_ir,
                    op=mul_op,
                    input_index=int(branch["mul_data_input_index"]),
                    new_input_name=str(branch["source_nhwc_name"]),
                )

                _permute_tensor_metadata_if_rank_matches(
                    model_ir.tensors.get(str(branch["mul_out_name"]), None),
                    perm_nchw_to_nhwc,
                )
                _permute_tensor_metadata_if_rank_matches(
                    model_ir.tensors.get(str(branch["add_out_name"]), None),
                    perm_nchw_to_nhwc,
                )

                _permute_const_nchw_to_nhwc(str(branch["mul_const_name"]))
                _permute_const_nchw_to_nhwc(str(branch["add_const_name"]))

                reshape_op = model_ir.operators[int(branch["reshape_idx"])]
                shape_tensor = model_ir.tensors.get(str(branch["reshape_shape_name"]), None)
                shape_vals = _read_const_ints_from_tensor(shape_tensor)
                if shape_vals is None or len(shape_vals) != 3:
                    shape_rewrite_ok = False
                    break
                new_shape = [int(shape_vals[0]), int(shape_vals[2]), int(shape_vals[1])]
                _write_const_ints_to_tensor(shape_tensor, [int(v) for v in new_shape])

                if isinstance(reshape_op.options, dict):
                    reshape_opts = dict(reshape_op.options)
                    for key in ["newShape", "onnxRawNewShape"]:
                        value = reshape_opts.get(key, None)
                        if isinstance(value, list) and len(value) == 3:
                            reshape_opts[key] = [int(new_shape[0]), int(new_shape[1]), int(new_shape[2])]
                    reshape_op.options = reshape_opts

                reshape_out_tensor = model_ir.tensors.get(str(branch["reshape_out_name"]), None)
                if reshape_out_tensor is not None:
                    reshape_out_tensor.shape = [int(v) for v in new_shape]
                    reshape_out_tensor.shape_signature = [int(v) for v in new_shape]

            if not shape_rewrite_ok:
                continue

            _replace_operator_input_at(
                model_ir=model_ir,
                op=bmm_op,
                input_index=0,
                new_input_name=str(lhs_branch["reshape_out_name"]),
            )
            _replace_operator_input_at(
                model_ir=model_ir,
                op=bmm_op,
                input_index=1,
                new_input_name=str(rhs_branch["reshape_out_name"]),
            )
            bmm_options["adjY"] = True
            bmm_op.options = bmm_options

            remove_indices = {
                int(lhs_post_idx),
                int(lhs_branch["pre_idx"]),
                int(rhs_branch["pre_idx"]),
            }
            for remove_idx in sorted(remove_indices, reverse=True):
                del model_ir.operators[int(remove_idx)]

            rewritten += 1
            changed = True
            break

        if not changed:
            break

    _prune_unused_tensors(model_ir)
    return {"optimized_batchmatmul_affine_transpose_input_chains": int(rewritten)}
