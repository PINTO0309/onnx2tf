from __future__ import annotations

from typing import Dict, List, Optional

from onnx2tf.tflite_builder.core.model_ir_utils import (
    _build_tensor_consumer_map,
    _build_tensor_producer_map,
    _clone_quantization,
    _permute_shape,
    _permute_tensor_metadata_if_rank_matches,
    _prune_unused_tensors,
    _read_const_ints_from_tensor,
    _read_transpose_perm,
    _set_operator_inputs,
    _set_operator_outputs,
    _write_const_ints_to_tensor,
)
from onnx2tf.tflite_builder.ir import ModelIR, OperatorIR

def _optimize_transpose_3d_leaky_logistic_muladd_ndhwc_chains(model_ir: ModelIR) -> Dict[str, int]:
    """
    Eliminate NDHWC<->NCDHW bridges around 3D LEAKY/ADD + LOGISTIC/MUL/ADD blocks.

    Target:
      base_nhwc --T(0,3,1,2)--> base_nchw --RESHAPE--> base_ncdhw
      skip_nhdwc --T(0,4,1,2,3)--> skip_ncdhw --LEAKY_RELU--> skip_leaky_ncdhw
      ADD(skip_leaky_ncdhw, base_ncdhw) -> add0_ncdhw --T(0,2,3,4,1)--> add0_ndhwc
      gate_ndhwc --T(0,4,1,2,3)--> gate_ncdhw --LOGISTIC--> gate_sig_ncdhw
      MUL(gate_sig_ncdhw, base_ncdhw) -> mul1_ncdhw
      ADD(mul1_ncdhw, skip_leaky_ncdhw) -> add1_ncdhw --T(0,2,3,4,1)--> add1_ndhwc

    Rewrite:
      - Remove the 5D pre/post transposes and keep the chain in NDHWC.
      - Rewrite base RESHAPE to consume base_nhwc directly and emit NDHWC.
      - Remove the 4D base transpose feeding that RESHAPE.

    Safety:
      - Uses a strict single-path motif with local consumers only.
      - Refuses graph/model-output boundary tensors.
    """
    rewritten = 0
    perm_ndhwc_to_ncdhw = [0, 4, 1, 2, 3]
    perm_ncdhw_to_ndhwc = [0, 2, 3, 4, 1]
    perm_nhwc_to_nchw = [0, 3, 1, 2]
    perm_nhwc_to_nchw = [0, 3, 1, 2]
    perm_nhwc_to_nchw = [0, 3, 1, 2]

    def _copy_tensor_metadata_with_perm(
        *,
        src_name: str,
        dst_name: str,
        perm: List[int],
    ) -> None:
        src_tensor = model_ir.tensors.get(str(src_name), None)
        dst_tensor = model_ir.tensors.get(str(dst_name), None)
        if src_tensor is None or dst_tensor is None:
            return
        dst_tensor.dtype = str(src_tensor.dtype)
        dst_tensor.quantization = _clone_quantization(src_tensor.quantization)
        dst_tensor.shape = [int(v) for v in list(src_tensor.shape)]
        dst_tensor.shape_signature = (
            [int(v) for v in list(src_tensor.shape_signature)]
            if src_tensor.shape_signature is not None
            else [int(v) for v in list(src_tensor.shape)]
        )
        _permute_tensor_metadata_if_rank_matches(
            dst_tensor,
            perm,
        )

    while True:
        changed = False
        consumers = _build_tensor_consumer_map(model_ir)
        producers = _build_tensor_producer_map(model_ir)
        model_outputs = set(str(v) for v in model_ir.outputs)

        for post1_idx, post1_op in enumerate(model_ir.operators):
            if str(post1_op.op_type) != "TRANSPOSE" or len(post1_op.inputs) < 2 or len(post1_op.outputs) != 1:
                continue
            if _read_transpose_perm(model_ir, post1_op) != perm_ncdhw_to_ndhwc:
                continue

            add1_out_name = str(post1_op.inputs[0])
            add1_post_out_name = str(post1_op.outputs[0])
            if add1_out_name in model_outputs or add1_post_out_name in model_outputs:
                continue

            add1_idx = producers.get(add1_out_name, None)
            if add1_idx is None:
                continue
            add1_op = model_ir.operators[int(add1_idx)]
            if (
                str(add1_op.op_type) != "ADD"
                or len(add1_op.inputs) != 2
                or len(add1_op.outputs) != 1
                or str(add1_op.outputs[0]) != add1_out_name
            ):
                continue
            if set(int(v) for v in consumers.get(add1_out_name, [])) != {int(post1_idx)}:
                continue

            add1_inputs = [str(v) for v in list(add1_op.inputs)]

            skip_name: Optional[str] = None
            base_name: Optional[str] = None
            for lhs_name, rhs_name in [(add1_inputs[0], add1_inputs[1]), (add1_inputs[1], add1_inputs[0])]:
                lhs_prod_idx = producers.get(str(lhs_name), None)
                if lhs_prod_idx is None:
                    continue
                lhs_prod_op = model_ir.operators[int(lhs_prod_idx)]
                if str(lhs_prod_op.op_type) != "LEAKY_RELU":
                    continue
                skip_name = str(lhs_name)
                base_name = str(rhs_name)
                break
            if skip_name is None or base_name is None:
                continue

            # Skip path: NDHWC -> TRANSPOSE -> LEAKY_RELU
            skip_leaky_idx = producers.get(skip_name, None)
            if skip_leaky_idx is None:
                continue
            skip_leaky_op = model_ir.operators[int(skip_leaky_idx)]
            if (
                str(skip_leaky_op.op_type) != "LEAKY_RELU"
                or len(skip_leaky_op.inputs) != 1
                or len(skip_leaky_op.outputs) != 1
                or str(skip_leaky_op.outputs[0]) != skip_name
            ):
                continue
            pre_skip_out_name = str(skip_leaky_op.inputs[0])
            if pre_skip_out_name in model_outputs:
                continue
            pre_skip_idx = producers.get(pre_skip_out_name, None)
            if pre_skip_idx is None:
                continue
            pre_skip_op = model_ir.operators[int(pre_skip_idx)]
            if (
                str(pre_skip_op.op_type) != "TRANSPOSE"
                or len(pre_skip_op.inputs) < 2
                or len(pre_skip_op.outputs) != 1
                or str(pre_skip_op.outputs[0]) != pre_skip_out_name
                or _read_transpose_perm(model_ir, pre_skip_op) != perm_ndhwc_to_ncdhw
            ):
                continue
            skip_ndhwc_name = str(pre_skip_op.inputs[0])
            if skip_ndhwc_name in model_outputs:
                continue
            if set(int(v) for v in consumers.get(pre_skip_out_name, [])) != {int(skip_leaky_idx)}:
                continue

            # Find second block:
            #   ADD(mul, skip_name) -> post2 transpose
            add2_candidates = [
                int(v)
                for v in consumers.get(skip_name, [])
                if int(v) != int(add1_idx)
                and str(model_ir.operators[int(v)].op_type) == "ADD"
            ]
            if len(add2_candidates) != 1:
                continue
            add2_idx = int(add2_candidates[0])
            add2_op = model_ir.operators[int(add2_idx)]
            if len(add2_op.inputs) != 2 or len(add2_op.outputs) != 1:
                continue
            add2_out_name = str(add2_op.outputs[0])
            if add2_out_name in model_outputs:
                continue
            add2_users = [int(v) for v in consumers.get(add2_out_name, [])]
            if len(add2_users) != 1:
                continue
            post2_idx = int(add2_users[0])
            post2_op = model_ir.operators[int(post2_idx)]
            if (
                str(post2_op.op_type) != "TRANSPOSE"
                or len(post2_op.inputs) < 2
                or len(post2_op.outputs) != 1
                or str(post2_op.inputs[0]) != add2_out_name
                or _read_transpose_perm(model_ir, post2_op) != perm_ncdhw_to_ndhwc
                or str(post2_op.outputs[0]) in model_outputs
            ):
                continue
            add2_post_out_name = str(post2_op.outputs[0])

            # add2 must be ADD(mul2, skip_name)
            add2_inputs = [str(v) for v in list(add2_op.inputs)]
            mul2_idx: Optional[int] = None
            if str(add2_inputs[0]) == str(skip_name):
                cand_idx = producers.get(str(add2_inputs[1]), None)
                if cand_idx is not None and str(model_ir.operators[int(cand_idx)].op_type) == "MUL":
                    mul2_idx = int(cand_idx)
            elif str(add2_inputs[1]) == str(skip_name):
                cand_idx = producers.get(str(add2_inputs[0]), None)
                if cand_idx is not None and str(model_ir.operators[int(cand_idx)].op_type) == "MUL":
                    mul2_idx = int(cand_idx)
            if mul2_idx is None:
                continue
            mul2_op = model_ir.operators[int(mul2_idx)]
            if len(mul2_op.inputs) != 2 or len(mul2_op.outputs) != 1:
                continue
            mul2_out_name = str(mul2_op.outputs[0])
            if mul2_out_name in model_outputs:
                continue
            if set(int(v) for v in consumers.get(mul2_out_name, [])) != {int(add2_idx)}:
                continue

            # mul2 must consume LOGISTIC(pre_gate) and base_name.
            mul2_inputs = [str(v) for v in list(mul2_op.inputs)]
            logistic_idx: Optional[int] = None
            if str(mul2_inputs[0]) == str(base_name):
                cand_idx = producers.get(str(mul2_inputs[1]), None)
                if cand_idx is not None and str(model_ir.operators[int(cand_idx)].op_type) == "LOGISTIC":
                    logistic_idx = int(cand_idx)
            elif str(mul2_inputs[1]) == str(base_name):
                cand_idx = producers.get(str(mul2_inputs[0]), None)
                if cand_idx is not None and str(model_ir.operators[int(cand_idx)].op_type) == "LOGISTIC":
                    logistic_idx = int(cand_idx)
            if logistic_idx is None:
                continue
            logistic_op = model_ir.operators[int(logistic_idx)]
            if (
                str(logistic_op.op_type) != "LOGISTIC"
                or len(logistic_op.inputs) != 1
                or len(logistic_op.outputs) != 1
            ):
                continue
            logistic_out_name = str(logistic_op.outputs[0])
            pre_gate_out_name = str(logistic_op.inputs[0])
            if pre_gate_out_name in model_outputs or logistic_out_name in model_outputs:
                continue
            if set(int(v) for v in consumers.get(logistic_out_name, [])) != {int(mul2_idx)}:
                continue
            pre_gate_idx = producers.get(pre_gate_out_name, None)
            if pre_gate_idx is None:
                continue
            pre_gate_op = model_ir.operators[int(pre_gate_idx)]
            if (
                str(pre_gate_op.op_type) != "TRANSPOSE"
                or len(pre_gate_op.inputs) < 2
                or len(pre_gate_op.outputs) != 1
                or str(pre_gate_op.outputs[0]) != pre_gate_out_name
                or _read_transpose_perm(model_ir, pre_gate_op) != perm_ndhwc_to_ncdhw
            ):
                continue
            gate_ndhwc_name = str(pre_gate_op.inputs[0])
            if gate_ndhwc_name in model_outputs:
                continue
            if set(int(v) for v in consumers.get(pre_gate_out_name, [])) != {int(logistic_idx)}:
                continue

            # base path:
            #   base_nhwc --T(0,3,1,2)--> base_nchw --RESHAPE--> base_name(NCDHW)
            base_reshape_idx = producers.get(base_name, None)
            if base_reshape_idx is None:
                continue
            base_reshape_op = model_ir.operators[int(base_reshape_idx)]
            if (
                str(base_reshape_op.op_type) != "RESHAPE"
                or len(base_reshape_op.inputs) < 2
                or len(base_reshape_op.outputs) != 1
                or str(base_reshape_op.outputs[0]) != str(base_name)
            ):
                continue
            base_shape_name = str(base_reshape_op.inputs[1])
            base_shape_tensor = model_ir.tensors.get(base_shape_name, None)
            base_shape_vals = _read_const_ints_from_tensor(base_shape_tensor)
            if base_shape_vals is None or len(base_shape_vals) != 5:
                continue
            pre_base_out_name = str(base_reshape_op.inputs[0])
            if pre_base_out_name in model_outputs:
                continue
            pre_base_idx = producers.get(pre_base_out_name, None)
            if pre_base_idx is None:
                continue
            pre_base_op = model_ir.operators[int(pre_base_idx)]
            if (
                str(pre_base_op.op_type) != "TRANSPOSE"
                or len(pre_base_op.inputs) < 2
                or len(pre_base_op.outputs) != 1
                or str(pre_base_op.outputs[0]) != pre_base_out_name
                or _read_transpose_perm(model_ir, pre_base_op) != perm_nhwc_to_nchw
            ):
                continue
            base_nhwc_name = str(pre_base_op.inputs[0])
            if base_nhwc_name in model_outputs:
                continue
            if set(int(v) for v in consumers.get(pre_base_out_name, [])) != {int(base_reshape_idx)}:
                continue

            # Strict local-consumer checks for safety.
            if set(int(v) for v in consumers.get(str(base_name), [])) != {int(add1_idx), int(mul2_idx)}:
                continue
            if set(int(v) for v in consumers.get(str(skip_name), [])) != {int(add1_idx), int(add2_idx)}:
                continue

            # Rewrite base reshape to NDHWC.
            _set_operator_inputs(
                model_ir=model_ir,
                op=base_reshape_op,
                new_inputs=[str(base_nhwc_name), str(base_shape_name)],
            )
            mapped_base_shape = _permute_shape(
                [int(v) for v in list(base_shape_vals)],
                perm_ncdhw_to_ndhwc,
            )
            if mapped_base_shape is None:
                continue
            _write_const_ints_to_tensor(base_shape_tensor, [int(v) for v in list(mapped_base_shape)])
            if isinstance(base_reshape_op.options, dict):
                reshape_opts = dict(base_reshape_op.options)
                reshape_opts_changed = False
                for key in ["newShape", "onnxRawNewShape"]:
                    value = reshape_opts.get(key, None)
                    if isinstance(value, list) and len(value) == 5:
                        remapped = _permute_shape([int(v) for v in list(value)], perm_ncdhw_to_ndhwc)
                        if remapped is not None and [int(v) for v in remapped] != [int(v) for v in value]:
                            reshape_opts[key] = [int(v) for v in list(remapped)]
                            reshape_opts_changed = True
                if reshape_opts_changed:
                    base_reshape_op.options = reshape_opts
            _permute_tensor_metadata_if_rank_matches(
                model_ir.tensors.get(str(base_name), None),
                perm_ncdhw_to_ndhwc,
            )

            # Bypass 5D pre-transposes.
            _set_operator_inputs(
                model_ir=model_ir,
                op=skip_leaky_op,
                new_inputs=[str(skip_ndhwc_name)],
            )
            _permute_tensor_metadata_if_rank_matches(
                model_ir.tensors.get(str(skip_name), None),
                perm_ncdhw_to_ndhwc,
            )

            _set_operator_inputs(
                model_ir=model_ir,
                op=logistic_op,
                new_inputs=[str(gate_ndhwc_name)],
            )
            _permute_tensor_metadata_if_rank_matches(
                model_ir.tensors.get(str(logistic_out_name), None),
                perm_ncdhw_to_ndhwc,
            )
            _permute_tensor_metadata_if_rank_matches(
                model_ir.tensors.get(str(mul2_out_name), None),
                perm_ncdhw_to_ndhwc,
            )

            # Bypass post-transpose outputs by letting ADD emit NDHWC tensors directly.
            _set_operator_outputs(
                model_ir=model_ir,
                op=add1_op,
                new_outputs=[str(add1_post_out_name)],
            )
            _copy_tensor_metadata_with_perm(
                src_name=str(add1_out_name),
                dst_name=str(add1_post_out_name),
                perm=perm_ncdhw_to_ndhwc,
            )

            _set_operator_outputs(
                model_ir=model_ir,
                op=add2_op,
                new_outputs=[str(add2_post_out_name)],
            )
            _copy_tensor_metadata_with_perm(
                src_name=str(add2_out_name),
                dst_name=str(add2_post_out_name),
                perm=perm_ncdhw_to_ndhwc,
            )

            remove_indices = sorted(
                list(
                    {
                        int(pre_base_idx),
                        int(pre_skip_idx),
                        int(pre_gate_idx),
                        int(post1_idx),
                        int(post2_idx),
                    }
                ),
                reverse=True,
            )
            for remove_idx in remove_indices:
                del model_ir.operators[int(remove_idx)]

            rewritten += 1
            changed = True
            break

        if not changed:
            break

    _prune_unused_tensors(model_ir)
    return {"optimized_transpose_3d_leaky_logistic_muladd_ndhwc_chains": int(rewritten)}
