from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np

from onnx2tf.tflite_builder.core.model_ir_utils import (
    _build_tensor_consumer_map,
    _build_tensor_producer_map,
    _clone_quantization,
    _is_fully_known_positive_shape,
    _permute_shape,
    _permute_tensor_metadata_if_rank_matches,
    _prune_unused_tensors,
    _read_transpose_perm,
    _replace_operator_input_at,
)
from onnx2tf.tflite_builder.ir import ModelIR, TensorIR

def optimize_terminal_transpose_prelu_reshape_batchmatmul_nhwc_chains(
    model_ir: ModelIR,
) -> Dict[str, int]:
    """
    Remove terminal NHWC->NCHW transpose before PRELU/RESHAPE/BATCH_MATMUL.

    Target:
      x_nhwc --T(0,3,1,2)--> x_nchw
      x_nchw --PRELU(alpha_nchw)--> p_nchw
      p_nchw --RESHAPE([N,-1]|[N,K])--> r
      r --BATCH_MATMUL(W)--> o

    Rewrite:
      x_nhwc --PRELU(alpha_nhwc)--> p_nhwc
      p_nhwc --RESHAPE([N,-1]|[N,K])--> r
      r --BATCH_MATMUL(W')--> o

    where W' is input-axis permuted for NHWC flatten ordering.
    """
    optimized = 0
    perm_nhwc_to_nchw = [0, 3, 1, 2]
    perm_nchw_to_nhwc = [0, 2, 3, 1]

    def _unique_tensor_name(base: str) -> str:
        name = str(base)
        suffix = 1
        while name in model_ir.tensors:
            name = f"{base}_{suffix}"
            suffix += 1
        return name

    def _materialize_prelu_alpha_nhwc(
        *,
        alpha_name: str,
        channel_dim: int,
        consumers: Dict[str, List[int]],
        chain_indices: set[int],
    ) -> Optional[str]:
        alpha_tensor = model_ir.tensors.get(str(alpha_name), None)
        if alpha_tensor is None or alpha_tensor.data is None:
            return None
        alpha_data = np.asarray(alpha_tensor.data)
        if int(alpha_data.size) == 1:
            return str(alpha_name)

        alpha_nhwc: Optional[np.ndarray] = None
        alpha_shape = [int(v) for v in list(alpha_data.shape)]
        if alpha_data.ndim == 4:
            if (
                int(alpha_shape[0]) == 1
                and int(alpha_shape[1]) == int(channel_dim)
                and int(alpha_shape[2]) == 1
                and int(alpha_shape[3]) == 1
            ):
                alpha_nhwc = np.transpose(alpha_data, perm_nchw_to_nhwc).astype(alpha_data.dtype, copy=False)
            elif (
                int(alpha_shape[0]) == 1
                and int(alpha_shape[1]) == 1
                and int(alpha_shape[2]) == 1
                and int(alpha_shape[3]) == int(channel_dim)
            ):
                return str(alpha_name)
        elif alpha_data.ndim == 3:
            if (
                int(alpha_shape[0]) == int(channel_dim)
                and int(alpha_shape[1]) == 1
                and int(alpha_shape[2]) == 1
            ):
                alpha_nhwc = np.transpose(alpha_data, (1, 2, 0)).astype(alpha_data.dtype, copy=False)
        elif alpha_data.ndim == 1:
            # 1D slope is ambiguous for rank-4 layout in this converter; keep strict.
            return None

        if alpha_nhwc is None:
            return None

        alpha_users = [int(v) for v in consumers.get(str(alpha_name), [])]
        shared_outside_chain = any(int(u) not in chain_indices for u in alpha_users)
        if shared_outside_chain:
            replacement_name = _unique_tensor_name(f"{alpha_name}_nhwc")
            replacement_shape = [int(v) for v in list(alpha_nhwc.shape)]
            model_ir.tensors[replacement_name] = TensorIR(
                name=replacement_name,
                dtype=str(alpha_tensor.dtype),
                shape=replacement_shape,
                shape_signature=replacement_shape,
                data=np.asarray(alpha_nhwc),
                is_variable=False,
                quantization=_clone_quantization(alpha_tensor.quantization),
            )
            return str(replacement_name)

        alpha_tensor.data = np.asarray(alpha_nhwc)
        alpha_tensor.shape = [int(v) for v in list(alpha_nhwc.shape)]
        alpha_tensor.shape_signature = [int(v) for v in list(alpha_nhwc.shape)]
        return str(alpha_name)

    while True:
        changed = False
        consumers = _build_tensor_consumer_map(model_ir)
        producers = _build_tensor_producer_map(model_ir)  # noqa: F841 - compatibility scan
        model_outputs = set(str(v) for v in model_ir.outputs)

        for pre_idx, pre_op in enumerate(model_ir.operators):
            if (
                str(pre_op.op_type) != "TRANSPOSE"
                or len(pre_op.inputs) < 2
                or len(pre_op.outputs) != 1
                or _read_transpose_perm(model_ir, pre_op) != perm_nhwc_to_nchw
            ):
                continue

            pre_in_name = str(pre_op.inputs[0])
            pre_out_name = str(pre_op.outputs[0])
            if pre_out_name in model_outputs:
                continue

            pre_users = [int(v) for v in consumers.get(pre_out_name, [])]
            if len(pre_users) != 1:
                continue
            prelu_idx = int(pre_users[0])
            prelu_op = model_ir.operators[int(prelu_idx)]
            if (
                str(prelu_op.op_type) != "PRELU"
                or len(prelu_op.inputs) != 2
                or len(prelu_op.outputs) != 1
                or str(prelu_op.inputs[0]) != pre_out_name
            ):
                continue
            prelu_out_name = str(prelu_op.outputs[0])
            if prelu_out_name in model_outputs:
                continue

            prelu_users = [int(v) for v in consumers.get(prelu_out_name, [])]
            if len(prelu_users) != 1:
                continue
            reshape_idx = int(prelu_users[0])
            reshape_op = model_ir.operators[int(reshape_idx)]
            if (
                str(reshape_op.op_type) != "RESHAPE"
                or len(reshape_op.inputs) < 2
                or len(reshape_op.outputs) != 1
                or str(reshape_op.inputs[0]) != prelu_out_name
            ):
                continue
            reshape_out_name = str(reshape_op.outputs[0])
            if reshape_out_name in model_outputs:
                continue

            reshape_users = [int(v) for v in consumers.get(reshape_out_name, [])]
            if len(reshape_users) != 1:
                continue
            bmm_idx = int(reshape_users[0])
            bmm_op = model_ir.operators[int(bmm_idx)]
            if (
                str(bmm_op.op_type) != "BATCH_MATMUL"
                or len(bmm_op.inputs) != 2
                or len(bmm_op.outputs) != 1
                or str(bmm_op.inputs[0]) != reshape_out_name
            ):
                continue
            bmm_opts = dict(bmm_op.options) if isinstance(bmm_op.options, dict) else {}
            if bool(bmm_opts.get("adjX", False)) or bool(bmm_opts.get("adjY", False)):
                continue

            pre_out_tensor = model_ir.tensors.get(pre_out_name, None)
            pre_in_tensor = model_ir.tensors.get(pre_in_name, None)
            prelu_out_tensor = model_ir.tensors.get(prelu_out_name, None)
            if (
                pre_out_tensor is None
                or pre_in_tensor is None
                or prelu_out_tensor is None
                or len(list(pre_out_tensor.shape)) != 4
                or len(list(pre_in_tensor.shape)) != 4
                or len(list(prelu_out_tensor.shape)) != 4
            ):
                continue
            nchw_shape = [int(v) for v in list(pre_out_tensor.shape)]
            nhwc_shape = [int(v) for v in list(pre_in_tensor.shape)]
            if not _is_fully_known_positive_shape(nchw_shape):
                continue
            if _permute_shape(nchw_shape, perm_nchw_to_nhwc) != nhwc_shape:
                continue
            _, c, h, w = [int(v) for v in nchw_shape]
            input_size = int(c * h * w)

            rhs_name = str(bmm_op.inputs[1])
            rhs_tensor = model_ir.tensors.get(rhs_name, None)
            if rhs_tensor is None or rhs_tensor.data is None:
                continue
            rhs_data = np.asarray(rhs_tensor.data)
            if rhs_data.ndim != 2 or int(rhs_data.shape[0]) != int(input_size):
                continue

            perm_index = np.transpose(
                np.arange(input_size, dtype=np.int64).reshape(c, h, w),
                (1, 2, 0),
            ).reshape(-1)
            rhs_nhwc = np.asarray(rhs_data[perm_index, :])

            chain_indices = {int(pre_idx), int(prelu_idx), int(reshape_idx), int(bmm_idx)}
            alpha_name = str(prelu_op.inputs[1])
            alpha_replacement = _materialize_prelu_alpha_nhwc(
                alpha_name=str(alpha_name),
                channel_dim=int(c),
                consumers=consumers,
                chain_indices=chain_indices,
            )
            if alpha_replacement is None:
                continue

            rhs_users = [int(v) for v in consumers.get(rhs_name, [])]
            selected_rhs_name = str(rhs_name)
            if len(rhs_users) == 1 and int(rhs_users[0]) == int(bmm_idx):
                rhs_tensor.data = np.asarray(rhs_nhwc)
                rhs_tensor.shape = [int(v) for v in list(rhs_nhwc.shape)]
                rhs_tensor.shape_signature = [int(v) for v in list(rhs_nhwc.shape)]
            else:
                selected_rhs_name = _unique_tensor_name(f"{rhs_name}_nhwc")
                model_ir.tensors[selected_rhs_name] = TensorIR(
                    name=selected_rhs_name,
                    dtype=str(rhs_tensor.dtype),
                    shape=[int(v) for v in list(rhs_nhwc.shape)],
                    shape_signature=[int(v) for v in list(rhs_nhwc.shape)],
                    data=np.asarray(rhs_nhwc),
                    is_variable=False,
                    quantization=_clone_quantization(rhs_tensor.quantization),
                )

            _replace_operator_input_at(
                model_ir=model_ir,
                op=prelu_op,
                input_index=0,
                new_input_name=str(pre_in_name),
            )
            _replace_operator_input_at(
                model_ir=model_ir,
                op=prelu_op,
                input_index=1,
                new_input_name=str(alpha_replacement),
            )
            _replace_operator_input_at(
                model_ir=model_ir,
                op=bmm_op,
                input_index=1,
                new_input_name=str(selected_rhs_name),
            )

            _permute_tensor_metadata_if_rank_matches(
                model_ir.tensors.get(prelu_out_name, None),
                perm_nchw_to_nhwc,
            )

            del model_ir.operators[int(pre_idx)]
            optimized += 1
            changed = True
            break

        if not changed:
            break

    _prune_unused_tensors(model_ir)
    return {"optimized_terminal_transpose_prelu_reshape_batchmatmul_nhwc_chains": int(optimized)}
