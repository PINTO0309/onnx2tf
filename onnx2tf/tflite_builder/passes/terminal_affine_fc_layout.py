from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np

from onnx2tf.tflite_builder.core.model_ir_utils import (
    _broadcast_static_shapes,
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

def optimize_terminal_transpose_mul_add_reshape_fc_nhwc_chains(
    model_ir: ModelIR,
) -> Dict[str, int]:
    """
    Remove terminal NHWC->NCHW transpose before MUL/ADD/RESHAPE/FULLY_CONNECTED.

    Target:
      x_nhwc --T(0,3,1,2)--> x_nchw
      x_nchw --MUL(c1)--> m_nchw
      m_nchw --ADD(c2)--> y_nchw
      y_nchw --RESHAPE([N,-1]|[N,K])--> r
      r --FULLY_CONNECTED(W,b)--> o

    Rewrite:
      x_nhwc --MUL(c1_nhwc)--> m_nhwc
      m_nhwc --ADD(c2_nhwc)--> y_nhwc
      y_nhwc --RESHAPE([N,-1]|[N,K])--> r
      r --FULLY_CONNECTED(W',b)--> o

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

    def _materialize_nhwc_const(
        *,
        tensor_name: str,
        target_nhwc_shape: List[int],
        consumers: Dict[str, List[int]],
        chain_indices: set[int],
    ) -> Optional[str]:
        side_tensor = model_ir.tensors.get(str(tensor_name), None)
        if side_tensor is None or side_tensor.data is None:
            return None
        side_data = np.asarray(side_tensor.data)
        if int(side_data.size) == 1:
            return str(tensor_name)

        if side_data.ndim != 4:
            return None
        side_shape = [int(v) for v in list(side_data.shape)]
        if _broadcast_static_shapes(target_nhwc_shape, side_shape) is not None:
            return str(tensor_name)

        rotated = np.transpose(side_data, perm_nchw_to_nhwc).astype(side_data.dtype, copy=False)
        rotated_shape = [int(v) for v in list(rotated.shape)]
        if _broadcast_static_shapes(target_nhwc_shape, rotated_shape) is None:
            return None

        side_users = [int(v) for v in consumers.get(str(tensor_name), [])]
        shared_outside_chain = any(int(u) not in chain_indices for u in side_users)
        if shared_outside_chain:
            replacement_name = _unique_tensor_name(f"{tensor_name}_nhwc")
            model_ir.tensors[replacement_name] = TensorIR(
                name=replacement_name,
                dtype=str(side_tensor.dtype),
                shape=[int(v) for v in list(rotated_shape)],
                shape_signature=[int(v) for v in list(rotated_shape)],
                data=np.asarray(rotated),
                is_variable=False,
                quantization=_clone_quantization(side_tensor.quantization),
            )
            return str(replacement_name)

        side_tensor.data = np.asarray(rotated)
        side_tensor.shape = [int(v) for v in list(rotated_shape)]
        side_tensor.shape_signature = [int(v) for v in list(rotated_shape)]
        return str(tensor_name)

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
            mul_idx = int(pre_users[0])
            mul_op = model_ir.operators[int(mul_idx)]
            if str(mul_op.op_type) != "MUL" or len(mul_op.inputs) != 2 or len(mul_op.outputs) != 1:
                continue

            mul_side_index: Optional[int] = None
            mul_side_name: Optional[str] = None
            if str(mul_op.inputs[0]) == pre_out_name:
                mul_side_index = 1
                mul_side_name = str(mul_op.inputs[1])
            elif str(mul_op.inputs[1]) == pre_out_name:
                mul_side_index = 0
                mul_side_name = str(mul_op.inputs[0])
            if mul_side_index is None or mul_side_name is None:
                continue

            mul_out_name = str(mul_op.outputs[0])
            if mul_out_name in model_outputs:
                continue
            mul_users = [int(v) for v in consumers.get(mul_out_name, [])]
            if len(mul_users) != 1:
                continue
            add_idx = int(mul_users[0])
            add_op = model_ir.operators[int(add_idx)]
            if str(add_op.op_type) != "ADD" or len(add_op.inputs) != 2 or len(add_op.outputs) != 1:
                continue

            add_side_index: Optional[int] = None
            add_side_name: Optional[str] = None
            if str(add_op.inputs[0]) == mul_out_name:
                add_side_index = 1
                add_side_name = str(add_op.inputs[1])
            elif str(add_op.inputs[1]) == mul_out_name:
                add_side_index = 0
                add_side_name = str(add_op.inputs[0])
            if add_side_index is None or add_side_name is None:
                continue

            add_out_name = str(add_op.outputs[0])
            if add_out_name in model_outputs:
                continue
            add_users = [int(v) for v in consumers.get(add_out_name, [])]
            if len(add_users) != 1:
                continue
            reshape_idx = int(add_users[0])
            reshape_op = model_ir.operators[int(reshape_idx)]
            if (
                str(reshape_op.op_type) != "RESHAPE"
                or len(reshape_op.inputs) < 2
                or len(reshape_op.outputs) != 1
                or str(reshape_op.inputs[0]) != add_out_name
            ):
                continue

            reshape_out_name = str(reshape_op.outputs[0])
            if reshape_out_name in model_outputs:
                continue
            reshape_users = [int(v) for v in consumers.get(reshape_out_name, [])]
            if len(reshape_users) != 1:
                continue
            fc_idx = int(reshape_users[0])
            fc_op = model_ir.operators[int(fc_idx)]
            if (
                str(fc_op.op_type) != "FULLY_CONNECTED"
                or len(fc_op.inputs) < 2
                or len(fc_op.outputs) != 1
                or str(fc_op.inputs[0]) != reshape_out_name
            ):
                continue

            pre_out_tensor = model_ir.tensors.get(pre_out_name, None)
            pre_in_tensor = model_ir.tensors.get(pre_in_name, None)
            add_out_tensor = model_ir.tensors.get(add_out_name, None)
            if (
                pre_out_tensor is None
                or pre_in_tensor is None
                or add_out_tensor is None
                or len(list(pre_out_tensor.shape)) != 4
                or len(list(pre_in_tensor.shape)) != 4
                or len(list(add_out_tensor.shape)) != 4
            ):
                continue

            nchw_shape = [int(v) for v in list(pre_out_tensor.shape)]
            nhwc_shape = [int(v) for v in list(pre_in_tensor.shape)]
            if _permute_shape(nchw_shape, perm_nchw_to_nhwc) != nhwc_shape:
                continue
            if not _is_fully_known_positive_shape(nchw_shape):
                continue

            n, c, h, w = [int(v) for v in nchw_shape]
            input_size = int(c * h * w)

            weight_name = str(fc_op.inputs[1])
            weight_tensor = model_ir.tensors.get(weight_name, None)
            if weight_tensor is None or weight_tensor.data is None:
                continue
            weight_data = np.asarray(weight_tensor.data)
            if weight_data.ndim != 2:
                continue

            perm_index = np.transpose(
                np.arange(input_size, dtype=np.int64).reshape(c, h, w),
                (1, 2, 0),
            ).reshape(-1)
            if weight_data.shape[1] == input_size:
                weight_nhwc = np.asarray(weight_data[:, perm_index])
            elif weight_data.shape[0] == input_size:
                weight_nhwc = np.asarray(weight_data[perm_index, :])
            else:
                continue

            chain_indices = {int(pre_idx), int(mul_idx), int(add_idx), int(reshape_idx), int(fc_idx)}
            mul_side_replacement = _materialize_nhwc_const(
                tensor_name=str(mul_side_name),
                target_nhwc_shape=[int(v) for v in list(nhwc_shape)],
                consumers=consumers,
                chain_indices=chain_indices,
            )
            if mul_side_replacement is None:
                continue
            add_side_replacement = _materialize_nhwc_const(
                tensor_name=str(add_side_name),
                target_nhwc_shape=[int(v) for v in list(nhwc_shape)],
                consumers=consumers,
                chain_indices=chain_indices,
            )
            if add_side_replacement is None:
                continue

            weight_users = [int(v) for v in consumers.get(weight_name, [])]
            selected_weight_name = str(weight_name)
            if len(weight_users) == 1 and int(weight_users[0]) == int(fc_idx):
                weight_tensor.data = np.asarray(weight_nhwc)
                weight_tensor.shape = [int(v) for v in list(weight_nhwc.shape)]
                weight_tensor.shape_signature = [int(v) for v in list(weight_nhwc.shape)]
            else:
                selected_weight_name = _unique_tensor_name(f"{weight_name}_nhwc")
                model_ir.tensors[selected_weight_name] = TensorIR(
                    name=selected_weight_name,
                    dtype=str(weight_tensor.dtype),
                    shape=[int(v) for v in list(weight_nhwc.shape)],
                    shape_signature=[int(v) for v in list(weight_nhwc.shape)],
                    data=np.asarray(weight_nhwc),
                    is_variable=False,
                    quantization=_clone_quantization(weight_tensor.quantization),
                )

            _replace_operator_input_at(
                model_ir=model_ir,
                op=mul_op,
                input_index=0 if int(mul_side_index) == 1 else 1,
                new_input_name=str(pre_in_name),
            )
            _replace_operator_input_at(
                model_ir=model_ir,
                op=mul_op,
                input_index=int(mul_side_index),
                new_input_name=str(mul_side_replacement),
            )
            _replace_operator_input_at(
                model_ir=model_ir,
                op=add_op,
                input_index=int(add_side_index),
                new_input_name=str(add_side_replacement),
            )
            _replace_operator_input_at(
                model_ir=model_ir,
                op=fc_op,
                input_index=1,
                new_input_name=str(selected_weight_name),
            )

            _permute_tensor_metadata_if_rank_matches(
                model_ir.tensors.get(mul_out_name, None),
                perm_nchw_to_nhwc,
            )
            _permute_tensor_metadata_if_rank_matches(
                model_ir.tensors.get(add_out_name, None),
                perm_nchw_to_nhwc,
            )

            del model_ir.operators[int(pre_idx)]
            optimized += 1
            changed = True
            break

        if not changed:
            break

    _prune_unused_tensors(model_ir)
    return {"optimized_terminal_transpose_mul_add_reshape_fc_nhwc_chains": int(optimized)}
