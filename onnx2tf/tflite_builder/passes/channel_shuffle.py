from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from onnx2tf.tflite_builder.core.model_ir_utils import (
    _clone_quantization,
    _is_fully_known_positive_shape,
    _permute_tensor_metadata_if_rank_matches,
    _permute_shape,
    _prune_unused_tensors,
    _read_transpose_perm,
    _read_const_ints_from_tensor,
    _replace_tensor_inputs,
    _set_operator_inputs,
    _set_operator_outputs,
)
from onnx2tf.tflite_builder.core.graph import ModelIRGraphIndex
from onnx2tf.tflite_builder.core.layout import LayoutState
from onnx2tf.tflite_builder.core.model_ir_pass_state import (
    ModelIRPreflightResult,
    ModelIRPassState,
    ModelIRPassStateScope,
    run_model_ir_pass_group,
)
from onnx2tf.tflite_builder.core.passes import PassPhase, PassSpec
from onnx2tf.tflite_builder.ir import ModelIR, OperatorIR, TensorIR


_NHWC_TO_NCHW_PERM = [0, 3, 1, 2]
_NCHW_TO_NHWC_PERM = [0, 2, 3, 1]
_CHANNEL_SHUFFLE_SWAP_PERM = [0, 2, 1, 3, 4]
_NHWC_CHANNEL_SHUFFLE_UNARY_OPS = frozenset(
    {
        "RELU",
        "RELU6",
        "LOGISTIC",
        "TANH",
        "LEAKY_RELU",
        "HARD_SWISH",
        "ABS",
        "EXP",
        "NEG",
        "SQRT",
    }
)


def _optimize_shufflenet_transpose_shuffle_chains(
    model_ir: ModelIR,
    *,
    graph_index: ModelIRGraphIndex | None = None,
    layout_state: LayoutState | None = None,
) -> Dict[str, int]:
    """
    Optimize ShuffleNet-style channel-shuffle blocks that rely on NCHW transpose chains.

    Target (representative):
      x_nhwc
        -> TRANSPOSE(0,3,1,2)
        -> RESHAPE -> TRANSPOSE(1,0,2) -> RESHAPE
        -> GATHER(idx=0/1)
      one gather -> TRANSPOSE(0,2,3,1) -> conv branch -> TRANSPOSE(0,3,1,2) -> (optional unary)
      other gather -> skip branch
      CONCAT(axis=1, [skip_nchw, branch_nchw]) -> y_nchw

    Rewrite:
      x_nhwc -> GATHER(axis=3, even/odd channels) -> skip_nhwc / branch_nhwc
      conv branch runs in NHWC directly
      CONCAT(axis=3, [skip_nhwc, branch_nhwc]) -> y_nhwc
      y_nhwc -> TRANSPOSE(0,3,1,2) -> y_nchw

    This preserves downstream NCHW contracts while reducing redundant transpose chains.
    """
    optimized = 0
    graph_index = graph_index or ModelIRGraphIndex(model_ir)
    perm_nhwc_to_nchw = [0, 3, 1, 2]
    perm_nchw_to_nhwc = [0, 2, 3, 1]
    unary_ops = {
        "LOGISTIC",
        "RELU",
        "RELU6",
        "RELU_0_TO_1",
        "LEAKY_RELU",
        "TANH",
        "GELU",
        "HARD_SWISH",
        "ABS",
        "EXP",
        "NEG",
        "SQRT",
    }
    branch_passthrough_ops = {
        "CONV_2D",
        "DEPTHWISE_CONV_2D",
        "ADD",
        "SUB",
        "MUL",
        "DIV",
        "PAD",
        "PADV2",
        "MAX_POOL_2D",
        "AVERAGE_POOL_2D",
        "CAST",
    }.union(unary_ops)

    def _unique_tensor_name(base: str) -> str:
        if base not in model_ir.tensors:
            return base
        suffix = 1
        while True:
            candidate = f"{base}_{suffix}"
            if candidate not in model_ir.tensors:
                return candidate
            suffix += 1

    def _op_index(op_ref: OperatorIR) -> Optional[int]:
        return graph_index.operator_index(op_ref)

    def _is_singleton_channel_nhwc_to_nchw_reshape(
        *,
        input_name: str,
        output_name: str,
    ) -> bool:
        input_tensor = model_ir.tensors.get(str(input_name), None)
        output_tensor = model_ir.tensors.get(str(output_name), None)
        if input_tensor is None or output_tensor is None:
            return False
        in_shape = [int(v) for v in list(input_tensor.shape)]
        out_shape = [int(v) for v in list(output_tensor.shape)]
        if len(in_shape) != 4 or len(out_shape) != 4:
            return False
        return (
            int(in_shape[0]) == int(out_shape[0])
            and int(in_shape[1]) == int(out_shape[2])
            and int(in_shape[2]) == int(out_shape[3])
            and int(in_shape[3]) == 1
            and int(out_shape[1]) == 1
        )

    def _parse_shuffle_split_candidate(
        *,
        op: OperatorIR,
        split_base_name: str,
        half_channels: int,
    ) -> Optional[Tuple[int, str]]:
        """
        Parse one shuffle split branch selector and return:
          (selector_idx_value, output_name), where selector_idx_value in {0, 1}.

        Supported forms:
        - GATHER(axis=0, indices scalar in {0,1}) on rank-5 split base
        - SLICE(begin/size const) selecting channel-halves on NCHW rank-4 split base
        """
        if len(op.outputs) != 1:
            return None
        output_name = str(op.outputs[0])

        if str(op.op_type) == "GATHER":
            if len(op.inputs) < 2:
                return None
            if int(op.options.get("axis", 0)) != 0:
                return None
            idx_vals = _read_const_ints_from_tensor(model_ir.tensors.get(str(op.inputs[1]), None))
            if idx_vals is None or len(idx_vals) != 1:
                return None
            idx_value = int(idx_vals[0])
            if idx_value not in [0, 1]:
                return None
            return int(idx_value), output_name

        if str(op.op_type) == "SLICE":
            if len(op.inputs) < 3:
                return None
            if str(op.inputs[0]) != str(split_base_name):
                return None
            begin_vals = _read_const_ints_from_tensor(model_ir.tensors.get(str(op.inputs[1]), None))
            size_vals = _read_const_ints_from_tensor(model_ir.tensors.get(str(op.inputs[2]), None))
            if begin_vals is None or size_vals is None:
                return None
            if len(begin_vals) != len(size_vals) or len(begin_vals) != 4:
                return None

            axis = 1
            begin = [int(v) for v in begin_vals]
            size = [int(v) for v in size_vals]
            start = int(begin[axis])
            if start not in [0, int(half_channels)]:
                return None
            idx_value = 0 if int(start) == 0 else 1
            if int(size[axis]) != int(half_channels):
                return None
            for dim_idx in range(4):
                if dim_idx == axis:
                    continue
                if int(begin[dim_idx]) != 0:
                    return None
                # Non-channel dims can be full extent (-1) or explicit full size.
                # Anything else means this is not a simple channel-halving split.
                if int(size[dim_idx]) not in [-1]:
                    split_base_tensor = model_ir.tensors.get(str(split_base_name), None)
                    if split_base_tensor is None or len(list(split_base_tensor.shape)) != 4:
                        return None
                    split_dim = int(split_base_tensor.shape[dim_idx])
                    if int(split_dim) <= 0 or int(size[dim_idx]) != int(split_dim):
                        return None
            return int(idx_value), output_name

        return None

    while True:
        changed = False
        consumers = graph_index.consumers
        producers = graph_index.producers
        model_outputs = set(str(v) for v in model_ir.outputs)

        for concat0_idx, concat0_op in enumerate(model_ir.operators):
            if str(concat0_op.op_type) != "CONCATENATION" or len(concat0_op.outputs) != 1:
                continue
            if int(concat0_op.options.get("axis", -1)) != 3:
                continue
            x_nhwc_name = str(concat0_op.outputs[0])
            x_nhwc_tensor = model_ir.tensors.get(x_nhwc_name, None)
            if x_nhwc_tensor is None or len(list(x_nhwc_tensor.shape)) != 4:
                continue
            x_shape = [int(v) for v in list(x_nhwc_tensor.shape)]
            channels = int(x_shape[3])
            if int(channels) <= 0 or int(channels % 2) != 0:
                continue
            half_channels = int(channels // 2)

            t0_users = [int(v) for v in consumers.get(x_nhwc_name, [])]
            if len(t0_users) != 1:
                continue
            t0_idx = int(t0_users[0])
            t0_op = model_ir.operators[t0_idx]
            if str(t0_op.op_type) != "TRANSPOSE" or len(t0_op.inputs) < 2 or len(t0_op.outputs) != 1:
                continue
            if _read_transpose_perm(model_ir, t0_op) != perm_nhwc_to_nchw:
                continue
            x_nchw_name = str(t0_op.outputs[0])
            if x_nchw_name in model_outputs:
                continue

            r1_users = [int(v) for v in consumers.get(x_nchw_name, [])]
            if len(r1_users) != 1:
                continue
            r1_op = model_ir.operators[int(r1_users[0])]
            if str(r1_op.op_type) != "RESHAPE" or len(r1_op.inputs) < 1 or len(r1_op.outputs) != 1:
                continue
            r1_out = str(r1_op.outputs[0])
            if r1_out in model_outputs:
                continue

            t1_users = [int(v) for v in consumers.get(r1_out, [])]
            if len(t1_users) != 1:
                continue
            t1_op = model_ir.operators[int(t1_users[0])]
            if str(t1_op.op_type) != "TRANSPOSE" or len(t1_op.inputs) < 2 or len(t1_op.outputs) != 1:
                continue
            t1_perm = _read_transpose_perm(model_ir, t1_op)
            if t1_perm not in ([1, 0, 2], [0, 2, 1, 3, 4]):
                continue
            is_rank5_shuffle_swap = bool(t1_perm == [0, 2, 1, 3, 4])
            t1_out = str(t1_op.outputs[0])
            if t1_out in model_outputs:
                continue

            r2_users = [int(v) for v in consumers.get(t1_out, [])]
            if len(r2_users) != 1:
                continue
            r2_op = model_ir.operators[int(r2_users[0])]
            if str(r2_op.op_type) != "RESHAPE" or len(r2_op.inputs) < 1 or len(r2_op.outputs) != 1:
                continue
            split_base = str(r2_op.outputs[0])
            if split_base in model_outputs:
                continue

            split_user_indices = [int(v) for v in consumers.get(split_base, [])]
            if len(split_user_indices) != 2:
                continue
            split_candidates: Dict[int, Tuple[int, OperatorIR, str]] = {}
            split_ids: set[int] = set()
            valid_split_candidates = True
            for split_idx in split_user_indices:
                split_op = model_ir.operators[int(split_idx)]
                parsed = _parse_shuffle_split_candidate(
                    op=split_op,
                    split_base_name=split_base,
                    half_channels=int(half_channels),
                )
                if parsed is None:
                    valid_split_candidates = False
                    break
                idx_value, out_name = parsed
                if int(idx_value) in split_ids:
                    valid_split_candidates = False
                    break
                split_ids.add(int(idx_value))
                split_candidates[int(idx_value)] = (
                    int(split_idx),
                    split_op,
                    str(out_name),
                )
            if not valid_split_candidates or set(split_candidates.keys()) != {0, 1}:
                continue

            # Determine which split candidate feeds NHWC conv-branch through NCHW->NHWC transpose.
            conv_split_idx_value: Optional[int] = None
            pre_conv_t_idx: Optional[int] = None
            pre_conv_t_op: Optional[OperatorIR] = None
            pre_conv_out_name: Optional[str] = None
            for idx_value in [0, 1]:
                _, _, split_out = split_candidates[idx_value]
                users = [int(v) for v in consumers.get(split_out, [])]
                if len(users) != 1:
                    continue
                maybe_t_idx = int(users[0])
                maybe_t_op = model_ir.operators[maybe_t_idx]
                if (
                    str(maybe_t_op.op_type) == "TRANSPOSE"
                    and len(maybe_t_op.inputs) >= 2
                    and len(maybe_t_op.outputs) == 1
                    and _read_transpose_perm(model_ir, maybe_t_op) == perm_nchw_to_nhwc
                    and str(maybe_t_op.outputs[0]) not in model_outputs
                ):
                    conv_split_idx_value = int(idx_value)
                    pre_conv_t_idx = int(maybe_t_idx)
                    pre_conv_t_op = maybe_t_op
                    pre_conv_out_name = str(maybe_t_op.outputs[0])
                    break
            if (
                conv_split_idx_value is None
                or pre_conv_t_idx is None
                or pre_conv_t_op is None
                or pre_conv_out_name is None
            ):
                continue
            skip_split_idx_value = 1 - int(conv_split_idx_value)
            _, _, skip_split_out_name = split_candidates[int(skip_split_idx_value)]
            if skip_split_out_name in model_outputs:
                continue

            pre_conv_out_users = [int(v) for v in consumers.get(pre_conv_out_name, [])]
            if len(pre_conv_out_users) != 1:
                continue
            first_branch_op = model_ir.operators[int(pre_conv_out_users[0])]

            # Trace branch NHWC flow until NHWC->NCHW transpose.
            transpose_post_idx: Optional[int] = None
            transpose_post_op: Optional[OperatorIR] = None
            branch_nhwc_cursor = str(pre_conv_out_name)
            visited_tensors: set[str] = set()
            branch_trace_valid = True
            while True:
                if branch_nhwc_cursor in visited_tensors:
                    branch_trace_valid = False
                    break
                visited_tensors.add(branch_nhwc_cursor)
                users = [int(v) for v in consumers.get(branch_nhwc_cursor, [])]
                if len(users) != 1:
                    branch_trace_valid = False
                    break
                op_idx = int(users[0])
                op = model_ir.operators[op_idx]
                if (
                    str(op.op_type) == "TRANSPOSE"
                    and len(op.inputs) >= 2
                    and len(op.outputs) == 1
                    and _read_transpose_perm(model_ir, op) == perm_nhwc_to_nchw
                ):
                    transpose_post_idx = int(op_idx)
                    transpose_post_op = op
                    break
                if str(op.op_type) not in branch_passthrough_ops or len(op.outputs) != 1:
                    branch_trace_valid = False
                    break
                branch_nhwc_cursor = str(op.outputs[0])
            if (
                not branch_trace_valid
                or transpose_post_idx is None
                or transpose_post_op is None
            ):
                continue

            branch_nhwc_tail_name = str(transpose_post_op.inputs[0])
            branch_nchw_start_name = str(transpose_post_op.outputs[0])
            if branch_nchw_start_name in model_outputs:
                continue

            skip_users = [int(v) for v in consumers.get(skip_split_out_name, [])]
            if len(skip_users) != 1:
                continue
            concat1_idx = int(skip_users[0])
            concat1_op = model_ir.operators[concat1_idx]
            if str(concat1_op.op_type) != "CONCATENATION" or len(concat1_op.outputs) != 1:
                continue
            if int(concat1_op.options.get("axis", -1)) != 1:
                continue
            concat1_inputs = [str(v) for v in list(concat1_op.inputs)]
            if len(concat1_inputs) != 2 or skip_split_out_name not in concat1_inputs:
                continue

            branch_nchw_cursor = branch_nchw_start_name
            unary_chain_indices: List[int] = []
            branch_trace_to_concat_ok = True
            while True:
                users = [int(v) for v in consumers.get(branch_nchw_cursor, [])]
                if len(users) != 1:
                    branch_trace_to_concat_ok = False
                    break
                user_idx = int(users[0])
                if user_idx == int(concat1_idx):
                    break
                user_op = model_ir.operators[user_idx]
                if (
                    str(user_op.op_type) in unary_ops
                    and len(user_op.inputs) == 1
                    and len(user_op.outputs) == 1
                    and str(user_op.inputs[0]) == branch_nchw_cursor
                    and branch_nchw_cursor not in model_outputs
                ):
                    unary_chain_indices.append(int(user_idx))
                    branch_nchw_cursor = str(user_op.outputs[0])
                    continue
                branch_trace_to_concat_ok = False
                break
            if not branch_trace_to_concat_ok:
                continue
            branch_concat_input_nchw = str(branch_nchw_cursor)
            if branch_concat_input_nchw not in concat1_inputs:
                continue

            # Build NHWC branch gather tensors from concat0 output.
            x_signature = (
                list(x_nhwc_tensor.shape_signature)
                if x_nhwc_tensor.shape_signature is not None
                else list(x_shape)
            )
            even_odd_shape = [int(x_shape[0]), int(x_shape[1]), int(x_shape[2]), int(half_channels)]
            even_odd_signature = [int(v) for v in list(x_signature)]
            if len(even_odd_signature) == 4 and int(even_odd_signature[3]) > 0:
                even_odd_signature[3] = int(half_channels)
            even_name = _unique_tensor_name(f"{x_nhwc_name}_shuffle_even_nhwc")
            odd_name = _unique_tensor_name(f"{x_nhwc_name}_shuffle_odd_nhwc")
            model_ir.tensors[even_name] = TensorIR(
                name=even_name,
                dtype=str(x_nhwc_tensor.dtype),
                shape=list(even_odd_shape),
                shape_signature=[int(v) for v in list(even_odd_signature)],
                quantization=_clone_quantization(x_nhwc_tensor.quantization),
            )
            model_ir.tensors[odd_name] = TensorIR(
                name=odd_name,
                dtype=str(x_nhwc_tensor.dtype),
                shape=list(even_odd_shape),
                shape_signature=[int(v) for v in list(even_odd_signature)],
                quantization=_clone_quantization(x_nhwc_tensor.quantization),
            )

            if is_rank5_shuffle_swap:
                r1_tensor = model_ir.tensors.get(str(r1_out), None)
                if r1_tensor is None:
                    continue
                r1_shape = [int(v) for v in list(r1_tensor.shape)]
                if len(r1_shape) != 5 or not _is_fully_known_positive_shape(r1_shape):
                    continue
                groups = int(r1_shape[1])
                cpg = int(r1_shape[2])
                if (
                    int(groups) <= 1
                    or int(cpg) <= 0
                    or int(groups) != 2
                    or int(groups * cpg) != int(channels)
                ):
                    continue
                shuffle_indices = np.asarray(
                    [int((k % groups) * cpg + (k // groups)) for k in range(int(channels))],
                    dtype=np.int32,
                )
                first_split_indices = np.asarray(shuffle_indices[:int(half_channels)], dtype=np.int32)
                second_split_indices = np.asarray(shuffle_indices[int(half_channels):], dtype=np.int32)
            else:
                first_split_indices = np.asarray(
                    [int(v) for v in range(0, int(channels), 2)],
                    dtype=np.int32,
                )
                second_split_indices = np.asarray(
                    [int(v) for v in range(1, int(channels), 2)],
                    dtype=np.int32,
                )
            if (
                len(first_split_indices) != int(half_channels)
                or len(second_split_indices) != int(half_channels)
            ):
                continue

            even_indices_name = _unique_tensor_name(f"{x_nhwc_name}_shuffle_even_indices")
            odd_indices_name = _unique_tensor_name(f"{x_nhwc_name}_shuffle_odd_indices")
            model_ir.tensors[even_indices_name] = TensorIR(
                name=even_indices_name,
                dtype="INT32",
                shape=[int(half_channels)],
                shape_signature=[int(half_channels)],
                data=np.asarray(first_split_indices, dtype=np.int32),
                is_variable=False,
            )
            model_ir.tensors[odd_indices_name] = TensorIR(
                name=odd_indices_name,
                dtype="INT32",
                shape=[int(half_channels)],
                shape_signature=[int(half_channels)],
                data=np.asarray(second_split_indices, dtype=np.int32),
                is_variable=False,
            )

            even_gather_op = OperatorIR(
                op_type="GATHER",
                inputs=[x_nhwc_name, even_indices_name],
                outputs=[even_name],
                options={"axis": 3, "batchDims": 0},
            )
            odd_gather_op = OperatorIR(
                op_type="GATHER",
                inputs=[x_nhwc_name, odd_indices_name],
                outputs=[odd_name],
                options={"axis": 3, "batchDims": 0},
            )

            conv_branch_input_nhwc = even_name if int(conv_split_idx_value) == 0 else odd_name
            skip_branch_input_nhwc = odd_name if int(conv_split_idx_value) == 0 else even_name

            # Rewrite branch start input.
            _replace_tensor_inputs(
                model_ir,
                pre_conv_out_name,
                conv_branch_input_nhwc,
                graph_index=graph_index,
            )

            # Rewire unary tail (if present) to NHWC by bypassing post transpose.
            if len(unary_chain_indices) > 0:
                first_unary_idx = int(unary_chain_indices[0])
                _set_operator_inputs(
                    model_ir=model_ir,
                    op=model_ir.operators[first_unary_idx],
                    new_inputs=[branch_nhwc_tail_name],
                    graph_index=graph_index,
                )
                for unary_idx in unary_chain_indices:
                    unary_out_name = str(model_ir.operators[int(unary_idx)].outputs[0])
                    _permute_tensor_metadata_if_rank_matches(
                        model_ir.tensors.get(unary_out_name, None),
                        perm_nchw_to_nhwc,
                    )
                branch_concat_input_nhwc = str(model_ir.operators[int(unary_chain_indices[-1])].outputs[0])
            else:
                branch_concat_input_nhwc = branch_nhwc_tail_name

            concat1_out_name = str(concat1_op.outputs[0])
            concat1_out_tensor = model_ir.tensors.get(concat1_out_name, None)
            if concat1_out_tensor is None:
                continue
            concat1_out_signature = (
                list(concat1_out_tensor.shape_signature)
                if concat1_out_tensor.shape_signature is not None
                else list(concat1_out_tensor.shape)
            )
            concat1_out_nhwc_shape = _permute_shape(list(concat1_out_tensor.shape), perm_nchw_to_nhwc)
            concat1_out_nhwc_signature = _permute_shape(list(concat1_out_signature), perm_nchw_to_nhwc)
            if concat1_out_nhwc_shape is None or concat1_out_nhwc_signature is None:
                continue

            concat1_out_nhwc_name = _unique_tensor_name(f"{concat1_out_name}_nhwc")
            model_ir.tensors[concat1_out_nhwc_name] = TensorIR(
                name=concat1_out_nhwc_name,
                dtype=str(concat1_out_tensor.dtype),
                shape=[int(v) for v in list(concat1_out_nhwc_shape)],
                shape_signature=[int(v) for v in list(concat1_out_nhwc_signature)],
                quantization=_clone_quantization(concat1_out_tensor.quantization),
            )

            new_concat1_inputs: List[str] = []
            for input_name in concat1_inputs:
                if str(input_name) == str(skip_split_out_name):
                    new_concat1_inputs.append(str(skip_branch_input_nhwc))
                elif str(input_name) == str(branch_concat_input_nchw):
                    new_concat1_inputs.append(str(branch_concat_input_nhwc))
                else:
                    new_concat1_inputs.append(str(input_name))
            if str(skip_branch_input_nhwc) not in set(new_concat1_inputs):
                continue
            if str(branch_concat_input_nhwc) not in set(new_concat1_inputs):
                continue

            _set_operator_inputs(
                model_ir=model_ir,
                op=concat1_op,
                new_inputs=[str(v) for v in list(new_concat1_inputs)],
                graph_index=graph_index,
            )
            concat1_op.options["axis"] = 3
            _set_operator_outputs(
                model_ir=model_ir,
                op=concat1_op,
                new_outputs=[concat1_out_nhwc_name],
                graph_index=graph_index,
            )

            post_perm_name = str(transpose_post_op.inputs[1]) if len(transpose_post_op.inputs) >= 2 else ""
            post_perm_tensor = model_ir.tensors.get(post_perm_name, None)
            post_perm_vals = _read_const_ints_from_tensor(post_perm_tensor)
            if post_perm_vals != perm_nhwc_to_nchw:
                post_perm_name = _unique_tensor_name("shufflenet_concat_post_perm")
                model_ir.tensors[post_perm_name] = TensorIR(
                    name=post_perm_name,
                    dtype="INT32",
                    shape=[4],
                    shape_signature=[4],
                    data=np.asarray(perm_nhwc_to_nchw, dtype=np.int32),
                    is_variable=False,
                )

            concat1_current_idx = _op_index(concat1_op)
            if concat1_current_idx is None:
                continue
            graph_index.insert_operator(
                int(concat1_current_idx) + 1,
                OperatorIR(
                    op_type="TRANSPOSE",
                    inputs=[concat1_out_nhwc_name, post_perm_name],
                    outputs=[concat1_out_name],
                ),
            )

            concat0_current_idx = _op_index(concat0_op)
            if concat0_current_idx is None:
                continue
            graph_index.insert_operator(
                int(concat0_current_idx) + 1,
                even_gather_op,
            )
            graph_index.insert_operator(
                int(concat0_current_idx) + 2,
                odd_gather_op,
            )

            remove_ops = {
                id(op): op
                for op in (
                    t0_op,
                    r1_op,
                    t1_op,
                    r2_op,
                    split_candidates[0][1],
                    split_candidates[1][1],
                    pre_conv_t_op,
                    transpose_post_op,
                )
            }
            remove_indices = [
                int(operator_index)
                for op in remove_ops.values()
                if (operator_index := graph_index.operator_index(op)) is not None
            ]
            for remove_idx in sorted(list(set(remove_indices)), reverse=True):
                graph_index.remove_operator(int(remove_idx))

            optimized += 1
            changed = True
            break

        if not changed:
            break

    _prune_unused_tensors(model_ir, layout_state=layout_state)
    if layout_state is not None:
        layout_state.sync_from_model_ir(model_ir)
    return {"optimized_shufflenet_transpose_shuffle_chains": int(optimized)}


def run_two_way_channel_shuffle_cleanup(
    model_ir: ModelIR,
    *,
    layout_state: LayoutState | None = None,
    diagnostics: List[Dict[str, Any]] | None = None,
    state_scope: ModelIRPassStateScope | None = None,
) -> Dict[str, int]:
    """Canonicalize guarded two-way channel-shuffle branch/Concat graphs."""

    def _preflight(candidate_model: ModelIR) -> ModelIRPreflightResult:
        found_concat = False
        found_reshape = False
        found_transpose = False
        found_selector = False
        for visited, operator in enumerate(candidate_model.operators, start=1):
            operator_type = str(operator.op_type)
            found_concat = found_concat or operator_type == "CONCATENATION"
            found_reshape = found_reshape or operator_type == "RESHAPE"
            found_transpose = found_transpose or operator_type == "TRANSPOSE"
            found_selector = found_selector or operator_type in {"GATHER", "SLICE"}
            if found_concat and found_reshape and found_transpose and found_selector:
                return ModelIRPreflightResult(True, visited)
        return ModelIRPreflightResult(False, len(candidate_model.operators))

    def _has_candidate(pass_state: ModelIRPassState) -> bool:
        candidate_model = pass_state.model_ir
        model_outputs = set(str(name) for name in candidate_model.outputs)
        for concat_op in candidate_model.operators:
            if (
                str(concat_op.op_type) != "CONCATENATION"
                or len(concat_op.outputs) != 1
                or int(concat_op.options.get("axis", -1)) != 3
            ):
                continue
            concat_output_name = str(concat_op.outputs[0])
            concat_tensor = candidate_model.tensors.get(concat_output_name)
            if concat_tensor is None or len(concat_tensor.shape) != 4:
                continue
            channels = int(concat_tensor.shape[3])
            if channels <= 0 or channels % 2 != 0:
                continue
            concat_users = pass_state.graph_index.consumer_indices(
                concat_output_name
            )
            if len(concat_users) != 1:
                continue
            t0_op = candidate_model.operators[int(concat_users[0])]
            if (
                str(t0_op.op_type) != "TRANSPOSE"
                or len(t0_op.inputs) < 2
                or len(t0_op.outputs) != 1
                or _read_transpose_perm(candidate_model, t0_op)
                != _NHWC_TO_NCHW_PERM
                or str(t0_op.outputs[0]) in model_outputs
            ):
                continue
            t0_users = pass_state.graph_index.consumer_indices(str(t0_op.outputs[0]))
            if len(t0_users) != 1:
                continue
            r1_op = candidate_model.operators[int(t0_users[0])]
            if (
                str(r1_op.op_type) != "RESHAPE"
                or len(r1_op.inputs) < 1
                or len(r1_op.outputs) != 1
                or str(r1_op.outputs[0]) in model_outputs
            ):
                continue
            r1_users = pass_state.graph_index.consumer_indices(str(r1_op.outputs[0]))
            if len(r1_users) != 1:
                continue
            t1_op = candidate_model.operators[int(r1_users[0])]
            if (
                str(t1_op.op_type) != "TRANSPOSE"
                or len(t1_op.inputs) < 2
                or len(t1_op.outputs) != 1
                or _read_transpose_perm(candidate_model, t1_op)
                not in ([1, 0, 2], _CHANNEL_SHUFFLE_SWAP_PERM)
                or str(t1_op.outputs[0]) in model_outputs
            ):
                continue
            t1_users = pass_state.graph_index.consumer_indices(str(t1_op.outputs[0]))
            if len(t1_users) != 1:
                continue
            r2_op = candidate_model.operators[int(t1_users[0])]
            if (
                str(r2_op.op_type) != "RESHAPE"
                or len(r2_op.inputs) < 1
                or len(r2_op.outputs) != 1
                or str(r2_op.outputs[0]) in model_outputs
            ):
                continue
            selector_users = pass_state.graph_index.consumer_indices(
                str(r2_op.outputs[0])
            )
            if len(selector_users) != 2:
                continue
            selector_types = {
                str(candidate_model.operators[int(index)].op_type)
                for index in selector_users
            }
            if selector_types <= {"GATHER", "SLICE"}:
                return True
        return False

    def _run(pass_state: ModelIRPassState) -> Dict[str, int | bool]:
        stats = _optimize_shufflenet_transpose_shuffle_chains(
            pass_state.model_ir,
            graph_index=pass_state.graph_index,
            layout_state=pass_state.layout_state,
        )
        return {
            **stats,
            "changed": bool(
                stats.get("optimized_shufflenet_transpose_shuffle_chains", 0)
            ),
        }

    details, _ = run_model_ir_pass_group(
        model_ir,
        specs=[
            PassSpec(
                pass_id="canonicalize.two_way_channel_shuffle_branch",
                phase=PassPhase.CANONICALIZE,
                callback=_run,
                precondition=_has_candidate,
                priority=10,
                transactional=True,
            )
        ],
        layout_state=layout_state,
        default_details={"optimized_shufflenet_transpose_shuffle_chains": 0},
        diagnostics=diagnostics,
        state_scope=state_scope,
        preflight=_preflight,
    )
    return {str(key): int(value) for key, value in details.items()}



def _optimize_shufflenet_reshape_transpose_shuffle_nhwc_chains(
    model_ir: ModelIR,
    *,
    graph_index: ModelIRGraphIndex | None = None,
    layout_state: LayoutState | None = None,
) -> Dict[str, int]:
    """
    Collapse ShuffleNet channel-shuffle chains expressed as:

      x_nhwc
        -> TRANSPOSE(0,3,1,2)
        -> RESHAPE([N,g,cpg,H,W])
        -> TRANSPOSE(0,2,1,3,4)
        -> RESHAPE([N,C,H,W])
        -> TRANSPOSE(0,2,3,1)
        -> y_nhwc

    into:

      x_nhwc -> GATHER(axis=3, shuffle_indices) -> y_nhwc

    where C=g*cpg and shuffle_indices[k] = (k % g) * cpg + (k // g).
    """
    optimized = 0
    graph_index = graph_index or ModelIRGraphIndex(model_ir)

    def _unique_tensor_name(base: str) -> str:
        name = str(base)
        suffix = 1
        while name in model_ir.tensors:
            name = f"{base}_{suffix}"
            suffix += 1
        return name

    while True:
        changed = False
        consumers = graph_index.consumers
        model_outputs = set(str(v) for v in model_ir.outputs)

        for t0_idx, t0_op in enumerate(model_ir.operators):
            if str(t0_op.op_type) != "TRANSPOSE" or len(t0_op.inputs) < 2 or len(t0_op.outputs) != 1:
                continue
            if _read_transpose_perm(model_ir, t0_op) != _NHWC_TO_NCHW_PERM:
                continue

            x_nhwc_name = str(t0_op.inputs[0])
            x_nchw_name = str(t0_op.outputs[0])
            if x_nchw_name in model_outputs:
                continue

            t0_users = [int(v) for v in consumers.get(x_nchw_name, [])]
            if len(t0_users) == 0:
                continue

            for t0_user_idx in t0_users:
                gather_src_name = str(x_nhwc_name)
                unary_bridge_op: Optional[OperatorIR] = None
                r1_idx: Optional[int] = None
                r1_op: Optional[OperatorIR] = None

                t0_user_op = model_ir.operators[int(t0_user_idx)]
                if (
                    str(t0_user_op.op_type) == "RESHAPE"
                    and len(t0_user_op.inputs) >= 1
                    and len(t0_user_op.outputs) == 1
                    and str(t0_user_op.inputs[0]) == x_nchw_name
                ):
                    r1_idx = int(t0_user_idx)
                    r1_op = t0_user_op
                elif (
                    str(t0_user_op.op_type) in _NHWC_CHANNEL_SHUFFLE_UNARY_OPS
                    and len(t0_user_op.inputs) == 1
                    and len(t0_user_op.outputs) == 1
                    and str(t0_user_op.inputs[0]) == x_nchw_name
                    and str(t0_user_op.outputs[0]) not in model_outputs
                ):
                    unary_out_name = str(t0_user_op.outputs[0])
                    unary_users = [int(v) for v in consumers.get(unary_out_name, [])]
                    if len(unary_users) != 1:
                        continue
                    unary_r1_idx = int(unary_users[0])
                    unary_r1_op = model_ir.operators[int(unary_r1_idx)]
                    if (
                        str(unary_r1_op.op_type) != "RESHAPE"
                        or len(unary_r1_op.inputs) < 1
                        or len(unary_r1_op.outputs) != 1
                        or str(unary_r1_op.inputs[0]) != unary_out_name
                    ):
                        continue
                    r1_idx = int(unary_r1_idx)
                    r1_op = unary_r1_op
                    unary_bridge_op = t0_user_op
                    gather_src_name = str(unary_out_name)
                else:
                    continue

                if r1_idx is None or r1_op is None:
                    continue
                r1_out_name = str(r1_op.outputs[0])
                if r1_out_name in model_outputs:
                    continue

                r1_users = [int(v) for v in consumers.get(r1_out_name, [])]
                if len(r1_users) != 1:
                    continue
                t1_idx = int(r1_users[0])
                t1_op = model_ir.operators[int(t1_idx)]
                if (
                    str(t1_op.op_type) != "TRANSPOSE"
                    or len(t1_op.inputs) < 2
                    or len(t1_op.outputs) != 1
                    or str(t1_op.inputs[0]) != r1_out_name
                    or _read_transpose_perm(model_ir, t1_op) != _CHANNEL_SHUFFLE_SWAP_PERM
                ):
                    continue
                t1_out_name = str(t1_op.outputs[0])
                if t1_out_name in model_outputs:
                    continue

                t1_users = [int(v) for v in consumers.get(t1_out_name, [])]
                if len(t1_users) != 1:
                    continue
                r2_idx = int(t1_users[0])
                r2_op = model_ir.operators[int(r2_idx)]
                if (
                    str(r2_op.op_type) != "RESHAPE"
                    or len(r2_op.inputs) < 1
                    or len(r2_op.outputs) != 1
                    or str(r2_op.inputs[0]) != t1_out_name
                ):
                    continue
                r2_out_name = str(r2_op.outputs[0])
                if r2_out_name in model_outputs:
                    continue

                r2_users = [int(v) for v in consumers.get(r2_out_name, [])]
                if len(r2_users) != 1:
                    continue
                t2_idx = int(r2_users[0])
                t2_op = model_ir.operators[int(t2_idx)]
                if (
                    str(t2_op.op_type) != "TRANSPOSE"
                    or len(t2_op.inputs) < 2
                    or len(t2_op.outputs) != 1
                    or str(t2_op.inputs[0]) != r2_out_name
                    or _read_transpose_perm(model_ir, t2_op) != _NCHW_TO_NHWC_PERM
                ):
                    continue

                x_tensor = model_ir.tensors.get(x_nhwc_name, None)
                r1_tensor = model_ir.tensors.get(r1_out_name, None)
                t1_tensor = model_ir.tensors.get(t1_out_name, None)
                y_tensor = model_ir.tensors.get(str(t2_op.outputs[0]), None)
                if x_tensor is None or r1_tensor is None or t1_tensor is None:
                    continue

                x_shape = [int(v) for v in list(x_tensor.shape)]
                r1_shape = [int(v) for v in list(r1_tensor.shape)]
                t1_shape = [int(v) for v in list(t1_tensor.shape)]
                if (
                    not _is_fully_known_positive_shape(x_shape)
                    or not _is_fully_known_positive_shape(r1_shape)
                    or not _is_fully_known_positive_shape(t1_shape)
                ):
                    continue
                if len(x_shape) != 4 or len(r1_shape) != 5 or len(t1_shape) != 5:
                    continue

                n, h, w, c = [int(v) for v in x_shape]
                if int(c) <= 1:
                    continue

                # r1:[N,g,cpg,H,W], t1:[N,cpg,g,H,W]
                if (
                    int(r1_shape[0]) != int(n)
                    or int(r1_shape[3]) != int(h)
                    or int(r1_shape[4]) != int(w)
                    or int(t1_shape[0]) != int(n)
                    or int(t1_shape[3]) != int(h)
                    or int(t1_shape[4]) != int(w)
                    or int(t1_shape[1]) != int(r1_shape[2])
                    or int(t1_shape[2]) != int(r1_shape[1])
                ):
                    continue
                groups = int(r1_shape[1])
                cpg = int(r1_shape[2])
                if int(groups) <= 1 or int(cpg) <= 0 or int(groups * cpg) != int(c):
                    continue

                shuffle_indices = np.asarray(
                    [int((k % groups) * cpg + (k // groups)) for k in range(int(c))],
                    dtype=np.int32,
                )
                if np.array_equal(shuffle_indices, np.arange(int(c), dtype=np.int32)):
                    continue

                gather_idx_name = _unique_tensor_name(f"{x_nhwc_name}_shuffle_indices")
                model_ir.tensors[gather_idx_name] = TensorIR(
                    name=gather_idx_name,
                    dtype="INT32",
                    shape=[int(c)],
                    shape_signature=[int(c)],
                    data=shuffle_indices,
                    is_variable=False,
                )

                t2_op.op_type = "GATHER"
                t2_op.version = 1
                _set_operator_inputs(
                    model_ir=model_ir,
                    op=t2_op,
                    new_inputs=[gather_src_name, gather_idx_name],
                    graph_index=graph_index,
                )
                t2_op.options = {"axis": 3, "batchDims": 0}

                if unary_bridge_op is not None:
                    _set_operator_inputs(
                        model_ir=model_ir,
                        op=unary_bridge_op,
                        new_inputs=[x_nhwc_name],
                        graph_index=graph_index,
                    )
                    _permute_tensor_metadata_if_rank_matches(
                        model_ir.tensors.get(str(unary_bridge_op.outputs[0]), None),
                        _NCHW_TO_NHWC_PERM,
                    )

                if y_tensor is not None:
                    y_tensor.shape = [int(v) for v in list(x_shape)]
                    y_signature = (
                        [int(v) for v in list(x_tensor.shape_signature)]
                        if x_tensor.shape_signature is not None
                        else [int(v) for v in list(x_shape)]
                    )
                    y_tensor.shape_signature = [int(v) for v in list(y_signature)]

                remove_indices = {int(r1_idx), int(t1_idx), int(r2_idx)}
                t0_remaining_users = [
                    int(v)
                    for v in t0_users
                    if int(v) != int(t0_user_idx)
                ]
                if len(t0_remaining_users) == 0:
                    remove_indices.add(int(t0_idx))

                for remove_idx in sorted(list(remove_indices), reverse=True):
                    graph_index.remove_operator(int(remove_idx))

                optimized += 1
                changed = True
                break

            if changed:
                break

        if not changed:
            break

    _prune_unused_tensors(model_ir, layout_state=layout_state)
    if layout_state is not None:
        layout_state.sync_from_model_ir(model_ir)
    return {"optimized_shufflenet_reshape_transpose_shuffle_nhwc_chains": int(optimized)}


def run_nhwc_channel_shuffle_cleanup(
    model_ir: ModelIR,
    *,
    layout_state: LayoutState | None = None,
    diagnostics: List[Dict[str, Any]] | None = None,
    state_scope: ModelIRPassStateScope | None = None,
) -> Dict[str, int]:
    """Canonicalize strict ShuffleNet NHWC channel shuffle to Gather."""

    def _preflight(candidate_model: ModelIR) -> ModelIRPreflightResult:
        found_reshape = False
        found_transpose = False
        for visited, operator in enumerate(candidate_model.operators, start=1):
            operator_type = str(operator.op_type)
            found_reshape = found_reshape or operator_type == "RESHAPE"
            found_transpose = found_transpose or operator_type == "TRANSPOSE"
            if found_reshape and found_transpose:
                return ModelIRPreflightResult(True, visited)
        return ModelIRPreflightResult(False, len(candidate_model.operators))

    def _has_candidate(pass_state: ModelIRPassState) -> bool:
        candidate_model = pass_state.model_ir
        model_outputs = set(str(name) for name in candidate_model.outputs)
        for t0_op in candidate_model.operators:
            if (
                str(t0_op.op_type) != "TRANSPOSE"
                or len(t0_op.inputs) < 2
                or len(t0_op.outputs) != 1
                or _read_transpose_perm(candidate_model, t0_op)
                != _NHWC_TO_NCHW_PERM
            ):
                continue
            x_nhwc_name = str(t0_op.inputs[0])
            x_nchw_name = str(t0_op.outputs[0])
            if x_nchw_name in model_outputs:
                continue
            t0_users = pass_state.graph_index.consumer_indices(x_nchw_name)
            for t0_user_index in t0_users:
                t0_user_op = candidate_model.operators[int(t0_user_index)]
                if (
                    str(t0_user_op.op_type) == "RESHAPE"
                    and len(t0_user_op.inputs) >= 1
                    and len(t0_user_op.outputs) == 1
                    and str(t0_user_op.inputs[0]) == x_nchw_name
                ):
                    r1_op = t0_user_op
                elif (
                    str(t0_user_op.op_type) in _NHWC_CHANNEL_SHUFFLE_UNARY_OPS
                    and len(t0_user_op.inputs) == 1
                    and len(t0_user_op.outputs) == 1
                    and str(t0_user_op.inputs[0]) == x_nchw_name
                    and str(t0_user_op.outputs[0]) not in model_outputs
                ):
                    unary_output_name = str(t0_user_op.outputs[0])
                    unary_users = pass_state.graph_index.consumer_indices(
                        unary_output_name
                    )
                    if len(unary_users) != 1:
                        continue
                    r1_op = candidate_model.operators[int(unary_users[0])]
                    if (
                        str(r1_op.op_type) != "RESHAPE"
                        or len(r1_op.inputs) < 1
                        or len(r1_op.outputs) != 1
                        or str(r1_op.inputs[0]) != unary_output_name
                    ):
                        continue
                else:
                    continue
                r1_output_name = str(r1_op.outputs[0])
                if r1_output_name in model_outputs:
                    continue
                r1_users = pass_state.graph_index.consumer_indices(r1_output_name)
                if len(r1_users) != 1:
                    continue
                t1_op = candidate_model.operators[int(r1_users[0])]
                if (
                    str(t1_op.op_type) != "TRANSPOSE"
                    or len(t1_op.inputs) < 2
                    or len(t1_op.outputs) != 1
                    or str(t1_op.inputs[0]) != r1_output_name
                    or _read_transpose_perm(candidate_model, t1_op)
                    != _CHANNEL_SHUFFLE_SWAP_PERM
                ):
                    continue
                t1_output_name = str(t1_op.outputs[0])
                if t1_output_name in model_outputs:
                    continue
                t1_users = pass_state.graph_index.consumer_indices(t1_output_name)
                if len(t1_users) != 1:
                    continue
                r2_op = candidate_model.operators[int(t1_users[0])]
                if (
                    str(r2_op.op_type) != "RESHAPE"
                    or len(r2_op.inputs) < 1
                    or len(r2_op.outputs) != 1
                    or str(r2_op.inputs[0]) != t1_output_name
                ):
                    continue
                r2_output_name = str(r2_op.outputs[0])
                if r2_output_name in model_outputs:
                    continue
                r2_users = pass_state.graph_index.consumer_indices(r2_output_name)
                if len(r2_users) != 1:
                    continue
                t2_op = candidate_model.operators[int(r2_users[0])]
                if (
                    str(t2_op.op_type) != "TRANSPOSE"
                    or len(t2_op.inputs) < 2
                    or len(t2_op.outputs) != 1
                    or str(t2_op.inputs[0]) != r2_output_name
                    or _read_transpose_perm(candidate_model, t2_op)
                    != _NCHW_TO_NHWC_PERM
                ):
                    continue
                x_tensor = candidate_model.tensors.get(x_nhwc_name)
                r1_tensor = candidate_model.tensors.get(r1_output_name)
                t1_tensor = candidate_model.tensors.get(t1_output_name)
                if x_tensor is None or r1_tensor is None or t1_tensor is None:
                    continue
                x_shape = [int(value) for value in x_tensor.shape]
                r1_shape = [int(value) for value in r1_tensor.shape]
                t1_shape = [int(value) for value in t1_tensor.shape]
                if (
                    not _is_fully_known_positive_shape(x_shape)
                    or not _is_fully_known_positive_shape(r1_shape)
                    or not _is_fully_known_positive_shape(t1_shape)
                    or [len(x_shape), len(r1_shape), len(t1_shape)] != [4, 5, 5]
                ):
                    continue
                n, height, width, channels = x_shape
                groups = r1_shape[1]
                channels_per_group = r1_shape[2]
                if (
                    channels > 1
                    and groups > 1
                    and channels_per_group > 0
                    and groups * channels_per_group == channels
                    and r1_shape
                    == [n, groups, channels_per_group, height, width]
                    and t1_shape
                    == [n, channels_per_group, groups, height, width]
                ):
                    shuffle_indices = np.asarray(
                        [
                            (index % groups) * channels_per_group
                            + (index // groups)
                            for index in range(channels)
                        ],
                        dtype=np.int32,
                    )
                    if not np.array_equal(
                        shuffle_indices,
                        np.arange(channels, dtype=np.int32),
                    ):
                        return True
        return False

    def _run(pass_state: ModelIRPassState) -> Dict[str, int | bool]:
        stats = _optimize_shufflenet_reshape_transpose_shuffle_nhwc_chains(
            pass_state.model_ir,
            graph_index=pass_state.graph_index,
            layout_state=pass_state.layout_state,
        )
        return {
            **stats,
            "changed": bool(
                stats.get(
                    "optimized_shufflenet_reshape_transpose_shuffle_nhwc_chains",
                    0,
                )
            ),
        }

    details, _ = run_model_ir_pass_group(
        model_ir,
        specs=[
            PassSpec(
                pass_id="canonicalize.nhwc_channel_shuffle_gather",
                phase=PassPhase.CANONICALIZE,
                callback=_run,
                precondition=_has_candidate,
                priority=10,
                transactional=True,
            )
        ],
        layout_state=layout_state,
        default_details={
            "optimized_shufflenet_reshape_transpose_shuffle_nhwc_chains": 0,
        },
        diagnostics=diagnostics,
        state_scope=state_scope,
        preflight=_preflight,
    )
    return {str(key): int(value) for key, value in details.items()}



def _optimize_nchw_channel_shuffle_reshape_transpose_reshape_to_gather(
    model_ir: ModelIR,
    *,
    graph_index: ModelIRGraphIndex | None = None,
    layout_state: LayoutState | None = None,
) -> Dict[str, int]:
    """
    Collapse NCHW channel-shuffle blocks into a single GATHER(axis=1).

    Target:
      x_nchw
        -> RESHAPE([N,g,cpg,H,W])
        -> TRANSPOSE([0,2,1,3,4])
        -> RESHAPE([N,C,H,W]) -> y_nchw

    Rewrite:
      x_nchw -> GATHER(axis=1, shuffle_indices) -> y_nchw

    where C=g*cpg and shuffle_indices[k] = (k % g) * cpg + (k // g).
    """
    optimized = 0
    graph_index = graph_index or ModelIRGraphIndex(model_ir)
    perm_shuffle_swap = [0, 2, 1, 3, 4]

    def _unique_tensor_name(base: str) -> str:
        name = str(base)
        suffix = 1
        while name in model_ir.tensors:
            name = f"{base}_{suffix}"
            suffix += 1
        return name

    while True:
        changed = False
        consumers = graph_index.consumers

        for r1_idx, r1_op in enumerate(model_ir.operators):
            if str(r1_op.op_type) != "RESHAPE" or len(r1_op.inputs) < 1 or len(r1_op.outputs) != 1:
                continue
            x_nchw_name = str(r1_op.inputs[0])
            r1_out_name = str(r1_op.outputs[0])

            r1_users = [int(v) for v in consumers.get(r1_out_name, [])]
            if len(r1_users) != 1:
                continue
            t1_idx = int(r1_users[0])
            t1_op = model_ir.operators[int(t1_idx)]
            if (
                str(t1_op.op_type) != "TRANSPOSE"
                or len(t1_op.inputs) < 2
                or len(t1_op.outputs) != 1
                or str(t1_op.inputs[0]) != r1_out_name
                or _read_transpose_perm(model_ir, t1_op) != perm_shuffle_swap
            ):
                continue
            t1_out_name = str(t1_op.outputs[0])

            t1_users = [int(v) for v in consumers.get(t1_out_name, [])]
            if len(t1_users) != 1:
                continue
            r2_idx = int(t1_users[0])
            r2_op = model_ir.operators[int(r2_idx)]
            if (
                str(r2_op.op_type) != "RESHAPE"
                or len(r2_op.inputs) < 1
                or len(r2_op.outputs) != 1
                or str(r2_op.inputs[0]) != t1_out_name
            ):
                continue
            y_nchw_name = str(r2_op.outputs[0])

            x_tensor = model_ir.tensors.get(x_nchw_name, None)
            r1_tensor = model_ir.tensors.get(r1_out_name, None)
            t1_tensor = model_ir.tensors.get(t1_out_name, None)
            y_tensor = model_ir.tensors.get(y_nchw_name, None)
            if x_tensor is None or r1_tensor is None or t1_tensor is None or y_tensor is None:
                continue

            x_shape = [int(v) for v in list(x_tensor.shape)]
            r1_shape = [int(v) for v in list(r1_tensor.shape)]
            t1_shape = [int(v) for v in list(t1_tensor.shape)]
            y_shape = [int(v) for v in list(y_tensor.shape)]
            if (
                not _is_fully_known_positive_shape(x_shape)
                or not _is_fully_known_positive_shape(r1_shape)
                or not _is_fully_known_positive_shape(t1_shape)
                or not _is_fully_known_positive_shape(y_shape)
            ):
                continue
            if len(x_shape) != 4 or len(r1_shape) != 5 or len(t1_shape) != 5 or len(y_shape) != 4:
                continue

            n, c, h, w = [int(v) for v in x_shape]
            groups = int(r1_shape[1])
            cpg = int(r1_shape[2])
            if (
                int(groups) <= 1
                or int(cpg) <= 0
                or int(groups * cpg) != int(c)
                or int(r1_shape[0]) != int(n)
                or int(r1_shape[3]) != int(h)
                or int(r1_shape[4]) != int(w)
                or int(t1_shape[0]) != int(n)
                or int(t1_shape[1]) != int(cpg)
                or int(t1_shape[2]) != int(groups)
                or int(t1_shape[3]) != int(h)
                or int(t1_shape[4]) != int(w)
                or [int(v) for v in list(y_shape)] != [int(n), int(c), int(h), int(w)]
            ):
                continue

            shuffle_indices = np.asarray(
                [int((k % groups) * cpg + (k // groups)) for k in range(int(c))],
                dtype=np.int32,
            )
            if np.array_equal(shuffle_indices, np.arange(int(c), dtype=np.int32)):
                continue

            gather_idx_name = _unique_tensor_name(f"{x_nchw_name}_shuffle_indices_nchw")
            model_ir.tensors[gather_idx_name] = TensorIR(
                name=gather_idx_name,
                dtype="INT32",
                shape=[int(c)],
                shape_signature=[int(c)],
                data=np.asarray(shuffle_indices, dtype=np.int32),
                is_variable=False,
            )

            r2_op.op_type = "GATHER"
            r2_op.version = 1
            _set_operator_inputs(
                model_ir=model_ir,
                op=r2_op,
                new_inputs=[x_nchw_name, gather_idx_name],
                graph_index=graph_index,
            )
            r2_op.options = {"axis": 1, "batchDims": 0}

            for remove_idx in sorted([int(r1_idx), int(t1_idx)], reverse=True):
                graph_index.remove_operator(int(remove_idx))

            optimized += 1
            changed = True
            break

        if not changed:
            break

    if optimized > 0:
        _prune_unused_tensors(model_ir, layout_state=layout_state)
        if layout_state is not None:
            layout_state.sync_from_model_ir(model_ir)
    return {"optimized_nchw_channel_shuffle_reshape_transpose_reshape_to_gather": int(optimized)}


def _repair_nchw_channel_shuffle_concat_gathers(
    model_ir: ModelIR,
    *,
    graph_index: ModelIRGraphIndex | None = None,
    layout_state: LayoutState | None = None,
) -> Dict[str, int]:
    """Restore NCHW concat axis before an NCHW channel-shuffle GATHER.

    A later layout pass can remap CONCATENATION to axis=3 after the original
    reshape/transpose channel-shuffle has already been collapsed to
    GATHER(axis=1). The gather index count is the exact expected channel count,
    so it safely distinguishes this stale mixed-layout state.
    """

    repaired = 0
    graph_index = graph_index or ModelIRGraphIndex(model_ir)
    producers = graph_index.producers
    for gather_op in model_ir.operators:
        if (
            str(gather_op.op_type) != "GATHER"
            or len(gather_op.inputs) < 2
            or len(gather_op.outputs) != 1
            or int(gather_op.options.get("axis", -1)) != 1
            or int(gather_op.options.get("batchDims", 0)) != 0
        ):
            continue
        data_name = str(gather_op.inputs[0])
        indices_tensor = model_ir.tensors.get(str(gather_op.inputs[1]), None)
        concat_idx = producers.get(data_name, None)
        if concat_idx is None or indices_tensor is None or indices_tensor.data is None:
            continue
        concat_op = model_ir.operators[int(concat_idx)]
        if (
            str(concat_op.op_type) != "CONCATENATION"
            or len(concat_op.inputs) < 2
            or len(concat_op.outputs) != 1
            or int(concat_op.options.get("axis", 1)) == 1
        ):
            continue
        input_tensors = [model_ir.tensors.get(str(name), None) for name in concat_op.inputs]
        if any(tensor is None for tensor in input_tensors):
            continue
        input_shapes = [[int(v) for v in list(tensor.shape)] for tensor in input_tensors if tensor is not None]
        if not input_shapes or any(len(shape) != 4 for shape in input_shapes):
            continue
        reference = input_shapes[0]
        if any(
            any(int(shape[axis]) != int(reference[axis]) for axis in [0, 2, 3])
            for shape in input_shapes[1:]
        ):
            continue
        expected_channels = int(sum(int(shape[1]) for shape in input_shapes))
        gather_index_count = int(np.asarray(indices_tensor.data).size)
        if expected_channels <= 0 or gather_index_count != expected_channels:
            continue

        concat_op.options["axis"] = 1
        repaired_shape = [int(v) for v in reference]
        repaired_shape[1] = int(expected_channels)
        concat_tensor = model_ir.tensors.get(data_name, None)
        gather_tensor = model_ir.tensors.get(str(gather_op.outputs[0]), None)
        for tensor in [concat_tensor, gather_tensor]:
            if tensor is None:
                continue
            tensor.shape = [int(v) for v in repaired_shape]
            tensor.shape_signature = [int(v) for v in repaired_shape]
        repaired += 1
    if repaired > 0 and layout_state is not None:
        layout_state.sync_from_model_ir(model_ir)
    return {"repaired_nchw_channel_shuffle_concat_gathers": int(repaired)}


def run_stale_nchw_channel_shuffle_repair(
    model_ir: ModelIR,
    *,
    layout_state: LayoutState | None = None,
    diagnostics: List[Dict[str, Any]] | None = None,
) -> Dict[str, int]:
    """Repair stale NHWC Concat metadata before NCHW shuffle Gather."""

    def _preflight(candidate_model: ModelIR) -> ModelIRPreflightResult:
        found_concat = False
        found_gather = False
        for visited, operator in enumerate(candidate_model.operators, start=1):
            operator_type = str(operator.op_type)
            found_concat = found_concat or operator_type == "CONCATENATION"
            found_gather = found_gather or operator_type == "GATHER"
            if found_concat and found_gather:
                return ModelIRPreflightResult(True, visited)
        return ModelIRPreflightResult(False, len(candidate_model.operators))

    def _has_candidate(pass_state: ModelIRPassState) -> bool:
        candidate_model = pass_state.model_ir
        for gather_op in candidate_model.operators:
            if (
                str(gather_op.op_type) != "GATHER"
                or len(gather_op.inputs) < 2
                or len(gather_op.outputs) != 1
                or int(gather_op.options.get("axis", -1)) != 1
                or int(gather_op.options.get("batchDims", 0)) != 0
            ):
                continue
            concat_op = pass_state.graph_index.producer(str(gather_op.inputs[0]))
            indices_tensor = candidate_model.tensors.get(str(gather_op.inputs[1]))
            if (
                concat_op is None
                or indices_tensor is None
                or indices_tensor.data is None
                or str(concat_op.op_type) != "CONCATENATION"
                or len(concat_op.inputs) < 2
                or len(concat_op.outputs) != 1
                or int(concat_op.options.get("axis", 1)) == 1
            ):
                continue
            input_tensors = [
                candidate_model.tensors.get(str(name))
                for name in concat_op.inputs
            ]
            if any(tensor is None for tensor in input_tensors):
                continue
            input_shapes = [
                [int(value) for value in tensor.shape]
                for tensor in input_tensors
                if tensor is not None
            ]
            if not input_shapes or any(len(shape) != 4 for shape in input_shapes):
                continue
            reference = input_shapes[0]
            if any(
                any(shape[axis] != reference[axis] for axis in (0, 2, 3))
                for shape in input_shapes[1:]
            ):
                continue
            expected_channels = sum(shape[1] for shape in input_shapes)
            if (
                expected_channels > 0
                and int(np.asarray(indices_tensor.data).size) == expected_channels
            ):
                return True
        return False

    def _run(pass_state: ModelIRPassState) -> Dict[str, int | bool]:
        stats = _repair_nchw_channel_shuffle_concat_gathers(
            pass_state.model_ir,
            graph_index=pass_state.graph_index,
            layout_state=pass_state.layout_state,
        )
        return {
            **stats,
            "changed": bool(
                stats.get("repaired_nchw_channel_shuffle_concat_gathers", 0)
            ),
        }

    details, _ = run_model_ir_pass_group(
        model_ir,
        specs=[
            PassSpec(
                pass_id="layout.repair_nchw_channel_shuffle_concat",
                phase=PassPhase.LAYOUT_PLAN,
                callback=_run,
                precondition=_has_candidate,
                priority=10,
                transactional=True,
            )
        ],
        layout_state=layout_state,
        default_details={"repaired_nchw_channel_shuffle_concat_gathers": 0},
        diagnostics=diagnostics,
        preflight=_preflight,
    )
    return {str(key): int(value) for key, value in details.items()}



def run_nchw_channel_shuffle_cleanup(
    model_ir: ModelIR,
    *,
    layout_state: LayoutState | None = None,
    diagnostics: List[Dict[str, Any]] | None = None,
    state_scope: ModelIRPassStateScope | None = None,
) -> Dict[str, int]:
    """Canonicalize strict static NCHW channel shuffle to Gather."""

    def _preflight(candidate_model: ModelIR) -> ModelIRPreflightResult:
        found_reshape = False
        found_transpose = False
        for visited, operator in enumerate(candidate_model.operators, start=1):
            operator_type = str(operator.op_type)
            found_reshape = found_reshape or operator_type == "RESHAPE"
            found_transpose = found_transpose or operator_type == "TRANSPOSE"
            if found_reshape and found_transpose:
                return ModelIRPreflightResult(True, visited)
        return ModelIRPreflightResult(False, len(candidate_model.operators))

    def _has_candidate(pass_state: ModelIRPassState) -> bool:
        candidate_model = pass_state.model_ir
        for r1_op in candidate_model.operators:
            if (
                str(r1_op.op_type) != "RESHAPE"
                or len(r1_op.inputs) < 1
                or len(r1_op.outputs) != 1
            ):
                continue
            x_name = str(r1_op.inputs[0])
            r1_output_name = str(r1_op.outputs[0])
            r1_users = pass_state.graph_index.consumer_indices(r1_output_name)
            if len(r1_users) != 1:
                continue
            transpose_op = candidate_model.operators[int(r1_users[0])]
            if (
                str(transpose_op.op_type) != "TRANSPOSE"
                or len(transpose_op.inputs) < 2
                or len(transpose_op.outputs) != 1
                or str(transpose_op.inputs[0]) != r1_output_name
                or _read_transpose_perm(candidate_model, transpose_op)
                != [0, 2, 1, 3, 4]
            ):
                continue
            transpose_output_name = str(transpose_op.outputs[0])
            transpose_users = pass_state.graph_index.consumer_indices(
                transpose_output_name
            )
            if len(transpose_users) != 1:
                continue
            r2_op = candidate_model.operators[int(transpose_users[0])]
            if (
                str(r2_op.op_type) != "RESHAPE"
                or len(r2_op.inputs) < 1
                or len(r2_op.outputs) != 1
                or str(r2_op.inputs[0]) != transpose_output_name
            ):
                continue
            tensors = [
                candidate_model.tensors.get(name)
                for name in (
                    x_name,
                    r1_output_name,
                    transpose_output_name,
                    str(r2_op.outputs[0]),
                )
            ]
            if any(tensor is None for tensor in tensors):
                continue
            shapes = [
                [int(value) for value in tensor.shape]
                for tensor in tensors
                if tensor is not None
            ]
            if not all(_is_fully_known_positive_shape(shape) for shape in shapes):
                continue
            x_shape, r1_shape, transpose_shape, y_shape = shapes
            if [len(shape) for shape in shapes] != [4, 5, 5, 4]:
                continue
            n, channels, height, width = x_shape
            groups = int(r1_shape[1])
            channels_per_group = int(r1_shape[2])
            if (
                groups > 1
                and channels_per_group > 0
                and groups * channels_per_group == channels
                and r1_shape == [n, groups, channels_per_group, height, width]
                and transpose_shape
                == [n, channels_per_group, groups, height, width]
                and y_shape == [n, channels, height, width]
            ):
                shuffle_indices = np.asarray(
                    [
                        (index % groups) * channels_per_group
                        + (index // groups)
                        for index in range(channels)
                    ],
                    dtype=np.int32,
                )
                if not np.array_equal(
                    shuffle_indices,
                    np.arange(channels, dtype=np.int32),
                ):
                    return True
        return False

    def _run(pass_state: ModelIRPassState) -> Dict[str, int | bool]:
        stats = _optimize_nchw_channel_shuffle_reshape_transpose_reshape_to_gather(
            pass_state.model_ir,
            graph_index=pass_state.graph_index,
            layout_state=pass_state.layout_state,
        )
        return {
            **stats,
            "changed": bool(
                stats.get(
                    "optimized_nchw_channel_shuffle_reshape_transpose_reshape_to_gather",
                    0,
                )
            ),
        }

    details, _ = run_model_ir_pass_group(
        model_ir,
        specs=[
            PassSpec(
                pass_id="canonicalize.nchw_channel_shuffle_gather",
                phase=PassPhase.CANONICALIZE,
                callback=_run,
                precondition=_has_candidate,
                priority=10,
                transactional=True,
            )
        ],
        layout_state=layout_state,
        default_details={
            "optimized_nchw_channel_shuffle_reshape_transpose_reshape_to_gather": 0,
        },
        diagnostics=diagnostics,
        state_scope=state_scope,
        preflight=_preflight,
    )
    return {str(key): int(value) for key, value in details.items()}
