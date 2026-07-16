from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np

from onnx2tf.tflite_builder.core.graph import ModelIRGraphIndex
from onnx2tf.tflite_builder.core.layout import LayoutState
from onnx2tf.tflite_builder.core.model_ir_pass_state import (
    ModelIRPreflightResult,
    ModelIRPassState,
    ModelIRPassStateScope,
    run_model_ir_pass_group,
)
from onnx2tf.tflite_builder.core.model_ir_utils import (
    _clone_quantization,
    _permute_shape,
    _prune_unused_tensors,
    _read_const_ints_from_tensor,
    _set_operator_inputs,
    _set_operator_outputs,
)
from onnx2tf.tflite_builder.core.passes import PassPhase, PassSpec
from onnx2tf.tflite_builder.ir import ModelIR, OperatorIR, TensorIR

def _optimize_singleton_layout_reshape_maxpool_binary_cast_chains(
    model_ir: ModelIR,
    *,
    graph_index: ModelIRGraphIndex | None = None,
    layout_state: LayoutState | None = None,
) -> Dict[str, int]:
    """
    Remove redundant singleton layout RESHAPE bridges around consecutive MAX_POOL blocks.

    Target:
      a_nchw --RESHAPE--> a_nhwc --MAX_POOL--> b_nhwc --RESHAPE--> b_nchw
      BIN(a_nchw, b_nchw) --(optional CAST)--> c_nchw
      c_nchw --RESHAPE--> c_nhwc --MAX_POOL--> d_nhwc --RESHAPE--> d_nchw

    Rewrite:
      a_nchw --RESHAPE--> a_nhwc --MAX_POOL--> b_nhwc
      BIN(a_nhwc, b_nhwc) --(optional CAST)--> c_nhwc
      c_nhwc --MAX_POOL--> d_nhwc --RESHAPE--> d_nchw

    This removes the middle `RESHAPE->...->RESHAPE` layout roundtrip while
    preserving external NCHW contracts.
    """
    rewritten = 0
    graph_index = graph_index or ModelIRGraphIndex(model_ir)
    perm_nhwc_to_nchw = [0, 3, 1, 2]
    binary_ops = {
        "EQUAL",
        "NOT_EQUAL",
        "GREATER",
        "GREATER_EQUAL",
        "LESS",
        "LESS_EQUAL",
        "MAXIMUM",
        "MINIMUM",
        "ADD",
        "SUB",
        "MUL",
        "DIV",
    }

    def _read_reshape_target_shape(op: OperatorIR) -> Optional[List[int]]:
        if str(op.op_type) != "RESHAPE" or len(op.inputs) < 2:
            return None
        shape_vals = _read_const_ints_from_tensor(model_ir.tensors.get(str(op.inputs[1]), None))
        if shape_vals is None or len(shape_vals) != 4:
            return None
        return [int(v) for v in list(shape_vals)]

    def _set_tensor_shape_metadata(tensor_name: str, shape: List[int]) -> None:
        tensor = model_ir.tensors.get(str(tensor_name), None)
        if tensor is None:
            return
        normalized = [int(v) for v in list(shape)]
        tensor.shape = [int(v) for v in list(normalized)]
        tensor.shape_signature = [int(v) for v in list(normalized)]

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
        producers = graph_index.producers
        model_outputs = set(str(v) for v in model_ir.outputs)

        for post1_idx, post1_op in enumerate(model_ir.operators):
            if str(post1_op.op_type) != "RESHAPE" or len(post1_op.inputs) < 1 or len(post1_op.outputs) != 1:
                continue
            post1_in_name = str(post1_op.inputs[0])
            post1_out_name = str(post1_op.outputs[0])
            if post1_out_name in model_outputs:
                continue

            pool1_idx = producers.get(post1_in_name, None)
            if pool1_idx is None:
                continue
            pool1_op = model_ir.operators[int(pool1_idx)]
            if str(pool1_op.op_type) != "MAX_POOL_2D" or len(pool1_op.inputs) != 1 or len(pool1_op.outputs) != 1:
                continue
            if str(pool1_op.outputs[0]) != post1_in_name:
                continue

            pre1_out_name = str(pool1_op.inputs[0])
            pre1_idx = producers.get(pre1_out_name, None)
            if pre1_idx is None:
                continue
            pre1_op = model_ir.operators[int(pre1_idx)]
            if str(pre1_op.op_type) != "RESHAPE" or len(pre1_op.inputs) < 1 or len(pre1_op.outputs) != 1:
                continue
            if str(pre1_op.outputs[0]) != pre1_out_name:
                continue

            a_nchw_name = str(pre1_op.inputs[0])
            pre1_target_shape = _read_reshape_target_shape(pre1_op)
            post1_target_shape = _read_reshape_target_shape(post1_op)
            if pre1_target_shape is None or post1_target_shape is None:
                continue
            if pre1_target_shape[3] != 1 or post1_target_shape[1] != 1:
                continue
            expected_post1 = _permute_shape(pre1_target_shape, perm_nhwc_to_nchw)
            if expected_post1 is None or [int(v) for v in list(expected_post1)] != post1_target_shape:
                continue

            post1_users = [int(v) for v in consumers.get(post1_out_name, [])]
            if len(post1_users) != 1:
                continue
            bin_idx = int(post1_users[0])
            bin_op = model_ir.operators[int(bin_idx)]
            if str(bin_op.op_type) not in binary_ops or len(bin_op.inputs) != 2 or len(bin_op.outputs) != 1:
                continue
            if a_nchw_name not in set(str(v) for v in bin_op.inputs):
                continue
            if post1_out_name not in set(str(v) for v in bin_op.inputs):
                continue
            bin_out_name = str(bin_op.outputs[0])
            if bin_out_name in model_outputs:
                continue

            cast_path: Optional[Dict[str, Any]] = None
            non_cast_user_indices: List[int] = []
            invalid_fanout = False
            bin_users = [int(v) for v in consumers.get(str(bin_out_name), [])]
            for bin_user_idx in list(bin_users):
                bin_user_op = model_ir.operators[int(bin_user_idx)]
                if str(bin_user_op.op_type) != "CAST":
                    non_cast_user_indices.append(int(bin_user_idx))
                    continue
                if len(bin_user_op.inputs) != 1 or len(bin_user_op.outputs) != 1:
                    invalid_fanout = True
                    break
                if str(bin_user_op.inputs[0]) != str(bin_out_name):
                    invalid_fanout = True
                    break
                cast_out_name = str(bin_user_op.outputs[0])
                if cast_out_name in model_outputs:
                    invalid_fanout = True
                    break

                pre2_users = [int(v) for v in consumers.get(cast_out_name, [])]
                if len(pre2_users) != 1:
                    invalid_fanout = True
                    break
                pre2_idx = int(pre2_users[0])
                pre2_op = model_ir.operators[int(pre2_idx)]
                if str(pre2_op.op_type) != "RESHAPE" or len(pre2_op.inputs) < 1 or len(pre2_op.outputs) != 1:
                    invalid_fanout = True
                    break
                if str(pre2_op.inputs[0]) != cast_out_name:
                    invalid_fanout = True
                    break
                pre2_out_name = str(pre2_op.outputs[0])
                if pre2_out_name in model_outputs:
                    invalid_fanout = True
                    break

                pool2_users = [int(v) for v in consumers.get(pre2_out_name, [])]
                if len(pool2_users) != 1:
                    invalid_fanout = True
                    break
                pool2_idx = int(pool2_users[0])
                pool2_op = model_ir.operators[int(pool2_idx)]
                if (
                    str(pool2_op.op_type) != "MAX_POOL_2D"
                    or len(pool2_op.inputs) != 1
                    or len(pool2_op.outputs) != 1
                    or str(pool2_op.inputs[0]) != pre2_out_name
                ):
                    invalid_fanout = True
                    break
                pool2_out_name = str(pool2_op.outputs[0])
                if pool2_out_name in model_outputs:
                    invalid_fanout = True
                    break

                post2_users = [int(v) for v in consumers.get(pool2_out_name, [])]
                if len(post2_users) != 1:
                    invalid_fanout = True
                    break
                post2_idx = int(post2_users[0])
                post2_op = model_ir.operators[int(post2_idx)]
                if (
                    str(post2_op.op_type) != "RESHAPE"
                    or len(post2_op.inputs) < 1
                    or len(post2_op.outputs) != 1
                    or str(post2_op.inputs[0]) != pool2_out_name
                ):
                    invalid_fanout = True
                    break

                pre2_target_shape = _read_reshape_target_shape(pre2_op)
                post2_target_shape = _read_reshape_target_shape(post2_op)
                if pre2_target_shape is None or post2_target_shape is None:
                    invalid_fanout = True
                    break
                if pre2_target_shape[3] != 1 or post2_target_shape[1] != 1:
                    invalid_fanout = True
                    break
                expected_post2 = _permute_shape(pre2_target_shape, perm_nhwc_to_nchw)
                if expected_post2 is None or [int(v) for v in list(expected_post2)] != post2_target_shape:
                    invalid_fanout = True
                    break

                if cast_path is not None:
                    invalid_fanout = True
                    break
                cast_path = {
                    "cast_idx": int(bin_user_idx),
                    "cast_op": bin_user_op,
                    "cast_out_name": str(cast_out_name),
                    "pre2_idx": int(pre2_idx),
                    "pre2_op": pre2_op,
                    "pool2_op": pool2_op,
                    "pre2_target_shape": [int(v) for v in list(pre2_target_shape)],
                    "post2_target_shape": [int(v) for v in list(post2_target_shape)],
                }

            if invalid_fanout or cast_path is None:
                continue

            # Rewire binary inputs to NHWC path.
            bin_inputs = [str(v) for v in list(bin_op.inputs)]
            replaced = False
            for input_index, input_name in enumerate(bin_inputs):
                if input_name == post1_out_name:
                    bin_inputs[input_index] = str(post1_in_name)
                    replaced = True
                elif input_name == a_nchw_name:
                    bin_inputs[input_index] = str(pre1_out_name)
                    replaced = True
            if not replaced:
                continue
            _set_operator_inputs(
                model_ir=model_ir,
                op=bin_op,
                new_inputs=bin_inputs,
                graph_index=graph_index,
            )

            pre2_target_shape = [int(v) for v in list(cast_path["pre2_target_shape"])]
            _set_tensor_shape_metadata(str(bin_out_name), pre2_target_shape)

            if len(non_cast_user_indices) > 0:
                bin_out_tensor = model_ir.tensors.get(str(bin_out_name), None)
                adapter_shape_name = _unique_tensor_name(f"{bin_out_name}_nchw_shape")
                adapter_out_name = _unique_tensor_name(f"{bin_out_name}_nchw")
                model_ir.tensors[adapter_shape_name] = TensorIR(
                    name=adapter_shape_name,
                    dtype="INT32",
                    shape=[4],
                    shape_signature=[4],
                    data=np.asarray(post1_target_shape, dtype=np.int32),
                    is_variable=False,
                    quantization=None,
                )
                model_ir.tensors[adapter_out_name] = TensorIR(
                    name=adapter_out_name,
                    dtype=(
                        str(bin_out_tensor.dtype)
                        if bin_out_tensor is not None
                        else "BOOL"
                    ),
                    shape=[int(v) for v in list(post1_target_shape)],
                    shape_signature=[int(v) for v in list(post1_target_shape)],
                    data=None,
                    is_variable=False,
                    quantization=(
                        _clone_quantization(bin_out_tensor.quantization)
                        if bin_out_tensor is not None
                        else None
                    ),
                )

                for user_idx in list(non_cast_user_indices):
                    user_op = model_ir.operators[int(user_idx)]
                    user_inputs = [str(v) for v in list(user_op.inputs)]
                    if str(bin_out_name) not in set(user_inputs):
                        continue
                    _set_operator_inputs(
                        model_ir=model_ir,
                        op=user_op,
                        new_inputs=[
                            str(adapter_out_name) if str(v) == str(bin_out_name) else str(v)
                            for v in list(user_inputs)
                        ],
                        graph_index=graph_index,
                    )

                bin_current_idx = next(
                    (int(idx) for idx, op_ref in enumerate(model_ir.operators) if op_ref is bin_op),
                    None,
                )
                if bin_current_idx is None:
                    continue
                graph_index.insert_operator(
                    int(bin_current_idx) + 1,
                    OperatorIR(
                        op_type="RESHAPE",
                        inputs=[str(bin_out_name), str(adapter_shape_name)],
                        outputs=[str(adapter_out_name)],
                        options={"newShape": [int(v) for v in list(post1_target_shape)]},
                    ),
                )

            _set_operator_inputs(
                model_ir=model_ir,
                op=cast_path["pool2_op"],
                new_inputs=[str(cast_path["cast_out_name"])],
                graph_index=graph_index,
            )
            _set_tensor_shape_metadata(str(cast_path["cast_out_name"]), pre2_target_shape)

            remove_indices: List[int] = []
            for op_idx, op_ref in enumerate(model_ir.operators):
                if op_ref is post1_op or op_ref is cast_path["pre2_op"]:
                    remove_indices.append(int(op_idx))
            for remove_idx in sorted(list(set(remove_indices)), reverse=True):
                graph_index.remove_operator(int(remove_idx))

            rewritten += 1
            changed = True
            break

        if not changed:
            break

    if rewritten > 0:
        _prune_unused_tensors(model_ir, layout_state=layout_state)
        if layout_state is not None:
            layout_state.sync_from_model_ir(model_ir)
    return {"rewritten_singleton_layout_reshape_maxpool_binary_cast_chains": int(rewritten)}


def _optimize_singleton_nms_maxpool_nhwc_chains(
    model_ir: ModelIR,
    *,
    graph_index: ModelIRGraphIndex | None = None,
    layout_state: LayoutState | None = None,
) -> Dict[str, int]:
    """
    Collapse SuperPoint-like singleton-channel NMS MAX_POOL reshape ladders into NHWC.

    Target (representative):
      ... -> base_nchw
      base_nchw --RESHAPE--> base_nhwc --MAX_POOL--> ...
      (EQUAL/CAST/GREATER/SELECT/NOT/AND/OR with repeated
       RESHAPE->MAX_POOL->RESHAPE wrappers in NCHW)

    Rewrite:
      Keep the whole logical NMS subgraph in NHWC and remove intermediate
      singleton layout wrappers; if downstream expects NCHW, add one terminal
      NHWC->NCHW adapter.
    """
    rewritten = 0
    graph_index = graph_index or ModelIRGraphIndex(model_ir)

    def _unique_tensor_name(base: str) -> str:
        name = str(base)
        suffix = 1
        while name in model_ir.tensors:
            name = f"{base}_{suffix}"
            suffix += 1
        return name

    def _read_reshape_target_shape(op: OperatorIR) -> Optional[List[int]]:
        if str(op.op_type) != "RESHAPE" or len(op.inputs) < 2:
            return None
        vals = _read_const_ints_from_tensor(model_ir.tensors.get(str(op.inputs[1]), None))
        if vals is None or len(vals) != 4:
            return None
        return [int(v) for v in list(vals)]

    def _set_rank4_shape(tensor_name: str, shape: List[int]) -> None:
        tensor = model_ir.tensors.get(str(tensor_name), None)
        if tensor is None:
            return
        normalized = [int(v) for v in list(shape)]
        if len(normalized) != 4:
            return
        tensor.shape = [int(v) for v in list(normalized)]
        tensor.shape_signature = [int(v) for v in list(normalized)]

    def _find_single_consumer_op(
        tensor_name: str,
        consumers: Dict[str, List[int]],
        *,
        expected_op_type: str,
        operators: List[OperatorIR],
        required_input_index: Optional[int] = None,
    ) -> Optional[int]:
        user_indices = [int(v) for v in consumers.get(str(tensor_name), [])]
        matched_indices: List[int] = []
        for user_idx in user_indices:
            user_op = operators[int(user_idx)]
            if str(user_op.op_type) != str(expected_op_type):
                continue
            if required_input_index is not None:
                if (
                    int(required_input_index) < 0
                    or len(user_op.inputs) <= int(required_input_index)
                    or str(user_op.inputs[int(required_input_index)]) != str(tensor_name)
                ):
                    continue
            matched_indices.append(int(user_idx))
        if len(matched_indices) != 1:
            return None
        return int(matched_indices[0])

    while True:
        changed = False
        consumers = graph_index.consumers
        producers = graph_index.producers
        model_outputs = set(str(v) for v in model_ir.outputs)

        for post1_idx, post1_op in enumerate(model_ir.operators):
            # post1: pool1b_nhwc -> pool1b_nchw
            if str(post1_op.op_type) != "RESHAPE" or len(post1_op.inputs) < 2 or len(post1_op.outputs) != 1:
                continue
            post1_in_name = str(post1_op.inputs[0])
            post1_out_name = str(post1_op.outputs[0])
            if post1_out_name in model_outputs:
                continue
            post1_target_shape = _read_reshape_target_shape(post1_op)
            if post1_target_shape is None or int(post1_target_shape[1]) != 1:
                continue

            # pool1b and cast0 path
            pool1b_idx = producers.get(post1_in_name, None)
            if pool1b_idx is None:
                continue
            pool1b_op = model_ir.operators[int(pool1b_idx)]
            if str(pool1b_op.op_type) != "MAX_POOL_2D" or len(pool1b_op.inputs) != 1 or len(pool1b_op.outputs) != 1:
                continue
            if str(pool1b_op.outputs[0]) != post1_in_name:
                continue
            cast0_out_name = str(pool1b_op.inputs[0])
            cast0_idx = producers.get(cast0_out_name, None)
            if cast0_idx is None:
                continue
            cast0_op = model_ir.operators[int(cast0_idx)]
            if str(cast0_op.op_type) != "CAST" or len(cast0_op.inputs) != 1 or len(cast0_op.outputs) != 1:
                continue
            if str(cast0_op.outputs[0]) != cast0_out_name:
                continue
            eq0_nhwc_name = str(cast0_op.inputs[0])
            if eq0_nhwc_name in model_outputs:
                continue

            # eq0 path: eq0_nhwc = EQUAL(base_nhwc, pool1_nhwc)
            eq0_idx = producers.get(eq0_nhwc_name, None)
            if eq0_idx is None:
                continue
            eq0_op = model_ir.operators[int(eq0_idx)]
            if str(eq0_op.op_type) != "EQUAL" or len(eq0_op.inputs) != 2 or len(eq0_op.outputs) != 1:
                continue
            if str(eq0_op.outputs[0]) != eq0_nhwc_name:
                continue
            eq0_in0 = str(eq0_op.inputs[0])
            eq0_in1 = str(eq0_op.inputs[1])

            pool1_idx: Optional[int] = None
            pool1_out_name: Optional[str] = None
            base_nhwc_name: Optional[str] = None
            for candidate in [eq0_in0, eq0_in1]:
                cand_prod_idx = producers.get(str(candidate), None)
                if cand_prod_idx is None:
                    continue
                cand_op = model_ir.operators[int(cand_prod_idx)]
                if str(cand_op.op_type) == "MAX_POOL_2D" and len(cand_op.inputs) == 1 and len(cand_op.outputs) == 1:
                    if str(cand_op.outputs[0]) == str(candidate):
                        pool1_idx = int(cand_prod_idx)
                        pool1_out_name = str(candidate)
                        other = eq0_in1 if str(candidate) == eq0_in0 else eq0_in0
                        base_nhwc_name = str(other)
                        break
            if pool1_idx is None or pool1_out_name is None or base_nhwc_name is None:
                continue

            pool1_op = model_ir.operators[int(pool1_idx)]
            if str(pool1_op.inputs[0]) != str(base_nhwc_name):
                continue
            base_nhwc_shape = [int(v) for v in _read_reshape_target_shape(model_ir.operators[int(producers.get(base_nhwc_name, -1))]) or []]
            if len(base_nhwc_shape) != 4:
                base_nhwc_tensor = model_ir.tensors.get(base_nhwc_name, None)
                if base_nhwc_tensor is None or base_nhwc_tensor.shape is None or len(list(base_nhwc_tensor.shape)) != 4:
                    continue
                base_nhwc_shape = [int(v) for v in list(base_nhwc_tensor.shape)]
            if int(base_nhwc_shape[3]) != 1:
                continue

            pre_base_idx = producers.get(base_nhwc_name, None)
            if pre_base_idx is None:
                continue
            pre_base_op = model_ir.operators[int(pre_base_idx)]
            if str(pre_base_op.op_type) != "RESHAPE" or len(pre_base_op.inputs) < 2 or len(pre_base_op.outputs) != 1:
                continue
            if str(pre_base_op.outputs[0]) != str(base_nhwc_name):
                continue
            base_nchw_name = str(pre_base_op.inputs[0])
            if base_nchw_name in model_outputs:
                continue
            base_nchw_shape = _read_reshape_target_shape(post1_op)
            if base_nchw_shape is None:
                base_nchw_tensor = model_ir.tensors.get(base_nchw_name, None)
                if base_nchw_tensor is None or base_nchw_tensor.shape is None or len(list(base_nchw_tensor.shape)) != 4:
                    continue
                base_nchw_shape = [int(v) for v in list(base_nchw_tensor.shape)]
            if int(base_nchw_shape[1]) != 1:
                continue

            # greater0/select0/not0
            greater0_idx = _find_single_consumer_op(
                post1_out_name,
                consumers,
                expected_op_type="GREATER",
                operators=model_ir.operators,
            )
            if greater0_idx is None:
                continue
            greater0_op = model_ir.operators[int(greater0_idx)]
            if len(greater0_op.inputs) != 2 or len(greater0_op.outputs) != 1:
                continue
            cond0_name = str(greater0_op.outputs[0])
            cond0_users = [int(v) for v in consumers.get(cond0_name, [])]
            if len(cond0_users) != 2:
                continue
            select0_idx: Optional[int] = None
            not0_idx: Optional[int] = None
            for user_idx in cond0_users:
                user_op = model_ir.operators[int(user_idx)]
                if str(user_op.op_type) == "SELECT" and len(user_op.inputs) == 3 and str(user_op.inputs[0]) == cond0_name:
                    select0_idx = int(user_idx)
                elif str(user_op.op_type) == "LOGICAL_NOT" and len(user_op.inputs) == 1 and str(user_op.inputs[0]) == cond0_name:
                    not0_idx = int(user_idx)
            if select0_idx is None or not0_idx is None:
                continue
            select0_op = model_ir.operators[int(select0_idx)]
            not0_op = model_ir.operators[int(not0_idx)]
            select0_out_name = str(select0_op.outputs[0])
            not0_out_name = str(not0_op.outputs[0])
            zero_nchw_name = str(select0_op.inputs[1])
            base_nchw_from_select0 = str(select0_op.inputs[2])
            if base_nchw_from_select0 != base_nchw_name:
                continue

            # where0 -> reshape40 -> pool2 -> reshape42 -> equal1 -> and1
            reshape40_idx = _find_single_consumer_op(
                select0_out_name,
                consumers,
                expected_op_type="RESHAPE",
                operators=model_ir.operators,
                required_input_index=0,
            )
            if reshape40_idx is None:
                continue
            reshape40_op = model_ir.operators[int(reshape40_idx)]
            reshape40_out_name = str(reshape40_op.outputs[0])
            pool2_idx = _find_single_consumer_op(
                reshape40_out_name,
                consumers,
                expected_op_type="MAX_POOL_2D",
                operators=model_ir.operators,
            )
            if pool2_idx is None:
                continue
            pool2_op = model_ir.operators[int(pool2_idx)]
            if len(pool2_op.inputs) != 1 or len(pool2_op.outputs) != 1:
                continue
            pool2_out_name = str(pool2_op.outputs[0])
            reshape42_idx = _find_single_consumer_op(
                pool2_out_name,
                consumers,
                expected_op_type="RESHAPE",
                operators=model_ir.operators,
            )
            if reshape42_idx is None:
                continue
            reshape42_op = model_ir.operators[int(reshape42_idx)]
            reshape42_out_name = str(reshape42_op.outputs[0])
            equal1_idx = _find_single_consumer_op(
                reshape42_out_name,
                consumers,
                expected_op_type="EQUAL",
                operators=model_ir.operators,
            )
            if equal1_idx is None:
                continue
            equal1_op = model_ir.operators[int(equal1_idx)]
            if len(equal1_op.inputs) != 2 or len(equal1_op.outputs) != 1:
                continue
            if str(equal1_op.inputs[0]) != select0_out_name and str(equal1_op.inputs[1]) != select0_out_name:
                continue
            if str(equal1_op.inputs[0]) != reshape42_out_name and str(equal1_op.inputs[1]) != reshape42_out_name:
                continue
            equal1_out_name = str(equal1_op.outputs[0])
            and1_idx = _find_single_consumer_op(
                equal1_out_name,
                consumers,
                expected_op_type="LOGICAL_AND",
                operators=model_ir.operators,
            )
            if and1_idx is None:
                continue
            and1_op = model_ir.operators[int(and1_idx)]
            if len(and1_op.inputs) != 2 or len(and1_op.outputs) != 1:
                continue
            if not0_out_name not in set(str(v) for v in and1_op.inputs):
                continue
            and1_out_name = str(and1_op.outputs[0])

            # or0 with eq0_nchw adapter
            or0_idx = _find_single_consumer_op(
                and1_out_name,
                consumers,
                expected_op_type="LOGICAL_OR",
                operators=model_ir.operators,
            )
            if or0_idx is None:
                continue
            or0_op = model_ir.operators[int(or0_idx)]
            if len(or0_op.inputs) != 2 or len(or0_op.outputs) != 1:
                continue
            eq0_nchw_name = (
                str(or0_op.inputs[0])
                if str(or0_op.inputs[1]) == and1_out_name
                else str(or0_op.inputs[1])
            )
            eq0_nchw_adapter_idx = producers.get(eq0_nchw_name, None)
            if eq0_nchw_adapter_idx is None:
                continue
            eq0_nchw_adapter_op = model_ir.operators[int(eq0_nchw_adapter_idx)]
            if (
                str(eq0_nchw_adapter_op.op_type) != "RESHAPE"
                or len(eq0_nchw_adapter_op.inputs) < 1
                or len(eq0_nchw_adapter_op.outputs) != 1
                or str(eq0_nchw_adapter_op.outputs[0]) != eq0_nchw_name
                or str(eq0_nchw_adapter_op.inputs[0]) != eq0_nhwc_name
            ):
                continue
            or0_out_name = str(or0_op.outputs[0])

            # cast1 -> reshape47 -> pool3 -> reshape49 -> greater1 -> select1/not1
            cast1_idx = _find_single_consumer_op(
                or0_out_name,
                consumers,
                expected_op_type="CAST",
                operators=model_ir.operators,
                required_input_index=0,
            )
            if cast1_idx is None:
                continue
            cast1_op = model_ir.operators[int(cast1_idx)]
            if len(cast1_op.inputs) != 1 or len(cast1_op.outputs) != 1:
                continue
            cast1_out_name = str(cast1_op.outputs[0])
            reshape47_idx = _find_single_consumer_op(
                cast1_out_name,
                consumers,
                expected_op_type="RESHAPE",
                operators=model_ir.operators,
            )
            if reshape47_idx is None:
                continue
            reshape47_op = model_ir.operators[int(reshape47_idx)]
            reshape47_out_name = str(reshape47_op.outputs[0])
            pool3_idx = _find_single_consumer_op(
                reshape47_out_name,
                consumers,
                expected_op_type="MAX_POOL_2D",
                operators=model_ir.operators,
            )
            if pool3_idx is None:
                continue
            pool3_op = model_ir.operators[int(pool3_idx)]
            if len(pool3_op.inputs) != 1 or len(pool3_op.outputs) != 1:
                continue
            pool3_out_name = str(pool3_op.outputs[0])
            reshape49_idx = _find_single_consumer_op(
                pool3_out_name,
                consumers,
                expected_op_type="RESHAPE",
                operators=model_ir.operators,
            )
            if reshape49_idx is None:
                continue
            reshape49_op = model_ir.operators[int(reshape49_idx)]
            reshape49_out_name = str(reshape49_op.outputs[0])
            greater1_idx = _find_single_consumer_op(
                reshape49_out_name,
                consumers,
                expected_op_type="GREATER",
                operators=model_ir.operators,
            )
            if greater1_idx is None:
                continue
            greater1_op = model_ir.operators[int(greater1_idx)]
            if len(greater1_op.inputs) != 2 or len(greater1_op.outputs) != 1:
                continue
            cond1_name = str(greater1_op.outputs[0])
            cond1_users = [int(v) for v in consumers.get(cond1_name, [])]
            if len(cond1_users) != 2:
                continue
            select1_idx: Optional[int] = None
            not1_idx: Optional[int] = None
            for user_idx in cond1_users:
                user_op = model_ir.operators[int(user_idx)]
                if str(user_op.op_type) == "SELECT" and len(user_op.inputs) == 3 and str(user_op.inputs[0]) == cond1_name:
                    select1_idx = int(user_idx)
                elif str(user_op.op_type) == "LOGICAL_NOT" and len(user_op.inputs) == 1 and str(user_op.inputs[0]) == cond1_name:
                    not1_idx = int(user_idx)
            if select1_idx is None or not1_idx is None:
                continue
            select1_op = model_ir.operators[int(select1_idx)]
            not1_op = model_ir.operators[int(not1_idx)]
            select1_out_name = str(select1_op.outputs[0])
            not1_out_name = str(not1_op.outputs[0])
            if str(select1_op.inputs[1]) != zero_nchw_name or str(select1_op.inputs[2]) != base_nchw_name:
                continue

            # where1 -> reshape53 -> pool4 -> reshape55 -> equal2 -> and2
            reshape53_idx = _find_single_consumer_op(
                select1_out_name,
                consumers,
                expected_op_type="RESHAPE",
                operators=model_ir.operators,
                required_input_index=0,
            )
            if reshape53_idx is None:
                continue
            reshape53_op = model_ir.operators[int(reshape53_idx)]
            reshape53_out_name = str(reshape53_op.outputs[0])
            pool4_idx = _find_single_consumer_op(
                reshape53_out_name,
                consumers,
                expected_op_type="MAX_POOL_2D",
                operators=model_ir.operators,
            )
            if pool4_idx is None:
                continue
            pool4_op = model_ir.operators[int(pool4_idx)]
            if len(pool4_op.inputs) != 1 or len(pool4_op.outputs) != 1:
                continue
            pool4_out_name = str(pool4_op.outputs[0])
            reshape55_idx = _find_single_consumer_op(
                pool4_out_name,
                consumers,
                expected_op_type="RESHAPE",
                operators=model_ir.operators,
            )
            if reshape55_idx is None:
                continue
            reshape55_op = model_ir.operators[int(reshape55_idx)]
            reshape55_out_name = str(reshape55_op.outputs[0])
            equal2_idx = _find_single_consumer_op(
                reshape55_out_name,
                consumers,
                expected_op_type="EQUAL",
                operators=model_ir.operators,
            )
            if equal2_idx is None:
                continue
            equal2_op = model_ir.operators[int(equal2_idx)]
            if len(equal2_op.inputs) != 2 or len(equal2_op.outputs) != 1:
                continue
            if str(equal2_op.inputs[0]) != select1_out_name and str(equal2_op.inputs[1]) != select1_out_name:
                continue
            if str(equal2_op.inputs[0]) != reshape55_out_name and str(equal2_op.inputs[1]) != reshape55_out_name:
                continue
            equal2_out_name = str(equal2_op.outputs[0])
            and2_idx = _find_single_consumer_op(
                equal2_out_name,
                consumers,
                expected_op_type="LOGICAL_AND",
                operators=model_ir.operators,
            )
            if and2_idx is None:
                continue
            and2_op = model_ir.operators[int(and2_idx)]
            if len(and2_op.inputs) != 2 or len(and2_op.outputs) != 1:
                continue
            if not1_out_name not in set(str(v) for v in and2_op.inputs):
                continue
            and2_out_name = str(and2_op.outputs[0])

            # or1 -> select2
            or1_idx = _find_single_consumer_op(
                and2_out_name,
                consumers,
                expected_op_type="LOGICAL_OR",
                operators=model_ir.operators,
            )
            if or1_idx is None:
                continue
            or1_op = model_ir.operators[int(or1_idx)]
            if len(or1_op.inputs) != 2 or len(or1_op.outputs) != 1:
                continue
            if or0_out_name not in set(str(v) for v in or1_op.inputs):
                continue
            or1_out_name = str(or1_op.outputs[0])
            select2_idx = _find_single_consumer_op(
                or1_out_name,
                consumers,
                expected_op_type="SELECT",
                operators=model_ir.operators,
            )
            if select2_idx is None:
                continue
            select2_op = model_ir.operators[int(select2_idx)]
            if len(select2_op.inputs) != 3 or len(select2_op.outputs) != 1:
                continue
            if str(select2_op.inputs[0]) != or1_out_name:
                continue
            if set([str(select2_op.inputs[1]), str(select2_op.inputs[2])]) != set([base_nchw_name, zero_nchw_name]):
                continue
            select2_out_nchw_name = str(select2_op.outputs[0])
            if select2_out_nchw_name in model_outputs:
                continue

            # Create zero_nhwc adapter from zero_nchw.
            zero_nhwc_shape = [int(v) for v in list(base_nhwc_shape)]
            zero_nhwc_shape_name = _unique_tensor_name(f"{zero_nchw_name}_nhwc_shape")
            zero_nhwc_name = _unique_tensor_name(f"{zero_nchw_name}_nhwc")
            model_ir.tensors[zero_nhwc_shape_name] = TensorIR(
                name=zero_nhwc_shape_name,
                dtype="INT32",
                shape=[4],
                shape_signature=[4],
                data=np.asarray(zero_nhwc_shape, dtype=np.int32),
                is_variable=False,
                quantization=None,
            )
            zero_nchw_tensor = model_ir.tensors.get(zero_nchw_name, None)
            model_ir.tensors[zero_nhwc_name] = TensorIR(
                name=zero_nhwc_name,
                dtype=(
                    str(zero_nchw_tensor.dtype)
                    if zero_nchw_tensor is not None
                    else "FLOAT32"
                ),
                shape=[int(v) for v in list(zero_nhwc_shape)],
                shape_signature=[int(v) for v in list(zero_nhwc_shape)],
                data=None,
                is_variable=False,
                quantization=(
                    _clone_quantization(zero_nchw_tensor.quantization)
                    if zero_nchw_tensor is not None
                    else None
                ),
            )

            select0_current_idx = next((int(i) for i, op in enumerate(model_ir.operators) if op is select0_op), None)
            if select0_current_idx is None:
                continue
            graph_index.insert_operator(
                int(select0_current_idx),
                OperatorIR(
                    op_type="RESHAPE",
                    inputs=[str(zero_nchw_name), str(zero_nhwc_shape_name)],
                    outputs=[str(zero_nhwc_name)],
                    options={"newShape": [int(v) for v in list(zero_nhwc_shape)]},
                ),
            )

            # Rewire to NHWC.
            greater0_inputs = [str(v) for v in list(greater0_op.inputs)]
            greater0_inputs = [
                str(post1_in_name) if str(v) == str(post1_out_name) else str(v)
                for v in list(greater0_inputs)
            ]
            _set_operator_inputs(
                model_ir=model_ir,
                op=greater0_op,
                new_inputs=greater0_inputs,
                graph_index=graph_index,
            )
            _set_operator_inputs(
                model_ir=model_ir,
                op=select0_op,
                new_inputs=[str(cond0_name), str(zero_nhwc_name), str(base_nhwc_name)],
                graph_index=graph_index,
            )

            _set_operator_inputs(
                model_ir=model_ir,
                op=pool2_op,
                new_inputs=[str(select0_out_name)],
                graph_index=graph_index,
            )
            _set_operator_inputs(
                model_ir=model_ir,
                op=equal1_op,
                new_inputs=[str(select0_out_name), str(pool2_out_name)],
                graph_index=graph_index,
            )

            _set_operator_inputs(
                model_ir=model_ir,
                op=or0_op,
                new_inputs=[
                    str(eq0_nhwc_name) if str(v) == str(eq0_nchw_name) else str(v)
                    for v in list(or0_op.inputs)
                ],
                graph_index=graph_index,
            )

            _set_operator_inputs(
                model_ir=model_ir,
                op=pool3_op,
                new_inputs=[str(cast1_out_name)],
                graph_index=graph_index,
            )
            greater1_inputs = [str(v) for v in list(greater1_op.inputs)]
            greater1_inputs = [
                str(pool3_out_name) if str(v) == str(reshape49_out_name) else str(v)
                for v in list(greater1_inputs)
            ]
            _set_operator_inputs(
                model_ir=model_ir,
                op=greater1_op,
                new_inputs=greater1_inputs,
                graph_index=graph_index,
            )
            _set_operator_inputs(
                model_ir=model_ir,
                op=select1_op,
                new_inputs=[str(cond1_name), str(zero_nhwc_name), str(base_nhwc_name)],
                graph_index=graph_index,
            )

            _set_operator_inputs(
                model_ir=model_ir,
                op=pool4_op,
                new_inputs=[str(select1_out_name)],
                graph_index=graph_index,
            )
            _set_operator_inputs(
                model_ir=model_ir,
                op=equal2_op,
                new_inputs=[str(select1_out_name), str(pool4_out_name)],
                graph_index=graph_index,
            )
            _set_operator_inputs(
                model_ir=model_ir,
                op=select2_op,
                new_inputs=[str(or1_out_name), str(base_nhwc_name), str(zero_nhwc_name)],
                graph_index=graph_index,
            )

            # Convert NHWC output of final SELECT back to NCHW for downstream.
            select2_out_nhwc_name = _unique_tensor_name(f"{select2_out_nchw_name}_nhwc")
            model_ir.tensors[select2_out_nhwc_name] = TensorIR(
                name=select2_out_nhwc_name,
                dtype=str(model_ir.tensors[select2_out_nchw_name].dtype)
                if select2_out_nchw_name in model_ir.tensors
                else "BOOL",
                shape=[int(v) for v in list(base_nhwc_shape)],
                shape_signature=[int(v) for v in list(base_nhwc_shape)],
                data=None,
                is_variable=False,
                quantization=(
                    _clone_quantization(model_ir.tensors[select2_out_nchw_name].quantization)
                    if select2_out_nchw_name in model_ir.tensors
                    else None
                ),
            )
            _set_operator_outputs(
                model_ir=model_ir,
                op=select2_op,
                new_outputs=[str(select2_out_nhwc_name)],
                graph_index=graph_index,
            )

            final_nchw_shape_name = _unique_tensor_name(f"{select2_out_nchw_name}_shape_nchw")
            model_ir.tensors[final_nchw_shape_name] = TensorIR(
                name=final_nchw_shape_name,
                dtype="INT32",
                shape=[4],
                shape_signature=[4],
                data=np.asarray(base_nchw_shape, dtype=np.int32),
                is_variable=False,
                quantization=None,
            )
            select2_current_idx = next((int(i) for i, op in enumerate(model_ir.operators) if op is select2_op), None)
            if select2_current_idx is None:
                continue
            graph_index.insert_operator(
                int(select2_current_idx) + 1,
                OperatorIR(
                    op_type="RESHAPE",
                    inputs=[str(select2_out_nhwc_name), str(final_nchw_shape_name)],
                    outputs=[str(select2_out_nchw_name)],
                    options={"newShape": [int(v) for v in list(base_nchw_shape)]},
                ),
            )

            # Update tensor metadata for NHWC-propagated logical tensors.
            nhwc_tensors = [
                str(cond0_name),
                str(select0_out_name),
                str(not0_out_name),
                str(equal1_out_name),
                str(and1_out_name),
                str(or0_out_name),
                str(cast1_out_name),
                str(cond1_name),
                str(select1_out_name),
                str(not1_out_name),
                str(equal2_out_name),
                str(and2_out_name),
                str(or1_out_name),
                str(select2_out_nhwc_name),
            ]
            for tensor_name in nhwc_tensors:
                _set_rank4_shape(tensor_name, [int(v) for v in list(base_nhwc_shape)])

            # Remove redundant middle NCHW adapters.
            remove_refs: List[OperatorIR] = [
                eq0_nchw_adapter_op,
                post1_op,
                reshape40_op,
                reshape42_op,
                reshape47_op,
                reshape49_op,
                reshape53_op,
                reshape55_op,
            ]
            remove_indices: List[int] = []
            for op_idx, op_ref in enumerate(model_ir.operators):
                if any(op_ref is target for target in remove_refs):
                    remove_indices.append(int(op_idx))
            for remove_idx in sorted(list(set(remove_indices)), reverse=True):
                graph_index.remove_operator(int(remove_idx))

            rewritten += 1
            changed = True
            break

        if not changed:
            break

    if rewritten > 0:
        _prune_unused_tensors(model_ir, layout_state=layout_state)
        if layout_state is not None:
            layout_state.sync_from_model_ir(model_ir)
    return {"rewritten_singleton_nms_maxpool_nhwc_chains": int(rewritten)}


def run_singleton_maxpool_layout_cleanup(
    model_ir: ModelIR,
    *,
    include_binary_cast: bool = True,
    include_nms: bool = True,
    layout_state: LayoutState | None = None,
    diagnostics: List[Dict[str, Any]] | None = None,
    state_scope: ModelIRPassStateScope | None = None,
) -> Dict[str, int]:
    """Run adjacent singleton MaxPool layout rewrites in legacy order."""

    binary_ops = {
        "EQUAL",
        "NOT_EQUAL",
        "GREATER",
        "GREATER_EQUAL",
        "LESS",
        "LESS_EQUAL",
        "MAXIMUM",
        "MINIMUM",
        "ADD",
        "SUB",
        "MUL",
        "DIV",
    }

    def _preflight(candidate_model: ModelIR) -> ModelIRPreflightResult:
        required = {"RESHAPE", "MAX_POOL_2D"}
        for visited, operator in enumerate(candidate_model.operators, start=1):
            required.discard(str(operator.op_type))
            if len(required) == 0:
                return ModelIRPreflightResult(True, visited)
        return ModelIRPreflightResult(False, len(candidate_model.operators))

    def _reshape_shape(candidate_model: ModelIR, op: OperatorIR) -> List[int] | None:
        if str(op.op_type) != "RESHAPE" or len(op.inputs) < 2:
            return None
        values = _read_const_ints_from_tensor(
            candidate_model.tensors.get(str(op.inputs[1]))
        )
        if values is None or len(values) != 4:
            return None
        return [int(value) for value in values]

    def _single_user(
        pass_state: ModelIRPassState,
        tensor_name: str,
        op_type: str,
    ) -> tuple[int, OperatorIR] | None:
        users = pass_state.graph_index.consumer_indices(str(tensor_name))
        if len(users) != 1:
            return None
        index = int(users[0])
        op = pass_state.model_ir.operators[index]
        if str(op.op_type) != str(op_type):
            return None
        return index, op

    def _has_binary_cast_candidate(pass_state: ModelIRPassState) -> bool:
        candidate_model = pass_state.model_ir
        model_outputs = set(str(value) for value in candidate_model.outputs)
        for post1_op in candidate_model.operators:
            post1_shape = _reshape_shape(candidate_model, post1_op)
            if (
                post1_shape is None
                or len(post1_op.outputs) != 1
                or int(post1_shape[1]) != 1
                or str(post1_op.outputs[0]) in model_outputs
            ):
                continue
            pool1 = pass_state.graph_index.producer(str(post1_op.inputs[0]))
            if (
                pool1 is None
                or str(pool1.op_type) != "MAX_POOL_2D"
                or len(pool1.inputs) != 1
                or len(pool1.outputs) != 1
            ):
                continue
            pre1 = pass_state.graph_index.producer(str(pool1.inputs[0]))
            pre1_shape = (
                _reshape_shape(candidate_model, pre1) if pre1 is not None else None
            )
            if pre1 is None or pre1_shape is None or int(pre1_shape[3]) != 1:
                continue
            binary = _single_user(
                pass_state,
                str(post1_op.outputs[0]),
                next(
                    (
                        str(candidate_model.operators[index].op_type)
                        for index in pass_state.graph_index.consumer_indices(
                            str(post1_op.outputs[0])
                        )
                        if str(candidate_model.operators[index].op_type) in binary_ops
                    ),
                    "",
                ),
            )
            if binary is None or str(pre1.inputs[0]) not in [str(v) for v in binary[1].inputs]:
                continue
            cast = _single_user(pass_state, str(binary[1].outputs[0]), "CAST")
            if cast is None or len(cast[1].outputs) != 1:
                continue
            pre2 = _single_user(pass_state, str(cast[1].outputs[0]), "RESHAPE")
            if pre2 is None or _reshape_shape(candidate_model, pre2[1]) is None:
                continue
            pool2 = _single_user(pass_state, str(pre2[1].outputs[0]), "MAX_POOL_2D")
            if pool2 is None or len(pool2[1].outputs) != 1:
                continue
            post2 = _single_user(pass_state, str(pool2[1].outputs[0]), "RESHAPE")
            if post2 is None:
                continue
            pre2_shape = _reshape_shape(candidate_model, pre2[1])
            post2_shape = _reshape_shape(candidate_model, post2[1])
            if (
                pre2_shape is not None
                and post2_shape is not None
                and int(pre2_shape[3]) == 1
                and int(post2_shape[1]) == 1
            ):
                return True
        return False

    def _has_nms_candidate(pass_state: ModelIRPassState) -> bool:
        candidate_model = pass_state.model_ir
        for post1_op in candidate_model.operators:
            post1_shape = _reshape_shape(candidate_model, post1_op)
            if post1_shape is None or int(post1_shape[1]) != 1:
                continue
            pool = pass_state.graph_index.producer(str(post1_op.inputs[0]))
            if pool is None or str(pool.op_type) != "MAX_POOL_2D" or len(pool.inputs) != 1:
                continue
            cast = pass_state.graph_index.producer(str(pool.inputs[0]))
            if cast is None or str(cast.op_type) != "CAST" or len(cast.inputs) != 1:
                continue
            equal = pass_state.graph_index.producer(str(cast.inputs[0]))
            if equal is not None and str(equal.op_type) == "EQUAL":
                return True
        return False

    def _run_binary_cast(pass_state: ModelIRPassState) -> Dict[str, int | bool]:
        stats = _optimize_singleton_layout_reshape_maxpool_binary_cast_chains(
            pass_state.model_ir,
            graph_index=pass_state.graph_index,
            layout_state=pass_state.layout_state,
        )
        return {
            **stats,
            "changed": bool(
                stats.get(
                    "rewritten_singleton_layout_reshape_maxpool_binary_cast_chains",
                    0,
                )
            ),
        }

    def _run_nms(pass_state: ModelIRPassState) -> Dict[str, int | bool]:
        stats = _optimize_singleton_nms_maxpool_nhwc_chains(
            pass_state.model_ir,
            graph_index=pass_state.graph_index,
            layout_state=pass_state.layout_state,
        )
        return {
            **stats,
            "changed": bool(stats.get("rewritten_singleton_nms_maxpool_nhwc_chains", 0)),
        }

    specs: List[PassSpec[ModelIRPassState]] = []
    if include_binary_cast:
        specs.append(
            PassSpec(
                pass_id="layout.singleton_maxpool_binary_cast",
                phase=PassPhase.LAYOUT_PLAN,
                callback=_run_binary_cast,
                precondition=_has_binary_cast_candidate,
                priority=10,
                transactional=True,
            )
        )
    if include_nms:
        specs.append(
            PassSpec(
                pass_id="layout.singleton_nms_maxpool_nhwc",
                phase=PassPhase.LAYOUT_PLAN,
                callback=_run_nms,
                precondition=_has_nms_candidate,
                priority=20,
                transactional=True,
            )
        )
    default_details = {
        "rewritten_singleton_layout_reshape_maxpool_binary_cast_chains": 0,
        "rewritten_singleton_nms_maxpool_nhwc_chains": 0,
    }
    if len(specs) == 0:
        return default_details
    details, _ = run_model_ir_pass_group(
        model_ir,
        specs=specs,
        layout_state=layout_state,
        default_details=default_details,
        diagnostics=diagnostics,
        state_scope=state_scope,
        preflight=_preflight,
    )
    return {str(key): int(value) for key, value in details.items()}
