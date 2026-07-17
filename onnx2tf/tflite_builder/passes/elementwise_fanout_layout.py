from __future__ import annotations

import copy
from typing import Dict, List

import numpy as np

from onnx2tf.tflite_builder.core.model_ir_utils import (
    _broadcast_static_shapes,
    _build_tensor_consumer_map,
    _build_tensor_producer_map,
    _clone_quantization,
    _permute_shape,
    _permute_tensor_metadata_if_rank_matches,
    _prune_unused_tensors,
    _read_transpose_perm,
    _replace_tensor_inputs,
    _set_operator_inputs,
    _set_operator_outputs,
)
from onnx2tf.tflite_builder.ir import ModelIR, OperatorIR, TensorIR
from onnx2tf.tflite_builder.passes.unbound_input_layout import (
    find_unbound_nonconstant_operator_inputs as _find_unbound_nonconstant_operator_inputs,
)


def optimize_transpose_elementwise_roundtrip_nhwc_nchw_fanout_chains(
    model_ir: ModelIR,
) -> Dict[str, int]:
    """
    Remove NHWC->NCHW and inverse NCHW->NHWC transpose bridges around an
    elementwise-only fanout subgraph.

    Pattern:
      x_nhwc --T(0,3,1,2)--> x_nchw
      x_nchw --(elementwise DAG; const side inputs allowed)--> y*_nchw
      y*_nchw --T(0,2,3,1)--> y*_nhwc

    Rewrite:
      - Keep the full elementwise DAG in NHWC by bypassing the leading transpose.
      - Remove all trailing inverse transposes at subgraph boundaries.

    Safety:
      - Subgraph ops are limited to layout-agnostic elementwise ops.
      - Per-channel rank-4 constants are remapped only when they are local to the subgraph.
      - Legacy NCHW users are preserved via local adapter TRANSPOSE ops.
    """
    rewritten = 0
    perm_nhwc_to_nchw = [0, 3, 1, 2]
    perm_nchw_to_nhwc = [0, 2, 3, 1]
    allowed_elementwise_ops = {
        "ABS",
        "ADD",
        "DIV",
        "ERF",
        "EXP",
        "FLOOR",
        "LOG",
        "MAXIMUM",
        "MINIMUM",
        "MUL",
        "NEG",
        "POW",
        "RSQRT",
        "SIGN",
        "SQRT",
        "SUB",
    }

    def _unique_tensor_name(base_name: str) -> str:
        candidate = str(base_name)
        serial = 0
        while candidate in model_ir.tensors:
            serial += 1
            candidate = f"{base_name}_{serial}"
        return candidate

    while True:
        changed = False
        consumers = _build_tensor_consumer_map(model_ir)
        producers = _build_tensor_producer_map(model_ir)
        model_outputs = set(str(v) for v in model_ir.outputs)

        for pre_idx, pre_op in enumerate(model_ir.operators):
            if str(pre_op.op_type) != "TRANSPOSE" or len(pre_op.inputs) < 2 or len(pre_op.outputs) != 1:
                continue
            if _read_transpose_perm(model_ir, pre_op) != perm_nhwc_to_nchw:
                continue

            pre_input_name = str(pre_op.inputs[0])
            pre_output_name = str(pre_op.outputs[0])
            if pre_output_name in model_outputs:
                continue

            # Forward-collect elementwise-only reachable ops.
            subgraph_indices: set[int] = set()
            frontier_tensors: List[str] = [pre_output_name]
            seen_tensors: set[str] = {pre_output_name}
            while len(frontier_tensors) > 0:
                current_tensor = str(frontier_tensors.pop())
                for user_idx_raw in consumers.get(current_tensor, []):
                    user_idx = int(user_idx_raw)
                    user_op = model_ir.operators[user_idx]
                    if str(user_op.op_type) not in allowed_elementwise_ops:
                        continue
                    if user_idx in subgraph_indices:
                        continue
                    subgraph_indices.add(user_idx)
                    for out_name_raw in list(user_op.outputs):
                        out_name = str(out_name_raw)
                        if out_name not in seen_tensors:
                            seen_tensors.add(out_name)
                            frontier_tensors.append(out_name)

            if len(subgraph_indices) == 0:
                continue

            # Validate local shape and collect runtime inputs coming from outside
            # the local elementwise region. Those external NCHW tensors are
            # adapted locally into NHWC.
            valid = True
            external_runtime_inputs: set[str] = set()
            for op_idx in sorted(list(subgraph_indices)):
                op = model_ir.operators[op_idx]
                if len(op.outputs) != 1:
                    valid = False
                    break
                for input_name_raw in list(op.inputs):
                    input_name = str(input_name_raw)
                    input_tensor = model_ir.tensors.get(input_name, None)
                    if input_tensor is not None and input_tensor.data is not None:
                        continue

                    if input_name == pre_output_name:
                        continue

                    prod_idx = producers.get(input_name, None)
                    if prod_idx is None or int(prod_idx) not in subgraph_indices:
                        external_runtime_inputs.add(str(input_name))
                if not valid:
                    break
            if not valid:
                continue

            # Ensure pre-transpose output is consumed only by this local subgraph.
            pre_users = set(int(v) for v in consumers.get(pre_output_name, []))
            if len(pre_users) == 0 or not pre_users.issubset(subgraph_indices):
                continue

            # Detect subgraph boundaries and classify external users:
            # - inverse post-transpose consumers (remove)
            # - legacy users (preserve via NHWC->NCHW adapter)
            boundary_posts: Dict[str, List[int]] = {}
            boundary_legacy_users: Dict[str, List[int]] = {}
            for op_idx in sorted(list(subgraph_indices)):
                op = model_ir.operators[op_idx]
                out_name = str(op.outputs[0])
                if out_name in model_outputs:
                    valid = False
                    break

                users = set(int(v) for v in consumers.get(out_name, []))
                external_users = sorted(list(users - subgraph_indices))
                if len(external_users) == 0:
                    continue

                for user_idx in external_users:
                    user_op = model_ir.operators[int(user_idx)]
                    if (
                        str(user_op.op_type) != "TRANSPOSE"
                        or len(user_op.inputs) < 2
                        or len(user_op.outputs) != 1
                        or str(user_op.inputs[0]) != out_name
                    ):
                        boundary_legacy_users.setdefault(out_name, []).append(int(user_idx))
                        continue
                    if (
                        _read_transpose_perm(model_ir, user_op) == perm_nchw_to_nhwc
                        and str(user_op.outputs[0]) not in model_outputs
                    ):
                        boundary_posts.setdefault(out_name, []).append(int(user_idx))
                    else:
                        boundary_legacy_users.setdefault(out_name, []).append(int(user_idx))
                if not valid:
                    break
            if not valid:
                continue
            if sum(len(v) for v in boundary_posts.values()) <= 0:
                continue
            # Conservative guard:
            # Rewriting mixed fanout islands that require external runtime
            # NCHW->NHWC adapter insertion has produced dangling `__to_nhwc`
            # tensors in some large graphs (e.g. network.7 branches).
            # Skip this candidate and leave it to other safer transpose passes.
            if len(external_runtime_inputs) > 0:
                continue

            # Safety guard:
            # This pass can be very aggressive on wide fanout graphs. Snapshot one
            # candidate rewrite and roll back if it introduces new unbound runtime
            # inputs (which later appears as "Input tensor N lacks data").
            candidate_snapshot = copy.deepcopy(model_ir)
            unbound_before_count = int(len(_find_unbound_nonconstant_operator_inputs(model_ir)))
            def _rollback_candidate() -> None:
                model_ir.tensors = candidate_snapshot.tensors
                model_ir.operators = candidate_snapshot.operators
                model_ir.inputs = candidate_snapshot.inputs
                model_ir.outputs = candidate_snapshot.outputs
                model_ir.subgraphs = candidate_snapshot.subgraphs
                model_ir.metadata = candidate_snapshot.metadata

            # Remap per-channel rank-4 constants from NCHW->NHWC where required.
            remapped_const_names: set[str] = set()
            for op_idx in sorted(list(subgraph_indices)):
                op = model_ir.operators[op_idx]
                runtime_input_names = []
                const_input_names = []
                for input_name_raw in list(op.inputs):
                    input_name = str(input_name_raw)
                    input_tensor = model_ir.tensors.get(input_name, None)
                    if input_tensor is not None and input_tensor.data is not None:
                        const_input_names.append(input_name)
                    else:
                        runtime_input_names.append(input_name)
                if len(runtime_input_names) == 0:
                    continue

                ref_name = next((name for name in runtime_input_names if name == pre_output_name), runtime_input_names[0])
                ref_tensor = model_ir.tensors.get(str(ref_name), None)
                if ref_tensor is None or ref_tensor.shape is None or len(list(ref_tensor.shape)) != 4:
                    continue
                ref_shape_nchw = [int(v) for v in list(ref_tensor.shape)]
                if any(int(v) <= 0 for v in ref_shape_nchw):
                    continue
                ref_shape_nhwc = _permute_shape(ref_shape_nchw, perm_nchw_to_nhwc)
                if ref_shape_nhwc is None:
                    continue

                for const_name in const_input_names:
                    if const_name in remapped_const_names:
                        continue
                    const_tensor = model_ir.tensors.get(str(const_name), None)
                    if const_tensor is None or const_tensor.data is None:
                        continue
                    const_data = np.asarray(const_tensor.data)
                    if int(const_data.ndim) != 4:
                        continue
                    const_shape = [int(v) for v in list(const_data.shape)]
                    # Strictly target NCHW per-channel broadcast constants [1,C,1,1].
                    if not (
                        const_shape[0] == 1
                        and const_shape[1] == ref_shape_nchw[1]
                        and const_shape[2] == 1
                        and const_shape[3] == 1
                    ):
                        continue
                    # Shared constants cannot be remapped safely because non-local
                    # users still expect NCHW semantics.
                    const_users = set(int(v) for v in consumers.get(str(const_name), []))
                    rotated_data = np.transpose(const_data, axes=perm_nchw_to_nhwc)
                    rotated_shape = [int(v) for v in list(rotated_data.shape)]
                    if _broadcast_static_shapes(ref_shape_nhwc, rotated_shape) is None:
                        valid = False
                        break
                    if const_users.issubset(subgraph_indices):
                        const_tensor.data = rotated_data.astype(const_data.dtype, copy=False)
                        const_tensor.shape = [int(v) for v in list(rotated_shape)]
                        const_tensor.shape_signature = (
                            [int(v) for v in list(rotated_shape)]
                            if const_tensor.shape_signature is None
                            else [int(v) for v in list(rotated_shape)]
                        )
                        remapped_const_names.add(str(const_name))
                    else:
                        # Shared constants are cloned for this rewritten branch.
                        clone_name = _unique_tensor_name(f"{const_name}__nhwc")
                        model_ir.tensors[str(clone_name)] = TensorIR(
                            name=str(clone_name),
                            dtype=str(const_tensor.dtype),
                            shape=[int(v) for v in list(rotated_shape)],
                            shape_signature=[int(v) for v in list(rotated_shape)],
                            data=rotated_data.astype(const_data.dtype, copy=False),
                            is_variable=False,
                            quantization=_clone_quantization(const_tensor.quantization),
                        )
                        op_inputs = [str(v) for v in list(op.inputs)]
                        op_inputs = [str(clone_name) if str(v) == str(const_name) else str(v) for v in op_inputs]
                        _set_operator_inputs(
                            model_ir=model_ir,
                            op=op,
                            new_inputs=op_inputs,
                        )
                        remapped_const_names.add(str(clone_name))
                if not valid:
                    break
            if not valid:
                _rollback_candidate()
                continue

            # Rewire all subgraph inputs from pre-transpose output to NHWC source.
            for op_idx in sorted(list(subgraph_indices)):
                op = model_ir.operators[int(op_idx)]
                new_inputs = [
                    pre_input_name if str(v) == pre_output_name else str(v)
                    for v in list(op.inputs)
                ]
                _set_operator_inputs(
                    model_ir=model_ir,
                    op=op,
                    new_inputs=new_inputs,
                )

            # Bridge external NCHW runtime inputs into NHWC for rewritten ops.
            if len(external_runtime_inputs) > 0:
                for external_input_name in sorted(list(external_runtime_inputs)):
                    external_tensor = model_ir.tensors.get(str(external_input_name), None)
                    if (
                        external_tensor is None
                        or external_tensor.shape is None
                        or len(list(external_tensor.shape)) != 4
                    ):
                        valid = False
                        break
                    source_shape = [int(v) for v in list(external_tensor.shape)]
                    target_shape = _permute_shape(source_shape, perm_nchw_to_nhwc)
                    if target_shape is None:
                        valid = False
                        break
                    adapter_perm_name = _unique_tensor_name(
                        f"{external_input_name}__to_nhwc_perm",
                    )
                    adapter_output_name = _unique_tensor_name(
                        f"{external_input_name}__to_nhwc",
                    )
                    model_ir.tensors[str(adapter_perm_name)] = TensorIR(
                        name=str(adapter_perm_name),
                        dtype="INT32",
                        shape=[4],
                        shape_signature=[4],
                        data=np.asarray(perm_nchw_to_nhwc, dtype=np.int32),
                        is_variable=False,
                        quantization=None,
                    )
                    model_ir.tensors[str(adapter_output_name)] = TensorIR(
                        name=str(adapter_output_name),
                        dtype=str(external_tensor.dtype),
                        shape=[int(v) for v in list(target_shape)],
                        shape_signature=[int(v) for v in list(target_shape)],
                        data=None,
                        is_variable=False,
                        quantization=_clone_quantization(external_tensor.quantization),
                    )
                    model_ir.operators.append(
                        OperatorIR(
                            op_type="TRANSPOSE",
                            inputs=[str(external_input_name), str(adapter_perm_name)],
                            outputs=[str(adapter_output_name)],
                        )
                    )
                    for op_idx in sorted(list(subgraph_indices)):
                        op = model_ir.operators[int(op_idx)]
                        new_inputs = [
                            str(adapter_output_name) if str(v) == str(external_input_name) else str(v)
                            for v in list(op.inputs)
                        ]
                        _set_operator_inputs(
                            model_ir=model_ir,
                            op=op,
                            new_inputs=new_inputs,
                        )
            if not valid:
                _rollback_candidate()
                continue

            # Subgraph tensors were NCHW; convert metadata to NHWC.
            for op_idx in sorted(list(subgraph_indices)):
                op = model_ir.operators[int(op_idx)]
                out_name = str(op.outputs[0])
                _permute_tensor_metadata_if_rank_matches(
                    model_ir.tensors.get(out_name, None),
                    perm_nchw_to_nhwc,
                )

            # Final broadcast-safety guard:
            # if an elementwise op now consumes NHWC data with a legacy
            # [1,C,1,1] constant, rotate (or clone+rotate) that constant.
            for op_idx in sorted(list(subgraph_indices)):
                op = model_ir.operators[int(op_idx)]
                if str(op.op_type) not in {"ADD", "SUB", "MUL", "DIV", "MAXIMUM", "MINIMUM", "POW"}:
                    continue
                input_names = [str(v) for v in list(op.inputs)]
                for const_input_name in list(input_names):
                    const_tensor = model_ir.tensors.get(str(const_input_name), None)
                    if const_tensor is None or const_tensor.data is None:
                        continue
                    const_data = np.asarray(const_tensor.data)
                    if int(const_data.ndim) != 4:
                        continue
                    data_input_name = next((name for name in input_names if str(name) != str(const_input_name)), None)
                    if data_input_name is None:
                        continue
                    data_tensor = model_ir.tensors.get(str(data_input_name), None)
                    if data_tensor is None or data_tensor.shape is None or len(list(data_tensor.shape)) != 4:
                        continue
                    data_shape = [int(v) for v in list(data_tensor.shape)]
                    if any(int(v) <= 0 for v in data_shape):
                        continue
                    const_shape = [int(v) for v in list(const_data.shape)]
                    if _broadcast_static_shapes(data_shape, const_shape) is not None:
                        continue
                    rotated_data = np.transpose(const_data, axes=perm_nchw_to_nhwc)
                    rotated_shape = [int(v) for v in list(rotated_data.shape)]
                    if _broadcast_static_shapes(data_shape, rotated_shape) is None:
                        continue
                    const_users = set(int(v) for v in consumers.get(str(const_input_name), []))
                    if const_users.issubset(subgraph_indices):
                        const_tensor.data = rotated_data.astype(const_data.dtype, copy=False)
                        const_tensor.shape = [int(v) for v in list(rotated_shape)]
                        const_tensor.shape_signature = [int(v) for v in list(rotated_shape)]
                    else:
                        clone_name = _unique_tensor_name(f"{const_input_name}__nhwc")
                        model_ir.tensors[str(clone_name)] = TensorIR(
                            name=str(clone_name),
                            dtype=str(const_tensor.dtype),
                            shape=[int(v) for v in list(rotated_shape)],
                            shape_signature=[int(v) for v in list(rotated_shape)],
                            data=rotated_data.astype(const_data.dtype, copy=False),
                            is_variable=False,
                            quantization=_clone_quantization(const_tensor.quantization),
                        )
                        _set_operator_inputs(
                            model_ir=model_ir,
                            op=op,
                            new_inputs=[
                                str(clone_name) if str(v) == str(const_input_name) else str(v)
                                for v in list(op.inputs)
                            ],
                        )

            # Replace each boundary with one canonical post-transpose output.
            post_indices_to_remove: set[int] = set()
            boundary_output_names = sorted(
                set(list(boundary_posts.keys()) + list(boundary_legacy_users.keys()))
            )
            for boundary_name in boundary_output_names:
                post_indices = [int(v) for v in list(boundary_posts.get(boundary_name, []))]
                legacy_users = [int(v) for v in list(boundary_legacy_users.get(boundary_name, []))]
                producer_idx = producers.get(str(boundary_name), None)
                if producer_idx is None or int(producer_idx) not in subgraph_indices:
                    valid = False
                    break
                producer_op = model_ir.operators[int(producer_idx)]
                if len(producer_op.outputs) != 1 or str(producer_op.outputs[0]) != str(boundary_name):
                    valid = False
                    break

                if len(post_indices) > 0:
                    canonical_post_idx = int(sorted(set(int(v) for v in post_indices))[0])
                    canonical_post_op = model_ir.operators[canonical_post_idx]
                    canonical_output_name = str(canonical_post_op.outputs[0])
                else:
                    canonical_post_idx = None
                    canonical_output_name = _unique_tensor_name(f"{boundary_name}__nhwc")
                    old_boundary_tensor = model_ir.tensors.get(str(boundary_name), None)
                    if old_boundary_tensor is not None:
                        model_ir.tensors[str(canonical_output_name)] = TensorIR(
                            name=str(canonical_output_name),
                            dtype=str(old_boundary_tensor.dtype),
                            shape=[int(v) for v in list(old_boundary_tensor.shape)],
                            shape_signature=(
                                [int(v) for v in list(old_boundary_tensor.shape_signature)]
                                if old_boundary_tensor.shape_signature is not None
                                else [int(v) for v in list(old_boundary_tensor.shape)]
                            ),
                            data=None,
                            is_variable=False,
                            quantization=_clone_quantization(old_boundary_tensor.quantization),
                        )

                _set_operator_outputs(
                    model_ir=model_ir,
                    op=producer_op,
                    new_outputs=[canonical_output_name],
                )
                _replace_tensor_inputs(model_ir, str(boundary_name), canonical_output_name)

                old_boundary_tensor = model_ir.tensors.get(str(boundary_name), None)
                canonical_tensor = model_ir.tensors.get(canonical_output_name, None)
                if old_boundary_tensor is not None and canonical_tensor is not None:
                    canonical_tensor.dtype = str(old_boundary_tensor.dtype)
                    canonical_tensor.quantization = _clone_quantization(old_boundary_tensor.quantization)
                    canonical_tensor.shape = [int(v) for v in list(old_boundary_tensor.shape)]
                    canonical_tensor.shape_signature = (
                        [int(v) for v in list(old_boundary_tensor.shape_signature)]
                        if old_boundary_tensor.shape_signature is not None
                        else [int(v) for v in list(old_boundary_tensor.shape)]
                    )

                for post_idx in sorted(set(int(v) for v in list(post_indices))):
                    post_indices_to_remove.add(int(post_idx))
                    if canonical_post_idx is not None and int(post_idx) == int(canonical_post_idx):
                        continue
                    post_output_name = str(model_ir.operators[int(post_idx)].outputs[0])
                    _replace_tensor_inputs(model_ir, post_output_name, canonical_output_name)

                # Preserve external NCHW users by appending a local NHWC->NCHW adapter.
                if len(legacy_users) > 0:
                    adapter_perm_name = _unique_tensor_name(
                        f"{boundary_name}__to_nchw_perm",
                    )
                    adapter_output_name = str(boundary_name)
                    canonical_tensor = model_ir.tensors.get(str(canonical_output_name), None)
                    if canonical_tensor is None or canonical_tensor.shape is None:
                        valid = False
                        break
                    canonical_shape = [int(v) for v in list(canonical_tensor.shape)]
                    legacy_shape = _permute_shape(canonical_shape, perm_nhwc_to_nchw)
                    if legacy_shape is None:
                        valid = False
                        break
                    model_ir.tensors[str(adapter_perm_name)] = TensorIR(
                        name=str(adapter_perm_name),
                        dtype="INT32",
                        shape=[4],
                        shape_signature=[4],
                        data=np.asarray(perm_nhwc_to_nchw, dtype=np.int32),
                        is_variable=False,
                        quantization=None,
                    )
                    model_ir.tensors[str(adapter_output_name)] = TensorIR(
                        name=str(adapter_output_name),
                        dtype=str(canonical_tensor.dtype),
                        shape=[int(v) for v in list(legacy_shape)],
                        shape_signature=[int(v) for v in list(legacy_shape)],
                        data=None,
                        is_variable=False,
                        quantization=_clone_quantization(canonical_tensor.quantization),
                    )
                    model_ir.operators.append(
                        OperatorIR(
                            op_type="TRANSPOSE",
                            inputs=[str(canonical_output_name), str(adapter_perm_name)],
                            outputs=[str(adapter_output_name)],
                        )
                    )
                    # Restore legacy users to NCHW tensor name.
                    for legacy_idx in list(legacy_users):
                        legacy_op = model_ir.operators[int(legacy_idx)]
                        new_inputs = [
                            str(adapter_output_name) if str(v) == str(canonical_output_name) else str(v)
                            for v in list(legacy_op.inputs)
                        ]
                        _set_operator_inputs(
                            model_ir=model_ir,
                            op=legacy_op,
                            new_inputs=new_inputs,
                        )

            if not valid:
                _rollback_candidate()
                continue

            remove_indices = set([int(pre_idx)])
            remove_indices.update(int(v) for v in list(post_indices_to_remove))
            for remove_idx in sorted(list(remove_indices), reverse=True):
                del model_ir.operators[int(remove_idx)]

            unbound_after_count = int(len(_find_unbound_nonconstant_operator_inputs(model_ir)))
            if unbound_after_count > unbound_before_count:
                _rollback_candidate()
                continue

            rewritten += 1
            changed = True
            break

        if not changed:
            break

    _prune_unused_tensors(model_ir)
    return {"optimized_transpose_elementwise_roundtrip_nhwc_nchw_fanout_chains": int(rewritten)}
