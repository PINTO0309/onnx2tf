from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from onnx2tf.tflite_builder.core.graph import ModelIRGraphIndex
from onnx2tf.tflite_builder.core.model_ir_utils import (
    _broadcast_shape_signatures,
    _clone_quantization,
    _is_per_tensor_quantization,
    _permute_shape,
    _prune_unused_tensors,
    _replace_tensor_inputs,
    _set_operator_inputs,
    _set_operator_outputs,
)
from onnx2tf.tflite_builder.ir import (
    ModelIR,
    QuantParamIR,
    TensorIR,
    remap_layout_through_permute,
)


def _optimize_transpose_elementwise_roundtrip_nchw_nhwc_chains(model_ir: ModelIR) -> Dict[str, int]:
    """
    Remove NCHW->NHWC->NCHW bridge pairs around a layout-agnostic elementwise subgraph.

    Pattern:
      (one or more) pre_transpose_i:  T(0,2,3,1)(x_i_nchw) -> x_i_nhwc
      elementwise-only subgraph in NHWC
      root_nhwc --T(0,3,1,2)--> root_nchw

    Rewrite:
      - Bypass each pre-transpose into the NHWC elementwise subgraph.
      - Keep the subgraph in NCHW.
      - Remove pre-transposes and the trailing inverse post-transpose.
    """
    rewritten = 0
    perm_nchw_to_nhwc = [0, 2, 3, 1]
    perm_nhwc_to_nchw = [0, 3, 1, 2]
    # Restrict to ops whose semantics are independent of channel axis placement.
    allowed_elementwise_ops = {
        "ABS",
        "ADD",
        "DIV",
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
    unary_elementwise_ops = {
        "ABS",
        "EXP",
        "FLOOR",
        "LOG",
        "NEG",
        "RSQRT",
        "SIGN",
        "SQRT",
    }
    graph_index = ModelIRGraphIndex(model_ir)

    def _broadcast_shape(
        shape_a: List[int],
        shape_b: List[int],
    ) -> Optional[List[int]]:
        if any(int(value) <= 0 for value in list(shape_a) + list(shape_b)):
            return None
        rank = max(len(shape_a), len(shape_b))
        values_a = [1] * (rank - len(shape_a)) + [int(v) for v in shape_a]
        values_b = [1] * (rank - len(shape_b)) + [int(v) for v in shape_b]
        output: List[int] = []
        for value_a, value_b in zip(values_a, values_b):
            if value_a == value_b:
                output.append(value_a)
            elif value_a == 1:
                output.append(value_b)
            elif value_b == 1:
                output.append(value_a)
            else:
                return None
        return output

    def _rank_four_metadata(
        tensor: Optional[TensorIR],
        expected_layout: str,
        *,
        runtime: bool,
    ) -> Optional[Tuple[List[int], List[int]]]:
        if tensor is None:
            return None
        shape = [int(value) for value in list(tensor.shape)]
        signature = (
            [int(value) for value in list(tensor.shape_signature)]
            if tensor.shape_signature is not None
            else None
        )
        if (
            len(shape) != 4
            or any(int(value) <= 0 for value in shape)
            or signature is None
            or len(signature) != 4
            or any(int(value) == 0 or int(value) < -1 for value in signature)
            or (runtime and tensor.data is not None)
        ):
            return None
        for layout in (tensor.logical_layout, tensor.physical_layout):
            if str(layout or "UNKNOWN").upper() not in {"UNKNOWN", expected_layout}:
                return None
        return shape, signature

    def _safe_permutation_constant(
        tensor_name: str,
        expected: List[int],
        model_inputs: set[str],
        model_outputs: set[str],
    ) -> bool:
        tensor = model_ir.tensors.get(tensor_name, None)
        if (
            tensor is None
            or tensor_name in model_inputs
            or tensor_name in model_outputs
            or bool(tensor.is_variable)
            or tensor.quantization is not None
            or str(tensor.dtype).upper() != "INT32"
            or list(tensor.shape) != [4]
            or list(tensor.shape_signature or []) != [4]
            or tensor.data is None
            or graph_index.producers.get(tensor_name, None) is not None
            or tensor_name in graph_index.duplicate_producers
        ):
            return False
        values = np.asarray(tensor.data)
        return bool(
            values.dtype == np.dtype(np.int32)
            and list(values.shape) == [4]
            and [int(value) for value in values.tolist()] == expected
        )

    def _permuted_quantization(
        quantization: Any,
        source_rank: int,
    ) -> Optional[Any]:
        cloned = _clone_quantization(quantization)
        if cloned is None or _is_per_tensor_quantization(cloned):
            return cloned
        if isinstance(cloned, QuantParamIR):
            dimension = int(cloned.quantized_dimension)
        elif isinstance(cloned, dict) and "quantized_dimension" in cloned:
            dimension = int(cloned["quantized_dimension"])
        else:
            return None
        if dimension < 0:
            dimension += int(source_rank)
        if dimension < 0 or dimension >= int(source_rank):
            return None
        mapped = int(perm_nhwc_to_nchw.index(dimension + 4 - source_rank))
        if isinstance(cloned, QuantParamIR):
            cloned.quantized_dimension = mapped
        else:
            cloned["quantized_dimension"] = mapped
        return cloned

    def _unique_tensor_name(base_name: str, reserved_names: set[str]) -> str:
        candidate = str(base_name)
        serial = 0
        while candidate in model_ir.tensors or candidate in reserved_names:
            serial += 1
            candidate = f"{base_name}_{serial}"
        reserved_names.add(candidate)
        return candidate

    while True:
        changed = False
        consumers = graph_index.consumers
        producers = graph_index.producers
        model_inputs = set(str(v) for v in model_ir.inputs)
        model_outputs = set(str(v) for v in model_ir.outputs)

        for post_idx in graph_index.operator_indices("TRANSPOSE"):
            post_op = model_ir.operators[int(post_idx)]
            if len(post_op.inputs) != 2 or len(post_op.outputs) != 1:
                continue
            if not _safe_permutation_constant(
                str(post_op.inputs[1]),
                perm_nhwc_to_nchw,
                model_inputs,
                model_outputs,
            ):
                continue

            root_nhwc_name = str(post_op.inputs[0])
            post_output_name = str(post_op.outputs[0])
            if (
                root_nhwc_name in model_inputs
                or root_nhwc_name in model_outputs
                or post_output_name in model_inputs
                or post_output_name in model_outputs
                or root_nhwc_name in graph_index.duplicate_producers
                or post_output_name in graph_index.duplicate_producers
                or producers.get(post_output_name, None) != int(post_idx)
            ):
                continue

            root_idx = producers.get(root_nhwc_name, None)
            if root_idx is None or int(root_idx) >= int(post_idx):
                continue
            root_tensor = model_ir.tensors.get(root_nhwc_name, None)
            canonical_tensor = model_ir.tensors.get(post_output_name, None)
            root_metadata = _rank_four_metadata(
                root_tensor,
                "NHWC",
                runtime=True,
            )
            canonical_metadata = _rank_four_metadata(
                canonical_tensor,
                "NCHW",
                runtime=True,
            )
            if root_metadata is None or canonical_metadata is None:
                continue
            root_shape, root_signature = root_metadata
            canonical_shape, canonical_signature = canonical_metadata
            if (
                _permute_shape(root_shape, perm_nhwc_to_nchw) != canonical_shape
                or _permute_shape(root_signature, perm_nhwc_to_nchw)
                != canonical_signature
                or str(root_tensor.dtype) != str(canonical_tensor.dtype)
                or any(
                    int(user_idx) <= int(post_idx)
                    for user_idx in consumers.get(post_output_name, [])
                )
            ):
                continue

            # Backward traverse from root producer to collect an elementwise-only subgraph.
            subgraph_indices: set[int] = set()
            pre_transpose_output_to_index: Dict[str, int] = {}
            constant_uses: Dict[str, set[int]] = {}
            stack: List[int] = [int(root_idx)]
            valid = True

            while len(stack) > 0 and valid:
                op_idx = int(stack.pop())
                if op_idx in subgraph_indices:
                    continue
                if op_idx >= int(post_idx):
                    valid = False
                    break
                op = model_ir.operators[int(op_idx)]
                op_type = str(op.op_type)
                expected_input_count = 1 if op_type in unary_elementwise_ops else 2
                if (
                    op_type not in allowed_elementwise_ops
                    or len(op.inputs) != expected_input_count
                    or len(op.outputs) != 1
                ):
                    valid = False
                    break

                output_name = str(op.outputs[0])
                if (
                    _rank_four_metadata(
                        model_ir.tensors.get(output_name, None),
                        "NHWC",
                        runtime=True,
                    )
                    is None
                    or output_name in model_inputs
                    or output_name in model_outputs
                    or output_name in graph_index.duplicate_producers
                    or producers.get(output_name, None) != op_idx
                ):
                    valid = False
                    break
                subgraph_indices.add(int(op_idx))
                for input_name_raw in list(op.inputs):
                    input_name = str(input_name_raw)
                    input_tensor = model_ir.tensors.get(input_name, None)
                    if input_tensor is None or input_name in graph_index.duplicate_producers:
                        valid = False
                        break
                    prod_idx = producers.get(input_name, None)
                    if prod_idx is None:
                        if input_tensor.data is not None:
                            constant_uses.setdefault(input_name, set()).add(op_idx)
                            continue
                        valid = False
                        break

                    if int(prod_idx) >= op_idx:
                        valid = False
                        break
                    prod_op = model_ir.operators[int(prod_idx)]
                    if (
                        str(prod_op.op_type) == "TRANSPOSE"
                        and len(prod_op.inputs) == 2
                        and len(prod_op.outputs) == 1
                        and str(prod_op.outputs[0]) == input_name
                    ):
                        pre_input_name = str(prod_op.inputs[0])
                        if (
                            input_name in model_inputs
                            or input_name in model_outputs
                            or pre_input_name in model_outputs
                            or not _safe_permutation_constant(
                                str(prod_op.inputs[1]),
                                perm_nchw_to_nhwc,
                                model_inputs,
                                model_outputs,
                            )
                        ):
                            valid = False
                            break
                        pre_input_tensor = model_ir.tensors.get(pre_input_name, None)
                        pre_input_metadata = _rank_four_metadata(
                            pre_input_tensor,
                            "NCHW",
                            runtime=False,
                        )
                        pre_output_metadata = _rank_four_metadata(
                            input_tensor,
                            "NHWC",
                            runtime=True,
                        )
                        if pre_input_metadata is None or pre_output_metadata is None:
                            valid = False
                            break
                        pre_input_shape, pre_input_signature = pre_input_metadata
                        pre_output_shape, pre_output_signature = pre_output_metadata
                        if (
                            _permute_shape(pre_input_shape, perm_nchw_to_nhwc)
                            != pre_output_shape
                            or _permute_shape(
                                pre_input_signature,
                                perm_nchw_to_nhwc,
                            )
                            != pre_output_signature
                            or str(pre_input_tensor.dtype) != str(input_tensor.dtype)
                        ):
                            valid = False
                            break
                        pre_source_idx = producers.get(pre_input_name, None)
                        if pre_source_idx is not None and int(pre_source_idx) >= int(prod_idx):
                            valid = False
                            break
                        pre_transpose_output_to_index[str(input_name)] = int(prod_idx)
                        continue

                    if str(prod_op.op_type) in allowed_elementwise_ops:
                        stack.append(int(prod_idx))
                        continue

                    valid = False
                    break

            if not valid or len(pre_transpose_output_to_index) == 0:
                continue

            # Root output must be consumed only by the matched post-transpose.
            if set(int(v) for v in consumers.get(root_nhwc_name, [])) != {int(post_idx)}:
                continue

            # Subgraph outputs must not leak outside subgraph except root->post.
            valid_outputs = True
            for op_idx in list(subgraph_indices):
                op = model_ir.operators[int(op_idx)]
                for output_name_raw in list(op.outputs):
                    output_name = str(output_name_raw)
                    users = set(int(v) for v in consumers.get(output_name, []))
                    if output_name == root_nhwc_name:
                        if users != {int(post_idx)}:
                            valid_outputs = False
                            break
                    else:
                        if not users.issubset(set(int(v) for v in list(subgraph_indices))):
                            valid_outputs = False
                            break
                if not valid_outputs:
                    break
            if not valid_outputs:
                continue

            # Each pre-transpose output must be local to this subgraph.
            valid_pre_consumers = True
            for pre_output_name, _ in pre_transpose_output_to_index.items():
                users = set(int(v) for v in consumers.get(str(pre_output_name), []))
                if len(users) == 0:
                    valid_pre_consumers = False
                    break
                if not users.issubset(set(int(v) for v in list(subgraph_indices))):
                    valid_pre_consumers = False
                    break
            if not valid_pre_consumers:
                continue

            # Prove the original elementwise shapes and dtypes before planning
            # the coordinate change.
            for op_idx in sorted(subgraph_indices):
                op = model_ir.operators[int(op_idx)]
                output_tensor = model_ir.tensors[str(op.outputs[0])]
                input_tensors = [model_ir.tensors[str(name)] for name in op.inputs]
                if any(
                    str(input_tensor.dtype) != str(output_tensor.dtype)
                    for input_tensor in input_tensors
                ):
                    valid = False
                    break
                if str(op.op_type) in unary_elementwise_ops:
                    input_tensor = input_tensors[0]
                    if (
                        list(input_tensor.shape) != list(output_tensor.shape)
                        or list(input_tensor.shape_signature or input_tensor.shape)
                        != list(output_tensor.shape_signature or output_tensor.shape)
                    ):
                        valid = False
                        break
                else:
                    if (
                        _broadcast_shape(
                            list(input_tensors[0].shape),
                            list(input_tensors[1].shape),
                        )
                        != list(output_tensor.shape)
                        or _broadcast_shape_signatures(
                            list(
                                input_tensors[0].shape_signature
                                or input_tensors[0].shape
                            ),
                            list(
                                input_tensors[1].shape_signature
                                or input_tensors[1].shape
                            ),
                        )
                        != list(output_tensor.shape_signature or output_tensor.shape)
                    ):
                        valid = False
                        break
            if not valid:
                continue

            reserved_names: set[str] = set()
            constant_actions: Dict[str, Dict[str, Any]] = {}
            for constant_name, use_indices in sorted(constant_uses.items()):
                constant_tensor = model_ir.tensors[constant_name]
                constant_data = np.asarray(constant_tensor.data)
                source_rank = int(constant_data.ndim)
                source_shape = [int(value) for value in list(constant_tensor.shape)]
                source_signature = (
                    [int(value) for value in list(constant_tensor.shape_signature)]
                    if constant_tensor.shape_signature is not None
                    else None
                )
                if (
                    source_rank > 4
                    or source_shape != [int(value) for value in constant_data.shape]
                    or source_signature != source_shape
                    or any(int(value) <= 0 for value in source_shape)
                ):
                    valid = False
                    break
                if int(constant_data.size) == 1:
                    continue
                if bool(constant_tensor.is_variable):
                    valid = False
                    break
                expanded_shape = [1] * (4 - source_rank) + source_shape
                remapped_data = np.transpose(
                    np.reshape(constant_data, expanded_shape),
                    axes=perm_nhwc_to_nchw,
                )
                remapped_shape = [int(value) for value in remapped_data.shape]
                remapped_quantization = _permuted_quantization(
                    constant_tensor.quantization,
                    source_rank,
                )
                if (
                    constant_tensor.quantization is not None
                    and not _is_per_tensor_quantization(constant_tensor.quantization)
                    and remapped_quantization is None
                ):
                    valid = False
                    break
                all_users = set(consumers.get(constant_name, []))
                must_clone = bool(
                    all_users - subgraph_indices
                    or constant_name in model_inputs
                    or constant_name in model_outputs
                )
                target_name = (
                    _unique_tensor_name(
                        f"{constant_name}__nchw",
                        reserved_names,
                    )
                    if must_clone
                    else constant_name
                )
                constant_actions[constant_name] = {
                    "target_name": target_name,
                    "use_indices": set(int(value) for value in use_indices),
                    "data": remapped_data.astype(constant_data.dtype, copy=False),
                    "shape": remapped_shape,
                    "signature": list(remapped_shape),
                    "quantization": remapped_quantization,
                    "logical_layout": (
                        remap_layout_through_permute(
                            layout=constant_tensor.logical_layout,
                            perm=perm_nhwc_to_nchw,
                        )
                        if source_rank == 4
                        else "UNKNOWN"
                    ),
                    "physical_layout": (
                        remap_layout_through_permute(
                            layout=constant_tensor.physical_layout,
                            perm=perm_nhwc_to_nchw,
                        )
                        if source_rank == 4
                        else "UNKNOWN"
                    ),
                }
            if not valid:
                continue

            pre_input_map = {
                str(pre_output_name): str(
                    model_ir.operators[int(pre_op_idx)].inputs[0]
                )
                for pre_output_name, pre_op_idx in pre_transpose_output_to_index.items()
            }
            output_metadata_plans: Dict[str, Dict[str, Any]] = {}
            for op_idx in sorted(subgraph_indices):
                output_name = str(model_ir.operators[int(op_idx)].outputs[0])
                output_tensor = model_ir.tensors[output_name]
                remapped_quantization = _permuted_quantization(
                    output_tensor.quantization,
                    4,
                )
                if (
                    output_tensor.quantization is not None
                    and not _is_per_tensor_quantization(output_tensor.quantization)
                    and remapped_quantization is None
                ):
                    valid = False
                    break
                output_metadata_plans[output_name] = {
                    "shape": _permute_shape(
                        list(output_tensor.shape),
                        perm_nhwc_to_nchw,
                    ),
                    "signature": _permute_shape(
                        list(output_tensor.shape_signature or output_tensor.shape),
                        perm_nhwc_to_nchw,
                    ),
                    "quantization": remapped_quantization,
                    "logical_layout": remap_layout_through_permute(
                        layout=output_tensor.logical_layout,
                        perm=perm_nhwc_to_nchw,
                    ),
                    "physical_layout": remap_layout_through_permute(
                        layout=output_tensor.physical_layout,
                        perm=perm_nhwc_to_nchw,
                    ),
                }
            if not valid:
                continue

            # Validate the complete post-rewrite broadcast plan before mutation.
            for op_idx in sorted(subgraph_indices):
                op = model_ir.operators[int(op_idx)]
                planned_shapes: List[List[int]] = []
                planned_signatures: List[List[int]] = []
                for input_name_raw in op.inputs:
                    input_name = str(input_name_raw)
                    if input_name in pre_input_map:
                        source_tensor = model_ir.tensors[pre_input_map[input_name]]
                        planned_shapes.append(list(source_tensor.shape))
                        planned_signatures.append(
                            list(source_tensor.shape_signature or source_tensor.shape)
                        )
                    elif input_name in output_metadata_plans:
                        planned_shapes.append(
                            list(output_metadata_plans[input_name]["shape"])
                        )
                        planned_signatures.append(
                            list(output_metadata_plans[input_name]["signature"])
                        )
                    elif input_name in constant_actions:
                        planned_shapes.append(
                            list(constant_actions[input_name]["shape"])
                        )
                        planned_signatures.append(
                            list(constant_actions[input_name]["signature"])
                        )
                    else:
                        input_tensor = model_ir.tensors[input_name]
                        planned_shapes.append(list(input_tensor.shape))
                        planned_signatures.append(
                            list(input_tensor.shape_signature or input_tensor.shape)
                        )
                output_plan = output_metadata_plans[str(op.outputs[0])]
                if str(op.op_type) in unary_elementwise_ops:
                    valid = bool(
                        planned_shapes[0] == output_plan["shape"]
                        and planned_signatures[0] == output_plan["signature"]
                    )
                else:
                    valid = bool(
                        _broadcast_shape(planned_shapes[0], planned_shapes[1])
                        == output_plan["shape"]
                        and _broadcast_shape_signatures(
                            planned_signatures[0],
                            planned_signatures[1],
                        )
                        == output_plan["signature"]
                    )
                if not valid:
                    break
            if not valid:
                continue

            # Commit the fully preflighted constant and tensor actions.
            for constant_name, action in constant_actions.items():
                source_tensor = model_ir.tensors[constant_name]
                target_name = str(action["target_name"])
                if target_name == constant_name:
                    target_tensor = source_tensor
                else:
                    target_tensor = TensorIR(
                        name=target_name,
                        dtype=str(source_tensor.dtype),
                        shape=list(action["shape"]),
                        shape_signature=list(action["signature"]),
                        data=action["data"],
                        is_variable=False,
                        quantization=_clone_quantization(action["quantization"]),
                        logical_layout=str(action["logical_layout"]),
                        physical_layout=str(action["physical_layout"]),
                        onnx_tensor_name=source_tensor.onnx_tensor_name,
                    )
                    model_ir.tensors[target_name] = target_tensor
                target_tensor.data = action["data"]
                target_tensor.shape = list(action["shape"])
                target_tensor.shape_signature = list(action["signature"])
                target_tensor.quantization = _clone_quantization(
                    action["quantization"]
                )
                target_tensor.logical_layout = str(action["logical_layout"])
                target_tensor.physical_layout = str(action["physical_layout"])

            for op_idx in sorted(subgraph_indices):
                op = model_ir.operators[int(op_idx)]
                new_inputs = []
                for input_name_raw in op.inputs:
                    input_name = str(input_name_raw)
                    if input_name in constant_actions:
                        input_name = str(
                            constant_actions[input_name]["target_name"]
                        )
                    else:
                        input_name = pre_input_map.get(input_name, input_name)
                    new_inputs.append(input_name)
                _set_operator_inputs(
                    model_ir=model_ir,
                    op=op,
                    new_inputs=new_inputs,
                    graph_index=graph_index,
                )

            for output_name, metadata_plan in output_metadata_plans.items():
                if output_name == root_nhwc_name:
                    continue
                tensor = model_ir.tensors[output_name]
                tensor.shape = list(metadata_plan["shape"])
                tensor.shape_signature = list(metadata_plan["signature"])
                tensor.quantization = _clone_quantization(
                    metadata_plan["quantization"]
                )
                tensor.logical_layout = str(metadata_plan["logical_layout"])
                tensor.physical_layout = str(metadata_plan["physical_layout"])

            # Replace root NHWC output with canonical post-transpose output.
            root_op = model_ir.operators[int(root_idx)]
            _set_operator_outputs(
                model_ir=model_ir,
                op=root_op,
                new_outputs=[post_output_name],
                graph_index=graph_index,
            )
            _replace_tensor_inputs(
                model_ir,
                root_nhwc_name,
                post_output_name,
                graph_index=graph_index,
            )

            canonical_plan = output_metadata_plans[root_nhwc_name]
            canonical_tensor.dtype = str(root_tensor.dtype)
            canonical_tensor.quantization = _clone_quantization(
                canonical_plan["quantization"]
            )
            canonical_tensor.shape = list(canonical_plan["shape"])
            canonical_tensor.shape_signature = list(canonical_plan["signature"])
            canonical_tensor.logical_layout = str(canonical_plan["logical_layout"])
            canonical_tensor.physical_layout = str(canonical_plan["physical_layout"])

            remove_indices = set([int(post_idx)])
            remove_indices.update(int(v) for v in list(pre_transpose_output_to_index.values()))
            graph_index.remove_operators(remove_indices)

            rewritten += 1
            changed = True
            break

        if not changed:
            break

    if rewritten > 0:
        _prune_unused_tensors(model_ir)
    return {"optimized_transpose_elementwise_roundtrip_nchw_nhwc_chains": int(rewritten)}

