from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np

from onnx2tf.tflite_builder.core.graph import ModelIRGraphIndex
from onnx2tf.tflite_builder.ir import ModelIR


def _resolve_reshape_new_shape_from_static_input(
    new_shape: List[int],
    input_signature: Optional[List[int]],
    allow_zero: Optional[bool] = None,
) -> Optional[List[int]]:
    if len(new_shape) == 0:
        return None
    candidate = [int(v) for v in new_shape]

    if any(int(dim) == 0 for dim in candidate):
        # Without allowzero metadata we avoid rewriting zero-dim requests.
        if allow_zero is None:
            return None
        if allow_zero:
            return None
        if input_signature is None or len(input_signature) == 0:
            return None
        for idx, dim in enumerate(candidate):
            if int(dim) != 0:
                continue
            if idx >= len(input_signature):
                return None
            in_dim = int(input_signature[idx])
            if in_dim <= 0:
                return None
            candidate[idx] = int(in_dim)

    if all(int(dim) >= 0 for dim in candidate):
        if input_signature is None or len(input_signature) == 0:
            return list(candidate)
        if any(int(dim) <= 0 for dim in input_signature):
            return list(candidate)

        input_product = int(np.prod(np.asarray(input_signature, dtype=np.int64)))
        if input_product <= 0:
            return list(candidate)
        target_product = int(np.prod(np.asarray(candidate, dtype=np.int64)))
        if target_product == input_product:
            return list(candidate)

        # Stale reshape constants can appear after late layout rewrites.
        # If exactly one dimension can be re-inferred from static input size,
        # repair that dimension instead of keeping an invalid constant.
        inferable_dims: List[Tuple[int, int]] = []
        for idx in range(len(candidate)):
            other_product = 1
            valid = True
            for j, dim in enumerate(candidate):
                if j == idx:
                    continue
                d = int(dim)
                if d <= 0:
                    valid = False
                    break
                other_product *= d
            if not valid or other_product <= 0:
                continue
            if input_product % other_product != 0:
                continue
            inferred_dim = int(input_product // other_product)
            if inferred_dim <= 0:
                continue
            inferable_dims.append((int(idx), int(inferred_dim)))

        if len(inferable_dims) == 1:
            dim_idx, inferred_dim = inferable_dims[0]
            repaired = [int(v) for v in candidate]
            repaired[dim_idx] = int(inferred_dim)
            return repaired

        return None

    minus_one_indices = [idx for idx, dim in enumerate(candidate) if int(dim) == -1]
    if len(minus_one_indices) != 1:
        return None
    if input_signature is None or len(input_signature) == 0:
        return None
    if any(int(dim) <= 0 for dim in input_signature):
        return None

    known_product = 1
    for dim in candidate:
        if int(dim) == -1:
            continue
        if int(dim) <= 0:
            return None
        known_product *= int(dim)
    if known_product <= 0:
        return None

    input_product = int(np.prod(np.asarray(input_signature, dtype=np.int64)))
    if input_product <= 0 or input_product % known_product != 0:
        return None
    inferred = int(input_product // known_product)
    if inferred <= 0:
        return None
    candidate[minus_one_indices[0]] = inferred
    return candidate


def resolve_dynamic_reshape_shapes(
    model_ir: ModelIR,
    prefer_runtime_inferable_from_onnx_raw: bool = False,
    *,
    graph_index: Optional[ModelIRGraphIndex] = None,
) -> Dict[str, int]:
    def _sanitize_reshape_template(
        *,
        template: List[int],
        input_dims: List[int],
        allow_zero: Optional[bool] = None,
    ) -> List[int]:
        sanitized = [int(v) for v in list(template)]
        for idx, dim in enumerate(list(sanitized)):
            if int(dim) == 0:
                if allow_zero is False:
                    copied = (
                        int(input_dims[idx])
                        if idx < len(input_dims) and int(input_dims[idx]) > 0
                        else None
                    )
                    if copied is not None:
                        sanitized[idx] = int(copied)
                    else:
                        sanitized[idx] = -1
            elif int(dim) < -1:
                sanitized[idx] = 1

        return [int(v) for v in list(sanitized)]

    active_index = (
        graph_index
        if graph_index is not None and graph_index.model_ir is model_ir
        else None
    )
    candidate_operators = (
        (
            model_ir.operators[int(operator_index)]
            for operator_index in active_index.operator_indices("RESHAPE")
        )
        if active_index is not None
        else iter(model_ir.operators)
    )

    resolved_count = 0
    for op in candidate_operators:
        if str(op.op_type) != "RESHAPE":
            continue
        if bool(op.options.get("layoutTransposeAsReshape", False)):
            input_tensor = (
                model_ir.tensors.get(str(op.inputs[0]), None)
                if len(op.inputs) > 0
                else None
            )
            output_tensor = (
                model_ir.tensors.get(str(op.outputs[0]), None)
                if len(op.outputs) == 1
                else None
            )
            input_signature = (
                [int(v) for v in list(input_tensor.shape_signature)]
                if input_tensor is not None
                and input_tensor.shape_signature is not None
                else []
            )
            output_shape = (
                [int(v) for v in list(output_tensor.shape)]
                if output_tensor is not None
                else []
            )
            if (
                len(input_signature) == len(output_shape)
                and len(output_shape) > 0
                and int(input_signature[0]) <= 0
                and all(int(v) > 0 for v in output_shape[1:])
            ):
                dynamic_target = [-1, *output_shape[1:]]
                op.options["newShape"] = list(dynamic_target)
                op.options["preserveDynamicShape"] = True
                if output_tensor is not None:
                    output_signature = (
                        [int(v) for v in list(output_tensor.shape_signature)]
                        if output_tensor.shape_signature is not None
                        else list(output_shape)
                    )
                    if len(output_signature) == len(dynamic_target):
                        output_signature[0] = -1
                        output_tensor.shape_signature = output_signature
                if len(op.inputs) >= 2:
                    shape_tensor = model_ir.tensors.get(str(op.inputs[1]), None)
                    if shape_tensor is not None and shape_tensor.data is not None:
                        shape_tensor.data = np.asarray(dynamic_target, dtype=np.int32)
                resolved_count += 1
                continue
        if bool(op.options.get("preserveDynamicShape", False)):
            continue
        if len(op.inputs) < 1 or len(op.outputs) != 1:
            continue

        has_onnx_raw_new_shape = "onnxRawNewShape" in op.options
        raw_new_shape = op.options.get("onnxRawNewShape", op.options.get("newShape", []))
        try:
            new_shape = [int(v) for v in np.asarray(raw_new_shape).reshape(-1).tolist()]
        except Exception:
            continue
        shape_tensor_new_shape: List[int] = []
        if len(op.inputs) >= 2:
            shape_tensor = model_ir.tensors.get(op.inputs[1], None)
            if shape_tensor is not None and shape_tensor.data is not None:
                try:
                    shape_tensor_new_shape = [
                        int(v) for v in np.asarray(shape_tensor.data).reshape(-1).tolist()
                    ]
                except Exception:
                    shape_tensor_new_shape = []
        if len(new_shape) == 0 and len(shape_tensor_new_shape) > 0:
            new_shape = [int(v) for v in list(shape_tensor_new_shape)]
        existing_new_shape = op.options.get("newShape", [])
        try:
            existing_new_shape_list = [
                int(v) for v in np.asarray(existing_new_shape).reshape(-1).tolist()
            ]
        except Exception:
            existing_new_shape_list = []
        if len(new_shape) == 0:
            # Runtime shape-driven RESHAPE (e.g. SHAPE/GATHER/CONCAT pipelines)
            # must remain dynamic. Do not force-resolve from empty templates.
            continue

        input_tensor = model_ir.tensors.get(op.inputs[0], None)
        if input_tensor is None:
            continue
        input_signature = (
            list(input_tensor.shape_signature)
            if input_tensor.shape_signature is not None
            else list(input_tensor.shape)
        )
        input_shape_fallback = (
            list(input_tensor.shape)
            if input_tensor.shape is not None
            else []
        )
        effective_input_signature: List[int] = []
        if len(input_signature) > 0:
            for idx, dim in enumerate(input_signature):
                sig_dim = int(dim)
                if sig_dim > 0:
                    effective_input_signature.append(int(sig_dim))
                    continue
                if idx < len(input_shape_fallback) and int(input_shape_fallback[idx]) > 0:
                    effective_input_signature.append(int(input_shape_fallback[idx]))
                else:
                    effective_input_signature.append(int(sig_dim))
        else:
            effective_input_signature = [int(v) for v in list(input_shape_fallback)]
        has_zero_dim = any(int(dim) == 0 for dim in new_shape)
        has_multi_minus_one = sum(1 for dim in new_shape if int(dim) == -1) > 1
        signature_for_resolve = (
            [int(v) for v in list(effective_input_signature)]
            if has_zero_dim or has_multi_minus_one
            else [int(v) for v in list(input_signature)]
        )
        if has_onnx_raw_new_shape and len(new_shape) > 0:
            # ONNX-exported raw reshape constants are semantically authoritative.
            # Do not re-infer them from possibly stale intermediate static metadata.
            # If a concrete newShape already exists, keep it as the most stable form.
            raw_has_minus_one = any(int(dim) == -1 for dim in new_shape)
            raw_has_zero_dim = any(int(dim) == 0 for dim in new_shape)
            if (
                bool(prefer_runtime_inferable_from_onnx_raw)
                and raw_has_minus_one
                and not raw_has_zero_dim
                and len(new_shape) >= 5
                and int(next(idx for idx, dim in enumerate(new_shape) if int(dim) == -1)) > 0
            ):
                # Final-stage safety mode:
                # keep ONNX's runtime-inferable `-1` instead of stale
                # concretized values. High-rank shape metadata can appear
                # fully static while still being a placeholder propagated
                # through dynamic Slice/MatMul/broadcast chains. TFLite
                # RESHAPE supports one `-1`, so the raw ONNX template is the
                # authoritative and safer final serialization contract.
                resolved_shape = _sanitize_reshape_template(
                    template=new_shape,
                    input_dims=signature_for_resolve,
                    allow_zero=(
                        bool(op.options.get("allowZero"))
                        if "allowZero" in op.options
                        else None
                    ),
                )
            elif (
                len(existing_new_shape_list) > 0
                and all(int(dim) > 0 for dim in existing_new_shape_list)
            ):
                resolved_shape = [int(v) for v in existing_new_shape_list]
            elif all(int(dim) > 0 for dim in new_shape):
                resolved_shape = [int(v) for v in new_shape]
            elif (
                any(int(dim) == -1 for dim in new_shape)
                and any(int(dim) <= 0 for dim in input_signature)
                and not any(int(dim) == 0 for dim in new_shape)
            ):
                # Keep runtime "-1" inference dimensions when input signature is dynamic.
                resolved_shape = [int(v) for v in new_shape]
            else:
                resolved_shape = _resolve_reshape_new_shape_from_static_input(
                    new_shape=_sanitize_reshape_template(
                        template=new_shape,
                        input_dims=signature_for_resolve,
                        allow_zero=(
                            bool(op.options.get("allowZero"))
                            if "allowZero" in op.options
                            else None
                        ),
                    ),
                    input_signature=signature_for_resolve,
                    allow_zero=(
                        bool(op.options.get("allowZero"))
                        if "allowZero" in op.options
                        else None
                    ),
                )
        else:
            if (
                any(int(dim) == -1 for dim in new_shape)
                and any(int(dim) <= 0 for dim in input_signature)
                and not any(int(dim) == 0 for dim in new_shape)
            ):
                # Keep runtime-inferable `-1` templates for shape-tensor driven
                # reshapes when the source rank is still dynamic. Concretizing
                # them from placeholder static metadata can freeze valid dynamic
                # extents to `1` and break runtime element counts.
                resolved_shape = [int(v) for v in new_shape]
            else:
                resolved_shape = _resolve_reshape_new_shape_from_static_input(
                    new_shape=_sanitize_reshape_template(
                        template=new_shape,
                        input_dims=signature_for_resolve,
                        allow_zero=(
                            bool(op.options.get("allowZero"))
                            if "allowZero" in op.options
                            else None
                        ),
                    ),
                    input_signature=signature_for_resolve,
                    allow_zero=(
                        bool(op.options.get("allowZero"))
                        if "allowZero" in op.options
                        else None
                    ),
                )
        if resolved_shape is None and len(new_shape) > 0:
            fallback_shape = _sanitize_reshape_template(
                template=new_shape,
                input_dims=signature_for_resolve,
                allow_zero=(
                    bool(op.options.get("allowZero"))
                    if "allowZero" in op.options
                    else None
                ),
            )
            if (
                sum(1 for dim in fallback_shape if int(dim) == -1) <= 1
                and all(int(dim) >= -1 for dim in fallback_shape)
                and not any(int(dim) == 0 for dim in fallback_shape)
            ):
                resolved_shape = [int(v) for v in list(fallback_shape)]
        if resolved_shape is None:
            continue
        if (
            has_onnx_raw_new_shape
            and len(new_shape) == len(resolved_shape)
            and any(int(dim) == 0 for dim in new_shape)
            and sum(1 for dim in new_shape if int(dim) == -1) == 1
            and any(int(dim) < 0 for dim in input_signature)
            and all(int(dim) > 0 for dim in resolved_shape)
        ):
            zero_axes = [int(i) for i, dim in enumerate(new_shape) if int(dim) == 0]
            inferred_axis = next(
                (int(i) for i, dim in enumerate(new_shape) if int(dim) == -1),
                None,
            )
            if inferred_axis is not None and len(zero_axes) > 0:
                dynamic_candidate_axes = [
                    int(axis) for axis in zero_axes if int(resolved_shape[axis]) > 1
                ]
                if len(dynamic_candidate_axes) == 0:
                    dynamic_candidate_axes = [int(v) for v in list(zero_axes)]
                dynamic_axis = int(dynamic_candidate_axes[0])
                if int(dynamic_axis) != int(inferred_axis):
                    relaxed_shape = [int(v) for v in list(resolved_shape)]
                    relaxed_shape[dynamic_axis] = -1
                    resolved_shape = [int(v) for v in list(relaxed_shape)]

        changed = False
        should_update_newshape_option = True
        if (
            len(existing_new_shape_list) == 0
            and any(int(dim) < 0 for dim in resolved_shape)
        ):
            # Keep runtime-driven dynamic reshape intent (newShape=[]) when the
            # resolved template still contains unknown dimensions.
            should_update_newshape_option = False
        if should_update_newshape_option:
            if existing_new_shape_list != [int(v) for v in resolved_shape]:
                op.options["newShape"] = [int(v) for v in resolved_shape]
                changed = True

        if len(op.inputs) >= 2:
            shape_tensor = model_ir.tensors.get(op.inputs[1], None)
            if shape_tensor is not None:
                existing_shape_data = None
                if shape_tensor.data is not None:
                    existing_shape_data = [
                        int(v) for v in np.asarray(shape_tensor.data).reshape(-1).tolist()
                    ]
                if existing_shape_data != [int(v) for v in resolved_shape]:
                    shape_tensor.data = np.asarray(resolved_shape, dtype=np.int32)
                    shape_tensor.dtype = "INT32"
                    shape_tensor.shape = [int(len(resolved_shape))]
                    shape_tensor.shape_signature = [int(len(resolved_shape))]
                    changed = True

        output_tensor = model_ir.tensors.get(op.outputs[0], None)
        if output_tensor is not None:
            output_shape = list(output_tensor.shape)
            output_signature = (
                list(output_tensor.shape_signature)
                if output_tensor.shape_signature is not None
                else None
            )
            materialized_shape = [int(v) if int(v) > 0 else 1 for v in resolved_shape]
            resolved_signature = [int(v) for v in resolved_shape]
            if output_shape != materialized_shape or output_signature != resolved_signature:
                output_tensor.shape = [int(v) for v in materialized_shape]
                output_tensor.shape_signature = [int(v) for v in resolved_shape]
                changed = True

        if changed:
            resolved_count += 1

    return {"resolved_dynamic_reshape_shapes": int(resolved_count)}
