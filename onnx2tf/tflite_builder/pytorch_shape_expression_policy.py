from __future__ import annotations

from typing import Callable, Dict, List, Optional, Set

from onnx2tf.tflite_builder.ir import ModelIR, OperatorIR
from onnx2tf.tflite_builder.pytorch_codegen_utils import (
    _constant_int_list,
    _product_expr,
)


def _shape_tensor_length_for_codegen(
    *,
    model_ir: ModelIR,
    tensor_name: str,
) -> Optional[int]:
    tensor = model_ir.tensors.get(str(tensor_name), None)
    if tensor is None:
        return None
    shape_values = [int(v) for v in list(tensor.shape)]
    if len(shape_values) == 0:
        return 0
    if len(shape_values) != 1 or int(shape_values[0]) < 0:
        return None
    return int(shape_values[0])


def _reconstruct_shape_scalar_expr_for_codegen(
    *,
    model_ir: ModelIR,
    producer_by_output_name: Dict[str, OperatorIR],
    tensor_exact_static_shape_list_fn: Callable[[str], Optional[List[int]]],
    tensor_expr_fn: Callable[[str], str],
    runtime_imports: Set[str],
    tensor_name: str,
    seen: Optional[Set[str]] = None,
) -> Optional[str]:
    current_name = str(tensor_name)
    if seen is None:
        seen = set()
    if current_name in seen:
        return None
    next_seen = set(seen)
    next_seen.add(current_name)

    tensor = model_ir.tensors.get(current_name, None)
    constant_values = _constant_int_list(tensor)
    if constant_values is not None and len(constant_values) == 1:
        return repr(int(constant_values[0]))

    producer = producer_by_output_name.get(current_name, None)
    if producer is None:
        return None
    op_type = str(producer.op_type)
    inputs = [str(v) for v in list(producer.inputs)]

    def _shape_dim_expr_from_shape_input(shape_tensor_name: str, dim_index: int) -> Optional[str]:
        shape_producer = producer_by_output_name.get(str(shape_tensor_name), None)
        if shape_producer is None:
            return None
        if str(shape_producer.op_type) != "SHAPE" or len(list(shape_producer.inputs)) < 1:
            return None
        source_name = str(shape_producer.inputs[0])
        exact_input_shape = tensor_exact_static_shape_list_fn(source_name)
        if exact_input_shape is not None and int(dim_index) < len(exact_input_shape):
            return repr(int(exact_input_shape[int(dim_index)]))
        return f"{tensor_expr_fn(source_name)}.shape[{int(dim_index)}]"

    if op_type in {"CAST", "IDENTITY", "RESHAPE", "SQUEEZE"} and len(inputs) >= 1:
        return _reconstruct_shape_scalar_expr_for_codegen(
            model_ir=model_ir,
            producer_by_output_name=producer_by_output_name,
            tensor_exact_static_shape_list_fn=tensor_exact_static_shape_list_fn,
            tensor_expr_fn=tensor_expr_fn,
            runtime_imports=runtime_imports,
            tensor_name=inputs[0],
            seen=next_seen,
        )
    if op_type == "SLICE" and len(inputs) >= 3:
        base_expr = _reconstruct_shape_list_expr_for_codegen(
            model_ir=model_ir,
            producer_by_output_name=producer_by_output_name,
            tensor_exact_static_shape_list_fn=tensor_exact_static_shape_list_fn,
            tensor_expr_fn=tensor_expr_fn,
            runtime_imports=runtime_imports,
            tensor_name=inputs[0],
            seen=next_seen,
        )
        begin_values = _constant_int_list(model_ir.tensors.get(inputs[1], None))
        size_values = _constant_int_list(model_ir.tensors.get(inputs[2], None))
        output_len = _shape_tensor_length_for_codegen(model_ir=model_ir, tensor_name=current_name)
        if base_expr is not None and begin_values is not None and len(begin_values) == 1 and size_values is not None and len(size_values) == 1 and output_len == 1:
            shape_dim_expr = _shape_dim_expr_from_shape_input(inputs[0], int(begin_values[0]))
            if shape_dim_expr is not None:
                return shape_dim_expr
            return f"({base_expr})[{int(begin_values[0])}]"
    if op_type == "STRIDED_SLICE" and len(inputs) >= 4:
        base_expr = _reconstruct_shape_list_expr_for_codegen(
            model_ir=model_ir,
            producer_by_output_name=producer_by_output_name,
            tensor_exact_static_shape_list_fn=tensor_exact_static_shape_list_fn,
            tensor_expr_fn=tensor_expr_fn,
            runtime_imports=runtime_imports,
            tensor_name=inputs[0],
            seen=next_seen,
        )
        begin_values = _constant_int_list(model_ir.tensors.get(inputs[1], None))
        stride_values = _constant_int_list(model_ir.tensors.get(inputs[3], None))
        output_len = _shape_tensor_length_for_codegen(model_ir=model_ir, tensor_name=current_name)
        if base_expr is not None and begin_values is not None and len(begin_values) == 1 and stride_values is not None and len(stride_values) == 1 and output_len == 1:
            shape_dim_expr = _shape_dim_expr_from_shape_input(inputs[0], int(begin_values[0]))
            if shape_dim_expr is not None:
                return shape_dim_expr
            return f"({base_expr})[{int(begin_values[0])}]"
    if op_type == "GATHER" and len(inputs) >= 2 and int(producer.options.get("axis", 0)) == 0:
        base_expr = _reconstruct_shape_list_expr_for_codegen(
            model_ir=model_ir,
            producer_by_output_name=producer_by_output_name,
            tensor_exact_static_shape_list_fn=tensor_exact_static_shape_list_fn,
            tensor_expr_fn=tensor_expr_fn,
            runtime_imports=runtime_imports,
            tensor_name=inputs[0],
            seen=next_seen,
        )
        index_values = _constant_int_list(model_ir.tensors.get(inputs[1], None))
        if base_expr is not None and index_values is not None and len(index_values) == 1:
            shape_dim_expr = _shape_dim_expr_from_shape_input(inputs[0], int(index_values[0]))
            if shape_dim_expr is not None:
                return shape_dim_expr
            return f"({base_expr})[{int(index_values[0])}]"
    if op_type == "EQUAL" and len(inputs) >= 2:
        lhs_expr = _reconstruct_shape_scalar_expr_for_codegen(
            model_ir=model_ir,
            producer_by_output_name=producer_by_output_name,
            tensor_exact_static_shape_list_fn=tensor_exact_static_shape_list_fn,
            tensor_expr_fn=tensor_expr_fn,
            runtime_imports=runtime_imports,
            tensor_name=inputs[0],
            seen=next_seen,
        )
        rhs_expr = _reconstruct_shape_scalar_expr_for_codegen(
            model_ir=model_ir,
            producer_by_output_name=producer_by_output_name,
            tensor_exact_static_shape_list_fn=tensor_exact_static_shape_list_fn,
            tensor_expr_fn=tensor_expr_fn,
            runtime_imports=runtime_imports,
            tensor_name=inputs[1],
            seen=next_seen,
        )
        if lhs_expr is not None and rhs_expr is not None:
            return f"({lhs_expr} == {rhs_expr})"
    if op_type == "NOT_EQUAL" and len(inputs) >= 2:
        lhs_expr = _reconstruct_shape_scalar_expr_for_codegen(
            model_ir=model_ir,
            producer_by_output_name=producer_by_output_name,
            tensor_exact_static_shape_list_fn=tensor_exact_static_shape_list_fn,
            tensor_expr_fn=tensor_expr_fn,
            runtime_imports=runtime_imports,
            tensor_name=inputs[0],
            seen=next_seen,
        )
        rhs_expr = _reconstruct_shape_scalar_expr_for_codegen(
            model_ir=model_ir,
            producer_by_output_name=producer_by_output_name,
            tensor_exact_static_shape_list_fn=tensor_exact_static_shape_list_fn,
            tensor_expr_fn=tensor_expr_fn,
            runtime_imports=runtime_imports,
            tensor_name=inputs[1],
            seen=next_seen,
        )
        if lhs_expr is not None and rhs_expr is not None:
            return f"({lhs_expr} != {rhs_expr})"
    if op_type in {"MAXIMUM", "MINIMUM"} and len(inputs) >= 2:
        lhs_expr = _reconstruct_shape_scalar_expr_for_codegen(
            model_ir=model_ir,
            producer_by_output_name=producer_by_output_name,
            tensor_exact_static_shape_list_fn=tensor_exact_static_shape_list_fn,
            tensor_expr_fn=tensor_expr_fn,
            runtime_imports=runtime_imports,
            tensor_name=inputs[0],
            seen=next_seen,
        )
        rhs_expr = _reconstruct_shape_scalar_expr_for_codegen(
            model_ir=model_ir,
            producer_by_output_name=producer_by_output_name,
            tensor_exact_static_shape_list_fn=tensor_exact_static_shape_list_fn,
            tensor_expr_fn=tensor_expr_fn,
            runtime_imports=runtime_imports,
            tensor_name=inputs[1],
            seen=next_seen,
        )
        if lhs_expr is not None and rhs_expr is not None:
            fn_name = "max" if op_type == "MAXIMUM" else "min"
            return f"{fn_name}({lhs_expr}, {rhs_expr})"
    if op_type in {"ADD", "SUB", "MUL"} and len(inputs) >= 2:
        lhs_expr = _reconstruct_shape_scalar_expr_for_codegen(
            model_ir=model_ir,
            producer_by_output_name=producer_by_output_name,
            tensor_exact_static_shape_list_fn=tensor_exact_static_shape_list_fn,
            tensor_expr_fn=tensor_expr_fn,
            runtime_imports=runtime_imports,
            tensor_name=inputs[0],
            seen=next_seen,
        )
        rhs_expr = _reconstruct_shape_scalar_expr_for_codegen(
            model_ir=model_ir,
            producer_by_output_name=producer_by_output_name,
            tensor_exact_static_shape_list_fn=tensor_exact_static_shape_list_fn,
            tensor_expr_fn=tensor_expr_fn,
            runtime_imports=runtime_imports,
            tensor_name=inputs[1],
            seen=next_seen,
        )
        if lhs_expr is not None and rhs_expr is not None:
            op_symbol = "+" if op_type == "ADD" else "-" if op_type == "SUB" else "*"
            return f"({lhs_expr} {op_symbol} {rhs_expr})"
    if op_type == "SELECT" and len(inputs) >= 3:
        return None
    if op_type == "REDUCE_PROD" and len(inputs) >= 1:
        base_expr = _reconstruct_shape_list_expr_for_codegen(
            model_ir=model_ir,
            producer_by_output_name=producer_by_output_name,
            tensor_exact_static_shape_list_fn=tensor_exact_static_shape_list_fn,
            tensor_expr_fn=tensor_expr_fn,
            runtime_imports=runtime_imports,
            tensor_name=inputs[0],
            seen=next_seen,
        )
        axes_values = _constant_int_list(model_ir.tensors.get(inputs[1], None)) if len(inputs) >= 2 else None
        output_len = _shape_tensor_length_for_codegen(model_ir=model_ir, tensor_name=current_name)
        input_len = _shape_tensor_length_for_codegen(model_ir=model_ir, tensor_name=inputs[0])
        if (
            base_expr is not None
            and axes_values == [0]
            and output_len == 1
            and input_len is not None
            and input_len >= 0
        ):
            return _product_expr([f"({base_expr})[{index}]" for index in range(int(input_len))])
    return None


def _reconstruct_shape_list_expr_for_codegen(
    *,
    model_ir: ModelIR,
    producer_by_output_name: Dict[str, OperatorIR],
    tensor_exact_static_shape_list_fn: Callable[[str], Optional[List[int]]],
    tensor_expr_fn: Callable[[str], str],
    runtime_imports: Set[str],
    tensor_name: str,
    seen: Optional[Set[str]] = None,
) -> Optional[str]:
    current_name = str(tensor_name)
    if seen is None:
        seen = set()
    if current_name in seen:
        return None
    next_seen = set(seen)
    next_seen.add(current_name)

    tensor = model_ir.tensors.get(current_name, None)
    constant_values = _constant_int_list(tensor)
    if constant_values is not None:
        return repr([int(v) for v in list(constant_values)])

    producer = producer_by_output_name.get(current_name, None)
    if producer is None:
        return None
    op_type = str(producer.op_type)
    inputs = [str(v) for v in list(producer.inputs)]
    if op_type == "CAST" and len(inputs) >= 1:
        input_expr = _reconstruct_shape_list_expr_for_codegen(
            model_ir=model_ir,
            producer_by_output_name=producer_by_output_name,
            tensor_exact_static_shape_list_fn=tensor_exact_static_shape_list_fn,
            tensor_expr_fn=tensor_expr_fn,
            runtime_imports=runtime_imports,
            tensor_name=inputs[0],
            seen=next_seen,
        )
        if input_expr is not None:
            return input_expr
        input_scalar_expr = _reconstruct_shape_scalar_expr_for_codegen(
            model_ir=model_ir,
            producer_by_output_name=producer_by_output_name,
            tensor_exact_static_shape_list_fn=tensor_exact_static_shape_list_fn,
            tensor_expr_fn=tensor_expr_fn,
            runtime_imports=runtime_imports,
            tensor_name=inputs[0],
            seen=next_seen,
        )
        if input_scalar_expr is not None:
            return f"[{input_scalar_expr}]"
        return None
    if op_type == "IDENTITY" and len(inputs) >= 1:
        return _reconstruct_shape_list_expr_for_codegen(
            model_ir=model_ir,
            producer_by_output_name=producer_by_output_name,
            tensor_exact_static_shape_list_fn=tensor_exact_static_shape_list_fn,
            tensor_expr_fn=tensor_expr_fn,
            runtime_imports=runtime_imports,
            tensor_name=inputs[0],
            seen=next_seen,
        )
    if op_type == "RESHAPE" and len(inputs) >= 2:
        target_expr = _reconstruct_shape_list_expr_for_codegen(
            model_ir=model_ir,
            producer_by_output_name=producer_by_output_name,
            tensor_exact_static_shape_list_fn=tensor_exact_static_shape_list_fn,
            tensor_expr_fn=tensor_expr_fn,
            runtime_imports=runtime_imports,
            tensor_name=inputs[1],
            seen=next_seen,
        )
        if target_expr is not None:
            return target_expr
        target_scalar_expr = _reconstruct_shape_scalar_expr_for_codegen(
            model_ir=model_ir,
            producer_by_output_name=producer_by_output_name,
            tensor_exact_static_shape_list_fn=tensor_exact_static_shape_list_fn,
            tensor_expr_fn=tensor_expr_fn,
            runtime_imports=runtime_imports,
            tensor_name=inputs[1],
            seen=next_seen,
        )
        if target_scalar_expr is not None:
            return f"[{target_scalar_expr}]"
        return None
    if op_type == "SHAPE" and len(inputs) >= 1:
        exact_input_shape = tensor_exact_static_shape_list_fn(inputs[0])
        if exact_input_shape is not None:
            return repr([int(v) for v in list(exact_input_shape)])
        tensor_expr = tensor_expr_fn(inputs[0])
        runtime_imports.add("_tensor_shape_list")
        return f"_tensor_shape_list({tensor_expr})"
    if op_type == "SLICE" and len(inputs) >= 3:
        base_expr = _reconstruct_shape_list_expr_for_codegen(
            model_ir=model_ir,
            producer_by_output_name=producer_by_output_name,
            tensor_exact_static_shape_list_fn=tensor_exact_static_shape_list_fn,
            tensor_expr_fn=tensor_expr_fn,
            runtime_imports=runtime_imports,
            tensor_name=inputs[0],
            seen=next_seen,
        )
        begin_values = _constant_int_list(model_ir.tensors.get(inputs[1], None))
        size_values = _constant_int_list(model_ir.tensors.get(inputs[2], None))
        if base_expr is not None and begin_values is not None and len(begin_values) == 1 and size_values is not None and len(size_values) == 1:
            start = int(begin_values[0])
            length = int(size_values[0])
            stop_expr = "" if length < 0 else str(start + length)
            return f"({base_expr})[{start}:{stop_expr}]"
    if op_type == "STRIDED_SLICE" and len(inputs) >= 4:
        base_expr = _reconstruct_shape_list_expr_for_codegen(
            model_ir=model_ir,
            producer_by_output_name=producer_by_output_name,
            tensor_exact_static_shape_list_fn=tensor_exact_static_shape_list_fn,
            tensor_expr_fn=tensor_expr_fn,
            runtime_imports=runtime_imports,
            tensor_name=inputs[0],
            seen=next_seen,
        )
        begin_values = _constant_int_list(model_ir.tensors.get(inputs[1], None))
        end_values = _constant_int_list(model_ir.tensors.get(inputs[2], None))
        stride_values = _constant_int_list(model_ir.tensors.get(inputs[3], None))
        if (
            base_expr is not None
            and begin_values is not None and len(begin_values) == 1
            and end_values is not None and len(end_values) == 1
            and stride_values is not None and len(stride_values) == 1
        ):
            start = int(begin_values[0])
            end_value = int(end_values[0])
            step = int(stride_values[0])
            stop_expr = "" if bool(producer.options.get("endMask", 0)) else str(end_value)
            if step == 1:
                return f"({base_expr})[{start}:{stop_expr}]"
            return f"({base_expr})[{start}:{stop_expr}:{step}]"
    if op_type == "GATHER" and len(inputs) >= 2 and int(producer.options.get("axis", 0)) == 0:
        base_expr = _reconstruct_shape_list_expr_for_codegen(
            model_ir=model_ir,
            producer_by_output_name=producer_by_output_name,
            tensor_exact_static_shape_list_fn=tensor_exact_static_shape_list_fn,
            tensor_expr_fn=tensor_expr_fn,
            runtime_imports=runtime_imports,
            tensor_name=inputs[0],
            seen=next_seen,
        )
        index_values = _constant_int_list(model_ir.tensors.get(inputs[1], None))
        if base_expr is not None and index_values is not None:
            parts = ", ".join(f"({base_expr})[{int(index)}]" for index in index_values)
            return f"[{parts}]"
    if op_type == "CONCATENATION":
        part_exprs: List[str] = []
        for input_name in inputs:
            input_expr = _reconstruct_shape_list_expr_for_codegen(
                model_ir=model_ir,
                producer_by_output_name=producer_by_output_name,
                tensor_exact_static_shape_list_fn=tensor_exact_static_shape_list_fn,
                tensor_expr_fn=tensor_expr_fn,
                runtime_imports=runtime_imports,
                tensor_name=input_name,
                seen=next_seen,
            )
            if input_expr is None:
                input_scalar_expr = _reconstruct_shape_scalar_expr_for_codegen(
                    model_ir=model_ir,
                    producer_by_output_name=producer_by_output_name,
                    tensor_exact_static_shape_list_fn=tensor_exact_static_shape_list_fn,
                    tensor_expr_fn=tensor_expr_fn,
                    runtime_imports=runtime_imports,
                    tensor_name=input_name,
                    seen=next_seen,
                )
                if input_scalar_expr is None:
                    return None
                input_expr = f"[{input_scalar_expr}]"
            part_exprs.append(f"({input_expr})")
        if len(part_exprs) == 0:
            return "[]"
        combined_expr = part_exprs[0]
        for part_expr in part_exprs[1:]:
            combined_expr = f"({combined_expr} + {part_expr})"
        return combined_expr
    if op_type in {"SELECT", "REDUCE_PROD"}:
        scalar_expr = _reconstruct_shape_scalar_expr_for_codegen(
            model_ir=model_ir,
            producer_by_output_name=producer_by_output_name,
            tensor_exact_static_shape_list_fn=tensor_exact_static_shape_list_fn,
            tensor_expr_fn=tensor_expr_fn,
            runtime_imports=runtime_imports,
            tensor_name=current_name,
            seen=next_seen,
        )
        if scalar_expr is not None:
            return f"[{scalar_expr}]"
    return None
