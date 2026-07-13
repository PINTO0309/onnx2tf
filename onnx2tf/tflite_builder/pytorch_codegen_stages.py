from __future__ import annotations

import ast
import re
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple, cast

from onnx2tf.tflite_builder.ir import ModelIR
from onnx2tf.tflite_builder.pytorch_codegen_utils import (
    _extract_statement_assignments,
    _extract_statement_loads,
)
from onnx2tf.tflite_builder.pytorch_layout_utils import (
    _can_emit_direct_torch_reshape_shape,
    _preferred_reshape_target_values,
)


def _fold_single_use_static_reshape_chains(
    lines: Sequence[str],
    *,
    tensor_var_names: Dict[str, str],
    model_ir: ModelIR,
) -> List[str]:
    rewritten = [str(line) for line in lines]
    if len(rewritten) < 2:
        return rewritten

    def _reshape_call(value: ast.AST) -> Optional[Tuple[ast.expr, ast.expr]]:
        if not isinstance(value, ast.Call):
            return None
        if len(value.keywords) != 0 or len(value.args) < 2:
            return None
        func = value.func
        if not (
            isinstance(func, ast.Attribute)
            and func.attr == "reshape"
            and isinstance(func.value, ast.Name)
            and func.value.id == "torch"
        ):
            return None
        return cast(ast.expr, value.args[0]), cast(ast.expr, value.args[1])

    def _static_shape_from_expr(
        shape_expr: ast.AST,
        *,
        input_name: str,
        input_shape: Sequence[int],
    ) -> Optional[List[int]]:
        try:
            literal_value = ast.literal_eval(shape_expr)
        except Exception:
            literal_value = None
        if isinstance(literal_value, (list, tuple)):
            values = [int(v) for v in list(literal_value)]
            if len(values) > 0 and all(int(v) > 0 for v in values):
                return values
        if not isinstance(shape_expr, ast.Call):
            return None
        if not (
            isinstance(shape_expr.func, ast.Name)
            and str(shape_expr.func.id) == "_resolve_reshape_shape"
            and len(shape_expr.args) >= 2
            and isinstance(shape_expr.args[1], ast.Name)
            and str(shape_expr.args[1].id) == input_name
        ):
            return None
        try:
            raw_new_shape = ast.literal_eval(shape_expr.args[0])
        except Exception:
            return None
        if not isinstance(raw_new_shape, (list, tuple)):
            return None
        raw_values = [int(v) for v in list(raw_new_shape)]
        if len(raw_values) == 0:
            return None
        unknown_index: Optional[int] = None
        known_product = 1
        for index, dim in enumerate(raw_values):
            if int(dim) == -1:
                if unknown_index is not None:
                    return None
                unknown_index = int(index)
                continue
            if int(dim) <= 0:
                return None
            known_product *= int(dim)
        input_product = 1
        for dim in list(input_shape):
            if int(dim) <= 0:
                return None
            input_product *= int(dim)
        resolved = [int(v) for v in raw_values]
        if unknown_index is not None:
            if known_product <= 0 or input_product % known_product != 0:
                return None
            resolved[int(unknown_index)] = int(input_product // known_product)
        if not all(int(v) > 0 for v in resolved):
            return None
        return resolved

    changed = True
    while changed:
        changed = False
        parsed_statements = [ast.parse(line).body[0] for line in rewritten]
        used_names_by_line = [
            _extract_statement_loads(statement) for statement in parsed_statements
        ]

        for producer_index_in_lines, statement in enumerate(parsed_statements):
            if not isinstance(statement, ast.Assign) or len(statement.targets) != 1:
                continue
            producer_target = statement.targets[0]
            if not isinstance(producer_target, ast.Name):
                continue
            producer_name = str(producer_target.id)
            producer_reshape = _reshape_call(statement.value)
            if producer_reshape is None:
                continue
            producer_input_expr, producer_shape_expr = producer_reshape
            producer_static_shape = _static_shape_from_expr(
                producer_shape_expr,
                input_name=producer_name,
                input_shape=[],
            )
            if producer_static_shape is None:
                continue
            later_use_lines = [
                int(line_index)
                for line_index in range(
                    int(producer_index_in_lines) + 1, len(used_names_by_line)
                )
                if producer_name in {str(v) for v in used_names_by_line[line_index]}
            ]
            if len(later_use_lines) != 1:
                continue
            consumer_index_in_lines = int(later_use_lines[0])
            if int(consumer_index_in_lines) <= int(producer_index_in_lines):
                continue
            consumer_statement = parsed_statements[int(consumer_index_in_lines)]
            if (
                not isinstance(consumer_statement, ast.Assign)
                or len(consumer_statement.targets) != 1
            ):
                continue
            consumer_target = consumer_statement.targets[0]
            if not isinstance(consumer_target, ast.Name):
                continue
            consumer_name = str(consumer_target.id)
            consumer_reshape = _reshape_call(consumer_statement.value)
            if consumer_reshape is None:
                continue
            consumer_input_expr, consumer_shape_expr = consumer_reshape
            if not (
                isinstance(consumer_input_expr, ast.Name)
                and str(consumer_input_expr.id) == producer_name
            ):
                continue
            final_static_shape = _static_shape_from_expr(
                consumer_shape_expr,
                input_name=producer_name,
                input_shape=producer_static_shape,
            )
            if final_static_shape is None:
                continue
            if not _can_emit_direct_torch_reshape_shape(
                final_static_shape, allow_zero=False
            ):
                continue
            rewritten[producer_index_in_lines] = (
                f"{consumer_name} = torch.reshape({ast.unparse(producer_input_expr)}, {repr(final_static_shape)})"
            )
            del rewritten[int(consumer_index_in_lines)]
            changed = True
            break
    return rewritten


def _build_forward_stage_methods(
    lines: Sequence[str],
    *,
    tensor_var_names: Dict[str, str],
    model_ir: ModelIR,
) -> Tuple[str, List[str], List[Dict[str, Any]]]:
    if len(lines) < 80:
        return "", [f"        {line}" for line in lines], []

    parsed_statements: List[ast.stmt] = []
    top_level_assigned_names: List[List[str]] = []
    raw_used_names: List[List[str]] = []
    for line in lines:
        statement = ast.parse(str(line)).body[0]
        parsed_statements.append(statement)
        top_level_assigned_names.append(_extract_statement_assignments(statement))
        raw_used_names.append(_extract_statement_loads(statement))

    local_name_candidates: Set[str] = {
        str(tensor_var_names[str(name)]) for name in model_ir.inputs
    }
    local_name_candidates.update(
        str(tensor_var_names[str(name)]) for name in model_ir.outputs
    )
    for assigned_names in top_level_assigned_names:
        local_name_candidates.update(str(name) for name in assigned_names)

    assigned_names_by_line: List[List[str]] = []
    used_names_by_line: List[List[str]] = []
    for assigned_names, used_names in zip(top_level_assigned_names, raw_used_names):
        assigned_filtered = [
            str(name) for name in assigned_names if str(name) in local_name_candidates
        ]
        used_filtered = [
            str(name) for name in used_names if str(name) in local_name_candidates
        ]
        assigned_names_by_line.append(assigned_filtered)
        used_names_by_line.append(used_filtered)

    tensor_name_by_var_name = {
        str(var_name): str(tensor_name)
        for tensor_name, var_name in tensor_var_names.items()
    }

    def _reshape_call_from_statement(statement: ast.stmt) -> Optional[Tuple[str, str]]:
        if not isinstance(statement, ast.Assign) or len(statement.targets) != 1:
            return None
        target = statement.targets[0]
        if not isinstance(target, ast.Name):
            return None
        value = statement.value
        if (
            not isinstance(value, ast.Call)
            or len(value.keywords) != 0
            or len(value.args) < 2
        ):
            return None
        func = value.func
        if not (
            isinstance(func, ast.Attribute)
            and func.attr == "reshape"
            and isinstance(func.value, ast.Name)
            and func.value.id == "torch"
        ):
            return None
        input_expr = value.args[0]
        if not isinstance(input_expr, ast.Name):
            return None
        return str(target.id), str(input_expr.id)

    total_lines = len(lines)

    def _is_adjacent_single_use_static_reshape_chain_boundary(line_index: int) -> bool:
        if int(line_index) < 0 or int(line_index) + 1 >= total_lines:
            return False
        producer_pair = _reshape_call_from_statement(parsed_statements[int(line_index)])
        consumer_pair = _reshape_call_from_statement(
            parsed_statements[int(line_index) + 1]
        )
        if producer_pair is None or consumer_pair is None:
            return False
        producer_name, _ = producer_pair
        consumer_name, consumer_input_name = consumer_pair
        if consumer_input_name != producer_name:
            return False
        later_use_lines = [
            int(candidate_index)
            for candidate_index in range(int(line_index) + 1, total_lines)
            if producer_name in {str(v) for v in used_names_by_line[candidate_index]}
        ]
        if later_use_lines != [int(line_index) + 1]:
            return False
        consumer_tensor_name = tensor_name_by_var_name.get(consumer_name, None)
        if consumer_tensor_name is None:
            return False
        consumer_tensor = model_ir.tensors.get(str(consumer_tensor_name), None)
        if consumer_tensor is None:
            return False
        preferred_shape = _preferred_reshape_target_values(consumer_tensor)
        if preferred_shape is None:
            preferred_shape = [int(v) for v in list(consumer_tensor.shape)]
        return (
            len(preferred_shape) > 0
            and all(int(v) > 0 for v in list(preferred_shape))
            and _can_emit_direct_torch_reshape_shape(preferred_shape, allow_zero=False)
        )

    final_output_names = {str(tensor_var_names[str(name)]) for name in model_ir.outputs}
    stage_min_lines = 18
    stage_target_lines = 28
    stage_max_lines = 36
    stage_methods: List[str] = []
    forward_stage_calls: List[str] = []
    stage_specs: List[Dict[str, Any]] = []
    stage_index = 0
    start_index = 0

    def _chunk_io(start: int, end: int) -> Tuple[List[str], List[str]]:
        defined: Set[str] = set()
        assigned_order: List[str] = []
        inputs: List[str] = []
        seen_inputs: Set[str] = set()
        for line_index in range(start, end + 1):
            for name in used_names_by_line[line_index]:
                if name not in defined and name not in seen_inputs:
                    seen_inputs.add(name)
                    inputs.append(name)
            for name in assigned_names_by_line[line_index]:
                if name not in defined:
                    defined.add(name)
                    assigned_order.append(name)
        later_needed: Set[str] = set(final_output_names)
        for line_index in range(end + 1, total_lines):
            later_needed.update(used_names_by_line[line_index])
        outputs = [name for name in assigned_order if name in later_needed]
        return inputs, outputs

    def _chunk_io_for_stage_lines(
        stage_lines_local: Sequence[str], end_index: int
    ) -> Tuple[List[str], List[str]]:
        defined: Set[str] = set()
        assigned_order: List[str] = []
        inputs: List[str] = []
        seen_inputs: Set[str] = set()
        stage_statements = [ast.parse(line).body[0] for line in stage_lines_local]
        for statement in stage_statements:
            for name in [
                str(v)
                for v in _extract_statement_loads(statement)
                if str(v) in local_name_candidates
            ]:
                if name not in defined and name not in seen_inputs:
                    seen_inputs.add(name)
                    inputs.append(name)
            for name in [
                str(v)
                for v in _extract_statement_assignments(statement)
                if str(v) in local_name_candidates
            ]:
                if name not in defined:
                    defined.add(name)
                    assigned_order.append(name)
        later_needed: Set[str] = set(final_output_names)
        for line_index in range(end_index + 1, total_lines):
            later_needed.update(used_names_by_line[line_index])
        outputs = [name for name in assigned_order if name in later_needed]
        return inputs, outputs

    def _append_stage(start: int, end: int) -> None:
        nonlocal stage_index
        stage_lines = _fold_single_use_static_reshape_chains(
            lines[start : end + 1],
            tensor_var_names=tensor_var_names,
            model_ir=model_ir,
        )
        stage_inputs, stage_outputs = _chunk_io_for_stage_lines(stage_lines, end)
        if len(stage_outputs) == 0:
            forward_stage_calls.extend(f"        {line}" for line in stage_lines)
            return
        method_name = f"_forward_stage_{stage_index}"
        arg_list = ", ".join(f"{name}: torch.Tensor" for name in stage_inputs)
        signature = (
            f"    def {method_name}(self, {arg_list})"
            if len(arg_list) > 0
            else f"    def {method_name}(self)"
        )
        if len(stage_outputs) == 1:
            signature += " -> torch.Tensor:\n"
        else:
            signature += (
                " -> tuple[" + ", ".join("torch.Tensor" for _ in stage_outputs) + "]:\n"
            )
        stage_body = "\n".join(f"        {line}" for line in stage_lines)
        if len(stage_outputs) == 1:
            stage_return = f"        return {stage_outputs[0]}\n"
        else:
            stage_return = f"        return ({', '.join(stage_outputs)})\n"
        stage_methods.append(f"{signature}{stage_body}\n{stage_return}")

        call_args = ", ".join(stage_inputs)
        call_expr = (
            f"self.{method_name}({call_args})"
            if len(call_args) > 0
            else f"self.{method_name}()"
        )
        if len(stage_outputs) == 1:
            forward_stage_calls.append(f"        {stage_outputs[0]} = {call_expr}")
        else:
            forward_stage_calls.append(
                f"        {', '.join(stage_outputs)} = {call_expr}"
            )
        stage_specs.append(
            {
                "stage_index": int(stage_index),
                "method_name": str(method_name),
                "inputs": list(stage_inputs),
                "outputs": list(stage_outputs),
            }
        )
        stage_index += 1

    while start_index < total_lines:
        remaining = total_lines - start_index
        if remaining < 80 or remaining <= stage_max_lines:
            _append_stage(start_index, total_lines - 1)
            break
        candidate_min_end = start_index + stage_min_lines - 1
        candidate_max_end = min(
            start_index + stage_max_lines - 1, total_lines - stage_min_lines - 1
        )
        if candidate_min_end > candidate_max_end:
            _append_stage(start_index, total_lines - 1)
            break
        best_candidate: Optional[
            Tuple[int, List[str], List[str], Tuple[int, int, int]]
        ] = None
        for end_index in range(candidate_min_end, candidate_max_end + 1):
            if _is_adjacent_single_use_static_reshape_chain_boundary(end_index):
                continue
            inputs, outputs = _chunk_io(start_index, end_index)
            if len(outputs) == 0:
                continue
            score = (
                len(inputs) + len(outputs),
                abs((end_index - start_index + 1) - stage_target_lines),
                len(outputs),
            )
            if best_candidate is None or score < best_candidate[3]:
                best_candidate = (end_index, inputs, outputs, score)
        if best_candidate is None:
            _append_stage(start_index, total_lines - 1)
            break
        end_index, _, _, _ = best_candidate
        _append_stage(start_index, end_index)
        start_index = end_index + 1

    stage_methods_source = "\n".join(stage_methods)
    if len(stage_methods_source) > 0:
        stage_methods_source += "\n"
    return stage_methods_source, forward_stage_calls, stage_specs


def _build_named_encoder_methods_composite(
    stage_specs: Sequence[Dict[str, Any]],
    *,
    final_output_names: Set[str],
) -> Tuple[str, List[str], List[str]]:
    if len(stage_specs) == 0:
        return "", [], []

    def _call_line_from_spec(spec: Dict[str, Any]) -> str:
        outputs = [str(name) for name in list(spec.get("outputs", []))]
        call_args = ", ".join(str(name) for name in list(spec.get("inputs", [])))
        call_expr = (
            f"self.{spec['method_name']}({call_args})"
            if len(call_args) > 0
            else f"self.{spec['method_name']}()"
        )
        if len(outputs) == 1:
            return f"        {outputs[0]} = {call_expr}"
        return f"        {', '.join(outputs)} = {call_expr}"

    layer_pattern = re.compile(r"bert_encoder_layer_(\d+)", flags=re.IGNORECASE)
    default_forward_lines = [_call_line_from_spec(spec) for spec in stage_specs]
    if not any(
        any(
            layer_pattern.search(str(name)) is not None
            and "attention_self_mul_1" not in str(name)
            for name in list(spec.get("outputs", []))
        )
        for spec in stage_specs
    ):
        return "", [], default_forward_lines

    def _stage_layer_index(spec: Dict[str, Any]) -> Optional[int]:
        matches: List[int] = []
        for name in list(spec.get("outputs", [])):
            if "attention_self_mul_1" in str(name):
                continue
            match = layer_pattern.search(str(name))
            if match is not None:
                matches.append(int(match.group(1)))
        if len(matches) == 0:
            return None
        return min(matches)

    grouped_ranges: List[Tuple[int, int, int]] = []
    start_spec_index: Optional[int] = None
    active_layer_index: Optional[int] = None
    for spec_index, spec in enumerate(stage_specs):
        layer_index = _stage_layer_index(spec)
        if layer_index is None:
            if start_spec_index is not None and active_layer_index is not None:
                grouped_ranges.append(
                    (
                        int(active_layer_index),
                        int(start_spec_index),
                        int(spec_index - 1),
                    )
                )
                start_spec_index = None
                active_layer_index = None
            continue
        if start_spec_index is None:
            start_spec_index = int(spec_index)
            active_layer_index = int(layer_index)
            continue
        if active_layer_index is None:
            active_layer_index = int(layer_index)
            continue
        if int(layer_index) != int(active_layer_index):
            grouped_ranges.append(
                (int(active_layer_index), int(start_spec_index), int(spec_index - 1))
            )
            start_spec_index = int(spec_index)
            active_layer_index = int(layer_index)
    if start_spec_index is not None and active_layer_index is not None:
        grouped_ranges.append(
            (int(active_layer_index), int(start_spec_index), int(len(stage_specs) - 1))
        )

    if len(grouped_ranges) == 0:
        return "", [], default_forward_lines

    class_chunks: List[str] = []
    init_lines: List[str] = []
    forward_lines_local: List[str] = []
    previous_end = 0

    def _group_io(start_idx: int, end_idx: int) -> Tuple[List[str], List[str]]:
        defined: Set[str] = set()
        assigned_order: List[str] = []
        seen_inputs: Set[str] = set()
        method_inputs: List[str] = []
        for spec_index in range(start_idx, end_idx + 1):
            spec = stage_specs[spec_index]
            for name in list(spec.get("inputs", [])):
                normalized = str(name)
                if normalized not in defined and normalized not in seen_inputs:
                    seen_inputs.add(normalized)
                    method_inputs.append(normalized)
            for name in list(spec.get("outputs", [])):
                normalized = str(name)
                if normalized not in defined:
                    defined.add(normalized)
                    assigned_order.append(normalized)
        later_needed = set(final_output_names)
        for spec_index in range(end_idx + 1, len(stage_specs)):
            later_needed.update(
                str(name) for name in list(stage_specs[spec_index].get("inputs", []))
            )
        method_outputs = [name for name in assigned_order if name in later_needed]
        return method_inputs, method_outputs

    def _emit_submodule_class_from_stage_range(
        *,
        class_name: str,
        start_idx: int,
        end_idx: int,
    ) -> Optional[Tuple[str, List[str], List[str], List[str]]]:
        method_inputs, method_outputs = _group_io(int(start_idx), int(end_idx))
        if len(method_outputs) == 0:
            return None
        class_stage_specs = list(stage_specs[start_idx : end_idx + 1])
        init_signature_lines = [
            f"class {class_name}(torch.nn.Module):",
            "    def __init__(",
            "        self,",
            "        *,",
        ]
        init_body_lines = ["        super().__init__()"]
        for spec in class_stage_specs:
            stage_name = str(spec["method_name"])
            init_signature_lines.append(f"        {stage_name}: Callable[..., Any],")
            init_body_lines.append(f"        self.{stage_name} = {stage_name}")
        init_signature_lines.append("    ) -> None:")
        arg_list = ", ".join(f"{name}: torch.Tensor" for name in method_inputs)
        signature = "    def forward(self"
        if len(arg_list) > 0:
            signature += f", {arg_list}"
        signature += ")"
        if len(method_outputs) == 1:
            signature += " -> torch.Tensor:\n"
        else:
            signature += (
                " -> tuple["
                + ", ".join("torch.Tensor" for _ in method_outputs)
                + "]:\n"
            )
        method_body_lines: List[str] = []
        init_call_lines: List[str] = []
        for spec in class_stage_specs:
            outputs = [str(name) for name in list(spec.get("outputs", []))]
            call_args = ", ".join(str(name) for name in list(spec.get("inputs", [])))
            call_expr = (
                f"self.{spec['method_name']}({call_args})"
                if len(call_args) > 0
                else f"self.{spec['method_name']}()"
            )
            if len(outputs) == 1:
                method_body_lines.append(f"        {outputs[0]} = {call_expr}")
            else:
                method_body_lines.append(f"        {', '.join(outputs)} = {call_expr}")
            init_call_lines.append(
                f"            {spec['method_name']}=self.{spec['method_name']},"
            )
        if len(method_outputs) == 1:
            return_line = f"        return {method_outputs[0]}\n"
        else:
            return_line = f"        return ({', '.join(method_outputs)})\n"
        class_source = (
            "\n".join(init_signature_lines)
            + "\n"
            + "\n".join(init_body_lines)
            + "\n\n"
            + f"{signature}"
            + "\n".join(method_body_lines)
            + "\n"
            + return_line
        )
        return class_source, method_inputs, method_outputs, init_call_lines

    for layer_index, start_idx, end_idx in grouped_ranges:
        while previous_end < start_idx:
            spec = stage_specs[previous_end]
            forward_lines_local.append(_call_line_from_spec(spec))
            previous_end += 1

        method_inputs, method_outputs = _group_io(int(start_idx), int(end_idx))
        if len(method_outputs) == 0:
            for spec_index in range(start_idx, end_idx + 1):
                spec = stage_specs[spec_index]
                forward_lines_local.append(_call_line_from_spec(spec))
            previous_end = int(end_idx + 1)
            continue

        layer_prefix = f"bert_encoder_layer_{int(layer_index)}_"
        split_idx: Optional[int] = None
        for spec_index in range(start_idx, end_idx + 1):
            output_names = [
                str(name) for name in list(stage_specs[spec_index].get("outputs", []))
            ]
            if any(
                output_name.startswith(layer_prefix + marker)
                for output_name in output_names
                for marker in ("ffn_", "output_", "output_bottleneck_")
            ):
                split_idx = int(spec_index)
                break

        composite_init_lines: Optional[List[str]] = None
        layer_attr_name = f"encoder_layer_{int(layer_index)}"
        if split_idx is not None and int(split_idx) > int(start_idx):
            attention_class_name = f"_GeneratedEncoderLayer{int(layer_index)}Attention"
            ffn_class_name = f"_GeneratedEncoderLayer{int(layer_index)}FFN"
            layer_class_name = f"_GeneratedEncoderLayer{int(layer_index)}"
            attention_emitted = _emit_submodule_class_from_stage_range(
                class_name=attention_class_name,
                start_idx=int(start_idx),
                end_idx=int(split_idx - 1),
            )
            ffn_emitted = _emit_submodule_class_from_stage_range(
                class_name=ffn_class_name,
                start_idx=int(split_idx),
                end_idx=int(end_idx),
            )
            if attention_emitted is not None and ffn_emitted is not None:
                (
                    attention_source,
                    attention_inputs,
                    attention_outputs,
                    attention_init_call_lines,
                ) = attention_emitted
                ffn_source, ffn_inputs, ffn_outputs, ffn_init_call_lines = ffn_emitted
                class_chunks.append(attention_source)
                class_chunks.append(ffn_source)
                class_chunks.append(
                    "class {layer_class_name}(torch.nn.Module):\n"
                    "    def __init__(self, *, attention: torch.nn.Module, ffn: torch.nn.Module) -> None:\n"
                    "        super().__init__()\n"
                    "        self.attention = attention\n"
                    "        self.ffn = ffn\n\n"
                    "    def forward({signature_args}){signature_return}"
                    "{body}"
                    "{return_line}".format(
                        layer_class_name=layer_class_name,
                        signature_args=(
                            "self"
                            + (
                                ", "
                                + ", ".join(
                                    f"{name}: torch.Tensor" for name in method_inputs
                                )
                                if len(method_inputs) > 0
                                else ""
                            )
                        ),
                        signature_return=(
                            " -> torch.Tensor:\n"
                            if len(method_outputs) == 1
                            else " -> tuple["
                            + ", ".join("torch.Tensor" for _ in method_outputs)
                            + "]:\n"
                        ),
                        body="".join(
                            [
                                (
                                    f"        {attention_outputs[0]} = self.attention({', '.join(attention_inputs)})\n"
                                    if len(attention_outputs) == 1
                                    else f"        {', '.join(attention_outputs)} = self.attention({', '.join(attention_inputs)})\n"
                                ),
                                (
                                    f"        {ffn_outputs[0]} = self.ffn({', '.join(ffn_inputs)})\n"
                                    if len(ffn_outputs) == 1
                                    else f"        {', '.join(ffn_outputs)} = self.ffn({', '.join(ffn_inputs)})\n"
                                ),
                            ]
                        ),
                        return_line=(
                            f"        return {method_outputs[0]}\n"
                            if len(method_outputs) == 1
                            else f"        return ({', '.join(method_outputs)})\n"
                        ),
                    )
                )
                composite_init_lines = [
                    f"self.{layer_attr_name} = {layer_class_name}(",
                    f"    attention={attention_class_name}(",
                    *attention_init_call_lines,
                    "    ),",
                    f"    ffn={ffn_class_name}(",
                    *ffn_init_call_lines,
                    "    ),",
                    ")",
                ]

        if composite_init_lines is None:
            layer_class_name = f"_GeneratedEncoderLayer{int(layer_index)}"
            emitted = _emit_submodule_class_from_stage_range(
                class_name=layer_class_name,
                start_idx=int(start_idx),
                end_idx=int(end_idx),
            )
            if emitted is None:
                for spec_index in range(start_idx, end_idx + 1):
                    spec = stage_specs[spec_index]
                    forward_lines_local.append(_call_line_from_spec(spec))
                previous_end = int(end_idx + 1)
                continue
            layer_source, _, _, layer_init_call_lines = emitted
            class_chunks.append(layer_source)
            composite_init_lines = [
                f"self.{layer_attr_name} = {layer_class_name}(",
                *layer_init_call_lines,
                ")",
            ]
        if composite_init_lines is not None:
            init_lines.extend(composite_init_lines)

        call_args = ", ".join(method_inputs)
        call_expr = (
            f"self.{layer_attr_name}({call_args})"
            if len(call_args) > 0
            else f"self.{layer_attr_name}()"
        )
        if len(method_outputs) == 1:
            forward_lines_local.append(f"        {method_outputs[0]} = {call_expr}")
        else:
            forward_lines_local.append(
                f"        {', '.join(method_outputs)} = {call_expr}"
            )
        previous_end = int(end_idx + 1)

    while previous_end < len(stage_specs):
        spec = stage_specs[previous_end]
        forward_lines_local.append(_call_line_from_spec(spec))
        previous_end += 1

    named_class_source = "\n".join(class_chunks)
    if len(named_class_source) > 0:
        named_class_source += "\n"
    return named_class_source, init_lines, forward_lines_local
