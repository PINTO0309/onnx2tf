from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple


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
