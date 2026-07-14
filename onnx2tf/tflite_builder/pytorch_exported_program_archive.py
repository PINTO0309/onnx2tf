from __future__ import annotations

import dataclasses
import json
import re
import tempfile
import zipfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple


def _strip_stack_traces_from_exported_program_archive(
    exported_program_path: Path,
) -> None:
    archive_path = Path(exported_program_path)
    if not archive_path.exists():
        raise FileNotFoundError(
            f"ExportedProgram archive not found. path={archive_path}"
        )
    with tempfile.NamedTemporaryFile(
        prefix="onnx2tf_exported_program_strip_",
        suffix=".pt2",
        delete=False,
        dir=str(archive_path.parent),
    ) as tmp_file:
        temp_archive_path = Path(tmp_file.name)
    try:
        removed_count = 0
        with (
            zipfile.ZipFile(str(archive_path), "r") as source_archive,
            zipfile.ZipFile(
                str(temp_archive_path),
                "w",
                compression=zipfile.ZIP_DEFLATED,
            ) as stripped_archive,
        ):
            for info in source_archive.infolist():
                payload = source_archive.read(info.filename)
                if info.filename.endswith("models/model.json"):
                    model_json = json.loads(payload)

                    def _strip_stack_trace_fields(value: Any) -> None:
                        nonlocal removed_count
                        if isinstance(value, dict):
                            if "stack_trace" in value:
                                del value["stack_trace"]
                                removed_count += 1
                            for child in value.values():
                                _strip_stack_trace_fields(child)
                            return
                        if isinstance(value, list):
                            for child in value:
                                _strip_stack_trace_fields(child)

                    _strip_stack_trace_fields(model_json)
                    payload = json.dumps(model_json, separators=(",", ":")).encode(
                        "utf-8"
                    )
                stripped_archive.writestr(info, payload)
        if removed_count == 0:
            temp_archive_path.unlink(missing_ok=True)
            return
        temp_archive_path.replace(archive_path)
    except Exception:
        temp_archive_path.unlink(missing_ok=True)
        raise



def _fold_inverse_permute_round_trips_in_exported_program_archive(
    exported_program_path: Path,
) -> None:
    archive_path = Path(exported_program_path)
    if not archive_path.exists():
        raise FileNotFoundError(f"ExportedProgram archive not found. path={archive_path}")

    import torch

    exported_program = torch.export.load(str(archive_path))
    graph_module = getattr(exported_program, "graph_module", None)
    if graph_module is None:
        return
    graph = graph_module.graph
    graph_signature = getattr(exported_program, "graph_signature", None)
    changed = False

    user_output_names = set()
    if graph_signature is not None:
        try:
            user_output_names = {str(name) for name in list(graph_signature.user_outputs)}
        except Exception:
            user_output_names = set()

    def _normalize_perm_simple(arg: Any) -> Optional[List[int]]:
        if not isinstance(arg, (list, tuple)):
            return None
        return [int(v) for v in arg]

    def _inverse_perm_simple(perm: Sequence[int]) -> List[int]:
        inverse = [0] * len(list(perm))
        for idx, value in enumerate(list(perm)):
            inverse[int(value)] = int(idx)
        return inverse

    while True:
        local_changed = False
        for node in list(graph.nodes):
            if (
                node.op == "call_function"
                and str(node.target) == "aten.alias.default"
                and len(node.args) >= 1
                and isinstance(node.args[0], torch.fx.Node)
                and str(node.name) not in user_output_names
            ):
                node.replace_all_uses_with(node.args[0])
                graph.erase_node(node)
                local_changed = True
                changed = True
        for node in list(graph.nodes):
            if (
                node.op != "call_function"
                or str(node.target) != "aten.permute.default"
                or len(node.args) < 2
            ):
                continue
            perm = _normalize_perm_simple(node.args[1])
            source = node.args[0] if len(node.args) >= 1 and isinstance(node.args[0], torch.fx.Node) else None
            if perm is None or source is None:
                continue
            node_users = list(node.users)
            if (
                len(node_users) != 1
                or node_users[0].op != "call_function"
                or str(node_users[0].target) != "aten.contiguous.default"
            ):
                continue
            contiguous_node = node_users[0]
            contiguous_users = list(contiguous_node.users)
            if len(contiguous_users) != 1:
                continue
            inverse_node = contiguous_users[0]
            if (
                inverse_node.op != "call_function"
                or str(inverse_node.target) != "aten.permute.default"
                or len(inverse_node.args) < 2
                or _normalize_perm_simple(inverse_node.args[1]) != _inverse_perm_simple(perm)
                or str(inverse_node.name) in user_output_names
            ):
                continue
            inverse_users = list(inverse_node.users)
            if (
                len(inverse_users) == 1
                and inverse_users[0].op == "call_function"
                and str(inverse_users[0].target) == "aten.contiguous.default"
            ):
                inverse_users[0].replace_all_uses_with(source)
            else:
                inverse_node.replace_all_uses_with(source)
            local_changed = True
            changed = True
        if not local_changed:
            break
        graph.eliminate_dead_code()
        graph.lint()
        graph_module.recompile()

    def _normalize_perm(arg: Any) -> Optional[List[int]]:
        if not isinstance(arg, (list, tuple)):
            return None
        return [int(v) for v in arg]

    def _is_scalar_like(value: Any) -> bool:
        if not isinstance(value, torch.fx.Node):
            return True
        meta_val = getattr(value, "meta", {}).get("val", None)
        if isinstance(meta_val, torch.Tensor):
            return int(meta_val.numel()) == 1
        if value.op == "get_attr" and isinstance(value.target, str):
            tensor = getattr(graph_module, value.target, None)
            if isinstance(tensor, torch.Tensor):
                return int(tensor.numel()) == 1
        return False

    def _rank_shape(node: Any) -> Optional[List[int]]:
        if not isinstance(node, torch.fx.Node):
            return None
        meta_val = getattr(node, "meta", {}).get("val", None)
        if isinstance(meta_val, torch.Tensor):
            return [int(v) for v in list(meta_val.shape)]
        if node.op == "get_attr" and isinstance(node.target, str):
            tensor = getattr(graph_module, node.target, None)
            if isinstance(tensor, torch.Tensor):
                return [int(v) for v in list(tensor.shape)]
        return None

    def _shape_meta_from_node(node: torch.fx.Node, shape: Sequence[int]) -> Dict[str, Any]:
        meta = dict(getattr(node, "meta", {}))
        meta_val = meta.get("val", None)
        if isinstance(meta_val, torch.Tensor):
            try:
                meta["val"] = torch.empty(
                    tuple(int(v) for v in shape),
                    dtype=meta_val.dtype,
                    device="meta",
                )
            except Exception:
                pass
        return meta

    def _nchw_singleton_shape_from_nhwc(node: Any) -> Optional[List[int]]:
        source_shape = _rank_shape(node)
        if source_shape is None or len(source_shape) != 4 or int(source_shape[3]) != 1:
            return None
        return [int(source_shape[0]), 1, int(source_shape[1]), int(source_shape[2])]

    scalar_binary_targets = {
        "aten.add.Tensor",
        "aten.maximum.default",
        "aten.minimum.default",
        "aten.mul.Tensor",
        "aten.sub.Tensor",
        "aten.div.Tensor",
    }

    def _broadcast_shapes(lhs: Optional[List[int]], rhs: Optional[List[int]]) -> Optional[List[int]]:
        if lhs is None:
            return rhs
        if rhs is None:
            return lhs
        result: List[int] = []
        for lhs_dim, rhs_dim in zip(reversed(lhs), reversed(rhs)):
            if lhs_dim == rhs_dim:
                result.append(int(lhs_dim))
            elif lhs_dim == 1:
                result.append(int(rhs_dim))
            elif rhs_dim == 1:
                result.append(int(lhs_dim))
            else:
                return None
        if len(lhs) > len(rhs):
            result.extend(reversed(lhs[: len(lhs) - len(rhs)]))
        elif len(rhs) > len(lhs):
            result.extend(reversed(rhs[: len(rhs) - len(lhs)]))
        result.reverse()
        return [int(v) for v in result]

    def _shape_arg_list(arg: Any) -> Optional[List[int]]:
        if not isinstance(arg, (list, tuple)):
            return None
        shape: List[int] = []
        for dim in arg:
            if not isinstance(dim, (int, bool)):
                return None
            shape.append(int(dim))
        return shape

    def _scalar_arg_value(arg: Any) -> Optional[float]:
        if isinstance(arg, (int, float, bool)):
            return float(arg)
        if not isinstance(arg, torch.fx.Node):
            return None
        tensor_value = _constant_tensor_value(arg)
        if isinstance(tensor_value, torch.Tensor) and int(tensor_value.numel()) == 1:
            try:
                return float(tensor_value.reshape(-1)[0].item())
            except Exception:
                return None
        return None

    def _first_meta_source_node(node: torch.fx.Node) -> Optional[torch.fx.Node]:
        for arg in node.args:
            if isinstance(arg, torch.fx.Node) and _rank_shape(arg) is not None:
                return arg
            if isinstance(arg, (list, tuple)):
                for item in arg:
                    if isinstance(item, torch.fx.Node) and _rank_shape(item) is not None:
                        return item
        return None

    placeholder_constant_tensors: Dict[str, torch.Tensor] = {}
    mutable_input_specs: List[torch.export.graph_signature.InputSpec] = list(
        getattr(exported_program.graph_signature, "input_specs", ())
    )
    if graph_signature is not None:
        for input_spec in list(mutable_input_specs):
            arg = getattr(input_spec, "arg", None)
            placeholder_name = str(getattr(arg, "name", ""))
            if placeholder_name == "":
                continue
            kind_name = str(getattr(input_spec, "kind", ""))
            target_name = str(getattr(input_spec, "target", ""))
            tensor_value: Optional[torch.Tensor] = None
            if "PARAMETER" in kind_name or "BUFFER" in kind_name:
                candidate = exported_program.state_dict.get(target_name, None)
                if isinstance(candidate, torch.Tensor):
                    tensor_value = candidate
            elif "CONSTANT_TENSOR" in kind_name:
                candidate = exported_program.constants.get(target_name, None)
                if not isinstance(candidate, torch.Tensor):
                    candidate = exported_program.tensor_constants.get(target_name, None)
                if isinstance(candidate, torch.Tensor):
                    tensor_value = candidate
            if isinstance(tensor_value, torch.Tensor):
                placeholder_constant_tensors[placeholder_name] = tensor_value.detach()

    def _constant_meta_from_tensor(reference_node: torch.fx.Node, value: torch.Tensor) -> Dict[str, Any]:
        meta = dict(getattr(reference_node, "meta", {}))
        try:
            meta["val"] = torch.empty(
                tuple(int(v) for v in list(value.shape)),
                dtype=value.dtype,
                device="meta",
            )
        except Exception:
            pass
        return meta

    def _make_unique_folded_constant_names(base_name: str) -> Tuple[str, str]:
        sanitized = re.sub(r"[^0-9A-Za-z_]+", "_", str(base_name)).strip("_")
        if sanitized == "":
            sanitized = "folded_const"
        if sanitized[0].isdigit():
            sanitized = f"const_{sanitized}"
        target_candidate = f"folded_const_{sanitized}"
        suffix = 0
        existing_placeholder_names = {str(node.name) for node in graph.nodes if node.op == "placeholder"}
        existing_targets = {
            str(getattr(input_spec, "target", ""))
            for input_spec in mutable_input_specs
            if getattr(input_spec, "target", None) is not None
        }
        placeholder_candidate = f"c_{target_candidate}"
        while target_candidate in existing_targets or placeholder_candidate in existing_placeholder_names:
            suffix += 1
            target_candidate = f"folded_const_{sanitized}_{suffix}"
            placeholder_candidate = f"c_{target_candidate}"
        return placeholder_candidate, target_candidate

    def _insert_folded_constant_placeholder(reference_node: torch.fx.Node, value: torch.Tensor) -> torch.fx.Node:
        placeholder_name, target_name = _make_unique_folded_constant_names(str(reference_node.name))
        first_user_input_name = next(
            (
                str(getattr(getattr(input_spec, "arg", None), "name", ""))
                for input_spec in mutable_input_specs
                if str(getattr(input_spec, "kind", "")) == "InputKind.USER_INPUT"
                and str(getattr(getattr(input_spec, "arg", None), "name", "")) != ""
            ),
            "",
        )
        insert_before_node = next(
            (
                node
                for node in graph.nodes
                if node.op == "placeholder" and str(node.name) == first_user_input_name
            ),
            next((node for node in graph.nodes if node.op != "placeholder"), None),
        )
        if insert_before_node is None:
            raise RuntimeError("Could not find insertion point for folded constant placeholder.")
        with graph.inserting_before(insert_before_node):
            placeholder_node = graph.placeholder(placeholder_name)
        placeholder_node.meta = _constant_meta_from_tensor(reference_node, value)
        placeholder_constant_tensors[placeholder_name] = value.detach().clone()
        exported_program.constants[target_name] = value.detach().clone()
        exported_program.tensor_constants[target_name] = value.detach().clone()
        insert_index = next(
            (
                index
                for index, input_spec in enumerate(mutable_input_specs)
                if str(getattr(input_spec, "kind", "")) == "InputKind.USER_INPUT"
            ),
            len(mutable_input_specs),
        )
        mutable_input_specs.insert(
            insert_index,
            torch.export.graph_signature.InputSpec(
                kind=torch.export.graph_signature.InputKind.CONSTANT_TENSOR,
                arg=torch.export.graph_signature.TensorArgument(name=placeholder_name),
                target=target_name,
            ),
        )
        return placeholder_node

    constant_foldable_targets = {
        "aten.add.Tensor",
        "aten.cat.default",
        "aten.clamp.default",
        "aten.contiguous.default",
        "aten.div.Tensor",
        "aten.exp.default",
        "aten.leaky_relu.default",
        "aten.lift_fresh_copy.default",
        "aten.matmul.default",
        "aten.maximum.default",
        "aten.minimum.default",
        "aten.mul.Tensor",
        "aten.neg.default",
        "aten.pad.default",
        "aten.permute.default",
        "aten.relu.default",
        "aten.reshape.default",
        "aten.sigmoid.default",
        "aten.slice.Tensor",
        "aten.sub.Tensor",
    }

    constant_value_cache: Dict[str, Optional[torch.Tensor]] = {}

    def _constant_value_from_arg(arg: Any) -> Optional[Any]:
        if isinstance(arg, torch.fx.Node):
            return _constant_tensor_value(arg)
        if isinstance(arg, dict):
            dict_values: Dict[Any, Any] = {}
            for key, value in arg.items():
                folded_value = _constant_value_from_arg(value)
                if folded_value is None and isinstance(value, torch.fx.Node):
                    return None
                if folded_value is None and not isinstance(value, torch.fx.Node):
                    folded_value = value
                dict_values[key] = folded_value
            return dict_values
        if isinstance(arg, (list, tuple)):
            values: List[Any] = []
            for item in list(arg):
                item_value = _constant_value_from_arg(item)
                if item_value is None and isinstance(item, torch.fx.Node):
                    return None
                if item_value is None and not isinstance(item, torch.fx.Node):
                    item_value = item
                values.append(item_value)
            return type(arg)(values)
        return arg

    def _constant_tensor_value(node: torch.fx.Node) -> Optional[torch.Tensor]:
        cached = constant_value_cache.get(str(node.name), None)
        if str(node.name) in constant_value_cache:
            return cached
        result: Optional[torch.Tensor] = None
        if node.op == "placeholder":
            candidate = placeholder_constant_tensors.get(str(node.name), None)
            if isinstance(candidate, torch.Tensor):
                result = candidate
        elif node.op == "get_attr" and isinstance(node.target, str):
            candidate = getattr(graph_module, node.target, None)
            if isinstance(candidate, torch.Tensor):
                result = candidate.detach()
        elif (
            node.op == "call_function"
            and callable(node.target)
            and str(node.target) in constant_foldable_targets
        ):
            evaluated_args = _constant_value_from_arg(node.args)
            evaluated_kwargs = _constant_value_from_arg(dict(node.kwargs))
            if evaluated_args is not None and evaluated_kwargs is not None:
                try:
                    with torch.no_grad():
                        candidate = node.target(*tuple(evaluated_args), **dict(evaluated_kwargs))
                    if isinstance(candidate, torch.Tensor):
                        result = candidate.detach()
                except Exception:
                    result = None
        constant_value_cache[str(node.name)] = result
        return result

    def _infer_tensor_shape(node: torch.fx.Node) -> Optional[List[int]]:
        if node.op != "call_function":
            return _rank_shape(node)
        target = str(node.target)
        if target in {"aten.contiguous.default", "aten.lift_fresh_copy.default"} and len(node.args) >= 1:
            return _rank_shape(node.args[0])
        if target in {
            "aten.clamp.default",
            "aten.relu.default",
            "aten.exp.default",
            "aten.neg.default",
            "aten.sigmoid.default",
            "aten.leaky_relu.default",
        } and len(node.args) >= 1:
            return _rank_shape(node.args[0])
        if target in scalar_binary_targets and len(node.args) == 2:
            return _broadcast_shapes(_rank_shape(node.args[0]), _rank_shape(node.args[1]))
        if target == "aten.reshape.default" and len(node.args) >= 2:
            return _shape_arg_list(node.args[1])
        if target == "aten.permute.default" and len(node.args) >= 2:
            source_shape = _rank_shape(node.args[0])
            perm = _shape_arg_list(node.args[1])
            if source_shape is None or perm is None or len(source_shape) != len(perm):
                return None
            return [int(source_shape[int(axis)]) for axis in perm]
        if target == "aten.cat.default" and len(node.args) >= 2:
            cat_inputs = node.args[0]
            cat_dim_arg = node.args[1]
            if not isinstance(cat_inputs, (list, tuple)) or not isinstance(cat_dim_arg, (int, bool)):
                return None
            input_shapes = [_rank_shape(inp) for inp in cat_inputs if isinstance(inp, torch.fx.Node)]
            if len(input_shapes) != len(cat_inputs) or len(input_shapes) == 0 or any(shape is None for shape in input_shapes):
                return None
            cat_dim = int(cat_dim_arg)
            base_shape = list(input_shapes[0] or [])
            if cat_dim < 0:
                cat_dim += len(base_shape)
            if cat_dim < 0 or cat_dim >= len(base_shape):
                return None
            base_shape[cat_dim] = sum(int((shape or base_shape)[cat_dim]) for shape in input_shapes)
            return base_shape
        if target == "aten.slice.Tensor" and len(node.args) >= 4:
            source_shape = _rank_shape(node.args[0])
            if source_shape is None:
                return None
            dim_arg = node.args[1]
            start_arg = node.args[2]
            end_arg = node.args[3]
            if not isinstance(dim_arg, (int, bool)):
                return None
            if not isinstance(start_arg, (int, bool)):
                return None
            if not isinstance(end_arg, (int, bool)):
                return None
            dim = int(dim_arg)
            start = int(start_arg)
            end = int(end_arg)
            if dim < 0:
                dim += len(source_shape)
            if dim < 0 or dim >= len(source_shape):
                return None
            output_shape = list(source_shape)
            output_shape[dim] = max(0, end - start)
            return output_shape
        if target == "aten.pad.default" and len(node.args) >= 2:
            source_shape = _rank_shape(node.args[0])
            pad = _shape_arg_list(node.args[1])
            if source_shape is None or pad is None or len(source_shape) != 4 or len(pad) != 4:
                return None
            output_shape = list(source_shape)
            output_shape[3] += int(pad[0]) + int(pad[1])
            output_shape[2] += int(pad[2]) + int(pad[3])
            return output_shape
        return _rank_shape(node)

    def _repair_inferred_meta_shapes() -> bool:
        inferred_changed = False
        for node in list(graph.nodes):
            if not isinstance(node, torch.fx.Node):
                continue
            inferred_shape = _infer_tensor_shape(node)
            if inferred_shape is None:
                continue
            current_shape = _rank_shape(node)
            if current_shape == inferred_shape:
                continue
            meta_source = _first_meta_source_node(node)
            if meta_source is None:
                continue
            node.meta = _shape_meta_from_node(meta_source, inferred_shape)
            inferred_changed = True
        if inferred_changed:
            graph_module.recompile()
        return inferred_changed

    def _run_one_pass() -> bool:
        local_changed = False

        def _resolve_dim3_cat_input_as_cf(
            cat_input: Any,
            *,
            insert_before: torch.fx.Node,
        ) -> Optional[torch.fx.Node]:
            if (
                isinstance(cat_input, torch.fx.Node)
                and cat_input.op == "call_function"
                and str(cat_input.target) == "aten.contiguous.default"
                and len(cat_input.args) >= 1
                and isinstance(cat_input.args[0], torch.fx.Node)
            ):
                permute_node = cat_input.args[0]
                if (
                    permute_node.op == "call_function"
                    and str(permute_node.target) == "aten.permute.default"
                    and len(permute_node.args) >= 2
                    and _normalize_perm(permute_node.args[1]) == [0, 2, 3, 1]
                    and isinstance(permute_node.args[0], torch.fx.Node)
                    and _rank_shape(permute_node.args[0]) is not None
                    and len(_rank_shape(permute_node.args[0]) or []) == 4
                ):
                    return permute_node.args[0]
            if (
                not isinstance(cat_input, torch.fx.Node)
                or cat_input.op != "call_function"
                or str(cat_input.target) != "aten.index.Tensor"
                or len(cat_input.args) < 2
                or not isinstance(cat_input.args[0], torch.fx.Node)
                or not isinstance(cat_input.args[1], (list, tuple))
            ):
                return None
            source = cat_input.args[0]
            source_dim_arg = source.args[1] if len(source.args) >= 2 else None
            if (
                source.op != "call_function"
                or str(source.target) != "aten.cat.default"
                or not isinstance(source_dim_arg, (int, bool))
                or int(source_dim_arg) != 1
            ):
                return None
            index_spec = list(cat_input.args[1])
            if len(index_spec) != 4 or index_spec[0] is not None or index_spec[2] is not None:
                return None
            if index_spec[1] is not None and index_spec[3] is None:
                return cat_input
            if index_spec[1] is None and index_spec[3] is not None:
                channel_indices = index_spec[3]
                if not isinstance(channel_indices, (torch.fx.Node, torch.Tensor)):
                    return None
                with graph.inserting_before(insert_before):
                    cf_index_node = graph.call_function(
                        torch.ops.aten.index.Tensor,
                        args=(source, [None, channel_indices, None, None]),
                        kwargs={},
                    )
                cf_index_node.meta = dict(getattr(cat_input, "meta", {}))
                return cf_index_node
            return None

        for node in list(graph.nodes):
            if (
                node.op != "call_function"
                or str(node.target) != "aten.permute.default"
                or len(node.args) < 2
                or _normalize_perm(node.args[1]) != [0, 3, 1, 2]
            ):
                continue
            clamp_node = node.args[0]
            if (
                not isinstance(clamp_node, torch.fx.Node)
                or clamp_node.op != "call_function"
                or str(clamp_node.target) != "aten.clamp.default"
                or len(clamp_node.args) < 3
            ):
                continue
            clamp_min = _scalar_arg_value(clamp_node.args[1])
            clamp_max = _scalar_arg_value(clamp_node.args[2])
            if clamp_min is None or clamp_max is None or abs(clamp_min) > 1e-6 or abs(clamp_max - 1.0) > 1e-6:
                continue
            clamp_input = clamp_node.args[0]
            if (
                not isinstance(clamp_input, torch.fx.Node)
                or clamp_input.op != "call_function"
                or str(clamp_input.target) != "aten.contiguous.default"
                or len(clamp_input.args) < 1
                or not isinstance(clamp_input.args[0], torch.fx.Node)
            ):
                continue
            permute_node = clamp_input.args[0]
            if (
                permute_node.op != "call_function"
                or str(permute_node.target) != "aten.permute.default"
                or len(permute_node.args) < 2
                or _normalize_perm(permute_node.args[1]) != [0, 2, 3, 1]
                or not isinstance(permute_node.args[0], torch.fx.Node)
            ):
                continue
            add_node = permute_node.args[0]
            if (
                add_node.op != "call_function"
                or str(add_node.target) != "aten.add.Tensor"
                or len(add_node.args) != 2
            ):
                continue
            add_const_index: Optional[int] = None
            add_input_index: Optional[int] = None
            for input_index, arg in enumerate(list(add_node.args)):
                scalar_value = _scalar_arg_value(arg)
                if scalar_value is not None and abs(scalar_value - 0.5) <= 1e-6:
                    add_const_index = int(input_index)
                elif isinstance(arg, torch.fx.Node):
                    add_input_index = int(input_index)
            if add_const_index is None or add_input_index is None:
                continue
            mul_node = add_node.args[add_input_index]
            if (
                not isinstance(mul_node, torch.fx.Node)
                or mul_node.op != "call_function"
                or str(mul_node.target) != "aten.mul.Tensor"
                or len(mul_node.args) != 2
            ):
                continue
            mul_const_index: Optional[int] = None
            mul_input_index: Optional[int] = None
            for input_index, arg in enumerate(list(mul_node.args)):
                scalar_value = _scalar_arg_value(arg)
                if scalar_value is not None and abs(scalar_value - (1.0 / 6.0)) <= 1e-6:
                    mul_const_index = int(input_index)
                elif isinstance(arg, torch.fx.Node):
                    mul_input_index = int(input_index)
            if mul_const_index is None or mul_input_index is None:
                continue
            source = mul_node.args[mul_input_index]
            if not isinstance(source, torch.fx.Node):
                continue
            if len(list(mul_node.users)) != 1 or add_node not in mul_node.users:
                continue
            if len(list(add_node.users)) != 1 or permute_node not in add_node.users:
                continue
            if len(list(permute_node.users)) != 1 or clamp_input not in permute_node.users:
                continue
            if len(list(clamp_input.users)) != 1 or clamp_node not in clamp_input.users:
                continue
            if len(list(clamp_node.users)) != 1 or node not in clamp_node.users:
                continue
            with graph.inserting_before(mul_node):
                folded_hardsigmoid = graph.call_function(
                    torch.ops.aten.hardsigmoid.default,
                    args=(source,),
                    kwargs={},
                )
            folded_hardsigmoid.meta = _shape_meta_from_node(source, _rank_shape(source) or [])
            node_users = list(node.users)
            if (
                len(node_users) == 1
                and node_users[0].op == "call_function"
                and str(node_users[0].target) == "aten.contiguous.default"
                and str(node_users[0].name) not in user_output_names
            ):
                node_users[0].replace_all_uses_with(folded_hardsigmoid)
            elif str(node.name) not in user_output_names:
                node.replace_all_uses_with(folded_hardsigmoid)
            else:
                continue
            local_changed = True

        for node in list(graph.nodes):
            if (
                node.op != "call_function"
                or str(node.target) != "aten.reshape.default"
                or len(node.args) < 2
                or not isinstance(node.args[0], torch.fx.Node)
            ):
                continue
            source_node = node.args[0]
            source_shape = _rank_shape(source_node)
            reshape_shape = _shape_arg_list(node.args[1])
            if (
                source_shape is None
                or reshape_shape is None
                or len(source_shape) != len(reshape_shape)
                or [int(v) for v in source_shape] != [int(v) for v in reshape_shape]
            ):
                continue
            node.replace_all_uses_with(source_node)
            local_changed = True

        for node in list(graph.nodes):
            if node.op != "call_function" or str(node.target) not in constant_foldable_targets:
                continue
            constant_value = _constant_tensor_value(node)
            if not isinstance(constant_value, torch.Tensor):
                continue
            folded_const_node = _insert_folded_constant_placeholder(node, constant_value)
            node.replace_all_uses_with(folded_const_node)
            local_changed = True

        for width_matmul_node in list(graph.nodes):
            if (
                width_matmul_node.op != "call_function"
                or str(width_matmul_node.target) != "aten.matmul.default"
                or len(width_matmul_node.args) != 2
                or not isinstance(width_matmul_node.args[0], torch.fx.Node)
                or not isinstance(width_matmul_node.args[1], torch.fx.Node)
            ):
                continue
            width_const_node = width_matmul_node.args[0]
            trailing_reshape_node = width_matmul_node.args[1]
            if (
                trailing_reshape_node.op != "call_function"
                or str(trailing_reshape_node.target) != "aten.reshape.default"
                or len(trailing_reshape_node.args) < 2
                or not isinstance(trailing_reshape_node.args[0], torch.fx.Node)
            ):
                continue
            trailing_reshape_shape = _shape_arg_list(trailing_reshape_node.args[1])
            if (
                trailing_reshape_shape is None
                or len(trailing_reshape_shape) != 4
                or int(trailing_reshape_shape[3]) != 1
            ):
                continue

            height_matmul_node = trailing_reshape_node.args[0]
            if (
                height_matmul_node.op != "call_function"
                or str(height_matmul_node.target) != "aten.matmul.default"
                or len(height_matmul_node.args) != 2
                or not isinstance(height_matmul_node.args[0], torch.fx.Node)
                or not isinstance(height_matmul_node.args[1], torch.fx.Node)
            ):
                continue
            height_const_node = height_matmul_node.args[0]
            leading_reshape_node = height_matmul_node.args[1]
            if (
                leading_reshape_node.op != "call_function"
                or str(leading_reshape_node.target) != "aten.reshape.default"
                or len(leading_reshape_node.args) < 2
                or not isinstance(leading_reshape_node.args[0], torch.fx.Node)
            ):
                continue

            source_node = leading_reshape_node.args[0]
            source_shape = _rank_shape(source_node)
            leading_reshape_shape = _shape_arg_list(leading_reshape_node.args[1])
            height_const_shape = _rank_shape(height_const_node)
            width_const_shape = _rank_shape(width_const_node)
            if (
                source_shape is None
                or len(source_shape) != 4
                or int(source_shape[1]) != 1
                or leading_reshape_shape is None
                or len(leading_reshape_shape) != 3
                or int(leading_reshape_shape[0]) != int(source_shape[0])
                or int(leading_reshape_shape[1]) != int(source_shape[2])
                or int(leading_reshape_shape[2]) != int(source_shape[3])
                or height_const_shape is None
                or len(height_const_shape) != 3
                or int(height_const_shape[2]) != int(source_shape[2])
                or int(trailing_reshape_shape[0]) != int(source_shape[0])
                or int(trailing_reshape_shape[1]) != int(height_const_shape[1])
                or int(trailing_reshape_shape[2]) != int(source_shape[3])
                or width_const_shape is None
                or len(width_const_shape) != 4
                or int(width_const_shape[3]) != int(source_shape[3])
            ):
                continue

            height_const_nchw_shape = [
                int(height_const_shape[0]),
                1,
                int(height_const_shape[1]),
                int(height_const_shape[2]),
            ]
            width_const_nchw_shape = [
                int(width_const_shape[0]),
                int(width_const_shape[1]),
                int(width_const_shape[3]),
                int(width_const_shape[2]),
            ]
            folded_height_shape = [
                int(source_shape[0]),
                int(source_shape[1]),
                int(height_const_shape[1]),
                int(source_shape[3]),
            ]
            folded_width_shape = [
                int(source_shape[0]),
                int(source_shape[1]),
                int(height_const_shape[1]),
                int(width_const_shape[2]),
            ]

            with graph.inserting_before(height_matmul_node):
                height_const_nchw = graph.call_function(
                    torch.ops.aten.reshape.default,
                    args=(height_const_node, height_const_nchw_shape),
                    kwargs={},
                )
            height_const_nchw.meta = _shape_meta_from_node(height_const_node, height_const_nchw_shape)

            with graph.inserting_before(height_matmul_node):
                folded_height_matmul = graph.call_function(
                    height_matmul_node.target,
                    args=(height_const_nchw, source_node),
                    kwargs=dict(height_matmul_node.kwargs),
                )
            folded_height_matmul.meta = _shape_meta_from_node(source_node, folded_height_shape)

            with graph.inserting_before(width_matmul_node):
                width_const_nchw = graph.call_function(
                    torch.ops.aten.permute.default,
                    args=(width_const_node, [0, 1, 3, 2]),
                    kwargs={},
                )
            width_const_nchw.meta = _shape_meta_from_node(width_const_node, width_const_nchw_shape)

            with graph.inserting_before(width_matmul_node):
                folded_width_matmul = graph.call_function(
                    width_matmul_node.target,
                    args=(folded_height_matmul, width_const_nchw),
                    kwargs=dict(width_matmul_node.kwargs),
                )
            folded_width_matmul.meta = _shape_meta_from_node(source_node, folded_width_shape)

            width_matmul_node.replace_all_uses_with(folded_width_matmul)
            local_changed = True

        for node in list(graph.nodes):
            if (
                node.op != "call_function"
                or str(node.target) != "aten.contiguous.default"
                or len(node.args) < 1
                or not isinstance(node.args[0], torch.fx.Node)
            ):
                continue
            permute_node = node.args[0]
            if (
                permute_node.op != "call_function"
                or str(permute_node.target) != "aten.permute.default"
                or len(permute_node.args) < 2
                or _normalize_perm(permute_node.args[1]) != [0, 2, 3, 1]
                or not isinstance(permute_node.args[0], torch.fx.Node)
            ):
                continue
            source = permute_node.args[0]
            for user in list(node.users):
                if (
                    user.op == "call_function"
                    and str(user.target) == "aten.permute.default"
                    and len(user.args) >= 2
                    and _normalize_perm(user.args[1]) == [0, 3, 1, 2]
                ):
                    user_users = list(user.users)
                    if (
                        len(user_users) == 1
                        and user_users[0].op == "call_function"
                        and str(user_users[0].target) == "aten.contiguous.default"
                    ):
                        user_users[0].replace_all_uses_with(source)
                    else:
                        user.replace_all_uses_with(source)
                    local_changed = True

        for node in list(graph.nodes):
            if (
                node.op != "call_function"
                or str(node.target) != "aten.permute.default"
                or len(node.args) < 2
                or _normalize_perm(node.args[1]) != [0, 3, 1, 2]
                or not isinstance(node.args[0], torch.fx.Node)
            ):
                continue
            source = node.args[0]
            source_shape = _rank_shape(source)
            node_users = list(node.users)
            if (
                len(node_users) != 1
                or node_users[0].op != "call_function"
                or str(node_users[0].target) != "aten.contiguous.default"
            ):
                continue
            contiguous_node = node_users[0]
            contiguous_shape = _rank_shape(contiguous_node)
            if contiguous_shape is not None and len(contiguous_shape) != 4:
                continue
            contiguous_users = list(contiguous_node.users)
            if (
                len(contiguous_users) != 1
                or contiguous_users[0].op != "call_function"
                or str(contiguous_users[0].target) != "aten.pad.default"
                or len(contiguous_users[0].args) < 2
                or not isinstance(contiguous_users[0].args[1], (list, tuple))
            ):
                continue
            pad_node = contiguous_users[0]
            pad_values = [int(v) for v in list(pad_node.args[1])]
            if len(pad_values) != 4:
                continue
            pad_users = list(pad_node.users)
            if (
                len(pad_users) != 1
                or pad_users[0].op != "call_function"
                or str(pad_users[0].target) != "aten.conv2d.default"
                or len(pad_users[0].args) < 7
            ):
                continue
            conv_node = pad_users[0]
            groups_arg = conv_node.args[6]
            if not isinstance(groups_arg, (int, bool)):
                continue
            groups = int(groups_arg)
            if groups <= 1:
                continue
            if source_shape is not None and (len(source_shape) != 4 or int(source_shape[1]) != groups):
                continue
            if contiguous_shape is not None and int(contiguous_shape[1]) == groups:
                continue
            sibling_conv_users = [
                user
                for user in list(source.users)
                if user is not node
                and user.op == "call_function"
                and str(user.target) == "aten.conv2d.default"
            ]
            if len(sibling_conv_users) == 0:
                continue
            pad_node.args = (source, *tuple(pad_node.args[1:]))
            if source_shape is not None:
                repaired_pad_shape = [
                    int(source_shape[0]),
                    int(source_shape[1]),
                    int(source_shape[2]) + pad_values[2] + pad_values[3],
                    int(source_shape[3]) + pad_values[0] + pad_values[1],
                ]
                pad_node.meta = _shape_meta_from_node(source, repaired_pad_shape)
            local_changed = True

        for node in list(graph.nodes):
            if (
                node.op != "call_function"
                or str(node.target) != "aten.contiguous.default"
                or len(node.args) < 1
                or not isinstance(node.args[0], torch.fx.Node)
            ):
                continue
            permute_node = node.args[0]
            if (
                permute_node.op != "call_function"
                or str(permute_node.target) != "aten.permute.default"
                or len(permute_node.args) < 2
                or _normalize_perm(permute_node.args[1]) != [0, 2, 3, 1]
                or not isinstance(permute_node.args[0], torch.fx.Node)
            ):
                continue
            source = permute_node.args[0]
            node_users = list(node.users)
            if len(node_users) != 1:
                continue
            binary_node = node_users[0]
            if (
                binary_node.op != "call_function"
                or str(binary_node.target) not in scalar_binary_targets
                or len(binary_node.args) != 2
            ):
                continue
            lhs_is_source = binary_node.args[0] is node and _is_scalar_like(binary_node.args[1])
            rhs_is_source = binary_node.args[1] is node and _is_scalar_like(binary_node.args[0])
            if not lhs_is_source and not rhs_is_source:
                continue
            binary_users = list(binary_node.users)
            if len(binary_users) != 1:
                continue
            inverse_node = binary_users[0]
            if (
                inverse_node.op != "call_function"
                or str(inverse_node.target) != "aten.permute.default"
                or len(inverse_node.args) < 2
                or _normalize_perm(inverse_node.args[1]) != [0, 3, 1, 2]
            ):
                continue
            with graph.inserting_before(binary_node):
                folded_binary = graph.call_function(
                    binary_node.target,
                    args=(
                        source if lhs_is_source else binary_node.args[0],
                        binary_node.args[1] if lhs_is_source else source,
                    ),
                    kwargs=dict(binary_node.kwargs),
                )
            folded_binary.meta = dict(getattr(inverse_node, "meta", {}))
            inverse_users = list(inverse_node.users)
            if (
                len(inverse_users) == 1
                and inverse_users[0].op == "call_function"
                and str(inverse_users[0].target) == "aten.contiguous.default"
            ):
                inverse_users[0].replace_all_uses_with(folded_binary)
            else:
                inverse_node.replace_all_uses_with(folded_binary)
            local_changed = True

        for node in list(graph.nodes):
            if (
                node.op != "call_function"
                or str(node.target) != "aten.permute.default"
                or len(node.args) < 2
                or _normalize_perm(node.args[1]) != [0, 3, 1, 2]
                or not isinstance(node.args[0], torch.fx.Node)
            ):
                continue
            source = node.args[0]
            nchw_shape = _nchw_singleton_shape_from_nhwc(source)
            if nchw_shape is None:
                continue
            with graph.inserting_before(node):
                reshape_node = graph.call_function(
                    torch.ops.aten.reshape.default,
                    args=(source, nchw_shape),
                    kwargs={},
                )
            reshape_node.meta = _shape_meta_from_node(source, nchw_shape)
            node_users = list(node.users)
            if (
                len(node_users) == 1
                and node_users[0].op == "call_function"
                and str(node_users[0].target) == "aten.contiguous.default"
            ):
                node_users[0].replace_all_uses_with(reshape_node)
            else:
                node.replace_all_uses_with(reshape_node)
            local_changed = True

        for node in list(graph.nodes):
            if (
                node.op != "call_function"
                or str(node.target) != "aten.permute.default"
                or len(node.args) < 2
                or _normalize_perm(node.args[1]) != [0, 2, 3, 1]
                or not isinstance(node.args[0], torch.fx.Node)
            ):
                continue
            reshape_node = node.args[0]
            reshape_shape_arg = reshape_node.args[1] if len(reshape_node.args) >= 2 else None
            if (
                reshape_node.op != "call_function"
                or str(reshape_node.target) != "aten.reshape.default"
                or len(reshape_node.args) < 2
                or not isinstance(reshape_node.args[0], torch.fx.Node)
                or not isinstance(reshape_shape_arg, (list, tuple))
            ):
                continue
            source = reshape_node.args[0]
            source_shape = _rank_shape(source)
            if source_shape is None or len(source_shape) != 4 or int(source_shape[3]) != 1:
                continue
            expected_shape = _nchw_singleton_shape_from_nhwc(source)
            if expected_shape is None:
                continue
            if [int(v) for v in list(reshape_shape_arg)] != expected_shape:
                continue
            node_users = list(node.users)
            if (
                len(node_users) == 1
                and node_users[0].op == "call_function"
                and str(node_users[0].target) == "aten.contiguous.default"
            ):
                node_users[0].replace_all_uses_with(source)
            else:
                node.replace_all_uses_with(source)
            local_changed = True

        for node in list(graph.nodes):
            if (
                node.op != "call_function"
                or str(node.target) not in scalar_binary_targets
                or len(node.args) != 2
            ):
                continue
            nhwc_arg = None
            nchw_arg = None
            nhwc_index = -1
            if (
                isinstance(node.args[0], torch.fx.Node)
                and isinstance(node.args[1], torch.fx.Node)
                and _nchw_singleton_shape_from_nhwc(node.args[0]) is not None
            ):
                nhwc_arg = node.args[0]
                nchw_arg = node.args[1]
                nhwc_index = 0
            elif (
                isinstance(node.args[0], torch.fx.Node)
                and isinstance(node.args[1], torch.fx.Node)
                and _nchw_singleton_shape_from_nhwc(node.args[1]) is not None
            ):
                nhwc_arg = node.args[1]
                nchw_arg = node.args[0]
                nhwc_index = 1
            if nhwc_arg is None or nchw_arg is None:
                continue
            nchw_shape = _rank_shape(nchw_arg)
            target_shape = _nchw_singleton_shape_from_nhwc(nhwc_arg)
            if (
                nchw_shape is None
                or target_shape is None
                or len(nchw_shape) != 4
                or nchw_shape != target_shape
            ):
                continue
            with graph.inserting_before(node):
                reshaped_nhwc = graph.call_function(
                    torch.ops.aten.reshape.default,
                    args=(nhwc_arg, target_shape),
                    kwargs={},
                )
            reshaped_nhwc.meta = _shape_meta_from_node(nhwc_arg, target_shape)
            node.args = (
                (reshaped_nhwc, nchw_arg)
                if nhwc_index == 0
                else (nchw_arg, reshaped_nhwc)
            )
            node.meta = _shape_meta_from_node(nchw_arg, target_shape)
            local_changed = True

        for node in list(graph.nodes):
            if (
                node.op != "call_function"
                or str(node.target) != "aten.permute.default"
                or len(node.args) < 2
                or _normalize_perm(node.args[1]) != [0, 3, 1, 2]
            ):
                continue
            cat_node = node.args[0]
            cat_dim_arg = cat_node.args[1] if isinstance(cat_node, torch.fx.Node) and len(cat_node.args) >= 2 else None
            if (
                not isinstance(cat_node, torch.fx.Node)
                or cat_node.op != "call_function"
                or str(cat_node.target) != "aten.cat.default"
                or len(cat_node.args) < 2
                or not isinstance(cat_node.args[0], (list, tuple))
                or not isinstance(cat_dim_arg, (int, bool))
                or int(cat_dim_arg) != 3
            ):
                continue
            folded_inputs = []
            valid = True
            for cat_input in list(cat_node.args[0]):
                if (
                    not isinstance(cat_input, torch.fx.Node)
                    or cat_input.op != "call_function"
                    or str(cat_input.target) != "aten.contiguous.default"
                    or len(cat_input.args) < 1
                    or not isinstance(cat_input.args[0], torch.fx.Node)
                ):
                    valid = False
                    break
                permute_node = cat_input.args[0]
                if (
                    permute_node.op != "call_function"
                    or str(permute_node.target) != "aten.permute.default"
                    or len(permute_node.args) < 2
                    or _normalize_perm(permute_node.args[1]) != [0, 2, 3, 1]
                    or not isinstance(permute_node.args[0], torch.fx.Node)
                    or _rank_shape(permute_node.args[0]) is None
                    or len(_rank_shape(permute_node.args[0]) or []) != 4
                ):
                    valid = False
                    break
                folded_inputs.append(permute_node.args[0])
            if not valid:
                continue
            cat_node.args = (folded_inputs, 1)
            cat_input_shape = _rank_shape(folded_inputs[0])
            if cat_input_shape is not None and len(cat_input_shape) == 4:
                cat_shape = list(cat_input_shape)
                cat_shape[1] = sum(
                    int((_rank_shape(inp) or cat_input_shape)[1])
                    for inp in folded_inputs
                    if _rank_shape(inp) is not None and len(_rank_shape(inp) or []) == 4
                )
                cat_node.meta = _shape_meta_from_node(folded_inputs[0], cat_shape)
            else:
                cat_node.meta = dict(getattr(node, "meta", {}))
            node_users = list(node.users)
            if (
                len(node_users) == 1
                and node_users[0].op == "call_function"
                and str(node_users[0].target) == "aten.contiguous.default"
            ):
                node_users[0].replace_all_uses_with(cat_node)
            else:
                node.replace_all_uses_with(cat_node)
            local_changed = True

        for index_node in list(graph.nodes):
            index_spec_arg = index_node.args[1] if len(index_node.args) >= 2 else None
            cat_node = index_node.args[0] if len(index_node.args) >= 1 else None
            cat_dim_arg = cat_node.args[1] if isinstance(cat_node, torch.fx.Node) and len(cat_node.args) >= 2 else None
            if (
                index_node.op != "call_function"
                or str(index_node.target) != "aten.index.Tensor"
                or not isinstance(cat_node, torch.fx.Node)
                or cat_node.op != "call_function"
                or str(cat_node.target) != "aten.cat.default"
                or len(cat_node.args) < 2
                or not isinstance(cat_node.args[0], (list, tuple))
                or not isinstance(cat_dim_arg, (int, bool))
                or int(cat_dim_arg) != 3
                or not isinstance(index_spec_arg, (list, tuple))
            ):
                continue
            index_spec = list(index_spec_arg)
            if len(index_spec) != 4 or index_spec[:3] != [None, None, None]:
                continue
            channel_indices = index_spec[3]
            if not isinstance(channel_indices, (torch.fx.Node, torch.Tensor)):
                continue

            folded_inputs = []
            valid_inputs = True
            for cat_input in list(cat_node.args[0]):
                folded_input = _resolve_dim3_cat_input_as_cf(
                    cat_input,
                    insert_before=index_node,
                )
                if folded_input is None:
                    valid_inputs = False
                    break
                folded_inputs.append(folded_input)
            if not valid_inputs:
                continue

            index_users = list(index_node.users)
            if len(index_users) == 0:
                continue
            replacement_meta_node: Optional[torch.fx.Node] = None
            valid_users = True
            for user in index_users:
                if (
                    user.op != "call_function"
                    or str(user.target) != "aten.permute.default"
                    or len(user.args) < 2
                    or _normalize_perm(user.args[1]) != [0, 3, 1, 2]
                ):
                    valid_users = False
                    break
                replacement_meta_node = user
            if not valid_users or replacement_meta_node is None:
                continue

            cat_node.args = (folded_inputs, 1)
            cat_input_shape = _rank_shape(folded_inputs[0])
            if cat_input_shape is not None and len(cat_input_shape) == 4:
                cat_shape = list(cat_input_shape)
                cat_shape[1] = sum(
                    int((_rank_shape(inp) or cat_input_shape)[1])
                    for inp in folded_inputs
                    if _rank_shape(inp) is not None and len(_rank_shape(inp) or []) == 4
                )
                cat_node.meta = _shape_meta_from_node(folded_inputs[0], cat_shape)
            index_node.args = (cat_node, [None, channel_indices, None, None])
            replacement_shape = _rank_shape(replacement_meta_node)
            if replacement_shape is not None:
                index_node.meta = _shape_meta_from_node(replacement_meta_node, replacement_shape)
            else:
                index_node.meta = dict(getattr(replacement_meta_node, "meta", {}))
            for sibling_user in list(cat_node.users):
                sibling_index_spec_arg = sibling_user.args[1] if len(sibling_user.args) >= 2 else None
                if (
                    not isinstance(sibling_user, torch.fx.Node)
                    or sibling_user.op != "call_function"
                    or str(sibling_user.target) != "aten.index.Tensor"
                    or not isinstance(sibling_index_spec_arg, (list, tuple))
                ):
                    continue
                sibling_index_spec = list(sibling_index_spec_arg)
                if (
                    len(sibling_index_spec) == 4
                    and sibling_index_spec[0] is None
                    and sibling_index_spec[1] is None
                    and sibling_index_spec[2] is None
                    and sibling_index_spec[3] is not None
                ):
                    sibling_user.args = (cat_node, [None, sibling_index_spec[3], None, None])
                    sibling_users = list(sibling_user.users)
                    for sibling_perm_user in sibling_users:
                        if (
                            sibling_perm_user.op == "call_function"
                            and str(sibling_perm_user.target) == "aten.permute.default"
                            and len(sibling_perm_user.args) >= 2
                            and _normalize_perm(sibling_perm_user.args[1]) == [0, 3, 1, 2]
                        ):
                            perm_users = list(sibling_perm_user.users)
                            if (
                                len(perm_users) == 1
                                and perm_users[0].op == "call_function"
                                and str(perm_users[0].target) == "aten.contiguous.default"
                            ):
                                perm_users[0].replace_all_uses_with(sibling_user)
                            else:
                                sibling_perm_user.replace_all_uses_with(sibling_user)
            for user in index_users:
                user_users = list(user.users)
                if (
                    len(user_users) == 1
                    and user_users[0].op == "call_function"
                    and str(user_users[0].target) == "aten.contiguous.default"
                ):
                    user_users[0].replace_all_uses_with(index_node)
                else:
                    user.replace_all_uses_with(index_node)
            local_changed = True

        for node in list(graph.nodes):
            if (
                node.op != "call_function"
                or str(node.target) != "aten.permute.default"
                or len(node.args) < 2
                or _normalize_perm(node.args[1]) != [0, 3, 1, 2]
            ):
                continue
            cat_node = node.args[0]
            cat_dim_arg = cat_node.args[1] if isinstance(cat_node, torch.fx.Node) and len(cat_node.args) >= 2 else None
            if (
                not isinstance(cat_node, torch.fx.Node)
                or cat_node.op != "call_function"
                or str(cat_node.target) != "aten.cat.default"
                or len(cat_node.args) < 2
                or not isinstance(cat_node.args[0], (list, tuple))
                or not isinstance(cat_dim_arg, (int, bool))
                or int(cat_dim_arg) != 3
            ):
                continue
            folded_inputs = []
            valid = True
            for cat_input in list(cat_node.args[0]):
                if not isinstance(cat_input, torch.fx.Node):
                    valid = False
                    break
                nchw_shape = _nchw_singleton_shape_from_nhwc(cat_input)
                if nchw_shape is None:
                    valid = False
                    break
                with graph.inserting_before(cat_node):
                    reshaped_input = graph.call_function(
                        torch.ops.aten.reshape.default,
                        args=(cat_input, nchw_shape),
                        kwargs={},
                    )
                reshaped_input.meta = _shape_meta_from_node(cat_input, nchw_shape)
                folded_inputs.append(reshaped_input)
            if not valid:
                continue
            cat_node.args = (folded_inputs, 1)
            cat_input_shape = _rank_shape(folded_inputs[0])
            if cat_input_shape is not None and len(cat_input_shape) == 4:
                cat_shape = list(cat_input_shape)
                cat_shape[1] = sum(int((_rank_shape(inp) or cat_input_shape)[1]) for inp in folded_inputs)
                cat_node.meta = _shape_meta_from_node(folded_inputs[0], cat_shape)
            node_users = list(node.users)
            if (
                len(node_users) == 1
                and node_users[0].op == "call_function"
                and str(node_users[0].target) == "aten.contiguous.default"
            ):
                node_users[0].replace_all_uses_with(cat_node)
            else:
                node.replace_all_uses_with(cat_node)
            local_changed = True

        layout_preserving_unary_targets = {
            "aten.clamp.default",
            "aten.relu.default",
        }
        for node in list(graph.nodes):
            if (
                node.op != "call_function"
                or str(node.target) not in layout_preserving_unary_targets
                or len(node.args) < 1
                or not isinstance(node.args[0], torch.fx.Node)
            ):
                continue
            permute_node = node.args[0]
            if (
                permute_node.op != "call_function"
                or str(permute_node.target) != "aten.permute.default"
                or len(permute_node.args) < 2
                or _normalize_perm(permute_node.args[1]) != [0, 2, 3, 1]
                or not isinstance(permute_node.args[0], torch.fx.Node)
            ):
                continue
            source = permute_node.args[0]
            source_shape = _rank_shape(source)
            if source_shape is None or len(source_shape) != 4 or int(source_shape[1]) != 1:
                continue
            node_users = list(node.users)
            if len(node_users) == 0:
                continue
            folded_shape = list(source_shape)
            valid = True
            for user in node_users:
                reshape_arg = user.args[1] if len(user.args) >= 2 else None
                if (
                    user.op != "call_function"
                    or str(user.target) != "aten.reshape.default"
                    or len(user.args) < 2
                    or not isinstance(reshape_arg, (list, tuple))
                ):
                    valid = False
                    break
                reshape_shape = []
                for dim in reshape_arg:
                    if not isinstance(dim, (int, bool)):
                        valid = False
                        break
                    reshape_shape.append(int(dim))
                if not valid or reshape_shape != folded_shape:
                    valid = False
                    break
            if not valid:
                continue
            with graph.inserting_before(node):
                folded_unary = graph.call_function(
                    node.target,
                    args=(source, *tuple(node.args[1:])),
                    kwargs=dict(node.kwargs),
                )
            folded_unary.meta = _shape_meta_from_node(source, folded_shape)
            for user in node_users:
                user.replace_all_uses_with(folded_unary)
            local_changed = True

        for node in list(graph.nodes):
            if (
                node.op != "call_function"
                or str(node.target) != "aten.permute.default"
                or len(node.args) < 2
                or _normalize_perm(node.args[1]) != [0, 3, 1, 2]
            ):
                continue
            unary_node = node.args[0]
            if (
                not isinstance(unary_node, torch.fx.Node)
                or unary_node.op != "call_function"
                or str(unary_node.target) not in layout_preserving_unary_targets
                or len(unary_node.args) < 1
                or not isinstance(unary_node.args[0], torch.fx.Node)
            ):
                continue
            permute_source = unary_node.args[0]
            if (
                permute_source.op != "call_function"
                or str(permute_source.target) != "aten.permute.default"
                or len(permute_source.args) < 2
                or _normalize_perm(permute_source.args[1]) != [0, 2, 3, 1]
                or not isinstance(permute_source.args[0], torch.fx.Node)
            ):
                continue
            source = permute_source.args[0]
            with graph.inserting_before(unary_node):
                folded_unary = graph.call_function(
                    unary_node.target,
                    args=(source, *tuple(unary_node.args[1:])),
                    kwargs=dict(unary_node.kwargs),
                )
            folded_unary.meta = dict(getattr(node, "meta", {}))
            node_users = list(node.users)
            if (
                len(node_users) == 1
                and node_users[0].op == "call_function"
                and str(node_users[0].target) == "aten.contiguous.default"
            ):
                node_users[0].replace_all_uses_with(folded_unary)
            else:
                node.replace_all_uses_with(folded_unary)
            local_changed = True

        for split_node in list(graph.nodes):
            if (
                split_node.op != "call_function"
                or str(split_node.target) != "aten.tensor_split.sections"
                or len(split_node.args) < 3
                or int(split_node.args[1]) != 3
                or int(split_node.args[2]) != 3
            ):
                continue
            cat_node = split_node.args[0]
            cat_dim_arg = cat_node.args[1] if len(cat_node.args) >= 2 else None
            if (
                not isinstance(cat_node, torch.fx.Node)
                or cat_node.op != "call_function"
                or str(cat_node.target) != "aten.cat.default"
                or len(cat_node.args) < 2
                or not isinstance(cat_node.args[0], (list, tuple))
                or not isinstance(cat_dim_arg, (int, bool))
                or int(cat_dim_arg) != 3
                or len(cat_node.args[0]) != 3
            ):
                continue
            replacement_inputs = list(cat_node.args[0])
            getitems = []
            for user in list(split_node.users):
                if (
                    user.op != "call_function"
                    or str(user.target) != "<built-in function getitem>"
                    or len(user.args) < 2
                    or user.args[0] is not split_node
                ):
                    getitems = []
                    break
                getitems.append(user)
            if len(getitems) != 3:
                continue
            valid = True
            for getitem_node in getitems:
                split_index = int(getitem_node.args[1])
                if split_index < 0 or split_index >= 3:
                    valid = False
                    break
                getitem_node.replace_all_uses_with(replacement_inputs[split_index])
                local_changed = True
            if not valid:
                continue

        def _match_nchw_singleton_source(node: Any) -> Optional[torch.fx.Node]:
            if not isinstance(node, torch.fx.Node):
                return None
            if (
                node.op == "call_function"
                and str(node.target) == "<built-in function getitem>"
                and len(node.args) >= 2
                and isinstance(node.args[0], torch.fx.Node)
            ):
                split_node = node.args[0]
                split_dim_arg = split_node.args[2] if len(split_node.args) >= 3 else None
                if (
                    split_node.op == "call_function"
                    and str(split_node.target) == "aten.tensor_split.sections"
                    and len(split_node.args) >= 3
                    and isinstance(split_dim_arg, (int, bool))
                    and int(split_dim_arg) == 3
                    and isinstance(split_node.args[0], torch.fx.Node)
                ):
                    split_input = split_node.args[0]
                    if (
                        split_input.op == "call_function"
                        and str(split_input.target) == "aten.contiguous.default"
                        and len(split_input.args) >= 1
                        and isinstance(split_input.args[0], torch.fx.Node)
                    ):
                        permute_node = split_input.args[0]
                        if (
                            permute_node.op == "call_function"
                            and str(permute_node.target) == "aten.permute.default"
                            and len(permute_node.args) >= 2
                            and _normalize_perm(permute_node.args[1]) == [0, 2, 3, 1]
                            and isinstance(permute_node.args[0], torch.fx.Node)
                        ):
                            base = permute_node.args[0]
                            split_index_arg = node.args[1] if len(node.args) >= 2 else None
                            if not isinstance(split_index_arg, (int, bool)):
                                return None
                            split_index = int(split_index_arg)
                            with graph.inserting_before(node):
                                return graph.call_function(
                                    torch.ops.aten.slice.Tensor,
                                    args=(base, 1, split_index, split_index + 1),
                                    kwargs={},
                                )
            if (
                node.op == "call_function"
                and str(node.target) in layout_preserving_unary_targets
                and len(node.args) >= 1
                and isinstance(node.args[0], torch.fx.Node)
            ):
                permute_input = node.args[0]
                if (
                    permute_input.op == "call_function"
                    and str(permute_input.target) == "aten.permute.default"
                    and len(permute_input.args) >= 2
                    and _normalize_perm(permute_input.args[1]) == [0, 2, 3, 1]
                    and isinstance(permute_input.args[0], torch.fx.Node)
                ):
                    with graph.inserting_before(node):
                        return graph.call_function(
                            node.target,
                            args=(permute_input.args[0], *tuple(node.args[1:])),
                            kwargs=dict(node.kwargs),
                        )
            if (
                node.op == "call_function"
                and str(node.target) == "aten.permute.default"
                and len(node.args) >= 2
                and _normalize_perm(node.args[1]) == [0, 2, 3, 1]
                and isinstance(node.args[0], torch.fx.Node)
            ):
                return node.args[0]
            return None

        def _nchw_singleton_meta_like(node: torch.fx.Node) -> Dict[str, Any]:
            meta = dict(getattr(node, "meta", {}))
            meta_val = meta.get("val", None)
            if isinstance(meta_val, torch.Tensor) and len(list(meta_val.shape)) == 4:
                nhwc_shape = [int(v) for v in list(meta_val.shape)]
                if int(nhwc_shape[3]) == 1:
                    try:
                        meta["val"] = torch.empty(
                            (int(nhwc_shape[0]), 1, int(nhwc_shape[1]), int(nhwc_shape[2])),
                            dtype=meta_val.dtype,
                            device="meta",
                        )
                    except Exception:
                        pass
            return meta

        for split_node in list(graph.nodes):
            split_dim_arg = split_node.args[2] if len(split_node.args) >= 3 else None
            split_sections_arg = split_node.args[1] if len(split_node.args) >= 2 else None
            if (
                split_node.op != "call_function"
                or str(split_node.target) != "aten.tensor_split.sections"
                or len(split_node.args) < 3
                or not isinstance(split_sections_arg, (int, bool))
                or int(split_sections_arg) != 3
                or not isinstance(split_dim_arg, (int, bool))
                or int(split_dim_arg) != 3
                or not isinstance(split_node.args[0], torch.fx.Node)
            ):
                continue
            split_input = split_node.args[0]
            if (
                split_input.op != "call_function"
                or str(split_input.target) != "aten.contiguous.default"
                or len(split_input.args) < 1
                or not isinstance(split_input.args[0], torch.fx.Node)
            ):
                continue
            permute_node = split_input.args[0]
            if (
                permute_node.op != "call_function"
                or str(permute_node.target) != "aten.permute.default"
                or len(permute_node.args) < 2
                or _normalize_perm(permute_node.args[1]) != [0, 2, 3, 1]
                or not isinstance(permute_node.args[0], torch.fx.Node)
            ):
                continue
            base = permute_node.args[0]
            getitem_nodes: Dict[int, torch.fx.Node] = {}
            valid = True
            for user in list(split_node.users):
                split_index_arg = user.args[1] if len(user.args) >= 2 else None
                if (
                    user.op != "call_function"
                    or str(user.target) != "<built-in function getitem>"
                    or user.args[0] is not split_node
                    or not isinstance(split_index_arg, (int, bool))
                ):
                    valid = False
                    break
                split_index = int(split_index_arg)
                if split_index < 0 or split_index >= 3:
                    valid = False
                    break
                getitem_nodes[split_index] = user
            if not valid or len(getitem_nodes) != 3:
                continue
            split_node.args = (base, 3, 1)
            for getitem_node in getitem_nodes.values():
                getitem_node.meta = _nchw_singleton_meta_like(getitem_node)
                for user in list(getitem_node.users):
                    if (
                        user.op == "call_function"
                        and str(user.target) == "aten.permute.default"
                        and len(user.args) >= 2
                        and _normalize_perm(user.args[1]) == [0, 3, 1, 2]
                    ):
                        permute_users = list(user.users)
                        if (
                            len(permute_users) == 1
                            and permute_users[0].op == "call_function"
                            and str(permute_users[0].target) == "aten.contiguous.default"
                        ):
                            permute_users[0].replace_all_uses_with(getitem_node)
                        else:
                            user.replace_all_uses_with(getitem_node)
                        local_changed = True
                        continue
                    if (
                        user.op == "call_function"
                        and str(user.target) in scalar_binary_targets
                        and len(user.args) == 2
                        and (
                            (user.args[0] is getitem_node and _is_scalar_like(user.args[1]))
                            or (user.args[1] is getitem_node and _is_scalar_like(user.args[0]))
                        )
                    ):
                        user.meta = _nchw_singleton_meta_like(getitem_node)
                        user_users = list(user.users)
                        if len(user_users) == 1:
                            reshape_node = user_users[0]
                            reshape_arg = reshape_node.args[1] if len(reshape_node.args) >= 2 else None
                            reshape_shape = None
                            if (
                                reshape_node.op == "call_function"
                                and str(reshape_node.target) == "aten.reshape.default"
                                and len(reshape_node.args) >= 2
                                and isinstance(reshape_arg, (list, tuple))
                            ):
                                reshape_shape = []
                                for dim in reshape_arg:
                                    if not isinstance(dim, (int, bool)):
                                        reshape_shape = None
                                        break
                                    reshape_shape.append(int(dim))
                            user_shape = _rank_shape(user)
                            if user_shape is not None and reshape_shape == user_shape:
                                reshape_node.replace_all_uses_with(user)
                                local_changed = True
            local_changed = True

        for node in list(graph.nodes):
            if (
                node.op != "call_function"
                or str(node.target) != "aten.permute.default"
                or len(node.args) < 2
                or _normalize_perm(node.args[1]) != [0, 3, 1, 2]
            ):
                continue
            cat_node = node.args[0]
            cat_dim_arg = cat_node.args[1] if len(cat_node.args) >= 2 else None
            if (
                not isinstance(cat_node, torch.fx.Node)
                or cat_node.op != "call_function"
                or str(cat_node.target) != "aten.cat.default"
                or len(cat_node.args) < 2
                or not isinstance(cat_node.args[0], (list, tuple))
                or not isinstance(cat_dim_arg, (int, bool))
                or int(cat_dim_arg) != 3
            ):
                continue
            folded_inputs = []
            valid = True
            for cat_input in list(cat_node.args[0]):
                if not isinstance(cat_input, torch.fx.Node):
                    valid = False
                    break
                folded_input = _match_nchw_singleton_source(cat_input)
                if folded_input is None:
                    valid = False
                    break
                folded_input.meta = _nchw_singleton_meta_like(cat_input)
                folded_inputs.append(folded_input)
            if not valid:
                continue
            cat_node.args = (folded_inputs, 1)
            cat_node.meta = dict(getattr(node, "meta", {}))
            node_users = list(node.users)
            if (
                len(node_users) == 1
                and node_users[0].op == "call_function"
                and str(node_users[0].target) == "aten.contiguous.default"
            ):
                node_users[0].replace_all_uses_with(cat_node)
            else:
                node.replace_all_uses_with(cat_node)
            local_changed = True

        for node in list(graph.nodes):
            if (
                node.op != "call_function"
                or str(node.target) != "aten.reshape.default"
                or len(node.args) < 2
                or not isinstance(node.args[0], torch.fx.Node)
            ):
                continue
            reshape_shape = [int(v) for v in list(node.args[1])] if isinstance(node.args[1], (list, tuple)) else None
            if reshape_shape is None or len(reshape_shape) != 4 or int(reshape_shape[1]) != 1:
                continue
            binary_node = node.args[0]
            if (
                binary_node.op != "call_function"
                or str(binary_node.target) not in scalar_binary_targets
                or len(binary_node.args) != 2
            ):
                continue
            permute_input = None
            scalar_arg = None
            if isinstance(binary_node.args[0], torch.fx.Node) and _is_scalar_like(binary_node.args[1]):
                permute_input = binary_node.args[0]
                scalar_arg = binary_node.args[1]
            elif isinstance(binary_node.args[1], torch.fx.Node) and _is_scalar_like(binary_node.args[0]):
                permute_input = binary_node.args[1]
                scalar_arg = binary_node.args[0]
            if (
                not isinstance(permute_input, torch.fx.Node)
                or permute_input.op != "call_function"
                or str(permute_input.target) != "aten.permute.default"
                or len(permute_input.args) < 2
                or _normalize_perm(permute_input.args[1]) != [0, 2, 3, 1]
                or not isinstance(permute_input.args[0], torch.fx.Node)
            ):
                continue
            source = permute_input.args[0]
            with graph.inserting_before(binary_node):
                folded_binary = graph.call_function(
                    binary_node.target,
                    args=(
                        source if binary_node.args[0] is permute_input else scalar_arg,
                        scalar_arg if binary_node.args[0] is permute_input else source,
                    ),
                    kwargs=dict(binary_node.kwargs),
                )
            folded_binary.meta = dict(getattr(node, "meta", {}))
            node.replace_all_uses_with(folded_binary)
            local_changed = True

        for node in list(graph.nodes):
            if (
                node.op != "call_function"
                or str(node.target) != "aten.reshape.default"
                or len(node.args) < 2
                or not isinstance(node.args[0], torch.fx.Node)
                or not isinstance(node.args[1], (list, tuple))
            ):
                continue
            source_shape = _rank_shape(node.args[0])
            if source_shape is None:
                continue
            try:
                target_shape = [int(v) for v in list(node.args[1])]
            except Exception:
                continue
            if source_shape != target_shape:
                continue
            node.replace_all_uses_with(node.args[0])
            local_changed = True

        for node in list(graph.nodes):
            if (
                node.op != "call_function"
                or str(node.target) != "aten.slice.Tensor"
                or len(node.args) < 4
                or not isinstance(node.args[0], torch.fx.Node)
            ):
                continue
            source_shape = _rank_shape(node.args[0])
            if source_shape is None or len(source_shape) != 4:
                continue
            try:
                dim = int(node.args[1])
                start = int(node.args[2])
                end = int(node.args[3])
            except Exception:
                continue
            if dim != 1 or end - start != 1:
                continue
            meta_val = getattr(node, "meta", {}).get("val", None)
            if isinstance(meta_val, torch.Tensor) and list(meta_val.shape) == [source_shape[0], 1, source_shape[2], source_shape[3]]:
                continue
            meta = dict(getattr(node, "meta", {}))
            if isinstance(meta_val, torch.Tensor):
                try:
                    meta["val"] = torch.empty(
                        (int(source_shape[0]), 1, int(source_shape[2]), int(source_shape[3])),
                        dtype=meta_val.dtype,
                        device="meta",
                    )
                except Exception:
                    pass
            node.meta = meta
            local_changed = True

        for split_node in list(graph.nodes):
            split_sections_arg = split_node.args[1] if len(split_node.args) >= 2 else None
            split_dim_arg = split_node.args[2] if len(split_node.args) >= 3 else None
            if (
                split_node.op != "call_function"
                or str(split_node.target) != "aten.tensor_split.sections"
                or len(split_node.args) < 3
                or not isinstance(split_node.args[0], torch.fx.Node)
                or not isinstance(split_sections_arg, (int, bool))
                or not isinstance(split_dim_arg, (int, bool))
            ):
                continue
            source_shape = _rank_shape(split_node.args[0])
            if source_shape is None:
                continue
            split_dim = int(split_dim_arg)
            if split_dim < 0:
                split_dim += len(source_shape)
            if split_dim < 0 or split_dim >= len(source_shape):
                continue
            section_count = int(split_sections_arg)
            if section_count <= 0:
                continue
            dim_size = int(source_shape[split_dim])
            if dim_size < 0:
                continue
            base_extent, remainder = divmod(dim_size, section_count)
            getitem_users = []
            valid = True
            for user in list(split_node.users):
                split_index_arg = user.args[1] if len(user.args) >= 2 else None
                if (
                    user.op != "call_function"
                    or str(user.target) != "<built-in function getitem>"
                    or user.args[0] is not split_node
                    or not isinstance(split_index_arg, (int, bool))
                ):
                    valid = False
                    break
                split_index = int(split_index_arg)
                if split_index < 0 or split_index >= section_count:
                    valid = False
                    break
                getitem_users.append((split_index, user))
            if not valid:
                continue
            offset = 0
            split_bounds = []
            for split_index in range(section_count):
                extent = base_extent + (1 if split_index < remainder else 0)
                split_bounds.append((offset, offset + extent))
                offset += extent
            if getitem_users:
                for split_index, user in getitem_users:
                    start, end = split_bounds[split_index]
                    output_shape = list(source_shape)
                    output_shape[split_dim] = end - start
                    with graph.inserting_before(user):
                        slice_node = graph.call_function(
                            torch.ops.aten.slice.Tensor,
                            args=(split_node.args[0], split_dim, start, end),
                            kwargs={},
                        )
                    slice_node.meta = _shape_meta_from_node(split_node.args[0], output_shape)
                    user.replace_all_uses_with(slice_node)
                    local_changed = True
                continue
            for user in list(split_node.users):
                split_index_arg = user.args[1] if len(user.args) >= 2 else None
                if (
                    user.op != "call_function"
                    or str(user.target) != "<built-in function getitem>"
                    or user.args[0] is not split_node
                    or not isinstance(split_index_arg, (int, bool))
                ):
                    continue
                split_index = int(split_index_arg)
                if split_index < 0 or split_index >= section_count:
                    continue
                extent = base_extent + (1 if split_index < remainder else 0)
                output_shape = list(source_shape)
                output_shape[split_dim] = extent
                current_shape = getattr(getattr(user, "meta", {}).get("val", None), "shape", None)
                if current_shape is not None and list(current_shape) == output_shape:
                    continue
                user.meta = _shape_meta_from_node(split_node.args[0], output_shape)
                local_changed = True

        if local_changed:
            graph.eliminate_dead_code()
            graph.lint()
            graph_module.recompile()
        return local_changed

    while True:
        pass_changed = _run_one_pass()
        repair_changed = _repair_inferred_meta_shapes()
        if pass_changed or repair_changed:
            changed = True
        if not pass_changed and not repair_changed:
            break

    if not changed:
        return

    graph.eliminate_dead_code()
    graph.lint()
    graph_module.recompile()

    output_node = next((node for node in graph.nodes if node.op == "output"), None)
    if output_node is not None and len(output_node.args) > 0:
        output_args = output_node.args[0]
        if not isinstance(output_args, (list, tuple)):
            output_args = (output_args,)
        output_specs = []
        for arg in output_args:
            if not isinstance(arg, torch.fx.Node):
                output_specs = []
                break
            output_specs.append(
                torch.export.graph_signature.OutputSpec(
                    kind=torch.export.graph_signature.OutputKind.USER_OUTPUT,
                    arg=torch.export.graph_signature.TensorArgument(name=str(arg.name)),
                    target=None,
                )
            )
        if output_specs:
            exported_program._graph_signature = dataclasses.replace(
                exported_program.graph_signature,
                input_specs=mutable_input_specs,
                output_specs=output_specs,
            )
    elif list(getattr(exported_program.graph_signature, "input_specs", ())) != mutable_input_specs:
        exported_program._graph_signature = dataclasses.replace(
            exported_program.graph_signature,
            input_specs=mutable_input_specs,
        )

    with tempfile.NamedTemporaryFile(
        prefix="onnx2tf_exported_program_post_",
        suffix=".pt2",
        delete=False,
        dir=str(archive_path.parent),
    ) as tmp_file:
        temp_archive_path = Path(tmp_file.name)
    try:
        torch.export.save(exported_program, str(temp_archive_path))
        temp_archive_path.replace(archive_path)
    except Exception:
        temp_archive_path.unlink(missing_ok=True)
        raise
