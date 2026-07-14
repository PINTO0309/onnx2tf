from __future__ import annotations


_EXPORTED_PROGRAM_CHILD_SCRIPT = """
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import torch

package_path = Path(sys.argv[1])
package_init_path = package_path / "__init__.py"
inputs_path = Path(sys.argv[2])
exported_program_path = Path(sys.argv[3])

module_name = (
    "_onnx2tf_generated_exported_program_child_"
    + hashlib.sha256(str(package_path.resolve()).encode("utf-8")).hexdigest()
)
if module_name in sys.modules:
    del sys.modules[module_name]
spec = importlib.util.spec_from_file_location(
    module_name,
    str(package_init_path),
    submodule_search_locations=[str(package_path)],
)
if spec is None or spec.loader is None:
    raise ImportError(
        f"Failed to create an import spec for the generated PyTorch package. path={package_init_path}"
    )
module = importlib.util.module_from_spec(spec)
sys.modules[module_name] = module
spec.loader.exec_module(module)
if not hasattr(module, "load_model"):
    raise RuntimeError(
        "Generated native PyTorch package does not expose load_model(). "
        f"package_dir={package_path}"
    )
payload = torch.load(str(inputs_path), map_location="cpu")
example_inputs = tuple(payload["inputs"])
model = module.load_model(device="cpu", eval_mode=True)
if hasattr(model, "cpu"):
    model = model.cpu()

def _prune_alias_nodes(exported_program):
    graph_module = getattr(exported_program, "graph_module", None)
    if graph_module is None:
        return exported_program
    graph = graph_module.graph
    graph_signature = getattr(exported_program, "graph_signature", None)
    user_output_names = set()
    if graph_signature is not None:
        try:
            user_output_names = {str(name) for name in list(graph_signature.user_outputs)}
        except Exception:
            user_output_names = set()
    changed = False
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
            changed = True
    if changed:
        graph.eliminate_dead_code()
        graph.lint()
        graph_module.recompile()
    return exported_program

def _fold_singleton_channel_split_permute_bridges_in_exported_program(exported_program):
    graph_module = getattr(exported_program, "graph_module", None)
    if graph_module is None:
        return exported_program
    graph = graph_module.graph
    changed = False

    def _normalize_perm(arg):
        if not isinstance(arg, (list, tuple)):
            return None
        return [int(v) for v in arg]

    def _rank4_shape(node):
        if not isinstance(node, torch.fx.Node):
            return None
        meta_val = getattr(node, "meta", {}).get("val", None)
        if isinstance(meta_val, torch.Tensor):
            shape = [int(v) for v in list(meta_val.shape)]
            return shape if len(shape) == 4 else None
        if node.op == "get_attr" and isinstance(node.target, str):
            tensor = getattr(graph_module, node.target, None)
            if isinstance(tensor, torch.Tensor):
                shape = [int(v) for v in list(tensor.shape)]
                return shape if len(shape) == 4 else None
        return None

    def _shape_meta_like(node, shape):
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

    def _shape_list_from_arg(arg):
        if not isinstance(arg, (list, tuple)):
            return None
        try:
            return [int(v) for v in list(arg)]
        except Exception:
            return None

    def _is_scalar_like(value):
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

    def _sync_user_output_signature():
        graph_signature = getattr(exported_program, "graph_signature", None)
        if graph_signature is None:
            return
        output_node = None
        for maybe_output in graph.nodes:
            if maybe_output.op == "output":
                output_node = maybe_output
                break
        if output_node is None or len(output_node.args) == 0:
            return
        output_args = output_node.args[0]
        if not isinstance(output_args, (list, tuple)):
            output_args = (output_args,)
        output_specs = []
        for arg in output_args:
            if not isinstance(arg, torch.fx.Node):
                return
            output_specs.append(
                torch.export.graph_signature.OutputSpec(
                    kind=torch.export.graph_signature.OutputKind.USER_OUTPUT,
                    arg=torch.export.graph_signature.TensorArgument(name=str(arg.name)),
                    target=None,
                )
            )
        try:
            exported_program._graph_signature = dataclasses.replace(
                graph_signature,
                output_specs=output_specs,
            )
        except Exception:
            pass

    def _shape_meta_like(node, shape):
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

    while True:
        local_changed = False
        for split_node in list(graph.nodes):
            if (
                split_node.op != "call_function"
                or str(split_node.target) != "aten.tensor_split.sections"
                or len(split_node.args) < 3
            ):
                continue
            split_input = split_node.args[0]
            if not (
                isinstance(split_input, torch.fx.Node)
                and split_input.op == "call_function"
                and str(split_input.target) == "aten.contiguous.default"
                and len(split_input.args) >= 1
                and isinstance(split_input.args[0], torch.fx.Node)
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
            if int(split_node.args[2]) != 3:
                continue
            base_source = permute_node.args[0]
            base_shape = _rank4_shape(base_source)
            permuted_shape = _rank4_shape(split_input)
            if (
                base_shape is None
                or permuted_shape is None
                or int(base_shape[1]) <= 0
                or int(permuted_shape[3]) != 1
                or int(base_shape[1]) != int(permuted_shape[3]) * len(list(split_node.users))
            ):
                pass
            if base_shape is None or len(base_shape) != 4:
                continue
            getitem_users = []
            for user in list(split_node.users):
                if (
                    user.op != "call_function"
                    or str(user.target) != "<built-in function getitem>"
                    or len(user.args) < 2
                    or user.args[0] is not split_node
                ):
                    getitem_users = []
                    break
                item_shape = _rank4_shape(user)
                if item_shape is None or item_shape != [int(base_shape[0]), int(base_shape[2]), int(base_shape[3]), 1]:
                    getitem_users = []
                    break
                getitem_users.append(user)
            if len(getitem_users) == 0:
                continue

            with graph.inserting_before(split_node):
                new_split = graph.call_function(
                    torch.ops.aten.tensor_split.sections,
                    args=(base_source, 1, 1),
                    kwargs=dict(split_node.kwargs),
                )
            new_split.meta = dict(getattr(split_node, "meta", {}))

            for getitem_node in list(getitem_users):
                split_index = int(getitem_node.args[1])
                with graph.inserting_before(getitem_node):
                    new_getitem = graph.call_function(
                        getitem_node.target,
                        args=(new_split, split_index),
                        kwargs=dict(getitem_node.kwargs),
                    )
                cf_shape = [int(base_shape[0]), 1, int(base_shape[2]), int(base_shape[3])]
                nhwc_shape = [int(base_shape[0]), int(base_shape[2]), int(base_shape[3]), 1]
                new_getitem.meta = _shape_meta_like(getitem_node, cf_shape)
                nhwc_view = None

                def _get_nhwc_view(insert_before_node):
                    nonlocal nhwc_view
                    if nhwc_view is not None:
                        return nhwc_view
                    with graph.inserting_before(insert_before_node):
                        nhwc_view = graph.call_function(
                            torch.ops.aten.reshape.default,
                            args=(new_getitem, nhwc_shape),
                            kwargs={},
                        )
                    nhwc_view.meta = _shape_meta_like(getitem_node, nhwc_shape)
                    return nhwc_view

                for user in list(getitem_node.users):
                    if (
                        user.op == "call_function"
                        and str(user.target) == "aten.permute.default"
                        and len(user.args) >= 2
                        and user.args[0] is getitem_node
                        and _normalize_perm(user.args[1]) == [0, 3, 1, 2]
                    ):
                        permute_users = list(user.users)
                        if (
                            len(permute_users) == 1
                            and permute_users[0].op == "call_function"
                            and str(permute_users[0].target) == "aten.contiguous.default"
                        ):
                            permute_users[0].replace_all_uses_with(new_getitem)
                        else:
                            user.replace_all_uses_with(new_getitem)
                        local_changed = True
                        changed = True
                        continue
                    if (
                        user.op == "call_function"
                        and str(user.target) == "aten.reshape.default"
                        and len(user.args) >= 2
                        and _shape_list_from_arg(user.args[1]) == cf_shape
                    ):
                        user.replace_all_uses_with(new_getitem)
                        local_changed = True
                        changed = True
                        continue
                    if (
                        user.op == "call_function"
                        and str(user.target) in {
                            "aten.add.Tensor",
                            "aten.div.Tensor",
                            "aten.mul.Tensor",
                            "aten.sub.Tensor",
                        }
                        and len(user.args) == 2
                        and (
                            (user.args[0] is getitem_node and _is_scalar_like(user.args[1]))
                            or (user.args[1] is getitem_node and _is_scalar_like(user.args[0]))
                        )
                    ):
                        user.replace_input_with(getitem_node, new_getitem)
                        user.meta = _shape_meta_like(user, cf_shape)
                        local_changed = True
                        changed = True
                        continue
                    replacement = _get_nhwc_view(user)
                    user.replace_input_with(getitem_node, replacement)
                    local_changed = True
                    changed = True

            for getitem_node in list(getitem_users):
                if len(list(getitem_node.users)) == 0:
                    graph.erase_node(getitem_node)
            if len(list(split_node.users)) == 0:
                graph.erase_node(split_node)

            if local_changed:
                graph.eliminate_dead_code()
                graph.lint()
                graph_module.recompile()
                break

        if not local_changed:
            break

    if changed:
        graph.eliminate_dead_code()
        graph.lint()
        graph_module.recompile()
        _sync_user_output_signature()
    return exported_program

def _fold_inverse_permute_round_trips(exported_program):
    graph_module = getattr(exported_program, "graph_module", None)
    if graph_module is None:
        return exported_program
    graph = graph_module.graph
    graph_signature = getattr(exported_program, "graph_signature", None)
    user_output_names = set()
    if graph_signature is not None:
        try:
            user_output_names = {str(name) for name in list(graph_signature.user_outputs)}
        except Exception:
            user_output_names = set()
    changed = False

    def _normalize_perm(arg):
        if not isinstance(arg, (list, tuple)):
            return None
        return [int(v) for v in arg]

    def _is_user_output_node(node):
        return isinstance(node, torch.fx.Node) and str(node.name) in user_output_names

    def _inverse_perm(perm):
        inverse = [0] * len(perm)
        for idx, value in enumerate(perm):
            inverse[int(value)] = int(idx)
        return inverse

    def _convert_nhwc_pad_to_nchw_pad(pad_values):
        values = [int(v) for v in list(pad_values)]
        if len(values) % 2 != 0 or len(values) > 8:
            return None
        nhwc_inner_to_outer = ["C", "W", "H", "N"]
        nchw_inner_to_outer = ["W", "H", "C", "N"]
        semantic_pairs = {name: [0, 0] for name in nhwc_inner_to_outer}
        pair_count = len(values) // 2
        for idx in range(pair_count):
            semantic_pairs[nhwc_inner_to_outer[idx]] = [
                values[idx * 2],
                values[idx * 2 + 1],
            ]
        output_pairs = [semantic_pairs[name] for name in nchw_inner_to_outer]
        last_nonzero = -1
        for idx, pair in enumerate(output_pairs):
            if pair != [0, 0]:
                last_nonzero = idx
        if last_nonzero < 0:
            return []
        flattened = []
        for pair in output_pairs[: last_nonzero + 1]:
            flattened.extend(pair)
        return flattened

    def _match_permute_chain_source(node, perm):
        if not isinstance(node, torch.fx.Node):
            return None
        if (
            node.op == "call_function"
            and str(node.target) == "aten.permute.default"
            and len(node.args) >= 2
            and _normalize_perm(node.args[1]) == perm
            and isinstance(node.args[0], torch.fx.Node)
        ):
            return node.args[0]
        if (
            node.op == "call_function"
            and str(node.target) == "aten.contiguous.default"
            and len(node.args) >= 1
            and isinstance(node.args[0], torch.fx.Node)
        ):
            input_node = node.args[0]
            if (
                input_node.op == "call_function"
                and str(input_node.target) == "aten.permute.default"
                and len(input_node.args) >= 2
                and _normalize_perm(input_node.args[1]) == perm
                and isinstance(input_node.args[0], torch.fx.Node)
            ):
                return input_node.args[0]
        return None

    def _match_binary_input_source(node, perm):
        return _match_permute_chain_source(node, perm)

    def _match_cat_input_fold_source(node):
        if not isinstance(node, torch.fx.Node):
            return None
        source = _match_permute_chain_source(node, [0, 2, 3, 1])
        if source is not None:
            return source
        direct_shape = _rank4_shape(node)
        if direct_shape is not None and int(direct_shape[3]) == 1:
            return ("reshape", node, [int(direct_shape[0]), 1, int(direct_shape[1]), int(direct_shape[2])])
        if (
            node.op == "call_function"
            and str(node.target) == "aten.pad.default"
            and len(node.args) >= 2
            and isinstance(node.args[0], torch.fx.Node)
        ):
            pad_source = _match_permute_chain_source(node.args[0], [0, 2, 3, 1])
            if pad_source is None:
                return None
            nchw_pad = _convert_nhwc_pad_to_nchw_pad(node.args[1])
            if nchw_pad is None:
                return None
            return (pad_source, nchw_pad, tuple(node.args[2:]), dict(node.kwargs))
        return None

    def _rank4_shape(node):
        if not isinstance(node, torch.fx.Node):
            return None
        meta_val = getattr(node, "meta", {}).get("val", None)
        if isinstance(meta_val, torch.Tensor):
            shape = [int(v) for v in list(meta_val.shape)]
            return shape if len(shape) == 4 else None
        if node.op == "get_attr" and isinstance(node.target, str):
            tensor = getattr(graph_module, node.target, None)
            if isinstance(tensor, torch.Tensor):
                shape = [int(v) for v in list(tensor.shape)]
                return shape if len(shape) == 4 else None
        return None

    def _match_nhwc_singleton_view_source(node):
        if not isinstance(node, torch.fx.Node):
            return None
        if (
            node.op == "call_function"
            and str(node.target) == "aten.reshape.default"
            and len(node.args) >= 2
            and isinstance(node.args[0], torch.fx.Node)
            and isinstance(node.args[1], (list, tuple))
        ):
            input_shape = _rank4_shape(node.args[0])
            view_shape = [int(v) for v in list(node.args[1])]
            if (
                input_shape is not None
                and view_shape == [int(input_shape[0]), int(input_shape[2]), int(input_shape[3]), 1]
                and int(input_shape[1]) == 1
            ):
                return node.args[0]
        return _match_permute_chain_source(node, [0, 2, 3, 1])

    for node in list(graph.nodes):
        if (
            node.op != "call_function"
            or str(node.target) != "aten.permute.default"
            or len(node.args) < 2
        ):
            continue
        source = node.args[0]
        if not isinstance(source, torch.fx.Node):
            continue
        perm = _normalize_perm(node.args[1])
        if perm is None:
            continue
        root_replacement = source
        root_users = list(node.users)
        if len(root_users) == 1 and (
            root_users[0].op == "call_function"
            and str(root_users[0].target) == "aten.contiguous.default"
        ):
            branch_input = root_users[0]
        else:
            branch_input = node
        branch_users = list(branch_input.users)
        if len(branch_users) == 1:
            inverse_node = branch_users[0]
            if (
                inverse_node.op == "call_function"
                and str(inverse_node.target) == "aten.permute.default"
                and len(inverse_node.args) >= 2
            ):
                inverse_perm = _normalize_perm(inverse_node.args[1])
                if inverse_perm is not None and inverse_perm == _inverse_perm(perm):
                    inverse_users = list(inverse_node.users)
                    if (
                        len(inverse_users) == 1
                        and inverse_users[0].op == "call_function"
                        and str(inverse_users[0].target) == "aten.contiguous.default"
                    ):
                        if _is_user_output_node(inverse_users[0]):
                            continue
                        inverse_users[0].replace_all_uses_with(root_replacement)
                    else:
                        if _is_user_output_node(inverse_node):
                            continue
                        inverse_node.replace_all_uses_with(root_replacement)
                    changed = True

    for node in list(graph.nodes):
        if (
            node.op != "call_function"
            or str(node.target) != "aten.permute.default"
            or len(node.args) < 2
        ):
            continue
        inverse_perm = _normalize_perm(node.args[1])
        if inverse_perm is None:
            continue
        source = _match_permute_chain_source(node.args[0], _inverse_perm(inverse_perm))
        if source is None:
            continue
        node_users = list(node.users)
        if (
            len(node_users) == 1
            and node_users[0].op == "call_function"
            and str(node_users[0].target) == "aten.contiguous.default"
        ):
            if _is_user_output_node(node_users[0]):
                continue
            node_users[0].replace_all_uses_with(source)
        else:
            if _is_user_output_node(node):
                continue
            node.replace_all_uses_with(source)
        changed = True

    for node in list(graph.nodes):
        if (
            node.op != "call_function"
            or str(node.target) != "aten.permute.default"
            or len(node.args) < 2
            or _normalize_perm(node.args[1]) != [0, 3, 1, 2]
        ):
            continue
        cat_node = node.args[0]
        if (
            not isinstance(cat_node, torch.fx.Node)
            or cat_node.op != "call_function"
            or str(cat_node.target) != "aten.cat.default"
            or len(cat_node.args) < 1
        ):
            continue
        cat_inputs_arg = cat_node.args[0]
        if not isinstance(cat_inputs_arg, (list, tuple)):
            continue
        cat_dim = None
        if len(cat_node.args) >= 2:
            cat_dim = int(cat_node.args[1])
        elif "dim" in cat_node.kwargs:
            cat_dim = int(cat_node.kwargs["dim"])
        if cat_dim != 3:
            continue
        folded_inputs = []
        for cat_input in list(cat_inputs_arg):
            folded_source = _match_cat_input_fold_source(cat_input)
            if folded_source is None:
                folded_inputs = []
                break
            if isinstance(folded_source, tuple):
                if len(folded_source) == 3 and folded_source[0] == "reshape":
                    _, source_node, reshaped_shape = folded_source
                    with graph.inserting_before(cat_node):
                        folded_reshape = graph.call_function(
                            torch.ops.aten.reshape.default,
                            args=(source_node, reshaped_shape),
                            kwargs={},
                        )
                    folded_reshape.meta = _shape_meta_like(cat_input, reshaped_shape)
                    folded_inputs.append(folded_reshape)
                else:
                    source_node, nchw_pad, pad_args_tail, pad_kwargs = folded_source
                    with graph.inserting_before(cat_node):
                        folded_pad = graph.call_function(
                            torch.ops.aten.pad.default,
                            args=(source_node, nchw_pad, *pad_args_tail),
                            kwargs=pad_kwargs,
                        )
                    folded_pad.meta = dict(getattr(cat_input, "meta", {}))
                    folded_inputs.append(folded_pad)
            else:
                folded_inputs.append(folded_source)
        if len(folded_inputs) != len(list(cat_inputs_arg)):
            continue
        folded_cat_kwargs = dict(cat_node.kwargs)
        folded_cat_args = (folded_inputs, 1)
        if "dim" in folded_cat_kwargs:
            folded_cat_kwargs["dim"] = 1
            folded_cat_args = (folded_inputs,)
        cat_node.args = folded_cat_args
        cat_node.kwargs = folded_cat_kwargs
        cat_node.meta = dict(getattr(node, "meta", {}))
        inverse_users = list(node.users)
        if (
            len(inverse_users) == 1
            and inverse_users[0].op == "call_function"
            and str(inverse_users[0].target) == "aten.contiguous.default"
        ):
            if _is_user_output_node(inverse_users[0]):
                continue
            inverse_users[0].replace_all_uses_with(cat_node)
        else:
            if _is_user_output_node(node):
                continue
            node.replace_all_uses_with(cat_node)
        changed = True

    for node in list(graph.nodes):
        if (
            node.op != "call_function"
            or str(node.target) != "aten.permute.default"
            or len(node.args) < 2
            or _normalize_perm(node.args[1]) != [0, 3, 1, 2]
        ):
            continue
        pad_node = node.args[0]
        if not isinstance(pad_node, torch.fx.Node):
            continue
        if (
            pad_node.op != "call_function"
            or str(pad_node.target) != "aten.pad.default"
            or len(pad_node.args) < 4
        ):
            continue
        source = _match_permute_chain_source(pad_node.args[0], [0, 2, 3, 1])
        if source is None:
            continue
        pad_values = list(pad_node.args[1])
        if len(pad_values) != 6:
            continue
        if [int(v) for v in pad_values[:2]] != [0, 0]:
            continue
        cf_pad = [int(pad_values[2]), int(pad_values[3]), int(pad_values[4]), int(pad_values[5])]
        with graph.inserting_before(pad_node):
            folded_pad = graph.call_function(
                pad_node.target,
                args=(source, cf_pad, *tuple(pad_node.args[2:])),
                kwargs=dict(pad_node.kwargs),
            )
        folded_pad.meta = dict(getattr(node, "meta", {}))
        node_users = list(node.users)
        if (
            len(node_users) == 1
            and node_users[0].op == "call_function"
            and str(node_users[0].target) == "aten.contiguous.default"
        ):
            if _is_user_output_node(node_users[0]):
                continue
            node_users[0].replace_all_uses_with(folded_pad)
        else:
            if _is_user_output_node(node):
                continue
            node.replace_all_uses_with(folded_pad)
        changed = True

    for node in list(graph.nodes):
        if (
            node.op != "call_function"
            or str(node.target) != "aten.permute.default"
            or len(node.args) < 2
            or _normalize_perm(node.args[1]) != [0, 2, 3, 1]
        ):
            continue
        source = node.args[0]
        if not isinstance(source, torch.fx.Node):
            continue
        node_users = list(node.users)
        if len(node_users) != 1:
            continue
        contiguous_node = node_users[0]
        if (
            contiguous_node.op != "call_function"
            or str(contiguous_node.target) != "aten.contiguous.default"
        ):
            continue
        contiguous_users = list(contiguous_node.users)
        if len(contiguous_users) != 1:
            continue
        sum_node = contiguous_users[0]
        if (
            sum_node.op != "call_function"
            or str(sum_node.target) != "aten.sum.dim_IntList"
            or len(sum_node.args) < 3
            or list(sum_node.args[1]) != [3]
            or bool(sum_node.args[2]) is not True
        ):
            continue
        sum_users = list(sum_node.users)
        if len(sum_users) != 1:
            continue
        sigmoid_node = sum_users[0]
        if (
            sigmoid_node.op != "call_function"
            or str(sigmoid_node.target) != "aten.sigmoid.default"
        ):
            continue
        sigmoid_users = list(sigmoid_node.users)
        if len(sigmoid_users) != 1:
            continue
        inverse_node = sigmoid_users[0]
        if (
            inverse_node.op != "call_function"
            or str(inverse_node.target) != "aten.permute.default"
            or len(inverse_node.args) < 2
            or _normalize_perm(inverse_node.args[1]) != [0, 3, 1, 2]
        ):
            continue
        with graph.inserting_before(node):
            folded_sum = graph.call_function(
                sum_node.target,
                args=(source, [1], True),
                kwargs=dict(sum_node.kwargs),
            )
            folded_sigmoid = graph.call_function(
                sigmoid_node.target,
                args=(folded_sum,),
                kwargs=dict(sigmoid_node.kwargs),
            )
        folded_sum.meta = dict(getattr(sum_node, "meta", {}))
        folded_sigmoid.meta = dict(getattr(inverse_node, "meta", {}))
        inverse_users = list(inverse_node.users)
        if (
            len(inverse_users) == 1
            and inverse_users[0].op == "call_function"
            and str(inverse_users[0].target) == "aten.contiguous.default"
        ):
            if _is_user_output_node(inverse_users[0]):
                continue
            inverse_users[0].replace_all_uses_with(folded_sigmoid)
        else:
            if _is_user_output_node(inverse_node):
                continue
            inverse_node.replace_all_uses_with(folded_sigmoid)
        changed = True

    for node in list(graph.nodes):
        if (
            node.op != "call_function"
            or str(node.target) != "aten.permute.default"
            or len(node.args) < 2
            or _normalize_perm(node.args[1]) != [0, 3, 1, 2]
        ):
            continue
        source = node.args[0]
        if not isinstance(source, torch.fx.Node):
            continue
        source_shape = None
        source_meta_val = getattr(source, "meta", {}).get("val", None)
        if isinstance(source_meta_val, torch.Tensor):
            source_shape = [int(v) for v in list(source_meta_val.shape)]
        elif source.op == "get_attr" and isinstance(source.target, str):
            source_tensor = getattr(graph_module, source.target, None)
            if isinstance(source_tensor, torch.Tensor):
                source_shape = [int(v) for v in list(source_tensor.shape)]
        if source_shape is None or len(source_shape) != 4:
            continue
        if int(source_shape[3]) != 1:
            continue
        reshaped_shape = [int(source_shape[0]), int(source_shape[3]), int(source_shape[1]), int(source_shape[2])]
        with graph.inserting_before(node):
            folded_reshape = graph.call_function(
                torch.ops.aten.reshape.default,
                args=(source, reshaped_shape),
                kwargs={},
            )
        folded_reshape.meta = dict(getattr(node, "meta", {}))
        node_users = list(node.users)
        if (
            len(node_users) == 1
            and node_users[0].op == "call_function"
            and str(node_users[0].target) == "aten.contiguous.default"
        ):
            if _is_user_output_node(node_users[0]):
                continue
            node_users[0].replace_all_uses_with(folded_reshape)
        else:
            if _is_user_output_node(node):
                continue
            node.replace_all_uses_with(folded_reshape)
        changed = True

    for node in list(graph.nodes):
        if (
            node.op != "call_function"
            or str(node.target) != "aten.permute.default"
            or len(node.args) < 2
            or _normalize_perm(node.args[1]) != [0, 3, 1, 2]
        ):
            continue
        mul_node = node.args[0]
        if (
            not isinstance(mul_node, torch.fx.Node)
            or mul_node.op != "call_function"
            or str(mul_node.target) != "aten.mul.Tensor"
            or len(mul_node.args) != 2
        ):
            continue
        nhwc_source = None
        scalar_arg = None
        if isinstance(mul_node.args[0], torch.fx.Node):
            nhwc_source = _match_nhwc_singleton_view_source(mul_node.args[0])
            scalar_arg = mul_node.args[1]
        if nhwc_source is None and isinstance(mul_node.args[1], torch.fx.Node):
            nhwc_source = _match_nhwc_singleton_view_source(mul_node.args[1])
            scalar_arg = mul_node.args[0]
        if nhwc_source is None or isinstance(scalar_arg, torch.fx.Node):
            continue
        with graph.inserting_before(mul_node):
            folded_mul = graph.call_function(
                mul_node.target,
                args=(
                    nhwc_source if mul_node.args[0] is not scalar_arg else scalar_arg,
                    scalar_arg if mul_node.args[0] is not scalar_arg else nhwc_source,
                ),
                kwargs=dict(mul_node.kwargs),
            )
        folded_mul.meta = dict(getattr(node, "meta", {}))
        inverse_users = list(node.users)
        if (
            len(inverse_users) == 1
            and inverse_users[0].op == "call_function"
            and str(inverse_users[0].target) == "aten.contiguous.default"
        ):
            if _is_user_output_node(inverse_users[0]):
                continue
            inverse_users[0].replace_all_uses_with(folded_mul)
        else:
            if _is_user_output_node(node):
                continue
            node.replace_all_uses_with(folded_mul)
        changed = True

    for node in list(graph.nodes):
        if (
            node.op != "call_function"
            or str(node.target) != "aten.permute.default"
            or len(node.args) < 2
            or _normalize_perm(node.args[1]) != [0, 3, 1, 2]
        ):
            continue
        mul_node = node.args[0]
        if (
            not isinstance(mul_node, torch.fx.Node)
            or mul_node.op != "call_function"
            or str(mul_node.target) != "aten.mul.Tensor"
            or len(mul_node.args) != 2
        ):
            continue
        mean_node = None
        const_node = None
        for arg in mul_node.args:
            if (
                isinstance(arg, torch.fx.Node)
                and arg.op == "call_function"
                and str(arg.target) == "aten.mean.dim"
                and len(arg.args) >= 3
                and list(arg.args[1]) == [1, 2]
                and bool(arg.args[2]) is True
            ):
                mean_node = arg
            elif isinstance(arg, torch.fx.Node):
                const_node = arg
        if mean_node is None or const_node is None:
            continue
        mean_input = mean_node.args[0]
        if not (
            isinstance(mean_input, torch.fx.Node)
            and mean_input.op == "call_function"
            and str(mean_input.target) == "aten.contiguous.default"
            and len(mean_input.args) >= 1
            and isinstance(mean_input.args[0], torch.fx.Node)
            and mean_input.args[0].op == "call_function"
            and str(mean_input.args[0].target) == "aten.permute.default"
            and len(mean_input.args[0].args) >= 2
            and _normalize_perm(mean_input.args[0].args[1]) == [0, 2, 3, 1]
            and isinstance(mean_input.args[0].args[0], torch.fx.Node)
        ):
            continue
        source = mean_input.args[0].args[0]
        const_shape = None
        const_meta_val = getattr(const_node, "meta", {}).get("val", None)
        if isinstance(const_meta_val, torch.Tensor):
            const_shape = [int(v) for v in list(const_meta_val.shape)]
        if const_node.op == "get_attr" and isinstance(const_node.target, str):
            const_tensor = getattr(graph_module, const_node.target, None)
            if isinstance(const_tensor, torch.Tensor):
                const_shape = [int(v) for v in list(const_tensor.shape)]
        if const_shape is None or len(const_shape) != 4 or const_shape[:3] != [1, 1, 1]:
            continue
        reshaped_const_shape = [1, int(const_shape[3]), 1, 1]
        with graph.inserting_before(mean_node):
            folded_mean = graph.call_function(
                mean_node.target,
                args=(source, [2, 3], True),
                kwargs=dict(mean_node.kwargs),
            )
            folded_const = graph.call_function(
                torch.ops.aten.reshape.default,
                args=(const_node, reshaped_const_shape),
                kwargs={},
            )
            folded_mul = graph.call_function(
                mul_node.target,
                args=(
                    folded_mean if mul_node.args[0] is mean_node else folded_const,
                    folded_const if mul_node.args[0] is mean_node else folded_mean,
                ),
                kwargs=dict(mul_node.kwargs),
            )
        folded_mean.meta = dict(getattr(mean_node, "meta", {}))
        folded_const.meta = dict(getattr(const_node, "meta", {}))
        folded_mul.meta = dict(getattr(node, "meta", {}))
        inverse_users = list(node.users)
        if (
            len(inverse_users) == 1
            and inverse_users[0].op == "call_function"
            and str(inverse_users[0].target) == "aten.contiguous.default"
        ):
            if _is_user_output_node(inverse_users[0]):
                continue
            inverse_users[0].replace_all_uses_with(folded_mul)
        else:
            if _is_user_output_node(node):
                continue
            node.replace_all_uses_with(folded_mul)
        changed = True

    binary_targets = {
        "aten.add.Tensor",
        "aten.div.Tensor",
        "aten.mul.Tensor",
        "aten.sub.Tensor",
    }
    for node in list(graph.nodes):
        if (
            node.op != "call_function"
            or str(node.target) != "aten.permute.default"
            or len(node.args) < 2
        ):
            continue
        inverse_perm = _normalize_perm(node.args[1])
        if inverse_perm is None:
            continue
        binary_node = node.args[0]
        if not isinstance(binary_node, torch.fx.Node):
            continue
        if (
            binary_node.op != "call_function"
            or str(binary_node.target) not in binary_targets
            or len(binary_node.args) != 2
        ):
            continue
        input_perm = _inverse_perm(inverse_perm)
        lhs_source = _match_binary_input_source(binary_node.args[0], input_perm)
        rhs_source = _match_binary_input_source(binary_node.args[1], input_perm)
        if lhs_source is None or rhs_source is None:
            continue
        with graph.inserting_before(binary_node):
            folded_binary = graph.call_function(
                binary_node.target,
                args=(lhs_source, rhs_source),
                kwargs=dict(binary_node.kwargs),
            )
        folded_binary.meta = dict(getattr(node, "meta", {}))
        inverse_users = list(node.users)
        if (
            len(inverse_users) == 1
            and inverse_users[0].op == "call_function"
            and str(inverse_users[0].target) == "aten.contiguous.default"
        ):
            if _is_user_output_node(inverse_users[0]):
                continue
            inverse_users[0].replace_all_uses_with(folded_binary)
        else:
            if _is_user_output_node(node):
                continue
            node.replace_all_uses_with(folded_binary)
        changed = True
    if changed:
        graph.eliminate_dead_code()
        graph.lint()
        graph_module.recompile()
    return exported_program

def _fold_layout_preserving_permute_chains(exported_program):
    graph_module = getattr(exported_program, "graph_module", None)
    if graph_module is None:
        return exported_program
    graph = graph_module.graph
    changed = False
    unary_targets = {
        "aten.relu.default",
    }
    for node in list(graph.nodes):
        if (
            node.op != "call_function"
            or str(node.target) != "aten.permute.default"
            or len(node.args) < 2
            or list(node.args[1]) != [0, 2, 3, 1]
        ):
            continue
        source = node.args[0]
        if not isinstance(source, torch.fx.Node):
            continue
        permute_users = list(node.users)
        if len(permute_users) != 1:
            continue
        contiguous_node = permute_users[0]
        if (
            contiguous_node.op != "call_function"
            or str(contiguous_node.target) != "aten.contiguous.default"
        ):
            continue
        contiguous_users = list(contiguous_node.users)
        if len(contiguous_users) != 1:
            continue
        unary_node = contiguous_users[0]
        if (
            unary_node.op != "call_function"
            or str(unary_node.target) not in unary_targets
        ):
            continue
        inverse_permute_nodes = list(unary_node.users)
        if len(inverse_permute_nodes) == 0:
            continue
        if any(
            inverse_node.op != "call_function"
            or str(inverse_node.target) != "aten.permute.default"
            or len(inverse_node.args) < 2
            or list(inverse_node.args[1]) != [0, 3, 1, 2]
            for inverse_node in inverse_permute_nodes
        ):
            continue
        with graph.inserting_before(node):
            folded_unary = graph.call_function(
                unary_node.target,
                args=(source, *tuple(unary_node.args[1:])),
                kwargs=dict(unary_node.kwargs),
            )
        folded_unary.meta = dict(getattr(source, "meta", {}))
        for inverse_node in inverse_permute_nodes:
            inverse_users = list(inverse_node.users)
            if (
                len(inverse_users) == 1
                and inverse_users[0].op == "call_function"
                and str(inverse_users[0].target) == "aten.contiguous.default"
            ):
                inverse_users[0].replace_all_uses_with(folded_unary)
            else:
                inverse_node.replace_all_uses_with(folded_unary)
        changed = True
    if changed:
        graph.eliminate_dead_code()
        graph.lint()
        graph_module.recompile()
    return exported_program


def _fold_nhwc_binary_expr_trees_in_exported_program(exported_program):
    graph_module = getattr(exported_program, "graph_module", None)
    if graph_module is None:
        return exported_program
    graph = graph_module.graph
    graph_signature = getattr(exported_program, "graph_signature", None)
    user_output_names = set()
    if graph_signature is not None:
        try:
            user_output_names = {str(name) for name in list(graph_signature.user_outputs)}
        except Exception:
            user_output_names = set()
    changed = False
    binary_targets = {
        "aten.add.Tensor",
        "aten.mul.Tensor",
        "aten.sub.Tensor",
        "aten.div.Tensor",
    }

    def _normalize_perm(arg):
        if not isinstance(arg, (list, tuple)):
            return None
        return [int(v) for v in arg]

    def _is_user_output_node(node):
        return isinstance(node, torch.fx.Node) and str(node.name) in user_output_names

    def _rank4_shape(node):
        if not isinstance(node, torch.fx.Node):
            return None
        meta_val = getattr(node, "meta", {}).get("val", None)
        if isinstance(meta_val, torch.Tensor):
            shape = [int(v) for v in list(meta_val.shape)]
            return shape if len(shape) == 4 else None
        if node.op == "get_attr" and isinstance(node.target, str):
            tensor = getattr(graph_module, node.target, None)
            if isinstance(tensor, torch.Tensor):
                shape = [int(v) for v in list(tensor.shape)]
                return shape if len(shape) == 4 else None
            return None

    def _shape_meta_from_node(node, shape):
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

    def _match_nhwc_materialize_source(node):
        if not isinstance(node, torch.fx.Node):
            return None
        if (
            node.op == "call_function"
            and str(node.target) in binary_targets
            and len(node.args) == 2
            and _rank4_shape(node) is not None
            and all(isinstance(arg, torch.fx.Node) and _rank4_shape(arg) is not None for arg in node.args)
        ):
            lhs_from_nhwc = _match_nhwc_materialize_source(node.args[0])
            rhs_from_nhwc = _match_nhwc_materialize_source(node.args[1])
            if lhs_from_nhwc is None and rhs_from_nhwc is None:
                return node
        if (
            node.op == "call_function"
            and str(node.target) == "aten.permute.default"
            and len(node.args) >= 2
            and _normalize_perm(node.args[1]) == [0, 2, 3, 1]
            and isinstance(node.args[0], torch.fx.Node)
            and _rank4_shape(node.args[0]) is not None
        ):
            return node.args[0]
        if (
            node.op == "call_function"
            and str(node.target) == "aten.contiguous.default"
            and len(node.args) >= 1
        ):
            return _match_nhwc_materialize_source(node.args[0])
        return None

    folded_expr_cache = {}

    def _build_folded_nchw_expr(node):
        if not isinstance(node, torch.fx.Node):
            return None
        cached = folded_expr_cache.get(node, None)
        if cached is not None:
            return cached
        source = _match_nhwc_materialize_source(node)
        if source is not None:
            folded_expr_cache[node] = source
            return source
        if (
            node.op == "call_function"
            and str(node.target) in binary_targets
            and len(node.args) == 2
        ):
            lhs = _build_folded_nchw_expr(node.args[0])
            rhs = _build_folded_nchw_expr(node.args[1])
            if lhs is None or rhs is None:
                return None
            if _rank4_shape(lhs) is None or _rank4_shape(rhs) is None:
                return None
            with graph.inserting_before(node):
                folded_binary = graph.call_function(
                    node.target,
                    args=(lhs, rhs),
                    kwargs=dict(node.kwargs),
                )
            lhs_shape = _rank4_shape(lhs)
            folded_binary.meta = (
                _shape_meta_from_node(lhs, lhs_shape)
                if lhs_shape is not None
                else dict(getattr(lhs, "meta", {}))
            )
            folded_expr_cache[node] = folded_binary
            return folded_binary
        return None

    def _materialize_nhwc(node, insert_before_node):
        source_shape = _rank4_shape(node)
        nhwc_shape = (
            [int(source_shape[0]), int(source_shape[2]), int(source_shape[3]), int(source_shape[1])]
            if source_shape is not None
            else None
        )
        with graph.inserting_before(insert_before_node):
            permute_node = graph.call_function(
                torch.ops.aten.permute.default,
                args=(node, [0, 2, 3, 1]),
                kwargs={},
            )
            contiguous_node = graph.call_function(
                torch.ops.aten.contiguous.default,
                args=(permute_node,),
                kwargs={},
            )
        permute_node.meta = (
            _shape_meta_from_node(node, nhwc_shape)
            if nhwc_shape is not None
            else dict(getattr(insert_before_node, "meta", {}))
        )
        contiguous_node.meta = (
            _shape_meta_from_node(node, nhwc_shape)
            if nhwc_shape is not None
            else dict(getattr(insert_before_node, "meta", {}))
        )
        return contiguous_node

    while True:
        local_changed = False
        for node in list(graph.nodes):
            if (
                node.op != "call_function"
                or str(node.target) not in binary_targets
                or len(node.args) != 2
            ):
                continue
            lhs_source = _match_nhwc_materialize_source(node.args[0])
            rhs_source = _match_nhwc_materialize_source(node.args[1])
            if lhs_source is None or rhs_source is None:
                continue
            if lhs_source is node.args[0] and rhs_source is node.args[1]:
                continue
            lhs_shape = _rank4_shape(lhs_source)
            rhs_shape = _rank4_shape(rhs_source)
            if lhs_shape is None or rhs_shape is None or lhs_shape != rhs_shape:
                continue
            supported = True
            for user in list(node.users):
                if user.op == "call_function" and str(user.target) in binary_targets:
                    continue
                if (
                    user.op == "call_function"
                    and str(user.target) == "aten.reshape.default"
                    and len(user.args) >= 1
                    and user.args[0] is node
                ):
                    continue
                if (
                    user.op == "call_function"
                    and str(user.target) == "aten.permute.default"
                    and len(user.args) >= 2
                    and user.args[0] is node
                    and _normalize_perm(user.args[1]) == [0, 3, 1, 2]
                ):
                    continue
                supported = False
                break
            if not supported:
                continue
            node.args = (lhs_source, rhs_source)
            node.meta = _shape_meta_from_node(lhs_source, lhs_shape)
            local_changed = True
            changed = True
        if not local_changed:
            break

    for node in list(graph.nodes):
        if (
            node.op == "call_function"
            and str(node.target) == "aten.reshape.default"
            and len(node.args) >= 2
            and isinstance(node.args[0], torch.fx.Node)
        ):
            folded_source = _build_folded_nchw_expr(node.args[0])
            if folded_source is None:
                continue
            if _rank4_shape(folded_source) is None:
                continue
            node.args = (_materialize_nhwc(folded_source, node),) + tuple(node.args[1:])
            changed = True
            continue
        if (
            node.op == "call_function"
            and str(node.target) == "aten.permute.default"
            and len(node.args) >= 2
            and _normalize_perm(node.args[1]) == [0, 3, 1, 2]
            and isinstance(node.args[0], torch.fx.Node)
        ):
            folded_source = _build_folded_nchw_expr(node.args[0])
            if folded_source is None:
                continue
            if _rank4_shape(folded_source) is None:
                continue
            node_users = list(node.users)
            if (
                len(node_users) == 1
                and node_users[0].op == "call_function"
                and str(node_users[0].target) == "aten.contiguous.default"
            ):
                if _is_user_output_node(node_users[0]):
                    continue
                node_users[0].replace_all_uses_with(folded_source)
            else:
                if _is_user_output_node(node):
                    continue
                node.replace_all_uses_with(folded_source)
            changed = True

    if changed:
        graph.eliminate_dead_code()
        graph.lint()
        graph_module.recompile()
    return exported_program

def _optimize_exported_program(exported_program):
    exported_program = _prune_alias_nodes(exported_program)
    exported_program = _fold_singleton_channel_split_permute_bridges_in_exported_program(exported_program)
    exported_program = _fold_inverse_permute_round_trips(exported_program)
    exported_program = _fold_layout_preserving_permute_chains(exported_program)
    exported_program = _fold_nhwc_binary_expr_trees_in_exported_program(exported_program)
    return exported_program

def _postprocess_saved_exported_program(exported_program):
    graph_module = getattr(exported_program, "graph_module", None)
    if graph_module is None:
        return exported_program
    graph = graph_module.graph
    graph_signature = getattr(exported_program, "graph_signature", None)
    user_output_names = set()
    if graph_signature is not None:
        try:
            user_output_names = {str(name) for name in list(graph_signature.user_outputs)}
        except Exception:
            user_output_names = set()
    changed = False

    def _normalize_perm(arg):
        if not isinstance(arg, (list, tuple)):
            return None
        return [int(v) for v in arg]

    def _is_user_output_node(node):
        return isinstance(node, torch.fx.Node) and str(node.name) in user_output_names

    def _rank4_shape(node):
        if not isinstance(node, torch.fx.Node):
            return None
        meta_val = getattr(node, "meta", {}).get("val", None)
        if isinstance(meta_val, torch.Tensor):
            shape = [int(v) for v in list(meta_val.shape)]
            return shape if len(shape) == 4 else None
        return None

    def _shape_meta_like(node, shape):
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

    def _shape_list_from_arg(arg):
        if not isinstance(arg, (list, tuple)):
            return None
        try:
            return [int(v) for v in list(arg)]
        except Exception:
            return None

    def _is_scalar_like(value):
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

    def _match_permute_chain_source(node, perm):
        if not isinstance(node, torch.fx.Node):
            return None
        if (
            node.op == "call_function"
            and str(node.target) == "aten.permute.default"
            and len(node.args) >= 2
            and _normalize_perm(node.args[1]) == perm
            and isinstance(node.args[0], torch.fx.Node)
        ):
            return node.args[0]
        if (
            node.op == "call_function"
            and str(node.target) == "aten.contiguous.default"
            and len(node.args) >= 1
        ):
            return _match_permute_chain_source(node.args[0], perm)
        return None

    def _convert_nhwc_pad_to_nchw_pad(pad_values):
        values = [int(v) for v in list(pad_values)]
        if len(values) % 2 != 0 or len(values) > 8:
            return None
        nhwc_inner_to_outer = ["C", "W", "H", "N"]
        nchw_inner_to_outer = ["W", "H", "C", "N"]
        semantic_pairs = {name: [0, 0] for name in nhwc_inner_to_outer}
        pair_count = len(values) // 2
        for idx in range(pair_count):
            semantic_pairs[nhwc_inner_to_outer[idx]] = [
                values[idx * 2],
                values[idx * 2 + 1],
            ]
        output_pairs = [semantic_pairs[name] for name in nchw_inner_to_outer]
        last_nonzero = -1
        for idx, pair in enumerate(output_pairs):
            if pair != [0, 0]:
                last_nonzero = idx
        if last_nonzero < 0:
            return []
        flattened = []
        for pair in output_pairs[: last_nonzero + 1]:
            flattened.extend(pair)
        return flattened

    def _match_cat_input_fold_source(node):
        source = _match_permute_chain_source(node, [0, 2, 3, 1])
        if source is not None:
            return source
        if (
            isinstance(node, torch.fx.Node)
            and node.op == "call_function"
            and str(node.target) == "aten.pad.default"
            and len(node.args) >= 2
            and isinstance(node.args[0], torch.fx.Node)
        ):
            pad_source = _match_permute_chain_source(node.args[0], [0, 2, 3, 1])
            if pad_source is None:
                return None
            nchw_pad = _convert_nhwc_pad_to_nchw_pad(node.args[1])
            if nchw_pad is None:
                return None
            return (pad_source, nchw_pad, tuple(node.args[2:]), dict(node.kwargs))
        return None

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
                changed = True

    while True:
        local_changed = False
        for split_node in list(graph.nodes):
            if (
                split_node.op != "call_function"
                or str(split_node.target) != "aten.tensor_split.sections"
                or len(split_node.args) < 3
            ):
                continue
            split_input = split_node.args[0]
            if not (
                isinstance(split_input, torch.fx.Node)
                and split_input.op == "call_function"
                and str(split_input.target) == "aten.contiguous.default"
                and len(split_input.args) >= 1
                and isinstance(split_input.args[0], torch.fx.Node)
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
            if int(split_node.args[2]) != 3:
                continue
            base_source = permute_node.args[0]
            base_shape = _rank4_shape(base_source)
            if base_shape is None or int(base_shape[1]) <= 0:
                continue
            getitem_users = []
            for user in list(split_node.users):
                if (
                    user.op != "call_function"
                    or str(user.target) != "<built-in function getitem>"
                    or len(user.args) < 2
                    or user.args[0] is not split_node
                ):
                    getitem_users = []
                    break
                getitem_users.append(user)
            if len(getitem_users) == 0:
                continue

            with graph.inserting_before(split_node):
                new_split = graph.call_function(
                    torch.ops.aten.tensor_split.sections,
                    args=(base_source, 1, 1),
                    kwargs=dict(split_node.kwargs),
                )
            new_split.meta = dict(getattr(split_node, "meta", {}))

            cf_shape = [int(base_shape[0]), 1, int(base_shape[2]), int(base_shape[3])]
            nhwc_shape = [int(base_shape[0]), int(base_shape[2]), int(base_shape[3]), 1]

            for getitem_node in list(getitem_users):
                split_index = int(getitem_node.args[1])
                with graph.inserting_before(getitem_node):
                    new_getitem = graph.call_function(
                        getitem_node.target,
                        args=(new_split, split_index),
                        kwargs=dict(getitem_node.kwargs),
                    )
                new_getitem.meta = _shape_meta_like(getitem_node, cf_shape)
                nhwc_holder = [None]

                def _get_nhwc_view(insert_before_node):
                    if nhwc_holder[0] is not None:
                        return nhwc_holder[0]
                    with graph.inserting_before(insert_before_node):
                        nhwc_holder[0] = graph.call_function(
                            torch.ops.aten.reshape.default,
                            args=(new_getitem, nhwc_shape),
                            kwargs={},
                        )
                    nhwc_holder[0].meta = _shape_meta_like(getitem_node, nhwc_shape)
                    return nhwc_holder[0]

                for user in list(getitem_node.users):
                    if (
                        user.op == "call_function"
                        and str(user.target) == "aten.reshape.default"
                        and len(user.args) >= 2
                        and _shape_list_from_arg(user.args[1]) == cf_shape
                    ):
                        user.replace_all_uses_with(new_getitem)
                        local_changed = True
                        changed = True
                        continue
                    if (
                        user.op == "call_function"
                        and str(user.target) in {
                            "aten.add.Tensor",
                            "aten.div.Tensor",
                            "aten.mul.Tensor",
                            "aten.sub.Tensor",
                        }
                        and len(user.args) == 2
                        and (
                            (user.args[0] is getitem_node and _is_scalar_like(user.args[1]))
                            or (user.args[1] is getitem_node and _is_scalar_like(user.args[0]))
                        )
                    ):
                        user.replace_input_with(getitem_node, new_getitem)
                        user.meta = _shape_meta_like(user, cf_shape)
                        local_changed = True
                        changed = True
                        continue
                    replacement = _get_nhwc_view(user)
                    user.replace_input_with(getitem_node, replacement)
                    local_changed = True
                    changed = True

            if local_changed:
                graph.eliminate_dead_code()
                graph.lint()
                graph_module.recompile()
                break
        if not local_changed:
            break

    for node in list(graph.nodes):
        if (
            node.op != "call_function"
            or str(node.target) != "aten.permute.default"
            or len(node.args) < 2
            or _normalize_perm(node.args[1]) != [0, 3, 1, 2]
        ):
            continue
        source = node.args[0]
        source_shape = _rank4_shape(source)
        if source_shape is None or int(source_shape[3]) != 1:
            continue
        reshaped_shape = [int(source_shape[0]), 1, int(source_shape[1]), int(source_shape[2])]
        with graph.inserting_before(node):
            folded_reshape = graph.call_function(
                torch.ops.aten.reshape.default,
                args=(source, reshaped_shape),
                kwargs={},
            )
        folded_reshape.meta = _shape_meta_like(node, reshaped_shape)
        node_users = list(node.users)
        if (
            len(node_users) == 1
            and node_users[0].op == "call_function"
            and str(node_users[0].target) == "aten.contiguous.default"
        ):
            node_users[0].replace_all_uses_with(folded_reshape)
        else:
            node.replace_all_uses_with(folded_reshape)
        changed = True

    for node in list(graph.nodes):
        if (
            node.op != "call_function"
            or str(node.target) != "aten.permute.default"
            or len(node.args) < 2
            or _normalize_perm(node.args[1]) != [0, 3, 1, 2]
        ):
            continue
        cat_node = node.args[0]
        if (
            not isinstance(cat_node, torch.fx.Node)
            or cat_node.op != "call_function"
            or str(cat_node.target) != "aten.cat.default"
            or len(cat_node.args) < 1
        ):
            continue
        cat_inputs_arg = cat_node.args[0]
        if not isinstance(cat_inputs_arg, (list, tuple)):
            continue
        cat_dim = None
        if len(cat_node.args) >= 2:
            cat_dim = int(cat_node.args[1])
        elif "dim" in cat_node.kwargs:
            cat_dim = int(cat_node.kwargs["dim"])
        if cat_dim != 3:
            continue
        folded_inputs = []
        for cat_input in list(cat_inputs_arg):
            folded_source = _match_cat_input_fold_source(cat_input)
            if folded_source is None:
                folded_inputs = []
                break
            if isinstance(folded_source, tuple):
                source_node, nchw_pad, pad_args_tail, pad_kwargs = folded_source
                if _rank4_shape(source_node) is None:
                    folded_inputs = []
                    break
                with graph.inserting_before(cat_node):
                    folded_pad = graph.call_function(
                        torch.ops.aten.pad.default,
                        args=(source_node, nchw_pad, *pad_args_tail),
                        kwargs=pad_kwargs,
                    )
                folded_pad.meta = dict(getattr(cat_input, "meta", {}))
                folded_inputs.append(folded_pad)
            else:
                if _rank4_shape(folded_source) is None:
                    folded_inputs = []
                    break
                folded_inputs.append(folded_source)
        if len(folded_inputs) != len(list(cat_inputs_arg)):
            continue
        folded_cat_kwargs = dict(cat_node.kwargs)
        folded_cat_args = (folded_inputs, 1)
        if "dim" in folded_cat_kwargs:
            folded_cat_kwargs["dim"] = 1
            folded_cat_args = (folded_inputs,)
        cat_node.args = folded_cat_args
        cat_node.kwargs = folded_cat_kwargs
        cat_node.meta = dict(getattr(node, "meta", {}))
        node_users = list(node.users)
        if (
            len(node_users) == 1
            and node_users[0].op == "call_function"
            and str(node_users[0].target) == "aten.contiguous.default"
        ):
            if _is_user_output_node(node_users[0]):
                continue
            node_users[0].replace_all_uses_with(cat_node)
        else:
            if _is_user_output_node(node):
                continue
            node.replace_all_uses_with(cat_node)
        changed = True

    for node in list(graph.nodes):
        if (
            node.op != "call_function"
            or str(node.target) != "aten.permute.default"
            or len(node.args) < 2
            or _normalize_perm(node.args[1]) != [0, 3, 1, 2]
        ):
            continue
        cat_node = node.args[0]
        if (
            not isinstance(cat_node, torch.fx.Node)
            or cat_node.op != "call_function"
            or str(cat_node.target) != "aten.cat.default"
            or len(cat_node.args) < 2
            or not isinstance(cat_node.args[0], (list, tuple))
            or int(cat_node.args[1]) != 3
        ):
            continue
        folded_inputs = []
        for cat_input in list(cat_node.args[0]):
            input_shape = _rank4_shape(cat_input)
            if input_shape is None or int(input_shape[3]) != 1:
                folded_inputs = []
                break
            reshaped_shape = [int(input_shape[0]), 1, int(input_shape[1]), int(input_shape[2])]
            with graph.inserting_before(cat_node):
                folded_input = graph.call_function(
                    torch.ops.aten.reshape.default,
                    args=(cat_input, reshaped_shape),
                    kwargs={},
                )
            folded_input.meta = _shape_meta_like(cat_input, reshaped_shape)
            folded_inputs.append(folded_input)
        if len(folded_inputs) != len(list(cat_node.args[0])):
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
        changed = True

    if changed:
        graph.eliminate_dead_code()
        graph.lint()
        graph_module.recompile()
    return exported_program

with torch.no_grad():
    exported = torch.export.export(model, example_inputs)
torch.export.save(exported, str(exported_program_path))
print(json.dumps({"file_name": exported_program_path.name}))
"""
