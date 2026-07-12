from __future__ import annotations

from typing import Any, List, Optional, Tuple

import numpy as np
import onnx

from onnx2tf.tflite_builder.pytorch_onnx_layout_passes import (
    _onnx_convert_pads_nhwc_to_nchw,
    _onnx_resolve_rank4_shape,
    _onnx_rewrite_slice_axis_to_nchw_in_place,
)
from onnx2tf.tflite_builder.pytorch_onnx_utils import (
    _onnx_get_initializer_array,
    _onnx_get_initializer_scalar,
    _onnx_make_unique_initializer_name,
    _onnx_node_attr,
    _onnx_node_maps,
    _onnx_remove_nodes_by_name,
    _onnx_replace_all_node_inputs,
    _onnx_set_initializer_array,
    _onnx_set_node_attr,
)


def _onnx_fold_concat_layout_bridges_in_place(graph: onnx.GraphProto) -> None:
    producer_map, consumer_map = _onnx_node_maps(graph)
    remove_node_names: List[str] = []
    for inverse_transpose_node in list(graph.node):
        if str(inverse_transpose_node.op_type) != "Transpose":
            continue
        if list(_onnx_node_attr(inverse_transpose_node, "perm") or []) != [0, 3, 1, 2]:
            continue
        concat_node_name = str(inverse_transpose_node.input[0]) if inverse_transpose_node.input else ""
        concat_node = producer_map.get(concat_node_name)
        if concat_node is None or str(concat_node.op_type) != "Concat":
            continue
        if int(_onnx_node_attr(concat_node, "axis") or -1) != 3:
            continue
        if len(consumer_map.get(str(concat_node.output[0]), [])) != 1:
            continue
        input_transpose_nodes: List[onnx.NodeProto] = []
        passthrough_inverse_transpose_nodes: List[Optional[onnx.NodeProto]] = []
        for input_name in concat_node.input:
            transpose_node = producer_map.get(str(input_name))
            if transpose_node is None:
                input_transpose_nodes = []
                break
            input_transpose_nodes.append(transpose_node)
        if not input_transpose_nodes:
            continue
        if any(str(node.op_type) != "Transpose" for node in input_transpose_nodes):
            continue
        if any(list(_onnx_node_attr(node, "perm") or []) != [0, 2, 3, 1] for node in input_transpose_nodes):
            continue
        valid = True
        for transpose_node in input_transpose_nodes:
            transpose_consumers = list(consumer_map.get(str(transpose_node.output[0]), []))
            if len(transpose_consumers) == 1 and transpose_consumers[0] == concat_node:
                passthrough_inverse_transpose_nodes.append(None)
                continue
            if len(transpose_consumers) != 2 or concat_node not in transpose_consumers:
                valid = False
                break
            passthrough_consumer = next(
                (
                    node
                    for node in transpose_consumers
                    if node != concat_node
                ),
                None,
            )
            if (
                passthrough_consumer is None
                or str(passthrough_consumer.op_type) != "Transpose"
                or list(_onnx_node_attr(passthrough_consumer, "perm") or []) != [0, 3, 1, 2]
                or len(passthrough_consumer.input) != 1
                or len(passthrough_consumer.output) != 1
            ):
                valid = False
                break
            passthrough_inverse_transpose_nodes.append(passthrough_consumer)
        if not valid:
            continue

        for input_index, transpose_node in enumerate(input_transpose_nodes):
            concat_node.input[input_index] = str(transpose_node.input[0])
        for input_transpose_node, passthrough_inverse_transpose_node in zip(
            input_transpose_nodes,
            passthrough_inverse_transpose_nodes,
        ):
            if passthrough_inverse_transpose_node is None:
                continue
            _onnx_replace_all_node_inputs(
                graph,
                old_name=str(passthrough_inverse_transpose_node.output[0]),
                new_name=str(input_transpose_node.input[0]),
            )
        _onnx_set_node_attr(concat_node, "axis", 1)
        _onnx_replace_all_node_inputs(
            graph,
            old_name=str(inverse_transpose_node.output[0]),
            new_name=str(concat_node.output[0]),
        )
        remove_node_names.extend(
            [str(node.name) for node in input_transpose_nodes]
            + [
                str(node.name)
                for node in passthrough_inverse_transpose_nodes
                if node is not None
            ]
            + [str(inverse_transpose_node.name)]
        )
    _onnx_remove_nodes_by_name(graph, remove_node_names)


def _onnx_fold_channel_front_concat_layout_bridges_in_place(graph: onnx.GraphProto) -> None:
    producer_map, consumer_map = _onnx_node_maps(graph)
    remove_node_names: List[str] = []
    for trailing_transpose_node in list(graph.node):
        if str(trailing_transpose_node.op_type) != "Transpose":
            continue
        if list(_onnx_node_attr(trailing_transpose_node, "perm") or []) != [3, 0, 1, 2]:
            continue
        concat_node_name = str(trailing_transpose_node.input[0]) if trailing_transpose_node.input else ""
        concat_node = producer_map.get(concat_node_name)
        if concat_node is None or str(concat_node.op_type) != "Concat":
            continue
        if int(_onnx_node_attr(concat_node, "axis") or -1) != 3:
            continue
        if len(consumer_map.get(str(concat_node.output[0]), [])) != 1:
            continue

        rewritten_inputs: List[str] = []
        input_transforms: List[Tuple[onnx.NodeProto, List[int]]] = []
        local_remove_node_names: List[str] = []
        expected_nhw_shape: Optional[List[int]] = None
        transformed_cf_shapes: List[List[int]] = []
        valid = True
        rewritten_any = False
        for input_name in list(concat_node.input):
            input_node = producer_map.get(str(input_name))
            if (
                input_node is None
                or str(input_node.op_type) != "Transpose"
                or len(input_node.input) != 1
                or len(consumer_map.get(str(input_node.output[0]), [])) != 1
            ):
                valid = False
                break
            source_name = str(input_node.input[0])
            source_shape = _onnx_resolve_rank4_shape(
                graph,
                source_name,
                producer_map=producer_map,
            )
            if source_shape is None:
                valid = False
                break
            input_perm = [int(v) for v in list(_onnx_node_attr(input_node, "perm") or [])]
            transformed_cf_shape: Optional[List[int]] = None
            input_nhw_shape: Optional[List[int]] = None
            if input_perm == [0, 2, 3, 1]:
                transformed_cf_shape = [int(v) for v in list(source_shape)]
                input_nhw_shape = [
                    int(source_shape[0]),
                    int(source_shape[2]),
                    int(source_shape[3]),
                    int(source_shape[1]),
                ]
                rewritten_inputs.append(source_name)
                local_remove_node_names.append(str(input_node.name))
                rewritten_any = True
            elif input_perm == [1, 2, 3, 0]:
                transformed_cf_shape = [
                    int(source_shape[1]),
                    int(source_shape[0]),
                    int(source_shape[2]),
                    int(source_shape[3]),
                ]
                input_nhw_shape = [
                    int(source_shape[1]),
                    int(source_shape[2]),
                    int(source_shape[3]),
                    int(source_shape[0]),
                ]
                rewritten_inputs.append(str(input_node.output[0]))
                input_transforms.append((input_node, [1, 0, 2, 3]))
                rewritten_any = True
            else:
                valid = False
                break
            assert transformed_cf_shape is not None
            assert input_nhw_shape is not None
            transformed_cf_shapes.append(transformed_cf_shape)
            if expected_nhw_shape is None:
                expected_nhw_shape = [int(v) for v in list(input_nhw_shape)]
            else:
                if (
                    int(expected_nhw_shape[0]) != int(input_nhw_shape[0])
                    or int(expected_nhw_shape[1]) != int(input_nhw_shape[1])
                    or int(expected_nhw_shape[2]) != int(input_nhw_shape[2])
                ):
                    valid = False
                    break
        if not valid or not rewritten_any or len(rewritten_inputs) != len(list(concat_node.input)):
            continue

        expected_cf_output_shape = [int(v) for v in list(transformed_cf_shapes[0])]
        expected_cf_output_shape[1] = int(sum(int(shape[1]) for shape in transformed_cf_shapes))
        trailing_output_shape = _onnx_resolve_rank4_shape(
            graph,
            str(trailing_transpose_node.output[0]),
            producer_map=producer_map,
        )
        if trailing_output_shape is not None:
            expected_trailing_output_shape = [
                int(expected_cf_output_shape[1]),
                int(expected_cf_output_shape[0]),
                int(expected_cf_output_shape[2]),
                int(expected_cf_output_shape[3]),
            ]
            if [int(v) for v in list(trailing_output_shape)] != expected_trailing_output_shape:
                continue

        for input_index, rewritten_input_name in enumerate(rewritten_inputs):
            concat_node.input[input_index] = str(rewritten_input_name)
        for input_node, new_perm in input_transforms:
            _onnx_set_node_attr(input_node, "perm", new_perm)
        _onnx_set_node_attr(concat_node, "axis", 1)
        _onnx_set_node_attr(trailing_transpose_node, "perm", [1, 0, 2, 3])
        remove_node_names.extend(local_remove_node_names)
    _onnx_remove_nodes_by_name(graph, remove_node_names)


def _onnx_fold_channel_front_gathernd_transpose_bridges_in_place(graph: onnx.GraphProto) -> None:
    producer_map, consumer_map = _onnx_node_maps(graph)
    remove_node_names: List[str] = []
    for source_transpose_node in list(graph.node):
        if str(source_transpose_node.op_type) != "Transpose":
            continue
        if list(_onnx_node_attr(source_transpose_node, "perm") or []) != [1, 0, 2, 3]:
            continue
        if len(source_transpose_node.input) != 1 or len(source_transpose_node.output) != 1:
            continue
        source_input_name = str(source_transpose_node.input[0])
        source_output_name = str(source_transpose_node.output[0])
        source_input_shape = _onnx_resolve_rank4_shape(
            graph,
            source_input_name,
            producer_map=producer_map,
        )
        if (
            source_input_shape is None
            or len(source_input_shape) != 4
            or int(source_input_shape[0]) != 1
        ):
            continue
        gather_nodes = list(consumer_map.get(source_output_name, []))
        if len(gather_nodes) == 0:
            continue
        if not all(str(node.op_type) == "GatherND" for node in gather_nodes):
            continue

        local_remove_node_names: List[str] = [str(source_transpose_node.name)]
        valid = True
        for gather_node in gather_nodes:
            if len(gather_node.input) != 2 or len(gather_node.output) != 1:
                valid = False
                break
            gather_consumers = consumer_map.get(str(gather_node.output[0]), [])
            if len(gather_consumers) != 1:
                valid = False
                break
            trailing_transpose_node = gather_consumers[0]
            trailing_output_name: Optional[str] = None
            trailing_remove_node_names: List[str] = []
            if (
                str(trailing_transpose_node.op_type) == "Transpose"
                and list(_onnx_node_attr(trailing_transpose_node, "perm") or []) == [1, 0, 2, 3]
                and len(trailing_transpose_node.output) == 1
            ):
                trailing_output_name = str(trailing_transpose_node.output[0])
                trailing_remove_node_names.append(str(trailing_transpose_node.name))
            elif (
                str(trailing_transpose_node.op_type) == "Transpose"
                and list(_onnx_node_attr(trailing_transpose_node, "perm") or []) == [1, 2, 3, 0]
                and len(trailing_transpose_node.output) == 1
            ):
                second_consumers = consumer_map.get(str(trailing_transpose_node.output[0]), [])
                if len(second_consumers) != 1:
                    valid = False
                    break
                second_transpose_node = second_consumers[0]
                if (
                    str(second_transpose_node.op_type) != "Transpose"
                    or list(_onnx_node_attr(second_transpose_node, "perm") or []) != [0, 3, 1, 2]
                    or len(second_transpose_node.output) != 1
                ):
                    valid = False
                    break
                trailing_output_name = str(second_transpose_node.output[0])
                trailing_remove_node_names.extend(
                    [
                        str(trailing_transpose_node.name),
                        str(second_transpose_node.name),
                    ]
                )
            else:
                valid = False
                break
            assert trailing_output_name is not None
            indices_name = str(gather_node.input[1])
            indices_array = _onnx_get_initializer_array(graph, indices_name)
            if indices_array is None:
                valid = False
                break
            indices_values = np.asarray(indices_array, dtype=np.int64)
            if indices_values.ndim != 2 or indices_values.shape[1] != 1:
                valid = False
                break
            flat_indices = indices_values.reshape(-1)
            if np.any(flat_indices < 0):
                valid = False
                break
            if int(source_input_shape[1]) > 0 and np.any(flat_indices >= int(source_input_shape[1])):
                valid = False
                break
            new_indices_name = _onnx_make_unique_initializer_name(graph, f"{indices_name}_axis1")
            _onnx_set_initializer_array(
                graph,
                name=new_indices_name,
                array=flat_indices.astype(np.int64),
            )
            gather_node.op_type = "Gather"
            gather_node.input[:] = [source_input_name, str(new_indices_name)]
            gather_node.output[:] = [trailing_output_name]
            del gather_node.attribute[:]
            _onnx_set_node_attr(gather_node, "axis", 1)
            local_remove_node_names.extend(trailing_remove_node_names)
        if not valid:
            continue
        remove_node_names.extend(local_remove_node_names)
    _onnx_remove_nodes_by_name(graph, remove_node_names)


def _onnx_fold_pad_concat_layout_bridges_in_place(graph: onnx.GraphProto) -> None:
    producer_map, consumer_map = _onnx_node_maps(graph)
    remove_node_names: List[str] = []
    for inverse_transpose_node in list(graph.node):
        if str(inverse_transpose_node.op_type) != "Transpose":
            continue
        if list(_onnx_node_attr(inverse_transpose_node, "perm") or []) != [0, 3, 1, 2]:
            continue
        concat_node_name = str(inverse_transpose_node.input[0]) if inverse_transpose_node.input else ""
        concat_node = producer_map.get(concat_node_name)
        if concat_node is None or str(concat_node.op_type) != "Concat":
            continue
        if int(_onnx_node_attr(concat_node, "axis") or -1) != 3:
            continue
        if len(consumer_map.get(str(concat_node.output[0]), [])) != 1:
            continue

        rewritten_inputs: List[str] = []
        local_remove_node_names: List[str] = [str(inverse_transpose_node.name)]
        valid = True
        for input_name in list(concat_node.input):
            input_node = producer_map.get(str(input_name))
            if input_node is None:
                valid = False
                break
            if (
                str(input_node.op_type) == "Transpose"
                and list(_onnx_node_attr(input_node, "perm") or []) == [0, 2, 3, 1]
                and len(consumer_map.get(str(input_node.output[0]), [])) == 1
            ):
                rewritten_inputs.append(str(input_node.input[0]))
                local_remove_node_names.append(str(input_node.name))
                continue
            if str(input_node.op_type) != "Pad" or len(input_node.input) < 2:
                valid = False
                break
            pad_input_node = producer_map.get(str(input_node.input[0]))
            if (
                pad_input_node is None
                or str(pad_input_node.op_type) != "Transpose"
                or list(_onnx_node_attr(pad_input_node, "perm") or []) != [0, 2, 3, 1]
                or len(consumer_map.get(str(pad_input_node.output[0]), [])) != 1
            ):
                valid = False
                break
            pad_values = _onnx_get_initializer_array(graph, str(input_node.input[1]))
            nchw_pad_values = (
                _onnx_convert_pads_nhwc_to_nchw(pad_values)
                if pad_values is not None
                else None
            )
            if nchw_pad_values is None:
                valid = False
                break
            new_pad_name = _onnx_make_unique_initializer_name(graph, f"{input_node.input[1]}_nchw")
            _onnx_set_initializer_array(graph, name=new_pad_name, array=nchw_pad_values)
            input_node.input[0] = str(pad_input_node.input[0])
            input_node.input[1] = str(new_pad_name)
            rewritten_inputs.append(str(input_node.output[0]))
            local_remove_node_names.append(str(pad_input_node.name))
        if not valid or not rewritten_inputs:
            continue

        for input_index, rewritten_input_name in enumerate(rewritten_inputs):
            concat_node.input[input_index] = str(rewritten_input_name)
        _onnx_set_node_attr(concat_node, "axis", 1)
        _onnx_replace_all_node_inputs(
            graph,
            old_name=str(inverse_transpose_node.output[0]),
            new_name=str(concat_node.output[0]),
        )
        remove_node_names.extend(local_remove_node_names)
    _onnx_remove_nodes_by_name(graph, remove_node_names)


def _onnx_fold_singleton_resize_matmul_transpose_bridges_in_place(graph: onnx.GraphProto) -> None:
    while True:
        producer_map, consumer_map = _onnx_node_maps(graph)
        remove_node_names: List[str] = []
        optimized = False
        for matmul_node in list(graph.node):
            if str(matmul_node.op_type) != "MatMul" or len(matmul_node.input) != 2:
                continue

            reshape_node: Optional[onnx.NodeProto] = None
            reshape_input_index: Optional[int] = None
            for input_index, input_name in enumerate(list(matmul_node.input)):
                candidate_node = producer_map.get(str(input_name))
                if candidate_node is not None and str(candidate_node.op_type) == "Reshape" and len(candidate_node.input) >= 2:
                    reshape_node = candidate_node
                    reshape_input_index = int(input_index)
                    break
            if reshape_node is None or reshape_input_index is None:
                continue

            reshape_shape_name = str(reshape_node.input[1]) if len(reshape_node.input) >= 2 else ""
            reshape_shape_array = _onnx_get_initializer_array(graph, reshape_shape_name)
            if reshape_shape_array is None:
                continue
            reshape_shape = [int(v) for v in reshape_shape_array.reshape(-1)]
            constant_input_index = 1 - int(reshape_input_index)
            constant_input_name = str(matmul_node.input[constant_input_index])
            constant_array = _onnx_get_initializer_array(graph, constant_input_name)
            if constant_array is None or constant_array.ndim < 2:
                continue

            if len(reshape_shape) == 4 and int(reshape_shape[1]) != 1 and int(reshape_shape[3]) == 1:
                nchw_shape = np.asarray(
                    [int(reshape_shape[0]), 1, int(reshape_shape[1]), int(reshape_shape[2])],
                    dtype=np.int64,
                )
                transposed_constant_array = np.swapaxes(constant_array, -1, -2)
                new_constant_name = _onnx_make_unique_initializer_name(graph, f"{constant_input_name}_nchw")
                _onnx_set_initializer_array(
                    graph,
                    name=new_constant_name,
                    array=np.asarray(transposed_constant_array),
                )

                new_shape_name = _onnx_make_unique_initializer_name(graph, f"{reshape_shape_name}_nchw")
                _onnx_set_initializer_array(graph, name=new_shape_name, array=nchw_shape)
                reshape_node.input[1] = str(new_shape_name)
                matmul_node.input[0] = str(reshape_node.output[0])
                matmul_node.input[1] = str(new_constant_name)
                for user_node in list(consumer_map.get(str(matmul_node.output[0]), [])):
                    if (
                        str(user_node.op_type) == "Transpose"
                        and list(_onnx_node_attr(user_node, "perm") or []) == [0, 3, 1, 2]
                    ):
                        _onnx_replace_all_node_inputs(
                            graph,
                            old_name=str(user_node.output[0]),
                            new_name=str(matmul_node.output[0]),
                        )
                        remove_node_names.append(str(user_node.name))
                _onnx_remove_nodes_by_name(graph, remove_node_names)
                optimized = True
                break

            if len(reshape_shape) != 3 or constant_array.ndim != 3:
                continue
            matmul_output_name = str(matmul_node.output[0]) if matmul_node.output else ""
            if matmul_output_name == "":
                continue
            matmul_users = consumer_map.get(matmul_output_name, [])
            if len(matmul_users) != 1:
                continue
            trailing_reshape_node = matmul_users[0]
            if str(trailing_reshape_node.op_type) != "Reshape" or len(trailing_reshape_node.input) < 2:
                continue
            trailing_shape_name = str(trailing_reshape_node.input[1])
            trailing_shape_array = _onnx_get_initializer_array(graph, trailing_shape_name)
            if trailing_shape_array is None:
                continue
            trailing_shape = [int(v) for v in trailing_shape_array.reshape(-1)]
            if len(trailing_shape) != 4 or int(trailing_shape[1]) != 1:
                continue
            if int(trailing_shape[2]) != int(constant_array.shape[-2]) or int(trailing_shape[3]) != int(reshape_shape[-1]):
                continue

            nchw_shape = np.asarray(
                [int(reshape_shape[0]), 1, int(reshape_shape[1]), int(reshape_shape[2])],
                dtype=np.int64,
            )
            expanded_constant_array = np.expand_dims(np.asarray(constant_array), axis=1)
            new_constant_name = _onnx_make_unique_initializer_name(graph, f"{constant_input_name}_nchw")
            _onnx_set_initializer_array(
                graph,
                name=new_constant_name,
                array=expanded_constant_array,
            )
            new_shape_name = _onnx_make_unique_initializer_name(graph, f"{reshape_shape_name}_nchw")
            _onnx_set_initializer_array(graph, name=new_shape_name, array=nchw_shape)
            reshape_node.input[1] = str(new_shape_name)
            matmul_node.input[constant_input_index] = str(new_constant_name)
            _onnx_replace_all_node_inputs(
                graph,
                old_name=str(trailing_reshape_node.output[0]),
                new_name=str(matmul_node.output[0]),
            )
            remove_node_names.append(str(trailing_reshape_node.name))
            _onnx_remove_nodes_by_name(graph, remove_node_names)
            optimized = True
            break
        if not optimized:
            break


def _onnx_fold_singleton_slice_layout_bridges_in_place(graph: onnx.GraphProto) -> None:
    while True:
        producer_map, consumer_map = _onnx_node_maps(graph)
        remove_node_names: List[str] = []
        optimized = False

        for transpose_node in list(graph.node):
            if str(transpose_node.op_type) != "Transpose":
                continue
            transpose_perm = list(_onnx_node_attr(transpose_node, "perm") or [])
            if transpose_perm not in ([0, 2, 3, 1], [0, 3, 1, 2]):
                continue
            if not transpose_node.input or not transpose_node.output:
                continue
            transpose_input_name = str(transpose_node.input[0])
            transpose_output_name = str(transpose_node.output[0])
            transpose_consumers = consumer_map.get(transpose_output_name, [])
            if not transpose_consumers:
                continue
            inverse_perm = [0, 3, 1, 2] if transpose_perm == [0, 2, 3, 1] else [0, 2, 3, 1]
            slice_nodes: List[onnx.NodeProto] = []
            inverse_transpose_nodes: List[onnx.NodeProto] = []
            supported = True
            for consumer_node in transpose_consumers:
                if str(consumer_node.op_type) == "Slice":
                    slice_nodes.append(consumer_node)
                    continue
                if (
                    str(consumer_node.op_type) == "Transpose"
                    and list(_onnx_node_attr(consumer_node, "perm") or []) == inverse_perm
                ):
                    inverse_transpose_nodes.append(consumer_node)
                    continue
                supported = False
                break
            if not supported or not slice_nodes:
                continue
            if not all(_onnx_rewrite_slice_axis_to_nchw_in_place(graph, slice_node=node) for node in slice_nodes):
                continue
            for slice_node in slice_nodes:
                slice_node.input[0] = str(transpose_input_name)

            remove_node_names.append(str(transpose_node.name))
            for inverse_transpose_node in inverse_transpose_nodes:
                _onnx_replace_all_node_inputs(
                    graph,
                    old_name=str(inverse_transpose_node.output[0]),
                    new_name=transpose_input_name,
                )
                remove_node_names.append(str(inverse_transpose_node.name))

            for slice_node in slice_nodes:
                slice_output_name = str(slice_node.output[0]) if slice_node.output else ""
                if slice_output_name == "":
                    continue
                for user_node in list(consumer_map.get(slice_output_name, [])):
                    if str(user_node.op_type) != "Transpose":
                        continue
                    user_perm = list(_onnx_node_attr(user_node, "perm") or [])
                    if transpose_perm == [0, 2, 3, 1] and user_perm != [0, 3, 1, 2]:
                        continue
                    if transpose_perm == [0, 3, 1, 2] and user_perm != [0, 2, 3, 1]:
                        continue
                    _onnx_replace_all_node_inputs(
                        graph,
                        old_name=str(user_node.output[0]),
                        new_name=slice_output_name,
                    )
                    remove_node_names.append(str(user_node.name))

            for user_node in list(transpose_consumers):
                slice_output_name = str(user_node.output[0]) if user_node.output else ""
                if slice_output_name == "":
                    continue
                passthrough_nodes = consumer_map.get(slice_output_name, [])
                for passthrough_node in list(passthrough_nodes):
                    if str(passthrough_node.op_type) not in {"Mul", "Add", "Sub", "Div", "Clip"}:
                        continue
                    passthrough_output_name = str(passthrough_node.output[0]) if passthrough_node.output else ""
                    if passthrough_output_name == "":
                        continue
                    for inverse_node in list(consumer_map.get(passthrough_output_name, [])):
                        if str(inverse_node.op_type) != "Transpose":
                            continue
                        inverse_perm = list(_onnx_node_attr(inverse_node, "perm") or [])
                        if transpose_perm == [0, 2, 3, 1] and inverse_perm != [0, 3, 1, 2]:
                            continue
                        if transpose_perm == [0, 3, 1, 2] and inverse_perm != [0, 2, 3, 1]:
                            continue
                        _onnx_replace_all_node_inputs(
                            graph,
                            old_name=str(inverse_node.output[0]),
                            new_name=passthrough_output_name,
                        )
                        remove_node_names.append(str(inverse_node.name))

            _onnx_remove_nodes_by_name(graph, remove_node_names)
            optimized = True
            break

        if not optimized:
            break


def _onnx_fold_singleton_concat_layout_bridges_in_place(graph: onnx.GraphProto) -> None:
    layout_preserving_unary_op_types = {
        "Clip",
        "Exp",
        "Identity",
        "LeakyRelu",
        "Relu",
        "Sigmoid",
    }
    scalar_binary_op_types = {
        "Add",
        "Div",
        "Mul",
        "Sub",
    }
    while True:
        producer_map, consumer_map = _onnx_node_maps(graph)
        optimized = False
        for inverse_transpose_node in list(graph.node):
            if str(inverse_transpose_node.op_type) != "Transpose":
                continue
            if list(_onnx_node_attr(inverse_transpose_node, "perm") or []) != [0, 3, 1, 2]:
                continue
            concat_node_name = str(inverse_transpose_node.input[0]) if inverse_transpose_node.input else ""
            concat_node = producer_map.get(concat_node_name)
            if concat_node is None or str(concat_node.op_type) != "Concat":
                continue
            if int(_onnx_node_attr(concat_node, "axis") or -1) != 3:
                continue
            if len(consumer_map.get(str(concat_node.output[0]), [])) != 1:
                continue

            rewritten_inputs: List[str] = []
            remove_node_names: List[str] = [str(inverse_transpose_node.name)]
            valid = True
            for input_name in list(concat_node.input):
                input_producer = producer_map.get(str(input_name))
                if input_producer is None:
                    valid = False
                    break
                if str(input_producer.op_type) == "Slice":
                    axes_array = _onnx_get_initializer_array(
                        graph,
                        str(input_producer.input[3]) if len(input_producer.input) >= 4 else "",
                    )
                    if axes_array is not None and [int(v) for v in axes_array.reshape(-1)] == [1]:
                        rewritten_inputs.append(str(input_name))
                        continue
                input_shape = _onnx_resolve_rank4_shape(
                    graph,
                    str(input_name),
                    producer_map=producer_map,
                )
                if (
                    input_shape is not None
                    and len(input_shape) == 4
                    and int(input_shape[1]) == 1
                    and int(input_shape[2]) > 0
                    and int(input_shape[3]) > 0
                ):
                    rewritten_inputs.append(str(input_name))
                    continue
                if (
                    str(input_producer.op_type) == "Transpose"
                    and list(_onnx_node_attr(input_producer, "perm") or []) == [0, 2, 3, 1]
                    and len(consumer_map.get(str(input_producer.output[0]), [])) == 1
                ):
                    rewritten_inputs.append(str(input_producer.input[0]))
                    remove_node_names.append(str(input_producer.name))
                    continue
                if str(input_producer.op_type) in layout_preserving_unary_op_types and len(input_producer.input) >= 1:
                    unary_input_name = str(input_producer.input[0])
                    unary_input_producer = producer_map.get(unary_input_name)
                    if (
                        unary_input_producer is not None
                        and str(unary_input_producer.op_type) == "Transpose"
                        and list(_onnx_node_attr(unary_input_producer, "perm") or []) == [0, 2, 3, 1]
                        and len(consumer_map.get(str(unary_input_producer.output[0]), [])) == 1
                    ):
                        input_producer.input[0] = str(unary_input_producer.input[0])
                        rewritten_inputs.append(str(input_producer.output[0]))
                        remove_node_names.append(str(unary_input_producer.name))
                        continue
                if str(input_producer.op_type) in scalar_binary_op_types and len(input_producer.input) == 2:
                    transpose_input_index: Optional[int] = None
                    for input_index, candidate_input_name in enumerate(list(input_producer.input)):
                        candidate_producer = producer_map.get(str(candidate_input_name))
                        if (
                            candidate_producer is not None
                            and str(candidate_producer.op_type) == "Transpose"
                            and list(_onnx_node_attr(candidate_producer, "perm") or []) == [0, 2, 3, 1]
                            and len(consumer_map.get(str(candidate_producer.output[0]), [])) == 1
                        ):
                            transpose_input_index = int(input_index)
                            break
                    if transpose_input_index is not None:
                        transpose_input_name = str(input_producer.input[transpose_input_index])
                        transpose_input_producer = producer_map.get(transpose_input_name)
                        if transpose_input_producer is not None:
                            input_producer.input[transpose_input_index] = str(transpose_input_producer.input[0])
                            rewritten_inputs.append(str(input_producer.output[0]))
                            remove_node_names.append(str(transpose_input_producer.name))
                            continue
                valid = False
                break
            if not valid or not rewritten_inputs:
                continue

            for input_index, rewritten_input_name in enumerate(rewritten_inputs):
                concat_node.input[input_index] = str(rewritten_input_name)
            _onnx_set_node_attr(concat_node, "axis", 1)
            _onnx_replace_all_node_inputs(
                graph,
                old_name=str(inverse_transpose_node.output[0]),
                new_name=str(concat_node.output[0]),
            )
            _onnx_remove_nodes_by_name(graph, remove_node_names)
            optimized = True
            break
        if not optimized:
            break


def _onnx_fold_singleton_concat_slice_layout_bridges_in_place(graph: onnx.GraphProto) -> None:
    while True:
        producer_map, consumer_map = _onnx_node_maps(graph)
        optimized = False
        for concat_node in list(graph.node):
            if str(concat_node.op_type) != "Concat":
                continue
            if int(_onnx_node_attr(concat_node, "axis") or -1) != 3:
                continue

            rewritten_inputs: List[str] = []
            remove_node_names: List[str] = []
            valid_inputs = True
            for input_name in list(concat_node.input):
                input_str = str(input_name)
                input_shape = _onnx_resolve_rank4_shape(
                    graph,
                    input_str,
                    producer_map=producer_map,
                )
                if (
                    input_shape is not None
                    and len(input_shape) == 4
                    and int(input_shape[1]) == 1
                    and int(input_shape[2]) > 0
                    and int(input_shape[3]) > 0
                ):
                    rewritten_inputs.append(input_str)
                    continue
                input_producer = producer_map.get(input_str)
                if (
                    input_producer is not None
                    and str(input_producer.op_type) == "Transpose"
                    and list(_onnx_node_attr(input_producer, "perm") or []) == [0, 2, 3, 1]
                    and len(consumer_map.get(str(input_producer.output[0]), [])) == 1
                ):
                    source_name = str(input_producer.input[0]) if input_producer.input else ""
                    source_shape = _onnx_resolve_rank4_shape(
                        graph,
                        source_name,
                        producer_map=producer_map,
                    )
                    if (
                        source_name != ""
                        and source_shape is not None
                        and len(source_shape) == 4
                        and int(source_shape[1]) == 1
                    ):
                        rewritten_inputs.append(source_name)
                        remove_node_names.append(str(input_producer.name))
                        continue
                valid_inputs = False
                break
            if not valid_inputs or not rewritten_inputs:
                continue

            concat_consumers = consumer_map.get(str(concat_node.output[0]), [])
            if not concat_consumers:
                continue
            if not all(str(node.op_type) == "Slice" for node in concat_consumers):
                continue
            if not all(_onnx_rewrite_slice_axis_to_nchw_in_place(graph, slice_node=node) for node in concat_consumers):
                continue

            for input_index, rewritten_input_name in enumerate(rewritten_inputs):
                concat_node.input[input_index] = str(rewritten_input_name)
            _onnx_set_node_attr(concat_node, "axis", 1)
            _onnx_remove_nodes_by_name(graph, remove_node_names)
            optimized = True
            break
        if not optimized:
            break


def _onnx_fold_singleton_binary_layout_bridges_in_place(graph: onnx.GraphProto) -> None:
    binary_op_types = {
        "Add",
        "Div",
        "Mul",
        "Sub",
    }
    producer_map, consumer_map = _onnx_node_maps(graph)
    remove_node_names: List[str] = []
    for transpose_node in list(graph.node):
        if str(transpose_node.op_type) != "Transpose":
            continue
        if list(_onnx_node_attr(transpose_node, "perm") or []) != [0, 3, 1, 2]:
            continue
        transpose_input_name = str(transpose_node.input[0]) if transpose_node.input else ""
        transpose_output_name = str(transpose_node.output[0]) if transpose_node.output else ""
        if transpose_input_name == "" or transpose_output_name == "":
            continue
        transpose_input_shape = _onnx_resolve_rank4_shape(
            graph,
            transpose_input_name,
            producer_map=producer_map,
        )
        if (
            transpose_input_shape is None
            or len(transpose_input_shape) != 4
            or int(transpose_input_shape[1]) != 1
        ):
            continue
        transpose_consumers = consumer_map.get(transpose_output_name, [])
        if not transpose_consumers:
            continue
        rewrites: List[Tuple[onnx.NodeProto, int]] = []
        valid = True
        for binary_node in list(transpose_consumers):
            if str(binary_node.op_type) not in binary_op_types or len(binary_node.input) != 2:
                valid = False
                break
            binary_input_index = None
            other_input_name = ""
            for input_index, input_name in enumerate(list(binary_node.input)):
                if str(input_name) == transpose_output_name:
                    binary_input_index = int(input_index)
                    other_input_name = str(binary_node.input[1 - input_index])
                    break
            if binary_input_index is None or other_input_name == "":
                valid = False
                break
            other_input_shape = _onnx_resolve_rank4_shape(
                graph,
                other_input_name,
                producer_map=producer_map,
            )
            if other_input_shape is None:
                scalar_initializer = _onnx_get_initializer_array(graph, other_input_name)
                if scalar_initializer is not None and int(np.asarray(scalar_initializer).size) == 1:
                    rewrites.append((binary_node, int(binary_input_index)))
                    continue
                other_input_producer = producer_map.get(other_input_name)
                if (
                    other_input_producer is None
                    or str(other_input_producer.op_type) != "Concat"
                    or int(_onnx_node_attr(other_input_producer, "axis") or -1) != 1
                ):
                    valid = False
                    break
            else:
                if (
                    len(other_input_shape) != 4
                    or int(other_input_shape[1]) <= 0
                    or int(other_input_shape[2]) != int(transpose_input_shape[2])
                    or int(other_input_shape[3]) != int(transpose_input_shape[3])
                ):
                    valid = False
                    break
            rewrites.append((binary_node, int(binary_input_index)))
        if not valid:
            continue
        for binary_node, binary_input_index in rewrites:
            binary_node.input[binary_input_index] = transpose_input_name
        remove_node_names.append(str(transpose_node.name))
    _onnx_remove_nodes_by_name(graph, remove_node_names)


def _onnx_fold_binary_layout_bridges_in_place(graph: onnx.GraphProto) -> None:
    binary_op_types = {
        "Add",
        "Div",
        "Max",
        "Min",
        "Mul",
        "Sub",
    }
    layout_preserving_unary_op_types = {
        "Clip",
        "Exp",
        "Identity",
        "LeakyRelu",
        "Relu",
        "Sigmoid",
    }
    producer_map, consumer_map = _onnx_node_maps(graph)
    remove_node_names: List[str] = []
    for binary_node in list(graph.node):
        if str(binary_node.op_type) not in binary_op_types or len(binary_node.input) != 2:
            continue
        rewritten_inputs: List[str] = []
        local_remove_node_names: List[str] = []
        rewritten_any = False
        expected_shape: Optional[List[int]] = None
        valid = True
        for input_name in list(binary_node.input):
            input_str = str(input_name)
            scalar_initializer = _onnx_get_initializer_array(graph, input_str)
            if scalar_initializer is not None and int(np.asarray(scalar_initializer).size) == 1:
                rewritten_inputs.append(input_str)
                continue
            input_producer = producer_map.get(input_str)
            if (
                input_producer is not None
                and str(input_producer.op_type) == "Transpose"
                and list(_onnx_node_attr(input_producer, "perm") or []) == [0, 2, 3, 1]
                and len(consumer_map.get(str(input_producer.output[0]), [])) == 1
                and input_producer.input
            ):
                source_name = str(input_producer.input[0])
                source_shape = _onnx_resolve_rank4_shape(
                    graph,
                    source_name,
                    producer_map=producer_map,
                )
                if source_shape is None or len(source_shape) != 4:
                    valid = False
                    break
                if expected_shape is None:
                    expected_shape = [int(v) for v in list(source_shape)]
                elif [int(v) for v in list(source_shape)] != expected_shape:
                    valid = False
                    break
                rewritten_inputs.append(source_name)
                local_remove_node_names.append(str(input_producer.name))
                rewritten_any = True
                continue
            input_shape = _onnx_resolve_rank4_shape(
                graph,
                input_str,
                producer_map=producer_map,
            )
            input_producer = producer_map.get(input_str)
            if (
                input_producer is not None
                and str(input_producer.op_type) in layout_preserving_unary_op_types
                and input_producer.input
            ):
                unary_input_producer = producer_map.get(str(input_producer.input[0]))
                if (
                    unary_input_producer is not None
                    and str(unary_input_producer.op_type) == "Transpose"
                    and list(_onnx_node_attr(unary_input_producer, "perm") or []) == [0, 2, 3, 1]
                ):
                    valid = False
                    break
            if input_shape is None or len(input_shape) != 4:
                valid = False
                break
            if expected_shape is None:
                expected_shape = [int(v) for v in list(input_shape)]
            elif [int(v) for v in list(input_shape)] != expected_shape:
                valid = False
                break
            rewritten_inputs.append(input_str)
        if not valid or not rewritten_any or len(rewritten_inputs) != 2:
            continue
        for input_index, rewritten_input_name in enumerate(rewritten_inputs):
            binary_node.input[input_index] = str(rewritten_input_name)
        remove_node_names.extend(local_remove_node_names)
    _onnx_remove_nodes_by_name(graph, remove_node_names)


def _onnx_fold_unary_binary_layout_bridges_in_place(graph: onnx.GraphProto) -> None:
    binary_op_types = {
        "Add",
        "Div",
        "Max",
        "Min",
        "Mul",
        "Sub",
    }
    layout_preserving_unary_op_types = {
        "Clip",
        "Exp",
        "Identity",
        "LeakyRelu",
        "Relu",
        "Sigmoid",
    }
    producer_map, consumer_map = _onnx_node_maps(graph)
    remove_node_names: List[str] = []
    for inverse_transpose_node in list(graph.node):
        if str(inverse_transpose_node.op_type) != "Transpose":
            continue
        if list(_onnx_node_attr(inverse_transpose_node, "perm") or []) != [0, 3, 1, 2]:
            continue
        binary_output_name = str(inverse_transpose_node.input[0]) if inverse_transpose_node.input else ""
        if binary_output_name == "":
            continue
        binary_node = producer_map.get(binary_output_name)
        if binary_node is None or str(binary_node.op_type) not in binary_op_types or len(binary_node.input) != 2:
            continue
        if len(consumer_map.get(str(binary_node.output[0]), [])) != 1:
            continue
        rewritten_inputs: List[str] = []
        local_remove_node_names: List[str] = [str(inverse_transpose_node.name)]
        valid = True
        rewritten_any = False
        for input_name in list(binary_node.input):
            input_str = str(input_name)
            scalar_initializer = _onnx_get_initializer_array(graph, input_str)
            if scalar_initializer is not None and int(np.asarray(scalar_initializer).size) == 1:
                rewritten_inputs.append(input_str)
                continue
            input_producer = producer_map.get(input_str)
            if (
                input_producer is not None
                and str(input_producer.op_type) == "Transpose"
                and list(_onnx_node_attr(input_producer, "perm") or []) == [0, 2, 3, 1]
                and len(consumer_map.get(str(input_producer.output[0]), [])) == 1
                and input_producer.input
            ):
                rewritten_inputs.append(str(input_producer.input[0]))
                local_remove_node_names.append(str(input_producer.name))
                rewritten_any = True
                continue
            if (
                input_producer is not None
                and str(input_producer.op_type) in layout_preserving_unary_op_types
                and len(consumer_map.get(str(input_producer.output[0]), [])) == 1
                and input_producer.input
            ):
                unary_input_name = str(input_producer.input[0])
                unary_input_producer = producer_map.get(unary_input_name)
                if (
                    unary_input_producer is not None
                    and str(unary_input_producer.op_type) == "Transpose"
                    and list(_onnx_node_attr(unary_input_producer, "perm") or []) == [0, 2, 3, 1]
                    and len(consumer_map.get(str(unary_input_producer.output[0]), [])) == 1
                    and unary_input_producer.input
                ):
                    input_producer.input[0] = str(unary_input_producer.input[0])
                    rewritten_inputs.append(str(input_producer.output[0]))
                    local_remove_node_names.append(str(unary_input_producer.name))
                    rewritten_any = True
                    continue
            valid = False
            break
        if not valid or not rewritten_any or len(rewritten_inputs) != 2:
            continue
        for input_index, rewritten_input_name in enumerate(rewritten_inputs):
            binary_node.input[input_index] = str(rewritten_input_name)
        binary_node.output[0] = str(inverse_transpose_node.output[0])
        remove_node_names.extend(local_remove_node_names)
    _onnx_remove_nodes_by_name(graph, remove_node_names)


def _onnx_fold_unary_binary_reduce_mean_layout_bridges_in_place(graph: onnx.GraphProto) -> None:
    binary_op_types = {
        "Add",
        "Div",
        "Max",
        "Min",
        "Mul",
        "Sub",
    }
    layout_preserving_unary_op_types = {
        "Clip",
        "Exp",
        "Identity",
        "LeakyRelu",
        "Relu",
        "Sigmoid",
        "Tanh",
    }
    producer_map, consumer_map = _onnx_node_maps(graph)
    remove_node_names: List[str] = []
    for inverse_transpose_node in list(graph.node):
        if str(inverse_transpose_node.op_type) != "Transpose":
            continue
        if list(_onnx_node_attr(inverse_transpose_node, "perm") or []) != [0, 3, 1, 2]:
            continue
        binary_output_name = str(inverse_transpose_node.input[0]) if inverse_transpose_node.input else ""
        if binary_output_name == "":
            continue
        binary_node = producer_map.get(binary_output_name)
        if binary_node is None or str(binary_node.op_type) not in binary_op_types or len(binary_node.input) != 2:
            continue
        rewritten_inputs: List[str] = []
        local_remove_node_names: List[str] = [str(inverse_transpose_node.name)]
        rewritten_any = False
        valid = True
        for input_name in list(binary_node.input):
            input_str = str(input_name)
            scalar_initializer = _onnx_get_initializer_array(graph, input_str)
            if scalar_initializer is not None and int(np.asarray(scalar_initializer).size) == 1:
                rewritten_inputs.append(input_str)
                continue
            input_producer = producer_map.get(input_str)
            if (
                input_producer is not None
                and str(input_producer.op_type) == "Transpose"
                and list(_onnx_node_attr(input_producer, "perm") or []) == [0, 2, 3, 1]
                and len(consumer_map.get(str(input_producer.output[0]), [])) == 1
                and input_producer.input
            ):
                rewritten_inputs.append(str(input_producer.input[0]))
                local_remove_node_names.append(str(input_producer.name))
                rewritten_any = True
                continue
            if (
                input_producer is not None
                and str(input_producer.op_type) in layout_preserving_unary_op_types
                and len(consumer_map.get(str(input_producer.output[0]), [])) == 1
                and input_producer.input
            ):
                unary_input_name = str(input_producer.input[0])
                unary_input_producer = producer_map.get(unary_input_name)
                if (
                    unary_input_producer is not None
                    and str(unary_input_producer.op_type) == "Transpose"
                    and list(_onnx_node_attr(unary_input_producer, "perm") or []) == [0, 2, 3, 1]
                    and len(consumer_map.get(str(unary_input_producer.output[0]), [])) == 1
                    and unary_input_producer.input
                ):
                    input_producer.input[0] = str(unary_input_producer.input[0])
                    rewritten_inputs.append(str(input_producer.output[0]))
                    local_remove_node_names.append(str(unary_input_producer.name))
                    rewritten_any = True
                    continue
            valid = False
            break
        if not valid or not rewritten_any or len(rewritten_inputs) != 2:
            continue

        binary_consumers = list(consumer_map.get(str(binary_node.output[0]), []))
        for consumer_node in binary_consumers:
            if str(consumer_node.name) == str(inverse_transpose_node.name):
                continue
            if str(consumer_node.op_type) != "ReduceMean":
                valid = False
                break
            if len(consumer_map.get(str(consumer_node.output[0]), [])) != 1:
                valid = False
                break
            axes_name = str(consumer_node.input[1]) if len(consumer_node.input) >= 2 else ""
            axes_array = _onnx_get_initializer_array(graph, axes_name)
            if axes_array is None or [int(v) for v in axes_array.reshape(-1)] != [1, 2]:
                valid = False
                break
            if int(_onnx_node_attr(consumer_node, "keepdims") or 0) != 1:
                valid = False
                break
            reduce_inverse_node = consumer_map.get(str(consumer_node.output[0]), [None])[0]
            if (
                reduce_inverse_node is None
                or str(reduce_inverse_node.op_type) != "Transpose"
                or list(_onnx_node_attr(reduce_inverse_node, "perm") or []) != [0, 3, 1, 2]
            ):
                valid = False
                break
            new_axes_name = _onnx_make_unique_initializer_name(graph, f"{axes_name}_nchw")
            _onnx_set_initializer_array(graph, name=new_axes_name, array=np.asarray([2, 3], dtype=np.int64))
            consumer_node.input[1] = str(new_axes_name)
            _onnx_replace_all_node_inputs(
                graph,
                old_name=str(reduce_inverse_node.output[0]),
                new_name=str(consumer_node.output[0]),
            )
            local_remove_node_names.append(str(reduce_inverse_node.name))

        if not valid:
            continue

        for input_index, rewritten_input_name in enumerate(rewritten_inputs):
            binary_node.input[input_index] = str(rewritten_input_name)
        _onnx_replace_all_node_inputs(
            graph,
            old_name=str(inverse_transpose_node.output[0]),
            new_name=str(binary_node.output[0]),
        )
        remove_node_names.extend(local_remove_node_names)
    _onnx_remove_nodes_by_name(graph, remove_node_names)


def _onnx_fold_mul_add_clip_to_hardsigmoid_in_place(graph: onnx.GraphProto) -> None:
    producer_map, consumer_map = _onnx_node_maps(graph)
    graph_output_names = {str(output.name) for output in graph.output}
    remove_node_names: List[str] = []
    alpha_ref = float(1.0 / 6.0)
    beta_ref = 0.5
    eps = 1e-6
    for clip_node in list(graph.node):
        if str(clip_node.op_type) != "Clip" or len(clip_node.input) != 3 or len(clip_node.output) != 1:
            continue
        clip_min = _onnx_get_initializer_scalar(graph, str(clip_node.input[1]))
        clip_max = _onnx_get_initializer_scalar(graph, str(clip_node.input[2]))
        if (
            clip_min is None
            or clip_max is None
            or abs(float(clip_min) - 0.0) > eps
            or abs(float(clip_max) - 1.0) > eps
        ):
            continue
        add_node = producer_map.get(str(clip_node.input[0]))
        if add_node is None or str(add_node.op_type) != "Add" or len(add_node.input) != 2 or len(add_node.output) != 1:
            continue
        if len(consumer_map.get(str(add_node.output[0]), [])) != 1:
            continue
        if str(add_node.output[0]) in graph_output_names:
            continue
        add_const_index: Optional[int] = None
        add_const_value: Optional[float] = None
        mul_output_index: Optional[int] = None
        for input_index, input_name in enumerate(list(add_node.input)):
            scalar_value = _onnx_get_initializer_scalar(graph, str(input_name))
            if scalar_value is not None:
                add_const_index = int(input_index)
                add_const_value = float(scalar_value)
            else:
                mul_output_index = int(input_index)
        if (
            add_const_index is None
            or mul_output_index is None
            or add_const_value is None
            or abs(float(add_const_value) - beta_ref) > eps
        ):
            continue
        mul_node = producer_map.get(str(add_node.input[mul_output_index]))
        if mul_node is None or str(mul_node.op_type) != "Mul" or len(mul_node.input) != 2 or len(mul_node.output) != 1:
            continue
        if len(consumer_map.get(str(mul_node.output[0]), [])) != 1:
            continue
        if str(mul_node.output[0]) in graph_output_names:
            continue
        mul_const_index: Optional[int] = None
        mul_const_value: Optional[float] = None
        source_input_index: Optional[int] = None
        for input_index, input_name in enumerate(list(mul_node.input)):
            scalar_value = _onnx_get_initializer_scalar(graph, str(input_name))
            if scalar_value is not None:
                mul_const_index = int(input_index)
                mul_const_value = float(scalar_value)
            else:
                source_input_index = int(input_index)
        if (
            mul_const_index is None
            or source_input_index is None
            or mul_const_value is None
            or abs(float(mul_const_value) - alpha_ref) > eps
        ):
            continue
        add_node.op_type = "HardSigmoid"
        add_node.input[:] = [str(mul_node.input[source_input_index])]
        add_node.output[:] = [str(clip_node.output[0])]
        del add_node.attribute[:]
        _onnx_set_node_attr(add_node, "alpha", float(alpha_ref))
        _onnx_set_node_attr(add_node, "beta", float(beta_ref))
        remove_node_names.extend([str(mul_node.name), str(clip_node.name)])
    _onnx_remove_nodes_by_name(graph, remove_node_names)


def _onnx_fold_softmax_layout_bridges_in_place(graph: onnx.GraphProto) -> None:
    producer_map, consumer_map = _onnx_node_maps(graph)
    remove_node_names: List[str] = []
    for inverse_transpose_node in list(graph.node):
        if str(inverse_transpose_node.op_type) != "Transpose":
            continue
        if list(_onnx_node_attr(inverse_transpose_node, "perm") or []) != [0, 3, 1, 2]:
            continue
        softmax_node_name = str(inverse_transpose_node.input[0]) if inverse_transpose_node.input else ""
        softmax_node = producer_map.get(softmax_node_name)
        if softmax_node is None or str(softmax_node.op_type) != "Softmax":
            continue
        if int(_onnx_node_attr(softmax_node, "axis") or -1) != 3:
            continue
        if len(consumer_map.get(str(softmax_node.output[0]), [])) != 1:
            continue
        if not softmax_node.input:
            continue
        input_transpose_node = producer_map.get(str(softmax_node.input[0]))
        if input_transpose_node is None or str(input_transpose_node.op_type) != "Transpose":
            continue
        if list(_onnx_node_attr(input_transpose_node, "perm") or []) != [0, 2, 3, 1]:
            continue
        if len(consumer_map.get(str(input_transpose_node.output[0]), [])) != 1:
            continue

        softmax_node.input[0] = str(input_transpose_node.input[0])
        _onnx_set_node_attr(softmax_node, "axis", 1)
        _onnx_replace_all_node_inputs(
            graph,
            old_name=str(inverse_transpose_node.output[0]),
            new_name=str(softmax_node.output[0]),
        )
        remove_node_names.extend(
            [
                str(input_transpose_node.name),
                str(inverse_transpose_node.name),
            ]
        )
    _onnx_remove_nodes_by_name(graph, remove_node_names)
