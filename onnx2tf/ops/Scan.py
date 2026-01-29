import re
import sys
import random
random.seed(0)
import numpy as np
np.random.seed(0)
import importlib
import tensorflow as tf
import tf_keras
import onnx_graphsurgeon as gs
from onnx2tf.utils.common_functions import (
    get_constant_or_variable,
    convert_axis,
    print_node_info,
    inverted_operation_enable_disable,
    make_tf_node_info,
    get_replacement_parameter,
    pre_process_transpose,
    post_process_transpose,
)
from onnx2tf.utils.enums import NUMPY_DTYPES_TO_TF_DTYPES
from onnx2tf.utils.logging import *


class While_Loop_CustomLayer(tf_keras.layers.Layer):
    def __init__(self):
        super(While_Loop_CustomLayer, self).__init__()

    def call(self, cond, body, loop_vars, shape_invariants, maximum_iterations=None):
        return tf.while_loop(
            cond=cond,
            body=body,
            loop_vars=loop_vars,
            shape_invariants=shape_invariants,
            maximum_iterations=maximum_iterations,
        )


def _to_tf_dtype(dtype):
    return NUMPY_DTYPES_TO_TF_DTYPES[dtype] if isinstance(dtype, np.dtype) else dtype


def _as_tensor(value):
    if isinstance(value, np.ndarray):
        return tf.convert_to_tensor(value)
    if isinstance(value, (np.generic, int, float, bool, str, bytes)):
        return tf.convert_to_tensor(value)
    return value


def _shape_invariant(value):
    try:
        shape = value.shape
    except Exception:
        return tf.TensorShape(None)
    if shape is None:
        return tf.TensorShape(None)
    if isinstance(shape, tf.TensorShape):
        if shape.rank is None:
            return tf.TensorShape(None)
        return tf.TensorShape([None for _ in range(shape.rank)])
    return tf.TensorShape([None for _ in range(len(shape))])


def _sanitize(name, output_signaturedefs):
    name = name.replace(':', '__')
    if output_signaturedefs:
        name = re.sub('^/', 'wa/', name)
    return name


@print_node_info
@inverted_operation_enable_disable
@get_replacement_parameter
def make_node(
    *,
    graph_node: gs.Node,
    tf_layers_dict: dict,
    **kwargs: dict,
):
    """Scan

    Parameters
    ----------
    graph_node: gs.Node
        graph_surgeon Node

    tf_layers_dict: dict
        optype, shape, dtype, tensorflow graph
    """
    num_scan_inputs = int(graph_node.attrs.get('num_scan_inputs', 0))
    if num_scan_inputs <= 0:
        error(
            f'num_scan_inputs must be > 0 for Scan.\n' +
            f'graph_node.name: {graph_node.name}'
        )
        sys.exit(1)

    total_inputs = len(graph_node.inputs)
    num_state_vars = total_inputs - num_scan_inputs
    if num_state_vars < 0:
        error(
            f'Invalid num_scan_inputs for Scan.\n' +
            f'graph_node.name: {graph_node.name}'
        )
        sys.exit(1)

    state_inputs = list(graph_node.inputs[:num_state_vars])
    scan_inputs = list(graph_node.inputs[num_state_vars:])

    state_values = []
    state_input_meta = []
    for state_input in state_inputs:
        before_op_output_shape_trans = \
            tf_layers_dict.get(state_input.name, {}).get('before_op_output_shape_trans', True)
        state_node = get_constant_or_variable(
            state_input,
            before_op_output_shape_trans,
        )
        state_val = tf_layers_dict[state_node.name]['tf_node'] \
            if isinstance(state_node, gs.Variable) else state_node
        state_val = _as_tensor(state_val)
        if isinstance(state_node, gs.Variable) \
            and state_node.shape is not None \
            and len(state_node.shape) == 0:
            state_val = tf.reshape(state_val, [])
        state_values.append(state_val)
        if isinstance(state_node, gs.Variable):
            state_input_meta.append(tf_layers_dict.get(state_node.name, {}))
        else:
            state_input_meta.append({})

    scan_input_tensors = []
    scan_input_meta = []
    scan_input_axes = graph_node.attrs.get('scan_input_axes', None)
    if scan_input_axes is None:
        scan_input_axes = [0 for _ in range(num_scan_inputs)]
    scan_input_directions = graph_node.attrs.get('scan_input_directions', None)
    if scan_input_directions is None:
        scan_input_directions = [0 for _ in range(num_scan_inputs)]
    if len(scan_input_axes) != num_scan_inputs or len(scan_input_directions) != num_scan_inputs:
        error(
            f'Invalid scan_input_axes or scan_input_directions for Scan.\n' +
            f'graph_node.name: {graph_node.name}'
        )
        sys.exit(1)

    converted_scan_input_axes = []
    for idx, scan_input in enumerate(scan_inputs):
        before_op_output_shape_trans = \
            tf_layers_dict.get(scan_input.name, {}).get('before_op_output_shape_trans', True)
        scan_node = get_constant_or_variable(
            scan_input,
            before_op_output_shape_trans,
        )
        scan_tensor = tf_layers_dict[scan_node.name]['tf_node'] \
            if isinstance(scan_node, gs.Variable) else scan_node
        scan_tensor = pre_process_transpose(
            value_before_transpose=scan_tensor,
            param_target='inputs',
            param_name=scan_input.name,
            **kwargs,
        )
        scan_tensor = _as_tensor(scan_tensor)
        scan_input_tensors.append(scan_tensor)
        if isinstance(scan_node, gs.Variable):
            scan_input_meta.append(tf_layers_dict.get(scan_node.name, {}))
        else:
            scan_input_meta.append({})
        scan_rank = scan_tensor.shape.rank
        if scan_rank is None and scan_input.shape is not None:
            scan_rank = len(scan_input.shape)
        if scan_rank is None:
            error(
                f'Scan input rank must be known.\n' +
                f'graph_node.name: {graph_node.name}'
            )
            sys.exit(1)
        converted_scan_input_axes.append(
            convert_axis(
                axis=int(scan_input_axes[idx]),
                tensor_rank=scan_rank,
                before_op_output_shape_trans=before_op_output_shape_trans,
            )
        )

    scan_outputs = list(graph_node.outputs[num_state_vars:])
    num_scan_outputs = len(scan_outputs)
    scan_output_axes = graph_node.attrs.get('scan_output_axes', None)
    if scan_output_axes is None:
        scan_output_axes = [0 for _ in range(num_scan_outputs)]
    scan_output_directions = graph_node.attrs.get('scan_output_directions', None)
    if scan_output_directions is None:
        scan_output_directions = [0 for _ in range(num_scan_outputs)]
    if len(scan_output_axes) != num_scan_outputs or len(scan_output_directions) != num_scan_outputs:
        error(
            f'Invalid scan_output_axes or scan_output_directions for Scan.\n' +
            f'graph_node.name: {graph_node.name}'
        )
        sys.exit(1)

    graph_node_outputs = list(graph_node.outputs)
    for graph_node_output in graph_node_outputs:
        tf_layers_dict[graph_node_output.name] = {
            'optype': graph_node.op,
            'shape': graph_node_output.shape,
            'dtype': graph_node_output.dtype,
        }

    body: gs.Graph = graph_node.attrs["body"]

    def _register_graph_output_constants(graph: gs.Graph):
        for output in graph.outputs:
            if output.name not in tf_layers_dict and isinstance(output, gs.Constant):
                tf_layers_dict[output.name] = {
                    'optype': 'Constant',
                    'shape': output.values.shape,
                    'dtype': output.values.dtype,
                }
                tf_layers_dict[output.name]['tf_node'] = \
                    tf.constant(
                        output.values,
                        dtype=_to_tf_dtype(output.values.dtype),
                    )
        for node in graph.nodes:
            for attr_val in node.attrs.values():
                if isinstance(attr_val, gs.Graph):
                    _register_graph_output_constants(attr_val)
                elif isinstance(attr_val, (list, tuple)):
                    for sub_val in attr_val:
                        if isinstance(sub_val, gs.Graph):
                            _register_graph_output_constants(sub_val)

    _register_graph_output_constants(body)

    if len(body.inputs) != (num_state_vars + num_scan_inputs):
        error(
            f'Body input count mismatch for Scan.\n' +
            f'graph_node.name: {graph_node.name}'
        )
        sys.exit(1)
    if len(body.outputs) < num_state_vars:
        error(
            f'Body output count mismatch for Scan.\n' +
            f'graph_node.name: {graph_node.name}'
        )
        sys.exit(1)

    scan_out_start_index = num_state_vars
    if len(body.outputs) != (num_state_vars + num_scan_outputs):
        error(
            f'Body output count mismatch for Scan.\n' +
            f'graph_node.name: {graph_node.name}'
        )
        sys.exit(1)

    # Determine sequence length from the first scan input
    sequence_length = tf.shape(scan_input_tensors[0])[converted_scan_input_axes[0]]
    sequence_length = tf.cast(sequence_length, tf.int32)

    scan_outputs_init = []
    for i in range(scan_out_start_index, len(body.outputs)):
        elem_shape = body.outputs[i].shape
        if elem_shape is not None:
            elem_shape = [
                dim if isinstance(dim, int) else None for dim in elem_shape
            ]
            elem_shape = tf.TensorShape(elem_shape)
        scan_outputs_init.append(
            tf.TensorArray(
                dtype=_to_tf_dtype(body.outputs[i].dtype),
                size=sequence_length,
                dynamic_size=False,
                element_shape=elem_shape,
            )
        )
    scan_outputs_shapes = [tf.TensorShape(None) for _ in scan_outputs_init]

    state_shapes = [_shape_invariant(v) for v in state_values]
    iter_cnt_init = tf.constant(0, dtype=tf.int32)

    def run_subgraph(iter_cnt, state_vals, scan_outputs_vals):
        scan_elems = []
        for i, scan_tensor in enumerate(scan_input_tensors):
            axis = converted_scan_input_axes[i]
            direction = int(scan_input_directions[i])
            if direction == 1:
                idx = sequence_length - 1 - iter_cnt
            else:
                idx = iter_cnt
            scan_elems.append(
                tf.gather(
                    params=scan_tensor,
                    indices=idx,
                    axis=axis,
                )
            )

        loop_inputs = list(state_vals) + scan_elems
        for idx, (body_input, loop_val) in enumerate(zip(body.inputs, loop_inputs)):
            body_input.name = _sanitize(body_input.name, kwargs.get('output_signaturedefs', False))
            target_dtype = _to_tf_dtype(body_input.dtype) if body_input.dtype is not None else None
            loop_val_cast = loop_val
            if target_dtype is not None \
                and isinstance(loop_val, tf.Tensor) \
                and loop_val.dtype != target_dtype:
                loop_val_cast = tf.cast(loop_val, target_dtype)
            if body_input.shape is not None \
                and len(body_input.shape) == 0:
                loop_val_cast = tf.reshape(loop_val_cast, [])
            tf_layers_dict[body_input.name] = {
                'optype': 'Input',
                'shape': body_input.shape,
                'dtype': body_input.dtype,
                'tf_node': loop_val_cast,
                'before_op_output_shape_trans': True,
            }
            meta = None
            if idx < num_state_vars:
                meta = state_input_meta[idx]
            else:
                meta = scan_input_meta[idx - num_state_vars]
            for key in ('before_op_output_shape_trans', 'nhwc'):
                if key in meta:
                    tf_layers_dict[body_input.name][key] = meta[key]

        subgraph_kwargs = dict(kwargs)
        subgraph_kwargs['suppress_log'] = True
        for body_node in body.nodes:
            optype = body_node.op
            try:
                op = importlib.import_module(f'onnx2tf.ops.{optype}')
            except ModuleNotFoundError:
                error(
                    f'{optype} OP is not yet implemented.'
                )
                sys.exit(1)
            body_node.name = _sanitize(body_node.name, kwargs.get('output_signaturedefs', False))
            op.make_node(
                graph_node=body_node,
                tf_layers_dict=tf_layers_dict,
                **subgraph_kwargs,
            )

        outputs = [tf_layers_dict[output.name]['tf_node'] for output in body.outputs]
        new_state_vals = outputs[:num_state_vars]
        scan_out_elems = outputs[scan_out_start_index:]

        updated_scan_outputs = []
        for i, ta in enumerate(scan_outputs_vals):
            direction = int(scan_output_directions[i])
            if direction == 1:
                write_idx = sequence_length - 1 - iter_cnt
            else:
                write_idx = iter_cnt
            updated_scan_outputs.append(
                ta.write(write_idx, scan_out_elems[i])
            )

        return [iter_cnt + 1, new_state_vals, updated_scan_outputs]

    def condition(iter_cnt, state_vals, scan_outputs_vals):
        return tf.less(iter_cnt, sequence_length)

    while_loop_layer = While_Loop_CustomLayer()
    iter_cnt_final, state_vals_final, scan_outputs_final = while_loop_layer(
        cond=condition,
        body=run_subgraph,
        loop_vars=[
            iter_cnt_init,
            state_values,
            scan_outputs_init,
        ],
        shape_invariants=[
            tf.TensorShape([]),
            state_shapes,
            scan_outputs_shapes,
        ],
        maximum_iterations=sequence_length,
    )

    scan_output_tensors = []
    for i, ta in enumerate(scan_outputs_final):
        out_tensor = ta.stack()
        out_rank = out_tensor.shape.rank
        if out_rank is None and scan_outputs[i].shape is not None:
            out_rank = len(scan_outputs[i].shape)
        axis = int(scan_output_axes[i])
        if out_rank is not None:
            axis = axis if axis >= 0 else axis + out_rank
        if out_rank is not None and axis != 0:
            perm = list(range(1, out_rank))
            perm.insert(axis, 0)
            out_tensor = tf.transpose(out_tensor, perm=perm)
        scan_output_tensors.append(out_tensor)

    final_outputs = list(state_vals_final) + scan_output_tensors
    if len(final_outputs) != len(graph_node_outputs):
        error(
            f'Scan output count mismatch. expected={len(graph_node_outputs)} actual={len(final_outputs)}\n' +
            f'graph_node.name: {graph_node.name}'
        )
        sys.exit(1)

    for idx, (graph_node_output, output_tensor) in enumerate(zip(graph_node_outputs, final_outputs)):
        tf_layers_dict[graph_node_output.name]['tf_node'] = output_tensor
        body_output = body.outputs[idx]
        body_meta = tf_layers_dict.get(body_output.name, {})
        for key in ('before_op_output_shape_trans', 'nhwc'):
            if key in body_meta:
                tf_layers_dict[graph_node_output.name][key] = body_meta[key]

        tf_layers_dict[graph_node_output.name]['tf_node'] = post_process_transpose(
            value_before_transpose=tf_layers_dict[graph_node_output.name]['tf_node'],
            param_target='outputs',
            param_name=graph_node_output.name,
            **kwargs,
        )

    tf_outputs = {f"output{idx}": value for idx, value in enumerate(final_outputs)}
    for graph_node_output in graph_node_outputs:
        tf_layers_dict[graph_node_output.name]['tf_node_info'] = \
            make_tf_node_info(
                node_info={
                    'tf_op_type': tf.while_loop,
                    'tf_inputs': {
                        'scan_input_axes': scan_input_axes,
                        'scan_input_directions': scan_input_directions,
                        'scan_output_axes': scan_output_axes,
                        'scan_output_directions': scan_output_directions,
                        'state_inputs': state_values,
                        'scan_inputs': scan_input_tensors,
                    },
                    'tf_outputs': {
                        'output': tf_outputs,
                    },
                }
            )
