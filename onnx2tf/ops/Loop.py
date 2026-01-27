import re
import sys
import random
random.seed(0)
import numpy as np
np.random.seed(0)
import tensorflow as tf
import tf_keras
import onnx_graphsurgeon as gs
from onnx2tf.utils.common_functions import (
    get_constant_or_variable,
    print_node_info,
    inverted_operation_enable_disable,
    make_tf_node_info,
)
from onnx2tf.utils.enums import NUMPY_DTYPES_TO_TF_DTYPES
import importlib
from onnx2tf.utils.logging import *


class While_Loop_CustomLayer(tf_keras.layers.Layer):
    def __init__(self):
        super(While_Loop_CustomLayer, self).__init__()

    def call(self, cond, body, loop_vars, shape_invariants, maximum_iterations):
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
    if isinstance(value, (np.generic, int, float, bool)):
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


@print_node_info
@inverted_operation_enable_disable
def make_node(
    *,
    graph_node: gs.Node,
    tf_layers_dict: dict,
    **kwargs: dict,
):
    """Loop

    Parameters
    ----------
    graph_node: gs.Node
        graph_surgeon Node

    tf_layers_dict: dict
        optype, shape, dtype, tensorflow graph
    """
    before_op_output_shape_trans_1 = \
        tf_layers_dict.get(graph_node.inputs[0].name, {}).get('before_op_output_shape_trans', True)
    before_op_output_shape_trans_2 = \
        tf_layers_dict.get(graph_node.inputs[1].name, {}).get('before_op_output_shape_trans', True)
    before_op_output_shape_trans = \
        before_op_output_shape_trans_1 \
        and before_op_output_shape_trans_2

    graph_node_input_1 = get_constant_or_variable(
        graph_node.inputs[0],
        before_op_output_shape_trans,
    )
    graph_node_input_2 = get_constant_or_variable(
        graph_node.inputs[1],
        before_op_output_shape_trans,
    )
    graph_node_input_n_list = []
    for graph_node_input in graph_node.inputs[2:]:
        graph_node_input_n = get_constant_or_variable(
            graph_node_input,
            before_op_output_shape_trans,
        )
        graph_node_input_n_list.append(graph_node_input_n)

    def _sanitize(name):
        name = name.replace(':', '__')
        if kwargs.get('output_signaturedefs', False):
            name = re.sub('^/', 'wa/', name)
        return name

    # M: maximum trip-count
    M = tf_layers_dict[graph_node_input_1.name]['tf_node'] \
        if isinstance(graph_node_input_1, gs.Variable) else graph_node_input_1
    M = None if isinstance(M, str) and M == "" else M
    M = _as_tensor(M) if M is not None else None
    if M is not None:
        M = tf.cast(M, tf.int64)
        max_i32 = tf.constant(tf.int32.max, dtype=tf.int64)
        M = tf.where(tf.greater(M, max_i32), max_i32, M)
        M = tf.cast(M, tf.int32)
        if M.shape is not None and M.shape.rank not in (None, 0):
            M = tf.reshape(M, [])

    # cond: loop continuation condition (optional)
    cond = tf_layers_dict[graph_node_input_2.name]['tf_node'] \
        if isinstance(graph_node_input_2, gs.Variable) else graph_node_input_2
    cond = None if isinstance(cond, str) and cond == "" else cond
    cond_init = None if cond is None else tf.cast(_as_tensor(cond), tf.bool)
    if cond_init is not None \
        and isinstance(graph_node_input_2, gs.Variable) \
        and graph_node_input_2.shape is not None \
        and len(graph_node_input_2.shape) == 0:
        cond_init = tf.reshape(cond_init, [])
    cond_provided = cond_init is not None
    if not cond_provided:
        cond_init = tf.constant(True, dtype=tf.bool)

    v_init = []
    v_input_meta = []
    for graph_node_input_n in graph_node_input_n_list:
        v_val = tf_layers_dict[graph_node_input_n.name]['tf_node'] \
            if isinstance(graph_node_input_n, gs.Variable) else graph_node_input_n
        v_val = _as_tensor(v_val)
        if isinstance(graph_node_input_n, gs.Variable) \
            and graph_node_input_n.shape is not None \
            and len(graph_node_input_n.shape) == 0:
            v_val = tf.reshape(v_val, [])
        v_init.append(v_val)
        if isinstance(graph_node_input_n, gs.Variable):
            v_input_meta.append(tf_layers_dict.get(graph_node_input_n.name, {}))
        else:
            v_input_meta.append({})

    v_shapes = [_shape_invariant(v) for v in v_init]

    body: gs.Graph = graph_node.attrs["body"]

    iter_cnt_init = tf.constant(0, dtype=tf.int32)

    scan_outputs_start_index = 1 + len(v_init)
    scan_outputs_init = []
    for i in range(scan_outputs_start_index, len(body.outputs)):
        elem_shape = body.outputs[i].shape
        if elem_shape is not None:
            elem_shape = [
                dim if isinstance(dim, int) else None for dim in elem_shape
            ]
            elem_shape = tf.TensorShape(elem_shape)
        scan_outputs_init.append(
            tf.TensorArray(
                dtype=_to_tf_dtype(body.outputs[i].dtype),
                size=0,
                dynamic_size=True,
                element_shape=elem_shape,
            )
        )
    scan_outputs_shapes = [tf.TensorShape(None) for _ in scan_outputs_init]

    graph_node_outputs = list(graph_node.outputs)
    for graph_node_output in graph_node_outputs:
        tf_layers_dict[graph_node_output.name] = {
            'optype': graph_node.op,
            'shape': graph_node_output.shape,
            'dtype': graph_node_output.dtype,
        }

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

    # Register subgraph constants outside the while_loop to avoid scope issues.
    _register_graph_output_constants(body)

    def run_subgraph(iter_cnt, cond, v, scan_outputs):
        # Bind loop vars to body graph inputs
        loop_inputs = [iter_cnt, cond] + list(v)
        for idx, (body_input, loop_val) in enumerate(zip(body.inputs, loop_inputs)):
            body_input.name = _sanitize(body_input.name)
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
            if idx >= 2:
                meta = v_input_meta[idx - 2]
                for key in ('before_op_output_shape_trans', 'nhwc'):
                    if key in meta:
                        tf_layers_dict[body_input.name][key] = meta[key]

        subgraph_kwargs = dict(kwargs)
        subgraph_kwargs['suppress_log'] = True
        for body_node in body.nodes:
            optype = body_node.op
            try:
                op = importlib.import_module(f'onnx2tf.ops.{optype}')
            except ModuleNotFoundError as ex:
                error(
                    f'{optype} OP is not yet implemented.'
                )
                sys.exit(1)
            body_node.name = _sanitize(body_node.name)
            op.make_node(
                graph_node=body_node,
                tf_layers_dict=tf_layers_dict,
                **subgraph_kwargs,
            )
        outputs = [tf_layers_dict[output.name]['tf_node'] for output in body.outputs]
        for i in range(scan_outputs_start_index, len(outputs)):
            s_index = i - scan_outputs_start_index
            scan_outputs[s_index] = scan_outputs[s_index].write(
                scan_outputs[s_index].size(), outputs[i]
            )
        cond_out = outputs[0]
        if isinstance(cond_out, tf.Tensor) and cond_out.dtype != tf.bool:
            cond_out = tf.cast(cond_out, tf.bool)
        iter_cnt = iter_cnt + 1
        return [iter_cnt, cond_out, outputs[1:scan_outputs_start_index], scan_outputs]

    if M is None and not cond_provided:
        error(
            f'Both M and cond in Loop are not set at the same time ' +
            f'Tensorflow.(PS. if you want to create a do-while loop ' +
            f'then please set cond to True or 1)\n' +
            f'graph_node.name: {graph_node.name}'
        )
        sys.exit(1)

    cond_true = tf.constant(True, dtype=tf.bool)
    if M is not None and not cond_provided:
        condition = lambda iter_cnt, cond, v, scan_outputs: cond_true
        while_loop_layer = While_Loop_CustomLayer()
        iter_cnt_final, _, v_final, scan_outputs_final = while_loop_layer(
            cond=condition,
            body=run_subgraph,
            loop_vars=[
                iter_cnt_init,
                cond_init,
                v_init,
                scan_outputs_init,
            ],
            shape_invariants=[
                tf.TensorShape([]),
                tf.TensorShape(None),
                v_shapes,
                scan_outputs_shapes,
            ],
            maximum_iterations=M,
        )
    elif M is None and cond_provided:
        condition = lambda iter_cnt, cond, v, scan_outputs: tf.reduce_all(cond)
        while_loop_layer = While_Loop_CustomLayer()
        iter_cnt_final, cond_final, v_final, scan_outputs_final = while_loop_layer(
            cond=condition,
            body=run_subgraph,
            loop_vars=[
                iter_cnt_init,
                cond_init,
                v_init,
                scan_outputs_init,
            ],
            shape_invariants=[
                tf.TensorShape([]),
                tf.TensorShape(None),
                v_shapes,
                scan_outputs_shapes,
            ],
        )
    else:
        condition = lambda iter_cnt, cond, v, scan_outputs: tf.reduce_all(cond)
        while_loop_layer = While_Loop_CustomLayer()
        iter_cnt_final, cond_final, v_final, scan_outputs_final = while_loop_layer(
            cond=condition,
            body=run_subgraph,
            loop_vars=[
                iter_cnt_init,
                cond_init,
                v_init,
                scan_outputs_init,
            ],
            shape_invariants=[
                tf.TensorShape([]),
                tf.TensorShape(None),
                v_shapes,
                scan_outputs_shapes,
            ],
            maximum_iterations=M,
        )

    if scan_outputs_start_index == len(body.outputs):
        final_outputs = list(v_final)
    else:
        def true_fn():
            return scan_outputs_final

        def false_fn():
            empty_scan_outputs = []
            for ta in scan_outputs_init:
                empty_scan_outputs.append(
                    tf.TensorArray(
                        dtype=ta.dtype,
                        size=0,
                        element_shape=ta.element_shape,
                    )
                )
            return empty_scan_outputs

        scan_out_final = tf.cond(tf.greater(iter_cnt_final, 0), true_fn, false_fn)
        scan_outputs_tensors = [o.stack() for o in scan_out_final]
        final_outputs = list(v_final) + scan_outputs_tensors

    if len(final_outputs) != len(graph_node_outputs):
        error(
            f'Loop output count mismatch. expected={len(graph_node_outputs)} actual={len(final_outputs)}\n' +
            f'graph_node.name: {graph_node.name}'
        )
        sys.exit(1)

    for idx, (graph_node_output, output_tensor) in enumerate(zip(graph_node_outputs, final_outputs)):
        tf_layers_dict[graph_node_output.name]['tf_node'] = output_tensor
        if idx < len(v_init):
            body_output = body.outputs[1 + idx]
        else:
            body_output = body.outputs[scan_outputs_start_index + (idx - len(v_init))]
        body_meta = tf_layers_dict.get(body_output.name, {})
        for key in ('before_op_output_shape_trans', 'nhwc'):
            if key in body_meta:
                tf_layers_dict[graph_node_output.name][key] = body_meta[key]

    tf_outputs = {f"output{idx}": value for idx, value in enumerate(final_outputs)}
    for graph_node_output in graph_node_outputs:
        tf_layers_dict[graph_node_output.name]['tf_node_info'] = \
            make_tf_node_info(
                node_info={
                    'tf_op_type': tf.while_loop,
                    'tf_inputs': {
                        'condition': condition,
                        'M': M,
                        'cond': cond_init,
                        'v_initial': v_init,
                    },
                    'tf_outputs': {
                        'output': tf_outputs,
                    },
                }
            )
