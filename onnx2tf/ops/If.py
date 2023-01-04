import re
import sys
import random
random.seed(0)
import numpy as np
np.random.seed(0)
import importlib
import tensorflow as tf
from tensorflow.python.keras.backend import switch
import onnx_graphsurgeon as gs
from onnx2tf.utils.common_functions import (
    get_constant_or_variable,
    print_node_info,
    inverted_operation_enable_disable,
    make_tf_node_info,
    get_replacement_parameter,
    pre_process_transpose,
    post_process_transpose,
)
from onnx2tf.utils.enums import NUMPY_DTYPES_TO_TF_DTYPES
from onnx2tf.utils.colors import Color


@print_node_info
@inverted_operation_enable_disable
@get_replacement_parameter
def make_node(
    *,
    graph_node: gs.Node,
    tf_layers_dict: dict,
    **kwargs: dict,
):
    """If

    Parameters
    ----------
    graph_node: gs.Node
        graph_surgeon Node

    tf_layers_dict: dict
        optype, shape, dtype, tensorflow graph
    """
    before_op_output_shape_trans_1 = \
        tf_layers_dict.get(graph_node.inputs[0].name, {}).get('before_op_output_shape_trans', True)
    before_op_output_shape_trans = \
        before_op_output_shape_trans_1

    graph_node_input = get_constant_or_variable(
        graph_node.inputs[0],
        before_op_output_shape_trans,
    )
    input_tensor = tf_layers_dict[graph_node_input.name]['tf_node'] \
        if isinstance(graph_node_input, gs.Variable) else graph_node_input
    graph_node_outputs = [] + graph_node.outputs

    # Then branch
    then_branch_graph: gs.Graph = graph_node.attrs['then_branch']
    then_branch_graph_outputs = then_branch_graph.outputs
    for then_branch_graph_node in then_branch_graph.nodes:
        optype = then_branch_graph_node.op
        try:
            op = importlib.import_module(f'onnx2tf.ops.{optype}')
        except ModuleNotFoundError as ex:
            print(
                f'{Color.RED}ERROR:{Color.RESET} {optype} OP is not yet implemented.'
            )
            sys.exit(1)
        # substitution because saved_model does not allow colons
        then_branch_graph_node.name = then_branch_graph_node.name.replace(':','_')
        # Substitution because saved_model does not allow leading slashes in op names
        if kwargs['output_signaturedefs']:
            then_branch_graph_node.name = re.sub('^/', 'wa/', then_branch_graph_node.name)
        op.make_node(
            graph_node=then_branch_graph_node,
            tf_layers_dict=tf_layers_dict,
            **kwargs,
        )
    # Then branch - Resister constant
    for output in then_branch_graph_outputs:
        if output.name not in tf_layers_dict and isinstance(output, gs.Constant):
            tf_layers_dict[output.name] = {
                'optype': 'Constant',
                'shape': output.values.shape,
                'dtype': output.values.dtype,
            }
            tf_layers_dict[output.name]['tf_node'] = \
                tf.constant(
                    output.values,
                    dtype=NUMPY_DTYPES_TO_TF_DTYPES[output.values.dtype],
                )
    then_branch_ops = []
    for then_branch_graph_output in then_branch_graph_outputs:
        then_branch_ops.append(
            tf_layers_dict[then_branch_graph_output.name]['tf_node']
        )

    # Else branch
    else_branch_graph: gs.Graph = graph_node.attrs['else_branch']
    else_branch_graph_outputs = else_branch_graph.outputs
    for else_branch_graph_node in else_branch_graph.nodes:
        optype = else_branch_graph_node.op
        try:
            op = importlib.import_module(f'onnx2tf.ops.{optype}')
        except ModuleNotFoundError as ex:
            print(
                f'{Color.RED}ERROR:{Color.RESET} {optype} OP is not yet implemented.'
            )
            sys.exit(1)
        # substitution because saved_model does not allow colons
        else_branch_graph_node.name = else_branch_graph_node.name.replace(':','_')
        # Substitution because saved_model does not allow leading slashes in op names
        if kwargs['output_signaturedefs']:
            else_branch_graph_node.name = re.sub('^/', 'wa/', else_branch_graph_node.name)
        op.make_node(
            graph_node=else_branch_graph_node,
            tf_layers_dict=tf_layers_dict,
            **kwargs,
        )
    # Else branch - Resister constant
    for output in else_branch_graph_outputs:
        if output.name not in tf_layers_dict and isinstance(output, gs.Constant):
            tf_layers_dict[output.name] = {
                'optype': 'Constant',
                'shape': output.values.shape,
                'dtype': output.values.dtype,
            }
            tf_layers_dict[output.name]['tf_node'] = \
                tf.constant(
                    output.values,
                    dtype=NUMPY_DTYPES_TO_TF_DTYPES[output.values.dtype],
                )
    else_branch_ops = []
    for else_branch_graph_output in else_branch_graph_outputs:
        else_branch_ops.append(
            tf_layers_dict[else_branch_graph_output.name]['tf_node']
        )

    # Pre-process transpose
    input_tensor = pre_process_transpose(
        value_before_transpose=input_tensor,
        param_target='inputs',
        param_name=graph_node.inputs[0].name,
        **kwargs,
    )

    # Generation of TF OP
    for graph_node_output in graph_node_outputs:
        tf_layers_dict[graph_node_output.name] = {
            'optype': graph_node.op,
            'shape': graph_node_output.shape,
            'dtype': graph_node_output.dtype,
        }

    if_cond_outputs = [] + switch(
        condition=input_tensor,
        then_expression=then_branch_ops,
        else_expression=else_branch_ops,
    )

    for graph_node_output, if_cond_output in zip(graph_node_outputs, if_cond_outputs):
        tf_layers_dict[graph_node_output.name]['tf_node'] = if_cond_output
        # Post-process transpose
        tf_layers_dict[graph_node_output.name]['tf_node'] = post_process_transpose(
            value_before_transpose=tf_layers_dict[graph_node_output.name]['tf_node'],
            param_target='outputs',
            param_name=graph_node_output.name,
            **kwargs,
        )

    # Generation of Debug Info
    tf_outputs = {f"output{idx}": value for idx, value in enumerate(if_cond_outputs)}
    tf_layers_dict[graph_node_output.name]['tf_node_info'] = \
        make_tf_node_info(
            node_info={
                'tf_op_type': tf.cond,
                'tf_inputs': {
                    'input': input_tensor,
                    'true_fn': then_branch_ops,
                    'false_fn': else_branch_ops,
                },
                'tf_outputs': {
                    'output': tf_outputs,
                },
            }
        )
