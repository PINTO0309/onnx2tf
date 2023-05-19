import random
random.seed(0)
import numpy as np
np.random.seed(0)
import tensorflow as tf
from onnx import TensorProto
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
from onnx2tf.utils.enums import ONNX_DTYPES_TO_TF_DTYPES


@print_node_info
@inverted_operation_enable_disable
@get_replacement_parameter
def make_node(
    *,
    graph_node: gs.Node,
    tf_layers_dict: dict,
    **kwargs: dict,
):
    """MelWeightMatrix

    Parameters
    ----------
    graph_node: gs.Node
        graph_surgeon Node

    tf_layers_dict: dict
        optype, shape, dtype, tensorflow graph
    """
    graph_node_input_1 = get_constant_or_variable(
        graph_node.inputs[0],
        before_op_output_shape_trans=False,
    )
    graph_node_input_2 = get_constant_or_variable(
        graph_node.inputs[1],
        before_op_output_shape_trans=False,
    )
    graph_node_input_3 = get_constant_or_variable(
        graph_node.inputs[2],
        before_op_output_shape_trans=False,
    )
    graph_node_input_4 = get_constant_or_variable(
        graph_node.inputs[3],
        before_op_output_shape_trans=False,
    )
    graph_node_input_5 = get_constant_or_variable(
        graph_node.inputs[4],
        before_op_output_shape_trans=False,
    )
    num_mel_bins = tf_layers_dict[graph_node_input_1.name]['tf_node'] \
        if isinstance(graph_node_input_1, gs.Variable) else graph_node_input_1
    dft_length = tf_layers_dict[graph_node_input_2.name]['tf_node'] \
        if isinstance(graph_node_input_2, gs.Variable) else graph_node_input_2
    sample_rate = tf_layers_dict[graph_node_input_3.name]['tf_node'] \
        if isinstance(graph_node_input_3, gs.Variable) else graph_node_input_3
    lower_edge_hertz = tf_layers_dict[graph_node_input_4.name]['tf_node'] \
        if isinstance(graph_node_input_4, gs.Variable) else graph_node_input_4
    upper_edge_hertz = tf_layers_dict[graph_node_input_5.name]['tf_node'] \
        if isinstance(graph_node_input_5, gs.Variable) else graph_node_input_5

    graph_node_output: gs.Variable = graph_node.outputs[0]

    shape = graph_node_output.shape
    dtype = graph_node_output.dtype

    output_datatype = int(graph_node.attrs.get('output_datatype', TensorProto.FLOAT))
    output_datatype = ONNX_DTYPES_TO_TF_DTYPES[output_datatype]

    # Preserving Graph Structure (Dict)
    tf_layers_dict[graph_node_output.name] = {
        'optype': graph_node.op,
        'shape': shape,
        'dtype': dtype,
    }

    # Pre-process transpose
    num_mel_bins = pre_process_transpose(
        value_before_transpose=num_mel_bins,
        param_target='inputs',
        param_name=graph_node.inputs[0].name,
        **kwargs,
    )
    dft_length = pre_process_transpose(
        value_before_transpose=dft_length,
        param_target='inputs',
        param_name=graph_node.inputs[1].name,
        **kwargs,
    )
    sample_rate = pre_process_transpose(
        value_before_transpose=sample_rate,
        param_target='inputs',
        param_name=graph_node.inputs[2].name,
        **kwargs,
    )
    lower_edge_hertz = pre_process_transpose(
        value_before_transpose=lower_edge_hertz,
        param_target='inputs',
        param_name=graph_node.inputs[3].name,
        **kwargs,
    )
    upper_edge_hertz = pre_process_transpose(
        value_before_transpose=upper_edge_hertz,
        param_target='inputs',
        param_name=graph_node.inputs[4].name,
        **kwargs,
    )

    # Generation of TF OP
    # https://github.com/onnx/onnx/blob/main/onnx/reference/ops/op_mel_weight_matrix.py
    # https://www.tensorflow.org/api_docs/python/tf/signal/linear_to_mel_weight_matrix
    num_spectrogram_bins = tf.math.add(tf.math.floordiv(dft_length, 2), 1)
    tf_layers_dict[graph_node_output.name]['tf_node'] = \
        tf.signal.linear_to_mel_weight_matrix(
            num_mel_bins=num_mel_bins,
            num_spectrogram_bins=num_spectrogram_bins,
            sample_rate=sample_rate,
            lower_edge_hertz=lower_edge_hertz,
            upper_edge_hertz=upper_edge_hertz,
            dtype=output_datatype,
            name=graph_node.name,
        )

    # Post-process transpose
    tf_layers_dict[graph_node_output.name]['tf_node'] = post_process_transpose(
        value_before_transpose=tf_layers_dict[graph_node_output.name]['tf_node'],
        param_target='outputs',
        param_name=graph_node.outputs[0].name,
        **kwargs,
    )

    # Generation of Debug Info
    tf_layers_dict[graph_node_output.name]['tf_node_info'] = \
        make_tf_node_info(
            node_info={
                'tf_op_type': tf.signal.linear_to_mel_weight_matrix,
                'tf_inputs': {
                    'num_mel_bins': num_mel_bins,
                    'dft_length': dft_length,
                    'num_spectrogram_bins': num_spectrogram_bins,
                    'sample_rate': sample_rate,
                    'lower_edge_hertz': lower_edge_hertz,
                    'upper_edge_hertz': upper_edge_hertz,
                    'output_datatype': output_datatype,
                },
                'tf_outputs': {
                    'output': tf_layers_dict[graph_node_output.name]['tf_node'],
                },
            }
        )
