from typing import Any, cast
import random
random.seed(0)
import numpy as np
np.random.seed(0)
import tensorflow as tf
import tf_keras
import onnx2tf.gs as gs
from onnx2tf.utils.common_functions import (
    get_constant_or_variable,
    print_node_info,
    inverted_operation_enable_disable,
    make_tf_node_info,
    get_replacement_parameter,
)
from onnx2tf.utils.enums import ONNX_DTYPES_TO_TF_DTYPES

_OPTIONAL_PROXY_HAS_VALUE_KEY = "__onnx_optional_has_value__"
_OPTIONAL_PROXY_VALUE_KEY = "__onnx_optional_value__"


def _type_proto_to_spec(type_proto):
    if type_proto is None:
        return None

    if hasattr(type_proto, 'optional_type') and type_proto.HasField('optional_type'):
        return _type_proto_to_spec(type_proto.optional_type.elem_type)

    if hasattr(type_proto, 'tensor_type') and type_proto.HasField('tensor_type'):
        elem_type = type_proto.tensor_type.elem_type
        tf_dtype = ONNX_DTYPES_TO_TF_DTYPES.get(elem_type, tf.float32)
        dims = []
        for dim in type_proto.tensor_type.shape.dim:
            if dim.HasField('dim_value'):
                dims.append(dim.dim_value)
            else:
                dims.append(None)
        tensor_spec = cast(Any, tf.TensorSpec)
        return tensor_spec(shape=dims, dtype=tf_dtype)

    if hasattr(type_proto, 'sequence_type') and type_proto.HasField('sequence_type'):
        elem_spec = _type_proto_to_spec(type_proto.sequence_type.elem_type)
        if isinstance(elem_spec, tf.TensorSpec):
            elem_shape = list(elem_spec.shape)
            ragged_tensor_spec = cast(Any, tf.RaggedTensorSpec)
            return ragged_tensor_spec(
                shape=[None] + elem_shape,
                dtype=elem_spec.dtype,
                ragged_rank=1,
            )
    return None


@print_node_info
@inverted_operation_enable_disable
@get_replacement_parameter
def make_node(
    *,
    graph_node: gs.Node,
    tf_layers_dict: dict,
    **kwargs: Any,
):
    """Optional

    Parameters
    ----------
    graph_node: gs.Node
        graph_surgeon Node

    tf_layers_dict: dict
        optype, shape, dtype, tensorflow graph
    """
    graph_node_input = None
    input_tensor = None
    if len(graph_node.inputs) >= 1 and graph_node.inputs[0].name != '':
        graph_node_input = get_constant_or_variable(
            graph_node.inputs[0],
            before_op_output_shape_trans=False,
        )
        input_tensor = tf_layers_dict[graph_node_input.name]['tf_node'] \
            if isinstance(graph_node_input, gs.Variable) else graph_node_input

    graph_node_output: gs.Variable = graph_node.outputs[0]
    shape = graph_node_output.shape
    dtype = graph_node_output.dtype

    # Preserving Graph Structure (Dict)
    tf_layers_dict[graph_node_output.name] = {
        'optype': graph_node.op,
        'shape': shape,
        'dtype': dtype,
        'nhwc': tf_layers_dict[graph_node_input.name]['nhwc'] \
            if isinstance(graph_node_input, gs.Variable) \
                and 'nhwc' in tf_layers_dict[graph_node_input.name].keys() else False
    }

    # Generation of TF OP
    # NOTE:
    # tf.experimental.Optional.from_value does not accept KerasTensor on recent TF/Keras,
    # so Optional values are represented with a lightweight proxy dict unless the
    # upstream node already produced tf.experimental.Optional.
    if input_tensor is None:
        tf_layers_dict[graph_node_output.name]['tf_node'] = {
            _OPTIONAL_PROXY_HAS_VALUE_KEY: False,
            _OPTIONAL_PROXY_VALUE_KEY: None,
        }
    elif isinstance(input_tensor, tf.experimental.Optional):
        tf_layers_dict[graph_node_output.name]['tf_node'] = input_tensor
    elif isinstance(input_tensor, dict) and _OPTIONAL_PROXY_HAS_VALUE_KEY in input_tensor:
        tf_layers_dict[graph_node_output.name]['tf_node'] = input_tensor
    else:
        value = input_tensor
        if isinstance(input_tensor, np.ndarray):
            value = tf.convert_to_tensor(input_tensor)
        tf_layers_dict[graph_node_output.name]['tf_node'] = {
            _OPTIONAL_PROXY_HAS_VALUE_KEY: True,
            _OPTIONAL_PROXY_VALUE_KEY: value,
        }

    # Generation of Debug Info
    tf_layers_dict[graph_node_output.name]['tf_node_info'] = \
        make_tf_node_info(
            node_info={
                'tf_op_type': tf.experimental.Optional,
                'tf_inputs': {
                    'input': input_tensor,
                },
                'tf_outputs': {
                    'output': tf_layers_dict[graph_node_output.name]['tf_node'],
                },
            }
        )
