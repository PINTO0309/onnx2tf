import random
random.seed(0)
import numpy as np
np.random.seed(0)
import tensorflow as tf
from onnx import numpy_helper
import onnx_graphsurgeon as gs
from onnx2tf.utils.enums import NUMPY_DTYPES_TO_TF_DTYPES
from onnx2tf.utils.common_functions import (
    print_node_info,
    make_tf_node_info,
    transpose_with_flexing_deterrence,
)


@print_node_info
def _make_tf_constant(
    *,
    value,
    dtype,
    name,
    kwargs,
):
    constant_tensor = tf.constant(
        value=value,
        dtype=dtype,
    )
    # NCW->NWC, NCHW->NHWC, NCDHW->NDHWC
    transposed_tensor = None
    if len(value.shape) == 3:
        transposed_tensor = \
            transpose_with_flexing_deterrence(
                input_tensor=constant_tensor,
                perm=[0,2,1],
                name=name,
                **kwargs,
            )
    elif len(value.shape) == 4:
        transposed_tensor = \
            transpose_with_flexing_deterrence(
                input_tensor=constant_tensor,
                perm=[0,2,3,1],
                name=name,
                **kwargs,
            )
    elif len(value.shape) == 5:
        transposed_tensor = \
            transpose_with_flexing_deterrence(
                input_tensor=constant_tensor,
                perm=[0,2,3,4,1],
                name=name,
                **kwargs,
            )
    else:
        transposed_tensor = constant_tensor
    return transposed_tensor


def _make_tf_sparsetensor(
    *,
    indices,
    values,
    dense_shape,
):
    sparse_tensor = tf.SparseTensor(
        indices=indices,
        values=values,
        dense_shape=dense_shape,
    )
    # NCW->NWC, NCHW->NHWC, NCDHW->NDHWC
    transposed_tensor = None
    if len(sparse_tensor.shape) == 3:
        transposed_tensor = \
            tf.transpose(
                a=sparse_tensor,
                perm=[0,2,1],
            )
    elif len(sparse_tensor.shape) == 4:
        transposed_tensor = \
            tf.transpose(
                a=sparse_tensor,
                perm=[0,2,3,1],
            )
    elif len(sparse_tensor.shape) == 5:
        transposed_tensor = \
            tf.transpose(
                a=sparse_tensor,
                perm=[0,2,3,4,1],
            )
    else:
        transposed_tensor = sparse_tensor
    return transposed_tensor


def make_node(
    *,
    graph_node: gs.Node,
    tf_layers_dict: dict,
    **kwargs: dict,
):
    """Constant

    Parameters
    ----------
    graph_node: gs.Node
        graph_surgeon Node

    tf_layers_dict: dict
        optype, shape, dtype, tensorflow graph
    """
    graph_node_output: gs.Variable = graph_node.outputs[0]

    shape = graph_node_output.shape
    dtype = graph_node_output.dtype

    opset = kwargs['opset']

    # Preserving Graph Structure (Dict)
    tf_layers_dict[graph_node_output.name] = {
        'optype': graph_node.op,
        'shape': shape,
        'dtype': dtype,
    }

    # Generation of TF OP
    # https://github.com/onnx/onnx-tensorflow/blob/main/onnx_tf/handlers/backend/constant.py

    sparse_value = None
    indices = None
    value = None
    shape = None
    const_dtype = None

    if opset <= 11:
        # either value or sparse_value
        if "value" in graph_node.attrs:
            attr_value = graph_node.attrs["value"]
            value = attr_value.values \
                if isinstance(attr_value, gs.Constant) and hasattr(attr_value, 'values') else numpy_helper.to_array(attr_value)
            const_dtype = NUMPY_DTYPES_TO_TF_DTYPES[attr_value.dtype] \
                if attr_value.dtype in NUMPY_DTYPES_TO_TF_DTYPES else attr_value.dtype
            tf_layers_dict[graph_node_output.name]['tf_node'] = \
                _make_tf_constant(
                    value=value,
                    dtype=const_dtype,
                    name=graph_node.name,
                    kwargs=kwargs,
                )
            # Generation of Debug Info
            tf_layers_dict[graph_node_output.name]['tf_node_info'] = \
                make_tf_node_info(
                    node_info={
                        'tf_op_type': tf.constant,
                        'tf_inputs': {
                            'sparse_value': sparse_value,
                            'indices': indices,
                            'value': value,
                            'shape': shape,
                            'const_dtype': const_dtype,
                        },
                        'tf_outputs': {
                            'output': tf_layers_dict[graph_node_output.name]['tf_node'],
                        },
                    }
                )
            return
        else:
            sparse_value = graph_node.attrs["sparse_value"]
            indices = numpy_helper.to_array(sparse_value.indices)
            values = numpy_helper.to_array(sparse_value.values)
            shape = np.array(sparse_value.dims)
            tf_layers_dict[graph_node_output.name]['tf_node'] = \
                _make_tf_sparsetensor(
                    indices=indices,
                    values=values,
                    dense_shape=shape,
                )
            # Generation of Debug Info
            tf_layers_dict[graph_node_output.name]['tf_node_info'] = \
                make_tf_node_info(
                    node_info={
                        'tf_op_type': tf.constant,
                        'tf_inputs': {
                            'sparse_value': sparse_value,
                            'indices': indices,
                            'value': value,
                            'shape': shape,
                            'const_dtype': const_dtype,
                        },
                        'tf_outputs': {
                            'output': tf_layers_dict[graph_node_output.name]['tf_node'],
                        },
                    }
                )
            return
    elif opset >= 12:
        if "value" in graph_node.attrs or "sparse_value" in graph_node.attrs:
            # either value or sparse_value
            if "value" in graph_node.attrs:
                attr_value = graph_node.attrs["value"]
                value = attr_value.values \
                    if isinstance(attr_value, gs.Constant) and hasattr(attr_value, 'values') else numpy_helper.to_array(attr_value)
                const_dtype = NUMPY_DTYPES_TO_TF_DTYPES[attr_value.dtype] \
                    if attr_value.dtype in NUMPY_DTYPES_TO_TF_DTYPES else attr_value.dtype
                tf_layers_dict[graph_node_output.name]['tf_node'] = \
                    _make_tf_constant(
                        value=value,
                        dtype=const_dtype,
                        name=graph_node.name,
                        kwargs=kwargs,
                    )
                # Generation of Debug Info
                tf_layers_dict[graph_node_output.name]['tf_node_info'] = \
                    make_tf_node_info(
                        node_info={
                            'tf_op_type': tf.constant,
                            'tf_inputs': {
                                'sparse_value': sparse_value,
                                'indices': indices,
                                'value': value,
                                'shape': shape,
                                'const_dtype': const_dtype,
                            },
                            'tf_outputs': {
                                'output': tf_layers_dict[graph_node_output.name]['tf_node'],
                            },
                        }
                    )
                return
            else:
                sparse_value = graph_node.attrs["sparse_value"]
                indices = numpy_helper.to_array(sparse_value.indices)
                values = numpy_helper.to_array(sparse_value.values)
                shape = np.array(sparse_value.dims)
                tf_layers_dict[graph_node_output.name]['tf_node'] = \
                    _make_tf_sparsetensor(
                        indices=indices,
                        values=values,
                        dense_shape=shape,
                    )
                # Generation of Debug Info
                tf_layers_dict[graph_node_output.name]['tf_node_info'] = \
                    make_tf_node_info(
                        node_info={
                            'tf_op_type': tf.constant,
                            'tf_inputs': {
                                'sparse_value': sparse_value,
                                'indices': indices,
                                'value': value,
                                'shape': shape,
                                'const_dtype': const_dtype,
                            },
                            'tf_outputs': {
                                'output': tf_layers_dict[graph_node_output.name]['tf_node'],
                            },
                        }
                    )
                return
        elif "value_float" in graph_node.attrs:
            value = graph_node.attrs["value_float"]
            const_dtype = tf.float32
        elif "value_floats" in graph_node.attrs:
            value = graph_node.attrs["value_floats"]
            const_dtype = tf.float32
        elif "value_int" in graph_node.attrs:
            value = graph_node.attrs["value_int"]
            const_dtype = tf.int64
        elif "value_ints" in graph_node.attrs:
            value = graph_node.attrs["value_ints"]
            const_dtype = tf.int64
        elif "value_string" in graph_node.attrs:
            value = graph_node.attrs["value_string"]
            const_dtype = tf.string
        elif "value_strings" in graph_node.attrs:
            value = graph_node.attrs["value_strings"]
            const_dtype = tf.string
        tf_layers_dict[graph_node_output.name]['tf_node'] = \
            _make_tf_constant(
                value=value,
                dtype=const_dtype,
                name=graph_node.name,
                kwargs=kwargs,
            )
        # Generation of Debug Info
        tf_layers_dict[graph_node_output.name]['tf_node_info'] = \
            make_tf_node_info(
                node_info={
                    'tf_op_type': tf.constant,
                    'tf_inputs': {
                        'sparse_value': sparse_value,
                        'indices': indices,
                        'value': value,
                        'shape': shape,
                        'const_dtype': const_dtype,
                    },
                    'tf_outputs': {
                        'output': tf_layers_dict[graph_node_output.name]['tf_node'],
                    },
                }
            )