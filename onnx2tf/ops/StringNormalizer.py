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
    get_replacement_parameter,
    pre_process_transpose,
    post_process_transpose,
)
from onnx2tf.utils.logging import *


class StringNormalizer(tf_keras.layers.Layer):
    def __init__(
        self,
        case_change_action=None,
        is_case_sensitive=True,
        locale=None,
        stopwords=[],
        **kwargs
    ):
        super().__init__(**kwargs)
        # self.tokenizer = text.WhitespaceTokenizer()
        self.case_change_action = case_change_action
        self.is_case_sensitive = is_case_sensitive
        self.locale = locale
        self.stopwords = list(stopwords) if stopwords is not None else []

    def _apply_case_action(self, inputs):
        if self.case_change_action == "LOWER":
            return tf.strings.lower(inputs)
        if self.case_change_action == "UPPER":
            return tf.strings.upper(inputs)
        return inputs

    def _stopword_mask(self, inputs):
        if len(self.stopwords) == 0:
            return tf.ones_like(inputs, dtype=tf.bool)
        stopwords = tf.constant(self.stopwords, dtype=tf.string)
        compare_inputs = inputs
        compare_stopwords = stopwords
        if not self.is_case_sensitive:
            compare_inputs = tf.strings.lower(inputs)
            compare_stopwords = tf.strings.lower(stopwords)
        matches = tf.reduce_any(
            tf.equal(
                tf.expand_dims(compare_inputs, axis=-1),
                compare_stopwords,
            ),
            axis=-1,
        )
        return tf.logical_not(matches)

    def call(self, inputs):
        def process_1d():
            mask = self._stopword_mask(inputs)
            filtered = tf.boolean_mask(inputs, mask)
            filtered = self._apply_case_action(filtered)
            return tf.cond(
                tf.equal(tf.size(filtered), 0),
                lambda: tf.constant([""], dtype=tf.string),
                lambda: filtered,
            )

        def process_2d():
            row = inputs[0]
            mask = self._stopword_mask(row)
            filtered = tf.boolean_mask(row, mask)
            filtered = self._apply_case_action(filtered)
            filtered = tf.expand_dims(filtered, axis=0)
            return tf.cond(
                tf.equal(tf.size(filtered), 0),
                lambda: tf.constant([[""]], dtype=tf.string),
                lambda: filtered,
            )

        input_rank = inputs.shape.rank
        if input_rank is None:
            input_rank = tf.rank(inputs)
        if isinstance(input_rank, int):
            if input_rank == 1:
                return process_1d()
            return process_2d()
        return tf.cond(tf.equal(input_rank, 1), process_1d, process_2d)


@print_node_info
@inverted_operation_enable_disable
@get_replacement_parameter
def make_node(
    *,
    graph_node: gs.Node,
    tf_layers_dict: dict,
    **kwargs: dict,
):
    """StringNormalizer

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
    input_tensor = tf_layers_dict[graph_node_input_1.name]['tf_node'] \
        if isinstance(graph_node_input_1, gs.Variable) else graph_node_input_1

    graph_node_output: gs.Variable = graph_node.outputs[0]

    shape = graph_node_output.shape
    dtype = graph_node_output.dtype

    case_change_action = graph_node.attrs.get('case_change_action', 'NONE')
    is_case_sensitive = bool(graph_node.attrs.get('is_case_sensitive', 0))
    locale = graph_node.attrs.get('locale', 'en_US')
    if locale != 'en_US':
        error_msg = f'' + \
                    Color.RED(f'WARNING:') + ' ' + \
                    f'locale option in StringNormalizer ops is not implemented yet.'
        print(error_msg)
    stopwords = graph_node.attrs.get('stopwords', [])
    if stopwords == []:
        stopwords = np.asarray([], dtype=np.string_)

    # Preserving Graph Structure (Dict)
    tf_layers_dict[graph_node_output.name] = {
        'optype': graph_node.op,
        'shape': shape,
        'dtype': dtype,
    }

    # Pre-process transpose
    input_tensor = pre_process_transpose(
        value_before_transpose=input_tensor,
        param_target='inputs',
        param_name=graph_node.inputs[0].name,
        **kwargs,
    )

    # Generation of TF OP
    tf_layers_dict[graph_node_output.name]['tf_node'] = \
        StringNormalizer(
            case_change_action=case_change_action,
            is_case_sensitive=is_case_sensitive,
            locale=locale,
            stopwords=stopwords,
        )(input_tensor)

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
                'tf_op_type': 'StringNormalizer',
                'tf_inputs': {
                    'case_change_action': case_change_action,
                    'is_case_sensitive': is_case_sensitive,
                    'locale': locale,
                    'stopwords': stopwords,
                },
                'tf_outputs': {
                    'output': tf_layers_dict[graph_node_output.name]['tf_node'],
                },
            }
        )
