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
        self.stopwords = set(stopwords)

    def call(self, inputs):
        if not self.is_case_sensitive:
            # if self.locale:
            #     inputs = text.case_fold_utf8(inputs)
            # else:
            #     inputs = tf.strings.lower(inputs)
            inputs = tf.strings.lower(inputs)
        elif self.case_change_action == "LOWER":
            inputs = tf.strings.lower(inputs)
        elif self.case_change_action == "UPPER":
            inputs = tf.strings.upper(inputs)

        # if self.tokenizer:
        #     tokenized = self.tokenizer.tokenize(inputs)
        # else:
        #     tokenized = tf.strings.split(inputs)
        tokenized = tf.strings.split(inputs)

        if not self.is_case_sensitive:
            # stopwords = [
            #     tf.strings.lower(word) \
            #         if self.locale is None else text.case_fold_utf8(word) \
            #             for word in self.stopwords
            # ]
            stopwords = [
                tf.strings.lower(word) for word in self.stopwords
            ]
        else:
            stopwords = self.stopwords

        return \
            tf.ragged.boolean_mask(
                tokenized,
                ~tf.reduce_any(
                    tf.equal(
                        tf.expand_dims(tokenized, axis=-1),
                        stopwords,
                    ),
                    axis=-1,
                )
            )


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
