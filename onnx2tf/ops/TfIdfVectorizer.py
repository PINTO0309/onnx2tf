import random
random.seed(0)
import collections
from enum import IntEnum
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
    pre_process_transpose,
    post_process_transpose,
)


class IntMap(collections.UserDict):
    def __init__(self):
        super().__init__()
        self.added_keys = []

    def emplace(self, key, value):
        if not isinstance(key, (int, str, bytes, np.bytes_)):
            raise TypeError(f"key must be a int or str not {type(key)}.")
        if not isinstance(value, NgramPart):
            raise TypeError(f"value must be a NGramPart not {type(value)}.")
        if key not in self:
            self.added_keys.append(key)
            self.data[key] = value
        return self.data[key]

    @property
    def first_key(self):
        if len(self) == 0:
            raise ValueError("IntMap is empty.")
        return self.added_keys[0]


class NgramPart:
    def __init__(self, nid: int):
        self.id_ = nid  # 0 - means no entry, search for a bigger N
        self._leafs_ = None

    def init(self):
        self._leafs_ = IntMap()

    def empty(self):
        return self._leafs_ is None

    def has_leaves(self):
        return self._leafs_ is not None and len(self._leafs_) > 0

    @property
    def leafs_(self):
        if self._leafs_ is None:
            raise RuntimeError("NgramPart was not initialized.")
        return self._leafs_

    def find(self, key):
        if not self.has_leaves():
            return None
        if key in self._leafs_:
            return key
        return None

    def emplace(self, key, value):
        return self.leafs_.emplace(key, value)

    def __getitem__(self, key):
        return self._leafs_[key]


class WeightingCriteria(IntEnum):
    NONE = 0
    TF = 1
    IDF = 2
    TFIDF = 3


def populate_grams(
    els,
    els_index,
    n_ngrams: int,
    ngram_size: int,
    ngram_id: int,
    c,
):
    for _ngrams in range(n_ngrams, 0, -1):
        n = 1
        m = c
        while els_index < len(els):
            p = m.emplace(els[els_index], NgramPart(0))
            if n == ngram_size:
                p.id_ = ngram_id
                ngram_id += 1
                els_index += 1
                break
            if p.empty():
                p.init()
            m = p.leafs_
            n += 1
            els_index += 1
    return ngram_id


class _TfIdfVectorizerImpl:
    def __init__(
        self,
        *,
        max_gram_length,
        max_skip_count,
        min_gram_length,
        mode,
        ngram_counts,
        ngram_indexes,
        pool_int64s,
        pool_strings,
        weights,
    ):
        if mode == "TF":
            self.weighting_criteria_ = WeightingCriteria.TF
        elif mode == "IDF":
            self.weighting_criteria_ = WeightingCriteria.IDF
        elif mode == "TFIDF":
            self.weighting_criteria_ = WeightingCriteria.TFIDF
        else:
            self.weighting_criteria_ = WeightingCriteria.NONE

        self.min_gram_length_ = int(min_gram_length)
        self.max_gram_length_ = int(max_gram_length)
        self.max_skip_count_ = int(max_skip_count)
        self.ngram_counts_ = list(ngram_counts)
        self.ngram_indexes_ = list(ngram_indexes)
        self.output_size_ = max(self.ngram_indexes_) + 1 if len(self.ngram_indexes_) > 0 else 0
        self.weights_ = list(weights) if weights is not None else []
        self.pool_int64s_ = list(pool_int64s) if pool_int64s is not None else []
        self.pool_strings_ = list(pool_strings) if pool_strings is not None else []

        self.int64_map_ = NgramPart(-10)
        self.int64_map_.init()

        total_items = len(self.pool_int64s_ or self.pool_strings_)
        ngram_id = 1  # start with 1, 0 - means no n-gram
        ngram_size = 1
        for i in range(len(self.ngram_counts_)):
            start_idx = self.ngram_counts_[i]
            end_idx = (
                self.ngram_counts_[i + 1]
                if (i + 1) < len(self.ngram_counts_)
                else total_items
            )
            items = end_idx - start_idx
            if items > 0:
                ngrams = items // ngram_size
                if (
                    ngram_size >= self.min_gram_length_
                    and ngram_size <= self.max_gram_length_
                ):
                    ngram_id = populate_grams(
                        self.pool_int64s_ or self.pool_strings_,
                        start_idx,
                        ngrams,
                        ngram_size,
                        ngram_id,
                        self.int64_map_,
                    )
                else:
                    ngram_id += ngrams
            ngram_size += 1

    def increment_count(self, ngram_id: int, row_num: int, frequencies: np.ndarray) -> None:
        ngram_id -= 1
        output_idx = row_num * self.output_size_ + self.ngram_indexes_[ngram_id]
        frequencies[output_idx] += 1

    def output_result(self, B: int, frequencies: np.ndarray) -> np.ndarray:
        if B == 0:
            output_dims = (self.output_size_,)
            B = 1
        else:
            output_dims = (B, self.output_size_)

        row_size = self.output_size_
        total_dims = int(np.prod(output_dims))
        Y = np.empty((total_dims,), dtype=np.float32)

        w = self.weights_
        if self.weighting_criteria_ == WeightingCriteria.TF:
            for i, f in enumerate(frequencies):
                Y[i] = f
        elif self.weighting_criteria_ == WeightingCriteria.IDF:
            if len(w) > 0:
                p = 0
                for _batch in range(B):
                    for i in range(row_size):
                        Y[p] = w[i] if frequencies[p] > 0 else 0
                        p += 1
            else:
                p = 0
                for f in frequencies:
                    Y[p] = 1 if f > 0 else 0
                    p += 1
        elif self.weighting_criteria_ == WeightingCriteria.TFIDF:
            if len(w) > 0:
                p = 0
                for _batch in range(B):
                    for i in range(row_size):
                        Y[p] = w[i] * frequencies[p]
                        p += 1
            else:
                p = 0
                for f in frequencies:
                    Y[p] = f
                    p += 1
        else:
            raise RuntimeError("Unexpected weighting_criteria.")

        return Y.reshape(output_dims)

    def compute_impl(self, X: np.ndarray, row_num: int, row_size: int, frequencies: np.ndarray) -> None:
        X_flat = X[row_num] if len(X.shape) > 1 else X
        row_begin = 0
        row_end = row_begin + row_size

        max_skip_distance = self.max_skip_count_ + 1
        start_ngram_size = self.min_gram_length_

        for skip_distance in range(1, max_skip_distance + 1):
            ngram_start = row_begin
            ngram_row_end = row_end

            while ngram_start < ngram_row_end:
                at_least_this = ngram_start + skip_distance * (start_ngram_size - 1)
                if at_least_this >= ngram_row_end:
                    break

                ngram_item = ngram_start
                int_map = self.int64_map_
                ngram_size = 1
                while (
                    int_map.has_leaves()
                    and ngram_size <= self.max_gram_length_
                    and ngram_item < ngram_row_end
                ):
                    val = X_flat[ngram_item]
                    hit = int_map.find(val)
                    if hit is None:
                        break
                    hit = int_map[val].id_
                    if ngram_size >= start_ngram_size and hit != 0:
                        self.increment_count(hit, row_num, frequencies)
                    int_map = int_map[val]
                    ngram_size += 1
                    ngram_item += skip_distance

                ngram_start += 1

            if start_ngram_size == 1:
                start_ngram_size += 1
                if start_ngram_size > self.max_gram_length_:
                    break

    def run(self, X: np.ndarray) -> np.ndarray:
        total_items = int(np.prod(X.shape))

        num_rows = 0
        B = 0
        C = 0
        input_dims = X.shape
        if len(input_dims) == 0:
            num_rows = 1
            C = 1
            if total_items != 1:
                raise ValueError(f"Unexpected total of items {total_items}.")
        elif len(input_dims) == 1:
            num_rows = 1
            C = input_dims[0]
        elif len(input_dims) == 2:
            B = input_dims[0]
            C = input_dims[1]
            num_rows = B
            if B < 1:
                raise ValueError(
                    f"Input shape must have either [C] or [B,C] dimensions with B > 0, B={B}, C={C}."
                )
        else:
            raise ValueError(
                f"Input shape must have either [C] or [B,C] dimensions with B > 0, B={B}, C={C}."
            )

        if num_rows * C != total_items:
            raise ValueError(
                f"Unexpected total of items, num_rows * C = {num_rows * C} != total_items = {total_items}."
            )

        frequencies = np.zeros((num_rows * self.output_size_,), dtype=np.int64)

        if total_items == 0 or self.int64_map_.empty():
            return self.output_result(B, frequencies)

        for i in range(num_rows):
            self.compute_impl(X, i, C, frequencies)

        return self.output_result(B, frequencies)


@print_node_info
@inverted_operation_enable_disable
@get_replacement_parameter
def make_node(
    *,
    graph_node: gs.Node,
    tf_layers_dict: dict,
    **kwargs: dict,
):
    """TfIdfVectorizer

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

    # Pre-process transpose
    input_tensor = pre_process_transpose(
        value_before_transpose=input_tensor,
        param_target='inputs',
        param_name=graph_node.inputs[0].name,
        **kwargs,
    )

    max_gram_length = int(graph_node.attrs.get('max_gram_length', 1))
    max_skip_count = int(graph_node.attrs.get('max_skip_count', 0))
    min_gram_length = int(graph_node.attrs.get('min_gram_length', 1))
    mode = graph_node.attrs.get('mode', 'TF')
    ngram_counts = graph_node.attrs.get('ngram_counts', [])
    ngram_indexes = graph_node.attrs.get('ngram_indexes', [])
    pool_int64s = graph_node.attrs.get('pool_int64s', None)
    pool_strings = graph_node.attrs.get('pool_strings', None)
    weights = graph_node.attrs.get('weights', None)

    impl = _TfIdfVectorizerImpl(
        max_gram_length=max_gram_length,
        max_skip_count=max_skip_count,
        min_gram_length=min_gram_length,
        mode=mode,
        ngram_counts=ngram_counts,
        ngram_indexes=ngram_indexes,
        pool_int64s=pool_int64s,
        pool_strings=pool_strings,
        weights=weights,
    )

    def _tfidf_numpy(x):
        return impl.run(x)

    tf_layers_dict[graph_node_output.name]['tf_node'] = \
        tf.numpy_function(
            func=_tfidf_numpy,
            inp=[input_tensor],
            Tout=tf.float32,
            name=graph_node.name,
        )
    if shape is not None:
        tf_layers_dict[graph_node_output.name]['tf_node'] = \
            tf.ensure_shape(
                tf_layers_dict[graph_node_output.name]['tf_node'],
                shape,
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
                'tf_op_type': 'TfIdfVectorizer',
                'tf_inputs': {
                    'max_gram_length': max_gram_length,
                    'max_skip_count': max_skip_count,
                    'min_gram_length': min_gram_length,
                    'mode': mode,
                    'ngram_counts': ngram_counts,
                    'ngram_indexes': ngram_indexes,
                    'pool_int64s': pool_int64s,
                    'pool_strings': pool_strings,
                    'weights': weights,
                },
                'tf_outputs': {
                    'output': tf_layers_dict[graph_node_output.name]['tf_node'],
                },
            }
        )
