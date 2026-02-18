import numpy as np

from onnx2tf.tflite_builder.accuracy_evaluator import _adapt_input_layout_for_tflite_input


def _detail(shape_signature):
    return {
        "shape_signature": np.asarray(shape_signature, dtype=np.int32),
    }


def test_adapt_input_layout_rank3_ncw_to_nwc() -> None:
    x = np.arange(1 * 17 * 2, dtype=np.float32).reshape(1, 17, 2)
    adapted = _adapt_input_layout_for_tflite_input(x, _detail([1, 2, 17]))
    np.testing.assert_array_equal(adapted, np.transpose(x, (0, 2, 1)))


def test_adapt_input_layout_rank4_nchw_to_nhwc() -> None:
    x = np.arange(1 * 3 * 4 * 5, dtype=np.float32).reshape(1, 3, 4, 5)
    adapted = _adapt_input_layout_for_tflite_input(x, _detail([1, 4, 5, 3]))
    np.testing.assert_array_equal(adapted, np.transpose(x, (0, 2, 3, 1)))


def test_adapt_input_layout_rank5_ncdhw_to_ndhwc() -> None:
    x = np.arange(1 * 2 * 3 * 4 * 5, dtype=np.float32).reshape(1, 2, 3, 4, 5)
    adapted = _adapt_input_layout_for_tflite_input(x, _detail([1, 3, 4, 5, 2]))
    np.testing.assert_array_equal(adapted, np.transpose(x, (0, 2, 3, 4, 1)))


def test_adapt_input_layout_noop_when_shape_matches() -> None:
    x = np.arange(1 * 2 * 17, dtype=np.float32).reshape(1, 2, 17)
    adapted = _adapt_input_layout_for_tflite_input(x, _detail([1, 2, 17]))
    np.testing.assert_array_equal(adapted, x)
