import numpy as np

from onnx2tf.tflite_builder.accuracy_evaluator import _generate_seeded_input


def test_generate_seeded_input_float_image_shape_defaults_uniform_0_1(
    monkeypatch,
) -> None:
    monkeypatch.delenv("ONNX2TF_EVAL_FLOAT_RANDOM_DISTRIBUTION", raising=False)
    x = _generate_seeded_input(
        shape=(1, 3, 8, 8),
        np_dtype=np.dtype(np.float32),
        rng=np.random.default_rng(0),
    )
    assert x.dtype == np.float32
    assert float(np.min(x)) >= 0.0
    assert float(np.max(x)) <= 1.0


def test_generate_seeded_input_float_non_image_shape_defaults_normal(
    monkeypatch,
) -> None:
    monkeypatch.delenv("ONNX2TF_EVAL_FLOAT_RANDOM_DISTRIBUTION", raising=False)
    rng_actual = np.random.default_rng(123)
    rng_expected = np.random.default_rng(123)
    actual = _generate_seeded_input(
        shape=(1, 16),
        np_dtype=np.dtype(np.float32),
        rng=rng_actual,
    )
    expected = rng_expected.standard_normal((1, 16)).astype(np.float32)
    np.testing.assert_array_equal(actual, expected)


def test_generate_seeded_input_float_env_override_uniform_m1_p1(
    monkeypatch,
) -> None:
    monkeypatch.setenv("ONNX2TF_EVAL_FLOAT_RANDOM_DISTRIBUTION", "uniform_-1_1")
    x = _generate_seeded_input(
        shape=(1, 16),
        np_dtype=np.dtype(np.float32),
        rng=np.random.default_rng(0),
    )
    assert x.dtype == np.float32
    assert float(np.min(x)) >= -1.0
    assert float(np.max(x)) <= 1.0

