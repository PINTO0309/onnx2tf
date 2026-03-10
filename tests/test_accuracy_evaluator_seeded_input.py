import numpy as np
from onnx import TensorProto, helper

from onnx2tf.tflite_builder.accuracy_evaluator import (
    _build_eval_inputs_for_sample,
    _collect_onnx_input_specs,
    _generate_seeded_input,
    _judge_metrics,
    _max_abs_error,
    _MetricAccumulator,
)


def test_generate_seeded_input_float_nchw_image_shape_defaults_normal(
    monkeypatch,
) -> None:
    monkeypatch.delenv("ONNX2TF_EVAL_FLOAT_RANDOM_DISTRIBUTION", raising=False)
    rng_actual = np.random.default_rng(0)
    rng_expected = np.random.default_rng(0)
    actual = _generate_seeded_input(
        shape=(1, 3, 8, 8),
        np_dtype=np.dtype(np.float32),
        rng=rng_actual,
    )
    expected = rng_expected.standard_normal((1, 3, 8, 8)).astype(np.float32)
    np.testing.assert_array_equal(actual, expected)


def test_generate_seeded_input_float_nhwc_image_shape_defaults_uniform_0_1(
    monkeypatch,
) -> None:
    monkeypatch.delenv("ONNX2TF_EVAL_FLOAT_RANDOM_DISTRIBUTION", raising=False)
    x = _generate_seeded_input(
        shape=(1, 8, 8, 3),
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


def test_build_eval_inputs_for_sample_fills_length_like_integer_input_from_peer_shape() -> None:
    inputs = _build_eval_inputs_for_sample(
        input_specs=[
            ("x", np.dtype(np.float32), (1, 7, 80)),
            ("x_lens", np.dtype(np.int64), (1,)),
        ],
        custom_inputs={},
        sample_index=0,
        rng=np.random.default_rng(0),
    )
    np.testing.assert_array_equal(inputs["x_lens"], np.asarray([7], dtype=np.int64))


def test_build_eval_inputs_for_sample_fills_length_from_audio_feature_last_axis() -> None:
    inputs = _build_eval_inputs_for_sample(
        input_specs=[
            ("audio_signal", np.dtype(np.float32), (1, 80, 16)),
            ("length", np.dtype(np.int64), (1,)),
        ],
        custom_inputs={},
        sample_index=0,
        rng=np.random.default_rng(0),
    )
    np.testing.assert_array_equal(inputs["length"], np.asarray([16], dtype=np.int64))


def test_build_eval_inputs_for_sample_fills_mask_like_input_with_ones() -> None:
    inputs = _build_eval_inputs_for_sample(
        input_specs=[
            ("pixel_values", np.dtype(np.float32), (1, 2, 3, 4)),
            ("pixel_attention_mask", np.dtype(np.bool_), (1, 2, 3)),
        ],
        custom_inputs={},
        sample_index=0,
        rng=np.random.default_rng(0),
    )
    np.testing.assert_array_equal(
        inputs["pixel_attention_mask"],
        np.ones((1, 2, 3), dtype=np.bool_),
    )


def test_collect_onnx_input_specs_uses_unit_time_axis_for_rank5_image_sequence() -> None:
    pixel_values = helper.make_tensor_value_info(
        "pixel_values",
        TensorProto.FLOAT,
        ["batch_size", "num_images", 3, 512, 512],
    )
    pixel_attention_mask = helper.make_tensor_value_info(
        "pixel_attention_mask",
        TensorProto.BOOL,
        ["batch_size", "num_images", 512, 512],
    )
    graph = helper.make_graph(
        [],
        "vision_like_graph",
        [pixel_values, pixel_attention_mask],
        [],
    )
    model = helper.make_model(graph)
    specs = dict((name, shape) for name, _dtype, shape in _collect_onnx_input_specs(model))
    assert specs["pixel_values"] == (1, 1, 3, 512, 512)
    assert specs["pixel_attention_mask"] == (1, 1, 512, 512)


def test_collect_onnx_input_specs_uses_longer_sequence_axis_for_rank3_feature_tensor() -> None:
    x = helper.make_tensor_value_info(
        "x",
        TensorProto.FLOAT,
        ["batch_size", "time", 80],
    )
    graph = helper.make_graph([], "seq_graph", [x], [])
    model = helper.make_model(graph)
    specs = dict((name, shape) for name, _dtype, shape in _collect_onnx_input_specs(model))
    assert specs["x"] == (1, 64, 80)


def test_collect_onnx_input_specs_uses_longer_time_axis_for_channel_major_audio_tensor() -> None:
    audio_signal = helper.make_tensor_value_info(
        "audio_signal",
        TensorProto.FLOAT,
        ["batch_size", 80, "time"],
    )
    graph = helper.make_graph([], "audio_graph", [audio_signal], [])
    model = helper.make_model(graph)
    specs = dict((name, shape) for name, _dtype, shape in _collect_onnx_input_specs(model))
    assert specs["audio_signal"] == (1, 80, 64)


def test_collect_onnx_input_specs_uses_unit_sequence_axis_when_peer_length_input_exists() -> None:
    x = helper.make_tensor_value_info(
        "x",
        TensorProto.FLOAT,
        ["batch_size", 80, "time"],
    )
    x_lens = helper.make_tensor_value_info(
        "x_lens",
        TensorProto.INT64,
        ["batch_size"],
    )
    graph = helper.make_graph([], "encoder_graph", [x, x_lens], [])
    model = helper.make_model(graph)
    specs = dict((name, shape) for name, _dtype, shape in _collect_onnx_input_specs(model))
    assert specs["x"] == (1, 80, 1)
    assert specs["x_lens"] == (1,)


def test_collect_onnx_input_specs_uses_unit_sequence_axis_for_audio_when_length_input_exists() -> None:
    audio_signal = helper.make_tensor_value_info(
        "audio_signal",
        TensorProto.FLOAT,
        ["batch_size", "time", 80],
    )
    length = helper.make_tensor_value_info(
        "length",
        TensorProto.INT64,
        ["batch_size"],
    )
    graph = helper.make_graph([], "audio_with_length_graph", [audio_signal, length], [])
    model = helper.make_model(graph)
    specs = dict((name, shape) for name, _dtype, shape in _collect_onnx_input_specs(model))
    assert specs["audio_signal"] == (1, 1, 80)
    assert specs["length"] == (1,)


def test_metric_accumulator_treats_equal_nonfinite_pairs_as_zero_error() -> None:
    ref = np.asarray([np.nan, np.inf, -np.inf, 1.25], dtype=np.float32)
    pred = np.asarray([np.nan, np.inf, -np.inf, 1.25], dtype=np.float32)

    acc = _MetricAccumulator()
    acc.update(ref, pred)
    metrics = acc.to_dict()

    assert metrics["max_abs"] == 0.0
    assert metrics["mean_abs"] == 0.0
    assert metrics["rmse"] == 0.0
    assert metrics["cosine_similarity"] == 1.0
    assert _max_abs_error(ref, pred) == 0.0


def test_metric_accumulator_keeps_metrics_finite_on_nonfinite_mismatch() -> None:
    ref = np.asarray([np.nan, 1.0, np.inf, -np.inf], dtype=np.float32)
    pred = np.asarray([0.0, 1.0, -np.inf, np.inf], dtype=np.float32)

    acc = _MetricAccumulator()
    acc.update(ref, pred)
    metrics = acc.to_dict()

    for key in ["max_abs", "mean_abs", "rmse", "cosine_similarity"]:
        assert np.isfinite(float(metrics[key]))


def test_judge_metrics_relaxes_thresholds_for_large_dynamic_range_outputs() -> None:
    result = _judge_metrics(
        metrics={
            "max_abs": 0.49,
            "ref_max_abs": 255.0,
            "ref_rms": 136.0,
            "mean_abs": 0.0055,
            "rmse": 0.0154,
            "cosine_similarity": 0.9999999,
        },
        thresholds={
            "max_abs": 0.05,
            "mean_abs": 0.005,
            "rmse": 0.006,
            "cosine_similarity": 0.999,
        },
        rtol=1.0e-4,
    )
    assert result["pass"] is True
    assert result["effective_thresholds"]["max_abs"] >= 0.49
    assert result["effective_thresholds"]["mean_abs"] >= 0.0055
    assert result["effective_thresholds"]["rmse"] >= 0.0154
