import numpy as np
from onnx import TensorProto, helper, numpy_helper

from onnx2tf.tflite_builder.accuracy_evaluator import (
    _FLOAT_METRIC_THRESHOLDS,
    _QUANT_METRIC_THRESHOLDS,
    _build_seeded_input_distribution_overrides,
    _build_static_control_input_overrides,
    _build_eval_inputs_for_sample,
    _collect_onnx_input_specs,
    _generate_seeded_input,
    _judge_per_output_metrics,
    _judge_metrics,
    _max_abs_error,
    _MetricAccumulator,
    _prepare_onnx_graph_for_onnxruntime,
    _resolve_tflite_evaluation_pass,
)


def test_default_accuracy_thresholds_match_pytorch_style() -> None:
    expected = {
        "max_abs": 5.0e-2,
        "mean_abs": 5.0e-3,
        "rmse": 6.0e-3,
        "cosine_similarity": 0.9990,
    }
    assert _FLOAT_METRIC_THRESHOLDS == expected
    assert _QUANT_METRIC_THRESHOLDS == expected


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


def test_build_seeded_input_distribution_overrides_prefers_uniform_for_image_to_image_nchw_model(
    monkeypatch,
) -> None:
    monkeypatch.delenv("ONNX2TF_EVAL_FLOAT_RANDOM_DISTRIBUTION", raising=False)
    graph = helper.make_graph(
        [],
        "image_to_image",
        [
            helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 3, 32, 48]),
        ],
        [
            helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 3, 32, 48]),
        ],
    )
    model = helper.make_model(graph)
    input_specs = _collect_onnx_input_specs(model)
    overrides = _build_seeded_input_distribution_overrides(
        onnx_graph=model,
        input_specs=input_specs,
    )
    assert overrides == {"input": "uniform_0_1"}


def test_build_seeded_input_distribution_overrides_detects_image_normalization_prefix(
    monkeypatch,
) -> None:
    monkeypatch.delenv("ONNX2TF_EVAL_FLOAT_RANDOM_DISTRIBUTION", raising=False)
    mean = numpy_helper.from_array(
        np.asarray([0.4, 0.45, 0.5], dtype=np.float32).reshape(3, 1, 1),
        name="mean",
    )
    std = numpy_helper.from_array(
        np.asarray([0.2, 0.25, 0.3], dtype=np.float32).reshape(3, 1, 1),
        name="std",
    )
    graph = helper.make_graph(
        [
            helper.make_node("Squeeze", ["input"], ["squeezed"], axes=[0]),
            helper.make_node("Sub", ["squeezed", "mean"], ["centered"]),
            helper.make_node("Div", ["centered", "std"], ["normalized"]),
            helper.make_node("ReduceMean", ["normalized"], ["output"], keepdims=0),
        ],
        "normalized_image_classifier",
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 3, 32, 48])],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, [])],
        initializer=[mean, std],
    )
    model = helper.make_model(graph)
    overrides = _build_seeded_input_distribution_overrides(
        onnx_graph=model,
        input_specs=_collect_onnx_input_specs(model),
    )
    assert overrides == {"input": "uniform_0_1"}


def test_build_eval_inputs_for_sample_uses_uniform_override_for_nchw_image_input() -> None:
    inputs = _build_eval_inputs_for_sample(
        input_specs=[
            ("input", np.dtype(np.float32), (1, 3, 16, 16)),
        ],
        custom_inputs={},
        sample_index=0,
        rng=np.random.default_rng(0),
        distribution_overrides={"input": "uniform_0_1"},
    )
    assert float(np.min(inputs["input"])) >= 0.0
    assert float(np.max(inputs["input"])) <= 1.0


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


def test_build_static_control_input_overrides_infers_topk_k_from_output_shape() -> None:
    data = helper.make_tensor_value_info("data", TensorProto.FLOAT, [1, 8400])
    k = helper.make_tensor_value_info("k", TensorProto.INT64, [1])
    values = helper.make_tensor_value_info("values", TensorProto.FLOAT, [1, 1250])
    indices = helper.make_tensor_value_info("indices", TensorProto.INT64, [1, 1250])
    topk = helper.make_node(
        "TopK",
        ["data", "k"],
        ["values", "indices"],
        axis=1,
        largest=1,
        sorted=1,
    )
    model = helper.make_model(
        helper.make_graph([topk], "topk_control_input", [data, k], [values, indices]),
        opset_imports=[helper.make_operatorsetid("", 13)],
    )
    input_specs = _collect_onnx_input_specs(model)

    overrides = _build_static_control_input_overrides(
        onnx_graph=model,
        input_specs=input_specs,
    )
    inputs = _build_eval_inputs_for_sample(
        input_specs=input_specs,
        custom_inputs={},
        sample_index=0,
        rng=np.random.default_rng(0),
        generated_input_overrides=overrides,
    )

    np.testing.assert_array_equal(inputs["k"], np.asarray([1250], dtype=np.int64))


def test_build_static_control_input_overrides_uses_audio_sample_rate() -> None:
    sample_rate = helper.make_tensor_value_info("sr", TensorProto.INT64, [])
    output = helper.make_tensor_value_info("output", TensorProto.INT64, [])
    model = helper.make_model(
        helper.make_graph(
            [helper.make_node("Identity", ["sr"], ["output"])],
            "sample_rate_control_input",
            [sample_rate],
            [output],
        )
    )
    input_specs = _collect_onnx_input_specs(model)

    overrides = _build_static_control_input_overrides(
        onnx_graph=model,
        input_specs=input_specs,
    )

    np.testing.assert_array_equal(
        overrides["sr"],
        np.asarray(16000, dtype=np.int64),
    )


def test_prepare_onnx_graph_for_onnxruntime_upgrades_legacy_default_opset() -> None:
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 2])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 2])
    model = helper.make_model(
        helper.make_graph(
            [helper.make_node("Relu", ["x"], ["y"])],
            "legacy_opset",
            [x],
            [y],
        ),
        opset_imports=[helper.make_operatorsetid("", 6)],
    )

    prepared = _prepare_onnx_graph_for_onnxruntime(model)

    assert next(opset.version for opset in model.opset_import if opset.domain == "") == 6
    assert next(opset.version for opset in prepared.opset_import if opset.domain == "") == 7


def test_prepare_onnx_graph_for_onnxruntime_preserves_standard_opset20_gelu() -> None:
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 4])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 4])
    model = helper.make_model(
        helper.make_graph(
            [
                helper.make_node(
                    "Gelu",
                    ["x"],
                    ["y"],
                    approximate="none",
                )
            ],
            "standard_gelu",
            [x],
            [y],
        ),
        opset_imports=[helper.make_operatorsetid("", 20)],
    )

    prepared = _prepare_onnx_graph_for_onnxruntime(model)

    assert prepared.graph.node[0].domain == ""
    assert [attr.name for attr in prepared.graph.node[0].attribute] == [
        "approximate"
    ]


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


def test_collect_onnx_input_specs_reuses_shared_symbolic_batch_across_axes() -> None:
    data = helper.make_tensor_value_info(
        "data",
        TensorProto.FLOAT,
        ["batch_size", "seq_length", 3],
    )
    initial_h = helper.make_tensor_value_info(
        "initial_h",
        TensorProto.FLOAT,
        [1, "batch_size", 5],
    )
    model = helper.make_model(
        helper.make_graph(
            [helper.make_node("GRU", ["data"], ["sequence"])],
            "shared_batch",
            [initial_h, data],
            [],
        ),
    )

    specs = {
        name: shape
        for name, _dtype, shape in _collect_onnx_input_specs(model)
    }

    assert specs["data"] == (1, 1, 3)
    assert specs["initial_h"] == (1, 1, 5)


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


def test_judge_per_output_metrics_passes_when_each_output_meets_thresholds() -> None:
    acc = _MetricAccumulator()
    ref = np.asarray([1.0, 2.0, 3.0], dtype=np.float32)
    pred = np.asarray([1.001, 2.001, 3.001], dtype=np.float32)
    acc.update(ref, pred)

    numeric_outputs_pass, per_output_metric_judgements = _judge_per_output_metrics(
        output_names=["y"],
        per_output_metrics={"y": acc},
        thresholds=_FLOAT_METRIC_THRESHOLDS,
        rtol=1.0e-4,
    )

    assert numeric_outputs_pass is True
    assert per_output_metric_judgements["y"] is not None
    assert per_output_metric_judgements["y"]["pass"] is True


def test_resolve_tflite_evaluation_pass_matches_pytorch_style_when_allclose_fails() -> None:
    metric_judgement = {
        "pass": True,
        "checks": {
            "max_abs": True,
            "mean_abs": True,
            "rmse": True,
            "cosine_similarity": True,
        },
    }

    assert _resolve_tflite_evaluation_pass(
        metric_judgement=metric_judgement,
        numeric_outputs_pass=True,
    ) is True


def test_resolve_tflite_evaluation_pass_fails_when_metric_checks_fail() -> None:
    metric_judgement = {
        "pass": False,
        "checks": {
            "max_abs": False,
            "mean_abs": True,
            "rmse": True,
            "cosine_similarity": True,
        },
    }

    assert _resolve_tflite_evaluation_pass(
        metric_judgement=metric_judgement,
        numeric_outputs_pass=False,
    ) is False


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


def test_judge_metrics_bypasses_cosine_for_low_energy_outputs() -> None:
    result = _judge_metrics(
        metrics={
            "max_abs": 2.2e-4,
            "ref_max_abs": 1.7e-4,
            "ref_rms": 2.7e-5,
            "mean_abs": 3.1e-5,
            "rmse": 4.0e-5,
            "cosine_similarity": 0.02,
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
    assert result["checks"]["cosine_similarity"] is True
    assert result["cosine_similarity_bypassed_for_low_energy"] is True


def test_judge_metrics_keeps_cosine_for_nontrivial_energy_outputs() -> None:
    result = _judge_metrics(
        metrics={
            "max_abs": 2.2e-4,
            "ref_max_abs": 0.2,
            "ref_rms": 0.05,
            "mean_abs": 3.1e-5,
            "rmse": 4.0e-5,
            "cosine_similarity": 0.02,
        },
        thresholds={
            "max_abs": 0.05,
            "mean_abs": 0.005,
            "rmse": 0.006,
            "cosine_similarity": 0.999,
        },
        rtol=1.0e-4,
    )
    assert result["pass"] is False
    assert result["checks"]["cosine_similarity"] is False
    assert result["cosine_similarity_bypassed_for_low_energy"] is False
