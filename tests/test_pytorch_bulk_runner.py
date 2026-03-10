import json
import subprocess
from pathlib import Path
from typing import Any, Dict, List

from onnx2tf.utils.pytorch_bulk_runner import run_bulk_pytorch_verification


def _write_dummy(path: Path) -> None:
    path.write_bytes(b"dummy")


def _write_pytorch_accuracy_report(
    *,
    path: Path,
    evaluation_pass: bool,
) -> None:
    payload: Dict[str, Any] = {
        "schema_version": 1,
        "backend": "pytorch_package",
        "evaluation_pass": bool(evaluation_pass),
        "overall_metrics": {
            "max_abs": 0.0 if evaluation_pass else 1.0,
            "mean_abs": 0.0 if evaluation_pass else 1.0,
            "rmse": 0.0 if evaluation_pass else 1.0,
            "cosine_similarity": 1.0 if evaluation_pass else 0.0,
            "ref_max_abs": 1.0,
            "ref_rms": 1.0,
        },
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _write_pytorch_smoke_report(
    *,
    path: Path,
    inference_pass: bool,
    error_type: str = "RuntimeError",
    error_message: str = "boom",
) -> None:
    payload: Dict[str, Any] = {
        "schema_version": 1,
        "backend": "pytorch_package_smoke",
        "inference_pass": bool(inference_pass),
        "error": None if inference_pass else {
            "sample_index": 0,
            "error_type": str(error_type),
            "error_message": str(error_message),
        },
        "samples": [],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def test_pytorch_bulk_runner_preserves_order_and_skips_missing(
    tmp_path,
    monkeypatch,
) -> None:
    model_a = tmp_path / "a.onnx"
    _write_dummy(model_a)
    missing = tmp_path / "missing.onnx"
    list_path = tmp_path / "list.txt"
    list_path.write_text(f"{model_a}\n{missing}\n{model_a}\n", encoding="utf-8")

    calls: List[str] = []

    def _fake_run(cmd, cwd=None, stdout=None, stderr=None, text=None, timeout=None):
        calls.append(str(cmd[cmd.index("-i") + 1]))
        output_dir = Path(cmd[cmd.index("-o") + 1])
        stem = Path(cmd[cmd.index("-i") + 1]).stem
        (output_dir / f"{stem}_float32.tflite").write_bytes(b"tflite")
        return type("CP", (), {"returncode": 0, "stdout": "ok", "stderr": ""})()

    def _fake_eval(**kwargs):
        _write_pytorch_accuracy_report(
            path=Path(kwargs["output_report_path"]),
            evaluation_pass=True,
        )
        return json.loads(Path(kwargs["output_report_path"]).read_text(encoding="utf-8"))

    def _fake_smoke(**kwargs):
        _write_pytorch_smoke_report(
            path=Path(kwargs["output_report_path"]),
            inference_pass=True,
        )
        return json.loads(Path(kwargs["output_report_path"]).read_text(encoding="utf-8"))

    monkeypatch.setattr(subprocess, "run", _fake_run)
    monkeypatch.setattr("onnx2tf.utils.pytorch_bulk_runner.smoke_test_pytorch_package_inference", _fake_smoke)
    monkeypatch.setattr("onnx2tf.utils.pytorch_bulk_runner.evaluate_pytorch_package_outputs", _fake_eval)
    monkeypatch.setattr("onnx.load", lambda *_args, **_kwargs: object())

    state = run_bulk_pytorch_verification(
        list_path=str(list_path),
        output_dir=str(tmp_path / "out"),
    )
    entries = state["entries"]
    assert [entry["model_line"] for entry in entries] == [str(model_a), str(missing), str(model_a)]
    assert entries[0]["classification"] == "pass"
    assert entries[1]["classification"] == "missing_model"
    assert entries[1]["strict_pass"] is True
    assert entries[2]["classification"] == "pass"
    assert len(calls) == 2


def test_pytorch_bulk_runner_resume_skips_completed_entries(
    tmp_path,
    monkeypatch,
) -> None:
    model_a = tmp_path / "a.onnx"
    model_b = tmp_path / "b.onnx"
    _write_dummy(model_a)
    _write_dummy(model_b)
    list_path = tmp_path / "list.txt"
    list_path.write_text(f"{model_a}\n{model_b}\n", encoding="utf-8")
    output_dir = tmp_path / "out"

    first_calls: List[str] = []

    def _fake_run_first(cmd, cwd=None, stdout=None, stderr=None, text=None, timeout=None):
        first_calls.append(str(cmd[cmd.index("-i") + 1]))
        output_dir = Path(cmd[cmd.index("-o") + 1])
        stem = Path(cmd[cmd.index("-i") + 1]).stem
        (output_dir / f"{stem}_float32.tflite").write_bytes(b"tflite")
        return type("CP", (), {"returncode": 0, "stdout": "ok", "stderr": ""})()

    def _fake_eval_first(**kwargs):
        _write_pytorch_accuracy_report(
            path=Path(kwargs["output_report_path"]),
            evaluation_pass=True,
        )
        return json.loads(Path(kwargs["output_report_path"]).read_text(encoding="utf-8"))

    def _fake_smoke_first(**kwargs):
        _write_pytorch_smoke_report(
            path=Path(kwargs["output_report_path"]),
            inference_pass=True,
        )
        return json.loads(Path(kwargs["output_report_path"]).read_text(encoding="utf-8"))

    monkeypatch.setattr(subprocess, "run", _fake_run_first)
    monkeypatch.setattr("onnx2tf.utils.pytorch_bulk_runner.smoke_test_pytorch_package_inference", _fake_smoke_first)
    monkeypatch.setattr("onnx2tf.utils.pytorch_bulk_runner.evaluate_pytorch_package_outputs", _fake_eval_first)
    monkeypatch.setattr("onnx.load", lambda *_args, **_kwargs: object())
    first_state = run_bulk_pytorch_verification(
        list_path=str(list_path),
        output_dir=str(output_dir),
    )
    assert len(first_state["entries"]) == 2
    assert len(first_calls) == 2

    second_calls: List[str] = []

    def _fake_run_second(cmd, cwd=None, stdout=None, stderr=None, text=None, timeout=None):
        second_calls.append("called")
        return type("CP", (), {"returncode": 0, "stdout": "", "stderr": ""})()

    monkeypatch.setattr(subprocess, "run", _fake_run_second)
    second_state = run_bulk_pytorch_verification(
        list_path=str(list_path),
        output_dir=str(output_dir),
        resume=True,
    )
    assert len(second_state["entries"]) == 2
    assert len(second_calls) == 0


def test_pytorch_bulk_runner_skips_requested_models(
    tmp_path,
    monkeypatch,
) -> None:
    model_a = tmp_path / "a.onnx"
    model_b = tmp_path / "b.onnx"
    _write_dummy(model_a)
    _write_dummy(model_b)
    list_path = tmp_path / "list.txt"
    list_path.write_text(f"{model_a}\n{model_b}\n", encoding="utf-8")

    calls: List[str] = []

    def _fake_run(cmd, cwd=None, stdout=None, stderr=None, text=None, timeout=None):
        calls.append(str(cmd[cmd.index("-i") + 1]))
        output_dir = Path(cmd[cmd.index("-o") + 1])
        stem = Path(cmd[cmd.index("-i") + 1]).stem
        (output_dir / f"{stem}_float32.tflite").write_bytes(b"tflite")
        return type("CP", (), {"returncode": 0, "stdout": "ok", "stderr": ""})()

    def _fake_smoke(**kwargs):
        _write_pytorch_smoke_report(
            path=Path(kwargs["output_report_path"]),
            inference_pass=True,
        )
        return json.loads(Path(kwargs["output_report_path"]).read_text(encoding="utf-8"))

    def _fake_eval(**kwargs):
        _write_pytorch_accuracy_report(
            path=Path(kwargs["output_report_path"]),
            evaluation_pass=True,
        )
        return json.loads(Path(kwargs["output_report_path"]).read_text(encoding="utf-8"))

    monkeypatch.setattr(subprocess, "run", _fake_run)
    monkeypatch.setattr("onnx2tf.utils.pytorch_bulk_runner.smoke_test_pytorch_package_inference", _fake_smoke)
    monkeypatch.setattr("onnx2tf.utils.pytorch_bulk_runner.evaluate_pytorch_package_outputs", _fake_eval)
    monkeypatch.setattr("onnx.load", lambda *_args, **_kwargs: object())

    state = run_bulk_pytorch_verification(
        list_path=str(list_path),
        output_dir=str(tmp_path / "out"),
        skip_model_names=["b.onnx"],
    )
    assert [entry["classification"] for entry in state["entries"]] == ["pass", "skipped_model"]
    assert state["entries"][1]["reason"] == "skipped_by_request"
    assert calls == [str(model_a)]


def test_pytorch_bulk_runner_marks_timeout(
    tmp_path,
    monkeypatch,
) -> None:
    model = tmp_path / "slow.onnx"
    _write_dummy(model)
    list_path = tmp_path / "list.txt"
    list_path.write_text(f"{model}\n", encoding="utf-8")

    def _fake_run(cmd, cwd=None, stdout=None, stderr=None, text=None, timeout=None):
        raise subprocess.TimeoutExpired(cmd=cmd, timeout=timeout)

    monkeypatch.setattr(subprocess, "run", _fake_run)
    state = run_bulk_pytorch_verification(
        list_path=str(list_path),
        output_dir=str(tmp_path / "out"),
        timeout_sec=1,
    )
    entry = state["entries"][0]
    assert entry["classification"] == "timeout"
    assert entry["strict_pass"] is False


def test_pytorch_bulk_runner_applies_default_skip_model_names(
    tmp_path,
    monkeypatch,
) -> None:
    model = tmp_path / "dehaze_maxim_2022aug_opt_sim_special_05_max_to_relu.onnx"
    _write_dummy(model)
    list_path = tmp_path / "list.txt"
    list_path.write_text(f"{model}\n", encoding="utf-8")

    calls: List[str] = []

    def _fake_run(cmd, cwd=None, stdout=None, stderr=None, text=None, timeout=None):
        calls.append(str(cmd[cmd.index("-i") + 1]))
        raise AssertionError("default-skip model should not invoke onnx2tf")

    monkeypatch.setattr(subprocess, "run", _fake_run)

    state = run_bulk_pytorch_verification(
        list_path=str(list_path),
        output_dir=str(tmp_path / "out"),
    )
    entry = state["entries"][0]
    assert entry["classification"] == "skipped_model"
    assert entry["reason"] == "skipped_by_request"
    assert calls == []


def test_pytorch_bulk_runner_marks_accuracy_failure(
    tmp_path,
    monkeypatch,
) -> None:
    model = tmp_path / "mismatch.onnx"
    _write_dummy(model)
    list_path = tmp_path / "list.txt"
    list_path.write_text(f"{model}\n", encoding="utf-8")

    def _fake_run(cmd, cwd=None, stdout=None, stderr=None, text=None, timeout=None):
        output_dir = Path(cmd[cmd.index("-o") + 1])
        stem = Path(cmd[cmd.index("-i") + 1]).stem
        (output_dir / f"{stem}_float32.tflite").write_bytes(b"tflite")
        return type("CP", (), {"returncode": 0, "stdout": "ok", "stderr": ""})()

    def _fake_eval(**kwargs):
        _write_pytorch_accuracy_report(
            path=Path(kwargs["output_report_path"]),
            evaluation_pass=False,
        )
        return json.loads(Path(kwargs["output_report_path"]).read_text(encoding="utf-8"))

    def _fake_smoke(**kwargs):
        _write_pytorch_smoke_report(
            path=Path(kwargs["output_report_path"]),
            inference_pass=True,
        )
        return json.loads(Path(kwargs["output_report_path"]).read_text(encoding="utf-8"))

    monkeypatch.setattr(subprocess, "run", _fake_run)
    monkeypatch.setattr("onnx2tf.utils.pytorch_bulk_runner.smoke_test_pytorch_package_inference", _fake_smoke)
    monkeypatch.setattr("onnx2tf.utils.pytorch_bulk_runner.evaluate_pytorch_package_outputs", _fake_eval)
    monkeypatch.setattr("onnx.load", lambda *_args, **_kwargs: object())

    state = run_bulk_pytorch_verification(
        list_path=str(list_path),
        output_dir=str(tmp_path / "out"),
    )
    entry = state["entries"][0]
    assert entry["classification"] == "pytorch_accuracy_mismatch"
    assert entry["strict_pass"] is False


def test_pytorch_bulk_runner_accepts_gmflow_accuracy_mismatch(
    tmp_path,
    monkeypatch,
) -> None:
    model = tmp_path / "gmflow-scale1-test.onnx"
    _write_dummy(model)
    list_path = tmp_path / "list.txt"
    list_path.write_text(f"{model}\n", encoding="utf-8")

    def _fake_run(cmd, cwd=None, stdout=None, stderr=None, text=None, timeout=None):
        output_dir = Path(cmd[cmd.index("-o") + 1])
        stem = Path(cmd[cmd.index("-i") + 1]).stem
        (output_dir / f"{stem}_float32.tflite").write_bytes(b"tflite")
        return type("CP", (), {"returncode": 0, "stdout": "ok", "stderr": ""})()

    def _fake_eval(**kwargs):
        _write_pytorch_accuracy_report(
            path=Path(kwargs["output_report_path"]),
            evaluation_pass=False,
        )
        return json.loads(Path(kwargs["output_report_path"]).read_text(encoding="utf-8"))

    def _fake_smoke(**kwargs):
        _write_pytorch_smoke_report(
            path=Path(kwargs["output_report_path"]),
            inference_pass=True,
        )
        return json.loads(Path(kwargs["output_report_path"]).read_text(encoding="utf-8"))

    monkeypatch.setattr(subprocess, "run", _fake_run)
    monkeypatch.setattr("onnx2tf.utils.pytorch_bulk_runner.smoke_test_pytorch_package_inference", _fake_smoke)
    monkeypatch.setattr("onnx2tf.utils.pytorch_bulk_runner.evaluate_pytorch_package_outputs", _fake_eval)
    monkeypatch.setattr("onnx.load", lambda *_args, **_kwargs: object())

    state = run_bulk_pytorch_verification(
        list_path=str(list_path),
        output_dir=str(tmp_path / "out"),
    )
    entry = state["entries"][0]
    assert entry["classification"] == "pass"
    assert entry["strict_pass"] is True
    assert entry["reason"] == "accepted_accuracy_mismatch"


def test_pytorch_bulk_runner_marks_tflite_generation_missing(
    tmp_path,
    monkeypatch,
) -> None:
    model = tmp_path / "no_tflite.onnx"
    _write_dummy(model)
    list_path = tmp_path / "list.txt"
    list_path.write_text(f"{model}\n", encoding="utf-8")

    def _fake_run(cmd, cwd=None, stdout=None, stderr=None, text=None, timeout=None):
        return type("CP", (), {"returncode": 0, "stdout": "ok", "stderr": ""})()

    monkeypatch.setattr(subprocess, "run", _fake_run)
    monkeypatch.setattr("onnx.load", lambda *_args, **_kwargs: object())

    state = run_bulk_pytorch_verification(
        list_path=str(list_path),
        output_dir=str(tmp_path / "out"),
    )
    entry = state["entries"][0]
    assert entry["classification"] == "tflite_generation_missing"
    assert entry["strict_pass"] is False


def test_pytorch_bulk_runner_marks_pytorch_inference_failure_when_tflite_exists(
    tmp_path,
    monkeypatch,
) -> None:
    model = tmp_path / "failing.onnx"
    _write_dummy(model)
    list_path = tmp_path / "list.txt"
    list_path.write_text(f"{model}\n", encoding="utf-8")

    def _fake_run(cmd, cwd=None, stdout=None, stderr=None, text=None, timeout=None):
        output_dir = Path(cmd[cmd.index("-o") + 1])
        stem = Path(cmd[cmd.index("-i") + 1]).stem
        (output_dir / f"{stem}_float32.tflite").write_bytes(b"tflite")
        return type("CP", (), {"returncode": 0, "stdout": "ok", "stderr": ""})()

    def _fake_smoke(**kwargs):
        _write_pytorch_smoke_report(
            path=Path(kwargs["output_report_path"]),
            inference_pass=False,
            error_type="RuntimeError",
            error_message="shape mismatch",
        )
        return json.loads(Path(kwargs["output_report_path"]).read_text(encoding="utf-8"))

    monkeypatch.setattr(subprocess, "run", _fake_run)
    monkeypatch.setattr("onnx2tf.utils.pytorch_bulk_runner.smoke_test_pytorch_package_inference", _fake_smoke)
    monkeypatch.setattr("onnx.load", lambda *_args, **_kwargs: object())

    state = run_bulk_pytorch_verification(
        list_path=str(list_path),
        output_dir=str(tmp_path / "out"),
    )
    entry = state["entries"][0]
    assert entry["classification"] == "pytorch_inference_error"
    assert entry["strict_pass"] is False
    assert entry["tflite_generated"] is True
    assert entry["reason"] == "RuntimeError:shape mismatch"


def test_pytorch_bulk_runner_ignores_onnxruntime_reference_failure_when_smoke_passes(
    tmp_path,
    monkeypatch,
) -> None:
    model = tmp_path / "ort_fail.onnx"
    _write_dummy(model)
    list_path = tmp_path / "list.txt"
    list_path.write_text(f"{model}\n", encoding="utf-8")

    def _fake_run(cmd, cwd=None, stdout=None, stderr=None, text=None, timeout=None):
        output_dir = Path(cmd[cmd.index("-o") + 1])
        stem = Path(cmd[cmd.index("-i") + 1]).stem
        (output_dir / f"{stem}_float32.tflite").write_bytes(b"tflite")
        return type("CP", (), {"returncode": 0, "stdout": "ok", "stderr": ""})()

    def _fake_smoke(**kwargs):
        _write_pytorch_smoke_report(
            path=Path(kwargs["output_report_path"]),
            inference_pass=True,
        )
        return json.loads(Path(kwargs["output_report_path"]).read_text(encoding="utf-8"))

    def _fake_eval(**kwargs):
        payload = {
            "schema_version": 1,
            "backend": "pytorch_package",
            "evaluation_pass": None,
            "evaluation_skipped": True,
            "skip_reason": "onnxruntime_reference_error",
            "onnxruntime_reference_error": {
                "stage": "inference",
                "error_type": "RuntimeError",
                "error_message": "onnxruntime failed",
            },
        }
        Path(kwargs["output_report_path"]).write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return payload

    monkeypatch.setattr(subprocess, "run", _fake_run)
    monkeypatch.setattr("onnx2tf.utils.pytorch_bulk_runner.smoke_test_pytorch_package_inference", _fake_smoke)
    monkeypatch.setattr("onnx2tf.utils.pytorch_bulk_runner.evaluate_pytorch_package_outputs", _fake_eval)
    monkeypatch.setattr("onnx.load", lambda *_args, **_kwargs: object())

    state = run_bulk_pytorch_verification(
        list_path=str(list_path),
        output_dir=str(tmp_path / "out"),
    )
    entry = state["entries"][0]
    assert entry["classification"] == "pass"
    assert entry["strict_pass"] is True
    assert entry["reason"] == "onnxruntime_reference_error"


def test_pytorch_bulk_runner_smoke_only_skips_accuracy_evaluation(
    tmp_path,
    monkeypatch,
) -> None:
    model = tmp_path / "smoke_only.onnx"
    _write_dummy(model)
    list_path = tmp_path / "list.txt"
    list_path.write_text(f"{model}\n", encoding="utf-8")

    def _fake_run(cmd, cwd=None, stdout=None, stderr=None, text=None, timeout=None):
        output_dir = Path(cmd[cmd.index("-o") + 1])
        stem = Path(cmd[cmd.index("-i") + 1]).stem
        (output_dir / f"{stem}_float32.tflite").write_bytes(b"tflite")
        return type("CP", (), {"returncode": 0, "stdout": "ok", "stderr": ""})()

    def _fake_smoke(**kwargs):
        _write_pytorch_smoke_report(
            path=Path(kwargs["output_report_path"]),
            inference_pass=True,
        )
        return json.loads(Path(kwargs["output_report_path"]).read_text(encoding="utf-8"))

    def _fake_eval(**_kwargs):
        raise AssertionError("accuracy evaluation should not be called in smoke_only mode")

    monkeypatch.setattr(subprocess, "run", _fake_run)
    monkeypatch.setattr("onnx2tf.utils.pytorch_bulk_runner.smoke_test_pytorch_package_inference", _fake_smoke)
    monkeypatch.setattr("onnx2tf.utils.pytorch_bulk_runner.evaluate_pytorch_package_outputs", _fake_eval)
    monkeypatch.setattr("onnx.load", lambda *_args, **_kwargs: object())

    state = run_bulk_pytorch_verification(
        list_path=str(list_path),
        output_dir=str(tmp_path / "out"),
        smoke_only=True,
    )
    entry = state["entries"][0]
    assert entry["classification"] == "pass"
    assert entry["strict_pass"] is True
    assert entry["reason"] == "smoke_only"
