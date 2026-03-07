import json
import os
from typing import Any, Dict, List

from onnx2tf.utils.tflite2sm_bulk_runner import run_bulk_verification


def _write_dummy(path: str) -> None:
    with open(path, "wb") as f:
        f.write(b"dummy")


def _write_saved_model_validation_report(
    *,
    run_dir: str,
    model_path: str,
    inference_status: str = "passed",
    inference_reason: str = "",
    comparison_status: str = "passed",
    comparison_pass: bool = True,
    comparison_reason: str = "",
) -> None:
    model_stem = os.path.splitext(os.path.basename(model_path))[0]
    report_path = os.path.join(
        run_dir,
        "saved_model",
        f"{model_stem}_saved_model_validation_report.json",
    )
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    payload: Dict[str, Any] = {
        "schema_version": 1,
        "inference": {
            "status": inference_status,
            "reason": inference_reason,
            "outputs": ["y"],
        },
        "comparison": {
            "status": comparison_status,
            "reason": comparison_reason,
            "pass": bool(comparison_pass),
            "matched": 1 if comparison_pass else 0,
            "total": 1,
            "max_abs": 0.0 if comparison_pass else 1.0,
            "unmatched_outputs": [] if comparison_pass else ["y"],
        },
        "overall_pass": bool(inference_status == "passed" and comparison_status == "passed" and comparison_pass),
    }
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def _write_accuracy_report(
    *,
    run_dir: str,
    model_path: str,
    evaluation_pass: bool,
) -> None:
    model_stem = os.path.splitext(os.path.basename(model_path))[0]
    report_path = os.path.join(
        run_dir,
        "saved_model",
        f"{model_stem}_accuracy_report.json",
    )
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    payload: Dict[str, Any] = {
        "schema_version": 1,
        "evaluation_pass": bool(evaluation_pass),
        "overall_metrics": {
            "max_abs": 0.0 if evaluation_pass else 1.0,
            "rmse": 0.0 if evaluation_pass else 1.0,
            "cosine_similarity": 1.0 if evaluation_pass else 0.0,
        },
    }
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def test_bulk_runner_preserves_order_and_skips_missing(
    tmp_path,
    monkeypatch,
) -> None:
    model_a = tmp_path / "a.onnx"
    _write_dummy(str(model_a))
    missing = tmp_path / "missing.onnx"
    list_path = tmp_path / "list.txt"
    list_path.write_text(
        "\n".join(
            [
                str(model_a),
                str(missing),
                str(model_a),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    calls: List[Dict[str, Any]] = []

    def _fake_run(cmd, cwd=None, stdout=None, stderr=None, text=None):
        model_path = cmd[cmd.index("-i") + 1]
        _write_saved_model_validation_report(
            run_dir=str(cwd),
            model_path=str(model_path),
        )
        calls.append({"cmd": list(cmd), "cwd": str(cwd)})
        return type("CP", (), {"returncode": 0, "stdout": "ok", "stderr": ""})()

    monkeypatch.setattr("subprocess.run", _fake_run)
    state = run_bulk_verification(
        list_path=str(list_path),
        output_dir=str(tmp_path / "out"),
    )
    entries = state["entries"]
    assert len(entries) == 3
    assert [entry["model_line"] for entry in entries] == [str(model_a), str(missing), str(model_a)]
    assert entries[0]["classification"] == "pass"
    assert entries[1]["classification"] == "missing_model"
    assert entries[2]["classification"] == "pass"
    assert len(calls) == 2


def test_bulk_runner_strict_mismatch_when_report_comparison_fails(
    tmp_path,
    monkeypatch,
) -> None:
    model = tmp_path / "mismatch.onnx"
    _write_dummy(str(model))
    list_path = tmp_path / "list.txt"
    list_path.write_text(f"{model}\n", encoding="utf-8")

    def _fake_run(cmd, cwd=None, stdout=None, stderr=None, text=None):
        model_path = cmd[cmd.index("-i") + 1]
        _write_saved_model_validation_report(
            run_dir=str(cwd),
            model_path=str(model_path),
            comparison_status="failed",
            comparison_pass=False,
            comparison_reason="output_mismatch",
        )
        return type("CP", (), {"returncode": 0, "stdout": "ok", "stderr": ""})()

    monkeypatch.setattr("subprocess.run", _fake_run)
    state = run_bulk_verification(
        list_path=str(list_path),
        output_dir=str(tmp_path / "out"),
    )
    assert state["entries"][0]["classification"] == "saved_model_tflite_mismatch"
    assert state["entries"][0]["strict_pass"] is False
    assert state["summary"]["in_scope_fail_count"] == 1


def test_bulk_runner_resume_skips_completed_entries(
    tmp_path,
    monkeypatch,
) -> None:
    model_a = tmp_path / "a.onnx"
    model_b = tmp_path / "b.onnx"
    _write_dummy(str(model_a))
    _write_dummy(str(model_b))
    list_path = tmp_path / "list.txt"
    list_path.write_text(f"{model_a}\n{model_b}\n", encoding="utf-8")
    output_dir = tmp_path / "out"

    first_calls: List[str] = []

    def _fake_run_first(cmd, cwd=None, stdout=None, stderr=None, text=None):
        model_path = cmd[cmd.index("-i") + 1]
        _write_saved_model_validation_report(
            run_dir=str(cwd),
            model_path=str(model_path),
        )
        first_calls.append(str(model_path))
        return type("CP", (), {"returncode": 0, "stdout": "ok", "stderr": ""})()

    monkeypatch.setattr("subprocess.run", _fake_run_first)
    state_first = run_bulk_verification(
        list_path=str(list_path),
        output_dir=str(output_dir),
    )
    assert len(state_first["entries"]) == 2
    assert len(first_calls) == 2

    second_calls: List[str] = []

    def _fake_run_second(cmd, cwd=None, stdout=None, stderr=None, text=None):
        second_calls.append("called")
        return type("CP", (), {"returncode": 0, "stdout": "", "stderr": ""})()

    monkeypatch.setattr("subprocess.run", _fake_run_second)
    state_second = run_bulk_verification(
        list_path=str(list_path),
        output_dir=str(output_dir),
        resume=True,
    )
    assert len(state_second["entries"]) == 2
    assert len(second_calls) == 0


def test_bulk_runner_fails_strict_when_report_missing_on_success_exit(
    tmp_path,
    monkeypatch,
) -> None:
    model = tmp_path / "no_report.onnx"
    _write_dummy(str(model))
    list_path = tmp_path / "list.txt"
    list_path.write_text(f"{model}\n", encoding="utf-8")

    def _fake_run(cmd, cwd=None, stdout=None, stderr=None, text=None):
        return type("CP", (), {"returncode": 0, "stdout": "ok", "stderr": ""})()

    monkeypatch.setattr("subprocess.run", _fake_run)
    state = run_bulk_verification(
        list_path=str(list_path),
        output_dir=str(tmp_path / "out"),
    )
    entry = state["entries"][0]
    assert entry["classification"] == "conversion_error"
    assert entry["strict_pass"] is False
    assert entry["reason"] == "saved_model_validation_report_missing"


def test_bulk_runner_accuracy_fail_is_accepted_when_saved_model_comparison_passes(
    tmp_path,
    monkeypatch,
) -> None:
    model = tmp_path / "upstream_fail.onnx"
    _write_dummy(str(model))
    list_path = tmp_path / "list.txt"
    list_path.write_text(f"{model}\n", encoding="utf-8")

    def _fake_run(cmd, cwd=None, stdout=None, stderr=None, text=None):
        model_path = cmd[cmd.index("-i") + 1]
        _write_saved_model_validation_report(
            run_dir=str(cwd),
            model_path=str(model_path),
        )
        _write_accuracy_report(
            run_dir=str(cwd),
            model_path=str(model_path),
            evaluation_pass=False,
        )
        return type("CP", (), {"returncode": 0, "stdout": "ok", "stderr": ""})()

    monkeypatch.setattr("subprocess.run", _fake_run)
    state = run_bulk_verification(
        list_path=str(list_path),
        output_dir=str(tmp_path / "out"),
    )
    entry = state["entries"][0]
    assert entry["classification"] == "pass"
    assert entry["strict_pass"] is True
    assert entry["reason"] == ""


def test_bulk_runner_accuracy_fail_still_fails_when_saved_model_comparison_fails(
    tmp_path,
    monkeypatch,
) -> None:
    model = tmp_path / "upstream_fail_mismatch.onnx"
    _write_dummy(str(model))
    list_path = tmp_path / "list.txt"
    list_path.write_text(f"{model}\n", encoding="utf-8")

    def _fake_run(cmd, cwd=None, stdout=None, stderr=None, text=None):
        model_path = cmd[cmd.index("-i") + 1]
        _write_saved_model_validation_report(
            run_dir=str(cwd),
            model_path=str(model_path),
            comparison_status="failed",
            comparison_pass=False,
            comparison_reason="output_mismatch",
        )
        _write_accuracy_report(
            run_dir=str(cwd),
            model_path=str(model_path),
            evaluation_pass=False,
        )
        return type("CP", (), {"returncode": 0, "stdout": "ok", "stderr": ""})()

    monkeypatch.setattr("subprocess.run", _fake_run)
    state = run_bulk_verification(
        list_path=str(list_path),
        output_dir=str(tmp_path / "out"),
    )
    entry = state["entries"][0]
    assert entry["classification"] == "saved_model_tflite_mismatch"
    assert entry["strict_pass"] is False
    assert entry["reason"] == "output_mismatch"


def test_bulk_runner_policy_skips_deformable_detr_one_input_simple(
    tmp_path,
    monkeypatch,
) -> None:
    model = tmp_path / "deformable_detr_one_input_simple.onnx"
    _write_dummy(str(model))
    list_path = tmp_path / "list.txt"
    list_path.write_text(f"{model}\n", encoding="utf-8")

    calls: List[List[str]] = []

    def _fake_run(cmd, cwd=None, stdout=None, stderr=None, text=None):
        calls.append(list(cmd))
        return type("CP", (), {"returncode": 0, "stdout": "ok", "stderr": ""})()

    monkeypatch.setattr("subprocess.run", _fake_run)
    state = run_bulk_verification(
        list_path=str(list_path),
        output_dir=str(tmp_path / "out"),
    )
    entry = state["entries"][0]
    assert entry["classification"] == "skipped_model"
    assert entry["strict_pass"] is True
    assert entry["reason"] == "policy_skipped_model"
    assert len(calls) == 0
