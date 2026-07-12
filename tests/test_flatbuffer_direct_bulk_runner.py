import json
import subprocess
from pathlib import Path
from typing import Any, Dict, List

import pytest

from onnx2tf.utils import flatbuffer_direct_bulk_runner as bulk_runner
from onnx2tf.utils.flatbuffer_direct_bulk_runner import (
    run_flatbuffer_direct_bulk_verification,
)


def _write_dummy(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"dummy")


def _write_accuracy_report(
    *,
    path: Path,
    evaluation_pass: bool,
) -> None:
    payload: Dict[str, Any] = {
        "schema_version": 1,
        "evaluation_pass": bool(evaluation_pass),
        "overall_metrics": {
            "max_abs": 0.0 if evaluation_pass else 1.0,
            "rmse": 0.0 if evaluation_pass else 1.0,
            "cosine_similarity": 1.0 if evaluation_pass else 0.0,
        },
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def test_flatbuffer_direct_bulk_runner_preserves_sorted_discovery_order(
    tmp_path,
    monkeypatch,
) -> None:
    model_b = tmp_path / "z_dir" / "b.onnx"
    model_a = tmp_path / "a_dir" / "a.onnx"
    _write_dummy(model_b)
    _write_dummy(model_a)

    calls: List[str] = []

    def _fake_run(cmd, cwd=None, stdout=None, stderr=None, text=None, timeout=None):
        model_path = str(cmd[cmd.index("-i") + 1])
        artifact_dir = Path(cmd[cmd.index("-o") + 1])
        stem = Path(model_path).stem
        calls.append(model_path)
        _write_accuracy_report(
            path=artifact_dir / f"{stem}_accuracy_report.json",
            evaluation_pass=True,
        )
        _write_accuracy_report(
            path=artifact_dir / f"{stem}_pytorch_accuracy_report.json",
            evaluation_pass=True,
        )
        return type("CP", (), {"returncode": 0, "stdout": "ok", "stderr": ""})()

    monkeypatch.setattr(subprocess, "run", _fake_run)
    state = run_flatbuffer_direct_bulk_verification(
        root_dir=str(tmp_path),
        output_dir=str(tmp_path / "out"),
    )
    assert [entry["model_path"] for entry in state["entries"]] == sorted(
        [str(model_a.resolve()), str(model_b.resolve())]
    )
    assert [Path(path).name for path in calls] == ["a.onnx", "b.onnx"]
    assert all(Path(path).parent.parent.name == "runs" for path in calls)


def test_flatbuffer_direct_bulk_runner_skips_missing_models_cleanly(
    tmp_path,
    monkeypatch,
) -> None:
    existing = tmp_path / "a.onnx"
    missing = tmp_path / "missing.onnx"
    _write_dummy(existing)

    monkeypatch.setattr(
        "onnx2tf.utils.flatbuffer_direct_bulk_runner._discover_onnx_models",
        lambda _root_dir: [str(existing), str(missing)],
    )

    calls: List[str] = []

    def _fake_run(cmd, cwd=None, stdout=None, stderr=None, text=None, timeout=None):
        model_path = str(cmd[cmd.index("-i") + 1])
        artifact_dir = Path(cmd[cmd.index("-o") + 1])
        stem = Path(model_path).stem
        calls.append(model_path)
        _write_accuracy_report(
            path=artifact_dir / f"{stem}_accuracy_report.json",
            evaluation_pass=True,
        )
        _write_accuracy_report(
            path=artifact_dir / f"{stem}_pytorch_accuracy_report.json",
            evaluation_pass=True,
        )
        return type("CP", (), {"returncode": 0, "stdout": "ok", "stderr": ""})()

    monkeypatch.setattr(subprocess, "run", _fake_run)
    state = run_flatbuffer_direct_bulk_verification(
        root_dir=str(tmp_path),
        output_dir=str(tmp_path / "out"),
    )
    assert [entry["classification"] for entry in state["entries"]] == ["pass", "missing_model"]
    assert state["entries"][1]["strict_pass"] is False
    assert [Path(path).name for path in calls] == ["a.onnx"]
    assert Path(calls[0]).parent.parent.name == "runs"


def test_flatbuffer_direct_bulk_runner_resume_skips_completed_entries(
    tmp_path,
    monkeypatch,
) -> None:
    model_a = tmp_path / "a.onnx"
    model_b = tmp_path / "b.onnx"
    _write_dummy(model_a)
    _write_dummy(model_b)
    output_dir = tmp_path / "out"

    first_calls: List[str] = []

    def _fake_run_first(cmd, cwd=None, stdout=None, stderr=None, text=None, timeout=None):
        model_path = str(cmd[cmd.index("-i") + 1])
        artifact_dir = Path(cmd[cmd.index("-o") + 1])
        stem = Path(model_path).stem
        first_calls.append(model_path)
        _write_accuracy_report(
            path=artifact_dir / f"{stem}_accuracy_report.json",
            evaluation_pass=True,
        )
        _write_accuracy_report(
            path=artifact_dir / f"{stem}_pytorch_accuracy_report.json",
            evaluation_pass=True,
        )
        return type("CP", (), {"returncode": 0, "stdout": "ok", "stderr": ""})()

    monkeypatch.setattr(subprocess, "run", _fake_run_first)
    first_state = run_flatbuffer_direct_bulk_verification(
        root_dir=str(tmp_path),
        output_dir=str(output_dir),
    )
    assert len(first_state["entries"]) == 2
    assert len(first_calls) == 2

    second_calls: List[str] = []

    def _fake_run_second(cmd, cwd=None, stdout=None, stderr=None, text=None, timeout=None):
        second_calls.append("called")
        return type("CP", (), {"returncode": 0, "stdout": "", "stderr": ""})()

    monkeypatch.setattr(subprocess, "run", _fake_run_second)
    second_state = run_flatbuffer_direct_bulk_verification(
        root_dir=str(tmp_path),
        output_dir=str(output_dir),
        resume=True,
    )
    assert len(second_state["entries"]) == 2
    assert second_calls == []


def test_flatbuffer_direct_bulk_runner_marks_pass_only_when_both_reports_pass(
    tmp_path,
    monkeypatch,
) -> None:
    model = tmp_path / "ok.onnx"
    _write_dummy(model)

    def _fake_run(cmd, cwd=None, stdout=None, stderr=None, text=None, timeout=None):
        artifact_dir = Path(cmd[cmd.index("-o") + 1])
        stem = Path(cmd[cmd.index("-i") + 1]).stem
        _write_accuracy_report(
            path=artifact_dir / f"{stem}_accuracy_report.json",
            evaluation_pass=True,
        )
        _write_accuracy_report(
            path=artifact_dir / f"{stem}_pytorch_accuracy_report.json",
            evaluation_pass=True,
        )
        return type("CP", (), {"returncode": 0, "stdout": "ok", "stderr": ""})()

    monkeypatch.setattr(subprocess, "run", _fake_run)
    state = run_flatbuffer_direct_bulk_verification(
        root_dir=str(tmp_path),
        output_dir=str(tmp_path / "out"),
    )
    entry = state["entries"][0]
    assert entry["classification"] == "pass"
    assert entry["strict_pass"] is True


def test_flatbuffer_direct_bulk_runner_records_stable_failure_signature_and_timing(
    tmp_path,
    monkeypatch,
) -> None:
    model = tmp_path / "broken.onnx"
    _write_dummy(model)

    def _fake_run(cmd, cwd=None, stdout=None, stderr=None, text=None, timeout=None):
        return type(
            "CP",
            (),
            {
                "returncode": 1,
                "stdout": "",
                "stderr": f"Traceback\nValueError: failed in {cwd}/temporary.bin",
            },
        )()

    monkeypatch.setattr(subprocess, "run", _fake_run)
    state = run_flatbuffer_direct_bulk_verification(
        root_dir=str(tmp_path),
        output_dir=str(tmp_path / "out"),
        min_nodes=None,
        max_nodes=None,
        include_pytorch_artifacts=False,
        recursive=False,
    )
    entry = state["entries"][0]
    assert entry["classification"] == "conversion_error"
    assert "ValueError: failed in <PATH>/temporary.bin" in entry["error_signature"]
    assert len(entry["error_signature_sha256"]) == 64

    summary = state["summary"]
    assert summary["filters"] == {
        "min_nodes": None,
        "max_nodes": None,
        "recursive": False,
        "include_pytorch_artifacts": False,
    }
    assert summary["timing"]["total_duration_sec"] >= 0.0
    assert summary["failed_models"][0]["error_signature_sha256"] == entry[
        "error_signature_sha256"
    ]


def test_flatbuffer_direct_bulk_runner_filters_node_count_tier(
    tmp_path,
    monkeypatch,
) -> None:
    from onnx import TensorProto, helper, save

    def _model(path, count):
        x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1])
        previous = "x"
        nodes = []
        for index in range(count):
            output = f"v{index}"
            nodes.append(helper.make_node("Relu", [previous], [output]))
            previous = output
        y = helper.make_tensor_value_info(previous, TensorProto.FLOAT, [1])
        save(helper.make_model(helper.make_graph(nodes, "g", [x], [y])), path)

    _model(tmp_path / "small.onnx", 2)
    _model(tmp_path / "medium.onnx", 55)
    calls = []

    def _fake_run(cmd, cwd=None, stdout=None, stderr=None, text=None, timeout=None):
        calls.append(cmd)
        artifact_dir = Path(cmd[cmd.index("-o") + 1])
        stem = Path(cmd[cmd.index("-i") + 1]).stem
        _write_accuracy_report(
            path=artifact_dir / f"{stem}_accuracy_report.json",
            evaluation_pass=True,
        )
        _write_accuracy_report(
            path=artifact_dir / f"{stem}_pytorch_accuracy_report.json",
            evaluation_pass=True,
        )
        return type("CP", (), {"returncode": 0, "stdout": "", "stderr": ""})()

    monkeypatch.setattr(subprocess, "run", _fake_run)
    state = run_flatbuffer_direct_bulk_verification(
        root_dir=str(tmp_path),
        output_dir=str(tmp_path / "out"),
        min_nodes=50,
        max_nodes=199,
    )
    assert [entry["model"] for entry in state["entries"]] == ["medium.onnx"]
    assert len(calls) == 1


def test_flatbuffer_direct_bulk_runner_enforces_max_abs_one_tenth(
    tmp_path,
    monkeypatch,
) -> None:
    model = tmp_path / "accuracy.onnx"
    _write_dummy(model)

    def _fake_run(cmd, cwd=None, stdout=None, stderr=None, text=None, timeout=None):
        artifact_dir = Path(cmd[cmd.index("-o") + 1])
        stem = Path(cmd[cmd.index("-i") + 1]).stem
        for suffix, max_abs in [("accuracy_report", 0.10001), ("pytorch_accuracy_report", 0.01)]:
            path = artifact_dir / f"{stem}_{suffix}.json"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(
                json.dumps({"evaluation_pass": True, "overall_metrics": {"max_abs": max_abs}}),
                encoding="utf-8",
            )
        return type("CP", (), {"returncode": 0, "stdout": "", "stderr": ""})()

    monkeypatch.setattr(subprocess, "run", _fake_run)
    state = run_flatbuffer_direct_bulk_verification(
        root_dir=str(tmp_path),
        output_dir=str(tmp_path / "out"),
    )
    entry = state["entries"][0]
    assert entry["classification"] == "tflite_fail"
    assert entry["tflite_max_abs"] == 0.10001


def test_flatbuffer_direct_bulk_runner_tflite_only_does_not_require_pytorch(
    tmp_path,
    monkeypatch,
) -> None:
    model = tmp_path / "core.onnx"
    _write_dummy(model)
    commands = []

    def _fake_run(cmd, cwd=None, stdout=None, stderr=None, text=None, timeout=None):
        commands.append(cmd)
        artifact_dir = Path(cmd[cmd.index("-o") + 1])
        stem = Path(cmd[cmd.index("-i") + 1]).stem
        _write_accuracy_report(
            path=artifact_dir / f"{stem}_accuracy_report.json",
            evaluation_pass=True,
        )
        return type("CP", (), {"returncode": 0, "stdout": "", "stderr": ""})()

    monkeypatch.setattr(subprocess, "run", _fake_run)
    state = run_flatbuffer_direct_bulk_verification(
        root_dir=str(tmp_path),
        output_dir=str(tmp_path / "out"),
        include_pytorch_artifacts=False,
    )
    assert state["entries"][0]["classification"] == "pass"
    assert "-fdopt" not in commands[0]


def test_flatbuffer_direct_bulk_runner_stages_model_before_conversion(
    tmp_path,
    monkeypatch,
) -> None:
    model = tmp_path / "immutable.onnx"
    original = b"original corpus bytes"
    model.write_bytes(original)

    def _fake_run(cmd, cwd=None, stdout=None, stderr=None, text=None, timeout=None):
        staged = Path(cmd[cmd.index("-i") + 1])
        assert staged != model
        assert staged.parent == Path(cwd)
        staged.write_bytes(b"converter mutation")
        artifact_dir = Path(cmd[cmd.index("-o") + 1])
        _write_accuracy_report(
            path=artifact_dir / "immutable_accuracy_report.json",
            evaluation_pass=True,
        )
        return type("CP", (), {"returncode": 0, "stdout": "", "stderr": ""})()

    monkeypatch.setattr(subprocess, "run", _fake_run)
    state = run_flatbuffer_direct_bulk_verification(
        root_dir=str(tmp_path),
        output_dir=str(tmp_path / "out"),
        include_pytorch_artifacts=False,
    )
    assert state["entries"][0]["classification"] == "pass"
    assert model.read_bytes() == original


def test_flatbuffer_direct_bulk_runner_root_only_excludes_nested_models(
    tmp_path,
    monkeypatch,
) -> None:
    root_model = tmp_path / "root.onnx"
    nested_model = tmp_path / "nested" / "nested.onnx"
    _write_dummy(root_model)
    _write_dummy(nested_model)

    def _fake_run(cmd, cwd=None, stdout=None, stderr=None, text=None, timeout=None):
        artifact_dir = Path(cmd[cmd.index("-o") + 1])
        _write_accuracy_report(
            path=artifact_dir / "root_accuracy_report.json",
            evaluation_pass=True,
        )
        return type("CP", (), {"returncode": 0, "stdout": "", "stderr": ""})()

    monkeypatch.setattr(subprocess, "run", _fake_run)
    state = run_flatbuffer_direct_bulk_verification(
        root_dir=str(tmp_path),
        output_dir=str(tmp_path / "out"),
        include_pytorch_artifacts=False,
        recursive=False,
    )
    assert [entry["model"] for entry in state["entries"]] == ["root.onnx"]


def test_flatbuffer_direct_bulk_runner_marks_tflite_failure(
    tmp_path,
    monkeypatch,
) -> None:
    model = tmp_path / "tflite_fail.onnx"
    _write_dummy(model)

    def _fake_run(cmd, cwd=None, stdout=None, stderr=None, text=None, timeout=None):
        artifact_dir = Path(cmd[cmd.index("-o") + 1])
        stem = Path(cmd[cmd.index("-i") + 1]).stem
        _write_accuracy_report(
            path=artifact_dir / f"{stem}_accuracy_report.json",
            evaluation_pass=False,
        )
        _write_accuracy_report(
            path=artifact_dir / f"{stem}_pytorch_accuracy_report.json",
            evaluation_pass=True,
        )
        return type("CP", (), {"returncode": 0, "stdout": "ok", "stderr": ""})()

    monkeypatch.setattr(subprocess, "run", _fake_run)
    state = run_flatbuffer_direct_bulk_verification(
        root_dir=str(tmp_path),
        output_dir=str(tmp_path / "out"),
    )
    entry = state["entries"][0]
    assert entry["classification"] == "tflite_fail"
    assert entry["strict_pass"] is False


def test_flatbuffer_direct_bulk_runner_marks_pytorch_failure(
    tmp_path,
    monkeypatch,
) -> None:
    model = tmp_path / "pytorch_fail.onnx"
    _write_dummy(model)

    def _fake_run(cmd, cwd=None, stdout=None, stderr=None, text=None, timeout=None):
        artifact_dir = Path(cmd[cmd.index("-o") + 1])
        stem = Path(cmd[cmd.index("-i") + 1]).stem
        _write_accuracy_report(
            path=artifact_dir / f"{stem}_accuracy_report.json",
            evaluation_pass=True,
        )
        _write_accuracy_report(
            path=artifact_dir / f"{stem}_pytorch_accuracy_report.json",
            evaluation_pass=False,
        )
        return type("CP", (), {"returncode": 0, "stdout": "ok", "stderr": ""})()

    monkeypatch.setattr(subprocess, "run", _fake_run)
    state = run_flatbuffer_direct_bulk_verification(
        root_dir=str(tmp_path),
        output_dir=str(tmp_path / "out"),
    )
    entry = state["entries"][0]
    assert entry["classification"] == "pytorch_fail"
    assert entry["strict_pass"] is False


def test_flatbuffer_direct_bulk_runner_marks_both_failures(
    tmp_path,
    monkeypatch,
) -> None:
    model = tmp_path / "both_fail.onnx"
    _write_dummy(model)

    def _fake_run(cmd, cwd=None, stdout=None, stderr=None, text=None, timeout=None):
        artifact_dir = Path(cmd[cmd.index("-o") + 1])
        stem = Path(cmd[cmd.index("-i") + 1]).stem
        _write_accuracy_report(
            path=artifact_dir / f"{stem}_accuracy_report.json",
            evaluation_pass=False,
        )
        _write_accuracy_report(
            path=artifact_dir / f"{stem}_pytorch_accuracy_report.json",
            evaluation_pass=False,
        )
        return type("CP", (), {"returncode": 0, "stdout": "ok", "stderr": ""})()

    monkeypatch.setattr(subprocess, "run", _fake_run)
    state = run_flatbuffer_direct_bulk_verification(
        root_dir=str(tmp_path),
        output_dir=str(tmp_path / "out"),
    )
    entry = state["entries"][0]
    assert entry["classification"] == "both_fail"
    assert entry["strict_pass"] is False


def test_flatbuffer_direct_bulk_runner_marks_missing_single_report(
    tmp_path,
    monkeypatch,
) -> None:
    model = tmp_path / "missing_report.onnx"
    _write_dummy(model)

    def _fake_run(cmd, cwd=None, stdout=None, stderr=None, text=None, timeout=None):
        artifact_dir = Path(cmd[cmd.index("-o") + 1])
        stem = Path(cmd[cmd.index("-i") + 1]).stem
        _write_accuracy_report(
            path=artifact_dir / f"{stem}_accuracy_report.json",
            evaluation_pass=True,
        )
        return type("CP", (), {"returncode": 0, "stdout": "ok", "stderr": ""})()

    monkeypatch.setattr(subprocess, "run", _fake_run)
    state = run_flatbuffer_direct_bulk_verification(
        root_dir=str(tmp_path),
        output_dir=str(tmp_path / "out"),
    )
    entry = state["entries"][0]
    assert entry["classification"] == "missing_pytorch_report"
    assert entry["strict_pass"] is False


def test_flatbuffer_direct_bulk_runner_marks_timeout(
    tmp_path,
    monkeypatch,
) -> None:
    model = tmp_path / "slow.onnx"
    _write_dummy(model)

    def _fake_run(cmd, cwd=None, stdout=None, stderr=None, text=None, timeout=None):
        raise subprocess.TimeoutExpired(cmd=cmd, timeout=timeout)

    monkeypatch.setattr(subprocess, "run", _fake_run)
    state = run_flatbuffer_direct_bulk_verification(
        root_dir=str(tmp_path),
        output_dir=str(tmp_path / "out"),
        timeout_sec=1,
    )
    entry = state["entries"][0]
    assert entry["classification"] == "timeout"
    assert entry["strict_pass"] is False


def test_flatbuffer_direct_bulk_runner_marks_conversion_error(
    tmp_path,
    monkeypatch,
) -> None:
    model = tmp_path / "error.onnx"
    _write_dummy(model)

    def _fake_run(cmd, cwd=None, stdout=None, stderr=None, text=None, timeout=None):
        return type("CP", (), {"returncode": 1, "stdout": "", "stderr": "failed"})()

    monkeypatch.setattr(subprocess, "run", _fake_run)
    state = run_flatbuffer_direct_bulk_verification(
        root_dir=str(tmp_path),
        output_dir=str(tmp_path / "out"),
    )
    entry = state["entries"][0]
    assert entry["classification"] == "conversion_error"
    assert entry["strict_pass"] is False


def test_flatbuffer_direct_bulk_runner_passes_native_pytorch_generation_timeout(
    tmp_path,
    monkeypatch,
) -> None:
    model = tmp_path / "ok.onnx"
    _write_dummy(model)
    seen_cmds: List[List[str]] = []

    def _fake_run(cmd, cwd=None, stdout=None, stderr=None, text=None, timeout=None):
        seen_cmds.append(list(cmd))
        artifact_dir = Path(cmd[cmd.index("-o") + 1])
        stem = Path(cmd[cmd.index("-i") + 1]).stem
        _write_accuracy_report(
            path=artifact_dir / f"{stem}_accuracy_report.json",
            evaluation_pass=True,
        )
        _write_accuracy_report(
            path=artifact_dir / f"{stem}_pytorch_accuracy_report.json",
            evaluation_pass=True,
        )
        return type("CP", (), {"returncode": 0, "stdout": "ok", "stderr": ""})()

    monkeypatch.setattr(subprocess, "run", _fake_run)
    run_flatbuffer_direct_bulk_verification(
        root_dir=str(tmp_path),
        output_dir=str(tmp_path / "out"),
        native_pytorch_generation_timeout_sec=37,
    )
    assert len(seen_cmds) == 1
    assert "--native_pytorch_generation_timeout_sec" in seen_cmds[0]
    timeout_index = seen_cmds[0].index("--native_pytorch_generation_timeout_sec")
    assert seen_cmds[0][timeout_index + 1] == "37"


def test_flatbuffer_direct_bulk_runner_updates_tqdm_progress(
    tmp_path,
    monkeypatch,
) -> None:
    model_a = tmp_path / "a.onnx"
    model_b = tmp_path / "b.onnx"
    _write_dummy(model_a)
    _write_dummy(model_b)

    class _FakeProgressBar:
        def __init__(self) -> None:
            self.postfixes: List[str] = []
            self.updates: List[int] = []
            self.closed = False

        def set_postfix_str(self, text, refresh=True):
            self.postfixes.append(str(text))

        def update(self, value):
            self.updates.append(int(value))

        def close(self):
            self.closed = True

    created_progress_bars: List[_FakeProgressBar] = []

    def _fake_create_progress_bar(*, total, initial=0, desc):
        assert total == 2
        assert initial == 0
        assert desc == "flatbuffer_direct bulk"
        bar = _FakeProgressBar()
        created_progress_bars.append(bar)
        return bar

    def _fake_run(cmd, cwd=None, stdout=None, stderr=None, text=None, timeout=None):
        artifact_dir = Path(cmd[cmd.index("-o") + 1])
        stem = Path(cmd[cmd.index("-i") + 1]).stem
        _write_accuracy_report(
            path=artifact_dir / f"{stem}_accuracy_report.json",
            evaluation_pass=True,
        )
        _write_accuracy_report(
            path=artifact_dir / f"{stem}_pytorch_accuracy_report.json",
            evaluation_pass=True,
        )
        return type("CP", (), {"returncode": 0, "stdout": "ok", "stderr": ""})()

    monkeypatch.setattr(
        "onnx2tf.utils.flatbuffer_direct_bulk_runner._create_progress_bar",
        _fake_create_progress_bar,
    )
    monkeypatch.setattr(subprocess, "run", _fake_run)
    run_flatbuffer_direct_bulk_verification(
        root_dir=str(tmp_path),
        output_dir=str(tmp_path / "out"),
    )
    assert len(created_progress_bars) == 1
    assert created_progress_bars[0].updates == [1, 1]
    assert created_progress_bars[0].closed is True
    assert created_progress_bars[0].postfixes[-1].endswith("[pass]")


def test_flatbuffer_direct_bulk_runner_starts_spinner_while_running(
    tmp_path,
    monkeypatch,
) -> None:
    model = tmp_path / "spin.onnx"
    _write_dummy(model)

    class _FakeSpinner:
        instances: List["_FakeSpinner"] = []

        def __init__(self, progress_bar) -> None:
            self.progress_bar = progress_bar
            self.contexts: List[tuple[str, str]] = []
            self.starts = 0
            self.stops = 0
            _FakeSpinner.instances.append(self)

        def set_context(self, *, model_name: str, status: str = "running") -> None:
            self.contexts.append((str(model_name), str(status)))

        def start(self) -> None:
            self.starts += 1

        def stop(self) -> None:
            self.stops += 1

    def _fake_run(cmd, cwd=None, stdout=None, stderr=None, text=None, timeout=None):
        artifact_dir = Path(cmd[cmd.index("-o") + 1])
        stem = Path(cmd[cmd.index("-i") + 1]).stem
        _write_accuracy_report(
            path=artifact_dir / f"{stem}_accuracy_report.json",
            evaluation_pass=True,
        )
        _write_accuracy_report(
            path=artifact_dir / f"{stem}_pytorch_accuracy_report.json",
            evaluation_pass=True,
        )
        return type("CP", (), {"returncode": 0, "stdout": "ok", "stderr": ""})()

    monkeypatch.setattr(bulk_runner, "_ProgressSpinner", _FakeSpinner)
    monkeypatch.setattr(subprocess, "run", _fake_run)
    run_flatbuffer_direct_bulk_verification(
        root_dir=str(tmp_path),
        output_dir=str(tmp_path / "out"),
    )
    assert len(_FakeSpinner.instances) == 1
    spinner = _FakeSpinner.instances[0]
    assert spinner.contexts == [("spin.onnx", "running")]
    assert spinner.starts == 1
    assert spinner.stops >= 2


def test_flatbuffer_direct_bulk_runner_summary_lists_failed_models_only(
    tmp_path,
    monkeypatch,
) -> None:
    ok_model = tmp_path / "ok.onnx"
    fail_model = tmp_path / "fail.onnx"
    _write_dummy(ok_model)
    _write_dummy(fail_model)

    def _fake_run(cmd, cwd=None, stdout=None, stderr=None, text=None, timeout=None):
        model_path = Path(cmd[cmd.index("-i") + 1])
        artifact_dir = Path(cmd[cmd.index("-o") + 1])
        if model_path.stem == "ok":
            _write_accuracy_report(
                path=artifact_dir / "ok_accuracy_report.json",
                evaluation_pass=True,
            )
            _write_accuracy_report(
                path=artifact_dir / "ok_pytorch_accuracy_report.json",
                evaluation_pass=True,
            )
        else:
            _write_accuracy_report(
                path=artifact_dir / "fail_accuracy_report.json",
                evaluation_pass=False,
            )
            _write_accuracy_report(
                path=artifact_dir / "fail_pytorch_accuracy_report.json",
                evaluation_pass=True,
            )
        return type("CP", (), {"returncode": 0, "stdout": "ok", "stderr": ""})()

    monkeypatch.setattr(subprocess, "run", _fake_run)
    state = run_flatbuffer_direct_bulk_verification(
        root_dir=str(tmp_path),
        output_dir=str(tmp_path / "out"),
    )
    failed_models = state["summary"]["failed_models"]
    assert len(failed_models) == 1
    assert failed_models[0]["model"] == "fail.onnx"
    assert failed_models[0]["classification"] == "tflite_fail"


def test_regression_profile_runs_recorded_tier_zero_to_three_successes_and_failures(
    tmp_path,
    monkeypatch,
) -> None:
    from onnx import TensorProto, helper, save

    def _model(path: Path, count: int) -> None:
        x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1])
        previous = "x"
        nodes = []
        for index in range(count):
            output = f"v{index}"
            nodes.append(helper.make_node("Relu", [previous], [output]))
            previous = output
        y = helper.make_tensor_value_info(previous, TensorProto.FLOAT, [1])
        save(helper.make_model(helper.make_graph(nodes, "g", [x], [y])), path)

    _model(tmp_path / "baseline_pass.onnx", 2)
    _model(tmp_path / "known_failure.onnx", 2)
    profile_path = tmp_path / "profile.json"
    profile_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "name": "test-tier0-3-all-models",
                "min_nodes": 1,
                "max_nodes": 999,
                "models": [
                    {
                        "tier": 0,
                        "model": "baseline_pass.onnx",
                        "baseline_classification": "pass",
                    },
                    {
                        "tier": 0,
                        "model": "known_failure.onnx",
                        "baseline_classification": "conversion_error",
                    },
                ],
            }
        ),
        encoding="utf-8",
    )
    calls: List[str] = []

    def _fake_run(cmd, cwd=None, stdout=None, stderr=None, text=None, timeout=None):
        model_path = Path(cmd[cmd.index("-i") + 1])
        artifact_dir = Path(cmd[cmd.index("-o") + 1])
        calls.append(model_path.name)
        _write_accuracy_report(
            path=artifact_dir / f"{model_path.stem}_accuracy_report.json",
            evaluation_pass=True,
        )
        return type("CP", (), {"returncode": 0, "stdout": "ok", "stderr": ""})()

    monkeypatch.setattr(subprocess, "run", _fake_run)
    state = run_flatbuffer_direct_bulk_verification(
        root_dir=str(tmp_path),
        output_dir=str(tmp_path / "out"),
        include_pytorch_artifacts=False,
        regression_profile=str(profile_path),
    )

    assert calls == ["baseline_pass.onnx", "known_failure.onnx"]
    assert [entry["model"] for entry in state["entries"]] == [
        "baseline_pass.onnx",
        "known_failure.onnx",
    ]
    profile_filter = state["summary"]["filters"]["regression_profile"]
    assert profile_filter["model_count"] == 2
    assert profile_filter["tiers"] == [0]
    assert profile_filter["baseline_classification_counts"] == {
        "conversion_error": 1,
        "pass": 1,
    }
    assert "model_names" not in profile_filter
    assert state["summary"]["filters"]["max_nodes"] == 999
    assert state["summary"]["filters"]["recursive"] is False


def test_regression_profile_excludes_recorded_timeouts_from_future_runs(
    tmp_path,
    monkeypatch,
) -> None:
    from onnx import TensorProto, helper, save

    def _model(path: Path) -> None:
        x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1])
        y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1])
        node = helper.make_node("Relu", ["x"], ["y"])
        save(helper.make_model(helper.make_graph([node], "g", [x], [y])), path)

    _model(tmp_path / "active.onnx")
    _model(tmp_path / "timed_out.onnx")
    profile_path = tmp_path / "profile.json"
    profile_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "min_nodes": 1,
                "max_nodes": 49,
                "models": [
                    {
                        "tier": 0,
                        "model": "active.onnx",
                        "baseline_classification": "pass",
                        "shape_hints": ["input:0:1,16,16,3"],
                        "overwrite_input_shape": ["image:1,3,16,16"],
                        "keep_shape_absolutely_input_names": ["state:0"],
                    },
                    {
                        "tier": 0,
                        "model": "timed_out.onnx",
                        "baseline_classification": "timeout",
                    },
                ],
            }
        ),
        encoding="utf-8",
    )
    calls: List[str] = []
    commands: List[List[str]] = []

    def _fake_run(cmd, cwd=None, stdout=None, stderr=None, text=None, timeout=None):
        model_path = Path(cmd[cmd.index("-i") + 1])
        artifact_dir = Path(cmd[cmd.index("-o") + 1])
        calls.append(model_path.name)
        commands.append(list(cmd))
        _write_accuracy_report(
            path=artifact_dir / f"{model_path.stem}_accuracy_report.json",
            evaluation_pass=True,
        )
        return type("CP", (), {"returncode": 0, "stdout": "ok", "stderr": ""})()

    monkeypatch.setattr(subprocess, "run", _fake_run)
    state = run_flatbuffer_direct_bulk_verification(
        root_dir=str(tmp_path),
        output_dir=str(tmp_path / "out"),
        include_pytorch_artifacts=False,
        regression_profile=str(profile_path),
    )

    assert calls == ["active.onnx"]
    assert commands[0][commands[0].index("-sh") + 1] == "input:0:1,16,16,3"
    assert commands[0][commands[0].index("-ois") + 1] == "image:1,3,16,16"
    assert commands[0][commands[0].index("-kat") + 1] == "state:0"
    assert [entry["model"] for entry in state["entries"]] == ["active.onnx"]
    profile_filter = state["summary"]["filters"]["regression_profile"]
    assert profile_filter["model_count"] == 2
    assert profile_filter["active_model_count"] == 1
    assert profile_filter["excluded_model_count"] == 1
    assert profile_filter["excluded_baseline_classification_counts"] == {
        "timeout": 1,
    }


def test_regression_profile_accepts_tier_four_models(tmp_path) -> None:
    profile_path = tmp_path / "profile.json"
    profile_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "min_nodes": 1000,
                "max_nodes": 1999,
                "models": [{"tier": 4, "model": "large.onnx"}],
            }
        ),
        encoding="utf-8",
    )

    profile = bulk_runner._load_regression_profile(str(profile_path))
    assert profile["tiers"] == [4]
    assert profile["max_nodes"] == 1999


def test_regression_profile_rejects_tier_five_models(tmp_path) -> None:
    profile_path = tmp_path / "profile.json"
    profile_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "min_nodes": 2000,
                "max_nodes": 4000,
                "models": [{"tier": 5, "model": "huge.onnx"}],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="only Tier 0-4 models"):
        bulk_runner._load_regression_profile(str(profile_path))


def test_managed_regression_profile_includes_all_tier_zero_to_four_models() -> None:
    profile_path = (
        Path(__file__).resolve().parents[1]
        / "docs"
        / "baselines"
        / "flatbuffer_direct_active_tier0_4.json"
    )
    profile = bulk_runner._load_regression_profile(str(profile_path))

    assert profile["model_count"] == 420
    assert profile["active_model_count"] == 394
    assert profile["excluded_model_count"] == 26
    assert profile["excluded_baseline_classification_counts"] == {"timeout": 26}
    assert profile["tiers"] == [0, 1, 2, 3, 4]
    assert profile["min_nodes"] == 1
    assert profile["max_nodes"] == 1999
    assert profile["baseline_classification_counts"] == {
        "missing_tflite_report": 14,
        "pass": 348,
        "tflite_fail": 32,
        "timeout": 26,
    }
    assert profile["model_options"]["silero_vad.onnx"] == {
        "keep_shape_absolutely_input_names": ["input", "state", "sr"],
    }
    profile_payload = json.loads(profile_path.read_text(encoding="utf-8"))
    inverse_entry = next(
        entry
        for entry in profile_payload["models"]
        if entry["model"] == "inverse_11.onnx"
    )
    assert inverse_entry["baseline_classification"] == "missing_tflite_report"
    assert inverse_entry["baseline_reason"] == (
        "unsupported_exact_inverse_matrix_size_224"
    )
    assert inverse_entry["error_signature_sha256"] == (
        "c1bdaa58f8b1dda9b86c24ce66f509320a6201e08f6ebfd35b4ecb058109d885"
    )
    string_normalizer_entry = next(
        entry
        for entry in profile_payload["models"]
        if entry["model"] == "string_normalizer_11.onnx"
    )
    assert string_normalizer_entry["baseline_classification"] == (
        "missing_tflite_report"
    )
    assert string_normalizer_entry["baseline_reason"] == (
        "unsupported_stock_tflite_string_normalizer"
    )
    assert string_normalizer_entry["error_signature_sha256"] == (
        "2623d1f687770e4aade4749af32963fca446f80b3dd131e7061bcb13ae225722"
    )
    ssd_mobilenet_entry = next(
        entry
        for entry in profile_payload["models"]
        if entry["model"] == "ssd_mobilenet_v1_12-int8.onnx"
    )
    assert ssd_mobilenet_entry["baseline_classification"] == (
        "missing_tflite_report"
    )
    assert ssd_mobilenet_entry["baseline_reason"] == (
        "invalid_onnx_missing_loop_captures_186"
    )
    assert ssd_mobilenet_entry["error_signature_sha256"] == (
        "a2db2a961bcee7b5af536f985dff9ceee1ed8ef8fc8115f8a57dbf64b5f1fe26"
    )
    assert profile["model_options"]["ssd_mobilenet_v1_12-int8.onnx"] == {
        "overwrite_input_shape": ["inputs:1,300,300,3"],
        "keep_shape_absolutely_input_names": ["inputs"],
    }
    efficientnet_lite4_entry = next(
        entry
        for entry in profile_payload["models"]
        if entry["model"] == "efficientnet-lite4-11-int8.onnx"
    )
    assert efficientnet_lite4_entry["baseline_classification"] == "tflite_fail"
    assert efficientnet_lite4_entry["baseline_reason"] == (
        "onnxruntime_u8s8_saturating_pair_accumulation"
    )
    assert efficientnet_lite4_entry["error_signature_sha256"] == (
        "da92ac2e2f0e69a6b68654c2152fadfadb079a20d73aaf8b33fc3a81af4afc15"
    )
    arcfaceresnet_entry = next(
        entry
        for entry in profile_payload["models"]
        if entry["model"] == "arcfaceresnet100-11-int8.onnx"
    )
    assert arcfaceresnet_entry["baseline_classification"] == "tflite_fail"
    assert arcfaceresnet_entry["baseline_reason"] == (
        "onnxruntime_u8s8_saturating_pair_accumulation"
    )
    assert arcfaceresnet_entry["error_signature_sha256"] == (
        "f1495fb3c58cf0f48521281b92113fa31d9dd301329f552c0cd80dcbc1687c3a"
    )
    assert arcfaceresnet_entry["tflite_max_abs"] == 0.3681950643658638
    dynamics_rife_entry = next(
        entry
        for entry in profile_payload["models"]
        if entry["model"] == "dynamics_rife_sim.onnx"
    )
    assert dynamics_rife_entry["baseline_classification"] == (
        "missing_tflite_report"
    )
    assert dynamics_rife_entry["baseline_reason"] == (
        "invalid_onnx_concat_spatial_mismatch_64_128"
    )
    assert dynamics_rife_entry["error_signature_sha256"] == (
        "603ca474b8eee210cb3bb2df39bb20791becc95c66d60a1ec68e1a8a2744c109"
    )
    yolov3_entry = next(
        entry
        for entry in profile_payload["models"]
        if entry["model"] == "yolov3-12-int8.onnx"
    )
    assert yolov3_entry["baseline_classification"] == "tflite_fail"
    assert yolov3_entry["baseline_reason"] == (
        "u8s8_detector_strict_metric_mismatch"
    )
    assert yolov3_entry["error_signature_sha256"] == (
        "042bc0ec6020004d0f33640af6ad3197daf7d2e012c2306e64ef42a999b29de9"
    )
    assert yolov3_entry["tflite_max_abs"] == 0.09563881158828735
    assert profile["model_options"]["yolov3-12-int8.onnx"] == {
        "overwrite_input_shape": [
            "input_1:1,3,416,416",
            "image_shape:1,2",
        ],
    }
    nanodet_entry = next(
        entry
        for entry in profile_payload["models"]
        if entry["model"] == "object_detection_nanodet_2022nov_int8.onnx"
    )
    assert nanodet_entry == {
        "tier": 2,
        "model": "object_detection_nanodet_2022nov_int8.onnx",
        "baseline_classification": "pass",
        "tflite_max_abs": 4.76837158203125e-07,
    }
    yolox_int8_entry = next(
        entry
        for entry in profile_payload["models"]
        if entry["model"] == "object_detection_yolox_2022nov_int8.onnx"
    )
    assert yolox_int8_entry == {
        "tier": 2,
        "model": "object_detection_yolox_2022nov_int8.onnx",
        "baseline_classification": "pass",
        "tflite_max_abs": 2.384185791015625e-07,
    }
    ppocrv3_entry = next(
        entry
        for entry in profile_payload["models"]
        if entry["model"] == "text_detection_en_ppocrv3_2023may_int8.onnx"
    )
    assert ppocrv3_entry["baseline_classification"] == "tflite_fail"
    assert ppocrv3_entry["baseline_reason"] == (
        "int8_requantization_outliers_amplified_by_transpose_conv"
    )
    assert ppocrv3_entry["error_signature_sha256"] == (
        "70b5455f472dbbc6067111117087dfd042b249969d88555128fe4aaf1ae9c64a"
    )
    assert ppocrv3_entry["overwrite_input_shape"] == ["x:1,3,480,640"]
    assert ppocrv3_entry["tflite_max_abs"] == 0.7411765307188034
    yolov5s_entry = next(
        entry
        for entry in profile_payload["models"]
        if entry["model"] == "yolov5s.onnx"
    )
    assert yolov5s_entry["baseline_classification"] == "tflite_fail"
    assert yolov5s_entry["baseline_reason"] == (
        "float16_decode_rounding_boundary"
    )
    assert yolov5s_entry["error_signature_sha256"] == (
        "3d19e30742aaefcf03909adb23b0cfcbe24641b67bd5183d4c50f5998d8cb8ed"
    )
    assert yolov5s_entry["tflite_max_abs"] == 0.32965087890625
    dequantize_linear_entry = next(
        entry
        for entry in profile_payload["models"]
        if entry["model"] == "dequantize_linear.onnx"
    )
    assert dequantize_linear_entry["baseline_classification"] == "tflite_fail"
    assert dequantize_linear_entry["baseline_reason"] == (
        "onnxruntime_qdq_fusion_and_float_conv_decode_amplification"
    )
    assert dequantize_linear_entry["error_signature_sha256"] == (
        "aca9525029ba7a608c692da1d3944c7ba88e2c55d50b7b54a6bc35b237aa2a42"
    )
    assert dequantize_linear_entry["tflite_max_abs"] == 81.25048828125
    conv_tasnet_entry = next(
        entry
        for entry in profile_payload["models"]
        if entry["model"] == "conv_tasnet.onnx"
    )
    assert conv_tasnet_entry["baseline_classification"] == (
        "missing_tflite_report"
    )
    assert conv_tasnet_entry["baseline_reason"] == (
        "invalid_onnx_scatterelements_rank_mismatch_4_6"
    )
    assert conv_tasnet_entry["error_signature_sha256"] == (
        "1a12a88b8f42185e545973a18cc67b274f6b1c7a01b21319ffeaeecf4fdb6052"
    )
    rtdetrv4_entry = next(
        entry
        for entry in profile_payload["models"]
        if entry["model"] == "rtdetrv4_s.onnx"
    )
    assert rtdetrv4_entry["baseline_classification"] == "tflite_fail"
    assert rtdetrv4_entry["baseline_reason"] == (
        "builtin_conv_accumulation_amplified_by_topk"
    )
    assert rtdetrv4_entry["error_signature_sha256"] == (
        "743900046d8e38c14684c26f9df0c0fcd60a95f014698a135d2162ef77b8ae05"
    )
    assert rtdetrv4_entry["tflite_max_abs"] == 79.0
    yolo11x_obb_entry = next(
        entry
        for entry in profile_payload["models"]
        if entry["model"] == "yolo11x-obb.onnx"
    )
    assert yolo11x_obb_entry == {
        "tier": 3,
        "model": "yolo11x-obb.onnx",
        "baseline_classification": "pass",
        "tflite_max_abs": 6.103515625e-05,
        "overwrite_input_shape": ["input_image:1,3,1024,1024"],
    }
    assert profile["model_options"]["yolo11x-obb.onnx"] == {
        "overwrite_input_shape": ["input_image:1,3,1024,1024"],
    }
    yolov9_n_entry = next(
        entry
        for entry in profile_payload["models"]
        if entry["model"] == "yolov9_n_wholebody15_Nx3HxW.onnx"
    )
    assert yolov9_n_entry == {
        "tier": 3,
        "model": "yolov9_n_wholebody15_Nx3HxW.onnx",
        "baseline_classification": "pass",
        "tflite_max_abs": 0.0008544921875,
        "overwrite_input_shape": ["images:1,3,640,640"],
    }
    assert profile["model_options"]["yolov9_n_wholebody15_Nx3HxW.onnx"] == {
        "overwrite_input_shape": ["images:1,3,640,640"],
    }
    yolov9_t_entry = next(
        entry
        for entry in profile_payload["models"]
        if entry["model"] == "yolov9_t_wholebody28_Nx3HxW.onnx"
    )
    assert yolov9_t_entry == {
        "tier": 3,
        "model": "yolov9_t_wholebody28_Nx3HxW.onnx",
        "baseline_classification": "pass",
        "tflite_max_abs": 0.0081634521484375,
        "overwrite_input_shape": ["images:1,3,640,640"],
    }
    assert profile["model_options"]["yolov9_t_wholebody28_Nx3HxW.onnx"] == {
        "overwrite_input_shape": ["images:1,3,640,640"],
    }
    version_rfb_entry = next(
        entry
        for entry in profile_payload["models"]
        if entry["model"] == "version-RFB-320-int8.onnx"
    )
    assert version_rfb_entry["baseline_classification"] == "tflite_fail"
    assert version_rfb_entry["baseline_reason"] == (
        "onnxruntime_u8s8_saturating_pair_accumulation"
    )
    assert version_rfb_entry["error_signature_sha256"] == (
        "da92ac2e2f0e69a6b68654c2152fadfadb079a20d73aaf8b33fc3a81af4afc15"
    )
    assert version_rfb_entry["tflite_max_abs"] == 0.14972958900034428
    crnn_entry = next(
        entry
        for entry in profile_payload["models"]
        if entry["model"] == "text_recognition_CRNN_CN_2021nov_int8.onnx"
    )
    assert crnn_entry["baseline_classification"] == "tflite_fail"
    assert crnn_entry["baseline_reason"] == (
        "qlinear_matmul_single_quantum_outlier"
    )
    assert crnn_entry["error_signature_sha256"] == (
        "42f6b758e04d423511002b64e18281c40f1c2fe6eb72f20065608fb10bca90a1"
    )
    assert crnn_entry["tflite_max_abs"] == 0.14842605590820312
    fcn_resnet_entry = next(
        entry
        for entry in profile_payload["models"]
        if entry["model"] == "fcn-resnet50-12-int8.onnx"
    )
    assert fcn_resnet_entry["baseline_classification"] == "tflite_fail"
    assert fcn_resnet_entry["baseline_reason"] == (
        "onnxruntime_u8s8_saturating_pair_accumulation"
    )
    assert fcn_resnet_entry["error_signature_sha256"] == (
        "da92ac2e2f0e69a6b68654c2152fadfadb079a20d73aaf8b33fc3a81af4afc15"
    )
    assert fcn_resnet_entry["tflite_max_abs"] == 0.5471203327178955
    yolox_nano_entry = next(
        entry
        for entry in profile_payload["models"]
        if entry["model"] == "yolox_nano.onnx"
    )
    assert yolox_nano_entry["baseline_classification"] == "tflite_fail"
    assert yolox_nano_entry["baseline_reason"] == (
        "float_conv_accumulation_amplified_by_exp_stride"
    )
    assert yolox_nano_entry["error_signature_sha256"] == (
        "9516f4001a1361f52f4218a2585bedb5371d3be64116a8182cccfd91a3c0d5c0"
    )
    assert yolox_nano_entry["tflite_max_abs"] == 0.1362457275390625
    alike_entry = next(
        entry
        for entry in profile_payload["models"]
        if entry["model"] == "alike_l_opset11_192x320_post.onnx"
    )
    assert alike_entry["baseline_classification"] == "tflite_fail"
    assert alike_entry["baseline_reason"] == (
        "topk_index_instability_from_float_ties"
    )
    assert alike_entry["error_signature_sha256"] == (
        "9a77ab8f63d543c221797af2d444b22c59a1cbb4a483e4c4caed79f45fcc8c9b"
    )
    assert alike_entry["tflite_max_abs"] == 290.0000057220459
    for debug_model_name, expected_signature in (
        (
            "tmp_alike_debug3.onnx",
            "224de9b141fb9878c1b1ef289b511c6f87d8844c4eac617b0e897cf0ade38f75",
        ),
        (
            "tmp_alike_debug4.onnx",
            "55596567db57cb322d3d8364fd9a668b24e93b882334c6d876a1b3af07506601",
        ),
    ):
        debug_entry = next(
            entry
            for entry in profile_payload["models"]
            if entry["model"] == debug_model_name
        )
        assert debug_entry["baseline_classification"] == "tflite_fail"
        assert debug_entry["baseline_reason"] == (
            "exact_equality_mask_instability_from_float_accumulation"
        )
        assert debug_entry["error_signature_sha256"] == expected_signature
        assert debug_entry["tflite_max_abs"] == 1.0
    assert profile["model_options"]["tiny_decoder_11.onnx"] == {
        "shape_hints": [
            "tokens:1,1",
            "audio_features:1,1500,384",
            "kv_cache:8,1,1,384",
            "offset:1",
        ],
        "keep_shape_absolutely_input_names": [
            "tokens",
            "audio_features",
            "kv_cache",
            "offset",
        ],
    }
    assert profile["model_options"]["d3net_dnn_double_44.onnx"] == {
        "keep_shape_absolutely_input_names": ["input"],
    }
    assert profile["model_options"]["G_180000.onnx"] == {
        "overwrite_input_shape": ["specs:1,257,73"],
        "keep_shape_absolutely_input_names": [
            "specs",
            "lengths",
            "sid_src",
            "sid_tgt",
        ],
    }
    assert profile["model_options"]["LibreRFDETRn.onnx"] == {
        "overwrite_input_shape": ["input:1,3,384,384"],
    }
    assert profile["model_options"]["conv_tasnet.onnx"] == {
        "keep_shape_absolutely_input_names": ["onnx::Unsqueeze_0"],
    }
    for model_name in ["best.onnx", "best_org.onnx"]:
        assert profile["model_options"][model_name] == {
            "overwrite_input_shape": ["images:1,3,512,640"],
        }
