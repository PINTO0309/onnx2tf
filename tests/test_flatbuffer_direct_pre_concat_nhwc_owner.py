from __future__ import annotations

import ast
from pathlib import Path

import pytest

import onnx2tf.tflite_builder.lower_from_onnx2tf as lowerer_module
import onnx2tf.tflite_builder.passes.pre_concat_nhwc_layout as owner_module
from onnx2tf.tflite_builder.ir import ModelIR


REPO_ROOT = Path(__file__).resolve().parents[1]
LOWERER_PATH = (
    REPO_ROOT / "onnx2tf" / "tflite_builder" / "lower_from_onnx2tf.py"
)
OWNER_PATH = (
    REPO_ROOT
    / "onnx2tf"
    / "tflite_builder"
    / "passes"
    / "pre_concat_nhwc_layout.py"
)
COMPOSITE = "_optimize_transpose_pre_concat_nhwc_chains"
PASS_OWNER = "optimize_transpose_pre_concat_nhwc_chains"
PASS_ALIAS = f"{COMPOSITE}_pass"
OWNER_STAGE_NAMES = (
    "run_nhwc_concat_layout_cleanup",
    "run_nhwc_concat_quantized_layout_cleanup",
    "optimize_transpose_pre_concat_nhwc_chains_legacy",
)


def _functions(path: Path) -> dict[str, ast.FunctionDef]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    return {
        node.name: node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
    }


def _call_name(call: ast.Call) -> str | None:
    return call.func.id if isinstance(call.func, ast.Name) else None


def test_pre_concat_pass_owner_order_and_schema_are_explicit() -> None:
    owner_tree = ast.parse(OWNER_PATH.read_text(encoding="utf-8"))
    composite = _functions(OWNER_PATH)[PASS_OWNER]
    stage_calls = sorted(
        (
            node
            for node in ast.walk(composite)
            if isinstance(node, ast.Call)
            and _call_name(node) in OWNER_STAGE_NAMES
        ),
        key=lambda node: node.lineno,
    )
    assert tuple(_call_name(call) for call in stage_calls) == OWNER_STAGE_NAMES
    assert [ast.unparse(argument) for argument in stage_calls[0].args] == [
        "model_ir"
    ]
    assert [ast.unparse(argument) for argument in stage_calls[1].args] == [
        "model_ir"
    ]
    assert {
        keyword.arg: ast.unparse(keyword.value)
        for keyword in stage_calls[0].keywords
    } == {
        "layout_state": "layout_state",
        "diagnostics": "diagnostics",
    }
    assert {
        keyword.arg: ast.unparse(keyword.value)
        for keyword in stage_calls[1].keywords
    } == {
        "layout_state": "layout_state",
        "diagnostics": "diagnostics",
    }
    assert ast.unparse(stage_calls[2]) == (
        "optimize_transpose_pre_concat_nhwc_chains_legacy(model_ir)"
    )
    stats_key = next(
        node
        for node in owner_tree.body
        if isinstance(node, ast.Assign)
        and len(node.targets) == 1
        and isinstance(node.targets[0], ast.Name)
        and node.targets[0].id == "_STATS_KEY"
    )
    assert ast.literal_eval(stats_key.value) == (
        "optimized_transpose_pre_concat_nhwc_chains"
    )
    assert ast.unparse(composite.body[-1]) == (
        "return {_STATS_KEY: int(optimized)}"
    )


def test_pre_concat_pass_owner_runtime_preserves_aggregation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model_ir = ModelIR("pre_concat_nhwc_owner")
    layout_state = object()
    diagnostics: list[dict[str, object]] = []
    observed: list[tuple[str, object, object | None, object | None]] = []

    def _indexed(
        candidate: ModelIR,
        *,
        layout_state: object,
        diagnostics: object,
    ) -> dict[str, int]:
        observed.append(
            (OWNER_STAGE_NAMES[0], candidate, layout_state, diagnostics)
        )
        return {
            "optimized_transpose_pre_concat_nhwc_direct_chains": 2,
            "ignored_indexed_counter": 100,
        }

    def _quantized(
        candidate: ModelIR,
        *,
        layout_state: object,
        diagnostics: object,
    ) -> dict[str, int]:
        observed.append(
            (OWNER_STAGE_NAMES[1], candidate, layout_state, diagnostics)
        )
        return {
            "optimized_transpose_pre_concat_nhwc_quantized_direct_chains": 3,
            "ignored_quantized_counter": 100,
        }

    def _legacy(candidate: ModelIR) -> dict[str, int]:
        observed.append((OWNER_STAGE_NAMES[2], candidate, None, None))
        return {"optimized_transpose_pre_concat_nhwc_chains": 5}

    monkeypatch.setattr(owner_module, OWNER_STAGE_NAMES[0], _indexed)
    monkeypatch.setattr(owner_module, OWNER_STAGE_NAMES[1], _quantized)
    monkeypatch.setattr(owner_module, OWNER_STAGE_NAMES[2], _legacy)

    assert lowerer_module._optimize_transpose_pre_concat_nhwc_chains(
        model_ir,
        layout_state=layout_state,
        diagnostics=diagnostics,
    ) == {"optimized_transpose_pre_concat_nhwc_chains": 10}
    assert [entry[0] for entry in observed] == list(OWNER_STAGE_NAMES)
    assert all(entry[1] is model_ir for entry in observed)
    assert observed[0][2:] == (layout_state, diagnostics)
    assert observed[1][2:] == (layout_state, diagnostics)
    assert observed[2][2:] == (None, None)


def test_pre_concat_composite_moves_behind_pass_module_compat_wrapper() -> None:
    assert OWNER_PATH.exists()
    owner = _functions(OWNER_PATH)[PASS_OWNER]
    owner_calls = sorted(
        (
            node
            for node in ast.walk(owner)
            if isinstance(node, ast.Call)
            and _call_name(node)
            in {
                OWNER_STAGE_NAMES[0],
                OWNER_STAGE_NAMES[1],
                "optimize_transpose_pre_concat_nhwc_chains_legacy",
            }
        ),
        key=lambda node: node.lineno,
    )
    assert tuple(_call_name(call) for call in owner_calls) == (
        OWNER_STAGE_NAMES[0],
        OWNER_STAGE_NAMES[1],
        "optimize_transpose_pre_concat_nhwc_chains_legacy",
    )

    wrapper = _functions(LOWERER_PATH)[COMPOSITE]
    assert len(wrapper.body) == 1
    dispatch = wrapper.body[0]
    assert isinstance(dispatch, ast.Return)
    assert isinstance(dispatch.value, ast.Call)
    assert _call_name(dispatch.value) == PASS_ALIAS
    assert [ast.unparse(argument) for argument in dispatch.value.args] == [
        "model_ir"
    ]
    assert {
        keyword.arg: ast.unparse(keyword.value)
        for keyword in dispatch.value.keywords
    } == {
        "layout_state": "layout_state",
        "diagnostics": "diagnostics",
    }

    lowerer_tree = ast.parse(LOWERER_PATH.read_text(encoding="utf-8"))
    imports = [
        node
        for node in lowerer_tree.body
        if isinstance(node, ast.ImportFrom)
        and node.module
        == "onnx2tf.tflite_builder.passes.pre_concat_nhwc_layout"
    ]
    assert len(imports) == 1
    assert [(alias.name, alias.asname) for alias in imports[0].names] == [
        (PASS_OWNER, PASS_ALIAS)
    ]
