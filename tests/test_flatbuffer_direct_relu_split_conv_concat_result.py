from __future__ import annotations

import ast
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
LOWERER_PATH = REPO_ROOT / "onnx2tf" / "tflite_builder" / "lower_from_onnx2tf.py"
OWNER_PATH = (
    REPO_ROOT
    / "onnx2tf"
    / "tflite_builder"
    / "passes"
    / "split_all_outputs_layout.py"
)
RELU_SPLIT_ALL = "_optimize_transpose_relu_split_all_outputs_to_nhwc_chains"
RELU_SPLIT_CONV_CONCAT = (
    "_optimize_transpose_relu_split_conv_relu_concat_posttranspose_to_nhwc_chains"
)
OWNER_NAME = (
    "optimize_transpose_relu_split_conv_relu_concat_posttranspose_to_nhwc_chains"
)
SPLIT_CONV_CONCAT_BRIDGE = (
    "_optimize_split_conv_concat_transpose_bridge_to_single_post_nchw"
)
SPLIT_MIXED_PRE_CONCAT = (
    "_optimize_transpose_split_mixed_pre_concat_to_single_post_adapter_nhwc_chains"
)
POST_SINET_RESULT_TARGETS = (
    "_post_sinet_relu_split_all_outputs_stats",
    "_post_sinet_relu_split_conv_concat_stats",
    "_post_sinet_split_conv_concat_bridge_stats",
)
POST_SINET_PHASE_IDS = (
    "cleanup.post_sinet.relu_split_all_outputs",
    "cleanup.post_sinet.relu_split_conv_concat",
    "cleanup.post_sinet.split_conv_concat_bridge",
)
POST_SINET_OWNER_EXPRESSIONS = (
    (
        "_optimize_transpose_relu_split_all_outputs_to_nhwc_chains("
        "model_ir, layout_state=session.layout_state)"
    ),
    (
        "_optimize_transpose_relu_split_conv_relu_concat_posttranspose_to_nhwc_chains("
        "model_ir, layout_state=session.layout_state)"
    ),
    (
        "_optimize_split_conv_concat_transpose_bridge_to_single_post_nchw("
        "model_ir, layout_state=session.layout_state)"
    ),
)


def _functions(path: Path) -> dict[str, ast.FunctionDef]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    return {
        node.name: node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
    }


def _statement_call(statement: ast.stmt) -> ast.Call | None:
    if not isinstance(statement, (ast.Assign, ast.Expr)):
        return None
    if not isinstance(statement.value, ast.Call):
        return None
    call = statement.value
    if (
        isinstance(call.func, ast.Attribute)
        and isinstance(call.func.value, ast.Name)
        and call.func.value.id == "session"
        and call.func.attr == "record_phase_result"
        and len(call.args) == 2
        and isinstance(call.args[1], ast.Call)
    ):
        return call.args[1]
    return call


def _call_name(statement: ast.stmt) -> str | None:
    call = _statement_call(statement)
    if call is None or not isinstance(call.func, ast.Name):
        return None
    return call.func.id


def _single_target(statement: ast.stmt) -> str | None:
    if not isinstance(statement, ast.Assign) or len(statement.targets) != 1:
        return None
    target = statement.targets[0]
    return target.id if isinstance(target, ast.Name) else None


def _phase_id(statement: ast.stmt) -> str | None:
    if not isinstance(statement, ast.Expr) or not isinstance(statement.value, ast.Call):
        return None
    call = statement.value
    if (
        not isinstance(call.func, ast.Attribute)
        or not isinstance(call.func.value, ast.Name)
        or call.func.value.id != "session"
        or call.func.attr != "record_phase_result"
        or len(call.args) != 2
    ):
        return None
    return ast.literal_eval(call.args[0])


def test_relu_split_conv_concat_schema_and_positive_cleanup_are_explicit() -> None:
    lowerer_wrapper = _functions(LOWERER_PATH)[RELU_SPLIT_CONV_CONCAT]
    assert len(lowerer_wrapper.body) == 1
    wrapper_return = lowerer_wrapper.body[0]
    assert isinstance(wrapper_return, ast.Return)
    assert isinstance(wrapper_return.value, ast.Call)
    assert isinstance(wrapper_return.value.func, ast.Name)
    assert wrapper_return.value.func.id == f"{RELU_SPLIT_CONV_CONCAT}_pass"
    assert [ast.unparse(argument) for argument in wrapper_return.value.args] == [
        "model_ir",
    ]
    assert {
        keyword.arg: ast.unparse(keyword.value)
        for keyword in wrapper_return.value.keywords
    } == {
        "graph_index": "graph_index",
        "layout_state": "layout_state",
        "max_rewrites": "max_rewrites",
        "candidate": "candidate",
    }

    owner_tree = ast.parse(OWNER_PATH.read_text(encoding="utf-8"))
    stats_key = next(
        statement
        for statement in owner_tree.body
        if isinstance(statement, ast.Assign)
        and _single_target(statement) == "_CONV_CONCAT_STATS_KEY"
    )
    assert isinstance(stats_key.value, ast.Constant)
    assert stats_key.value.value == (
        "optimized_transpose_relu_split_conv_relu_concat_posttranspose_to_nhwc_chains"
    )

    owner = _functions(OWNER_PATH)[OWNER_NAME]
    cleanup_guards = [
        statement
        for statement in owner.body
        if isinstance(statement, ast.If)
        and any(
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "_prune_unused_tensors"
            for node in ast.walk(statement)
        )
    ]
    assert len(cleanup_guards) == 1
    cleanup_guard = cleanup_guards[0]
    assert ast.unparse(cleanup_guard.test) == "rewritten > 0"
    assert [
        node.func.id
        for node in ast.walk(cleanup_guard)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    ] == ["_prune_unused_tensors"]
    assert not any(
        isinstance(statement, ast.Expr)
        and _call_name(statement) == "_prune_unused_tensors"
        for statement in owner.body
    )


def test_lowerer_records_post_sinet_relu_split_conv_concat_result() -> None:
    lowerer = _functions(LOWERER_PATH)["lower_onnx_to_ir"]
    direct_results = [
        statement
        for statement in lowerer.body
        if _call_name(statement) == RELU_SPLIT_CONV_CONCAT
    ]
    assert len(direct_results) == 1
    expected_targets = [None]
    assert [_single_target(statement) for statement in direct_results] == (
        expected_targets
    )
    assert _phase_id(direct_results[0]) == (
        "cleanup.post_sinet.relu_split_conv_concat"
    )
    for statement in direct_results:
        call = _statement_call(statement)
        assert call is not None
        assert [ast.unparse(argument) for argument in call.args] == ["model_ir"]
        assert {
            keyword.arg: ast.unparse(keyword.value)
            for keyword in call.keywords
        } == {"layout_state": "session.layout_state"}
    for target in (target for target in expected_targets if target is not None):
        assert not any(
            isinstance(node, ast.Name)
            and node.id == target
            and isinstance(node.ctx, ast.Load)
            for node in ast.walk(lowerer)
        )

    first_index = lowerer.body.index(direct_results[0])
    first_previous = lowerer.body[first_index - 1]
    first_following = lowerer.body[first_index + 1]
    assert _phase_id(first_previous) == (
        "cleanup.post_sinet.relu_split_all_outputs"
    )
    assert _call_name(first_previous) == RELU_SPLIT_ALL
    assert _phase_id(first_following) == (
        "cleanup.post_sinet.split_conv_concat_bridge"
    )
    assert _call_name(first_following) == SPLIT_CONV_CONCAT_BRIDGE



def test_post_sinet_relu_split_results_use_phase_result_store() -> None:
    lowerer = _functions(LOWERER_PATH)["lower_onnx_to_ir"]
    records = [
        statement
        for statement in lowerer.body
        if _phase_id(statement) in POST_SINET_PHASE_IDS
    ]
    indices = [lowerer.body.index(statement) for statement in records]

    assert tuple(_phase_id(statement) for statement in records) == POST_SINET_PHASE_IDS
    assert tuple(ast.unparse(statement.value.args[1]) for statement in records) == (
        POST_SINET_OWNER_EXPRESSIONS
    )
    assert indices == list(range(indices[0], indices[0] + 3))
    predecessor = lowerer.body[indices[0] - 1]
    assert _single_target(predecessor) == "_post_sinet_qkv_attention_results"
    successor = lowerer.body[indices[-1] + 1]
    assert _phase_id(successor) == "cleanup.post_sinet.mix_attention"
    assert not any(
        isinstance(node, ast.Name)
        and node.id in POST_SINET_RESULT_TARGETS
        and isinstance(node.ctx, ast.Load)
        for node in ast.walk(lowerer)
    )
