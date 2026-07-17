from __future__ import annotations

import ast
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
LOWERER_PATH = REPO_ROOT / "onnx2tf" / "tflite_builder" / "lower_from_onnx2tf.py"
LAYOUT_PREFIX = "_run_layout_recovery_prefix_pass_sequence"
ATTENTION_PREFIX = "_run_layout_reshape_attention_recovery_prefix"


def _lowerer_and_helper(helper_name: str) -> tuple[ast.FunctionDef, ast.FunctionDef]:
    tree = ast.parse(LOWERER_PATH.read_text(encoding="utf-8"))
    lowerer = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "lower_onnx_to_ir"
    )
    helper = next(
        node
        for node in lowerer.body
        if isinstance(node, ast.FunctionDef) and node.name == helper_name
    )
    return lowerer, helper


def _expression_path(node: ast.expr) -> Any:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return f"{_expression_path(node.value)}.{node.attr}"
    if isinstance(node, ast.Constant):
        return node.value
    raise AssertionError(f"unexpected call expression: {ast.dump(node)}")


def _call_contracts(
    helper: ast.FunctionDef,
) -> dict[str, tuple[tuple[Any, ...], dict[str, Any]]]:
    contracts: dict[str, tuple[tuple[Any, ...], dict[str, Any]]] = {}
    for statement in helper.body:
        assert isinstance(statement, ast.Expr)
        call = statement.value
        assert isinstance(call, ast.Call)
        assert isinstance(call.func, ast.Name)
        assert call.func.id not in contracts
        contracts[call.func.id] = (
            tuple(_expression_path(argument) for argument in call.args),
            {
                str(keyword.arg): _expression_path(keyword.value)
                for keyword in call.keywords
            },
        )
    return contracts


def _model_layout_contract(
    *, diagnostics: bool = False
) -> tuple[tuple[str, ...], dict[str, str]]:
    keywords = {"layout_state": "session.layout_state"}
    if diagnostics:
        keywords["diagnostics"] = "session.diagnostics"
    return ("model_ir",), keywords


def test_layout_recovery_helpers_are_straight_line_closures() -> None:
    expected_lines = {
        LAYOUT_PREFIX: 66,
        ATTENTION_PREFIX: 51,
    }
    control_flow_nodes = (
        ast.AsyncFor,
        ast.AsyncWith,
        ast.For,
        ast.If,
        ast.Match,
        ast.Try,
        ast.While,
        ast.With,
    )
    for helper_name, line_count in expected_lines.items():
        _, helper = _lowerer_and_helper(helper_name)

        assert helper.end_lineno is not None
        assert helper.end_lineno - helper.lineno + 1 == line_count
        assert helper.args.args == []
        assert helper.args.posonlyargs == []
        assert helper.args.kwonlyargs == []
        assert helper.args.vararg is None
        assert helper.args.kwarg is None
        assert not any(
            isinstance(node, control_flow_nodes) for node in ast.walk(helper)
        )
        assert not any(
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "ModelIRPassStateScope"
            for node in ast.walk(helper)
        )


def test_layout_recovery_prefix_preserves_exact_argument_contracts() -> None:
    _, helper = _lowerer_and_helper(LAYOUT_PREFIX)
    contracts = _call_contracts(helper)
    model_only = (("model_ir",), {})
    no_arguments = ((), {})
    model_layout = _model_layout_contract()
    model_layout_diagnostics = _model_layout_contract(diagnostics=True)

    assert contracts == {
        "_optimize_transpose_quant_dequant_bridges": model_only,
        "_run_boundary_batchmatmul_unary_layout_pass_cluster": no_arguments,
        "_optimize_transpose_elementwise_roundtrip_nchw_nhwc_chains": model_only,
        "_optimize_transpose_elementwise_roundtrip_nhwc_nchw_fanout_chains": model_only,
        "run_hard_activation_passthrough_cleanup": model_layout_diagnostics,
        "_optimize_swish_transpose_passthrough_chains": model_layout,
        "_optimize_gelu_tanh_transpose_passthrough_chains": model_layout,
        "_optimize_center_size_offset_terminal_transpose_chains": model_layout,
        "_optimize_leakyrelu_transpose_passthrough_chains": model_layout,
        "_optimize_prelu_transpose_passthrough_chains": model_layout,
        "_optimize_transpose_elementwise_concat_conv_nhwc_groups": model_layout,
        "run_spp_layout_cleanup": model_layout_diagnostics,
        "_optimize_transpose_pre_concat_nhwc_chains": model_layout_diagnostics,
        "run_ndhwc_concat_layout_cleanup": model_layout_diagnostics,
        "_optimize_transpose_stridedslice_pre_concat_nhwc_chains": model_layout,
        "_optimize_transpose_split_mixed_pre_concat_to_single_post_adapter_nhwc_chains": model_layout,
        "_optimize_transpose_input_chains_pre_concat_to_single_post_adapter": model_layout,
        "_optimize_transpose_slice_logistic_concat_reshape_tail_nhwc_chains": model_layout,
        "_run_channel_shuffle_gather_layout_pass_cluster": no_arguments,
    }


def test_attention_recovery_prefix_preserves_exact_argument_contracts() -> None:
    _, helper = _lowerer_and_helper(ATTENTION_PREFIX)
    contracts = _call_contracts(helper)
    model_only = (("model_ir",), {})
    model_layout = _model_layout_contract()

    assert contracts == {
        LAYOUT_PREFIX: ((), {}),
        "_optimize_transpose_pre_add_nhwc_chains": model_layout,
        "_optimize_transpose_pre_add_mulconst_reshape_transpose_suffix_nhwc_chains": model_layout,
        "_optimize_transpose_pre_unary_reshape_transpose_suffix_nhwc_chains": model_layout,
        "_optimize_transpose_reshape_transpose_to_expanddims_nhwc_chains": model_layout,
        "_optimize_transpose_reshape_transpose_to_flatten_hw_nhwc_chains": model_layout,
        "_optimize_reshape_transpose_reshape_transpose_to_nhwc_reshape_chains": model_only,
        "_optimize_attention_qkv_reshape_transpose_reshape_to_reshape_transpose_chains": model_layout,
        "_optimize_attention_gather_transpose_reshape_cleanup_chains": model_only,
        "_optimize_gather_axis0_singleton_to_reshape_input_chains": model_layout,
        "_optimize_attention_preproj_reshape_to_batchmatmul_ranklift_chains": model_only,
        "_optimize_window_partition_reshape_transpose_to_space_to_depth_chains": model_layout,
        "_optimize_window_reverse_reshape_transpose_to_depth_to_space_chains": model_layout,
        "_optimize_transpose_pre_unary_squeeze_transpose_suffix_nhwc_chains": model_layout,
        "run_squeeze_reshape_identity_cleanup": (
            ("model_ir",),
            {
                "include_unary_passthrough": True,
                "layout_state": "session.layout_state",
                "diagnostics": "session.diagnostics",
            },
        ),
    }


def test_layout_recovery_helpers_only_capture_model_and_session_state() -> None:
    for helper_name in (LAYOUT_PREFIX, ATTENTION_PREFIX):
        _, helper = _lowerer_and_helper(helper_name)
        called_names = {
            node.func.id
            for node in ast.walk(helper)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
        }
        loaded_data_names = {
            node.id
            for node in ast.walk(helper)
            if isinstance(node, ast.Name)
            and isinstance(node.ctx, ast.Load)
            and node.id not in called_names
        }

        assert loaded_data_names == {"model_ir", "session"}
