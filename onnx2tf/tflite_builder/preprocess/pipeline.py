from __future__ import annotations

from collections import OrderedDict
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import onnx

PreprocessRuleCallback = Callable[[onnx.ModelProto], Optional[Dict[str, Any]]]


class _PreprocessRule:
    def __init__(
        self,
        rule_id: str,
        callback: PreprocessRuleCallback,
    ) -> None:
        self.rule_id = str(rule_id)
        self.callback = callback


_REGISTERED_PREPROCESS_RULES: "OrderedDict[str, _PreprocessRule]" = OrderedDict()


def register_preprocess_rule(
    *,
    rule_id: str,
    callback: PreprocessRuleCallback,
    overwrite: bool = False,
) -> None:
    rid = str(rule_id).strip()
    if rid == "":
        raise ValueError("preprocess rule_id must not be empty.")
    if not callable(callback):
        raise TypeError("preprocess callback must be callable.")
    if rid in _REGISTERED_PREPROCESS_RULES and not overwrite:
        raise ValueError(f"preprocess rule already exists: {rid}")
    _REGISTERED_PREPROCESS_RULES[rid] = _PreprocessRule(
        rule_id=rid,
        callback=callback,
    )


def clear_preprocess_rules() -> None:
    _REGISTERED_PREPROCESS_RULES.clear()


def get_registered_preprocess_rule_ids() -> List[str]:
    return list(_REGISTERED_PREPROCESS_RULES.keys())


def run_preprocess_pipeline(
    *,
    onnx_graph: onnx.ModelProto,
    enabled_rule_ids: Optional[Sequence[str]] = None,
) -> Tuple[onnx.ModelProto, Dict[str, Any]]:
    working_graph = onnx.ModelProto()
    working_graph.CopyFrom(onnx_graph)

    registered_rule_ids = get_registered_preprocess_rule_ids()
    if enabled_rule_ids is None:
        target_rule_ids = list(registered_rule_ids)
    else:
        target_rule_ids = [str(v) for v in enabled_rule_ids]
        unknown_rule_ids = sorted(
            list(set(target_rule_ids) - set(registered_rule_ids))
        )
        if len(unknown_rule_ids) > 0:
            raise ValueError(
                f"Unknown preprocess rule id(s): {unknown_rule_ids}"
            )

    applied_rules: List[Dict[str, Any]] = []
    total_matched_nodes = 0
    total_rewritten_nodes = 0
    for rule_id in target_rule_ids:
        rule = _REGISTERED_PREPROCESS_RULES[rule_id]
        raw_result = rule.callback(working_graph)
        result = raw_result if isinstance(raw_result, dict) else {}
        matched_nodes = int(result.get("matched_nodes", 0))
        rewritten_nodes = int(result.get("rewritten_nodes", 0))
        if matched_nodes < 0:
            matched_nodes = 0
        if rewritten_nodes < 0:
            rewritten_nodes = 0
        changed = bool(
            result.get(
                "changed",
                rewritten_nodes > 0,
            )
        )
        applied_rules.append(
            {
                "rule_id": rule_id,
                "matched_nodes": matched_nodes,
                "rewritten_nodes": rewritten_nodes,
                "changed": changed,
                "message": str(result.get("message", "")),
            }
        )
        total_matched_nodes += matched_nodes
        total_rewritten_nodes += rewritten_nodes

    report = {
        "schema_version": 1,
        "pipeline_version": 1,
        "registered_rule_ids": registered_rule_ids,
        "enabled_rule_ids": target_rule_ids,
        "applied_rules": applied_rules,
        "summary": {
            "registered_rule_count": int(len(registered_rule_ids)),
            "enabled_rule_count": int(len(target_rule_ids)),
            "executed_rule_count": int(len(applied_rules)),
            "changed_rule_count": int(
                len([r for r in applied_rules if bool(r.get("changed", False))])
            ),
            "total_matched_nodes": int(total_matched_nodes),
            "total_rewritten_nodes": int(total_rewritten_nodes),
        },
    }
    return working_graph, report

