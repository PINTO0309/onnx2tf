from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Sequence, Tuple


@dataclass(frozen=True)
class RecoveryInvocation:
    pass_id: str
    callback: Callable[..., Any]
    args: Tuple[Any, ...] = ()
    keyword_args: Tuple[Tuple[str, Any], ...] = ()

    def run(self) -> Any:
        return self.callback(*self.args, **dict(self.keyword_args))


def run_recovery_invocations(
    invocations: Sequence[RecoveryInvocation],
    *,
    expected_pass_ids: Sequence[str],
    phase_name: str,
) -> None:
    actual_pass_ids = tuple(invocation.pass_id for invocation in invocations)
    if actual_pass_ids != tuple(expected_pass_ids):
        raise RuntimeError(f"{phase_name} pass IDs diverged from their order")
    for invocation in invocations:
        invocation.run()
