from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any, Callable, Dict, Generic, Iterable, List, Optional, TypeVar


StateT = TypeVar("StateT")
PassCallback = Callable[[StateT], Optional[Dict[str, Any]]]
PreconditionCallback = Callable[[StateT], bool]
FingerprintCallback = Callable[[StateT], bytes]
ValidatorCallback = Callable[[StateT], Iterable[str]]
CloneCallback = Callable[[StateT], Any]
RestoreCallback = Callable[[StateT, Any], None]


class PassPhase(IntEnum):
    NORMALIZE = 10
    CANONICALIZE = 20
    LAYOUT_PLAN = 30
    FUSION = 40
    LOWERING = 50
    POST_LOWERING_CLEANUP = 60
    VALIDATE = 70


@dataclass(frozen=True)
class PassSpec(Generic[StateT]):
    pass_id: str
    phase: PassPhase
    callback: PassCallback[StateT]
    precondition: Optional[PreconditionCallback[StateT]] = None
    priority: int = 100
    max_iterations: int = 1
    transactional: bool = False


@dataclass(frozen=True)
class PassResult:
    pass_id: str
    phase: str
    iterations: int
    changed: bool
    stopped_by_cycle: bool
    details: Dict[str, Any] = field(default_factory=dict)


class PassInvariantError(RuntimeError):
    """Invariant failure raised after a transactional pass is rolled back."""

    def __init__(
        self,
        *,
        pass_id: str,
        phase: PassPhase,
        iterations: int,
        problems: Iterable[str],
    ) -> None:
        self.pass_id = str(pass_id)
        self.phase = phase.name.lower()
        self.iterations = int(iterations)
        self.problems = tuple(str(problem) for problem in problems)
        super().__init__(
            "pass invariant violation: "
            f"pass_id={self.pass_id} problems={list(self.problems[:8])}"
        )


class OrderedPassManager(Generic[StateT]):
    """Deterministic pass runner with optional rollback and cycle detection."""

    def __init__(
        self,
        *,
        fingerprint: Optional[FingerprintCallback[StateT]] = None,
        validator: Optional[ValidatorCallback[StateT]] = None,
        clone: Optional[CloneCallback[StateT]] = None,
        restore: Optional[RestoreCallback[StateT]] = None,
    ) -> None:
        self._specs: List[PassSpec[StateT]] = []
        self._fingerprint = fingerprint
        self._validator = validator
        self._clone = clone
        self._restore = restore

    def register(self, spec: PassSpec[StateT]) -> None:
        pass_id = str(spec.pass_id).strip()
        if not pass_id:
            raise ValueError("pass_id must not be empty")
        if any(existing.pass_id == pass_id for existing in self._specs):
            raise ValueError(f"pass already registered: {pass_id}")
        if spec.max_iterations < 1:
            raise ValueError("max_iterations must be >= 1")
        if spec.transactional and (self._clone is None or self._restore is None):
            raise ValueError("transactional pass requires clone and restore callbacks")
        self._specs.append(spec)

    def ordered_specs(self) -> List[PassSpec[StateT]]:
        return sorted(
            self._specs,
            key=lambda spec: (int(spec.phase), int(spec.priority), str(spec.pass_id)),
        )

    def _digest(self, state: StateT) -> Optional[str]:
        if self._fingerprint is None:
            return None
        return hashlib.sha256(self._fingerprint(state)).hexdigest()

    def run(self, state: StateT) -> List[PassResult]:
        results: List[PassResult] = []
        for spec in self.ordered_specs():
            seen: set[str] = set()
            changed = False
            stopped_by_cycle = False
            details: Dict[str, Any] = {}
            iterations = 0
            for _ in range(int(spec.max_iterations)):
                if spec.precondition is not None and not spec.precondition(state):
                    details["skipped_by_precondition"] = True
                    break
                before = self._digest(state)
                if before is not None:
                    if before in seen:
                        stopped_by_cycle = True
                        break
                    seen.add(before)
                snapshot = self._clone(state) if spec.transactional and self._clone else None
                raw = spec.callback(state)
                current = dict(raw or {})
                details.update(current)
                iterations += 1
                problems = list(self._validator(state)) if self._validator else []
                if problems:
                    if snapshot is not None and self._restore is not None:
                        self._restore(state, snapshot)
                    raise PassInvariantError(
                        pass_id=spec.pass_id,
                        phase=spec.phase,
                        iterations=iterations,
                        problems=problems,
                    )
                after = self._digest(state)
                iteration_changed = bool(current.get("changed", before != after))
                changed = changed or iteration_changed
                if not iteration_changed:
                    break
            results.append(
                PassResult(
                    pass_id=spec.pass_id,
                    phase=spec.phase.name.lower(),
                    iterations=iterations,
                    changed=changed,
                    stopped_by_cycle=stopped_by_cycle,
                    details=details,
                )
            )
        return results
