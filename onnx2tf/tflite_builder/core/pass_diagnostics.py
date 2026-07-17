from __future__ import annotations

from typing import Any, Dict, Iterable, Optional, Union


class ModelIRPassDiagnostics(list[Dict[str, Any]]):
    """List-compatible diagnostics with lazily maintained pass numbering."""

    def __init__(
        self,
        values: Iterable[Dict[str, Any]] = (),
    ) -> None:
        super().__init__(values)
        self._model_ir_numbering_valid = len(self) == 0
        self._model_ir_event_count = 0
        self._maximum_model_ir_group_sequence: Optional[int] = None
        self._model_ir_invocations: Dict[str, int] = {}

    def _invalidate_model_ir_numbering(self) -> None:
        self._model_ir_numbering_valid = False

    def _reset_empty_model_ir_numbering(self) -> None:
        self._model_ir_numbering_valid = True
        self._model_ir_event_count = 0
        self._maximum_model_ir_group_sequence = None
        self._model_ir_invocations = {}

    def _update_model_ir_numbering_for_append(
        self,
        event: Dict[str, Any],
    ) -> None:
        if not self._model_ir_numbering_valid:
            return
        if not isinstance(event, dict):
            self._invalidate_model_ir_numbering()
            return
        try:
            if str(event.get("stage", "")) != "model_ir_pass":
                return
            group_sequence = int(event.get("group_sequence", 0))
        except (TypeError, ValueError, OverflowError):
            self._invalidate_model_ir_numbering()
            return
        code = str(event.get("code", ""))
        self._model_ir_event_count += 1
        self._model_ir_invocations[code] = int(
            self._model_ir_invocations.get(code, 0) + 1
        )
        if (
            self._maximum_model_ir_group_sequence is None
            or group_sequence > self._maximum_model_ir_group_sequence
        ):
            self._maximum_model_ir_group_sequence = group_sequence

    def _rebuild_model_ir_numbering(self) -> None:
        event_count = 0
        maximum_group_sequence: Optional[int] = None
        invocations: Dict[str, int] = {}
        for event in list.__iter__(self):
            if str(event.get("stage", "")) != "model_ir_pass":
                continue
            event_count += 1
            code = str(event.get("code", ""))
            invocations[code] = int(invocations.get(code, 0) + 1)
            group_sequence = int(event.get("group_sequence", 0))
            if (
                maximum_group_sequence is None
                or group_sequence > maximum_group_sequence
            ):
                maximum_group_sequence = group_sequence
        self._model_ir_event_count = event_count
        self._maximum_model_ir_group_sequence = maximum_group_sequence
        self._model_ir_invocations = invocations
        self._model_ir_numbering_valid = True

    def model_ir_numbering_snapshot(
        self,
    ) -> tuple[int, Optional[int], Dict[str, int]]:
        if not self._model_ir_numbering_valid:
            self._rebuild_model_ir_numbering()
        return (
            int(self._model_ir_event_count),
            self._maximum_model_ir_group_sequence,
            dict(self._model_ir_invocations),
        )

    def append(self, event: Dict[str, Any]) -> None:
        list.append(self, event)
        self._update_model_ir_numbering_for_append(event)

    def extend(self, values: Iterable[Dict[str, Any]]) -> None:
        appended = list(values)
        list.extend(self, appended)
        for event in appended:
            self._update_model_ir_numbering_for_append(event)

    def insert(self, index: int, event: Dict[str, Any]) -> None:
        list.insert(self, index, event)
        self._update_model_ir_numbering_for_append(event)

    def __iadd__(
        self,
        values: Iterable[Dict[str, Any]],
    ) -> ModelIRPassDiagnostics:
        self.extend(values)
        return self

    def __imul__(self, value: int) -> ModelIRPassDiagnostics:
        list.__imul__(self, value)
        if len(self) == 0:
            self._reset_empty_model_ir_numbering()
        else:
            self._invalidate_model_ir_numbering()
        return self

    def __setitem__(
        self,
        index: Union[int, slice],
        value: Union[Dict[str, Any], Iterable[Dict[str, Any]]],
    ) -> None:
        list.__setitem__(self, index, value)
        if len(self) == 0:
            self._reset_empty_model_ir_numbering()
        else:
            self._invalidate_model_ir_numbering()

    def __delitem__(self, index: Union[int, slice]) -> None:
        list.__delitem__(self, index)
        if len(self) == 0:
            self._reset_empty_model_ir_numbering()
        else:
            self._invalidate_model_ir_numbering()

    def pop(self, index: int = -1) -> Dict[str, Any]:
        value = list.pop(self, index)
        if len(self) == 0:
            self._reset_empty_model_ir_numbering()
        else:
            self._invalidate_model_ir_numbering()
        return value

    def remove(self, value: Dict[str, Any]) -> None:
        list.remove(self, value)
        if len(self) == 0:
            self._reset_empty_model_ir_numbering()
        else:
            self._invalidate_model_ir_numbering()

    def clear(self) -> None:
        list.clear(self)
        self._reset_empty_model_ir_numbering()
