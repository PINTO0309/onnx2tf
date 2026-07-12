from __future__ import annotations

import threading
from typing import Any, Optional


def create_progress_bar(*, total: int, desc: str, enabled: bool):
    if not enabled or int(total) <= 0:
        return None
    try:
        from tqdm.auto import tqdm
    except Exception:
        return None
    return tqdm(total=int(total), desc=str(desc), dynamic_ncols=True)


class ProgressSpinner:
    def __init__(self, progress_bar: Any) -> None:
        self._progress_bar = progress_bar
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def start(self) -> None:
        self.stop()
        if self._progress_bar is None:
            return
        self._stop_event = threading.Event()
        self._progress_bar.set_postfix_str("|", refresh=True)
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        thread = self._thread
        if thread is not None:
            self._stop_event.set()
            thread.join(timeout=0.5)
        self._thread = None
        if self._progress_bar is not None:
            self._progress_bar.set_postfix_str("", refresh=True)

    def _run(self) -> None:
        frames = ["|", "/", "-", "\\"]
        frame_index = 0
        while not self._stop_event.wait(0.1):
            if self._progress_bar is None:
                return
            frame_index = (frame_index + 1) % len(frames)
            self._progress_bar.set_postfix_str(frames[frame_index], refresh=True)
