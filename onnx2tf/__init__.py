from __future__ import annotations

from typing import Any

__version__ = "2.6.0"


def _load_impl():
    from onnx2tf.onnx2tf import convert as _convert, main as _main

    return _convert, _main


def convert(*args: Any, **kwargs: Any):
    _convert, _ = _load_impl()
    return _convert(*args, **kwargs)


def main(*args: Any, **kwargs: Any):
    _, _main = _load_impl()
    return _main(*args, **kwargs)
