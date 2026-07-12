from __future__ import annotations


class ModelIRPyTorchExportError(RuntimeError):
    """Raised when a requested ModelIR-backed PyTorch artifact cannot be built."""


class NativePyTorchGenerationTimeoutError(ModelIRPyTorchExportError):
    """Raised when native PyTorch package generation exceeds its time limit."""
