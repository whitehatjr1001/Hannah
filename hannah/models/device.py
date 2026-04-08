"""Helpers for selecting torch execution devices."""

from __future__ import annotations

from typing import Any


def get_torch_device_name(torch_module: Any | None = None) -> str:
    """Return the preferred torch device name for the current machine.

    Preference order matches the runtime expectation for accelerated training:
    CUDA first, then Apple Metal (MPS), then CPU.
    """
    if torch_module is None:
        try:
            import torch as imported_torch
        except Exception:
            return "cpu"
        torch_module = imported_torch

    cuda = getattr(torch_module, "cuda", None)
    if cuda is not None:
        is_available = getattr(cuda, "is_available", None)
        if callable(is_available) and bool(is_available()):
            return "cuda"

    backends = getattr(torch_module, "backends", None)
    mps = getattr(backends, "mps", None) if backends is not None else None
    if mps is not None:
        is_available = getattr(mps, "is_available", None)
        if callable(is_available) and bool(is_available()):
            return "mps"

    return "cpu"
