"""Tests for torch device selection helpers."""

from __future__ import annotations

from types import SimpleNamespace

from hannah.models.device import get_torch_device_name


def _torch_stub(*, cuda: bool, mps: bool) -> SimpleNamespace:
    return SimpleNamespace(
        cuda=SimpleNamespace(is_available=lambda: cuda),
        backends=SimpleNamespace(
            mps=SimpleNamespace(is_available=lambda: mps),
        ),
    )


def test_get_torch_device_name_prefers_cuda_over_mps() -> None:
    torch_stub = _torch_stub(cuda=True, mps=True)

    assert get_torch_device_name(torch_stub) == "cuda"


def test_get_torch_device_name_uses_mps_when_cuda_is_unavailable() -> None:
    torch_stub = _torch_stub(cuda=False, mps=True)

    assert get_torch_device_name(torch_stub) == "mps"


def test_get_torch_device_name_falls_back_to_cpu() -> None:
    torch_stub = _torch_stub(cuda=False, mps=False)

    assert get_torch_device_name(torch_stub) == "cpu"


def test_get_torch_device_name_handles_missing_torch_module() -> None:
    assert get_torch_device_name(None) == "cpu"
