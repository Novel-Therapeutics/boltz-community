"""Intel GPU (XPU) support for PyTorch Lightning.

PyTorch has shipped native Intel GPU support (``torch.xpu``) since 2.5, but
PyTorch Lightning only registers ``cpu``, ``cuda``, ``mps`` and ``tpu``
accelerators. This module registers a minimal single-device ``xpu``
accelerator so ``Trainer(accelerator="xpu")`` works.
"""

from __future__ import annotations

from typing import Any, Union

import torch
from pytorch_lightning.accelerators import Accelerator, AcceleratorRegistry


def xpu_available() -> bool:
    """Return True if PyTorch can see an Intel GPU."""
    return hasattr(torch, "xpu") and torch.xpu.is_available()


class XPUAccelerator(Accelerator):
    """Accelerator for Intel GPUs via ``torch.xpu``."""

    def setup_device(self, device: torch.device) -> None:
        if device.type != "xpu":
            msg = f"Device should be XPU, got {device} instead."
            raise ValueError(msg)
        torch.xpu.set_device(device)

    def teardown(self) -> None:
        torch.xpu.empty_cache()

    def get_device_stats(self, device: Union[str, torch.device]) -> dict[str, Any]:
        return {}

    @staticmethod
    def parse_devices(devices: Union[int, str, list[int]]) -> list[int]:
        if isinstance(devices, str):
            devices = [int(d) for d in devices.split(",") if d.strip()]
        if isinstance(devices, int):
            return list(range(max(1, devices)))
        return list(devices)

    @staticmethod
    def get_parallel_devices(devices: Union[int, str, list[int]]) -> list[torch.device]:
        return [
            torch.device("xpu", i) for i in XPUAccelerator.parse_devices(devices)
        ]

    @staticmethod
    def auto_device_count() -> int:
        return torch.xpu.device_count()

    @staticmethod
    def is_available() -> bool:
        return xpu_available()

    @classmethod
    def name(cls) -> str:
        return "xpu"

    @classmethod
    def register_accelerators(cls, accelerator_registry: AcceleratorRegistry) -> None:
        accelerator_registry.register(
            "xpu", cls, description="Intel GPU (XPU) accelerator"
        )


def register_xpu_accelerator() -> None:
    """Register the ``xpu`` accelerator with Lightning if it is not already known."""
    if "xpu" not in AcceleratorRegistry.available_accelerators():
        XPUAccelerator.register_accelerators(AcceleratorRegistry)
