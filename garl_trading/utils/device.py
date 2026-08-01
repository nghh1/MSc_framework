from __future__ import annotations
import torch

def resolve_torch_device(device: str | torch.device | None = "auto") -> torch.device:
    """Resolve auto/CUDA/MPS/CPU and fail clearly for unavailable explicit devices."""
    if isinstance(device, torch.device):
        return device
    requested = (device or "auto").lower()
    if requested == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    if requested.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(f"Requested {device!r}, but CUDA is unavailable.")
    if requested == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("Requested 'mps', but Apple Metal acceleration is unavailable.")
    return torch.device(requested)