"""
Augmentations for sequence data used in contrastive learning.

All transforms take and return a torch.Tensor of shape (T, D).

They should preserve the underlying "semantic" behavior while introducing
small variations, so that the encoder learns invariant representations.

Include five augmentations:
    1) GaussianJitter:
       - Models measurement noise / small numerical perturbations.
    2) Scaling:
       - Models calibration drift, unit changes, and gain variations.
    3) TimeShift:
       - Models small misalignment caused by window boundaries or latency.
    4) TimeMask:
       - Models missing log entries / dropped samples / partial observability.
    5) RandomCrop (+ padding):
       - Models different segmentation boundaries and partial observation of a behavior.

"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Optional, Sequence, Union, Dict, Any
import torch


@dataclass
class RandomApply:
    """
    Apply a transform with probability p. To control augmentation strength.

    """
    transform: Callable[[torch.Tensor], torch.Tensor]
    p: float = 0.5

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        if self.p <= 0.0:
            return x
        if self.p >= 1.0:
            return self.transform(x)
        if torch.rand(1, device=x.device).item() < self.p:
            return self.transform(x)
        return x


class Compose:
    """Compose multiple transforms into a single callable."""

    def __init__(self, transforms: Sequence[Callable[[torch.Tensor], torch.Tensor]]):
        self.transforms = list(transforms)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        for t in self.transforms:
            x = t(x)
        return x


@dataclass
class GaussianJitter:
    """
    Add small Gaussian noise.
    """
    sigma: float = 0.01

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        if self.sigma <= 0:
            return x
        noise = torch.randn_like(x) * self.sigma
        return x + noise


@dataclass
class Scaling:
    """
    Multiply each feature by a random factor.

    Notes:
        For log-derived features containing one-hot-like dimensions, scaling can distort semantics.
        Consider using very conservative ranges (e.g., 0.98~1.02) or disabling this transform.
    """
    min_scale: float = 0.9
    max_scale: float = 1.1

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        if self.min_scale == 1.0 and self.max_scale == 1.0:
            return x

        scale = torch.empty((1, x.shape[1]), device=x.device).uniform_(self.min_scale, self.max_scale)
        return x * scale


@dataclass
class TimeShift:
    """
    Shift the sequence along the time dimension.

    By default uses circular shift (torch.roll). For some tasks, circular shift may be unrealistic.
    You can set circular=False to use zero-padding shift instead.
    """
    max_shift: int = 2
    circular: bool = True

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        if self.max_shift <= 0:
            return x

        T = x.shape[0]
        if T <= 1:
            return x

        shift = int(torch.randint(low=-self.max_shift, high=self.max_shift + 1, size=(1,)).item())
        if shift == 0:
            return x

        if self.circular:
            return torch.roll(x, shifts=shift, dims=0)

        # Non-circular shift: pad with zeros
        out = torch.zeros_like(x)
        if shift > 0:
            out[shift:] = x[: T - shift]
        else:
            s = -shift
            out[: T - s] = x[s:]
        return out


@dataclass
class TimeMask:
    """Randomly mask (set to zero) a contiguous time segment."""
    max_mask_ratio: float = 0.2
    mask_value: float = 0.0

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        T = x.shape[0]
        if self.max_mask_ratio <= 0 or T <= 1:
            return x

        max_len = max(1, int(T * self.max_mask_ratio))
        mask_len = int(torch.randint(low=1, high=max_len + 1, size=(1,)).item())

        start = int(torch.randint(low=0, high=T - mask_len + 1, size=(1,)).item())
        out = x.clone()
        out[start:start + mask_len] = self.mask_value
        return out


@dataclass
class RandomCrop:
    """
    Randomly crop a subsequence and pad back to original length.

    Implementation details:
        - Crop a random contiguous subsequence.
        - Pad zeros (or a chosen value) to restore length T
    """
    min_crop_ratio: float = 0.8
    pad_value: float = 0.0

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        T, D = x.shape
        if self.min_crop_ratio >= 1.0 or T <= 1:
            return x

        # Sample crop length in [min_crop_ratio*T, T]
        rand_ratio = float(torch.empty(1).uniform_(self.min_crop_ratio, 1.0).item())
        crop_len = max(1, int(T * rand_ratio))
        if crop_len >= T:
            return x

        start = int(torch.randint(low=0, high=T - crop_len + 1, size=(1,)).item())
        cropped = x[start:start + crop_len]

        pad_total = T - crop_len
        pad_left = pad_total // 2
        pad_right = pad_total - pad_left

        left_pad = torch.full((pad_left, D), self.pad_value, device=x.device, dtype=x.dtype)
        right_pad = torch.full((pad_right, D), self.pad_value, device=x.device, dtype=x.dtype)
        return torch.cat([left_pad, cropped, right_pad], dim=0)


def build_augmentation(
    profile: str,
    *,
    # Common knobs
    jitter_sigma: float = 0.01,
    jitter_p: float = 1.0,
    mask_ratio: float = 0.05,
    mask_p: float = 0.3,
    shift_steps: int = 1,
    shift_p: float = 0.3,
    crop_ratio: float = 0.9,
    crop_p: float = 0.2,
    # Scaling knobs
    scale_min: float = 0.98,
    scale_max: float = 1.02,
    scale_p: float = 0.2,

) -> Callable[[torch.Tensor], torch.Tensor]:
    """
    Build an augmentation pipeline by a named profile.
    """
    if profile == "log_weak":
        return Compose([
            RandomApply(GaussianJitter(sigma=jitter_sigma), p=jitter_p),
            RandomApply(TimeMask(max_mask_ratio=mask_ratio), p=mask_p),
            RandomApply(TimeShift(max_shift=shift_steps, circular=False), p=shift_p),
        ])

    if profile == "log_strong":
        return Compose([
            RandomApply(GaussianJitter(sigma=jitter_sigma), p=jitter_p),
            RandomApply(TimeMask(max_mask_ratio=max(mask_ratio, 0.15)), p=max(mask_p, 0.5)),
            RandomApply(TimeShift(max_shift=max(shift_steps, 2), circular=False), p=max(shift_p, 0.5)),
            RandomApply(RandomCrop(min_crop_ratio=min(crop_ratio, 0.85)), p=max(crop_p, 0.4)),
            RandomApply(Scaling(min_scale=scale_min, max_scale=scale_max), p=scale_p),
        ])

    raise ValueError(f"Unknown augmentation profile: {profile}")