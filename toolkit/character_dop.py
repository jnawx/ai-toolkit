import math
from dataclasses import dataclass
from typing import Literal, Optional

import torch
import torch.nn.functional as F


CharacterDOPModality = Literal["visual", "audio"]


@dataclass(frozen=True)
class CharacterDOPLosses:
    total: torch.Tensor
    visual: Optional[torch.Tensor]
    audio: Optional[torch.Tensor]


def character_dop_multiplier(*, base_multiplier: float, every_n_steps: int) -> float:
    """Keep character DOP's expected weight stable when it runs sparsely."""
    if every_n_steps < 1:
        raise ValueError("character DOP every_n_steps must be at least 1")
    return float(base_multiplier) * every_n_steps


def _visual_preservation_weights(
    character_mask: torch.Tensor,
    token_error: torch.Tensor,
) -> torch.Tensor:
    """Convert a character-positive image/video mask into outside weights."""
    mask = character_mask.to(device=token_error.device, dtype=torch.float32).clamp(0.0, 1.0)
    if token_error.ndim == 3:
        # Image token errors are BHW. Collapse video masks across time when a
        # caller supplies one, then resize in image space.
        if mask.ndim == 5:
            mask = mask.amax(dim=2)
        if mask.ndim != 4:
            raise ValueError("image character masks must be BCHW or BCTHW")
        mask = mask.amax(dim=1, keepdim=True)
        mask = F.interpolate(mask, size=token_error.shape[-2:], mode="nearest")
        return 1.0 - mask[:, 0]

    if token_error.ndim != 4:
        raise ValueError("video character DOP expects BTHW token errors")
    if mask.ndim == 4:
        # A static BCHW mask applies to every frame.
        mask = mask.amax(dim=1, keepdim=True).unsqueeze(2)
    elif mask.ndim == 5:
        mask = mask.amax(dim=1, keepdim=True)
    else:
        raise ValueError("video character masks must be BCHW or BCTHW")
    mask = F.interpolate(mask, size=token_error.shape[-3:], mode="nearest")
    return 1.0 - mask[:, 0]


def _focused_token_mean(
    token_error: torch.Tensor,
    focus_fraction: float,
    weights: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    flat_error = token_error.flatten(start_dim=1)
    flat_weights = (
        torch.ones_like(flat_error, dtype=torch.float32)
        if weights is None
        else weights.flatten(start_dim=1)
    )
    sample_losses = []
    for sample_error, sample_weights in zip(flat_error, flat_weights):
        eligible = sample_weights > 1e-6
        if not torch.any(eligible):
            sample_losses.append(sample_error.sum() * 0.0)
            continue
        eligible_error = sample_error[eligible]
        eligible_weights = sample_weights[eligible]
        weighted_error = eligible_error * eligible_weights
        keep = max(1, math.ceil(weighted_error.numel() * focus_fraction))
        selected = weighted_error.topk(keep, sorted=False).indices
        selected_weights = eligible_weights[selected]
        sample_losses.append(
            weighted_error[selected].sum() / selected_weights.sum().clamp_min(1e-6)
        )
    return torch.stack(sample_losses).mean()


def character_dop_loss(
    prediction: torch.Tensor,
    prior: torch.Tensor,
    *,
    modality: CharacterDOPModality,
    focus_fraction: float,
    character_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Preserve the base model where a character LoRA drifts most.

    The error is reduced to visual or audio tokens before selecting the largest
    fraction. This keeps a small face, person, or speech interval from being
    averaged away by the rest of a video or soundtrack.
    """
    if prediction.shape != prior.shape:
        raise ValueError(
            "character DOP prediction and prior must have matching shapes, "
            f"got {tuple(prediction.shape)} and {tuple(prior.shape)}"
        )
    if not 0.0 < focus_fraction <= 1.0:
        raise ValueError("character DOP focus_fraction must be in the range (0, 1]")
    if modality == "visual":
        if prediction.ndim not in (4, 5):
            raise ValueError("visual character DOP expects BCHW or BCTHW predictions")
        token_error = (prediction - prior).square().mean(dim=1, dtype=torch.float32)
        weights = (
            _visual_preservation_weights(character_mask, token_error)
            if character_mask is not None
            else None
        )
    elif modality == "audio":
        if prediction.ndim != 3:
            raise ValueError("audio character DOP expects BSC predictions")
        if character_mask is not None:
            raise ValueError("character masks are only supported for visual predictions")
        token_error = (prediction - prior).square().mean(dim=-1, dtype=torch.float32)
        weights = None
    else:
        raise ValueError(f"unsupported character DOP modality: {modality}")

    return _focused_token_mean(token_error, focus_fraction, weights)


def character_dop_losses(
    *,
    visual_prediction: Optional[torch.Tensor] = None,
    visual_primary_prediction: Optional[torch.Tensor] = None,
    visual_prior: Optional[torch.Tensor] = None,
    audio_prediction: Optional[torch.Tensor] = None,
    audio_prior: Optional[torch.Tensor] = None,
    focus_fraction: float,
    base_multiplier: float,
    every_n_steps: int,
    visual_multiplier: float,
    audio_multiplier: float,
    character_mask: Optional[torch.Tensor] = None,
) -> CharacterDOPLosses:
    """Build interval-compensated visual and audio character DOP losses.

    With a character mask, visual preservation compares the trigger-conditioned
    primary prediction to the base-model class prior outside the mask. This is
    the direct anti-bleed constraint: the trigger may change the character, but
    not other people or the scene. Without a mask, the counterfactual class
    prediction is used so the character itself remains learnable.
    """
    if (visual_prediction is None) != (visual_prior is None):
        raise ValueError("visual character DOP requires both prediction and prior")
    if (audio_prediction is None) != (audio_prior is None):
        raise ValueError("audio character DOP requires both prediction and prior")
    if visual_prediction is None and audio_prediction is None:
        raise ValueError("character DOP requires a visual or audio prediction")

    scale = character_dop_multiplier(
        base_multiplier=base_multiplier,
        every_n_steps=every_n_steps,
    )
    visual_loss = None
    if visual_prediction is not None:
        visual_prediction_to_preserve = (
            visual_primary_prediction
            if character_mask is not None and visual_primary_prediction is not None
            else visual_prediction
        )
        visual_loss = character_dop_loss(
            visual_prediction_to_preserve,
            visual_prior,
            modality="visual",
            focus_fraction=focus_fraction,
            character_mask=character_mask,
        ) * scale * visual_multiplier

    audio_loss = None
    if audio_prediction is not None:
        audio_loss = character_dop_loss(
            audio_prediction,
            audio_prior,
            modality="audio",
            focus_fraction=focus_fraction,
        ) * scale * audio_multiplier

    if visual_loss is None:
        total = audio_loss
    elif audio_loss is None:
        total = visual_loss
    else:
        total = visual_loss + audio_loss
    return CharacterDOPLosses(total=total, visual=visual_loss, audio=audio_loss)
