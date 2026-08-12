import math
import json
import os
from dataclasses import dataclass
from typing import List, Literal, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F

from toolkit.character_dop_schema import validate_character_audio_intervals


CharacterDOPModality = Literal["visual", "audio"]
AudioIntervals = Sequence[Sequence[Tuple[float, float]]]


def _validated_audio_intervals(raw_intervals) -> List[Tuple[float, float]]:
    return validate_character_audio_intervals(raw_intervals)


def load_character_audio_intervals(
    *,
    media_path: str,
    intervals_path: str,
    dataset_root: Optional[str] = None,
) -> Optional[List[Tuple[float, float]]]:
    """Load absolute source-time speaking intervals for one media item."""
    if os.path.isdir(intervals_path):
        stem = os.path.splitext(os.path.basename(media_path))[0]
        sidecar_path = None
        if dataset_root is not None:
            try:
                relative_media = os.path.relpath(media_path, dataset_root)
                if not relative_media.startswith('..' + os.sep) and relative_media != '..':
                    relative_stem = os.path.splitext(relative_media)[0]
                    nested_sidecar = os.path.join(intervals_path, f"{relative_stem}.json")
                    if os.path.exists(nested_sidecar):
                        sidecar_path = nested_sidecar
            except ValueError:
                pass
        if sidecar_path is None:
            sidecar_path = os.path.join(intervals_path, f"{stem}.json")
    else:
        sidecar_path = intervals_path
    if not os.path.exists(sidecar_path):
        interval_dir = os.path.normpath(intervals_path)
        is_dataset_editor_path = (
            os.path.basename(interval_dir) == 'audio'
            and os.path.basename(os.path.dirname(interval_dir)) == '_character_dop'
        )
        if is_dataset_editor_path:
            return None
        raise FileNotFoundError(
            f"Character DOP audio interval sidecar not found: {sidecar_path}"
        )
    with open(sidecar_path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if isinstance(payload, dict):
        if "character_intervals" not in payload:
            raise ValueError(
                f"Character DOP audio sidecar must contain 'character_intervals': {sidecar_path}"
            )
        payload = payload["character_intervals"]
    return _validated_audio_intervals(payload)


def map_audio_intervals_to_training_clip(
    intervals: Sequence[Tuple[float, float]],
    *,
    source_start_seconds: float,
    source_duration_seconds: float,
    target_duration_seconds: float,
) -> List[Tuple[float, float]]:
    """Clip absolute source intervals and map them through training time-stretch."""
    if source_duration_seconds <= 0.0 or target_duration_seconds <= 0.0:
        return []
    source_end = source_start_seconds + source_duration_seconds
    scale = target_duration_seconds / source_duration_seconds
    mapped = []
    for start, end in intervals:
        clipped_start = max(start, source_start_seconds)
        clipped_end = min(end, source_end)
        if clipped_end <= clipped_start:
            continue
        mapped.append(
            (
                (clipped_start - source_start_seconds) * scale,
                (clipped_end - source_start_seconds) * scale,
            )
        )
    return mapped


def prepare_temporal_character_mask(
    mask: torch.Tensor,
    *,
    frame_indices: Optional[Sequence[int]],
    scale_size: Tuple[int, int],
    crop: Tuple[int, int, int, int],
    flip_x: bool,
    flip_y: bool,
    min_value: float,
) -> torch.Tensor:
    """Prepare a THW/BTHW mask with the video's frame and spatial transform."""
    mask = torch.as_tensor(mask, dtype=torch.float32)
    if mask.ndim == 2:
        mask = mask.unsqueeze(0)
    elif mask.ndim == 4:
        if mask.shape[0] == 1:
            mask = mask[0]
        elif mask.shape[1] == 1:
            mask = mask[:, 0]
        else:
            raise ValueError("temporal character masks must have one channel")
    if mask.ndim != 3:
        raise ValueError("temporal character masks must be HW, THW, 1THW, or T1HW")
    if mask.shape[0] > 1 and frame_indices is None:
        raise ValueError(
            "Temporal character mask needs the video's sampled frame indices. "
            "Delete and rebuild this item's latent cache."
        )
    if frame_indices is not None and mask.shape[0] != len(frame_indices):
        if len(frame_indices) == 0 or max(frame_indices) >= mask.shape[0]:
            raise ValueError("temporal character mask has fewer frames than the source video")
        mask = mask[list(frame_indices)]

    mask = mask.unsqueeze(1)
    if flip_x:
        mask = mask.flip(-1)
    if flip_y:
        mask = mask.flip(-2)
    scale_width, scale_height = scale_size
    mask = F.interpolate(mask, size=(scale_height, scale_width), mode="nearest")
    crop_x, crop_y, crop_width, crop_height = crop
    mask = mask[
        :,
        :,
        crop_y:crop_y + crop_height,
        crop_x:crop_x + crop_width,
    ]
    mask = mask.clamp(0.0, 1.0)
    mask = min_value + mask * (1.0 - min_value)
    return mask.permute(1, 0, 2, 3).contiguous()


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


def _presence_tensor(
    presence: Optional[Sequence[bool]],
    *,
    batch_size: int,
    device: torch.device,
) -> Optional[torch.Tensor]:
    if presence is None:
        return None
    presence_tensor = torch.as_tensor(presence, device=device, dtype=torch.bool).flatten()
    if presence_tensor.numel() != batch_size:
        raise ValueError(
            f"character annotation presence needs {batch_size} values, "
            f"got {presence_tensor.numel()}"
        )
    return presence_tensor


def _select_per_sample_prediction(
    *,
    primary: torch.Tensor,
    fallback: torch.Tensor,
    annotation_present: Optional[Sequence[bool]],
) -> torch.Tensor:
    presence = _presence_tensor(
        annotation_present,
        batch_size=primary.shape[0],
        device=primary.device,
    )
    if presence is None:
        return primary
    selection_shape = (primary.shape[0],) + (1,) * (primary.ndim - 1)
    return torch.where(presence.view(selection_shape), primary, fallback)


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


def audio_character_mask_from_intervals(
    prediction: torch.Tensor,
    intervals: AudioIntervals,
    *,
    latents_per_second: int,
) -> torch.Tensor:
    """Build a BSC-compatible character-speaking mask for channel-major audio."""
    if prediction.ndim != 3:
        raise ValueError("audio character masks require BSC predictions")
    if latents_per_second < 1:
        raise ValueError("audio latents_per_second must be positive")
    if len(intervals) != prediction.shape[0]:
        raise ValueError("audio character intervals must contain one entry per batch item")
    sequence_length = prediction.shape[1]
    if sequence_length % 2 != 0:
        raise ValueError("MiniMax H3 audio rows must contain two channel-major halves")
    channel_length = sequence_length // 2
    times = torch.arange(
        channel_length,
        device=prediction.device,
        dtype=torch.float32,
    ) / float(latents_per_second)
    masks = []
    for sample_intervals in intervals:
        channel_mask = torch.zeros_like(times)
        for start, end in sample_intervals:
            if start < 0.0 or end <= start:
                raise ValueError("audio character intervals must satisfy 0 <= start < end")
            channel_mask = torch.maximum(
                channel_mask,
                ((times >= start) & (times < end)).to(torch.float32),
            )
        masks.append(channel_mask.repeat(2))
    return torch.stack(masks)


def character_dop_loss(
    prediction: torch.Tensor,
    prior: torch.Tensor,
    *,
    modality: CharacterDOPModality,
    focus_fraction: float,
    character_mask: Optional[torch.Tensor] = None,
    character_mask_present: Optional[Sequence[bool]] = None,
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
        token_error = (prediction - prior).square().mean(dim=-1, dtype=torch.float32)
        if character_mask is not None:
            if character_mask.shape != token_error.shape:
                raise ValueError(
                    "audio character mask must match the B,S prediction rows, "
                    f"got {tuple(character_mask.shape)} and {tuple(token_error.shape)}"
                )
            weights = 1.0 - character_mask.to(
                device=token_error.device,
                dtype=torch.float32,
            ).clamp(0.0, 1.0)
        else:
            weights = None
    else:
        raise ValueError(f"unsupported character DOP modality: {modality}")

    if weights is not None:
        presence = _presence_tensor(
            character_mask_present,
            batch_size=token_error.shape[0],
            device=token_error.device,
        )
        if presence is not None:
            presence_shape = (token_error.shape[0],) + (1,) * (token_error.ndim - 1)
            weights = torch.where(
                presence.view(presence_shape),
                weights,
                torch.ones_like(weights),
            )

    return _focused_token_mean(token_error, focus_fraction, weights)


def character_dop_losses(
    *,
    visual_prediction: Optional[torch.Tensor] = None,
    visual_primary_prediction: Optional[torch.Tensor] = None,
    visual_prior: Optional[torch.Tensor] = None,
    audio_prediction: Optional[torch.Tensor] = None,
    audio_primary_prediction: Optional[torch.Tensor] = None,
    audio_prior: Optional[torch.Tensor] = None,
    audio_character_intervals: Optional[AudioIntervals] = None,
    visual_character_mask_present: Optional[Sequence[bool]] = None,
    audio_character_mask_present: Optional[Sequence[bool]] = None,
    audio_latents_per_second: int = 40,
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
            _select_per_sample_prediction(
                primary=visual_primary_prediction,
                fallback=visual_prediction,
                annotation_present=visual_character_mask_present,
            )
            if character_mask is not None and visual_primary_prediction is not None
            else visual_prediction
        )
        visual_loss = character_dop_loss(
            visual_prediction_to_preserve,
            visual_prior,
            modality="visual",
            focus_fraction=focus_fraction,
            character_mask=character_mask,
            character_mask_present=visual_character_mask_present,
        ) * scale * visual_multiplier

    audio_loss = None
    if audio_prediction is not None:
        audio_character_mask = None
        audio_prediction_to_preserve = audio_prediction
        if audio_character_intervals is not None and audio_primary_prediction is not None:
            audio_prediction_to_preserve = _select_per_sample_prediction(
                primary=audio_primary_prediction,
                fallback=audio_prediction,
                annotation_present=audio_character_mask_present,
            )
            audio_character_mask = audio_character_mask_from_intervals(
                audio_prediction_to_preserve,
                audio_character_intervals,
                latents_per_second=audio_latents_per_second,
            )
        audio_loss = character_dop_loss(
            audio_prediction_to_preserve,
            audio_prior,
            modality="audio",
            focus_fraction=focus_fraction,
            character_mask=audio_character_mask,
            character_mask_present=audio_character_mask_present,
        ) * scale * audio_multiplier

    if visual_loss is None:
        total = audio_loss
    elif audio_loss is None:
        total = visual_loss
    else:
        total = visual_loss + audio_loss
    return CharacterDOPLosses(total=total, visual=visual_loss, audio=audio_loss)
