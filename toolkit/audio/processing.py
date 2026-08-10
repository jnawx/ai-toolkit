import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class AudioSegment:
    start_seconds: float
    duration_seconds: float
    target_duration_seconds: float


def plan_audio_segments(
    source_duration_seconds: float,
    max_segment_seconds: float,
    bucket_step_seconds: float = 1.0,
) -> list[AudioSegment]:
    """Plan contiguous audio segments and their padded duration buckets."""
    if source_duration_seconds <= 0:
        raise ValueError("source audio duration must be greater than zero")
    if max_segment_seconds <= 0:
        raise ValueError("maximum audio segment duration must be greater than zero")
    if bucket_step_seconds <= 0:
        raise ValueError("audio duration bucket step must be greater than zero")

    segment_count = max(1, math.ceil(source_duration_seconds / max_segment_seconds))
    nominal_duration_seconds = source_duration_seconds / segment_count
    segments = []
    for segment_index in range(segment_count):
        start_seconds = segment_index * nominal_duration_seconds
        end_seconds = min(
            source_duration_seconds,
            (segment_index + 1) * nominal_duration_seconds,
        )
        duration_seconds = end_seconds - start_seconds
        target_duration_seconds = min(
            max_segment_seconds,
            max(1, math.ceil((duration_seconds / bucket_step_seconds) - 1e-9))
            * bucket_step_seconds,
        )
        segments.append(
            AudioSegment(
                start_seconds=start_seconds,
                duration_seconds=duration_seconds,
                target_duration_seconds=target_duration_seconds,
            )
        )
    return segments


def audio_segment_frame_range(
    segment: AudioSegment,
    sample_rate: int,
) -> tuple[int, int]:
    """Return the source frame offset and length for an audio segment."""
    if sample_rate <= 0:
        raise ValueError("audio sample rate must be greater than zero")
    frame_offset = round(segment.start_seconds * sample_rate)
    end_frame = round(
        (segment.start_seconds + segment.duration_seconds) * sample_rate
    )
    return frame_offset, max(1, end_frame - frame_offset)


def load_audio_segment(
    path: str,
    segment: AudioSegment,
    source_sample_rate: int | None,
) -> tuple[torch.Tensor, int]:
    """Load one source range, falling back for containers that cannot seek."""
    import torchaudio

    if source_sample_rate is not None:
        frame_offset, num_frames = audio_segment_frame_range(
            segment,
            source_sample_rate,
        )
        try:
            waveform, sample_rate = torchaudio.load(
                path,
                frame_offset=frame_offset,
                num_frames=num_frames,
            )
            if sample_rate == source_sample_rate and waveform.shape[-1] > 0:
                return waveform, sample_rate
        except Exception:
            # Seeking is backend/container dependent. A full decode below is the
            # compatibility path and will surface any actual decode failure.
            pass

    waveform, sample_rate = torchaudio.load(path)
    frame_offset, num_frames = audio_segment_frame_range(segment, sample_rate)
    return waveform[..., frame_offset : frame_offset + num_frames], sample_rate


def stack_audio_latents(latents: list[torch.Tensor]) -> torch.Tensor:
    """Stack packed audio latents, padding shorter sequences with silence rows."""
    if not latents:
        raise ValueError("cannot stack an empty audio latent batch")
    max_rows = max(latent.shape[0] for latent in latents)
    return torch.stack(
        [F.pad(latent, (0, 0, 0, max_rows - latent.shape[0])) for latent in latents]
    )


def waveform_to_stereo(waveform: torch.Tensor) -> torch.Tensor:
    """Convert mono or common surround layouts to stereo."""
    channels = waveform.shape[0]
    if channels == 2:
        return waveform
    if channels == 1:
        return waveform.expand(2, -1)
    if channels == 6:  # 5.1: FL, FR, FC, LFE, BL, BR
        fl, fr, fc, _, bl, br = waveform
        scale = 0.7071
        return torch.stack([fl + scale * fc + scale * bl, fr + scale * fc + scale * br])
    if channels == 8:  # 7.1: FL, FR, FC, LFE, BL, BR, SL, SR
        fl, fr, fc, _, bl, br, sl, sr = waveform
        scale = 0.7071
        return torch.stack(
            [
                fl + scale * fc + scale * (bl + sl),
                fr + scale * fc + scale * (br + sr),
            ]
        )
    return waveform.mean(0, keepdim=True).expand(2, -1)


def prepare_audio_for_training(
    waveform: torch.Tensor,
    sample_rate: int,
    target_sample_rate: int,
    duration_seconds: float,
    normalize: bool = False,
) -> torch.Tensor:
    """Prepare a deterministic stereo waveform for one duration bucket."""
    if duration_seconds <= 0:
        raise ValueError("audio_duration_seconds must be greater than zero")

    waveform = waveform_to_stereo(waveform)
    if sample_rate != target_sample_rate:
        import torchaudio

        waveform = torchaudio.functional.resample(
            waveform, sample_rate, target_sample_rate
        )

    target_samples = max(1, round(duration_seconds * target_sample_rate))
    current_samples = waveform.shape[-1]
    if current_samples > target_samples:
        start = (current_samples - target_samples) // 2
        waveform = waveform[..., start : start + target_samples]
    elif current_samples < target_samples:
        waveform = F.pad(waveform, (0, target_samples - current_samples))

    if normalize:
        peak = waveform.abs().amax()
        if peak > 0:
            waveform = waveform * (0.999 / peak)

    return waveform.contiguous()
