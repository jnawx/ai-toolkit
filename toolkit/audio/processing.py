import torch
import torch.nn.functional as F


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
    """Prepare a deterministic, fixed-duration stereo training waveform."""
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
