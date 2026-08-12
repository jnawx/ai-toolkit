from pathlib import Path
from typing import Callable

import numpy as np
from PIL import Image

from toolkit.character_mask_models import (
    DEFAULT_SAM2_TRACKER_MODEL,
    validate_character_mask_model,
)


DEFAULT_SAM2_MODEL = DEFAULT_SAM2_TRACKER_MODEL
MAX_VIDEO_FRAMES = 1200
MAX_SAVED_MASK_EDGE = 768


def _load_media_frames(media_path: Path) -> tuple[list[Image.Image], list[float]]:
    suffix = media_path.suffix.lower()
    if suffix in {".jpg", ".jpeg", ".png", ".webp", ".bmp"}:
        with Image.open(media_path) as image:
            return [image.convert("RGB")], [0.0]

    import av

    frames = []
    times = []
    with av.open(str(media_path)) as container:
        if not container.streams.video:
            raise ValueError("SAM2 character tracking requires an image or video")
        stream = container.streams.video[0]
        fallback_fps = float(stream.average_rate or stream.base_rate or 24.0)
        for index, frame in enumerate(container.decode(stream)):
            if index >= MAX_VIDEO_FRAMES:
                raise ValueError(
                    f"Character mask videos are limited to {MAX_VIDEO_FRAMES} source frames"
                )
            frames.append(frame.to_image().convert("RGB"))
            times.append(float(frame.time) if frame.time is not None else index / fallback_fps)
    if not frames:
        raise ValueError("No video frames could be decoded for character tracking")
    return frames, times


def _prompt_frame_index(time_seconds: float, frame_times: list[float]) -> int:
    return min(
        range(len(frame_times)),
        key=lambda index: abs(frame_times[index] - time_seconds),
    )


def _mask_from_output(processor, inference_session, output) -> np.ndarray:
    masks = processor.post_process_masks(
        [output.pred_masks],
        original_sizes=[
            [inference_session.video_height, inference_session.video_width]
        ],
        binarize=True,
    )[0]
    mask = masks.detach().to("cpu").numpy()
    while mask.ndim > 2:
        mask = mask[0]
    return (mask > 0).astype(np.uint8)


def _resize_mask_stack(mask: np.ndarray, max_edge: int) -> np.ndarray:
    height, width = mask.shape[-2:]
    if max(height, width) <= max_edge:
        return mask
    scale = max_edge / float(max(height, width))
    target = (max(1, round(width * scale)), max(1, round(height * scale)))
    resized = [
        np.asarray(
            Image.fromarray(frame * 255, mode="L").resize(target, Image.Resampling.NEAREST)
        )
        > 0
        for frame in mask
    ]
    return np.stack(resized).astype(np.uint8)


def track_with_sam2(
    media_path: Path,
    prompts: list[dict],
    initial_mask: np.ndarray | None,
    initial_time_seconds: float | None,
    progress: Callable[[str], None],
    *,
    model_id: str = DEFAULT_SAM2_MODEL,
) -> np.ndarray:
    """Track one prompted character through an image/video using Transformers SAM2."""
    import torch
    from transformers import Sam2VideoModel, Sam2VideoProcessor

    model_id = validate_character_mask_model(model_id, kind="tracker")
    progress("Decoding source media")
    frames, frame_times = _load_media_frames(media_path)
    width, height = frames[0].size
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32

    progress(f"Loading {model_id} on {device.type}")
    model = Sam2VideoModel.from_pretrained(model_id, dtype=dtype).to(device).eval()
    processor = Sam2VideoProcessor.from_pretrained(model_id)
    session = processor.init_video_session(
        video=frames,
        inference_device=device,
        video_storage_device="cpu",
        inference_state_device="cpu",
        dtype=dtype,
    )

    prompts_by_frame: dict[int, list[dict]] = {}
    for prompt in prompts:
        frame_index = _prompt_frame_index(prompt["time_seconds"], frame_times)
        prompts_by_frame.setdefault(frame_index, []).extend(prompt["points"])

    initial_mask_frame = None
    if initial_mask is not None:
        if initial_time_seconds is None:
            raise ValueError("SAM2 auto-mask tracking requires its source time")
        initial_mask_frame = _prompt_frame_index(initial_time_seconds, frame_times)
        if initial_mask.ndim != 2:
            raise ValueError("SAM2 initial character mask must have shape HW")
        if initial_mask_frame in prompts_by_frame:
            del prompts_by_frame[initial_mask_frame]

    progress("Applying the character seed mask and manual corrections")
    with torch.inference_mode():
        if initial_mask is not None and initial_mask_frame is not None:
            processor.add_inputs_to_inference_session(
                inference_session=session,
                frame_idx=initial_mask_frame,
                obj_ids=1,
                input_masks=[initial_mask],
            )
            model(inference_session=session, frame_idx=initial_mask_frame)
        for frame_index, points in sorted(prompts_by_frame.items()):
            processor.add_inputs_to_inference_session(
                inference_session=session,
                frame_idx=frame_index,
                obj_ids=1,
                input_points=[[[[point["x"] * width, point["y"] * height] for point in points]]],
                input_labels=[[[point["label"] for point in points]]],
            )
            model(inference_session=session, frame_idx=frame_index)

        masks: dict[int, np.ndarray] = {}
        seed_frames = list(prompts_by_frame)
        if initial_mask_frame is not None:
            seed_frames.append(initial_mask_frame)
        if not seed_frames:
            raise ValueError("SAM2 tracking requires points or an initial mask")
        first_prompt_frame = min(seed_frames)
        progress(f"Tracking character through {len(frames)} frames")
        for output in model.propagate_in_video_iterator(
            session,
            start_frame_idx=first_prompt_frame,
            show_progress_bar=False,
        ):
            masks[int(output.frame_idx)] = _mask_from_output(processor, session, output)
        if first_prompt_frame > 0:
            for output in model.propagate_in_video_iterator(
                session,
                start_frame_idx=first_prompt_frame,
                reverse=True,
                show_progress_bar=False,
            ):
                masks[int(output.frame_idx)] = _mask_from_output(processor, session, output)

    missing_frames = [index for index in range(len(frames)) if index not in masks]
    if missing_frames:
        raise RuntimeError(
            f"SAM2 did not return masks for {len(missing_frames)} source frames"
        )
    progress("Saving tracked character mask")
    mask_stack = np.stack([masks[index] for index in range(len(frames))])
    return _resize_mask_stack(mask_stack, MAX_SAVED_MASK_EDGE)
