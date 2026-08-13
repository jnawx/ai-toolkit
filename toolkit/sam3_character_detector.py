from contextlib import nullcontext
import os
from pathlib import Path
from typing import Callable

import numpy as np
from PIL import Image


MAX_CHARACTER_CANDIDATES = 64
MIN_CHARACTER_AREA_FRACTION = 0.0005


def _load_media_frame(media_path: Path, time_seconds: float) -> Image.Image:
    if media_path.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp", ".bmp"}:
        with Image.open(media_path) as image:
            return image.convert("RGB")

    import av

    best_frame = None
    best_distance = float("inf")
    with av.open(str(media_path)) as container:
        if not container.streams.video:
            raise ValueError("SAM 3 auto-mask requires an image or video")
        stream = container.streams.video[0]
        fallback_fps = float(stream.average_rate or stream.base_rate or 24.0)
        if time_seconds > 0 and stream.time_base is not None:
            seek_time = max(0.0, time_seconds - 1.0)
            container.seek(int(seek_time / float(stream.time_base)), stream=stream)
        for index, frame in enumerate(container.decode(stream)):
            frame_time = float(frame.time) if frame.time is not None else index / fallback_fps
            distance = abs(frame_time - time_seconds)
            if distance < best_distance:
                best_frame = frame.to_image().convert("RGB")
                best_distance = distance
            if frame_time >= time_seconds and best_frame is not None:
                break
    if best_frame is None:
        raise ValueError("No video frame could be decoded for SAM 3 auto-mask")
    return best_frame


def detect_with_sam3(
    media_path: Path,
    time_seconds: float,
    concept: str,
    model_id: str,
    progress: Callable[[str], None],
) -> list[dict]:
    """Detect every visible instance matching a text concept on one frame."""
    import torch
    from transformers import Sam3Model, Sam3Processor

    progress("Decoding the selected source frame")
    image = _load_media_frame(media_path, time_seconds)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32

    progress(f"Loading {model_id} on {device.type}")
    try:
        token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN") or None
        model = Sam3Model.from_pretrained(model_id, dtype=dtype, token=token).to(device).eval()
        processor = Sam3Processor.from_pretrained(model_id, token=token)
    except (OSError, ValueError, ImportError) as exc:
        raise RuntimeError(
            f"SAM 3 ({model_id}) could not be loaded: {exc}. Verify that your saved "
            "Hugging Face token can access the gated model and that the installed "
            "transformers version includes SAM 3."
        ) from exc

    inputs = processor(images=image, text=concept, return_tensors="pt").to(device)
    progress(f'Detecting every instance matching "{concept}"')
    autocast = (
        torch.autocast(device_type="cuda", dtype=dtype)
        if device.type == "cuda"
        else nullcontext()
    )
    with torch.inference_mode(), autocast:
        outputs = model(**inputs)
    result = processor.post_process_instance_segmentation(
        outputs,
        threshold=0.3,
        mask_threshold=0.5,
        target_sizes=inputs["original_sizes"].tolist(),
    )[0]

    image_area = image.width * image.height
    detections = []
    for mask, score, box in zip(result["masks"], result["scores"], result["boxes"]):
        binary_mask = (mask.detach().to("cpu").numpy() > 0).astype(np.uint8)
        if int(binary_mask.sum()) < image_area * MIN_CHARACTER_AREA_FRACTION:
            continue
        detections.append(
            {
                "mask": binary_mask,
                "score": float(score.detach().to("cpu").item()),
                "box": [float(value) for value in box.detach().to("cpu").tolist()],
            }
        )
    detections.sort(key=lambda detection: detection["score"], reverse=True)
    if len(detections) > MAX_CHARACTER_CANDIDATES:
        progress(
            f"Found {len(detections)} matching instances; returning the "
            f"{MAX_CHARACTER_CANDIDATES} highest-confidence masks"
        )
    else:
        progress(f"Found {len(detections)} matching instance(s)")
    return detections[:MAX_CHARACTER_CANDIDATES]
