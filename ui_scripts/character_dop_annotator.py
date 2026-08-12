import argparse
import json
import sys
from pathlib import Path

TOOLKIT_ROOT = Path(__file__).resolve().parents[1]
if str(TOOLKIT_ROOT) not in sys.path:
    sys.path.insert(0, str(TOOLKIT_ROOT))

from toolkit.character_dop_annotation import (
    detect_character_instances,
    get_character_annotation_state,
    get_character_mask_preview,
    save_character_audio_intervals,
    track_character_visual_mask,
)
from toolkit.character_mask_models import (
    DEFAULT_SAM2_TRACKER_MODEL,
    DEFAULT_SAM3_DETECTOR_MODEL,
    get_character_mask_model_catalog,
)
from toolkit.sam2_character_tracker import track_with_sam2
from toolkit.sam3_character_detector import detect_with_sam3


def _json_arg(value: str):
    try:
        return json.loads(value)
    except json.JSONDecodeError as exc:
        raise argparse.ArgumentTypeError("expected valid JSON") from exc


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare built-in Character DOP annotations")
    parser.add_argument("action", choices=("models", "state", "save-audio", "detect", "track", "preview"))
    parser.add_argument("--dataset-dir")
    parser.add_argument("--media-path")
    parser.add_argument("--intervals", type=_json_arg)
    parser.add_argument("--prompts", type=_json_arg)
    parser.add_argument("--frame-index", type=int, default=0)
    parser.add_argument("--model-id")
    parser.add_argument("--payload-stdin", action="store_true")
    args = parser.parse_args()
    if args.action == "models":
        print(json.dumps(get_character_mask_model_catalog()), flush=True)
        return
    if args.dataset_dir is None or args.media_path is None:
        parser.error(f"{args.action} requires --dataset-dir and --media-path")
    dataset_dir = Path(args.dataset_dir)
    media_path = Path(args.media_path)
    payload = json.load(sys.stdin) if args.payload_stdin else {}

    if args.action == "state":
        result = get_character_annotation_state(dataset_dir=dataset_dir, media_path=media_path)
    elif args.action == "save-audio":
        intervals = payload.get("intervals", args.intervals)
        if intervals is None:
            parser.error("save-audio requires intervals")
        save_character_audio_intervals(
            dataset_dir=dataset_dir,
            media_path=media_path,
            intervals=intervals,
        )
        result = get_character_annotation_state(dataset_dir=dataset_dir, media_path=media_path)
    elif args.action == "preview":
        result = get_character_mask_preview(
            dataset_dir=dataset_dir,
            media_path=media_path,
            frame_index=args.frame_index,
        )
    elif args.action == "detect":
        result = detect_character_instances(
            dataset_dir=dataset_dir,
            media_path=media_path,
            time_seconds=payload.get("time_seconds", 0.0),
            concept=payload.get("concept", "person"),
            model_id=payload.get("model_id", args.model_id or DEFAULT_SAM3_DETECTOR_MODEL),
            detector=detect_with_sam3,
            progress=lambda message: print(message, flush=True),
        )
    else:
        prompts = payload.get("prompts", args.prompts)
        initial_masks = payload.get("initial_masks", [])
        if prompts is None and not initial_masks:
            parser.error("track requires prompts or initial_masks")

        def report(message: str) -> None:
            print(message, flush=True)

        result = track_character_visual_mask(
            dataset_dir=dataset_dir,
            media_path=media_path,
            prompts=prompts or [],
            initial_mask_data_urls=initial_masks,
            initial_time_seconds=payload.get("initial_time_seconds"),
            tracker=lambda path, prompts, initial_mask, initial_time, progress: track_with_sam2(
                path,
                prompts,
                initial_mask,
                initial_time,
                progress,
                model_id=payload.get("model_id", args.model_id or DEFAULT_SAM2_TRACKER_MODEL),
            ),
            progress=report,
        )
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
