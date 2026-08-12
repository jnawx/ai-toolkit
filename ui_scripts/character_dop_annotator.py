import argparse
import json
import sys
from pathlib import Path

TOOLKIT_ROOT = Path(__file__).resolve().parents[1]
if str(TOOLKIT_ROOT) not in sys.path:
    sys.path.insert(0, str(TOOLKIT_ROOT))

from toolkit.character_dop_annotation import (
    get_character_annotation_state,
    get_character_mask_preview,
    save_character_audio_intervals,
    track_character_visual_mask,
)
from toolkit.sam2_character_tracker import DEFAULT_SAM2_MODEL, track_with_sam2


def _json_arg(value: str):
    try:
        return json.loads(value)
    except json.JSONDecodeError as exc:
        raise argparse.ArgumentTypeError("expected valid JSON") from exc


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare built-in Character DOP annotations")
    parser.add_argument("action", choices=("state", "save-audio", "track", "preview"))
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument("--media-path", required=True)
    parser.add_argument("--intervals", type=_json_arg)
    parser.add_argument("--prompts", type=_json_arg)
    parser.add_argument("--frame-index", type=int, default=0)
    parser.add_argument("--model-id", default=DEFAULT_SAM2_MODEL)
    args = parser.parse_args()
    dataset_dir = Path(args.dataset_dir)
    media_path = Path(args.media_path)

    if args.action == "state":
        result = get_character_annotation_state(dataset_dir=dataset_dir, media_path=media_path)
    elif args.action == "save-audio":
        if args.intervals is None:
            parser.error("save-audio requires --intervals")
        save_character_audio_intervals(
            dataset_dir=dataset_dir,
            media_path=media_path,
            intervals=args.intervals,
        )
        result = get_character_annotation_state(dataset_dir=dataset_dir, media_path=media_path)
    elif args.action == "preview":
        result = get_character_mask_preview(
            dataset_dir=dataset_dir,
            media_path=media_path,
            frame_index=args.frame_index,
        )
    else:
        if args.prompts is None:
            parser.error("track requires --prompts")

        def report(message: str) -> None:
            print(message, flush=True)

        result = track_character_visual_mask(
            dataset_dir=dataset_dir,
            media_path=media_path,
            prompts=args.prompts,
            tracker=lambda path, prompts, progress: track_with_sam2(
                path,
                prompts,
                progress,
                model_id=args.model_id,
            ),
            progress=report,
        )
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
