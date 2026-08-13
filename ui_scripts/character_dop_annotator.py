import argparse
import json
import sys
from pathlib import Path

TOOLKIT_ROOT = Path(__file__).resolve().parents[1]
if str(TOOLKIT_ROOT) not in sys.path:
    sys.path.insert(0, str(TOOLKIT_ROOT))

from toolkit.character_dop_annotation import (
    create_character_identity,
    create_shared_character_identity,
    delete_character_identity,
    delete_shared_character_identity,
    detect_character_instances,
    get_character_annotation_state,
    get_character_mask_preview,
    get_character_mask_overlays,
    save_character_audio_intervals,
    track_character_visual_mask,
    update_character_identity,
    update_shared_character_identity,
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
    parser.add_argument(
        "action",
        choices=(
            "models", "list-identities", "create-global-identity", "update-global-identity",
            "delete-global-identity", "state", "save-identity", "update-identity",
            "delete-identity", "save-audio", "save-description", "detect", "track", "preview", "overlays",
        ),
    )
    parser.add_argument("--dataset-dir")
    parser.add_argument("--datasets-root")
    parser.add_argument("--media-path")
    parser.add_argument("--intervals", type=_json_arg)
    parser.add_argument("--prompts", type=_json_arg)
    parser.add_argument("--frame-index", type=int, default=0)
    parser.add_argument("--time-fraction", type=float, default=0.0)
    parser.add_argument("--model-id")
    parser.add_argument("--identity-id")
    parser.add_argument("--payload-stdin", action="store_true")
    args = parser.parse_args()
    if args.action == "models":
        print(json.dumps(get_character_mask_model_catalog()), flush=True)
        return
    payload = json.load(sys.stdin) if args.payload_stdin else {}
    if args.action in {
        "list-identities", "create-global-identity", "update-global-identity", "delete-global-identity"
    }:
        if not args.datasets_root:
            parser.error(f"{args.action} requires --datasets-root")
        datasets_root = Path(args.datasets_root)
        if args.action == "list-identities":
            from toolkit.character_dop_annotation import list_available_character_identities
            result = {"identities": list_available_character_identities(datasets_root)}
        elif args.action == "delete-global-identity":
            result = delete_shared_character_identity(
                datasets_root=datasets_root,
                identity_id=payload.get("identity_id", args.identity_id),
            )
        else:
            save_identity = (
                create_shared_character_identity
                if args.action == "create-global-identity"
                else update_shared_character_identity
            )
            identity = save_identity(
                datasets_root=datasets_root,
                identity_id=payload.get("identity_id", ""),
                display_name=payload.get("display_name", ""),
                trigger_word=payload.get("trigger_word", ""),
                class_prompt=payload.get("class_prompt", ""),
                caption_description=payload.get("caption_description"),
            )
            from toolkit.character_dop_annotation import list_available_character_identities
            result = {
                "identity": identity,
                "identities": list_available_character_identities(datasets_root),
            }
        print(json.dumps(result), flush=True)
        return
    if args.dataset_dir is None or args.media_path is None:
        parser.error(f"{args.action} requires --dataset-dir and --media-path")
    dataset_dir = Path(args.dataset_dir)
    datasets_root = Path(args.datasets_root) if args.datasets_root else None
    media_path = Path(args.media_path)

    identity_id = payload.get("identity_id", args.identity_id)
    if args.action == "state":
        result = get_character_annotation_state(
            dataset_dir=dataset_dir,
            media_path=media_path,
            identity_id=identity_id,
            datasets_root=datasets_root,
        )
    elif args.action == "save-identity":
        create_identity = (
            create_shared_character_identity
            if datasets_root is not None
            else create_character_identity
        )
        identity = create_identity(
            **(
                {"datasets_root": datasets_root}
                if datasets_root is not None
                else {"dataset_dir": dataset_dir}
            ),
            identity_id=payload.get("identity_id", ""),
            display_name=payload.get("display_name", ""),
            trigger_word=payload.get("trigger_word", ""),
            class_prompt=payload.get("class_prompt", ""),
            caption_description=payload.get("caption_description"),
        )
        result = get_character_annotation_state(
            dataset_dir=dataset_dir,
            media_path=media_path,
            identity_id=identity["id"],
            datasets_root=datasets_root,
        )
    elif args.action == "update-identity":
        update_identity = (
            update_shared_character_identity
            if datasets_root is not None
            else update_character_identity
        )
        identity = update_identity(
            **(
                {"datasets_root": datasets_root}
                if datasets_root is not None
                else {"dataset_dir": dataset_dir}
            ),
            identity_id=payload.get("identity_id", ""),
            display_name=payload.get("display_name", ""),
            trigger_word=payload.get("trigger_word", ""),
            class_prompt=payload.get("class_prompt", ""),
            caption_description=payload.get("caption_description"),
        )
        result = get_character_annotation_state(
            dataset_dir=dataset_dir,
            media_path=media_path,
            identity_id=identity["id"],
            datasets_root=datasets_root,
        )
    elif args.action == "delete-identity":
        deleted = (
            delete_shared_character_identity(
                datasets_root=datasets_root,
                identity_id=identity_id,
            )
            if datasets_root is not None
            else delete_character_identity(
                dataset_dir=dataset_dir,
                identity_id=identity_id,
            )
        )
        fallback_identity_id = (
            deleted["identities"][0]["id"] if deleted["identities"] else None
        )
        result = get_character_annotation_state(
            dataset_dir=dataset_dir,
            media_path=media_path,
            identity_id=fallback_identity_id,
            datasets_root=datasets_root,
        )
        result["deleted_identity"] = deleted["deleted_identity"]
        result["cleanup_pending"] = deleted["cleanup_pending"]
        result["cleanup_errors"] = deleted.get("cleanup_errors", [])
    elif args.action == "save-audio":
        intervals = payload.get("intervals", args.intervals)
        if intervals is None:
            parser.error("save-audio requires intervals")
        save_character_audio_intervals(
            dataset_dir=dataset_dir,
            media_path=media_path,
            intervals=intervals,
            identity_id=identity_id,
            datasets_root=datasets_root,
        )
        result = get_character_annotation_state(
            dataset_dir=dataset_dir,
            media_path=media_path,
            identity_id=identity_id,
            datasets_root=datasets_root,
        )
    elif args.action == "save-description":
        from toolkit.character_dop_annotation import save_character_caption_description
        save_character_caption_description(
            dataset_dir=dataset_dir,
            media_path=media_path,
            identity_id=identity_id,
            caption_description=payload.get("caption_description"),
            datasets_root=datasets_root,
        )
        result = get_character_annotation_state(
            dataset_dir=dataset_dir,
            media_path=media_path,
            identity_id=identity_id,
            datasets_root=datasets_root,
        )
    elif args.action == "preview":
        result = get_character_mask_preview(
            dataset_dir=dataset_dir,
            media_path=media_path,
            frame_index=args.frame_index,
            identity_id=identity_id,
            datasets_root=datasets_root,
        )
    elif args.action == "overlays":
        result = {
            "overlays": get_character_mask_overlays(
                dataset_dir=dataset_dir,
                media_path=media_path,
                time_fraction=args.time_fraction,
                datasets_root=datasets_root,
            )
        }
    elif args.action == "detect":
        result = detect_character_instances(
            dataset_dir=dataset_dir,
            media_path=media_path,
            time_seconds=payload.get("time_seconds", 0.0),
            concept=payload.get("concept", "person"),
            model_id=payload.get("model_id", args.model_id or DEFAULT_SAM3_DETECTOR_MODEL),
            detector=detect_with_sam3,
            progress=lambda message: print(message, flush=True),
            datasets_root=datasets_root,
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
            identity_id=identity_id,
            datasets_root=datasets_root,
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
