import json
import tempfile
import unittest
import base64
import io
import subprocess
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest.mock import patch

import numpy as np
from PIL import Image

from toolkit.character_dop_annotation import (
    MAX_CHARACTER_IDENTITIES,
    create_shared_character_identity,
    delete_character_identity,
    delete_shared_character_identity,
    detect_character_instances,
    find_matching_character_visual_mask,
    get_character_mask_preview,
    get_character_mask_overlays,
    get_character_annotation_paths,
    get_character_identity_views,
    is_character_annotation_artifact,
    get_character_annotation_state,
    list_character_identities,
    list_available_character_identities,
    save_character_audio_intervals,
    save_character_caption_description,
    save_character_identity,
    save_character_visual_mask,
    track_character_visual_mask,
    update_character_identity,
    update_shared_character_identity,
)


class CharacterDOPAnnotationStorageTests(unittest.TestCase):
    def test_identity_stores_a_rich_caption_description_separately_from_its_dop_class(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            dataset_dir = Path(tmp_dir) / "dataset"
            dataset_dir.mkdir()

            identity = save_character_identity(
                dataset_dir=dataset_dir,
                identity_id="alice",
                display_name="Alice",
                trigger_word="AliceToken",
                class_prompt="a woman",
                caption_description="a tall woman with a silver bob haircut and a red wool coat",
            )

            self.assertEqual(identity["class_prompt"], "a woman")
            self.assertEqual(
                identity["caption_description"],
                "a tall woman with a silver bob haircut and a red wool coat",
            )

    def test_media_caption_description_overrides_the_global_identity_description(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            dataset_dir = Path(tmp_dir) / "dataset"
            dataset_dir.mkdir()
            media_path = dataset_dir / "alice.jpg"
            Image.new("RGB", (4, 4)).save(media_path)
            save_character_identity(
                dataset_dir=dataset_dir,
                identity_id="alice",
                display_name="Alice",
                trigger_word="AliceToken",
                class_prompt="a woman",
                caption_description="a woman with short silver hair",
            )
            save_character_visual_mask(
                dataset_dir=dataset_dir,
                media_path=media_path,
                identity_id="alice",
                mask=np.ones((4, 4), dtype=np.uint8),
            )

            save_character_caption_description(
                dataset_dir=dataset_dir,
                media_path=media_path,
                identity_id="alice",
                caption_description="a woman wearing a red raincoat",
            )

            view = get_character_identity_views(
                dataset_dir=dataset_dir,
                media_path=media_path,
            )[0]
            self.assertEqual(view.caption_description, "a woman wearing a red raincoat")

    def test_updating_identity_metadata_preserves_its_annotations(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            dataset_dir = Path(tmp_dir) / "dataset"
            dataset_dir.mkdir()
            media_path = dataset_dir / "portrait.jpg"
            Image.new("RGB", (4, 4)).save(media_path)
            save_character_identity(
                dataset_dir=dataset_dir,
                identity_id="alice",
                display_name="Alice",
                trigger_word="OldAliceToken",
                class_prompt="a person",
            )
            mask_path = save_character_visual_mask(
                dataset_dir=dataset_dir,
                media_path=media_path,
                identity_id="alice",
                mask=np.ones((4, 4), dtype=np.uint8),
            )

            updated = update_character_identity(
                dataset_dir=dataset_dir,
                identity_id="alice",
                display_name="Alice Example",
                trigger_word="AliceToken",
                class_prompt="a woman",
            )

            self.assertEqual(
                updated,
                {
                    "id": "alice",
                    "display_name": "Alice Example",
                    "trigger_word": "AliceToken",
                    "class_prompt": "a woman",
                    "caption_description": "a woman",
                },
            )
            self.assertTrue(mask_path.is_file())
            self.assertTrue(
                get_character_annotation_state(
                    dataset_dir=dataset_dir,
                    media_path=media_path,
                    identity_id="alice",
                )["visual"]["exists"]
            )

    def test_stale_identity_update_cannot_recreate_a_deleted_identity(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            dataset_dir = Path(tmp_dir) / "dataset"
            dataset_dir.mkdir()
            save_character_identity(
                dataset_dir=dataset_dir,
                identity_id="alice",
                display_name="Alice",
                trigger_word="AliceToken",
                class_prompt="a person",
            )
            with self.assertRaisesRegex(ValueError, "already exists"):
                save_character_identity(
                    dataset_dir=dataset_dir,
                    identity_id="alice",
                    display_name="Alice overwritten",
                    trigger_word="AliceOtherToken",
                    class_prompt="a woman",
                )
            delete_character_identity(dataset_dir=dataset_dir, identity_id="alice")

            with self.assertRaisesRegex(ValueError, "unknown character identity"):
                update_character_identity(
                    dataset_dir=dataset_dir,
                    identity_id="alice",
                    display_name="Alice stale tab",
                    trigger_word="AliceToken",
                    class_prompt="a woman",
                )

            self.assertEqual(list_character_identities(dataset_dir), [])

    def test_failed_catalog_commit_restores_staged_identity_annotations(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            dataset_dir = Path(tmp_dir) / "dataset"
            dataset_dir.mkdir()
            media_path = dataset_dir / "portrait.jpg"
            Image.new("RGB", (4, 4)).save(media_path)
            save_character_identity(
                dataset_dir=dataset_dir,
                identity_id="alice",
                display_name="Alice",
                trigger_word="AliceToken",
                class_prompt="a person",
            )
            mask_path = save_character_visual_mask(
                dataset_dir=dataset_dir,
                media_path=media_path,
                identity_id="alice",
                mask=np.ones((4, 4), dtype=np.uint8),
            )

            with patch(
                "toolkit.character_dop_annotation._write_character_identity_catalog",
                side_effect=OSError("catalog unavailable"),
            ):
                with self.assertRaisesRegex(OSError, "catalog unavailable"):
                    delete_character_identity(dataset_dir=dataset_dir, identity_id="alice")

            self.assertTrue(mask_path.is_file())
            self.assertEqual(list_character_identities(dataset_dir)[0]["id"], "alice")

    def test_annotation_write_after_delete_cannot_recreate_identity_tree(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            dataset_dir = Path(tmp_dir) / "dataset"
            dataset_dir.mkdir()
            media_path = dataset_dir / "voice.wav"
            media_path.touch()
            save_character_identity(
                dataset_dir=dataset_dir,
                identity_id="alice",
                display_name="Alice",
                trigger_word="AliceToken",
                class_prompt="a person",
            )
            delete_character_identity(dataset_dir=dataset_dir, identity_id="alice")

            with self.assertRaisesRegex(ValueError, "unknown character identity"):
                save_character_audio_intervals(
                    dataset_dir=dataset_dir,
                    media_path=media_path,
                    identity_id="alice",
                    intervals=[[0.0, 1.0]],
                )

            self.assertFalse(
                (dataset_dir / "_character_dop" / "identities" / "alice").exists()
            )

    def test_delete_waits_for_in_progress_identity_annotation_write(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            dataset_dir = Path(tmp_dir) / "dataset"
            dataset_dir.mkdir()
            media_path = dataset_dir / "voice.wav"
            media_path.touch()
            save_character_identity(
                dataset_dir=dataset_dir,
                identity_id="alice",
                display_name="Alice",
                trigger_word="AliceToken",
                class_prompt="a person",
            )
            write_started = threading.Event()
            allow_write = threading.Event()
            from toolkit import character_dop_annotation as annotation_module

            original_write_json = annotation_module._write_json

            def blocking_write_json(path, payload):
                if path.name == "voice.json":
                    write_started.set()
                    self.assertTrue(allow_write.wait(timeout=5))
                return original_write_json(path, payload)

            with patch(
                "toolkit.character_dop_annotation._write_json",
                side_effect=blocking_write_json,
            ), ThreadPoolExecutor(max_workers=2) as executor:
                save_future = executor.submit(
                    save_character_audio_intervals,
                    dataset_dir=dataset_dir,
                    media_path=media_path,
                    identity_id="alice",
                    intervals=[[0.0, 1.0]],
                )
                self.assertTrue(write_started.wait(timeout=5))
                delete_future = executor.submit(
                    delete_character_identity,
                    dataset_dir=dataset_dir,
                    identity_id="alice",
                )
                self.assertFalse(delete_future.done())
                allow_write.set()
                save_future.result(timeout=5)
                delete_future.result(timeout=5)

            self.assertFalse(
                (dataset_dir / "_character_dop" / "identities" / "alice").exists()
            )

    def test_deleting_identity_removes_only_its_catalog_entry_and_annotations(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            dataset_dir = Path(tmp_dir) / "dataset"
            dataset_dir.mkdir()
            media_path = dataset_dir / "together.mp4"
            media_path.touch()
            for identity_id, trigger_word in (("alice", "AliceToken"), ("bob", "BobToken")):
                save_character_identity(
                    dataset_dir=dataset_dir,
                    identity_id=identity_id,
                    display_name=identity_id.title(),
                    trigger_word=trigger_word,
                    class_prompt="a person",
                )
                save_character_audio_intervals(
                    dataset_dir=dataset_dir,
                    media_path=media_path,
                    identity_id=identity_id,
                    intervals=[[0.0, 1.0]],
                )

            result = delete_character_identity(
                dataset_dir=dataset_dir,
                identity_id="alice",
            )

            self.assertEqual(result["deleted_identity"]["id"], "alice")
            self.assertEqual(
                [identity["id"] for identity in result["identities"]],
                ["bob"],
            )
            self.assertFalse(
                (dataset_dir / "_character_dop" / "identities" / "alice").exists()
            )
            self.assertTrue(
                (dataset_dir / "_character_dop" / "identities" / "bob").is_dir()
            )
            with self.assertRaisesRegex(ValueError, "unknown character identity"):
                get_character_annotation_state(
                    dataset_dir=dataset_dir,
                    media_path=media_path,
                    identity_id="alice",
                )

    def test_delete_reports_when_staged_annotation_cleanup_is_pending(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            dataset_dir = Path(tmp_dir) / "dataset"
            dataset_dir.mkdir()
            media_path = dataset_dir / "portrait.jpg"
            Image.new("RGB", (4, 4)).save(media_path)
            save_character_identity(
                dataset_dir=dataset_dir,
                identity_id="alice",
                display_name="Alice",
                trigger_word="AliceToken",
                class_prompt="a person",
            )
            save_character_visual_mask(
                dataset_dir=dataset_dir,
                media_path=media_path,
                identity_id="alice",
                mask=np.ones((4, 4), dtype=np.uint8),
            )

            with patch(
                "toolkit.character_dop_annotation.shutil.rmtree",
                side_effect=OSError("file is locked"),
            ):
                result = delete_character_identity(
                    dataset_dir=dataset_dir,
                    identity_id="alice",
                )

            self.assertTrue(result["cleanup_pending"])
            self.assertEqual(list_character_identities(dataset_dir), [])
            self.assertTrue(
                any(
                    (dataset_dir / "_character_dop" / ".deleted-identities").iterdir()
                )
            )

    def test_concurrent_identity_writers_preserve_every_catalog_entry(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            dataset_dir = Path(tmp_dir) / "dataset"
            dataset_dir.mkdir()

            def save_identity(index):
                save_character_identity(
                    dataset_dir=dataset_dir,
                    identity_id=f"person-{index}",
                    display_name=f"Person {index}",
                    trigger_word=f"Person{index}Token",
                    class_prompt="a person",
                )

            with ThreadPoolExecutor(max_workers=8) as executor:
                list(executor.map(save_identity, range(12)))

            self.assertEqual(
                {identity["id"] for identity in list_character_identities(dataset_dir)},
                {f"person-{index}" for index in range(12)},
            )

    def test_identity_catalog_enforces_text_and_count_limits(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            dataset_dir = Path(tmp_dir) / "dataset"
            dataset_dir.mkdir()
            with self.assertRaisesRegex(ValueError, "trigger word must be at most"):
                save_character_identity(
                    dataset_dir=dataset_dir,
                    identity_id="alice",
                    display_name="Alice",
                    trigger_word="x" * 129,
                    class_prompt="a woman",
                )

            annotation_dir = dataset_dir / "_character_dop"
            annotation_dir.mkdir()
            oversized_catalog = {
                "version": 1,
                "identities": [
                    {
                        "id": f"person-{index}",
                        "display_name": f"Person {index}",
                        "trigger_word": f"Person{index}Token",
                        "class_prompt": "a person",
                    }
                    for index in range(MAX_CHARACTER_IDENTITIES + 1)
                ],
            }
            (annotation_dir / "identities.json").write_text(
                json.dumps(oversized_catalog),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "supports at most"):
                list_character_identities(dataset_dir)

    def test_externally_edited_catalog_rejects_overlapping_triggers(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            dataset_dir = Path(tmp_dir) / "dataset"
            annotation_dir = dataset_dir / "_character_dop"
            annotation_dir.mkdir(parents=True)
            (annotation_dir / "identities.json").write_text(
                json.dumps(
                    {
                        "version": 1,
                        "identities": [
                            {
                                "id": "ann",
                                "display_name": "Ann",
                                "trigger_word": "AnnToken",
                                "class_prompt": "a woman",
                            },
                            {
                                "id": "anna",
                                "display_name": "Anna",
                                "trigger_word": "AnnToken2",
                                "class_prompt": "a woman",
                            },
                        ],
                    }
                ),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "cannot contain one another"):
                list_character_identities(dataset_dir)

    def test_annotation_storage_rejects_symlink_escape(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            temp_root = Path(tmp_dir)
            dataset_dir = temp_root / "dataset"
            outside_dir = temp_root / "outside"
            dataset_dir.mkdir()
            outside_dir.mkdir()
            media_path = dataset_dir / "portrait.jpg"
            media_path.touch()
            try:
                (dataset_dir / "_character_dop").symlink_to(
                    outside_dir,
                    target_is_directory=True,
                )
            except OSError as exc:
                self.skipTest(f"directory symlinks are unavailable: {exc}")

            with self.assertRaisesRegex(ValueError, "escapes the dataset"):
                save_character_identity(
                    dataset_dir=dataset_dir,
                    identity_id="alice",
                    display_name="Alice",
                    trigger_word="AliceToken",
                    class_prompt="a woman",
                )

    def test_two_identities_can_annotate_the_same_media_without_duplicate_files(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            dataset_dir = Path(tmp_dir) / "dataset"
            dataset_dir.mkdir()
            media_path = dataset_dir / "together.mp4"
            media_path.touch()

            save_character_identity(
                dataset_dir=dataset_dir,
                identity_id="alice",
                display_name="Alice",
                trigger_word="AliceToken",
                class_prompt="a woman",
            )
            save_character_identity(
                dataset_dir=dataset_dir,
                identity_id="bob",
                display_name="Bob",
                trigger_word="BobToken",
                class_prompt="a man",
            )
            alice_mask = np.zeros((1, 3, 4), dtype=np.uint8)
            alice_mask[:, :, :2] = 1
            bob_mask = np.zeros((1, 3, 4), dtype=np.uint8)
            bob_mask[:, :, 2:] = 1

            alice_path = save_character_visual_mask(
                dataset_dir=dataset_dir,
                media_path=media_path,
                identity_id="alice",
                mask=alice_mask,
            )
            bob_path = save_character_visual_mask(
                dataset_dir=dataset_dir,
                media_path=media_path,
                identity_id="bob",
                mask=bob_mask,
            )
            save_character_audio_intervals(
                dataset_dir=dataset_dir,
                media_path=media_path,
                identity_id="alice",
                intervals=[[0.0, 1.0]],
            )
            save_character_audio_intervals(
                dataset_dir=dataset_dir,
                media_path=media_path,
                identity_id="bob",
                intervals=[[1.0, 2.0]],
            )

            self.assertNotEqual(alice_path, bob_path)
            self.assertEqual(
                [identity["id"] for identity in list_character_identities(dataset_dir)],
                ["alice", "bob"],
            )
            self.assertEqual(
                get_character_annotation_state(
                    dataset_dir=dataset_dir,
                    media_path=media_path,
                    identity_id="alice",
                )["audio"]["intervals"],
                [[0.0, 1.0]],
            )
            self.assertEqual(
                get_character_annotation_state(
                    dataset_dir=dataset_dir,
                    media_path=media_path,
                    identity_id="bob",
                )["audio"]["intervals"],
                [[1.0, 2.0]],
            )

    def test_identity_catalog_rejects_unsafe_ids_and_duplicate_triggers(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            dataset_dir = Path(tmp_dir) / "dataset"
            dataset_dir.mkdir()
            with self.assertRaisesRegex(ValueError, "identity id"):
                save_character_identity(
                    dataset_dir=dataset_dir,
                    identity_id="../alice",
                    display_name="Alice",
                    trigger_word="AliceToken",
                    class_prompt="a woman",
                )
            save_character_identity(
                dataset_dir=dataset_dir,
                identity_id="alice",
                display_name="Alice",
                trigger_word="AliceToken",
                class_prompt="a woman",
            )
            with self.assertRaisesRegex(ValueError, "already used"):
                save_character_identity(
                    dataset_dir=dataset_dir,
                    identity_id="alice-copy",
                    display_name="Alice copy",
                    trigger_word="alicetoken",
                    class_prompt="a woman",
                )

    def test_named_annotations_become_independent_training_views(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            dataset_dir = Path(tmp_dir) / "dataset"
            dataset_dir.mkdir()
            media_path = dataset_dir / "together.mp4"
            media_path.touch()
            for identity_id, trigger_word, class_prompt, interval in (
                ("alice", "AliceToken", "a woman", [0.0, 1.0]),
                ("bob", "BobToken", "a man", [1.0, 2.0]),
            ):
                save_character_identity(
                    dataset_dir=dataset_dir,
                    identity_id=identity_id,
                    display_name=identity_id.title(),
                    trigger_word=trigger_word,
                    class_prompt=class_prompt,
                )
                save_character_visual_mask(
                    dataset_dir=dataset_dir,
                    media_path=media_path,
                    identity_id=identity_id,
                    mask=np.ones((1, 2, 2), dtype=np.uint8),
                )
                save_character_audio_intervals(
                    dataset_dir=dataset_dir,
                    media_path=media_path,
                    identity_id=identity_id,
                    intervals=[interval],
                )

            views = get_character_identity_views(
                dataset_dir=dataset_dir,
                media_path=media_path,
            )

            self.assertEqual(
                [
                    (view.identity_id, view.trigger_word, view.class_prompt, view.audio_intervals)
                    for view in views
                ],
                [
                    ("alice", "AliceToken", "a woman", [(0.0, 1.0)]),
                    ("bob", "BobToken", "a man", [(1.0, 2.0)]),
                ],
            )
            self.assertTrue(all(view.visual_path.is_file() for view in views))

    def test_auto_mask_returns_each_detected_person_as_a_selectable_mask(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            dataset_dir = Path(tmp_dir) / "dataset"
            dataset_dir.mkdir()
            media_path = dataset_dir / "group.jpg"
            Image.new("RGB", (8, 6)).save(media_path)
            first = np.zeros((6, 8), dtype=np.uint8)
            first[:, :3] = 1
            second = np.zeros((6, 8), dtype=np.uint8)
            second[:, 5:] = 1
            received = []

            def detector(path, time_seconds, concept, model_id, _progress):
                received.append((path, time_seconds, concept, model_id))
                return [
                    {"mask": first, "score": 0.91, "box": [0, 0, 3, 6]},
                    {"mask": second, "score": 0.84, "box": [5, 0, 8, 6]},
                ]

            result = detect_character_instances(
                dataset_dir=dataset_dir,
                media_path=media_path,
                time_seconds=0.0,
                concept="person",
                model_id="facebook/sam3",
                detector=detector,
            )

            self.assertEqual(received[0][2:], ("person", "facebook/sam3"))
            self.assertEqual(len(result["candidates"]), 2)
            self.assertEqual(result["candidates"][0]["id"], 1)
            self.assertTrue(result["candidates"][0]["mask_data_url"].startswith("data:image/png;base64,"))
            self.assertEqual(result["candidates"][1]["box"], [5.0, 0.0, 8.0, 6.0])

    def test_auto_mask_downsizes_large_candidate_payloads_and_boxes_together(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            dataset_dir = Path(tmp_dir) / "dataset"
            dataset_dir.mkdir()
            media_path = dataset_dir / "large.jpg"
            Image.new("RGB", (1000, 500)).save(media_path)
            mask = np.ones((500, 1000), dtype=np.uint8)

            result = detect_character_instances(
                dataset_dir=dataset_dir,
                media_path=media_path,
                time_seconds=0,
                concept="person",
                model_id="facebook/sam3",
                detector=lambda *_args: [
                    {"mask": mask, "score": 0.95, "box": [100, 50, 900, 450]}
                ],
            )

            candidate = result["candidates"][0]
            encoded = candidate["mask_data_url"].split(",", 1)[1]
            with Image.open(io.BytesIO(base64.b64decode(encoded))) as decoded:
                self.assertEqual(decoded.size, (768, 384))
            self.assertEqual((result["width"], result["height"]), (768, 384))
            self.assertEqual(candidate["box"], [76.8, 38.4, 691.2, 345.6])

    def test_auto_mask_marks_candidates_that_overlap_an_existing_identity_mask(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            dataset_dir = Path(tmp_dir) / "dataset"
            dataset_dir.mkdir()
            media_path = dataset_dir / "people.jpg"
            Image.new("RGB", (6, 4)).save(media_path)
            save_character_identity(
                dataset_dir=dataset_dir,
                identity_id="alice",
                display_name="Alice",
                trigger_word="AliceToken",
                class_prompt="a woman",
            )
            alice_mask = np.zeros((4, 6), dtype=np.uint8)
            alice_mask[:, :3] = 1
            save_character_visual_mask(
                dataset_dir=dataset_dir,
                media_path=media_path,
                identity_id="alice",
                mask=alice_mask,
            )
            other_mask = np.zeros((4, 6), dtype=np.uint8)
            other_mask[:, 3:] = 1

            candidates = detect_character_instances(
                dataset_dir=dataset_dir,
                media_path=media_path,
                time_seconds=0,
                concept="person",
                model_id="facebook/sam3",
                detector=lambda *_args: [
                    {"mask": alice_mask, "score": 0.9, "box": [0, 0, 3, 4]},
                    {"mask": other_mask, "score": 0.8, "box": [3, 0, 6, 4]},
                ],
            )["candidates"]

            self.assertEqual(candidates[0]["existing_identity_id"], "alice")
            self.assertIsNone(candidates[1]["existing_identity_id"])

    def test_user_can_save_and_reload_speaking_intervals_for_a_dataset_item(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            dataset_dir = Path(tmp_dir) / "dataset"
            dataset_dir.mkdir()
            media_path = dataset_dir / "scene.mp4"
            media_path.touch()

            saved_path = save_character_audio_intervals(
                dataset_dir=dataset_dir,
                media_path=media_path,
                intervals=[[0.4, 1.8], [3.1, 4.0]],
            )
            state = get_character_annotation_state(
                dataset_dir=dataset_dir,
                media_path=media_path,
            )

            self.assertEqual(
                json.loads(saved_path.read_text(encoding="utf-8")),
                {"character_intervals": [[0.4, 1.8], [3.1, 4.0]]},
            )
            self.assertEqual(state["audio"]["intervals"], [[0.4, 1.8], [3.1, 4.0]])
            self.assertTrue(state["audio"]["exists"])

    def test_user_can_preview_a_saved_video_mask_at_a_specific_frame(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            dataset_dir = Path(tmp_dir) / "dataset"
            dataset_dir.mkdir()
            media_path = dataset_dir / "scene.mp4"
            media_path.touch()
            mask = np.zeros((3, 4, 5), dtype=np.uint8)
            mask[1, 1:3, 2:4] = 1

            save_character_visual_mask(
                dataset_dir=dataset_dir,
                media_path=media_path,
                mask=mask,
            )
            preview = get_character_mask_preview(
                dataset_dir=dataset_dir,
                media_path=media_path,
                frame_index=1,
            )

            encoded = preview["data_url"].split(",", 1)[1]
            image = Image.open(io.BytesIO(base64.b64decode(encoded)))
            pixels = np.asarray(image)
            self.assertEqual(tuple(pixels.shape), (4, 5))
            self.assertEqual(int((pixels > 0).sum()), 4)

    def test_all_identity_overlays_use_a_shared_relative_video_time(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            dataset_dir = Path(tmp_dir) / "dataset"
            dataset_dir.mkdir()
            media_path = dataset_dir / "scene.mp4"
            media_path.touch()
            for identity_id, trigger in (("alice", "AliceToken"), ("bob", "BobToken")):
                save_character_identity(
                    dataset_dir=dataset_dir,
                    identity_id=identity_id,
                    display_name=identity_id.title(),
                    trigger_word=trigger,
                    class_prompt="a person",
                )
            alice = np.zeros((3, 4, 4), dtype=np.uint8)
            alice[2, 0, 0] = 1
            bob = np.zeros((5, 4, 4), dtype=np.uint8)
            bob[4, 3, 3] = 1
            save_character_visual_mask(dataset_dir=dataset_dir, media_path=media_path, identity_id="alice", mask=alice)
            save_character_visual_mask(dataset_dir=dataset_dir, media_path=media_path, identity_id="bob", mask=bob)

            overlays = get_character_mask_overlays(
                dataset_dir=dataset_dir,
                media_path=media_path,
                time_fraction=1.0,
            )

            self.assertEqual([overlay["identity_id"] for overlay in overlays], ["alice", "bob"])
            self.assertEqual(overlays[0]["centroid"], [0.0, 0.0])
            self.assertEqual(overlays[1]["centroid"], [1.0, 1.0])

    def test_image_tracking_saves_a_normal_png_mask_for_existing_image_dataloaders(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            dataset_dir = Path(tmp_dir) / "dataset"
            dataset_dir.mkdir()
            media_path = dataset_dir / "portrait.jpg"
            media_path.touch()

            mask_path = save_character_visual_mask(
                dataset_dir=dataset_dir,
                media_path=media_path,
                mask=np.ones((1, 3, 4), dtype=np.uint8),
            )
            state = get_character_annotation_state(
                dataset_dir=dataset_dir,
                media_path=media_path,
            )

            self.assertEqual(mask_path.suffix, ".png")
            with Image.open(mask_path) as saved_mask:
                self.assertEqual(saved_mask.size, (4, 3))
            self.assertEqual(state["visual"]["shape"], [1, 3, 4])

    def test_visual_tracking_validates_prompts_and_saves_the_trackers_mask(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            dataset_dir = Path(tmp_dir) / "dataset"
            dataset_dir.mkdir()
            media_path = dataset_dir / "scene.mp4"
            media_path.touch()
            received = []

            def tracker(path, prompts, initial_mask, initial_time, _progress):
                received.append((path, prompts, initial_mask, initial_time))
                return np.ones((2, 3, 4), dtype=np.uint8)

            state = track_character_visual_mask(
                dataset_dir=dataset_dir,
                media_path=media_path,
                prompts=[
                    {
                        "time_seconds": 0.5,
                        "points": [
                            {"x": 0.25, "y": 0.75, "label": 1},
                            {"x": 0.9, "y": 0.1, "label": 0},
                        ],
                    }
                ],
                tracker=tracker,
            )

            self.assertEqual(received[0][0], media_path.resolve())
            self.assertEqual(received[0][1][0]["points"][1]["label"], 0)
            self.assertIsNone(received[0][2])
            self.assertTrue(state["visual"]["exists"])
            self.assertEqual(state["visual"]["shape"], [2, 3, 4])

    def test_visual_tracking_writes_only_the_selected_named_identity(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            dataset_dir = Path(tmp_dir) / "dataset"
            dataset_dir.mkdir()
            media_path = dataset_dir / "together.mp4"
            media_path.touch()
            save_character_identity(
                dataset_dir=dataset_dir,
                identity_id="alice",
                display_name="Alice",
                trigger_word="AliceToken",
                class_prompt="a woman",
            )
            save_character_identity(
                dataset_dir=dataset_dir,
                identity_id="bob",
                display_name="Bob",
                trigger_word="BobToken",
                class_prompt="a man",
            )

            state = track_character_visual_mask(
                dataset_dir=dataset_dir,
                media_path=media_path,
                identity_id="bob",
                prompts=[
                    {
                        "time_seconds": 0.0,
                        "points": [{"x": 0.5, "y": 0.5, "label": 1}],
                    }
                ],
                tracker=lambda *_args: np.ones((2, 3, 4), dtype=np.uint8),
            )

            self.assertEqual(state["identity"]["id"], "bob")
            self.assertTrue(state["visual"]["exists"])
            self.assertFalse(
                get_character_annotation_state(
                    dataset_dir=dataset_dir,
                    media_path=media_path,
                    identity_id="alice",
                )["visual"]["exists"]
            )

    def test_selected_auto_masks_are_combined_and_seed_visual_tracking(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            dataset_dir = Path(tmp_dir) / "dataset"
            dataset_dir.mkdir()
            media_path = dataset_dir / "group.mp4"
            media_path.touch()
            left = np.zeros((4, 6), dtype=np.uint8)
            left[:, :2] = 1
            right = np.zeros((4, 6), dtype=np.uint8)
            right[:, 4:] = 1
            received = []

            def detector(_path, _time, _concept, _model_id, _progress):
                return [
                    {"mask": left, "score": 0.9, "box": [0, 0, 2, 4]},
                    {"mask": right, "score": 0.8, "box": [4, 0, 6, 4]},
                ]

            candidates = detect_character_instances(
                dataset_dir=dataset_dir,
                media_path=media_path,
                time_seconds=1.25,
                concept="person",
                model_id="facebook/sam3",
                detector=detector,
            )["candidates"]

            def tracker(path, prompts, initial_mask, initial_time, _progress):
                received.append((path, prompts, initial_mask, initial_time))
                return initial_mask[None]

            state = track_character_visual_mask(
                dataset_dir=dataset_dir,
                media_path=media_path,
                prompts=[],
                initial_mask_data_urls=[candidate["mask_data_url"] for candidate in candidates],
                initial_time_seconds=1.25,
                tracker=tracker,
            )

            expected = np.maximum(left, right)
            np.testing.assert_array_equal(received[0][2], expected)
            self.assertEqual(received[0][3], 1.25)
            self.assertTrue(state["visual"]["exists"])

    def test_safe_automask_refuses_to_overwrite_an_existing_identity_mask(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            dataset_dir = Path(tmp_dir) / "dataset"
            dataset_dir.mkdir()
            media_path = dataset_dir / "person.jpg"
            media_path.touch()
            save_character_identity(
                dataset_dir=dataset_dir,
                identity_id="alice",
                display_name="Alice",
                trigger_word="AliceToken",
                class_prompt="a woman",
            )
            save_character_visual_mask(
                dataset_dir=dataset_dir,
                media_path=media_path,
                identity_id="alice",
                mask=np.ones((4, 4), dtype=np.uint8),
            )
            tracker_called = False

            def tracker(*_args):
                nonlocal tracker_called
                tracker_called = True
                return np.zeros((1, 4, 4), dtype=np.uint8)

            with self.assertRaisesRegex(FileExistsError, "already has a visual mask"):
                track_character_visual_mask(
                    dataset_dir=dataset_dir,
                    media_path=media_path,
                    identity_id="alice",
                    prompts=[{"time_seconds": 0, "points": [{"x": 0.5, "y": 0.5, "label": 1}]}],
                    tracker=tracker,
                    preserve_existing=True,
                )

            self.assertFalse(tracker_called)

    def test_training_finds_a_nested_visual_annotation_created_by_the_ui(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            dataset_dir = Path(tmp_dir) / "dataset"
            media_path = dataset_dir / "scenes" / "scene.mp4"
            mask_root = dataset_dir / "_character_dop" / "visual"
            media_path.parent.mkdir(parents=True)
            media_path.touch()
            mask_path = mask_root / "scenes" / "scene.npy"
            mask_path.parent.mkdir(parents=True)
            np.save(mask_path, np.ones((1, 2, 2), dtype=np.uint8))

            found = find_matching_character_visual_mask(
                media_path=media_path,
                mask_root=mask_root,
                dataset_dir=dataset_dir,
                is_video=True,
            )

            self.assertEqual(found, mask_path)

    def test_ui_script_exposes_saved_annotation_state(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            dataset_dir = Path(tmp_dir) / "dataset"
            dataset_dir.mkdir()
            media_path = dataset_dir / "voice.wav"
            media_path.touch()
            save_character_audio_intervals(
                dataset_dir=dataset_dir,
                media_path=media_path,
                intervals=[[0.25, 0.75]],
            )

            completed = subprocess.run(
                [
                    sys.executable,
                    str(Path(__file__).parents[1] / "ui_scripts" / "character_dop_annotator.py"),
                    "state",
                    "--dataset-dir",
                    str(dataset_dir),
                    "--media-path",
                    str(media_path),
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            state = json.loads(completed.stdout.strip().splitlines()[-1])

            self.assertEqual(state["audio"]["intervals"], [[0.25, 0.75]])

    def test_ui_script_creates_and_selects_a_named_identity(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            dataset_dir = Path(tmp_dir) / "dataset"
            dataset_dir.mkdir()
            media_path = dataset_dir / "portrait.jpg"
            Image.new("RGB", (4, 4)).save(media_path)
            script_path = Path(__file__).parents[1] / "ui_scripts" / "character_dop_annotator.py"

            created = subprocess.run(
                [
                    sys.executable,
                    str(script_path),
                    "save-identity",
                    "--dataset-dir",
                    str(dataset_dir),
                    "--media-path",
                    str(media_path),
                    "--payload-stdin",
                ],
                input=json.dumps(
                    {
                        "identity_id": "alice",
                        "display_name": "Alice",
                        "trigger_word": "AliceToken",
                        "class_prompt": "a woman",
                    }
                ),
                check=True,
                capture_output=True,
                text=True,
            )
            state = json.loads(created.stdout.strip().splitlines()[-1])

            self.assertEqual(state["identity"]["id"], "alice")
            self.assertEqual(state["identities"][0]["trigger_word"], "AliceToken")

            updated = subprocess.run(
                [
                    sys.executable,
                    str(script_path),
                    "update-identity",
                    "--dataset-dir",
                    str(dataset_dir),
                    "--media-path",
                    str(media_path),
                    "--payload-stdin",
                ],
                input=json.dumps(
                    {
                        "identity_id": "alice",
                        "display_name": "Alice Example",
                        "trigger_word": "AliceUpdatedToken",
                        "class_prompt": "a person",
                    }
                ),
                check=True,
                capture_output=True,
                text=True,
            )
            updated_state = json.loads(updated.stdout.strip().splitlines()[-1])

            self.assertEqual(updated_state["identity"]["display_name"], "Alice Example")
            self.assertEqual(updated_state["identity"]["trigger_word"], "AliceUpdatedToken")

    def test_ui_script_shares_identity_definitions_across_datasets(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            datasets_root = Path(tmp_dir) / "datasets"
            first_dataset = datasets_root / "first"
            second_dataset = datasets_root / "second"
            first_dataset.mkdir(parents=True)
            second_dataset.mkdir()
            first_media = first_dataset / "portrait.jpg"
            second_media = second_dataset / "portrait.jpg"
            Image.new("RGB", (4, 4)).save(first_media)
            Image.new("RGB", (4, 4)).save(second_media)
            script_path = Path(__file__).parents[1] / "ui_scripts" / "character_dop_annotator.py"

            subprocess.run(
                [
                    sys.executable,
                    str(script_path),
                    "save-identity",
                    "--datasets-root",
                    str(datasets_root),
                    "--dataset-dir",
                    str(first_dataset),
                    "--media-path",
                    str(first_media),
                    "--payload-stdin",
                ],
                input=json.dumps(
                    {
                        "identity_id": "alice",
                        "display_name": "Alice",
                        "trigger_word": "AliceToken",
                        "class_prompt": "a woman",
                    }
                ),
                check=True,
                capture_output=True,
                text=True,
            )

            loaded = subprocess.run(
                [
                    sys.executable,
                    str(script_path),
                    "state",
                    "--datasets-root",
                    str(datasets_root),
                    "--dataset-dir",
                    str(second_dataset),
                    "--media-path",
                    str(second_media),
                    "--identity-id",
                    "alice",
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            state = json.loads(loaded.stdout.strip().splitlines()[-1])

            self.assertEqual(state["identity"]["id"], "alice")
            self.assertEqual(
                [identity["id"] for identity in state["identities"]],
                ["alice"],
            )

    def test_shared_identity_activates_per_dataset_on_first_annotation(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            datasets_root = Path(tmp_dir) / "datasets"
            first_dataset = datasets_root / "first"
            second_dataset = datasets_root / "second"
            first_dataset.mkdir(parents=True)
            second_dataset.mkdir()
            second_media = second_dataset / "portrait.jpg"
            Image.new("RGB", (4, 4)).save(second_media)
            create_shared_character_identity(
                datasets_root=datasets_root,
                identity_id="alice",
                display_name="Alice",
                trigger_word="AliceToken",
                class_prompt="a woman",
            )

            self.assertEqual(list_character_identities(second_dataset), [])
            state = get_character_annotation_state(
                datasets_root=datasets_root,
                dataset_dir=second_dataset,
                media_path=second_media,
                identity_id="alice",
            )
            self.assertEqual(state["identity"]["id"], "alice")
            self.assertFalse(state["visual"]["exists"])

            save_character_visual_mask(
                datasets_root=datasets_root,
                dataset_dir=second_dataset,
                media_path=second_media,
                identity_id="alice",
                mask=np.ones((4, 4), dtype=np.uint8),
            )

            self.assertEqual(
                [identity["id"] for identity in list_character_identities(second_dataset)],
                ["alice"],
            )
            self.assertEqual(
                get_character_identity_views(
                    dataset_dir=second_dataset,
                    media_path=second_media,
                )[0].trigger_word,
                "AliceToken",
            )
            self.assertEqual(list_character_identities(first_dataset), [])

    def test_existing_dataset_identity_is_migrated_to_shared_registry(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            datasets_root = Path(tmp_dir) / "datasets"
            first_dataset = datasets_root / "first"
            second_dataset = datasets_root / "second"
            first_dataset.mkdir(parents=True)
            second_dataset.mkdir()
            second_media = second_dataset / "portrait.jpg"
            Image.new("RGB", (4, 4)).save(second_media)
            save_character_identity(
                dataset_dir=first_dataset,
                identity_id="alice",
                display_name="Alice",
                trigger_word="AliceToken",
                class_prompt="a woman",
            )

            state = get_character_annotation_state(
                datasets_root=datasets_root,
                dataset_dir=second_dataset,
                media_path=second_media,
                identity_id="alice",
            )

            self.assertEqual(state["identity"]["id"], "alice")
            self.assertEqual(
                json.loads(
                    (datasets_root / "_character_dop_identities.json").read_text(
                        encoding="utf-8"
                    )
                )["identities"][0]["trigger_word"],
                "AliceToken",
            )

    def test_shared_identity_update_is_used_by_every_activated_dataset(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            datasets_root = Path(tmp_dir) / "datasets"
            dataset_dir = datasets_root / "portraits"
            dataset_dir.mkdir(parents=True)
            media_path = dataset_dir / "portrait.jpg"
            Image.new("RGB", (4, 4)).save(media_path)
            create_shared_character_identity(
                datasets_root=datasets_root,
                identity_id="alice",
                display_name="Alice",
                trigger_word="AliceToken",
                class_prompt="a woman",
            )
            save_character_visual_mask(
                datasets_root=datasets_root,
                dataset_dir=dataset_dir,
                media_path=media_path,
                identity_id="alice",
                mask=np.ones((4, 4), dtype=np.uint8),
            )

            update_shared_character_identity(
                datasets_root=datasets_root,
                identity_id="alice",
                display_name="Alice Example",
                trigger_word="AliceUpdatedToken",
                class_prompt="a person",
            )

            identity = list_character_identities(dataset_dir)[0]
            self.assertEqual(identity["display_name"], "Alice Example")
            self.assertEqual(identity["trigger_word"], "AliceUpdatedToken")
            self.assertEqual(
                get_character_identity_views(
                    dataset_dir=dataset_dir,
                    media_path=media_path,
                )[0].class_prompt,
                "a person",
            )

    def test_shared_identity_delete_removes_annotations_from_every_dataset(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            datasets_root = Path(tmp_dir) / "datasets"
            datasets = [datasets_root / "first", datasets_root / "second"]
            media_paths = []
            for dataset_dir in datasets:
                dataset_dir.mkdir(parents=True, exist_ok=True)
                media_path = dataset_dir / "portrait.jpg"
                Image.new("RGB", (4, 4)).save(media_path)
                media_paths.append(media_path)
            create_shared_character_identity(
                datasets_root=datasets_root,
                identity_id="alice",
                display_name="Alice",
                trigger_word="AliceToken",
                class_prompt="a woman",
            )
            for dataset_dir, media_path in zip(datasets, media_paths):
                save_character_visual_mask(
                    datasets_root=datasets_root,
                    dataset_dir=dataset_dir,
                    media_path=media_path,
                    identity_id="alice",
                    mask=np.ones((4, 4), dtype=np.uint8),
                )

            result = delete_shared_character_identity(
                datasets_root=datasets_root,
                identity_id="alice",
            )

            self.assertEqual(result["deleted_identity"]["id"], "alice")
            self.assertFalse(result["cleanup_pending"])
            self.assertEqual(list_available_character_identities(datasets_root), [])
            for dataset_dir in datasets:
                self.assertEqual(list_character_identities(dataset_dir), [])
                self.assertFalse(
                    (dataset_dir / "_character_dop" / "identities" / "alice").exists()
                )

    def test_ui_script_deletes_a_named_identity_and_selects_a_safe_fallback(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            dataset_dir = Path(tmp_dir) / "dataset"
            dataset_dir.mkdir()
            media_path = dataset_dir / "portrait.jpg"
            Image.new("RGB", (4, 4)).save(media_path)
            for identity_id, trigger_word in (("alice", "AliceToken"), ("bob", "BobToken")):
                save_character_identity(
                    dataset_dir=dataset_dir,
                    identity_id=identity_id,
                    display_name=identity_id.title(),
                    trigger_word=trigger_word,
                    class_prompt="a person",
                )
            script_path = Path(__file__).parents[1] / "ui_scripts" / "character_dop_annotator.py"

            deleted = subprocess.run(
                [
                    sys.executable,
                    str(script_path),
                    "delete-identity",
                    "--dataset-dir",
                    str(dataset_dir),
                    "--media-path",
                    str(media_path),
                    "--identity-id",
                    "alice",
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            state = json.loads(deleted.stdout.strip().splitlines()[-1])

            self.assertEqual(state["deleted_identity"]["id"], "alice")
            self.assertFalse(state["cleanup_pending"])
            self.assertEqual(state["identity"]["id"], "bob")
            self.assertEqual([identity["id"] for identity in state["identities"]], ["bob"])

    def test_ui_script_exposes_supported_character_mask_models(self):
        completed = subprocess.run(
            [
                sys.executable,
                str(Path(__file__).parents[1] / "ui_scripts" / "character_dop_annotator.py"),
                "models",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        catalog = json.loads(completed.stdout.strip().splitlines()[-1])

        self.assertEqual(catalog["detectors"][0]["id"], "facebook/sam3")
        self.assertEqual(
            [model["id"] for model in catalog["trackers"]],
            [
                "facebook/sam2.1-hiera-tiny",
                "facebook/sam2.1-hiera-small",
                "facebook/sam2.1-hiera-base-plus",
                "facebook/sam2.1-hiera-large",
            ],
        )

    def test_saving_annotations_invalidates_only_that_items_generated_latents(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            dataset_dir = Path(tmp_dir) / "dataset"
            dataset_dir.mkdir()
            media_path = dataset_dir / "scene.mp4"
            media_path.touch()
            cache_dir = dataset_dir / "_latent_cache"
            cache_dir.mkdir()
            stale = cache_dir / "scene_oldhash.safetensors"
            unrelated = cache_dir / "other_oldhash.safetensors"
            stale.touch()
            unrelated.touch()

            save_character_audio_intervals(
                dataset_dir=dataset_dir,
                media_path=media_path,
                intervals=[],
            )

            self.assertFalse(stale.exists())
            self.assertTrue(unrelated.exists())

    def test_annotation_paths_preserve_dots_in_media_stems(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            dataset_dir = Path(tmp_dir) / "dataset"
            dataset_dir.mkdir()
            media_path = dataset_dir / "take.v2.mp4"
            media_path.touch()

            paths = get_character_annotation_paths(
                dataset_dir=dataset_dir,
                media_path=media_path,
            )

            self.assertEqual(paths.visual.name, "take.v2.npy")
            self.assertEqual(paths.audio.name, "take.v2.json")

    def test_generated_visual_masks_are_excluded_from_dataset_media_discovery(self):
        dataset_dir = Path("dataset")

        self.assertTrue(
            is_character_annotation_artifact(
                dataset_dir / "_character_dop" / "visual" / "portrait.png",
                dataset_dir,
            )
        )
        self.assertFalse(
            is_character_annotation_artifact(dataset_dir / "portrait.png", dataset_dir)
        )

    def test_annotation_storage_rejects_symlink_escape_from_dataset(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            dataset_dir = root / "dataset"
            outside_dir = root / "outside"
            dataset_dir.mkdir()
            outside_dir.mkdir()
            outside_media = outside_dir / "scene.mp4"
            outside_media.touch()
            link = dataset_dir / "linked.mp4"
            try:
                link.symlink_to(outside_media)
            except OSError:
                self.skipTest("symlinks are unavailable on this platform")

            with self.assertRaisesRegex(ValueError, "inside the dataset"):
                get_character_annotation_paths(
                    dataset_dir=dataset_dir,
                    media_path=link,
                )


if __name__ == "__main__":
    unittest.main()
