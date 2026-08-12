import json
import tempfile
import unittest
import base64
import io
import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image

from toolkit.character_dop_annotation import (
    find_matching_character_visual_mask,
    get_character_mask_preview,
    get_character_annotation_paths,
    is_character_annotation_artifact,
    get_character_annotation_state,
    save_character_audio_intervals,
    save_character_visual_mask,
    track_character_visual_mask,
)


class CharacterDOPAnnotationStorageTests(unittest.TestCase):
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

            def tracker(path, prompts, _progress):
                received.append((path, prompts))
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
            self.assertTrue(state["visual"]["exists"])
            self.assertEqual(state["visual"]["shape"], [2, 3, 4])

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
