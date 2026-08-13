import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from toolkit.character_dop_annotation import (
    save_character_identity,
    save_character_visual_mask,
)
from toolkit.config_modules import DatasetConfig
from toolkit.data_loader import expand_character_identity_file_items
from toolkit.data_transfer_object.data_loader import FileItemDTO


class CharacterIdentityTrainingViewTests(unittest.TestCase):
    def test_named_identity_catalog_is_a_valid_character_dop_trigger_source(self):
        from extensions_built_in.sd_trainer.SDTrainer import has_dop_trigger_source

        with tempfile.TemporaryDirectory() as tmp_dir:
            save_character_identity(
                dataset_dir=Path(tmp_dir),
                identity_id="alice",
                display_name="Alice",
                trigger_word="AliceToken",
                class_prompt="a woman",
            )
            dataset = type(
                "Dataset",
                (),
                {
                    "trigger_word": None,
                    "folder_path": tmp_dir,
                    "character_dop_use_dataset_annotations": True,
                    "is_reg": False,
                },
            )()

            self.assertTrue(
                has_dop_trigger_source(
                    trigger_word=None,
                    dataset_configs=[dataset],
                    preservation_mode="character",
                )
            )
            self.assertFalse(
                has_dop_trigger_source(
                    trigger_word=None,
                    dataset_configs=[dataset],
                    preservation_mode="standard",
                )
            )
            catalog_free_dir = Path(tmp_dir) / "catalog-free"
            catalog_free_dir.mkdir()
            catalog_free_dataset = type(
                "Dataset",
                (),
                {
                    "trigger_word": None,
                    "folder_path": str(catalog_free_dir),
                    "character_dop_use_dataset_annotations": True,
                    "is_reg": False,
                },
            )()
            self.assertFalse(
                has_dop_trigger_source(
                    trigger_word=None,
                    dataset_configs=[dataset, catalog_free_dataset],
                    preservation_mode="character",
                )
            )
            dataset.character_dop_use_dataset_annotations = False
            self.assertFalse(
                has_dop_trigger_source(
                    trigger_word=None,
                    dataset_configs=[dataset],
                    preservation_mode="character",
                )
            )

    def test_uncached_counterfactual_uses_the_active_identity_class_prompt(self):
        from extensions_built_in.sd_trainer.SDTrainer import build_dop_prompt

        file_item = type(
            "FileItem",
            (),
            {
                "trigger_word": "BobToken",
                "character_dop_class_prompt": "a man",
            },
        )()

        self.assertEqual(
            build_dop_prompt(
                "AliceToken talks to BobToken.",
                file_item=file_item,
                fallback_trigger="JobToken",
                fallback_class_prompt="a person",
            ),
            "AliceToken talks to a man.",
        )

    def test_catalog_managed_dataset_rejects_unassigned_training_media(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            dataset_dir = Path(tmp_dir)
            media_path = dataset_dir / "unassigned.jpg"
            Image.new("RGB", (8, 6)).save(media_path)
            save_character_identity(
                dataset_dir=dataset_dir,
                identity_id="alice",
                display_name="Alice",
                trigger_word="AliceToken",
                class_prompt="a woman",
            )
            dataset_config = DatasetConfig(
                folder_path=str(dataset_dir),
                resolution=[8],
                diff_output_preservation=True,
                diff_output_preservation_class="a person",
            )
            file_item = FileItemDTO(
                path=str(media_path),
                dataset_config=dataset_config,
                size_database={},
                dataset_root=str(dataset_dir),
            )

            with self.assertRaisesRegex(ValueError, "not assigned to any named identity"):
                expand_character_identity_file_items([file_item])

    def test_one_shared_image_becomes_one_counterfactual_view_per_identity(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            dataset_dir = Path(tmp_dir)
            media_path = dataset_dir / "together.jpg"
            Image.new("RGB", (8, 6)).save(media_path)
            media_path.with_suffix(".txt").write_text(
                "AliceToken stands beside BobToken.",
                encoding="utf-8",
            )
            for identity_id, trigger_word, class_prompt in (
                ("alice", "AliceToken", "a woman"),
                ("bob", "BobToken", "a man"),
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
                    mask=np.ones((6, 8), dtype=np.uint8),
                )

            dataset_config = DatasetConfig(
                folder_path=str(dataset_dir),
                resolution=[8],
                diff_output_preservation=True,
                diff_output_preservation_class="a person",
            )
            file_item = FileItemDTO(
                path=str(media_path),
                dataset_config=dataset_config,
                size_database={},
                dataset_root=str(dataset_dir),
            )

            views = expand_character_identity_file_items([file_item])
            for view in views:
                view.load_caption()
                view.load_character_dop_visual_mask()

            self.assertEqual(len(views), 2)
            self.assertEqual([view.path for view in views], [str(media_path), str(media_path)])
            self.assertEqual([view.trigger_word for view in views], ["AliceToken", "BobToken"])
            self.assertEqual(
                [view.caption_dop for view in views],
                [
                    "a woman stands beside BobToken.",
                    "AliceToken stands beside a man.",
                ],
            )
            self.assertEqual(
                [view.get_dop_dropout_caption().strip() for view in views],
                ["a woman", "a man"],
            )
            self.assertTrue(all(view.has_character_dop_visual_mask for view in views))
            self.assertNotEqual(
                views[0].character_dop_visual_mask_path,
                views[1].character_dop_visual_mask_path,
            )
            self.assertTrue(
                all(tuple(view.character_dop_visual_mask_tensor.shape) == (1, 6, 8) for view in views)
            )

    def test_named_views_use_identity_distinct_latent_cache_keys(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            dataset_dir = Path(tmp_dir)
            media_path = dataset_dir / "shared.jpg"
            Image.new("RGB", (8, 6)).save(media_path)
            for identity_id, trigger_word in (("alice", "AliceToken"), ("bob", "BobToken")):
                save_character_identity(
                    dataset_dir=dataset_dir,
                    identity_id=identity_id,
                    display_name=identity_id.title(),
                    trigger_word=trigger_word,
                    class_prompt="a person",
                )
                save_character_visual_mask(
                    dataset_dir=dataset_dir,
                    media_path=media_path,
                    identity_id=identity_id,
                    mask=np.ones((6, 8), dtype=np.uint8),
                )

            dataset_config = DatasetConfig(
                folder_path=str(dataset_dir),
                resolution=[8],
                diff_output_preservation=True,
            )
            source = FileItemDTO(
                path=str(media_path),
                dataset_config=dataset_config,
                size_database={},
                dataset_root=str(dataset_dir),
            )
            # Simulate independently selected windows on an uncached shared video.
            source.is_video = True
            views = expand_character_identity_file_items([source])
            views[0].video_frames_to_extract = [0, 1, 2]
            views[1].video_frames_to_extract = [8, 9, 10]

            paths = [view.get_latent_path(recalculate=True) for view in views]

            self.assertNotEqual(paths[0], paths[1])
            self.assertEqual(views[0].video_frames_to_extract, [0, 1, 2])
            self.assertEqual(views[1].video_frames_to_extract, [8, 9, 10])
            self.assertEqual(
                [view.get_latent_info_dict()["character_dop_identity_id"] for view in views],
                ["alice", "bob"],
            )


if __name__ == "__main__":
    unittest.main()
