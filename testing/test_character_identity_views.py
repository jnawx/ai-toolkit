import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

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
    def test_regularization_media_remains_one_physical_view(self):
        source = SimpleNamespace(
            is_reg=True,
            character_dop_identity_views=[SimpleNamespace(identity_id="alice"), SimpleNamespace(identity_id="bob")],
            dataset_config=SimpleNamespace(character_training=None),
        )

        self.assertEqual(expand_character_identity_file_items([source]), [source])

    def test_video_views_only_use_annotations_for_enabled_modalities(self):
        class Source:
            def __init__(self, *, do_audio):
                self.is_audio_only = False
                self.is_video = True
                self.is_reg = False
                self.dataset_config = SimpleNamespace(
                    do_audio=do_audio,
                    character_training={
                        "identities": [{"id": "alice", "weight": 1}],
                        "joint_training_fraction": 0,
                    },
                )
                self.character_dop_identity_views = [
                    SimpleNamespace(
                        identity_id="alice",
                        trigger_word="AliceToken",
                        caption_description="Alice description",
                        visual_path=None,
                        audio_intervals=[(0.0, 1.0)],
                    )
                ]

            def bind_character_dop_identity(self, identity_view, **kwargs):
                self.character_dop_identity_id = identity_view.identity_id
                self.character_training_view_mode = kwargs["view_mode"]

        self.assertEqual(expand_character_identity_file_items([Source(do_audio=False)]), [])
        views = expand_character_identity_file_items([Source(do_audio=True)])
        self.assertEqual(len(views), 1)
        self.assertEqual(views[0].character_dop_identity_id, "alice")

    def test_focus_caption_replaces_annotated_identities_outside_enabled_modalities(self):
        class Source:
            is_audio_only = False
            is_video = True
            is_reg = False
            dataset_config = SimpleNamespace(
                do_audio=False,
                character_training={
                    "identities": [{"id": "alice", "weight": 1}],
                    "joint_training_fraction": 0,
                },
            )
            character_dop_identity_views = [
                SimpleNamespace(
                    identity_id="alice", trigger_word="AliceToken",
                    caption_description="Alice description", visual_path=Path("alice.npy"),
                    audio_intervals=None,
                ),
                SimpleNamespace(
                    identity_id="bob", trigger_word="BobToken",
                    caption_description="Bob description", visual_path=None,
                    audio_intervals=[(0.0, 1.0)],
                ),
            ]

            def bind_character_dop_identity(self, identity_view, **kwargs):
                self.character_dop_identity_id = identity_view.identity_id
                self.character_caption_replacements = kwargs["caption_replacements"]

        views = expand_character_identity_file_items([Source()])

        self.assertEqual(views[0].character_caption_replacements, {"BobToken": "Bob description"})

    def test_selected_identities_create_isolated_focus_and_joint_caption_views(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            dataset_dir = Path(tmp_dir)
            media_path = dataset_dir / "group.jpg"
            Image.new("RGB", (8, 6)).save(media_path)
            media_path.with_suffix(".txt").write_text(
                "AliceToken stands beside BobToken and CarlToken.",
                encoding="utf-8",
            )
            for identity_id, trigger_word, description in (
                ("alice", "AliceToken", "a tall woman with silver hair"),
                ("bob", "BobToken", "a bearded man in a blue shirt"),
                ("carl", "CarlToken", "an older man wearing glasses"),
            ):
                save_character_identity(
                    dataset_dir=dataset_dir,
                    identity_id=identity_id,
                    display_name=identity_id.title(),
                    trigger_word=trigger_word,
                    class_prompt="a person",
                    caption_description=description,
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
                character_training={
                    "identities": [
                        {"id": "alice", "weight": 1},
                        {"id": "bob", "weight": 1},
                    ],
                    "joint_training_fraction": 0.25,
                },
            )
            source = FileItemDTO(
                path=str(media_path),
                dataset_config=dataset_config,
                size_database={},
                dataset_root=str(dataset_dir),
            )

            views = expand_character_identity_file_items([source])
            for view in views:
                view.load_caption()

            self.assertEqual(
                [(view.character_dop_identity_id, view.character_training_view_mode) for view in views],
                [("alice", "focus"), ("bob", "focus"), ("alice", "joint"), ("bob", "joint")],
            )
            self.assertEqual(
                [view.caption for view in views],
                [
                    "AliceToken stands beside a bearded man in a blue shirt and an older man wearing glasses.",
                    "a tall woman with silver hair stands beside BobToken and an older man wearing glasses.",
                    "AliceToken stands beside BobToken and an older man wearing glasses.",
                    "AliceToken stands beside BobToken and an older man wearing glasses.",
                ],
            )

    def test_selected_identity_job_discards_media_without_a_selected_annotation(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            dataset_dir = Path(tmp_dir)
            media_path = dataset_dir / "carl.jpg"
            Image.new("RGB", (8, 6)).save(media_path)
            save_character_identity(
                dataset_dir=dataset_dir,
                identity_id="carl",
                display_name="Carl",
                trigger_word="CarlToken",
                class_prompt="a man",
            )
            save_character_visual_mask(
                dataset_dir=dataset_dir,
                media_path=media_path,
                identity_id="carl",
                mask=np.ones((6, 8), dtype=np.uint8),
            )
            dataset_config = DatasetConfig(
                folder_path=str(dataset_dir),
                resolution=[8],
                diff_output_preservation=True,
                character_training={
                    "identities": [{"id": "alice", "weight": 1}],
                    "joint_training_fraction": 0,
                },
            )
            source = FileItemDTO(
                path=str(media_path),
                dataset_config=dataset_config,
                size_database={},
                dataset_root=str(dataset_dir),
            )

            self.assertEqual(expand_character_identity_file_items([source]), [])

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

    def test_focus_and_joint_views_use_distinct_video_latent_cache_keys(self):
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
                character_training={
                    "identities": [{"id": "alice", "weight": 1}, {"id": "bob", "weight": 1}],
                    "joint_training_fraction": 0.5,
                },
            )
            source = FileItemDTO(
                path=str(media_path), dataset_config=dataset_config, size_database={}, dataset_root=str(dataset_dir)
            )
            views = expand_character_identity_file_items([source])

            alice_views = [view for view in views if view.character_dop_identity_id == "alice"]
            paths = [view.get_latent_path(recalculate=True) for view in alice_views]

            self.assertEqual([view.character_training_view_mode for view in alice_views], ["focus", "joint"])
            self.assertNotEqual(paths[0], paths[1])


if __name__ == "__main__":
    unittest.main()
