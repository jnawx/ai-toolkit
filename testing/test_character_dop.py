import unittest
import json
import tempfile
import types
from pathlib import Path

import torch

from toolkit.character_dop import (
    apply_character_training_weight,
    character_dop_loss,
    character_dop_losses,
    character_dop_multiplier,
    load_character_audio_intervals,
    map_audio_intervals_to_training_clip,
    prepare_temporal_character_mask,
)
from toolkit.config_modules import (
    DatasetConfig,
    ModelConfig,
    SaveConfig,
    TrainConfig,
    validate_configs,
)
from toolkit.data_transfer_object.data_loader import DataLoaderBatchDTO


class CharacterDOPVisualLossTests(unittest.TestCase):
    def test_character_training_weight_upweights_only_the_masked_visual_region(self):
        element_loss = torch.ones((1, 2, 1, 2, 2), dtype=torch.float32)
        character_mask = torch.tensor(
            [[[[1.0, 0.0], [0.5, 0.0]]]], dtype=torch.float32
        )

        weighted = apply_character_training_weight(
            element_loss,
            modality="visual",
            multiplier=3.0,
            character_mask=character_mask,
        )

        expected_spatial_weights = torch.tensor(
            [[[[3.0, 1.0], [2.0, 1.0]]]], dtype=torch.float32
        )
        torch.testing.assert_close(
            weighted,
            expected_spatial_weights.unsqueeze(1).expand_as(element_loss),
        )

    def test_character_masks_collate_separately_from_ordinary_loss_masks(self):
        dataset_config = types.SimpleNamespace(
            load_image_when_caching_latents=False,
            cache_tensors_to_disk=False,
        )

        def item(character_mask):
            return types.SimpleNamespace(
                is_latent_cached=False,
                is_audio_only=False,
                dataset_config=dataset_config,
                extra_values=[],
                audio_data=None,
                audio_tensor=None,
                num_frames=1,
                tensor=torch.zeros((1, 1, 1)),
                control_tensor=None,
                control_tensor_list=None,
                inpaint_tensor=None,
                loss_multiplier=1.0,
                clip_image_tensor=None,
                mask_tensor=None,
                character_dop_visual_mask_tensor=character_mask,
                unaugmented_tensor=None,
                unconditional_tensor=None,
                clip_image_embeds=None,
                clip_image_embeds_unconditional=None,
                prompt_embeds=None,
                dop_prompt_embeds=None,
            )

        batch = DataLoaderBatchDTO(
            file_items=[item(torch.ones((1, 1, 1))), item(None)]
        )

        self.assertIsNone(batch.mask_tensor)
        self.assertEqual(batch.character_dop_visual_mask_present, [True, False])
        torch.testing.assert_close(
            batch.character_dop_visual_mask_tensor,
            torch.tensor([[[[1.0]]], [[[0.0]]]]),
        )

    def test_visual_loss_focuses_on_the_largest_token_drift(self):
        prior = torch.zeros((1, 1, 1, 2, 2), dtype=torch.float32)
        prediction = torch.tensor(
            [[[[[4.0, 1.0], [1.0, 1.0]]]]], dtype=torch.float32
        )

        loss = character_dop_loss(
            prediction,
            prior,
            modality="visual",
            focus_fraction=0.25,
        )

        self.assertEqual(loss.item(), 16.0)

    def test_visual_loss_preserves_outside_the_character_mask(self):
        prior = torch.zeros((1, 1, 1, 2, 2), dtype=torch.float32)
        prediction = torch.tensor(
            [[[[[10.0, 2.0], [2.0, 2.0]]]]], dtype=torch.float32
        )
        character_mask = torch.tensor(
            [[[[1.0, 0.0], [0.0, 0.0]]]], dtype=torch.float32
        )

        loss = character_dop_loss(
            prediction,
            prior,
            modality="visual",
            focus_fraction=1.0,
            character_mask=character_mask,
        )

        self.assertEqual(loss.item(), 4.0)

    def test_visual_loss_backpropagates_from_bfloat16_h3_predictions(self):
        prior = torch.zeros((1, 2, 1, 2, 2), dtype=torch.bfloat16)
        prediction = torch.ones_like(prior, requires_grad=True)

        loss = character_dop_loss(
            prediction,
            prior,
            modality="visual",
            focus_fraction=0.25,
        )
        loss.backward()

        self.assertIsNotNone(prediction.grad)
        self.assertGreater(prediction.grad.abs().sum().item(), 0.0)

    def test_mixed_visual_batch_uses_counterfactual_fallback_for_unannotated_items(self):
        prior = torch.zeros((2, 1, 1, 1), dtype=torch.float32)
        primary = torch.tensor([[[[10.0]]], [[[20.0]]]])
        counterfactual = torch.tensor([[[[2.0]]], [[[3.0]]]])
        character_mask = torch.tensor([[[[1.0]]], [[[0.0]]]])

        losses = character_dop_losses(
            visual_prediction=counterfactual,
            visual_primary_prediction=primary,
            visual_prior=prior,
            visual_character_mask_present=[True, False],
            character_mask=character_mask,
            focus_fraction=1.0,
            base_multiplier=1.0,
            every_n_steps=1,
            visual_multiplier=1.0,
            audio_multiplier=1.0,
        )

        self.assertEqual(losses.visual.item(), 4.5)

    def test_temporal_masks_select_the_same_source_frames_as_the_video(self):
        source_mask = torch.tensor(
            [
                [[0.0, 0.0], [0.0, 0.0]],
                [[0.5, 0.5], [0.5, 0.5]],
                [[1.0, 1.0], [1.0, 1.0]],
            ]
        )

        mask = prepare_temporal_character_mask(
            source_mask,
            frame_indices=[0, 2],
            scale_size=(2, 2),
            crop=(0, 0, 2, 2),
            flip_x=False,
            flip_y=False,
            min_value=0.0,
        )

        self.assertEqual(tuple(mask.shape), (1, 2, 2, 2))
        self.assertEqual(mask[:, 0].sum().item(), 0.0)
        self.assertEqual(mask[:, 1].sum().item(), 4.0)

    def test_temporal_masks_reject_legacy_caches_without_frame_indices(self):
        source_mask = torch.zeros((3, 2, 2), dtype=torch.float32)

        with self.assertRaisesRegex(ValueError, "rebuild.*latent cache"):
            prepare_temporal_character_mask(
                source_mask,
                frame_indices=None,
                scale_size=(2, 2),
                crop=(0, 0, 2, 2),
                flip_x=False,
                flip_y=False,
                min_value=0.0,
            )


class CharacterDOPAudioLossTests(unittest.TestCase):
    def test_character_training_weight_upweights_only_annotated_speech(self):
        element_loss = torch.ones((1, 4, 2), dtype=torch.float32)

        weighted = apply_character_training_weight(
            element_loss,
            modality="audio",
            multiplier=2.0,
            audio_character_intervals=[[(0.0, 1.0)]],
            audio_latents_per_second=1,
        )

        expected_row_weights = torch.tensor(
            [[[2.0], [1.0], [2.0], [1.0]]], dtype=torch.float32
        )
        torch.testing.assert_close(
            weighted,
            expected_row_weights.expand_as(element_loss),
        )

    def test_audio_loss_focuses_on_the_largest_temporal_drift(self):
        prior = torch.zeros((1, 4, 2), dtype=torch.float32)
        prediction = torch.tensor(
            [[[4.0, 4.0], [3.0, 3.0], [1.0, 1.0], [1.0, 1.0]]],
            dtype=torch.float32,
        )

        loss = character_dop_loss(
            prediction,
            prior,
            modality="audio",
            focus_fraction=0.5,
        )

        self.assertEqual(loss.item(), 12.5)

    def test_mixed_audio_batch_uses_counterfactual_fallback_for_unannotated_items(self):
        prior = torch.zeros((2, 2, 1), dtype=torch.float32)
        primary = torch.tensor([[[10.0], [10.0]], [[20.0], [20.0]]])
        counterfactual = torch.tensor([[[2.0], [2.0]], [[3.0], [3.0]]])

        losses = character_dop_losses(
            audio_prediction=counterfactual,
            audio_primary_prediction=primary,
            audio_prior=prior,
            audio_character_intervals=[[(0.0, 1.0)], []],
            audio_character_mask_present=[True, False],
            audio_latents_per_second=1,
            focus_fraction=1.0,
            base_multiplier=1.0,
            every_n_steps=1,
            visual_multiplier=1.0,
            audio_multiplier=1.0,
        )

        self.assertEqual(losses.audio.item(), 4.5)

    def test_audio_intervals_load_from_a_matching_json_sidecar(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "scene.json").write_text(
                json.dumps({"character_intervals": [[1.0, 2.5]]}),
                encoding="utf-8",
            )

            intervals = load_character_audio_intervals(
                media_path=str(root / "scene.mp4"),
                intervals_path=str(root),
            )

        self.assertEqual(intervals, [(1.0, 2.5)])

    def test_audio_intervals_load_from_nested_dataset_editor_sidecar(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            dataset = Path(temp_dir) / "dataset"
            media = dataset / "scenes" / "scene.mp4"
            sidecars = dataset / "_character_dop" / "audio"
            media.parent.mkdir(parents=True)
            media.touch()
            nested_sidecar = sidecars / "scenes" / "scene.json"
            nested_sidecar.parent.mkdir(parents=True)
            nested_sidecar.write_text(
                json.dumps({"character_intervals": [[1.0, 2.5]]}),
                encoding="utf-8",
            )

            intervals = load_character_audio_intervals(
                media_path=str(media),
                intervals_path=str(sidecars),
                dataset_root=str(dataset),
            )

        self.assertEqual(intervals, [(1.0, 2.5)])

    def test_unannotated_items_fall_back_when_dataset_has_other_ui_sidecars(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            dataset = Path(temp_dir) / "dataset"
            media = dataset / "unannotated.mp4"
            sidecars = dataset / "_character_dop" / "audio"
            sidecars.mkdir(parents=True)
            media.touch()

            intervals = load_character_audio_intervals(
                media_path=str(media),
                intervals_path=str(sidecars),
                dataset_root=str(dataset),
            )

        self.assertIsNone(intervals)

    def test_source_intervals_map_into_a_stretched_training_clip(self):
        intervals = map_audio_intervals_to_training_clip(
            [(11.0, 13.0), (18.0, 22.0)],
            source_start_seconds=10.0,
            source_duration_seconds=10.0,
            target_duration_seconds=5.0,
        )

        self.assertEqual(intervals, [(0.5, 1.5), (4.0, 5.0)])


class CharacterDOPScheduleTests(unittest.TestCase):
    def test_sparse_steps_are_compensated_by_the_interval(self):
        self.assertEqual(
            character_dop_multiplier(base_multiplier=1.5, every_n_steps=4),
            6.0,
        )

    def test_joint_losses_apply_visual_audio_and_interval_multipliers(self):
        losses = character_dop_losses(
            visual_prediction=torch.full((1, 1, 1, 1, 1), 2.0),
            visual_prior=torch.zeros((1, 1, 1, 1, 1)),
            audio_prediction=torch.full((1, 1, 1), 3.0),
            audio_prior=torch.zeros((1, 1, 1)),
            focus_fraction=1.0,
            base_multiplier=1.0,
            every_n_steps=4,
            visual_multiplier=2.0,
            audio_multiplier=3.0,
        )

        self.assertEqual(losses.visual.item(), 32.0)
        self.assertEqual(losses.audio.item(), 108.0)
        self.assertEqual(losses.total.item(), 140.0)

    def test_character_mask_binds_primary_trigger_effect_to_the_mask(self):
        losses = character_dop_losses(
            visual_prediction=torch.full((1, 1, 1, 1, 2), 2.0),
            visual_primary_prediction=torch.tensor([[[[[10.0, 5.0]]]]]),
            visual_prior=torch.zeros((1, 1, 1, 1, 2)),
            focus_fraction=1.0,
            base_multiplier=1.0,
            every_n_steps=1,
            visual_multiplier=1.0,
            audio_multiplier=1.0,
            character_mask=torch.tensor([[[[1.0, 0.0]]]]),
        )

        self.assertEqual(losses.visual.item(), 25.0)

    def test_speaking_intervals_bind_primary_voice_effect_outside_the_character(self):
        losses = character_dop_losses(
            audio_prediction=torch.full((1, 4, 1), 2.0),
            audio_primary_prediction=torch.tensor([[[10.0], [5.0], [10.0], [5.0]]]),
            audio_prior=torch.zeros((1, 4, 1)),
            audio_character_intervals=[[(0.0, 0.025)]],
            audio_latents_per_second=40,
            focus_fraction=1.0,
            base_multiplier=1.0,
            every_n_steps=1,
            visual_multiplier=1.0,
            audio_multiplier=1.0,
        )

        self.assertEqual(losses.audio.item(), 25.0)


class CharacterDOPConfigTests(unittest.TestCase):
    def test_character_mode_has_modality_specific_defaults(self):
        config = TrainConfig(diff_output_preservation_mode="character")

        self.assertEqual(config.diff_output_preservation_mode, "character")
        self.assertEqual(config.diff_output_preservation_focus_fraction, 0.25)
        self.assertEqual(config.diff_output_preservation_visual_multiplier, 1.0)
        self.assertEqual(config.diff_output_preservation_audio_multiplier, 1.0)
        self.assertEqual(config.character_training_visual_multiplier, 1.0)
        self.assertEqual(config.character_training_audio_multiplier, 1.0)

    def test_standard_mode_remains_the_default(self):
        config = TrainConfig()

        self.assertEqual(config.diff_output_preservation_mode, "standard")

    def test_invalid_character_focus_is_rejected(self):
        for value in (0, -0.1, 1.1, "invalid"):
            with self.subTest(value=value):
                with self.assertRaisesRegex(ValueError, "focus_fraction"):
                    TrainConfig(diff_output_preservation_focus_fraction=value)

    def test_character_training_multipliers_cannot_downweight_the_character(self):
        for key in (
            "character_training_visual_multiplier",
            "character_training_audio_multiplier",
        ):
            for value in (0.5, float("nan"), float("inf"), float("-inf")):
                with self.subTest(key=key, value=value):
                    with self.assertRaisesRegex(ValueError, key):
                        TrainConfig(**{key: value})

    def test_character_training_weight_rejects_non_finite_multiplier(self):
        for value in (float("nan"), float("inf"), float("-inf")):
            with self.subTest(value=value):
                with self.assertRaisesRegex(ValueError, "at least 1"):
                    apply_character_training_weight(
                        torch.ones((1, 1, 1, 1)),
                        modality="visual",
                        multiplier=value,
                    )

    def test_character_mode_is_restricted_to_minimax_h3(self):
        train = TrainConfig(diff_output_preservation_mode="character")
        model = ModelConfig(name_or_path="unused", arch="flux")
        dataset = DatasetConfig(folder_path="unused", resolution=[512])

        with self.assertRaisesRegex(ValueError, "MiniMax-H3"):
            validate_configs(train, model, SaveConfig(), [dataset])

    def test_selected_identity_curriculum_requires_dop_to_be_enabled(self):
        with self.assertRaisesRegex(ValueError, "requires diff_output_preservation"):
            TrainConfig(
                diff_output_preservation=False,
                diff_output_preservation_mode="character",
                character_training={
                    "identities": [{"id": "alice", "weight": 1}],
                    "joint_training_fraction": 0,
                },
            )

    def test_speaking_masks_require_an_audio_dataset(self):
        train = TrainConfig(diff_output_preservation_mode="character")
        model = ModelConfig(name_or_path="unused", arch="minimax_h3")
        dataset = DatasetConfig(
            folder_path="unused",
            resolution=[512],
            character_dop_audio_mask_path="intervals",
        )

        with self.assertRaisesRegex(ValueError, "audio-enabled"):
            validate_configs(train, model, SaveConfig(), [dataset])

    def test_speaking_masks_accept_a_preprocessed_audio_only_dataset(self):
        train = TrainConfig(diff_output_preservation_mode="character")
        model = ModelConfig(name_or_path="unused", arch="minimax_h3")
        dataset = DatasetConfig(
            folder_path="unused",
            resolution=768,
            do_audio=False,
            is_audio_only=True,
            character_dop_audio_mask_path="intervals",
        )

        validate_configs(train, model, SaveConfig(), [dataset])

    def test_character_mode_automatically_uses_annotations_created_by_the_dataset_ui(self):
        with tempfile.TemporaryDirectory() as dataset_dir:
            annotation_root = Path(dataset_dir) / "_character_dop"
            visual_dir = annotation_root / "visual"
            audio_dir = annotation_root / "audio"
            visual_dir.mkdir(parents=True)
            audio_dir.mkdir()
            train = TrainConfig(diff_output_preservation_mode="character")
            model = ModelConfig(name_or_path="unused", arch="minimax_h3")
            dataset = DatasetConfig(
                folder_path=dataset_dir,
                resolution=[512],
                do_audio=True,
            )

            validate_configs(train, model, SaveConfig(), [dataset])

            self.assertIsNone(dataset.mask_path)
            self.assertEqual(dataset.character_dop_visual_mask_path, str(visual_dir))
            self.assertEqual(dataset.character_dop_audio_mask_path, str(audio_dir))


class CharacterTrainingIntegrationTests(unittest.TestCase):
    def test_default_training_multipliers_do_not_touch_character_annotations(self):
        from extensions_built_in.sd_trainer.SDTrainer import SDTrainer

        class FakeModel:
            is_flow_matching = True
            prediction_type = "epsilon"
            x0_pred = False

            @staticmethod
            def get_loss_target(**kwargs):
                return torch.zeros_like(kwargs["noise"])

            @staticmethod
            def scale_loss(loss):
                return loss

            @staticmethod
            def get_character_training_inputs(**_kwargs):
                raise AssertionError("1x character training must remain a no-op")

        batch = types.SimpleNamespace(
            get_is_reg_list=lambda: [False],
            loss_multiplier_list=[1.0],
            mask_tensor=None,
            latents=torch.zeros((1, 1, 1, 1)),
            dataset_config=types.SimpleNamespace(is_audio_only=False),
            sigmas=None,
            audio_pred=None,
            audio_target=None,
        )
        trainer = SDTrainer.__new__(SDTrainer)
        trainer.train_config = TrainConfig(
            diff_output_preservation=True,
            diff_output_preservation_mode="character",
        )
        trainer.device_torch = torch.device("cpu")
        trainer.sd = FakeModel()
        trainer.dfe = None
        trainer.adapter = None

        loss = trainer.calculate_loss(
            noise_pred=torch.ones((1, 1, 1, 1)),
            noise=torch.zeros((1, 1, 1, 1)),
            noisy_latents=torch.zeros((1, 1, 1, 1)),
            timesteps=torch.tensor([500.0]),
            batch=batch,
        )

        self.assertEqual(loss.item(), 1.0)

    def test_trainer_applies_visual_character_weight_before_reduction(self):
        from extensions_built_in.sd_trainer.SDTrainer import SDTrainer

        character_mask = torch.tensor([[[[1.0, 0.0]]]], dtype=torch.float32)

        class FakeModel:
            is_flow_matching = True
            prediction_type = "epsilon"
            x0_pred = False

            @staticmethod
            def get_loss_target(**kwargs):
                return torch.zeros_like(kwargs["noise"])

            @staticmethod
            def scale_loss(loss):
                return loss

            @staticmethod
            def get_character_training_inputs(**_kwargs):
                return {
                    "visual_character_mask": character_mask,
                    "audio_character_intervals": None,
                    "audio_latents_per_second": 40,
                }

        batch = types.SimpleNamespace(
            get_is_reg_list=lambda: [False],
            loss_multiplier_list=[1.0],
            mask_tensor=None,
            latents=torch.zeros((1, 1, 1, 2)),
            dataset_config=types.SimpleNamespace(is_audio_only=False),
            sigmas=None,
            audio_pred=None,
            audio_target=None,
        )
        trainer = SDTrainer.__new__(SDTrainer)
        trainer.train_config = TrainConfig(
            diff_output_preservation=True,
            diff_output_preservation_mode="character",
            character_training_visual_multiplier=3.0,
        )
        trainer.device_torch = torch.device("cpu")
        trainer.sd = FakeModel()
        trainer.dfe = None
        trainer.adapter = None

        loss = trainer.calculate_loss(
            noise_pred=torch.tensor([[[[2.0, 1.0]]]], dtype=torch.float32),
            noise=torch.zeros((1, 1, 1, 2), dtype=torch.float32),
            noisy_latents=torch.zeros((1, 1, 1, 2), dtype=torch.float32),
            timesteps=torch.tensor([500.0]),
            batch=batch,
        )

        self.assertEqual(loss.item(), 6.5)

    def test_trainer_applies_audio_character_weight_to_joint_h3_loss(self):
        from extensions_built_in.sd_trainer.SDTrainer import SDTrainer

        class FakeModel:
            is_flow_matching = True
            prediction_type = "epsilon"
            x0_pred = False

            @staticmethod
            def get_loss_target(**kwargs):
                return torch.zeros_like(kwargs["noise"])

            @staticmethod
            def scale_loss(loss):
                return loss

            @staticmethod
            def get_character_training_inputs(**_kwargs):
                return {
                    "visual_character_mask": None,
                    "audio_character_intervals": [[(0.0, 1.0)]],
                    "audio_latents_per_second": 1,
                }

        batch = types.SimpleNamespace(
            get_is_reg_list=lambda: [False],
            loss_multiplier_list=[1.0],
            mask_tensor=None,
            latents=torch.zeros((1, 1, 1, 1)),
            dataset_config=types.SimpleNamespace(is_audio_only=False),
            sigmas=None,
            audio_pred=torch.tensor([[[2.0], [1.0], [2.0], [1.0]]]),
            audio_target=torch.zeros((1, 4, 1)),
        )
        trainer = SDTrainer.__new__(SDTrainer)
        trainer.train_config = TrainConfig(
            diff_output_preservation=True,
            diff_output_preservation_mode="character",
            character_training_audio_multiplier=2.0,
        )
        trainer.device_torch = torch.device("cpu")
        trainer.sd = FakeModel()
        trainer.dfe = None
        trainer.adapter = None
        trainer.additional_logs = {}

        loss = trainer.calculate_loss(
            noise_pred=torch.zeros((1, 1, 1, 1)),
            noise=torch.zeros((1, 1, 1, 1)),
            noisy_latents=torch.zeros((1, 1, 1, 1)),
            timesteps=torch.tensor([500.0]),
            batch=batch,
        )

        self.assertEqual(loss.item(), 4.5)

    def test_trainer_applies_audio_character_weight_to_audio_only_h3_loss(self):
        from extensions_built_in.sd_trainer.SDTrainer import SDTrainer

        class FakeModel:
            is_flow_matching = True
            prediction_type = "epsilon"
            x0_pred = False

            @staticmethod
            def get_loss_target(**kwargs):
                return torch.zeros_like(kwargs["noise"])

            @staticmethod
            def scale_loss(loss):
                return loss

            @staticmethod
            def get_character_training_inputs(**_kwargs):
                return {
                    "visual_character_mask": None,
                    "audio_character_intervals": [[(0.0, 1.0)]],
                    "audio_latents_per_second": 1,
                }

        batch = types.SimpleNamespace(
            get_is_reg_list=lambda: [False],
            loss_multiplier_list=[1.0],
            mask_tensor=None,
            latents=torch.zeros((1, 4, 1)),
            dataset_config=types.SimpleNamespace(is_audio_only=True),
            sigmas=None,
            audio_pred=None,
            audio_target=None,
        )
        trainer = SDTrainer.__new__(SDTrainer)
        trainer.train_config = TrainConfig(
            diff_output_preservation=True,
            diff_output_preservation_mode="character",
            character_training_audio_multiplier=2.0,
        )
        trainer.device_torch = torch.device("cpu")
        trainer.sd = FakeModel()
        trainer.dfe = None
        trainer.adapter = None

        loss = trainer.calculate_loss(
            noise_pred=torch.tensor([[[2.0], [1.0], [2.0], [1.0]]]),
            noise=torch.zeros((1, 4, 1)),
            noisy_latents=torch.zeros((1, 4, 1)),
            timesteps=torch.tensor([500.0]),
            batch=batch,
        )

        self.assertEqual(loss.item(), 4.5)


if __name__ == "__main__":
    unittest.main()
