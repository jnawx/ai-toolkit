import unittest
import json
import tempfile
from pathlib import Path

import torch

from toolkit.character_dop import (
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


class CharacterDOPVisualLossTests(unittest.TestCase):
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


class CharacterDOPAudioLossTests(unittest.TestCase):
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

    def test_standard_mode_remains_the_default(self):
        config = TrainConfig()

        self.assertEqual(config.diff_output_preservation_mode, "standard")

    def test_invalid_character_focus_is_rejected(self):
        for value in (0, -0.1, 1.1, "invalid"):
            with self.subTest(value=value):
                with self.assertRaisesRegex(ValueError, "focus_fraction"):
                    TrainConfig(diff_output_preservation_focus_fraction=value)

    def test_character_mode_is_restricted_to_minimax_h3(self):
        train = TrainConfig(diff_output_preservation_mode="character")
        model = ModelConfig(name_or_path="unused", arch="flux")
        dataset = DatasetConfig(folder_path="unused", resolution=[512])

        with self.assertRaisesRegex(ValueError, "MiniMax-H3"):
            validate_configs(train, model, SaveConfig(), [dataset])

    def test_speaking_masks_require_an_audio_dataset(self):
        train = TrainConfig(diff_output_preservation_mode="character")
        model = ModelConfig(name_or_path="unused", arch="minimax_h3")
        dataset = DatasetConfig(
            folder_path="unused",
            resolution=[512],
            character_dop_audio_mask_path="intervals",
        )

        with self.assertRaisesRegex(ValueError, "do_audio"):
            validate_configs(train, model, SaveConfig(), [dataset])


if __name__ == "__main__":
    unittest.main()
