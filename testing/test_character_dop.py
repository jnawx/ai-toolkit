import unittest

import torch

from toolkit.character_dop import (
    character_dop_loss,
    character_dop_losses,
    character_dop_multiplier,
)
from toolkit.config_modules import TrainConfig


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


if __name__ == "__main__":
    unittest.main()
