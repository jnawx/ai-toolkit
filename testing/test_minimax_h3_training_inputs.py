import types
import unittest

import torch

from extensions_built_in.diffusion_models.minimax_h3.minimax_h3 import MinimaxH3Model
from toolkit.audio.processing import AudioSegment


class _RecordingTransformer:
    def __init__(self):
        self.device = torch.device("cpu")
        self.calls = []

    def to(self, device):
        self.device = torch.device(device)
        return self

    def __call__(self, *, hidden_states, audio_hidden_states, **_kwargs):
        self.calls.append(
            (hidden_states.detach().clone(), audio_hidden_states.detach().clone())
        )
        return torch.zeros_like(hidden_states), torch.zeros_like(audio_hidden_states)


class _Batch:
    def __init__(self, *, do_i2v=False):
        self.dataset_config = types.SimpleNamespace(
            is_audio_only=False,
            do_i2v=do_i2v,
            do_audio=False,
        )
        self.num_frames = 5
        self.first_frame_latents = (
            torch.ones((1, 24, 1, 2, 2), dtype=torch.float32)
            if do_i2v
            else None
        )
        self.tensor = None
        self.audio_latents = None
        self.audio_data = None
        self.audio_noise = None
        self.audio_target = None
        self.audio_pred_slot = "test_secondary"
        self.audio_pred = None
        self.audio_pred_prior = None
        self.audio_pred_preservation = None
        self.mask_tensor = None
        self.character_dop_visual_mask_tensor = None
        self.character_dop_visual_mask_present = None
        self.file_items = [
            types.SimpleNamespace(
                character_dop_audio_intervals=None,
                audio_segment=None,
            )
        ]

    def set_secondary_audio_pred(self, _prediction):
        pass


def _run_two_h3_predictions(*, do_i2v=False):
    h3 = MinimaxH3Model.__new__(MinimaxH3Model)
    h3.device_torch = torch.device("cpu")
    h3.torch_dtype = torch.float32
    h3.model = _RecordingTransformer()
    batch = _Batch(do_i2v=do_i2v)
    latents = torch.zeros((1, 24, 1, 2, 2), dtype=torch.float32)
    timestep = torch.tensor([500.0])
    text_embeddings = types.SimpleNamespace(
        text_token_tags=[torch.tensor([1], dtype=torch.long)],
        text_embeds=[torch.zeros((1, 4), dtype=torch.float32)],
    )

    h3.get_noise_prediction(latents, timestep, text_embeddings, batch=batch)
    h3.get_noise_prediction(latents, timestep, text_embeddings, batch=batch)
    return h3.model.calls


class MiniMaxH3SharedTrainingInputTests(unittest.TestCase):
    def test_silent_audio_noise_is_shared_across_counterfactual_passes(self):
        calls = _run_two_h3_predictions()

        torch.testing.assert_close(calls[0][1], calls[1][1])

    def test_i2v_conditioning_noise_is_shared_across_counterfactual_passes(self):
        calls = _run_two_h3_predictions(do_i2v=True)

        torch.testing.assert_close(calls[0][0], calls[1][0])


class MiniMaxH3CharacterDOPRoutingTests(unittest.TestCase):
    def test_joint_video_audio_routes_primary_predictions_and_speaking_intervals(self):
        h3 = MinimaxH3Model.__new__(MinimaxH3Model)
        batch = _Batch()
        batch.audio_pred = torch.full((1, 4, 1), 3.0)
        batch.audio_pred_prior = torch.zeros((1, 4, 1))
        batch.audio_pred_preservation = torch.ones((1, 4, 1))
        batch.character_dop_visual_mask_tensor = torch.ones((1, 1, 1, 1))
        batch.character_dop_visual_mask_present = [True]
        batch.file_items[0].character_dop_audio_intervals = [(11.0, 13.0)]
        batch.file_items[0].audio_segment = AudioSegment(10.0, 10.0, 5.0)
        primary = torch.full((1, 1, 1, 1, 1), 4.0)
        preservation = torch.full_like(primary, 2.0)
        prior = torch.zeros_like(primary)

        inputs = h3.get_character_dop_inputs(
            batch=batch,
            primary_prediction=primary,
            preservation_prediction=preservation,
            prior_prediction=prior,
        )

        self.assertIs(inputs["visual_primary_prediction"], primary)
        self.assertIs(inputs["audio_primary_prediction"], batch.audio_pred)
        self.assertIs(inputs["character_mask"], batch.character_dop_visual_mask_tensor)
        self.assertEqual(inputs["visual_character_mask_present"], [True])
        self.assertEqual(inputs["audio_character_mask_present"], [True])
        self.assertEqual(inputs["audio_character_intervals"], [[(0.5, 1.5)]])
        self.assertEqual(inputs["audio_latents_per_second"], 40)

    def test_audio_only_routes_the_main_h3_predictions_as_audio(self):
        h3 = MinimaxH3Model.__new__(MinimaxH3Model)
        batch = _Batch()
        batch.dataset_config.is_audio_only = True
        primary = torch.full((1, 4, 1), 4.0)
        preservation = torch.full_like(primary, 2.0)
        prior = torch.zeros_like(primary)

        inputs = h3.get_character_dop_inputs(
            batch=batch,
            primary_prediction=primary,
            preservation_prediction=preservation,
            prior_prediction=prior,
        )

        self.assertIsNone(inputs["visual_prediction"])
        self.assertIs(inputs["audio_prediction"], preservation)
        self.assertIs(inputs["audio_primary_prediction"], primary)
        self.assertIs(inputs["audio_prior"], prior)


if __name__ == "__main__":
    unittest.main()
