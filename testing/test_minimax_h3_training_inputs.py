import types
import unittest

import torch

from extensions_built_in.diffusion_models.minimax_h3.minimax_h3 import MinimaxH3Model


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


if __name__ == "__main__":
    unittest.main()
