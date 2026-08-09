import sys
import types
import unittest
import importlib.util
from pathlib import Path

import torch


def _stub_optional_imports():
    """Load config parsing without requiring a full training environment."""
    hf = types.ModuleType("huggingface_hub")
    hf_utils = types.ModuleType("huggingface_hub.utils")
    hf_tqdm = types.ModuleType("huggingface_hub.utils.tqdm")
    hf_tqdm.is_tqdm_disabled = lambda _level: None
    sys.modules.setdefault("huggingface_hub", hf)
    sys.modules.setdefault("huggingface_hub.utils", hf_utils)
    sys.modules.setdefault("huggingface_hub.utils.tqdm", hf_tqdm)

    prompt_utils = types.ModuleType("toolkit.prompt_utils")
    prompt_utils.PromptEmbeds = object
    sys.modules.setdefault("toolkit.prompt_utils", prompt_utils)

    album_artwork = types.ModuleType("toolkit.audio.album_artwork")
    album_artwork.add_album_artwork = lambda *_args, **_kwargs: None
    sys.modules.setdefault("toolkit.audio.album_artwork", album_artwork)

    torchao = types.ModuleType("torchao")
    torchao_quantization = types.ModuleType("torchao.quantization")
    quant_primitives = types.ModuleType("torchao.quantization.quant_primitives")
    quant_primitives._DTYPE_TO_BIT_WIDTH = {}
    sys.modules.setdefault("torchao", torchao)
    sys.modules.setdefault("torchao.quantization", torchao_quantization)
    sys.modules.setdefault("torchao.quantization.quant_primitives", quant_primitives)


_stub_optional_imports()

from toolkit.config_modules import DatasetConfig, preprocess_dataset_raw_config


class AudioWaveformPreparationTests(unittest.TestCase):
    def test_audio_is_stereo_and_padded_to_the_configured_duration(self):
        from toolkit.audio.processing import prepare_audio_for_training

        waveform = torch.tensor([[0.25, -0.25]], dtype=torch.float32)

        prepared = prepare_audio_for_training(
            waveform,
            sample_rate=2,
            target_sample_rate=2,
            duration_seconds=2.0,
            normalize=False,
        )

        self.assertEqual(tuple(prepared.shape), (2, 4))
        torch.testing.assert_close(prepared[0], torch.tensor([0.25, -0.25, 0.0, 0.0]))
        torch.testing.assert_close(prepared[0], prepared[1])

    def test_audio_is_center_cropped_and_optionally_normalized(self):
        from toolkit.audio.processing import prepare_audio_for_training

        waveform = torch.tensor([[0.1, 0.2, 0.4, -0.2, -0.1, 0.0]], dtype=torch.float32)

        prepared = prepare_audio_for_training(
            waveform,
            sample_rate=2,
            target_sample_rate=2,
            duration_seconds=2.0,
            normalize=True,
        )

        self.assertEqual(tuple(prepared.shape), (2, 4))
        self.assertAlmostEqual(float(prepared.abs().max()), 0.999, places=5)
        self.assertGreater(float(prepared[0, 0]), 0.0)
        self.assertLess(float(prepared[0, -1]), 0.0)


def _load_h3_source_module(name: str):
    source_path = (
        Path(__file__).parents[1]
        / "extensions_built_in"
        / "diffusion_models"
        / "minimax_h3"
        / "src"
        / f"{name}.py"
    )
    spec = importlib.util.spec_from_file_location(f"test_minimax_h3_{name}", source_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class MiniMaxH3AudioOnlyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.packing = _load_h3_source_module("packing")
        cls.transformer_module = _load_h3_source_module("transformer")

    def test_primary_audio_uses_the_audio_sigma_schedule(self):
        clean = torch.zeros(2, 4, 3)
        noise = torch.ones_like(clean)
        timesteps = torch.tensor([250.0, 750.0])

        noisy = self.packing.add_audio_noise(clean, noise, timesteps)
        expected_sigma = self.packing.remap_sigma(timesteps / 1000.0)

        torch.testing.assert_close(noisy[:, 0, 0], expected_sigma)

    def test_transformer_trains_audio_with_zero_video_rows(self):
        params = self.transformer_module.MiniMaxH3TransformerParams(
            hidden_size=16,
            num_layers=1,
            token_refiner_num_layers=1,
            num_attention_heads=2,
            attention_head_dim=8,
            ffn_hidden_size=32,
            latents_dim=4,
            audio_latents_dim=4,
            patch_size=(1, 2, 2),
            text_dim=12,
            timestep_input_dim=8,
            time_embed_hidden_size=16,
            time_embed_dim=8,
            rope_inv_freq_len=1,
        )
        model = self.transformer_module.MiniMaxH3Transformer(params)
        layout = self.packing.build_packed_sequence(
            torch.ones(3, dtype=torch.long),
            num_latent_frames=0,
            latent_height=48,
            latent_width=48,
            num_audio_latents=5,
        )
        position_ids, token_tags, video_indices, audio_indices, text_indices, _ = (
            self.packing.pad_layouts_to_batch([layout])
        )
        audio = torch.randn(1, len(audio_indices), 4, requires_grad=True)
        video = torch.empty(1, 0, 16)
        text = torch.randn(1, len(text_indices), 12)
        row_timesteps = self.packing.build_row_timesteps(layout, 0.5, 0.7).unsqueeze(0)

        video_out, audio_out = model(
            video,
            audio,
            text,
            row_timesteps,
            token_tags,
            position_ids,
            video_indices,
            audio_indices,
            text_indices,
        )
        audio_out.square().mean().backward()

        self.assertEqual(tuple(video_out.shape), (1, 0, 16))
        self.assertEqual(tuple(audio_out.shape), (1, 10, 4))
        self.assertGreater(float(audio.grad.abs().sum()), 0.0)


class AudioOnlyDatasetConfigTests(unittest.TestCase):
    def test_empty_resolutions_without_audio_still_remove_dataset(self):
        processed = preprocess_dataset_raw_config(
            [{"folder_path": "unused", "resolution": [], "do_audio": False}]
        )

        self.assertEqual(processed, [])

    def test_empty_resolutions_with_audio_create_one_audio_only_dataset(self):
        processed = preprocess_dataset_raw_config(
            [{"folder_path": "audio", "resolution": [], "do_audio": True}]
        )

        self.assertEqual(len(processed), 1)
        config = DatasetConfig(**processed[0])
        self.assertTrue(config.is_audio_only)
        self.assertTrue(config.do_audio)
        self.assertEqual(config.resolution, 768)

    def test_selected_resolutions_without_audio_keep_visual_behavior(self):
        processed = preprocess_dataset_raw_config(
            [{"folder_path": "images", "resolution": [512, 768], "do_audio": False}]
        )

        self.assertEqual([item["resolution"] for item in processed], [512, 768])
        self.assertTrue(all(not DatasetConfig(**item).is_audio_only for item in processed))

    def test_selected_resolution_with_audio_keeps_joint_behavior(self):
        processed = preprocess_dataset_raw_config(
            [{"folder_path": "videos", "resolution": [768], "do_audio": True}]
        )

        self.assertEqual(len(processed), 1)
        config = DatasetConfig(**processed[0])
        self.assertFalse(config.is_audio_only)
        self.assertTrue(config.do_audio)
        self.assertEqual(config.resolution, 768)


if __name__ == "__main__":
    unittest.main()
