import copy
import sys
import tempfile
import types
import unittest
import importlib.util
import wave
from pathlib import Path
from unittest.mock import patch

import torch


def _stub_optional_imports():
    """Load config parsing without requiring a full training environment."""
    added_modules = []

    def add_stub(name, module):
        if name not in sys.modules:
            sys.modules[name] = module
            added_modules.append(name)

    hf = types.ModuleType("huggingface_hub")
    hf_utils = types.ModuleType("huggingface_hub.utils")
    hf_tqdm = types.ModuleType("huggingface_hub.utils.tqdm")
    hf_tqdm.is_tqdm_disabled = lambda _level: None
    add_stub("huggingface_hub", hf)
    add_stub("huggingface_hub.utils", hf_utils)
    add_stub("huggingface_hub.utils.tqdm", hf_tqdm)

    prompt_utils = types.ModuleType("toolkit.prompt_utils")
    prompt_utils.PromptEmbeds = object
    add_stub("toolkit.prompt_utils", prompt_utils)

    album_artwork = types.ModuleType("toolkit.audio.album_artwork")
    album_artwork.add_album_artwork = lambda *_args, **_kwargs: None
    add_stub("toolkit.audio.album_artwork", album_artwork)

    torchao = types.ModuleType("torchao")
    torchao_quantization = types.ModuleType("torchao.quantization")
    quant_primitives = types.ModuleType("torchao.quantization.quant_primitives")
    quant_primitives._DTYPE_TO_BIT_WIDTH = {}
    add_stub("torchao", torchao)
    add_stub("torchao.quantization", torchao_quantization)
    add_stub("torchao.quantization.quant_primitives", quant_primitives)
    return added_modules


_config_was_loaded = "toolkit.config_modules" in sys.modules
_stubbed_modules = _stub_optional_imports()
try:
    from toolkit.config_modules import DatasetConfig, preprocess_dataset_raw_config
finally:
    if not _config_was_loaded:
        sys.modules.pop("toolkit.config_modules", None)
    for _module_name in reversed(_stubbed_modules):
        sys.modules.pop(_module_name, None)


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


class AudioSegmentationTests(unittest.TestCase):
    def test_video_audio_loader_imports_the_segment_metadata_it_constructs(self):
        import toolkit.dataloader_mixins as dataloader_mixins
        from toolkit.audio.processing import AudioSegment

        self.assertIs(dataloader_mixins.AudioSegment, AudioSegment)

    def test_audio_uses_next_whole_second_duration_bucket(self):
        from toolkit.audio.processing import plan_audio_segment

        segment = plan_audio_segment(
            source_duration_seconds=2.4,
        )

        self.assertAlmostEqual(segment.start_seconds, 0.0)
        self.assertAlmostEqual(segment.duration_seconds, 2.4)
        self.assertAlmostEqual(segment.target_duration_seconds, 3.0)

    def test_variable_audio_latents_are_padded_to_the_batch_maximum(self):
        from toolkit.audio.processing import stack_audio_latents

        short = torch.ones(4, 32)
        long = torch.full((6, 32), 2.0)

        batch = stack_audio_latents([short, long])

        self.assertEqual(tuple(batch.shape), (2, 6, 32))
        torch.testing.assert_close(batch[0, :4], short)
        torch.testing.assert_close(batch[0, 4:], torch.zeros(2, 32))
        torch.testing.assert_close(batch[1], long)

    def test_long_audio_remains_one_segment_in_its_duration_bucket(self):
        from toolkit.audio.processing import (
            audio_segment_frame_range,
            plan_audio_segment,
        )

        segment = plan_audio_segment(
            source_duration_seconds=12.4,
        )

        self.assertAlmostEqual(segment.start_seconds, 0.0)
        self.assertAlmostEqual(segment.duration_seconds, 12.4)
        self.assertAlmostEqual(segment.target_duration_seconds, 13.0)
        self.assertEqual(audio_segment_frame_range(segment, 10), (0, 124))

    def test_audio_segment_loader_reads_only_the_planned_source_range(self):
        from toolkit.audio.processing import AudioSegment, load_audio_segment

        calls = []
        fake_torchaudio = types.ModuleType("torchaudio")

        def fake_load(path, frame_offset=0, num_frames=-1):
            calls.append((path, frame_offset, num_frames))
            source = torch.arange(100, dtype=torch.float32).unsqueeze(0)
            return source[..., frame_offset : frame_offset + num_frames], 10

        fake_torchaudio.load = fake_load
        segment = AudioSegment(
            start_seconds=2.0,
            duration_seconds=3.0,
            target_duration_seconds=3.0,
        )

        with patch.dict(sys.modules, {"torchaudio": fake_torchaudio}):
            waveform, sample_rate = load_audio_segment("example.mp4", segment, 10)

        self.assertEqual(calls, [("example.mp4", 20, 30)])
        self.assertEqual(sample_rate, 10)
        torch.testing.assert_close(waveform, torch.arange(20, 50).unsqueeze(0).float())

    def test_audio_segment_loader_falls_back_when_container_cannot_seek(self):
        from toolkit.audio.processing import AudioSegment, load_audio_segment

        calls = []
        fake_torchaudio = types.ModuleType("torchaudio")

        def fake_load(path, frame_offset=0, num_frames=-1):
            calls.append((frame_offset, num_frames))
            if num_frames != -1:
                raise RuntimeError("backend does not support seeking")
            return torch.arange(100, dtype=torch.float32).unsqueeze(0), 10

        fake_torchaudio.load = fake_load
        segment = AudioSegment(2.0, 3.0, 3.0)

        with patch.dict(sys.modules, {"torchaudio": fake_torchaudio}):
            waveform, sample_rate = load_audio_segment("example.mp4", segment, 10)

        self.assertEqual(calls, [(20, 30), (0, -1)])
        self.assertEqual(sample_rate, 10)
        torch.testing.assert_close(waveform, torch.arange(20, 50).unsqueeze(0).float())

    @unittest.skipUnless(importlib.util.find_spec("av"), "PyAV is not installed")
    def test_audio_only_file_item_uses_its_segment_bucket_width(self):
        from toolkit.audio.processing import (
            AudioSegment,
            load_audio_segment,
            plan_audio_segment,
            prepare_audio_for_training,
        )
        from toolkit.data_transfer_object.data_loader import (
            DataLoaderBatchDTO,
            FileItemDTO,
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            audio_path = Path(temp_dir) / "variable.wav"
            with wave.open(str(audio_path), "wb") as audio_file:
                audio_file.setnchannels(1)
                audio_file.setsampwidth(2)
                audio_file.setframerate(100)
                audio_file.writeframes(b"\x00\x00" * 510)

            dataset_config = DatasetConfig(
                folder_path=temp_dir,
                resolution=[],
                do_audio=True,
                buckets=False,
            )
            file_item = FileItemDTO(
                path=str(audio_path),
                dataset_config=dataset_config,
                size_database={},
                dataset_root=temp_dir,
                sample_rate=32000,
            )
            segment = plan_audio_segment(
                file_item.audio_source_duration_seconds,
            )
            segment_item = copy.deepcopy(file_item)
            segment_item.set_audio_segment(segment)
            waveform, sample_rate = load_audio_segment(
                str(audio_path),
                segment_item.audio_segment,
                file_item.audio_source_sample_rate,
            )
            prepared = prepare_audio_for_training(
                waveform,
                sample_rate,
                target_sample_rate=32000,
                duration_seconds=segment_item.audio_segment.target_duration_seconds,
            )
            short_item = copy.deepcopy(file_item)
            short_item.set_audio_segment(AudioSegment(0.0, 1.5, 2.0))
            raw_items = [segment_item, short_item]
            for item in raw_items:
                item.load_and_process_audio()
            raw_batch = DataLoaderBatchDTO(file_items=raw_items)
            for item, row_count in zip(raw_items, (4, 6)):
                item._encoded_latent = torch.ones(row_count, 32)
                item.is_latent_cached = True
            cached_batch = DataLoaderBatchDTO(file_items=raw_items)

        self.assertAlmostEqual(file_item.audio_source_duration_seconds, 5.1)
        self.assertEqual(segment_item.width, 6000)
        self.assertEqual(segment_item.crop_width, 6000)
        self.assertAlmostEqual(segment_item.audio_segment.duration_seconds, 5.1)
        self.assertAlmostEqual(segment_item.audio_segment.target_duration_seconds, 6.0)
        self.assertNotEqual(
            segment_item.get_latent_path(),
            short_item.get_latent_path(),
        )
        self.assertEqual(tuple(prepared.shape), (2, 192000))
        self.assertIsNone(raw_batch.tensor)
        self.assertIsNone(raw_batch.audio_tensor)
        self.assertEqual(len(raw_batch.audio_data), 2)
        self.assertIsNone(cached_batch.tensor)
        self.assertEqual(tuple(cached_batch.latents.shape), (2, 6, 32))


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

    def test_legacy_audio_maximum_is_ignored(self):
        config = DatasetConfig(
            folder_path="audio",
            resolution=768,
            do_audio=True,
            is_audio_only=True,
            audio_duration_seconds=0.1,
        )

        self.assertFalse(hasattr(config, "audio_duration_seconds"))

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
