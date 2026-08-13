import json
import os
import subprocess
import tempfile
import unittest
from pathlib import Path


class DatasetBalanceUITests(unittest.TestCase):
    def test_missing_datasets_root_returns_an_empty_inventory(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            missing_root = Path(tmp_dir) / "datasets"
            module_path = (
                Path(__file__).parents[1] / "ui" / "src" / "server" / "datasetStats.ts"
            ).resolve()
            script = f"""
import {{ pathToFileURL }} from 'node:url';
const {{ collectDatasetInventories }} = await import(pathToFileURL({json.dumps(str(module_path))}).href);
console.log(JSON.stringify(await collectDatasetInventories({json.dumps(str(missing_root))}, [{json.dumps(str(missing_root / 'one'))}])));
"""
            completed = subprocess.run(
                ["node", "--experimental-strip-types", "--input-type=module", "-e", script],
                check=True,
                capture_output=True,
                text=True,
            )

            self.assertEqual(json.loads(completed.stdout), {})

    def test_inventory_counts_source_media_and_named_character_views(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            datasets_root = Path(tmp_dir) / "datasets"
            dataset_dir = datasets_root / "shared-scenes"
            dataset_dir.mkdir(parents=True)
            unselected_dataset = datasets_root / "unselected"
            unselected_dataset.mkdir()
            (unselected_dataset / "ignored.jpg").touch()
            (dataset_dir / "one.jpg").touch()
            (dataset_dir / "two.png").touch()
            (dataset_dir / "scene.mp4").touch()
            (dataset_dir / "voice.wav").touch()
            annotation_root = dataset_dir / "_character_dop"
            annotation_root.mkdir()
            (annotation_root / "identities.json").write_text(
                json.dumps(
                    {
                        "version": 1,
                        "identities": [
                            {
                                "id": "alice",
                                "display_name": "Alice",
                                "trigger_word": "AliceToken",
                                "class_prompt": "a woman",
                            },
                            {
                                "id": "bob",
                                "display_name": "Bob",
                                "trigger_word": "BobToken",
                                "class_prompt": "a man",
                            },
                        ],
                    }
                ),
                encoding="utf-8",
            )
            for identity_id in ("alice", "bob"):
                visual_dir = annotation_root / "identities" / identity_id / "visual"
                visual_dir.mkdir(parents=True)
                (visual_dir / "one.png").touch()
            audio_dir = annotation_root / "identities" / "alice" / "audio"
            audio_dir.mkdir(parents=True)
            (audio_dir / "scene.json").write_text('{"character_intervals": []}', encoding="utf-8")
            module_path = (
                Path(__file__).parents[1] / "ui" / "src" / "server" / "datasetStats.ts"
            ).resolve()
            script = f"""
import {{ pathToFileURL }} from 'node:url';
const {{ collectDatasetInventories }} = await import(pathToFileURL({json.dumps(str(module_path))}).href);
console.log(JSON.stringify(await collectDatasetInventories({json.dumps(str(datasets_root))}, [{json.dumps(str(dataset_dir))}])));
"""
            completed = subprocess.run(
                ["node", "--experimental-strip-types", "--input-type=module", "-e", script],
                check=True,
                capture_output=True,
                text=True,
            )
            inventories = json.loads(completed.stdout)
            self.assertEqual(list(inventories), [str(dataset_dir)])
            inventory = inventories[str(dataset_dir)]

            self.assertEqual(inventory["images"], {"sources": 2, "assignedSources": 1, "characterViews": 2})
            self.assertEqual(inventory["videos"], {"sources": 1, "assignedSources": 1, "characterViews": 1})
            self.assertEqual(inventory["audio"], {"sources": 1, "assignedSources": 0, "characterViews": 0})
            self.assertEqual(inventory["identityCount"], 2)

    def test_inventory_ignores_local_identity_removed_from_shared_registry(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            datasets_root = Path(tmp_dir) / "datasets"
            dataset_dir = datasets_root / "characters"
            dataset_dir.mkdir(parents=True)
            (dataset_dir / "one.jpg").touch()
            identity = {
                "id": "alice",
                "display_name": "Alice",
                "trigger_word": "AliceToken",
                "class_prompt": "a woman",
            }
            (datasets_root / "_character_dop_identities.json").write_text(
                json.dumps({"version": 1, "identities": []}),
                encoding="utf-8",
            )
            annotation_root = dataset_dir / "_character_dop"
            annotation_root.mkdir()
            (annotation_root / "identities.json").write_text(
                json.dumps({"version": 1, "identities": [identity]}),
                encoding="utf-8",
            )
            visual_dir = annotation_root / "identities" / "alice" / "visual"
            visual_dir.mkdir(parents=True)
            (visual_dir / "one.png").touch()
            module_path = (
                Path(__file__).parents[1] / "ui" / "src" / "server" / "datasetStats.ts"
            ).resolve()
            script = f"""
import {{ pathToFileURL }} from 'node:url';
const {{ collectDatasetInventories }} = await import(pathToFileURL({json.dumps(str(module_path))}).href);
console.log(JSON.stringify(await collectDatasetInventories({json.dumps(str(datasets_root))}, [{json.dumps(str(dataset_dir))}])));
"""
            completed = subprocess.run(
                ["node", "--experimental-strip-types", "--input-type=module", "-e", script],
                check=True,
                capture_output=True,
                text=True,
            )
            inventory = json.loads(completed.stdout)[str(dataset_dir)]

            self.assertEqual(inventory["identityCount"], 0)
            self.assertEqual(inventory["images"]["assignedSources"], 0)
            self.assertEqual(inventory["images"]["characterViews"], 0)

    def test_inventory_reports_invalid_catalog_metadata(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            dataset_dir = Path(tmp_dir) / "datasets" / "invalid-catalog"
            dataset_dir.mkdir(parents=True)
            (dataset_dir / "one.jpg").touch()
            annotation_root = dataset_dir / "_character_dop"
            annotation_root.mkdir()
            (annotation_root / "identities.json").write_text(
                json.dumps(
                    {
                        "version": 1,
                        "identities": [
                            {
                                "id": "alice",
                                "display_name": "Alice",
                                "trigger_word": "AliceToken",
                                "class_prompt": "a person",
                            },
                            {
                                "id": "alice-copy",
                                "display_name": "Alice copy",
                                "trigger_word": "alicetoken-extra",
                                "class_prompt": "a person",
                            },
                        ],
                    }
                ),
                encoding="utf-8",
            )
            inventory = self._collect_single_inventory(dataset_dir)

            self.assertEqual(inventory["identityCount"], 0)
            self.assertIn("cannot contain one another", inventory["error"])

    def test_inventory_ignores_wrong_visual_suffix_and_reports_bad_audio_json(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            dataset_dir = Path(tmp_dir) / "datasets" / "invalid-annotations"
            dataset_dir.mkdir(parents=True)
            (dataset_dir / "one.jpg").touch()
            annotation_root = dataset_dir / "_character_dop"
            annotation_root.mkdir()
            (annotation_root / "identities.json").write_text(
                json.dumps(
                    {
                        "version": 1,
                        "identities": [
                            {
                                "id": "alice",
                                "display_name": "Alice",
                                "trigger_word": "AliceToken",
                                "class_prompt": "a person",
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            visual_dir = annotation_root / "identities" / "alice" / "visual"
            audio_dir = annotation_root / "identities" / "alice" / "audio"
            visual_dir.mkdir(parents=True)
            audio_dir.mkdir(parents=True)
            (visual_dir / "one.npy").touch()
            (audio_dir / "one.json").write_text('{"wrong_key": []}', encoding="utf-8")

            inventory = self._collect_single_inventory(dataset_dir)

            self.assertEqual(inventory["images"]["assignedSources"], 0)
            self.assertEqual(inventory["images"]["characterViews"], 0)
            self.assertIn("missing character_intervals", inventory["error"])

    def test_inventory_rejects_annotation_symlink_escape(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            datasets_root = Path(tmp_dir) / "datasets"
            dataset_dir = datasets_root / "characters"
            outside_dir = Path(tmp_dir) / "outside"
            dataset_dir.mkdir(parents=True)
            outside_dir.mkdir()
            (dataset_dir / "one.jpg").touch()
            (outside_dir / "identities.json").write_text(
                json.dumps({"version": 1, "identities": []}),
                encoding="utf-8",
            )
            try:
                os.symlink(outside_dir, dataset_dir / "_character_dop", target_is_directory=True)
            except OSError as exc:
                self.skipTest(f"directory symlinks are unavailable: {exc}")
            module_path = (
                Path(__file__).parents[1] / "ui" / "src" / "server" / "datasetStats.ts"
            ).resolve()
            script = f"""
import {{ pathToFileURL }} from 'node:url';
const {{ collectDatasetInventories }} = await import(pathToFileURL({json.dumps(str(module_path))}).href);
console.log(JSON.stringify(await collectDatasetInventories({json.dumps(str(datasets_root))}, [{json.dumps(str(dataset_dir))}])));
"""
            completed = subprocess.run(
                ["node", "--experimental-strip-types", "--input-type=module", "-e", script],
                check=True,
                capture_output=True,
                text=True,
            )
            inventory = json.loads(completed.stdout)[str(dataset_dir)]

            self.assertIn("escapes its dataset", inventory["error"])
            self.assertEqual(inventory["identityCount"], 0)

    def _collect_single_inventory(self, dataset_dir):
        module_path = (
            Path(__file__).parents[1] / "ui" / "src" / "server" / "datasetStats.ts"
        ).resolve()
        script = f"""
import {{ pathToFileURL }} from 'node:url';
const {{ collectDatasetInventory }} = await import(pathToFileURL({json.dumps(str(module_path))}).href);
console.log(JSON.stringify(await collectDatasetInventory({json.dumps(str(dataset_dir))})));
"""
        completed = subprocess.run(
            ["node", "--experimental-strip-types", "--input-type=module", "-e", script],
            check=True,
            capture_output=True,
            text=True,
        )
        return json.loads(completed.stdout)

    def test_balance_projection_matches_loader_repeats_flips_and_character_views(self):
        module_path = (
            Path(__file__).parents[1]
            / "ui"
            / "src"
            / "app"
            / "jobs"
            / "new"
            / "datasetBalance.ts"
        ).resolve()
        script = f"""
import {{ pathToFileURL }} from 'node:url';
const {{ calculateDatasetBalance }} = await import(pathToFileURL({json.dumps(str(module_path))}).href);
const stats = {{
  '/datasets/characters': {{
    path: '/datasets/characters', identityCount: 2,
    images: {{ sources: 10, assignedSources: 4, characterViews: 8 }},
    videos: {{ sources: 0, assignedSources: 0, characterViews: 0 }},
    audio: {{ sources: 0, assignedSources: 0, characterViews: 0 }}
  }},
  '/datasets/regularization': {{
    path: '/datasets/regularization', identityCount: 0,
    images: {{ sources: 20, assignedSources: 0, characterViews: 0 }},
    videos: {{ sources: 0, assignedSources: 0, characterViews: 0 }},
    audio: {{ sources: 0, assignedSources: 0, characterViews: 0 }}
  }}
}};
const rows = calculateDatasetBalance([
  {{ folder_path: '/datasets/characters', resolution: [512], do_audio: false, num_frames: 1, num_repeats: 2, flip_x: true, flip_y: false, network_weight: 1, is_reg: false }},
  {{ folder_path: '/datasets/regularization', resolution: [512], do_audio: false, num_frames: 1, num_repeats: 1, flip_x: false, flip_y: false, network_weight: 0.5, is_reg: true }}
], stats, {{ modelGroup: 'video', characterDop: true, globalTrigger: true }});
console.log(JSON.stringify(rows));
"""
        completed = subprocess.run(
            ["node", "--experimental-strip-types", "--input-type=module", "-e", script],
            check=True,
            capture_output=True,
            text=True,
        )
        rows = json.loads(completed.stdout)

        self.assertEqual(rows[0]["sourceItems"], 10)
        self.assertEqual(rows[0]["trainingViews"], 14)
        self.assertEqual(rows[0]["augmentationFactor"], 4)
        self.assertEqual(rows[0]["effectiveItems"], 56)
        self.assertAlmostEqual(rows[0]["samplingShare"], 0.5)
        self.assertEqual(rows[1]["effectiveItems"], 20)
        self.assertAlmostEqual(rows[1]["samplingShare"], 0.5)
        self.assertEqual(rows[1]["networkWeight"], 0.5)

    def test_audio_model_flips_match_non_audio_only_loader_behavior(self):
        module_path = (
            Path(__file__).parents[1]
            / "ui"
            / "src"
            / "app"
            / "jobs"
            / "new"
            / "datasetBalance.ts"
        ).resolve()
        script = f"""
import {{ pathToFileURL }} from 'node:url';
const {{ calculateDatasetBalance }} = await import(pathToFileURL({json.dumps(str(module_path))}).href);
const path = '/datasets/audio';
const rows = calculateDatasetBalance([
  {{ folder_path: path, resolution: [512], do_audio: false, num_frames: 1, num_repeats: 1, flip_x: true, flip_y: true, network_weight: 1, is_reg: false }}
], {{ [path]: {{
  path, identityCount: 0,
  images: {{ sources: 0, assignedSources: 0, characterViews: 0 }},
  videos: {{ sources: 0, assignedSources: 0, characterViews: 0 }},
  audio: {{ sources: 3, assignedSources: 0, characterViews: 0 }}
}} }}, {{ modelGroup: 'audio', characterDop: false, globalTrigger: false }});
console.log(JSON.stringify(rows));
"""
        completed = subprocess.run(
            ["node", "--experimental-strip-types", "--input-type=module", "-e", script],
            check=True,
            capture_output=True,
            text=True,
        )
        row = json.loads(completed.stdout)[0]

        self.assertEqual(row["flipFactor"], 4)
        self.assertEqual(row["effectiveItems"], 12)

    def test_unassigned_regularization_sources_do_not_raise_character_dop_warning(self):
        module_path = (
            Path(__file__).parents[1]
            / "ui"
            / "src"
            / "app"
            / "jobs"
            / "new"
            / "datasetBalance.ts"
        ).resolve()
        script = f"""
import {{ pathToFileURL }} from 'node:url';
const {{ calculateDatasetBalance }} = await import(pathToFileURL({json.dumps(str(module_path))}).href);
const path = '/datasets/regularization';
const rows = calculateDatasetBalance([
  {{ folder_path: path, resolution: [512], num_frames: 1, num_repeats: 1, flip_x: false, flip_y: false, network_weight: 1, is_reg: true }}
], {{ [path]: {{
  path, identityCount: 1,
  images: {{ sources: 5, assignedSources: 2, characterViews: 2 }},
  videos: {{ sources: 0, assignedSources: 0, characterViews: 0 }},
  audio: {{ sources: 0, assignedSources: 0, characterViews: 0 }}
}} }}, {{ modelGroup: 'video', characterDop: true, globalTrigger: false }});
console.log(JSON.stringify(rows));
"""
        completed = subprocess.run(
            ["node", "--experimental-strip-types", "--input-type=module", "-e", script],
            check=True,
            capture_output=True,
            text=True,
        )
        rows = json.loads(completed.stdout)

        self.assertEqual(rows[0]["trainingViews"], 5)
        self.assertEqual(rows[0]["unassignedCharacterSources"], 0)

    def test_dataset_trigger_prevents_false_unassigned_character_warning(self):
        module_path = (
            Path(__file__).parents[1]
            / "ui"
            / "src"
            / "app"
            / "jobs"
            / "new"
            / "datasetBalance.ts"
        ).resolve()
        script = f"""
import {{ pathToFileURL }} from 'node:url';
const {{ calculateDatasetBalance }} = await import(pathToFileURL({json.dumps(str(module_path))}).href);
const path = '/datasets/characters';
const rows = calculateDatasetBalance([
  {{ folder_path: path, resolution: [512], num_frames: 1, num_repeats: 1, flip_x: false, flip_y: false, network_weight: 1, is_reg: false, trigger_word: 'FallbackToken' }}
], {{ [path]: {{
  path, identityCount: 1,
  images: {{ sources: 4, assignedSources: 3, characterViews: 3 }},
  videos: {{ sources: 0, assignedSources: 0, characterViews: 0 }},
  audio: {{ sources: 0, assignedSources: 0, characterViews: 0 }}
}} }}, {{ modelGroup: 'video', characterDop: true, globalTrigger: false }});
console.log(JSON.stringify(rows));
"""
        completed = subprocess.run(
            ["node", "--experimental-strip-types", "--input-type=module", "-e", script],
            check=True,
            capture_output=True,
            text=True,
        )

        self.assertEqual(json.loads(completed.stdout)[0]["unassignedCharacterSources"], 0)

    def test_dataset_can_disable_named_annotation_expansion(self):
        module_path = (
            Path(__file__).parents[1]
            / "ui"
            / "src"
            / "app"
            / "jobs"
            / "new"
            / "datasetBalance.ts"
        ).resolve()
        script = f"""
import {{ pathToFileURL }} from 'node:url';
const {{ calculateDatasetBalance }} = await import(pathToFileURL({json.dumps(str(module_path))}).href);
const path = '/datasets/characters';
const rows = calculateDatasetBalance([
  {{ folder_path: path, resolution: [512], num_frames: 1, num_repeats: 1, flip_x: false, flip_y: false, network_weight: 1, is_reg: false, character_dop_use_dataset_annotations: false }}
], {{ [path]: {{
  path, identityCount: 2,
  images: {{ sources: 5, assignedSources: 3, characterViews: 6 }},
  videos: {{ sources: 0, assignedSources: 0, characterViews: 0 }},
  audio: {{ sources: 0, assignedSources: 0, characterViews: 0 }}
}} }}, {{ modelGroup: 'video', characterDop: true, globalTrigger: true }});
console.log(JSON.stringify(rows));
"""
        completed = subprocess.run(
            ["node", "--experimental-strip-types", "--input-type=module", "-e", script],
            check=True,
            capture_output=True,
            text=True,
        )
        rows = json.loads(completed.stdout)

        self.assertEqual(rows[0]["trainingViews"], 5)
        self.assertFalse(rows[0]["usesCharacterViews"])

    def test_resolution_expansion_matches_dataset_preprocessing(self):
        module_path = (
            Path(__file__).parents[1]
            / "ui"
            / "src"
            / "app"
            / "jobs"
            / "new"
            / "datasetBalance.ts"
        ).resolve()
        script = f"""
import {{ pathToFileURL }} from 'node:url';
const {{ calculateDatasetBalance }} = await import(pathToFileURL({json.dumps(str(module_path))}).href);
const path = '/datasets/images';
const inventory = {{
  path, identityCount: 0,
  images: {{ sources: 10, assignedSources: 0, characterViews: 0 }},
  videos: {{ sources: 0, assignedSources: 0, characterViews: 0 }},
  audio: {{ sources: 0, assignedSources: 0, characterViews: 0 }}
}};
const rows = calculateDatasetBalance([
  {{ folder_path: path, resolution: [512, 768], num_frames: 1, num_repeats: 1, flip_x: false, flip_y: false, network_weight: 1, is_reg: false }},
  {{ folder_path: path, resolution: [], do_audio: false, num_frames: 1, num_repeats: 1, flip_x: false, flip_y: false, network_weight: 1, is_reg: false }}
], {{ [path]: inventory }}, {{ modelGroup: 'video', characterDop: false, globalTrigger: false }});
console.log(JSON.stringify(rows));
"""
        completed = subprocess.run(
            ["node", "--experimental-strip-types", "--input-type=module", "-e", script],
            check=True,
            capture_output=True,
            text=True,
        )
        rows = json.loads(completed.stdout)

        self.assertEqual(rows[0]["resolutionFactor"], 2)
        self.assertEqual(rows[0]["effectiveItems"], 20)
        self.assertEqual(rows[1]["resolutionFactor"], 0)
        self.assertEqual(rows[1]["effectiveItems"], 0)
        self.assertEqual(rows[0]["samplingShare"], 1)

    def test_job_creation_exposes_dataset_balance_and_inventory_route(self):
        root = Path(__file__).parents[1]
        source = (root / "ui" / "src" / "app" / "jobs" / "new" / "SimpleJob.tsx").read_text(
            encoding="utf-8"
        )
        route = root / "ui" / "src" / "app" / "api" / "datasets" / "stats" / "route.ts"

        self.assertTrue(route.is_file())
        self.assertIn("Dataset balance", source)
        self.assertIn("Sampling share", source)
        self.assertIn("LoRA weight does not change sampling share", source)
        self.assertIn("Character DOP views", source)


if __name__ == "__main__":
    unittest.main()
