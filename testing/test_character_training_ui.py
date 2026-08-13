import json
import subprocess
import unittest
from pathlib import Path


class CharacterTrainingUITests(unittest.TestCase):
    def run_validation(self, strategy, datasets, stats):
        module_path = (
            Path(__file__).parents[1]
            / "ui"
            / "src"
            / "app"
            / "jobs"
            / "new"
            / "characterTrainingBalance.ts"
        ).resolve()
        script = f"""
import {{ pathToFileURL }} from 'node:url';
const {{ validateCharacterTrainingCoverage }} = await import(pathToFileURL({json.dumps(str(module_path))}).href);
console.log(JSON.stringify(validateCharacterTrainingCoverage(
  {json.dumps(strategy)}, {json.dumps(datasets)}, {json.dumps(stats)}
)));
"""
        completed = subprocess.run(
            ["node", "--experimental-strip-types", "--input-type=module", "-e", script],
            check=True,
            capture_output=True,
            text=True,
        )
        return json.loads(completed.stdout)

    def test_every_selected_identity_must_have_representation_in_non_regularization_sources(self):
        strategy = {
            "identities": [
                {"id": "alice", "weight": 1, "solo_fraction": 1},
                {"id": "bob", "weight": 1, "solo_fraction": 1},
            ],
            "joint_training_fraction": 0,
        }
        datasets = [
            {"folder_path": "/train", "is_reg": False},
            {"folder_path": "/reg", "is_reg": True},
        ]
        empty = {"sources": 0, "solo": 0, "group": 0}
        stats = {
            "/train": {
                "path": "/train",
                "identities": [
                    {
                        "id": "alice",
                        "images": {"sources": 2, "solo": 2, "group": 0},
                        "videosVisual": empty,
                        "videosAudio": empty,
                        "audio": empty,
                    }
                ],
            },
            "/reg": {
                "path": "/reg",
                "identities": [
                    {
                        "id": "bob",
                        "images": {"sources": 10, "solo": 10, "group": 0},
                        "videosVisual": empty,
                        "videosAudio": empty,
                        "audio": empty,
                    }
                ],
            },
        }

        errors = self.run_validation(strategy, datasets, stats)

        self.assertEqual(len(errors), 1)
        self.assertIn("bob", errors[0].lower())

    def test_nonzero_joint_and_group_requests_block_when_coverage_is_missing(self):
        strategy = {
            "identities": [{"id": "alice", "weight": 1, "solo_fraction": 0.5}],
            "joint_training_fraction": 0.25,
        }
        datasets = [{"folder_path": "/train", "is_reg": False}]
        empty = {"sources": 0, "solo": 0, "group": 0}
        stats = {
            "/train": {
                "path": "/train",
                "identities": [{
                    "id": "alice",
                    "images": {"sources": 2, "solo": 2, "group": 0},
                    "videosVisual": empty,
                    "videosAudio": empty,
                    "audio": empty,
                }],
            }
        }

        errors = self.run_validation(strategy, datasets, stats)

        self.assertTrue(any("group" in error.lower() for error in errors))
        self.assertTrue(any("joint" in error.lower() for error in errors))

    def test_joint_training_requires_a_shared_source_for_two_selected_identities(self):
        strategy = {
            "identities": [
                {"id": "alice", "weight": 1, "solo_fraction": 0},
                {"id": "bob", "weight": 1, "solo_fraction": 0},
            ],
            "joint_training_fraction": 0.25,
        }
        datasets = [{"folder_path": "/train", "is_reg": False}]
        empty = {"sources": 0, "solo": 0, "group": 0}
        group = {"sources": 1, "solo": 0, "group": 1}
        stats = {
            "/train": {
                "path": "/train",
                "identities": [
                    {"id": "alice", "images": group, "videosVisual": empty, "videosAudio": empty, "audio": empty},
                    {"id": "bob", "images": group, "videosVisual": empty, "videosAudio": empty, "audio": empty},
                ],
                "jointIdentityPairs": [["alice", "carl"], ["bob", "danielle"]],
            }
        }

        errors = self.run_validation(strategy, datasets, stats)
        self.assertTrue(any("joint" in error.lower() for error in errors))

        stats["/train"]["jointIdentityPairs"].append(["alice", "bob"])
        errors = self.run_validation(strategy, datasets, stats)
        self.assertFalse(any("joint" in error.lower() for error in errors))


if __name__ == "__main__":
    unittest.main()
