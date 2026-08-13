import unittest
from unittest.mock import patch


from toolkit.character_training_sampler import (
    CharacterTrainingCandidate,
    CoverageWeightedSampler,
    build_character_sampling_plan,
)
from toolkit.data_loader import build_character_training_sampler


class CharacterTrainingSamplerTests(unittest.TestCase):
    def test_joint_views_respect_active_identity_weights_across_uneven_groups(self):
        candidates = []
        groups = (
            ("alice-bob", 5, ("alice", "bob")),
            ("bob-carl", 3, ("bob", "carl")),
            ("alice-danielle", 1, ("alice", "danielle")),
            ("alice-bob-danielle", 4, ("alice", "bob", "danielle")),
        )
        for group_name, count, identities in groups:
            for media_index in range(count):
                source_id = f"{group_name}-{media_index}"
                for identity_id in identities:
                    candidates.append(
                        CharacterTrainingCandidate(
                            key=f"{source_id}:{identity_id}",
                            source_id=source_id,
                            identity_id=identity_id,
                            identity_ids=identities,
                            view_mode="joint",
                            media_type="image",
                            dataset_path="/datasets/groups",
                        )
                    )

        plan = build_character_sampling_plan(
            candidates,
            {
                "identities": [
                    {"id": "alice", "weight": 40},
                    {"id": "bob", "weight": 30},
                    {"id": "carl", "weight": 20},
                    {"id": "danielle", "weight": 10},
                ],
                "joint_training_fraction": 1.0,
            },
        )

        identity_share = {identity_id: 0.0 for identity_id in ("alice", "bob", "carl", "danielle")}
        for candidate, probability in zip(candidates, plan.probabilities):
            identity_share[candidate.identity_id] += probability

        self.assertAlmostEqual(identity_share["alice"], 0.40)
        self.assertAlmostEqual(identity_share["bob"], 0.30)
        self.assertAlmostEqual(identity_share["carl"], 0.20)
        self.assertAlmostEqual(identity_share["danielle"], 0.10)
        self.assertAlmostEqual(sum(plan.probabilities), 1.0)

    def test_joint_and_solo_group_percentages_are_independent(self):
        candidates = [
            CharacterTrainingCandidate(
                key=f"{mode}-{context}-{index}",
                source_id=f"{mode}-{context}-{index}",
                identity_id="alice",
                identity_ids=("alice",) if context == "solo" else ("alice", "bob"),
                view_mode=mode,
                media_type="image",
                dataset_path="/datasets/people",
            )
            for mode, context, count in (
                ("focus", "solo", 2),
                ("focus", "group", 2),
                ("joint", "group", 2),
            )
            for index in range(count)
        ]

        plan = build_character_sampling_plan(
            candidates,
            {
                "identities": [{"id": "alice", "weight": 1, "solo_fraction": 0.75}],
                "joint_training_fraction": 0.20,
            },
        )

        totals = {"focus_solo": 0.0, "focus_group": 0.0, "joint": 0.0}
        for candidate, probability in zip(candidates, plan.probabilities):
            if candidate.view_mode == "joint":
                totals["joint"] += probability
            elif len(candidate.identity_ids) == 1:
                totals["focus_solo"] += probability
            else:
                totals["focus_group"] += probability

        self.assertAlmostEqual(totals["joint"], 0.20)
        self.assertAlmostEqual(totals["focus_solo"], 0.60)
        self.assertAlmostEqual(totals["focus_group"], 0.20)

    def test_solo_group_mix_can_be_set_independently_for_each_modality(self):
        candidates = [
            CharacterTrainingCandidate(
                key=f"{modality}-{context}",
                source_id=f"{modality}-{context}",
                identity_id="alice",
                identity_ids=("alice",) if context == "solo" else ("alice", "bob"),
                view_mode="focus",
                media_type=modality,
                dataset_path="/datasets/people",
            )
            for modality in ("image", "video", "audio")
            for context in ("solo", "group")
        ]

        plan = build_character_sampling_plan(
            candidates,
            {
                "identities": [{
                    "id": "alice",
                    "weight": 1,
                    "context_fractions": {"image": 1, "video": 0.25, "audio": 0},
                }],
                "joint_training_fraction": 0,
            },
        )

        self.assertEqual(plan.probabilities, (1 / 3, 0, 1 / 12, 1 / 4, 0, 1 / 3))

    def test_identity_source_mix_allocates_an_explicit_dataset_and_auto_remainder(self):
        candidates = [
            CharacterTrainingCandidate(
                key=f"{dataset}-{index}",
                source_id=f"{dataset}-{index}",
                identity_id="alice",
                identity_ids=("alice",),
                view_mode="focus",
                media_type="image",
                dataset_path=dataset,
            )
            for dataset, count in (("/datasets/a", 2), ("/datasets/b", 3), ("/datasets/c", 1))
            for index in range(count)
        ]

        plan = build_character_sampling_plan(
            candidates,
            {
                "identities": [
                    {
                        "id": "alice",
                        "weight": 1,
                        "solo_fraction": 1,
                        "source_weights": {"/datasets/a": 0.8, "*": 0.2},
                    }
                ],
                "joint_training_fraction": 0,
            },
        )

        source_share = {"/datasets/a": 0.0, "/datasets/b": 0.0, "/datasets/c": 0.0}
        for candidate, probability in zip(candidates, plan.probabilities):
            source_share[candidate.dataset_path] += probability

        self.assertAlmostEqual(source_share["/datasets/a"], 0.8)
        self.assertAlmostEqual(source_share["/datasets/b"], 0.15)
        self.assertAlmostEqual(source_share["/datasets/c"], 0.05)

    def test_source_and_context_targets_are_jointly_balanced_instead_of_applied_to_every_bucket(self):
        candidates = [
            CharacterTrainingCandidate(
                key="a-solo",
                source_id="a-solo",
                identity_id="alice",
                identity_ids=("alice",),
                view_mode="focus",
                media_type="image",
                dataset_path="/datasets/a",
            ),
            CharacterTrainingCandidate(
                key="b-group",
                source_id="b-group",
                identity_id="alice",
                identity_ids=("alice", "bob"),
                view_mode="focus",
                media_type="image",
                dataset_path="/datasets/b",
            ),
        ]

        plan = build_character_sampling_plan(
            candidates,
            {
                "identities": [{
                    "id": "alice",
                    "weight": 1,
                    "solo_fraction": 0.8,
                    "source_weights": {"/datasets/a": 0.8, "*": 0.2},
                }],
                "joint_training_fraction": 0,
            },
        )

        self.assertAlmostEqual(plan.probabilities[0], 0.8)
        self.assertAlmostEqual(plan.probabilities[1], 0.2)

    def test_runtime_sampler_covers_every_virtual_view_before_weighted_repeats(self):
        sampler = CoverageWeightedSampler(
            probabilities=[0.7, 0.2, 0.1],
            epoch_size=12,
            seed=123,
        )

        indices = list(iter(sampler))

        self.assertEqual(len(indices), 12)
        self.assertEqual(set(indices[:3]), {0, 1, 2})
        self.assertGreater(indices.count(0), indices.count(2))

    def test_dataloader_builds_sampler_from_expanded_views_and_strictly_validates_each_identity(self):
        strategy = {
            "identities": [
                {"id": "alice", "weight": 3, "solo_fraction": 1},
                {"id": "bob", "weight": 1, "solo_fraction": 1},
            ],
            "joint_training_fraction": 0,
        }
        dataset = type(
            "Dataset",
            (),
            {
                "file_list": [
                    type(
                        "Item",
                        (),
                        {
                            "path": f"/{identity}.jpg",
                            "character_dop_identity_id": identity,
                            "character_training_identity_ids": (identity,),
                            "character_training_view_mode": "focus",
                            "is_video": False,
                            "is_audio_only": False,
                            "dataset_config": type(
                                "Config",
                                (),
                                {"folder_path": "/datasets/people", "character_training": strategy},
                            )(),
                        },
                    )()
                    for identity in ("alice", "bob")
                ]
            },
        )()
        dataset.dataset_config = dataset.file_list[0].dataset_config
        concat = type("Concat", (), {"datasets": [dataset]})()

        sampler = build_character_training_sampler(concat, seed=7)

        self.assertIsNotNone(sampler)
        self.assertEqual(len(sampler), 4)

        dataset.file_list = [dataset.file_list[0]]
        with self.assertRaisesRegex(ValueError, "bob"):
            build_character_training_sampler(concat, seed=7)


if __name__ == "__main__":
    unittest.main()
