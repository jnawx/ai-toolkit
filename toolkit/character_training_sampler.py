"""Plan selected-identity Character LoRA sampling behind one small interface.

The UI and data loader both use :func:`build_character_sampling_plan`.  Keeping
the balancing rules here prevents job previews and runtime sampling from
quietly disagreeing as more dimensions (context, modality, and source) are
added.
"""

from dataclasses import dataclass
import math
import os
import random
from typing import Any, Sequence

from torch.utils.data import Sampler


@dataclass(frozen=True)
class CharacterTrainingCandidate:
    """One virtual training view of a physical media source."""

    key: str
    source_id: str
    identity_id: str
    identity_ids: tuple[str, ...]
    view_mode: str
    media_type: str
    dataset_path: str


@dataclass(frozen=True)
class CharacterSamplingPlan:
    """Normalized probability for each input candidate, in input order."""

    probabilities: tuple[float, ...]


class CoverageWeightedSampler(Sampler[int]):
    """Cover every eligible view once, then fill the epoch by target weight."""

    def __init__(self, probabilities: Sequence[float], epoch_size: int, seed: int = 0):
        self.probabilities = tuple(float(value) for value in probabilities)
        self.eligible_indices = [
            index for index, probability in enumerate(self.probabilities) if probability > 0.0
        ]
        if not self.eligible_indices:
            raise ValueError("character sampling plan has no eligible views")
        self.epoch_size = int(epoch_size)
        if self.epoch_size < len(self.eligible_indices):
            raise ValueError(
                "character sampling epoch must be large enough to cover every eligible view"
            )
        total = sum(self.probabilities)
        if not math.isfinite(total) or abs(total - 1.0) > 1e-6:
            raise ValueError("character sampling probabilities must total 1")
        self.seed = int(seed)
        self.epoch = 0

    def __len__(self) -> int:
        return self.epoch_size

    def __iter__(self):
        rng = random.Random(self.seed + self.epoch)
        self.epoch += 1
        coverage = list(self.eligible_indices)
        rng.shuffle(coverage)

        remaining = self.epoch_size - len(coverage)
        residual_weights = [
            max(0.0, probability * self.epoch_size - (1 if probability > 0 else 0))
            for probability in self.probabilities
        ]
        residual_total = sum(residual_weights)
        if residual_total <= 0:
            residual_weights = list(self.probabilities)
            residual_total = sum(residual_weights)
        exact_counts = [weight / residual_total * remaining for weight in residual_weights]
        repeat_counts = [math.floor(value) for value in exact_counts]
        unallocated = remaining - sum(repeat_counts)
        remainder_order = sorted(
            self.eligible_indices,
            key=lambda index: (exact_counts[index] - repeat_counts[index], self.probabilities[index]),
            reverse=True,
        )
        for index in remainder_order[:unallocated]:
            repeat_counts[index] += 1
        repeats = [
            index
            for index, count in enumerate(repeat_counts)
            for _ in range(count)
        ]
        rng.shuffle(repeats)
        return iter(coverage + repeats)


def _identity_weights(raw_strategy: dict[str, Any]) -> dict[str, float]:
    identities = raw_strategy.get("identities", [])
    if not isinstance(identities, list) or not identities:
        raise ValueError("character training requires at least one selected identity")
    weights: dict[str, float] = {}
    for raw_identity in identities:
        if not isinstance(raw_identity, dict):
            raise ValueError("character training identities must be objects")
        identity_id = str(raw_identity.get("id", "")).strip()
        if not identity_id or identity_id in weights:
            raise ValueError("character training identity ids must be non-empty and unique")
        try:
            weight = float(raw_identity.get("weight", 1.0))
        except (TypeError, ValueError) as exc:
            raise ValueError(f"character training weight is invalid for {identity_id}") from exc
        if not weight > 0.0:
            raise ValueError(f"character training weight must be positive for {identity_id}")
        weights[identity_id] = weight
    total = sum(weights.values())
    return {identity_id: weight / total for identity_id, weight in weights.items()}


def validate_character_training_strategy(raw_strategy: dict[str, Any]) -> None:
    """Validate the public selected-identity curriculum interface."""
    _identity_weights(raw_strategy)
    _fraction(raw_strategy.get("joint_training_fraction", 0.0), "joint training fraction")
    for identity_id, identity_config in _identity_configs(raw_strategy).items():
        if "solo_fraction" in identity_config:
            _fraction(identity_config["solo_fraction"], f"solo fraction for {identity_id}")
        context_fractions = identity_config.get("context_fractions")
        if context_fractions is not None:
            if not isinstance(context_fractions, dict):
                raise ValueError(f"context fractions for {identity_id} must be an object")
            unknown_modalities = set(context_fractions) - {"image", "video", "audio"}
            if unknown_modalities:
                raise ValueError(
                    f"unknown context modalities for {identity_id}: "
                    + ", ".join(sorted(unknown_modalities))
                )
            for modality, raw_fraction in context_fractions.items():
                _fraction(raw_fraction, f"{modality} solo fraction for {identity_id}")
        raw_source_weights = identity_config.get("source_weights")
        if raw_source_weights is not None:
            if not isinstance(raw_source_weights, dict) or not raw_source_weights:
                raise ValueError(f"source weights for {identity_id} must be a non-empty object")
            total = 0.0
            for source_path, raw_weight in raw_source_weights.items():
                try:
                    weight = float(raw_weight)
                except (TypeError, ValueError) as exc:
                    raise ValueError(f"source weight is invalid for {identity_id}: {source_path}") from exc
                if not math.isfinite(weight) or weight < 0.0:
                    raise ValueError(f"source weight must be non-negative for {identity_id}: {source_path}")
                total += weight
            if total > 1.0 + 1e-6 or ("*" in raw_source_weights and abs(total - 1.0) > 1e-6):
                raise ValueError(
                    f"source weights for {identity_id} must not exceed 1; "
                    "an explicit '*' remainder must make the total exactly 1"
                )


def _identity_configs(raw_strategy: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {
        str(raw_identity["id"]).strip(): raw_identity
        for raw_identity in raw_strategy.get("identities", [])
    }


def _fraction(value: Any, field: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field} must be between 0 and 1") from exc
    if not 0.0 <= result <= 1.0:
        raise ValueError(f"{field} must be between 0 and 1")
    return result


def _normalized_path(value: str) -> str:
    return os.path.normcase(os.path.normpath(str(value)))


def _allocate_source_probabilities(
    *,
    probabilities: list[float],
    candidates: Sequence[CharacterTrainingCandidate],
    indices: list[int],
    total_probability: float,
    raw_source_weights: Any,
    identity_id: str,
) -> None:
    targets = _source_probability_targets(
        candidates=candidates,
        indices=indices,
        total_probability=total_probability,
        raw_source_weights=raw_source_weights,
        identity_id=identity_id,
    )
    source_indices: dict[str, list[int]] = {}
    for index in indices:
        source_key = _normalized_path(candidates[index].dataset_path)
        source_indices.setdefault(source_key, []).append(index)

    for source_key, source_candidate_indices in source_indices.items():
        per_candidate = targets[source_key] / len(source_candidate_indices)
        for index in source_candidate_indices:
            probabilities[index] = per_candidate


def _source_probability_targets(
    *,
    candidates: Sequence[CharacterTrainingCandidate],
    indices: list[int],
    total_probability: float,
    raw_source_weights: Any,
    identity_id: str,
) -> dict[str, float]:
    source_indices: dict[str, list[int]] = {}
    for index in indices:
        source_key = _normalized_path(candidates[index].dataset_path)
        source_indices.setdefault(source_key, []).append(index)
    if not isinstance(raw_source_weights, dict) or not raw_source_weights:
        return {
            source: total_probability * len(source_candidates) / len(indices)
            for source, source_candidates in source_indices.items()
        }

    explicit: dict[str, float] = {}
    remainder_weight = None
    for raw_path, raw_weight in raw_source_weights.items():
        try:
            weight = float(raw_weight)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"source weight is invalid for {identity_id}: {raw_path}") from exc
        if not math.isfinite(weight) or weight < 0.0:
            raise ValueError(f"source weight must be non-negative for {identity_id}: {raw_path}")
        if raw_path == "*":
            remainder_weight = weight
        else:
            explicit[_normalized_path(raw_path)] = weight
    explicit_total = sum(explicit.values())
    if remainder_weight is None:
        remainder_weight = 1.0 - explicit_total
    if abs(explicit_total + remainder_weight - 1.0) > 1e-6:
        raise ValueError(f"source weights for {identity_id} must total 1")

    for source_key, weight in explicit.items():
        if weight > 0.0 and source_key not in source_indices:
            raise ValueError(
                f"{identity_id} has no eligible representation in requested source {source_key}"
            )
    unspecified = [source for source in source_indices if source not in explicit]
    if remainder_weight > 0.0 and not unspecified:
        raise ValueError(f"{identity_id} has no eligible source for the automatic remainder")
    unspecified_count = sum(len(source_indices[source]) for source in unspecified)

    targets = {}
    for source_key, source_candidate_indices in source_indices.items():
        if source_key in explicit:
            targets[source_key] = total_probability * explicit[source_key]
        else:
            targets[source_key] = (
                total_probability
                * remainder_weight
                * len(source_candidate_indices)
                / unspecified_count
            )
    return targets


def _allocate_context_source_probabilities(
    *,
    probabilities: list[float],
    candidates: Sequence[CharacterTrainingCandidate],
    row_targets: dict[tuple[str, str], tuple[list[int], float]],
    total_indices: list[int],
    total_probability: float,
    raw_source_weights: Any,
    identity_id: str,
) -> None:
    """Fit context and source marginals jointly across structural-zero cells."""
    if not isinstance(raw_source_weights, dict) or not raw_source_weights:
        for row_indices, row_probability in row_targets.values():
            per_candidate = row_probability / len(row_indices)
            for index in row_indices:
                probabilities[index] = per_candidate
        return

    source_targets = _source_probability_targets(
        candidates=candidates,
        indices=total_indices,
        total_probability=total_probability,
        raw_source_weights=raw_source_weights,
        identity_id=identity_id,
    )
    cells: dict[tuple[tuple[str, str], str], list[int]] = {}
    for row_key, (row_indices, _row_probability) in row_targets.items():
        for index in row_indices:
            source = _normalized_path(candidates[index].dataset_path)
            cells.setdefault((row_key, source), []).append(index)
    masses = {cell: float(len(indices)) for cell, indices in cells.items()}

    for _iteration in range(500):
        for row_key, (_indices, target) in row_targets.items():
            row_cells = [cell for cell in masses if cell[0] == row_key]
            current = sum(masses[cell] for cell in row_cells)
            if target > 0 and current <= 0:
                raise ValueError(f"{identity_id} cannot satisfy its requested context mix")
            scale = target / current if current else 0.0
            for cell in row_cells:
                masses[cell] *= scale
        for source, target in source_targets.items():
            source_cells = [cell for cell in masses if cell[1] == source]
            current = sum(masses[cell] for cell in source_cells)
            if target > 0 and current <= 0:
                raise ValueError(f"{identity_id} cannot satisfy its requested source mix")
            scale = target / current if current else 0.0
            for cell in source_cells:
                masses[cell] *= scale
        row_error = max(
            abs(sum(mass for (cell_row, _source), mass in masses.items() if cell_row == row) - target)
            for row, (_indices, target) in row_targets.items()
        )
        source_error = max(
            abs(sum(mass for (_row, cell_source), mass in masses.items() if cell_source == source) - target)
            for source, target in source_targets.items()
        )
        if max(row_error, source_error) <= 1e-9:
            break
    else:
        raise ValueError(
            f"{identity_id} cannot simultaneously satisfy the requested solo/group and dataset source percentages"
        )

    for cell, candidate_indices in cells.items():
        per_candidate = masses[cell] / len(candidate_indices)
        for index in candidate_indices:
            probabilities[index] = per_candidate


def build_character_sampling_plan(
    candidates: Sequence[CharacterTrainingCandidate],
    raw_strategy: dict[str, Any],
) -> CharacterSamplingPlan:
    """Return a strict, normalized sampling plan for virtual character views."""

    validate_character_training_strategy(raw_strategy)
    identity_weights = _identity_weights(raw_strategy)
    identity_configs = _identity_configs(raw_strategy)
    joint_fraction = _fraction(
        raw_strategy.get("joint_training_fraction", 0.0),
        "joint training fraction",
    )
    candidates_by_identity: dict[str, list[int]] = {identity_id: [] for identity_id in identity_weights}
    for index, candidate in enumerate(candidates):
        if candidate.identity_id in candidates_by_identity:
            candidates_by_identity[candidate.identity_id].append(index)

    missing = [
        identity_id
        for identity_id, indices in candidates_by_identity.items()
        if not indices
    ]
    if missing:
        raise ValueError(
            "selected character identities have no eligible representation: "
            + ", ".join(missing)
        )

    probabilities = [0.0] * len(candidates)
    for identity_id, identity_probability in identity_weights.items():
        indices = candidates_by_identity[identity_id]
        identity_config = identity_configs[identity_id]
        for view_mode, mode_fraction in (
            ("focus", 1.0 - joint_fraction),
            ("joint", joint_fraction),
        ):
            if mode_fraction == 0.0:
                continue
            mode_indices = [
                index for index in indices if candidates[index].view_mode == view_mode
            ]
            if not mode_indices:
                raise ValueError(
                    f"{identity_id} has no eligible {view_mode} representation, "
                    f"but its requested fraction is {mode_fraction:.3f}"
                )
            mode_probability = identity_probability * mode_fraction
            if view_mode == "joint":
                _allocate_source_probabilities(
                    probabilities=probabilities,
                    candidates=candidates,
                    indices=mode_indices,
                    total_probability=mode_probability,
                    raw_source_weights=identity_config.get("source_weights"),
                    identity_id=identity_id,
                )
                continue

            modality_indices = {
                modality: [
                    index for index in mode_indices
                    if candidates[index].media_type == modality
                ]
                for modality in ("image", "video", "audio")
            }
            context_fractions = identity_config.get("context_fractions", {})
            row_targets: dict[tuple[str, str], tuple[list[int], float]] = {}
            for modality, scoped_indices in modality_indices.items():
                if not scoped_indices:
                    continue
                modality_probability = mode_probability * len(scoped_indices) / len(mode_indices)
                solo_indices = [
                    index for index in scoped_indices
                    if len(candidates[index].identity_ids) == 1
                ]
                group_indices = [
                    index for index in scoped_indices
                    if len(candidates[index].identity_ids) > 1
                ]
                if modality in context_fractions:
                    solo_fraction = _fraction(
                        context_fractions[modality],
                        f"{modality} solo fraction for {identity_id}",
                    )
                elif "solo_fraction" in identity_config:
                    solo_fraction = _fraction(
                        identity_config["solo_fraction"],
                        f"solo fraction for {identity_id}",
                    )
                else:
                    solo_fraction = len(solo_indices) / len(scoped_indices)
                for context, context_indices, context_fraction in (
                    ("solo", solo_indices, solo_fraction),
                    ("group", group_indices, 1.0 - solo_fraction),
                ):
                    if context_fraction == 0.0:
                        continue
                    if not context_indices:
                        raise ValueError(
                            f"{identity_id} has no eligible {modality} {context} representation, "
                            f"but its requested fraction is {context_fraction:.3f}"
                        )
                    row_targets[(modality, context)] = (
                        context_indices,
                        modality_probability * context_fraction,
                    )
            _allocate_context_source_probabilities(
                probabilities=probabilities,
                candidates=candidates,
                row_targets=row_targets,
                total_indices=mode_indices,
                total_probability=mode_probability,
                raw_source_weights=identity_config.get("source_weights"),
                identity_id=identity_id,
            )
    return CharacterSamplingPlan(probabilities=tuple(probabilities))
