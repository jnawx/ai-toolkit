import json
import base64
import io
import os
import re
import shutil
import tempfile
import uuid
from contextlib import ExitStack, contextmanager, nullcontext
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable, Optional, Sequence

import numpy as np
from filelock import FileLock
from PIL import Image

from toolkit.character_dop_schema import validate_character_audio_intervals
from toolkit.character_mask_models import validate_character_mask_model


ANNOTATION_DIRECTORY = "_character_dop"
SHARED_IDENTITY_CATALOG = "_character_dop_identities.json"
SHARED_IDENTITY_LOCK = ".character_dop_identities.lock"
VISUAL_MASK_EXTENSIONS = (".png", ".jpg", ".jpeg", ".webp", ".bmp")
MAX_AUTO_MASK_EDGE = 768
MAX_AUTO_MASK_COUNT = 64
MAX_AUTO_MASK_DATA_URL_LENGTH = 2 * 1024 * 1024
MAX_AUTO_MASK_TOTAL_DATA_URL_LENGTH = 24 * 1024 * 1024
CHARACTER_IDENTITY_CATALOG_VERSION = 1
CHARACTER_IDENTITY_ID_PATTERN = re.compile(r"^[a-z0-9][a-z0-9_-]{0,63}$")
MAX_CHARACTER_IDENTITIES = 64
MAX_CHARACTER_IDENTITY_CATALOG_BYTES = 128 * 1024
CHARACTER_IDENTITY_TEXT_LIMITS = {
    "display name": (128, 512),
    "trigger word": (128, 512),
    "class prompt": (256, 1024),
}


def is_character_annotation_artifact(file_path: Path, dataset_dir: Path) -> bool:
    """Return whether a path belongs to the dataset's private annotation tree."""
    try:
        relative_path = Path(file_path).resolve().relative_to(Path(dataset_dir).resolve())
    except ValueError:
        return False
    return ANNOTATION_DIRECTORY in relative_path.parts[:-1]


def _append_suffix(path: Path, suffix: str) -> Path:
    return path.parent / f"{path.name}{suffix}"


def _annotation_storage_path(dataset_dir: Path, *relative_parts: Any) -> Path:
    """Resolve an annotation path and reject symlink/junction escapes."""
    dataset_root = Path(dataset_dir).resolve(strict=True)
    relative_path = Path(*relative_parts)
    if relative_path.is_absolute() or ".." in relative_path.parts:
        raise ValueError("character annotation storage path must be inside the dataset")
    storage_path = (dataset_root / relative_path).resolve(strict=False)
    def containment_form(path: Path) -> Path:
        normalized = os.path.normcase(str(path))
        if normalized.startswith("\\\\?\\UNC\\"):
            normalized = "\\\\" + normalized[8:]
        elif normalized.startswith("\\\\?\\"):
            normalized = normalized[4:]
        return Path(normalized)

    try:
        containment_form(storage_path).relative_to(containment_form(dataset_root))
    except ValueError as exc:
        raise ValueError("character annotation storage path escapes the dataset") from exc
    return storage_path


def _identity_catalog_path(dataset_dir: Path) -> Path:
    return _annotation_storage_path(dataset_dir, ANNOTATION_DIRECTORY, "identities.json")


def _identity_catalog_lock_path(dataset_dir: Path) -> Path:
    return _annotation_storage_path(dataset_dir, ANNOTATION_DIRECTORY, ".identities.lock")


def _validate_identity_id(identity_id: str) -> str:
    identity_id = str(identity_id).strip()
    if not CHARACTER_IDENTITY_ID_PATTERN.fullmatch(identity_id):
        raise ValueError(
            "character identity id must use 1-64 lowercase letters, numbers, hyphens, or underscores"
        )
    return identity_id


def _required_identity_text(value: str, field: str) -> str:
    value = str(value).strip()
    if not value:
        raise ValueError(f"character identity {field} cannot be blank")
    max_chars, max_bytes = CHARACTER_IDENTITY_TEXT_LIMITS[field]
    if len(value) > max_chars or len(value.encode("utf-8")) > max_bytes:
        raise ValueError(
            f"character identity {field} must be at most {max_chars} characters "
            f"and {max_bytes} UTF-8 bytes"
        )
    return value


@lru_cache(maxsize=128)
def _read_character_identity_catalog(
    catalog_path: str,
    modified_ns: int,
    size: int,
) -> Any:
    del modified_ns, size
    return json.loads(Path(catalog_path).read_text(encoding="utf-8"))


def _list_character_identities_at_path(catalog_path: Path) -> list[dict]:
    if not catalog_path.exists():
        return []
    stat = catalog_path.stat()
    if stat.st_size > MAX_CHARACTER_IDENTITY_CATALOG_BYTES:
        raise ValueError(
            f"character identity catalog exceeds {MAX_CHARACTER_IDENTITY_CATALOG_BYTES} bytes"
        )
    payload = _read_character_identity_catalog(
        str(catalog_path),
        stat.st_mtime_ns,
        stat.st_size,
    )
    if not isinstance(payload, dict) or payload.get("version") != CHARACTER_IDENTITY_CATALOG_VERSION:
        raise ValueError("unsupported character identity catalog version")
    raw_identities = payload.get("identities")
    if not isinstance(raw_identities, list):
        raise ValueError("character identity catalog must contain an identities list")
    if len(raw_identities) > MAX_CHARACTER_IDENTITIES:
        raise ValueError(
            f"character identity catalog supports at most {MAX_CHARACTER_IDENTITIES} identities"
        )
    identities = []
    seen_ids = set()
    seen_triggers = set()
    for raw_identity in raw_identities:
        if not isinstance(raw_identity, dict):
            raise ValueError("each character identity must be an object")
        identity = {
            "id": _validate_identity_id(raw_identity.get("id", "")),
            "display_name": _required_identity_text(
                raw_identity.get("display_name", ""), "display name"
            ),
            "trigger_word": _required_identity_text(
                raw_identity.get("trigger_word", ""), "trigger word"
            ),
            "class_prompt": _required_identity_text(
                raw_identity.get("class_prompt", ""), "class prompt"
            ),
        }
        if identity["id"] in seen_ids:
            raise ValueError(f"duplicate character identity id: {identity['id']}")
        trigger_key = identity["trigger_word"].casefold()
        if trigger_key in seen_triggers:
            raise ValueError(
                f"duplicate character identity trigger word: {identity['trigger_word']}"
            )
        if any(
            existing_trigger in trigger_key or trigger_key in existing_trigger
            for existing_trigger in seen_triggers
        ):
            raise ValueError(
                "character identity trigger words cannot contain one another because "
                "DOP replaces one active trigger at a time"
            )
        seen_ids.add(identity["id"])
        seen_triggers.add(trigger_key)
        identities.append(identity)
    return [dict(identity) for identity in identities]


def list_character_identities(dataset_dir: Path) -> list[dict]:
    """Return identities activated for one dataset in display order."""
    dataset_dir = Path(dataset_dir).resolve(strict=True)
    local_identities = _list_character_identities_at_path(_identity_catalog_path(dataset_dir))
    shared_catalog = next(
        (
            _annotation_storage_path(ancestor, SHARED_IDENTITY_CATALOG)
            for ancestor in dataset_dir.parents
            if (ancestor / SHARED_IDENTITY_CATALOG).exists()
        ),
        None,
    )
    if shared_catalog is None:
        return local_identities
    shared_identities = _list_character_identities_at_path(shared_catalog)
    shared_by_id = {identity["id"]: identity for identity in shared_identities}
    return [
        dict(shared_by_id[identity["id"]])
        for identity in local_identities
        if identity["id"] in shared_by_id
    ]


def _list_local_character_identities(dataset_dir: Path) -> list[dict]:
    return _list_character_identities_at_path(_identity_catalog_path(dataset_dir))


def _resolved_datasets_root(datasets_root: Path, dataset_dir: Optional[Path] = None) -> Path:
    root = Path(datasets_root).resolve(strict=True)
    if dataset_dir is not None:
        dataset = Path(dataset_dir).resolve(strict=True)
        try:
            dataset.relative_to(root)
        except ValueError as exc:
            raise ValueError("character annotation dataset must be inside the datasets root") from exc
        if dataset == root:
            raise ValueError("character annotation dataset cannot be the datasets root")
    return root


def _shared_identity_catalog_path(datasets_root: Path) -> Path:
    root = _resolved_datasets_root(datasets_root)
    return _annotation_storage_path(root, SHARED_IDENTITY_CATALOG)


def _shared_identity_lock_path(datasets_root: Path) -> Path:
    root = _resolved_datasets_root(datasets_root)
    return _annotation_storage_path(root, SHARED_IDENTITY_LOCK)


def _identity_conflict(identities: Sequence[dict], identity: dict) -> Optional[str]:
    trigger_key = identity["trigger_word"].casefold()
    for existing in identities:
        if existing["id"] == identity["id"]:
            continue
        existing_key = existing["trigger_word"].casefold()
        if existing_key == trigger_key:
            return f"character identity trigger word is already used by {existing['display_name']}"
        if existing_key in trigger_key or trigger_key in existing_key:
            return (
                "character identity trigger words cannot contain one another because "
                "DOP replaces one active trigger at a time"
            )
    return None


def _validated_identity(
    *, identity_id: str, display_name: str, trigger_word: str, class_prompt: str
) -> dict:
    return {
        "id": _validate_identity_id(identity_id),
        "display_name": _required_identity_text(display_name, "display name"),
        "trigger_word": _required_identity_text(trigger_word, "trigger word"),
        "class_prompt": _required_identity_text(class_prompt, "class prompt"),
    }


def _write_character_identity_catalog(catalog_path: Path, identities: list[dict]) -> None:
    payload = {
        "version": CHARACTER_IDENTITY_CATALOG_VERSION,
        "identities": identities,
    }
    if len(_json_text(payload).encode("utf-8")) > MAX_CHARACTER_IDENTITY_CATALOG_BYTES:
        raise ValueError(
            f"character identity catalog exceeds {MAX_CHARACTER_IDENTITY_CATALOG_BYTES} bytes"
        )
    _write_json(catalog_path, payload)
    _read_character_identity_catalog.cache_clear()


def _save_identity_catalog_entry(
    *,
    catalog_path: Path,
    lock_path: Path,
    identity: dict,
    require_existing: Optional[bool],
) -> dict:
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with FileLock(str(lock_path), timeout=30):
        identities = _list_character_identities_at_path(catalog_path)
        conflict = _identity_conflict(identities, identity)
        if conflict:
            raise ValueError(conflict)
        matching_index = next(
            (index for index, existing in enumerate(identities) if existing["id"] == identity["id"]),
            None,
        )
        if require_existing is True and matching_index is None:
            raise ValueError(f"unknown character identity: {identity['id']}")
        if require_existing is False and matching_index is not None:
            raise ValueError(f"character identity already exists: {identity['id']}")
        if matching_index is None:
            if len(identities) >= MAX_CHARACTER_IDENTITIES:
                raise ValueError(
                    f"character identity catalog supports at most {MAX_CHARACTER_IDENTITIES} identities"
                )
            identities.append(identity)
        else:
            identities[matching_index] = identity
        _write_character_identity_catalog(catalog_path, identities)
    return dict(identity)


def _save_character_identity(
    *,
    dataset_dir: Path,
    identity_id: str,
    display_name: str,
    trigger_word: str,
    class_prompt: str,
    require_existing: Optional[bool],
) -> dict:
    identity = _validated_identity(
        identity_id=identity_id,
        display_name=display_name,
        trigger_word=trigger_word,
        class_prompt=class_prompt,
    )
    return _save_identity_catalog_entry(
        catalog_path=_identity_catalog_path(dataset_dir),
        lock_path=_identity_catalog_lock_path(dataset_dir),
        identity=identity,
        require_existing=require_existing,
    )


def _legacy_identity_catalogs(datasets_root: Path) -> list[list[dict]]:
    """Find direct child dataset catalogs without following links outside the root."""
    root = _resolved_datasets_root(datasets_root)
    catalogs = []
    for entry in root.iterdir():
        if entry.is_symlink() or not entry.is_dir():
            continue
        try:
            entry.resolve(strict=True).relative_to(root)
        except (OSError, ValueError):
            continue
        identities = _list_local_character_identities(entry)
        if identities:
            catalogs.append(identities)
    return catalogs


def list_available_character_identities(datasets_root: Path) -> list[dict]:
    """Return workspace identities, migrating existing dataset catalogs once."""
    catalog_path = _shared_identity_catalog_path(datasets_root)
    lock_path = _shared_identity_lock_path(datasets_root)
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with FileLock(str(lock_path), timeout=30):
        if catalog_path.exists():
            return _list_character_identities_at_path(catalog_path)
        identities = []
        for legacy_identities in _legacy_identity_catalogs(datasets_root):
            for identity in legacy_identities:
                existing = next(
                    (candidate for candidate in identities if candidate["id"] == identity["id"]),
                    None,
                )
                if existing is not None:
                    if existing != identity:
                        raise ValueError(
                            f"conflicting character identity definitions found for {identity['id']}"
                        )
                    continue
                conflict = _identity_conflict(identities, identity)
                if conflict:
                    raise ValueError(conflict)
                if len(identities) >= MAX_CHARACTER_IDENTITIES:
                    raise ValueError(
                        f"character identity catalog supports at most {MAX_CHARACTER_IDENTITIES} identities"
                    )
                identities.append(identity)
        if identities:
            _write_character_identity_catalog(catalog_path, identities)
        return [dict(identity) for identity in identities]


def create_shared_character_identity(
    *,
    datasets_root: Path,
    identity_id: str,
    display_name: str,
    trigger_word: str,
    class_prompt: str,
) -> dict:
    list_available_character_identities(datasets_root)
    return _save_identity_catalog_entry(
        catalog_path=_shared_identity_catalog_path(datasets_root),
        lock_path=_shared_identity_lock_path(datasets_root),
        identity=_validated_identity(
            identity_id=identity_id,
            display_name=display_name,
            trigger_word=trigger_word,
            class_prompt=class_prompt,
        ),
        require_existing=False,
    )


def update_shared_character_identity(
    *,
    datasets_root: Path,
    identity_id: str,
    display_name: str,
    trigger_word: str,
    class_prompt: str,
) -> dict:
    list_available_character_identities(datasets_root)
    return _save_identity_catalog_entry(
        catalog_path=_shared_identity_catalog_path(datasets_root),
        lock_path=_shared_identity_lock_path(datasets_root),
        identity=_validated_identity(
            identity_id=identity_id,
            display_name=display_name,
            trigger_word=trigger_word,
            class_prompt=class_prompt,
        ),
        require_existing=True,
    )


def save_character_identity(
    *,
    dataset_dir: Path,
    identity_id: str,
    display_name: str,
    trigger_word: str,
    class_prompt: str,
) -> dict:
    """Create a new named identity and reject an existing immutable id."""
    return _save_character_identity(
        dataset_dir=dataset_dir,
        identity_id=identity_id,
        display_name=display_name,
        trigger_word=trigger_word,
        class_prompt=class_prompt,
        require_existing=False,
    )


def create_character_identity(
    *,
    dataset_dir: Path,
    identity_id: str,
    display_name: str,
    trigger_word: str,
    class_prompt: str,
) -> dict:
    """Create a new named identity and reject an existing immutable id."""
    return _save_character_identity(
        dataset_dir=dataset_dir,
        identity_id=identity_id,
        display_name=display_name,
        trigger_word=trigger_word,
        class_prompt=class_prompt,
        require_existing=False,
    )


def update_character_identity(
    *,
    dataset_dir: Path,
    identity_id: str,
    display_name: str,
    trigger_word: str,
    class_prompt: str,
) -> dict:
    """Update metadata for an existing identity without recreating stale ids."""
    return _save_character_identity(
        dataset_dir=dataset_dir,
        identity_id=identity_id,
        display_name=display_name,
        trigger_word=trigger_word,
        class_prompt=class_prompt,
        require_existing=True,
    )


def delete_character_identity(*, dataset_dir: Path, identity_id: str) -> dict:
    """Delete one named identity and its owned annotations from a dataset."""
    identity_id = _validate_identity_id(identity_id)
    catalog_path = _identity_catalog_path(dataset_dir)
    lock_path = _identity_catalog_lock_path(dataset_dir)
    identity_root = _annotation_storage_path(
        dataset_dir,
        ANNOTATION_DIRECTORY,
        "identities",
        identity_id,
    )
    trash_root = _annotation_storage_path(
        dataset_dir,
        ANNOTATION_DIRECTORY,
        ".deleted-identities",
    )
    staged_identity_root = None
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with FileLock(str(lock_path), timeout=30):
        identities = _list_local_character_identities(dataset_dir)
        deleted_identity = next(
            (identity for identity in identities if identity["id"] == identity_id),
            None,
        )
        if deleted_identity is None:
            raise ValueError(f"unknown character identity: {identity_id}")
        remaining_identities = [
            identity for identity in identities if identity["id"] != identity_id
        ]
        if identity_root.exists():
            trash_root.mkdir(parents=True, exist_ok=True)
            staged_identity_root = _annotation_storage_path(
                dataset_dir,
                ANNOTATION_DIRECTORY,
                ".deleted-identities",
                f"{identity_id}-{uuid.uuid4().hex}",
            )
            identity_root.replace(staged_identity_root)
        try:
            _write_character_identity_catalog(catalog_path, remaining_identities)
        except Exception:
            if staged_identity_root is not None and staged_identity_root.exists():
                staged_identity_root.replace(identity_root)
            raise
    cleanup_pending = False
    if staged_identity_root is not None:
        try:
            if staged_identity_root.is_dir():
                shutil.rmtree(staged_identity_root)
            else:
                staged_identity_root.unlink()
        except OSError:
            cleanup_pending = True
        try:
            trash_root.rmdir()
        except OSError:
            pass
    return {
        "deleted_identity": dict(deleted_identity),
        "identities": [dict(identity) for identity in remaining_identities],
        "cleanup_pending": cleanup_pending,
    }


def delete_shared_character_identity(*, datasets_root: Path, identity_id: str) -> dict:
    """Delete one shared definition and its annotations from every direct child dataset."""
    identity_id = _validate_identity_id(identity_id)
    list_available_character_identities(datasets_root)
    root = _resolved_datasets_root(datasets_root)
    catalog_path = _shared_identity_catalog_path(root)
    lock_path = _shared_identity_lock_path(root)
    cleanup_pending = False
    cleanup_errors = []
    with FileLock(str(lock_path), timeout=30):
        identities = _list_character_identities_at_path(catalog_path)
        deleted_identity = next(
            (identity for identity in identities if identity["id"] == identity_id),
            None,
        )
        if deleted_identity is None:
            raise ValueError(f"unknown character identity: {identity_id}")
        remaining_identities = [
            identity for identity in identities if identity["id"] != identity_id
        ]
        _write_character_identity_catalog(catalog_path, remaining_identities)
        for entry in root.iterdir():
            if entry.is_symlink() or not entry.is_dir():
                continue
            try:
                entry.resolve(strict=True).relative_to(root)
                local_ids = {
                    identity["id"] for identity in _list_local_character_identities(entry)
                }
                if identity_id not in local_ids:
                    continue
                result = delete_character_identity(
                    dataset_dir=entry,
                    identity_id=identity_id,
                )
                cleanup_pending = cleanup_pending or result["cleanup_pending"]
            except Exception as exc:
                cleanup_pending = True
                cleanup_errors.append(f"{entry.name}: {exc}")
    return {
        "deleted_identity": dict(deleted_identity),
        "identities": [dict(identity) for identity in remaining_identities],
        "cleanup_pending": cleanup_pending,
        "cleanup_errors": cleanup_errors,
    }


def _require_character_identity(
    dataset_dir: Path,
    identity_id: Optional[str],
    datasets_root: Optional[Path] = None,
    *,
    shared_catalog_locked: bool = False,
) -> Optional[dict]:
    if identity_id is None:
        return None
    identity_id = _validate_identity_id(identity_id)
    if datasets_root is None:
        identities = list_character_identities(dataset_dir)
    elif shared_catalog_locked:
        identities = _list_character_identities_at_path(
            _shared_identity_catalog_path(datasets_root)
        )
    else:
        _resolved_datasets_root(datasets_root, dataset_dir)
        identities = list_available_character_identities(datasets_root)
    identity = next(
        (
            candidate
            for candidate in identities
            if candidate["id"] == identity_id
        ),
        None,
    )
    if identity is None:
        raise ValueError(f"unknown character identity: {identity_id}")
    return identity


def _activate_character_identity_unlocked(dataset_dir: Path, identity: Optional[dict]) -> None:
    if identity is None:
        return
    identities = _list_local_character_identities(dataset_dir)
    matching_index = next(
        (index for index, existing in enumerate(identities) if existing["id"] == identity["id"]),
        None,
    )
    conflict = _identity_conflict(identities, identity)
    if conflict:
        raise ValueError(conflict)
    if matching_index is None:
        if len(identities) >= MAX_CHARACTER_IDENTITIES:
            raise ValueError(
                f"character identity catalog supports at most {MAX_CHARACTER_IDENTITIES} identities"
            )
        identities.append(dict(identity))
    else:
        identities[matching_index] = dict(identity)
    _write_character_identity_catalog(_identity_catalog_path(dataset_dir), identities)


@contextmanager
def _character_identity_annotation_lock(
    dataset_dir: Path,
    identity_id: Optional[str],
    datasets_root: Optional[Path] = None,
):
    """Serialize named-identity writes with catalog updates and deletion."""
    if identity_id is None:
        with nullcontext():
            yield
        return
    if datasets_root is not None:
        _resolved_datasets_root(datasets_root, dataset_dir)
        list_available_character_identities(datasets_root)
    lock_path = _identity_catalog_lock_path(dataset_dir)
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with ExitStack() as stack:
        if datasets_root is not None:
            stack.enter_context(FileLock(str(_shared_identity_lock_path(datasets_root)), timeout=30))
        stack.enter_context(FileLock(str(lock_path), timeout=30))
        yield


@dataclass(frozen=True)
class CharacterAnnotationPaths:
    root: Path
    visual: Path
    audio: Path
    prompts: Path


@dataclass(frozen=True)
class CharacterIdentityView:
    """One named identity's training annotations for a physical media item."""

    identity_id: str
    display_name: str
    trigger_word: str
    class_prompt: str
    visual_path: Optional[Path]
    audio_intervals: Optional[list[tuple[float, float]]]


def _resolved_media(dataset_dir: Path, media_path: Path) -> tuple[Path, Path]:
    dataset_dir = Path(dataset_dir).resolve(strict=True)
    media_path = Path(media_path).resolve(strict=True)
    try:
        relative_media = media_path.relative_to(dataset_dir)
    except ValueError as exc:
        raise ValueError("character annotation media must be inside the dataset") from exc
    if not media_path.is_file():
        raise FileNotFoundError(f"character annotation media not found: {media_path}")
    return dataset_dir, relative_media


def get_character_annotation_paths(
    *,
    dataset_dir: Path,
    media_path: Path,
    identity_id: Optional[str] = None,
) -> CharacterAnnotationPaths:
    """Resolve one item's built-in annotation files without trusting client paths."""
    dataset_dir, relative_media = _resolved_media(dataset_dir, media_path)
    relative_stem = relative_media.with_suffix("")
    root_parts = [ANNOTATION_DIRECTORY]
    if identity_id is not None:
        root_parts.extend(("identities", _validate_identity_id(identity_id)))
    root = _annotation_storage_path(dataset_dir, ANNOTATION_DIRECTORY)
    visual_suffix = ".png" if relative_media.suffix.lower() in VISUAL_MASK_EXTENSIONS else ".npy"
    return CharacterAnnotationPaths(
        root=root,
        visual=_annotation_storage_path(
            dataset_dir, *root_parts, "visual", _append_suffix(relative_stem, visual_suffix)
        ),
        audio=_annotation_storage_path(
            dataset_dir, *root_parts, "audio", _append_suffix(relative_stem, ".json")
        ),
        prompts=_annotation_storage_path(
            dataset_dir, *root_parts, "prompts", _append_suffix(relative_stem, ".json")
        ),
    )


def get_character_identity_views(
    *,
    dataset_dir: Path,
    media_path: Path,
) -> list[CharacterIdentityView]:
    """Return annotated named identities as independent views of one media item."""
    views = []
    for identity in list_character_identities(dataset_dir):
        paths = get_character_annotation_paths(
            dataset_dir=dataset_dir,
            media_path=media_path,
            identity_id=identity["id"],
        )
        audio_intervals = None
        if paths.audio.exists():
            payload = json.loads(paths.audio.read_text(encoding="utf-8"))
            if isinstance(payload, dict) and "character_intervals" not in payload:
                raise ValueError(
                    f"Character DOP audio sidecar must contain 'character_intervals': {paths.audio}"
                )
            audio_intervals = validate_character_audio_intervals(
                payload.get("character_intervals", []) if isinstance(payload, dict) else payload
            )
        visual_path = paths.visual if paths.visual.exists() else None
        if visual_path is None and audio_intervals is None:
            continue
        views.append(
            CharacterIdentityView(
                identity_id=identity["id"],
                display_name=identity["display_name"],
                trigger_word=identity["trigger_word"],
                class_prompt=identity["class_prompt"],
                visual_path=visual_path,
                audio_intervals=audio_intervals,
            )
        )
    return views


def find_matching_character_visual_mask(
    *,
    media_path: Path,
    mask_root: Path,
    dataset_dir: Path,
    is_video: bool,
) -> Optional[Path]:
    """Find a nested UI annotation, with legacy flat mask folders as fallback."""
    media_path = Path(media_path).resolve()
    dataset_dir = Path(dataset_dir).resolve()
    mask_root = Path(mask_root)
    extensions = VISUAL_MASK_EXTENSIONS + ((".npy",) if is_video else ())
    candidate_stems = []
    try:
        candidate_stems.append(mask_root / media_path.relative_to(dataset_dir).with_suffix(""))
    except ValueError:
        pass
    candidate_stems.append(mask_root / media_path.stem)
    for candidate_stem in candidate_stems:
        for extension in extensions:
            candidate = _append_suffix(candidate_stem, extension)
            if candidate.is_file():
                return candidate
    return None


def _json_text(payload: Any) -> str:
    return json.dumps(payload, indent=2) + "\n"


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary_path = Path(handle.name)
            handle.write(_json_text(payload))
        temporary_path.replace(path)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def _mask_data_url(mask: np.ndarray) -> str:
    frame = (np.asarray(mask) > 0).astype(np.uint8) * 255
    if frame.ndim != 2:
        raise ValueError("character instance masks must have shape HW")
    buffer = io.BytesIO()
    Image.fromarray(frame, mode="L").save(buffer, format="PNG", optimize=True)
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def _decode_mask_data_url(data_url: str) -> np.ndarray:
    prefix = "data:image/png;base64,"
    if not isinstance(data_url, str) or not data_url.startswith(prefix):
        raise ValueError("character auto-mask must be a base64 PNG data URL")
    encoded = data_url[len(prefix):]
    if len(encoded) > MAX_AUTO_MASK_DATA_URL_LENGTH:
        raise ValueError("character auto-mask payload is too large")
    try:
        raw = base64.b64decode(encoded, validate=True)
        with Image.open(io.BytesIO(raw)) as image:
            if max(image.size) > MAX_AUTO_MASK_EDGE:
                raise ValueError("character auto-mask dimensions are too large")
            mask = (np.asarray(image.convert("L")) > 0).astype(np.uint8)
    except Exception as exc:
        raise ValueError("character auto-mask PNG could not be decoded") from exc
    return mask


def _combine_mask_data_urls(data_urls: Sequence[str]) -> np.ndarray:
    if not data_urls:
        raise ValueError("select at least one detected character mask")
    if len(data_urls) > MAX_AUTO_MASK_COUNT:
        raise ValueError(f"select no more than {MAX_AUTO_MASK_COUNT} character masks")
    if sum(len(data_url) for data_url in data_urls) > MAX_AUTO_MASK_TOTAL_DATA_URL_LENGTH:
        raise ValueError("selected character mask payload is too large")
    masks = [_decode_mask_data_url(data_url) for data_url in data_urls]
    if any(mask.shape != masks[0].shape for mask in masks[1:]):
        raise ValueError("selected character masks have different dimensions")
    return np.maximum.reduce(masks)


def detect_character_instances(
    *,
    dataset_dir: Path,
    media_path: Path,
    time_seconds: float,
    concept: str,
    model_id: str,
    detector: Callable[[Path, float, str, str, Callable[[str], None]], list[dict]],
    progress: Callable[[str], None] = lambda _message: None,
) -> dict:
    """Detect selectable instances of a text concept on one source frame."""
    _resolved_media(dataset_dir, media_path)
    media_path = Path(media_path).resolve(strict=True)
    model_id = validate_character_mask_model(model_id, kind="detector")
    concept = str(concept).strip()
    if not concept:
        raise ValueError("character auto-mask concept cannot be blank")
    time_seconds = float(time_seconds)
    if not np.isfinite(time_seconds) or time_seconds < 0.0:
        raise ValueError("character auto-mask time must be non-negative")

    detections = detector(media_path, time_seconds, concept, model_id, progress)
    candidates = []
    expected_shape = None
    for index, detection in enumerate(detections[:MAX_AUTO_MASK_COUNT], start=1):
        mask = (np.asarray(detection["mask"]) > 0).astype(np.uint8)
        if mask.ndim != 2 or min(mask.shape) < 1:
            raise ValueError("character detector returned a mask without shape HW")
        source_height, source_width = mask.shape
        scale = min(1.0, MAX_AUTO_MASK_EDGE / float(max(mask.shape)))
        if scale < 1.0:
            target_size = (
                max(1, round(source_width * scale)),
                max(1, round(source_height * scale)),
            )
            mask = (
                np.asarray(
                    Image.fromarray(mask * 255).resize(
                        target_size,
                        Image.Resampling.NEAREST,
                    )
                )
                > 0
            ).astype(np.uint8)
        if expected_shape is None:
            expected_shape = mask.shape
        elif mask.shape != expected_shape:
            raise ValueError("character detector returned masks with different dimensions")
        box = [round(float(value) * scale, 4) for value in detection.get("box", [])]
        if len(box) != 4 or not all(np.isfinite(value) for value in box):
            raise ValueError("character detector boxes must be xyxy")
        score = float(detection.get("score", 0.0))
        if not np.isfinite(score):
            raise ValueError("character detector scores must be finite")
        candidates.append(
            {
                "id": index,
                "score": score,
                "box": box,
                "area": int(mask.sum()),
                "mask_data_url": _mask_data_url(mask),
            }
        )
    return {
        "concept": concept,
        "model_id": model_id,
        "time_seconds": time_seconds,
        "width": int(expected_shape[1]) if expected_shape else 0,
        "height": int(expected_shape[0]) if expected_shape else 0,
        "candidates": candidates,
    }


def invalidate_character_annotation_latents(media_path: Path) -> int:
    """Remove only this media item's derived latent caches after annotations change."""
    media_path = Path(media_path).resolve()
    cache_dir = media_path.parent / "_latent_cache"
    removed = 0
    if cache_dir.is_dir():
        for cache_path in cache_dir.glob(f"{media_path.stem}_*.safetensors"):
            if cache_path.is_file():
                cache_path.unlink()
                removed += 1
    return removed


def save_character_audio_intervals(
    *,
    dataset_dir: Path,
    media_path: Path,
    intervals: Sequence[Sequence[float]],
    identity_id: Optional[str] = None,
    datasets_root: Optional[Path] = None,
) -> Path:
    validated = validate_character_audio_intervals(list(intervals))
    with _character_identity_annotation_lock(dataset_dir, identity_id, datasets_root):
        identity = _require_character_identity(
            dataset_dir,
            identity_id,
            datasets_root,
            shared_catalog_locked=datasets_root is not None,
        )
        _activate_character_identity_unlocked(dataset_dir, identity)
        paths = get_character_annotation_paths(
            dataset_dir=dataset_dir,
            media_path=media_path,
            identity_id=identity_id,
        )
        _write_json(
            paths.audio,
            {"character_intervals": [[start, end] for start, end in validated]},
        )
        invalidate_character_annotation_latents(media_path)
    return paths.audio


def _write_character_visual_mask_file(paths: CharacterAnnotationPaths, binary_mask: np.ndarray) -> None:
    paths.visual.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = paths.visual.with_suffix(paths.visual.suffix + ".tmp")
    if paths.visual.suffix == ".png":
        Image.fromarray(binary_mask[0] * 255, mode="L").save(temporary_path, format="PNG", optimize=True)
    else:
        with temporary_path.open("wb") as handle:
            np.save(handle, binary_mask, allow_pickle=False)
    temporary_path.replace(paths.visual)


def save_character_visual_mask(
    *,
    dataset_dir: Path,
    media_path: Path,
    mask: np.ndarray,
    identity_id: Optional[str] = None,
    datasets_root: Optional[Path] = None,
) -> Path:
    mask = np.asarray(mask)
    if mask.ndim == 2:
        mask = mask[None]
    if mask.ndim != 3 or min(mask.shape) < 1:
        raise ValueError("character visual mask must have shape HW or THW")
    binary_mask = (mask > 0).astype(np.uint8)
    with _character_identity_annotation_lock(dataset_dir, identity_id, datasets_root):
        identity = _require_character_identity(
            dataset_dir,
            identity_id,
            datasets_root,
            shared_catalog_locked=datasets_root is not None,
        )
        _activate_character_identity_unlocked(dataset_dir, identity)
        paths = get_character_annotation_paths(
            dataset_dir=dataset_dir,
            media_path=media_path,
            identity_id=identity_id,
        )
        _write_character_visual_mask_file(paths, binary_mask)
        invalidate_character_annotation_latents(media_path)
    return paths.visual


def validate_character_visual_prompts(raw_prompts: Any) -> list[dict]:
    if not isinstance(raw_prompts, list) or not raw_prompts:
        raise ValueError("character visual tracking requires at least one prompted frame")
    prompts = []
    for raw_prompt in raw_prompts:
        if not isinstance(raw_prompt, dict):
            raise ValueError("each character visual prompt must be an object")
        time_seconds = float(raw_prompt.get("time_seconds", -1.0))
        raw_points = raw_prompt.get("points")
        if time_seconds < 0.0 or not isinstance(raw_points, list) or not raw_points:
            raise ValueError("each prompted frame needs a non-negative time and at least one point")
        points = []
        for raw_point in raw_points:
            if not isinstance(raw_point, dict):
                raise ValueError("each character visual point must be an object")
            x = float(raw_point.get("x", -1.0))
            y = float(raw_point.get("y", -1.0))
            label = int(raw_point.get("label", -1))
            if not 0.0 <= x <= 1.0 or not 0.0 <= y <= 1.0 or label not in (0, 1):
                raise ValueError("character visual points require normalized x/y and label 0 or 1")
            points.append({"x": x, "y": y, "label": label})
        prompts.append({"time_seconds": time_seconds, "points": points})
    return prompts


def track_character_visual_mask(
    *,
    dataset_dir: Path,
    media_path: Path,
    identity_id: Optional[str] = None,
    datasets_root: Optional[Path] = None,
    prompts: Any,
    tracker: Callable[
        [Path, list[dict], Optional[np.ndarray], Optional[float], Callable[[str], None]],
        np.ndarray,
    ],
    initial_mask_data_urls: Sequence[str] = (),
    initial_time_seconds: Optional[float] = None,
    progress: Callable[[str], None] = lambda _message: None,
) -> dict:
    _require_character_identity(dataset_dir, identity_id, datasets_root)
    paths = get_character_annotation_paths(
        dataset_dir=dataset_dir,
        media_path=media_path,
        identity_id=identity_id,
    )
    initial_mask = (
        _combine_mask_data_urls(initial_mask_data_urls)
        if initial_mask_data_urls
        else None
    )
    if prompts:
        validated_prompts = validate_character_visual_prompts(prompts)
    elif initial_mask is not None:
        validated_prompts = []
    else:
        raise ValueError("character visual tracking requires points or an auto-mask")
    if initial_mask is not None:
        if initial_time_seconds is None:
            raise ValueError("character auto-mask tracking requires its source time")
        initial_time_seconds = float(initial_time_seconds)
        if not np.isfinite(initial_time_seconds) or initial_time_seconds < 0.0:
            raise ValueError("character auto-mask time must be non-negative")
    mask = tracker(
        Path(media_path).resolve(),
        validated_prompts,
        initial_mask,
        initial_time_seconds,
        progress,
    )
    mask = np.asarray(mask)
    if mask.ndim == 2:
        mask = mask[None]
    if mask.ndim != 3 or min(mask.shape) < 1:
        raise ValueError("character visual mask must have shape HW or THW")
    binary_mask = (mask > 0).astype(np.uint8)
    with _character_identity_annotation_lock(dataset_dir, identity_id, datasets_root):
        identity = _require_character_identity(
            dataset_dir,
            identity_id,
            datasets_root,
            shared_catalog_locked=datasets_root is not None,
        )
        _activate_character_identity_unlocked(dataset_dir, identity)
        paths = get_character_annotation_paths(
            dataset_dir=dataset_dir,
            media_path=media_path,
            identity_id=identity_id,
        )
        _write_character_visual_mask_file(paths, binary_mask)
        _write_json(paths.prompts, {"prompts": validated_prompts})
        invalidate_character_annotation_latents(media_path)
    return get_character_annotation_state(
        dataset_dir=dataset_dir,
        media_path=media_path,
        identity_id=identity_id,
        datasets_root=datasets_root,
    )


def get_character_mask_preview(
    *,
    dataset_dir: Path,
    media_path: Path,
    frame_index: int,
    identity_id: Optional[str] = None,
    datasets_root: Optional[Path] = None,
) -> dict:
    _require_character_identity(dataset_dir, identity_id, datasets_root)
    paths = get_character_annotation_paths(
        dataset_dir=dataset_dir,
        media_path=media_path,
        identity_id=identity_id,
    )
    if not paths.visual.exists():
        raise FileNotFoundError("character visual mask has not been generated")
    if paths.visual.suffix == ".png":
        with Image.open(paths.visual) as image:
            mask = (np.asarray(image.convert("L")) > 0).astype(np.uint8)[None]
    else:
        mask = np.load(paths.visual, mmap_mode="r", allow_pickle=False)
        if mask.ndim != 3:
            raise ValueError("saved character visual mask must have shape THW")
    frame_index = max(0, min(int(frame_index), mask.shape[0] - 1))
    frame = (np.asarray(mask[frame_index]) > 0).astype(np.uint8) * 255
    buffer = io.BytesIO()
    Image.fromarray(frame, mode="L").save(buffer, format="PNG", optimize=True)
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return {
        "data_url": f"data:image/png;base64,{encoded}",
        "frame_index": frame_index,
        "frame_count": int(mask.shape[0]),
        "width": int(mask.shape[2]),
        "height": int(mask.shape[1]),
    }


def get_character_annotation_state(
    *,
    dataset_dir: Path,
    media_path: Path,
    identity_id: Optional[str] = None,
    datasets_root: Optional[Path] = None,
) -> dict:
    identity = _require_character_identity(dataset_dir, identity_id, datasets_root)
    paths = get_character_annotation_paths(
        dataset_dir=dataset_dir,
        media_path=media_path,
        identity_id=identity_id,
    )
    intervals = []
    if paths.audio.exists():
        payload = json.loads(paths.audio.read_text(encoding="utf-8"))
        raw_intervals = payload.get("character_intervals", [])
        intervals = [list(interval) for interval in validate_character_audio_intervals(raw_intervals)]
    prompts = []
    if paths.prompts.exists():
        payload = json.loads(paths.prompts.read_text(encoding="utf-8"))
        prompts = payload.get("prompts", [])
    visual_shape = None
    if paths.visual.exists():
        if paths.visual.suffix == ".png":
            with Image.open(paths.visual) as image:
                visual_shape = [1, image.height, image.width]
        else:
            visual_shape = list(np.load(paths.visual, mmap_mode="r", allow_pickle=False).shape)
    return {
        "root": str(paths.root),
        "identity": identity,
        "identities": (
            list_available_character_identities(datasets_root)
            if datasets_root is not None
            else list_character_identities(dataset_dir)
        ),
        "visual": {
            "exists": paths.visual.exists(),
            "path": str(paths.visual),
            "shape": visual_shape,
        },
        "audio": {
            "exists": paths.audio.exists(),
            "path": str(paths.audio),
            "intervals": intervals,
        },
        "prompts": prompts,
    }
