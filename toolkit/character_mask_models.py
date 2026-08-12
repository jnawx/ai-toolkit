from typing import Literal


DEFAULT_SAM2_TRACKER_MODEL = "facebook/sam2.1-hiera-tiny"
DEFAULT_SAM3_DETECTOR_MODEL = "facebook/sam3"

SAM2_TRACKER_MODELS = (
    {
        "id": "facebook/sam2.1-hiera-tiny",
        "label": "SAM 2.1 Tiny",
        "description": "Fastest; good for review and most clean shots.",
    },
    {
        "id": "facebook/sam2.1-hiera-small",
        "label": "SAM 2.1 Small",
        "description": "Better boundaries with a modest speed cost.",
    },
    {
        "id": "facebook/sam2.1-hiera-base-plus",
        "label": "SAM 2.1 Base+",
        "description": "High-quality tracking for difficult hair and occlusion.",
    },
    {
        "id": "facebook/sam2.1-hiera-large",
        "label": "SAM 2.1 Large",
        "description": "Highest SAM 2.1 quality; slowest and most VRAM intensive.",
    },
)

SAM3_DETECTOR_MODELS = (
    {
        "id": DEFAULT_SAM3_DETECTOR_MODEL,
        "label": "SAM 3 Text",
        "description": "Finds matching instances from a phrase such as person (up to 64 per frame).",
        "gated": True,
    },
)


def get_character_mask_model_catalog() -> dict:
    return {
        "trackers": [dict(model) for model in SAM2_TRACKER_MODELS],
        "detectors": [dict(model) for model in SAM3_DETECTOR_MODELS],
        "defaults": {
            "tracker": DEFAULT_SAM2_TRACKER_MODEL,
            "detector": DEFAULT_SAM3_DETECTOR_MODEL,
            "concept": "person",
        },
    }


def validate_character_mask_model(
    model_id: str,
    *,
    kind: Literal["tracker", "detector"],
) -> str:
    models = SAM2_TRACKER_MODELS if kind == "tracker" else SAM3_DETECTOR_MODELS
    allowed = {model["id"] for model in models}
    if model_id not in allowed:
        raise ValueError(f"Unsupported character mask {kind} model: {model_id}")
    return model_id
