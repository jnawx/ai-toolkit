import math
from typing import Any


def validate_character_audio_intervals(raw_intervals: Any) -> list[tuple[float, float]]:
    intervals = []
    if not isinstance(raw_intervals, list):
        raise ValueError("character audio intervals must be a JSON list")
    for interval in raw_intervals:
        if not isinstance(interval, list) or len(interval) != 2:
            raise ValueError("each character audio interval must be [start_seconds, end_seconds]")
        start, end = float(interval[0]), float(interval[1])
        if not math.isfinite(start) or not math.isfinite(end) or start < 0.0 or end <= start:
            raise ValueError("character audio intervals must satisfy 0 <= start < end")
        intervals.append((start, end))
    return intervals
