"""Immutable future formal-training boundaries derived from the FT0 contract."""

from __future__ import annotations

FUTURE_SEGMENTS = ((5000, 10000), (10000, 15000), (15000, 20000), (20000, 25000), (25000, 30000))


def validate_future_segment(start: int, end: int) -> None:
    if (start, end) not in FUTURE_SEGMENTS:
        raise ValueError("segment is not one pre-registered future FT3-FT7 boundary")
