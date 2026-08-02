"""Pure-Python order comparison labels shared by validation and test."""

from __future__ import annotations


WINNER_TOLERANCE = 1.0e-8


def winner(delta: float, *, tolerance: float = WINNER_TOLERANCE) -> str:
    if delta > tolerance:
        return "color_then_scatter"
    if delta < -tolerance:
        return "scatter_then_color"
    return "tie"
