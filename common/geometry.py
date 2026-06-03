"""Small vectorized geometry helpers shared by the demos."""

from __future__ import annotations

import numpy as np


def point_to_segment_distance(
    points: np.ndarray, a: np.ndarray, b: np.ndarray
) -> np.ndarray:
    """Shortest distance from each point to the segment ``a -> b``.

    Args:
        points: ``(N, 2)`` array of points.
        a: ``(2,)`` segment start.
        b: ``(2,)`` segment end.

    Returns:
        ``(N,)`` array of distances.
    """
    points = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    ab = b - a
    denom = float(ab @ ab)
    if denom < 1e-9:
        return np.linalg.norm(points - a, axis=1)
    t = ((points - a) @ ab) / denom
    t = np.clip(t, 0.0, 1.0)
    proj = a + t[:, None] * ab
    return np.linalg.norm(points - proj, axis=1)


def unit(vector: np.ndarray) -> np.ndarray:
    """Return the unit vector; zero vector maps to zero."""
    vector = np.asarray(vector, dtype=np.float64)
    norm = float(np.linalg.norm(vector))
    if norm < 1e-9:
        return np.zeros_like(vector)
    return vector / norm
