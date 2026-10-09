"""Speaker separation quality assessment via embedding cosine similarity."""

from __future__ import annotations

import numpy as np


def compute_sep_cosine(turn_emb: np.ndarray, enroll_emb: np.ndarray) -> float:
    """Compute cosine similarity between a turn's speaker embedding and enrollment embedding."""
    u = turn_emb.flatten().astype(np.float64)
    v = enroll_emb.flatten().astype(np.float64)

    norm_u = np.linalg.norm(u)
    norm_v = np.linalg.norm(v)

    if norm_u < 1e-8 or norm_v < 1e-8:
        return 0.0

    sim = float(np.dot(u, v) / (norm_u * norm_v))
    return float(np.clip(sim, -1.0, 1.0))
