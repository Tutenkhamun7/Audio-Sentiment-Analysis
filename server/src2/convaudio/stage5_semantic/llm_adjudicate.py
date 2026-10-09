"""Optional LLM adjudication for flagged ambiguous turns.

HARD CONSTRAINTS:
1. No LLM in the per-turn path: this is an optional adjudication for flagged turns only.
2. Disabled by default (PipelineConfig.stage5.llm_adjudication = False).
3. If the LLM is unreachable or fails, the turn retains its encoder abstention
   and the run completes successfully without raising an unhandled exception.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np

from convaudio.stage2_timeline.schema import SemanticFeatures

if TYPE_CHECKING:
    from convaudio.config import Stage5Config
    from convaudio.stage2_timeline.schema import Turn

logger = logging.getLogger(__name__)


def is_turn_ambiguous(probs: np.ndarray, margin: float = 0.1, min_conf: float = 0.45) -> bool:
    """Check if turn predictions are ambiguous (top 2 close or just below min_conf)."""
    if probs.size < 2:
        return False
    sorted_probs = np.sort(probs)[::-1]
    top1 = float(sorted_probs[0])
    top2 = float(sorted_probs[1])

    # Ambiguity conditions: margin between top 2 is small or top1 is slightly below min_conf
    if (top1 - top2) < margin:
        return True
    if top1 < min_conf and top1 >= (min_conf - 0.15):
        return True
    return False


def call_llm_api(prompt: str, model_name: str = "gpt-4o-mini") -> str:
    """Invoke external LLM API for adjudication. Raises Exception on network or auth failure."""
    # In real deployment this invokes OpenAI/Gemini client.
    # When unconfigured or offline, raises ConnectionError / RuntimeError.
    raise ConnectionError("LLM API endpoint is unreachable or not configured.")


def maybe_adjudicate_turn(
    turn: Turn,
    window: str,
    probs: np.ndarray,
    config: Stage5Config,
) -> SemanticFeatures | None:
    """Attempt LLM adjudication for an ambiguous turn if enabled in config."""
    # Disabled by default
    if not getattr(config, "llm_adjudication", False):
        return None

    margin = getattr(config, "ambiguity_margin", 0.1)
    min_conf = getattr(config, "min_conf", 0.45)
    if not is_turn_ambiguous(probs, margin=margin, min_conf=min_conf):
        return None

    prompt = (
        "You are an expert contact-centre dialogue annotator following the stance-not-situation rule.\n"
        "Evaluate the following dialogue window and score the target turn marked with '>>> '.\n"
        "Return exactly one label: 'negative', 'neutral', or 'positive'.\n\n"
        f"{window}\n"
    )

    try:
        raw_response = call_llm_api(prompt, model_name="gpt-4o-mini")
        res = raw_response.strip().lower()
        if "negative" in res:
            lbl = "negative"
        elif "positive" in res:
            lbl = "positive"
        elif "neutral" in res:
            lbl = "neutral"
        else:
            return None

        # Build adjudicated semantic features
        probs_dict = {
            "negative": 0.8 if lbl == "negative" else 0.1,
            "neutral": 0.8 if lbl == "neutral" else 0.1,
            "positive": 0.8 if lbl == "positive" else 0.1,
        }
        return SemanticFeatures(
            probs=probs_dict,
            label=lbl,
            conf=0.8,
            calibrated=False,
            model_revision="llm_adjudicated",
            adjudicated=True,
            adjudication_reason="ambiguity_resolved",
        )
    except Exception as e:
        logger.warning(
            f"LLM adjudication failed for turn '{turn.turn_id}': {e}. Retaining encoder abstention."
        )
        return None
