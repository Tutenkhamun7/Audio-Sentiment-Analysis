"""Context window assembly for Stage 5 semantic classifier."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from convaudio.stage2_timeline.schema import Turn


def _count_tokens(text: str, tokenizer: Any = None) -> int:
    """Count tokens in text using provided tokenizer or simple token splitting."""
    if tokenizer is None:
        # Default simple estimation: approximate by words
        return len(text.split())
    if callable(tokenizer):
        res = tokenizer(text)
        if isinstance(res, list):
            return len(res)
        if hasattr(res, "input_ids"):
            return len(res.input_ids)
        return len(res)
    if hasattr(tokenizer, "encode"):
        tokens = tokenizer.encode(text)
        return len(tokens)
    return len(text.split())


def build_context_window(
    turns: list[Turn],
    target_idx: int,
    n_context: int = 3,
    speaker_roles: dict[str, str] | None = None,
    tokenizer: Any = None,
    max_tokens: int | None = None,
) -> str:
    """Build a speaker-tagged context window ending with the target turn.

    Format:
        SPEAKER_00: the refund window closed in March
        SPEAKER_01: no, that's not what I was told
        SPEAKER_00: I understand, let me check the notes
        >>> SPEAKER_01: fine.

    Rules:
    - Target turn is prefixed with '>>> ' and is always last.
    - Exactly one '>>> ' marker is present in the window.
    - Turns with empty or null text are skipped, walking further back to fill n_context.
    - Speaker labels use speaker_roles if provided, otherwise raw speaker IDs.
    - If window exceeds max_tokens, turns are truncated from the oldest end.
      The target turn is never truncated.
    """
    if target_idx < 0 or target_idx >= len(turns):
        raise IndexError(f"Target index {target_idx} is out of bounds for {len(turns)} turns")

    target_turn = turns[target_idx]

    def _resolve_spk(spk: str) -> str:
        if speaker_roles:
            return speaker_roles.get(spk, spk)
        return spk

    # Walk backwards from target_idx - 1 to find up to n_context non-empty turns
    context_lines: list[str] = []
    curr = target_idx - 1
    while curr >= 0 and len(context_lines) < n_context:
        t = turns[curr]
        if t.text is not None and t.text.strip():
            spk = _resolve_spk(t.speaker)
            context_lines.append(f"{spk}: {t.text.strip()}")
        curr -= 1

    # Reverse so context is in chronological order
    context_lines.reverse()

    target_spk = _resolve_spk(target_turn.speaker)
    target_text = (target_turn.text or "").strip()
    target_line = f">>> {target_spk}: {target_text}"

    lines = context_lines + [target_line]

    # Truncate oldest turns if max_tokens is specified
    if max_tokens is not None and max_tokens > 0:
        while len(lines) > 1:
            window_candidate = "\n".join(lines)
            if _count_tokens(window_candidate, tokenizer=tokenizer) <= max_tokens:
                break
            # Drop oldest line
            lines.pop(0)

    window = "\n".join(lines)

    # Invariants verification
    marker_count = window.count(">>> ")
    if marker_count != 1:
        raise AssertionError(f"Window must contain exactly one '>>> ' marker, found {marker_count}")

    if not lines[-1].startswith(">>> "):
        raise AssertionError("Target turn must be the final turn in the window")

    return window
