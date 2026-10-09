"""Timeline validation against JSON schema and semantic business invariants."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

import jsonschema

from convaudio.errors import TimelineValidationError
from convaudio.stage2_timeline.schema import ConversationTimeline

DEFAULT_SCHEMA_DIR = Path(__file__).resolve().parent.parent.parent / "schema"


def _load_schema(schema_version: str = "1.0.0", schema_path: Path | str | None = None) -> dict[str, Any]:
    if schema_path:
        path = Path(schema_path)
    else:
        filename = f"timeline-{schema_version}.json"
        path = DEFAULT_SCHEMA_DIR / filename
        if not path.exists():
            alt_paths = [
                Path.cwd() / "schema" / filename,
                Path.cwd() / "src2" / "schema" / filename,
                DEFAULT_SCHEMA_DIR / "timeline-1.0.0.json",
            ]
            for alt in alt_paths:
                if alt.exists():
                    path = alt
                    break
    if not path.exists():
        raise FileNotFoundError(f"Timeline schema file not found at {path}")
    with open(path, "r", encoding="utf-8") as f:
        schema = json.load(f)
    return cast(dict[str, Any], schema)


def validate_timeline(
    timeline_or_path: dict[str, Any] | Path | str | ConversationTimeline,
    schema_path: Path | str | None = None,
) -> None:
    """Validate timeline object or serialized file against schema and domain rules.

    Raises TimelineValidationError on any violation.
    """
    if isinstance(timeline_or_path, ConversationTimeline):
        data = timeline_or_path.to_dict()
    elif isinstance(timeline_or_path, dict):
        data = timeline_or_path
    else:
        p = Path(timeline_or_path)
        if p.exists() and p.is_file():
            text = p.read_text(encoding="utf-8")
        else:
            text = str(timeline_or_path)
        try:
            data = json.loads(text)
        except json.JSONDecodeError as e:
            raise TimelineValidationError(f"Invalid JSON format: {e}") from e

    # 1. Structural schema validation
    version = str(data.get("schema_version", "1.0.0"))
    schema = _load_schema(schema_version=version, schema_path=schema_path)
    try:
        jsonschema.validate(instance=data, schema=schema)
    except jsonschema.ValidationError as e:
        raise TimelineValidationError(f"JSON schema validation failed: {e.message}") from e

    # 2. Semantic business invariants
    # 2a. frame_rate_hz & n_frames are mandatory and non-empty
    frame_rate_hz = data.get("frame_rate_hz")
    if frame_rate_hz is None or not isinstance(frame_rate_hz, (int, float)) or frame_rate_hz <= 0:
        raise TimelineValidationError(
            f"Mandatory field 'frame_rate_hz' is missing, invalid, or <= 0: {frame_rate_hz}"
        )

    n_frames = data.get("n_frames")
    if n_frames is None or not isinstance(n_frames, int) or n_frames < 0:
        raise TimelineValidationError(
            f"Mandatory field 'n_frames' is missing or invalid: {n_frames}"
        )

    # 2b. Turn invariants
    turns = data.get("turns", [])
    for idx, turn in enumerate(turns):
        turn_id = turn.get("turn_id", f"turn_{idx}")
        acoustic = turn.get("acoustic")
        abstain_reason = turn.get("acoustic_abstain_reason")

        # Mutually exclusive: exactly one must be non-null
        if acoustic is not None and abstain_reason is not None:
            raise TimelineValidationError(
                f"Turn '{turn_id}' has both 'acoustic' and 'acoustic_abstain_reason' non-null. "
                "They are mutually exclusive: exactly one must be non-null."
            )
        if acoustic is None and abstain_reason is None:
            raise TimelineValidationError(
                f"Turn '{turn_id}' has both 'acoustic' and 'acoustic_abstain_reason' null. "
                "Exactly one must be non-null."
            )

        # Word timings source rule: never absent when text is non-null
        text = turn.get("text")
        wt_source = turn.get("word_timings_source")
        if text is not None and text.strip() != "":
            if wt_source not in ("forced_align", "asr_fallback"):
                raise TimelineValidationError(
                    f"Turn '{turn_id}' has text but missing or invalid 'word_timings_source': {wt_source}"
                )

        # Stage 5 semantic invariants
        semantic = turn.get("semantic")
        sem_abstain = turn.get("semantic_abstain_reason")
        if version >= "1.1.0":
            if semantic is not None and sem_abstain is not None:
                raise TimelineValidationError(
                    f"Turn '{turn_id}' has both 'semantic' and 'semantic_abstain_reason' non-null. "
                    "They are mutually exclusive: exactly one must be non-null."
                )
            if semantic is None and sem_abstain is None:
                raise TimelineValidationError(
                    f"Turn '{turn_id}' has both 'semantic' and 'semantic_abstain_reason' null. "
                    "Exactly one must be non-null."
                )
        else:
            if semantic is not None and sem_abstain is not None:
                raise TimelineValidationError(
                    f"Turn '{turn_id}' has both 'semantic' and 'semantic_abstain_reason' non-null. "
                    "They are mutually exclusive: exactly one must be non-null."
                )

        if semantic is not None and isinstance(semantic, dict):
            probs = semantic.get("probs")
            label = semantic.get("label")
            if isinstance(probs, dict) and label is not None and label not in probs:
                raise TimelineValidationError(
                    f"Turn '{turn_id}' semantic label '{label}' is not in probs keys: {list(probs.keys())}"
                )
