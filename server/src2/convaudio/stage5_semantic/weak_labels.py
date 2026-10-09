"""CRM bootstrap weak-labelling module for Stage 5 semantic classifier.

CRUDE BOOTSTRAP WARNING:
This method is deliberately crude — calm escalations and cheerful unresolved
calls both exist. CRM call-level outcomes do not guarantee turn-level ground truth.
It serves solely as an initial training bootstrap prior to acquiring sufficient
hand-annotated turns. Hand annotations must always override weak labels, and
weak labels must never enter evaluation/test splits.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from convaudio.stage2_timeline.schema import ConversationTimeline


def parse_crm_manifest(crm_path: Path | str) -> dict[str, bool]:
    """Parse CSV or JSON CRM manifest mapping call_id to escalated boolean."""
    p = Path(crm_path)
    if not p.exists():
        raise FileNotFoundError(f"CRM manifest not found: {p}")

    call_escalation: dict[str, bool] = {}

    if p.suffix.lower() == ".json":
        with open(p, "r", encoding="utf-8") as f:
            data = json.load(f)
            if isinstance(data, list):
                for item in data:
                    cid = str(item.get("call_id", ""))
                    esc = bool(item.get("escalated", False))
                    if cid:
                        call_escalation[cid] = esc
            elif isinstance(data, dict):
                for cid, esc in data.items():
                    call_escalation[str(cid)] = bool(esc)
    else:
        # Default CSV
        with open(p, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                cid = row.get("call_id", "").strip()
                esc_str = row.get("escalated", "false").strip().lower()
                is_esc = esc_str in ("true", "1", "yes", "t")
                if cid:
                    call_escalation[cid] = is_esc

    return call_escalation


def generate_weak_labels(
    crm_file: Path | str,
    runs_dir: Path | str,
    out_file: Path | str,
    default_weight: float = 0.3,
    customer_role: str = "CUSTOMER",
    speaker_roles: dict[str, str] | None = None,
) -> int:
    """Generate weak turn labels from call CRM escalation outcomes."""
    call_escalation = parse_crm_manifest(crm_file)
    runs_path = Path(runs_dir)

    timeline_paths: list[Path] = []
    if runs_path.is_file() and runs_path.name.endswith(".json"):
        timeline_paths = [runs_path]
    else:
        timeline_paths = sorted(runs_path.rglob("timeline.json"))

    weak_records: list[dict[str, Any]] = []

    for tl_path in timeline_paths:
        try:
            timeline = ConversationTimeline.from_json(tl_path)
        except Exception:
            continue

        cid = timeline.call_id
        if cid not in call_escalation:
            continue

        is_escalated = call_escalation[cid]
        weak_label = "negative" if is_escalated else "neutral"

        for turn in timeline.turns:
            if turn.text is None or not turn.text.strip():
                continue

            # Determine speaker role
            spk = turn.speaker
            role = spk
            if speaker_roles and spk in speaker_roles:
                role = speaker_roles[spk]
            elif spk in ("SPEAKER_01", "spk_1", "customer", "CUSTOMER"):
                role = "CUSTOMER"

            # Apply weak label primarily to customer turns
            if role.upper() == customer_role.upper():
                weak_records.append({
                    "turn_id": turn.turn_id,
                    "call_id": cid,
                    "timeline": str(tl_path.resolve()),
                    "label": weak_label,
                    "weight": default_weight,
                    "annotator": "weak_crm",
                    "provenance": turn.provenance,
                    "overlap_flag": turn.overlap.ratio > 0.0 or turn.overlap.interrupts is not None,
                })

    out_p = Path(out_file)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    with open(out_p, "w", encoding="utf-8") as f:
        for r in weak_records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    return len(weak_records)
