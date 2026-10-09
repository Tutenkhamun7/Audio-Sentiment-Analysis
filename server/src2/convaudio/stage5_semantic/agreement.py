"""Inter-annotator agreement evaluation using Cohen's kappa."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def calculate_cohens_kappa(
    labels_a: list[str],
    labels_b: list[str],
    categories: list[str] | None = None,
) -> dict[str, Any]:
    """Calculate Cohen's kappa between two lists of aligned annotations."""
    if len(labels_a) != len(labels_b):
        raise ValueError(
            f"Annotation lengths mismatch: {len(labels_a)} vs {len(labels_b)}"
        )

    n = len(labels_a)
    if n == 0:
        return {
            "kappa": 1.0,
            "observed_agreement": 1.0,
            "expected_agreement": 1.0,
            "n_items": 0,
            "confusion_matrix": {},
        }

    if categories is None:
        cats = sorted(list(set(labels_a) | set(labels_b)))
    else:
        cats = sorted(list(set(categories)))

    cat_to_idx = {c: i for i, c in enumerate(cats)}
    k = len(cats)

    # Contingency matrix
    matrix = [[0 for _ in range(k)] for _ in range(k)]
    for la, lb in zip(labels_a, labels_b):
        if la in cat_to_idx and lb in cat_to_idx:
            matrix[cat_to_idx[la]][cat_to_idx[lb]] += 1

    # Observed agreement
    agreed = sum(matrix[i][i] for i in range(k))
    po = agreed / n

    # Expected agreement
    pe = 0.0
    for i in range(k):
        row_total = sum(matrix[i][j] for j in range(k))
        col_total = sum(matrix[j][i] for j in range(k))
        pe += (row_total / n) * (col_total / n)

    if pe >= 1.0:
        kappa = 1.0
    else:
        kappa = (po - pe) / (1.0 - pe)

    conf_dict: dict[str, dict[str, int]] = {}
    for i, ca in enumerate(cats):
        conf_dict[ca] = {}
        for j, cb in enumerate(cats):
            conf_dict[ca][cb] = matrix[i][j]

    return {
        "kappa": float(kappa),
        "observed_agreement": float(po),
        "expected_agreement": float(pe),
        "n_items": n,
        "categories": cats,
        "confusion_matrix": conf_dict,
    }


def compute_file_agreement(
    file_a: Path | str,
    file_b: Path | str,
) -> dict[str, Any]:
    """Load two annotation JSONL files and compute Cohen's kappa on matching turn_ids."""
    def _load_labels(p: Path | str) -> dict[str, str]:
        labels: dict[str, str] = {}
        with open(p, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                obj = json.loads(line)
                tid = obj.get("turn_id")
                lbl = obj.get("label")
                if tid and lbl:
                    labels[str(tid)] = str(lbl).strip().lower()
        return labels

    data_a = _load_labels(file_a)
    data_b = _load_labels(file_b)

    common_keys = sorted(list(set(data_a.keys()) & set(data_b.keys())))
    if not common_keys:
        raise ValueError("No common turn_ids found between annotation files.")

    aligned_a = [data_a[k] for k in common_keys]
    aligned_b = [data_b[k] for k in common_keys]

    res = calculate_cohens_kappa(aligned_a, aligned_b)
    res["common_turn_ids_count"] = len(common_keys)
    return res
