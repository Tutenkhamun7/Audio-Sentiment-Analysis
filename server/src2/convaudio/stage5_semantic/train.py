"""Training pipeline for Stage 5 semantic classifier."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, f1_score

from convaudio.errors import ConvaudioError, LicenceViolation
from convaudio.licences import is_licence_denied, load_licence_policy
from convaudio.stage2_timeline.schema import ConversationTimeline
from convaudio.stage5_semantic.features import DYNAMICS_FEATURE_NAMES, extract_turn_dynamics
from convaudio.stage5_semantic.gates import scale_temperature
from convaudio.stage5_semantic.scaler import DynamicsScaler
from convaudio.stage5_semantic.window import build_context_window

LABELS: tuple[str, ...] = ("negative", "neutral", "positive")


def assert_corpus_licence_allowed(
    manifest_data: dict[str, Any] | None = None,
    policy: dict[str, Any] | None = None,
) -> None:
    """Verify that dataset corpus licence conforms to licence policy."""
    if manifest_data is None:
        return

    corpus_lic = str(manifest_data.get("corpus_licence", "")).strip()
    if not corpus_lic:
        return

    if policy is None:
        policy = load_licence_policy()

    denied = policy.get("denied_patterns", [])
    if is_licence_denied(corpus_lic, denied):
        raise LicenceViolation(
            f"Corpus licence '{corpus_lic}' violates project licence policy (non-commercial or restricted)."
        )


def load_dataset_records(
    labels_file: Path | str,
    weak_labels_file: Path | str | None = None,
    manifest_file: Path | str | None = None,
) -> list[dict[str, Any]]:
    """Load labels, ensuring hand labels override weak labels for matching turns."""
    if manifest_file and Path(manifest_file).exists():
        with open(manifest_file, "r", encoding="utf-8") as f:
            manifest = json.load(f)
        assert_corpus_licence_allowed(manifest)

    # 1. Load weak labels if available
    records_by_turn: dict[str, dict[str, Any]] = {}
    if weak_labels_file and Path(weak_labels_file).exists():
        with open(weak_labels_file, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                rec = json.loads(line)
                rec["is_weak"] = True
                rec["weight"] = float(rec.get("weight", 0.3))
                tid = str(rec["turn_id"])
                records_by_turn[tid] = rec

    # 2. Load hand labels — hand label always overrides weak label
    with open(labels_file, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            rec["is_weak"] = False
            rec["weight"] = 1.0
            tid = str(rec["turn_id"])
            records_by_turn[tid] = rec

    return list(records_by_turn.values())


def split_records_by_call_id(
    records: list[dict[str, Any]],
    test_ratio: float = 0.2,
    seed: int = 42,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Split dataset by call_id so turns from the same call never span train and test.

    Weak labels NEVER enter the test split.
    """
    import random
    rng = random.Random(seed)

    # Group records by call_id
    call_records: dict[str, list[dict[str, Any]]] = {}
    for r in records:
        cid = r.get("call_id")
        if not cid:
            # Derive call_id from timeline path if omitted
            cid = Path(r["timeline"]).parent.name
            r["call_id"] = cid
        call_records.setdefault(cid, []).append(r)

    # Identify calls with hand labels vs calls with only weak labels
    hand_calls: list[str] = []
    weak_only_calls: list[str] = []

    for cid, turns in call_records.items():
        if any(not t.get("is_weak", False) for t in turns):
            hand_calls.append(cid)
        else:
            weak_only_calls.append(cid)

    # Shuffle hand calls
    rng.shuffle(hand_calls)
    n_test = max(1, int(len(hand_calls) * test_ratio)) if len(hand_calls) > 1 else 0
    test_calls = set(hand_calls[:n_test])
    train_calls = set(hand_calls[n_test:]) | set(weak_only_calls)

    # Assert no call appears in both splits
    assert not (train_calls & test_calls), "Call ID overlap detected between train and test splits!"

    train_records: list[dict[str, Any]] = []
    test_records: list[dict[str, Any]] = []

    for cid, turns in call_records.items():
        if cid in test_calls:
            for t in turns:
                # Extra check: weak labels must NEVER enter the test split
                if not t.get("is_weak", False):
                    test_records.append(t)
                else:
                    train_records.append(t)
        else:
            train_records.extend(turns)

    return train_records, test_records


def build_samples_from_records(
    records: list[dict[str, Any]],
    n_context: int = 3,
    speaker_roles: dict[str, str] | None = None,
) -> tuple[list[str], np.ndarray, np.ndarray, np.ndarray, list[dict[str, Any]]]:
    """Rebuild windows and features verbatim from referenced timelines using pure functions."""
    timeline_cache: dict[str, ConversationTimeline] = {}

    windows: list[str] = []
    dynamics_rows: list[list[float]] = []
    labels: list[int] = []
    weights: list[float] = []
    metas: list[dict[str, Any]] = []

    label_to_idx = {name: idx for idx, name in enumerate(LABELS)}

    for rec in records:
        tl_path = str(rec["timeline"])
        if tl_path not in timeline_cache:
            timeline_cache[tl_path] = ConversationTimeline.from_json(tl_path)
        tl = timeline_cache[tl_path]

        target_id = rec["turn_id"]
        # Find index in timeline
        target_idx = -1
        for i, t in enumerate(tl.turns):
            if t.turn_id == target_id:
                target_idx = i
                break

        if target_idx == -1:
            continue

        turn = tl.turns[target_idx]
        win = build_context_window(
            tl.turns,
            target_idx=target_idx,
            n_context=n_context,
            speaker_roles=speaker_roles,
        )
        feats = extract_turn_dynamics(turn, target_idx, len(tl.turns))
        feat_vector = [feats[name] for name in DYNAMICS_FEATURE_NAMES]

        lbl_str = str(rec["label"]).strip().lower()
        if lbl_str not in label_to_idx:
            continue

        windows.append(win)
        dynamics_rows.append(feat_vector)
        labels.append(label_to_idx[lbl_str])
        weights.append(float(rec.get("weight", 1.0)))
        metas.append({
            "turn_id": target_id,
            "call_id": tl.call_id,
            "provenance": turn.provenance,
            "overlap_flag": turn.overlap.ratio > 0.0 or turn.overlap.interrupts is not None,
        })

    return (
        windows,
        np.array(dynamics_rows, dtype=np.float32),
        np.array(labels, dtype=np.int64),
        np.array(weights, dtype=np.float32),
        metas,
    )


def fit_temperature(logits: np.ndarray, y_true: np.ndarray) -> float:
    """Fit optimal temperature scalar minimizing negative log-likelihood."""
    from scipy.optimize import minimize_scalar

    def _nll(T: float) -> float:
        if T <= 0:
            return 1e6
        probs = scale_temperature(logits, float(T))
        eps = 1e-12
        log_probs = np.log(np.clip(probs, eps, 1.0))
        # Negative log-likelihood
        loss = -np.mean(log_probs[np.arange(len(y_true)), y_true])
        return float(loss)

    res = minimize_scalar(_nll, bounds=(0.1, 10.0), method="bounded")
    return float(res.x) if res.success else 1.0


def train_semantic_model(
    labels_file: Path | str,
    out_dir: Path | str,
    weak_labels_file: Path | str | None = None,
    manifest_file: Path | str | None = None,
    n_context: int = 3,
    test_ratio: float = 0.2,
    seed: int = 42,
) -> dict[str, Any]:
    """Train logistic classification head over sentence embeddings + scaled dynamics features."""
    from convaudio.stage5_semantic.setfit_classifier import SetFitSemanticClassifier

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    records = load_dataset_records(
        labels_file=labels_file,
        weak_labels_file=weak_labels_file,
        manifest_file=manifest_file,
    )
    if not records:
        raise ConvaudioError("No labelled training records found.")

    train_recs, test_recs = split_records_by_call_id(records, test_ratio=test_ratio, seed=seed)

    # Build windows and dynamics
    tr_windows, tr_feats, tr_y, tr_weights, tr_meta = build_samples_from_records(
        train_recs, n_context=n_context
    )
    te_windows, te_feats, te_y, te_weights, te_meta = build_samples_from_records(
        test_recs, n_context=n_context
    )

    if len(tr_y) == 0:
        raise ConvaudioError("Zero valid training turns extracted.")
    if len(np.unique(tr_y)) < 2:
        raise ConvaudioError("Training data must contain samples from at least 2 distinct classes.")

    # 1. Fit scaler on training data ONLY
    scaler = DynamicsScaler(feature_names=DYNAMICS_FEATURE_NAMES)
    tr_feats_scaled = scaler.fit_transform(tr_feats)
    scaler.save(out_path / "scaler.json")

    # 2. Extract text embeddings
    base_model = SetFitSemanticClassifier()
    tr_embs = base_model._get_text_embeddings(tr_windows)
    tr_X = np.concatenate([tr_embs, tr_feats_scaled], axis=1)

    # 3. Fit classifier head
    clf = LogisticRegression(max_iter=1000, random_state=seed)
    clf.fit(tr_X, tr_y, sample_weight=tr_weights)

    # Ensure full (len(LABELS), n_features) weights and bias regardless of binary or multiclass fit
    n_features = tr_X.shape[1]
    full_W = np.zeros((len(LABELS), n_features), dtype=np.float32)
    full_b = np.full((len(LABELS),), -10.0, dtype=np.float32)

    if clf.coef_.shape[0] == 1:
        c0 = int(clf.classes_[0])
        c1 = int(clf.classes_[1])
        full_W[c1] = clf.coef_[0] / 2.0
        full_W[c0] = -clf.coef_[0] / 2.0
        full_b[c1] = float(clf.intercept_[0]) / 2.0
        full_b[c0] = -float(clf.intercept_[0]) / 2.0
    elif clf.coef_.shape[0] == len(LABELS):
        full_W = clf.coef_.astype(np.float32)
        full_b = clf.intercept_.astype(np.float32)
    else:
        for idx, c in enumerate(clf.classes_):
            full_W[int(c)] = clf.coef_[idx]
            full_b[int(c)] = clf.intercept_[idx]

    # 4. Fit temperature calibration
    tr_logits = np.dot(tr_X, full_W.T) + full_b
    temperature = fit_temperature(tr_logits, tr_y)

    # Save head weights
    np.savez(
        out_path / "head_weights.npz",
        weights=full_W,
        bias=full_b,
    )

    # Save artifact metadata
    artefact = {
        "labels": list(LABELS),
        "feature_names": list(DYNAMICS_FEATURE_NAMES),
        "temperature": temperature,
        "base_model": base_model.checkpoint_revision,
        "n_train_samples": len(tr_y),
    }
    with open(out_path / "artefact.json", "w", encoding="utf-8") as f:
        json.dump(artefact, f, indent=2)

    # 5. Evaluate on test split if present
    metrics: dict[str, Any] = {
        "n_train": len(tr_y),
        "n_test": len(te_y),
        "temperature": temperature,
    }

    if len(te_y) > 0:
        te_feats_scaled = scaler.transform(te_feats)
        te_embs = base_model._get_text_embeddings(te_windows)
        te_X = np.concatenate([te_embs, te_feats_scaled], axis=1)

        te_logits = np.dot(te_X, full_W.T) + full_b
        te_probs = scale_temperature(te_logits, temperature)
        te_preds = np.argmax(te_probs, axis=1)

        macro_f1 = f1_score(te_y, te_preds, average="macro", zero_division=0)
        c_matrix = confusion_matrix(te_y, te_preds, labels=list(range(len(LABELS))))
        abstention_rate = float(np.mean(np.max(te_probs, axis=1) < 0.45))

        # Stratified evaluation
        strat_prov: dict[str, Any] = {}
        for prov in ("clean", "separated"):
            mask = [m["provenance"] == prov for m in te_meta]
            if any(mask):
                sub_y = te_y[mask]
                sub_preds = te_preds[mask]
                strat_prov[prov] = {
                    "count": int(np.sum(mask)),
                    "macro_f1": float(f1_score(sub_y, sub_preds, average="macro", zero_division=0)),
                }

        strat_overlap: dict[str, Any] = {}
        for is_ov, name in ((True, "overlapped"), (False, "non_overlapped")):
            mask = [m["overlap_flag"] == is_ov for m in te_meta]
            if any(mask):
                sub_y = te_y[mask]
                sub_preds = te_preds[mask]
                strat_overlap[name] = {
                    "count": int(np.sum(mask)),
                    "macro_f1": float(f1_score(sub_y, sub_preds, average="macro", zero_division=0)),
                }

        metrics.update({
            "test_macro_f1": float(macro_f1),
            "abstention_rate": abstention_rate,
            "confusion_matrix": c_matrix.tolist(),
            "stratified_by_provenance": strat_prov,
            "stratified_by_overlap": strat_overlap,
        })

    with open(out_path / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    return metrics
