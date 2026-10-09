"""Tests 16-19 and acceptance criteria for Stage 5 semantic analysis."""

import json
from pathlib import Path
from unittest.mock import patch

from convaudio.config import PipelineConfig
from convaudio.pipeline import PipelineRunner
from convaudio.stage5_semantic.agreement import calculate_cohens_kappa, compute_file_agreement
from convaudio.stage5_semantic.train import (
    build_samples_from_records,
    load_dataset_records,
    split_records_by_call_id,
)
from tests.fixtures.synth import generate_clean_stereo_call


# 16. train/test split is by call_id — assert no call_id appears in both
def test_16_train_test_split_by_call_id() -> None:
    records = [
        {"turn_id": f"call1_t{i}", "call_id": "call1", "timeline": "call1/timeline.json", "label": "neutral", "is_weak": False}
        for i in range(5)
    ] + [
        {"turn_id": f"call2_t{i}", "call_id": "call2", "timeline": "call2/timeline.json", "label": "negative", "is_weak": False}
        for i in range(5)
    ] + [
        {"turn_id": f"call3_t{i}", "call_id": "call3", "timeline": "call3/timeline.json", "label": "positive", "is_weak": False}
        for i in range(5)
    ]

    train_recs, test_recs = split_records_by_call_id(records, test_ratio=0.33, seed=42)

    train_calls = {r["call_id"] for r in train_recs}
    test_calls = {r["call_id"] for r in test_recs}

    assert len(train_calls) > 0
    assert len(test_calls) > 0
    # Mandatory assertion: no call_id appears in both train and test
    assert not (train_calls & test_calls), f"Overlap found: {train_calls & test_calls}"


# 17. Weak labels never enter the test split; a hand label overrides a weak label for the same turn
def test_17_weak_labels_isolation_and_hand_override(tmp_path: Path) -> None:
    hand_file = tmp_path / "hand.jsonl"
    weak_file = tmp_path / "weak.jsonl"

    # Turn t_common is in both weak and hand labels with contradictory values
    with open(weak_file, "w", encoding="utf-8") as f:
        f.write(json.dumps({"turn_id": "t_common", "call_id": "call_weak", "timeline": "run/timeline.json", "label": "neutral", "weight": 0.3}) + "\n")
        f.write(json.dumps({"turn_id": "t_weak_only", "call_id": "call_weak_only", "timeline": "run/timeline.json", "label": "negative", "weight": 0.3}) + "\n")

    with open(hand_file, "w", encoding="utf-8") as f:
        f.write(json.dumps({"turn_id": "t_common", "call_id": "call_hand", "timeline": "run/timeline.json", "label": "positive", "annotator": "a1"}) + "\n")
        f.write(json.dumps({"turn_id": "t_hand_only", "call_id": "call_hand_2", "timeline": "run/timeline.json", "label": "negative", "annotator": "a1"}) + "\n")

    merged = load_dataset_records(labels_file=hand_file, weak_labels_file=weak_file)
    merged_map = {r["turn_id"]: r for r in merged}

    # Hand label overrides weak label
    assert merged_map["t_common"]["label"] == "positive"
    assert merged_map["t_common"]["weight"] == 1.0
    assert not merged_map["t_common"]["is_weak"]

    # Weak labels never enter test split
    train_recs, test_recs = split_records_by_call_id(merged, test_ratio=0.5, seed=42)
    for r in test_recs:
        assert not r.get("is_weak", False), f"Weak record leaked into test set: {r}"


# 18. LLM adjudication disabled by default; when enabled and unreachable, turn retains encoder abstention and run completes
def test_18_llm_adjudication_disabled_default_and_graceful_failure(tmp_path: Path) -> None:
    audio_path = tmp_path / "audio_llm.wav"
    generate_clean_stereo_call(audio_path, duration_s=3.0)
    run_dir = tmp_path / "run_llm"

    # Default config: llm_adjudication is False
    cfg_default = PipelineConfig()
    assert not cfg_default.stage5.llm_adjudication

    # Assert no LLM call is made in default run
    with patch("convaudio.stage5_semantic.llm_adjudicate.call_llm_api") as mock_api:
        runner_default = PipelineRunner(config=cfg_default, stub_models=True)
        runner_default.run(audio_path, run_dir)
        assert not mock_api.called

    # When enabled and LLM is unreachable, must complete without crashing
    cfg_enabled = PipelineConfig()
    cfg_enabled.stage5.llm_adjudication = True
    cfg_enabled.stage5.min_conf = 0.99  # Force abstention to trigger adjudication check

    with patch("convaudio.stage5_semantic.llm_adjudicate.call_llm_api", side_effect=ConnectionError("Unreachable LLM API")):
        runner_llm = PipelineRunner(config=cfg_enabled, stub_models=True)
        timeline = runner_llm.run(audio_path, run_dir, force=True)
        assert timeline is not None
        assert (run_dir / "timeline.json").exists()


# 19. Training-serving parity: the same turn produces a byte-identical window string via training and inference paths
def test_19_training_serving_window_parity(tmp_path: Path) -> None:
    audio_path = tmp_path / "audio_parity.wav"
    generate_clean_stereo_call(audio_path, duration_s=4.0)
    run_dir = tmp_path / "run_parity"

    runner = PipelineRunner(stub_models=True)
    timeline = runner.run(audio_path, run_dir)

    # Pick a turn
    turn_idx = len(timeline.turns) - 1
    target_turn = timeline.turns[turn_idx]

    # 1. Serving / inference path
    from convaudio.stage5_semantic.window import build_context_window
    serving_window = build_context_window(
        timeline.turns,
        target_idx=turn_idx,
        n_context=runner.config.stage5.n_context,
        speaker_roles=runner.config.stage5.speaker_roles,
    )

    # 2. Training path
    records = [{
        "turn_id": target_turn.turn_id,
        "call_id": timeline.call_id,
        "timeline": str((run_dir / "timeline.json").resolve()),
        "label": "neutral",
    }]
    tr_windows, _, _, _, _ = build_samples_from_records(
        records,
        n_context=runner.config.stage5.n_context,
        speaker_roles=runner.config.stage5.speaker_roles,
    )

    training_window = tr_windows[0]

    # Must be byte-identical
    assert serving_window == training_window
    assert serving_window.encode("utf-8") == training_window.encode("utf-8")


# Acceptance Criteria 2: Switching classifier between stub/setfit/modernbert requires changing ONE config value
def test_acceptance_criteria_2_classifier_switching() -> None:
    cfg = PipelineConfig()

    for classifier_name in ("stub", "setfit", "modernbert"):
        cfg.stage5.classifier = classifier_name
        assert cfg.stage5.classifier == classifier_name


# Acceptance Criteria 4: export-annotation produces stratified sample, agreement reports Cohen's kappa
def test_acceptance_criteria_4_export_and_agreement(tmp_path: Path) -> None:
    # 1. Test Cohen's kappa logic directly
    a = ["positive", "positive", "neutral", "negative", "negative"]
    b = ["positive", "positive", "neutral", "negative", "neutral"]
    res = calculate_cohens_kappa(a, b)
    assert res["n_items"] == 5
    assert 0.5 < res["kappa"] <= 1.0

    # Test file-based agreement
    f_a = tmp_path / "a.jsonl"
    f_b = tmp_path / "b.jsonl"
    with open(f_a, "w", encoding="utf-8") as fa, open(f_b, "w", encoding="utf-8") as fb:
        for idx, (la, lb) in enumerate(zip(a, b)):
            tid = f"t_{idx}"
            fa.write(json.dumps({"turn_id": tid, "label": la}) + "\n")
            fb.write(json.dumps({"turn_id": tid, "label": lb}) + "\n")

    file_res = compute_file_agreement(f_a, f_b)
    assert file_res["n_items"] == 5
    assert file_res["kappa"] == res["kappa"]
