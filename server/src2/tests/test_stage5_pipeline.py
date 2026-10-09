"""Tests 7, 14, 15 for Stage 5 pipeline execution, caching, and scaler invariants."""

import json
import time
from pathlib import Path
from unittest.mock import MagicMock

from convaudio.pipeline import PipelineRunner
from convaudio.stage2_timeline.schema import ConversationTimeline
from convaudio.stage2_timeline.validate import validate_timeline
from tests.fixtures.synth import generate_clean_stereo_call


# 7. Scaler is loaded at inference, never fitted — assert on a mocked scaler
def test_7_scaler_never_fitted_at_inference(tmp_path: Path) -> None:
    audio_path = tmp_path / "call.wav"
    generate_clean_stereo_call(audio_path, duration_s=3.0)
    run_dir = tmp_path / "run"

    runner = PipelineRunner(stub_models=True)
    timeline = runner.run(audio_path, run_dir)

    # Now simulate a classifier with a mock scaler
    mock_scaler = MagicMock()
    mock_scaler.transform.side_effect = lambda x: x

    # Execute stage 5 with a custom classifier that has mock_scaler
    from convaudio.stage5_semantic.stub import StubSemanticClassifier

    stub_cls = StubSemanticClassifier()
    setattr(stub_cls, "scaler", mock_scaler)

    # Run _execute_stage5 with injected classifier
    runner._execute_stage5(timeline, run_dir, "test_input_hash", "test_text_hash", classifier=stub_cls)

    # Assert transform was called, fit and fit_transform were never called
    assert mock_scaler.transform.called
    assert not mock_scaler.fit.called
    assert not hasattr(mock_scaler, "fit_transform") or not mock_scaler.fit_transform.called


# 14. --stub-models runs Stages 1–5 end-to-end, schema-valid, no GPU, no network, under 10 seconds
def test_14_stub_models_end_to_end_under_10s(tmp_path: Path) -> None:
    audio_path = tmp_path / "audio_stub.wav"
    generate_clean_stereo_call(audio_path, duration_s=4.0)
    run_dir = tmp_path / "run_stub"

    runner = PipelineRunner(stub_models=True)
    t0 = time.perf_counter()
    timeline = runner.run(audio_path, run_dir)
    elapsed = time.perf_counter() - t0

    assert elapsed < 10.0, f"Stub pipeline took {elapsed:.2f}s, expected < 10.0s"
    assert (run_dir / "stage5.json").exists()
    assert (run_dir / "timeline.json").exists()

    # Schema validation
    validate_timeline(run_dir / "timeline.json")
    assert timeline.schema_version == "1.1.0"

    # Invariant: every turn has semantic or semantic_abstain_reason, mutually exclusive
    assert len(timeline.turns) > 0
    for t in timeline.turns:
        has_sem = t.semantic is not None
        has_abs = t.semantic_abstain_reason is not None
        assert (has_sem and not has_abs) or (has_abs and not has_sem)


# 15. Stage 5 re-runs without re-running Stages 1–4; changing Stage 3 text invalidates Stage 5 checkpoint
def test_15_stage5_resumability_and_text_invalidation(tmp_path: Path) -> None:
    audio_path = tmp_path / "audio_resumable.wav"
    generate_clean_stereo_call(audio_path, duration_s=4.0)
    run_dir = tmp_path / "run_resumable"

    runner = PipelineRunner(stub_models=True)
    runner.run(audio_path, run_dir)

    # Initial mtimes
    s1_mtime = (run_dir / "stage1.json").stat().st_mtime_ns
    s2_mtime = (run_dir / "stage2.json").stat().st_mtime_ns
    s3_mtime = (run_dir / "stage3.json").stat().st_mtime_ns
    s4_mtime = (run_dir / "stage4.json").stat().st_mtime_ns
    s5_mtime = (run_dir / "stage5.json").stat().st_mtime_ns

    time.sleep(0.05)

    # Part A: Running stage 5 individually must NOT re-run Stages 1–4
    runner.run_stage(stage_num=5, run_dir=run_dir, force=True)
    assert (run_dir / "stage1.json").stat().st_mtime_ns == s1_mtime
    assert (run_dir / "stage2.json").stat().st_mtime_ns == s2_mtime
    assert (run_dir / "stage3.json").stat().st_mtime_ns == s3_mtime
    assert (run_dir / "stage4.json").stat().st_mtime_ns == s4_mtime
    assert (run_dir / "stage5.json").stat().st_mtime_ns > s5_mtime

    # Part B: Changing Stage 3 text invalidates Stage 5 checkpoint
    with open(run_dir / "stage5.json", "r", encoding="utf-8") as f:
        old_s5_data = json.load(f)
    old_text_hash = old_s5_data["text_hash"]

    # Modify text of first turn in timeline.json
    tl = ConversationTimeline.from_json(run_dir / "timeline.json")
    assert len(tl.turns) > 0
    tl.turns[0].text = "Completely altered transcript text"
    tl.to_json(run_dir / "timeline.json")

    time.sleep(0.05)
    # Run stage 5 via runner.run (non-force)
    # Since text changed, text_hash is different -> Stage 5 MUST re-run, not skip
    runner.run(audio_path, run_dir, force=False)

    with open(run_dir / "stage5.json", "r", encoding="utf-8") as f:
        new_s5_data = json.load(f)
    new_text_hash = new_s5_data["text_hash"]

    assert new_text_hash != old_text_hash
