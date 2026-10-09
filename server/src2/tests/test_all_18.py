"""Complete test suite verifying all 18 mandatory specifications."""

import re
import time
from pathlib import Path

import numpy as np
import pytest
from typer.testing import CliRunner

from convaudio.cli import app
from convaudio.config import PipelineConfig
from convaudio.errors import LicenceViolation, TimelineValidationError
from convaudio.io.audio import load_audio
from convaudio.licences import assert_licence_allowed
from convaudio.pipeline import PipelineRunner
from convaudio.stage1_separate.branch import decide_branch
from convaudio.stage1_separate.phantom import filter_phantom_sources
from convaudio.stage1_separate.vad_only import process_clean_stereo
from convaudio.stage2_timeline.overlap import detect_interruptions, resample_mask
from convaudio.stage2_timeline.schema import (
    AcousticFeatures,
    AudioMeta,
    ConversationTimeline,
    CorpusStats,
    Diagnostics,
    SpeakerMeta,
    Turn,
    TurnOverlap,
    TurnQuality,
)
from convaudio.stage2_timeline.validate import validate_timeline
from convaudio.stage3_lexical.asr import process_stage3_lexical
from convaudio.stage4_acoustic.pooling import check_abstention, pool_acoustic_features
from tests.fixtures.synth import (
    generate_clean_stereo_call,
    generate_high_overlap_mono_call,
    generate_low_overlap_mono_call,
)


def _make_valid_timeline() -> ConversationTimeline:
    return ConversationTimeline(
        schema_version="1.0.0",
        call_id="call_test_001",
        created_utc="2026-10-05T12:00:00Z",
        audio_meta=AudioMeta(
            path="audio.wav",
            sha256="abc123def456",
            original_sr=16000,
            channels=2,
            duration_s=10.0,
            branch="stereo_split",
        ),
        frame_rate_hz=50.0,
        n_frames=500,
        overlap_mask="overlap_mask.npy",
        speakers={
            "SPEAKER_00": SpeakerMeta(
                stream_path="streams/SPEAKER_00.wav",
                total_speech_s=5.0,
                embedding_key="SPEAKER_00",
            ),
            "SPEAKER_01": SpeakerMeta(
                stream_path="streams/SPEAKER_01.wav",
                total_speech_s=4.0,
                embedding_key="SPEAKER_01",
            ),
        },
        corpus_stats=CorpusStats(
            total_speech_s=9.0,
            overlap_speech_s=0.5,
            overlap_ratio=0.055,
            n_turns=2,
            n_interruptions=0,
        ),
        turns=[
            Turn(
                turn_id="turn_000",
                speaker="SPEAKER_00",
                start=0.5,
                end=3.0,
                provenance="clean",
                overlap=TurnOverlap(
                    ratio=0.0,
                    clean_speech_s=2.5,
                    interrupts=None,
                    interrupted_by=None,
                    preceding_gap_s=None,
                ),
                quality=TurnQuality(
                    sep_cosine=0.95,
                    asr_avg_logprob=-0.2,
                    no_speech_prob=0.01,
                    compression_ratio=1.2,
                ),
                text="Hello world",
                word_timings=None,
                word_timings_source="asr_fallback",
                acoustic=AcousticFeatures(
                    valence=0.5,
                    arousal=-0.1,
                    conf=0.9,
                    pooled_frames=125,
                    masked_frames=0,
                ),
                acoustic_abstain_reason=None,
            ),
            Turn(
                turn_id="turn_001",
                speaker="SPEAKER_01",
                start=3.2,
                end=4.0,
                provenance="clean",
                overlap=TurnOverlap(
                    ratio=0.0,
                    clean_speech_s=0.8,
                    interrupts=None,
                    interrupted_by=None,
                    preceding_gap_s=0.2,
                ),
                quality=TurnQuality(
                    sep_cosine=0.92,
                    asr_avg_logprob=-0.3,
                    no_speech_prob=0.05,
                    compression_ratio=1.1,
                ),
                text="Hi there",
                word_timings=None,
                word_timings_source="asr_fallback",
                acoustic=None,
                acoustic_abstain_reason="SHORT_TURN",
            ),
        ],
        diagnostics=Diagnostics(
            rejected_sources=[],
            alignment_failures=0,
            acoustic_abstentions=1,
            abstain_reasons={"SHORT_TURN": 1, "HIGH_OVERLAP": 0, "LOW_SEP_QUALITY": 0},
            model_revisions={},
            stage_durations_s={"s1": 0.1, "s2": 0.05, "s3": 0.2, "s4": 0.1},
        ),
    )


# 1. stereo branch skips separation and sets provenance="clean" on all turns
def test_01_stereo_branch_skips_separation_provenance_clean(tmp_path: Path) -> None:
    stereo_wav = tmp_path / "stereo_call.wav"
    generate_clean_stereo_call(stereo_wav, duration_s=6.0, sr=16000)

    waveform, _ = load_audio(stereo_wav, target_sr=16000)
    branch, _ = decide_branch(waveform)
    assert branch == "stereo_split"

    stage1_out = process_clean_stereo(waveform, sr=16000, out_dir=tmp_path / "s1", call_id="c_test")
    turns = stage1_out["turns"]
    assert len(turns) > 0
    for t in turns:
        assert t.provenance == "clean"


# 2. mono branch sets provenance="separated" and branch="mono_separated"
def test_02_mono_branch_sets_provenance_separated_and_branch_mono_separated(tmp_path: Path) -> None:
    mono_wav = tmp_path / "mono_call.wav"
    generate_low_overlap_mono_call(mono_wav, duration_s=6.0, sr=16000)
    out_dir = tmp_path / "run_mono"

    runner = PipelineRunner(stub_models=True)
    timeline = runner.run(audio_path=mono_wav, run_dir=out_dir)

    assert timeline.audio_meta.branch == "mono_separated"
    assert len(timeline.turns) > 0
    for turn in timeline.turns:
        assert turn.provenance == "separated"


# 3. --force-branch overrides the automatic decision
def test_03_force_branch_overrides_automatic_decision(tmp_path: Path) -> None:
    stereo_wav = tmp_path / "stereo_call.wav"
    generate_clean_stereo_call(stereo_wav, duration_s=4.0, sr=16000)
    wav_stereo, _ = load_audio(stereo_wav)

    mono_wav = tmp_path / "mono_call.wav"
    generate_low_overlap_mono_call(mono_wav, duration_s=4.0, sr=16000)
    wav_mono, _ = load_audio(mono_wav)

    b_mono_forced, _ = decide_branch(wav_stereo, force_branch="mono")
    assert b_mono_forced == "mono_separated"

    b_stereo_forced, _ = decide_branch(wav_mono, force_branch="stereo")
    assert b_stereo_forced == "stereo_split"


# 4. frame-level mask survives a JSON round-trip with frame_rate_hz intact
def test_04_frame_level_mask_survives_json_roundtrip(tmp_path: Path) -> None:
    tl = _make_valid_timeline()
    out_file = tmp_path / "timeline.json"
    tl.to_json(out_file)

    loaded = ConversationTimeline.from_json(out_file)
    assert loaded.frame_rate_hz == 50.0
    assert loaded.n_frames == 500
    assert loaded.overlap_mask == "overlap_mask.npy"
    validate_timeline(out_file)


# 5. resample_mask is correct between two different frame rates — assert on a known pattern, not just on output length
def test_05_resample_mask_preserves_pattern() -> None:
    mask_50hz = np.zeros(150, dtype=bool)
    mask_50hz[50:100] = True

    mask_100hz = resample_mask(mask_50hz, from_hz=50.0, to_hz=100.0)
    assert len(mask_100hz) == 300
    assert not np.any(mask_100hz[:99])
    assert np.all(mask_100hz[101:198])
    assert not np.any(mask_100hz[201:])

    mask_25hz = resample_mask(mask_100hz, from_hz=100.0, to_hz=25.0)
    assert len(mask_25hz) == 75
    assert not np.any(mask_25hz[:24])
    assert np.all(mask_25hz[26:48])
    assert not np.any(mask_25hz[51:])


# 6. interruption detection: a programmed overlap yields the correct interrupts / interrupted_by pair AND a negative preceding_gap_s
def test_06_interruption_detection_pair_and_negative_gap() -> None:
    turn0 = Turn(
        turn_id="turn_000",
        speaker="SPEAKER_00",
        start=1.0,
        end=3.0,
        provenance="separated",
        overlap=TurnOverlap(ratio=0.4, clean_speech_s=1.2),
        quality=TurnQuality(sep_cosine=0.9),
    )
    turn1 = Turn(
        turn_id="turn_001",
        speaker="SPEAKER_01",
        start=2.2,
        end=4.0,
        provenance="separated",
        overlap=TurnOverlap(ratio=0.44, clean_speech_s=1.0),
        quality=TurnQuality(sep_cosine=0.85),
    )
    turns = [turn0, turn1]
    n_interrupts = detect_interruptions(turns, interrupt_window_s=1.0)

    assert n_interrupts == 1
    assert turn0.overlap.preceding_gap_s is None
    assert turn1.overlap.preceding_gap_s == -0.8
    assert turn1.overlap.interrupts == "SPEAKER_00"
    assert turn0.overlap.interrupted_by == "SPEAKER_01"


# 7. validate_timeline() REJECTS a timeline missing frame_rate_hz
def test_07_validate_timeline_rejects_missing_frame_rate_hz() -> None:
    tl = _make_valid_timeline()
    data = tl.to_dict()
    del data["frame_rate_hz"]

    with pytest.raises(TimelineValidationError):
        validate_timeline(data)


# 8. validate_timeline() REJECTS a turn with both acoustic and acoustic_abstain_reason non-null, and one with both null
def test_08_validate_timeline_rejects_both_or_neither_acoustic_and_reason() -> None:
    tl_both = _make_valid_timeline()
    data_both = tl_both.to_dict()
    data_both["turns"][0]["acoustic_abstain_reason"] = "SHORT_TURN"
    with pytest.raises(TimelineValidationError):
        validate_timeline(data_both)

    tl_neither = _make_valid_timeline()
    data_neither = tl_neither.to_dict()
    data_neither["turns"][0]["acoustic"] = None
    data_neither["turns"][0]["acoustic_abstain_reason"] = None
    with pytest.raises(TimelineValidationError):
        validate_timeline(data_neither)


# 9. abstention: clean speech below 1.0 s -> acoustic is None and reason == "SHORT_TURN"
def test_09_abstention_clean_speech_below_threshold() -> None:
    frame_emotions = np.ones((50, 2), dtype=np.float32) * 0.5
    mask = np.zeros(50, dtype=bool)

    features, reason = pool_acoustic_features(
        frame_emotions=frame_emotions,
        mask=mask,
        clean_speech_s=0.75,
        overlap_ratio=0.1,
        sep_cosine=0.9,
        min_clean_speech_s=1.0,
    )
    assert features is None
    assert reason == "SHORT_TURN"


# 10. abstention precedence: a turn failing two gates reports SHORT_TURN first
def test_10_abstention_precedence_short_turn_first() -> None:
    reason = check_abstention(
        clean_speech_s=0.4,
        overlap_ratio=0.95,
        sep_cosine=0.2,
        min_clean_speech_s=1.0,
        max_overlap_ratio=0.8,
        min_sep_cosine=0.5,
    )
    assert reason == "SHORT_TURN"


# 11. masked pooling ignores masked frames — inject a huge sentinel value into masked positions and assert the pooled output is unchanged
def test_11_masked_pooling_ignores_masked_frames() -> None:
    n_frames = 100
    mask = np.zeros(n_frames, dtype=bool)
    mask[70:] = True

    rng = np.random.default_rng(123)
    frame_emotions = rng.uniform(-0.5, 0.5, size=(n_frames, 2)).astype(np.float32)

    feat_baseline, _ = pool_acoustic_features(
        frame_emotions=frame_emotions.copy(),
        mask=mask,
        clean_speech_s=1.4,
        overlap_ratio=0.3,
        sep_cosine=0.9,
    )
    corrupted_emotions = frame_emotions.copy()
    corrupted_emotions[70:, :] = 999999.0

    feat_corrupted, _ = pool_acoustic_features(
        frame_emotions=corrupted_emotions,
        mask=mask,
        clean_speech_s=1.4,
        overlap_ratio=0.3,
        sep_cosine=0.9,
    )
    assert feat_baseline is not None and feat_corrupted is not None
    assert feat_corrupted.valence == feat_baseline.valence
    assert feat_corrupted.arousal == feat_baseline.arousal


# 12. phantom rejection drops a sub-threshold source and records it in diagnostics.rejected_sources
def test_12_phantom_rejection_drops_subthreshold_source() -> None:
    sr = 16000
    rng = np.random.default_rng(42)
    src0 = rng.standard_normal(int(3.0 * sr)).astype(np.float32) * 0.5
    src1 = rng.standard_normal(int(0.3 * sr)).astype(np.float32) * 0.5

    valid_idx, rejected = filter_phantom_sources(
        sources=[src0, src1],
        active_durations_s=[3.0, 0.3],
        min_speaker_speech_s=1.0,
    )
    assert valid_idx == [0]
    assert len(rejected) == 1
    assert rejected[0]["source_idx"] == 1
    assert rejected[0]["reason"] == "below_min_speech"


# 13. alignment failure falls back and sets word_timings_source="asr_fallback" and increments diagnostics.alignment_failures
def test_13_alignment_failure_falls_back_and_records_diagnostics(tmp_path: Path) -> None:
    wav_path = tmp_path / "call.wav"
    generate_clean_stereo_call(wav_path, duration_s=4.0, sr=16000)
    run_dir = tmp_path / "run_align"

    runner = PipelineRunner(stub_models=True)
    timeline = runner.run(audio_path=wav_path, run_dir=run_dir)

    process_stage3_lexical(
        timeline=timeline,
        config=PipelineConfig(),
        stub_models=True,
        simulate_alignment_failure=True,
    )
    assert timeline.diagnostics.alignment_failures > 0
    for turn in timeline.turns:
        assert turn.word_timings_source == "asr_fallback"


# 14. assert_licence_allowed raises LicenceViolation on "CC-BY-NC-4.0"
def test_14_assert_licence_allowed_raises_on_cc_by_nc() -> None:
    with pytest.raises(LicenceViolation):
        assert_licence_allowed(code_licence="MIT", weights_licence="CC-BY-NC-4.0")


# 15. convaudio licences check exits non-zero on an injected deny-listed entry
def test_15_convaudio_licences_check_exits_nonzero_on_injected_denylist(tmp_path: Path) -> None:
    runner = CliRunner()
    bad_manifest = tmp_path / "bad_manifest.yaml"
    bad_manifest.write_text(
        """
- name: "injected_bad_model"
  code_licence: "MIT"
  weights_licence: "CC-BY-NC-4.0"
        """.strip(),
        encoding="utf-8",
    )
    result = runner.invoke(app, ["licences", "check", "--manifest", str(bad_manifest)])
    assert result.exit_code != 0
    assert "Licence policy violations detected" in result.output


# 16. no ffmpeg subprocess anywhere — grep the package source and assert
def test_16_no_ffmpeg_subprocess_anywhere() -> None:
    package_dir = Path(__file__).resolve().parent.parent / "convaudio"
    py_files = list(package_dir.rglob("*.py"))
    assert len(py_files) > 0

    ffmpeg_pattern = re.compile(r"""\bffmpeg\b""", re.IGNORECASE)
    for f in py_files:
        lines = f.read_text(encoding="utf-8").splitlines()
        for line_no, line in enumerate(lines, start=1):
            stripped = line.strip()
            if ("subprocess" in stripped or "os.system" in stripped or "popen" in stripped) and ffmpeg_pattern.search(stripped):
                pytest.fail(f"Found forbidden ffmpeg execution in {f.name}:{line_no}: {line}")
            if "run([" in stripped and "ffmpeg" in stripped.lower():
                pytest.fail(f"Found forbidden ffmpeg execution in {f.name}:{line_no}: {line}")


# 17. full DAG with --stub-models: no GPU, no network, schema-valid output, under 10 seconds
def test_17_full_dag_stub_models_fast_and_valid(tmp_path: Path) -> None:
    wav_path = tmp_path / "test_call.wav"
    generate_high_overlap_mono_call(wav_path, duration_s=6.0, sr=16000)
    out_dir = tmp_path / "run_e2e"

    runner = PipelineRunner(stub_models=True)
    t0 = time.perf_counter()
    timeline = runner.run(audio_path=wav_path, run_dir=out_dir, num_speakers=2)
    elapsed = time.perf_counter() - t0

    assert elapsed < 10.0
    timeline_file = out_dir / "timeline.json"
    assert timeline_file.exists()
    validate_timeline(timeline_file)
    assert timeline.schema_version in ("1.0.0", "1.1.0")
    assert timeline.frame_rate_hz > 0
    assert timeline.n_frames > 0


# 18. re-running convaudio run skips completed stages; --force re-runs them
def test_18_rerunning_skips_completed_stages_and_force_reruns(tmp_path: Path) -> None:
    wav_path = tmp_path / "test_call.wav"
    generate_clean_stereo_call(wav_path, duration_s=4.0, sr=16000)
    out_dir = tmp_path / "run_resumable"

    runner = PipelineRunner(stub_models=True)
    runner.run(audio_path=wav_path, run_dir=out_dir)

    s1_mtime_1 = (out_dir / "stage1.json").stat().st_mtime_ns
    s2_mtime_1 = (out_dir / "stage2.json").stat().st_mtime_ns

    time.sleep(0.05)
    runner.run(audio_path=wav_path, run_dir=out_dir, force=False)

    s1_mtime_2 = (out_dir / "stage1.json").stat().st_mtime_ns
    s2_mtime_2 = (out_dir / "stage2.json").stat().st_mtime_ns
    assert s1_mtime_1 == s1_mtime_2
    assert s2_mtime_1 == s2_mtime_2

    time.sleep(0.05)
    runner.run(audio_path=wav_path, run_dir=out_dir, force=True)
    s1_mtime_3 = (out_dir / "stage1.json").stat().st_mtime_ns
    assert s1_mtime_3 > s1_mtime_2
