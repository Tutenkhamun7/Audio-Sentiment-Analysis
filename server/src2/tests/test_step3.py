"""Tests for Step 3: Audio ingest, channel branch, clean-stereo, and no-ffmpeg (Tests 1, 3, 16)."""

import re
from pathlib import Path

import pytest

from convaudio.io.audio import load_audio
from convaudio.stage1_separate.branch import decide_branch
from convaudio.stage1_separate.vad_only import process_clean_stereo
from tests.fixtures.synth import generate_clean_stereo_call, generate_low_overlap_mono_call


def test_1_stereo_branch_skips_separation_and_sets_provenance_clean(tmp_path: Path) -> None:
    """Test 1: stereo branch skips separation and sets provenance='clean' on all turns."""
    stereo_wav = tmp_path / "stereo_call.wav"
    generate_clean_stereo_call(stereo_wav, duration_s=6.0, sr=16000)

    waveform, meta = load_audio(stereo_wav, target_sr=16000)
    branch, metrics = decide_branch(waveform)
    assert branch == "stereo_split"

    stage1_out = process_clean_stereo(waveform, sr=16000, out_dir=tmp_path / "s1", call_id="c_test")
    turns = stage1_out["turns"]
    assert len(turns) > 0
    for t in turns:
        assert t.provenance == "clean"


def test_3_force_branch_overrides_automatic_decision(tmp_path: Path) -> None:
    """Test 3: --force-branch overrides the automatic decision."""
    stereo_wav = tmp_path / "stereo_call.wav"
    generate_clean_stereo_call(stereo_wav, duration_s=4.0, sr=16000)
    wav_stereo, _ = load_audio(stereo_wav)

    mono_wav = tmp_path / "mono_call.wav"
    generate_low_overlap_mono_call(mono_wav, duration_s=4.0, sr=16000)
    wav_mono, _ = load_audio(mono_wav)

    # Naturally stereo -> forced mono
    b_mono_forced, _ = decide_branch(wav_stereo, force_branch="mono")
    assert b_mono_forced == "mono_separated"

    # Naturally mono -> forced stereo
    b_stereo_forced, _ = decide_branch(wav_mono, force_branch="stereo")
    assert b_stereo_forced == "stereo_split"


def test_16_no_ffmpeg_subprocess_anywhere() -> None:
    """Test 16: no ffmpeg subprocess anywhere — grep the package source and assert."""
    package_dir = Path(__file__).resolve().parent.parent / "convaudio"
    py_files = list(package_dir.rglob("*.py"))
    assert len(py_files) > 0

    ffmpeg_pattern = re.compile(r"""\bffmpeg\b""", re.IGNORECASE)
    # Exclude comments/docstrings explaining that ffmpeg is forbidden
    for f in py_files:
        lines = f.read_text(encoding="utf-8").splitlines()
        for line_no, line in enumerate(lines, start=1):
            stripped = line.strip()
            # If line invokes subprocess or os.system or runs a command with ffmpeg
            if ("subprocess" in stripped or "os.system" in stripped or "popen" in stripped) and ffmpeg_pattern.search(stripped):
                pytest.fail(f"Found forbidden ffmpeg execution in {f.name}:{line_no}: {line}")
            # Also ensure no direct string literal "ffmpeg" in execution context
            if "run([" in stripped and "ffmpeg" in stripped.lower():
                pytest.fail(f"Found forbidden ffmpeg execution in {f.name}:{line_no}: {line}")
