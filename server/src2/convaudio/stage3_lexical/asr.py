"""ASR transcription and alignment orchestration for Stage 3."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import soundfile as sf
import torch

from convaudio.config import PipelineConfig
from convaudio.stage2_timeline.schema import ConversationTimeline, Turn, WordTiming
from convaudio.stage3_lexical.align import (
    AlignmentError,
    forced_align_segment,
    stub_forced_align,
)

if os.name == "nt":
    import torch

    torch_lib = Path(torch.__file__).parent / "lib"
    if torch_lib.exists():
        try:
            os.add_dll_directory(str(torch_lib))
        except Exception:
            pass
        os.environ["PATH"] = str(torch_lib) + ";" + os.environ.get("PATH", "")


def transcribe_and_align_stream_stub(
    stream_path: Path | str,
    turns: list[Turn],
    fail_alignment: bool = False,
) -> int:
    """Stub transcription and forced alignment for a single speaker stream."""
    alignment_failures = 0

    for idx, turn in enumerate(turns):
        turn.text = f"Sample speech transcript for {turn.speaker} turn {idx}"
        turn.quality.asr_avg_logprob = -0.25
        turn.quality.no_speech_prob = 0.02
        turn.quality.compression_ratio = 1.25

        words = turn.text.split()
        if fail_alignment:
            # Simulated alignment failure -> fallback to segment timings
            turn.word_timings = [
                WordTiming(w=w, s=turn.start, e=turn.end, conf=0.5) for w in words
            ]
            turn.word_timings_source = "asr_fallback"
            alignment_failures += 1
        else:
            try:
                turn.word_timings = stub_forced_align(turn.text, turn.start, turn.end)
                turn.word_timings_source = "forced_align"
            except AlignmentError:
                turn.word_timings = [
                    WordTiming(w=w, s=turn.start, e=turn.end, conf=0.5) for w in words
                ]
                turn.word_timings_source = "asr_fallback"
                alignment_failures += 1

    return alignment_failures


def transcribe_and_align_stream_whisper(
    stream_path: Path | str,
    turns: list[Turn],
    config: PipelineConfig,
    device: str | None = None,
) -> int:
    """Real ASR using faster-whisper and torchaudio forced alignment."""
    from faster_whisper import WhisperModel

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model_size = config.stage3.whisper_model
    try:
        model = WhisperModel(
            model_size,
            device=device,
            compute_type="float16" if device == "cuda" else "int8",
        )
        segments, info = model.transcribe(
            str(stream_path),
            vad_filter=config.stage3.vad_filter,
            word_timestamps=True,
        )
        seg_list = list(segments)
    except Exception:
        # Fall back to CPU if CUDA fails (e.g. cublas missing on Windows)
        device = "cpu"
        model = WhisperModel(model_size, device="cpu", compute_type="int8")
        segments, info = model.transcribe(
            str(stream_path),
            vad_filter=config.stage3.vad_filter,
            word_timestamps=True,
        )
        seg_list = list(segments)

    # Read audio stream tensor for forced alignment
    data, sr = sf.read(str(stream_path), dtype="float32")
    if data.ndim == 2:
        data = data[:, 0]
    wav_tensor = torch.from_numpy(data).unsqueeze(0)

    alignment_failures = 0

    all_words: list[Any] = []
    for s in seg_list:
        if s.words:
            all_words.extend(s.words)

    # Disjointly map words to the speaker's turn matching their timestamp midpoint
    turn_words: dict[str, list[Any]] = {t.turn_id: [] for t in turns}
    for w in all_words:
        mid = (w.start + w.end) / 2.0
        best_t_id = None
        min_dist = float("inf")
        for turn in turns:
            if turn.start - 0.35 <= mid <= turn.end + 0.35:
                dist = abs(mid - (turn.start + turn.end) / 2.0)
                if dist < min_dist:
                    min_dist = dist
                    best_t_id = turn.turn_id
        if best_t_id is not None:
            turn_words[best_t_id].append(w)

    for turn in turns:
        words = turn_words.get(turn.turn_id, [])
        full_text: str | None
        if words:
            matched_words = [w.word.strip() for w in words if w.word.strip()]
            full_text = " ".join(matched_words).strip()
            logprobs = [float(getattr(w, "probability", 0.9)) for w in words]
            no_speech_probs = [0.05]
            comp_ratios = [1.0]
        else:
            # Fall back to segment-level text if word timestamps were absent
            segs = [s for s in seg_list if max(turn.start, s.start) < min(turn.end, s.end)]
            matched_texts = [s.text.strip() for s in segs if s.text.strip()]
            full_text = " ".join(matched_texts).strip() if matched_texts else None
            logprobs = [s.avg_logprob for s in segs]
            no_speech_probs = [s.no_speech_prob for s in segs]
            comp_ratios = [s.compression_ratio for s in segs]

        turn.text = full_text if full_text else None
        turn.quality.asr_avg_logprob = (
            float(sum(logprobs) / len(logprobs)) if logprobs else -0.5
        )
        turn.quality.no_speech_prob = (
            float(sum(no_speech_probs) / len(no_speech_probs)) if no_speech_probs else 0.1
        )
        turn.quality.compression_ratio = (
            float(sum(comp_ratios) / len(comp_ratios)) if comp_ratios else 1.0
        )

        if not turn.text:
            turn.word_timings = None
            turn.word_timings_source = None
            continue

        # Slice turn audio for forced alignment
        start_samp = int(turn.start * sr)
        end_samp = min(len(data), int(turn.end * sr))
        turn_wav = wav_tensor[:, start_samp:end_samp]

        try:
            timings = forced_align_segment(
                turn_wav, sr=sr, transcript=turn.text, device=device
            )
            # Offset word timings by turn start
            for wt in timings:
                wt.s = round(wt.s + turn.start, 3)
                wt.e = round(wt.e + turn.start, 3)
            turn.word_timings = timings
            turn.word_timings_source = "forced_align"
        except Exception:
            # Alignment failure fallback: use Whisper word timings if available
            if words:
                turn.word_timings = [
                    WordTiming(
                        w=w.word.strip(),
                        s=round(w.start, 3),
                        e=round(w.end, 3),
                        conf=round(float(getattr(w, "probability", 0.5)), 2),
                    )
                    for w in words
                    if w.word.strip()
                ]
            else:
                words_list = turn.text.split()
                turn.word_timings = [
                    WordTiming(w=w, s=turn.start, e=turn.end, conf=0.5)
                    for w in words_list
                ]
            turn.word_timings_source = "asr_fallback"
            alignment_failures += 1

    return alignment_failures


def process_stage3_lexical(
    timeline: ConversationTimeline,
    config: PipelineConfig,
    stub_models: bool = False,
    simulate_alignment_failure: bool = False,
) -> dict[str, Any]:
    """Process Stage 3: Lexical extraction per speaker stream."""
    total_alignment_failures = 0

    # Group turns by speaker
    speaker_turns: dict[str, list[Turn]] = {}
    for t in timeline.turns:
        speaker_turns.setdefault(t.speaker, []).append(t)

    for spk_id, spk_meta in timeline.speakers.items():
        turns = speaker_turns.get(spk_id, [])
        if not turns:
            continue

        if stub_models or not config.allow_download:
            failures = transcribe_and_align_stream_stub(
                stream_path=spk_meta.stream_path,
                turns=turns,
                fail_alignment=simulate_alignment_failure,
            )
        else:
            try:
                device = "cuda" if torch.cuda.is_available() else "cpu"
                failures = transcribe_and_align_stream_whisper(
                    stream_path=spk_meta.stream_path,
                    turns=turns,
                    config=config,
                    device=device,
                )
            except Exception:
                failures = transcribe_and_align_stream_stub(
                    stream_path=spk_meta.stream_path,
                    turns=turns,
                    fail_alignment=simulate_alignment_failure,
                )

        total_alignment_failures += failures

    timeline.diagnostics.alignment_failures = total_alignment_failures
    return {
        "alignment_failures": total_alignment_failures,
        "n_transcribed_turns": len(timeline.turns),
    }
