"""Inspect Stage 5 (Faster-Whisper Transcription) output on test.wav.

This script executes Stages 1-4 to obtain the SepFormer-unmixed speaker audio,
then runs Stage 5 (transcribe_speaker_turns) and prints its exact data structures:
1. Turn-by-turn transcribed text and timestamps.
2. Word-level timings and confidence scores.
3. The exact overlap collision window (e.g. 29s - 32s) showing how both
   overlapping turns are preserved side-by-side.
4. The raw object/JSON representation of a Stage 5 RawTurn.
"""

from __future__ import annotations

import json
from pathlib import Path
import pprint
import sys
import torch

# Add server directory to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / \src\))

from app.core.audio import load_audio_from_file
from app.core.config import settings
from app.engine.diarizer import run_diarization
from app.engine.overlap_separator import separate_overlaps_in_memory
from app.engine.transcriber import transcribe_speaker_turns


def run_stage5_inspection(audio_path: str) -> None:
    print(f"\n{'='*85}")
    print(f" STAGE 5 (FASTER-WHISPER TRANSCRIPTION) INSPECTION ON: {Path(audio_path).name}")
    print(f"{'='*85}")

    # 1. Load Audio
    print("\n[*] Loading audio into memory...")
    mono_wav, sr, duration_s = load_audio_from_file(audio_path, target_sr=16000)
    print(f"    Duration: {duration_s:.2f}s | Sample Rate: {sr}Hz | Shape: {mono_wav.shape}")

    # 2. Stage 2: Diarization
    print("\n[*] Running Diarization (Pyannote)...")
    diar_result = run_diarization(mono_wav, sr)
    print(f"    Speakers detected: {diar_result.speakers}")
    print(f"    Initial raw diarization turns: {len(diar_result.turns)}")
    print(f"    Overlap collision windows: {len(diar_result.overlaps)}")

    # 3. Stage 3 & 4: SepFormer Overlap Separation
    print("\n[*] Running Stage 3 & 4: Targeted SepFormer Separation & Neural Permutation...")
    speaker_wavs = separate_overlaps_in_memory(
        mono=mono_wav,
        sr=sr,
        diarization=diar_result,
        overlap_padding_s=settings.overlap_padding_s,
    )
    print(f"    Speaker streams created: {list(speaker_wavs.keys())}")
    for spk, w in speaker_wavs.items():
        print(f"      - {spk}: {w.shape[-1]} samples ({w.shape[-1]/sr:.2f}s audio)")

    # 4. STAGE 5: FASTER-WHISPER TRANSCRIPTION
    print("\n" + "#"*85)
    print(" >>> EXECUTING STAGE 5: transcribe_speaker_turns() <<<")
    print("#"*85)

    all_turns = list(diar_result.turns)
    total_samples = int(duration_s * sr)
    for spk in diar_result.speakers:
        spk_audio = speaker_wavs.get(spk, torch.zeros(total_samples))
        spk_turns = [t for t in all_turns if t.speaker == spk]
        print(f"[*] Transcribing {spk} ({len(spk_turns)} turns) from its unmixed track...")
        transcribe_speaker_turns(
            speaker_wav=spk_audio,
            turns=spk_turns,
            sr=sr,
            vad_filter=settings.vad_filter,
            initial_prompt=settings.whisper_initial_prompt,
            collar_s=settings.asr_collar_s,
            all_turns=all_turns,
            min_word_prob=settings.min_word_prob,
        )

    all_turns.sort(key=lambda t: t.start)
    stage5_turns = all_turns

    print(f"\n[+] Stage 5 Completed! Returned {len(stage5_turns)} transcribed turns.\n")

    # Display Stage 5 Turn-by-Turn Output
    print(f"{'INDEX':<6} | {'SPEAKER':<12} | {'TIMESTAMPS':<22} | {'WORDS':<6} | {'TRANSCRIBED TEXT'}")
    print("-" * 85)

    for idx, t in enumerate(stage5_turns):
        t_start = round(t.start, 2)
        t_end = round(t.end, 2)
        time_str = f"[{t_start:06.2f}s -> {t_end:06.2f}s]"
        num_words = len(t.words)
        text_preview = t.text if t.text else "(no speech)"
        if len(text_preview) > 55:
            text_preview = text_preview[:52] + "..."

        print(f"#{idx:02d}   | {t.speaker:<12} | {time_str:<22} | {num_words:<6} | {text_preview}")

    # Zoom in on the Overlap Window (Turns around 30s)
    print("\n" + "="*85)
    print(" ZOOM IN: OVERLAPPING COLLISION WINDOW (AROUND 30 SECONDS)")
    print(" Notice how both turns exist with their own timestamps and independent words!")
    print("="*85)

    for idx, t in enumerate(stage5_turns):
        # Look for turns near the 29s - 33s mark
        if 28.0 <= t.start <= 33.0 or 28.0 <= t.end <= 33.0:
            print(f"\nTurn #{idx} ({t.speaker}): [{t.start:.2f}s -> {t.end:.2f}s] (Duration: {t.duration:.2f}s)")
            print(f"  Full Text: \"{t.text}\"")
            print("  Word Timings from Stage 5:")
            for w in t.words:
                print(f"    - \"{w.word}\" : [{w.start:.2f}s -> {w.end:.2f}s] (conf: {w.confidence:.2f})")

    # Display raw data structure of a sample turn
    print("\n" + "="*85)
    print(" RAW DATA STRUCTURE RETURNED BY STAGE 5 (Sample Turn #0):")
    print("="*85)
    sample_turn = stage5_turns[0]
    sample_dict = {
        "speaker": sample_turn.speaker,
        "start": sample_turn.start,
        "end": sample_turn.end,
        "duration": sample_turn.duration,
        "text": sample_turn.text,
        "words": [w.model_dump() for w in sample_turn.words],
    }
    print(json.dumps(sample_dict, indent=2))
    print("\n" + "="*85 + "\n")


if __name__ == "__main__":
    test_path = sys.argv[1] if len(sys.argv) > 1 else r"C:\Users\Anvay\Downloads\test.wav"
    run_stage5_inspection(test_path)

