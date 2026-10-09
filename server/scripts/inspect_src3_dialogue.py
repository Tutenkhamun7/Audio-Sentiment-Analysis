#!/usr/bin/env python3
"""CLI utility to test, inspect, and evaluate turn-by-turn dialogue & transcription using src3.

Usage:
    # Run in-process on default test.wav
    python scripts/inspect_src3_dialogue.py

    # Run on a custom audio file
    python scripts/inspect_src3_dialogue.py path/to/call.wav

    # Show word-level timestamps & probabilities
    python scripts/inspect_src3_dialogue.py --show-words

    # Slice and export individual turn audio clips to listen to
    python scripts/inspect_src3_dialogue.py --save-clips ./output_clips

    # Test via the running FastAPI backend
    python scripts/inspect_src3_dialogue.py --api http://localhost:8000/api/v1/analyze

    # Map speaker IDs to names
    python scripts/inspect_src3_dialogue.py --humanize
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

# Ensure stdout and stderr support UTF-8 on Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# Add 'server' directory to sys.path so 'src3' imports resolve smoothly
script_dir = Path(__file__).resolve().parent
server_dir = script_dir.parent
if str(server_dir) not in sys.path:
    sys.path.insert(0, str(server_dir))

# DEFAULT_AUDIO = Path("C:/Users/Anvay/Downloads/test.wav")
DEFAULT_AUDIO = Path("C:/Users/Anvay/Downloads/F_0101_10y4m_1.wav")


def format_timestamp(seconds: float) -> str:
    """Format seconds into MM:SS.ms string."""
    m = int(seconds // 60)
    s = seconds % 60
    return f"{m:02d}:{s:05.2f}"


def run_via_api(audio_path: Path, api_url: str, options_dict: Dict[str, Any]) -> Dict[str, Any]:
    """Send audio to running FastAPI backend."""
    import httpx

    print(f"\n[API] Sending {audio_path.name} to {api_url}...")
    with open(audio_path, "rb") as f:
        files = {"file": (audio_path.name, f, "audio/wav")}
        response = httpx.post(api_url, files=files, data=options_dict, timeout=300.0)

    if response.status_code != 200:
        print(f"[ERROR] API Error ({response.status_code}): {response.text}")
        sys.exit(1)

    return response.json()


def run_in_process(audio_path: Path, options_dict: Dict[str, Any]) -> Dict[str, Any]:
    """Run src3 pipeline directly in-process without needing a server."""
    from src3.core.audio import load_audio_from_file
    from src3.schemas.request import AnalyzeOptions
    from src3.services.orchestrator import AudioAnalysisOrchestrator

    print(f"\n[*] Running in-process src3 pipeline on: {audio_path}...")
    waveform, sr, duration_s = load_audio_from_file(str(audio_path))
    opts = AnalyzeOptions(**options_dict)
    orchestrator = AudioAnalysisOrchestrator()
    res = orchestrator.analyze(
        waveform=waveform,
        sr=sr,
        options=opts,
        call_id=audio_path.name,
    )
    return res.model_dump()


def save_turn_audio_clips(audio_path: Path, turns: List[Dict[str, Any]], output_dir: Path) -> None:
    """Slice turn audio segments and save as separate WAV clips for listening."""
    import soundfile as sf
    from src3.core.audio import load_audio_from_file

    output_dir.mkdir(parents=True, exist_ok=True)
    waveform, sr, _ = load_audio_from_file(str(audio_path))
    total_samples = waveform.shape[-1]

    print(f"\n[SAVE] Exporting {len(turns)} turn audio clips to: {output_dir.resolve()}")
    for idx, t in enumerate(turns):
        turn_id = t.get("turn_id", f"turn_{idx:03d}")
        spk = t.get("speaker", "SPK")
        s_sec = t.get("start", 0.0)
        e_sec = t.get("end", 0.0)

        s_sample = max(0, int(s_sec * sr))
        e_sample = min(total_samples, int(e_sec * sr))
        clip_data = waveform[0, s_sample:e_sample].cpu().numpy()

        clip_filename = f"{idx:02d}_{turn_id}_{spk}_{s_sec:.1f}s-{e_sec:.1f}s.wav"
        clip_path = output_dir / clip_filename
        sf.write(str(clip_path), clip_data, sr)

    print(f"[OK] Successfully exported all turn clips.")


def print_dialogue_transcript(
    result: Dict[str, Any],
    humanize: bool = False,
    speaker_map: Optional[Dict[str, str]] = None,
    show_words: bool = False,
) -> None:
    """Print beautifully formatted turn-by-turn dialogue and acoustic metadata."""
    metrics = result.get("metrics", {})
    turns = result.get("turns", [])
    call_id = result.get("call_id", "Audio")

    # Speaker mapping
    spk_names: Dict[str, str] = {}
    if speaker_map:
        spk_names.update(speaker_map)
    elif humanize:
        spk_names = {
            "SPEAKER_00": "SPEAKER_00",
            "SPEAKER_01": "SPEAKER_01",
        }

    # Color codes
    CYAN = "\033[96m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    MAGENTA = "\033[95m"
    RED = "\033[91m"
    BLUE = "\033[94m"
    BOLD = "\033[1m"
    DIM = "\033[2m"
    RESET = "\033[0m"

    print("\n" + "=" * 80)
    print(f"{BOLD}[ANALYSIS REPORT] {call_id}{RESET}")
    print("=" * 80)
    dur = metrics.get("duration_s", 0.0)
    sp_dur = metrics.get("total_speech_s", 0.0)
    ov_dur = metrics.get("overlap_s", 0.0)
    ov_ratio = metrics.get("overlap_ratio", 0.0) * 100
    spk_cnt = metrics.get("speaker_count", 0)
    branch = metrics.get("branch_used", "unknown")
    interrupts = metrics.get("interruption_count", 0)

    print(
        f"Duration: {BOLD}{dur:.2f}s{RESET} | "
        f"Speech: {sp_dur:.2f}s ({sp_dur/max(0.1, dur)*100:.1f}%) | "
        f"Speakers: {spk_cnt} | "
        f"Branch: {CYAN}{branch}{RESET}"
    )
    print(
        f"Overlaps: {ov_dur:.2f}s ({ov_ratio:.1f}%) | "
        f"Interruptions: {interrupts} | "
        f"Total Turns: {len(turns)}"
    )
    print("-" * 105)
    print(f"{BOLD}{'TIME':<18} | {'SPEAKER':<12} | {'SENTIMENT':<16} | {'AFFECT (V,A)':<18} | {'TRANSCRIPT'}{RESET}")
    # print(f"{BOLD}{'TIME':<18} | {'SPEAKER':<12} | {'TRANSCRIPT'}{RESET}")
    print("-" * 105)

    for idx, t in enumerate(turns):
        s_time = t.get("start", 0.0)
        e_time = t.get("end", 0.0)
        t_dur = t.get("duration", e_time - s_time)
        spk_id = t.get("speaker", "UNKNOWN")
        spk_label = spk_names.get(spk_id, spk_id)

        # Speaker color
        spk_color = GREEN if spk_id.endswith("00") else CYAN

        # Textual sentiment
        sent = t.get("sentiment", {})
        s_lbl = sent.get("label", "neutral")
        s_sc = sent.get("score", 1.0)
        sent_str = f"{s_lbl} ({s_sc:.2f})"
        sent_color = (
            GREEN if s_lbl == "positive"
            else (RED if s_lbl == "negative" else DIM)
        )

        # Acoustic emotion & coordinates
        emo = t.get("emotion", {})
        dom_emo = emo.get("dominant_emotion") or "neutral"
        val = emo.get("valence")
        aro = emo.get("arousal")
        if val is not None and aro is not None and not emo.get("abstained"):
            aff_str = f"{dom_emo} ({val:+.2f},{aro:+.2f})"
        elif emo.get("abstained"):
            reason = emo.get("abstain_reason", "abstained")
            aff_str = f"[{reason}]"
        else:
            aff_str = dom_emo

        emo_color = (
            GREEN
            if dom_emo in ("positive", "happy")
            else (RED if dom_emo in ("negative", "angry", "sad") else DIM)
        )

        time_str = f"[{format_timestamp(s_time)} → {format_timestamp(e_time)}]"

        # Interruption markers
        flags = []
        if t.get("is_interruption"):
            flags.append(f"{RED}[INTERRUPTS]{RESET}")
        if t.get("interrupted_by"):
            flags.append(f"{YELLOW}[INTERRUPTED]{RESET}")

        flag_str = " " + " ".join(flags) if flags else ""
        text = t.get("text", "") or f"{DIM}(no speech detected){RESET}"

        print(
            f"{DIM}{time_str:<18}{RESET} | "
            f"{spk_color}{BOLD}{spk_label:<12}{RESET} | "
            f"{sent_color}{sent_str:<16}{RESET} | "
            f"{emo_color}{aff_str:<18}{RESET} | "
            f"{text}{flag_str}"
        )

        # Word-level breakdown
        if show_words and t.get("words"):
            words_list = t.get("words", [])
            print(f"   {DIM}└─ Words ({len(words_list)}):{RESET}", end=" ")
            words_formatted = [
                f"{w.get('word')}({w.get('start', 0.0):.2f}-{w.get('end', 0.0):.2f}, conf={w.get('confidence', 0.0):.2f})"
                for w in words_list
            ]
            print(f"{DIM}{' '.join(words_formatted)}{RESET}")

    print("=" * 80 + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Inspect turn-by-turn speaker output and transcription using src3."
    )
    parser.add_argument(
        "audio",
        nargs="?",
        default=str(DEFAULT_AUDIO) if DEFAULT_AUDIO.exists() else None,
        help=f"Path to input WAV audio file (default: {DEFAULT_AUDIO})",
    )
    parser.add_argument(
        "--api",
        type=str,
        default=None,
        help="Target FastAPI analyze endpoint URL (e.g. http://localhost:8000/api/v1/analyze). If omitted, runs in-process.",
    )
    parser.add_argument(
        "--humanize",
        action="store_true",
        help="Label SPEAKER_00 as Agent and SPEAKER_01 as Customer.",
    )
    parser.add_argument(
        "--show-words",
        action="store_true",
        help="Display word-level timestamps and acoustic probabilities.",
    )
    parser.add_argument(
        "--save-clips",
        type=str,
        default=None,
        metavar="DIR",
        help="Slice each turn and save individual WAV clips to DIR for listening.",
    )
    parser.add_argument(
        "--output-json",
        type=str,
        default=None,
        metavar="FILE",
        help="Save raw analysis result JSON to FILE.",
    )
    parser.add_argument(
        "--no-separation",
        action="store_true",
        help="Disable SepFormer overlap separation.",
    )
    parser.add_argument(
        "--no-split",
        action="store_true",
        help="Force no_split branch (entire audio as single stream).",
    )
    parser.add_argument(
        "--speakers",
        type=int,
        default=None,
        help="Explicit speaker count if known in advance (e.g. 2).",
    )

    args = parser.parse_args()

    if not args.audio:
        print("❌ Error: No audio file provided. Usage: python inspect_src3_dialogue.py path/to/audio.wav")
        sys.exit(1)

    audio_path = Path(args.audio)
    if not audio_path.exists():
        print(f"❌ Error: Audio file not found at: {audio_path.resolve()}")
        sys.exit(1)

    # Options dictionary matching AnalyzeOptions schema
    options_dict: Dict[str, Any] = {
        "enable_overlap_separation": not args.no_separation,
        "force_branch": "no_split" if args.no_split else None,
        "num_speakers": args.speakers,
    }

    # Execute
    if args.api:
        result = run_via_api(audio_path, args.api, options_dict)
    else:
        result = run_in_process(audio_path, options_dict)

    # Print dialogue
    print_dialogue_transcript(
        result,
        humanize=args.humanize,
        show_words=args.show_words,
    )

    # Export clips if requested
    if args.save_clips:
        save_turn_audio_clips(audio_path, result.get("turns", []), Path(args.save_clips))

    # Save JSON if requested
    if args.output_json:
        out_p = Path(args.output_json)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        with open(out_p, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)
        print(f"📄 Raw JSON saved to: {out_p.resolve()}")


if __name__ == "__main__":
    main()
