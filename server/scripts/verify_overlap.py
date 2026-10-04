import os
import sys
import torch
from pyannote.audio import Pipeline

# Fix FFmpeg for Windows DLL loading
FFMPEG_BIN_PATH = r"C:\Users\Anvay\AppData\Local\Microsoft\WinGet\Packages\Gyan.FFmpeg.Shared_Microsoft.Winget.Source_8wekyb3d8bbwe\ffmpeg-9.0.2-full_build-shared\bin"
if os.name == "nt" and os.path.exists(FFMPEG_BIN_PATH):
    try:
        os.add_dll_directory(FFMPEG_BIN_PATH)
    except Exception as e:
        print(f"[FFmpeg] DLL warning: {e}")


def main():
    AUDIO_FILE = r"C:\Users\Anvay\Downloads\test.wav"
    HF_TOKEN = os.getenv("HF_TOKEN", "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")

    if not os.path.exists(AUDIO_FILE):
        print(f"[Error] Audio file not found at {AUDIO_FILE}")
        sys.exit(1)

    print("=" * 65)
    print(" PYANNOTE OVERLAPPING SPEECH VERIFICATION TEST")
    print("=" * 65)
    print(f"[*] Target Audio File : {AUDIO_FILE}")
    print("[*] Device            : CUDA (NVIDIA GPU)")
    print("[*] Loading Pipeline  : pyannote/speaker-diarization-community-1 ...")

    pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-community-1", token=HF_TOKEN)

    if torch.cuda.is_available():
        pipeline.to(torch.device("cuda"))

    print("\n[*] Running Pyannote Diarization on audio...")
    output = pipeline(AUDIO_FILE)

    # 1. Inspect the standard Diarization output
    diarization = output.speaker_diarization
    print(f"\n[+] Total speaker turns detected: {len(diarization)}")

    # 2. Extract overlapping speech segments natively detected by Pyannote
    overlap_timeline = diarization.get_overlap()
    overlap_segments = list(overlap_timeline)

    print("\n" + "=" * 65)
    print(" OVERLAP DETECTION RESULTS")
    print("=" * 65)

    if not overlap_segments:
        print("[-] No overlapping speech detected in this specific file.")
    else:
        print(
            f"[+] PROOF CONFIRMED: Pyannote detected {len(overlap_segments)} overlapping segments!"
        )
        print("-" * 65)
        print(f"{'Start Time':<12} | {'End Time':<12} | {'Duration':<10} | {'Active Speakers'}")
        print("-" * 65)

        for i, segment in enumerate(overlap_segments, 1):
            # Find which speakers are active during this overlap window
            active_speakers = set()
            for turn, track, speaker in diarization.itertracks(yield_label=True):
                # Check for temporal overlap between speaker turn and the overlap window
                if max(turn.start, segment.start) < min(turn.end, segment.end):
                    active_speakers.add(speaker)

            speaker_str = ", ".join(sorted(active_speakers))
            print(
                f"{segment.start:06.2f}s     | {segment.end:06.2f}s     | {segment.duration:05.2f}s    | {speaker_str}"
            )

    # 3. Compare speaker_diarization vs exclusive_speaker_diarization
    exclusive_diarization = output.exclusive_speaker_diarization
    total_speech_diar = sum(
        turn.duration for turn, track, _ in diarization.itertracks(yield_label=True)
    )
    total_speech_excl = sum(
        turn.duration for turn, track, _ in exclusive_diarization.itertracks(yield_label=True)
    )
    total_overlap_duration = sum(s.duration for s in overlap_segments)

    print("\n" + "=" * 65)
    print(" ARCHITECTURAL PROOF: DIARIZATION vs EXCLUSIVE DIARIZATION")
    print("=" * 65)
    print(f"Total Speech Time (with overlap)    : {total_speech_diar:.2f} seconds")
    print(f"Total Speech Time (exclusive mono)  : {total_speech_excl:.2f} seconds")
    print(f"Calculated Overlap Speech Duration : {total_overlap_duration:.2f} seconds")
    print(
        f"Difference (Speech in Overlap)      : {total_speech_diar - total_speech_excl:.2f} seconds"
    )
    print("=" * 65)


if __name__ == "__main__":
    main()
