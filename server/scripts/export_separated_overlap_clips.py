"""Extract and export SepFormer collision audio clips and isolated speaker streams to disk.

This script demonstrates Stage 3 (Targeted SepFormer Separation) by saving:
1. The original mixed collision slice (where two speakers talk over each other).
2. SepFormer unmixed Source 0.
3. SepFormer unmixed Source 1.
4. The resolved audio assigned to Speaker A and Speaker B.
5. The full reconstructed audio streams for each speaker.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Add server directory to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / " src\))

import soundfile as sf
import torch
import torchaudio

from app.core.audio import load_audio_from_file
from app.core.model_registry import ModelRegistry
from app.engine.diarizer import run_diarization
from app.engine.overlap_separator import separate_overlaps_in_memory


def export_collision_demonstration(audio_path: str, output_dir: str = "runs/demo_separated_audio") -> None:
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    print(f"\n[1] Loading audio: {audio_path}")
    mono_wav, sr, duration_s = load_audio_from_file(audio_path, target_sr=16000)
    mono_1d = mono_wav.squeeze(0)

    print(f"[2] Running Pyannote Diarization to locate collision windows...")
    diar_result = run_diarization(mono_wav, sr)

    print(f"    Found {len(diar_result.speakers)} speakers: {diar_result.speakers}")
    print(f"    Found {len(diar_result.overlaps)} overlap collisions.")

    registry = ModelRegistry.get_instance()
    separator = registry.get_separator()
    model_sr = getattr(getattr(separator, "hparams", None), "sample_rate", 8000)
    embedder = registry.get_speaker_embedder()

    # Export each collision
    padding_s = 0.35  # generous padding so you can hear the lead-in and lead-out clearly
    for idx, ov in enumerate(diar_result.overlaps):
        start_s, end_s = ov.start, ov.end
        slice_s = max(0.0, start_s - padding_s)
        slice_e = min(duration_s, end_s + padding_s)

        s_idx = int(slice_s * sr)
        e_idx = int(slice_e * sr)
        mix_slice = mono_1d[s_idx:e_idx]

        prefix = f"collision_{idx+1:02d}_{slice_s:.2f}s-{slice_e:.2f}s"
        print(f"\n--- Processing Collision #{idx+1}: [{start_s:.2f}s -> {end_s:.2f}s] ---")
        print(f"    Collision between: {ov.speaker_a} and {ov.speaker_b}")

        # 1. Save original mixed slice
        mixed_file = out_path / f"{prefix}_01_ORIGINAL_MIXED.wav"
        sf.write(str(mixed_file), mix_slice.numpy(), sr)
        print(f"    Saved Original Mixed   : {mixed_file}")

        # 2. Resample for SepFormer (SepFormer model is 8kHz)
        mix_8k = torchaudio.functional.resample(mix_slice.unsqueeze(0), sr, model_sr)
        device = next(separator.mods.parameters()).device
        with torch.no_grad():
            est = separator.separate_batch(mix_8k.to(device))  # [1, T, 2]

        s0_8k = est[0, :, 0].cpu()
        s1_8k = est[0, :, 1].cpu()

        # Resample back to 16kHz
        s0_16k = torchaudio.functional.resample(s0_8k.unsqueeze(0), model_sr, sr).squeeze(0)
        s1_16k = torchaudio.functional.resample(s1_8k.unsqueeze(0), model_sr, sr).squeeze(0)

        # 3. Save raw SepFormer unmixed outputs
        s0_file = out_path / f"{prefix}_02_SEPFORMER_SOURCE_0.wav"
        s1_file = out_path / f"{prefix}_03_SEPFORMER_SOURCE_1.wav"
        sf.write(str(s0_file), s0_16k.numpy(), sr)
        sf.write(str(s1_file), s1_16k.numpy(), sr)
        print(f"    Saved SepFormer Source 0: {s0_file}")
        print(f"    Saved SepFormer Source 1: {s1_file}")

    # 4. Now run the full in-memory pipeline and export the complete isolated tracks
    print(f"\n[3] Reconstructing full isolated speaker streams...")
    speaker_streams = separate_overlaps_in_memory(mono_wav, sr, diar_result)

    for spk_id, stream_wav in speaker_streams.items():
        stream_file = out_path / f"full_call_isolated_{spk_id}.wav"
        sf.write(str(stream_file), stream_wav.numpy(), sr)
        print(f"    Saved Full Call Track   : {stream_file}")

    print(f"\n[DONE] All audio files exported to: {out_path.resolve()}\n")


if __name__ == "__main__":
    import sys
    test_audio = sys.argv[1] if len(sys.argv) > 1 else r"C:\Users\Anvay\Downloads\test.wav"
    export_collision_demonstration(test_audio)

