import os
import json
import subprocess
import torch
from faster_whisper import WhisperModel
from pyannote.audio import Pipeline

# 1. FIX FFMPEG FOR PYTORCH ON WINDOWS
FFMPEG_BIN_PATH = r"C:\Users\Anvay\AppData\Local\Microsoft\WinGet\Packages\Gyan.FFmpeg.Shared_Microsoft.Winget.Source_8wekyb3d8bbwe\ffmpeg-9.0.2-full_build-shared\bin"
if os.name == "nt" and os.path.exists(FFMPEG_BIN_PATH):
    try:
        os.add_dll_directory(FFMPEG_BIN_PATH)
        print(f"[System] Registered FFmpeg DLLs from {FFMPEG_BIN_PATH}")
    except Exception as e:
        print(f"[System] Failed to register FFmpeg DLLs: {e}")


class AdvancedASREngine:
    def __init__(
        self, hf_token: str, whisper_model_size: str = "large-v3-turbo", device: str = "cuda"
    ):
        self.device = device

        # Load Faster-Whisper
        print(f"[Init] Loading Faster-Whisper ({whisper_model_size})...")
        self.whisper = WhisperModel(whisper_model_size, device=self.device, compute_type="float16")

        # Load Pyannote (Only needed for mono fallback)
        print("[Init] Loading Pyannote Diarization...")
        self.pyannote = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-community-1", token=hf_token
        )
        if self.pyannote and self.device == "cuda" and torch.cuda.is_available():
            self.pyannote.to(torch.device("cuda"))

    def _get_channel_count(self, file_path: str) -> int:
        """Probes the audio file to detect if it is mono (1) or stereo (2+)."""
        cmd = [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "a:0",
            "-show_entries",
            "stream=channels",
            "-of",
            "json",
            file_path,
        ]
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            info = json.loads(result.stdout)
            return int(info["streams"][0]["channels"])
        except Exception as e:
            print(f"[Probe Error] Could not determine channels, defaulting to Mono. {e}")
            return 1

    def _process_stereo(self, file_path: str, language: str = "en") -> list:
        """Splits stereo channels cleanly and transcribes each channel with full acoustic context."""
        print("[Engine] Stereo detected. Bypassing Pyannote. Splitting channels via FFmpeg...")

        agent_wav = "temp_agent.wav"
        customer_wav = "temp_customer.wav"

        # Split Left and Right channels using channelsplit filter (compatible with modern FFmpeg)
        cmd = [
            "ffmpeg",
            "-y",
            "-i",
            file_path,
            "-filter_complex",
            "[0:a]channelsplit=channel_layout=stereo[left][right]",
            "-map",
            "[left]",
            agent_wav,
            "-map",
            "[right]",
            customer_wav,
        ]
        try:
            subprocess.run(cmd, capture_output=True, check=True)
        except subprocess.CalledProcessError:
            # Fallback for older FFmpeg mapping syntax
            subprocess.run(
                [
                    "ffmpeg",
                    "-y",
                    "-i",
                    file_path,
                    "-map_channel",
                    "0.0.0",
                    agent_wav,
                    "-map_channel",
                    "0.0.1",
                    customer_wav,
                ],
                capture_output=True,
                check=True,
            )

        final_segments = []

        # Transcribe Left Channel (Agent)
        segments, _ = self.whisper.transcribe(
            agent_wav,
            language=language,
            beam_size=5,
            condition_on_previous_text=False,
            vad_filter=True,
            vad_parameters=dict(min_silence_duration_ms=500),
        )
        for seg in segments:
            text = seg.text.strip()
            if text:
                final_segments.append(
                    {
                        "speaker": "SPEAKER_00",
                        "start_time": seg.start,
                        "end_time": seg.end,
                        "text": text,
                    }
                )

        # Transcribe Right Channel (Customer)
        segments, _ = self.whisper.transcribe(
            customer_wav,
            language=language,
            beam_size=5,
            condition_on_previous_text=False,
            vad_filter=True,
            vad_parameters=dict(min_silence_duration_ms=500),
        )
        for seg in segments:
            text = seg.text.strip()
            if text:
                final_segments.append(
                    {
                        "speaker": "SPEAKER_01",
                        "start_time": seg.start,
                        "end_time": seg.end,
                        "text": text,
                    }
                )

        # Cleanup temp files
        if os.path.exists(agent_wav):
            os.remove(agent_wav)
        if os.path.exists(customer_wav):
            os.remove(customer_wav)

        # Sort chronologically
        return sorted(final_segments, key=lambda x: x["start_time"])

    def _assign_speaker(self, word_start: float, word_end: float, intervals: list) -> str:
        """Determines the active speaker for a word using temporal interval overlap."""
        if not intervals:
            return "SPEAKER_00"

        # 1. Overlap duration with each speaker
        speaker_overlaps = {}
        for inv in intervals:
            overlap = max(0.0, min(word_end, inv["end"]) - max(word_start, inv["start"]))
            if overlap > 0.0:
                speaker_overlaps[inv["speaker"]] = (
                    speaker_overlaps.get(inv["speaker"], 0.0) + overlap
                )

        if speaker_overlaps:
            return max(speaker_overlaps.items(), key=lambda x: x[1])[0]

        # 2. Nearest interval by midpoint if word falls in a micro-gap
        midpoint = (word_start + word_end) / 2.0
        closest = min(
            intervals,
            key=lambda x: min(abs(midpoint - x["start"]), abs(midpoint - x["end"])),
        )
        return closest["speaker"]

    def _process_mono(self, file_path: str, language: str = "en") -> list:
        """
        Transcribes full continuous audio with word timestamps to preserve Whisper's 30s context,
        then aligns word timestamps with Pyannote speaker intervals by temporal overlap.
        """
        print("[Engine] Mono detected. Transcribing continuous audio with Whisper...")

        # 1. Transcribe the whole audio continuously (avoids chopping phonemes / losing context)
        segments_gen, _ = self.whisper.transcribe(
            file_path,
            language=language,
            word_timestamps=True,
            beam_size=5,
            condition_on_previous_text=False,
            vad_filter=True,
            vad_parameters=dict(min_silence_duration_ms=500),
            no_speech_threshold=0.6,
            compression_ratio_threshold=2.4,
        )

        words = []
        for seg in segments_gen:
            if seg.words:
                for w in seg.words:
                    word_clean = w.word.strip()
                    if word_clean:
                        words.append({"start": w.start, "end": w.end, "word": word_clean})

        if not words:
            print("[Engine] No words detected.")
            return []

        # 2. Run Pyannote on the continuous audio for speaker boundaries
        print("[Engine] Running Pyannote Diarization...")
        intervals = []
        if self.pyannote:
            diarization = self.pyannote(file_path)
            for turn, speaker in diarization.speaker_diarization:
                intervals.append({"speaker": speaker, "start": turn.start, "end": turn.end})

        # 3. Associate each word with its speaker via temporal overlap
        aligned_words = []
        for w in words:
            speaker = self._assign_speaker(w["start"], w["end"], intervals)
            aligned_words.append({**w, "speaker": speaker})

        # 4. Smooth isolated single-word jitter (e.g. A -> B -> A for 1 word)
        if len(aligned_words) >= 3:
            for i in range(1, len(aligned_words) - 1):
                prev_w = aligned_words[i - 1]
                curr_w = aligned_words[i]
                next_w = aligned_words[i + 1]
                if (
                    prev_w["speaker"] == next_w["speaker"]
                    and curr_w["speaker"] != prev_w["speaker"]
                ):
                    gap_before = curr_w["start"] - prev_w["end"]
                    gap_after = next_w["start"] - curr_w["end"]
                    if (
                        gap_before <= 0.6
                        and gap_after <= 0.6
                        and (curr_w["end"] - curr_w["start"]) <= 0.8
                    ):
                        curr_w["speaker"] = prev_w["speaker"]

        # 5. Group into natural conversational turns
        final_segments = []
        current_turn = None
        terminal_punctuation = {".", "!", "?"}

        for w in aligned_words:
            if not current_turn:
                current_turn = {
                    "speaker": w["speaker"],
                    "start_time": round(w["start"], 2),
                    "end_time": round(w["end"], 2),
                    "text": w["word"],
                }
                continue

            same_speaker = w["speaker"] == current_turn["speaker"]
            word_gap = w["start"] - current_turn["end_time"]
            turn_duration = current_turn["end_time"] - current_turn["start_time"]
            prev_ended_sentence = (
                current_turn["text"] and current_turn["text"][-1] in terminal_punctuation
            )

            # Split turn if speaker changed, or natural pause after sentence, or exceeded max duration (6s)
            should_split = (
                not same_speaker
                or (prev_ended_sentence and turn_duration >= 2.0)
                or (word_gap >= 0.8 and turn_duration >= 2.0)
                or (turn_duration >= 6.0)
            )

            if should_split:
                final_segments.append(current_turn)
                current_turn = {
                    "speaker": w["speaker"],
                    "start_time": round(w["start"], 2),
                    "end_time": round(w["end"], 2),
                    "text": w["word"],
                }
            else:
                current_turn["end_time"] = round(w["end"], 2)
                current_turn["text"] += f" {w['word']}"

        if current_turn:
            final_segments.append(current_turn)

        return final_segments

    def transcribe_file(self, file_path: str, language: str = "en") -> list:
        """Main routing function."""
        channels = self._get_channel_count(file_path)

        if channels >= 2:
            return self._process_stereo(file_path, language=language)
        else:
            return self._process_mono(file_path, language=language)


# ==========================================
# TEST EXECUTION
# ==========================================
if __name__ == "__main__":
    # Load from environment or fallback
    HF_TOKEN = os.getenv("HF_TOKEN", "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")
    AUDIO_FILE = r"C:\Users\Anvay\Downloads\test.wav"

    engine = AdvancedASREngine(hf_token=HF_TOKEN)

    print("\n--- Starting Processing ---")
    results = engine.transcribe_file(AUDIO_FILE)

    print("\n--- Final Clean Output ---")
    for r in results:
        print(f"[{r['start_time']:05.2f}s - {r['end_time']:05.2f}s] {r['speaker']}: {r['text']}")
