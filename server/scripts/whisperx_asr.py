import os
import json
import subprocess
from typing import Dict, List, Optional, Any
import torch
import whisperx
from whisperx.alignment import load_align_model, align
from whisperx.diarize import DiarizationPipeline, assign_word_speakers

# 1. FIX FFMPEG FOR PYTORCH / TORCHAUDIO ON WINDOWS
FFMPEG_BIN_PATH = r"C:\Users\Anvay\AppData\Local\Microsoft\WinGet\Packages\Gyan.FFmpeg.Shared_Microsoft.Winget.Source_8wekyb3d8bbwe\ffmpeg-9.0.2-full_build-shared\bin"
if os.name == "nt" and os.path.exists(FFMPEG_BIN_PATH):
    try:
        os.add_dll_directory(FFMPEG_BIN_PATH)
        print(f"[System] Registered FFmpeg DLLs from {FFMPEG_BIN_PATH}")
    except Exception as e:
        print(f"[System] Warning registering FFmpeg DLLs: {e}")


class WhisperXASREngine:
    """
    Advanced ASR Engine powered by WhisperX:
    - Faster-Whisper batched transcription
    - Wav2Vec2 phoneme-level forced alignment for ultra-precise word timestamps
    - Pyannote speaker diarization with word-level speaker attribution
    - Dual-channel (Stereo) automatic detection & split to eliminate speaker overlap errors
    """

    def __init__(
        self,
        hf_token: Optional[str] = None,
        model_size: str = "large-v3-turbo",
        device: Optional[str] = None,
        compute_type: Optional[str] = None,
        batch_size: int = 16,
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        if self.device == "cuda":
            self.compute_type = compute_type or "float16"
        else:
            self.compute_type = compute_type or "int8"

        self.batch_size = batch_size
        self.hf_token = hf_token or os.getenv("HF_TOKEN")

        print(
            f"[Init] Loading WhisperX model ({model_size}) on {self.device} ({self.compute_type})..."
        )
        self.model = whisperx.load_model(
            model_size,
            device=self.device,
            compute_type=self.compute_type,
            asr_options={
                "temperatures": [0.0],
                "initial_prompt": None,
                "condition_on_previous_text": False,
            },
        )

        # Diarization pipeline (lazy-loaded if token available)
        self.diarize_pipeline: Optional[DiarizationPipeline] = None
        if self.hf_token:
            print("[Init] Loading WhisperX Pyannote Diarization Pipeline...")
            try:
                self.diarize_pipeline = DiarizationPipeline(token=self.hf_token, device=self.device)
            except Exception as e:
                print(f"[Init Warning] Could not load Diarization Pipeline: {e}")

        # Cache alignment models per language: { "en": (model, metadata) }
        self._align_models: Dict[str, Any] = {}

    def _get_align_model(self, language_code: str):
        if language_code not in self._align_models:
            print(
                f"[Alignment] Loading Wav2Vec2 alignment model for language: '{language_code}'..."
            )
            align_model, metadata = load_align_model(
                language_code=language_code, device=self.device
            )
            self._align_models[language_code] = (align_model, metadata)
        return self._align_models[language_code]

    def _get_channel_count(self, file_path: str) -> int:
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
            print(f"[Probe Warning] Could not determine channels ({e}), defaulting to Mono.")
            return 1

    def _group_words_into_turns(
        self,
        words: List[Dict[str, Any]],
        max_pause_sec: float = 0.8,
        max_turn_duration_sec: float = 6.0,
    ) -> List[Dict[str, Any]]:
        """Groups sequentially aligned words by speaker into readable conversational turns."""
        turns: List[Dict[str, Any]] = []
        if not words:
            return turns

        current_turn: Optional[Dict[str, Any]] = None
        terminal_punctuation = {".", "!", "?"}

        for w in words:
            word_str = w.get("word", "").strip()
            if not word_str:
                continue

            start = float(w.get("start", 0.0))
            end = float(w.get("end", start + 0.1))
            speaker = w.get("speaker", "SPEAKER_00")

            if current_turn is None:
                current_turn = {
                    "speaker": speaker,
                    "start_time": round(start, 2),
                    "end_time": round(end, 2),
                    "text": word_str,
                    "words": [w],
                }
                continue

            same_speaker = speaker == current_turn["speaker"]
            word_gap = start - current_turn["end_time"]
            turn_duration = current_turn["end_time"] - current_turn["start_time"]
            prev_ended_sentence = bool(
                current_turn["text"] and current_turn["text"][-1] in terminal_punctuation
            )

            should_split = (
                not same_speaker
                or (prev_ended_sentence and turn_duration >= 2.0)
                or (word_gap >= max_pause_sec and turn_duration >= 2.0)
                or (turn_duration >= max_turn_duration_sec)
            )

            if should_split:
                turns.append(current_turn)
                current_turn = {
                    "speaker": speaker,
                    "start_time": round(start, 2),
                    "end_time": round(end, 2),
                    "text": word_str,
                    "words": [w],
                }
            else:
                current_turn["end_time"] = round(end, 2)
                current_turn["text"] += f" {word_str}"
                current_turn["words"].append(w)

        if current_turn:
            turns.append(current_turn)

        return turns

    def _process_stereo(
        self, file_path: str, language: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Stereo (2-channel) Call Center Audio:
        Separates Left (Agent) and Right (Customer) channels into discrete audio streams,
        runs WhisperX transcription + Wav2Vec2 forced alignment on each channel independently.
        This provides 100% speaker separation accuracy with zero diarization cross-talk.
        """
        print("[Engine] Stereo detected. Splitting Left (Agent) and Right (Customer) channels...")
        agent_wav = "temp_agent_wx.wav"
        customer_wav = "temp_customer_wx.wav"

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

        all_words: List[Dict[str, Any]] = []

        try:
            for speaker_label, ch_path in [("SPEAKER_00", agent_wav), ("SPEAKER_01", customer_wav)]:
                audio = whisperx.load_audio(ch_path)
                transcription = self.model.transcribe(
                    audio, batch_size=self.batch_size, language=language
                )
                lang = transcription.get("language", language or "en")
                if not transcription.get("segments"):
                    continue

                align_model, metadata = self._get_align_model(lang)
                aligned = align(
                    transcription["segments"],
                    align_model,
                    metadata,
                    audio,
                    self.device,
                    return_char_alignments=False,
                )

                for w in aligned.get("word_segments", []):
                    w["speaker"] = speaker_label
                    all_words.append(w)

        finally:
            for temp_f in [agent_wav, customer_wav]:
                if os.path.exists(temp_f):
                    try:
                        os.remove(temp_f)
                    except OSError:
                        pass

        # Sort all words chronologically across both channels
        all_words.sort(key=lambda w: w.get("start", 0.0))
        return self._group_words_into_turns(all_words)

    def _process_mono(
        self,
        file_path: str,
        language: Optional[str] = None,
        min_speakers: Optional[int] = None,
        max_speakers: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """
        Mono Audio:
        1. Transcribe continuous audio with batched faster-whisper.
        2. Perform CTC forced alignment using Wav2Vec2 to extract exact word boundaries.
        3. Run Pyannote speaker diarization.
        4. Assign speaker labels to individual words and merge into conversational turns.
        """
        print("[Engine] Mono audio detected. Running WhisperX transcription...")
        audio = whisperx.load_audio(file_path)

        # 1. Transcribe
        transcription = self.model.transcribe(audio, batch_size=self.batch_size, language=language)
        detected_lang = transcription.get("language", language or "en")
        segments = transcription.get("segments", [])

        if not segments:
            print("[Engine] No speech detected in audio.")
            return []

        # 2. Wav2Vec2 CTC Forced Alignment
        print(f"[Engine] Aligning phonemes for language '{detected_lang}'...")
        align_model, metadata = self._get_align_model(detected_lang)
        aligned_result = align(
            segments, align_model, metadata, audio, self.device, return_char_alignments=False
        )

        aligned_words = aligned_result.get("word_segments", [])

        # 3. Speaker Diarization
        if self.diarize_pipeline:
            print("[Engine] Running Pyannote Diarization...")
            diarize_segments = self.diarize_pipeline(
                audio, min_speakers=min_speakers, max_speakers=max_speakers
            )
            # 4. Assign speaker to each word
            result_with_speakers = assign_word_speakers(diarize_segments, aligned_result)
            aligned_words = result_with_speakers.get("word_segments", aligned_words)
        else:
            print(
                "[Engine] Diarization skipped (No HF token provided). Defaulting speaker to SPEAKER_00."
            )
            for w in aligned_words:
                w["speaker"] = "SPEAKER_00"

        # 5. Group words into conversational turns
        return self._group_words_into_turns(aligned_words)

    def transcribe(
        self,
        file_path: str,
        language: Optional[str] = None,
        min_speakers: Optional[int] = None,
        max_speakers: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """
        Transcribes an audio file (stereo or mono) with word-level forced alignment and diarization.
        Returns a list of conversational turn dictionaries:
        [
            {
                "speaker": "SPEAKER_00",
                "start_time": 0.12,
                "end_time": 3.45,
                "text": "Hello, how can I help you today?",
                "words": [
                    {"word": "Hello,", "start": 0.12, "end": 0.54, "score": 0.98, "speaker": "SPEAKER_00"},
                    ...
                ]
            }
        ]
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Audio file not found: {file_path}")

        channels = self._get_channel_count(file_path)
        if channels >= 2:
            return self._process_stereo(file_path, language=language)
        else:
            return self._process_mono(
                file_path, language=language, min_speakers=min_speakers, max_speakers=max_speakers
            )


# ==========================================
# TEST RUNNER
# ==========================================
if __name__ == "__main__":
    # Choose audio file: use test.wav if exists, otherwise fallback to local test_audio.wav
    default_test = r"C:\Users\Anvay\Downloads\test.wav"
    audio_path = default_test if os.path.exists(default_test) else "test_audio.wav"
    hf_token = os.getenv("HF_TOKEN", "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")

    print("=" * 60)
    print(" WHISPERX ASR & FORCED ALIGNMENT ENGINE")
    print("=" * 60)
    print(f"Target Audio : {audio_path}")
    print(f"CUDA Available: {torch.cuda.is_available()}")

    engine = WhisperXASREngine(hf_token=hf_token, model_size="large-v3-turbo")

    print("\n--- Running Transcription & Alignment ---")
    results = engine.transcribe(audio_path)

    print("\n" + "=" * 60)
    print(" FINAL RESULTS")
    print("=" * 60)
    for turn in results:
        print(
            f"[{turn['start_time']:05.2f}s - {turn['end_time']:05.2f}s] {turn['speaker']}: {turn['text']}"
        )
        # Optionally show word-level breakdown
        word_details = [
            f"{w.get('word')}({w.get('start', 0.0):.2f}-{w.get('end', 0.0):.2f})"
            for w in turn.get("words", [])
        ]
        # print(f"   Words: {' '.join(word_details)}")
