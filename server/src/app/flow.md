### End-to-End Pipeline Diagram

│ Diagram exceeds terminal width (535 > 205 cols)  
 │ Displayed as code block. Widen terminal to view inline.

    flowchart TD
        A["Raw Audio Input\n(FastAPI Upload / Memory Buffer)"] --> B["Stage 0: In-Memory Normalization\n(Decode, Mono/Stereo Check, Resample to 16kHz)"]

        B --> C{"Stage 1: Branch Decision"}
        C -->|"Stereo Call Center Recording"| D1["Branch A: Stereo Split\n(Ch 0 = Speaker 0, Ch 1 = Speaker 1)"]
        C -->|"Explicit 'no_split' / Single Speaker"| D2["Branch B: Monologue\n(Single Stream, Bypass Diarization)"]
        C -->|"Mono Multi-Speaker Call"| D3["Branch C: Mono Diarized"]

        D3 --> E["Stage 2: Pyannote Diarization\n(Voiceprint Clustering & Overlap Detection)"]
        E --> F["Stage 2b: Conversational Turn Merging\n(Merge adjacent turns of same speaker if pause < 0.8s)"]

        F --> G{"Any Collisions?\n(Overlaps > 0.05s)"}
        G -->|"No Overlaps"| H["Standard Masked Streams"]
        G -->|"Yes"| I["Stage 3: Targeted SepFormer Separation\n(Slice only overlap windows + 0.2s padding)"]
        I --> J["Stage 3b: Permutation Alignment\n(Match 13-dim MFCC timbre to correct speaker)"]
        J --> K["Stage 3c: Cosine Crossfade Splicing\n(Splice separated audio back into speaker streams)"]

        D1 --> L["Stage 4: In-Memory Speaker Streams\n(SPEAKER_00 Tensor, SPEAKER_01 Tensor)"]
        D2 --> L
        H --> L
        K --> L

        L --> M["Stage 5: Faster-Whisper Transcription\n(Word Timestamps + Vocabulary Prompt Context)"]
        M --> N["Stage 5b: Floor Handover Clamping\n(Prevent padding from leaking into neighbor speaker)"]
        N --> O["Stage 5c: Ghost Turn Pruning\n(Discard non-speech VAD pauses / breaths)"]

        O --> P["Stage 6: Wav2Vec2 CTC Forced Alignment\n(Snap turn start/end to exact millisecond word boundaries)"]
        P --> Q["Stage 7: Emotion & Sentiment\n(Acoustic Emotion2Vec + Text Sentiment)"]
        Q --> R["Stage 8: Dynamics & Metrics\n(Interruption Counts, Overlap Ratio, Turn Assembly)"]
        R --> S["Final Response: AnalyzeResponse JSON"]

──────

### Step-by-Step Breakdown

#### Stage 0: In-Memory Ingestion & Resampling (app/core/audio.py)

• What happens: The uploaded audio bytes (WAV, MP3, FLAC, M4A) are loaded directly into RAM using a BytesIO buffer.  
 • Normalization:  
 • Standardized to 16, 000 Hz (the universal sample rate expected by Pyannote, SepFormer, and Whisper).  
 • Kept as a pure PyTorch float tensor (channels, samples).  
 • Zero Disk I/O: No files are ever written to disk. This avoids SSD bottlenecks and prevents temp-file clutter in production.  
 ──────

#### Stage 1: Branch Detection (app/engine/branch.py)

Before running any heavy neural models, the system inspects the audio structure:

1. Stereo Branch (stereo_split): Many enterprise call centers record Agent on Channel 0 and Customer on Channel 1. If channels have independent energy, it bypasses neural diarization entirely.
2. No-Split Branch (no_split): If the user marks the file as a single-speaker voice note or lecture, diarization is skipped.
3. Mono Separated Branch (mono_separated): Standard mono audio where multiple people talk on the same microphone track.  
   ──────

#### Stage 2: Diarization & Turn Merging (app/engine/diarizer.py)

If running on mono audio:

1. Pyannote 4.0 Clustering: Extracts speaker voiceprint embeddings and segments the call into raw timestamps:  
   • [04.12s - 08.50s] SPEAKER_01  
   • [09.10s - 15.20s] SPEAKER_01  
   • [25.18s - 28.82s] SPEAKER_00
2. Conversational Turn Merging: In real life, humans take natural micro-breaths while speaking. If the same speaker has two consecutive segments separated by less than 0.8s, they are merged into one  
   continuous turn.  
    • Impact: Reduces fragmented transcripts from 40 choppy pieces down to 20–25 natural dialogue turns.
3. Overlap Speech Detection (OSD): Flags regions where both speakers are speaking at once (e.g. [30.36s - 30.90s]).  
   ──────

#### Stage 3: Targeted Overlap Separation (app/engine/overlap_separator.py)

Instead of separating the entire 3-minute file (which is slow and introduces phase artifacts):

1. Targeted Slicing: The system extracts only the collision slices (e.g. a 0.54s chunk) with a small 0.2s acoustic context padding.
2. SepFormer Inference: Runs SpeechBrain's SepFormer on just that chunk to split the blended sound into two distinct sources (Source A and Source B).
3. Timbre Permutation Matching (MFCCs): SepFormer doesn't know who is Speaker 0 or Speaker 1. The system computes a 13-dimensional MFCC acoustic timbre vector from clean audio of each speaker, computes
   cosine similarity, and matches Source A → Speaker 0 and Source B → Speaker 1.
4. Cosine Crossfade Splicing: The untangled pieces are smoothly blended back into each speaker's audio timeline using a 50ms raised-cosine crossfade curve to avoid audible clicks.  
   ──────

#### Stage 4 & 5: Transcription & Floor Handover Clamping (app/engine/transcriber.py)

Each speaker's separated audio stream is transcribed via faster-whisper:

1. Full-Stream Word Timestamps: Whisper transcribes the stream and outputs millisecond start/end timestamps for every word.
2. Vocabulary Guidance (initial_prompt): A configurable business domain prompt guides Whisper on proper nouns (e.g. "Marathahalli", "KA 9515").
3. Floor Handover Clamping:  
   • When Speaker 1 yields the floor to Speaker 2, Speaker 1's boundary is clamped so it never spills into Speaker 2's start time.  
   • Speaker 2 starts cleanly without absorbing boundary clicks from Speaker 1.
4. Ghost Turn Pruning: If a diarized segment contains no spoken words (just a breath or line click), it is cleanly discarded so it doesn't create empty turns or duplicate paragraphs.  
   ──────

#### Stage 6: Word Alignment & Boundary Snapping (app/engine/aligner.py)

1. Wav2Vec2 CTC Forced Alignment: Compares the transcribed words against the raw audio phoneme emissions.
2. Boundary Snapping: Snaps the turn's start and end times to the exact millisecond of the first spoken phoneme and last spoken phoneme (the technique pioneered by WhisperX).  
   ──────

#### Stage 7: Emotion & Sentiment (app/engine/acoustic.py & semantic.py)

1. Acoustic Affect: If emotion2vec is active, it extracts valence (positivity/negativity) and arousal (energy/excitement) directly from the audio pitch and cadence.
2. Text Sentiment: Runs sentiment analysis on the transcribed text to classify each turn as positive, negative, or neutral.  
   ──────

#### Stage 8: Conversational Dynamics & Response Serialization (app/services/orchestrator.py)

The orchestrator compiles the final metrics:

• Interruption Detection: Did Speaker B start talking before Speaker A finished, causing Speaker A to yield within 1.0s? Flags [INTERRUPTS] and [INTERRUPTED].  
 • Overlap Metrics: Total overlap duration in seconds and overlap percentage of the call.  
 • Turn Assembly: Serializes into the strongly-typed Pydantic AnalyzeResponse DTO.  
 ──────

### Summary of What Makes This Architecture Unique

Feature │ How Traditional Pipelines Do It │ How app Does It
──────────────────────────────────────────────────────────┼─────────────────────────────────────────────────────────┼────────────────────────────────────────────────────────────────────────────────────
Disk I/O │ Writes 50+ intermediate .wav files to disk. │ Zero disk writes (100% in-memory PyTorch tensors).
Speech Separation │ Separates the whole 3-minute file (slow, noisy). │ Targeted: Separates only the 1.5 seconds of overlapping speech.
Turn Boundary Spills │ Speaker A steals Speaker B's first words. │ Floor Handover Clamping: Clamps padding to next speaker's onset.
Hardware │ Hardcoded to GPU or CPU. │ Granular: Swap each model independently (ASR_DEVICE=cuda, DIARIZATION_DEVICE=cpu).
