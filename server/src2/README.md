# convaudio

Batch audio-analysis pipeline (Stages 1–4) for contact-centre call recordings.

`convaudio` implements deterministic, resumable, licence-compliant audio processing designed specifically for conversational speech with overlapping dialogue.

The primary deliverable of Stages 1–4 is a single validated canonical artefact: **`timeline.json`**. Stages 5+ (semantic sentiment, cross-modal fusion, and call roll-up) are deferred to downstream services while preserving a fully extensible data contract.

---

## Architecture Overview

```
                      +-----------------------------+
                      |       Input Audio           |
                      +--------------+--------------+
                                     |
                          [1a. Decode & Resample]
                                     |
                          [1b. Channel Branch]
                         /                    \
              (clean stereo)                (mixture)
                    /                          \
       +-----------------------+      +-------------------------+
       | [1b/1d] VAD Split     |      | [1c] Speech Separation  |
       | provenance="clean"    |      | (pyannote.audio / stub) |
       | SKIP separation       |      | Leakage removal ON      |
       +-----------+-----------+      | ASR collar dilation ON  |
                   |                  | [1d] Phantom Rejection  |
                   |                  | provenance="separated"  |
                   |                  +------------+------------+
                   |                               |
                   \                               /
            +-------v-----------------------------v-------+
            | Stage 2: Canonical ConversationTimeline    |
            | schema/timeline-1.0.0.json, frame-level    |
            | overlap mask, interruption detection       |
            +----------------------+----------------------+
                                   |
                   +---------------+---------------+
                   |                               |
      +------------v------------+     +------------v------------+
      | Stage 3: Lexical Path   |     | Stage 4: Acoustic Path  |
      | faster-whisper VAD ON   |     | FrameEmotionEncoder     |
      | torchaudio forced_align |     | WavLM / Stub encoder    |
      | ASR fallback on fail    |     | sep_cosine quality gate |
      | word_timings at ~20ms   |     | Masked pooling          |
      +------------+------------+     | Abstention precedence   |
                   |                  +------------+------------+
                   \                               /
                    +--------------+--------------+
                                   |
                      +------------v------------+
                      |   Final timeline.json   |
                      |   Validated Artefact    |
                      +-------------------------+
```

---

## Hard Constraints & Licence Policy

1. **Licence Policy Enforced in Code**:
   - **Allowed**: `MIT`, `Apache-2.0`, `BSD-2-Clause`, `BSD-3-Clause`, `ISC`
   - **Allowed with Attribution**: `CC-BY-4.0`
   - **Denied**: `*-NC*`, non-commercial, research-only, `BSD-4-Clause`, `GPL-*`, `LGPL-*`, `AGPL-*`, `UNKNOWN`, `UNRESOLVED`
   - `pyannote.audio` is approved per policy including gated HF weights.
   - Code licences and model weights licences are audited as separate fields.
   - Run `convaudio licences check` to scan dependencies and manifests.
   - See `THIRD_PARTY_NOTICES.md` for explicit per-component declarations.
2. **Zero `ffmpeg` Subprocess Dependency**:
   - Pure Python/C decoding via `soundfile` and `torchaudio`.
   - Unsupported audio raises `UnsupportedAudioFormat` with the detected format. Never shells out.
3. **Deterministic & Resumable**:
   - Each stage writes `stageN.json` and skips execution if previous output and input hashes match.
   - Use `--force` to invalidate caches and force re-execution.
4. **Models Behind Interfaces**:
   - All models adhere to `Protocol` definitions with fixture-returning stubs.
   - `--stub-models` executes the complete DAG offline in < 2 seconds with no GPU and no network.
   - Checkpoint resolution order: `CONVAUDIO_MODEL_DIR` → Hugging Face Cache → Hugging Face Download (guarded by `CONVAUDIO_ALLOW_DOWNLOAD=1`).

---

## CLI Commands

```bash
# 1. Run full pipeline (Stages 1-4)
convaudio run audio.wav --out runs/c_8812 --stub-models
convaudio run audio.wav --out runs/c_8812 --num-speakers 2 --force
convaudio run audio.wav --out runs/c_8812 --force-branch stereo

# 2. Run individual stage
convaudio stage 2 --run-dir runs/c_8812

# 3. Validate timeline.json against schema and domain invariants
convaudio validate runs/c_8812/timeline.json

# 4. Generate Markdown diagnostic report
convaudio report --run-dir runs/c_8812

# 5. Print formatted dialogue script (Speaker 1 - Dialogue)
convaudio dialogue runs/c_8812/timeline.json
convaudio dialogue runs/c_8812 --merge-consecutive

# 6. Check licence compliance across distributions and weights
convaudio licences check
```

---

## Running Tests

From `server/`:

```powershell
$env:PYTHONPATH = "src2"
uv run pytest src2/tests/test_all_18.py -v
uv run ruff check --config src2/pyproject.toml src2
uv run mypy --strict --config-file src2/pyproject.toml src2/convaudio
```
