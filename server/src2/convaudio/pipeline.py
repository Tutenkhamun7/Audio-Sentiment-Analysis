"""DAG orchestration, checkpointing, and pipeline execution for Stages 1-4."""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any

from convaudio.config import PipelineConfig, load_config
from convaudio.errors import ConvaudioError
from convaudio.io.audio import calculate_sha256, load_audio
from convaudio.stage1_separate.branch import decide_branch
from convaudio.stage1_separate.joint import separate_speech_pyannote, separate_speech_stub
from convaudio.stage1_separate.vad_only import process_clean_stereo, process_no_split
from convaudio.stage2_timeline.builder import build_timeline
from convaudio.stage2_timeline.schema import (
    ConversationTimeline,
    SemanticFeatures,
    Turn,
    TurnOverlap,
    TurnQuality,
)
from convaudio.stage2_timeline.validate import validate_timeline
from convaudio.stage3_lexical.asr import process_stage3_lexical
from convaudio.stage4_acoustic.pooling import run_acoustic_stage
from convaudio.stage4_acoustic.stub import StubFrameEmotionEncoder
from convaudio.stage5_semantic.features import extract_features_array
from convaudio.stage5_semantic.gates import (
    check_abstention_post_model,
    check_abstention_pre_model,
    scale_temperature,
)
from convaudio.stage5_semantic.stub import StubSemanticClassifier
from convaudio.stage5_semantic.window import build_context_window


def _json_hash(data: Any) -> str:
    """Calculate deterministic sha256 for a JSON-serializable object or string."""
    if isinstance(data, (dict, list)):
        payload = json.dumps(data, sort_keys=True, default=str)
    else:
        payload = str(data)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


class PipelineRunner:
    """Orchestrates Stages 1-4 DAG with resumability and deterministic caching."""

    def __init__(
        self,
        config: PipelineConfig | None = None,
        stub_models: bool = False,
    ) -> None:
        self.config = config or load_config()
        self.stub_models = stub_models

    def run(
        self,
        audio_path: Path | str,
        run_dir: Path | str,
        num_speakers: int | None = None,
        force: bool = False,
        force_branch: str | None = None,
    ) -> ConversationTimeline:
        """Execute full batch pipeline from Stage 1 through Stage 4."""
        run_path = Path(run_dir)
        run_path.mkdir(parents=True, exist_ok=True)
        call_id = Path(audio_path).stem

        # ---------------- STAGE 1 ----------------
        s1_file = run_path / "stage1.json"
        audio_hash = calculate_sha256(audio_path)
        s1_input_hash = _json_hash({
            "audio_hash": audio_hash,
            "num_speakers": num_speakers or self.config.stage1.num_speakers,
            "force_branch": force_branch,
            "split_by_speaker": self.config.stage1.split_by_speaker,
            "stub_models": self.stub_models,
        })

        if not force and s1_file.exists():
            with open(s1_file, "r", encoding="utf-8") as f:
                s1_data = json.load(f)
            if s1_data.get("input_hash") == s1_input_hash:
                print(f"[Stage 1] Skipping (resumed from {s1_file.name})")
                stage1_out = s1_data
            else:
                stage1_out = self._execute_stage1(
                    audio_path, run_path, call_id, s1_input_hash, num_speakers, force_branch
                )
        else:
            stage1_out = self._execute_stage1(
                audio_path, run_path, call_id, s1_input_hash, num_speakers, force_branch
            )

        # ---------------- STAGE 2 ----------------
        s2_file = run_path / "stage2.json"
        s2_input_hash = calculate_sha256(s1_file)

        if not force and s2_file.exists():
            with open(s2_file, "r", encoding="utf-8") as f:
                s2_data = json.load(f)
            if s2_data.get("input_hash") == s2_input_hash:
                print(f"[Stage 2] Skipping (resumed from {s2_file.name})")
                timeline = ConversationTimeline.from_json(run_path / "timeline.json")
            else:
                timeline = self._execute_stage2(run_path, call_id, stage1_out, s2_input_hash)
        else:
            timeline = self._execute_stage2(run_path, call_id, stage1_out, s2_input_hash)

        # ---------------- STAGE 3 ----------------
        s3_file = run_path / "stage3.json"
        s3_input_hash = _json_hash({
            "stage2_hash": calculate_sha256(s2_file),
            "stub_models": self.stub_models,
        })

        if not force and s3_file.exists():
            with open(s3_file, "r", encoding="utf-8") as f:
                s3_data = json.load(f)
            if s3_data.get("input_hash") == s3_input_hash:
                print(f"[Stage 3] Skipping (resumed from {s3_file.name})")
                timeline = ConversationTimeline.from_json(run_path / "timeline.json")
            else:
                timeline = self._execute_stage3(timeline, run_path, s3_input_hash)
        else:
            timeline = self._execute_stage3(timeline, run_path, s3_input_hash)

        # ---------------- STAGE 4 ----------------
        s4_file = run_path / "stage4.json"
        s4_input_hash = _json_hash({
            "stage3_hash": calculate_sha256(s3_file),
            "encoder": self.config.stage4.encoder,
            "stub_models": self.stub_models,
        })

        if not force and s4_file.exists():
            with open(s4_file, "r", encoding="utf-8") as f:
                s4_data = json.load(f)
            if s4_data.get("input_hash") == s4_input_hash:
                print(f"[Stage 4] Skipping (resumed from {s4_file.name})")
                timeline = ConversationTimeline.from_json(run_path / "timeline.json")
            else:
                timeline = self._execute_stage4(
                    timeline, run_path, stage1_out.get("embeddings_path"), s4_input_hash
                )
        else:
            timeline = self._execute_stage4(
                timeline, run_path, stage1_out.get("embeddings_path"), s4_input_hash
            )

        # ---------------- STAGE 5 ----------------
        s5_file = run_path / "stage5.json"
        text_payload = [{"turn_id": t.turn_id, "text": t.text} for t in timeline.turns]
        text_hash = hashlib.sha256(json.dumps(text_payload, sort_keys=True).encode("utf-8")).hexdigest()
        s5_input_hash = _json_hash({
            "text_hash": text_hash,
            "classifier": self.config.stage5.classifier,
            "model_path": str(self.config.stage5.model_path),
            "stub_models": self.stub_models,
        })

        if not force and s5_file.exists():
            with open(s5_file, "r", encoding="utf-8") as f:
                s5_data = json.load(f)
            if s5_data.get("input_hash") == s5_input_hash and s5_data.get("text_hash") == text_hash:
                print(f"[Stage 5] Skipping (resumed from {s5_file.name})")
                timeline = ConversationTimeline.from_json(run_path / "timeline.json")
            else:
                timeline = self._execute_stage5(timeline, run_path, s5_input_hash, text_hash)
        else:
            timeline = self._execute_stage5(timeline, run_path, s5_input_hash, text_hash)

        # Final validation
        timeline_file = run_path / "timeline.json"
        timeline.to_json(timeline_file)
        validate_timeline(timeline_file)
        return timeline

    def _execute_stage1(
        self,
        audio_path: Path | str,
        run_path: Path,
        call_id: str,
        input_hash: str,
        num_speakers: int | None,
        force_branch: str | None,
    ) -> dict[str, Any]:
        t0 = time.perf_counter()
        waveform, audio_meta = load_audio(audio_path, target_sr=self.config.audio.target_sr)

        if not self.config.stage1.split_by_speaker or (force_branch and force_branch.strip().lower() in ("no_split", "none", "single", "nosplit")):
            branch = "no_split"
        else:
            branch, _ = decide_branch(
                waveform,
                corr_threshold=self.config.branch.stereo_corr_threshold,
                min_energy_ratio=self.config.branch.min_channel_energy_ratio,
                force_branch=force_branch,
            )
        audio_meta.branch = branch

        if branch == "no_split":
            res = process_no_split(
                waveform=waveform,
                sr=self.config.audio.target_sr,
                out_dir=run_path,
                call_id=call_id,
                frame_rate_hz=self.config.stage2.default_frame_rate_hz,
            )
        elif branch == "stereo_split":
            res = process_clean_stereo(
                waveform=waveform,
                sr=self.config.audio.target_sr,
                out_dir=run_path,
                call_id=call_id,
                frame_rate_hz=self.config.stage2.default_frame_rate_hz,
            )
        else:
            if self.stub_models or not self.config.allow_download:
                res = separate_speech_stub(
                    waveform=waveform,
                    sr=self.config.audio.target_sr,
                    out_dir=run_path,
                    call_id=call_id,
                    num_speakers=num_speakers or self.config.stage1.num_speakers,
                    frame_rate_hz=self.config.stage2.default_frame_rate_hz,
                    min_speaker_speech_s=self.config.stage1.min_speaker_speech_s,
                )
            else:
                res = separate_speech_pyannote(
                    waveform=waveform,
                    sr=self.config.audio.target_sr,
                    out_dir=run_path,
                    call_id=call_id,
                    config=self.config,
                )

        from dataclasses import asdict
        dur = round(time.perf_counter() - t0, 3)

        # Serialize turns recursively
        turns_dict = [asdict(t) for t in res["turns"]]
        stage1_payload = {
            "input_hash": input_hash,
            "branch": branch,
            "audio_meta": audio_meta.__dict__,
            "streams": res["streams"],
            "diarization_rttm": res["diarization_rttm"],
            "overlap_mask_path": res["overlap_mask_path"],
            "embeddings_path": res["embeddings_path"],
            "speakers": res["speakers"],
            "turns": turns_dict,
            "frame_rate_hz": res["frame_rate_hz"],
            "n_frames": res["n_frames"],
            "rejected_sources": res.get("rejected_sources", []),
            "stage_duration_s": dur,
        }
        if "model_revision" in res:
            stage1_payload["model_revision"] = res["model_revision"]

        s1_file = run_path / "stage1.json"
        with open(s1_file, "w", encoding="utf-8") as f:
            json.dump(stage1_payload, f, indent=2)

        return stage1_payload

    def _execute_stage2(
        self,
        run_path: Path,
        call_id: str,
        stage1_out: dict[str, Any],
        input_hash: str,
    ) -> ConversationTimeline:
        t0 = time.perf_counter()
        from convaudio.stage2_timeline.schema import AudioMeta

        audio_meta = AudioMeta(**stage1_out["audio_meta"])

        # Reconstruct turns from dict
        turns: list[Turn] = []
        for td in stage1_out["turns"]:
            td_copy = dict(td)
            ov_data = td_copy.pop("overlap")
            q_data = td_copy.pop("quality", {})
            turns.append(
                Turn(
                    overlap=TurnOverlap(**ov_data),
                    quality=TurnQuality(**q_data),
                    **td_copy,
                )
            )

        stage1_artefacts = dict(stage1_out)
        stage1_artefacts["turns"] = turns

        timeline = build_timeline(
            call_id=call_id,
            audio_meta=audio_meta,
            stage1_artefacts=stage1_artefacts,
            interrupt_window_s=self.config.stage2.interrupt_window_s,
        )

        dur = round(time.perf_counter() - t0, 3)
        timeline.diagnostics.stage_durations_s["s1"] = stage1_out.get("stage_duration_s", 0.0)
        timeline.diagnostics.stage_durations_s["s2"] = dur
        if "model_revision" in stage1_out:
            timeline.diagnostics.model_revisions["diarizer"] = stage1_out["model_revision"]

        timeline_path = run_path / "timeline.json"
        timeline.to_json(timeline_path)

        s2_payload = {
            "input_hash": input_hash,
            "timeline_path": str(timeline_path),
            "stage_duration_s": dur,
        }
        with open(run_path / "stage2.json", "w", encoding="utf-8") as f:
            json.dump(s2_payload, f, indent=2)

        return timeline

    def _execute_stage3(
        self,
        timeline: ConversationTimeline,
        run_path: Path,
        input_hash: str,
    ) -> ConversationTimeline:
        t0 = time.perf_counter()
        res = process_stage3_lexical(
            timeline=timeline,
            config=self.config,
            stub_models=self.stub_models,
        )
        dur = round(time.perf_counter() - t0, 3)
        timeline.diagnostics.stage_durations_s["s3"] = dur

        timeline_path = run_path / "timeline.json"
        timeline.to_json(timeline_path)

        s3_payload = {
            "input_hash": input_hash,
            "alignment_failures": res["alignment_failures"],
            "stage_duration_s": dur,
        }
        with open(run_path / "stage3.json", "w", encoding="utf-8") as f:
            json.dump(s3_payload, f, indent=2)

        return timeline

    def _execute_stage4(
        self,
        timeline: ConversationTimeline,
        run_path: Path,
        embeddings_path: str | Path | None,
        input_hash: str,
    ) -> ConversationTimeline:
        t0 = time.perf_counter()

        from convaudio.stage4_acoustic.interfaces import FrameEmotionEncoder

        # Instantiate emotion encoder
        encoder: FrameEmotionEncoder
        if self.stub_models or self.config.stage4.encoder == "stub":
            encoder = StubFrameEmotionEncoder(
                frame_rate_hz=self.config.stage2.default_frame_rate_hz
            )
        elif self.config.stage4.encoder == "wavlm":
            from convaudio.stage4_acoustic.encoder_wavlm import WavLMValenceArousalEncoder

            encoder = WavLMValenceArousalEncoder(config=self.config)
        elif self.config.stage4.encoder in ("emotion2vec", "emotion2vec_plus"):
            from convaudio.stage4_acoustic.encoder_emotion2vec import Emotion2VecEncoder

            encoder = Emotion2VecEncoder(config=self.config)
        else:
            raise ValueError(f"Unknown encoder type: {self.config.stage4.encoder}")

        run_acoustic_stage(
            timeline=timeline,
            encoder=encoder,
            embeddings_path=embeddings_path,
            min_clean_speech_s=self.config.stage4.min_clean_speech_s,
            max_overlap_ratio=self.config.stage4.max_overlap_ratio,
            min_sep_cosine=self.config.stage4.min_sep_cosine,
        )

        dur = round(time.perf_counter() - t0, 3)
        timeline.diagnostics.stage_durations_s["s4"] = dur

        timeline_path = run_path / "timeline.json"
        timeline.to_json(timeline_path)

        s4_payload = {
            "input_hash": input_hash,
            "acoustic_abstentions": timeline.diagnostics.acoustic_abstentions,
            "abstain_reasons": timeline.diagnostics.abstain_reasons,
            "stage_duration_s": dur,
        }
        with open(run_path / "stage4.json", "w", encoding="utf-8") as f:
            json.dump(s4_payload, f, indent=2)

        return timeline

    def run_stage(
        self,
        stage_num: int,
        run_dir: Path | str,
        force: bool = False,
    ) -> None:
        """Run a single stage in an existing run directory."""
        run_path = Path(run_dir)
        if stage_num == 1:
            raise ConvaudioError("Stage 1 requires audio file input. Use 'convaudio run' instead.")
        elif stage_num == 2:
            s1_file = run_path / "stage1.json"
            if not s1_file.exists():
                raise FileNotFoundError(f"Missing {s1_file}")
            with open(s1_file, "r", encoding="utf-8") as f:
                s1_data = json.load(f)
            call_id = Path(s1_data["audio_meta"]["path"]).stem
            self._execute_stage2(run_path, call_id, s1_data, calculate_sha256(s1_file))
        elif stage_num == 3:
            timeline_file = run_path / "timeline.json"
            if not timeline_file.exists():
                raise FileNotFoundError(f"Missing {timeline_file}")
            timeline = ConversationTimeline.from_json(timeline_file)
            self._execute_stage3(timeline, run_path, "single-stage-exec")
        elif stage_num == 4:
            timeline_file = run_path / "timeline.json"
            if not timeline_file.exists():
                raise FileNotFoundError(f"Missing {timeline_file}")
            timeline = ConversationTimeline.from_json(timeline_file)
            emb_file = run_path / "embeddings.npz"
            self._execute_stage4(timeline, run_path, emb_file, "single-stage-exec")
        elif stage_num == 5:
            timeline_file = run_path / "timeline.json"
            if not timeline_file.exists():
                raise FileNotFoundError(f"Missing {timeline_file}")
            timeline = ConversationTimeline.from_json(timeline_file)
            text_payload = [{"turn_id": t.turn_id, "text": t.text} for t in timeline.turns]
            text_hash = hashlib.sha256(json.dumps(text_payload, sort_keys=True).encode("utf-8")).hexdigest()
            s5_input_hash = _json_hash({
                "text_hash": text_hash,
                "classifier": self.config.stage5.classifier,
                "model_path": str(self.config.stage5.model_path),
                "stub_models": self.stub_models,
            })
            self._execute_stage5(timeline, run_path, s5_input_hash, text_hash)
        else:
            raise ValueError(f"Invalid stage_num: {stage_num}. Must be 1, 2, 3, 4, or 5.")

    def _execute_stage5(
        self,
        timeline: ConversationTimeline,
        run_path: Path,
        input_hash: str,
        text_hash: str,
        classifier: Any = None,
    ) -> ConversationTimeline:
        """Execute Stage 5 semantic sentiment prediction, abstention gating, and calibration."""
        import numpy as np
        t0 = time.perf_counter()

        # Instantiate classifier if not injected
        if classifier is None:
            cls_name = "stub" if self.stub_models else self.config.stage5.classifier.lower()
            if cls_name == "stub":
                classifier = StubSemanticClassifier()
            elif cls_name == "setfit":
                from convaudio.stage5_semantic.setfit_classifier import SetFitSemanticClassifier
                classifier = SetFitSemanticClassifier(model_path=self.config.stage5.model_path)
            elif cls_name == "modernbert":
                from convaudio.stage5_semantic.modernbert_classifier import (
                    ModernBertSemanticClassifier,
                )
                classifier = ModernBertSemanticClassifier(model_path=self.config.stage5.model_path)
            else:
                raise ValueError(f"Unknown Stage 5 classifier: {cls_name}")
        else:
            cls_name = getattr(classifier, "classifier_name", "custom")

        active_indices: list[int] = []
        windows: list[str] = []

        # Gate check pre-model
        for idx, turn in enumerate(timeline.turns):
            pre_reason = check_abstention_pre_model(
                turn,
                min_words=self.config.stage5.min_words,
                min_asr_logprob=self.config.stage5.min_asr_logprob,
            )
            if pre_reason is not None:
                turn.semantic = None
                turn.semantic_abstain_reason = pre_reason
            else:
                win = build_context_window(
                    timeline.turns,
                    target_idx=idx,
                    n_context=self.config.stage5.n_context,
                    speaker_roles=self.config.stage5.speaker_roles,
                )
                active_indices.append(idx)
                windows.append(win)

        if active_indices:
            active_turns = [timeline.turns[i] for i in active_indices]
            features_raw = extract_features_array(
                active_turns, feature_names=classifier.feature_names
            )

            # Apply scaler if present on classifier (assert loaded, never fitted)
            if hasattr(classifier, "scaler") and classifier.scaler is not None:
                features_scaled = classifier.scaler.transform(features_raw)
            else:
                features_scaled = features_raw

            raw_probs = classifier.predict(windows, features_scaled)

            # Temperature calibration
            if hasattr(classifier, "temperature") and classifier.temperature is not None:
                calibrated_probs = scale_temperature(raw_probs, float(classifier.temperature))
                is_calibrated = True
            else:
                if self.config.stage5.require_calibration and not self.stub_models and cls_name != "stub":
                    raise ConvaudioError("Calibration artifact missing and require_calibration is True")
                calibrated_probs = raw_probs
                is_calibrated = True if (self.stub_models or cls_name == "stub") else False

            for k, turn_idx in enumerate(active_indices):
                turn = timeline.turns[turn_idx]
                turn_probs = calibrated_probs[k]

                post_reason = check_abstention_post_model(
                    turn_probs, min_conf=self.config.stage5.min_conf
                )
                if post_reason is not None:
                    # Optional LLM adjudication
                    adjudicated = False
                    if self.config.stage5.llm_adjudication:
                        try:
                            from convaudio.stage5_semantic.llm_adjudicate import (
                                maybe_adjudicate_turn,
                            )
                            adj_res = maybe_adjudicate_turn(
                                turn, windows[k], turn_probs, self.config.stage5
                            )
                            if adj_res is not None:
                                turn.semantic = adj_res
                                turn.semantic_abstain_reason = None
                                adjudicated = True
                        except Exception:
                            adjudicated = False
                    if not adjudicated:
                        turn.semantic = None
                        turn.semantic_abstain_reason = post_reason
                else:
                    winning_idx = int(np.argmax(turn_probs))
                    winning_label = classifier.labels[winning_idx]
                    winning_conf = float(turn_probs[winning_idx])
                    probs_dict = {
                        lbl: float(turn_probs[j]) for j, lbl in enumerate(classifier.labels)
                    }
                    turn.semantic = SemanticFeatures(
                        probs=probs_dict,
                        label=winning_label,
                        conf=winning_conf,
                        calibrated=is_calibrated,
                        model_revision=classifier.checkpoint_revision,
                    )
                    turn.semantic_abstain_reason = None

        # Bump schema version to 1.1.0
        timeline.schema_version = "1.1.0"

        dur = round(time.perf_counter() - t0, 3)
        timeline.diagnostics.model_revisions["stage5_semantic"] = classifier.checkpoint_revision
        timeline.diagnostics.stage_durations_s["s5"] = dur

        # Persist stage5.json
        s5_payload = {
            "input_hash": input_hash,
            "text_hash": text_hash,
            "classifier": cls_name,
            "model_revision": classifier.checkpoint_revision,
            "n_turns": len(timeline.turns),
            "n_abstained": sum(1 for t in timeline.turns if t.semantic_abstain_reason is not None),
            "stage_duration_s": dur,
        }
        with open(run_path / "stage5.json", "w", encoding="utf-8") as f:
            json.dump(s5_payload, f, indent=2)

        timeline_path = run_path / "timeline.json"
        timeline.to_json(timeline_path)
        validate_timeline(timeline_path)

        return timeline
