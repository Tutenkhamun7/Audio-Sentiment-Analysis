"""Multilingual conversational speaker turn segmentation engine."""

from __future__ import annotations

import logging
import re
from typing import Dict, List, Union
import torch
from app.engine.diarizer import RawTurn
from app.schemas.response import WordTiming

logger = logging.getLogger(__name__)

# Multilingual sentence terminal punctuation regex:
# Covers Latin (. ? !), CJK (。 ？ ！), Indic (। ॥), Arabic (؟ ۔), Greek (;), Armenian (։)
SENTENCE_TERMINAL_REGEX = re.compile(
    r'[\.\?\!\u3002\uff1f\uff01\u0964\u0965\u06D4\u061F\u037E\u0589]'
)

# Common honorifics and abbreviations across English, Spanish, French, German, and Hindi
COMMON_ABBREVIATIONS = {
    # English
    "mr", "mrs", "ms", "dr", "prof", "inc", "ltd", "corp", "no", "vs", "e.g", "i.e",
    # Spanish
    "sr", "sra", "srta", "dra", "ee.uu",
    # French
    "m", "mme", "mlle",
    # German
    "z.b", "bzw", "d.h", "u.a",
    # Indic / Romanized
    "shri", "smt", "pandit",
}


def is_terminal_word(token: str) -> bool:
    """Check if token ends with sentence terminal punctuation, ignoring abbreviations."""
    clean = token.strip()
    if not clean:
        return False

    if not SENTENCE_TERMINAL_REGEX.search(clean[-1]):
        return False

    # Check if token without punctuation is a known abbreviation (e.g., 'Dr.' or 'Mr.')
    bare = clean.rstrip(".,?!;:\u3002\uff1f\uff01\u0964\u0965\u06D4\u061F\u037E\u0589").lower()
    if bare in COMMON_ABBREVIATIONS:
        return False

    # Numbers like '3.5' or '10.0' should not split
    if re.match(r'^\d+(\.\d+)?$', clean):
        return False

    return True


def group_words_into_turns(
    words: List[WordTiming],
    speaker: str,
    max_pause_s: float = 1.2,
    soft_turn_duration_s: float = 4.0,
    max_turn_duration_s: float = 8.0,
) -> List[RawTurn]:
    """Segment a single speaker's word stream into natural conversational turns.

    Rules applied:
    1. Inter-Pausal Unit (IPU): Silence pause >= max_pause_s triggers a new turn.
    2. Sentence boundary: Terminal punctuation triggers a split once the turn exceeds soft_turn_duration_s.
    3. Maximum duration cap: Hard cap at max_turn_duration_s to avoid run-on monologue turns.
    """
    if not words:
        return []

    turns: List[RawTurn] = []
    current_words: List[WordTiming] = []

    for w in words:
        if not current_words:
            current_words.append(w)
            continue

        prev_w = current_words[-1]
        pause = w.start - prev_w.end
        turn_dur = prev_w.end - current_words[0].start
        has_terminal = is_terminal_word(prev_w.word)

        # Split conditions:
        should_split = (
            pause >= max_pause_s
            or (has_terminal and pause >= 0.25)
            or (has_terminal and turn_dur >= soft_turn_duration_s)
            or (turn_dur >= max_turn_duration_s)
        )

        if should_split:
            turn_start = round(current_words[0].start, 3)
            turn_end = round(current_words[-1].end, 3)
            turn = RawTurn(
                speaker=speaker,
                start=turn_start,
                end=max(turn_start + 0.05, turn_end),
                text=" ".join(item.word for item in current_words).strip(),
                words=list(current_words),
                turn_type="speech",
            )
            turns.append(turn)
            current_words = [w]
        else:
            current_words.append(w)

    if current_words:
        turn_start = round(current_words[0].start, 3)
        turn_end = round(current_words[-1].end, 3)
        turn = RawTurn(
            speaker=speaker,
            start=turn_start,
            end=max(turn_start + 0.05, turn_end),
            text=" ".join(item.word for item in current_words).strip(),
            words=list(current_words),
            turn_type="speech",
        )
        turns.append(turn)

    return turns


def group_aligned_words_into_turns(
    words: List[WordTiming],
    max_pause_s: float = 1.2,
    soft_turn_duration_s: float = 4.0,
    max_turn_duration_s: float = 8.0,
) -> List[RawTurn]:
    """Segment a continuous chronological word stream (with speaker labels) into conversational turns.

    Splits occur on:
    1. Speaker changes (turn transition)
    2. Silence pause >= max_pause_s (Inter-Pausal Unit)
    3. Terminal punctuation + pause >= 0.25s or duration >= soft_turn_duration_s
    4. Duration cap >= max_turn_duration_s
    """
    if not words:
        return []

    # Sort strictly chronologically
    sorted_words = sorted(words, key=lambda w: (w.start, w.end))

    turns: List[RawTurn] = []
    current_words: List[WordTiming] = []
    current_speaker: str = getattr(sorted_words[0], "speaker", "SPEAKER_00") or "SPEAKER_00"

    for w in sorted_words:
        w_spk = getattr(w, "speaker", current_speaker) or current_speaker

        if not current_words:
            current_words.append(w)
            current_speaker = w_spk
            continue

        prev_w = current_words[-1]
        pause = w.start - prev_w.end
        turn_dur = prev_w.end - current_words[0].start
        has_terminal = is_terminal_word(prev_w.word)

        # Split conditions
        speaker_changed = (w_spk != current_speaker)
        pause_exceeded = (pause >= max_pause_s)
        sentence_ended = (has_terminal and (pause >= 0.25 or turn_dur >= soft_turn_duration_s))
        duration_capped = (turn_dur >= max_turn_duration_s)

        should_split = speaker_changed or pause_exceeded or sentence_ended or duration_capped

        if should_split:
            turn_start = round(current_words[0].start, 3)
            turn_end = round(current_words[-1].end, 3)
            turn = RawTurn(
                speaker=current_speaker,
                start=turn_start,
                end=max(turn_start + 0.05, turn_end),
                text=" ".join(item.word for item in current_words).strip(),
                words=list(current_words),
                turn_type="speech",
            )
            turns.append(turn)

            current_words = [w]
            current_speaker = w_spk
        else:
            current_words.append(w)

    if current_words:
        turn_start = round(current_words[0].start, 3)
        turn_end = round(current_words[-1].end, 3)
        turn = RawTurn(
            speaker=current_speaker,
            start=turn_start,
            end=max(turn_start + 0.05, turn_end),
            text=" ".join(item.word for item in current_words).strip(),
            words=list(current_words),
            turn_type="speech",
        )
        turns.append(turn)

    return turns



def assemble_interleaved_dialogue(
    per_speaker_turns: Union[Dict[str, List[RawTurn]], List[RawTurn]],
    backchannel_duration_s: float = 0.8,
) -> List[RawTurn]:
    """Interleaves turns from all speakers chronologically and tags acoustic backchannels.

    Accepts either:
    - Dict[str, List[RawTurn]] (turns grouped by speaker)
    - List[RawTurn] (flat list of turns)

    A backchannel is identified when:
    - The turn duration is <= backchannel_duration_s (e.g., short "Yeah", "Right", "Uh-huh").
    - Another speaker was holding the floor immediately before.
    - The original speaker continues speaking shortly thereafter (floor was not yielded).
    """
    all_turns: List[RawTurn] = []
    if isinstance(per_speaker_turns, dict):
        for spk_turns in per_speaker_turns.values():
            all_turns.extend(spk_turns)
    else:
        all_turns = list(per_speaker_turns)

    all_turns.sort(key=lambda t: (t.start, t.end))

    # Tag backchannel conversational dynamics
    for idx, turn in enumerate(all_turns):
        if idx > 0 and idx < len(all_turns) - 1:
            prev_t = all_turns[idx - 1]
            next_t = all_turns[idx + 1]

            # Speaker B interjected briefly while Speaker A was speaking or resuming floor
            if (
                turn.duration <= backchannel_duration_s
                and turn.speaker != prev_t.speaker
                and next_t.speaker == prev_t.speaker
                and turn.start < prev_t.end + 0.5
            ):
                turn.turn_type = "backchannel"

    return all_turns


def reattribute_turns_by_timbre(
    turns: List[RawTurn],
    mono_wav: torch.Tensor,
    sr: int = 16000,
    margin: float = 0.0,
) -> List[RawTurn]:
    """Reattribute turns whose acoustic profile strongly matches another speaker.

    Guards against Pyannote boundary lag when speakers exchange rapid dialogue.
    """
    if not turns or mono_wav.ndim < 1:
        return turns

    wav_1d = mono_wav.squeeze(0).cpu().float()
    total_samples = wav_1d.shape[-1]

    # Collect reference profiles per speaker from clean turns (duration >= 2.0s)
    from app.engine.overlap_separator import extract_acoustic_profile

    spk_profiles: dict[str, list[torch.Tensor]] = {}
    for t in turns:
        if t.duration >= 2.0:
            s_idx = max(0, int(t.start * sr))
            e_idx = min(total_samples, int(t.end * sr))
            if e_idx - s_idx >= int(sr * 0.5):
                prof = extract_acoustic_profile(wav_1d[s_idx:e_idx], sr)
                spk_profiles.setdefault(t.speaker, []).append(prof)

    if len(spk_profiles) < 2:
        return turns

    # Average reference profile per speaker
    avg_profiles: dict[str, torch.Tensor] = {}
    for spk, plist in spk_profiles.items():
        stacked = torch.stack(plist, dim=0).mean(dim=0)
        norm = torch.linalg.norm(stacked).clamp(min=1e-8)
        avg_profiles[spk] = stacked / norm

    # Reattribute any turn where acoustic similarity to another speaker exceeds current by margin
    for t in turns:
        s_idx = max(0, int(t.start * sr))
        e_idx = min(total_samples, int(t.end * sr))
        if (e_idx - s_idx) < int(sr * 0.4):
            continue

        turn_prof = extract_acoustic_profile(wav_1d[s_idx:e_idx], sr)
        current_spk = t.speaker
        if current_spk not in avg_profiles:
            continue

        curr_sim = float(torch.cosine_similarity(turn_prof.unsqueeze(0), avg_profiles[current_spk].unsqueeze(0)).item())

        best_spk = current_spk
        best_sim = curr_sim

        for other_spk, other_prof in avg_profiles.items():
            if other_spk == current_spk:
                continue
            sim = float(torch.cosine_similarity(turn_prof.unsqueeze(0), other_prof.unsqueeze(0)).item())
            if sim > best_sim + margin:
                best_sim = sim
                best_spk = other_spk

        if best_spk != current_spk:
            logger.info(
                "Acoustic reattribution: turn [%.2fs -> %.2fs] '%s' reattributed from %s to %s (sim: %.4f vs %.4f)",
                t.start,
                t.end,
                t.text,
                current_spk,
                best_spk,
                curr_sim,
                best_sim,
            )
            t.speaker = best_spk

    return turns
