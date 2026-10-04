from typing import Any, Dict, List, Optional, Tuple
from app.interfaces import OverlapInterval, RawSegment, SpeakerInterval


class AlignmentService:
    """
    Pure domain service responsible for:
      1. Associating transcribed words with chronological speaker intervals.
      2. Splitting monologues into natural sentence / clause bounds (e.g. 2s - 6s).
      3. Merging microscopic conversational splinters within identical speaker turns.
    """

    @staticmethod
    def find_speaker_for_timestamp(
        midpoint: float, speaker_intervals: List[SpeakerInterval]
    ) -> str:
        """Finds which speaker was active at a given word midpoint timestamp."""
        for interval in speaker_intervals:
            if interval.start <= midpoint <= interval.end:
                return interval.speaker

        # Fallback: nearest interval if midpoint falls into a boundary micro-gap
        if speaker_intervals:
            closest = min(
                speaker_intervals,
                key=lambda x: min(abs(midpoint - x.start), abs(midpoint - x.end)),
            )
            return closest.speaker

        return "SPEAKER_00"

    @classmethod
    def find_speaker_for_word(
        cls,
        start: float,
        end: float,
        speaker_intervals: List[SpeakerInterval],
        fallback_speaker: Optional[str] = None,
    ) -> str:
        """
        Determines the active speaker for a word using temporal interval overlap.
        If the word overlaps with one or more speaker intervals, the speaker with the
        maximum duration of overlap is selected.
        If no interval overlaps (e.g. speech in a boundary gap), it selects the closest
        interval within 1.0 second, or falls back to the previous speaker.
        """
        if not speaker_intervals:
            return fallback_speaker or "SPEAKER_00"

        # 1. Calculate overlap duration with each interval
        speaker_overlaps: Dict[str, float] = {}
        for inv in speaker_intervals:
            overlap = max(0.0, min(end, inv.end) - max(start, inv.start))
            if overlap > 0.0:
                speaker_overlaps[inv.speaker] = speaker_overlaps.get(inv.speaker, 0.0) + overlap

        if speaker_overlaps:
            return max(speaker_overlaps.items(), key=lambda x: x[1])[0]

        # 2. If no direct overlap, check closest interval to midpoint
        midpoint = (start + end) / 2.0
        closest = min(
            speaker_intervals,
            key=lambda x: min(abs(midpoint - x.start), abs(midpoint - x.end)),
        )
        dist = min(abs(midpoint - closest.start), abs(midpoint - closest.end))
        if dist <= 1.0:
            return closest.speaker

        return fallback_speaker or closest.speaker

    def align_words_to_speakers(
        self,
        raw_segments: List[RawSegment],
        speaker_intervals: List[SpeakerInterval],
        max_duration: float = 6.0,
        min_split_duration: float = 2.0,
    ) -> List[Dict]:
        """
        Aligns words to speaker intervals and chunks consecutive speech into bounded turns.
        Splits when:
          - Speaker identity changes.
          - Previous word completed a sentence ('.', '!', '?') and chunk reached min_split_duration.
          - Chunk duration reached max_duration.
          - Conversational pause (gap >= 0.8s) occurred and chunk reached min_split_duration.
        """
        aligned_words: List[Dict[str, Any]] = []
        last_speaker = None

        for seg in raw_segments:
            if not seg.words:
                speaker = self.find_speaker_for_word(
                    seg.start, seg.end, speaker_intervals, fallback_speaker=last_speaker
                )
                last_speaker = speaker
                aligned_words.append(
                    {
                        "start": seg.start,
                        "end": seg.end,
                        "word": seg.text.strip(),
                        "speaker": speaker,
                    }
                )
                continue

            for word_obj in seg.words:
                word_clean = word_obj.word.strip()
                if not word_clean:
                    continue
                speaker = self.find_speaker_for_word(
                    word_obj.start, word_obj.end, speaker_intervals, fallback_speaker=last_speaker
                )
                last_speaker = speaker
                aligned_words.append(
                    {
                        "start": word_obj.start,
                        "end": word_obj.end,
                        "word": word_clean,
                        "speaker": speaker,
                    }
                )

        if not aligned_words:
            return []

        # Smooth isolated single-word jitter (A, B, A -> A, A, A when B is an isolated artifact)
        if len(aligned_words) >= 3:
            for i in range(1, len(aligned_words) - 1):
                prev_w = aligned_words[i - 1]
                curr_w = aligned_words[i]
                next_w = aligned_words[i + 1]

                if (
                    prev_w["speaker"] == next_w["speaker"]
                    and curr_w["speaker"] != prev_w["speaker"]
                ):
                    gap_before = float(curr_w["start"]) - float(prev_w["end"])
                    gap_after = float(next_w["start"]) - float(curr_w["end"])
                    curr_duration = float(curr_w["end"]) - float(curr_w["start"])
                    if gap_before <= 0.6 and gap_after <= 0.6 and curr_duration <= 0.8:
                        curr_w["speaker"] = prev_w["speaker"]

        # Refine speaker transition boundaries using linguistic cues (punctuation & capitalization)
        aligned_words = self.refine_boundary_word_alignment(aligned_words)

        speaker_turns: List[Dict] = []
        current_turn: Dict = {}
        terminal_punctuation = {".", "!", "?"}
        abbreviations = {
            "mr.",
            "mrs.",
            "ms.",
            "dr.",
            "prof.",
            "sr.",
            "jr.",
            "vs.",
            "etc.",
            "st.",
            "co.",
            "corp.",
            "ltd.",
        }

        for word_dict in aligned_words:
            if not current_turn:
                current_turn = {
                    "start": word_dict["start"],
                    "end": word_dict["end"],
                    "speaker": word_dict["speaker"],
                    "text": word_dict["word"],
                }
                continue

            same_speaker = word_dict["speaker"] == current_turn["speaker"]
            current_duration = float(current_turn["end"]) - float(current_turn["start"])
            word_gap = float(word_dict["start"]) - float(current_turn["end"])

            prev_text = current_turn["text"].rstrip()
            last_word = prev_text.split()[-1].lower() if prev_text else ""
            prev_ended_sentence = bool(
                prev_text
                and prev_text[-1] in terminal_punctuation
                and last_word not in abbreviations
            )

            # Significant conversational pause (>= 1.2s, or >= 0.8s if turn reached min_split_duration)
            is_conversational_pause = word_gap >= 1.2 or (
                word_gap >= 0.8 and current_duration >= min_split_duration
            )

            should_split = (
                not same_speaker
                or is_conversational_pause
                or (prev_ended_sentence and current_duration >= min_split_duration)
                or (current_duration >= max_duration)
            )

            if should_split:
                speaker_turns.append(current_turn)
                current_turn = {
                    "start": word_dict["start"],
                    "end": word_dict["end"],
                    "speaker": word_dict["speaker"],
                    "text": word_dict["word"],
                }
            else:
                current_turn["end"] = word_dict["end"]
                current_turn["text"] += f" {word_dict['word']}"

        if current_turn:
            speaker_turns.append(current_turn)

        return speaker_turns

    def merge_segments_by_speaker(
        self,
        turns: List[Dict],
        min_duration: float = 2.0,
        max_gap: float = 0.5,
        max_duration: float = 6.0,
    ) -> List[Dict]:
        """
        Merges short conversational splinters (< min_duration) ONLY if the speaker
        identity is the same, pause between utterances is <= max_gap, and combined
        duration does not exceed max_duration.
        """
        if not turns:
            return []

        merged: List[Dict] = []
        current: Dict = {}

        for turn in turns:
            if not current:
                current = dict(turn)
                continue

            gap = turn["start"] - current["end"]
            current_duration = current["end"] - current["start"]
            combined_duration = turn["end"] - current["start"]
            same_speaker = turn["speaker"] == current["speaker"]

            if (
                same_speaker
                and current_duration < min_duration
                and gap <= max_gap
                and combined_duration <= max_duration
            ):
                current["end"] = turn["end"]
                current["text"] += f" {turn['text']}"
            else:
                merged.append(current)
                current = dict(turn)

        if current:
            merged.append(current)

        return merged

    @staticmethod
    def refine_boundary_word_alignment(aligned_words: List[Dict]) -> List[Dict]:
        """
        Heals acoustic diarization boundary jitter using Whisper's syntactic cues
        (punctuation and capitalization):
        1. Trailing Word Recovery: If a word tagged with the new speaker actually completes
           the previous speaker's unclosed sentence (e.g. 'at' [spk0] -> 'it.' [spk1] -> 'Thank' [spk1]),
           reassign the boundary word back to the previous speaker.
        2. Leading Word Recovery: If a word tagged with the old speaker actually begins
           the next speaker's new sentence (e.g. 'number.' [spk0] -> 'I' [spk0] -> 'will' [spk1]),
           reassign the boundary word forward to the next speaker.
        """
        if len(aligned_words) < 3:
            return aligned_words

        terminal_punct = {".", "!", "?"}

        for i in range(1, len(aligned_words) - 1):
            prev_w = aligned_words[i - 1]
            curr_w = aligned_words[i]
            next_w = aligned_words[i + 1]

            prev_text = prev_w["word"].strip()
            curr_text = curr_w["word"].strip()
            next_text = next_w["word"].strip()

            # Case 1: Trailing word leak (belongs to PREVIOUS speaker)
            # prev_w is Speaker A, curr_w is Speaker B, next_w is Speaker B
            if prev_w["speaker"] != curr_w["speaker"] and curr_w["speaker"] == next_w["speaker"]:
                gap_to_prev = curr_w["start"] - prev_w["end"]
                if gap_to_prev <= 0.6:
                    prev_ended = bool(prev_text and prev_text[-1] in terminal_punct)
                    curr_ended = bool(curr_text and curr_text[-1] in terminal_punct)
                    next_capitalized = bool(next_text and next_text[0].isupper())

                    if not prev_ended and curr_ended and next_capitalized:
                        curr_w["speaker"] = prev_w["speaker"]
                        continue

            # Case 2: Leading word leak (belongs to NEXT speaker)
            # prev_w is Speaker A, curr_w is Speaker A, next_w is Speaker B
            if prev_w["speaker"] == curr_w["speaker"] and curr_w["speaker"] != next_w["speaker"]:
                gap_to_next = next_w["start"] - curr_w["end"]
                if gap_to_next <= 0.6:
                    prev_ended = bool(prev_text and prev_text[-1] in terminal_punct)
                    curr_capitalized = bool(curr_text and curr_text[0].isupper())
                    curr_ended = bool(curr_text and curr_text[-1] in terminal_punct)

                    if prev_ended and curr_capitalized and not curr_ended:
                        curr_w["speaker"] = next_w["speaker"]
                        continue

        return aligned_words

    @staticmethod
    def enrich_turns_with_overlaps(
        turns: List[Dict],
        overlap_intervals: Optional[List[OverlapInterval]] = None,
    ) -> List[Dict]:
        """
        Calculates overlapping speech metrics and determines interruption dynamics.
        If an overlap starts during Turn A (who started earlier) and Turn B cuts in,
        Turn B is tagged with is_interruption=True, and Turn A is tagged with interrupted_by=Turn B's speaker.
        """
        for t in turns:
            t["is_interruption"] = False
            t["interrupted_by"] = None
            t["overlap_duration"] = 0.0

        if not overlap_intervals or not turns:
            return turns

        for ov in overlap_intervals:
            intersecting = []
            for t in turns:
                overlap_time = max(0.0, min(t["end"], ov.end) - max(t["start"], ov.start))
                if overlap_time > 0.0:
                    intersecting.append((t, overlap_time))

            if not intersecting:
                continue

            for t, dur in intersecting:
                t["overlap_duration"] = round(t.get("overlap_duration", 0.0) + dur, 2)

            speakers_involved = {t["speaker"] for t, _ in intersecting}
            if len(speakers_involved) >= 2:
                sorted_turns = sorted([t for t, _ in intersecting], key=lambda x: x["start"])
                first_turn = sorted_turns[0]
                for subsequent_turn in sorted_turns[1:]:
                    if subsequent_turn["speaker"] != first_turn["speaker"]:
                        subsequent_turn["is_interruption"] = True
                        first_turn["interrupted_by"] = subsequent_turn["speaker"]

        return turns

    def process(
        self,
        raw_segments: List[RawSegment],
        speaker_intervals: List[SpeakerInterval],
        overlap_intervals: Optional[List[OverlapInterval]] = None,
        min_duration: float = 2.0,
        max_duration: float = 6.0,
        max_gap: float = 0.5,
    ) -> Tuple[List[Dict], List[str]]:
        """Coordinates full alignment, bounded chunking, and overlap enrichment."""
        raw_turns = self.align_words_to_speakers(
            raw_segments,
            speaker_intervals,
            max_duration=max_duration,
            min_split_duration=min_duration,
        )
        consolidated = self.merge_segments_by_speaker(
            raw_turns,
            min_duration=min_duration,
            max_gap=max_gap,
            max_duration=max_duration,
        )
        enriched = self.enrich_turns_with_overlaps(consolidated, overlap_intervals)
        speakers = sorted(list({t["speaker"] for t in enriched}))
        return enriched, speakers
