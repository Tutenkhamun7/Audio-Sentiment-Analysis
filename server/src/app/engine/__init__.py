"""Deep learning and DSP processing engines for app."""

from app.engine.acoustic import extract_turn_emotions
from app.engine.aligner import align_and_snap_turn_boundaries, assign_words_to_speakers
from app.engine.branch import analyze_channels, decide_branch
from app.engine.diarizer import DiarizationResult, RawTurn, run_diarization
from app.engine.overlap_separator import separate_overlaps_in_memory
from app.engine.segmenter import assemble_interleaved_dialogue, group_aligned_words_into_turns, group_words_into_turns
from app.engine.semantic import analyze_text_sentiment, analyze_texts_batch
from app.engine.transcriber import transcribe_speaker_turns, transcribe_stream_words

__all__ = [
    "extract_turn_emotions",
    "align_and_snap_turn_boundaries",
    "assign_words_to_speakers",
    "analyze_channels",
    "decide_branch",
    "DiarizationResult",
    "RawTurn",
    "run_diarization",
    "separate_overlaps_in_memory",
    "analyze_text_sentiment",
    "analyze_texts_batch",
    "transcribe_speaker_turns",
    "transcribe_stream_words",
    "group_words_into_turns",
    "group_aligned_words_into_turns",
    "assemble_interleaved_dialogue",
]
