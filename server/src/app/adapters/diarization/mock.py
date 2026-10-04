from typing import List, Optional, Tuple
from app.interfaces import BaseDiarizer, OverlapInterval, SpeakerInterval


class MockDiarizerAdapter(BaseDiarizer):
    """Mock Diarizer adapter for testing without loading neural models."""

    def __init__(
        self,
        predefined_intervals: Optional[List[SpeakerInterval]] = None,
        predefined_overlaps: Optional[List[OverlapInterval]] = None,
    ):
        self.predefined_intervals = predefined_intervals or [
            SpeakerInterval(start=0.0, end=3.0, speaker="SPEAKER_00")
        ]
        self.predefined_overlaps = predefined_overlaps or []

    def diarize(self, file_path: str) -> List[SpeakerInterval]:
        return self.predefined_intervals

    def diarize_with_overlaps(
        self, file_path: str
    ) -> Tuple[List[SpeakerInterval], List[OverlapInterval]]:
        return self.predefined_intervals, self.predefined_overlaps
