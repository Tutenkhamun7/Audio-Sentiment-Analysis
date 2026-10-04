from typing import List, Optional
from app.interfaces import BaseASR, RawSegment, WordTimestamp


class MockASRAdapter(BaseASR):
    """Mock ASR adapter for testing without loading neural models."""

    def __init__(self, predefined_segments: Optional[List[RawSegment]] = None):
        self.predefined_segments = predefined_segments or [
            RawSegment(
                start=0.0,
                end=3.0,
                text="Hello world! How are you?",
                words=[
                    WordTimestamp(start=0.0, end=0.8, word="Hello"),
                    WordTimestamp(start=0.8, end=1.5, word="world!"),
                    WordTimestamp(start=1.6, end=2.0, word="How"),
                    WordTimestamp(start=2.0, end=2.4, word="are"),
                    WordTimestamp(start=2.4, end=3.0, word="you?"),
                ],
            )
        ]

    def transcribe(
        self,
        file_path: str,
        beam_size: int = 5,
        language: Optional[str] = "en",
        **kwargs,
    ) -> List[RawSegment]:
        return self.predefined_segments
