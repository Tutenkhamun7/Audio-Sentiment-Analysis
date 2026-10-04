# Pluggable model adapters
from app.adapters.emotion.qwen import QwenSemanticAdapter
from app.adapters.emotion.distilroberta import DistilRobertaSemanticAdapter
from app.adapters.emotion.emotion2vec import Emotion2VecAcousticAdapter

__all__ = [
    "QwenSemanticAdapter",
    "DistilRobertaSemanticAdapter",
    "Emotion2VecAcousticAdapter",
]
