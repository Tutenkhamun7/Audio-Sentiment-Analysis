"""Response DTOs for the app API service."""

from __future__ import annotations

from typing import Dict, List, Optional
from pydantic import BaseModel, Field


class WordTiming(BaseModel):
    """Millisecond-level word timing and attribution."""

    word: str
    start: float
    end: float
    confidence: float = 1.0
    speaker: Optional[str] = None


class EmotionAnalysis(BaseModel):
    """Acoustic and textual sentiment attributes for a turn."""

    dominant_emotion: Optional[str] = None
    scores: Dict[str, float] = Field(default_factory=dict)
    valence: Optional[float] = None
    arousal: Optional[float] = None
    is_sarcastic: bool = False
    incongruence_score: float = 0.0
    abstained: bool = False
    abstain_reason: Optional[str] = None


class SentimentAnalysis(BaseModel):
    """Semantic text sentiment and affect classification."""

    label: str = "neutral"
    score: float = 1.0
    scores: Dict[str, float] = Field(default_factory=dict)


class TurnDetail(BaseModel):
    """Complete speaker turn with text, timing, conversational dynamics, and affect."""

    turn_id: str
    speaker: str
    start: float
    end: float
    duration: float
    text: Optional[str] = None
    words: List[WordTiming] = Field(default_factory=list)
    emotion: EmotionAnalysis = Field(default_factory=EmotionAnalysis)
    sentiment: Optional[SentimentAnalysis] = None
    is_interruption: bool = False
    interrupted_by: Optional[str] = None
    interrupts: Optional[str] = None
    overlap_ratio: float = 0.0
    turn_type: str = "speech"
    is_sarcastic: bool = False


class CallMetrics(BaseModel):
    """Aggregate statistics for the entire conversation."""

    duration_s: float
    total_speech_s: float
    overlap_s: float
    overlap_ratio: float
    interruption_count: int
    speaker_count: int
    branch_used: str


class AnalyzeResponse(BaseModel):
    """Unified API response for an audio analysis call."""

    call_id: str
    metrics: CallMetrics
    turns: List[TurnDetail]
    speakers: List[str]
    processing_time_s: float


class ActionItem(BaseModel):
    """Action item or agreed task from the call."""

    task: str
    owner: str = "Agent"
    priority: str = "Medium"  # "High" | "Medium" | "Low"


class SentimentTrajectory(BaseModel):
    """Customer mood and emotional transition trajectory across the dialogue."""

    start_emotion: str = "neutral"
    end_emotion: str = "neutral"
    trend: str = "stable"  # "improved" | "declined" | "stable" | "volatile"
    details: str = ""


class AgentPerformanceQA(BaseModel):
    """Quality assurance and agent behavioral scorecard."""

    empathy_score: float = 8.0  # 0.0 - 10.0
    professionalism_score: float = 8.5  # 0.0 - 10.0
    active_listening_score: float = 8.0  # 0.0 - 10.0
    overall_score: float = 8.2  # 0.0 - 10.0
    strengths: List[str] = Field(default_factory=list)
    areas_for_improvement: List[str] = Field(default_factory=list)


class RiskAssessment(BaseModel):
    """Call escalation and customer churn risk indicators."""

    escalation_risk: str = "Low"  # "High" | "Medium" | "Low"
    churn_risk: str = "Low"  # "High" | "Medium" | "Low"
    risk_factors: List[str] = Field(default_factory=list)


class KeyMoment(BaseModel):
    """Highlighted conversational turn with timestamp and insight."""

    timestamp: str  # e.g. "00:14"
    time_seconds: float = 0.0
    speaker: str = "Speaker 1"
    quote: str
    insight: str
    sentiment: Optional[str] = None


class PostCallAnalysisResponse(BaseModel):
    """LLM-generated comprehensive post-call analytical synthesis."""

    call_id: str
    executive_summary: str
    call_outcome: str  # "Resolved" | "Pending Follow-up" | "Escalated" | "Inquiry Answered"
    outcome_reasoning: str
    customer_satisfaction_score: float = 8.0  # 0.0 - 10.0
    sentiment_trajectory: SentimentTrajectory
    agent_performance: AgentPerformanceQA
    action_items: List[ActionItem] = Field(default_factory=list)
    risk_assessment: RiskAssessment
    key_moments: List[KeyMoment] = Field(default_factory=list)
    llm_model_used: str = "qwen/qwen3.8-27b:free"
    generated_at: str = ""
