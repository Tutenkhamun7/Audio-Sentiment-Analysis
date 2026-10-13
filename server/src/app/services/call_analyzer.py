"""LLM-powered Post-Call Analysis Service using OpenRouter or local intelligence."""

from __future__ import annotations

from datetime import datetime, timezone
import json
import logging
from typing import Any, Dict, List, Optional
import httpx

from app.core.config import settings
from app.schemas.request import PostAnalysisRequest
from app.schemas.response import (
    ActionItem,
    AgentPerformanceQA,
    KeyMoment,
    PostCallAnalysisResponse,
    RiskAssessment,
    SentimentTrajectory,
)

logger = logging.getLogger(__name__)


class CallAnalyzerService:
    """Orchestrates comprehensive post-call analytical synthesis via LLM."""

    def __init__(self) -> None:
        self.api_key = settings.openrouter_api_key
        self.base_url = settings.openrouter_base_url.rstrip("/")
        self.model = settings.llm_model

    def _format_time_str(self, seconds: float) -> str:
        """Format seconds into MM:SS string."""
        mins = int(seconds // 60)
        secs = int(seconds % 60)
        return f"{mins:02d}:{secs:02d}"

    def _build_dialogue_context(
        self,
        turns: List[Dict[str, Any]],
        speakers: List[str],
        metrics: Optional[Dict[str, Any]],
        overall_transcript: Optional[str] = None,
    ) -> str:
        """Format dialogue turns and metrics into prompt context."""
        lines = []

        if metrics:
            lines.append("=== CONVERSATION METRICS ===")
            lines.append(f"Duration: {metrics.get('duration_s', 0)}s")
            lines.append(f"Total Speech: {metrics.get('total_speech_s', 0)}s")
            lines.append(f"Overlap Collisions: {metrics.get('overlap_s', 0)}s ({metrics.get('overlap_ratio', 0) * 100:.1f}%)")
            lines.append(f"Interruption Count: {metrics.get('interruption_count', 0)}")
            lines.append(f"Speakers Identified: {', '.join(speakers)}")
            lines.append("")

        lines.append("=== CONVERSATION TURNS & MULTIMODAL AFFECT ===")
        for idx, t in enumerate(turns):
            spk = t.get("speaker", f"Speaker {idx+1}")
            start = float(t.get("start", t.get("start_time", 0.0)))
            end = float(t.get("end", t.get("end_time", 0.0)))
            text = t.get("text", "").strip()

            # Emotion attributes
            emotion_obj = t.get("emotion") or {}
            acoustic_em = emotion_obj.get("dominant_emotion")
            if not acoustic_em and isinstance(t.get("acoustic_emotion"), list) and t.get("acoustic_emotion"):
                acoustic_em = t["acoustic_emotion"][0].get("label")

            sentiment_obj = t.get("sentiment") or {}
            semantic_em = sentiment_obj.get("label")
            if not semantic_em and isinstance(t.get("semantic_emotion"), list) and t.get("semantic_emotion"):
                semantic_em = t["semantic_emotion"][0].get("label")

            affect_notes = []
            if acoustic_em:
                affect_notes.append(f"Acoustic Prosody: {acoustic_em}")
            if semantic_em:
                affect_notes.append(f"Text Sentiment: {semantic_em}")
            if t.get("is_sarcastic") or t.get("is_conflict"):
                affect_notes.append("WARNING: Cross-modal conflict / sarcasm detected")
            if t.get("is_interruption"):
                interrupted_by = t.get("interrupted_by") or "Other speaker"
                affect_notes.append(f"Interrupted by: {interrupted_by}")

            affect_str = f" [{', '.join(affect_notes)}]" if affect_notes else ""
            timestamp = f"{self._format_time_str(start)}-{self._format_time_str(end)}"
            lines.append(f"Turn #{idx+1} ({timestamp}) {spk}: \"{text}\"{affect_str}")

        if overall_transcript and not turns:
            lines.append("=== TRANSCRIPT ===")
            lines.append(overall_transcript)

        return "\n".join(lines)

    async def _call_openrouter_llm(self, context_prompt: str, custom_instructions: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """Call OpenRouter LLM and extract structured JSON."""
        if not self.api_key:
            logger.info("OpenRouter API key not provided; falling back to deterministic analyzer.")
            return None

        system_instruction = (
            "You are an expert Call Quality & Conversational Affect Analyst. "
            "Analyze the provided multi-turn customer/agent audio conversation. "
            "Evaluate acoustic prosody tone, verbal sentiment, conversational dynamics, resolutions, and quality assurance. "
            "Respond ONLY with a valid JSON object matching the exact schema specified."
        )

        schema_format = """
{
  "executive_summary": "Comprehensive 2-4 sentence executive overview of the conversation, caller need, and resolution.",
  "call_outcome": "Resolved" | "Pending Follow-up" | "Escalated" | "Inquiry Answered",
  "outcome_reasoning": "Clear explanation of why this outcome was assigned.",
  "customer_satisfaction_score": 8.5, // Float between 0.0 and 10.0
  "sentiment_trajectory": {
    "start_emotion": "e.g. Frustrated",
    "end_emotion": "e.g. Relieved / Satisfied",
    "trend": "improved" | "declined" | "stable" | "volatile",
    "details": "Explanation of customer mood evolution from beginning to end."
  },
  "agent_performance": {
    "empathy_score": 8.5, // Float 0.0 - 10.0
    "professionalism_score": 9.0, // Float 0.0 - 10.0
    "active_listening_score": 8.0, // Float 0.0 - 10.0
    "overall_score": 8.5, // Float 0.0 - 10.0
    "strengths": ["String strength 1", "String strength 2"],
    "areas_for_improvement": ["String area 1", "String area 2"]
  },
  "action_items": [
    {
      "task": "Specific concrete task to follow up on",
      "owner": "Agent" | "Customer" | "Support Team",
      "priority": "High" | "Medium" | "Low"
    }
  ],
  "risk_assessment": {
    "escalation_risk": "Low" | "Medium" | "High",
    "churn_risk": "Low" | "Medium" | "High",
    "risk_factors": ["Factor 1", "Factor 2"]
  },
  "key_moments": [
    {
      "timestamp": "MM:SS",
      "time_seconds": 12.5,
      "speaker": "Speaker Name",
      "quote": "Direct quote",
      "insight": "Why this moment is significant to the call outcome",
      "sentiment": "Emotion or mood tag"
    }
  ]
}
"""

        user_content = f"{context_prompt}\n\n"
        if custom_instructions:
            user_content += f"CUSTOM INSTRUCTION: {custom_instructions}\n\n"
        user_content += f"Required JSON Format:\n{schema_format}\nRespond ONLY with the JSON object."

        headers = {
            "Authorization": f"Bearer {self.api_key.strip()}",
            "Content-Type": "application/json",
            "HTTP-Referer": "http://localhost:5173",
            "X-Title": "Audio Sentiment Analysis Studio",
        }

        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_instruction},
                {"role": "user", "content": user_content},
            ],
            "response_format": {"type": "json_object"},
            "temperature": 0.2,
        }

        url = f"{self.base_url}/chat/completions"

        try:
            async with httpx.AsyncClient(timeout=35.0) as client:
                response = await client.post(url, headers=headers, json=payload)
                if response.status_code != 200:
                    logger.warning(
                        "OpenRouter returned status %s: %s. Falling back to deterministic analyzer.",
                        response.status_code,
                        response.text[:200],
                    )
                    return None

                data = response.json()
                raw_text = data["choices"][0]["message"]["content"].strip()

                # Clean markdown fences if any
                if raw_text.startswith("```"):
                    raw_text = raw_text.split("```")[1]
                    if raw_text.startswith("json"):
                        raw_text = raw_text[4:]
                    raw_text = raw_text.strip()

                parsed = json.loads(raw_text)
                return parsed if isinstance(parsed, dict) else None
        except Exception as err:
            logger.warning("OpenRouter LLM request failed (%s). Falling back to deterministic analyzer.", err)
            return None

    def _generate_deterministic_analysis(
        self,
        turns: List[Dict[str, Any]],
        speakers: List[str],
        metrics: Optional[Dict[str, Any]],
        call_id: str,
    ) -> PostCallAnalysisResponse:
        """Deterministic heuristic analysis when LLM is unavailable or offline."""
        now_iso = datetime.now(timezone.utc).isoformat()
        total_turns = len(turns)

        # Detect conflicts / interruptions
        conflict_turns = [t for t in turns if t.get("is_conflict") or t.get("is_sarcastic")]
        interrupt_turns = [t for t in turns if t.get("is_interruption")]
        interrupt_count = (metrics.get("interruption_count") if metrics else None) or len(interrupt_turns)

        # Classify first vs last turn emotions
        first_turns = turns[:2] if turns else []
        last_turns = turns[-2:] if turns else []

        start_affect = "neutral"
        if first_turns:
            emo = first_turns[0].get("emotion", {}).get("dominant_emotion")
            if emo:
                start_affect = emo.lower()

        end_affect = "neutral"
        if last_turns:
            emo = last_turns[-1].get("emotion", {}).get("dominant_emotion")
            if emo:
                end_affect = emo.lower()

        negative_emotions = {"angry", "sad", "fearful", "disgusted"}
        positive_emotions = {"happy", "pleasant"}

        has_negative_start = start_affect in negative_emotions
        has_positive_end = end_affect in positive_emotions or end_affect == "neutral"

        if has_negative_start and has_positive_end:
            trend = "improved"
            outcome = "Resolved"
            reasoning = "Caller started with elevated emotional distress or urgency but concluded neutrally/positively after interaction."
            csat = 8.2
        elif end_affect in negative_emotions:
            trend = "declined"
            outcome = "Pending Follow-up" if interrupt_count < 3 else "Escalated"
            reasoning = "Negative affect remained present toward the end of the call, indicating unaddressed friction or ongoing issue."
            csat = 4.5
        else:
            trend = "stable"
            outcome = "Inquiry Answered"
            reasoning = "Conversation maintained a calm, steady baseline throughout with structured exchange of information."
            csat = 8.5

        # Scores
        empathy_score = max(5.0, min(9.5, 8.5 - len(conflict_turns) * 0.8))
        prof_score = max(5.5, min(9.8, 9.0 - interrupt_count * 0.5))
        listening_score = max(5.0, min(9.5, 8.8 - interrupt_count * 0.6))
        overall_score = round((empathy_score + prof_score + listening_score) / 3.0, 1)

        strengths = [
            f"Successfully addressed {total_turns} conversational turns with consistent communication.",
            "Maintained active participation without conversational stalls.",
        ]
        if prof_score >= 8.0:
            strengths.append("Professional tone maintained throughout speaker handovers.")

        areas_for_improvement = []
        if interrupt_count > 0:
            areas_for_improvement.append(f"Reduce overtalk interruptions ({interrupt_count} collision events detected).")
        if conflict_turns:
            areas_for_improvement.append("Address subtle vocal prosody friction or potential sarcastic subtext promptly.")
        if not areas_for_improvement:
            areas_for_improvement.append("Continue maintaining high standard of customer rapport.")

        # Key Moments
        key_moments = []
        for idx, t in enumerate(turns):
            start = float(t.get("start", t.get("start_time", 0.0)))
            text = t.get("text", "").strip()
            if not text:
                continue

            spk = t.get("speaker", "Speaker")
            if t.get("is_conflict") or t.get("is_sarcastic"):
                key_moments.append(
                    KeyMoment(
                        timestamp=self._format_time_str(start),
                        time_seconds=round(start, 2),
                        speaker=spk,
                        quote=text[:80],
                        insight="Prosody-semantic dissonance detected; tonal prosody diverged from literal words.",
                        sentiment="Incongruence",
                    )
                )
            elif idx == 0:
                key_moments.append(
                    KeyMoment(
                        timestamp=self._format_time_str(start),
                        time_seconds=round(start, 2),
                        speaker=spk,
                        quote=text[:80],
                        insight="Call opening statement and initial intent formulation.",
                        sentiment=start_affect.capitalize(),
                    )
                )
            elif t.get("is_interruption"):
                key_moments.append(
                    KeyMoment(
                        timestamp=self._format_time_str(start),
                        time_seconds=round(start, 2),
                        speaker=spk,
                        quote=text[:80],
                        insight="Turn collision / interruption during dialogue handover.",
                        sentiment="Interruption",
                    )
                )

        if not key_moments and turns:
            first_t = turns[0]
            start = float(first_t.get("start", first_t.get("start_time", 0.0)))
            key_moments.append(
                KeyMoment(
                    timestamp=self._format_time_str(start),
                    time_seconds=round(start, 2),
                    speaker=first_t.get("speaker", "Speaker 1"),
                    quote=first_t.get("text", "")[:80],
                    insight="Initial opening exchange of the conversation.",
                    sentiment="Neutral",
                )
            )

        # Action items
        action_items = [
            ActionItem(
                task="Document call interaction summary and confirm customer record is updated.",
                owner="Agent",
                priority="Medium",
            )
        ]
        if outcome in {"Pending Follow-up", "Escalated"}:
            action_items.append(
                ActionItem(
                    task="Schedule customer follow-up to verify full resolution of requested issue.",
                    owner="Support Team",
                    priority="High",
                )
            )

        risk_level = "High" if outcome == "Escalated" else ("Medium" if outcome == "Pending Follow-up" else "Low")
        risk_factors = []
        if conflict_turns:
            risk_factors.append(f"{len(conflict_turns)} cross-modal tone/word dissonance turns identified")
        if interrupt_count > 1:
            risk_factors.append(f"{interrupt_count} conversational interruptions detected")
        if not risk_factors:
            risk_factors.append("No critical risk indicators detected during this dialogue.")

        speaker_list_str = ", ".join(speakers) if speakers else "identified speakers"
        summary = (
            f"Analyzed {total_turns} conversational turns between {speaker_list_str}. "
            f"The interaction concluded as '{outcome}' with an estimated CSAT of {csat}/10. "
            f"Dialogue exhibited a {trend} emotional progression with {interrupt_count} conversational interruptions."
        )

        return PostCallAnalysisResponse(
            call_id=call_id,
            executive_summary=summary,
            call_outcome=outcome,
            outcome_reasoning=reasoning,
            customer_satisfaction_score=csat,
            sentiment_trajectory=SentimentTrajectory(
                start_emotion=start_affect.capitalize(),
                end_emotion=end_affect.capitalize(),
                trend=trend,
                details=f"Transitioned from initial {start_affect} tone to {end_affect} conclusion.",
            ),
            agent_performance=AgentPerformanceQA(
                empathy_score=empathy_score,
                professionalism_score=prof_score,
                active_listening_score=listening_score,
                overall_score=overall_score,
                strengths=strengths,
                areas_for_improvement=areas_for_improvement,
            ),
            action_items=action_items,
            risk_assessment=RiskAssessment(
                escalation_risk=risk_level,
                churn_risk="Medium" if risk_level != "Low" else "Low",
                risk_factors=risk_factors,
            ),
            key_moments=key_moments[:5],
            llm_model_used="deterministic_analyzer",
            generated_at=now_iso,
        )

    async def analyze_call(self, request_data: PostAnalysisRequest) -> PostCallAnalysisResponse:
        """Main analysis pipeline: attempts LLM reasoning with deterministic fallback."""
        turns = request_data.turns or []
        speakers = request_data.speakers or []
        metrics = request_data.metrics
        call_id = request_data.call_id or "call"

        dialogue_context = self._build_dialogue_context(
            turns=turns,
            speakers=speakers,
            metrics=metrics,
            overall_transcript=request_data.overall_transcript,
        )

        llm_result = await self._call_openrouter_llm(
            context_prompt=dialogue_context,
            custom_instructions=request_data.custom_instructions,
        )

        now_iso = datetime.now(timezone.utc).isoformat()

        if llm_result and isinstance(llm_result, dict):
            try:
                # Safely parse trajectory
                traj_raw = llm_result.get("sentiment_trajectory") or {}
                trajectory = SentimentTrajectory(
                    start_emotion=str(traj_raw.get("start_emotion", "Neutral")),
                    end_emotion=str(traj_raw.get("end_emotion", "Neutral")),
                    trend=str(traj_raw.get("trend", "stable")),
                    details=str(traj_raw.get("details", "")),
                )

                # Safely parse QA
                qa_raw = llm_result.get("agent_performance") or {}
                agent_qa = AgentPerformanceQA(
                    empathy_score=float(qa_raw.get("empathy_score", 8.0)),
                    professionalism_score=float(qa_raw.get("professionalism_score", 8.5)),
                    active_listening_score=float(qa_raw.get("active_listening_score", 8.0)),
                    overall_score=float(qa_raw.get("overall_score", 8.2)),
                    strengths=list(qa_raw.get("strengths", [])),
                    areas_for_improvement=list(qa_raw.get("areas_for_improvement", [])),
                )

                # Safely parse Action Items
                actions = []
                for a in llm_result.get("action_items", []):
                    if isinstance(a, dict) and "task" in a:
                        actions.append(
                            ActionItem(
                                task=str(a.get("task")),
                                owner=str(a.get("owner", "Agent")),
                                priority=str(a.get("priority", "Medium")),
                            )
                        )

                # Safely parse Risk
                risk_raw = llm_result.get("risk_assessment") or {}
                risk = RiskAssessment(
                    escalation_risk=str(risk_raw.get("escalation_risk", "Low")),
                    churn_risk=str(risk_raw.get("churn_risk", "Low")),
                    risk_factors=list(risk_raw.get("risk_factors", [])),
                )

                # Safely parse Key Moments
                key_moments = []
                for m in llm_result.get("key_moments", []):
                    if isinstance(m, dict) and "quote" in m:
                        key_moments.append(
                            KeyMoment(
                                timestamp=str(m.get("timestamp", "00:00")),
                                time_seconds=float(m.get("time_seconds", 0.0)),
                                speaker=str(m.get("speaker", "Speaker")),
                                quote=str(m.get("quote", "")),
                                insight=str(m.get("insight", "")),
                                sentiment=m.get("sentiment"),
                            )
                        )

                return PostCallAnalysisResponse(
                    call_id=call_id,
                    executive_summary=str(llm_result.get("executive_summary", "")),
                    call_outcome=str(llm_result.get("call_outcome", "Resolved")),
                    outcome_reasoning=str(llm_result.get("outcome_reasoning", "")),
                    customer_satisfaction_score=float(llm_result.get("customer_satisfaction_score", 8.0)),
                    sentiment_trajectory=trajectory,
                    agent_performance=agent_qa,
                    action_items=actions,
                    risk_assessment=risk,
                    key_moments=key_moments,
                    llm_model_used=self.model,
                    generated_at=now_iso,
                )
            except Exception as parse_err:
                logger.warning("Failed to parse LLM structured output (%s); falling back to deterministic.", parse_err)

        # Fallback to deterministic analyzer
        return self._generate_deterministic_analysis(
            turns=turns,
            speakers=speakers,
            metrics=metrics,
            call_id=call_id,
        )
