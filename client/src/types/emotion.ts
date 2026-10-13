export type StandardEmotion =
  | "HAPPY"
  | "ANGRY"
  | "SAD"
  | "NEUTRAL"
  | "CONFUSED"
  | "FEARFUL"
  | "DISGUSTED"
  | "DISGUST"
  | "SURPRISED"
  | "AMBIGUOUS";

export type EmotionType = StandardEmotion | string;

export interface EmotionScore {
  label: EmotionType;
  original_label: string;
  score: number; // Float 0.0 - 1.0
}

export interface TimelineSegment {
  segment_id: number;
  speaker: string;
  start_time: number; // in seconds (e.g. 1.25)
  end_time: number; // in seconds (e.g. 4.80)
  text: string;
  semantic_emotion: EmotionScore[];
  acoustic_emotion: EmotionScore[];
  is_conflict: boolean;
  is_ambiguous: boolean;
  conflict_detail?: string | null;
  is_interruption?: boolean;
  interrupted_by?: string | null;
  overlap_duration?: number;
}

export interface AgentSynthesis {
  summary: string;
  escalation_detected: boolean;
  primary_speaker_sentiments: Record<string, string>;
  flagged_anomalies: string[];
  interruption_count: number;
  total_overtalk_seconds: number;
  overtalk_ratio: number;
}

export interface AudioAnalysisResult {
  segments: TimelineSegment[];
  filename: string;
  fileSize?: number;
  duration?: number;
  speakers: string[];
  overall_transcript?: string;
  agent_context?: AgentSynthesis | null;
}

export interface ActionItem {
  task: string;
  owner: string;
  priority: "High" | "Medium" | "Low" | string;
}

export interface SentimentTrajectory {
  start_emotion: string;
  end_emotion: string;
  trend: "improved" | "declined" | "stable" | "volatile" | string;
  details: string;
}

export interface AgentPerformanceQA {
  empathy_score: number;
  professionalism_score: number;
  active_listening_score: number;
  overall_score: number;
  strengths: string[];
  areas_for_improvement: string[];
}

export interface RiskAssessment {
  escalation_risk: "High" | "Medium" | "Low" | string;
  churn_risk: "High" | "Medium" | "Low" | string;
  risk_factors: string[];
}

export interface KeyMoment {
  timestamp: string;
  time_seconds: number;
  speaker: string;
  quote: string;
  insight: string;
  sentiment?: string | null;
}

export interface PostCallAnalysis {
  call_id: string;
  executive_summary: string;
  call_outcome: "Resolved" | "Pending Follow-up" | "Escalated" | "Inquiry Answered" | string;
  outcome_reasoning: string;
  customer_satisfaction_score: number;
  sentiment_trajectory: SentimentTrajectory;
  agent_performance: AgentPerformanceQA;
  action_items: ActionItem[];
  risk_assessment: RiskAssessment;
  key_moments: KeyMoment[];
  llm_model_used: string;
  generated_at: string;
}
