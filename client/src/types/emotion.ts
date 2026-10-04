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
