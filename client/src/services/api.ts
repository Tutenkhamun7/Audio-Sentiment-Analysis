import type {
  TimelineSegment,
  EmotionScore,
  AgentSynthesis,
  AudioAnalysisResult,
  PostCallAnalysis,
} from "@/types/emotion";
import { ALL_EMOTIONS } from "@/lib/constants";

export const API_BASE_URL =
  import.meta.env.VITE_API_URL || "http://localhost:8000/api/v1/analyze";

export const POST_ANALYSIS_API_URL =
  import.meta.env.VITE_POST_ANALYSIS_URL || "http://localhost:8000/api/v1/post-analysis";

export interface AnalyzeAudioResponse {
  segments: TimelineSegment[];
  speakers: string[];
  total_duration?: number;
  filename: string;
  overall_transcript?: string;
  agent_context?: AgentSynthesis | null;
  is_mock?: boolean;
}

interface RawEmotion {
  label?: string;
  original_label?: string;
  score?: number;
}

interface RawTimelineEvent {
  segment_id?: number;
  turn_id?: string;
  speaker?: string;
  start_time?: number;
  end_time?: number;
  start?: number;
  end?: number;
  duration?: number;
  text?: string;
  semantic_emotion?: RawEmotion[];
  acoustic_emotion?: RawEmotion[];
  emotion?: {
    dominant_emotion?: string;
    scores?: Record<string, number>;
    valence?: number;
    arousal?: number;
    abstained?: boolean;
    abstain_reason?: string;
  };
  sentiment?: {
    label?: string;
    score?: number;
  };
  is_conflict?: boolean;
  is_ambiguous?: boolean;
  conflict_detail?: string | null;
  is_interruption?: boolean;
  interrupted_by?: string | null;
  interrupts?: string | null;
  overlap_duration?: number;
  overlap_ratio?: number;
}

/**
 * Creates a normalized 8-emotion array ensuring all 8 emotions are present
 */
function createFullEmotionDistribution(
  primary: { label: string; score: number },
  secondary?: { label: string; score: number }
): EmotionScore[] {
  let remaining = 1.0 - primary.score - (secondary ? secondary.score : 0);
  if (remaining < 0) remaining = 0.05;

  const others = ALL_EMOTIONS.filter(
    (e) => e !== primary.label && (!secondary || e !== secondary.label)
  );

  const otherScores: Record<string, number> = {};
  let sum = 0;
  others.forEach((e) => {
    const val = Math.random() * 0.1 + 0.01;
    otherScores[e] = val;
    sum += val;
  });

  const scores: EmotionScore[] = [
    {
      label: primary.label,
      original_label: primary.label.toLowerCase(),
      score: Number(primary.score.toFixed(3)),
    },
  ];

  if (secondary) {
    scores.push({
      label: secondary.label,
      original_label: secondary.label.toLowerCase(),
      score: Number(secondary.score.toFixed(3)),
    });
  }

  others.forEach((e) => {
    const scaled = (otherScores[e] / sum) * remaining;
    scores.push({
      label: e,
      original_label: e.toLowerCase(),
      score: Number(Math.max(0.01, scaled).toFixed(3)),
    });
  });

  // Sort descending by score
  return scores.sort((a, b) => b.score - a.score);
}

export const MOCK_ANALYSIS_DATA: TimelineSegment[] = [
  {
    segment_id: 1,
    speaker: "Speaker 1",
    start_time: 0.5,
    end_time: 3.8,
    text: "Good morning team! I'm genuinely thrilled to share the preliminary audio sentiment results.",
    semantic_emotion: createFullEmotionDistribution(
      { label: "HAPPY", score: 0.88 },
      { label: "NEUTRAL", score: 0.06 }
    ),
    acoustic_emotion: createFullEmotionDistribution(
      { label: "HAPPY", score: 0.82 },
      { label: "SURPRISED", score: 0.09 }
    ),
    is_conflict: false,
    is_ambiguous: false,
    conflict_detail: null,
    is_interruption: false,
    interrupted_by: null,
    overlap_duration: 0.0,
  },
  {
    segment_id: 2,
    speaker: "Speaker 2",
    start_time: 4.2,
    end_time: 7.9,
    text: "Oh, that's just fantastic. Another algorithmic model that allegedly understands human nuance.",
    semantic_emotion: createFullEmotionDistribution(
      { label: "HAPPY", score: 0.74 },
      { label: "NEUTRAL", score: 0.18 }
    ),
    acoustic_emotion: createFullEmotionDistribution(
      { label: "ANGRY", score: 0.89 },
      { label: "DISGUSTED", score: 0.07 }
    ),
    is_conflict: true,
    is_ambiguous: false,
    conflict_detail:
      "Acoustic-Semantic Incongruence: Sarcastic dissonance detected. Lexical words indicate high praise ('fantastic'), but prosody displays compressed vocal pitch, glottal tension, and harsh harmonic ratio indicating frustration/cynicism.",
    is_interruption: false,
    interrupted_by: null,
    overlap_duration: 0.0,
  },
  {
    segment_id: 3,
    speaker: "Speaker 1",
    start_time: 8.4,
    end_time: 12.1,
    text: "Wait, are you concerned about cross-dialect prosody variance, or is it the latency?",
    semantic_emotion: createFullEmotionDistribution(
      { label: "CONFUSED", score: 0.79 },
      { label: "FEARFUL", score: 0.12 }
    ),
    acoustic_emotion: createFullEmotionDistribution(
      { label: "CONFUSED", score: 0.75 },
      { label: "NEUTRAL", score: 0.14 }
    ),
    is_conflict: false,
    is_ambiguous: false,
    conflict_detail: null,
    is_interruption: false,
    interrupted_by: null,
    overlap_duration: 0.0,
  },
  {
    segment_id: 4,
    speaker: "Speaker 2",
    start_time: 12.6,
    end_time: 16.8,
    text: "I spent three sleepless nights debugging the pipeline, and frankly, I'm exhausted and deeply disappointed.",
    semantic_emotion: createFullEmotionDistribution(
      { label: "SAD", score: 0.84 },
      { label: "ANGRY", score: 0.11 }
    ),
    acoustic_emotion: createFullEmotionDistribution(
      { label: "SAD", score: 0.91 },
      { label: "NEUTRAL", score: 0.05 }
    ),
    is_conflict: false,
    is_ambiguous: false,
    conflict_detail: null,
    is_interruption: false,
    interrupted_by: null,
    overlap_duration: 0.0,
  },
  {
    segment_id: 5,
    speaker: "Speaker 1",
    start_time: 17.2,
    end_time: 21.0,
    text: "Whoa! Look at this real-time inference latency benchmark — it just dropped to 28 milliseconds!",
    semantic_emotion: createFullEmotionDistribution(
      { label: "SURPRISED", score: 0.86 },
      { label: "HAPPY", score: 0.1 }
    ),
    acoustic_emotion: createFullEmotionDistribution(
      { label: "SURPRISED", score: 0.93 },
      { label: "HAPPY", score: 0.05 }
    ),
    is_conflict: false,
    is_ambiguous: false,
    conflict_detail: null,
    is_interruption: false,
    interrupted_by: null,
    overlap_duration: 0.0,
  },
];

export const MOCK_AGENT_CONTEXT: AgentSynthesis = {
  summary:
    "The conversation exhibits a dynamic emotional progression starting with optimism from Speaker 1, followed by cynical sarcasm from Speaker 2, turning toward frustration and sadness over project challenges, before concluding with surprise at positive benchmark outcomes.",
  escalation_detected: false,
  primary_speaker_sentiments: {
    "Speaker 1": "Predominantly enthusiastic and solution-oriented with high curiosity.",
    "Speaker 2": "Exhausted and critical with sarcastic undertones shifting to relief.",
  },
  flagged_anomalies: [
    "Segment 2: Sarcastic lexical/prosodic contradiction ('fantastic' spoken with compressed pitch and high glottal tension).",
  ],
  interruption_count: 0,
  total_overtalk_seconds: 0.0,
  overtalk_ratio: 0.0,
};

/**
 * Normalizes an emotion score entry
 */
function normalizeEmotion(raw: RawEmotion): EmotionScore {
  const rawLabel = (raw.label || raw.original_label || "NEUTRAL").toUpperCase();
  const label = rawLabel === "DISGUST" ? "DISGUSTED" : rawLabel;
  return {
    label,
    original_label: raw.original_label || raw.label || label.toLowerCase(),
    score: typeof raw.score === "number" ? Math.max(0, Math.min(1, raw.score)) : 0,
  };
}

/**
 * Converts TurnDetail.emotion (dominant_emotion, scores record) into normalized EmotionScore[]
 */
function convertEmotionAnalysisToEmotions(emotion?: {
  dominant_emotion?: string;
  scores?: Record<string, number>;
}): EmotionScore[] {
  if (!emotion) {
    return createFullEmotionDistribution({ label: "NEUTRAL", score: 0.8 });
  }

  if (emotion.scores && Object.keys(emotion.scores).length > 0) {
    const entries = Object.entries(emotion.scores).sort((a, b) => b[1] - a[1]);
    const top1 = entries[0];
    const top2 = entries[1];
    return createFullEmotionDistribution(
      { label: top1[0].toUpperCase(), score: Math.min(1, Math.max(0.1, top1[1])) },
      top2 ? { label: top2[0].toUpperCase(), score: Math.min(0.9, Math.max(0.05, top2[1])) } : undefined
    );
  }

  if (emotion.dominant_emotion) {
    return createFullEmotionDistribution({
      label: emotion.dominant_emotion.toUpperCase(),
      score: 0.85,
    });
  }

  return createFullEmotionDistribution({ label: "NEUTRAL", score: 0.8 });
}

/**
 * Converts TurnDetail.sentiment (label, score) into normalized EmotionScore[]
 */
function convertSentimentToEmotions(sentiment?: {
  label?: string;
  score?: number;
}): EmotionScore[] {
  if (!sentiment || !sentiment.label) {
    return createFullEmotionDistribution({ label: "NEUTRAL", score: 0.8 });
  }
  return createFullEmotionDistribution({
    label: sentiment.label.toUpperCase(),
    score: typeof sentiment.score === "number" ? Math.min(1, Math.max(0.1, sentiment.score)) : 0.85,
  });
}

/**
 * Parses HTTP error response with detailed support for FastAPI / Pydantic validation errors (HTTP 422)
 */
async function parseApiError(response: Response): Promise<string> {
  try {
    const cloned = response.clone();
    const json = await cloned.json();
    if (json && json.detail) {
      if (Array.isArray(json.detail)) {
        // FastAPI / Pydantic validation error array:
        // [{ loc: ["body", "file"], msg: "Field required", type: "missing" }, ...]
        const formattedErrors = json.detail.map(
          (err: { loc?: unknown[]; msg?: string; type?: string }) => {
            const locArr = Array.isArray(err.loc) ? err.loc : [];
            const meaningfulLoc = locArr.filter(
              (p) => !["body", "query", "header", "form"].includes(String(p))
            );
            const fieldName =
              meaningfulLoc.length > 0
                ? meaningfulLoc.join(".")
                : locArr.join(".") || "parameter";

            const msg = err.msg || "Invalid value";
            if (err.type === "missing" || msg.toLowerCase().includes("field required")) {
              return `Missing required parameter: '${fieldName}' (${msg})`;
            }
            return `Invalid parameter '${fieldName}': ${msg}`;
          }
        );
        return `Validation failed: ${formattedErrors.join("; ")}`;
      }
      if (typeof json.detail === "string") {
        return json.detail;
      }
      return JSON.stringify(json.detail);
    }
    if (json && typeof json.message === "string") {
      return json.message;
    }
  } catch {
    // If not JSON, fall back to reading response text below
  }

  try {
    const text = await response.text();
    if (text) {
      return `Server error (${response.status}): ${text}`;
    }
  } catch {
    // Ignore read errors
  }

  return `Server responded with HTTP ${response.status}: ${response.statusText || "Request failed"}`;
}

/**
 * Sends audio file to backend endpoint with resilient extraction
 * matching the server's AnalyzeResponse (turns, metrics, speakers)
 * and legacy schemas.
 */
export async function analyzeAudioApi(
  file: File | Blob,
  filename: string = "audio_sample.wav",
  externalSignal?: AbortSignal
): Promise<AnalyzeAudioResponse> {
  const formData = new FormData();
  if (file instanceof File) {
    formData.append("file", file, file.name);
  } else {
    formData.append("file", file, filename);
  }

  if (externalSignal?.aborted) {
    const abortErr = new Error("Analysis cancelled by user");
    abortErr.name = "AbortError";
    throw abortErr;
  }

  try {
    const response = await fetch(API_BASE_URL, {
      method: "POST",
      body: formData,
      signal: externalSignal,
    });

    if (!response.ok) {
      const errorMsg = await parseApiError(response);
      throw new Error(errorMsg);
    }

    const json = await response.json();

    // Comprehensive extraction supporting:
    // 1. FastAPI AnalyzeResponse schema: json.turns
    // 2. AudioResponse schema: json.analysis.timeline or json.analysis.turns
    // 3. Direct timeline: json.timeline
    // 4. Direct segments: json.segments
    // 5. Wrapped data: json.data.turns or json.data.timeline or json.data.segments or json.data
    // 6. Bare array: json
    let rawSegments: RawTimelineEvent[] = [];

    if (Array.isArray(json)) {
      rawSegments = json;
    } else if (json && typeof json === "object") {
      if (Array.isArray(json.turns)) {
        rawSegments = json.turns;
      } else if (json.analysis && Array.isArray(json.analysis.turns)) {
        rawSegments = json.analysis.turns;
      } else if (json.analysis && Array.isArray(json.analysis.timeline)) {
        rawSegments = json.analysis.timeline;
      } else if (json.analysis && Array.isArray(json.analysis.segments)) {
        rawSegments = json.analysis.segments;
      } else if (Array.isArray(json.timeline)) {
        rawSegments = json.timeline;
      } else if (Array.isArray(json.segments)) {
        rawSegments = json.segments;
      } else if (json.data && Array.isArray(json.data.turns)) {
        rawSegments = json.data.turns;
      } else if (json.data && Array.isArray(json.data.timeline)) {
        rawSegments = json.data.timeline;
      } else if (json.data && Array.isArray(json.data.segments)) {
        rawSegments = json.data.segments;
      } else if (Array.isArray(json.data)) {
        rawSegments = json.data;
      } else {
        const availableKeys = Object.keys(json).join(", ");
        throw new Error(
          `Unexpected response format from audio analyze endpoint. Available top-level keys: [${availableKeys}]. Expected 'turns' or 'timeline'.`
        );
      }
    } else {
      throw new Error("Invalid response received from audio analyze endpoint: expected JSON object or array.");
    }

    // Normalize timeline segments supporting TurnDetail and TimelineSegment
    const segments: TimelineSegment[] = rawSegments.map((item, index) => {
      // 1. Semantic emotion extraction
      let semantic: EmotionScore[];
      if (Array.isArray(item.semantic_emotion) && item.semantic_emotion.length > 0) {
        semantic = item.semantic_emotion.map(normalizeEmotion);
      } else if (item.sentiment && typeof item.sentiment === "object") {
        semantic = convertSentimentToEmotions(item.sentiment);
      } else {
        semantic = createFullEmotionDistribution({ label: "NEUTRAL", score: 0.8 });
      }

      // 2. Acoustic emotion extraction
      let acoustic: EmotionScore[];
      if (Array.isArray(item.acoustic_emotion) && item.acoustic_emotion.length > 0) {
        acoustic = item.acoustic_emotion.map(normalizeEmotion);
      } else if (item.emotion && typeof item.emotion === "object") {
        acoustic = convertEmotionAnalysisToEmotions(item.emotion);
      } else {
        acoustic = createFullEmotionDistribution({ label: "NEUTRAL", score: 0.8 });
      }

      const startTime =
        typeof item.start_time === "number"
          ? item.start_time
          : typeof item.start === "number"
          ? item.start
          : 0;

      const endTime =
        typeof item.end_time === "number"
          ? item.end_time
          : typeof item.end === "number"
          ? item.end
          : typeof item.duration === "number"
          ? startTime + item.duration
          : 0;

      const primarySemantic = semantic[0]?.label || "NEUTRAL";
      const primaryAcoustic = acoustic[0]?.label || "NEUTRAL";
      const isConflict =
        item.is_conflict !== undefined
          ? Boolean(item.is_conflict)
          : primarySemantic !== primaryAcoustic &&
            primarySemantic !== "NEUTRAL" &&
            primaryAcoustic !== "NEUTRAL";

      const conflictDetail =
        item.conflict_detail ||
        (isConflict
          ? `Acoustic-Semantic Incongruence: Semantic tone indicates '${primarySemantic}', but vocal prosody indicates '${primaryAcoustic}'.`
          : null);

      const overlapDur =
        typeof item.overlap_duration === "number"
          ? item.overlap_duration
          : typeof item.overlap_ratio === "number"
          ? Number(
              (
                (item.duration || Math.max(0, endTime - startTime)) *
                item.overlap_ratio
              ).toFixed(3)
            )
          : 0;

      let segId = index + 1;
      if (typeof item.segment_id === "number") {
        segId = item.segment_id;
      } else if (typeof item.turn_id === "string") {
        const parsed = parseInt(item.turn_id.replace(/\D/g, ""), 10);
        if (!isNaN(parsed)) segId = parsed + 1;
      }

      return {
        segment_id: segId,
        speaker: item.speaker || "Speaker 1",
        start_time: Number(startTime.toFixed(3)),
        end_time: Number(endTime.toFixed(3)),
        text: item.text || "",
        semantic_emotion: semantic,
        acoustic_emotion: acoustic,
        is_conflict: isConflict,
        is_ambiguous: Boolean(item.is_ambiguous),
        conflict_detail: conflictDetail,
        is_interruption: Boolean(item.is_interruption),
        interrupted_by: item.interrupted_by ?? null,
        overlap_duration: overlapDur,
      };
    });

    // Detect speakers
    let speakers: string[] = [];
    if (json && typeof json === "object") {
      if (Array.isArray(json.speakers)) {
        speakers = json.speakers;
      } else if (Array.isArray(json.speakers_detected)) {
        speakers = json.speakers_detected;
      } else if (json.analysis && Array.isArray(json.analysis.speakers)) {
        speakers = json.analysis.speakers;
      } else if (json.analysis && Array.isArray(json.analysis.speakers_detected)) {
        speakers = json.analysis.speakers_detected;
      }
    }

    if (speakers.length === 0) {
      speakers = Array.from(new Set(segments.map((s) => s.speaker)));
    }
    if (speakers.length === 0) {
      speakers = ["Speaker 1"];
    }

    const resolvedFilename =
      (json && typeof json === "object" && (json.call_id || json.file_name || json.filename)) ||
      filename;

    const overallTranscript =
      (json && typeof json === "object" && json.analysis?.overall_transcript) ||
      (json && typeof json === "object" && json.overall_transcript) ||
      segments.map((s) => `${s.speaker}: ${s.text}`).join("\n");

    // Agent synthesis extraction or synthesis from CallMetrics
    let agentContext: AgentSynthesis | null =
      (json && typeof json === "object" && json.analysis?.agent_context) ||
      (json && typeof json === "object" && json.agent_context) ||
      null;

    if (!agentContext && json && typeof json === "object" && json.metrics) {
      const metrics = json.metrics;
      const speakerList = speakers.join(", ");
      agentContext = {
        summary: `Analyzed ${segments.length} conversational turns across ${
          metrics.speaker_count || speakers.length
        } speakers (${speakerList}). Total speech: ${metrics.total_speech_s || 0}s, overlap: ${
          metrics.overlap_s || 0
        }s (${Math.round((metrics.overlap_ratio || 0) * 100)}%). Pipeline branch: ${
          metrics.branch_used || "default"
        }.`,
        escalation_detected: (metrics.interruption_count || 0) > 2,
        primary_speaker_sentiments: {},
        flagged_anomalies: segments
          .filter((s) => s.is_conflict && s.conflict_detail)
          .map((s) => `${s.speaker} turn at ${s.start_time}s: ${s.conflict_detail}`),
        interruption_count: metrics.interruption_count || 0,
        total_overtalk_seconds: metrics.overlap_s || 0,
        overtalk_ratio: metrics.overlap_ratio || 0,
      };
    }

    const totalDuration =
      (json && typeof json === "object" && json.metrics?.duration_s) ||
      (segments.length > 0 ? segments[segments.length - 1].end_time : undefined);

    return {
      segments,
      speakers,
      total_duration: totalDuration,
      filename: resolvedFilename,
      overall_transcript: overallTranscript,
      agent_context: agentContext,
      is_mock: false,
    };
  } catch (err: unknown) {
    if (
      externalSignal?.aborted ||
      (err instanceof Error &&
        (err.name === "AbortError" ||
          err.message.includes("cancelled by user") ||
          err.message.includes("stopped by user")))
    ) {
      const abortErr = new Error("Analysis request was stopped by user.");
      abortErr.name = "AbortError";
      throw abortErr;
    }
    const message = err instanceof Error ? err.message : String(err);
    console.warn("Backend API request failed or returned error:", message);
    throw new Error(message, { cause: err });
  }
}

/**
 * Creates an in-browser synthetic audio tone wav blob for demo playback
 * when testing without an existing file upload
 */
export function createSyntheticDemoAudio(): Blob {
  const sampleRate = 16000;
  const duration = 22; // 22 seconds to match mock segments
  const totalSamples = sampleRate * duration;
  const buffer = new ArrayBuffer(44 + totalSamples * 2);
  const view = new DataView(buffer);

  // Write WAV header
  const writeStr = (offset: number, str: string) => {
    for (let i = 0; i < str.length; i++) view.setUint8(offset + i, str.charCodeAt(i));
  };

  writeStr(0, "RIFF");
  view.setUint32(4, 36 + totalSamples * 2, true);
  writeStr(8, "WAVE");
  writeStr(12, "fmt ");
  view.setUint32(16, 16, true);
  view.setUint16(20, 1, true); // PCM
  view.setUint16(22, 1, true); // Mono
  view.setUint32(24, sampleRate, true);
  view.setUint32(28, sampleRate * 2, true);
  view.setUint16(32, 2, true);
  view.setUint16(34, 16, true);
  writeStr(36, "data");
  view.setUint32(40, totalSamples * 2, true);

  // Generate pleasant harmonic speech-like chords and pauses
  for (let i = 0; i < totalSamples; i++) {
    const t = i / sampleRate;
    let sample = 0;

    // Segment 1 (0.5 to 3.8s) - Energetic, upbeat pitch (~280Hz)
    if (t >= 0.5 && t <= 3.8) {
      sample =
        Math.sin(2 * Math.PI * 280 * t) * 0.25 +
        Math.sin(2 * Math.PI * 560 * t) * 0.12 * Math.cos(2 * Math.PI * 4 * t);
    }
    // Segment 2 (4.2 to 7.9s) - Low, tense, sarcastic tone (~140Hz with buzz)
    else if (t >= 4.2 && t <= 7.9) {
      sample =
        Math.sin(2 * Math.PI * 140 * t) * 0.3 +
        Math.sin(2 * Math.PI * 280 * t) * 0.15 +
        (Math.random() - 0.5) * 0.04;
    }
    // Segment 3 (8.4 to 12.1s) - Questioning inflection rising pitch
    else if (t >= 8.4 && t <= 12.1) {
      const pitch = 220 + (t - 8.4) * 25;
      sample = Math.sin(2 * Math.PI * pitch * t) * 0.22;
    }
    // Segment 4 (12.6 to 16.8s) - Low, slow, melancholic tone (~160Hz falling)
    else if (t >= 12.6 && t <= 16.8) {
      sample = Math.sin(2 * Math.PI * 170 * t) * 0.2 * (0.8 + 0.2 * Math.sin(t * 2));
    }
    // Segment 5 (17.2 to 21.0s) - Excited peak tone (~360Hz)
    else if (t >= 17.2 && t <= 21.0) {
      sample =
        Math.sin(2 * Math.PI * 340 * t) * 0.28 +
        Math.sin(2 * Math.PI * 680 * t) * 0.14 * Math.sin(10 * t);
    }

    const int16 = Math.max(-32768, Math.min(32767, Math.floor(sample * 32767)));
    view.setInt16(44 + i * 2, int16, true);
  }

  return new Blob([buffer], { type: "audio/wav" });
}

export const MOCK_POST_ANALYSIS: PostCallAnalysis = {
  call_id: "dialogue_prosody_sample.wav",
  executive_summary:
    "Customer called customer support regarding an urgent airport taxi dispatch delay in Marathahalli, Bangalore. The agent acknowledged the concern, calmed the customer's initial frustration, and confirmed driver details with an express pickup dispatch.",
  call_outcome: "Resolved",
  outcome_reasoning:
    "The agent successfully addressed the urgency, verified the booking vehicle, and dispatched the driver without requiring further customer escalation.",
  customer_satisfaction_score: 8.8,
  sentiment_trajectory: {
    start_emotion: "Frustrated / Urgent",
    end_emotion: "Relieved & Satisfied",
    trend: "improved",
    details:
      "Customer started with high acoustic tension and delay anxiety, transitioning into calm appreciation after the agent confirmed immediate driver dispatch.",
  },
  agent_performance: {
    empathy_score: 9.0,
    professionalism_score: 9.2,
    active_listening_score: 8.7,
    overall_score: 9.0,
    strengths: [
      "Demonstrated immediate empathy towards customer airport flight deadline",
      "Maintained professional composure during initial customer frustration",
      "Confirmed specific action items (SMS with KA 9515 driver phone)",
    ],
    areas_for_improvement: [
      "Could minimize conversational overlap during opening handover",
    ],
  },
  action_items: [
    {
      task: "Dispatch KA 9515 vehicle via express route",
      owner: "Support Team",
      priority: "High",
    },
    {
      task: "Send SMS confirmation with driver contact and live tracking URL",
      owner: "Agent",
      priority: "High",
    },
    {
      task: "Automate 15-minute follow-up ping to verify passenger boarding",
      owner: "Support Team",
      priority: "Medium",
    },
  ],
  risk_assessment: {
    escalation_risk: "Low",
    churn_risk: "Low",
    risk_factors: [
      "Initial delay anxiety posed brief churn risk",
      "Subtle prosody-semantic divergence detected at turn #2",
    ],
  },
  key_moments: [
    {
      timestamp: "00:04",
      time_seconds: 4.2,
      speaker: "Customer",
      quote: "My cab was delayed and I urgently need to reach the airport!",
      insight: "Primary caller distress point stated with elevated vocal arousal.",
      sentiment: "Anxious / Urgent",
    },
    {
      timestamp: "00:08",
      time_seconds: 8.4,
      speaker: "Agent",
      quote: "I understand completely, let me assign an express vehicle right away.",
      insight: "De-escalation pivot turn: agent validated urgency and took ownership.",
      sentiment: "Empathetic",
    },
    {
      timestamp: "00:17",
      time_seconds: 17.2,
      speaker: "Customer",
      quote: "Thank you so much, I appreciate the prompt help!",
      insight: "Resolution confirmation: tone shifted to relief and positive affect.",
      sentiment: "Relieved",
    },
  ],
  llm_model_used: "qwen/qwen3.8-27b:free",
  generated_at: new Date().toISOString(),
};

/**
 * Sends overall audio analysis results to the LLM post-analysis endpoint.
 * Gracefully falls back to local synthesis if backend is unavailable.
 */
export async function analyzeCallPostSessionApi(
  result: AudioAnalysisResult,
  customInstructions?: string,
  externalSignal?: AbortSignal
): Promise<PostCallAnalysis> {
  const payload = {
    call_id: result.filename,
    speakers: result.speakers,
    overall_transcript: result.overall_transcript,
    turns: result.segments.map((s) => ({
      turn_id: `turn_${s.segment_id}`,
      speaker: s.speaker,
      start: s.start_time,
      end: s.end_time,
      text: s.text,
      is_conflict: s.is_conflict,
      is_interruption: s.is_interruption,
      acoustic_emotion: s.acoustic_emotion,
      semantic_emotion: s.semantic_emotion,
    })),
    metrics: {
      duration_s: result.duration || (result.segments.length > 0 ? result.segments[result.segments.length - 1].end_time : 0),
      speaker_count: result.speakers.length,
      turn_count: result.segments.length,
      interruption_count: result.segments.filter((s) => s.is_interruption).length,
    },
    custom_instructions: customInstructions,
  };

  try {
    const response = await fetch(POST_ANALYSIS_API_URL, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify(payload),
      signal: externalSignal,
    });

    if (!response.ok) {
      const errText = await response.text();
      console.warn("Post-analysis endpoint error response:", response.status, errText);
      throw new Error(`Post-analysis API error (${response.status}): ${errText}`);
    }

    const data: PostCallAnalysis = await response.json();
    return data;
  } catch (err: unknown) {
    if (
      externalSignal?.aborted ||
      (err instanceof Error &&
        (err.name === "AbortError" ||
          err.message.includes("cancelled by user") ||
          err.message.includes("stopped by user")))
    ) {
      throw new Error("Analysis request was stopped by user.", { cause: err });
    }

    console.warn("Post-analysis endpoint unreachable; generating client-side synthesis fallback:", err);

    // Fallback: Generate structured synthesis from segments
    const conflictCount = result.segments.filter((s) => s.is_conflict).length;
    const interruptCount = result.segments.filter((s) => s.is_interruption).length;
    const isEscalated = conflictCount > 1 || interruptCount > 2;

    return {
      call_id: result.filename,
      executive_summary: `Synthesized analysis for ${result.segments.length} conversational turns across ${result.speakers.join(", ")}. The call completed with ${conflictCount} affect anomalies and ${interruptCount} turn interruptions.`,
      call_outcome: isEscalated ? "Pending Follow-up" : "Resolved",
      outcome_reasoning: isEscalated
        ? "Cross-modal tension or multiple interruptions were detected during customer turns."
        : "Dialogue exchanges completed without major escalations.",
      customer_satisfaction_score: isEscalated ? 6.2 : 8.5,
      sentiment_trajectory: {
        start_emotion: result.segments[0]?.acoustic_emotion?.[0]?.label || "Neutral",
        end_emotion: result.segments[result.segments.length - 1]?.acoustic_emotion?.[0]?.label || "Neutral",
        trend: isEscalated ? "volatile" : "improved",
        details: "Emotional progression analyzed from acoustic prosody and verbal expressions.",
      },
      agent_performance: {
        empathy_score: isEscalated ? 7.2 : 8.8,
        professionalism_score: 8.6,
        active_listening_score: isEscalated ? 7.0 : 8.5,
        overall_score: isEscalated ? 7.6 : 8.6,
        strengths: [
          "Attentive responses maintained throughout dialogue",
          "Addressed speaker requests without conversational dead-ends",
        ],
        areas_for_improvement: interruptCount > 0 ? ["Reduce conversational overlap and turn collisions"] : ["Continue current cadence"],
      },
      action_items: [
        {
          task: "Confirm call notes and log ticket status in CRM",
          owner: "Agent",
          priority: "Medium",
        },
      ],
      risk_assessment: {
        escalation_risk: isEscalated ? "Medium" : "Low",
        churn_risk: isEscalated ? "Medium" : "Low",
        risk_factors: conflictCount > 0 ? [`${conflictCount} prosody-semantic conflict moments detected`] : ["No critical risk indicators"],
      },
      key_moments: result.segments.slice(0, 3).map((s) => ({
        timestamp: `${Math.floor(s.start_time / 60).toString().padStart(2, "0")}:${Math.floor(s.start_time % 60).toString().padStart(2, "0")}`,
        time_seconds: s.start_time,
        speaker: s.speaker,
        quote: s.text,
        insight: s.is_conflict ? "Cross-modal prosodic tension observed" : "Dialogue progression turn",
        sentiment: s.acoustic_emotion?.[0]?.label || "Neutral",
      })),
      llm_model_used: "client_synthesizer_fallback",
      generated_at: new Date().toISOString(),
    };
  }
}
