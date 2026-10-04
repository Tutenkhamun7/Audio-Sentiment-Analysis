import type {
  TimelineSegment,
  EmotionScore,
  AgentSynthesis,
} from "@/types/emotion";
import { ALL_EMOTIONS } from "@/lib/constants";

export const API_BASE_URL =
  import.meta.env.VITE_API_URL || "http://localhost:8000/api/v1/audio/analyze";

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
  speaker?: string;
  start_time?: number;
  end_time?: number;
  text?: string;
  semantic_emotion?: RawEmotion[];
  acoustic_emotion?: RawEmotion[];
  is_conflict?: boolean;
  is_ambiguous?: boolean;
  conflict_detail?: string | null;
  is_interruption?: boolean;
  interrupted_by?: string | null;
  overlap_duration?: number;
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
 * Sends audio file to backend endpoint with resilient extraction
 * matching the server's AudioResponse (which houses timeline in analysis.timeline)
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

  const controller = new AbortController();
  const timeoutId = setTimeout(() => controller.abort(), 60000); // 60s timeout for ML model inference

  if (externalSignal) {
    if (externalSignal.aborted) {
      clearTimeout(timeoutId);
      const abortErr = new Error("Analysis cancelled by user");
      abortErr.name = "AbortError";
      throw abortErr;
    }
    externalSignal.addEventListener("abort", () => {
      clearTimeout(timeoutId);
      controller.abort();
    });
  }

  try {
    const response = await fetch(API_BASE_URL, {
      method: "POST",
      body: formData,
      signal: controller.signal,
    });

    clearTimeout(timeoutId);

    if (!response.ok) {
      const errorText = await response.text().catch(() => "");
      throw new Error(
        `Server responded with HTTP ${response.status}: ${errorText || response.statusText}`
      );
    }

    const json = await response.json();

    // Comprehensive extraction supporting:
    // 1. AudioResponse schema: json.analysis.timeline
    // 2. Direct timeline: json.timeline
    // 3. Direct segments: json.segments
    // 4. Wrapped data: json.data.timeline or json.data.segments or json.data
    // 5. Bare array: json
    let rawSegments: RawTimelineEvent[] = [];

    if (Array.isArray(json)) {
      rawSegments = json;
    } else if (json && typeof json === "object") {
      if (json.analysis && Array.isArray(json.analysis.timeline)) {
        rawSegments = json.analysis.timeline;
      } else if (json.analysis && Array.isArray(json.analysis.segments)) {
        rawSegments = json.analysis.segments;
      } else if (Array.isArray(json.timeline)) {
        rawSegments = json.timeline;
      } else if (Array.isArray(json.segments)) {
        rawSegments = json.segments;
      } else if (json.data && Array.isArray(json.data.timeline)) {
        rawSegments = json.data.timeline;
      } else if (json.data && Array.isArray(json.data.segments)) {
        rawSegments = json.data.segments;
      } else if (Array.isArray(json.data)) {
        rawSegments = json.data;
      } else {
        const availableKeys = Object.keys(json).join(", ");
        throw new Error(
          `Unexpected response format from audio analyze endpoint. Available top-level keys: [${availableKeys}]. Expected 'analysis.timeline' or 'segments'.`
        );
      }
    } else {
      throw new Error("Invalid response received from audio analyze endpoint: expected JSON object or array.");
    }

    // Normalize timeline segments
    const segments: TimelineSegment[] = rawSegments.map((item, index) => {
      const semantic = Array.isArray(item.semantic_emotion)
        ? item.semantic_emotion.map(normalizeEmotion)
        : [];
      const acoustic = Array.isArray(item.acoustic_emotion)
        ? item.acoustic_emotion.map(normalizeEmotion)
        : [];

      return {
        segment_id: typeof item.segment_id === "number" ? item.segment_id : index + 1,
        speaker: item.speaker || "Speaker 1",
        start_time: typeof item.start_time === "number" ? item.start_time : 0,
        end_time: typeof item.end_time === "number" ? item.end_time : 0,
        text: item.text || "",
        semantic_emotion: semantic,
        acoustic_emotion: acoustic,
        is_conflict: Boolean(item.is_conflict),
        is_ambiguous: Boolean(item.is_ambiguous),
        conflict_detail: item.conflict_detail ?? null,
        is_interruption: Boolean(item.is_interruption),
        interrupted_by: item.interrupted_by ?? null,
        overlap_duration: typeof item.overlap_duration === "number" ? item.overlap_duration : 0,
      };
    });

    // Detect speakers
    let speakers: string[] = [];
    if (json && typeof json === "object") {
      if (Array.isArray(json.speakers_detected)) {
        speakers = json.speakers_detected;
      } else if (json.analysis && Array.isArray(json.analysis.speakers_detected)) {
        speakers = json.analysis.speakers_detected;
      } else if (Array.isArray(json.speakers)) {
        speakers = json.speakers;
      }
    }

    if (speakers.length === 0) {
      speakers = Array.from(new Set(segments.map((s) => s.speaker)));
    }
    if (speakers.length === 0) {
      speakers = ["Speaker 1"];
    }

    const resolvedFilename =
      (json && typeof json === "object" && (json.file_name || json.filename)) || filename;

    const overallTranscript =
      (json && typeof json === "object" && json.analysis?.overall_transcript) ||
      (json && typeof json === "object" && json.overall_transcript) ||
      segments.map((s) => `${s.speaker}: ${s.text}`).join("\n");

    const agentContext =
      (json && typeof json === "object" && json.analysis?.agent_context) ||
      (json && typeof json === "object" && json.agent_context) ||
      null;

    const totalDuration =
      segments.length > 0 ? segments[segments.length - 1].end_time : undefined;

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
      (err instanceof Error && (err.name === "AbortError" || err.message.includes("cancelled by user")))
    ) {
      const abortErr = new Error("Analysis request was stopped by user.");
      abortErr.name = "AbortError";
      throw abortErr;
    }
    const message = err instanceof Error ? err.message : String(err);
    console.warn("Backend API request failed or returned unexpected shape:", message);
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
