import type { StandardEmotion } from "@/types/emotion";

export interface EmotionMeta {
  label: StandardEmotion;
  displayName: string;
  emoji: string;
  color: string;
  bgLight: string;
  textClass: string;
  badgeClass: string;
  borderClass: string;
  barColor: string;
}

export const ALL_EMOTIONS: StandardEmotion[] = [
  "HAPPY",
  "ANGRY",
  "SAD",
  "NEUTRAL",
  "CONFUSED",
  "FEARFUL",
  "DISGUSTED",
  "SURPRISED",
];

export const EMOTION_TAXONOMY: Record<StandardEmotion, EmotionMeta> = {
  HAPPY: {
    label: "HAPPY",
    displayName: "Happy",
    emoji: "😄",
    color: "#eab308",
    bgLight: "rgba(234, 179, 8, 0.08)",
    textClass: "text-amber-600 dark:text-amber-400/90",
    badgeClass: "bg-amber-500/8 text-amber-700 dark:text-amber-300/90 border-amber-500/20",
    borderClass: "border-amber-500/25",
    barColor: "bg-amber-500/60 dark:bg-amber-400/50",
  },
  ANGRY: {
    label: "ANGRY",
    displayName: "Angry",
    emoji: "😠",
    color: "#e11d48",
    bgLight: "rgba(225, 29, 72, 0.08)",
    textClass: "text-rose-600 dark:text-rose-400/90",
    badgeClass: "bg-rose-500/8 text-rose-700 dark:text-rose-300/90 border-rose-500/20",
    borderClass: "border-rose-500/25",
    barColor: "bg-rose-500/60 dark:bg-rose-400/50",
  },
  SAD: {
    label: "SAD",
    displayName: "Sad",
    emoji: "😢",
    color: "#2563eb",
    bgLight: "rgba(37, 99, 235, 0.08)",
    textClass: "text-blue-600 dark:text-blue-400/90",
    badgeClass: "bg-blue-500/8 text-blue-700 dark:text-blue-300/90 border-blue-500/20",
    borderClass: "border-blue-500/25",
    barColor: "bg-blue-500/60 dark:bg-blue-400/50",
  },
  NEUTRAL: {
    label: "NEUTRAL",
    displayName: "Neutral",
    emoji: "😐",
    color: "#64748b",
    bgLight: "rgba(100, 116, 139, 0.08)",
    textClass: "text-slate-600 dark:text-slate-400",
    badgeClass: "bg-slate-500/8 text-slate-700 dark:text-slate-300/90 border-slate-500/20",
    borderClass: "border-slate-500/25",
    barColor: "bg-slate-400/60 dark:bg-slate-500/50",
  },
  CONFUSED: {
    label: "CONFUSED",
    displayName: "Confused",
    emoji: "😕",
    color: "#4f46e5",
    bgLight: "rgba(79, 70, 229, 0.08)",
    textClass: "text-indigo-600 dark:text-indigo-400/90",
    badgeClass: "bg-indigo-500/8 text-indigo-700 dark:text-indigo-300/90 border-indigo-500/20",
    borderClass: "border-indigo-500/25",
    barColor: "bg-indigo-500/60 dark:bg-indigo-400/50",
  },
  FEARFUL: {
    label: "FEARFUL",
    displayName: "Fearful",
    emoji: "😨",
    color: "#9333ea",
    bgLight: "rgba(147, 51, 234, 0.08)",
    textClass: "text-purple-600 dark:text-purple-400/90",
    badgeClass: "bg-purple-500/8 text-purple-700 dark:text-purple-300/90 border-purple-500/20",
    borderClass: "border-purple-500/25",
    barColor: "bg-purple-500/60 dark:bg-purple-400/50",
  },
  DISGUSTED: {
    label: "DISGUSTED",
    displayName: "Disgusted",
    emoji: "🤢",
    color: "#059669",
    bgLight: "rgba(5, 150, 105, 0.08)",
    textClass: "text-emerald-600 dark:text-emerald-400/90",
    badgeClass: "bg-emerald-500/8 text-emerald-700 dark:text-emerald-300/90 border-emerald-500/20",
    borderClass: "border-emerald-500/25",
    barColor: "bg-emerald-500/60 dark:bg-emerald-400/50",
  },
  SURPRISED: {
    label: "SURPRISED",
    displayName: "Surprised",
    emoji: "😲",
    color: "#db2777",
    bgLight: "rgba(219, 39, 119, 0.08)",
    textClass: "text-pink-600 dark:text-pink-400/90",
    badgeClass: "bg-pink-500/8 text-pink-700 dark:text-pink-300/90 border-pink-500/20",
    borderClass: "border-pink-500/25",
    barColor: "bg-pink-500/60 dark:bg-pink-400/50",
  },
  DISGUST: {
    label: "DISGUST",
    displayName: "Disgusted",
    emoji: "🤢",
    color: "#059669",
    bgLight: "rgba(5, 150, 105, 0.08)",
    textClass: "text-emerald-600 dark:text-emerald-400/90",
    badgeClass: "bg-emerald-500/8 text-emerald-700 dark:text-emerald-300/90 border-emerald-500/20",
    borderClass: "border-emerald-500/25",
    barColor: "bg-emerald-500/60 dark:bg-emerald-400/50",
  },
  AMBIGUOUS: {
    label: "AMBIGUOUS",
    displayName: "Ambiguous",
    emoji: "🤔",
    color: "#7c3aed",
    bgLight: "rgba(124, 58, 237, 0.08)",
    textClass: "text-purple-600 dark:text-purple-400/90",
    badgeClass: "bg-purple-500/8 text-purple-700 dark:text-purple-300/90 border-purple-500/20",
    borderClass: "border-purple-500/25",
    barColor: "bg-purple-500/60 dark:bg-purple-400/50",
  },
};

export function getEmotionMeta(rawLabel: string): EmotionMeta {
  let normalized = (rawLabel || "").toUpperCase().trim() as StandardEmotion;
  if (normalized === "DISGUST") {
    normalized = "DISGUSTED";
  }
  if (EMOTION_TAXONOMY[normalized]) {
    return EMOTION_TAXONOMY[normalized];
  }
  if (normalized === "AMBIGUOUS") {
    return EMOTION_TAXONOMY.AMBIGUOUS;
  }
  // Subtle neutral fallback
  return {
    label: normalized,
    displayName: rawLabel.charAt(0).toUpperCase() + rawLabel.slice(1).toLowerCase(),
    emoji: "💬",
    color: "#64748b",
    bgLight: "rgba(100, 116, 139, 0.08)",
    textClass: "text-muted-foreground",
    badgeClass: "bg-muted text-muted-foreground border-border/60",
    borderClass: "border-border/60",
    barColor: "bg-muted-foreground/40",
  };
}

// Subtle, clean speaker styling
const SPEAKER_PALETTES = [
  {
    chip: "bg-muted/70 text-foreground/80 border-border/80 hover:bg-muted",
    dot: "bg-emerald-500/80",
    bubble: "bg-card hover:bg-muted/30 border-border/60",
    border: "border-border",
  },
  {
    chip: "bg-muted/70 text-foreground/80 border-border/80 hover:bg-muted",
    dot: "bg-indigo-500/80",
    bubble: "bg-card hover:bg-muted/30 border-border/60",
    border: "border-border",
  },
  {
    chip: "bg-muted/70 text-foreground/80 border-border/80 hover:bg-muted",
    dot: "bg-sky-500/80",
    bubble: "bg-card hover:bg-muted/30 border-border/60",
    border: "border-border",
  },
  {
    chip: "bg-muted/70 text-foreground/80 border-border/80 hover:bg-muted",
    dot: "bg-amber-500/80",
    bubble: "bg-card hover:bg-muted/30 border-border/60",
    border: "border-border",
  },
  {
    chip: "bg-muted/70 text-foreground/80 border-border/80 hover:bg-muted",
    dot: "bg-rose-500/80",
    bubble: "bg-card hover:bg-muted/30 border-border/60",
    border: "border-border",
  },
];

export function getSpeakerTheme(speaker: string) {
  let hash = 0;
  for (let i = 0; i < speaker.length; i++) {
    hash = (hash * 31 + speaker.charCodeAt(i)) >>> 0;
  }
  const index = hash % SPEAKER_PALETTES.length;
  return SPEAKER_PALETTES[index];
}

export function formatTime(seconds: number, includeMs = false): string {
  if (isNaN(seconds) || seconds < 0) seconds = 0;
  const mins = Math.floor(seconds / 60);
  const secs = Math.floor(seconds % 60);
  const ms = Math.floor((seconds % 1) * 10);

  const mm = mins.toString().padStart(2, "0");
  const ss = secs.toString().padStart(2, "0");

  if (includeMs) {
    return `${mm}:${ss}.${ms}`;
  }
  return `${mm}:${ss}`;
}
