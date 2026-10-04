import React, { useState } from "react";
import type { TimelineSegment, EmotionScore } from "@/types/emotion";
import {
  formatTime,
  getSpeakerTheme,
  getEmotionMeta,
  ALL_EMOTIONS,
} from "@/lib/constants";
import {
  Play,
  ChevronDown,
  ChevronUp,
  AlertCircle,
  Volume2,
  FileText,
  BarChart3,
  Layers,
} from "lucide-react";

interface EmotionTimelineCardsProps {
  segments: TimelineSegment[];
  activeSegmentId: number | null;
  onPlaySegment: (startTime: number, endTime?: number) => void;
}

export const EmotionTimelineCards: React.FC<EmotionTimelineCardsProps> = ({
  segments,
  activeSegmentId,
  onPlaySegment,
}) => {
  // Store expanded state per segment ID
  const [expandedMap, setExpandedMap] = useState<Record<number, boolean>>(() => {
    const init: Record<number, boolean> = {};
    segments.forEach((seg, idx) => {
      init[seg.segment_id] = idx === 0 || seg.is_conflict;
    });
    return init;
  });

  const toggleExpand = (id: number) => {
    setExpandedMap((prev) => ({
      ...prev,
      [id]: !prev[id],
    }));
  };

  const toggleAll = (expand: boolean) => {
    const next: Record<number, boolean> = {};
    segments.forEach((seg) => {
      next[seg.segment_id] = expand;
    });
    setExpandedMap(next);
  };

  // Helper to ensure 8 emotions are mapped for comparison bars
  const normalizeEmotionBars = (emotions: EmotionScore[] = []): EmotionScore[] => {
    const map = new Map<string, number>();
    emotions.forEach((e) => {
      map.set(e.label.toUpperCase(), e.score);
    });

    return ALL_EMOTIONS.map((label) => ({
      label,
      original_label: label.toLowerCase(),
      score: map.get(label) || 0,
    })).sort((a, b) => b.score - a.score);
  };

  return (
    <div className="w-full flex flex-col gap-3.5">
      {/* Section Header Controls */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 pb-2.5 border-b border-border/60">
        <div className="flex items-center gap-2">
          <Layers className="w-4 h-4 text-muted-foreground" />
          <h3 className="text-sm sm:text-base font-semibold text-foreground">
            Segmented Emotion Timeline
          </h3>
        </div>

        <div className="flex items-center gap-1.5">
          <button
            type="button"
            onClick={() => toggleAll(true)}
            className="px-2.5 py-1 text-xs font-medium rounded-md bg-secondary hover:bg-secondary/80 text-muted-foreground hover:text-foreground transition-colors cursor-pointer"
          >
            Expand All
          </button>
          <button
            type="button"
            onClick={() => toggleAll(false)}
            className="px-2.5 py-1 text-xs font-medium rounded-md bg-secondary hover:bg-secondary/80 text-muted-foreground hover:text-foreground transition-colors cursor-pointer"
          >
            Collapse All
          </button>
        </div>
      </div>

      {/* Vertical Segment Cards List */}
      <div className="space-y-3">
        {segments.map((segment) => {
          const isExpanded = !!expandedMap[segment.segment_id];
          const isActive = activeSegmentId === segment.segment_id;
          const speakerTheme = getSpeakerTheme(segment.speaker);

          const topSemantic = segment.semantic_emotion?.[0];
          const topAcoustic = segment.acoustic_emotion?.[0];

          const topSemanticMeta = topSemantic ? getEmotionMeta(topSemantic.label) : null;
          const topAcousticMeta = topAcoustic ? getEmotionMeta(topAcoustic.label) : null;

          const semanticBars = normalizeEmotionBars(segment.semantic_emotion);
          const acousticBars = normalizeEmotionBars(segment.acoustic_emotion);

          return (
            <div
              key={segment.segment_id}
              className={`rounded-2xl border transition-all duration-200 overflow-hidden ${
                isActive
                  ? "border-foreground/30 bg-muted/40 shadow-2xs ring-1 ring-foreground/10"
                  : "border-border/70 bg-card hover:border-border"
              }`}
            >
              {/* Card Header (Always Visible) */}
              <div
                onClick={() => toggleExpand(segment.segment_id)}
                className="p-4 sm:p-4.5 flex flex-col gap-2.5 cursor-pointer select-none group"
              >
                <div className="flex flex-wrap items-center justify-between gap-2">
                  <div className="flex flex-wrap items-center gap-2">
                    {/* Play Segment Button */}
                    <button
                      type="button"
                      onClick={(e) => {
                        e.stopPropagation();
                        onPlaySegment(segment.start_time, segment.end_time);
                      }}
                      className="p-1.5 rounded-lg bg-muted text-foreground/80 hover:bg-foreground hover:text-background transition-colors active:scale-95 cursor-pointer"
                      title="Play this segment"
                    >
                      <Play className="w-3 h-3 fill-current" />
                    </button>

                    {/* Speaker Tag */}
                    <div
                      className="inline-flex items-center gap-1.5 px-2.5 py-0.5 rounded text-xs font-medium bg-muted text-foreground/85 border border-border/70"
                    >
                      <span className={`w-1.5 h-1.5 rounded-full ${speakerTheme.dot}`} />
                      <span>{segment.speaker}</span>
                    </div>

                    {/* Time Range */}
                    <span className="text-xs font-mono text-muted-foreground/75 bg-muted/40 px-2 py-0.5 rounded border border-border/50">
                      {formatTime(segment.start_time, true)} →{" "}
                      {formatTime(segment.end_time, true)}
                    </span>

                    {/* Now Playing Animated Pill */}
                    {isActive && (
                      <span className="inline-flex items-center gap-1.5 px-2 py-0.5 rounded text-[11px] font-medium bg-foreground text-background shadow-2xs">
                        <span className="w-1 h-1 rounded-full bg-background animate-ping" />
                        Now Playing
                      </span>
                    )}

                    {/* Conflict Badge */}
                    {segment.is_conflict && (
                      <span className="inline-flex items-center gap-1 px-2 py-0.5 rounded text-[11px] font-medium bg-rose-500/10 text-rose-700 dark:text-rose-300 border border-rose-500/20">
                        <AlertCircle className="w-3 h-3" />
                        <span>Conflict</span>
                      </span>
                    )}
                  </div>

                  {/* Top Semantic & Acoustic emotion summary pills + Chevron */}
                  <div className="flex items-center gap-2 ml-auto">
                    {/* Top Acoustic Emotion Summary */}
                    {topAcousticMeta && topAcoustic && (
                      <div
                        className="hidden sm:inline-flex items-center gap-1 px-2 py-0.5 rounded text-xs font-medium bg-muted/60 text-muted-foreground border border-border/60"
                        title="Top Acoustic Prosody Emotion"
                      >
                        <Volume2 className="w-3 h-3 opacity-60" />
                        <span>{topAcousticMeta.emoji}</span>
                        <span>{topAcousticMeta.displayName}</span>
                        <span className="opacity-70 font-mono">
                          {Math.round(topAcoustic.score * 100)}%
                        </span>
                      </div>
                    )}

                    {/* Top Semantic Emotion Summary */}
                    {topSemanticMeta && topSemantic && (
                      <div
                        className="hidden md:inline-flex items-center gap-1 px-2 py-0.5 rounded text-xs font-medium bg-muted/60 text-muted-foreground border border-border/60"
                        title="Top Semantic NLP Emotion"
                      >
                        <FileText className="w-3 h-3 opacity-60" />
                        <span>{topSemanticMeta.emoji}</span>
                        <span>{topSemanticMeta.displayName}</span>
                        <span className="opacity-70 font-mono">
                          {Math.round(topSemantic.score * 100)}%
                        </span>
                      </div>
                    )}

                    {/* Expand/Collapse Chevron */}
                    <div className="p-1 rounded-md text-muted-foreground group-hover:text-foreground group-hover:bg-muted transition-colors">
                      {isExpanded ? (
                        <ChevronUp className="w-4 h-4" />
                      ) : (
                        <ChevronDown className="w-4 h-4" />
                      )}
                    </div>
                  </div>
                </div>

                {/* Spoken text snippet */}
                <p className="text-xs sm:text-sm font-normal text-foreground/85 leading-relaxed italic pl-2 border-l-2 border-border/80">
                  &ldquo;{segment.text}&rdquo;
                </p>
              </div>

              {/* Expanded Breakdown */}
              {isExpanded && (
                <div className="px-4.5 pb-5 pt-2 border-t border-border/60 bg-muted/15 space-y-4">
                  {/* Conflict Detail Alert Note if is_conflict is true */}
                  {segment.is_conflict && (
                    <div className="p-3 rounded-xl border border-rose-500/20 bg-rose-500/5 text-foreground/85 flex items-start gap-2.5">
                      <AlertCircle className="w-4 h-4 text-rose-500 shrink-0 mt-0.5" />
                      <div className="space-y-0.5">
                        <h4 className="text-[11px] font-semibold uppercase tracking-wider text-rose-700 dark:text-rose-400">
                          Cross-Modal Dissonance Detected
                        </h4>
                        <p className="text-xs leading-relaxed text-muted-foreground">
                          {segment.conflict_detail ||
                            "Acoustic vocal prosody contrasts significantly with the semantic sentiment of the spoken words."}
                        </p>
                      </div>
                    </div>
                  )}

                  {/* Side-by-Side Comparison Columns */}
                  <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
                    {/* Column 1: Semantic Emotion (NLP / Words) */}
                    <div className="p-3.5 rounded-xl border border-border/60 bg-card flex flex-col gap-2.5">
                      <div className="flex items-center justify-between pb-1.5 border-b border-border/50">
                        <div className="flex items-center gap-1.5">
                          <FileText className="w-3.5 h-3.5 text-muted-foreground" />
                          <span className="text-xs font-semibold text-foreground">
                            Semantic Emotion (NLP)
                          </span>
                        </div>
                        <span className="text-[10px] font-mono text-muted-foreground/75 flex items-center gap-1">
                          <BarChart3 className="w-3 h-3" />
                          Spectrum
                        </span>
                      </div>

                      <div className="space-y-2">
                        {semanticBars.map((item, idx) => {
                          const meta = getEmotionMeta(item.label);
                          const pct = Math.round(item.score * 100);
                          const isTop = idx === 0 && pct > 20;
                          return (
                            <div key={item.label} className="space-y-0.5">
                              <div className="flex items-center justify-between text-xs">
                                <span className="flex items-center gap-1.5 text-foreground/80">
                                  <span>{meta.emoji}</span>
                                  <span>{meta.displayName}</span>
                                </span>
                                <span className="font-mono text-muted-foreground text-[11px]">
                                  {pct}%
                                </span>
                              </div>
                              {/* Progress track */}
                              <div className="w-full h-1.5 rounded-full bg-muted overflow-hidden">
                                <div
                                  className={`h-full rounded-full transition-all duration-300 ${
                                    isTop
                                      ? "bg-foreground/75"
                                      : idx === 1
                                        ? "bg-muted-foreground/50"
                                        : "bg-muted-foreground/20"
                                  }`}
                                  style={{ width: `${Math.max(item.score > 0 ? 3 : 0, pct)}%` }}
                                />
                              </div>
                            </div>
                          );
                        })}
                      </div>
                    </div>

                    {/* Column 2: Acoustic Emotion (Vocal Prosody / Tone) */}
                    <div className="p-3.5 rounded-xl border border-border/60 bg-card flex flex-col gap-2.5">
                      <div className="flex items-center justify-between pb-1.5 border-b border-border/50">
                        <div className="flex items-center gap-1.5">
                          <Volume2 className="w-3.5 h-3.5 text-muted-foreground" />
                          <span className="text-xs font-semibold text-foreground">
                            Acoustic Emotion (Prosody)
                          </span>
                        </div>
                        <span className="text-[10px] font-mono text-muted-foreground/75 flex items-center gap-1">
                          <BarChart3 className="w-3 h-3" />
                          Prosodic
                        </span>
                      </div>

                      <div className="space-y-2">
                        {acousticBars.map((item, idx) => {
                          const meta = getEmotionMeta(item.label);
                          const pct = Math.round(item.score * 100);
                          const isTop = idx === 0 && pct > 20;
                          return (
                            <div key={item.label} className="space-y-0.5">
                              <div className="flex items-center justify-between text-xs">
                                <span className="flex items-center gap-1.5 text-foreground/80">
                                  <span>{meta.emoji}</span>
                                  <span>{meta.displayName}</span>
                                </span>
                                <span className="font-mono text-muted-foreground text-[11px]">
                                  {pct}%
                                </span>
                              </div>
                              {/* Progress track */}
                              <div className="w-full h-1.5 rounded-full bg-muted overflow-hidden">
                                <div
                                  className={`h-full rounded-full transition-all duration-300 ${
                                    isTop
                                      ? "bg-foreground/75"
                                      : idx === 1
                                        ? "bg-muted-foreground/50"
                                        : "bg-muted-foreground/20"
                                  }`}
                                  style={{ width: `${Math.max(item.score > 0 ? 3 : 0, pct)}%` }}
                                />
                              </div>
                            </div>
                          );
                        })}
                      </div>
                    </div>
                  </div>
                </div>
              )}
            </div>
          );
        })}
      </div>
    </div>
  );
};
