import React, { useRef, useEffect } from "react";
import type { TimelineSegment } from "@/types/emotion";
import { formatTime, getSpeakerTheme, getEmotionMeta } from "@/lib/constants";
import { MessageSquareText, Play, Pause, AlertCircle } from "lucide-react";

interface SynchronizedTranscriptProps {
  segments: TimelineSegment[];
  activeSegmentId: number | null;
  isPlaying?: boolean;
  onToggleFragment: (segment: TimelineSegment) => void;
  onSeek?: (startTime: number) => void;
}

export const SynchronizedTranscript: React.FC<SynchronizedTranscriptProps> = ({
  segments,
  activeSegmentId,
  isPlaying = false,
  onToggleFragment,
  onSeek,
}) => {
  const containerRef = useRef<HTMLDivElement | null>(null);
  const activeBubbleRef = useRef<HTMLDivElement | null>(null);

  // Auto-scroll active segment smoothly into view
  useEffect(() => {
    if (activeBubbleRef.current && containerRef.current) {
      const container = containerRef.current;
      const bubble = activeBubbleRef.current;
      const bubbleTop = bubble.offsetTop - container.offsetTop;
      const targetScroll = bubbleTop - container.clientHeight / 2 + bubble.clientHeight / 2;

      container.scrollTo({
        top: Math.max(0, targetScroll),
        behavior: "smooth",
      });
    }
  }, [activeSegmentId]);

  const handleRowClick = (seg: TimelineSegment) => {
    if (onToggleFragment) {
      onToggleFragment(seg);
    } else if (onSeek) {
      onSeek(seg.start_time);
    }
  };

  return (
    <div className="w-full flex flex-col gap-3 rounded-2xl border border-border/70 bg-card p-5 shadow-2xs">
      {/* Transcript Header */}
      <div className="flex items-center justify-between pb-2.5 border-b border-border/60">
        <div className="flex items-center gap-2">
          <MessageSquareText className="w-4 h-4 text-muted-foreground" />
          <h3 className="text-sm font-semibold text-foreground">
            Synchronized Transcript
          </h3>
        </div>
        <span className="text-[11px] text-muted-foreground/75 hidden sm:inline">
          Click fragment to play / pause
        </span>
      </div>

      {/* Sentence Bubbles Container */}
      <div
        ref={containerRef}
        className="max-h-[520px] overflow-y-auto space-y-2.5 pr-1.5 scrollbar-thin scrollbar-thumb-muted-foreground/15 hover:scrollbar-thumb-muted-foreground/30"
      >
        {segments.map((seg) => {
          const isActive = activeSegmentId === seg.segment_id;
          const isFragmentPlaying = isActive && isPlaying;
          const speakerTheme = getSpeakerTheme(seg.speaker);
          const topAcoustic = seg.acoustic_emotion?.[0];
          const acousticMeta = topAcoustic ? getEmotionMeta(topAcoustic.label) : null;

          return (
            <div
              key={seg.segment_id}
              ref={isActive ? activeBubbleRef : null}
              onClick={() => handleRowClick(seg)}
              className={`group relative p-3.5 rounded-xl cursor-pointer transition-all duration-150 border ${
                isActive
                  ? "border-foreground/30 bg-muted/60 shadow-2xs ring-1 ring-foreground/10"
                  : "border-border/60 hover:border-border bg-muted/20 hover:bg-muted/40"
              }`}
            >
              <div className="flex items-start justify-between gap-3 mb-1.5">
                {/* Speaker pill & Timestamp */}
                <div className="flex items-center gap-2">
                  <span
                    className="inline-flex items-center gap-1.5 px-2 py-0.5 rounded text-[11px] font-medium bg-muted text-foreground/80 border border-border/60"
                  >
                    <span className={`w-1.5 h-1.5 rounded-full ${speakerTheme.dot}`} />
                    <span>{seg.speaker}</span>
                  </span>

                  <span className="text-[11px] font-mono text-muted-foreground/70">
                    {formatTime(seg.start_time, true)} – {formatTime(seg.end_time, true)}
                  </span>
                </div>

                {/* Right badges & Click to play/pause icon */}
                <div className="flex items-center gap-1.5">
                  {/* Conflict badge */}
                  {seg.is_conflict && (
                    <span
                      title="Prosody Conflict"
                      className="inline-flex items-center gap-1 px-1.5 py-0.2 rounded text-[10px] font-medium bg-rose-500/10 text-rose-700 dark:text-rose-300 border border-rose-500/20"
                    >
                      <AlertCircle className="w-3 h-3" />
                      <span>Conflict</span>
                    </span>
                  )}

                  {/* Emotion chip */}
                  {acousticMeta && (
                    <span
                      className="hidden sm:inline-flex items-center gap-1 px-2 py-0.2 rounded text-[11px] text-muted-foreground bg-muted/60 border border-border/50"
                    >
                      <span>{acousticMeta.emoji}</span>
                      <span>{acousticMeta.displayName}</span>
                    </span>
                  )}

                  {/* Play / Pause Toggle Button */}
                  <button
                    type="button"
                    onClick={(e) => {
                      e.stopPropagation();
                      handleRowClick(seg);
                    }}
                    className={`p-1.5 rounded-lg transition-all active:scale-95 cursor-pointer ${
                      isFragmentPlaying
                        ? "bg-foreground text-background shadow-2xs"
                        : isActive
                          ? "bg-foreground/15 text-foreground hover:bg-foreground hover:text-background"
                          : "text-muted-foreground/60 hover:text-foreground hover:bg-muted"
                    }`}
                    title={
                      isFragmentPlaying
                        ? "Pause this fragment"
                        : isActive
                          ? "Resume this fragment"
                          : "Play this fragment"
                    }
                  >
                    {isFragmentPlaying ? (
                      <Pause className="w-3.5 h-3.5 fill-current" />
                    ) : (
                      <Play className="w-3.5 h-3.5 fill-current" />
                    )}
                  </button>
                </div>
              </div>

              {/* Transcribed Text */}
              <p
                className={`text-xs sm:text-sm leading-relaxed transition-colors ${
                  isActive ? "text-foreground font-medium" : "text-muted-foreground"
                }`}
              >
                &ldquo;{seg.text}&rdquo;
              </p>
            </div>
          );
        })}
      </div>
    </div>
  );
};
