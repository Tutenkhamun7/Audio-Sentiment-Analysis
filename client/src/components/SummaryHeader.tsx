import React from "react";
import { Check, FileAudio, Users, RotateCcw } from "lucide-react";
import { getSpeakerTheme } from "@/lib/constants";

interface SummaryHeaderProps {
  segmentCount: number;
  speakers: string[];
  filename: string;
  onReset: () => void;
}

export const SummaryHeader: React.FC<SummaryHeaderProps> = ({
  segmentCount,
  speakers,
  filename,
  onReset,
}) => {
  return (
    <div className="w-full flex flex-col md:flex-row md:items-center justify-between gap-4 p-5 rounded-2xl border border-border/70 bg-card shadow-2xs">
      <div className="flex flex-col gap-2">
        <div className="flex items-center gap-2.5">
          <div className="flex items-center justify-center w-6 h-6 rounded-md bg-muted text-foreground/80 shrink-0">
            <Check className="w-3.5 h-3.5" />
          </div>
          <h2 className="text-base sm:text-lg font-semibold tracking-tight text-foreground">
            Analysis Complete
          </h2>
          <span className="hidden sm:inline text-xs text-muted-foreground/60">•</span>
          <span className="text-xs sm:text-sm text-muted-foreground">
            {segmentCount} speech segment{segmentCount !== 1 ? "s" : ""} transcribed & emotion-mapped
          </span>
        </div>

        {/* Badges: Filename & Speaker Chips */}
        <div className="flex flex-wrap items-center gap-2 pt-0.5">
          {/* Filename badge */}
          <div className="inline-flex items-center gap-1.5 px-2.5 py-1 rounded-md text-xs font-mono text-muted-foreground bg-muted/60 border border-border/60">
            <FileAudio className="w-3 h-3 text-muted-foreground" />
            <span className="max-w-[220px] truncate">{filename}</span>
          </div>

          {/* Speakers count badge */}
          <div className="inline-flex items-center gap-1.5 px-2.5 py-1 rounded-md text-xs text-muted-foreground bg-muted/60 border border-border/60">
            <Users className="w-3 h-3" />
            <span>
              {speakers.length} Speaker{speakers.length !== 1 ? "s" : ""}
            </span>
          </div>

          {/* Individual Speaker chips */}
          {speakers.map((speaker) => {
            const theme = getSpeakerTheme(speaker);
            return (
              <div
                key={speaker}
                className={`inline-flex items-center gap-1.5 px-2.5 py-1 rounded-md text-xs font-medium border ${theme.chip}`}
              >
                <span className={`w-1.5 h-1.5 rounded-full ${theme.dot}`} />
                <span>{speaker}</span>
              </div>
            );
          })}
        </div>
      </div>

      {/* Reset / Analyze Another Audio Action */}
      <div className="shrink-0 flex items-center">
        <button
          type="button"
          onClick={onReset}
          className="inline-flex items-center gap-2 px-3.5 py-2 rounded-lg text-xs font-medium text-foreground bg-secondary hover:bg-secondary/80 border border-border/70 transition-colors cursor-pointer"
        >
          <RotateCcw className="w-3.5 h-3.5 text-muted-foreground" />
          <span>Analyze Another</span>
        </button>
      </div>
    </div>
  );
};
