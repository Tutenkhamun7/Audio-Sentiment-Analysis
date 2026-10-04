import React from "react";
import type { AgentSynthesis } from "@/types/emotion";
import {
  Brain,
  AlertCircle,
  ShieldCheck,
  Flame,
  Clock,
  Users,
} from "lucide-react";
import { getSpeakerTheme } from "@/lib/constants";

interface AgentSynthesisCardProps {
  agentContext: AgentSynthesis;
}

export const AgentSynthesisCard: React.FC<AgentSynthesisCardProps> = ({
  agentContext,
}) => {
  return (
    <div className="w-full rounded-2xl border border-border/70 bg-card p-5 sm:p-6 shadow-2xs flex flex-col gap-4">
      {/* Header */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 pb-3 border-b border-border/60">
        <div className="flex items-center gap-2.5">
          <div className="p-2 rounded-lg bg-muted text-muted-foreground">
            <Brain className="w-4 h-4" />
          </div>
          <div>
            <div className="flex items-center gap-2">
              <h3 className="text-sm sm:text-base font-semibold text-foreground">
                AI Conversational Synthesis
              </h3>
              <span className="px-1.5 py-0.5 rounded text-[10px] font-mono font-medium bg-muted text-muted-foreground border border-border/60">
                Reasoning Agent
              </span>
            </div>
            <p className="text-xs text-muted-foreground">
              Transcript & acoustic prosody cross-evaluation
            </p>
          </div>
        </div>

        {/* Escalation / Hostility Flag */}
        <div>
          {agentContext.escalation_detected ? (
            <div className="inline-flex items-center gap-1.5 px-2.5 py-1 rounded-md text-xs font-medium bg-rose-500/10 text-rose-700 dark:text-rose-300 border border-rose-500/20">
              <Flame className="w-3.5 h-3.5" />
              <span>Escalation Detected</span>
            </div>
          ) : (
            <div className="inline-flex items-center gap-1.5 px-2.5 py-1 rounded-md text-xs font-medium bg-muted/60 text-muted-foreground border border-border/60">
              <ShieldCheck className="w-3.5 h-3.5 text-muted-foreground" />
              <span>No Escalation</span>
            </div>
          )}
        </div>
      </div>

      {/* Summary Narrative */}
      {agentContext.summary && (
        <div className="p-3.5 rounded-xl bg-muted/40 border border-border/50 text-xs sm:text-sm leading-relaxed text-foreground/85">
          <p>{agentContext.summary}</p>
        </div>
      )}

      {/* Grid: Speaker Sentiments & Conversation Telemetry */}
      <div className="grid grid-cols-1 md:grid-cols-12 gap-4">
        {/* Speaker Sentiments */}
        {agentContext.primary_speaker_sentiments &&
          Object.keys(agentContext.primary_speaker_sentiments).length > 0 && (
            <div className="md:col-span-7 flex flex-col gap-2">
              <span className="text-[11px] font-medium uppercase tracking-wider text-muted-foreground flex items-center gap-1.5">
                <Users className="w-3 h-3" />
                <span>Speaker Behavioral Profiles</span>
              </span>
              <div className="space-y-2">
                {Object.entries(agentContext.primary_speaker_sentiments).map(
                  ([spk, sentiment]) => {
                    const theme = getSpeakerTheme(spk);
                    return (
                      <div
                        key={spk}
                        className="p-3 rounded-xl border border-border/60 bg-muted/20 text-xs sm:text-sm flex flex-col gap-1"
                      >
                        <div className="flex items-center gap-2">
                          <span
                            className="inline-flex items-center gap-1.5 px-2 py-0.5 rounded text-[11px] font-medium bg-muted text-foreground/80 border border-border/60"
                          >
                            <span className={`w-1.5 h-1.5 rounded-full ${theme.dot}`} />
                            {spk}
                          </span>
                        </div>
                        <p className="text-muted-foreground leading-normal mt-0.5">{sentiment}</p>
                      </div>
                    );
                  }
                )}
              </div>
            </div>
          )}

        {/* Conversational Telemetry (Interruptions & Overtalk) */}
        <div className="md:col-span-5 flex flex-col gap-2">
          <span className="text-[11px] font-medium uppercase tracking-wider text-muted-foreground flex items-center gap-1.5">
            <Clock className="w-3 h-3" />
            <span>Turn Dynamics & Overlap</span>
          </span>

          <div className="grid grid-cols-2 gap-2">
            <div className="p-3 rounded-xl border border-border/60 bg-muted/20 flex flex-col items-center justify-center text-center">
              <span className="text-lg font-semibold font-mono text-foreground">
                {agentContext.interruption_count}
              </span>
              <span className="text-[11px] text-muted-foreground mt-0.5">
                Interruptions
              </span>
            </div>

            <div className="p-3 rounded-xl border border-border/60 bg-muted/20 flex flex-col items-center justify-center text-center">
              <span className="text-lg font-semibold font-mono text-foreground">
                {Math.round(agentContext.overtalk_ratio * 100)}%
              </span>
              <span className="text-[11px] text-muted-foreground mt-0.5">
                Overlap ({agentContext.total_overtalk_seconds.toFixed(1)}s)
              </span>
            </div>
          </div>

          {/* Flagged Anomalies */}
          {agentContext.flagged_anomalies &&
            agentContext.flagged_anomalies.length > 0 && (
              <div className="p-3 rounded-xl border border-border/70 bg-muted/30 text-xs space-y-1 mt-1">
                <span className="font-medium flex items-center gap-1 text-[11px] uppercase tracking-wider text-foreground/75">
                  <AlertCircle className="w-3.5 h-3.5 text-muted-foreground" />
                  <span>Flagged Dissonance</span>
                </span>
                <ul className="list-disc list-inside space-y-0.5 pl-1 text-[11px] text-muted-foreground">
                  {agentContext.flagged_anomalies.map((anomaly, idx) => (
                    <li key={idx} className="leading-snug">
                      {anomaly}
                    </li>
                  ))}
                </ul>
              </div>
            )}
        </div>
      </div>
    </div>
  );
};
