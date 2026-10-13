import React, { useState } from "react";
import type { PostCallAnalysis } from "@/types/emotion";
import {
  Sparkles,
  CheckCircle2,
  AlertTriangle,
  Flame,
  TrendingUp,
  TrendingDown,
  Minus,
  RefreshCw,
  Loader2,
  Clock,
  ShieldCheck,
  ChevronRight,
  ListTodo,
  UserCheck,
  Award,
  HelpCircle,
} from "lucide-react";

interface PostAnalysisSectionProps {
  analysis: PostCallAnalysis | null;
  isLoading: boolean;
  onGenerate: () => void;
  onSeek: (timeSeconds: number) => void;
}

export const PostAnalysisSection: React.FC<PostAnalysisSectionProps> = ({
  analysis,
  isLoading,
  onGenerate,
  onSeek,
}) => {
  // Local state for checking off action items interactively
  const [checkedTasks, setCheckedTasks] = useState<Record<number, boolean>>({});

  const toggleTask = (idx: number) => {
    setCheckedTasks((prev) => ({
      ...prev,
      [idx]: !prev[idx],
    }));
  };

  const getOutcomeTheme = (outcome: string) => {
    const lower = outcome.toLowerCase();
    if (lower.includes("resolved") || lower.includes("answered")) {
      return {
        badge: "bg-emerald-500/10 text-emerald-700 dark:text-emerald-300 border-emerald-500/20",
        icon: CheckCircle2,
      };
    }
    if (lower.includes("escalat")) {
      return {
        badge: "bg-rose-500/10 text-rose-700 dark:text-rose-300 border-rose-500/20",
        icon: Flame,
      };
    }
    return {
      badge: "bg-amber-500/10 text-amber-700 dark:text-amber-300 border-amber-500/20",
      icon: AlertTriangle,
    };
  };

  const getPriorityBadge = (priority: string) => {
    const p = priority.toLowerCase();
    if (p === "high") {
      return "bg-rose-500/10 text-rose-600 dark:text-rose-300 border-rose-500/20";
    }
    if (p === "medium") {
      return "bg-amber-500/10 text-amber-600 dark:text-amber-300 border-amber-500/20";
    }
    return "bg-muted text-muted-foreground border-border/60";
  };

  const getTrendIcon = (trend: string) => {
    const t = trend.toLowerCase();
    if (t === "improved") return <TrendingUp className="w-3.5 h-3.5 text-emerald-500" />;
    if (t === "declined") return <TrendingDown className="w-3.5 h-3.5 text-rose-500" />;
    return <Minus className="w-3.5 h-3.5 text-muted-foreground" />;
  };

  // If no analysis generated yet and not loading, display a clean call-to-action card
  if (!analysis && !isLoading) {
    return (
      <div className="w-full rounded-2xl border border-dashed border-border/80 bg-muted/20 p-6 sm:p-8 flex flex-col sm:flex-row items-center justify-between gap-5 shadow-2xs">
        <div className="flex items-center gap-4">
          <div className="w-12 h-12 rounded-2xl bg-foreground/5 border border-border/70 flex items-center justify-center shrink-0 text-foreground">
            <Sparkles className="w-6 h-6" />
          </div>
          <div className="space-y-1">
            <h3 className="text-base sm:text-lg font-semibold text-foreground">
              LLM Post-Call Analysis & QA Intelligence
            </h3>
            <p className="text-xs sm:text-sm text-muted-foreground max-w-xl">
              Synthesize full call dialogue, extract customer sentiment trajectory, evaluate agent empathy
              and compliance scores, identify risks, and generate actionable follow-up items using LLM.
            </p>
          </div>
        </div>

        <button
          type="button"
          onClick={onGenerate}
          className="inline-flex items-center gap-2 px-5 py-2.5 rounded-xl text-xs sm:text-sm font-medium bg-foreground text-background hover:opacity-90 active:scale-95 shadow-xs transition-all cursor-pointer shrink-0"
        >
          <Sparkles className="w-4 h-4" />
          <span>Analyze Call with LLM</span>
        </button>
      </div>
    );
  }

  // Loading state
  if (isLoading) {
    return (
      <div className="w-full rounded-2xl border border-border/70 bg-card p-8 sm:p-10 shadow-2xs flex flex-col items-center justify-center gap-3.5 text-center animate-in fade-in-50 duration-200">
        <div className="w-12 h-12 rounded-2xl bg-muted text-foreground flex items-center justify-center">
          <Loader2 className="w-6 h-6 animate-spin text-foreground" />
        </div>
        <div className="space-y-1">
          <h4 className="text-base font-semibold text-foreground">
            Synthesizing Call with LLM...
          </h4>
          <p className="text-xs sm:text-sm text-muted-foreground max-w-md">
            Processing full conversational transcript, evaluating acoustic dissonance, scoring QA metrics, and extracting next steps.
          </p>
        </div>
      </div>
    );
  }

  if (!analysis) return null;

  const outcomeTheme = getOutcomeTheme(analysis.call_outcome);
  const OutcomeIcon = outcomeTheme.icon;

  return (
    <div className="w-full flex flex-col gap-6 rounded-2xl border border-border/70 bg-card p-5 sm:p-7 shadow-2xs animate-in fade-in-50 duration-300">
      {/* 1. Header Bar */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 pb-4 border-b border-border/60">
        <div className="flex items-center gap-3">
          <div className="w-10 h-10 rounded-xl bg-foreground text-background flex items-center justify-center shrink-0 shadow-2xs">
            <Sparkles className="w-5 h-5 fill-current" />
          </div>
          <div>
            <div className="flex flex-wrap items-center gap-2">
              <h3 className="text-base sm:text-lg font-bold text-foreground">
                Post-Call Intelligence & QA Evaluation
              </h3>
              <span className="px-2 py-0.5 rounded text-[11px] font-mono bg-muted text-muted-foreground border border-border/60">
                {analysis.llm_model_used || "LLM Engine"}
              </span>
            </div>
            <p className="text-xs text-muted-foreground">
              Deep multimodal synthesis of caller intent, emotional progression, and QA metrics
            </p>
          </div>
        </div>

        <div className="flex items-center gap-2 self-start sm:self-auto">
          <button
            type="button"
            onClick={onGenerate}
            disabled={isLoading}
            className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-medium border border-border/70 bg-card hover:bg-muted text-muted-foreground hover:text-foreground transition-all active:scale-95 cursor-pointer shadow-2xs"
            title="Re-run LLM analysis"
          >
            <RefreshCw className={`w-3.5 h-3.5 ${isLoading ? "animate-spin" : ""}`} />
            <span>Re-analyze</span>
          </button>
        </div>
      </div>

      {/* 2. Top Metric Cards: Resolution, CSAT, Overall QA, Escalation Risk */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-3.5">
        {/* Call Outcome */}
        <div className="p-4 rounded-xl border border-border/60 bg-muted/20 flex flex-col justify-between gap-2">
          <span className="text-[11px] uppercase tracking-wider font-medium text-muted-foreground">
            Call Outcome
          </span>
          <div className="flex items-center gap-2">
            <span
              className={`inline-flex items-center gap-1.5 px-2.5 py-1 rounded-lg text-xs font-semibold border ${outcomeTheme.badge}`}
            >
              <OutcomeIcon className="w-3.5 h-3.5" />
              <span>{analysis.call_outcome}</span>
            </span>
          </div>
        </div>

        {/* Customer Satisfaction Score */}
        <div className="p-4 rounded-xl border border-border/60 bg-muted/20 flex flex-col justify-between gap-2">
          <span className="text-[11px] uppercase tracking-wider font-medium text-muted-foreground">
            Estimated CSAT
          </span>
          <div className="flex items-baseline gap-1.5">
            <span className="text-2xl font-extrabold text-foreground font-mono">
              {analysis.customer_satisfaction_score.toFixed(1)}
            </span>
            <span className="text-xs text-muted-foreground font-mono">/ 10</span>
          </div>
        </div>

        {/* Agent Overall QA Score */}
        <div className="p-4 rounded-xl border border-border/60 bg-muted/20 flex flex-col justify-between gap-2">
          <span className="text-[11px] uppercase tracking-wider font-medium text-muted-foreground">
            Agent QA Score
          </span>
          <div className="flex items-baseline gap-1.5">
            <span className="text-2xl font-extrabold text-foreground font-mono">
              {(analysis.agent_performance.overall_score * 10).toFixed(0)}%
            </span>
            <span className="text-xs text-muted-foreground font-mono">
              ({analysis.agent_performance.overall_score.toFixed(1)}/10)
            </span>
          </div>
        </div>

        {/* Risk / Churn Signal */}
        <div className="p-4 rounded-xl border border-border/60 bg-muted/20 flex flex-col justify-between gap-2">
          <span className="text-[11px] uppercase tracking-wider font-medium text-muted-foreground">
            Escalation Risk
          </span>
          <div className="flex items-center gap-2">
            <span
              className={`inline-flex items-center gap-1 px-2.5 py-0.5 rounded-lg text-xs font-semibold border ${
                analysis.risk_assessment.escalation_risk.toLowerCase() === "high"
                  ? "bg-rose-500/10 text-rose-700 dark:text-rose-300 border-rose-500/20"
                  : analysis.risk_assessment.escalation_risk.toLowerCase() === "medium"
                    ? "bg-amber-500/10 text-amber-700 dark:text-amber-300 border-amber-500/20"
                    : "bg-emerald-500/10 text-emerald-700 dark:text-emerald-300 border-emerald-500/20"
              }`}
            >
              <ShieldCheck className="w-3.5 h-3.5" />
              <span>{analysis.risk_assessment.escalation_risk} Risk</span>
            </span>
          </div>
        </div>
      </div>

      {/* 3. Executive Summary & Outcome Reasoning */}
      <div className="p-4.5 rounded-xl border border-border/60 bg-muted/30 flex flex-col gap-2.5">
        <h4 className="text-xs uppercase tracking-wider font-semibold text-foreground/80 flex items-center gap-1.5">
          <Award className="w-3.5 h-3.5 text-muted-foreground" />
          <span>Executive Summary & Resolution Note</span>
        </h4>
        <p className="text-xs sm:text-sm text-foreground/90 leading-relaxed">
          {analysis.executive_summary}
        </p>
        {analysis.outcome_reasoning && (
          <p className="text-xs text-muted-foreground italic pt-1 border-t border-border/50">
            <strong>Outcome Rationale:</strong> {analysis.outcome_reasoning}
          </p>
        )}
      </div>

      {/* 4. Two-Column Layout: Sentiment Trajectory + Agent QA Scorecard */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-5 items-start">
        {/* Left Column: Sentiment Trajectory & Emotional Journey */}
        <div className="lg:col-span-5 p-4.5 rounded-xl border border-border/60 bg-card flex flex-col gap-4">
          <div className="flex items-center justify-between pb-2 border-b border-border/50">
            <span className="text-xs uppercase tracking-wider font-semibold text-foreground flex items-center gap-1.5">
              <TrendingUp className="w-3.5 h-3.5 text-muted-foreground" />
              <span>Customer Sentiment Trajectory</span>
            </span>
            <div className="flex items-center gap-1 px-2 py-0.5 rounded text-[11px] font-medium bg-muted border border-border/60">
              {getTrendIcon(analysis.sentiment_trajectory.trend)}
              <span className="capitalize">{analysis.sentiment_trajectory.trend}</span>
            </div>
          </div>

          {/* Journey Pills */}
          <div className="flex items-center justify-between gap-2 p-3 rounded-lg bg-muted/40 border border-border/40">
            <div className="flex flex-col items-center gap-1 text-center">
              <span className="text-[10px] text-muted-foreground uppercase tracking-wider">Start Mood</span>
              <span className="px-2.5 py-0.5 rounded-md text-xs font-semibold bg-muted text-foreground border border-border/60">
                {analysis.sentiment_trajectory.start_emotion}
              </span>
            </div>

            <ChevronRight className="w-4 h-4 text-muted-foreground/60 shrink-0" />

            <div className="flex flex-col items-center gap-1 text-center">
              <span className="text-[10px] text-muted-foreground uppercase tracking-wider">End Mood</span>
              <span className="px-2.5 py-0.5 rounded-md text-xs font-semibold bg-foreground text-background shadow-2xs">
                {analysis.sentiment_trajectory.end_emotion}
              </span>
            </div>
          </div>

          <p className="text-xs text-muted-foreground leading-relaxed">
            {analysis.sentiment_trajectory.details ||
              "Tracked caller emotion from initial turn through handover resolution."}
          </p>

          {/* Risk Factors Breakdown */}
          {analysis.risk_assessment.risk_factors.length > 0 && (
            <div className="pt-2 border-t border-border/50 space-y-1.5">
              <span className="text-[11px] uppercase tracking-wider font-medium text-muted-foreground flex items-center gap-1">
                <AlertTriangle className="w-3 h-3 text-amber-500" />
                <span>Detected Risk Indicators</span>
              </span>
              <ul className="space-y-1 text-xs text-muted-foreground">
                {analysis.risk_assessment.risk_factors.map((rf, idx) => (
                  <li key={idx} className="flex items-start gap-1.5">
                    <span className="text-amber-500">•</span>
                    <span>{rf}</span>
                  </li>
                ))}
              </ul>
            </div>
          )}
        </div>

        {/* Right Column: Agent QA Performance Scorecard */}
        <div className="lg:col-span-7 p-4.5 rounded-xl border border-border/60 bg-card flex flex-col gap-4">
          <div className="flex items-center justify-between pb-2 border-b border-border/50">
            <span className="text-xs uppercase tracking-wider font-semibold text-foreground flex items-center gap-1.5">
              <UserCheck className="w-3.5 h-3.5 text-muted-foreground" />
              <span>Agent Performance & QA Scorecard</span>
            </span>
            <span className="font-mono text-xs font-bold text-foreground">
              {analysis.agent_performance.overall_score.toFixed(1)} / 10
            </span>
          </div>

          {/* Detailed QA Bars */}
          <div className="space-y-2.5">
            {/* Empathy */}
            <div className="space-y-1">
              <div className="flex justify-between text-xs">
                <span className="text-foreground/80 font-medium">Empathy & Emotional Attunement</span>
                <span className="font-mono font-semibold text-foreground">
                  {analysis.agent_performance.empathy_score.toFixed(1)} / 10
                </span>
              </div>
              <div className="w-full h-1.5 rounded-full bg-muted overflow-hidden">
                <div
                  className="h-full rounded-full bg-foreground/80 transition-all duration-300"
                  style={{ width: `${Math.min(100, (analysis.agent_performance.empathy_score / 10) * 100)}%` }}
                />
              </div>
            </div>

            {/* Professionalism */}
            <div className="space-y-1">
              <div className="flex justify-between text-xs">
                <span className="text-foreground/80 font-medium">Professionalism & Tone Composure</span>
                <span className="font-mono font-semibold text-foreground">
                  {analysis.agent_performance.professionalism_score.toFixed(1)} / 10
                </span>
              </div>
              <div className="w-full h-1.5 rounded-full bg-muted overflow-hidden">
                <div
                  className="h-full rounded-full bg-foreground/80 transition-all duration-300"
                  style={{ width: `${Math.min(100, (analysis.agent_performance.professionalism_score / 10) * 100)}%` }}
                />
              </div>
            </div>

            {/* Active Listening */}
            <div className="space-y-1">
              <div className="flex justify-between text-xs">
                <span className="text-foreground/80 font-medium">Active Listening & Turn Adherence</span>
                <span className="font-mono font-semibold text-foreground">
                  {analysis.agent_performance.active_listening_score.toFixed(1)} / 10
                </span>
              </div>
              <div className="w-full h-1.5 rounded-full bg-muted overflow-hidden">
                <div
                  className="h-full rounded-full bg-foreground/80 transition-all duration-300"
                  style={{ width: `${Math.min(100, (analysis.agent_performance.active_listening_score / 10) * 100)}%` }}
                />
              </div>
            </div>
          </div>

          {/* Strengths & Improvement Areas */}
          <div className="grid grid-cols-1 sm:grid-cols-2 gap-3 pt-2 border-t border-border/50">
            {/* Strengths */}
            <div className="space-y-1.5">
              <span className="text-[11px] uppercase tracking-wider font-semibold text-emerald-700 dark:text-emerald-400 flex items-center gap-1">
                <CheckCircle2 className="w-3 h-3" />
                <span>Demonstrated Strengths</span>
              </span>
              <ul className="space-y-1 text-xs text-muted-foreground">
                {analysis.agent_performance.strengths.map((str, idx) => (
                  <li key={idx} className="flex items-start gap-1.5">
                    <span className="text-emerald-500">•</span>
                    <span>{str}</span>
                  </li>
                ))}
              </ul>
            </div>

            {/* Coaching Tips */}
            <div className="space-y-1.5">
              <span className="text-[11px] uppercase tracking-wider font-semibold text-amber-700 dark:text-amber-400 flex items-center gap-1">
                <HelpCircle className="w-3 h-3" />
                <span>Coaching / Improvement</span>
              </span>
              <ul className="space-y-1 text-xs text-muted-foreground">
                {analysis.agent_performance.areas_for_improvement.map((area, idx) => (
                  <li key={idx} className="flex items-start gap-1.5">
                    <span className="text-amber-500">•</span>
                    <span>{area}</span>
                  </li>
                ))}
              </ul>
            </div>
          </div>
        </div>
      </div>

      {/* 5. Action Items & Follow-up Tasks */}
      {analysis.action_items && analysis.action_items.length > 0 && (
        <div className="p-4.5 rounded-xl border border-border/60 bg-muted/15 flex flex-col gap-3">
          <div className="flex items-center justify-between pb-1.5 border-b border-border/50">
            <span className="text-xs uppercase tracking-wider font-semibold text-foreground flex items-center gap-1.5">
              <ListTodo className="w-3.5 h-3.5 text-muted-foreground" />
              <span>Agreed Action Items & Follow-up Checklist</span>
            </span>
            <span className="text-[11px] text-muted-foreground font-mono">
              {Object.values(checkedTasks).filter(Boolean).length} / {analysis.action_items.length} Completed
            </span>
          </div>

          <div className="space-y-2">
            {analysis.action_items.map((item, idx) => {
              const isChecked = !!checkedTasks[idx];
              return (
                <div
                  key={idx}
                  onClick={() => toggleTask(idx)}
                  className={`p-3 rounded-xl border transition-all cursor-pointer flex items-center justify-between gap-3 ${
                    isChecked
                      ? "bg-muted/30 border-border/40 opacity-70"
                      : "bg-card border-border/60 hover:border-border shadow-2xs"
                  }`}
                >
                  <div className="flex items-center gap-3">
                    <input
                      type="checkbox"
                      checked={isChecked}
                      onChange={() => toggleTask(idx)}
                      className="w-4 h-4 rounded border-border accent-foreground cursor-pointer"
                    />
                    <span
                      className={`text-xs sm:text-sm font-medium ${
                        isChecked ? "line-through text-muted-foreground" : "text-foreground"
                      }`}
                    >
                      {item.task}
                    </span>
                  </div>

                  <div className="flex items-center gap-2 shrink-0">
                    <span className="px-2 py-0.5 rounded text-[10px] font-medium bg-muted text-foreground/80 border border-border/60">
                      {item.owner || "Agent"}
                    </span>
                    <span
                      className={`px-2 py-0.5 rounded text-[10px] font-semibold border ${getPriorityBadge(
                        item.priority
                      )}`}
                    >
                      {item.priority}
                    </span>
                  </div>
                </div>
              );
            })}
          </div>
        </div>
      )}

      {/* 6. Critical Key Moments with Clickable Audio Seek Timestamps */}
      {analysis.key_moments && analysis.key_moments.length > 0 && (
        <div className="p-4.5 rounded-xl border border-border/60 bg-card flex flex-col gap-3">
          <div className="flex items-center justify-between pb-1.5 border-b border-border/50">
            <span className="text-xs uppercase tracking-wider font-semibold text-foreground flex items-center gap-1.5">
              <Clock className="w-3.5 h-3.5 text-muted-foreground" />
              <span>Critical Dialogue Highlights (Click timestamp to jump)</span>
            </span>
            <span className="text-[11px] text-muted-foreground">
              Direct audio navigation
            </span>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-3">
            {analysis.key_moments.map((km, idx) => (
              <div
                key={idx}
                onClick={() => onSeek(km.time_seconds)}
                className="group p-3.5 rounded-xl border border-border/60 bg-muted/20 hover:bg-muted/40 hover:border-border transition-all cursor-pointer flex flex-col justify-between gap-2.5"
                title={`Seek audio to ${km.timestamp}`}
              >
                <div className="flex items-center justify-between gap-2">
                  <button
                    type="button"
                    onClick={(e) => {
                      e.stopPropagation();
                      onSeek(km.time_seconds);
                    }}
                    className="inline-flex items-center gap-1 px-2 py-0.5 rounded text-xs font-mono font-semibold bg-foreground text-background shadow-2xs group-hover:scale-105 transition-transform cursor-pointer"
                  >
                    <span>{km.timestamp}</span>
                  </button>

                  <span className="text-xs font-medium text-muted-foreground">
                    {km.speaker}
                  </span>
                </div>

                <p className="text-xs text-foreground/90 italic line-clamp-2 leading-relaxed">
                  &ldquo;{km.quote}&rdquo;
                </p>

                <p className="text-[11px] text-muted-foreground border-t border-border/40 pt-1.5">
                  {km.insight}
                </p>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
};
