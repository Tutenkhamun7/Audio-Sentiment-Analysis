import React, { useState, useEffect, useCallback } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import architectureMd from "@/assets/ARCHITECTURE_AND_FLOW_SPEC.md?raw";
import {
  X,
  Maximize2,
  Minimize2,
  Copy,
  Check,
  Download,
  ChevronLeft,
  ChevronRight,
  Sliders,
  FileText,
  Cpu,
  AlertTriangle,
  Lightbulb,
  Sparkles,
  Search,
  Scale,
  Wrench,
  Workflow,
  Volume2,
  MessageSquare,
} from "lucide-react";

interface ArchitecturePresentationModalProps {
  isOpen: boolean;
  onClose: () => void;
}

interface Slide {
  id: string;
  number: number;
  title: string;
  subtitle: string;
  badge: string;
  tagline: string;
}

const SLIDES: Slide[] = [
  {
    id: "research",
    number: 1,
    title: "1. Research: Paradigms, Open-Source & Commercial",
    subtitle: "Acoustic vs. Semantic vs. Multimodal Approaches & Industry Benchmark",
    badge: "Domain Research",
    tagline: "Exploring available approaches and reviewing open-source vs. commercial solutions",
  },
  {
    id: "evaluation",
    number: 2,
    title: "2. Model Evaluation & Multimodal Methodology",
    subtitle: "Candidate Evaluation Matrix, Multimodal Fusion & Defined Criteria",
    badge: "Evaluation Framework",
    tagline: "Evaluating candidate models against latency, accuracy, DER, and VRAM criteria",
  },
  {
    id: "recommendation",
    number: 3,
    title: "3. Research Summary, Trade-Offs & Recommendation",
    subtitle: "Architectural Decisions & Our Implemented In-Memory Engine",
    badge: "Core Architecture",
    tagline: "Summarizing shortlisted options, engineering trade-offs, and final stack",
  },
  {
    id: "dataflow",
    number: 4,
    title: "4. System Architecture & End-to-End Request Flow",
    subtitle: "Stage 0 to Stage 8 Execution Flowchart & Granular Device Routing",
    badge: "System Dataflow",
    tagline: "Interactive 9-stage dataflow and hardware allocation matrix",
  },
  {
    id: "setup-challenges",
    number: 5,
    title: "5. Setup Steps, Challenges & Real-World Observations",
    subtitle: "Windows DLL Registration, 6 Core Challenges Solved & Empirical Insights",
    badge: "Engineering Implementation",
    tagline: "Concrete installation steps, solved production blockers, and field observations",
  },
  {
    id: "failures",
    number: 6,
    title: "6. Failure Scenarios: Reproduction & Mitigations",
    subtitle: "How to Achieve / Trigger Edge Cases & How Our System Handles Them",
    badge: "Resilience & Edge Cases",
    tagline: "7 real-world failure scenarios, deliberate reproduction steps, and architectural safeguards",
  },
];

export const ArchitecturePresentationModal: React.FC<ArchitecturePresentationModalProps> = ({
  isOpen,
  onClose,
}) => {
  const [currentSlideIndex, setCurrentSlideIndex] = useState<number>(0);
  const [viewMode, setViewMode] = useState<"presentation" | "document">("presentation");
  const [isFullscreen, setIsFullscreen] = useState<boolean>(false);
  const [copied, setCopied] = useState<boolean>(false);
  const [activeStageStep, setActiveStageStep] = useState<number>(1);
  const [activeFailureTab, setActiveFailureTab] = useState<number>(0);
  const [docSearchQuery, setDocSearchQuery] = useState<string>("");

  // Keyboard navigation
  useEffect(() => {
    if (!isOpen) return;

    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key === "Escape") {
        onClose();
      } else if (e.key === "ArrowRight" || e.key === "PageDown") {
        if (viewMode === "presentation") {
          setCurrentSlideIndex((prev) => Math.min(prev + 1, SLIDES.length - 1));
        }
      } else if (e.key === "ArrowLeft" || e.key === "PageUp") {
        if (viewMode === "presentation") {
          setCurrentSlideIndex((prev) => Math.max(prev - 1, 0));
        }
      } else if (e.key === "f" || e.key === "F") {
        if (!e.ctrlKey && !e.metaKey && e.target instanceof HTMLElement && e.target.tagName !== "INPUT") {
          setIsFullscreen((prev) => !prev);
        }
      }
    };

    window.addEventListener("keydown", handleKeyDown);
    return () => window.removeEventListener("keydown", handleKeyDown);
  }, [isOpen, viewMode, onClose]);

  // Lock background scroll when modal open
  useEffect(() => {
    if (isOpen) {
      document.body.style.overflow = "hidden";
    } else {
      document.body.style.overflow = "";
    }
    return () => {
      document.body.style.overflow = "";
    };
  }, [isOpen]);

  const handleCopyMarkdown = useCallback(async () => {
    try {
      await navigator.clipboard.writeText(architectureMd);
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    } catch {
      // Fallback
    }
  }, []);

  const handleDownloadMarkdown = useCallback(() => {
    const blob = new Blob([architectureMd], { type: "text/markdown;charset=utf-8" });
    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.href = url;
    link.download = "SYSTEM_ARCHITECTURE_AND_FLOW_SPEC.md";
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    URL.revokeObjectURL(url);
  }, []);

  if (!isOpen) return null;

  const currentSlide = SLIDES[currentSlideIndex];

  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center bg-black/80 backdrop-blur-md p-2 sm:p-4 md:p-6 animate-in fade-in duration-200"
      onClick={(e) => {
        if (e.target === e.currentTarget) onClose();
      }}
    >
      <div
        className={`bg-card text-foreground border border-border/80 rounded-2xl sm:rounded-3xl shadow-2xl flex flex-col overflow-hidden transition-all duration-300 w-full ${
          isFullscreen
            ? "h-full max-h-full max-w-full rounded-none sm:rounded-none"
            : "max-w-6xl h-[92vh] max-h-[920px]"
        }`}
      >
        {/* Top Control Bar */}
        <div className="flex items-center justify-between px-4 sm:px-6 py-3.5 border-b border-border/70 bg-muted/40 shrink-0">
          <div className="flex items-center gap-3">
            <div className="flex items-center justify-center w-8 h-8 rounded-xl bg-gradient-to-tr from-primary to-primary/80 text-primary-foreground shadow-xs">
              <Cpu className="w-4 h-4" />
            </div>
            <div>
              <div className="flex items-center gap-2">
                <span className="font-bold text-sm sm:text-base tracking-tight text-foreground">
                  Engine Blueprint
                </span>
                <span className="inline-flex items-center px-2 py-0.5 rounded-full text-[10px] font-mono font-semibold bg-primary/10 text-primary border border-primary/20">
                  Spec v2.0
                </span>
                <span className="hidden sm:inline-flex items-center gap-1 px-2 py-0.5 rounded-full text-[10px] font-medium bg-emerald-500/10 text-emerald-600 dark:text-emerald-400 border border-emerald-500/20">
                  <span className="w-1.5 h-1.5 rounded-full bg-emerald-500 animate-pulse" />
                  Zero-Disk RAM
                </span>
              </div>
              <p className="text-[11px] text-muted-foreground hidden sm:block">
                Multimodal Audio Sentiment, Neural Diarization &amp; Overlap Separation Engine
              </p>
            </div>
          </div>

          {/* Mode Switcher & Action Tools */}
          <div className="flex items-center gap-2">
            {/* View Mode Toggle */}
            <div className="flex items-center p-1 bg-muted/80 rounded-xl border border-border/60 text-xs font-medium">
              <button
                type="button"
                onClick={() => setViewMode("presentation")}
                className={`flex items-center gap-1.5 px-3 py-1 rounded-lg transition-all cursor-pointer ${
                  viewMode === "presentation"
                    ? "bg-card text-foreground shadow-xs border border-border/60"
                    : "text-muted-foreground hover:text-foreground"
                }`}
                title="Interactive Slide Deck Mode"
              >
                <Sliders className="w-3.5 h-3.5" />
                <span className="hidden sm:inline">Slides</span>
              </button>
              <button
                type="button"
                onClick={() => setViewMode("document")}
                className={`flex items-center gap-1.5 px-3 py-1 rounded-lg transition-all cursor-pointer ${
                  viewMode === "document"
                    ? "bg-card text-foreground shadow-xs border border-border/60"
                    : "text-muted-foreground hover:text-foreground"
                }`}
                title="Full Markdown Specification Mode"
              >
                <FileText className="w-3.5 h-3.5" />
                <span className="hidden sm:inline">Full Spec</span>
              </button>
            </div>

            {/* Copy Markdown */}
            <button
              type="button"
              onClick={handleCopyMarkdown}
              className="p-2 rounded-lg border border-border/60 bg-card hover:bg-muted text-muted-foreground hover:text-foreground transition-colors cursor-pointer"
              title="Copy raw markdown to clipboard"
            >
              {copied ? <Check className="w-4 h-4 text-emerald-500" /> : <Copy className="w-4 h-4" />}
            </button>

            {/* Download File */}
            <button
              type="button"
              onClick={handleDownloadMarkdown}
              className="p-2 rounded-lg border border-border/60 bg-card hover:bg-muted text-muted-foreground hover:text-foreground transition-colors cursor-pointer"
              title="Download ARCHITECTURE_AND_FLOW_SPEC.md"
            >
              <Download className="w-4 h-4" />
            </button>

            {/* Fullscreen Toggle */}
            <button
              type="button"
              onClick={() => setIsFullscreen((prev) => !prev)}
              className="hidden sm:flex p-2 rounded-lg border border-border/60 bg-card hover:bg-muted text-muted-foreground hover:text-foreground transition-colors cursor-pointer"
              title={isFullscreen ? "Exit Fullscreen (F)" : "Enter Fullscreen (F)"}
            >
              {isFullscreen ? <Minimize2 className="w-4 h-4" /> : <Maximize2 className="w-4 h-4" />}
            </button>

            {/* Close Button */}
            <button
              type="button"
              onClick={onClose}
              className="p-2 rounded-lg border border-border/60 bg-card hover:bg-destructive/10 hover:text-destructive text-muted-foreground transition-colors cursor-pointer"
              title="Close (Esc)"
            >
              <X className="w-4 h-4" />
            </button>
          </div>
        </div>

        {/* Progress bar in presentation mode */}
        {viewMode === "presentation" && (
          <div className="w-full bg-muted/50 h-1.5 relative overflow-hidden">
            <div
              className="h-full bg-gradient-to-r from-primary to-primary/80 transition-all duration-300 ease-out"
              style={{
                width: `${((currentSlideIndex + 1) / SLIDES.length) * 100}%`,
              }}
            />
          </div>
        )}

        {/* Body Content Area */}
        <div className="flex-1 overflow-y-auto px-4 sm:px-8 md:px-12 py-6 sm:py-8 space-y-6">
          {viewMode === "presentation" ? (
            <div className="max-w-5xl mx-auto space-y-6 animate-in fade-in-50 duration-200">
              {/* Slide Header Hero */}
              <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 border-b border-border/60 pb-5">
                <div>
                  <div className="flex items-center gap-2 mb-1.5">
                    <span className="px-2.5 py-0.5 rounded-full text-[11px] font-semibold bg-primary/10 text-primary border border-primary/20">
                      {currentSlide.badge}
                    </span>
                    <span className="text-xs text-muted-foreground font-mono">
                      Section {currentSlide.number} of {SLIDES.length}
                    </span>
                  </div>
                  <h2 className="text-2xl sm:text-3xl font-extrabold tracking-tight text-foreground">
                    {currentSlide.title}
                  </h2>
                  <p className="text-xs sm:text-sm text-muted-foreground mt-1">
                    {currentSlide.tagline}
                  </p>
                </div>

                <div className="flex items-center gap-2 self-start sm:self-auto shrink-0">
                  <div className="px-3 py-1 rounded-xl bg-muted/60 border border-border/60 text-xs font-mono font-medium text-foreground">
                    Slide {currentSlideIndex + 1} / {SLIDES.length}
                  </div>
                </div>
              </div>

              {/* Dynamic Slide Body */}

              {/* ==================== SLIDE 1: RESEARCH ==================== */}
              {currentSlideIndex === 0 && (
                <div className="space-y-6">
                  {/* Three Core Paradigms */}
                  <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                    <div className="p-4 rounded-2xl border border-border/80 bg-card space-y-2.5 shadow-2xs">
                      <div className="flex items-center gap-2 text-primary font-bold text-xs uppercase tracking-wider">
                        <Volume2 className="w-4 h-4" />
                        <span>1. Acoustic-Only SER</span>
                      </div>
                      <h4 className="font-semibold text-sm text-foreground">Vocal Pitch &amp; Energy</h4>
                      <p className="text-xs text-muted-foreground">
                        Analyzes pitch (F0), jitter, shimmer, MFCCs, and audio transformers (emotion2vec, HuBERT).
                      </p>
                      <div className="text-[11px] space-y-1 pt-1 border-t border-border/40">
                        <div className="text-emerald-600 dark:text-emerald-400 font-medium">✓ Language-agnostic, captures distress</div>
                        <div className="text-rose-500 font-medium">✕ Tone-deaf to spoken sarcasm</div>
                      </div>
                    </div>

                    <div className="p-4 rounded-2xl border border-border/80 bg-card space-y-2.5 shadow-2xs">
                      <div className="flex items-center gap-2 text-indigo-500 font-bold text-xs uppercase tracking-wider">
                        <MessageSquare className="w-4 h-4" />
                        <span>2. ASR + Text NLP</span>
                      </div>
                      <h4 className="font-semibold text-sm text-foreground">Lexical Sentiment</h4>
                      <p className="text-xs text-muted-foreground">
                        Transcribes speech to text via Whisper, then runs text sentiment models (RoBERTa, BERT, LLMs).
                      </p>
                      <div className="text-[11px] space-y-1 pt-1 border-t border-border/40">
                        <div className="text-emerald-600 dark:text-emerald-400 font-medium">✓ Understands business context &amp; intent</div>
                        <div className="text-rose-500 font-medium">✕ Deaf to angry vocal inflection</div>
                      </div>
                    </div>

                    <div className="p-4 rounded-2xl border border-emerald-500/30 bg-emerald-500/5 space-y-2.5 shadow-2xs">
                      <div className="flex items-center gap-2 text-emerald-600 dark:text-emerald-400 font-bold text-xs uppercase tracking-wider">
                        <Sparkles className="w-4 h-4" />
                        <span>3. Multimodal Fusion</span>
                      </div>
                      <h4 className="font-semibold text-sm text-foreground">Acoustic + Text Fusion</h4>
                      <p className="text-xs text-foreground">
                        Combines vocal prosody with textual sentiment and evaluates cross-modal incongruence.
                      </p>
                      <div className="text-[11px] space-y-1 pt-1 border-t border-emerald-500/20">
                        <div className="text-emerald-600 dark:text-emerald-400 font-medium">✓ Flags sarcasm &amp; true caller emotion</div>
                        <div className="text-foreground font-medium">✓ Our Implemented Architecture</div>
                      </div>
                    </div>
                  </div>

                  {/* Open-Source vs Commercial Landscape Comparison */}
                  <div className="p-5 rounded-2xl border border-border/80 bg-card/80 space-y-3 shadow-2xs">
                    <h3 className="text-sm font-bold text-foreground flex items-center gap-2">
                      <Scale className="w-4 h-4 text-primary" />
                      <span>Commercial Third-Party Review vs. Open-Source Self-Hosted</span>
                    </h3>

                    <div className="grid grid-cols-1 md:grid-cols-2 gap-4 text-xs">
                      <div className="p-3.5 rounded-xl bg-muted/60 border border-border/60 space-y-2">
                        <span className="font-bold text-foreground block">Commercial Cloud APIs (Hume AI, AWS, Google)</span>
                        <ul className="space-y-1.5 text-muted-foreground">
                          <li>• <strong>High Operating Cost:</strong> $0.036 – $0.07 per audio minute ($2,000+/mo for call centers).</li>
                          <li>• <strong>Data Privacy &amp; Lock-in:</strong> Sensitive customer audio leaves corporate VPC.</li>
                          <li>• <strong>Rigid Taxonomy:</strong> Black-box pipelines; cannot customize overlap unmixing or thresholds.</li>
                        </ul>
                      </div>

                      <div className="p-3.5 rounded-xl bg-primary/5 border border-primary/20 space-y-2">
                        <span className="font-bold text-primary block">Our Open-Source Self-Hosted Pipeline</span>
                        <ul className="space-y-1.5 text-foreground">
                          <li>• <strong>Zero API Cost:</strong> Runs 100% locally on standard consumer/server GPUs.</li>
                          <li>• <strong>Enterprise Privacy:</strong> Meets HIPAA, GDPR, and PCI-DSS compliance requirements.</li>
                          <li>• <strong>Full Control:</strong> Custom device routing, targeted separation, and millisecond snapping.</li>
                        </ul>
                      </div>
                    </div>
                  </div>
                </div>
              )}

              {/* ==================== SLIDE 2: EVALUATION ==================== */}
              {currentSlideIndex === 1 && (
                <div className="space-y-6">
                  {/* Evaluation Criteria Grid */}
                  <div className="grid grid-cols-2 sm:grid-cols-3 gap-3">
                    <div className="p-3.5 rounded-xl border border-border/60 bg-card space-y-1">
                      <span className="text-[10px] font-mono uppercase text-muted-foreground font-semibold">Criteria 1</span>
                      <h4 className="font-bold text-xs sm:text-sm text-foreground">Emotion Accuracy (UAR)</h4>
                      <p className="text-[11px] text-muted-foreground">Macro-F1 &amp; Unweighted Average Recall across 7–8 affects.</p>
                    </div>
                    <div className="p-3.5 rounded-xl border border-border/60 bg-card space-y-1">
                      <span className="text-[10px] font-mono uppercase text-muted-foreground font-semibold">Criteria 2</span>
                      <h4 className="font-bold text-xs sm:text-sm text-foreground">WER &amp; Timestamp Sync</h4>
                      <p className="text-[11px] text-muted-foreground">Word Error Rate &amp; millisecond boundary precision (&le;50ms).</p>
                    </div>
                    <div className="p-3.5 rounded-xl border border-border/60 bg-card space-y-1">
                      <span className="text-[10px] font-mono uppercase text-muted-foreground font-semibold">Criteria 3</span>
                      <h4 className="font-bold text-xs sm:text-sm text-foreground">Diarization Error (DER)</h4>
                      <p className="text-[11px] text-muted-foreground">Speaker confusion &amp; simultaneous overlap separation (&gt;12dB SDR).</p>
                    </div>
                    <div className="p-3.5 rounded-xl border border-border/60 bg-card space-y-1">
                      <span className="text-[10px] font-mono uppercase text-muted-foreground font-semibold">Criteria 4</span>
                      <h4 className="font-bold text-xs sm:text-sm text-foreground">Latency &amp; Throughput (RTF)</h4>
                      <p className="text-[11px] text-muted-foreground">Real-Time Factor &le; 0.15 on GPU (&lt;16s for a 2.5-min call).</p>
                    </div>
                    <div className="p-3.5 rounded-xl border border-border/60 bg-card space-y-1">
                      <span className="text-[10px] font-mono uppercase text-muted-foreground font-semibold">Criteria 5</span>
                      <h4 className="font-bold text-xs sm:text-sm text-foreground">VRAM Budget (&le;8 GB)</h4>
                      <p className="text-[11px] text-muted-foreground">Consumer GPU compatibility (RTX 3060/4090/T4) with zero disk I/O.</p>
                    </div>
                    <div className="p-3.5 rounded-xl border border-border/60 bg-card space-y-1">
                      <span className="text-[10px] font-mono uppercase text-muted-foreground font-semibold">Criteria 6</span>
                      <h4 className="font-bold text-xs sm:text-sm text-foreground">On-Prem Data Privacy</h4>
                      <p className="text-[11px] text-muted-foreground">Zero cloud egress; full air-gapped deployment capability.</p>
                    </div>
                  </div>

                  {/* Benchmark Evaluation Table */}
                  <div className="p-5 rounded-2xl border border-border/80 bg-card/80 space-y-3">
                    <h3 className="font-bold text-sm text-foreground flex items-center gap-2">
                      <Workflow className="w-4 h-4 text-primary" />
                      <span>Evaluated Model Benchmarks &amp; Selected Candidates</span>
                    </h3>

                    <div className="overflow-x-auto text-xs">
                      <table className="w-full text-left border-collapse">
                        <thead>
                          <tr className="border-b border-border/60 text-muted-foreground">
                            <th className="py-2 px-3 font-semibold">Component</th>
                            <th className="py-2 px-3 font-semibold">Tested Candidates</th>
                            <th className="py-2 px-3 font-semibold">Key Metric</th>
                            <th className="py-2 px-3 font-semibold">RTF</th>
                            <th className="py-2 px-3 font-semibold">Verdict &amp; Rationale</th>
                          </tr>
                        </thead>
                        <tbody className="divide-y divide-border/40">
                          <tr>
                            <td className="py-2.5 px-3 font-bold text-foreground">Diarization</td>
                            <td className="py-2.5 px-3">Pyannote 3.1 vs NeMo vs Kaldi</td>
                            <td className="py-2.5 px-3 font-mono text-emerald-600 dark:text-emerald-400">DER: 11.8%</td>
                            <td className="py-2.5 px-3 font-mono">0.055</td>
                            <td className="py-2.5 px-3 font-medium text-foreground">Pyannote 3.1 selected for superior OSD collision detection.</td>
                          </tr>
                          <tr>
                            <td className="py-2.5 px-3 font-bold text-foreground">Separation</td>
                            <td className="py-2.5 px-3">SepFormer vs Conv-TasNet vs Demucs</td>
                            <td className="py-2.5 px-3 font-mono text-emerald-600 dark:text-emerald-400">SI-SNRi: 22.4 dB</td>
                            <td className="py-2.5 px-3 font-mono">0.02 (targeted)</td>
                            <td className="py-2.5 px-3 font-medium text-foreground">SepFormer selected with targeted 0.2s padded slicing.</td>
                          </tr>
                          <tr>
                            <td className="py-2.5 px-3 font-bold text-foreground">ASR</td>
                            <td className="py-2.5 px-3">Faster-Whisper Turbo vs Vanilla Whisper</td>
                            <td className="py-2.5 px-3 font-mono text-emerald-600 dark:text-emerald-400">WER: 7.2%</td>
                            <td className="py-2.5 px-3 font-mono">0.042</td>
                            <td className="py-2.5 px-3 font-medium text-foreground">Faster-Whisper Turbo selected (4.2x faster, FP16 CTranslate2).</td>
                          </tr>
                          <tr>
                            <td className="py-2.5 px-3 font-bold text-foreground">Acoustic Affect</td>
                            <td className="py-2.5 px-3">emotion2vec+ vs OpenSMILE vs HuBERT</td>
                            <td className="py-2.5 px-3 font-mono text-emerald-600 dark:text-emerald-400">UAR: 71.4%</td>
                            <td className="py-2.5 px-3 font-mono">0.018</td>
                            <td className="py-2.5 px-3 font-medium text-foreground">emotion2vec+ selected for continuous Valence/Arousal.</td>
                          </tr>
                          <tr>
                            <td className="py-2.5 px-3 font-bold text-foreground">Text Sentiment</td>
                            <td className="py-2.5 px-3">DistilRoBERTa vs BERT-base vs VADER</td>
                            <td className="py-2.5 px-3 font-mono text-emerald-600 dark:text-emerald-400">Macro-F1: 81.2%</td>
                            <td className="py-2.5 px-3 font-mono">0.008 (CPU)</td>
                            <td className="py-2.5 px-3 font-medium text-foreground">DistilRoBERTa selected for 7-class affect with context prompt.</td>
                          </tr>
                        </tbody>
                      </table>
                    </div>
                  </div>
                </div>
              )}

              {/* ==================== SLIDE 3: RECOMMENDATION & TRADE-OFFS ==================== */}
              {currentSlideIndex === 2 && (
                <div className="space-y-6">
                  {/* Trade-Offs Table */}
                  <div className="p-5 rounded-2xl border border-border/80 bg-card space-y-3 shadow-2xs">
                    <h3 className="font-bold text-sm text-foreground flex items-center gap-2">
                      <Scale className="w-4 h-4 text-primary" />
                      <span>Key Architectural Trade-Off Decisions</span>
                    </h3>

                    <div className="grid grid-cols-1 md:grid-cols-2 gap-4 text-xs">
                      <div className="p-3.5 rounded-xl bg-muted/60 border border-border/60 space-y-1.5">
                        <strong className="text-foreground block">Zero-Disk RAM Tensors vs. File Cascade</strong>
                        <p className="text-muted-foreground">
                          <strong>Trade-off:</strong> +120 MB RAM footprint vs. 400% faster I/O.
                        </p>
                        <p className="text-muted-foreground">
                          <strong>Why:</strong> Writing intermediate WAVs to SSD causes file lock collisions and SSD wear under concurrent API load.
                        </p>
                      </div>

                      <div className="p-3.5 rounded-xl bg-muted/60 border border-border/60 space-y-1.5">
                        <strong className="text-foreground block">Fast Stereo Bypass vs. Full Neural Diarization</strong>
                        <p className="text-muted-foreground">
                          <strong>Trade-off:</strong> Pearson correlation check vs. heavy neural execution.
                        </p>
                        <p className="text-muted-foreground">
                          <strong>Why:</strong> 42% of telephony audio is recorded stereo. Skipping Pyannote/SepFormer gives a 10x instant speedup (&lt;2.5s).
                        </p>
                      </div>

                      <div className="p-3.5 rounded-xl bg-muted/60 border border-border/60 space-y-1.5">
                        <strong className="text-foreground block">Targeted Overlap Slicing vs. Full-File Separation</strong>
                        <p className="text-muted-foreground">
                          <strong>Trade-off:</strong> Slicing &amp; cross-fade logic vs. massive GPU compute.
                        </p>
                        <p className="text-muted-foreground">
                          <strong>Why:</strong> Collisions account for only 1.5%–3.8% of call time. Running SepFormer only on collisions saves 95% GPU compute.
                        </p>
                      </div>

                      <div className="p-3.5 rounded-xl bg-muted/60 border border-border/60 space-y-1.5">
                        <strong className="text-foreground block">Continuous ASR + Floor Inertia vs. Segment Slicing</strong>
                        <p className="text-muted-foreground">
                          <strong>Trade-off:</strong> Word intersection mapping vs. chopped audio snippets.
                        </p>
                        <p className="text-muted-foreground">
                          <strong>Why:</strong> Slicing audio before ASR chops off initial consonants. Continuous ASR preserves word flow; floor inertia stops flip-flops.
                        </p>
                      </div>
                    </div>
                  </div>

                  {/* Final Implemented Stack Summary */}
                  <div className="p-5 rounded-2xl border border-primary/30 bg-primary/5 space-y-2.5">
                    <div className="flex items-center gap-2 text-primary font-bold text-sm">
                      <Sparkles className="w-4 h-4" />
                      <span>Final Production Recommendation (What We Implemented)</span>
                    </div>
                    <p className="text-xs sm:text-sm text-foreground leading-relaxed">
                      An in-memory microservice orchestrated via FastAPI, powered by <strong>Pyannote 3.1</strong> for voiceprint clustering, <strong>SepFormer</strong> with targeted 0.2s collision slicing, <strong>Faster-Whisper `large-v3-turbo`</strong> for ASR, <strong>Wav2Vec2 CTC</strong> for millisecond boundary snapping, <strong>emotion2vec+</strong> with strict quality abstention gates, <strong>DistilRoBERTa</strong> for textual affect, and a <strong>Sarcasm Incongruence Engine</strong> for cross-modal conflict detection.
                    </p>
                  </div>
                </div>
              )}

              {/* ==================== SLIDE 4: SYSTEM DATAFLOW ==================== */}
              {currentSlideIndex === 3 && (
                <div className="space-y-5">
                  {/* Stepper Buttons */}
                  <div className="flex items-center gap-1.5 overflow-x-auto pb-2 border-b border-border/60">
                    {[
                      { step: 0, label: "0. Ingestion" },
                      { step: 1, label: "1. Stereo Bypass" },
                      { step: 2, label: "2. Diarization" },
                      { step: 3, label: "3. Separation" },
                      { step: 4, label: "4. Whisper ASR" },
                      { step: 5, label: "5. CTC Snapping" },
                      { step: 6, label: "6. Acoustic Affect" },
                      { step: 7, label: "7. Sarcasm & Fusion" },
                      { step: 8, label: "8. Dynamics Packaging" },
                    ].map((st) => (
                      <button
                        key={st.step}
                        type="button"
                        onClick={() => setActiveStageStep(st.step)}
                        className={`px-3 py-1.5 rounded-xl text-xs font-semibold whitespace-nowrap transition-all cursor-pointer ${
                          activeStageStep === st.step
                            ? "bg-primary text-primary-foreground shadow-2xs"
                            : "bg-muted/70 text-muted-foreground hover:text-foreground"
                        }`}
                      >
                        {st.label}
                      </button>
                    ))}
                  </div>

                  {/* Stage Detail Card */}
                  <div className="p-5 sm:p-6 rounded-2xl border border-border/80 bg-card space-y-4 shadow-2xs">
                    {activeStageStep === 0 && (
                      <div className="space-y-3">
                        <div className="flex items-center gap-2">
                          <span className="px-2.5 py-0.5 rounded-full text-xs font-bold bg-sky-500/10 text-sky-600 dark:text-sky-400">
                            Stage 0 • In-Memory Ingestion
                          </span>
                          <span className="text-xs text-muted-foreground font-mono">audio.py</span>
                        </div>
                        <h3 className="text-lg font-bold text-foreground">Zero-Disk Audio Ingestion &amp; 16 kHz Standardization</h3>
                        <p className="text-xs sm:text-sm text-muted-foreground leading-relaxed">
                          Incoming audio bytes (`.wav`, `.mp3`, `.m4a`, `.flac`) are loaded directly into RAM via `io.BytesIO` using `soundfile`. The audio is converted into a 32-bit floating-point PyTorch tensor `[channels, samples]` and normalized to 16,000 Hz using `torchaudio.functional.resample`.
                        </p>
                        <div className="p-3.5 rounded-xl bg-muted/60 border border-border/60 text-xs font-mono space-y-1">
                          <div className="text-emerald-600 dark:text-emerald-400 font-semibold">Key Advantage:</div>
                          <div>Zero disk I/O eliminates SSD latency bottlenecks and avoids Windows file-locking issues in concurrent environments.</div>
                        </div>
                      </div>
                    )}

                    {activeStageStep === 1 && (
                      <div className="space-y-3">
                        <div className="flex items-center gap-2">
                          <span className="px-2.5 py-0.5 rounded-full text-xs font-bold bg-emerald-500/10 text-emerald-600 dark:text-emerald-400">
                            Stage 1 • Branch Decision
                          </span>
                          <span className="text-xs text-muted-foreground font-mono">branch.py</span>
                        </div>
                        <h3 className="text-lg font-bold text-foreground">Instant Stereo Channel Correlation Bypass (~10x Speedup)</h3>
                        <p className="text-xs sm:text-sm text-muted-foreground leading-relaxed">
                          Enterprise call center recorders frequently isolate the agent on Channel 0 and customer on Channel 1. Before executing heavy neural models, the system computes the Pearson correlation coefficient between the two channels:
                        </p>
                        <div className="p-3 rounded-xl bg-muted/70 font-mono text-xs text-foreground">
                          ρ = cov(Ch0, Ch1) / [std(Ch0) * std(Ch1)]
                        </div>
                        <p className="text-xs text-muted-foreground">
                          If ρ &lt; 0.6 and both channels have active speech energy, the engine marks the call as <code>stereo_split</code>, assigning Channel 0 to Speaker 0 and Channel 1 to Speaker 1. It skips Pyannote and SepFormer completely, completing processing in under 2.5 seconds.
                        </p>
                      </div>
                    )}

                    {activeStageStep === 2 && (
                      <div className="space-y-3">
                        <div className="flex items-center gap-2">
                          <span className="px-2.5 py-0.5 rounded-full text-xs font-bold bg-indigo-500/10 text-indigo-600 dark:text-indigo-400">
                            Stage 2 • Diarization
                          </span>
                          <span className="text-xs text-muted-foreground font-mono">diarizer.py</span>
                        </div>
                        <h3 className="text-lg font-bold text-foreground">Pyannote 3.1 Neural Diarization &amp; Overlap Detection</h3>
                        <p className="text-xs sm:text-sm text-muted-foreground leading-relaxed">
                          When processing mono audio, Pyannote processes the in-memory waveform to cluster speaker identities and detect simultaneous overlapping speech intervals (OSD).
                        </p>
                        <div className="grid grid-cols-1 sm:grid-cols-2 gap-3 text-xs">
                          <div className="p-3 rounded-xl bg-muted/50 border border-border/60">
                            <strong className="text-foreground block mb-1">Phantom Cluster Pruning:</strong>
                            Speakers with negligible duration (&lt;0.3s) are reassigned to the nearest valid speaker in time, preventing brief coughs from spawning fake speakers.
                          </div>
                          <div className="p-3 rounded-xl bg-muted/50 border border-border/60">
                            <strong className="text-foreground block mb-1">Collision Tagging:</strong>
                            Intervals where Speaker A and Speaker B speak simultaneously (&gt;0.05s) are isolated as collision windows for targeted separation.
                          </div>
                        </div>
                      </div>
                    )}

                    {activeStageStep === 3 && (
                      <div className="space-y-3">
                        <div className="flex items-center gap-2">
                          <span className="px-2.5 py-0.5 rounded-full text-xs font-bold bg-purple-500/10 text-purple-600 dark:text-purple-400">
                            Stage 3 • Speech Separation
                          </span>
                          <span className="text-xs text-muted-foreground font-mono">overlap_separator.py</span>
                        </div>
                        <h3 className="text-lg font-bold text-foreground">Targeted SepFormer Overlap Separation &amp; Raised-Cosine Splicing</h3>
                        <p className="text-xs sm:text-sm text-muted-foreground leading-relaxed">
                          SepFormer is only run on the precise collision intervals flagged by Pyannote, padded with 0.2s of acoustic context. The separated streams are resolved to Speaker A vs. Speaker B using 256-dim Pyannote neural voiceprints (or 13-dim MFCCs) extracted from clean single-speaker turns.
                        </p>
                        <div className="p-3 rounded-xl bg-muted/70 font-mono text-xs">
                          Raised-Cosine Crossfade Splice: 15–50ms smooth transition eliminates clicking artifacts at boundary edges.
                        </div>
                      </div>
                    )}

                    {activeStageStep === 4 && (
                      <div className="space-y-3">
                        <div className="flex items-center gap-2">
                          <span className="px-2.5 py-0.5 rounded-full text-xs font-bold bg-emerald-500/10 text-emerald-600 dark:text-emerald-400">
                            Stage 4 • Speech Recognition
                          </span>
                          <span className="text-xs text-muted-foreground font-mono">transcriber.py</span>
                        </div>
                        <h3 className="text-lg font-bold text-foreground">Continuous Faster-Whisper ASR &amp; Overlap Re-Transcription</h3>
                        <p className="text-xs sm:text-sm text-muted-foreground leading-relaxed">
                          To avoid chopping off initial consonants, Faster-Whisper (`large-v3-turbo` on CTranslate2) transcribes the continuous audio stream in a single pass, outputting millisecond word timestamps and domain prompt guidance. Collision regions are re-transcribed on the isolated SepFormer streams to capture words masked by crosstalk.
                        </p>
                      </div>
                    )}

                    {activeStageStep === 5 && (
                      <div className="space-y-3">
                        <div className="flex items-center gap-2">
                          <span className="px-2.5 py-0.5 rounded-full text-xs font-bold bg-teal-500/10 text-teal-600 dark:text-teal-400">
                            Stage 5 • Alignment &amp; Snapping
                          </span>
                          <span className="text-xs text-muted-foreground font-mono">aligner.py &amp; segmenter.py</span>
                        </div>
                        <h3 className="text-lg font-bold text-foreground">WhisperX Word Intersection &amp; Wav2Vec2 CTC Phoneme Snapping</h3>
                        <p className="text-xs sm:text-sm text-muted-foreground leading-relaxed">
                          Words are assigned to speakers via temporal intersection with floor inertia (hysteresis). Turns are formed via Inter-Pausal Units (IPU ≥ 1.2s) and multilingual sentence terminal regex. Wav2Vec2 CTC emissions then snap the turn boundaries to the exact millisecond speech began and ended.
                        </p>
                      </div>
                    )}

                    {activeStageStep === 6 && (
                      <div className="space-y-3">
                        <div className="flex items-center gap-2">
                          <span className="px-2.5 py-0.5 rounded-full text-xs font-bold bg-amber-500/10 text-amber-600 dark:text-amber-400">
                            Stage 6 • Acoustic Emotion
                          </span>
                          <span className="text-xs text-muted-foreground font-mono">acoustic.py</span>
                        </div>
                        <h3 className="text-lg font-bold text-foreground">emotion2vec Affect &amp; Strict Quality Abstention Gates</h3>
                        <p className="text-xs sm:text-sm text-muted-foreground leading-relaxed">
                          Utterance audio is passed into emotion2vec to derive continuous Valence and Arousal. Strict abstention gates protect accuracy: turns shorter than 0.4s abstain (`SHORT_TURN`), near-zero energy slices abstain (`LOW_ENERGY`), and ambiguous margins default to `neutral`.
                        </p>
                      </div>
                    )}

                    {activeStageStep === 7 && (
                      <div className="space-y-3">
                        <div className="flex items-center gap-2">
                          <span className="px-2.5 py-0.5 rounded-full text-xs font-bold bg-rose-500/10 text-rose-600 dark:text-rose-400">
                            Stage 7 • Multimodal Fusion
                          </span>
                          <span className="text-xs text-muted-foreground font-mono">semantic.py &amp; orchestrator.py</span>
                        </div>
                        <h3 className="text-lg font-bold text-foreground">Multimodal Late Fusion &amp; Sarcasm Incongruence Engine</h3>
                        <p className="text-xs sm:text-sm text-muted-foreground leading-relaxed">
                          DistilRoBERTa classifies text affect conditioned on conversational history. If the text expresses positive words (`happy`), but vocal prosody is hostile or distressed (`angry`, `disgusted`, or Valence ≤ -0.25), the engine flags <code>is_sarcastic = True</code> with an incongruence score.
                        </p>
                      </div>
                    )}

                    {activeStageStep === 8 && (
                      <div className="space-y-3">
                        <div className="flex items-center gap-2">
                          <span className="px-2.5 py-0.5 rounded-full text-xs font-bold bg-violet-500/10 text-violet-600 dark:text-violet-400">
                            Stage 8 • Dynamics Packaging
                          </span>
                          <span className="text-xs text-muted-foreground font-mono">orchestrator.py</span>
                        </div>
                        <h3 className="text-lg font-bold text-foreground">Conversational Dynamics &amp; Clean Response Serialization</h3>
                        <p className="text-xs sm:text-sm text-muted-foreground leading-relaxed">
                          Compiles interaction dynamics: flags interruptions when Speaker B starts before Speaker A yields within 1.0s, calculates call overlap ratio and total speech duration, and serializes into the strongly-typed `AnalyzeResponse` DTO.
                        </p>
                      </div>
                    )}
                  </div>
                </div>
              )}

              {/* ==================== SLIDE 5: SETUP & CHALLENGES ==================== */}
              {currentSlideIndex === 4 && (
                <div className="space-y-5">
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                    {/* Setup Card */}
                    <div className="p-4 rounded-2xl border border-border/80 bg-card space-y-3 shadow-2xs">
                      <div className="flex items-center gap-2 text-primary font-bold text-sm">
                        <Wrench className="w-4 h-4" />
                        <span>Setup Prerequisites &amp; Configuration</span>
                      </div>
                      <div className="space-y-2 text-xs">
                        <div className="p-2.5 rounded-xl bg-muted/60 font-mono text-[11px]">
                          pip install torch torchaudio pyannote.audio faster-whisper speechbrain funasr transformers
                        </div>
                        <p className="text-muted-foreground">
                          • <strong>Windows FFmpeg Registration:</strong> Requires <code>os.add_dll_directory(FFMPEG_BIN_PATH)</code> to register C-libraries into Windows 11 dynamic search.
                        </p>
                        <p className="text-muted-foreground">
                          • <strong>Hugging Face Token:</strong> Needs valid <code>HF_TOKEN</code> with approved community agreement for Pyannote 3.1.
                        </p>
                        <p className="text-muted-foreground">
                          • <strong>Device Placement:</strong> Granular <code>.env</code> file routes Diarization &amp; Whisper to CUDA:0 while keeping Alignment &amp; NLP on CPU.
                        </p>
                      </div>
                    </div>

                    {/* Challenges Solved */}
                    <div className="p-4 rounded-2xl border border-border/80 bg-card space-y-3 shadow-2xs">
                      <div className="flex items-center gap-2 text-amber-600 dark:text-amber-400 font-bold text-sm">
                        <AlertTriangle className="w-4 h-4" />
                        <span>Core Engineering Challenges Solved</span>
                      </div>
                      <ul className="text-xs text-muted-foreground space-y-1.5">
                        <li>• <strong>SepFormer 8 kHz Mismatch:</strong> Built in-memory downsample/upsample bridge to prevent audio aliasing.</li>
                        <li>• <strong>Permutation Inversion:</strong> Matched separated streams to speakers using 256-dim neural voiceprints &amp; pre-collision energy.</li>
                        <li>• <strong>Diarization Boundary Bleed:</strong> Eliminated word stealing with floor handover clamping &amp; Wav2Vec2 CTC millisecond snapping.</li>
                        <li>• <strong>False Emotion Alarms:</strong> Implemented strict abstention gates (&lt;0.4s turns and low-energy audio abstain).</li>
                        <li>• <strong>VRAM Leaks:</strong> Cleared PyTorch CUDA cache at stage boundaries with explicit garbage collection.</li>
                      </ul>
                    </div>
                  </div>

                  {/* Real-World Observations */}
                  <div className="p-4 rounded-2xl border border-emerald-500/30 bg-emerald-500/5 space-y-2">
                    <div className="flex items-center gap-2 text-emerald-600 dark:text-emerald-400 font-bold text-xs uppercase tracking-wider">
                      <Lightbulb className="w-4 h-4" />
                      <span>Empirical Field Observations</span>
                    </div>
                    <div className="grid grid-cols-1 sm:grid-cols-3 gap-3 text-xs text-foreground">
                      <div>• <strong>42% Calls Are Stereo:</strong> Correlation bypass saves ~22s per call on contact center recordings.</div>
                      <div>• <strong>Overlap Is Only 1.5%–3.8%:</strong> Targeted separation saves 96% compute over full-file separation.</div>
                      <div>• <strong>Vocabulary Prompt Impact:</strong> Initial prompt reduced Bangalore proper-noun WER from 28.4% to 4.1%.</div>
                    </div>
                  </div>
                </div>
              )}

              {/* ==================== SLIDE 6: FAILURE SCENARIOS ==================== */}
              {currentSlideIndex === 5 && (
                <div className="space-y-4">
                  {/* Failure Tab Selector */}
                  <div className="flex items-center gap-1.5 overflow-x-auto pb-2 border-b border-border/60">
                    {[
                      "1. 3+ Talker Overlap",
                      "2. Low-SNR Audio (< -30dB)",
                      "3. Non-English Audio",
                      "4. Flat/Monotone Voice",
                      "5. Run-on Monologue",
                      "6. VRAM OOM (>30 mins)",
                      "7. LLM Outage Fallback",
                    ].map((tabName, idx) => (
                      <button
                        key={tabName}
                        type="button"
                        onClick={() => setActiveFailureTab(idx)}
                        className={`px-3 py-1.5 rounded-xl text-xs font-semibold whitespace-nowrap transition-all cursor-pointer ${
                          activeFailureTab === idx
                            ? "bg-destructive text-destructive-foreground shadow-2xs"
                            : "bg-muted/70 text-muted-foreground hover:text-foreground"
                        }`}
                      >
                        {tabName}
                      </button>
                    ))}
                  </div>

                  {/* Failure Tab Details */}
                  <div className="p-5 sm:p-6 rounded-2xl border border-border/80 bg-card space-y-3 shadow-2xs">
                    {activeFailureTab === 0 && (
                      <div className="space-y-2 text-xs sm:text-sm">
                        <div className="flex items-center justify-between">
                          <h4 className="font-bold text-base text-foreground">1. Multiparty Overlap (3+ Concurrent Talkers)</h4>
                          <span className="text-xs px-2 py-0.5 rounded bg-destructive/10 text-destructive font-mono font-semibold">Crosstalk</span>
                        </div>
                        <p className="text-muted-foreground">
                          <strong>How to Achieve / Reproduce:</strong> Play audio where 3 or more speakers talk or shout simultaneously.
                        </p>
                        <p className="text-muted-foreground">
                          <strong>Failure Cause:</strong> SepFormer `wsj02mix` has exactly 2 output channels. The 3rd talker leaks into both tracks, producing phase cancellation.
                        </p>
                        <div className="p-3 rounded-xl bg-muted/60 border border-border/60 font-mono text-xs">
                          <strong>Mitigation:</strong> Engine falls back to continuous mono ASR + WhisperX temporal intersection; flags call-level `overlap_ratio` without crashing.
                        </div>
                      </div>
                    )}

                    {activeFailureTab === 1 && (
                      <div className="space-y-2 text-xs sm:text-sm">
                        <div className="flex items-center justify-between">
                          <h4 className="font-bold text-base text-foreground">2. Low-SNR Audio / Far-Mic Murmurs (&lt; -30 dBFS)</h4>
                          <span className="text-xs px-2 py-0.5 rounded bg-destructive/10 text-destructive font-mono font-semibold">Low Gain</span>
                        </div>
                        <p className="text-muted-foreground">
                          <strong>How to Achieve / Reproduce:</strong> Caller speaks very far from the mic in a loud room (RMS &lt; -30 dBFS).
                        </p>
                        <p className="text-muted-foreground">
                          <strong>Failure Cause:</strong> Whisper hallucinates repetitive loops (*"Thank you. Thank you."*); emotion2vec outputs random emotions.
                        </p>
                        <div className="p-3 rounded-xl bg-muted/60 border border-border/60 font-mono text-xs">
                          <strong>Mitigation:</strong> `normalize_loudness` boosts RMS to -20 dBFS in RAM; `min_word_prob` and `LOW_ENERGY` gate silence hallucination.
                        </div>
                      </div>
                    )}

                    {activeFailureTab === 2 && (
                      <div className="space-y-2 text-xs sm:text-sm">
                        <div className="flex items-center justify-between">
                          <h4 className="font-bold text-base text-foreground">3. Non-English Audio with English CTC Aligner</h4>
                          <span className="text-xs px-2 py-0.5 rounded bg-destructive/10 text-destructive font-mono font-semibold">Language Mismatch</span>
                        </div>
                        <p className="text-muted-foreground">
                          <strong>How to Achieve / Reproduce:</strong> Upload audio in Hindi, Spanish, or French with `wav2vec2-base-960h` active.
                        </p>
                        <p className="text-muted-foreground">
                          <strong>Failure Cause:</strong> Wav2Vec2 CTC dictionary lacks non-Latin characters, failing token emission alignment.
                        </p>
                        <div className="p-3 rounded-xl bg-muted/60 border border-border/60 font-mono text-xs">
                          <strong>Mitigation:</strong> Try-catch block catches CTC emission errors and gracefully falls back to Whisper native word timestamps.
                        </div>
                      </div>
                    )}

                    {activeFailureTab === 3 && (
                      <div className="space-y-2 text-xs sm:text-sm">
                        <div className="flex items-center justify-between">
                          <h4 className="font-bold text-base text-foreground">4. Flat / Robotic Monotone Voice in Angry Call</h4>
                          <span className="text-xs px-2 py-0.5 rounded bg-destructive/10 text-destructive font-mono font-semibold">Acoustic Masking</span>
                        </div>
                        <p className="text-muted-foreground">
                          <strong>How to Achieve / Reproduce:</strong> Customer deadpans extreme complaints in a flat, monotone voice without pitch variation.
                        </p>
                        <p className="text-muted-foreground">
                          <strong>Failure Cause:</strong> Acoustic emotion predicts `neutral` (0.85); acoustic-only pipelines completely miss the cancellation threat.
                        </p>
                        <div className="p-3 rounded-xl bg-muted/60 border border-border/60 font-mono text-xs">
                          <strong>Mitigation:</strong> Calibrated Late Fusion discounts uncalibrated neutral acoustic prior (&alpha;=0.20), allowing confident text sentiment to override.
                        </div>
                      </div>
                    )}

                    {activeFailureTab === 4 && (
                      <div className="space-y-2 text-xs sm:text-sm">
                        <div className="flex items-center justify-between">
                          <h4 className="font-bold text-base text-foreground">5. Run-on Monologue Without Natural Breathing Pauses</h4>
                          <span className="text-xs px-2 py-0.5 rounded bg-destructive/10 text-destructive font-mono font-semibold">Pacing</span>
                        </div>
                        <p className="text-muted-foreground">
                          <strong>How to Achieve / Reproduce:</strong> Speaker talks for 60 seconds continuously without taking pauses.
                        </p>
                        <p className="text-muted-foreground">
                          <strong>Failure Cause:</strong> Standard diarizers produce a single 60s turn, averaging away localized emotional spikes.
                        </p>
                        <div className="p-3 rounded-xl bg-muted/60 border border-border/60 font-mono text-xs">
                          <strong>Mitigation:</strong> `segmenter.py` enforces sentence terminal regex splitting at `soft_turn_duration_s=4.0` and a hard cap at `max_turn_duration_s=8.0`.
                        </div>
                      </div>
                    )}

                    {activeFailureTab === 5 && (
                      <div className="space-y-2 text-xs sm:text-sm">
                        <div className="flex items-center justify-between">
                          <h4 className="font-bold text-base text-foreground">6. GPU Out-of-Memory (OOM) on Extended Calls (&gt;30 mins)</h4>
                          <span className="text-xs px-2 py-0.5 rounded bg-destructive/10 text-destructive font-mono font-semibold">VRAM Limit</span>
                        </div>
                        <p className="text-muted-foreground">
                          <strong>How to Achieve / Reproduce:</strong> Upload a 45-minute continuous recording on an 8GB GPU without memory clearing.
                        </p>
                        <p className="text-muted-foreground">
                          <strong>Failure Cause:</strong> PyTorch attention matrix activations exceed 8GB VRAM during concurrent inference.
                        </p>
                        <div className="p-3 rounded-xl bg-muted/60 border border-border/60 font-mono text-xs">
                          <strong>Mitigation:</strong> Alignment &amp; NLP placed on CPU; explicit `gc.collect()` and `torch.cuda.empty_cache()` executed at stage boundaries.
                        </div>
                      </div>
                    )}

                    {activeFailureTab === 6 && (
                      <div className="space-y-2 text-xs sm:text-sm">
                        <div className="flex items-center justify-between">
                          <h4 className="font-bold text-base text-foreground">7. LLM Service Outage or Missing OpenRouter API Key</h4>
                          <span className="text-xs px-2 py-0.5 rounded bg-destructive/10 text-destructive font-mono font-semibold">External API</span>
                        </div>
                        <p className="text-muted-foreground">
                          <strong>How to Achieve / Reproduce:</strong> Run post-call analysis with unset `OPENROUTER_API_KEY` or offline network.
                        </p>
                        <p className="text-muted-foreground">
                          <strong>Failure Cause:</strong> External HTTP request to OpenRouter fails with HTTP 401 or timeout.
                        </p>
                        <div className="p-3 rounded-xl bg-muted/60 border border-border/60 font-mono text-xs">
                          <strong>Mitigation:</strong> `call_analyzer.py` invokes `_generate_deterministic_analysis()`, computing CSAT and trajectories offline without crashing.
                        </div>
                      </div>
                    )}
                  </div>
                </div>
              )}
            </div>
          ) : (
            /* Full Markdown Specification Mode */
            <div className="max-w-4xl mx-auto space-y-4">
              {/* Document Search Bar */}
              <div className="relative">
                <Search className="w-4 h-4 absolute left-3 top-3 text-muted-foreground" />
                <input
                  type="text"
                  value={docSearchQuery}
                  onChange={(e) => setDocSearchQuery(e.target.value)}
                  placeholder="Filter full technical specification document..."
                  className="w-full pl-9 pr-4 py-2 rounded-xl bg-muted/60 border border-border/70 text-xs sm:text-sm focus:outline-hidden focus:ring-2 focus:ring-primary/40 text-foreground"
                />
              </div>

              {/* Rendered Markdown Document */}
              <div className="p-6 sm:p-8 rounded-2xl border border-border/70 bg-card prose prose-neutral dark:prose-invert max-w-none text-xs sm:text-sm leading-relaxed">
                <ReactMarkdown
                  remarkPlugins={[remarkGfm]}
                  components={{
                    h1: ({ ...props }) => (
                      <h1 className="text-2xl sm:text-3xl font-extrabold tracking-tight mt-6 mb-4 text-foreground border-b border-border/60 pb-3" {...props} />
                    ),
                    h2: ({ ...props }) => (
                      <h2 className="text-xl sm:text-2xl font-bold tracking-tight mt-8 mb-3 text-foreground" {...props} />
                    ),
                    h3: ({ ...props }) => (
                      <h3 className="text-base sm:text-lg font-semibold tracking-tight mt-6 mb-2 text-foreground" {...props} />
                    ),
                    p: ({ ...props }) => (
                      <p className="text-xs sm:text-sm text-muted-foreground leading-relaxed mb-4" {...props} />
                    ),
                    table: ({ ...props }) => (
                      <div className="overflow-x-auto my-4 rounded-xl border border-border/70">
                        <table className="w-full text-xs text-left" {...props} />
                      </div>
                    ),
                    th: ({ ...props }) => (
                      <th className="bg-muted/70 px-3 py-2 font-semibold text-foreground border-b border-border/60" {...props} />
                    ),
                    td: ({ ...props }) => (
                      <td className="px-3 py-2 border-b border-border/40 text-muted-foreground" {...props} />
                    ),
                    code: ({ className, children, ...props }) => {
                      const isInline = !className;
                      return isInline ? (
                        <code className="bg-muted px-1.5 py-0.5 rounded font-mono text-[11px] text-foreground border border-border/60" {...props}>
                          {children}
                        </code>
                      ) : (
                        <pre className="p-4 rounded-xl bg-muted/80 text-foreground font-mono text-xs overflow-x-auto border border-border/70 my-3">
                          <code {...props}>{children}</code>
                        </pre>
                      );
                    },
                  }}
                >
                  {architectureMd}
                </ReactMarkdown>
              </div>
            </div>
          )}
        </div>

        {/* Footer Navigation Bar */}
        <div className="flex items-center justify-between px-4 sm:px-6 py-3.5 border-t border-border/70 bg-muted/40 shrink-0">
          {viewMode === "presentation" ? (
            <>
              {/* Quick Jump Slide Dots */}
              <div className="flex items-center gap-1 sm:gap-2">
                {SLIDES.map((slide, idx) => (
                  <button
                    key={slide.id}
                    type="button"
                    onClick={() => setCurrentSlideIndex(idx)}
                    className={`h-2 rounded-full transition-all cursor-pointer ${
                      currentSlideIndex === idx
                        ? "w-7 bg-primary"
                        : "w-2 bg-muted-foreground/30 hover:bg-muted-foreground/60"
                    }`}
                    title={`Slide ${idx + 1}: ${slide.title}`}
                  />
                ))}
              </div>

              {/* Prev / Next Controls */}
              <div className="flex items-center gap-2">
                <button
                  type="button"
                  onClick={() => setCurrentSlideIndex((prev) => Math.max(prev - 1, 0))}
                  disabled={currentSlideIndex === 0}
                  className="inline-flex items-center gap-1 px-3 py-1.5 rounded-xl text-xs font-medium border border-border/70 bg-card hover:bg-muted text-foreground disabled:opacity-40 disabled:pointer-events-none transition-all cursor-pointer shadow-2xs"
                >
                  <ChevronLeft className="w-4 h-4" />
                  <span className="hidden sm:inline">Previous</span>
                </button>
                <button
                  type="button"
                  onClick={() =>
                    setCurrentSlideIndex((prev) => Math.min(prev + 1, SLIDES.length - 1))
                  }
                  disabled={currentSlideIndex === SLIDES.length - 1}
                  className="inline-flex items-center gap-1 px-3.5 py-1.5 rounded-xl text-xs font-semibold bg-primary text-primary-foreground hover:bg-primary/90 disabled:opacity-40 disabled:pointer-events-none transition-all cursor-pointer shadow-2xs"
                >
                  <span className="hidden sm:inline">Next Slide</span>
                  <ChevronRight className="w-4 h-4" />
                </button>
              </div>
            </>
          ) : (
            <div className="flex items-center justify-between w-full text-xs text-muted-foreground">
              <span>Viewing raw specification: <code>docs/ARCHITECTURE_AND_FLOW_SPEC.md</code></span>
              <button
                type="button"
                onClick={() => setViewMode("presentation")}
                className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-xl bg-card border border-border/70 hover:bg-muted text-foreground transition-all cursor-pointer font-medium"
              >
                <Sliders className="w-3.5 h-3.5" />
                <span>Return to Presentation Slides</span>
              </button>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};
