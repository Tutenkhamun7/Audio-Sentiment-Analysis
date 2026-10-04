import { useState, useCallback, useRef } from "react";
import { Header } from "@/components/Header";
import { UploadSection } from "@/components/UploadSection";
import { RecordSection } from "@/components/RecordSection";
import { SummaryHeader } from "@/components/SummaryHeader";
import { AudioPlayerHUD } from "@/components/AudioPlayerHUD";
import { SynchronizedTranscript } from "@/components/SynchronizedTranscript";
import { EmotionTimelineCards } from "@/components/EmotionTimelineCards";
import { AgentSynthesisCard } from "@/components/AgentSynthesisCard";
import {
  analyzeAudioApi,
  MOCK_ANALYSIS_DATA,
  MOCK_AGENT_CONTEXT,
  createSyntheticDemoAudio,
} from "@/services/api";
import type { AudioAnalysisResult, TimelineSegment } from "@/types/emotion";
import {
  Upload,
  Mic,
  AlertCircle,
  Sparkles,
  Info,
  Radio,
  RefreshCw,
  XCircle,
  Loader2,
} from "lucide-react";

type ActiveTab = "upload" | "record";

export function App() {
  const [activeTab, setActiveTab] = useState<ActiveTab>("upload");
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [isLoading, setIsLoading] = useState<boolean>(false);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);

  const abortControllerRef = useRef<AbortController | null>(null);

  // Analysis result state
  const [analysisResult, setAnalysisResult] = useState<AudioAnalysisResult | null>(null);
  const [audioBlob, setAudioBlob] = useState<Blob | null>(null);

  // Audio player synchronization state
  const [seekTime, setSeekTime] = useState<number | null>(null);
  const [stopAtTime, setStopAtTime] = useState<number | null>(null);
  const [activeSegmentId, setActiveSegmentId] = useState<number | null>(null);

  // Cancel ongoing request
  const handleCancelAnalysis = useCallback(() => {
    if (abortControllerRef.current) {
      abortControllerRef.current.abort();
      abortControllerRef.current = null;
    }
    setIsLoading(false);
  }, []);

  // Reset to initial input studio view
  const handleReset = useCallback(() => {
    handleCancelAnalysis();
    setAnalysisResult(null);
    setAudioBlob(null);
    setSelectedFile(null);
    setErrorMessage(null);
    setSeekTime(null);
    setStopAtTime(null);
    setActiveSegmentId(null);
  }, [handleCancelAnalysis]);

  // Switch input section tab and reset any processed audio state
  const handleTabChange = useCallback((tab: ActiveTab) => {
    handleReset();
    setActiveTab(tab);
  }, [handleReset]);

  // Run analysis for uploaded file
  const handleAnalyzeFile = async (file: File) => {
    const controller = new AbortController();
    abortControllerRef.current = controller;

    setIsLoading(true);
    setErrorMessage(null);
    setAudioBlob(file);

    try {
      const response = await analyzeAudioApi(file, file.name, controller.signal);
      setAnalysisResult({
        segments: response.segments,
        filename: response.filename || file.name,
        fileSize: file.size,
        duration: response.total_duration,
        speakers: response.speakers,
        overall_transcript: response.overall_transcript,
        agent_context: response.agent_context,
      });
    } catch (err: unknown) {
      if (err instanceof Error && (err.name === "AbortError" || err.message.includes("stopped by user"))) {
        setErrorMessage(null);
        return;
      }
      const msg = err instanceof Error ? err.message : String(err);
      setErrorMessage(msg);
    } finally {
      setIsLoading(false);
      abortControllerRef.current = null;
    }
  };

  // Run analysis for recorded audio (instant auto-submission)
  const handleRecordingComplete = async (wavBlob: Blob) => {
    const controller = new AbortController();
    abortControllerRef.current = controller;

    setIsLoading(true);
    setErrorMessage(null);
    setAudioBlob(wavBlob);

    const recordingFilename = `mic_recording_${new Date()
      .toISOString()
      .replace(/[:.]/g, "-")}.wav`;

    try {
      const response = await analyzeAudioApi(wavBlob, recordingFilename, controller.signal);
      setAnalysisResult({
        segments: response.segments,
        filename: response.filename || recordingFilename,
        fileSize: wavBlob.size,
        duration: response.total_duration,
        speakers: response.speakers,
        overall_transcript: response.overall_transcript,
        agent_context: response.agent_context,
      });
    } catch (err: unknown) {
      if (err instanceof Error && (err.name === "AbortError" || err.message.includes("stopped by user"))) {
        setErrorMessage(null);
        return;
      }
      const msg = err instanceof Error ? err.message : String(err);
      setErrorMessage(msg);
    } finally {
      setIsLoading(false);
      abortControllerRef.current = null;
    }
  };

  // Load interactive demo dataset (works seamlessly even when offline)
  const handleLoadDemo = useCallback(() => {
    setIsLoading(false);
    setErrorMessage(null);
    const demoBlob = createSyntheticDemoAudio();
    setAudioBlob(demoBlob);

    const speakers = Array.from(
      new Set(MOCK_ANALYSIS_DATA.map((s: TimelineSegment) => s.speaker))
    );

    setAnalysisResult({
      segments: MOCK_ANALYSIS_DATA,
      filename: "dialogue_prosody_sample.wav",
      fileSize: demoBlob.size,
      duration: 21.0,
      speakers,
      agent_context: MOCK_AGENT_CONTEXT,
    });
  }, []);

  const handleTranscriptSeek = (time: number) => {
    setStopAtTime(null);
    setSeekTime(time);
  };

  const handlePlaySegment = (startTime: number, endTime?: number) => {
    setStopAtTime(typeof endTime === "number" ? endTime : null);
    setSeekTime(startTime);
  };

  return (
    <div className="min-h-screen bg-background text-foreground flex flex-col antialiased selection:bg-primary/20 selection:text-primary">
      {/* Top Navbar */}
      <Header
        onLoadDemo={handleLoadDemo}
        hasResults={!!analysisResult}
        onReset={handleReset}
      />

      {/* Main Studio Interface */}
      <main className="flex-1 max-w-7xl w-full mx-auto px-4 sm:px-6 lg:px-8 py-8 flex flex-col gap-8">
        {/* Studio Page Header */}
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 border-b border-border/60 pb-6">
          <div className="space-y-1">
            <div className="flex items-center gap-3">
              <h1 className="text-2xl sm:text-3xl font-extrabold tracking-tight text-foreground">
                Audio Emotion Studio
              </h1>
              {/* Live Status Badge */}
              <div className="inline-flex items-center gap-1.5 px-2.5 py-0.5 rounded-full text-[11px] font-medium bg-muted/70 text-muted-foreground border border-border/70">
                <span className="w-1.5 h-1.5 rounded-full bg-emerald-500/80" />
                <span>Live Studio</span>
              </div>
            </div>
            <p className="text-xs sm:text-sm text-muted-foreground max-w-2xl">
              Multimodal speech emotion recognition engine. Extract acoustic vocal prosody, NLP
              semantic tone, and detect incongruent emotional conflicts across speech segments.
            </p>
          </div>

          {/* Tab Switcher (Pill-style) */}
          <div className="flex items-center p-1 bg-muted/60 rounded-xl border border-border/70 self-start sm:self-auto shadow-2xs">
            <button
              type="button"
              onClick={() => handleTabChange("upload")}
              className={`flex items-center gap-2 px-3.5 py-1.5 rounded-lg text-xs sm:text-sm font-medium transition-all duration-150 cursor-pointer ${
                activeTab === "upload"
                  ? "bg-card text-foreground shadow-2xs border border-border/60"
                  : "text-muted-foreground hover:text-foreground"
              }`}
            >
              <Upload className="w-4 h-4" />
              <span>Upload File</span>
            </button>
            <button
              type="button"
              onClick={() => handleTabChange("record")}
              className={`flex items-center gap-2 px-3.5 py-1.5 rounded-lg text-xs sm:text-sm font-medium transition-all duration-150 cursor-pointer ${
                activeTab === "record"
                  ? "bg-card text-foreground shadow-2xs border border-border/60"
                  : "text-muted-foreground hover:text-foreground"
              }`}
            >
              <Mic className="w-4 h-4" />
              <span>Record Audio</span>
            </button>
          </div>
        </div>

        {/* Offline / Backend Error Graceful Fallback Banner */}
        {errorMessage && (
          <div className="p-4 sm:p-5 rounded-2xl border border-border/80 bg-card text-foreground flex flex-col sm:flex-row items-start sm:items-center justify-between gap-4 shadow-2xs">
            <div className="flex items-start gap-3">
              <div className="p-2 rounded-lg bg-muted text-muted-foreground shrink-0 mt-0.5">
                <AlertCircle className="w-4 h-4 text-amber-600 dark:text-amber-400" />
              </div>
              <div className="space-y-1">
                <h4 className="text-xs sm:text-sm font-semibold text-foreground">
                  Backend Endpoint Connection Notice
                </h4>
                <p className="text-xs text-muted-foreground">
                  {errorMessage.includes("Failed to fetch") ||
                  errorMessage.includes("NetworkError") ||
                  errorMessage.includes("Connection refused")
                    ? "Could not reach backend API at http://localhost:8000/api/v1/audio/analyze. You can start your local FastAPI server, or load the pre-computed multimodal demo dataset below to test all studio features."
                    : errorMessage}
                </p>
              </div>
            </div>

            <div className="flex items-center gap-2 shrink-0 self-end sm:self-auto">
              <button
                type="button"
                onClick={handleLoadDemo}
                className="inline-flex items-center gap-1.5 px-3.5 py-2 rounded-xl text-xs font-medium bg-foreground text-background hover:opacity-90 shadow-2xs transition-all active:scale-95 cursor-pointer"
              >
                <Sparkles className="w-3.5 h-3.5" />
                <span>Load Demo Dataset</span>
              </button>
              {selectedFile && activeTab === "upload" && (
                <button
                  type="button"
                  onClick={() => handleAnalyzeFile(selectedFile)}
                  className="inline-flex items-center gap-1 px-3 py-2 rounded-xl text-xs font-medium border border-border text-foreground bg-card hover:bg-muted transition-all active:scale-95 cursor-pointer"
                >
                  <RefreshCw className="w-3.5 h-3.5" />
                  <span>Retry</span>
                </button>
              )}
            </div>
          </div>
        )}

        {/* Active Analysis in Progress Card with Stop Request Action */}
        {isLoading && (
          <div className="w-full p-4 sm:p-5 rounded-2xl border border-border/80 bg-card shadow-2xs flex flex-col sm:flex-row items-center justify-between gap-4 animate-in fade-in-50 duration-200">
            <div className="flex items-center gap-3.5">
              <div className="w-9 h-9 rounded-xl bg-muted text-foreground flex items-center justify-center shrink-0">
                <Loader2 className="w-4 h-4 animate-spin text-foreground" />
              </div>
              <div className="space-y-0.5">
                <h4 className="text-xs sm:text-sm font-semibold text-foreground">
                  Running Multimodal Emotion Analysis...
                </h4>
                <p className="text-xs text-muted-foreground">
                  Extracting acoustic prosody, Whisper transcript, and incongruence checks.
                </p>
              </div>
            </div>

            <button
              type="button"
              onClick={handleCancelAnalysis}
              className="inline-flex items-center gap-1.5 px-3.5 py-2 rounded-xl text-xs font-medium border border-border bg-card hover:bg-destructive/10 hover:text-destructive hover:border-destructive/30 text-muted-foreground transition-all cursor-pointer shadow-2xs shrink-0 self-end sm:self-auto"
              title="Stop ongoing request"
            >
              <XCircle className="w-3.5 h-3.5" />
              <span>Stop Request</span>
            </button>
          </div>
        )}

        {/* Input Methods Section (Collapsible or visible above results) */}
        {!analysisResult && (
          <div className="w-full">
            {activeTab === "upload" ? (
              <UploadSection
                onAnalyze={handleAnalyzeFile}
                isLoading={isLoading}
                selectedFile={selectedFile}
                setSelectedFile={setSelectedFile}
                onCancel={handleCancelAnalysis}
              />
            ) : (
              <RecordSection
                onRecordingComplete={handleRecordingComplete}
                isLoading={isLoading}
                onCancel={handleCancelAnalysis}
              />
            )}
          </div>
        )}

        {/* Analysis Results View */}
        {analysisResult && audioBlob && (
          <div className="flex flex-col gap-8 animate-in fade-in-50 duration-300">
            {/* A. Summary Header */}
            <SummaryHeader
              segmentCount={analysisResult.segments.length}
              speakers={analysisResult.speakers}
              filename={analysisResult.filename}
              onReset={handleReset}
            />

            {/* AI Agent Conversational Synthesis (if available from LangGraph agent) */}
            {analysisResult.agent_context && (
              <AgentSynthesisCard agentContext={analysisResult.agent_context} />
            )}

            {/* B. Synchronized Audio Player Card */}
            <AudioPlayerHUD
              audioBlob={audioBlob}
              segments={analysisResult.segments}
              seekTime={seekTime}
              stopAtTime={stopAtTime}
              onSeekHandled={() => {
                setSeekTime(null);
                setStopAtTime(null);
              }}
              activeSegmentId={activeSegmentId}
              setActiveSegmentId={setActiveSegmentId}
            />

            {/* Grid Layout: Synchronized Transcript + Segment Emotion Breakdown */}
            <div className="grid grid-cols-1 lg:grid-cols-12 gap-8 items-start">
              {/* C. Interactive Synchronized Transcript (5 Cols on LG) */}
              <div className="lg:col-span-5 w-full sticky top-24">
                <SynchronizedTranscript
                  segments={analysisResult.segments}
                  activeSegmentId={activeSegmentId}
                  onSeek={handleTranscriptSeek}
                />
              </div>

              {/* D. Segmented Emotion Timeline Cards (7 Cols on LG) */}
              <div className="lg:col-span-7 w-full">
                <EmotionTimelineCards
                  segments={analysisResult.segments}
                  activeSegmentId={activeSegmentId}
                  onPlaySegment={handlePlaySegment}
                />
              </div>
            </div>
          </div>
        )}

        {/* Quick Help Footer note */}
        <div className="mt-auto pt-8 border-t border-border/40 flex flex-col sm:flex-row items-center justify-between text-xs text-muted-foreground gap-3">
          <div className="flex items-center gap-2">
            <Radio className="w-3.5 h-3.5 text-primary" />
            <span>Target Endpoint: <code className="font-mono bg-muted px-1.5 py-0.5 rounded">POST http://localhost:8000/api/v1/audio/analyze</code></span>
          </div>
          <div className="flex items-center gap-1.5">
            <Info className="w-3.5 h-3.5" />
            <span>Audio automatically converted to 16kHz 16-bit Mono PCM WAV in-browser</span>
          </div>
        </div>
      </main>
    </div>
  );
}

export default App;
